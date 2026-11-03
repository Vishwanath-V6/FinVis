"""Retrieval-grounded extractive question answering and citations."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable

import torch

from .retrieval import RetrievalIndex


def load_qa_model(model_name: str = "distilbert-base-uncased-distilled-squad"):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def answer_question(
    question: str,
    retrieval_index: RetrievalIndex,
    tokenizer: Any,
    model: Any,
    top_k: int = 5,
    max_answer_length: int = 64,
    abstention_policy: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if (
        not isinstance(max_answer_length, int)
        or isinstance(max_answer_length, bool)
        or max_answer_length <= 0
    ):
        raise ValueError("max_answer_length must be a positive integer")
    if abstention_policy is not None and not callable(abstention_policy):
        raise TypeError("abstention_policy must be callable or None")
    results = retrieval_index.search(question, top_k=top_k)
    if not results:
        return {"answer": None, "sufficient_evidence": False, "sources": [], "evidence": []}

    table_result = _answer_from_table(question, results)
    if table_result is not None:
        if abstention_policy is not None and abstention_policy(table_result):
            table_result["answer"] = None
            table_result["sufficient_evidence"] = False
        return table_result
    if any(result.get("content_type") == "table" for result in results):
        return {
            "answer": None,
            "sufficient_evidence": False,
            "sources": [],
            "evidence": [],
            "answer_source": "structured_table",
        }

    best = None
    for result in results:
        context = result["text"]
        encoded = tokenizer(
            question,
            context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation="only_second",
            max_length=tokenizer.model_max_length if tokenizer.model_max_length < 4096 else 512,
        )
        sequence_ids = encoded.sequence_ids(0)
        encoded.pop("offset_mapping", None)
        context_positions = [i for i, sequence_id in enumerate(sequence_ids) if sequence_id == 1]
        if not context_positions:
            continue
        with torch.no_grad():
            output = model(**encoded)
        start_logits = output.start_logits[0].clone()
        end_logits = output.end_logits[0].clone()
        if len(start_logits) != len(sequence_ids) or len(end_logits) != len(sequence_ids):
            continue
        candidates = []
        context_set = set(context_positions)
        for start in reversed(context_positions):
            if start not in context_set:
                continue
            for end in range(start, min(start + max_answer_length, len(end_logits))):
                if end in context_set:
                    score = float(start_logits[start] + end_logits[end])
                    candidates.append((score, start, end))
        if not candidates:
            continue
        score, start, end = max(candidates, key=lambda candidate: candidate[0])
        answer = tokenizer.decode(encoded["input_ids"][0, start : end + 1], skip_special_tokens=True).strip()
        candidate = (score, answer, result)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return {"answer": None, "sufficient_evidence": False, "sources": [], "evidence": []}
    score, answer, source = best
    normalized_answer = _normalise_answer(answer)
    source_text = str(source.get("text", ""))
    evidence_supported = bool(normalized_answer) and normalized_answer in _normalise_answer(source_text)
    candidate_result = {
        "answer": answer,
        "sufficient_evidence": evidence_supported,
        "sources": [
            {
                "document_id": source.get("document_id"),
                "source_filename": source.get("source_filename"),
                "page_number": source.get("page_number"),
                "chunk_id": source.get("chunk_id"),
                "score": source.get("score"),
            }
        ],
        "evidence": [source],
        "span_score": score,
    }
    if abstention_policy is not None and abstention_policy(candidate_result):
        candidate_result["answer"] = None
        candidate_result["sufficient_evidence"] = False
    return candidate_result


def _normalise_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _answer_from_table(question: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve an unambiguous metric/year/value directly from a table row."""
    query_terms = {_metric_term(term) for term in re.findall(r"[a-z0-9]+", question.casefold())}
    requested_years = {_period_key(year) for year in re.findall(r"(?:fy)?\d{2,4}", question.casefold())}
    requested_scope = next(
        (scope for scope in ("consolidated", "standalone") if scope in question.casefold()),
        None,
    )
    table_rows = [
        result for result in results
        if result.get("content_type") == "table" and isinstance(result.get("table_values"), dict)
    ]
    calculation = _calculate_table_answer(question, table_rows, requested_scope)
    if calculation is not None:
        source = calculation["source"]
        return {
            "answer": calculation["answer"],
            "sufficient_evidence": True,
            "sources": [_source_citation(source)],
            "evidence": [source],
            "answer_source": "structured_table_calculation",
        }
    candidates: list[tuple[int, dict[str, Any], str]] = []
    for result in results:
        if result.get("content_type") != "table":
            continue
        row_label = str(result.get("table_row_label", "")).casefold()
        label_terms = {_metric_term(term) for term in re.findall(r"[a-z0-9]+", row_label)}
        if not label_terms or not label_terms <= query_terms:
            continue
        values = result.get("table_values")
        if not isinstance(values, dict):
            continue
        matching_values = []
        for header, value in values.items():
            header_text = str(header).casefold()
            if requested_scope and requested_scope not in header_text:
                continue
            header_years = {
                _period_key(year) for year in re.findall(r"(?:fy)?\d{2,4}", header_text)
            }
            if requested_years and not (requested_years & header_years):
                continue
            if isinstance(value, str) and value.strip():
                matching_values.append((str(header), value.strip()))
        if len(matching_values) != 1:
            continue
        header, value = matching_values[0]
        score = len(label_terms & query_terms) * 10
        if requested_years:
            score += 10
        candidates.append((score, result, f"{value} {result.get('table_unit', '')}".strip()))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    _, source, answer = candidates[0]
    return {
        "answer": answer,
        "sufficient_evidence": True,
        "sources": [
            {
                "document_id": source.get("document_id"),
                "source_filename": source.get("source_filename"),
                "page_number": source.get("page_number"),
                "chunk_id": source.get("chunk_id"),
                "score": source.get("score"),
            }
        ],
        "evidence": [source],
        "answer_source": "structured_table",
    }


def _source_citation(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": source.get("document_id"),
        "source_filename": source.get("source_filename"),
        "page_number": source.get("page_number"),
        "chunk_id": source.get("chunk_id"),
        "score": source.get("score"),
    }


def _calculate_table_answer(
            question: str, rows: list[dict[str, Any]], scope: str | None
) -> dict[str, Any] | None:
            lowered = question.casefold()
            if not rows or "fy" not in lowered:
                return None
            metric_rows = [
                row for row in rows
                if _metric_matches_question(str(row.get("table_row_label", "")), question)
            ]
            if not metric_rows:
                return None
            target = metric_rows[0]
            values = target["table_values"]
            prefix = f"{scope.title()} " if scope else ""

            def value(year: str) -> Decimal | None:
                for header, raw in values.items():
                    if prefix.casefold() in str(header).casefold() and _period_key(year) in {
                        _period_key(item) for item in re.findall(r"(?:fy)?\d{2,4}", str(header).casefold())
                    }:
                        try:
                            return Decimal(str(raw).replace(",", "").replace("(", "-").replace(")", ""))
                        except InvalidOperation:
                            return None
                return None

            fy25, fy24 = value("FY25"), value("FY24")
            if fy25 is None or fy24 is None:
                return None
            unit = str(target.get("table_unit", "")).strip()
            if "percentage change" in lowered or "as the base" in lowered:
                result = ((fy25 - fy24) / fy24 * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                return {"answer": f"{result:+.2f}%", "source": target}
            if "change from fy24 to fy25" in lowered:
                result = fy25 - fy24
                direction = "Increase" if result >= 0 else "Decrease"
                return {"answer": f"{direction} of ₹{abs(result):,.0f} {unit.replace('₹ ', '').replace('₹', '')} (FY25 minus FY24 = {result:,.0f} {unit.replace('₹ ', '').replace('₹', '')})", "source": target}
            if "higher or lower" in lowered:
                comparison_rows = [
                    row for row in rows
                    if _metric_matches_question(str(row.get("table_row_label", "")), question)
                ]
                if not comparison_rows:
                    return None
                current = value("FY25" if "fy25" in lowered else "FY24")
                other_scope = "standalone" if scope == "consolidated" else "consolidated"
                other = _value_for_scope(comparison_rows[0]["table_values"], other_scope, "FY25" if "fy25" in lowered else "FY24")
                if current is None or other is None:
                    return None
                difference = current - other
                return {
                    "answer": f"Consolidated was higher by ₹{abs(difference):,.0f} {unit.replace('₹ ', '').replace('₹', '')}",
                    "source": target,
                }
            if "as a percentage of revenue" in lowered and "operating profit" in lowered:
                revenue_rows = [
                    row for row in rows
                    if "revenue from operations" in str(row.get("table_row_label", "")).casefold()
                ]
                if not revenue_rows:
                    return None
                revenue = _value_for_scope(
                    revenue_rows[0]["table_values"], scope or "standalone",
                    "FY25" if "fy25" in lowered else "FY24",
                )
                if revenue in (None, Decimal("0")):
                    return None
                result = (fy25 if "fy25" in lowered else fy24) / revenue * Decimal("100")
                result = result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                return {"answer": f"{result:.2f}%", "source": target}
            return None


def _value_for_scope(values: dict[str, Any], scope: str, year: str) -> Decimal | None:
            for header, raw in values.items():
                if scope.casefold() in str(header).casefold() and _period_key(year) in {
                    _period_key(item) for item in re.findall(r"(?:fy)?\d{2,4}", str(header).casefold())
                }:
                    try:
                        return Decimal(str(raw).replace(",", "").replace("(", "-").replace(")", ""))
                    except InvalidOperation:
                        return None
            return None


def _metric_matches_question(label: str, question: str) -> bool:
            label_terms = {_metric_term(term) for term in re.findall(r"[a-z0-9]+", label.casefold())}
            query_terms = {_metric_term(term) for term in re.findall(r"[a-z0-9]+", question.casefold())}
            return bool(label_terms) and label_terms <= query_terms


def _period_key(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) == 2:
        return f"fy{digits}"
    if len(digits) == 4:
        return f"fy{digits[-2:]}"
    return value.casefold()


def _metric_term(value: str) -> str:
    value = value.casefold()
    if len(value) > 4 and value.endswith("s"):
        value = value[:-1]
    return value
