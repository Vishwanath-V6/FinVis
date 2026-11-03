"""Run the Tata Motors benchmark through FinVis and a TF-IDF retrieval baseline."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import answer_question, load_qa_model
from src.preprocessing import chunk_pages, clean_pages, extract_text
from src.retrieval import RetrievalIndex


TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"(?<!\w)-?\d[\d,]*(?:\.\d+)?")
EXPECTED_ANSWERABLE = 104
EXPECTED_UNANSWERABLE = 24


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def _numbers(text: str) -> set[str]:
    normalized = re.sub(r"\((\d[\d,]*(?:\.\d+)?)\)", r"-\1", text)
    return {match.replace(",", "") for match in NUMBER_RE.findall(normalized)}


def _answer_matches(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    expected_numbers = _numbers(expected)
    actual_numbers = _numbers(actual)
    return bool(expected_numbers) and expected_numbers <= actual_numbers


class TfidfBaseline:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        vocabulary = sorted({token for chunk in chunks for token in _tokens(chunk["text"])})
        self.vocabulary = {token: i for i, token in enumerate(vocabulary)}
        matrix = np.zeros((len(chunks), len(vocabulary)), dtype=np.float32)
        document_frequency = np.zeros(len(vocabulary), dtype=np.float32)
        for row, chunk in enumerate(chunks):
            counts: dict[str, int] = {}
            for token in _tokens(chunk["text"]):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                column = self.vocabulary[token]
                matrix[row, column] = count
                document_frequency[column] += 1
        if len(chunks):
            matrix *= (np.log((len(chunks) + 1) / (document_frequency + 1)) + 1)
            norms = np.linalg.norm(matrix, axis=1)
            matrix[norms > 0] /= norms[norms > 0, None]
        self.matrix = matrix

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        for token in _tokens(query):
            if token in self.vocabulary:
                query_vector[self.vocabulary[token]] += 1
        norm = np.linalg.norm(query_vector)
        if norm == 0:
            return []
        query_vector /= norm
        scores = self.matrix @ query_vector
        positions = np.argsort(-scores, kind="stable")[:top_k]
        output = []
        for position in positions:
            result = dict(self.chunks[int(position)])
            result["score"] = float(scores[int(position)])
            result["index_position"] = int(position)
            output.append(result)
        return output


def _map_relevant_chunks(cases: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    for case in cases:
        if not case.get("answerable"):
            case["mapped_chunk_ids"] = []
            continue
        evidence = str(case.get("evidence_row", ""))
        evidence_numbers = _numbers(evidence)
        evidence_lower = evidence.casefold()
        evidence_terms = set(_tokens(evidence_lower))
        matches = []
        for chunk in chunks:
            if chunk.get("content_type") != "table":
                continue
            row_tokens = set(_tokens(str(chunk.get("table_row_label", ""))))
            row_label = str(chunk.get("table_row_label", "")).casefold()
            row_terms = set(_tokens(row_label))
            label_match = row_label in evidence_lower or (
                row_terms and len(row_terms & evidence_terms) / len(row_terms) >= 0.8
            )
            if not label_match:
                continue
            if evidence_numbers <= _numbers(str(chunk.get("text", ""))):
                matches.append(chunk["chunk_id"])
            elif evidence_numbers & _numbers(str(chunk.get("text", ""))):
                matches.append(chunk["chunk_id"])
        if not matches:
            raise ValueError(f"could not map evidence to a chunk: {case['question_id']}")
        case["mapped_chunk_ids"] = matches


def _printed_page_map(pages: list[dict[str, Any]], printed_page: int) -> set[int]:
    return {
        int(page["page_number"])
        for page in pages
        if "FINANCIAL HIGHLIGHTS" in page.get("text", "").upper()
        and re.search(rf"(?m)^\s*{printed_page}\s*$", page.get("text", ""))
    }


def _benchmark_printed_page(cases: list[dict[str, Any]]) -> int:
    printed_pages = {
        case.get("printed_page")
        for case in cases
        if case.get("answerable") and isinstance(case.get("printed_page"), int)
    }
    if not printed_pages:
        raise ValueError("answerable benchmark questions must contain printed_page")
    if len(printed_pages) != 1:
        raise ValueError("benchmark questions must use one consistent printed_page")
    return printed_pages.pop()


def _validate_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases = payload.get("questions")
    if not isinstance(cases, list) or not cases:
        raise ValueError("benchmark input must contain a non-empty questions list")
    question_ids = [case.get("question_id") for case in cases]
    if any(not isinstance(question_id, str) or not question_id for question_id in question_ids):
        raise ValueError("every benchmark question must have a non-empty question_id")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("benchmark question_id values must be unique")
    if any(not isinstance(case.get("question"), str) or not case["question"].strip() for case in cases):
        raise ValueError("every benchmark question must have non-empty question text")
    if any(not isinstance(case.get("answerable"), bool) for case in cases):
        raise ValueError("every benchmark question must have a boolean answerable field")
    declared = payload.get("dataset_counts")
    actual = {
        "total": len(cases),
        "answerable": sum(bool(case.get("answerable")) for case in cases),
        "unanswerable": sum(not bool(case.get("answerable")) for case in cases),
    }
    if declared is not None and declared != actual:
        raise ValueError(f"dataset_counts {declared!r} do not match actual counts {actual!r}")
    if actual != {
        "total": EXPECTED_ANSWERABLE + EXPECTED_UNANSWERABLE,
        "answerable": EXPECTED_ANSWERABLE,
        "unanswerable": EXPECTED_UNANSWERABLE,
    }:
        raise ValueError(
            "benchmark must contain exactly "
            f"{EXPECTED_ANSWERABLE} answerable and {EXPECTED_UNANSWERABLE} unanswerable questions"
        )
    return cases


def _retrieval_metrics(
    retriever: Any, cases: list[dict[str, Any]], k: int
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    answerable = [case for case in cases if case.get("answerable")]
    hits = 0
    reciprocal = 0.0
    retrieved: dict[str, list[dict[str, Any]]] = {}
    for case in answerable:
        results = retriever.search(case["question"], top_k=k)
        retrieved[case["question_id"]] = results
        relevant = [
            rank for rank, result in enumerate(results, 1)
            if result.get("chunk_id") in case["mapped_chunk_ids"]
        ]
        if relevant:
            hits += 1
            reciprocal += 1 / relevant[0]
    denominator = len(answerable)
    return {
        "denominator": denominator,
        "hit_at_5": {"count": hits, "percent": 100 * hits / denominator if denominator else 0},
        "mrr": reciprocal / denominator if denominator else 0,
    }, retrieved


def _evaluate_system(
    name: str,
    retriever: Any,
    cases: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    printed_pages: set[int],
    top_k: int = 5,
) -> dict[str, Any]:
    retrieval, retrieved = _retrieval_metrics(retriever, cases, top_k)
    details = []
    cpu_latencies_ms = []
    for case in cases:
        started = time.process_time()
        result = answer_question(
            case["question"],
            retriever,
            tokenizer,
            model,
            top_k=top_k,
            abstention_policy=lambda candidate: candidate.get("answer_source") != "structured_table",
        )
        cpu_latencies_ms.append((time.process_time() - started) * 1000)
        answerable = bool(case.get("answerable"))
        answered = isinstance(result.get("answer"), str) and bool(result["answer"].strip())
        answer_correct = _answer_matches(result.get("answer"), case.get("reference_answer")) if answerable else not answered
        citation = False
        for source in result.get("sources", []):
            citation = (
                source.get("source_filename") == Path(retriever.chunks[0]["source_filename"]).name
                and source.get("page_number") in printed_pages
            )
            if citation:
                break
        abstention_correct = (not answered) if not answerable else answered
        errors = []
        if answerable and case["mapped_chunk_ids"] and not any(
            item.get("chunk_id") in case["mapped_chunk_ids"] for item in retrieved.get(case["question_id"], [])
        ):
            errors.append("retrieval_miss")
        if answerable and not answer_correct:
            errors.append("incorrect_answer")
        if answerable and answered and not citation:
            errors.append("wrong_citation")
        if not abstention_correct:
            errors.append("abstention_failure")
        details.append(
            {
                "question_id": case["question_id"],
                "answer": result.get("answer"),
                "reference_answer": case.get("reference_answer"),
                "answer_correct": answer_correct,
                "citation_correct": citation,
                "abstention_correct": abstention_correct,
                "returned_chunk_ids": [item.get("chunk_id") for item in result.get("evidence", [])],
                "errors": errors,
            }
        )
    answerable_details = [item for item, case in zip(details, cases) if case.get("answerable")]
    unanswerable_details = [item for item, case in zip(details, cases) if not case.get("answerable")]
    citation_count = sum(item["citation_correct"] for item in answerable_details)
    correct_abstentions = sum(item["abstention_correct"] for item in unanswerable_details)
    return {
        "retrieval": retrieval,
        "answer_correctness": {
            "count": sum(item["answer_correct"] for item in answerable_details),
            "denominator": len(answerable_details),
            "percent": 100 * sum(item["answer_correct"] for item in answerable_details) / len(answerable_details),
        },
        "citation_accuracy": {
            "count": citation_count,
            "denominator": len(answerable_details),
            "percent": 100 * citation_count / len(answerable_details),
        },
        "abstention_accuracy": {
            "count": correct_abstentions,
            "denominator": len(unanswerable_details),
            "percent": 100 * correct_abstentions / len(unanswerable_details),
        },
        "median_cpu_inference_latency_ms": median(cpu_latencies_ms),
        "error_counts": {
            error: sum(error in item["errors"] for item in details)
            for error in ("retrieval_miss", "incorrect_answer", "wrong_citation", "abstention_failure")
        },
        "details": details,
    }


def run(pdf_path: Path, input_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cases = _validate_payload(payload)
    if not pdf_path.exists():
        raise FileNotFoundError(f"benchmark PDF does not exist: {pdf_path}")
    pages = extract_text(pdf_path, document_id=pdf_path.stem, enable_ocr=False)
    chunks = chunk_pages(clean_pages(pages))
    _map_relevant_chunks(cases, chunks)
    printed_pages = _printed_page_map(pages, _benchmark_printed_page(cases))
    if not printed_pages:
        raise ValueError("could not locate the benchmark printed page in extracted PDF text")

    mini_index = RetrievalIndex().build(chunks)
    tfidf = TfidfBaseline(chunks)
    tokenizer, model = load_qa_model()
    systems = {
        "minilm": _evaluate_system("minilm", mini_index, cases, tokenizer, model, printed_pages),
        "tfidf": _evaluate_system("tfidf", tfidf, cases, tokenizer, model, printed_pages),
    }
    result = {
        "dataset_name": payload.get("dataset_name"),
        "pdf": str(pdf_path),
        "question_count": len(cases),
        "answerable_count": sum(bool(case.get("answerable")) for case in cases),
        "unanswerable_count": sum(not bool(case.get("answerable")) for case in cases),
        "chunk_count": len(chunks),
        "table_chunk_count": sum(chunk.get("content_type") == "table" for chunk in chunks),
        "printed_page_physical_pages": sorted(printed_pages),
        "ground_truth_chunk_mapping": [
            {
                "question_id": case["question_id"],
                "answerable": bool(case.get("answerable")),
                "relevant_chunk_ids": case["mapped_chunk_ids"],
            }
            for case in cases
        ],
        "systems": systems,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    updated_payload = json.loads(json.dumps(payload))
    for case in updated_payload["questions"]:
        mapped = next(
            item["relevant_chunk_ids"]
            for item in result["ground_truth_chunk_mapping"]
            if item["question_id"] == case["question_id"]
        )
        case["relevant_chunk_ids"] = mapped
    (output_dir / "benchmark_input_updated.json").write_text(
        json.dumps(updated_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(output_dir / "benchmark_report.md", result)
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Tata Motors FY25 FinVis Benchmark",
        "",
        f"- Questions: {result['question_count']} ({result['answerable_count']} answerable, {result['unanswerable_count']} unanswerable)",
        f"- Chunks: {result['chunk_count']} ({result['table_chunk_count']} table chunks)",
        "",
        "| System | Hit@5 | MRR | Answer correctness | Citation accuracy | Abstention accuracy | Median CPU ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result["systems"].items():
        retrieval = metrics["retrieval"]
        lines.append(
            f"| {name} | {retrieval['hit_at_5']['count']}/{retrieval['denominator']} "
            f"({retrieval['hit_at_5']['percent']:.2f}%) | {retrieval['mrr']:.4f} | "
            f"{metrics['answer_correctness']['count']}/{metrics['answer_correctness']['denominator']} "
            f"({metrics['answer_correctness']['percent']:.2f}%) | "
            f"{metrics['citation_accuracy']['count']}/{metrics['citation_accuracy']['denominator']} "
            f"({metrics['citation_accuracy']['percent']:.2f}%) | "
            f"{metrics['abstention_accuracy']['count']}/{metrics['abstention_accuracy']['denominator']} "
            f"({metrics['abstention_accuracy']['percent']:.2f}%) | "
            f"{metrics['median_cpu_inference_latency_ms']:.2f} |"
        )
    lines.extend(["", "## Error analysis", ""])
    for name, metrics in result["systems"].items():
        lines.append(f"### {name}")
        lines.append("")
        for error, count in metrics["error_counts"].items():
            lines.append(f"- {error}: {count}")
        lines.append("")
        for item in metrics["details"]:
            if item["errors"]:
                lines.append(
                    f"- `{item['question_id']}`: {', '.join(item['errors'])}; "
                    f"returned answer: {item['answer']!r}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_outputs")
    args = parser.parse_args()
    run(args.pdf, args.input, args.output_dir)


if __name__ == "__main__":
    main()
