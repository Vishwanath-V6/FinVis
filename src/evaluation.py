"""Reproducible retrieval and answer evaluation helpers."""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Iterable

from .retrieval import RetrievalIndex


def retrieval_metrics(index: RetrievalIndex, cases: Iterable[dict[str, Any]], k: int = 5) -> dict[str, float]:
    cases = list(cases)
    if not cases:
        raise ValueError("evaluation cases must not be empty")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    hits = reciprocal = precision = recall = 0.0
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("each case requires a non-empty question")
        expected = set(case.get("expected_chunk_ids", []))
        if not expected:
            raise ValueError("each retrieval case requires expected_chunk_ids")
        results = index.search(case["question"], top_k=k)
        found = [result.get("chunk_id") for result in results]
        relevant = [position for position, chunk_id in enumerate(found, 1) if chunk_id in expected]
        hits += bool(relevant)
        precision += len(relevant) / len(found) if found else 0
        recall += len(set(found) & expected) / len(expected)
        reciprocal += 1 / relevant[0] if relevant else 0
    n = len(cases)
    return {
        "hit_at_k": hits / n,
        "recall_at_k": recall / n,
        "precision_at_k": precision / n,
        "mrr": reciprocal / n,
    }


def evaluate_answers(
    cases: Iterable[dict[str, Any]], answer_fn: Callable[[str], dict[str, Any]]
) -> dict[str, float]:
    cases = list(cases)
    if not cases:
        raise ValueError("evaluation cases must not be empty")
    exact = grounded = abstention = 0
    latencies: list[float] = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("each case requires a non-empty question")
        started = time.perf_counter()
        result = answer_fn(case["question"])
        if not isinstance(result, dict):
            raise ValueError("answer_fn must return a dictionary")
        latencies.append(time.perf_counter() - started)
        expected = case.get("reference_answer")
        actual = result.get("answer")
        if expected and actual and _normalise(actual) == _normalise(expected):
            exact += 1
        if result.get("sufficient_evidence") and _answer_supported_by_evidence(result):
            grounded += 1
        if case.get("unanswerable") and not result.get("sufficient_evidence"):
            abstention += 1
    n = len(cases)
    unanswerable_cases = sum(bool(case.get("unanswerable")) for case in cases)
    return {
        "exact_match": exact / n,
        "grounded_rate": grounded / n,
        "unanswerable_abstention_rate": abstention / unanswerable_cases if unanswerable_cases else 0.0,
        "mean_latency_seconds": sum(latencies) / n,
    }


def _answer_supported_by_evidence(result: dict[str, Any]) -> bool:
    answer = result.get("answer")
    evidence = result.get("evidence", [])
    if not isinstance(answer, str) or not answer.strip() or not isinstance(evidence, list):
        return False
    answer_text = _normalise(answer)
    answer_numbers = _financial_numbers(answer)
    for item in evidence:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        normalized = _normalise(text)
        if answer_text in normalized and answer_numbers <= _financial_numbers(text):
            return True
    return False


def _normalise(value: str) -> str:
    return re.sub(r"\W+", " ", value.lower()).strip()


def _financial_numbers(value: str) -> set[str]:
    return {
        token.replace(",", "")
        for token in re.findall(r"(?<!\w)(?:[$€£]\s*)?-?\d[\d,]*(?:\.\d+)?%?", value)
    }
