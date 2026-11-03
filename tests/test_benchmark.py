import json

import pytest

from scripts.run_benchmark import (
    _answer_matches,
    _retrieval_metrics,
    _validate_payload,
    run,
)


def test_benchmark_numeric_answer_matching_is_format_tolerant():
    assert _answer_matches("₹439,695 crore", "₹439,695 crore")
    assert _answer_matches("439695 crore", "₹439,695 crore")
    assert not _answer_matches("₹434,016 crore", "₹439,695 crore")


def test_benchmark_rejects_malformed_input(tmp_path):
    pdf = tmp_path / "missing.pdf"
    payload = tmp_path / "input.json"
    payload.write_text(json.dumps({"questions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="questions"):
        run(pdf, payload, tmp_path / "out")


def test_benchmark_rejects_duplicate_question_ids():
    cases = [
        {"question_id": "Q1", "question": "first", "answerable": True},
        {"question_id": "Q1", "question": "second", "answerable": False},
    ]
    with pytest.raises(ValueError, match="unique"):
        _validate_payload({"questions": cases})


def test_benchmark_validates_declared_counts():
    cases = [{"question_id": "Q1", "question": "first", "answerable": True}]
    with pytest.raises(ValueError, match="dataset_counts"):
        _validate_payload(
            {
                "questions": cases,
                "dataset_counts": {"total": 2, "answerable": 1, "unanswerable": 1},
            }
        )


def _valid_cases():
    return [
        {
            "question_id": f"Q{i}",
            "question": f"question {i}",
            "answerable": i < 104,
            "mapped_chunk_ids": [i] if i < 104 else [],
        }
        for i in range(128)
    ]


def test_benchmark_requires_exact_dataset_split():
    cases = _valid_cases()[:-1]
    with pytest.raises(ValueError, match="exactly"):
        _validate_payload({"questions": cases})


def test_retrieval_metrics_use_answerable_denominator_and_reciprocal_rank():
    class Retriever:
        def search(self, question, top_k):
            number = int(question.split()[-1])
            return [{"chunk_id": number}] if number == 0 else [{"chunk_id": -1}]

    metrics, retrieved = _retrieval_metrics(Retriever(), _valid_cases(), 5)
    assert metrics["denominator"] == 104
    assert metrics["hit_at_5"] == {"count": 1, "percent": pytest.approx(100 / 104)}
    assert metrics["mrr"] == pytest.approx(1 / 104)
    assert len(retrieved) == 104


def test_retrieval_metrics_reject_invalid_k():
    with pytest.raises(ValueError, match="positive integer"):
        _retrieval_metrics(object(), _valid_cases(), 0)
