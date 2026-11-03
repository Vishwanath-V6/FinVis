import json
import asyncio
from io import BytesIO

import pytest
import torch
import numpy as np
from starlette.datastructures import UploadFile

import api
from src.evaluation import evaluate_answers, retrieval_metrics
from src.inference import answer_question
from src.preprocessing import chunk_pages, clean_text
from src.preprocessing import _extract_page_tables
from src.preprocessing import PreprocessingError, extract_text
from src.retrieval import RetrievalIndex


class FakeEncoding(dict):
    def sequence_ids(self, _batch):
        return [None, 0, None, 1, 1, 1, None]


class FakeTokenizer:
    model_max_length = 128

    def __call__(self, *_args, **_kwargs):
        return FakeEncoding(input_ids=torch.tensor([[0, 1, 2, 10, 11, 12, 3]]))

    def decode(self, tokens, **_kwargs):
        return " ".join(str(int(token)) for token in tokens)


class FakeModel:
    def __call__(self, **_kwargs):
        start = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0, 10.0, 0.0]])
        end = torch.tensor([[0.0, 0.0, 0.0, 1.0, 4.0, -10.0, 0.0]])
        return type("Output", (), {"start_logits": start, "end_logits": end})()


class FakeIndex:
    def search(self, _question, top_k):
        return [{"text": "10 11 answer context", "chunk_id": 1, "source_filename": "x.pdf"}][:top_k]


def test_answer_uses_highest_valid_span_and_does_not_use_logit_sign():
    result = answer_question("q", FakeIndex(), FakeTokenizer(), FakeModel(), max_answer_length=2)
    assert result["answer"] == "10 11"
    assert result["sufficient_evidence"] is True
    assert result["span_score"] == 9


def test_abstention_is_explicit_policy():
    result = answer_question(
        "q",
        FakeIndex(),
        FakeTokenizer(),
        FakeModel(),
        abstention_policy=lambda candidate: candidate["span_score"] < 100,
    )
    assert result["answer"] is None
    assert result["sufficient_evidence"] is False


def test_answer_handles_retrieved_context_without_context_tokens():
    class NoContextEncoding(FakeEncoding):
        def sequence_ids(self, _batch):
            return [None, 0, None]

    class NoContextTokenizer(FakeTokenizer):
        def __call__(self, *_args, **_kwargs):
            return NoContextEncoding(input_ids=torch.tensor([[0, 1, 2]]))

    result = answer_question("q", FakeIndex(), NoContextTokenizer(), FakeModel())
    assert result["answer"] is None
    assert result["sources"] == []


def test_chunk_boundaries_and_metadata():
    pages = [{"document_id": "doc", "source_filename": "x.pdf", "page_number": 2, "text": "a b c d e"}]
    chunks = chunk_pages(pages, chunk_size=3, overlap=1)
    assert [chunk["text"] for chunk in chunks] == ["a b c", "c d e"]
    assert all(chunk["page_number"] == 2 and chunk["document_id"] == "doc" for chunk in chunks)


def test_cleaning_does_not_join_numeric_hyphenated_values():
    assert clean_text("Value 1-\n2 and year fi-\nnance") == "Value 1- 2 and year finance"
    assert clean_text("Loss -1,234.50; margin 12.5%; date 2024-12-31") == (
        "Loss -1,234.50; margin 12.5%; date 2024-12-31"
    )


def test_table_rows_keep_headers_values_and_page_metadata():
    class Table:
        def extract(self):
            return [["Period", "Revenue"], ["FY24", "$1,234"]]

    class Page:
        def find_tables(self):
            return type("Tables", (), {"tables": [Table()]})()

    tables = _extract_page_tables(Page(), "doc", "report.pdf", 4)
    assert tables[0]["headers"] == ["Period", "Revenue"]
    assert tables[0]["rows"] == [["FY24", "$1,234"]]
    chunks = chunk_pages(
        [{"document_id": "doc", "source_filename": "report.pdf", "page_number": 4,
          "text": "", "tables": tables}]
    )
    assert chunks[0]["content_type"] == "table"
    assert "Revenue = $1,234" in chunks[0]["text"]
    assert chunks[0]["table_values"]["Revenue"] == "$1,234"


def test_structured_table_answer_associates_metric_and_year():
    class Index:
        def search(self, _question, top_k):
            return [
                {
                    "content_type": "table",
                    "table_row_label": "Revenue",
                    "table_values": {"Particulars": "Revenue", "FY2025": "2,450.00", "FY2024": "2,100.00"},
                    "table_unit": "INR lakh",
                    "text": "Financial table row. Metric: Revenue. FY2025 = 2,450.00 FY2024 = 2,100.00",
                    "document_id": "doc", "source_filename": "financial.pdf", "page_number": 3,
                    "chunk_id": 7, "score": 0.9,
                }
            ][:top_k]

    result = answer_question("Revenue operations in 2025", Index(), FakeTokenizer(), FakeModel())
    assert result["answer"] == "2,450.00 INR lakh"
    assert result["answer_source"] == "structured_table"


def test_structured_table_answer_abstains_for_missing_period_value():
    class Index:
        def search(self, _question, top_k):
            return [{
                "content_type": "table", "table_row_label": "Gross profit",
                "table_values": {"Particulars": "Gross profit", "FY2024": "900.00"},
                "table_unit": "INR lakh", "text": "Gross profit FY2024 = 900.00",
                "document_id": "doc", "source_filename": "financial.pdf", "page_number": 3,
                "chunk_id": 8, "score": 0.9,
            }][:top_k]

    result = answer_question("Gross profit in 2025", Index(), FakeTokenizer(), FakeModel())
    assert result["answer"] is None
    assert result["sufficient_evidence"] is False


def test_table_chunks_are_row_specific_and_preserve_period_value_unit():
    pages = [{
        "document_id": "doc", "source_filename": "financial.pdf", "page_number": 2, "text": "",
        "tables": [{
            "table_id": "t1",
            "headers": ["", "FY2025", "FY2024"],
            "rows": [["Revenue", "2,450.00", "2,100.00"], ["Gross profit", "980.00", "900.00"]],
            "unit": "INR lakh",
        }],
    }]
    chunks = chunk_pages(pages)
    assert len(chunks) == 2
    revenue = next(chunk for chunk in chunks if chunk["table_row_label"] == "Revenue")
    assert revenue["table_values"]["FY2025"] == "2,450.00"
    assert "Unit: INR lakh" in revenue["text"]
    assert "Source page: 2" in revenue["text"]


def test_ocr_is_used_for_short_text_pages(monkeypatch, tmp_path):
    class Page:
        def get_text(self, *_args, **_kwargs):
            return " "

        def get_textpage_ocr(self, **_kwargs):
            return object()

    class Document:
        is_encrypted = False

        def __init__(self, page):
            self.page = page

        def __len__(self):
            return 1

        def __iter__(self):
            return iter([self.page])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class PyMuPdf:
        @staticmethod
        def open(_path):
            return Document(Page())

    import src.preprocessing as preprocessing
    monkeypatch.setattr(preprocessing, "pymupdf", PyMuPdf)
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF")
    page = Page()
    monkeypatch.setattr(page, "get_text", lambda *_args, **kwargs: "OCR revenue $1,234" if kwargs else " ")
    document = Document(page)
    monkeypatch.setattr(PyMuPdf, "open", lambda _path: document)
    pages = preprocessing.extract_text(path, enable_ocr=True)
    assert pages[0]["ocr_used"] is True
    assert "OCR revenue" in pages[0]["text"]


def test_embedding_rejects_boolean_batch_size():
    from src.model_utils import encode_texts

    with pytest.raises(ValueError):
        encode_texts(["text"], batch_size=True)


def test_metrics_have_distinct_hit_recall_and_safe_denominators():
    class Index:
        def search(self, _question, top_k):
            return [{"chunk_id": 1}]

    metrics = retrieval_metrics(
        Index(),
        [{"question": "q", "expected_chunk_ids": [1, 2]}],
        k=5,
    )
    assert metrics["hit_at_k"] == 1
    assert metrics["recall_at_k"] == 0.5
    assert metrics["precision_at_k"] == 1


def test_abstention_rate_uses_unanswerable_cases_only():
    results = evaluate_answers(
        [
            {"question": "answerable", "unanswerable": False, "reference_answer": "yes"},
            {"question": "unknown", "unanswerable": True},
        ],
        lambda question: {
            "answer": "yes" if question == "answerable" else None,
            "sufficient_evidence": question == "answerable",
            "sources": [{}] if question == "answerable" else [],
        },
    )
    assert results["exact_match"] == 0.5
    assert results["unanswerable_abstention_rate"] == 1


def test_retrieval_rejects_invalid_chunk_metadata():
    with pytest.raises(ValueError, match="document_id"):
        RetrievalIndex().build([{"text": "x", "chunk_id": 1}])


def test_retrieval_preserves_vector_row_to_chunk_mapping(monkeypatch, tmp_path):
    vectors = {
        "first": np.array([[1.0, 0.0]], dtype=np.float32),
        "second": np.array([[0.0, 1.0]], dtype=np.float32),
    }

    def fake_encode(texts, _model_name):
        return np.vstack([vectors[text] for text in texts])

    monkeypatch.setattr("src.retrieval.encode_texts", fake_encode)
    chunks = [
        {"text": "first", "chunk_id": 10, "document_id": "d", "source_filename": "a.pdf", "page_number": 1},
        {"text": "second", "chunk_id": 20, "document_id": "d", "source_filename": "a.pdf", "page_number": 2},
    ]
    index = RetrievalIndex(model_name="test").build(chunks)
    assert index.search("second", top_k=1)[0]["chunk_id"] == 20
    index.save(tmp_path)
    loaded = RetrievalIndex.load(tmp_path)
    assert [chunk["chunk_id"] for chunk in loaded.chunks] == [10, 20]


def test_malformed_pdf_is_rejected(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(PreprocessingError):
        extract_text(path, enable_ocr=False)


def test_saved_index_rejects_row_mapping_mismatch(tmp_path, monkeypatch):
    index = RetrievalIndex()
    index.index = type("FakeFaiss", (), {"ntotal": 2, "d": 3, "is_trained": True})()
    index.chunks = []
    metadata = tmp_path / "chunks.json"
    metadata.write_text(json.dumps({"model_name": "m", "chunks": []}), encoding="utf-8")
    (tmp_path / "index.faiss").write_bytes(b"not an index")
    monkeypatch.setattr("src.retrieval.faiss.read_index", lambda _path: index.index)
    with pytest.raises(ValueError, match="row count"):
        RetrievalIndex.load(tmp_path)


def test_api_sanitizes_filename_and_replaces_index_only_after_success(monkeypatch):
    class FakeIndex:
        def build(self, chunks):
            return self

    monkeypatch.setattr(api, "RetrievalIndex", FakeIndex)
    monkeypatch.setattr(
        api,
        "extract_text",
        lambda _path, document_id: [
            {"document_id": document_id, "source_filename": "upload.pdf", "page_number": 1, "text": "x"}
        ],
    )
    monkeypatch.setattr(api, "clean_pages", lambda pages: pages)
    monkeypatch.setattr(api, "chunk_pages", lambda pages: [{"document_id": "d", "source_filename": "x.pdf",
                                                              "page_number": 1, "chunk_id": 1, "text": "x"}])
    upload = UploadFile(
        filename="../../outside.pdf",
        file=BytesIO(b"%PDF-1.7\ncontent"),
        headers={"content-type": "application/pdf"},
    )
    response = asyncio.run(api.index_document(upload))
    assert response["document"] == "outside.pdf"


def test_api_rejects_non_pdf_and_reads_only_bounded_size(monkeypatch):
    invalid = UploadFile(
        filename="document.pdf",
        file=BytesIO(b"not a pdf"),
        headers={"content-type": "application/pdf"},
    )
    with pytest.raises(api.HTTPException) as invalid_error:
        asyncio.run(api.index_document(invalid))
    assert invalid_error.value.status_code == 415

    monkeypatch.setattr(api, "MAX_UPLOAD_SIZE", 4)
    oversized = UploadFile(
        filename="document.pdf",
        file=BytesIO(b"%PDF-1.7"),
        headers={"content-type": "application/pdf"},
    )
    with pytest.raises(api.HTTPException) as size_error:
        asyncio.run(api.index_document(oversized))
    assert size_error.value.status_code == 413
