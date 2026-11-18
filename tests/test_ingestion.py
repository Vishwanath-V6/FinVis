import json

import pytest

from src.ingestion import fingerprint_bytes, select_page_numbers, validate_pdf_bytes


def test_fingerprint_is_content_based_not_filename_based():
    assert fingerprint_bytes(b"%PDF-a") == fingerprint_bytes(b"%PDF-a")
    assert fingerprint_bytes(b"%PDF-a") != fingerprint_bytes(b"%PDF-b")


def test_validate_pdf_bytes_rejects_bad_content_and_size():
    with pytest.raises(ValueError):
        validate_pdf_bytes(b"", 100)
    with pytest.raises(ValueError):
        validate_pdf_bytes(b"%PDF-x", 4)
    with pytest.raises(ValueError):
        validate_pdf_bytes(b"not-pdf", 100)


def test_select_page_numbers_uses_request_and_adjacent_context():
    scan = {
        "page_count": 4,
        "pages": [
            {"page_number": 1, "text": "contents"},
            {"page_number": 2, "text": "revenue table"},
            {"page_number": 3, "text": "notes"},
            {"page_number": 4, "text": "cash flow"},
        ],
    }
    assert select_page_numbers(scan, "revenue") == [1, 2, 3]


def test_cache_key_inputs_are_serializable():
    assert json.dumps({"fingerprint": fingerprint_bytes(b"%PDF")})
