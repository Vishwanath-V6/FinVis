"""PDF ingestion, cleaning, and metadata-preserving chunking."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pymupdf


class PreprocessingError(ValueError):
    """Raised when a document cannot be safely prepared for retrieval."""


def extract_text(
    pdf_path: str | Path,
    document_id: str | None = None,
    *,
    enable_ocr: bool = True,
    ocr_language: str = "eng",
    ocr_min_chars: int = 20,
) -> list[dict[str, Any]]:
    """Extract text page by page from a readable PDF.

    Returned page records contain one-based page numbers and the stable document ID.
    Empty pages are retained so page numbering remains faithful to the source.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise PreprocessingError(f"PDF path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise PreprocessingError(f"Expected a PDF file, got: {path.name}")

    if not isinstance(enable_ocr, bool):
        raise TypeError("enable_ocr must be a boolean")
    if not isinstance(ocr_language, str) or not ocr_language.strip():
        raise ValueError("ocr_language must be a non-empty string")
    if not isinstance(ocr_min_chars, int) or isinstance(ocr_min_chars, bool) or ocr_min_chars < 0:
        raise ValueError("ocr_min_chars must be a non-negative integer")
    doc_id = document_id or path.stem
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise PreprocessingError("document_id must be a non-empty string")
    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise PreprocessingError(f"Encrypted PDF is not supported: {path.name}")
            if len(document) == 0:
                raise PreprocessingError(f"PDF contains no pages: {path.name}")
            pages: list[dict[str, Any]] = []
            for page_index, page in enumerate(document):
                text = page.get_text("text") or ""
                ocr_used = False
                if enable_ocr and len(text.strip()) < ocr_min_chars:
                    try:
                        textpage = page.get_textpage_ocr(language=ocr_language, dpi=300, full=True)
                        ocr_text = page.get_text("text", textpage=textpage) or ""
                    except Exception as exc:
                        if not text.strip():
                            raise PreprocessingError(
                                f"OCR failed on page {page_index + 1} of {path.name}: {exc}"
                            ) from exc
                        ocr_text = ""
                    if ocr_text.strip():
                        text = ocr_text
                        ocr_used = True
                tables = _extract_page_tables(page, doc_id, path.name, page_index + 1, text)
                pages.append(
                    {
                        "document_id": doc_id,
                        "source_filename": path.name,
                        "page_number": page_index + 1,
                        "text": text,
                        "character_count": len(text),
                        "word_count": len(text.split()),
                        "ocr_used": ocr_used,
                        "tables": tables,
                    }
                )
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"Unable to read PDF {path.name}: {exc}") from exc
    if not any(page["text"].strip() or page["tables"] for page in pages):
        raise PreprocessingError(
            f"PDF has no extractable text: {path.name}. OCR is not enabled; "
            "provide a text-based PDF or install/configure an OCR pipeline."
        )
    return pages


def _extract_page_tables(
    page: Any, document_id: str, source_filename: str, page_number: int, page_text: str = ""
) -> list[dict[str, Any]]:
    """Extract tables into structured rows while retaining cell boundaries."""
    finder = getattr(page, "find_tables", None)
    if finder is None:
        return []
    try:
        found = finder()
        raw_tables = getattr(found, "tables", found or [])
    except Exception:
        return []
    tables: list[dict[str, Any]] = []
    for table_number, table in enumerate(raw_tables):
        try:
            rows = table.extract()
        except Exception:
            continue
        if not rows or not all(isinstance(row, (list, tuple)) for row in rows):
            continue
        normalized_rows = [
            [
                re.sub(r"\bafet\s+r\b", "after", cell.strip(), flags=re.IGNORECASE)
                if isinstance(cell, str)
                else ""
                for cell in row
            ]
            for row in rows
        ]
        if not any(cell for row in normalized_rows for cell in row):
            continue
        headers, data_rows, inferred = _repair_financial_table_headers(
            normalized_rows[0], normalized_rows[1:], page_text
        )
        unit = _infer_financial_unit(page_text)
        tables.append(
            {
                "document_id": document_id,
                "source_filename": source_filename,
                "page_number": page_number,
                "table_id": f"{document_id}:{page_number}:{table_number}",
                "headers": headers,
                "rows": data_rows,
                "unit": unit,
                "header_inferred": inferred,
            }
        )
    return tables


def _repair_financial_table_headers(
    headers: list[str], rows: list[list[str]], page_text: str
) -> tuple[list[str], list[list[str]], bool]:
    if not headers or not rows:
        return headers, rows, False
    numeric_header_cells = sum(bool(re.fullmatch(r"[\d,().-]+", cell)) for cell in headers[1:])
    if (
        len(headers) == 5
        and numeric_header_cells >= 3
        and re.search(r"\bStandalone\b", page_text, re.IGNORECASE)
        and re.search(r"\bConsolidated\b", page_text, re.IGNORECASE)
        and re.search(r"\bFY25\b", page_text)
        and re.search(r"\bFY24\b", page_text)
    ):
        return [
            "Particulars",
            "Standalone FY25",
            "Standalone FY24",
            "Consolidated FY25",
            "Consolidated FY24",
        ], [headers, *rows], True
    return headers, rows, False


def _infer_financial_unit(page_text: str) -> str:
    match = re.search(
        r"\((₹|Rs\.?)\s*in\s+(crore|crores|lakh|lakhs|million|billion)\)",
        page_text,
        re.IGNORECASE,
    )
    return f"{match.group(1)} {match.group(2)}" if match else ""


def clean_text(text: str) -> str:
    """Normalize PDF artifacts conservatively without changing financial values."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def clean_pages(pages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean page records while preserving document and page metadata."""
    cleaned: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            raise TypeError("each page must be a dictionary")
        if "page_number" not in page or "text" not in page:
            raise PreprocessingError("each page requires page_number and text")
        if (
            not isinstance(page["page_number"], int)
            or isinstance(page["page_number"], bool)
            or page["page_number"] < 1
        ):
            raise PreprocessingError("page_number must be a positive integer")
        if not isinstance(page.get("document_id"), str) or not page["document_id"].strip():
            raise PreprocessingError("each page requires a document_id")
        if not isinstance(page.get("source_filename"), str) or not page["source_filename"].strip():
            raise PreprocessingError("each page requires a source_filename")
        if not isinstance(page.get("tables", []), list):
            raise PreprocessingError("page tables must be a list")
        text = clean_text(page["text"])
        record = dict(page)
        record.update(
            text=text,
            character_count=len(text),
            word_count=len(text.split()),
        )
        cleaned.append(record)
    return cleaned


def chunk_pages(
    pages: Iterable[dict[str, Any]], chunk_size: int = 350, overlap: int = 40
) -> list[dict[str, Any]]:
    """Create word-based chunks with a stable metadata contract.

    Each chunk contains ``document_id``, ``source_filename``, ``page_number``,
    ``chunk_id``, ``text``, ``words`` (legacy alias), ``word_count``,
    ``character_count``, and ``preview``.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[dict[str, Any]] = []
    next_chunk_id = 0
    for page in pages:
        if not isinstance(page, dict):
            raise TypeError("each page must be a dictionary")
        raw_text = page.get("text")
        if not isinstance(raw_text, str):
            raise PreprocessingError("page text must be a string")
        if not isinstance(page.get("page_number"), int) or page["page_number"] < 1:
            raise PreprocessingError("page_number must be a positive integer")
        if not isinstance(page.get("document_id"), str) or not page["document_id"].strip():
            raise PreprocessingError("each page requires a document_id")
        if not isinstance(page.get("source_filename"), str) or not page["source_filename"].strip():
            raise PreprocessingError("each page requires a source_filename")
        words = raw_text.split()
        if words:
            for start in range(0, len(words), chunk_size - overlap):
                chunk_words = words[start : start + chunk_size]
                if not chunk_words:
                    continue
                chunk_text = " ".join(chunk_words)
                chunks.append(
                    {
                        "document_id": page.get("document_id", "document"),
                        "source_filename": page.get("source_filename", ""),
                        "page_number": page["page_number"],
                        "chunk_id": next_chunk_id,
                        "text": chunk_text,
                        "words": chunk_text,
                        "word_count": len(chunk_words),
                        "character_count": len(chunk_text),
                        "preview": chunk_text[:300],
                        "content_type": "text",
                    }
                )
                next_chunk_id += 1
                if start + chunk_size >= len(words):
                    break
        for table in page.get("tables", []):
            if not isinstance(table, dict):
                raise PreprocessingError("each table must be a dictionary")
            table_id = table.get("table_id")
            headers = table.get("headers")
            rows = table.get("rows")
            if not isinstance(table_id, str) or not table_id.strip():
                raise PreprocessingError("each table requires a table_id")
            if not isinstance(headers, list) or not isinstance(rows, list):
                raise PreprocessingError("table headers and rows must be lists")
            table_metadata = dict(table)
            table_metadata["page_number"] = page["page_number"]
            for table_row in _table_rows(headers, rows, table_metadata):
                row_text = table_row["search_text"]
                chunks.append(
                    {
                        "document_id": page["document_id"],
                        "source_filename": page["source_filename"],
                        "page_number": page["page_number"],
                        "chunk_id": next_chunk_id,
                        "table_id": table_id,
                        "text": row_text,
                        "words": row_text,
                        "word_count": len(row_text.split()),
                        "character_count": len(row_text),
                        "preview": row_text[:300],
                        "content_type": "table",
                        "table_row_label": table_row["row_label"],
                        "table_headers": table_row["headers"],
                        "table_values": table_row["values"],
                        "table_unit": table.get("unit", ""),
                        "table_reporting_period": table.get("reporting_period", ""),
                    }
                )
                next_chunk_id += 1
    return chunks


def _table_to_search_text(headers: list[Any], rows: list[Any]) -> str:
    if not headers or not all(isinstance(cell, str) for cell in headers):
        return ""
    lines = ["Table headers: " + " | ".join(headers)]
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, list) or len(row) != len(headers):
            continue
        cells = [cell if isinstance(cell, str) else "" for cell in row]
        if any(cell.strip() for cell in cells):
            lines.append(
                f"Row {row_number}: "
                + " | ".join(f"{header} = {cell}" for header, cell in zip(headers, cells))
            )
    return "\n".join(lines) if len(lines) > 1 else ""


def _table_rows(headers: list[Any], rows: list[Any], table: dict[str, Any]) -> list[dict[str, Any]]:
    """Create one searchable, self-contained chunk for each complete table row."""
    clean_headers = [cell.strip() if isinstance(cell, str) else "" for cell in headers]
    if not clean_headers:
        return []
    if not clean_headers[0]:
        clean_headers[0] = "Particulars"
    unit = str(table.get("unit", "")).strip()
    if not unit:
        unit_match = re.search(
            r"\b(?:INR|USD|EUR|GBP)\s+(?:lakh|lakhs|million|billion|crore|crores)\b",
            " ".join(clean_headers),
            flags=re.IGNORECASE,
        )
        unit = unit_match.group(0) if unit_match else ""
    reporting_period = str(table.get("reporting_period", "")).strip()
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(clean_headers):
            continue
        values = [cell.strip() if isinstance(cell, str) else "" for cell in row]
        row_label = values[0]
        if not row_label or not any(values[1:]):
            continue
        pairs = [
            f"{header} = {value}"
            for header, value in zip(clean_headers[1:], values[1:])
            if header and value
        ]
        if not pairs:
            continue
        period_text = f" Reporting period: {reporting_period}." if reporting_period else ""
        unit_text = f" Unit: {unit}." if unit else ""
        result.append(
            {
                "row_label": row_label,
                "headers": clean_headers,
                "values": dict(zip(clean_headers, values)),
                "search_text": (
                    f"Financial table row. Metric: {row_label}.{period_text}{unit_text} "
                    + " ".join(pairs)
                    + f" Source page: {table.get('page_number', '')}."
                ),
            }
        )
    return result


# Compatibility names used by the original notebook.
clean_data = clean_text
chunker = chunk_pages
