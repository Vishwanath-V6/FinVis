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
                narrative = _narrative_text(page, tables, text)
                pages.append(
                    {
                        "document_id": doc_id,
                        "source_filename": path.name,
                        "page_number": page_index + 1,
                        "text": text,
                        "narrative_text": narrative,
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


_NUMERIC_CELL_RE = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?$")
_MISSING_CELL = {"", "-", "–", "—", "na", "n.a.", "n.a", ".", "*"}
_SCOPE_RE = re.compile(r"\b(standalone|consolidated)\b", re.IGNORECASE)
_PERIOD_RE = re.compile(r"\bFY\s*[-/]?\s*(20\d{2}|\d{2})\b", re.IGNORECASE)
_UNIT_RE = re.compile(
    r"(₹|Rs\.?|INR|USD|EUR|GBP)\s*(?:in\s+)?(crore|crores|lakh|lakhs|million|billion)",
    re.IGNORECASE,
)


def page_likely_has_table(page_text: str) -> bool:
    """Skip expensive table detection on narrative pages with almost no figures."""
    stripped = page_text.strip()
    if len(stripped) < 40:
        return True
    return sum(character.isdigit() for character in stripped) >= 24


def _extract_page_tables(
    page: Any, document_id: str, source_filename: str, page_number: int, page_text: str = ""
) -> list[dict[str, Any]]:
    """Extract tables into structured rows while retaining cell boundaries."""
    if not page_likely_has_table(page_text):
        return []
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
        normalized_rows = [_normalise_table_row(row) for row in rows]
        if not any(cell for row in normalized_rows for cell in row):
            continue
        headers, data_rows, inferred = _repair_financial_table_headers(
            normalized_rows[0], normalized_rows[1:], page_text
        )
        bbox = _table_bbox(table)
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
                "reporting_period": " ".join(
                    _unique_in_order(
                        [f"FY{match.group(1)[-2:]}" for match in _PERIOD_RE.finditer(page_text)]
                    )
                ),
                "bbox": bbox,
            }
        )
    return tables


def _normalise_table_row(row: Any) -> list[str]:
    cells: list[str] = []
    previous = ""
    for index, cell in enumerate(row):
        text = cell.strip() if isinstance(cell, str) else ""
        text = re.sub(r"\bafet\s+r\b", "after", text, flags=re.IGNORECASE)
        # Forward-fill only empty header-like leading cells from merged labels.
        if not text and index == 0 and previous:
            text = previous
        if text:
            previous = text
        cells.append(text)
    return cells


def _table_bbox(table: Any) -> list[float] | None:
    bbox = getattr(table, "bbox", None)
    if bbox is None:
        return None
    try:
        return [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None


def _narrative_text(page: Any, tables: list[dict[str, Any]], fallback: str) -> str:
    """Keep page prose for chunking without duplicating extracted table cells."""
    boxes = [tuple(table["bbox"]) for table in tables if table.get("bbox")]
    if not boxes or not hasattr(page, "get_text"):
        return fallback
    try:
        blocks = page.get_text("blocks") or []
    except Exception:
        return fallback
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, (list, tuple)) or len(block) < 5:
            continue
        x0, y0, x1, y1, text = block[:5]
        if not isinstance(text, str) or not text.strip():
            continue
        region = (float(x0), float(y0), float(x1), float(y1))
        if any(_boxes_overlap(region, box) for box in boxes):
            continue
        parts.append(text.strip())
    return "\n".join(parts) if parts else fallback


def _boxes_overlap(left: tuple[float, ...], right: tuple[float, ...], pad: float = 2.0) -> bool:
    return not (
        left[2] < right[0] - pad
        or left[0] > right[2] + pad
        or left[3] < right[1] - pad
        or left[1] > right[3] + pad
    )


def _is_numeric_cell(cell: str) -> bool:
    text = cell.strip().replace(" ", "")
    if text.casefold() in _MISSING_CELL:
        return False
    return bool(_NUMERIC_CELL_RE.fullmatch(text))


def _is_numeric_data_row(row: list[str]) -> bool:
    values = [cell for cell in row[1:]]
    if not values:
        return False
    numeric = sum(_is_numeric_cell(cell) for cell in values)
    return numeric >= 2 and numeric / len(values) >= 0.5


def _merge_header_rows(rows: list[list[str]]) -> list[str]:
    width = max(len(row) for row in rows)
    merged: list[str] = []
    for column in range(width):
        parts: list[str] = []
        last = ""
        for row in rows:
            cell = row[column].strip() if column < len(row) and isinstance(row[column], str) else ""
            if not cell:
                cell = last
            elif cell:
                last = cell
            if cell and cell not in parts:
                parts.append(cell)
        merged.append(" ".join(parts).strip())
    if merged and not merged[0]:
        merged[0] = "Particulars"
    return merged


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _reconstruct_headers_from_page(column_count: int, page_text: str, first_row: list[str]) -> list[str] | None:
    if column_count < 2:
        return None
    metric = first_row[0] if first_row else ""
    index = page_text.find(metric) if metric else -1
    region = page_text[:index] if index > 0 else page_text[:2000]
    scopes = _unique_in_order([match.group(1).title() for match in _SCOPE_RE.finditer(region)])
    periods = _unique_in_order(
        [f"FY{match.group(1)[-2:]}" for match in _PERIOD_RE.finditer(region)]
    )
    value_columns = column_count - 1
    if len(scopes) >= 2 and len(periods) >= 2 and value_columns == len(scopes[:2]) * len(periods[:2]):
        headers = ["Particulars"]
        for scope in scopes[:2]:
            for period in periods[:2]:
                headers.append(f"{scope} {period}")
        return headers
    if not scopes and len(periods) == value_columns:
        return ["Particulars", *periods]
    if len(scopes) == value_columns and not periods:
        return ["Particulars", *scopes]
    if len(periods) == value_columns:
        return ["Particulars", *periods]
    return None


def _repair_financial_table_headers(
    headers: list[str], rows: list[list[str]], page_text: str
) -> tuple[list[str], list[list[str]], bool]:
    if not headers:
        return headers, rows, False
    all_rows = [headers, *rows]
    data_index = next((index for index, row in enumerate(all_rows) if _is_numeric_data_row(row)), 0)
    if data_index == 0 and _is_numeric_data_row(headers):
        reconstructed = _reconstruct_headers_from_page(len(headers), page_text, headers)
        if reconstructed and len(reconstructed) == len(headers):
            return reconstructed, all_rows, True
        numeric_header_cells = sum(_is_numeric_cell(cell) for cell in headers[1:])
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
            ], all_rows, True
        return headers, rows, False
    if data_index > 0:
        merged = _merge_header_rows(all_rows[:data_index])
        if len(merged) != len(all_rows[data_index]):
            width = len(all_rows[data_index])
            merged = (merged + [""] * width)[:width]
        return merged, all_rows[data_index:], data_index > 1
    return headers, rows, False


def _infer_financial_unit(page_text: str) -> str:
    matches = [
        (currency, scale.casefold())
        for currency, scale in _UNIT_RE.findall(page_text)
    ]
    if not matches:
        return ""
    normalised = {
        (
            "₹" if currency in {"₹", "Rs", "Rs.", "INR"} or currency.casefold().startswith("rs") else currency,
            "crore" if scale.startswith("crore") else "lakh" if scale.startswith("lakh") else scale,
        )
        for currency, scale in matches
    }
    if len(normalised) != 1:
        return ""
    currency, scale = next(iter(normalised))
    return f"{currency} {scale}"


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
        narrative = page.get("narrative_text")
        record.update(
            text=text,
            narrative_text=clean_text(narrative) if isinstance(narrative, str) else text,
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
    seen_chunk_text: set[str] = set()
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
        raw_text = page.get("narrative_text", raw_text)
        if not isinstance(raw_text, str):
            raise PreprocessingError("page text must be a string")
        words = raw_text.split()
        if words:
            for start in range(0, len(words), chunk_size - overlap):
                chunk_words = words[start : start + chunk_size]
                if not chunk_words:
                    continue
                chunk_text = " ".join(chunk_words)
                text_key = re.sub(r"\s+", " ", chunk_text.casefold()).strip()
                if text_key in seen_chunk_text:
                    continue
                seen_chunk_text.add(text_key)
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
                        "bbox": table.get("bbox"),
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
