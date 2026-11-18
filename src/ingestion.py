"""Staged, content-addressed document ingestion for FinVis."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import pymupdf

from .preprocessing import (
    PreprocessingError,
    _extract_page_tables,
    _narrative_text,
    chunk_pages,
    clean_pages,
    extract_text,
    page_likely_has_table,
)
from .retrieval import LexicalIndex, RetrievalIndex


PROCESSING_VERSION = "finvis-ingestion-v2"
DEFAULT_CACHE_DIR = Path(".finvis_cache")
_FINANCIAL_TERMS = {
    "revenue", "sales", "profit", "loss", "income", "expense", "cost",
    "asset", "liability", "debt", "cash", "flow", "ebit", "margin",
    "balance", "statement", "financial", "segment", "operations",
}


def fingerprint_bytes(content: bytes) -> str:
    if not isinstance(content, bytes) or not content:
        raise ValueError("PDF content must be non-empty bytes")
    return hashlib.sha256(content).hexdigest()


def validate_pdf_bytes(content: bytes, max_bytes: int) -> str:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if not content or len(content) > max_bytes:
        raise ValueError(f"PDF must be non-empty and at most {max_bytes} bytes")
    if not content.startswith(b"%PDF-"):
        raise PreprocessingError("uploaded content is not a valid PDF")
    return fingerprint_bytes(content)


def scan_pdf(path: str | Path) -> dict[str, Any]:
    """Perform the cheap native-text pass without table detection or OCR."""
    started = time.perf_counter()
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    pages: list[dict[str, Any]] = []
    try:
        with pymupdf.open(pdf_path) as document:
            if document.is_encrypted or len(document) == 0:
                raise PreprocessingError("PDF is encrypted or contains no pages")
            metadata = dict(document.metadata or {})
            for number, page in enumerate(document, 1):
                text = page.get_text("text") or ""
                pages.append(
                    {
                        "page_number": number,
                        "character_count": len(text),
                        "word_count": len(text.split()),
                        "text": text,
                        "has_native_text": bool(text.strip()),
                    }
                )
    except PreprocessingError:
        raise
    except Exception as exc:
        raise PreprocessingError(f"Unable to scan PDF {pdf_path.name}: {exc}") from exc
    return {
        "metadata": metadata,
        "page_count": len(pages),
        "pages": pages,
        "elapsed_seconds": time.perf_counter() - started,
    }


def select_page_numbers(scan: dict[str, Any], request: str = "") -> list[int]:
    """Select likely financial pages using scan text, request terms, and adjacency."""
    if not isinstance(request, str):
        raise TypeError("request must be a string")
    requested = set(re.findall(r"[a-z0-9]+", request.casefold()))
    # A targeted request must narrow work; only the summary workflow uses the
    # broad financial vocabulary. Adjacent pages preserve table headers/context.
    terms = requested if requested else _FINANCIAL_TERMS
    scored: list[tuple[int, int]] = []
    for page in scan.get("pages", []):
        text = str(page.get("text", "")).casefold()
        score = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", text))
        if score:
            scored.append((score, int(page["page_number"])))
    if not scored:
        return [int(page["page_number"]) for page in scan.get("pages", [])]
    threshold = 2 if len(requested) >= 3 else 1
    selected = {number for score, number in scored if score >= threshold}
    if not selected:
        selected = {number for _, number in scored}
    for score, number in scored:
        if score < threshold and selected:
            continue
        selected.update({number - 1, number + 1})
    page_count = int(scan.get("page_count", 0))
    return sorted(number for number in selected if 1 <= number <= page_count)


def _read_selected_pages(
    path: Path, document_id: str, selected_pages: list[int], *, enable_ocr: bool = True
) -> list[dict[str, Any]]:
    """Run the existing full-fidelity extraction only on selected pages."""
    selected = set(selected_pages)
    if not selected:
        raise PreprocessingError("no pages selected for extraction")
    with pymupdf.open(path) as document:
        if document.is_encrypted:
            raise PreprocessingError("encrypted PDFs are not supported")
        pages: list[dict[str, Any]] = []
        for number in sorted(selected):
            page = document[number - 1]
            text = page.get_text("text") or ""
            ocr_used = False
            ocr_error = ""
            if enable_ocr and len(text.strip()) < 20:
                try:
                    textpage = page.get_textpage_ocr(language="eng", dpi=300, full=True)
                    ocr_text = page.get_text("text", textpage=textpage) or ""
                except Exception as exc:
                    ocr_error = str(exc)
                    ocr_text = ""
                if ocr_text.strip():
                    text, ocr_used = ocr_text, True
            tables = (
                _extract_page_tables(page, document_id, path.name, number, text)
                if page_likely_has_table(text)
                else []
            )
            pages.append(
                {
                    "document_id": document_id,
                    "source_filename": path.name,
                    "page_number": number,
                    "text": text,
                    "narrative_text": _narrative_text(page, tables, text),
                    "character_count": len(text),
                    "word_count": len(text.split()),
                    "ocr_used": ocr_used,
                    "ocr_error": ocr_error,
                    "tables": tables,
                }
            )
    return pages


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path, Path]:
    root = cache_dir / key
    return root / "manifest.json", root / "chunks.json", root / "index"


def prepare_document(
    content: bytes,
    filename: str,
    request: str = "",
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    model_name: str | None = None,
    max_bytes: int = 50 * 1024 * 1024,
    progress: Callable[[str], None] | None = None,
    build_semantic: bool = False,
) -> dict[str, Any]:
    """Scan, target, extract, chunk, and cache a document by content hash."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    fingerprint = validate_pdf_bytes(content, max_bytes)
    request_key = " ".join(request.casefold().split())
    key = hashlib.sha256(
        f"{PROCESSING_VERSION}:{fingerprint}:{request_key}:{model_name or 'default'}".encode()
    ).hexdigest()
    cache_root = Path(cache_dir)
    manifest_path, chunks_path, index_dir = _cache_paths(cache_root, key)
    if manifest_path.exists() and chunks_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        index = RetrievalIndex.load(index_dir) if build_semantic and index_dir.exists() else None
        return {
            "cache_hit": True,
            "fingerprint": fingerprint,
            "manifest": manifest,
            "pages": manifest["pages"],
            "chunks": chunks,
            "index": index,
            "lexical_index": LexicalIndex(chunks),
            "timings": {"cache_load_seconds": manifest.get("cache_load_seconds", 0.0)},
        }

    cache_root.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "upload.pdf"
        path.write_bytes(content)
        started = time.perf_counter()
        scan = scan_pdf(path)
        timings["scan_seconds"] = time.perf_counter() - started
        if progress:
            progress(f"Fast scan complete: {scan['page_count']:,} pages.")
        selected = select_page_numbers(scan, request)
        for scanned_page in scan.get("pages", []):
            scanned_page.pop("text", None)
        started = time.perf_counter()
        pages = _read_selected_pages(path, fingerprint, selected)
        timings["targeted_extraction_seconds"] = time.perf_counter() - started
        if progress:
            progress(f"Extracted {len(selected):,} targeted pages.")
        started = time.perf_counter()
        chunks = chunk_pages(clean_pages(pages))
        timings["chunking_seconds"] = time.perf_counter() - started
        if progress:
            progress(f"Created {len(chunks):,} targeted chunks; building retrieval index.")
        index = None
        if build_semantic:
            started = time.perf_counter()
            index = RetrievalIndex(model_name=model_name) if model_name else RetrievalIndex()
            index.build(chunks)
            timings["embedding_index_seconds"] = time.perf_counter() - started

    manifest = {
        "processing_version": PROCESSING_VERSION,
        "fingerprint": fingerprint,
        "filename": Path(filename).name,
        "request": request,
        "page_count": scan["page_count"],
        "selected_pages": selected,
        "pages": pages,
        "timings": timings,
    }
    target = manifest_path.parent
    target.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    if index is not None:
        index.save(index_dir)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return {
        "cache_hit": False,
        "fingerprint": fingerprint,
        "manifest": manifest,
        "pages": pages,
        "chunks": chunks,
        "index": index,
        "lexical_index": LexicalIndex(chunks),
        "timings": timings,
    }


def build_semantic_index(
    prepared: dict[str, Any],
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    model_name: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> RetrievalIndex:
    """Build or reopen the deferred semantic index for an already prepared document."""
    manifest = prepared.get("manifest")
    if not isinstance(manifest, dict) or not isinstance(prepared.get("chunks"), list):
        raise ValueError("prepared document is missing manifest or chunks")
    key = hashlib.sha256(
        f"{PROCESSING_VERSION}:{manifest['fingerprint']}:{' '.join(str(manifest.get('request', '')).casefold().split())}:{model_name or 'default'}".encode()
    ).hexdigest()
    _, _, index_dir = _cache_paths(Path(cache_dir), key)
    if index_dir.exists():
        return RetrievalIndex.load(index_dir)
    if progress:
        progress("Building semantic embeddings for question answering...")
    index = RetrievalIndex(model_name=model_name) if model_name else RetrievalIndex()
    index.build(prepared["chunks"])
    index.save(index_dir)
    return index
