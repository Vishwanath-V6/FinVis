"""Before/after ingestion benchmark on the same PDF and request."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion import prepare_document
from src.preprocessing import chunk_pages, clean_pages, extract_text
from src.retrieval import RetrievalIndex


def legacy_ingestion(pdf: Path) -> dict:
    started = time.perf_counter()
    pages = clean_pages(extract_text(pdf, document_id=pdf.stem, enable_ocr=True))
    extraction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    chunks = chunk_pages(pages)
    chunking_seconds = time.perf_counter() - started
    started = time.perf_counter()
    RetrievalIndex().build(chunks)
    embedding_index_seconds = time.perf_counter() - started
    return {
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "table_chunk_count": sum(c.get("content_type") == "table" for c in chunks),
        "extraction_seconds": extraction_seconds,
        "chunking_seconds": chunking_seconds,
        "embedding_index_seconds": embedding_index_seconds,
        "total_seconds": extraction_seconds + chunking_seconds + embedding_index_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--request", default="revenue operating profit net profit debt")
    parser.add_argument("--cache-dir", type=Path, default=Path(".finvis_cache"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_outputs/performance_comparison.json"))
    args = parser.parse_args()
    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)
    legacy = legacy_ingestion(args.pdf)
    started = time.perf_counter()
    dashboard_started = time.perf_counter()
    dashboard = prepare_document(
        args.pdf.read_bytes(),
        args.pdf.name,
        args.request,
        cache_dir=args.cache_dir,
        build_semantic=False,
    )
    dashboard_wall = time.perf_counter() - dashboard_started
    started = time.perf_counter()
    optimized = prepare_document(
        args.pdf.read_bytes(),
        args.pdf.name,
        args.request,
        cache_dir=args.cache_dir,
        build_semantic=True,
    )
    optimized_wall = time.perf_counter() - started
    cached_started = time.perf_counter()
    cached = prepare_document(
        args.pdf.read_bytes(),
        args.pdf.name,
        args.request,
        cache_dir=args.cache_dir,
    )
    cached_wall = time.perf_counter() - cached_started
    result = {
        "pdf": str(args.pdf),
        "request": args.request,
        "legacy_full_ingestion": legacy,
        "optimized_targeted_ingestion": {
            "cache_hit": optimized["cache_hit"],
            "selected_page_count": len(optimized["manifest"]["selected_pages"]),
            "chunk_count": len(optimized["chunks"]),
            "stage_timings": optimized["timings"],
            "total_seconds": optimized_wall,
        },
        "dashboard_readiness": {
            "cache_hit": dashboard["cache_hit"],
            "selected_page_count": len(dashboard["manifest"]["selected_pages"]),
            "chunk_count": len(dashboard["chunks"]),
            "total_seconds": dashboard_wall,
        },
        "optimized_cache_reuse": {
            "cache_hit": cached["cache_hit"],
            "total_seconds": cached_wall,
            "stage_timings": cached["timings"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
