"""Measure staged FinVis ingestion and cache reuse without fabricating results."""

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


def run(pdf: Path, cache_dir: Path, request: str) -> dict:
    content = pdf.read_bytes()
    first_started = time.perf_counter()
    first = prepare_document(content, pdf.name, request, cache_dir=cache_dir)
    first_wall = time.perf_counter() - first_started
    second_started = time.perf_counter()
    second = prepare_document(content, pdf.name, request, cache_dir=cache_dir)
    second_wall = time.perf_counter() - second_started
    return {
        "pdf": str(pdf),
        "request": request,
        "document_fingerprint": first["fingerprint"],
        "page_count": first["manifest"]["page_count"],
        "selected_page_count": len(first["manifest"]["selected_pages"]),
        "chunk_count": len(first["chunks"]),
        "first_run_cache_hit": first["cache_hit"],
        "first_run_wall_seconds": first_wall,
        "first_run_stage_timings": first["timings"],
        "cache_run_cache_hit": second["cache_hit"],
        "cache_run_wall_seconds": second_wall,
        "cache_run_stage_timings": second["timings"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".finvis_cache"))
    parser.add_argument("--request", default="revenue operating profit net profit debt")
    parser.add_argument("--output", type=Path, default=Path("benchmark_outputs/ingestion_performance.json"))
    args = parser.parse_args()
    if not args.pdf.exists():
        raise FileNotFoundError(args.pdf)
    result = run(args.pdf, args.cache_dir, args.request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
