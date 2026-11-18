"""Small central configuration for local FinVis deployments."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.getenv("FINVIS_CACHE_DIR", str(ROOT / ".finvis_cache")))
MAX_UPLOAD_BYTES = int(os.getenv("FINVIS_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
EMBEDDING_MODEL = os.getenv(
    "FINVIS_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
QA_MODEL = os.getenv(
    "FINVIS_QA_MODEL",
    "distilbert-base-uncased-distilled-squad",
)
