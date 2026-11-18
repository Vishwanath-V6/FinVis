"""Embedding model loading and validated batch encoding."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 64


def configure_cpu_threads(max_threads: int = 4) -> int:
    """Cap BLAS/Torch threads so FAISS and embeddings do not oversubscribe."""
    cpu_count = os.cpu_count() or 1
    threads = max(1, min(max_threads, cpu_count))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        import faiss

        faiss.omp_set_num_threads(threads)
    except Exception:
        pass
    return threads


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load and cache a sentence-transformer model."""
    if not model_name:
        raise ValueError("model_name must not be empty")
    configure_cpu_threads()
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load embedding model {model_name!r}. "
            "The first use downloads it from Hugging Face."
        ) from exc


@lru_cache(maxsize=512)
def _cached_query_vector(text: str, model_name: str, batch_size: int) -> tuple[float, ...]:
    matrix = _encode_unchecked((text,), model_name, batch_size)
    return tuple(float(value) for value in matrix[0])


def _encode_unchecked(texts: Sequence[str], model_name: str, batch_size: int) -> np.ndarray:
    model = load_embedding_model(model_name)
    vectors = np.asarray(
        model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
        raise ValueError(f"unexpected embedding shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("embedding matrix contains zero vectors")
    return vectors / norms[:, None]


def encode_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """Encode texts as finite float32, L2-normalized vectors."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("texts must contain non-empty strings")
    if len(texts) == 1:
        return np.asarray([_cached_query_vector(texts[0], model_name, batch_size)], dtype=np.float32)
    return _encode_unchecked(texts, model_name, batch_size)
