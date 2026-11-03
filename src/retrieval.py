"""FAISS indexing, persistence, and semantic retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from .model_utils import DEFAULT_EMBEDDING_MODEL, encode_texts


class RetrievalIndex:
    """Cosine-similarity FAISS index with an explicit row-to-chunk mapping."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.index: faiss.Index | None = None
        self.chunks: list[dict[str, Any]] = []

    def build(self, chunks: Iterable[dict[str, Any]]) -> "RetrievalIndex":
        records = [self._validate_chunk(chunk) for chunk in chunks]
        if not records:
            raise ValueError("cannot build an index from zero chunks")
        chunk_ids = [record["chunk_id"] for record in records]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_id values must be unique")
        texts = [record.get("text") or record.get("words", "") for record in records]
        vectors = encode_texts(texts, self.model_name)
        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(records)
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("embedding/chunk count or dimension mismatch")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = records
        return self

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index is None or not self.chunks:
            raise RuntimeError("retrieval index is empty")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_vector = encode_texts([query], self.model_name)
        if query_vector.ndim != 2 or query_vector.shape != (1, self.index.d):
            raise ValueError("query embedding dimension does not match the index")
        candidate_count = min(len(self.chunks), max(top_k, 20))
        scores, positions = self.index.search(query_vector, candidate_count)
        results: list[dict[str, Any]] = []
        for score, position in zip(scores[0], positions[0]):
            if position < 0 or position >= len(self.chunks):
                continue
            result = dict(self.chunks[int(position)])
            result["score"] = float(score)
            result["index_position"] = int(position)
            results.append(result)
        results.sort(key=lambda result: self._table_query_score(query, result), reverse=True)
        return results[:top_k]

    @staticmethod
    def _table_query_score(query: str, result: dict[str, Any]) -> float:
        """Prefer table rows containing both the requested metric and period."""
        base = float(result.get("score", 0.0))
        if result.get("content_type") != "table":
            return base
        query_terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
        row_terms = set(
            re.findall(
                r"[a-z0-9]+",
                f"{result.get('table_row_label', '')} {result.get('text', '')}".casefold(),
            )
        )
        overlap = len(query_terms & row_terms)
        years = set(re.findall(r"(?:fy)?20\d{2}", query.casefold()))
        period_terms = set(re.findall(r"(?:fy)?20\d{2}", str(result.get("text", "")).casefold()))
        period_match = bool(years & period_terms)
        return base + overlap * 0.05 + (0.25 if period_match else 0.0)

    def save(self, directory: str | Path) -> None:
        if self.index is None or not self.chunks:
            raise RuntimeError("cannot save an empty index")
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target / "index.faiss"))
        (target / "chunks.json").write_text(
            json.dumps({"model_name": self.model_name, "chunks": self.chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "RetrievalIndex":
        target = Path(directory)
        index_path, metadata_path = target / "index.faiss", target / "chunks.json"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"index files are missing from {target}")
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("model_name"), str):
            raise ValueError("invalid index metadata")
        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ValueError("index metadata chunks must be a list")
        instance = cls(payload["model_name"])
        instance.index = faiss.read_index(str(index_path))
        instance.chunks = [instance._validate_chunk(chunk) for chunk in raw_chunks]
        if len({chunk["chunk_id"] for chunk in instance.chunks}) != len(instance.chunks):
            raise ValueError("chunk_id values must be unique")
        if (
            not instance.index.is_trained
            or instance.index.d <= 0
            or instance.index.ntotal != len(instance.chunks)
        ):
            raise ValueError("FAISS row count does not match chunk metadata")
        return instance

    @staticmethod
    def _validate_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(chunk, dict):
            raise ValueError("each chunk must be a dictionary")
        record = dict(chunk)
        text = record.get("text") or record.get("words")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each chunk requires non-empty text")
        if not isinstance(record.get("chunk_id"), int) or isinstance(record["chunk_id"], bool):
            raise ValueError("each chunk requires an integer chunk_id")
        if not isinstance(record.get("document_id"), str) or not record["document_id"].strip():
            raise ValueError("each chunk requires a document_id")
        if not isinstance(record.get("source_filename"), str) or not record["source_filename"].strip():
            raise ValueError("each chunk requires a source_filename")
        if (
            not isinstance(record.get("page_number"), int)
            or isinstance(record["page_number"], bool)
            or record["page_number"] < 1
        ):
            raise ValueError("each chunk requires a positive page_number")
        record["text"] = text
        if record.get("content_type", "text") not in {"text", "table"}:
            raise ValueError("content_type must be text or table")
        return record
