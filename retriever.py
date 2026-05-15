"""
retriever.py – Lightweight semantic retrieval over the SHL catalog.

Uses sentence-transformers for embedding and FAISS for ANN search.
Falls back to keyword TF-IDF if sentence-transformers is unavailable.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import Optional

log = logging.getLogger(__name__)

# ── Try to load heavy deps; fall back gracefully ─────────────────────────────

_HEAVY = False
log.warning("sentence-transformers or faiss not available; using keyword fallback")


# ── Keyword retriever (always available) ─────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9.#+]+", text.lower())


class KeywordRetriever:
    """Simple TF-IDF style scorer over catalog items."""

    def __init__(self, catalog: list[dict]) -> None:
        self.catalog = catalog
        self._build_index()

    def _doc_text(self, item: dict) -> str:
        parts = [
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("tags", [])),
            " ".join(item.get("job_levels", [])),
        ]
        return " ".join(parts)

    def _build_index(self) -> None:
        self._idf: dict[str, float] = {}
        self._docs: list[list[str]] = []
        df: dict[str, int] = defaultdict(int)
        N = len(self.catalog)

        for item in self.catalog:
            tokens = _tokenize(self._doc_text(item))
            self._docs.append(tokens)
            for tok in set(tokens):
                df[tok] += 1

        for tok, count in df.items():
            self._idf[tok] = math.log((N + 1) / (count + 1)) + 1.0

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        q_tokens = _tokenize(query)
        scores: list[tuple[float, dict]] = []

        for i, (tokens, item) in enumerate(zip(self._docs, self.catalog)):
            tok_set = set(tokens)
            score = sum(
                self._idf.get(t, 0.0)
                for t in q_tokens
                if t in tok_set
            )
            scores.append((score, item))

        scores.sort(key=lambda x: -x[0])
        # Return items with score > 0, capped at top_k
        return [item for score, item in scores if score > 0][:top_k]


# ── Semantic retriever (preferred when heavy deps available) ──────────────────

class SemanticRetriever:
    """Dense retrieval using sentence-transformers + FAISS."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, catalog: list[dict]) -> None:
        self.catalog = catalog
        self._model: Optional[SentenceTransformer] = None
        self._index = None
        self._ready = False

    def _doc_text(self, item: dict) -> str:
        return (
            f"{item.get('name', '')}. "
            f"{item.get('description', '')} "
            f"Tags: {', '.join(item.get('tags', []))}. "
            f"Levels: {', '.join(item.get('job_levels', []))}."
        )

    def build(self) -> None:
        if not _HEAVY:
            log.warning("Heavy deps missing; SemanticRetriever.build() is a no-op")
            return
        log.info("Loading embedding model %s …", self.MODEL_NAME)
        self._model = SentenceTransformer(self.MODEL_NAME)
        texts = [self._doc_text(item) for item in self.catalog]
        embeddings = self._model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._ready = True
        log.info("FAISS index built with %d vectors (dim=%d)", len(texts), dim)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self._ready or self._model is None:
            raise RuntimeError("Index not built")
        q_emb = self._model.encode([query])
        q_emb = np.array(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)
        distances, indices = self._index.search(q_emb, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or dist < 0.10:   # threshold to avoid weak matches
                continue
            results.append(self.catalog[idx])
        return results


# ── Public factory ────────────────────────────────────────────────────────────

class CatalogRetriever:
    """
    Facade that tries SemanticRetriever first; falls back to KeywordRetriever.
    """

    def __init__(self, catalog: list[dict]) -> None:
        self.catalog = catalog
        self._keyword = KeywordRetriever(catalog)
        self._semantic: Optional[SemanticRetriever] = None

        if _HEAVY:
            sem = SemanticRetriever(catalog)
            try:
                sem.build()
                self._semantic = sem
                log.info("Using semantic retriever")
            except Exception as exc:
                log.warning("Semantic retriever failed to build (%s); using keyword fallback", exc)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return up to top_k catalog items ranked by relevance to query."""
        if self._semantic is not None:
            try:
                results = self._semantic.search(query, top_k)
                if results:
                    return results
            except Exception as exc:
                log.warning("Semantic search error: %s; falling back to keyword", exc)

        return self._keyword.search(query, top_k)

    def get_by_name(self, name: str) -> Optional[dict]:
        """Exact (case-insensitive) or partial name lookup."""
        name_lower = name.lower()
        # exact match
        for item in self.catalog:
            if item["name"].lower() == name_lower:
                return item
        # partial match
        for item in self.catalog:
            if name_lower in item["name"].lower():
                return item
        return None
