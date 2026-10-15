"""Sentence embedding wrapper (sentence-transformers, open-source, runs locally)."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from app.core.config import get_settings
from app.core.singleton import once


class Embedder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 array of shape (len(texts), dim)."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vecs = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return vecs.astype("float32")


@once
def _load(model_name: str) -> Embedder:
    return SentenceTransformerEmbedder(model_name)


def get_embedder() -> Embedder:
    return _load(get_settings().EMBED_MODEL)
