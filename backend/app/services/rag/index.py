"""FAISS index management: one inner-product index per class, persisted under STORAGE_ROOT/indexes.

Vectors are L2-normalised so inner product == cosine similarity. Chunk DB ids are used as FAISS ids,
so search results map straight back to ``knowledge_base_chunks`` rows. The index is always
rebuildable from the vectors stored in the database (used after deletes and if the file is missing).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import faiss
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import KnowledgeBaseChunk, KnowledgeBaseDocument
from app.services.storage import get_storage

_lock = threading.RLock()
_cache: dict[int, faiss.Index] = {}


@dataclass
class Hit:
    chunk_id: int
    text: str
    score: float


def _key(class_id: int) -> str:
    return f"indexes/class_{class_id}.faiss"


def _new_index(dim: int) -> faiss.Index:
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))


def _persist(class_id: int, index: faiss.Index) -> None:
    storage = get_storage()
    with storage.local_path(_key(class_id)) as p:
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(p))


def _load(class_id: int) -> faiss.Index | None:
    storage = get_storage()
    if not storage.exists(_key(class_id)):
        return None
    with storage.local_path(_key(class_id)) as p:
        return faiss.read_index(str(p))


def add_vectors(class_id: int, ids: list[int], vectors: np.ndarray) -> None:
    if len(ids) == 0:
        return
    with _lock:
        index = _cache.get(class_id) or _load(class_id) or _new_index(vectors.shape[1])
        index.add_with_ids(vectors.astype("float32"), np.asarray(ids, dtype="int64"))
        _cache[class_id] = index
        _persist(class_id, index)


def rebuild(class_id: int, db: Session, dim: int) -> None:
    """Recreate a class index from the vectors stored in Postgres."""
    rows = db.execute(
        select(KnowledgeBaseChunk.id, KnowledgeBaseChunk.embedding_vector)
        .join(KnowledgeBaseDocument, KnowledgeBaseChunk.document_id == KnowledgeBaseDocument.id)
        .where(KnowledgeBaseDocument.class_id == class_id)
    ).all()
    with _lock:
        index = _new_index(dim)
        if rows:
            ids = np.asarray([r[0] for r in rows], dtype="int64")
            vecs = np.vstack([np.frombuffer(r[1], dtype="float32") for r in rows])
            index.add_with_ids(vecs, ids)
        _cache[class_id] = index
        _persist(class_id, index)


def search(class_id: int, db: Session, query_vec: np.ndarray, k: int) -> list[Hit]:
    """Top-k chunks for a single normalised query vector (shape (dim,))."""
    with _lock:
        index = _cache.get(class_id) or _load(class_id)
        if index is None:
            return []
        _cache[class_id] = index
        if index.ntotal == 0:
            return []
        scores, ids = index.search(query_vec.reshape(1, -1).astype("float32"), min(k, index.ntotal))
    found = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
    if not found:
        return []
    rows = {
        c.id: c.chunk_text
        for c in db.scalars(select(KnowledgeBaseChunk).where(KnowledgeBaseChunk.id.in_([i for i, _ in found])))
    }
    return [Hit(chunk_id=i, text=rows[i], score=s) for i, s in found if i in rows]


def drop_cache() -> None:
    with _lock:
        _cache.clear()

