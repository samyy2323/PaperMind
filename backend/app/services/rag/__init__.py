from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import KnowledgeBaseChunk, KnowledgeBaseDocument
from app.services.rag import index as faiss_index
from app.services.rag.chunking import chunk_text
from app.services.rag.embedder import get_embedder


def ingest_document(db: Session, class_id: int, filename: str, text: str, size_bytes: int) -> KnowledgeBaseDocument:
    """Chunk, embed and index a knowledge-base document. Raises ValueError if it has no usable text."""
    chunks = chunk_text(text, get_settings().CHUNK_TARGET_CHARS)
    if not chunks:
        raise ValueError("No extractable text found in the document.")
    vecs = get_embedder().encode(chunks)

    doc = KnowledgeBaseDocument(class_id=class_id, filename=filename, size_bytes=size_bytes, chunk_count=len(chunks))
    db.add(doc)
    db.flush()
    rows = [
        KnowledgeBaseChunk(document_id=doc.id, chunk_text=c, embedding_vector=v.astype("float32").tobytes())
        for c, v in zip(chunks, vecs)
    ]
    db.add_all(rows)
    db.flush()
    faiss_index.add_vectors(class_id, [r.id for r in rows], vecs)
    db.commit()
    return doc


def delete_document(db: Session, doc: KnowledgeBaseDocument) -> None:
    class_id = doc.class_id
    db.delete(doc)
    db.commit()
    faiss_index.rebuild(class_id, db, get_embedder().dim)


__all__ = ["ingest_document", "delete_document"]
