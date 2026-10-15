from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.user import utcnow


class KnowledgeBaseDocument(Base):
    __tablename__ = "knowledge_base_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("classes.id"), index=True)
    filename: Mapped[str] = mapped_column(String(300))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    upload_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    chunks: Mapped[list["KnowledgeBaseChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class KnowledgeBaseChunk(Base):
    __tablename__ = "knowledge_base_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("knowledge_base_documents.id"), index=True)
    chunk_text: Mapped[str] = mapped_column(Text)
    # float32 vector, little-endian bytes. The FAISS index is rebuildable from these.
    embedding_vector: Mapped[bytes] = mapped_column(LargeBinary)

    document: Mapped[KnowledgeBaseDocument] = relationship(back_populates="chunks")
