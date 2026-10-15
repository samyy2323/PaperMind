from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, owned_class, require_teacher
from app.models import Class, KnowledgeBaseDocument, User
from app.schemas import KnowledgeDocOut
from app.services import rag
from app.services.ingestion.answer_key import extract_text_from_upload

router = APIRouter(tags=["knowledge base"])
ALLOWED = (".pdf", ".txt", ".md")


def _doc_out(doc: KnowledgeBaseDocument, klass: Class) -> KnowledgeDocOut:
    return KnowledgeDocOut(
        id=doc.id, class_id=klass.id, file_name=doc.filename, sub_domain=klass.sub_domain,
        size_kb=max(1, round(doc.size_bytes / 1024)), chunks_indexed=doc.chunk_count, uploaded_at=doc.upload_date,
    )


@router.get("/knowledge-base", response_model=list[KnowledgeDocOut])
def list_documents(
    sub_domain: str | None = Query(default=None, alias="subDomain"),
    db: Session = Depends(get_db),
    teacher: User = Depends(require_teacher),
):
    """All knowledge-base documents across the teacher's classes, optionally filtered by sub-domain."""
    q = (
        select(KnowledgeBaseDocument, Class)
        .join(Class, KnowledgeBaseDocument.class_id == Class.id)
        .where(Class.teacher_id == teacher.id)
        .order_by(KnowledgeBaseDocument.upload_date.desc())
    )
    if sub_domain:
        q = q.where(Class.sub_domain == sub_domain)
    return [_doc_out(d, c) for d, c in db.execute(q)]


@router.post("/classes/{class_id}/knowledge-base", response_model=KnowledgeDocOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    class_id: int, file: UploadFile = File(...), db: Session = Depends(get_db), teacher: User = Depends(require_teacher)
):
    klass = owned_class(db, class_id, teacher)
    name = file.filename or "document"
    if not name.lower().endswith(ALLOWED):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload a PDF, .txt or .md file")
    data = file.file.read()
    try:
        text = extract_text_from_upload(name, data)
        doc = rag.ingest_document(db, class_id, name, text, len(data))
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    except Exception as exc:  # noqa: BLE001 - corrupt PDF etc.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Could not read the document: {exc}")
    return _doc_out(doc, klass)


@router.delete("/knowledge-base/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    doc = db.get(KnowledgeBaseDocument, doc_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    owned_class(db, doc.class_id, teacher)
    rag.delete_document(db, doc)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
