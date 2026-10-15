from contextlib import ExitStack
import io

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.api.deps import enrolled, get_current_user, get_db, owned_exam, require_teacher, student_can_see, visible_sheet
from app.models import AnswerSheet, AnswerSheetImage, Exam, StudentAnswer, User
from app.models.sheet import UPLOADED
from app.schemas import SheetDetailOut, SheetRow, StatusOut, UploadAccepted
from app.services import results
from app.services.pdf import build_reference_pdf
from app.services.storage import get_storage
from app.tasks.tasks import enqueue_pipeline

router = APIRouter(tags=["answer sheets"])

MAX_PAGES = 30
MAX_BYTES = 25 * 1024 * 1024


def _image_ext(data: bytes) -> str:
    """Validate the upload really is an image and return its extension."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
            fmt = (im.format or "").lower()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "One of the files is not a readable image")
    if fmt not in ("jpeg", "png", "webp", "bmp", "tiff", "mpo"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported image format: {fmt}")
    return {"jpeg": "jpg", "mpo": "jpg"}.get(fmt, fmt)


# ------------------------------------------------------------------ 3c teacher uploads a student's photos
@router.post("/students/{student_id}/answer-sheet", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
def upload_answer_sheet(
    student_id: int,
    exam_id: int = Form(..., alias="examId"),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    teacher: User = Depends(require_teacher),
):
    """Save the photos, assemble the human-readable reference PDF, and enqueue the grading pipeline.
    Returns 202 immediately; poll GET /answer-sheets/{id}/status."""
    exam = owned_exam(db, exam_id, teacher)
    if exam.status != "published":
        raise HTTPException(status.HTTP_409_CONFLICT, "Publish the exam before uploading answer sheets")
    student = db.get(User, student_id)
    if student is None or student.role != "student" or not enrolled(db, student_id, exam.class_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Student is not on this exam's class roster")
    if not files or len(files) > MAX_PAGES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Upload between 1 and {MAX_PAGES} photos")

    payloads = []
    for f in files:
        data = f.file.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"{f.filename} is larger than 25 MB")
        payloads.append((data, _image_ext(data)))

    sheet = AnswerSheet(exam_id=exam.id, student_id=student_id, uploaded_by=teacher.id, status=UPLOADED, stage="ocr")
    db.add(sheet)
    db.flush()

    storage = get_storage()
    keys = []
    for i, (data, ext) in enumerate(payloads, start=1):
        key = storage.save(f"answer_sheets/{sheet.id}/page_{i:03d}.{ext}", data)
        db.add(AnswerSheetImage(answer_sheet_id=sheet.id, image_path=key, page_order=i))
        keys.append(key)

    # Reference PDF: viewing/archival only. The OCR task receives the image keys, never this PDF.
    with ExitStack() as stack:
        paths = [stack.enter_context(storage.local_path(k)) for k in keys]
        pdf_bytes = build_reference_pdf(paths)
    sheet.reference_pdf_path = storage.save(f"answer_sheets/{sheet.id}/reference.pdf", pdf_bytes)
    db.commit()

    enqueue_pipeline(sheet.id, keys)
    return UploadAccepted(submission_id=sheet.id, exam_id=exam.id, status="processing", stage="ocr")


@router.post("/answer-sheets/{sheet_id}/reprocess", response_model=UploadAccepted, status_code=status.HTTP_202_ACCEPTED)
def reprocess(sheet_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    """Re-run the pipeline on the stored images (e.g. after an OCR failure or LLM outage)."""
    sheet = visible_sheet(db, sheet_id, teacher)
    db.query(StudentAnswer).filter_by(answer_sheet_id=sheet.id).delete()
    sheet.status, sheet.stage, sheet.error_message = UPLOADED, "ocr", None
    keys = [i.image_path for i in sheet.images]
    db.commit()
    enqueue_pipeline(sheet.id, keys)
    return UploadAccepted(submission_id=sheet.id, exam_id=sheet.exam_id, status="processing", stage="ocr")


# ------------------------------------------------------------------ status / results
@router.get("/answer-sheets/{sheet_id}/status", response_model=StatusOut)
def sheet_status(sheet_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    sheet = visible_sheet(db, sheet_id, user)  # students only reach a sheet once it has results
    ui = results.sheet_ui_status(sheet)
    stage = "done" if ui != "processing" else results.API_STAGE.get(sheet.stage, "ocr")
    return StatusOut(status=ui, stage=stage, error=None if user.role == "student" else sheet.error_message)


@router.get("/answer-sheets/{sheet_id}/results", response_model=SheetDetailOut)
def sheet_results(sheet_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    sheet = visible_sheet(db, sheet_id, user)
    return results.sheet_detail(db, sheet, user)


@router.get("/answer-sheets/{sheet_id}/reference-pdf")
def reference_pdf(sheet_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    sheet = visible_sheet(db, sheet_id, user)
    if not sheet.reference_pdf_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No reference PDF for this answer sheet")
    storage = get_storage()
    with storage.local_path(sheet.reference_pdf_path) as p:
        return FileResponse(p, media_type="application/pdf", filename=f"answer-sheet-{sheet.id}.pdf", content_disposition_type="inline")


@router.get("/students/{student_id}/answer-sheets", response_model=list[SheetRow])
def student_sheets(student_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """A student's own results list (teachers can list the sheets of their students too)."""
    if user.role == "student" and user.id != student_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You can only view your own results")
    sheets = db.scalars(
        select(AnswerSheet).where(AnswerSheet.student_id == student_id)
        .options(selectinload(AnswerSheet.answers), selectinload(AnswerSheet.images), selectinload(AnswerSheet.student))
        .order_by(AnswerSheet.created_at.desc(), AnswerSheet.id.desc())
    ).all()
    out = []
    for s in sheets:
        exam = db.get(Exam, s.exam_id)
        if user.role == "student":
            if not student_can_see(s):
                continue
        elif exam.klass.teacher_id != user.id:
            continue
        out.append(results.sheet_row(s, exam, for_student=user.role == "student"))
    return out
