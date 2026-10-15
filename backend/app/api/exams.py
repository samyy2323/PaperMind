from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.api.classes import get_or_create_class
from app.api.deps import get_db, owned_class, owned_exam, require_teacher
from app.models import AnswerSheet, Class, Exam, Question, User
from app.schemas import (
    ExamCreate, ExamOut, ExamPatch, QuestionIn, QuestionOut, QuestionPatch, SheetRow,
)
from app.services import results
from app.services.ingestion.answer_key import extract_pdf_text, parse_answer_key

router = APIRouter(prefix="/exams", tags=["exams"])


def _locked(exam: Exam) -> None:
    if exam.status == "published":
        raise HTTPException(status.HTTP_409_CONFLICT, "Published exams are locked")


@router.get("", response_model=list[ExamOut])
def list_exams(db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    exams = db.scalars(
        select(Exam).join(Class, Exam.class_id == Class.id).where(Class.teacher_id == teacher.id)
        .options(selectinload(Exam.questions), selectinload(Exam.klass)).order_by(Exam.created_at.desc(), Exam.id.desc())
    ).all()
    counts = dict(
        db.execute(
            select(AnswerSheet.exam_id, func.count(func.distinct(AnswerSheet.student_id)))
            .where(AnswerSheet.exam_id.in_([e.id for e in exams])).group_by(AnswerSheet.exam_id)
        ).all()
    )
    return [results.exam_out(e, counts.get(e.id, 0)) for e in exams]


@router.post("", response_model=ExamOut, status_code=status.HTTP_201_CREATED)
def create_exam(body: ExamCreate, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    if body.class_id is not None:
        klass = owned_class(db, body.class_id, teacher)
    elif body.subject and body.sub_domain:
        klass = get_or_create_class(db, teacher, body.subject.strip(), body.sub_domain.strip())
    else:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Provide classId, or subject and subDomain")
    exam = Exam(class_id=klass.id, title=body.title.strip(), total_marks=body.total_marks)
    db.add(exam)
    db.commit()
    return results.exam_out(owned_exam(db, exam.id, teacher), 0)


@router.get("/{exam_id}", response_model=ExamOut)
def get_exam(exam_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    return results.exam_out(owned_exam(db, exam_id, teacher))


@router.patch("/{exam_id}", response_model=ExamOut)
def patch_exam(exam_id: int, body: ExamPatch, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    exam = owned_exam(db, exam_id, teacher)
    _locked(exam)
    if body.title is not None:
        exam.title = body.title.strip()
    if body.total_marks is not None:
        exam.total_marks = body.total_marks
    db.commit()
    return results.exam_out(exam)


# ------------------------------------------------------------------ questions
@router.put("/{exam_id}/questions", response_model=list[QuestionOut])
def replace_questions(exam_id: int, body: list[QuestionIn], db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    """Replace the whole question list of a draft exam (used when the editor saves its draft)."""
    exam = owned_exam(db, exam_id, teacher)
    _locked(exam)
    exam.questions.clear()
    db.flush()
    for i, q in enumerate(body, start=1):
        exam.questions.append(
            Question(question_number=i, question_text=q.text, reference_answer=q.reference_answer, question_type=q.type, marks=q.marks)
        )
    db.commit()
    return [results.question_out(q) for q in owned_exam(db, exam_id, teacher).questions]


@router.patch("/{exam_id}/questions/{question_id}", response_model=QuestionOut)
def patch_question(
    exam_id: int, question_id: int, body: QuestionPatch, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)
):
    exam = owned_exam(db, exam_id, teacher)
    _locked(exam)
    q = next((q for q in exam.questions if q.id == question_id), None)
    if q is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Question not found")
    if body.text is not None:
        q.question_text = body.text
    if body.reference_answer is not None:
        q.reference_answer = body.reference_answer
    if body.type is not None:
        q.question_type = body.type
    if body.marks is not None:
        q.marks = body.marks
    db.commit()
    return results.question_out(q)


# ------------------------------------------------------------------ 3a answer key ingestion
@router.post("/{exam_id}/answer-key", response_model=list[QuestionOut])
def upload_answer_key(exam_id: int, file: UploadFile = File(...), db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    """Extract questions + reference answers from a typed answer-key PDF. Replaces the draft's questions;
    the teacher reviews/edits them (PATCH/PUT) before publishing."""
    exam = owned_exam(db, exam_id, teacher)
    _locked(exam)
    name = file.filename or "answer_key.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "The answer key must be a PDF")
    try:
        parsed = parse_answer_key(extract_pdf_text(file.file.read()), exam.total_marks)
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))
    except Exception as exc:  # noqa: BLE001 - corrupt/encrypted PDF
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Could not read the PDF: {exc}")

    exam.questions.clear()
    db.flush()
    conf = {}
    for p in parsed:
        q = Question(
            question_number=p.number, question_text=p.text, reference_answer=p.reference_answer,
            question_type=p.question_type, marks=p.marks,
        )
        exam.questions.append(q)
        conf[p.number] = p.extraction_confidence
    exam.answer_key_filename = name
    db.commit()
    return [results.question_out(q, conf.get(q.question_number)) for q in owned_exam(db, exam_id, teacher).questions]


@router.post("/{exam_id}/publish", response_model=ExamOut)
def publish_exam(exam_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    exam = owned_exam(db, exam_id, teacher)
    _locked(exam)
    problems = []
    if not exam.questions:
        problems.append("Add at least one question.")
    if any(not q.question_text.strip() or not q.reference_answer.strip() for q in exam.questions):
        problems.append("Every question needs question text and a reference answer.")
    total = sum(q.marks for q in exam.questions)
    if exam.questions and abs(total - exam.total_marks) > 1e-6:
        problems.append(f"Question marks add up to {total:g}, but total marks is {exam.total_marks:g}.")
    if problems:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, " ".join(problems))
    exam.status, exam.published_at = "published", datetime.now(timezone.utc)
    db.commit()
    return results.exam_out(exam)


# ------------------------------------------------------------------ review dashboard
@router.get("/{exam_id}/submissions", response_model=list[SheetRow])
def list_submissions(exam_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    exam = owned_exam(db, exam_id, teacher)
    return [results.sheet_row(s, exam) for s in results.load_sheets(db, exam_id)]


@router.get("/{exam_id}/review-queue", response_model=list[SheetRow])
def review_queue(exam_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    exam = owned_exam(db, exam_id, teacher)
    rows = [results.sheet_row(s, exam) for s in results.load_sheets(db, exam_id)]
    return [r for r in rows if r["status"] == "needs_review"]


@router.get("/{exam_id}/analytics")
def exam_analytics(exam_id: int, db: Session = Depends(get_db), teacher: User = Depends(require_teacher)):
    return results.analytics(db, owned_exam(db, exam_id, teacher))
