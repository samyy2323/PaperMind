"""Read-side helpers: turn ORM rows into the JSON shapes the frontend consumes, plus analytics."""
from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import AnswerSheet, Exam, StudentAnswer, User
from app.models.sheet import GRADED, NEEDS_REVIEW, PROCESSING, UPLOADED

API_STAGE = {"ocr": "ocr", "correction": "correction", "segmentation": "correction", "grading": "grading", "done": "done"}


def sheet_ui_status(sheet: AnswerSheet) -> str:
    if sheet.status in (UPLOADED, PROCESSING):
        return "processing"
    return "needs_review" if sheet.status == NEEDS_REVIEW else "graded"


def recompute_status(sheet: AnswerSheet) -> None:
    """After a teacher acts on a flagged answer, move the sheet to graded once nothing is pending."""
    if sheet.status in (UPLOADED, PROCESSING) or not sheet.answers:
        return
    if any(a.pending_review for a in sheet.answers):
        sheet.status = NEEDS_REVIEW
    else:
        sheet.status = GRADED
        sheet.error_message = None


def load_sheets(db: Session, exam_id: int, *, latest_per_student: bool = True) -> list[AnswerSheet]:
    sheets = list(
        db.scalars(
            select(AnswerSheet)
            .where(AnswerSheet.exam_id == exam_id)
            .options(selectinload(AnswerSheet.answers), selectinload(AnswerSheet.student), selectinload(AnswerSheet.images))
            .order_by(AnswerSheet.created_at.desc(), AnswerSheet.id.desc())
        )
    )
    if not latest_per_student:
        return sheets
    seen: set[int] = set()
    out = []
    for s in sheets:  # newest first
        if s.student_id not in seen:
            seen.add(s.student_id)
            out.append(s)
    return out


def question_out(q, extraction_confidence: float | None = None) -> dict:
    return {
        "id": q.id, "number": q.question_number, "text": q.question_text, "referenceAnswer": q.reference_answer,
        "type": q.question_type, "marks": q.marks, "extractionConfidence": extraction_confidence,
    }


def exam_out(exam: Exam, submission_count: int | None = None) -> dict:
    return {
        "id": exam.id, "classId": exam.class_id, "title": exam.title, "subject": exam.klass.subject,
        "subDomain": exam.klass.sub_domain, "totalMarks": exam.total_marks, "numQuestions": len(exam.questions),
        "status": exam.status, "answerKeyFile": exam.answer_key_filename, "createdAt": exam.created_at,
        "publishedAt": exam.published_at, "submissionCount": submission_count,
        "questions": [question_out(q) for q in exam.questions],
    }


def sheet_row(sheet: AnswerSheet, exam: Exam, *, for_student: bool = False) -> dict:
    ui = sheet_ui_status(sheet)
    pending = [a for a in sheet.answers if a.pending_review]
    total = None
    if ui != "processing" and sheet.answers:
        # Students never see AI marks for answers still awaiting the teacher.
        total = sum(a.effective_score for a in sheet.answers if not (for_student and a.pending_review))
    elif ui != "processing":
        total = 0.0
    return {
        "id": sheet.id, "examId": exam.id, "examTitle": exam.title, "examSubject": exam.klass.subject,
        "studentId": sheet.student_id, "studentName": sheet.student.name, "usn": sheet.student.roll_number,
        "submittedAt": sheet.created_at, "pages": len(sheet.images), "status": ui, "total": total,
        "max": exam.total_marks, "pending": len(pending),
        "mismatch": any(a.flagged_reason == "injection_suspected" for a in pending),
        "lowConfidence": any(a.flagged_reason in ("low_confidence", "parse_failure") for a in pending),
        "hasReferencePdf": bool(sheet.reference_pdf_path),
        "error": None if for_student else sheet.error_message,
    }


def answer_out(a: StudentAnswer, *, for_student: bool, names: dict[int, str]) -> dict:
    hidden = for_student and a.pending_review
    override = None
    if a.teacher_override_score is not None:
        override = {
            "score": a.teacher_override_score, "reason": a.teacher_override_reason,
            "by": names.get(a.overridden_by or -1, "Teacher"), "at": a.overridden_at,
        }
    return {
        "id": a.id, "questionId": a.question_id, "ocrText": a.corrected_text,
        "aiScore": None if hidden else a.extracted_score,
        "confidence": None if for_student else a.confidence,
        "relevance": None if for_student else a.embedding_similarity,
        "justification": None if hidden else a.llm_justification,
        "correct": None if hidden else a.objective_correct,
        "flaggedReason": None if for_student else a.flagged_reason,
        "pendingReview": a.pending_review,
        "reviewed": a.teacher_reviewed, "override": override,
    }


def sheet_detail(db: Session, sheet: AnswerSheet, viewer: User) -> dict:
    for_student = viewer.role == "student"
    exam = sheet.exam
    names = {u.id: u.name for u in db.scalars(select(User).where(User.id.in_([a.overridden_by for a in sheet.answers if a.overridden_by])))}
    row = sheet_row(sheet, exam, for_student=for_student)
    return {
        "exam": exam_out(exam),
        "submission": {
            "id": sheet.id, "examId": exam.id, "studentId": sheet.student_id, "studentName": sheet.student.name,
            "usn": sheet.student.roll_number, "submittedAt": sheet.created_at, "pages": row["pages"],
            "status": row["status"], "error": row["error"], "hasReferencePdf": row["hasReferencePdf"],
            "answers": [answer_out(a, for_student=for_student, names=names) for a in sheet.answers],
        },
    }


def analytics(db: Session, exam: Exam) -> dict:
    sheets = load_sheets(db, exam.id)
    graded = [s for s in sheets if sheet_ui_status(s) != "processing" and s.answers]
    total_marks = exam.total_marks
    totals = [sum(a.effective_score for a in s.answers) for s in graded]
    pcts = sorted(t / total_marks * 100 for t in totals)

    bins = [{"range": f"{i * 10}–{i * 10 + 10}%", "lo": i * 10, "count": 0} for i in range(10)]
    for p in pcts:
        bins[min(9, int(p // 10))]["count"] += 1

    by_q = defaultdict(list)
    for s in graded:
        for a in s.answers:
            by_q[a.question_id].append(a)
    per_question = []
    for q in exam.questions:
        rows = by_q.get(q.id, [])
        avg = sum(a.effective_score for a in rows) / len(rows) if rows else 0.0
        per_question.append({
            "question": f"Q{q.question_number}", "type": q.question_type, "marks": q.marks, "avg": round(avg, 1),
            "avgPct": round(avg / q.marks * 100) if q.marks else 0, "flagged": sum(1 for a in rows if a.pending_review),
        })

    low_conf = mismatch = 0
    for s in graded:
        for a in s.answers:
            if a.pending_review:
                if a.flagged_reason == "injection_suspected":
                    mismatch += 1
                else:
                    low_conf += 1

    flagged_sheets = sum(1 for s in graded if any(a.pending_review for a in s.answers))
    return {
        "exam": {"id": exam.id, "title": exam.title, "subject": exam.klass.subject, "totalMarks": total_marks},
        "totalSubmissions": len(sheets), "processing": len(sheets) - len(graded),
        "averageScore": (sum(totals) / len(totals)) if totals else 0.0,
        "averagePct": (sum(pcts) / len(pcts)) if pcts else 0.0,
        "medianPct": pcts[len(pcts) // 2] if pcts else 0.0, "highestPct": pcts[-1] if pcts else 0.0,
        "flaggedSubmissions": flagged_sheets, "cleanSubmissions": len(graded) - flagged_sheets,
        "flaggedAnswers": {"lowConfidence": low_conf, "mismatch": mismatch},
        "distribution": bins, "perQuestion": per_question,
    }
