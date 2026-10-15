from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.db import get_db
from app.core.security import get_current_user, require_teacher
from app.models import AnswerSheet, Class, Enrollment, Exam, StudentAnswer, User

__all__ = ["get_db", "get_current_user", "require_teacher", "owned_class", "owned_exam", "visible_sheet", "owned_answer"]


def owned_class(db: Session, class_id: int, teacher: User) -> Class:
    klass = db.get(Class, class_id)
    if klass is None or klass.teacher_id != teacher.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Class not found")
    return klass


def owned_exam(db: Session, exam_id: int, teacher: User) -> Exam:
    exam = db.scalars(
        select(Exam).where(Exam.id == exam_id).options(selectinload(Exam.questions), selectinload(Exam.klass))
    ).first()
    if exam is None or exam.klass.teacher_id != teacher.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Exam not found")
    return exam


def visible_sheet(db: Session, sheet_id: int, user: User) -> AnswerSheet:
    """A teacher sees sheets of their own classes; a student only their own, and only once graded."""
    sheet = db.scalars(
        select(AnswerSheet)
        .where(AnswerSheet.id == sheet_id)
        .options(selectinload(AnswerSheet.answers), selectinload(AnswerSheet.images), selectinload(AnswerSheet.student))
    ).first()
    if sheet is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer sheet not found")
    exam = db.get(Exam, sheet.exam_id)
    if user.role == "teacher":
        if exam.klass.teacher_id != user.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer sheet not found")
    elif sheet.student_id != user.id or not student_can_see(sheet):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer sheet not found")
    return sheet


def student_can_see(sheet: AnswerSheet) -> bool:
    """Students see a result only after the pipeline produced answers."""
    return sheet.status in ("graded", "needs_review") and bool(sheet.answers)


def owned_answer(db: Session, answer_id: int, teacher: User) -> StudentAnswer:
    ans = db.get(StudentAnswer, answer_id)
    if ans is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer not found")
    sheet = db.get(AnswerSheet, ans.answer_sheet_id)
    if db.get(Exam, sheet.exam_id).klass.teacher_id != teacher.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer not found")
    return ans


def enrolled(db: Session, student_id: int, class_id: int) -> bool:
    return db.scalar(select(Enrollment.id).where(Enrollment.student_id == student_id, Enrollment.class_id == class_id)) is not None
