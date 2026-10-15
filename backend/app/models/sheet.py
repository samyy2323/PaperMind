from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.exam import Exam, Question
from app.models.user import User, utcnow

# answer_sheets.status
UPLOADED, PROCESSING, GRADED, NEEDS_REVIEW = "uploaded", "processing", "graded", "needs_review"
# answer_sheets.stage (internal; the API maps segmentation -> "correction")
STAGES = ("ocr", "correction", "segmentation", "grading", "done")


class AnswerSheet(Base):
    __tablename__ = "answer_sheets"

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), index=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(20), default=UPLOADED)
    stage: Mapped[str] = mapped_column(String(20), default="ocr")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_pdf_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    exam: Mapped[Exam] = relationship()
    student: Mapped[User] = relationship(foreign_keys=[student_id])
    images: Mapped[list["AnswerSheetImage"]] = relationship(
        back_populates="answer_sheet", cascade="all, delete-orphan", order_by="AnswerSheetImage.page_order"
    )
    answers: Mapped[list["StudentAnswer"]] = relationship(back_populates="answer_sheet", cascade="all, delete-orphan")


class AnswerSheetImage(Base):
    __tablename__ = "answer_sheet_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    answer_sheet_id: Mapped[int] = mapped_column(ForeignKey("answer_sheets.id"), index=True)
    image_path: Mapped[str] = mapped_column(String(500))
    page_order: Mapped[int]

    answer_sheet: Mapped[AnswerSheet] = relationship(back_populates="images")


class StudentAnswer(Base):
    __tablename__ = "student_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    answer_sheet_id: Mapped[int] = mapped_column(ForeignKey("answer_sheets.id"), index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), index=True)
    ocr_raw_text: Mapped[str] = mapped_column(Text, default="")
    corrected_text: Mapped[str] = mapped_column(Text, default="")
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    extracted_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # AI-awarded
    max_score: Mapped[float] = mapped_column(Float, default=0)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding_similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_justification: Mapped[str | None] = mapped_column(Text, nullable=True)
    objective_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    flagged_reason: Mapped[str | None] = mapped_column(String(40), nullable=True)
    teacher_override_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    teacher_override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    overridden_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    overridden_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # True once a teacher has accepted the AI score for a flagged answer without changing it.
    teacher_reviewed: Mapped[bool] = mapped_column(Boolean, default=False)
    debug: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    answer_sheet: Mapped[AnswerSheet] = relationship(back_populates="answers")
    question: Mapped[Question] = relationship()

    @property
    def effective_score(self) -> float:
        if self.teacher_override_score is not None:
            return self.teacher_override_score
        return self.extracted_score or 0.0

    @property
    def pending_review(self) -> bool:
        return (
            self.flagged_reason is not None
            and self.teacher_override_score is None
            and not self.teacher_reviewed
        )
