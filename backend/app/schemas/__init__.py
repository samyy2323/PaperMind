"""Pydantic request/response schemas. JSON uses camelCase to match the React frontend."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, from_attributes=True)


# ------------------------------------------------------------------ auth
class LoginRequest(CamelModel):
    email: str
    password: str


class UserOut(CamelModel):
    id: int
    name: str
    email: str
    role: str
    usn: str | None = None  # roll number
    token: str | None = None


# ------------------------------------------------------------------ classes / roster
class ClassEnsure(CamelModel):
    subject: str
    sub_domain: str


class ClassOut(CamelModel):
    id: int
    name: str
    subject: str
    sub_domain: str


class RosterAdd(CamelModel):
    name: str = Field(min_length=1, max_length=200)
    roll_number: str = Field(min_length=1, max_length=50)
    email: str | None = None
    password: str | None = None  # defaults to the roll number


class RosterPatch(CamelModel):
    name: str | None = None
    roll_number: str | None = None


class RosterStudentOut(CamelModel):
    id: int
    name: str
    roll_number: str | None
    email: str


# ------------------------------------------------------------------ exams
QuestionType = Literal["objective", "subjective"]


class QuestionIn(CamelModel):
    text: str = ""
    reference_answer: str = ""
    type: QuestionType = "subjective"
    marks: float = Field(default=1, ge=0)


class QuestionPatch(CamelModel):
    text: str | None = None
    reference_answer: str | None = None
    type: QuestionType | None = None
    marks: float | None = Field(default=None, ge=0)


class QuestionOut(CamelModel):
    id: int
    number: int
    text: str
    reference_answer: str
    type: str
    marks: float
    extraction_confidence: float | None = None


class ExamCreate(CamelModel):
    title: str = Field(min_length=1, max_length=300)
    total_marks: float = Field(gt=0)
    class_id: int | None = None
    # Convenience for the UI: resolves (or creates) the teacher's class for this subject.
    subject: str | None = None
    sub_domain: str | None = None


class ExamPatch(CamelModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    total_marks: float | None = Field(default=None, gt=0)


class ExamOut(CamelModel):
    id: int
    class_id: int
    title: str
    subject: str
    sub_domain: str
    total_marks: float
    num_questions: int
    status: str
    answer_key_file: str | None
    created_at: datetime
    published_at: datetime | None
    submission_count: int | None = None
    questions: list[QuestionOut]


# ------------------------------------------------------------------ knowledge base
class KnowledgeDocOut(CamelModel):
    id: int
    class_id: int
    file_name: str
    sub_domain: str
    size_kb: int
    chunks_indexed: int
    uploaded_at: datetime
    status: str = "indexed"


# ------------------------------------------------------------------ answer sheets
class UploadAccepted(CamelModel):
    submission_id: int  # answer sheet id
    exam_id: int
    status: str
    stage: str


class StatusOut(CamelModel):
    status: str
    stage: str  # ocr | correction | grading | done
    error: str | None = None


class SheetRow(CamelModel):
    id: int
    exam_id: int
    exam_title: str
    exam_subject: str
    student_id: int
    student_name: str
    usn: str | None
    submitted_at: datetime
    pages: int
    status: str  # graded | needs_review | processing
    total: float | None
    max: float
    pending: int
    mismatch: bool
    low_confidence: bool
    has_reference_pdf: bool
    error: str | None = None


class OverrideIn(CamelModel):
    score: float = Field(ge=0)
    reason: str = Field(min_length=1, max_length=500)


class SheetDetailOut(CamelModel):
    exam: ExamOut
    submission: dict[str, Any]
