"""Grading pipeline steps: OCR -> correction -> segmentation -> grading -> aggregation.

Each step is a plain function (unit-testable) that loads the sheet, does its work, persists the results
and updates ``answer_sheets.stage``. Intermediate artifacts (per-line OCR text) are kept in storage as
JSON so the steps can run in different Celery workers. A failing step marks the sheet ``needs_review``
with an error message; later steps see that and skip, so a bad photo never crashes the queue.
"""
from __future__ import annotations

import functools
import json
import logging

from sqlalchemy import delete, select

from app.core.config import get_settings
from app.core.db import session_scope
from app.models import AnswerSheet, Question, StudentAnswer
from app.models.sheet import GRADED, NEEDS_REVIEW, PROCESSING
from app.services import correction as correction_service
from app.services import ocr as ocr_service
from app.services import segmentation
from app.services.grading import grade_answer
from app.services.grading.llm import get_llm_client
from app.services.grading.subjective import SubjectiveGrader
from app.services.rag import index as faiss_index
from app.services.rag.embedder import get_embedder
from app.services.storage import get_storage

log = logging.getLogger(__name__)


class PipelineError(Exception):
    """Expected failure with a message safe to show the teacher."""


def _art(sheet_id: int, name: str) -> str:
    return f"answer_sheets/{sheet_id}/{name}"


def _fail(sheet_id: int, message: str) -> None:
    with session_scope() as db:
        sheet = db.get(AnswerSheet, sheet_id)
        if sheet:
            sheet.status, sheet.error_message = NEEDS_REVIEW, message


def _guarded(stage_name: str):
    """Run a step; never raise. Skip if an earlier step already failed this sheet."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(sheet_id: int, *args, **kwargs) -> int:
            with session_scope() as db:
                sheet = db.get(AnswerSheet, sheet_id)
                if sheet is None or (sheet.status == NEEDS_REVIEW and sheet.error_message):
                    return sheet_id
                sheet.status, sheet.stage = PROCESSING, stage_name
            try:
                fn(sheet_id, *args, **kwargs)
            except PipelineError as exc:
                log.warning("sheet %s failed at %s: %s", sheet_id, stage_name, exc)
                _fail(sheet_id, str(exc))
            except Exception as exc:  # noqa: BLE001 - keep the queue alive
                log.exception("sheet %s crashed at %s", sheet_id, stage_name)
                _fail(sheet_id, f"Unexpected error while {stage_name}: {type(exc).__name__}: {exc}")
            return sheet_id

        return wrapper

    return deco


# --------------------------------------------------------------------------- 3d OCR
@_guarded("ocr")
def step_ocr(sheet_id: int, image_keys: list[str]) -> None:
    storage = get_storage()
    engine = ocr_service.get_ocr_engine()
    lines = []
    for page, key in enumerate(image_keys):
        try:
            with storage.local_path(key) as path:
                image = ocr_service.load_image(path)
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"Page {page + 1} could not be opened as an image ({type(exc).__name__}).") from exc
        lines.extend(engine.read_image(image, page))
    if not lines:
        raise PipelineError("No handwriting could be detected in the uploaded photos. Check focus and lighting.")
    storage.save(_art(sheet_id, "ocr.json"), json.dumps([l.to_dict() for l in lines]).encode())


# --------------------------------------------------------------------------- 3e correction
@_guarded("correction")
def step_correction(sheet_id: int) -> None:
    storage = get_storage()
    lines = json.loads(storage.read(_art(sheet_id, "ocr.json")))
    fixed = correction_service.get_corrector().correct([l["text"] for l in lines])
    out = [
        {"raw": l["text"], "text": t, "confidence": l["confidence"], "page": l["page"], "box": l.get("box")}
        for l, t in zip(lines, fixed)
    ]
    storage.save(_art(sheet_id, "corrected.json"), json.dumps(out).encode())


# --------------------------------------------------------------------------- 3f segmentation
@_guarded("segmentation")
def step_segmentation(sheet_id: int) -> None:
    storage = get_storage()
    lines = [
        segmentation.SegLine(
            raw=l["raw"],
            text=l["text"],
            confidence=l["confidence"],
            box=tuple(l["box"]) if l.get("box") is not None else None,
        )
        for l in json.loads(storage.read(_art(sheet_id, "corrected.json")))
    ]
    with session_scope() as db:
        sheet = db.get(AnswerSheet, sheet_id)
        questions = list(db.scalars(select(Question).where(Question.exam_id == sheet.exam_id).order_by(Question.question_number)))
        if not questions:
            raise PipelineError("The exam has no questions.")
        by_number = {q.question_number: q for q in questions}
        types = {n: q.question_type for n, q in by_number.items()}
        contexts = {
            n: f"{q.question_text}\n{q.reference_answer}".strip()
            for n, q in by_number.items()
        }
        embedder = get_embedder()
        segments = segmentation.get_segmenter("markers").segment(
            lines, list(by_number), types, question_contexts=contexts, embedder=embedder
        )
        db.execute(delete(StudentAnswer).where(StudentAnswer.answer_sheet_id == sheet_id))
        for number, q in by_number.items():
            seg = segments[number]
            db.add(
                StudentAnswer(
                    answer_sheet_id=sheet_id, question_id=q.id, ocr_raw_text=seg.raw_text,
                    corrected_text=seg.text, ocr_confidence=seg.confidence, max_score=q.marks,
                    debug={"segmentation": seg.method, "match_score": seg.match_score},
                )
            )


# --------------------------------------------------------------------------- 3g grading
def _make_retriever(db, class_id: int):
    k = get_settings().RAG_TOP_K

    def retrieve(vec):
        return [(h.text, h.score) for h in faiss_index.search(class_id, db, vec, k)]

    return retrieve


@_guarded("grading")
def step_grading(sheet_id: int) -> None:
    with session_scope() as db:
        sheet = db.get(AnswerSheet, sheet_id)
        answers = list(db.scalars(select(StudentAnswer).where(StudentAnswer.answer_sheet_id == sheet_id)))
        qs = {q.id: q for q in db.scalars(select(Question).where(Question.exam_id == sheet.exam_id))}
        grader = None
        if any(qs[a.question_id].question_type == "subjective" for a in answers):
            grader = SubjectiveGrader(get_llm_client(), get_embedder(), _make_retriever(db, sheet.exam.class_id))

        errors = 0
        for a in answers:
            q = qs[a.question_id]
            try:
                r = grade_answer(
                    question_type=q.question_type, question_text=q.question_text, reference_answer=q.reference_answer,
                    marks=q.marks, answer_text=a.corrected_text, ocr_confidence=a.ocr_confidence, subjective_grader=grader,
                )
            except Exception as exc:  # noqa: BLE001 - e.g. LLM server down: keep the sheet, flag the answer
                log.exception("grading answer %s failed", a.id)
                errors += 1
                a.extracted_score, a.confidence, a.flagged_reason = 0.0, 0.0, "parse_failure"
                a.llm_justification = f"Automatic grading failed ({type(exc).__name__}). Needs manual review."
                a.debug = {**(a.debug or {}), "error": f"{type(exc).__name__}: {exc}"}
                continue
            a.extracted_score, a.max_score, a.confidence = r.score, r.max_score, r.confidence
            a.embedding_similarity, a.llm_justification = r.similarity, r.justification
            a.objective_correct, a.flagged_reason = r.objective_correct, r.flagged_reason
            if (a.debug or {}).get("segmentation") == "positional" and not a.flagged_reason:
                a.flagged_reason = "segmentation_uncertain"  # answers were assigned by position, not by question number
            elif (a.debug or {}).get("segmentation") == "content_match" and not a.flagged_reason:
                m_score = (a.debug or {}).get("match_score")
                if m_score is not None and m_score < 0.50:
                    a.flagged_reason = "low_confidence"
            a.debug = {**(a.debug or {}), **r.debug}
        if errors:
            sheet.error_message = f"The grading model failed on {errors} answer(s). Reprocess once it is available."


# --------------------------------------------------------------------------- 3i aggregation
@_guarded("done")
def step_aggregate(sheet_id: int) -> None:
    with session_scope() as db:
        sheet = db.get(AnswerSheet, sheet_id)
        answers = list(db.scalars(select(StudentAnswer).where(StudentAnswer.answer_sheet_id == sheet_id)))
        sheet.status = NEEDS_REVIEW if any(a.pending_review for a in answers) else GRADED
        sheet.stage = "done"

