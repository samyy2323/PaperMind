"""Grading orchestration: route by question type, combine signals, decide flags."""
from __future__ import annotations

from dataclasses import dataclass

from app.core.config import get_settings
from app.services.grading.flagging import decide_flag
from app.services.grading.objective import grade_objective
from app.services.grading.subjective import SubjectiveGrader


@dataclass
class GradeOutcome:
    score: float
    max_score: float
    confidence: float
    similarity: float | None
    justification: str | None
    objective_correct: bool | None
    flagged_reason: str | None
    debug: dict


def grade_answer(
    *,
    question_type: str,
    question_text: str,
    reference_answer: str,
    marks: float,
    answer_text: str,
    ocr_confidence: float | None,
    subjective_grader: SubjectiveGrader | None,
) -> GradeOutcome:
    settings = get_settings()

    if question_type == "objective":
        r = grade_objective(answer_text, reference_answer, marks)
        conf, sim, just, correct, status, debug = r.confidence, None, None, r.correct, "ok", {"detail": r.detail}
        score = r.score
    else:
        if subjective_grader is None:
            raise RuntimeError("Subjective question but no grader configured")
        s = subjective_grader.grade(question_text, reference_answer, answer_text, marks)
        score, conf, sim, just, correct, status = s.score, s.confidence, s.similarity, s.justification, None, s.parse_status
        debug = {"parse_status": status, "retrieved_chunks": [c[:200] for c in s.chunks]}

    # OCR uncertainty caps overall confidence: a shaky transcription cannot be graded confidently.
    if ocr_confidence is not None:
        conf = min(conf, ocr_confidence)

    flag = decide_flag(
        question_type=question_type, score=score, max_score=marks, confidence=conf,
        similarity=sim, parse_status=status, settings=settings,
    )
    return GradeOutcome(score, marks, round(conf, 4), None if sim is None else round(sim, 4), just, correct, flag, debug)
