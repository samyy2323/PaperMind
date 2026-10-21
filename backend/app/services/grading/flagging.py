"""Confidence flagging: decide whether an AI-graded answer needs a human look."""
from __future__ import annotations

from app.core.config import Settings

INJECTION = "injection_suspected"
PARSE_FAILURE = "parse_failure"
LOW_CONFIDENCE = "low_confidence"
ANSWER_NOT_FOUND = "answer_not_found"


def decide_flag(
    *,
    question_type: str,
    score: float,
    max_score: float,
    confidence: float | None,
    similarity: float | None,
    parse_status: str,
    settings: Settings,
) -> str | None:
    # Empty extractions must never be silently marked 0 without alerting the teacher.
    if parse_status == "empty":
        return ANSWER_NOT_FOUND

    # High mark for an answer that has almost nothing to do with the syllabus/reference material: the
    # profile of a prompt-injection or gibberish answer that fooled the LLM.
    if (
        question_type == "subjective"
        and similarity is not None
        and max_score > 0
        and score / max_score > settings.HIGH_SCORE_RATIO
        and similarity < settings.LOW_SIMILARITY_THRESHOLD
    ):
        return INJECTION
    if parse_status == "failed":
        return PARSE_FAILURE
    if confidence is not None and confidence < settings.CONFIDENCE_THRESHOLD:
        return LOW_CONFIDENCE
    return None
