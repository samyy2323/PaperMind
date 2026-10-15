"""Subjective grading: RAG retrieval -> injection-safe prompt -> local LLM -> validated JSON."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.core.config import get_settings
from app.services.grading.llm import LLMClient
from app.services.grading.prompt import RETRY_REMINDER, SYSTEM_PROMPT, build_user_prompt
from app.services.rag.embedder import Embedder

log = logging.getLogger(__name__)

# retrieve(query_vec) -> [(chunk_text, cosine_score)]; injected so the grader has no DB dependency.
Retriever = Callable[[np.ndarray], list[tuple[str, float]]]


class InvalidLLMOutput(ValueError):
    pass


@dataclass
class SubjectiveResult:
    score: float
    justification: str
    confidence: float  # LLM-reported (or default) certainty, 0..1
    similarity: float | None  # embedding similarity to syllabus/reference material
    parse_status: str  # ok | retried | failed | empty
    chunks: list[str] = field(default_factory=list)


def parse_llm_json(raw: str, max_marks: float) -> dict:
    """Validate the model's reply against {"score": number, "justification": string, "confidence"?: number}."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise InvalidLLMOutput("no JSON object in reply")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise InvalidLLMOutput(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise InvalidLLMOutput("reply is not a JSON object")

    score = obj.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score):
        raise InvalidLLMOutput("'score' must be a number")
    just = obj.get("justification")
    if not isinstance(just, str) or not just.strip():
        raise InvalidLLMOutput("'justification' must be a non-empty string")
    conf = obj.get("confidence")
    if conf is not None and (isinstance(conf, bool) or not isinstance(conf, (int, float))):
        raise InvalidLLMOutput("'confidence' must be a number when present")

    clamped = min(max(float(score), 0.0), float(max_marks))
    return {
        "score": round(clamped * 2) / 2,
        "justification": just.strip(),
        "confidence": None if conf is None else min(max(float(conf), 0.0), 1.0),
        "clamped": clamped != float(score),
    }


class SubjectiveGrader:
    def __init__(self, llm: LLMClient, embedder: Embedder, retriever: Retriever | None = None):
        self.llm, self.embedder, self.retriever = llm, embedder, retriever
        self.settings = get_settings()

    def grade(self, question: str, reference: str, student_answer: str, max_marks: float) -> SubjectiveResult:
        answer = (student_answer or "").strip()
        if not answer:
            return SubjectiveResult(0.0, "No answer was found for this question.", 0.9, None, "empty")

        vecs = self.embedder.encode([answer, reference or question])
        ans_vec, ref_vec = vecs[0], vecs[1]

        hits = self.retriever(ans_vec)[: self.settings.RAG_TOP_K] if self.retriever else []
        # Topical similarity = closest of: syllabus chunks (retrieval scores) or the teacher's reference answer.
        similarity = float(max([float(ans_vec @ ref_vec)] + [s for _, s in hits]))
        chunks = [t for t, _ in hits]

        user_prompt = build_user_prompt(question, reference, max_marks, chunks, answer)
        parsed, status = self._ask(user_prompt, max_marks)
        if parsed is None:
            return SubjectiveResult(
                0.0, "Automatic grading failed (the model did not return valid JSON). Needs manual review.",
                0.0, similarity, "failed", chunks,
            )
        conf = parsed["confidence"] if parsed["confidence"] is not None else self.settings.DEFAULT_LLM_CONFIDENCE
        if status == "retried":
            conf = min(conf, 0.5)  # needed a second attempt => not a clean parse
        if parsed["clamped"]:
            conf = min(conf, 0.5)  # model gave an out-of-range score
        return SubjectiveResult(parsed["score"], parsed["justification"], conf, similarity, status, chunks)

    def _ask(self, user_prompt: str, max_marks: float) -> tuple[dict | None, str]:
        """Call the LLM; reject and retry once if the reply doesn't parse."""
        prompts = [user_prompt, user_prompt + "\n\n" + RETRY_REMINDER]
        for attempt, prompt in enumerate(prompts):
            try:
                return parse_llm_json(self.llm.generate(SYSTEM_PROMPT, prompt), max_marks), "ok" if attempt == 0 else "retried"
            except InvalidLLMOutput as exc:
                log.warning("LLM output rejected (attempt %d): %s", attempt + 1, exc)
        return None, "failed"
