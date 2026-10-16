"""OCR text correction.

A fine-tuned seq2seq model (ByT5 or BART) repairs likely TrOCR errors line by line before grading.
Training code lives in ``backend/training/correction`` (data builder, Colab notebook, evaluator). Until a checkpoint
is configured via CORRECTION_MODEL_PATH the stage is a pass-through so the rest of the pipeline is not blocked.

Safety: a correction model must only repair handwriting-recognition errors, never rewrite what the student wrote.
``apply_guard`` keeps the original OCR text whenever the model's output is implausibly far from it.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol

from rapidfuzz.distance import Levenshtein

from app.core.config import get_settings
from app.core.singleton import once

log = logging.getLogger(__name__)

# A genuine OCR repair changes a handful of characters; a hallucinated rewrite changes most of the line.
MAX_ABS_EDITS = 6
MAX_REL_EDITS = 0.4


# A leading question number ("24 . ", "3) ", "Q7.", "21" glued to the text). The correction model must never touch it:
# a wrong question number sends the whole answer to the wrong question.
_PREFIX = re.compile(r"^(\s*(?:\(?[A-Za-z]{0,3}\.?\s*)?\d{1,2}\s*(?:[\.\)\:\-]+|(?=[A-Za-z]{3}))\s*)")
_DIGITS = re.compile(r"\d")


def split_prefix(line: str) -> tuple[str, str]:
    """Split a line into (question-number prefix, body). The prefix is '' when the line has none."""
    m = _PREFIX.match(line)
    if not m or not line[m.end():].strip():
        return "", line
    return m.group(1), line[m.end():]


def finalize_body(body: str, fixed: str) -> str:
    """Accept the model's version of a line body only if it is a plausible repair that keeps every digit."""
    fixed = apply_guard(body, fixed)
    if _DIGITS.findall(fixed) != _DIGITS.findall(body):  # numbers are content (marks, values, years): never change them
        return body
    return fixed


def apply_guard(raw: str, corrected: str) -> str:
    """Return ``corrected`` unless it is empty or too different from ``raw``, in which case keep ``raw``."""
    if not corrected.strip():
        return raw
    edits = Levenshtein.distance(raw, corrected)
    if edits > max(MAX_ABS_EDITS, MAX_REL_EDITS * len(raw)) or len(corrected) > 1.6 * len(raw) + 10:
        log.warning("correction rejected (%d edits on a %d-char line): %r -> %r", edits, len(raw), raw[:60], corrected[:60])
        return raw
    return corrected


class Corrector(Protocol):
    name: str

    def correct(self, lines: list[str]) -> list[str]:
        """Return corrected text, same length and order as the input."""


class PassThroughCorrector:
    name = "passthrough"

    def correct(self, lines: list[str]) -> list[str]:
        return list(lines)


class Seq2SeqCorrector:
    """Wrapper for a fine-tuned ByT5 / BART checkpoint (AutoModelForSeq2SeqLM)."""

    def __init__(self, path: str, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = f"seq2seq:{path}"
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).eval()
        self.batch_size = batch_size

    def _generate(self, texts: list[str]) -> list[str]:
        out: list[str] = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
                gen = self.model.generate(**enc, max_new_tokens=256, num_beams=4)
                out.extend(t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True))
        return out

    def correct(self, lines: list[str]) -> list[str]:
        parts = [split_prefix(l) for l in lines]
        bodies = [b for _, b in parts]
        fixed = self._generate(bodies)
        return [prefix + finalize_body(body, f) for (prefix, body), f in zip(parts, fixed)]


@once
def _load(path: str) -> Corrector:
    if path:
        return Seq2SeqCorrector(path)
    log.info("CORRECTION_MODEL_PATH not set: text correction is a pass-through")
    return PassThroughCorrector()


def get_corrector() -> Corrector:
    return _load(get_settings().CORRECTION_MODEL_PATH)
