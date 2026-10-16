"""Rule-based grading for objective questions (option letters, true/false, short answers). No LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

ACCEPT_RATIO = 85  # fuzzy text match accepted as correct
AMBIGUOUS_RATIO = 60  # between this and ACCEPT_RATIO: marked wrong but flagged for review
LETTER_CONFLICT_TEXT_RATIO = 92  # letter disagrees but the option text matches this well => OCR misread the letter

_LETTER = re.compile(r"^\(?([a-e])\)?(?:[\s.:\-)]+(.*))?$", re.I | re.S)
_TRUE = {"true", "t"}
_FALSE = {"false", "f"}


@dataclass
class ObjectiveResult:
    correct: bool
    score: float
    confidence: float
    detail: str


def _norm(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _split_option(s: str) -> tuple[str | None, str]:
    """'(b) Method overriding' -> ('b', 'method overriding'); 'b' -> ('b', ''); 'Availability' -> (None, ...)."""
    s = s.strip()
    m = _LETTER.match(s)
    # A bare single word like "a" or "b" is a letter; a real word starting with a letter is not.
    if m and (m.group(2) is not None or len(re.sub(r"[^\w]", "", s)) == 1):
        return m.group(1).lower(), _norm(m.group(2) or "")
    return None, _norm(s)


def _bool_value(s: str) -> str | None:
    n = _norm(s)
    if n in _TRUE:
        return "true"
    if n in _FALSE:
        return "false"
    first = n.split(" ")[0] if n else ""
    return {"true": "true", "false": "false"}.get(first)


def grade_objective(student: str, reference: str, marks: float) -> ObjectiveResult:
    def result(correct: bool, conf: float, detail: str) -> ObjectiveResult:
        return ObjectiveResult(correct, marks if correct else 0.0, conf, detail)

    student = (student or "").strip()
    if not student:
        return result(False, 0.9, "No answer found.")

    ref_bool = _bool_value(reference)
    if ref_bool:
        stu_bool = _bool_value(student)
        if stu_bool is None:
            return result(False, 0.5, f"Could not read a true/false answer from '{student}'.")
        return result(stu_bool == ref_bool, 1.0, f"Expected {ref_bool}, read {stu_bool}.")

    ref_letter, ref_text = _split_option(reference)
    stu_letter, stu_text = _split_option(student)

    if ref_letter and not stu_letter and len(re.sub(r"[^\w]", "", student)) <= 2:
        return result(False, 0.4, f"Could not read an option letter from '{student}' - verify manually.")

    if ref_letter and stu_letter:
        if ref_letter == stu_letter:
            return result(True, 1.0, f"Option {stu_letter} matches.")
        if ref_text and stu_text and fuzz.ratio(ref_text, stu_text) >= LETTER_CONFLICT_TEXT_RATIO:
            return result(True, 0.7, "Option letter differs but option text matches (likely OCR misread).")
        return result(False, 0.9, f"Expected option {ref_letter}, read {stu_letter}.")

    a, b = (ref_text or _norm(reference)), (stu_text or _norm(student))
    ratio = max(fuzz.ratio(a, b), fuzz.ratio(a.replace(" ", ""), b.replace(" ", "")))
    if ratio >= ACCEPT_RATIO:
        return result(True, min(1.0, ratio / 100), f"Text match {ratio:.0f}%.")
    if ratio >= AMBIGUOUS_RATIO:
        return result(False, 0.5, f"Near match ({ratio:.0f}%) — verify manually.")
    return result(False, 0.9, f"Text match only {ratio:.0f}%.")
