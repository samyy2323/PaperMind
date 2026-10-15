"""Answer-key ingestion: PDF text extraction (PyMuPDF) and per-question parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass

import fitz  # PyMuPDF


def extract_pdf_text(data: bytes) -> str:
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text("text") for page in doc)


def extract_text_from_upload(filename: str, data: bytes) -> str:
    """PDF or plain-text/markdown upload -> text (used by the knowledge base)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return extract_pdf_text(data)
    return data.decode("utf-8", errors="replace")


@dataclass
class ParsedQuestion:
    number: int
    text: str
    reference_answer: str
    question_type: str
    marks: float
    extraction_confidence: float


_Q_START = re.compile(r"(?im)^[ \t]*(?:Q(?:uestion)?\.?[ \t]*)?(\d{1,3})[ \t]*[\.\):][ \t]+(?=\S)")
_ANSWER = re.compile(r"(?im)(?:^|\n)[ \t]*(?:ans(?:wer)?|solution|key|correct[ \t]+answer)[ \t]*[:\-–][ \t]*")
_MARKS = re.compile(r"[\[\(]\s*(\d+(?:\.\d+)?)\s*(?:marks?|m|pts?|points?)\s*[\]\)]|\b(\d+(?:\.\d+)?)\s*marks?\b", re.I)
_OPTION_REF = re.compile(r"^\(?[a-eA-E]\)?(?:[\s.:)\-]|$)")
_OPTIONS_IN_Q = re.compile(r"(?m)(?:^|\s)\(?[aA][\)\.]\s+\S.*\(?[bB][\)\.]\s+\S", re.S)
_BOOL = {"true", "false", "t", "f", "yes", "no"}


def _clean(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()


def _infer_type(question: str, reference: str) -> str:
    ref = reference.strip()
    words = re.findall(r"\w+", ref)
    if ref.lower().rstrip(".") in _BOOL or _OPTION_REF.match(ref):
        return "objective"
    if len(words) <= 4 and _OPTIONS_IN_Q.search(question):
        return "objective"
    return "subjective"


def parse_answer_key(text: str, total_marks: float) -> list[ParsedQuestion]:
    """Split extracted answer-key text into questions with reference answers.

    Questions are numbered 1., Q2), Question 3: ... Only markers that continue the running sequence
    (1, 2, 3 ...) start a new question, so numbered lists inside an answer are left alone. Within a
    question, text after an "Answer:" / "Ans -" label is the reference answer.
    """
    text = text.replace("\r\n", "\n")
    starts: list[tuple[int, int, int]] = []  # (number, match start, content start)
    expected = 1
    for m in _Q_START.finditer(text):
        if int(m.group(1)) == expected:
            starts.append((expected, m.start(), m.end()))
            expected += 1
    if not starts:
        raise ValueError("No numbered questions (1., Q2), Question 3: …) found in the PDF.")

    parsed: list[ParsedQuestion] = []
    for i, (num, _s, content_start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        block = text[content_start:end].strip()

        marks: float | None = None
        mm = _MARKS.search(block)
        if mm:
            marks = float(mm.group(1) or mm.group(2))
            block = (block[: mm.start()] + block[mm.end():]).strip()

        am = _ANSWER.search("\n" + block)
        if am:
            offset = len("\n")
            q_text = block[: max(0, am.start() - offset)]
            ref = block[am.end() - offset:]
            conf = 0.95
        else:
            q_text, ref, conf = block, "", 0.4  # no answer label: teacher must supply the reference answer
        q_text, ref = _clean_multiline(q_text), _clean_multiline(ref)
        parsed.append(
            ParsedQuestion(num, q_text, ref, _infer_type(q_text, ref), marks if marks is not None else -1, conf)
        )

    # Marks not written in the PDF are shared out from what remains of the exam total.
    known = sum(q.marks for q in parsed if q.marks >= 0)
    unknown = [q for q in parsed if q.marks < 0]
    if unknown:
        share = max(0.5, round((total_marks - known) / len(unknown) * 2) / 2)
        for q in unknown:
            q.marks = share
    return parsed


def _clean_multiline(s: str) -> str:
    lines = [_clean(l) for l in s.split("\n")]
    return "\n".join(l for l in lines if l).strip()
