"""Question segmentation: assign corrected OCR lines to exam questions.

The default heuristic looks for question numbers the student wrote ("1)", "Q2.", "Ans 3:"). If too few
markers are found it falls back to splitting the lines evenly in reading order. Implement the
``Segmenter`` protocol and register it in ``SEGMENTERS`` to swap the strategy.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SegLine:
    raw: str
    text: str  # corrected
    confidence: float


@dataclass
class Segment:
    lines: list[SegLine] = field(default_factory=list)
    method: str = "markers"

    @property
    def raw_text(self) -> str:
        return "\n".join(l.raw for l in self.lines)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    @property
    def confidence(self) -> float | None:
        if not self.lines:
            return None
        return sum(l.confidence for l in self.lines) / len(self.lines)


class Segmenter(Protocol):
    def segment(self, lines: list[SegLine], question_numbers: list[int]) -> dict[int, Segment]: ...


# Optional "Ans"/"Q" prefix, then a number in brackets/plain, then ) . : or -.  "3.5 times" is not a marker.
_MARKER = re.compile(
    r"^\s*(?:ans(?:wer)?\.?\s*)?(?:q(?:uestion|n)?\.?\s*)?[\(\[]?(\d{1,2}|[lI|])\s*(?:[\)\]:\-]|\.(?=\s|$))\s*",
    re.IGNORECASE,
)


def _marker_number(text: str) -> tuple[int, int] | None:
    m = _MARKER.match(text)
    if not m:
        return None
    tok = m.group(1)
    num = 1 if tok in ("l", "I", "|") else int(tok)  # OCR often reads "1" as l / I / |
    return num, m.end()


class MarkerSegmenter:
    def segment(self, lines: list[SegLine], question_numbers: list[int]) -> dict[int, Segment]:
        valid = set(question_numbers)
        result = {n: Segment() for n in question_numbers}
        if not lines:
            return result

        # Accept markers whose numbers strictly increase (ignores "1." bullets inside an answer).
        found: list[tuple[int, int, int]] = []  # (line index, question number, marker end)
        last = 0
        for i, line in enumerate(lines):
            hit = _marker_number(line.text)
            if hit and hit[0] in valid and hit[0] > last:
                found.append((i, hit[0], hit[1]))
                last = hit[0]

        if len(found) >= math.ceil(len(question_numbers) / 2):
            # Lines before the first recognised marker are usually the answer whose number OCR mangled
            # (e.g. "1)" read as "19 (") rather than a header: give them to the question just before it.
            first_line, first_num, _ = found[0]
            if first_line > 0 and first_num - 1 in valid:
                result[first_num - 1].lines.extend(lines[:first_line])
            for idx, (start, num, mend) in enumerate(found):
                end = found[idx + 1][0] if idx + 1 < len(found) else len(lines)
                seg = result[num]
                for j in range(start, end):
                    ln = lines[j]
                    if j == start:
                        rmark = _MARKER.match(ln.raw)
                        ln = SegLine(ln.raw[rmark.end():] if rmark else ln.raw, ln.text[mend:], ln.confidence)
                    if ln.text.strip() or ln.raw.strip():
                        seg.lines.append(ln)
            return result

        return PositionalSegmenter().segment(lines, question_numbers)


class PositionalSegmenter:
    """No reliable numbering: split lines into contiguous, near-equal groups in question order."""

    def segment(self, lines: list[SegLine], question_numbers: list[int]) -> dict[int, Segment]:
        n = len(question_numbers)
        result = {q: Segment(method="positional") for q in question_numbers}
        if n == 0:
            return result
        size = max(len(lines) / n, 1e-9)
        for i, line in enumerate(lines):
            result[question_numbers[min(n - 1, int(i // size))]].lines.append(line)
        return result


SEGMENTERS: dict[str, Segmenter] = {"markers": MarkerSegmenter(), "positional": PositionalSegmenter()}


def get_segmenter(name: str = "markers") -> Segmenter:
    return SEGMENTERS[name]
