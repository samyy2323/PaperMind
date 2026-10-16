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
    def segment(self, lines: list[SegLine], question_numbers: list[int], question_types: dict[int, str] | None = None) -> dict[int, Segment]: ...


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


# A line that is only a (possibly garbled) number followed by ONE character: "3 . B.", "d", "20 c", "S. B.", "is . B.".
_ANSWER_LINE = re.compile(r"^\W*(?:(?P<num>[A-Za-z0-9]{1,3})\W+)?(?P<ans>[A-Za-z0-9])\W*$")
_GLUED = re.compile(r"^\s*(\d{1,2})(?=[A-Za-z]{3})")  # "21supervised": the OCR lost the dot after the number
_SPACED = re.compile(r"^\s*(\d{1,2})\s+(?=[A-Za-z]{2})")  # "25 list out ...": only a space after the number

SKIP_COST, DROP_COST, WRONG_NUM_COST, UNKNOWN_NUM_COST = 0.6, 1.5, 1.0, 0.3


def _hint(num: str | None, valid: set[int]) -> int | None:
    return int(num) if num and num.isdigit() and int(num) in valid else None


def align_objective(candidates: list[int | None], numbers: list[int]) -> dict[int, int]:
    """Monotone alignment of answer-line candidates (hints = the number the OCR read, or None) to question numbers.

    Garbled numbers ("is" for 15, "49" for 19, "S" for 5) are repaired from the ORDER of the lines; a question the
    student skipped, or a stray line, costs a little. Returns {candidate index: question number}.
    """
    m, n = len(candidates), len(numbers)
    INF = float("inf")
    dp = [[INF] * (n + 1) for _ in range(m + 1)]
    back: dict[tuple[int, int], tuple[str, int, int]] = {}
    dp[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            cur = dp[i][j]
            if cur == INF:
                continue
            if i < m and j < n:  # candidate i answers question j
                hint = candidates[i]
                cost = 0.0 if hint == numbers[j] else (UNKNOWN_NUM_COST if hint is None else WRONG_NUM_COST)
                if cur + cost < dp[i + 1][j + 1]:
                    dp[i + 1][j + 1], back[(i + 1, j + 1)] = cur + cost, ("match", i, j)
            if j < n and cur + SKIP_COST < dp[i][j + 1]:  # question j was not answered
                dp[i][j + 1], back[(i, j + 1)] = cur + SKIP_COST, ("skip", i, j)
            if i < m and cur + DROP_COST < dp[i + 1][j]:  # candidate i is a stray line
                dp[i + 1][j], back[(i + 1, j)] = cur + DROP_COST, ("drop", i, j)
    mapping, i, j = {}, m, n
    while (i, j) != (0, 0):
        kind, pi, pj = back[(i, j)]
        if kind == "match":
            mapping[pi] = numbers[pj]
        i, j = pi, pj
    return mapping


def _answer_block(lines: list[SegLine], valid: set[int]) -> list[tuple[int, int | None]]:
    """(line index, number hint) of the first dense run of 'number + single character' lines (the MCQ block)."""
    hits = []
    for i, ln in enumerate(lines):
        m = _ANSWER_LINE.match(ln.text.strip())
        if m:
            hits.append((i, _hint(m.group("num"), valid)))
    if not hits:
        return []
    block = [hits[0]]
    for h in hits[1:]:
        if h[0] - block[-1][0] <= 3:
            block.append(h)
        elif len(block) >= 3:
            break
        else:
            block = [h]
    return block if len(block) >= 3 else []


class MarkerSegmenter:
    def segment(self, lines: list[SegLine], question_numbers: list[int], question_types: dict[int, str] | None = None) -> dict[int, Segment]:
        valid = set(question_numbers)
        result = {n: Segment() for n in question_numbers}
        if not lines:
            return result
        owner: dict[int, int] = {}  # line index -> question number whose answer starts on that line
        consumed: set[int] = set()

        # ---- 1. the objective (MCQ) block: aligned by order, numbers used only as hints --------------------------
        objective = sorted(n for n in question_numbers if (question_types or {}).get(n) == "objective")
        block_end = -1
        if objective:
            block = _answer_block(lines, set(objective))
            if len(block) >= max(3, len(objective) // 2):
                mapping = align_objective([h for _, h in block], objective)
                for ci, qn in mapping.items():
                    idx = block[ci][0]
                    owner[idx] = qn
                    consumed.add(idx)
                block_end = block[-1][0]

        # ---- 2. the remaining questions: explicit numbers, any order, each used once ------------------------------
        seen = set(owner.values())
        marker_len: dict[int, int] = {}
        for i, line in enumerate(lines):
            if i <= block_end:
                continue
            num = end = None
            mk = _marker_number(line.text)
            if mk:
                num, end = mk
            else:
                g = _GLUED.match(line.text) or _SPACED.match(line.text)
                if g:
                    num, end = int(g.group(1)), g.end()
            if num in valid and num not in seen:
                owner[i], marker_len[i] = num, end
                seen.add(num)
        if not owner:
            return PositionalSegmenter().segment(lines, question_numbers, question_types)

        # ---- 3. lines belong to the most recent question that started above them ----------------------------------
        order = sorted(owner)
        # lines above the first marker are usually the answer whose number OCR mangled ("1)" read as "19 (")
        prev = owner[order[0]] - 1
        if order[0] > 0 and prev in valid and prev not in seen:
            result[prev].lines.extend(lines[: order[0]])
        for k, start in enumerate(order):
            end = order[k + 1] if k + 1 < len(order) else len(lines)
            seg = result[owner[start]]
            for j in range(start, end):
                ln = lines[j]
                if j == start and start not in consumed and start in marker_len:
                    rmark = _MARKER.match(ln.raw) or _GLUED.match(ln.raw) or _SPACED.match(ln.raw)
                    ln = SegLine(ln.raw[rmark.end():] if rmark else ln.raw, ln.text[marker_len[start]:], ln.confidence)
                elif j == start and start in consumed:  # MCQ line: keep just the answer character
                    m = _ANSWER_LINE.match(ln.text.strip())
                    ln = SegLine(m.group("ans") if m else ln.raw, m.group("ans") if m else ln.text, ln.confidence)
                elif start in consumed:
                    continue  # noise between MCQ lines is not part of any answer
                if ln.text.strip() or ln.raw.strip():
                    seg.lines.append(ln)
        return result


class PositionalSegmenter:
    """No reliable numbering: split lines into contiguous, near-equal groups in question order."""

    def segment(self, lines: list[SegLine], question_numbers: list[int], question_types: dict[int, str] | None = None) -> dict[int, Segment]:
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
