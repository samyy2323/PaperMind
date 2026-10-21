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

import numpy as np


@dataclass
class SegLine:
    raw: str
    text: str  # corrected
    confidence: float
    box: tuple[float, float, float, float] | None = None

    @property
    def x0(self) -> float | None:
        return self.box[0] if self.box is not None else None

    @property
    def y0(self) -> float | None:
        return self.box[1] if self.box is not None else None

    @property
    def x1(self) -> float | None:
        return self.box[2] if self.box is not None else None

    @property
    def y1(self) -> float | None:
        return self.box[3] if self.box is not None else None


@dataclass
class PageGeometry:
    x_margin: float
    median_line_height: float
    indent_threshold: float
    vertical_gap_threshold: float


def estimate_page_geometry(lines: list[SegLine]) -> PageGeometry | None:
    """Estimate left margin baseline and line height / vertical gap thresholds.

    Returns None if fewer than 2 lines have bounding box coordinates.
    """
    boxes = [l.box for l in lines if l.box is not None]
    if len(boxes) < 2:
        return None
    x0s = [b[0] for b in boxes]
    heights = [max(b[3] - b[1], 1.0) for b in boxes]

    x_margin = float(np.percentile(x0s, 10))
    med_h = float(np.median(heights))

    # Indentation threshold: ~75% of line height or at least 25px
    indent_thr = max(25.0, 0.75 * med_h)
    # Significant vertical gap between separate answers / paragraphs: ~1.35x line height
    gap_thr = max(18.0, 1.35 * med_h)

    return PageGeometry(
        x_margin=x_margin,
        median_line_height=med_h,
        indent_threshold=indent_thr,
        vertical_gap_threshold=gap_thr,
    )


def is_line_indented(line: SegLine, geom: PageGeometry | None) -> bool:
    """Return True if line is indented relative to the left margin column."""
    if geom is None or line.x0 is None:
        return False
    return (line.x0 - geom.x_margin) >= geom.indent_threshold


def has_vertical_gap(prev_line: SegLine, curr_line: SegLine, geom: PageGeometry | None) -> bool:
    """Return True if a significant vertical gap exists between prev_line and curr_line."""
    if geom is None or prev_line.y1 is None or curr_line.y0 is None:
        return False
    gap = curr_line.y0 - prev_line.y1
    return gap >= geom.vertical_gap_threshold


@dataclass
class Segment:
    lines: list[SegLine] = field(default_factory=list)
    method: str = "markers"
    match_score: float | None = None

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
    def segment(
        self,
        lines: list[SegLine],
        question_numbers: list[int],
        question_types: dict[int, str] | None = None,
        question_contexts: dict[int, str] | None = None,
        embedder: object | None = None,
    ) -> dict[int, Segment]: ...


# Optional "Ans"/"Q" prefix, then a number in brackets/plain, then ) . : or -.  "3.5 times" is not a marker.
_MARKER = re.compile(
    r"^\s*(?:ans(?:wer)?\.?\s*)?(?:q(?:uestion|n)?\.?\s*)?[\(\[]?(\d{1,2}|[lI|])\s*(?:[\)\]:\-]|\.(?=\s|$))\s*",
    re.IGNORECASE,
)

# Potential or garbled marker pattern: matches garbled numbers ("38.", "49)"), hash/asterisk headers ("# ", "* "),
# or generic prefixes ("Ans:", "Q.") that indicate a student began an answer block.
_POTENTIAL_MARKER = re.compile(
    r"^\s*(?:(?:ans(?:wer)?|q(?:uestion|n)?)\.?\s*[:\.\-\)]\s*|(?:ans(?:wer)?\.?\s*)?(?:q(?:uestion|n)?\.?\s*)?[\(\[]?([A-Za-z0-9#*~lI|]{1,4})\s*(?:[\)\]:\-]|\.(?=\s|$))\s*|[#*~]\s+)",
    re.IGNORECASE,
)


def _strip_pseudo_marker(text: str) -> str:
    """Remove corrupted leading question markers (e.g. '38. ', '# ', 'Ans: ') from answer text."""
    m = _POTENTIAL_MARKER.match(text)
    if m:
        return text[m.end():]
    return text


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


def match_orphan_blocks_by_content(
    orphan_blocks: list[list[SegLine]],
    unassigned_questions: list[int],
    question_contexts: dict[int, str],
    embedder: object,
    threshold: float = 0.35,
) -> dict[int, tuple[list[SegLine], float, int]]:
    """Match orphan line blocks to unassigned questions using dense semantic similarity.

    Returns {question_number: (cleaned_lines, similarity_score, orphan_block_index)}.
    Only matches with similarity >= threshold are accepted.
    Uses greedy bipartite matching (highest similarity first, 1-to-1).
    """
    if not orphan_blocks or not unassigned_questions or not question_contexts or embedder is None:
        return {}

    cleaned_blocks: list[list[SegLine]] = []
    block_texts: list[str] = []
    valid_block_indices: list[int] = []

    for idx, b in enumerate(orphan_blocks):
        if not b:
            continue
        first = b[0]
        cleaned_first = SegLine(
            raw=_strip_pseudo_marker(first.raw),
            text=_strip_pseudo_marker(first.text),
            confidence=first.confidence,
        )
        cleaned_b = [cleaned_first] + list(b[1:])
        text = "\n".join(l.text for l in cleaned_b if l.text.strip()).strip()
        if text:
            cleaned_blocks.append(cleaned_b)
            block_texts.append(text)
            valid_block_indices.append(idx)

    if not block_texts:
        return {}

    valid_questions = [q for q in unassigned_questions if question_contexts.get(q, "").strip()]
    if not valid_questions:
        return {}
    q_texts = [question_contexts[q].strip() for q in valid_questions]

    b_vecs = np.asarray(embedder.encode(block_texts))
    q_vecs = np.asarray(embedder.encode(q_texts))
    if b_vecs.ndim == 1:
        b_vecs = b_vecs.reshape(1, -1)
    if q_vecs.ndim == 1:
        q_vecs = q_vecs.reshape(1, -1)

    # Compute cosine similarities
    b_norm = np.linalg.norm(b_vecs, axis=1, keepdims=True)
    b_norm[b_norm == 0] = 1.0
    b_unit = b_vecs / b_norm

    q_norm = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    q_unit = q_vecs / q_norm

    sim_matrix = np.matmul(b_unit, q_unit.T)  # shape: (len(block_texts), len(valid_questions))

    candidates: list[tuple[float, int, int]] = []
    for bi in range(len(block_texts)):
        for qi in range(len(valid_questions)):
            score = float(sim_matrix[bi, qi])
            if score >= threshold:
                candidates.append((score, bi, qi))

    # Highest similarity first
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched: dict[int, tuple[list[SegLine], float, int]] = {}
    matched_blocks: set[int] = set()
    matched_questions: set[int] = set()

    for score, bi, qi in candidates:
        orig_bi = valid_block_indices[bi]
        q_num = valid_questions[qi]
        if orig_bi not in matched_blocks and q_num not in matched_questions:
            matched_blocks.add(orig_bi)
            matched_questions.add(q_num)
            matched[q_num] = (cleaned_blocks[bi], score, orig_bi)

    return matched


class MarkerSegmenter:
    def segment(
        self,
        lines: list[SegLine],
        question_numbers: list[int],
        question_types: dict[int, str] | None = None,
        question_contexts: dict[int, str] | None = None,
        embedder: object | None = None,
    ) -> dict[int, Segment]:
        valid = set(question_numbers)
        result = {n: Segment() for n in question_numbers}
        if not lines:
            return result
        geom = estimate_page_geometry(lines)
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
                # If geometry is available and this line is indented inside an active answer without a gap,
                # do not mistake it for a top-level question marker.
                if is_line_indented(line, geom) and owner and not has_vertical_gap(lines[i - 1], line, geom):
                    continue
                owner[i], marker_len[i] = num, end
                seen.add(num)

        # Check if we should attempt orphan block semantic matching
        can_content_match = (
            question_contexts is not None
            and embedder is not None
            and len(question_contexts) > 0
            and len(seen) < len(question_numbers)
        )

        if not owner and not can_content_match:
            return PositionalSegmenter().segment(lines, question_numbers, question_types, question_contexts, embedder)

        if not can_content_match:
            # Standard marker-based partition (all markers found, or no embedder/contexts provided)
            order = sorted(owner)
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
                        ln = SegLine(ln.raw[rmark.end():] if rmark else ln.raw, ln.text[marker_len[start]:], ln.confidence, box=ln.box)
                    elif j == start and start in consumed:  # MCQ line: keep just the answer character
                        m = _ANSWER_LINE.match(ln.text.strip())
                        ln = SegLine(m.group("ans") if m else ln.raw, m.group("ans") if m else ln.text, ln.confidence, box=ln.box)
                    elif start in consumed:
                        continue  # noise between MCQ lines is not part of any answer
                    if ln.text.strip() or ln.raw.strip():
                        seg.lines.append(ln)
            return result

        # ---- 3. Content-aware partition with orphan block extraction --------------------------------------
        # Identify potential marker positions for orphan lines
        pot_starts: dict[int, int] = {}
        for i in range(block_end + 1, len(lines)):
            if i in owner:
                continue
            line = lines[i]
            prev_line = lines[i - 1] if i > 0 else None
            # If line is indented and not preceded by a vertical gap, it's an internal continuation/bullet
            if is_line_indented(line, geom) and not (prev_line and has_vertical_gap(prev_line, line, geom)):
                continue

            pmk = _POTENTIAL_MARKER.match(line.text)
            if pmk:
                pot_starts[i] = pmk.end()
            else:
                mk = _marker_number(line.text)
                if mk:
                    pot_starts[i] = mk[1]
                else:
                    g = _GLUED.match(line.text) or _SPACED.match(line.text)
                    if g:
                        pot_starts[i] = g.end()
                    elif prev_line and has_vertical_gap(prev_line, line, geom) and not is_line_indented(line, geom):
                        # Geometric answer boundary: margin-aligned line after a vertical gap
                        pot_starts[i] = 0

        all_splits = sorted(set(owner.keys()) | set(pot_starts.keys()))
        if not all_splits:
            return PositionalSegmenter().segment(lines, question_numbers, question_types, question_contexts, embedder)

        # Lines before the first split
        first_split = all_splits[0]
        if first_split > block_end + 1:
            pre_lines = lines[block_end + 1 : first_split]
            if first_split in owner:
                prev = owner[first_split] - 1
                if prev in valid and prev not in seen:
                    result[prev].lines.extend(pre_lines)
            elif pre_lines:
                pot_starts[block_end + 1] = 0
                all_splits = [block_end + 1] + all_splits

        # Partition into valid segments and orphan blocks
        orphan_blocks: list[dict] = []
        cur_valid_qn: int | None = None

        for k, start in enumerate(all_splits):
            end = all_splits[k + 1] if k + 1 < len(all_splits) else len(lines)
            chunk = lines[start:end]
            if not chunk:
                continue

            if start in owner:
                qn = owner[start]
                cur_valid_qn = qn
                seg = result[qn]
                for j, ln in enumerate(chunk):
                    if j == 0 and start not in consumed and start in marker_len:
                        rmark = _MARKER.match(ln.raw) or _GLUED.match(ln.raw) or _SPACED.match(ln.raw)
                        ln = SegLine(ln.raw[rmark.end():] if rmark else ln.raw, ln.text[marker_len[start]:], ln.confidence, box=ln.box)
                    elif j == 0 and start in consumed:
                        m = _ANSWER_LINE.match(ln.text.strip())
                        ln = SegLine(m.group("ans") if m else ln.raw, m.group("ans") if m else ln.text, ln.confidence, box=ln.box)
                    elif start in consumed:
                        continue
                    if ln.text.strip() or ln.raw.strip():
                        seg.lines.append(ln)
            else:
                orphan_blocks.append({
                    "lines": chunk,
                    "preceding_qn": cur_valid_qn,
                })

        # Match orphan blocks against unassigned questions
        unassigned = [q for q in question_numbers if not result[q].lines]
        matched: dict[int, tuple[list[SegLine], float, int]] = {}
        if orphan_blocks and unassigned and question_contexts and embedder:
            matched = match_orphan_blocks_by_content(
                [ob["lines"] for ob in orphan_blocks],
                unassigned,
                question_contexts,
                embedder,
            )

        matched_by_idx = {data[2]: (q_num, data[0], data[1]) for q_num, data in matched.items()}

        for idx, ob in enumerate(orphan_blocks):
            if idx in matched_by_idx:
                q_num, clines, score = matched_by_idx[idx]
                result[q_num] = Segment(
                    lines=[ln for ln in clines if ln.text.strip() or ln.raw.strip()],
                    method="content_match",
                    match_score=score,
                )
            elif ob["preceding_qn"] is not None:
                result[ob["preceding_qn"]].lines.extend(
                    [ln for ln in ob["lines"] if ln.text.strip() or ln.raw.strip()]
                )

        if not owner and not any(s.lines for s in result.values()):
            return PositionalSegmenter().segment(lines, question_numbers, question_types, question_contexts, embedder)

        return result


class PositionalSegmenter:
    """No reliable numbering: split lines into contiguous, near-equal groups in question order."""

    def segment(
        self,
        lines: list[SegLine],
        question_numbers: list[int],
        question_types: dict[int, str] | None = None,
        question_contexts: dict[int, str] | None = None,
        embedder: object | None = None,
    ) -> dict[int, Segment]:
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

