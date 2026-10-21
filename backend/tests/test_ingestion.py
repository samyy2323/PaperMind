import fitz
import numpy as np
import pytest

from app.services.ingestion.answer_key import extract_pdf_text, parse_answer_key
from app.services.rag.chunking import chunk_text
from app.services.segmentation import (
    MarkerSegmenter,
    PositionalSegmenter,
    SegLine,
    _strip_pseudo_marker,
    estimate_page_geometry,
    has_vertical_gap,
    is_line_indented,
    match_orphan_blocks_by_content,
)

KEY_TEXT = """Object-Oriented Programming - Answer Key

1. Which feature lets a subclass replace a superclass method? (a) Encapsulation (b) Method overriding (c) Abstraction [2 marks]
Answer: (b) Method overriding

2. An interface can contain constructors. True or False? [2 marks]
Answer: False

3. Explain the Liskov Substitution Principle. [6 marks]
Answer: Objects of a superclass must be replaceable with objects of a subclass without breaking the program.
For example, a Square that extends Rectangle violates it.
1. Fix: use a shared Shape abstraction.

4. Explain encapsulation.
Ans: Encapsulation bundles data and methods in a class and restricts direct access to internal state.
"""


def make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(40, 40, 555, 800), text, fontsize=10)
    return doc.tobytes()


def test_answer_key_parsing_from_real_pdf():
    qs = parse_answer_key(extract_pdf_text(make_pdf(KEY_TEXT)), total_marks=20)
    assert [q.number for q in qs] == [1, 2, 3, 4]  # the "1. Fix" list item inside Q3 is not a new question
    assert qs[0].question_type == "objective" and qs[0].reference_answer.startswith("(b) Method overriding")
    assert qs[1].question_type == "objective" and qs[1].reference_answer == "False"
    assert qs[2].question_type == "subjective" and "Square" in qs[2].reference_answer and "Shape" in qs[2].reference_answer
    assert [q.marks for q in qs[:3]] == [2, 2, 6]
    assert qs[3].marks == 10  # unlabelled question gets what remains of the total
    assert "[2 marks]" not in qs[0].text


def test_answer_key_without_answer_label_is_low_confidence():
    qs = parse_answer_key("1. Define a class.\n2. Define an object.", total_marks=10)
    assert all(q.extraction_confidence < 0.6 and q.reference_answer == "" for q in qs)
    assert [q.marks for q in qs] == [5, 5]


def test_answer_key_without_questions_raises():
    with pytest.raises(ValueError):
        parse_answer_key("Just some prose with no numbering.", 10)


def test_chunking_targets_paragraph_groups():
    para = "This is a sentence about object oriented programming. " * 6
    text = "\n\n".join([para] * 6)
    chunks = chunk_text(text, target_chars=700)
    assert 2 <= len(chunks) <= 6 and all(len(c) < 1400 for c in chunks)
    assert chunk_text("tiny", 700) == []


def _lines(*texts):
    return [SegLine(raw=t, text=t, confidence=0.9) for t in texts]


def test_segmentation_by_markers_strips_markers_and_preamble():
    lines = _lines("Name: Rohan", "1) (b)", "2) True", "3. LSP says subclasses must", "be substitutable.", "Q4: Encapsulation hides data")
    seg = MarkerSegmenter().segment(lines, [1, 2, 3, 4])
    assert seg[1].text == "(b)" and seg[2].text == "True"
    assert seg[3].text == "LSP says subclasses must\nbe substitutable."
    assert seg[4].text == "Encapsulation hides data"
    assert "Rohan" not in "".join(s.text for s in seg.values())


def test_segmentation_reads_l_as_one_and_ignores_decimals_and_bullets():
    lines = _lines("l) (b)", "2) The value is", "3.5 times larger", "1. bullet inside answer 2", "3) third")
    seg = MarkerSegmenter().segment(lines, [1, 2, 3])
    assert seg[1].text == "(b)"
    assert "3.5 times larger" in seg[2].text and "bullet inside" in seg[2].text
    assert seg[3].text == "third"


def test_segmentation_falls_back_to_position_without_numbers():
    lines = _lines("a", "b", "c", "d")
    seg = MarkerSegmenter().segment(lines, [1, 2])
    assert seg[1].method == "positional" and [l.text for l in seg[1].lines] == ["a", "b"] and [l.text for l in seg[2].lines] == ["c", "d"]
    assert PositionalSegmenter().segment([], [1, 2])[1].lines == []


def test_segmentation_keeps_lines_before_first_marker_when_number_was_mangled():
    # OCR read "1) (b)" as "19 ( 0b ) ." -> no marker for Q1; those lines must not be lost
    lines = _lines("19 ( 0b ) .", "2) False", "3) Liskov says", "4) Encapsulation hides")
    seg = MarkerSegmenter().segment(lines, [1, 2, 3, 4])
    assert seg[1].text == "19 ( 0b ) ." and seg[2].text == "False"
    # ...but a header before Q1's own marker is still dropped
    seg = MarkerSegmenter().segment(_lines("Name: Rohan", "1) (b)", "2) True"), [1, 2])
    assert seg[1].text == "(b)"


def test_segmentation_handles_answers_written_out_of_order():
    lines = _lines("22. Comment on Model M1: needs better data", "more text for 22", "24. Hard to find datasets",
                   "31. k is the pre-defined cluster", "29. Normalization is required", "23. Regression analysis is")
    seg = MarkerSegmenter().segment(lines, [22, 23, 24, 29, 31])
    assert seg[22].text.startswith("Comment on Model M1") and "more text for 22" in seg[22].text
    assert seg[24].text == "Hard to find datasets" and seg[31].text == "k is the pre-defined cluster"
    assert seg[29].text == "Normalization is required" and seg[23].text == "Regression analysis is"


# ---- real OCR lines from a scanned sheet of the public Mendeley dataset (Student_1) ---------------------------------
def _real_sheet():
    import json
    from pathlib import Path

    rows = json.loads((Path(__file__).parent / "fixtures" / "mendeley_student1_ocr.json").read_text())
    return [SegLine(raw=r["raw"], text=r["raw"], confidence=r["confidence"]) for r in rows]


def test_real_sheet_mcq_block_is_aligned_despite_garbled_numbers():
    types = {n: "objective" if n <= 20 else "subjective" for n in range(1, 36)}
    seg = MarkerSegmenter().segment(_real_sheet(), list(range(1, 36)), types)
    letters = {n: seg[n].text.strip().upper() for n in range(1, 21)}
    truth = "D C B A B B B D D D B C B C B C B C B C".split()  # what Student_1 wrote (teacher's CSV)
    # OCR read "5" as S, "11" as 19, "15" as "is", "19" as 49, and lost the letter of Q10 - order repairs all of those
    assert [letters[n] for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)] == \
           [truth[n - 1] for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)]
    assert letters[10] == "3"  # unreadable, but still assigned to Q10 (the grader will flag it)


def test_real_sheet_short_answers_follow_their_own_numbers():
    types = {n: "objective" if n <= 20 else "subjective" for n in range(1, 36)}
    seg = MarkerSegmenter().segment(_real_sheet(), list(range(1, 36)), types)
    assert seg[21].method == "markers"
    assert "learning" in seg[21].text.lower() and "supervised" in seg[21].text.lower()  # "21supervised": glued number
    assert "Model M1" in seg[22].text or "model m1" in seg[22].text.lower()
    assert "challenges" in seg[24].text.lower()
    assert "normalization" in seg[29].text.lower()
    assert "regression" in seg[23].text.lower()


def test_number_followed_only_by_a_space_is_a_marker():
    lines = _lines("21. Define it", "answer text", "25 list out any 2 major applications", "- finance", "26. distinguish")
    types = {n: "subjective" for n in range(21, 27)}
    seg = MarkerSegmenter().segment(lines, list(range(21, 27)), types)
    assert seg[25].text.startswith("list out any 2 major applications") and "finance" in seg[25].text
    assert seg[21].text == "Define it\nanswer text" and seg[26].text == "distinguish"


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]], default: list[float] | None = None):
        self.mapping = mapping
        self.default = default or [0.0, 0.0, 0.0]

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            matched_vec = None
            for key, vec in self.mapping.items():
                if key.lower() in t.lower():
                    matched_vec = vec
                    break
            vecs.append(matched_vec or self.default)
        return np.array(vecs, dtype="float32")


def test_strip_pseudo_marker_cleans_garbled_prefixes():
    assert _strip_pseudo_marker("38. Precision is TP/(TP+FP)") == "Precision is TP/(TP+FP)"
    assert _strip_pseudo_marker("# Recall is TP/(TP+FN)") == "Recall is TP/(TP+FN)"
    assert _strip_pseudo_marker("Ans: F1 score is harmonic mean") == "F1 score is harmonic mean"
    assert _strip_pseudo_marker("Q. Normalization scales features") == "Normalization scales features"
    assert _strip_pseudo_marker("Supervised learning uses labels") == "Supervised learning uses labels"


def test_match_orphan_blocks_by_content_matches_above_threshold():
    embedder = FakeEmbedder({
        "precision": [1.0, 0.0, 0.0],
        "recall": [0.0, 1.0, 0.0],
    })
    blocks = [
        _lines("38. Precision is calculated using true positives", "over predicted positives"),
        _lines("# Recall is calculated using true positives", "over actual positives"),
    ]
    unassigned = [33, 34]
    contexts = {
        33: "What is precision?\nPrecision is true positives divided by total predicted positives.",
        34: "What is recall?\nRecall is true positives divided by total actual positives.",
    }
    matches = match_orphan_blocks_by_content(blocks, unassigned, contexts, embedder, threshold=0.35)
    assert 33 in matches and 34 in matches
    assert matches[33][1] >= 0.35
    assert matches[34][1] >= 0.35
    # First line pseudo-markers are stripped
    assert matches[33][0][0].text.startswith("Precision is")
    assert matches[34][0][0].text.startswith("Recall is")


def test_segmentation_content_matching_recovers_garbled_numbers():
    # Student sheet where Q33 was OCR'd as "38." and Q34 was OCR'd as "#"
    lines = _lines(
        "31. Classification predicts discrete classes",
        "32. Regression predicts continuous values",
        "38. Precision is true positives over predicted positives",
        "it measures classifier exactness",
        "# Recall is true positives over actual positives",
        "it measures classifier completeness",
        "35. F1-score balances both precision and recall",
    )
    embedder = FakeEmbedder({
        "precision": [1.0, 0.0, 0.0],
        "recall": [0.0, 1.0, 0.0],
    })
    contexts = {
        31: "Classification definition",
        32: "Regression definition",
        33: "What is precision in machine learning?",
        34: "What is recall in machine learning?",
        35: "What is F1-score?",
    }
    seg = MarkerSegmenter().segment(
        lines,
        [31, 32, 33, 34, 35],
        question_contexts=contexts,
        embedder=embedder,
    )
    assert seg[31].method == "markers"
    assert seg[32].method == "markers"
    assert seg[33].method == "content_match"
    assert seg[33].match_score is not None and seg[33].match_score >= 0.35
    assert seg[33].text.startswith("Precision is true positives")
    assert "measures classifier exactness" in seg[33].text

    assert seg[34].method == "content_match"
    assert seg[34].match_score is not None and seg[34].match_score >= 0.35
    assert seg[34].text.startswith("Recall is true positives")
    assert "measures classifier completeness" in seg[34].text

    assert seg[35].method == "markers"
    assert "balances both" in seg[35].text


def test_orphan_blocks_stay_with_preceding_question_when_unmatched():
    # Stray line with low similarity should stay attached to previous answer
    lines = _lines(
        "31. Classification predicts discrete classes",
        "38. Irrelevant noise about weather and sports",
        "32. Regression predicts continuous values",
    )
    embedder = FakeEmbedder({"unrelated": [1.0, 0.0, 0.0]}, default=[0.0, 0.0, 0.0])
    contexts = {
        31: "Classification definition",
        32: "Regression definition",
        33: "What is deep learning?",
    }
    seg = MarkerSegmenter().segment(
        lines,
        [31, 32, 33],
        question_contexts=contexts,
        embedder=embedder,
    )
    # 38 was not matched to 33, so its lines stay attached to Q31
    assert "Irrelevant noise" in seg[31].text
    assert seg[33].lines == []


def test_geometry_margin_and_gap_estimation():
    lines = [
        SegLine("Q1", "Q1", 0.9, box=(80.0, 100.0, 400.0, 130.0)),
        SegLine("Body text line 1", "Body text line 1", 0.9, box=(82.0, 135.0, 450.0, 165.0)),
        SegLine("Indented bullet", "Indented bullet", 0.9, box=(140.0, 170.0, 380.0, 200.0)),
        SegLine("Q2 after gap", "Q2 after gap", 0.9, box=(78.0, 260.0, 420.0, 290.0)),
    ]
    geom = estimate_page_geometry(lines)
    assert geom is not None
    assert 78.0 <= geom.x_margin <= 82.0
    assert 25.0 <= geom.median_line_height <= 35.0

    # Line 2 at x0=140 is indented
    assert is_line_indented(lines[2], geom) is True
    # Line 1 at x0=82 is not indented
    assert is_line_indented(lines[1], geom) is False

    # Gap between line 1 (y1=165) and line 2 (y0=170) is 5px (no vertical gap)
    assert has_vertical_gap(lines[1], lines[2], geom) is False
    # Gap between line 2 (y1=200) and line 3 (y0=260) is 60px (vertical gap)
    assert has_vertical_gap(lines[2], lines[3], geom) is True


def test_indented_numbered_sub_item_does_not_split_or_claim_question():
    # Inside Q3 answer, student writes an indented numbered list "1. Fix..." and "2. Subclass..."
    # Even though Q1 is unassigned, geometry prevents these indented items from stealing Q1
    lines = [
        SegLine("3. Explain LSP", "3. Explain LSP", 0.9, box=(80.0, 100.0, 400.0, 130.0)),
        SegLine("Subclasses must be substitutable.", "Subclasses must be substitutable.", 0.9, box=(80.0, 135.0, 500.0, 165.0)),
        SegLine("1. Fix using Shape abstraction.", "1. Fix using Shape abstraction.", 0.9, box=(150.0, 170.0, 450.0, 200.0)),
        SegLine("2. Subclass Shape instead.", "2. Subclass Shape instead.", 0.9, box=(150.0, 205.0, 450.0, 235.0)),
        SegLine("4. Encapsulation hides state.", "4. Encapsulation hides state.", 0.9, box=(80.0, 290.0, 460.0, 320.0)),
    ]
    seg = MarkerSegmenter().segment(lines, [1, 2, 3, 4])
    # Q3 retains the indented list items
    assert "1. Fix using Shape" in seg[3].text
    assert "2. Subclass Shape" in seg[3].text
    # Q1 and Q2 were not stolen by the indented items
    assert seg[1].lines == []
    assert seg[2].lines == []
    assert seg[4].text.startswith("Encapsulation hides state")


def test_margin_aligned_line_with_vertical_gap_recovers_missing_question_number():
    # Student wrote Q33 at the left margin after a vertical gap, but OCR missed the number completely
    lines = [
        SegLine("32. Regression predicts continuous values", "32. Regression predicts continuous values", 0.9, box=(80.0, 100.0, 500.0, 130.0)),
        SegLine("It uses linear or polynomial models.", "It uses linear or polynomial models.", 0.9, box=(80.0, 135.0, 480.0, 165.0)),
        # Big vertical gap (y0=240 vs y1=165), at left margin (x0=80), but NO question number in text:
        SegLine("Precision is true positives divided by total predicted.", "Precision is true positives divided by total predicted.", 0.9, box=(80.0, 240.0, 520.0, 270.0)),
        SegLine("It measures exactness of classifications.", "It measures exactness of classifications.", 0.9, box=(80.0, 275.0, 490.0, 305.0)),
    ]
    embedder = FakeEmbedder({"precision": [1.0, 0.0, 0.0]})
    contexts = {
        32: "Regression definition",
        33: "What is precision in machine learning?",
    }
    seg = MarkerSegmenter().segment(lines, [32, 33], question_contexts=contexts, embedder=embedder)
    assert seg[32].method == "markers"
    assert "linear or polynomial" in seg[32].text

    # Q33 was recovered via geometric gap + margin alignment + MiniLM semantic matching!
    assert seg[33].method == "content_match"
    assert seg[33].match_score is not None and seg[33].match_score >= 0.35
    assert "Precision is true positives" in seg[33].text


