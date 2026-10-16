import fitz
import pytest

from app.services.ingestion.answer_key import extract_pdf_text, parse_answer_key
from app.services.rag.chunking import chunk_text
from app.services.segmentation import MarkerSegmenter, PositionalSegmenter, SegLine

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
