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
