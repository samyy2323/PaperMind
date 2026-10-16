"""Pure-logic tests for the correction stage and the training data helpers (no models, no GPU)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "correction"))

from app.services.correction import apply_guard  # noqa: E402
from pm_common import cer, fact_swap, fragment, group_bucket, split_of, split_sentences  # noqa: E402
from pm_noise import simulate  # noqa: E402


def test_guard_accepts_small_repairs_including_mangled_markers():
    assert apply_guard("3 ) Liskov . principle", "3) Liskov principle") == "3) Liskov principle"
    assert apply_guard("19 ( 0b ) .", "1) (b).") == "1) (b)."  # the real error seen in the demo


def test_guard_rejects_rewrites_and_empty_output():
    raw = "Encapsulation hides data in a class"
    assert apply_guard(raw, "Polymorphism lets objects take many forms across subclasses.") == raw
    assert apply_guard(raw, "") == raw
    assert apply_guard("ok", "x" * 40) == "ok"  # runaway length


def test_cer():
    assert cer(["abc"], ["abc"]) == 0
    assert round(cer(["abd"], ["abc"]), 3) == round(1 / 3, 3)


def test_split_is_deterministic_and_group_based():
    assert split_of("A sentence.") == split_of("A sentence.")
    assert {split_of(f"s{i}") for i in range(500)} == {"train", "val", "test"}
    assert 0 <= group_bucket("x") < 100
    train = sum(split_of(f"s{i}") == "train" for i in range(2000))
    assert 1700 < train < 1950  # about 90 %


def test_fragments_are_line_sized_and_lossless():
    rng = random.Random(1)
    s = "The Liskov Substitution Principle states that objects of a superclass must be replaceable with objects of a subclass."
    parts = fragment(s, rng)
    assert " ".join(parts) == s and len(parts) >= 2 and all(len(p) <= 100 for p in parts)
    assert fragment("Short line here.", rng) == ["Short line here."]


def test_fact_swap_flips_terms_and_keeps_case():
    assert fact_swap("A stack uses Last In First Out ordering.") == "A stack uses First In First Out ordering."
    assert fact_swap("True, because it works.") == "False, because it works."
    assert fact_swap("Nothing to flip in here at all.") is None


def test_split_sentences_filters_junk():
    text = "Binary search runs in O(log n) time on a sorted array.\nSee https://x.org now.\nShort.\nCafé society is great for everyone here."
    assert split_sentences(text) == ["Binary search runs in O(log n) time on a sorted array."]


def test_simulated_noise_changes_text_and_is_seeded():
    s = "Encapsulation bundles data and methods into a single class."
    assert simulate(s, 0, random.Random(0)) == s
    a, b = simulate(s, 0.8, random.Random(3)), simulate(s, 0.8, random.Random(3))
    assert a == b and a != s


def test_question_numbers_and_digits_are_never_changed_by_the_model():
    from app.services.correction import finalize_body, split_prefix

    assert split_prefix("24 . Elucidate any 2 challenges") == ("24 . ", "Elucidate any 2 challenges")
    assert split_prefix("21supervised learning is") == ("21", "supervised learning is")
    assert split_prefix("3 . B.") == ("3 . ", "B.")
    assert split_prefix("2002 was a year") == ("", "2002 was a year")
    # the model dropped the content digit "2": keep the original body
    assert finalize_body("Elucidate any 2 challenges of ML", "Elucidate any challenges of ML") == "Elucidate any 2 challenges of ML"
    assert finalize_body("we cwant .", "we want.") == "we want."  # a pure typo repair is accepted


def test_unreadable_mcq_answer_is_flagged_not_marked_wrong_confidently():
    from app.services.grading.objective import grade_objective

    r = grade_objective("3", "D", 1)
    assert not r.correct and r.confidence < 0.6
    assert grade_objective("B.", "B", 1).correct
