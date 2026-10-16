"""Shared helpers for the OCR-correction training pipeline (metrics, text normalisation, line fragments)."""
from __future__ import annotations

import hashlib
import random
import re

from rapidfuzz.distance import Levenshtein

# --------------------------------------------------------------------------- metrics
def edit_distance(a: str, b: str) -> int:
    return Levenshtein.distance(a, b)


def cer(preds: list[str], refs: list[str]) -> float:
    """Character error rate over a corpus: total edits / total reference characters."""
    if not refs:
        return 0.0
    edits = sum(edit_distance(p, r) for p, r in zip(preds, refs))
    return edits / max(sum(len(r) for r in refs), 1)


# --------------------------------------------------------------------------- text normalisation
_PUNCT_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def normalise(text: str) -> str:
    text = text.translate(_PUNCT_MAP)
    text = re.sub(r"\[\d+\]", "", text)  # [1] citations
    return re.sub(r"\s+", " ", text).strip()


def usable_sentence(s: str, lo: int = 30, hi: int = 260) -> bool:
    """Handwritten exam text: plain ASCII prose, no markup/URLs."""
    if not (lo <= len(s) <= hi) or not s.isascii():
        return False
    if re.search(r"https?://|[{}|=<>\\^~`@#]", s):
        return False
    letters = sum(c.isalpha() for c in s)
    return letters / len(s) > 0.65 and s[0].isalnum()


def split_sentences(text: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n|\n", text):
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", normalise(para)):
            if usable_sentence(s):
                out.append(s)
    return out


# --------------------------------------------------------------------------- group split
def group_bucket(group: str) -> int:
    """Deterministic 0-99 bucket for a sentence, so every line from one sentence lands in one split."""
    return int(hashlib.md5(group.lower().encode()).hexdigest(), 16) % 100


def split_of(group: str, val_pct: int = 5, test_pct: int = 5) -> str:
    b = group_bucket(group)
    if b < test_pct:
        return "test"
    if b < test_pct + val_pct:
        return "val"
    return "train"


# --------------------------------------------------------------------------- line fragments
_MARKERS = ["{n})", "{n} )", "{n}.", "Q{n}.", "Q{n})", "Ans {n}:", "({n})", "{n}-"]


def fragment(sentence: str, rng: random.Random, lo: int = 35, hi: int = 85) -> list[str]:
    """Cut a sentence into line-sized pieces at word boundaries.

    The backend corrects one *page line* at a time, and a student's sentence wraps across lines, so the model
    must be trained on line-sized fragments that start and end mid-sentence.
    """
    words = sentence.split()
    pieces, cur = [], ""
    width = rng.randint(lo, hi)
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            pieces.append(cur)
            cur, width = w, rng.randint(lo, hi)
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        if pieces and len(cur) < 12:  # avoid a one-word orphan line
            pieces[-1] += " " + cur
        else:
            pieces.append(cur)
    return pieces


def add_marker(line: str, rng: random.Random) -> str:
    n = rng.randint(1, 10)
    marker = rng.choice(_MARKERS).format(n=n)
    if rng.random() < 0.2:
        marker = f"({rng.choice('abcd')}) {marker}"
    return f"{marker} {line}"


def chunk_words(text: str, max_chars: int = 38) -> list[str]:
    """Split a line into segments the way the backend's detector does (bounded width) before TrOCR."""
    chunks, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > max_chars:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return chunks + ([cur] if cur else [])


# --------------------------------------------------------------------------- fact swaps
# Fluent-but-wrong variants of a sentence. The correction model must repair OCR damage in such a sentence
# WITHOUT "fixing" the student's mistake, otherwise it would silently change what gets graded.
_SWAP_PAIRS = [
    ("Last In First Out", "First In First Out"), ("LIFO", "FIFO"), ("stack", "queue"), ("true", "false"),
    ("O(log n)", "O(n)"), ("public", "private"), ("compile-time", "run-time"), ("symmetric", "asymmetric"),
    ("supervised", "unsupervised"), ("overfitting", "underfitting"), ("precision", "recall"),
    ("inheritance", "composition"), ("minimum", "maximum"), ("encrypt", "decrypt"), ("sorted", "unsorted"),
    ("increase", "decrease"), ("before", "after"), ("Agile", "Waterfall"), ("white-box", "black-box"),
    ("public key", "private key"), ("synchronous", "asynchronous"), ("abstract", "concrete"),
]


def fact_swap(sentence: str) -> str | None:
    """Swap the first matching term for its opposite (case preserved); None if nothing matches."""
    for a, b in _SWAP_PAIRS:
        for x, y in ((a, b), (b, a)):
            m = re.search(rf"(?<![A-Za-z]){re.escape(x)}(?![A-Za-z])", sentence, re.I)
            if m:
                rep = y.capitalize() if m.group(0)[0].isupper() and not y[0].isupper() else y
                return sentence[: m.start()] + rep + sentence[m.end():]
    return None


# --------------------------------------------------------------------------- real handwriting sources
def iam_detok(text: str) -> str:
    """IAM transcripts are word-tokenised with '|' separators and detached punctuation; restore normal spacing."""
    t = text.replace("|", " ")
    t = re.sub(r"\s+([,.;:!?%)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


def read_manifest(path: str) -> list[tuple[str, str]]:
    """TSV of  <image path relative to the manifest> TAB <transcript>  ->  [(absolute image path, transcript)]."""
    from pathlib import Path

    base = Path(path).resolve().parent
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        img, _, text = line.partition("\t")
        if text.strip() and (base / img.strip()).exists():
            out.append((str(base / img.strip()), text.strip()))
    return out


def read_iam(iam_dir: str, limit: int = 0, seed: int = 0) -> list[tuple[str, str, str]]:
    """Official IAM database (lines.txt + line PNGs) -> [(image path, transcript, form id)]."""
    from pathlib import Path

    root = Path(iam_dir)
    lines_txt = next(iter(sorted(root.rglob("lines.txt"))), None)
    if lines_txt is None:
        raise FileNotFoundError(f"no lines.txt under {iam_dir} (extract ascii.tgz and lines.tgz of the IAM download there)")
    images = {p.stem: str(p) for p in root.rglob("*.png")}
    rows = []
    for line in lines_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(None, 8)
        if len(parts) < 9 or parts[1] != "ok" or parts[0] not in images:
            continue
        text = iam_detok(parts[8])
        if 8 <= len(text) <= 120:
            rows.append((images[parts[0]], text, parts[0].rsplit("-", 1)[0]))
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows
