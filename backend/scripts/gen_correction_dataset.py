"""
gen_correction_dataset.py
─────────────────────────────────────────────────────────────────────────────
Generates a (noisy_trocr_output, clean_text) training dataset for fine-tuning
ByT5 as PaperMind's OCR correction model.

HOW IT WORKS
────────────
1. Builds a corpus of academic sentences covering the exam sub-domains.
2. Renders each sentence as a synthetic "handwritten" line image using Pillow
   with bundled/system fonts (no external font downloads needed).
3. Runs the real TrOCR model on each rendered image.
4. Saves (trocr_output, clean_text) pairs where TrOCR output != clean text,
   plus a controlled fraction of clean pairs so the model learns to pass
   through already-correct text.
5. Splits into train / val / test CSVs ready for the Colab fine-tuning script.

WHY THIS IS ACCURATE
────────────────────
- We render text and run the ACTUAL TrOCR model — error patterns are real,
  not simulated. The model learns exactly what microsoft/trocr-base-handwritten
  does wrong on this style of text.
- Domain matches exam content: CS theory, numbered question markers,
  true/false, option letters — exactly what the grader sees.
- Font diversity: uses multiple weights/sizes/slight rotations so the model
  generalises beyond a single rendering style.

REQUIREMENTS (already satisfied by backend/.venv)
────────────────────────────────────────────────
  pip install torch transformers Pillow numpy

USAGE
─────
  # From C:\\PaperMind\\backend — backend/.venv must be active
  python scripts/gen_correction_dataset.py --out data/correction --n 30000

  # Quick smoke-test (100 pairs, no saving):
  python scripts/gen_correction_dataset.py --n 100 --dry-run

OUTPUT
──────
  data/correction/train.csv   (~80 %)
  data/correction/val.csv     (~10 %)
  data/correction/test.csv    (~10 %)

  Columns: input (noisy), target (clean), edit_distance, trocr_conf
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import math
import os
import random
import sys
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CORPUS — academic text in the exact domains PaperMind grades
# ─────────────────────────────────────────────────────────────────────────────

# Sentence templates covering:
#   • numbered question markers  (1), 2) ... Q3. ...)
#   • option letters             ((a) (b) (c) (d))
#   • true/false                 (True / False)
#   • CS theory (OOP, SE, AI/ML, OS, DSA)
#   • numbers, abbreviations, code fragments

_MARKERS = [
    "1)", "2)", "3)", "4)", "5)", "6)",
    "Q1.", "Q2.", "Q3.", "Q4.",
    "Ans 1:", "Ans 2:", "Ans 3:",
]

_OPTIONS = ["(a)", "(b)", "(c)", "(d)", "(e)"]

_OOP_SENTENCES = [
    "Encapsulation bundles data and methods into a single class.",
    "Inheritance allows a subclass to reuse the behaviour of a superclass.",
    "Polymorphism enables objects of different classes to respond to the same method call.",
    "Abstraction hides implementation details and exposes only the interface.",
    "The Liskov Substitution Principle states that objects of a subclass should be replaceable with objects of the superclass.",
    "A constructor is a special method invoked automatically when an object is created.",
    "The final keyword in Java prevents a class from being subclassed.",
    "Method overloading allows multiple methods with the same name but different parameters.",
    "Method overriding occurs when a subclass provides a different implementation of a superclass method.",
    "An interface defines a contract that implementing classes must fulfil.",
    "Abstract classes cannot be instantiated directly.",
    "The singleton pattern ensures only one instance of a class exists at runtime.",
    "Composition is often preferred over inheritance to reduce coupling.",
    "Access modifiers: public, private, protected, and default.",
    "Static members belong to the class rather than to any instance.",
    "A stack uses Last In First Out (LIFO) ordering.",
    "A queue uses First In First Out (FIFO) ordering.",
    "Binary search runs in O(log n) time on a sorted array.",
    "Quicksort has an average time complexity of O(n log n).",
    "Heap sort uses a max-heap or min-heap data structure.",
    "The factory method pattern defines an interface for creating objects.",
    "The observer pattern defines a one-to-many dependency between objects.",
    "Tight coupling makes unit testing difficult.",
    "Dependency injection reduces tight coupling between components.",
    "An exception is an event that disrupts the normal flow of a program.",
    "Garbage collection automatically reclaims memory occupied by unused objects.",
    "Multithreading allows concurrent execution of two or more threads.",
    "Synchronisation prevents race conditions in multithreaded programs.",
    "The decorator pattern attaches additional responsibilities to an object dynamically.",
    "UML class diagrams show classes, attributes, methods, and relationships.",
]

_SE_SENTENCES = [
    "Agile methodology emphasises iterative development and customer collaboration.",
    "The waterfall model is a linear sequential approach to software development.",
    "A use case diagram captures functional requirements from a user's perspective.",
    "Regression testing verifies that new changes have not broken existing functionality.",
    "Continuous integration merges developer code changes into a shared repository frequently.",
    "Version control systems track changes to source code over time.",
    "Code review improves code quality and shares knowledge among team members.",
    "A software requirement specification (SRS) documents what a system must do.",
    "Black-box testing treats the system as an opaque unit and tests inputs and outputs.",
    "White-box testing examines the internal structure of the code.",
    "McCabe's cyclomatic complexity measures the number of linearly independent paths.",
    "SOLID stands for Single responsibility, Open-closed, Liskov, Interface segregation, Dependency inversion.",
    "Technical debt accumulates when short-term shortcuts are taken instead of proper solutions.",
    "A sprint in Scrum is a time-boxed iteration typically lasting two weeks.",
    "The product backlog is an ordered list of features and requirements for the product.",
    "Pair programming involves two developers working together at one workstation.",
    "Test-driven development writes tests before writing the implementation code.",
    "A deployment pipeline automates the building, testing, and deployment of software.",
    "Static analysis tools detect potential bugs without executing the program.",
    "Software entropy refers to the gradual decline in software quality over time.",
]

_AI_SENTENCES = [
    "A neural network is a computational model inspired by the human brain.",
    "Supervised learning uses labelled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds patterns in data without explicit labels.",
    "Reinforcement learning trains agents through reward and punishment signals.",
    "Gradient descent minimises a loss function by iteratively updating parameters.",
    "Overfitting occurs when a model learns the training data noise rather than the underlying pattern.",
    "Regularisation techniques such as L1 and L2 penalty reduce overfitting.",
    "A convolutional neural network is well suited for image recognition tasks.",
    "Recurrent neural networks process sequential data by maintaining hidden state.",
    "The attention mechanism allows the model to focus on relevant parts of the input.",
    "Precision measures the fraction of predicted positives that are actually positive.",
    "Recall measures the fraction of actual positives correctly predicted.",
    "The F1-score is the harmonic mean of precision and recall.",
    "K-means clustering partitions data into k clusters by minimising intra-cluster distance.",
    "Principal Component Analysis reduces dimensionality by projecting onto principal components.",
    "Decision trees split data based on feature values to make predictions.",
    "Random forests combine many decision trees to reduce variance.",
    "Support vector machines find the hyperplane with the maximum margin between classes.",
    "Natural language processing enables computers to understand human language.",
    "Transfer learning reuses a model trained on one task for a different but related task.",
]

_CYBERSEC_SENTENCES = [
    "Symmetric encryption uses the same key for encryption and decryption.",
    "Asymmetric encryption uses a public key to encrypt and a private key to decrypt.",
    "A firewall monitors and controls incoming and outgoing network traffic.",
    "SQL injection occurs when untrusted input is inserted into a SQL query.",
    "Cross-site scripting (XSS) injects malicious scripts into web pages viewed by other users.",
    "A buffer overflow overwrites adjacent memory beyond a buffer's boundary.",
    "Two-factor authentication requires two forms of identification.",
    "Hashing converts data into a fixed-length string that cannot be reversed.",
    "A man-in-the-middle attack intercepts communication between two parties.",
    "Penetration testing simulates cyberattacks to identify vulnerabilities.",
    "SSL/TLS encrypts data transmitted over a network.",
    "A VPN creates an encrypted tunnel between the client and the VPN server.",
    "Phishing attacks trick users into revealing sensitive information.",
    "Zero-day vulnerabilities are unknown to the software vendor.",
    "The principle of least privilege grants only the minimum access required.",
]

_TRUE_FALSE = [
    "True", "False",
    "True, because encapsulation hides internal state.",
    "False, because interfaces cannot have instance variables.",
    "True, because polymorphism allows runtime method dispatch.",
    "False, an abstract class can have concrete methods.",
    "True, multiple inheritance is supported in Python.",
    "False, Java does not support multiple class inheritance directly.",
]

_OPTION_SENTENCES = [
    "(a) Encapsulation", "(b) Inheritance", "(c) Polymorphism", "(d) Abstraction",
    "(a) True", "(b) False",
    "(a) O(1)", "(b) O(log n)", "(c) O(n)", "(d) O(n log n)", "(e) O(n^2)",
    "(a) Stack", "(b) Queue", "(c) Heap", "(d) Tree",
    "(a) Compile-time error", "(b) Runtime exception", "(c) Logical error", "(d) No error",
]


def _build_corpus(n_target: int) -> list[str]:
    """Build a list of sentence strings that cover the exam domains."""
    pool = (
        _OOP_SENTENCES * 4
        + _SE_SENTENCES * 2
        + _AI_SENTENCES * 2
        + _CYBERSEC_SENTENCES
        + _TRUE_FALSE * 6
        + _OPTION_SENTENCES * 8
    )
    # Add marker + sentence combos (the most failure-prone pattern for TrOCR)
    marker_combos = []
    for m in _MARKERS:
        for s in random.sample(_OOP_SENTENCES + _SE_SENTENCES + _AI_SENTENCES, 8):
            marker_combos.append(f"{m} {s}")
        for opt in _OPTION_SENTENCES:
            marker_combos.append(f"{m} {opt}")
    pool += marker_combos * 2

    # Shuffle and truncate/repeat to hit n_target
    random.shuffle(pool)
    if len(pool) >= n_target:
        return pool[:n_target]
    repeats = math.ceil(n_target / len(pool))
    extended = []
    for _ in range(repeats):
        shuffled = pool[:]
        random.shuffle(shuffled)
        extended.extend(shuffled)
    return extended[:n_target]


# ─────────────────────────────────────────────────────────────────────────────
# 2. RENDERING — simulate handwritten-style line images
# ─────────────────────────────────────────────────────────────────────────────

# Font sizes and styles to vary across examples
_FONT_SIZES = [24, 28, 32, 36]

# Map of (style, size) -> font path fallback chain.
# We try to use fonts that vaguely resemble handwriting; fall back to
# any available system font; ultimately fall back to PIL default.
_FALLBACK_FONT_NAMES = [
    "cour.ttf",          # Courier New (monospace, clear)
    "arial.ttf",         # Arial
    "times.ttf",         # Times New Roman
    "DejaVuSans.ttf",    # Linux fallback
    "LiberationMono-Regular.ttf",
]

def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system font paths; fall back to PIL default bitmap font."""
    search_dirs = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts/truetype"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/Library/Fonts"),
    ]
    for fname in _FALLBACK_FONT_NAMES:
        for d in search_dirs:
            p = d / fname
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    continue
    return ImageFont.load_default()


def render_line(text: str, font_size: int | None = None, noise_level: float = 0.0) -> Image.Image:
    """
    Render a text string as a greyscale line image similar to what
    TrOCR's line detector crops from a handwritten page.

    noise_level in [0, 1]:
      0.0 = clean print
      0.3 = mild noise (slight blur, minor rotation)
      0.6 = moderate (bigger blur, salt/pepper)
      0.9 = heavy (rotation + blur + speckle)
    """
    import numpy as np

    if font_size is None:
        font_size = random.choice(_FONT_SIZES)
    font = _find_font(font_size)

    # ---- measure text size ----
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = max(10, int(tw * 0.08)), max(6, int(th * 0.25))
    W, H = tw + 2 * pad_x, th + 2 * pad_y

    # ---- draw on white background ----
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # slight colour variation around black ink
    ink_r = random.randint(0, 30)
    ink_g = random.randint(0, 30)
    ink_b = random.randint(0, 50)
    draw.text((pad_x, pad_y), text, font=font, fill=(ink_r, ink_g, ink_b))

    # ---- noise / degradation ----
    if noise_level > 0:
        arr = np.asarray(img, dtype=np.float32)

        # Gaussian noise
        if noise_level > 0.2:
            sigma = noise_level * 25
            arr += np.random.normal(0, sigma, arr.shape)
            arr = np.clip(arr, 0, 255)

        # Salt-and-pepper speckle
        if noise_level > 0.4:
            mask = np.random.random(arr.shape[:2]) < noise_level * 0.03
            arr[mask] = np.random.choice([0, 255])

        img = Image.fromarray(arr.astype(np.uint8))

        # Slight rotation
        if noise_level > 0.3:
            angle = random.uniform(-noise_level * 4, noise_level * 4)
            img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

        # Blur (simulates slight focus loss)
        if noise_level > 0.5:
            from PIL import ImageFilter
            radius = noise_level * 1.5
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    return img.convert("RGB")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TROCR INFERENCE — run the real model to collect real error patterns
# ─────────────────────────────────────────────────────────────────────────────

class TrOCRRunner:
    MODEL = "microsoft/trocr-base-handwritten"

    def __init__(self):
        log.info("Loading TrOCR model: %s", self.MODEL)
        self.processor = TrOCRProcessor.from_pretrained(self.MODEL)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        log.info("TrOCR on %s", self.device)

    @torch.inference_mode()
    def run_batch(self, images: list[Image.Image]) -> list[tuple[str, float]]:
        """Returns (text, confidence) per image."""
        pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        out = self.model.generate(
            pixels,
            max_new_tokens=64,
            num_beams=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        logprobs = self.model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )
        pad = self.processor.tokenizer.pad_token_id
        tokens = out.sequences[:, 1:]
        texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        results = []
        for text, lp, tok in zip(texts, logprobs, tokens):
            mask = tok != pad
            mean_lp = float(lp[mask].mean()) if mask.any() else -5.0
            results.append((text.strip(), math.exp(mean_lp)))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN GENERATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    """Character-level edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def generate(
    n_pairs: int,
    out_dir: Path,
    batch_size: int = 16,
    clean_fraction: float = 0.15,
    dry_run: bool = False,
) -> None:
    """
    Generate `n_pairs` training examples and write train/val/test CSVs.

    clean_fraction: fraction of pairs where TrOCR was correct — kept so the
    model learns to pass through already-clean lines without altering them.
    """
    rng = random.Random(42)
    corpus = _build_corpus(n_pairs * 3)  # over-generate; we'll filter
    ocr = TrOCRRunner()

    pairs: list[dict] = []
    processed = 0
    corpus_iter = iter(corpus)

    # Noise schedule: mix of noise levels to cover easy and hard cases
    def _noise() -> float:
        return rng.choice([0.0, 0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8])

    log.info("Generating %d pairs (batch_size=%d)…", n_pairs, batch_size)

    while len(pairs) < n_pairs:
        batch_texts: list[str] = []
        batch_imgs: list[Image.Image] = []
        batch_noise: list[float] = []

        for _ in range(batch_size):
            try:
                text = next(corpus_iter)
            except StopIteration:
                # re-shuffle corpus and continue
                corpus = _build_corpus(n_pairs * 2)
                corpus_iter = iter(corpus)
                text = next(corpus_iter)

            # Vary font size per sample
            font_size = rng.choice(_FONT_SIZES)
            nl = _noise()
            img = render_line(text, font_size=font_size, noise_level=nl)
            batch_texts.append(text)
            batch_imgs.append(img)
            batch_noise.append(nl)

        results = ocr.run_batch(batch_imgs)
        processed += len(batch_texts)

        for clean, (noisy, conf), noise_lvl in zip(batch_texts, results, batch_noise):
            ed = levenshtein(noisy, clean)
            if ed == 0:
                # TrOCR was perfect — keep a fraction of clean pairs
                if rng.random() > clean_fraction:
                    continue
            pairs.append({
                "input": noisy,
                "target": clean,
                "edit_distance": ed,
                "trocr_conf": round(conf, 4),
                "noise_level": round(noise_lvl, 2),
            })

        if processed % 500 == 0 or len(pairs) >= n_pairs:
            pct = min(100, len(pairs) / n_pairs * 100)
            log.info(
                "  processed=%d  pairs_collected=%d / %d  (%.1f%%)",
                processed, len(pairs), n_pairs, pct,
            )

        if len(pairs) >= n_pairs:
            break

    pairs = pairs[:n_pairs]

    # ── Stats ──────────────────────────────────────────────────────────────
    perfect = sum(1 for p in pairs if p["edit_distance"] == 0)
    avg_ed = sum(p["edit_distance"] for p in pairs) / len(pairs)
    avg_conf = sum(p["trocr_conf"] for p in pairs) / len(pairs)
    log.info(
        "Dataset stats: total=%d  perfect(ed=0)=%d(%.1f%%)  avg_edit_dist=%.2f  avg_trocr_conf=%.3f",
        len(pairs), perfect, 100 * perfect / len(pairs), avg_ed, avg_conf,
    )

    if dry_run:
        log.info("Dry run — showing first 10 pairs:")
        for p in pairs[:10]:
            print(f"  IN : {p['input']!r}")
            print(f"  OUT: {p['target']!r}")
            print(f"  ED={p['edit_distance']}  conf={p['trocr_conf']}")
            print()
        return

    # ── Split and write ─────────────────────────────────────────────────────
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)
    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train : n_train + n_val],
        "test":  pairs[n_train + n_val :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["input", "target", "edit_distance", "trocr_conf", "noise_level"]
    for split_name, rows in splits.items():
        path = out_dir / f"{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Wrote %s (%d rows)", path, len(rows))

    log.info("Done. Dataset at %s", out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate TrOCR correction training data")
    parser.add_argument("--out", default="data/correction", help="Output directory for CSVs")
    parser.add_argument("--n", type=int, default=30_000, help="Target number of training pairs")
    parser.add_argument("--batch", type=int, default=16, help="TrOCR inference batch size (lower if OOM)")
    parser.add_argument("--clean-frac", type=float, default=0.15,
                        help="Fraction of pairs where TrOCR was correct (kept to prevent over-correction)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run 100 pairs, print results, do not save")
    args = parser.parse_args()

    random.seed(args.seed)

    generate(
        n_pairs=100 if args.dry_run else args.n,
        out_dir=Path(args.out),
        batch_size=args.batch,
        clean_fraction=args.clean_frac,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
