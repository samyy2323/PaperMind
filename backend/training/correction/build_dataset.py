#!/usr/bin/env python3
"""Build the OCR-correction dataset: (noisy TrOCR-style line  ->  clean line) pairs, as JSONL.

Differences from the first generator, each fixing a real weakness:
  * thousands of DISTINCT sentences (built-in seeds + Simple-English Wikipedia filtered to CS topics + your own
    syllabus / textbook / student-answer files) instead of ~20 repeated ones;
  * splits are made by SENTENCE, so test sentences are never seen in training (no memorisation-inflated scores);
  * examples are line-sized fragments (a wrapped sentence), because the backend corrects one page line at a time;
  * noise comes from the REAL TrOCR model on handwriting-style fonts (--trocr-frac) plus a fast simulated model;
  * REAL handwriting can be added: the official IAM database (--iam-dir) and/or your own scanned lines (--manifest);
    --test-manifest keeps your own real pages out of training so they give an honest real-world test score;
  * "fact-swapped" sentences (stack<->queue, LIFO<->FIFO ...) are included as targets, so the model learns to repair
    OCR damage WITHOUT correcting a student's factual mistakes (that would silently change grades).

    python build_dataset.py --out data --n-lines 50000 --font-dir fonts --trocr-frac 0.35
    python build_dataset.py --out /tmp/smoke --n-lines 300 --wiki-n 0 --trocr-frac 0     # tiny, no network/GPU
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pm_common import (add_marker, edit_distance, fact_swap, fragment, normalise, read_iam, read_manifest,  # noqa: E402
                       split_of, split_sentences)
from pm_noise import TrOCRNoiser, find_fonts, simulate  # noqa: E402

# Phrases that are specific to computing; ambiguous single words ("python", "web", "code", "memory") are left out.
WIKI_KEYWORDS = (
    "algorithm", "data structure", "software", "computer program", "programming language", "database", "computer network",
    "encrypt", "cryptograph", "cybersecurity", "computer security", "machine learning", "neural network",
    "artificial intelligence", "object-oriented", "compiler", "operating system", "source code", "sorting",
    "network protocol", "microprocessor", "computer memory", "hash function", "recursion", "firewall", "web server",
    "software development", "computer science",
)


def load_seed() -> list[str]:
    return [s for s in (HERE / "seed_sentences.txt").read_text(encoding="utf-8").splitlines() if s.strip()]


def load_files(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() == ".pdf":
            import fitz  # PyMuPDF

            with fitz.open(path) as doc:
                text = "\n".join(page.get_text("text") for page in doc)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        out += split_sentences(text)
    return out


def load_wiki(n: int, seed: int) -> list[str]:
    """Stream Simple-English Wikipedia and keep prose sentences about computing."""
    if n <= 0:
        return []
    out: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=2000)
        for row in ds:
            for s in split_sentences(row["text"]):
                if any(k in s.lower() for k in WIKI_KEYWORDS):
                    out.append(s)
            if len(out) >= n:
                break
    except Exception as exc:  # no internet (e.g. Kaggle with internet off) - carry on with the other sources
        print(f"WARNING: could not read Wikipedia ({type(exc).__name__}: {exc}); continuing without it", file=sys.stderr)
    return out[:n]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-lines", type=int, default=50_000, help="total examples to produce")
    ap.add_argument("--wiki-n", type=int, default=15_000, help="sentences to pull from Simple Wikipedia (0 = skip)")
    ap.add_argument("--files", nargs="*", default=[], help="extra .txt/.md/.pdf (syllabus, textbooks, student answers)")
    ap.add_argument("--font-dir", default=None, help="folder of handwriting-style .ttf/.otf fonts")
    ap.add_argument("--trocr-frac", type=float, default=0.35, help="share of examples noised by the real TrOCR model")
    ap.add_argument("--trocr-batch", type=int, default=64)
    ap.add_argument("--clean-frac", type=float, default=0.10, help="share of unchanged lines (anti over-correction)")
    ap.add_argument("--swap-frac", type=float, default=0.15, help="share of sentences also added as fact-swapped")
    ap.add_argument("--marker-frac", type=float, default=0.55)
    ap.add_argument("--iam-dir", default=None, help="folder with the official IAM lines.txt + line PNGs (real handwriting)")
    ap.add_argument("--iam-lines", type=int, default=8000, help="how many IAM lines to add")
    ap.add_argument("--iam-repeat", type=int, default=1, help="augmented passes over each IAM line (more real-handwriting errors)")
    ap.add_argument("--manifest", default=None, help="TSV 'image<TAB>transcript' of your own real line images (trained on)")
    ap.add_argument("--manifest-repeat", type=int, default=4, help="augmented passes over each manifest line")
    ap.add_argument("--test-manifest", default=None, help="TSV of real lines used ONLY for the test split (honest real-world score)")
    ap.add_argument("--seed", type=int, default=42)
    args, _ignored = ap.parse_known_args(argv)  # tolerate options that belong to the training stage
    rng = random.Random(args.seed)
    t0 = time.time()

    seed_sentences = [normalise(s) for s in load_seed()]
    # A fixed quarter of the hand-written seed sentences is reserved for the TEST split (never trained on). They are
    # in-domain exam text, and their fact-swapped variants form the "does the model leave student errors alone" test.
    held_out = set(seed_sentences[::4])
    sentences = seed_sentences + [normalise(s) for s in load_files(args.files) + load_wiki(args.wiki_n, args.seed)]
    sentences = list(dict.fromkeys(sentences))
    rng.shuffle(sentences)
    print(f"{len(sentences):,} distinct sentences ({time.time() - t0:.0f}s)")
    if len(sentences) < 200:
        print("WARNING: very small corpus - the model will memorise it. Use --wiki-n and/or --files.", file=sys.stderr)

    # ---- 1. clean fragments, grouped by their source sentence ---------------------------------------------
    rows: list[dict] = []
    seed_set = set(seed_sentences)
    si = 0
    while len(rows) < args.n_lines:
        sent = sentences[si % len(sentences)]
        si += 1
        variants = [(sent, "normal", None)]
        wanted = sent in held_out or (sent in seed_set and rng.random() < 0.8) or rng.random() < args.swap_frac
        swapped = fact_swap(sent) if wanted else None
        if swapped:
            variants.append((swapped, "swapped", sent))
        for text, kind, orig in variants:
            for j, frag in enumerate(fragment(text, rng)):
                if j == 0 and rng.random() < args.marker_frac:
                    frag = add_marker(frag, rng)
                rows.append({"target": frag, "group": sent, "kind": kind, "orig": orig})
    rows = rows[: args.n_lines]
    if si > len(sentences):
        print(f"note: corpus reused {si / len(sentences):.1f}x to reach {args.n_lines:,} lines")

    # ---- 2. choose a noise source for each line ------------------------------------------------------------
    for r in rows:
        x = rng.random()
        r["source"] = "clean" if x < args.clean_frac else ("trocr" if x < args.clean_frac + args.trocr_frac else "sim")

    trocr_rows = [r for r in rows if r["source"] == "trocr"]
    if trocr_rows:
        fonts = find_fonts(args.font_dir)
        if not fonts:
            print("WARNING: no fonts found - falling back to simulated noise for the TrOCR share", file=sys.stderr)
            for r in trocr_rows:
                r["source"] = "sim"
        else:
            print(f"Running real TrOCR on {len(trocr_rows):,} rendered lines with {len(fonts)} font(s)...")
            noiser = TrOCRNoiser(fonts, seed=args.seed, batch_size=args.trocr_batch)
            for s in range(0, len(trocr_rows), 2000):
                part = trocr_rows[s : s + 2000]
                for r, (text, conf) in zip(part, noiser.read_lines([p["target"] for p in part])):
                    r["input"], r["trocr_conf"] = text, round(conf, 4)
                print(f"  TrOCR {min(s + 2000, len(trocr_rows)):,}/{len(trocr_rows):,} ({time.time() - t0:.0f}s)")
            del noiser

    levels = [0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
    for r in rows:
        if r["source"] == "clean":
            r["input"] = r["target"]
        elif r["source"] == "sim":
            r["input"] = simulate(r["target"], rng.choice(levels), rng)
        r["input"] = r["input"].strip() or r["target"]
        r["edits"] = edit_distance(r["input"], r["target"])

    # ---- 2b. REAL handwriting (IAM, your own pages): OCR the actual images -------------------------------------
    real_jobs = []  # (image, transcript, group, source, test_only)
    if args.iam_dir:
        iam = read_iam(args.iam_dir, args.iam_lines, args.seed)
        print(f"IAM: {len(iam):,} usable lines")
        for _ in range(max(1, args.iam_repeat)):
            real_jobs += [(img, text, f"iam:{form}", "iam", False) for img, text, form in iam]
    if args.manifest:
        for _ in range(args.manifest_repeat):
            real_jobs += [(img, text, f"own:{text}", "own", False) for img, text in read_manifest(args.manifest)]
    if args.test_manifest:
        real_jobs += [(img, text, f"owntest:{text}", "own_test", True) for img, text in read_manifest(args.test_manifest)]
    forced_test_groups = set()
    if real_jobs:
        print(f"Running real TrOCR on {len(real_jobs):,} REAL handwriting lines...")
        noiser = TrOCRNoiser(["unused"], seed=args.seed, batch_size=args.trocr_batch)
        # augmented, so repeated passes over the same image give different errors
        outs = noiser.read_images([j[0] for j in real_jobs], augment_images=True)
        for (img, text, group, source, test_only), (ocr_text, conf) in zip(real_jobs, outs):
            if not ocr_text:
                continue
            rows.append({"target": text, "input": ocr_text, "group": group, "kind": "normal", "orig": None,
                         "source": source, "trocr_conf": round(conf, 4), "edits": edit_distance(ocr_text, text)})
            if test_only:
                forced_test_groups.add(group)
        del noiser

    # ---- 3. split by sentence group & write -----------------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        splits["test" if (r["group"] in held_out or r["group"] in forced_test_groups) else split_of(r["group"])].append(r)
    for name, part in splits.items():
        rng.shuffle(part)
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    test_groups, train_groups = {r["group"] for r in splits["test"]}, {r["group"] for r in splits["train"]}
    assert not (test_groups & train_groups), "split leak: a sentence is in both train and test"
    stats = {
        "lines": {k: len(v) for k, v in splits.items()},
        "distinct_sentences": len(sentences),
        "sources": dict(Counter(r["source"] for r in rows)),
        "kinds": dict(Counter(r["kind"] for r in rows)),
        "unchanged_pct": round(100 * sum(r["edits"] == 0 for r in rows) / len(rows), 1),
        "median_edits_changed": statistics.median([r["edits"] for r in rows if r["edits"]] or [0]),
        "baseline_cer": round(sum(r["edits"] for r in rows) / sum(len(r["target"]) for r in rows), 4),
        "seconds": round(time.time() - t0),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("\nSamples (input -> target):")
    for r in rng.sample(rows, min(8, len(rows))):
        print(f"  [{r['source']:5}/{r['kind']}] {r['input']!r}\n        -> {r['target']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
