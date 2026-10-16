#!/usr/bin/env python3
"""Evaluate an OCR-correction checkpoint on held-out sentences.

    python evaluate_correction.py --model /path/to/checkpoint --data data            # reads data/test.jsonl
    python evaluate_correction.py --model google/byt5-small --data data --limit 200   # an untrained model, for comparison

Reports, per noise source and overall:
  CER before / after correction   - did the model reduce character errors? (lower is better)
  exact match                     - lines corrected perfectly
  over-correction                 - lines that were ALREADY correct and got changed (must be ~0)
  preservation (fact-swapped)     - on fluent-but-wrong student sentences, how often the model rewrote the mistake
                                    back to the true statement. This is the grading-safety metric: a correction
                                    model must repair handwriting errors, never the student's content.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm_common import cer, edit_distance  # noqa: E402


def load_jsonl(path: Path, limit: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def correct_texts(model, tok, texts: list[str], batch_size: int = 64, beams: int = 1, max_len: int = 192) -> list[str]:
    import torch

    device = next(model.parameters()).device
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [""] * len(texts)
    model.eval()
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            gen = model.generate(**enc, max_new_tokens=max_len, num_beams=beams)
            for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
                out[i] = t.strip()
    return out


def score(rows: list[dict], preds: list[str]) -> dict:
    ins, refs = [r["input"] for r in rows], [r["target"] for r in rows]
    report: dict = {"lines": len(rows)}

    def block(sel: list[int]) -> dict:
        if not sel:
            return {}
        i_, r_, p_ = [ins[k] for k in sel], [refs[k] for k in sel], [preds[k] for k in sel]
        return {
            "lines": len(sel),
            "cer_before": round(cer(i_, r_), 4),
            "cer_after": round(cer(p_, r_), 4),
            "exact_before": round(sum(a == b for a, b in zip(i_, r_)) / len(sel), 4),
            "exact_after": round(sum(a == b for a, b in zip(p_, r_)) / len(sel), 4),
        }

    report["overall"] = block(list(range(len(rows))))
    for src in sorted({r["source"] for r in rows}):
        report[f"source_{src}"] = block([k for k, r in enumerate(rows) if r["source"] == src])

    clean = [k for k in range(len(rows)) if ins[k] == refs[k]]
    if clean:
        report["over_correction"] = {
            "already_correct_lines": len(clean),
            "changed_by_model": sum(preds[k] != ins[k] for k in clean),
            "rate": round(sum(preds[k] != ins[k] for k in clean) / len(clean), 4),
        }

    swapped = [k for k, r in enumerate(rows) if r.get("kind") == "swapped" and r.get("orig") and ins[k] != refs[k]]
    if swapped:
        # reverted = the model's output is closer to the TRUE sentence's wording than to what the student wrote
        reverted = 0
        for k in swapped:
            true_frag = rows[k]["orig"]
            if edit_distance(preds[k], refs[k]) > 0 and preds[k] and edit_distance(preds[k], true_frag) < edit_distance(preds[k], refs[k]):
                reverted += 1
        report["preservation"] = {**block(swapped), "rewritten_to_true_fact": reverted, "rate": round(reverted / len(swapped), 4)}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--model", required=True, help="checkpoint dir or HF id")
    ap.add_argument("--data", default="data", help="folder containing <split>.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--report", default=None, help="write the JSON report here")
    args, _ignored = ap.parse_known_args(argv)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    rows = load_jsonl(Path(args.data) / f"{args.split}.jsonl", args.limit)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if torch.cuda.is_available():
        model.to("cuda")
    preds = correct_texts(model, tok, [r["input"] for r in rows], args.batch, args.beams)
    report = score(rows, preds)
    print(json.dumps(report, indent=2))

    worst = sorted(range(len(rows)), key=lambda k: -edit_distance(preds[k], rows[k]["target"]))[:8]
    print("\nWorst remaining errors (input | prediction | target):")
    for k in worst:
        print(f"  {rows[k]['input']!r}\n   -> {preds[k]!r}\n   == {rows[k]['target']!r}")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
