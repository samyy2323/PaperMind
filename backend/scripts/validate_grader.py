"""Validate the subjective grader against a human-scored dataset (Mohler-style).

    python scripts/validate_grader.py path/to/eval.csv [--limit 50] [--max-score 5]

CSV columns: question, reference_answer, student_answer, human_score  (optional: max_score)
Runs the real grading engine (RAG retrieval is skipped: the reference answer is the only context) and
prints Pearson correlation and RMSE between model scores and human scores, both on the raw scale and
normalised to 0-1. For our own accuracy validation; not part of the live product.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.grading.llm import get_llm_client  # noqa: E402
from app.services.grading.subjective import SubjectiveGrader  # noqa: E402
from app.services.rag.embedder import get_embedder  # noqa: E402


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx, vy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
    return float("nan") if vx == 0 or vy == 0 else cov / math.sqrt(vx * vy)


def rmse(xs: list[float], ys: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="only grade the first N rows")
    ap.add_argument("--max-score", type=float, default=5.0, help="scale of human_score when no max_score column")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"question", "reference_answer", "student_answer", "human_score"}
    if not rows or not required <= set(rows[0]):
        print(f"CSV must have columns: {', '.join(sorted(required))}", file=sys.stderr)
        return 2
    if args.limit:
        rows = rows[: args.limit]

    grader = SubjectiveGrader(get_llm_client(), get_embedder(), retriever=None)
    human, model, failed = [], [], 0
    for i, r in enumerate(rows, start=1):
        max_score = float(r.get("max_score") or args.max_score)
        res = grader.grade(r["question"], r["reference_answer"], r["student_answer"], max_score)
        if res.parse_status == "failed":
            failed += 1
            continue
        human.append(float(r["human_score"]) / max_score)
        model.append(res.score / max_score)
        if args.verbose:
            print(f"[{i}/{len(rows)}] human={r['human_score']}/{max_score:g} model={res.score:g} :: {res.justification[:90]}")

    if len(human) < 2:
        print("Not enough gradable rows to compute correlation.", file=sys.stderr)
        return 1
    print(f"\nRows graded: {len(human)} (unparseable LLM output: {failed})")
    print(f"Pearson r  : {pearson(human, model):.3f}")
    print(f"RMSE (0-1) : {rmse(human, model):.3f}   (= {rmse(human, model) * args.max_score:.2f} on a {args.max_score:g}-point scale)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
