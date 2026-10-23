"""Grader ceiling test: grade HAND-TYPED answers (perfect OCR + perfect segmentation) and compare with the teacher.

If the grader scores badly even on perfect text, the problem is the grader / rubric / model.
If it scores well, the loss in the full pipeline comes from reading and locating the answers (OCR + segmentation).

    cd backend
    python scripts/oracle_grader_eval.py --data "<mendeley folder>"                 # uses LLM_MODEL from .env
    LLM_MODEL=qwen2.5:14b python scripts/oracle_grader_eval.py --data "<folder>"     # try another Ollama model (set the env var)

Needs Ollama running with the model pulled. Uses the real SubjectiveGrader (same prompt, JSON validation) with no RAG.
Data: tests/fixtures/mendeley_student1_typed_answers.json (Student_1, 15 short answers).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
logging.disable(logging.WARNING)

import run_mendeley_eval as mend  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.services.grading.llm import get_llm_client  # noqa: E402
from app.services.grading.subjective import SubjectiveGrader  # noqa: E402
from app.services.rag.embedder import get_embedder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Mendeley dataset folder (Question.txt, answerkey.txt, marks CSV)")
    ap.add_argument("--student", type=int, default=1)
    ap.add_argument("--answers", type=Path, default=HERE.parent / "tests" / "fixtures" / "mendeley_student1_typed_answers.json")
    args = ap.parse_args()

    data = Path(args.data)
    questions = mend.parse_questions(data / "Question.txt", data / "answerkey.txt")
    teacher = mend.read_marks(data / "Teacher_manual_marks_Anonymized.csv")[args.student]["short"]
    typed = {int(k): v for k, v in json.loads(args.answers.read_text(encoding="utf-8")).items() if k.isdigit()}

    s = get_settings()
    print(f"grader: {s.LLM_BACKEND}:{s.LLM_MODEL}  (perfect text, no RAG)\n")
    grader = SubjectiveGrader(get_llm_client(), get_embedder(), retriever=None)

    ours, theirs, t0 = [], [], time.time()
    print(" Q | teacher | ours | answer (start)                                            | LLM justification")
    for n in range(21, 36):
        q = questions[n - 1]
        res = grader.grade(q["text"], q["referenceAnswer"], typed.get(n, ""), q["marks"])
        ours.append(res.score)
        theirs.append(teacher[n])
        print(f"{n:>2} | {teacher[n]:>7} | {res.score:>4} | {typed.get(n, '')[:55]!r:<57} | {res.justification[:90]}")
    elapsed = time.time() - t0

    def pearson(x, y):
        mx, my = sum(x) / len(x), sum(y) / len(y)
        vx, vy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
        return float("nan") if vx == 0 or vy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)

    mae = sum(abs(a - b) for a, b in zip(ours, theirs)) / len(ours)
    within = sum(abs(a - b) <= 0.5 for a, b in zip(ours, theirs)) / len(ours)
    print(f"\nshort-answer total: ours {sum(ours):.1f}  teacher {sum(theirs):.1f}   (the full OCR pipeline scored about 8.5-9)")
    print(f"Pearson {pearson(ours, theirs):.2f}   MAE {mae:.2f} (0-2 scale)   within 0.5 mark: {within:.0%}   [{elapsed:.0f}s]")
    over = sum(1 for a, b in zip(ours, theirs) if a - b > 0.5)
    under = sum(1 for a, b in zip(ours, theirs) if b - a > 0.5)
    print(f"over-scored {over}, under-scored {under} (difference > 0.5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
