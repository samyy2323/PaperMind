"""Run PaperMind's WHOLE pipeline on the public Mendeley exam dataset and compare with the teacher's marks.

Dataset: "A Dataset of Digitized Student Examination Papers, Answer Keys, and Manual Evaluations for Automated Grading
Research" (Mendeley Data, DOI 10.17632/sf3kvjwknt.1, CC BY 4.0): 50 scanned handwritten sheets of a Data Science exam
(20 MCQ x 1 mark + 15 short answers x 2 marks), the question paper, the answer key and the teacher's per-question marks.

    python scripts/run_mendeley_eval.py --data "<dataset folder>" --students 1,2,3 [--api http://localhost:8010/api]

Needs the API running with the real models (TrOCR, ByT5 correction, Ollama LLM) and the demo teacher seeded
(`python scripts/seed_demo.py`). Writes mendeley_report.json / .md and per-question CSV next to --out.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
import time
from pathlib import Path

import httpx
import pymupdf


# ----------------------------------------------------------------------------- dataset parsing
def parse_questions(question_txt: Path, key_csv: Path) -> list[dict]:
    text = question_txt.read_text(encoding="utf-8", errors="replace")
    blocks: dict[int, list[str]] = {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^\s*(\d{1,2})\.\s+(.*\S)\s*$", line)
        if m and 1 <= int(m.group(1)) <= 35 and (int(m.group(1)) - 1 in blocks or int(m.group(1)) == 1):
            current = int(m.group(1))
            blocks[current] = [m.group(2)]
        elif current and line.strip() and not line.startswith("Part "):
            blocks[current].append(line.strip())
    key = {}
    with key_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key[int(row["Question_Number"])] = (row["Type"], row["Correct_Answer"].strip())
    questions = []
    for n in range(1, 36):
        kind, answer = key[n]
        mcq = kind.upper().startswith("MCQ")
        questions.append({
            "text": "\n".join(blocks[n]),
            "referenceAnswer": answer,
            "type": "objective" if mcq else "subjective",
            "marks": 1 if mcq else 2,
        })
    return questions


def read_marks(csv_path: Path) -> dict[int, dict]:
    """Student_k -> {'mcq': {q: letter}, 'short': {q: mark}, 'total': float}"""
    out = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            k = int(row["Student_ID"].split("_")[1])
            def num(s):  # blank / "NC" (left blank, not corrected) / "NM" -> 0 marks
                try:
                    return float(s)
                except (TypeError, ValueError):
                    return 0.0

            out[k] = {
                "mcq": {q: (row[str(q)] or "").strip().upper() for q in range(1, 21)},
                "short": {q: num(row[str(q)]) for q in range(21, 36)},
                "total": num(row["total"]),
            }
    return out


def sheet_images(pdf: Path) -> list[bytes]:
    """The scanned page images, as stored in the PDF (JPEG) - falls back to rendering."""
    out = []
    with pymupdf.open(pdf) as doc:
        for page in doc:
            imgs = page.get_images(full=True)
            if imgs:
                out.append(doc.extract_image(imgs[0][0])["image"])
            else:
                out.append(page.get_pixmap(dpi=150).tobytes("jpg"))
    return out


# ----------------------------------------------------------------------------- statistics
def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    vx, vy = sum((a - mx) ** 2 for a in x), sum((b - my) ** 2 for b in y)
    return float("nan") if vx == 0 or vy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def fmt(v) -> str:
    return "n/a" if isinstance(v, float) and math.isnan(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with Question.txt, answerkey.txt, Student_Pdf/ ...")
    ap.add_argument("--students", default="1,2,3", help="comma-separated student numbers")
    ap.add_argument("--api", default="http://localhost:8010/api")
    ap.add_argument("--out", default="mendeley_run")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds to wait per sheet")
    ap.add_argument("--reuse", default="", help="EXAM_ID:K=SHEET_ID[,K=SHEET_ID] - evaluate sheets already processed")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    students = [int(s) for s in args.students.split(",") if s.strip()]

    questions = parse_questions(data / "Question.txt", data / "answerkey.txt")
    truth = read_marks(data / "Teacher_manual_marks_Anonymized.csv")
    key_letter = {i + 1: q["referenceAnswer"].upper() for i, q in enumerate(questions[:20])}

    # the server drops idle keep-alive connections; retry transport errors instead of failing the whole run
    api = httpx.Client(base_url=args.api, timeout=600, transport=httpx.HTTPTransport(retries=5))
    tok = api.post("/auth/login", json={"email": "kulkarni@college.edu", "password": "demo1234"})
    tok.raise_for_status()
    api.headers["Authorization"] = f"Bearer {tok.json()['token']}"

    cls = api.post("/classes/ensure", json={"subject": "Data Science", "subDomain": "AI & ML"}).json()
    reuse: dict[int, int] = {}
    if args.reuse:  # evaluate sheets that were already processed (e.g. after improving a later pipeline stage)
        exam_id, _, pairs = args.reuse.partition(":")
        reuse = {int(k): int(v) for k, v in (p.split("=") for p in pairs.split(","))}
        exam = api.get(f"/exams/{int(exam_id)}").json()
    else:
        exam = api.post("/exams", json={"title": f"Data Science exam (Mendeley) {int(time.time())}", "totalMarks": 50,
                                        "classId": cls["id"]}).json()
        api.put(f"/exams/{exam['id']}/questions", json=questions).raise_for_status()
        api.post(f"/exams/{exam['id']}/publish").raise_for_status()
        exam = api.get(f"/exams/{exam['id']}").json()
    qids = [q["id"] for q in exam["questions"]]
    print(f"exam #{exam['id']} published: {len(qids)} questions, {exam['totalMarks']:g} marks")

    rows, per_student = [], []
    for k in students:
        t0 = time.time()
        if k in reuse:
            sheet, pages = reuse[k], []
            print(f"\n[Student_{k}] re-evaluating existing sheet #{sheet}")
        else:
            pdf = data / "Student_Pdf" / f"Student_{k}.pdf"
            roll = f"MEND-{k}"
            roster = api.get(f"/classes/{cls['id']}/roster").json()
            sid = next((s["id"] for s in roster if s["rollNumber"] == roll), None) or api.post(
                f"/classes/{cls['id']}/roster", json={"name": f"Student {k}", "rollNumber": roll}).json()["id"]
            pages = sheet_images(pdf)
            files = [("files", (f"s{k}_p{i + 1}.jpg", img, "image/jpeg")) for i, img in enumerate(pages)]
            up = api.post(f"/students/{sid}/answer-sheet", data={"examId": str(exam["id"])}, files=files)
            up.raise_for_status()
            sheet = up.json()["submissionId"]
            print(f"\n[Student_{k}] uploaded {len(pages)} pages -> sheet #{sheet}")

        last = None
        while time.time() - t0 < args.timeout:
            try:
                st = api.get(f"/answer-sheets/{sheet}/status").json()
            except httpx.TransportError:
                time.sleep(3)
                continue
            if st != last:
                print(f"  [{time.time() - t0:5.0f}s] {st}")
                last = st
            if st["stage"] == "done" or st.get("error"):
                break
            time.sleep(5)
        detail = api.get(f"/answer-sheets/{sheet}/results").json()["submission"]
        answers = {a["questionId"]: a for a in detail["answers"]}
        tr = truth[k]
        mcq_read_ok = mcq_total = 0
        ours_short, teacher_short = [], []
        our_total = 0.0
        for n, qid in enumerate(qids, start=1):
            a = answers.get(qid)
            if a is None:
                continue
            score = a["override"]["score"] if a.get("override") else (a["aiScore"] or 0.0)
            our_total += score
            if n <= 20:
                m = re.match(r"^\W*([A-Ea-e])\b", a["ocrText"] or "")
                pred = m.group(1).upper() if m else ""
                ok = pred == tr["mcq"][n]
                mcq_total += 1
                mcq_read_ok += ok
                rows.append({"student": k, "q": n, "type": "mcq", "ocr_text": a["ocrText"], "predicted": pred,
                             "true_letter": tr["mcq"][n], "letter_read_ok": int(ok), "our_score": score,
                             "teacher_score": int(tr["mcq"][n] == key_letter[n]), "flag": a["flaggedReason"]})
            else:
                ours_short.append(score)
                teacher_short.append(tr["short"][n])
                rows.append({"student": k, "q": n, "type": "short", "ocr_text": a["ocrText"], "predicted": "",
                             "true_letter": "", "letter_read_ok": "", "our_score": score,
                             "teacher_score": tr["short"][n], "flag": a["flaggedReason"]})
        per_student.append({
            "student": k, "seconds": round(time.time() - t0), "status": detail["status"], "error": detail.get("error"),
            "our_total": our_total, "teacher_total": tr["total"],
            "mcq_letters_read_correctly": f"{mcq_read_ok}/{mcq_total}",
            "short_mae": (sum(abs(a - b) for a, b in zip(ours_short, teacher_short)) / len(ours_short)) if ours_short else None,
            "empty_short_answers": sum(1 for r in rows if r["student"] == k and r["type"] == "short" and not (r["ocr_text"] or "").strip()),
        })
        print("  ->", json.dumps(per_student[-1]))

    # ---- aggregate ---------------------------------------------------------------------------------------
    mcq = [r for r in rows if r["type"] == "mcq"]
    short = [r for r in rows if r["type"] == "short"]
    summary = {
        "students": students,
        "mcq_letter_read_accuracy": round(sum(r["letter_read_ok"] for r in mcq) / max(len(mcq), 1), 3),
        "mcq_correctness_agreement": round(sum(r["our_score"] == r["teacher_score"] for r in mcq) / max(len(mcq), 1), 3),
        "short_pearson_vs_teacher": pearson([r["our_score"] for r in short], [r["teacher_score"] for r in short]),
        "short_mae": round(sum(abs(r["our_score"] - r["teacher_score"]) for r in short) / max(len(short), 1), 3),
        "short_within_half_mark": round(sum(abs(r["our_score"] - r["teacher_score"]) <= 0.5 for r in short) / max(len(short), 1), 3),
        "short_empty_extractions": sum(1 for r in short if not (r["ocr_text"] or "").strip()),
        "total_pearson_vs_teacher": pearson([s["our_total"] for s in per_student], [s["teacher_total"] for s in per_student]),
        "total_mae": round(sum(abs(s["our_total"] - s["teacher_total"]) for s in per_student) / max(len(per_student), 1), 2),
        "flagged_answers": sum(1 for r in rows if r["flag"]),
    }
    (out / "mendeley_report.json").write_text(json.dumps({"summary": summary, "students": per_student}, indent=2, default=str))
    with (out / "per_question.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:32} {fmt(v)}")
    print(f"\nPer-question details: {out / 'per_question.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
