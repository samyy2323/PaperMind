"""Download the public Mendeley exam dataset used to evaluate PaperMind's whole pipeline.

    python backend/scripts/fetch_mendeley_dataset.py              # -> testdata/mendeley/   (145 MB)
    python backend/scripts/fetch_mendeley_dataset.py --check      # only check the download is reachable and print its size

Dataset: "A Dataset of Digitized Student Examination Papers, Answer Keys, and Manual Evaluations for Automated Grading Research"
  Authors : Dinesh K P (Mendeley Data, version 1, published 14 April 2026)
  DOI     : 10.17632/sf3kvjwknt.1      https://data.mendeley.com/datasets/sf3kvjwknt/1
  Licence : CC BY 4.0 - attribution required (cite the DOI above wherever results are reported)
  Content : 50 scanned handwritten answer sheets (4 pages each) of a Data Science exam (20 MCQ x 1 mark + 15 short answers
            x 2 marks), the question paper, the answer key, and the teacher's per-question marks.

Then run the whole pipeline on it (API + models + Ollama must be running, see backend/README.md):
    python backend/scripts/run_mendeley_eval.py --data "testdata/mendeley/<dataset folder>" --students 1,2,3 --api http://localhost:8010/api
Standard library only. The folder is git-ignored: do not commit the dataset.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://data.mendeley.com/public-api/zip/sf3kvjwknt/download/1"
EXPECTED = ["Question.txt", "answerkey.txt", "Teacher_manual_marks_Anonymized.csv"]


def size_of_download() -> int:
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-0", "User-Agent": "papermind-fetch"})
    with urllib.request.urlopen(req) as resp:
        content_range = resp.headers.get("Content-Range", "")
        return int(content_range.rsplit("/", 1)[-1]) if "/" in content_range else int(resp.headers.get("Content-Length") or 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path("testdata/mendeley"))
    ap.add_argument("--check", action="store_true", help="only check reachability and size")
    args = ap.parse_args()

    total = size_of_download()
    print(f"download size: {total / 1e6:.0f} MB")
    if args.check:
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    zip_path = args.dest / "mendeley_sf3kvjwknt.zip"
    req = urllib.request.Request(URL, headers={"User-Agent": "papermind-fetch"})
    with urllib.request.urlopen(req) as resp, zip_path.open("wb") as out:
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.dest)
    folders = [p for p in args.dest.iterdir() if p.is_dir()]
    root = next((p for p in folders if (p / "Question.txt").exists()), None)
    if root is None:
        print("ERROR: unexpected archive layout", file=sys.stderr)
        return 1
    missing = [f for f in EXPECTED if not (root / f).exists()]
    pdfs = len(list((root / "Student_Pdf").glob("*.pdf")))
    print(f"OK: {root}\n  student PDFs: {pdfs} (expected 50)   missing files: {missing or 'none'}")
    print(f'\nRun:  python backend/scripts/run_mendeley_eval.py --data "{root}" --students 1,2,3')
    return 0 if not missing and pdfs == 50 else 1


if __name__ == "__main__":
    sys.exit(main())
