"""Download the trained ByT5 OCR-correction model from this repository's GitHub Release.

    python backend/scripts/fetch_model.py                    # -> backend/models/byt5-correction/
    python backend/scripts/fetch_model.py --tag model-v1 --dest some/other/folder

The 2.3 GB weights are attached to the release as two parts (GitHub limits a release file to 2 GB). This script downloads
them, joins them, and verifies the SHA-256 checksum. Standard library only - nothing to install.

Then set  CORRECTION_MODEL_PATH=./models/byt5-correction  in backend/.env  and restart the backend.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "samyy2323/PaperMind"
PARTS = ["model.safetensors.part00", "model.safetensors.part01"]
SMALL = ["config.json", "generation_config.json", "tokenizer_config.json", "added_tokens.json", "test_report.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "papermind-fetch-model"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:7.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()


def main() -> int:
    default_dest = Path(__file__).resolve().parent.parent / "models" / "byt5-correction"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="model-v1", help="release tag")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dest", type=Path, default=default_dest)
    args = ap.parse_args()
    base = f"https://github.com/{args.repo}/releases/download/{args.tag}"
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    sums_path = dest / "SHA256SUMS.txt"
    download(f"{base}/SHA256SUMS.txt", sums_path)
    expected = {}
    for line in sums_path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.strip().lstrip("*")] = digest

    model = dest / "model.safetensors"
    if model.exists() and sha256(model) == expected["model.safetensors"]:
        print("model.safetensors already present and verified")
    else:
        parts = []
        for name in PARTS:
            part = dest / name
            download(f"{base}/{name}", part)
            parts.append(part)
        print("joining parts ...")
        with model.open("wb") as out:
            for part in parts:
                with part.open("rb") as f:
                    while chunk := f.read(1 << 20):
                        out.write(chunk)
                part.unlink()
        if sha256(model) != expected["model.safetensors"]:
            model.unlink()
            print("ERROR: checksum mismatch - download corrupted, please run again", file=sys.stderr)
            return 1
    for name in SMALL:
        download(f"{base}/{name}", dest / name)
        if name in expected and sha256(dest / name) != expected[name]:
            print(f"ERROR: checksum mismatch for {name}", file=sys.stderr)
            return 1
    print(f"\nOK - model ready in {dest}\nNext: set CORRECTION_MODEL_PATH=./models/byt5-correction in backend/.env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
