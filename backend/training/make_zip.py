"""Zip the correction-training code for upload to Colab:  python backend/training/make_zip.py"""
import zipfile
from pathlib import Path

src = Path(__file__).resolve().parent / "correction"
out = Path(__file__).resolve().parent / "papermind_correction.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(src.iterdir()):
        if p.suffix in (".py", ".txt") and p.name != "__init__.py":
            z.write(p, p.name)
print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
