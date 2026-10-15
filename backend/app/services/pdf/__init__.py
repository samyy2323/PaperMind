"""Reference PDF assembly. The PDF is for human viewing/archival only — OCR always reads the original images."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps


def build_reference_pdf(image_paths: list[Path]) -> bytes:
    if not image_paths:
        raise ValueError("No images to assemble")
    pages = []
    for p in image_paths:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        img.thumbnail((2000, 2800))  # keep archival PDFs a reasonable size
        pages.append(img)
    buf = io.BytesIO()
    pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:], resolution=150.0)
    return buf.getvalue()
