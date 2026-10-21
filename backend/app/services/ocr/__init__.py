"""OCR / handwriting recognition: line detection -> TrOCR per line -> text lines in reading order."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from PIL import Image, ImageOps

from app.core.config import get_settings

MAX_SIDE = 2400  # phone photos are downscaled before detection


@dataclass
class OCRLine:
    page: int  # 0-based page index
    text: str
    confidence: float
    box: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class OCREngine(Protocol):
    def read_image(self, image: Image.Image, page: int) -> list[OCRLine]: ...


def load_image(path) -> Image.Image:
    """Open an uploaded photo, honouring EXIF rotation and bounding its size."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


class TrOCREngine:
    def __init__(self, detector, recognizer, batch_size: int):
        self.detector, self.recognizer, self.batch_size = detector, recognizer, batch_size

    def read_image(self, image: Image.Image, page: int) -> list[OCRLine]:
        from app.services.ocr.detectors import is_blank_page, suppress_bleed_through

        if is_blank_page(image):
            return []
        cleaned = suppress_bleed_through(image)
        if hasattr(self.detector, "detect_with_boxes"):
            detected = self.detector.detect_with_boxes(cleaned)
            lines = [crops for crops, _ in detected]
            boxes = [box for _, box in detected]
        else:
            lines = self.detector.detect(cleaned)
            boxes = [None] * len(lines)

        flat = [seg for line in lines for seg in line]
        if not flat:
            return []
        recognised = self.recognizer.recognize(flat, batch_size=self.batch_size)
        out, i = [], 0
        for line_idx, line in enumerate(lines):
            parts = recognised[i : i + len(line)]
            i += len(line)
            text = " ".join(t for t, _ in parts if t).strip()
            if not text:
                continue
            conf = sum(c for _, c in parts) / len(parts)
            if conf < 0.25 and len(text) < 3:
                continue
            line_box = boxes[line_idx] if line_idx < len(boxes) else None
            out.append(OCRLine(page=page, text=text, confidence=round(conf, 4), box=line_box))
        return out


def get_ocr_engine() -> OCREngine:
    s = get_settings()
    if s.OCR_BACKEND == "trocr":
        from app.services.ocr.detectors import get_line_detector
        from app.services.ocr.trocr import get_trocr

        return TrOCREngine(get_line_detector(), get_trocr(), s.OCR_BATCH_SIZE)
    raise ValueError(f"Unknown OCR_BACKEND: {s.OCR_BACKEND}")
