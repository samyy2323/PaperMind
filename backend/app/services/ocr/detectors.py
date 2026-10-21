"""Text-line detection. A detector turns a page image into reading-ordered *lines*, where each line
is a list of image *segments* (TrOCR degrades on very wide crops, so long lines are split at word gaps
into segments of bounded aspect ratio; the segments are recognised independently and re-joined).
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.core.singleton import once

log = logging.getLogger(__name__)

MAX_SEGMENT_ASPECT = 6.5  # width / height of a crop fed to TrOCR
MIN_LINE_HEIGHT = 12


class LineDetector(Protocol):
    def detect(self, image: Image.Image) -> list[list[Image.Image]]: ...


# --------------------------------------------------------------------------- shared helpers
def is_blank_page(image: Image.Image, min_ink_ratio: float = 0.0015, min_darkness_delta: float = 30.0) -> bool:
    """Return True if an image is essentially blank or has only faint bleed-through."""
    gray = np.asarray(image.convert("L"))
    if gray.size == 0:
        return True
    bg = float(np.percentile(gray, 95))
    if bg < 50:
        return False
    dark_mask = gray < (bg - min_darkness_delta)
    ink_ratio = float(dark_mask.sum()) / float(gray.size)
    return ink_ratio < min_ink_ratio


def suppress_bleed_through(image: Image.Image, ghost_threshold: float = 28.0) -> Image.Image:
    """Brighten faint ghost ink / bleed-through that is barely darker than the paper background."""
    arr = np.asarray(image.convert("RGB")).copy()
    gray = np.asarray(Image.fromarray(arr).convert("L"))
    bg = float(np.percentile(gray, 95))
    if bg < 80:
        return image
    ghost_mask = (gray >= (bg - ghost_threshold)) & (gray < bg)
    arr[ghost_mask] = int(min(255, bg))
    return Image.fromarray(arr)


def _pad_crop(img: Image.Image, box: tuple[float, float, float, float], pad: float = 0.15) -> Image.Image:
    x0, y0, x1, y1 = box
    h = y1 - y0
    px, py = h * pad, h * pad
    W, H = img.size
    return img.crop((max(0, int(x0 - px)), max(0, int(y0 - py)), min(W, int(x1 + px)), min(H, int(y1 + py))))


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Boolean ink mask via Otsu threshold."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = 0.0
    sum_b = 0.0
    best, thr = 0.0, 127
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best:
            best, thr = var, t
    return gray < thr


def split_wide(line_img: Image.Image, max_aspect: float = MAX_SEGMENT_ASPECT) -> list[Image.Image]:
    """Split a wide line crop at the emptiest columns so each piece has aspect <= max_aspect."""
    w, h = line_img.size
    if w / max(h, 1) <= max_aspect:
        return [line_img]
    ink = _ink_mask(np.asarray(line_img.convert("L"))).sum(axis=0).astype(float)
    ink = np.convolve(ink, np.ones(5) / 5, mode="same")
    max_w = int(max_aspect * h)
    cuts, start = [0], 0
    while w - start > max_w:
        lo, hi = start + int(max_w * 0.55), start + max_w
        cut = lo + int(np.argmin(ink[lo:hi]))
        cuts.append(cut)
        start = cut
    cuts.append(w)
    return [line_img.crop((a, 0, b, h)) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 4]


def group_words_into_lines(words: list[tuple[float, float, float, float]]) -> list[list[tuple[float, float, float, float]]]:
    """Cluster word boxes (x0,y0,x1,y1) into lines by vertical overlap; lines sorted top-to-bottom."""
    lines: list[dict] = []
    for box in sorted(words, key=lambda b: (b[1] + b[3]) / 2):
        x0, y0, x1, y1 = box
        h = y1 - y0
        for ln in lines:
            overlap = min(y1, ln["y1"]) - max(y0, ln["y0"])
            if overlap > 0.5 * min(h, ln["y1"] - ln["y0"]):
                ln["boxes"].append(box)
                ln["y0"], ln["y1"] = min(ln["y0"], y0), max(ln["y1"], y1)
                break
        else:
            lines.append({"y0": y0, "y1": y1, "boxes": [box]})
    lines.sort(key=lambda l: (l["y0"] + l["y1"]) / 2)
    return [sorted(l["boxes"], key=lambda b: b[0]) for l in lines]


# --------------------------------------------------------------------------- projection detector
class ProjectionLineDetector:
    """Dependency-free detector: horizontal ink-projection profile. Works on clean scans / ruled paper;
    used as a fallback when doctr is unavailable."""

    def detect(self, image: Image.Image) -> list[list[Image.Image]]:
        gray = np.asarray(image.convert("L"))
        ink = _ink_mask(gray)
        H, W = ink.shape
        # Remove ruled lines: rows where a single long horizontal run of ink spans most of the page.
        row_ink = ink.sum(axis=1)
        ink[row_ink > 0.6 * W] = False
        profile = np.convolve(ink.sum(axis=1).astype(float), np.ones(3) / 3, mode="same")
        thr = max(2.0, 0.004 * W)
        rows = profile > thr

        bands, start = [], None
        for y, on in enumerate(rows):
            if on and start is None:
                start = y
            elif not on and start is not None:
                bands.append([start, y])
                start = None
        if start is not None:
            bands.append([start, H])

        # merge bands separated by tiny gaps (i/j dots, descenders)
        merged: list[list[int]] = []
        for b in bands:
            if merged and b[0] - merged[-1][1] < 0.25 * max(merged[-1][1] - merged[-1][0], 8):
                merged[-1][1] = b[1]
            else:
                merged.append(b)

        out: list[tuple[list[Image.Image], tuple[float, float, float, float]]] = []
        for y0, y1 in merged:
            if y1 - y0 < MIN_LINE_HEIGHT:
                continue
            cols = np.where(ink[y0:y1].sum(axis=0) > 0)[0]
            if cols.size == 0:
                continue
            x0, x1 = float(cols[0]), float(cols[-1] + 1)
            box = (x0, float(y0), x1, float(y1))
            crops = split_wide(_pad_crop(image, (x0, y0, x1, y1)))
            out.append((crops, box))
        return out

    def detect(self, image: Image.Image) -> list[list[Image.Image]]:
        return [crops for crops, _ in self.detect_with_boxes(image)]


# --------------------------------------------------------------------------- doctr detector
class DoctrLineDetector:
    def __init__(self, arch: str):
        from doctr.models import detection_predictor

        self._pred = detection_predictor(arch=arch, pretrained=True, assume_straight_pages=True)

    def detect_with_boxes(self, image: Image.Image) -> list[tuple[list[Image.Image], tuple[float, float, float, float]]]:
        arr = np.asarray(image.convert("RGB"))
        H, W = arr.shape[:2]
        res = self._pred([arr])[0]
        boxes = res["words"] if isinstance(res, dict) else res
        words = [(b[0] * W, b[1] * H, b[2] * W, b[3] * H) for b in boxes]
        words = [b for b in words if (b[3] - b[1]) >= MIN_LINE_HEIGHT * 0.6]
        out: list[tuple[list[Image.Image], tuple[float, float, float, float]]] = []
        for line in group_words_into_lines(words):
            ly0, ly1 = min(b[1] for b in line), max(b[3] for b in line)
            lx0, lx1 = min(b[0] for b in line), max(b[2] for b in line)
            line_box = (float(lx0), float(ly0), float(lx1), float(ly1))
            lh = ly1 - ly0
            # accumulate words into segments bounded by aspect ratio
            seg: list[tuple[float, float, float, float]] = []
            segments: list[Image.Image] = []
            for b in line:
                trial = seg + [b]
                width = trial[-1][2] - trial[0][0]
                if seg and width / max(lh, 1) > MAX_SEGMENT_ASPECT:
                    segments.append(_pad_crop(image, (seg[0][0], ly0, seg[-1][2], ly1)))
                    seg = [b]
                else:
                    seg = trial
            if seg:
                segments.append(_pad_crop(image, (seg[0][0], ly0, seg[-1][2], ly1)))
            out.append((segments, line_box))
        return out

    def detect(self, image: Image.Image) -> list[list[Image.Image]]:
        return [crops for crops, _ in self.detect_with_boxes(image)]


@once
def _load_detector(kind: str, arch: str) -> LineDetector:
    if kind == "doctr":
        try:
            return DoctrLineDetector(arch)
        except Exception as exc:  # noqa: BLE001 - any load/import failure falls back
            log.warning("doctr line detector unavailable (%s); falling back to projection-profile detector", exc)
    return ProjectionLineDetector()


def get_line_detector() -> LineDetector:
    s = get_settings()
    return _load_detector(s.LINE_DETECTOR, s.DOCTR_DET_ARCH)
