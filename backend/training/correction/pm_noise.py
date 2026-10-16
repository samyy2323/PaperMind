"""Noise generators that turn clean text into TrOCR-like output.

1. ``simulate``   - fast character-level confusion model (no GPU / model needed).
2. ``TrOCRNoiser`` - renders text in handwriting-style fonts and runs the REAL TrOCR model, so the errors are
   the ones PaperMind's OCR stage actually makes.
"""
from __future__ import annotations

import math
import random
import re
from pathlib import Path

from pm_common import chunk_words

# --------------------------------------------------------------------------- 1. simulated confusions
CONFUSIONS = {
    "l": ["1", "I", "i", "!"], "I": ["l", "1", "i"], "1": ["l", "I", "i"], "0": ["O", "o", "D"], "O": ["0", "o", "Q"],
    "o": ["0", "a", "c"], "m": ["rn", "in", "nn"], "n": ["ri", "u", "h"], "a": ["o", "u", "d"], "e": ["c", "a", "o"],
    "g": ["q", "9", "y"], "q": ["g", "9"], "S": ["5", "s"], "s": ["5", "S", "r"], "Z": ["2", "z"], "z": ["2", "Z"],
    "B": ["8", "R"], "b": ["6", "h", "l"], "G": ["6", "C"], "h": ["b", "li", "n"], "d": ["cl", "a"], "u": ["n", "ii", "v"],
    "v": ["y", "u", "r"], "y": ["v", "g"], "c": ["e", "o", "r"], "t": ["f", "l", "+"], "f": ["t", "s"], "r": ["n", "v", "i"],
    "w": ["vv", "uu"], "k": ["h", "lc"], "i": ["l", "j", "e"], "p": ["P", "b"], "x": ["n", "k"], "H": ["N", "M"],
    "T": ["I", "F"], "U": ["V", "O"], "W": ["V", "N"], "E": ["F", "C"],
}
_PUNCT_SPACE = re.compile(r"([.,:;!?()])")


def simulate(text: str, level: float, rng: random.Random) -> str:
    """Return a noisy version of ``text``. ``level`` in [0, 1]; 0 returns the text unchanged."""
    if level <= 0:
        return text
    out: list[str] = []
    for ch in text:
        r = rng.random()
        if ch in CONFUSIONS and r < level * 0.22:
            out.append(rng.choice(CONFUSIONS[ch]))
        elif ch != " " and r > 1 - level * 0.035:
            continue  # dropped stroke
        elif ch.isalpha() and rng.random() < level * 0.015:
            out.append(ch + ch)  # doubled stroke
        else:
            out.append(ch)
    noisy = "".join(out)

    # TrOCR habit: detached punctuation ("Liskov . principle", "2 ) False")
    if rng.random() < 0.25 + level * 0.5:
        noisy = _PUNCT_SPACE.sub(lambda m: f" {m.group(1)} " if rng.random() < 0.55 else m.group(1), noisy)
    # split / merged words
    words = noisy.split(" ")
    for i in range(len(words)):
        if len(words[i]) > 6 and rng.random() < level * 0.05:
            k = rng.randint(2, len(words[i]) - 2)
            words[i] = words[i][:k] + " " + words[i][k:]
        elif i + 1 < len(words) and rng.random() < level * 0.03:
            words[i] = words[i] + words[i + 1]
            words[i + 1] = ""
    noisy = re.sub(r" {2,}", " ", " ".join(words)).strip()
    # mangled leading question number, e.g. "1) (b)" -> "19 ( 0b ) ."
    if re.match(r"^\(?[A-Za-z]{0,3}\d{1,2}\s*[\.\):\-]", noisy) and rng.random() < level * 0.35:
        noisy = re.sub(r"^(\(?[A-Za-z]{0,3})(\d{1,2})\s*([\.\):\-])", lambda m: f"{m.group(1)}{m.group(2)}{rng.choice(['9', '0', ' ', ''])} {m.group(3)} ", noisy, 1)
        noisy = re.sub(r" {2,}", " ", noisy).strip()
    return noisy


# --------------------------------------------------------------------------- 2. real TrOCR noise
MAX_SEGMENT_ASPECT = 6.5  # same bound the backend's line detector uses before TrOCR


def split_wide(img, max_aspect: float = MAX_SEGMENT_ASPECT):
    """Cut a wide line image at its emptiest columns so each piece has aspect <= max_aspect (as production does)."""
    import numpy as np

    w, h = img.size
    if w / max(h, 1) <= max_aspect:
        return [img]
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    ink = (gray < gray.mean() - 0.5 * gray.std()).sum(axis=0).astype(float)
    ink = np.convolve(ink, np.ones(5) / 5, mode="same")
    max_w = int(max_aspect * h)
    cuts, start = [0], 0
    while w - start > max_w:
        lo, hi = start + int(max_w * 0.55), start + max_w
        cut = lo + int(np.argmin(ink[lo:hi]))
        cuts.append(cut)
        start = cut
    cuts.append(w)
    return [img.crop((a, 0, b, h)) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 4]


def augment(img, rng: random.Random):
    """Light photo-like variation so repeated passes over the same real line give different TrOCR errors."""
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    img = img.convert("RGB")
    if rng.random() < 0.7:
        img = img.rotate(rng.uniform(-2.0, 2.0), fillcolor=(255, 255, 255), resample=Image.BICUBIC, expand=True)
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.4:
        arr = np.asarray(img, dtype=np.float32) + np.random.normal(0, rng.uniform(2, 10), (img.size[1], img.size[0], 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    return img
def find_fonts(font_dir: str | None) -> list[str]:
    paths: list[str] = []
    if font_dir and Path(font_dir).is_dir():
        paths += [str(p) for p in sorted(Path(font_dir).glob("*")) if p.suffix.lower() in (".ttf", ".otf")]
    for p in ("C:/Windows/Fonts/Inkfree.ttf", "C:/Windows/Fonts/segoesc.ttf", "C:/Windows/Fonts/LHANDW.TTF",
              "C:/Windows/Fonts/BRUSHSCI.TTF"):
        if Path(p).exists():
            paths.append(p)
    return paths


def render_segment(text: str, font_path: str, rng: random.Random):
    """Draw a text segment word by word with jitter so it looks hand-written rather than typeset."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    size = rng.randint(34, 58)
    font = ImageFont.truetype(font_path, size)
    pad = int(size * 0.5)
    words = text.split(" ")
    advances = [font.getlength(w + " ") * rng.uniform(0.92, 1.12) for w in words]
    width = int(sum(advances)) + 2 * pad
    height = int(size * 1.9)
    img = Image.new("RGB", (width, height), (rng.randint(235, 255),) * 3)
    d = ImageDraw.Draw(img)
    ink = (rng.randint(0, 50), rng.randint(0, 50), rng.randint(20, 110))
    x = float(pad)
    for w, adv in zip(words, advances):
        d.text((x, pad * 0.6 + rng.uniform(-3, 3)), w, font=font, fill=ink)
        x += adv
    if rng.random() < 0.7:
        img = img.rotate(rng.uniform(-2.5, 2.5), fillcolor=(255, 255, 255), resample=Image.BICUBIC)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.3)))
    if rng.random() < 0.5:
        arr = np.asarray(img, dtype=np.float32) + np.random.normal(0, rng.uniform(2, 14), (height, width, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    return img


class TrOCRNoiser:
    MODEL = "microsoft/trocr-base-handwritten"

    def __init__(self, fonts: list[str], seed: int = 0, batch_size: int = 64, model: str | None = None):
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        if not fonts:
            raise RuntimeError("TrOCR noise needs at least one handwriting-style font (see --font-dir).")
        self.fonts, self.rng, self.batch_size = fonts, random.Random(seed), batch_size
        self.torch = torch
        self.processor = TrOCRProcessor.from_pretrained(model or self.MODEL)
        self.model = VisionEncoderDecoderModel.from_pretrained(model or self.MODEL).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def read_lines(self, lines: list[str]) -> list[tuple[str, float]]:
        """Render each line as bounded-width segments (like the production detector), OCR them, re-join."""
        segs: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            font = self.rng.choice(self.fonts)  # one font per line, as one writer writes one line
            for chunk in chunk_words(line):
                segs.append((i, chunk, font))
        texts: list[tuple[str, float]] = []
        order = sorted(range(len(segs)), key=lambda k: len(segs[k][1]))  # similar widths batch together
        results: dict[int, tuple[str, float]] = {}
        for s in range(0, len(order), self.batch_size):
            idx = order[s : s + self.batch_size]
            imgs = [render_segment(segs[k][1], segs[k][2], self.rng) for k in idx]
            for k, res in zip(idx, self._ocr(imgs)):
                results[k] = res
        joined: dict[int, list[tuple[str, float]]] = {}
        for k, (i, _chunk, _font) in enumerate(segs):
            joined.setdefault(i, []).append(results[k])
        for i in range(len(lines)):
            parts = joined.get(i, [])
            text = " ".join(t for t, _ in parts).strip()
            conf = sum(c for _, c in parts) / max(len(parts), 1)
            texts.append((text, conf))
        return texts

    def read_images(self, paths: list[str], augment_images: bool = False) -> list[tuple[str, float]]:
        """OCR REAL line photos/scans (IAM, your own pages): bounded-width segments -> TrOCR -> re-join."""
        from PIL import Image, ImageOps

        segs = []
        for i, path in enumerate(paths):
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            if augment_images:
                img = augment(img, self.rng)
            segs += [(i, piece) for piece in split_wide(img)]
            if i % 2000 == 0:
                print(f"  loading real images {i:,}/{len(paths):,}", flush=True)
        order = sorted(range(len(segs)), key=lambda k: segs[k][1].size[0])
        results = {}
        total = len(order)
        for s in range(0, total, self.batch_size):
            idx = order[s : s + self.batch_size]
            for k, res in zip(idx, self._ocr([segs[k][1] for k in idx])):
                results[k] = res
            if (s // self.batch_size) % 40 == 0:
                print(f"  OCR real segments {min(s + self.batch_size, total):,}/{total:,}", flush=True)
        grouped = {}
        for k, (i, _) in enumerate(segs):
            grouped.setdefault(i, []).append(results[k])
        out = []
        for i in range(len(paths)):
            parts = grouped.get(i, [])
            out.append((" ".join(t for t, _ in parts).strip(), sum(c for _, c in parts) / max(len(parts), 1)))
        return out

    def _ocr(self, images) -> list[tuple[str, float]]:
        torch = self.torch
        with torch.inference_mode():
            pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
            out = self.model.generate(pixels, max_new_tokens=64, num_beams=1, output_scores=True, return_dict_in_generate=True)
            lp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            tokens = out.sequences[:, 1:]
            pad = self.processor.tokenizer.pad_token_id
            texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
        res = []
        for t, l, tok in zip(texts, lp, tokens):
            mask = tok != pad
            res.append((t.strip(), math.exp(float(l[mask].mean())) if mask.any() else 0.0))
        return res
