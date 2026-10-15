"""OCR text correction.

A fine-tuned seq2seq model (ByT5 or BART) repairs likely TrOCR errors line by line before grading.
TODO: fine-tune on our OCR error pairs (TrOCR output -> ground-truth transcription). Until a checkpoint is
configured via CORRECTION_MODEL_PATH the stage is a pass-through so the rest of the pipeline is not blocked.
"""
from __future__ import annotations

import logging
from typing import Protocol

from app.core.config import get_settings
from app.core.singleton import once

log = logging.getLogger(__name__)


class Corrector(Protocol):
    name: str

    def correct(self, lines: list[str]) -> list[str]:
        """Return corrected text, same length and order as the input."""


class PassThroughCorrector:
    name = "passthrough"

    def correct(self, lines: list[str]) -> list[str]:
        return list(lines)


class Seq2SeqCorrector:
    """Wrapper for a fine-tuned ByT5 / BART checkpoint (AutoModelForSeq2SeqLM)."""

    def __init__(self, path: str, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = f"seq2seq:{path}"
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).eval()
        self.batch_size = batch_size

    def correct(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        with self._torch.inference_mode():
            for i in range(0, len(lines), self.batch_size):
                batch = lines[i : i + self.batch_size]
                enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
                gen = self.model.generate(**enc, max_new_tokens=256, num_beams=4)
                out.extend(t.strip() for t in self.tok.batch_decode(gen, skip_special_tokens=True))
        return [c or orig for c, orig in zip(out, lines)]


@once
def _load(path: str) -> Corrector:
    if path:
        return Seq2SeqCorrector(path)
    log.info("CORRECTION_MODEL_PATH not set: text correction is a pass-through")
    return PassThroughCorrector()


def get_corrector() -> Corrector:
    return _load(get_settings().CORRECTION_MODEL_PATH)
