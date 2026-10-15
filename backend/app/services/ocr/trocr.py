"""TrOCR handwriting recognition (microsoft/trocr-base-handwritten via Hugging Face transformers)."""
from __future__ import annotations

import math
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from app.core.config import get_settings
from app.core.singleton import once


class TrOCRRecognizer:
    def __init__(self, model_name: str):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.inference_mode()
    def recognize(self, images: list[Image.Image], batch_size: int = 8) -> list[tuple[str, float]]:
        """Return (text, confidence) per image. Confidence is the geometric-mean token probability."""
        results: list[tuple[str, float]] = []
        pad = self.processor.tokenizer.pad_token_id
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            pixels = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            out = self.model.generate(
                pixels, max_new_tokens=64, num_beams=1, output_scores=True, return_dict_in_generate=True
            )
            logprobs = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
            tokens = out.sequences[:, 1:]  # drop decoder start token; aligns with transition scores
            texts = self.processor.batch_decode(out.sequences, skip_special_tokens=True)
            for text, lp, tok in zip(texts, logprobs, tokens):
                mask = tok != pad
                mean_lp = float(lp[mask].mean()) if mask.any() else -5.0
                results.append((text.strip(), math.exp(mean_lp)))
        return results


@once
def _load(model_name: str) -> TrOCRRecognizer:
    return TrOCRRecognizer(model_name)


def get_trocr() -> TrOCRRecognizer:
    return _load(get_settings().TROCR_MODEL)
