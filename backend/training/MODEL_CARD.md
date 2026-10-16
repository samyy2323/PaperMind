# PaperMind OCR-correction model v1 (ByT5-base)

A `google/byt5-base` model fine-tuned to repair **TrOCR** (`microsoft/trocr-base-handwritten`) recognition errors, one page
line at a time, before PaperMind grades an answer sheet. It is used by `backend/app/services/correction`.

**Get it:** `python backend/scripts/fetch_model.py`, then set `CORRECTION_MODEL_PATH=./models/byt5-correction` in `backend/.env`.

## Training
* 582M parameters, 4 epochs, ~31 min on one A100, fp32 weights with bf16 autocast (`backend/training/`).
* ~71k training lines, split **by sentence** (test sentences never seen in training):
  synthetic text rendered in handwriting-style fonts and read by the real TrOCR model, simulated recognition noise,
  and real handwriting lines from the IAM database (TrOCR output vs. the ground-truth transcript).
* "Fact-swapped" sentences (e.g. stack<->queue) are included so the model learns to repair recognition errors **without
  correcting a student's factual mistakes**.

## Held-out results (4,183 lines, greedy decoding)
| Data | Character error rate before -> after |
|---|---|
| Overall | 0.105 -> 0.036 |
| Real IAM handwriting | 0.135 -> 0.089 |
| Lines that were already correct | changed in 3.7% of cases |
| Student mistakes "fixed" back to the true fact | 0 of 54 |

## Known limitations (read before relying on it)
* **Two-digit question numbers.** v1 was trained with question markers numbered 1-10 only, so on a real exam it shortened
  numbers such as `24` to `2`. The backend therefore never lets the model touch a leading question number or any digit
  (`split_prefix` / `finalize_body`). A v2 with 1-50 numbering is planned.
* Gains on real handwriting are moderate (about a third fewer character errors on IAM); the large gains are on the
  synthetic data it was trained on. Always keep the backend guard (`apply_guard`).
* Checkpoint note: `config.json` has `tie_word_embeddings=false` and `tokenizer_config.json` omits `extra_special_tokens`
  so it loads on transformers 4.x (the backend pins `<5`).

## Licence and data terms
* Base model: `google/byt5-base` (Apache-2.0).
* Training data: the **IAM Handwriting Database** (Marti & Bunke, 2002), which is licensed for **non-commercial research
  use** - treat this model the same way and cite IAM; Simple English Wikipedia text (CC BY-SA 4.0); handwriting-style fonts
  under the SIL Open Font License. No IAM images or transcripts are included in this repository or release.
