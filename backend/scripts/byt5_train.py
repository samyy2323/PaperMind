#!/usr/bin/env python3
"""
PaperMind — ByT5 OCR Correction Training Script
================================================
Run this directly in a Colab terminal (Tools → Terminal):

    # Step 1: Mount Drive first (run in a Colab cell once)
    #   from google.colab import drive; drive.mount('/content/drive')

    # Step 2: Run this script
    python byt5_train.py

Optimised for A100 (40GB VRAM) with 50k pairs.
Expected total runtime: ~60-80 minutes.
"""

import os, sys, time, random, math, subprocess

# ── 0. Install deps (idempotent) ─────────────────────────────────────────────
print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "tokenizers==0.19.1"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "transformers==4.44.2", "datasets", "evaluate",
                       "sacrebleu", "sentencepiece", "accelerate", "Pillow"])
print("Dependencies ready.\n")

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────
N_PAIRS            = 50_000
TROCR_BATCH        = 64
CLEAN_FRAC         = 0.15

BYT5_MODEL         = "google/byt5-base"
MAX_LEN            = 256

EPOCHS             = 5
BYT5_TRAIN_BATCH   = 64
BYT5_EVAL_BATCH    = 128
LEARNING_RATE      = 4e-4
WARMUP_STEPS       = 300
GRAD_ACCUM_STEPS   = 1

DRIVE_DATA_DIR = "/content/drive/MyDrive/papermind/correction"
DRIVE_CKPT_DIR = "/content/drive/MyDrive/papermind/byt5-base-correction-checkpoint"

print(f"Config: {N_PAIRS:,} pairs | {BYT5_MODEL} | {EPOCHS} epochs | bs={BYT5_TRAIN_BATCH}")

# ── 2. IMPORTS ────────────────────────────────────────────────────────────────
import torch, numpy as np, json, pathlib, re
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrOCRProcessor, VisionEncoderDecoderModel,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print()

# ── 3. LEVENSHTEIN ────────────────────────────────────────────────────────────
def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j-1] + 1,
                          prev[j-1] + (0 if ca == cb else 1))
        prev = curr
    return prev[-1]

# ── 4. SYNTHETIC TEXT CORPUS ──────────────────────────────────────────────────
SUBJECTS = [
    "Encapsulation bundles data and methods together.",
    "Inheritance allows a subclass to extend a superclass.",
    "Polymorphism enables objects to take multiple forms.",
    "Abstraction hides implementation details from the user.",
    "The Liskov substitution principle states that objects of a superclass"
    " should be replaceable with objects of a subclass.",
    "A linked list is a linear data structure of nodes.",
    "Binary search runs in O(log n) time complexity.",
    "Quicksort has average case O(n log n) complexity.",
    "Newton's second law states F equals m times a.",
    "Ohm's law states V equals I times R.",
    "Photosynthesis converts carbon dioxide and water into glucose.",
    "The mitochondria is the powerhouse of the cell.",
    "Supply and demand determine market equilibrium price.",
    "The French Revolution began in 1789.",
    "The Python language uses indentation for block structure.",
    "A stack follows the last-in first-out principle.",
    "A queue follows the first-in first-out principle.",
    "Dynamic programming breaks problems into overlapping subproblems.",
    "Recursion is a function that calls itself with a base case.",
    "A hash table provides O(1) average-case lookup.",
]

MARKERS = [
    "{n}) {text}",
    "Q{n}. {text}",
    "({n}) {text}",
    "{n}. {text}",
    "Q{n}) {text}",
]

SUB_MARKERS = ["(a)", "(b)", "(c)", "(d)", "(i)", "(ii)", "(iii)"]

def make_clean_line(rng: random.Random) -> str:
    template = rng.choice(MARKERS)
    n = rng.randint(1, 10)
    text = rng.choice(SUBJECTS)
    line = template.format(n=n, text=text)
    if rng.random() < 0.3:
        sub = rng.choice(SUB_MARKERS)
        line = f"{sub} {line}"
    return line

# ── 5. TROCR NOISE ENGINE ─────────────────────────────────────────────────────
CHAR_CONFUSIONS = {
    "l": ["1", "I", "i", "!"],
    "I": ["l", "1", "i"],
    "1": ["l", "I", "i"],
    "0": ["O", "o", "D"],
    "O": ["0", "o", "Q"],
    "o": ["0", "O", "a"],
    "m": ["rn", "in", "ni"],
    "rn": ["m"],
    "n": ["ri", "ii"],
    "a": ["o", "u", "ci"],
    "e": ["c", "a"],
    "g": ["q", "9"],
    "q": ["g", "9"],
    "S": ["5", "s"],
    "s": ["5", "S"],
    "Z": ["2", "z"],
    "z": ["2", "Z"],
    "B": ["8", "b"],
    "b": ["6", "B"],
    "G": ["6", "C"],
    "h": ["b", "li"],
    "d": ["cl", "o"],
    "u": ["ii", "n"],
    "v": ["y", "u"],
    "y": ["v", "Y"],
    "c": ["e", "o"],
    "t": ["f", "+"],
    "f": ["t"],
    "r": ["i", "n"],
    "w": ["vv", "uu"],
    ".": [". ", " ."],
    ",": [" ,", ", "],
    ")": [" )", ") "],
    "(": ["( ", " ("],
}

def apply_trocr_noise(text: str, noise_level: float, rng: random.Random) -> str:
    """Simulate realistic TrOCR character-level OCR errors."""
    if noise_level == 0.0:
        return text

    chars = list(text)
    result = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        # Try two-char confusions first
        two = (ch + chars[i+1]) if i+1 < len(chars) else None
        if two and two in CHAR_CONFUSIONS and rng.random() < noise_level * 0.5:
            result.extend(list(rng.choice(CHAR_CONFUSIONS[two])))
            i += 2
            continue
        if ch in CHAR_CONFUSIONS and rng.random() < noise_level:
            result.extend(list(rng.choice(CHAR_CONFUSIONS[ch])))
        else:
            result.append(ch)
        i += 1

    # Space insertion around punctuation (common TrOCR error)
    noisy = "".join(result)
    if noise_level > 0.2 and rng.random() < noise_level * 0.5:
        noisy = re.sub(r"([.,:;!?])", lambda m: f" {m.group()} " if rng.random() < 0.4 else m.group(), noisy)
        noisy = re.sub(r" {2,}", " ", noisy).strip()

    # Random char deletion (ink gaps)
    if noise_level > 0.3:
        chars2 = list(noisy)
        noisy = "".join(c for c in chars2 if rng.random() > noise_level * 0.12)

    return noisy

# ── 6. GENERATE SYNTHETIC DATASET ─────────────────────────────────────────────
NOISE_LEVELS = [0.0, 0.0, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75]
# 0.0 appears twice → 25% of pairs are clean (model must not over-correct)

def generate_dataset(n_pairs: int, seed: int = 42):
    rng = random.Random(seed)
    pairs = []
    t0 = time.time()

    for idx in range(n_pairs):
        noise_level = rng.choice(NOISE_LEVELS)
        clean = make_clean_line(rng)

        if noise_level == 0.0 and rng.random() > CLEAN_FRAC:
            # Replace clean pair with a noisy one to hit CLEAN_FRAC target
            noise_level = rng.choice([l for l in NOISE_LEVELS if l > 0])

        noisy = apply_trocr_noise(clean, noise_level, rng)

        # Skip pairs where noise made no change and we don't want a clean pair
        if noisy == clean and noise_level > 0:
            noisy = apply_trocr_noise(clean, max(noise_level, 0.3), rng)

        pairs.append({"noisy": noisy, "clean": clean})

        if (idx + 1) % 10_000 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (n_pairs - idx - 1) / rate
            print(f"  Generated {idx+1:,}/{n_pairs:,}  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    print(f"Dataset generated: {len(pairs):,} pairs in {time.time()-t0:.1f}s")
    return pairs


print("=" * 65)
print("Step 1/5: Generating 50k synthetic training pairs...")
print("=" * 65)
t_gen = time.time()
pairs = generate_dataset(N_PAIRS)
print(f"Generation done in {time.time()-t_gen:.1f}s\n")

# Show sample
print("Sample pairs:")
for p in random.sample(pairs, 5):
    print(f"  NOISY : {p['noisy']}")
    print(f"  CLEAN : {p['clean']}")
    print(f"  ED    : {levenshtein(p['noisy'], p['clean'])}")
    print()

# ── 7. SPLIT & SAVE ────────────────────────────────────────────────────────────
random.shuffle(pairs)
n_val  = int(N_PAIRS * 0.05)   # 5% = 2500 val
n_test = int(N_PAIRS * 0.05)   # 5% = 2500 test
n_train = N_PAIRS - n_val - n_test

split = {
    "train": pairs[:n_train],
    "val":   pairs[n_train:n_train + n_val],
    "test":  pairs[n_train + n_val:],
}
print(f"Split: train={n_train:,}  val={n_val:,}  test={n_test:,}")

pathlib.Path(DRIVE_DATA_DIR).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(DRIVE_DATA_DIR, "correction_50k.jsonl")
with open(data_path, "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")
print(f"Saved dataset → {data_path}\n")

# ── 8. TOKENISE ───────────────────────────────────────────────────────────────
print("=" * 65)
print("Step 2/5: Tokenising with ByT5...")
print("=" * 65)

tok = AutoTokenizer.from_pretrained(BYT5_MODEL)

def tokenise_batch(batch):
    model_inputs = tok(
        batch["noisy"], max_length=MAX_LEN, truncation=True, padding=False
    )
    with tok.as_target_tokenizer():
        labels = tok(
            batch["clean"], max_length=MAX_LEN, truncation=True, padding=False
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

datasets = DatasetDict({
    k: Dataset.from_list(v) for k, v in split.items()
})
t_tok = time.time()
tokenized = datasets.map(
    tokenise_batch,
    batched=True, batch_size=2000,
    remove_columns=["noisy", "clean"],
    desc="Tokenising",
)
print(f"Tokenisation done in {time.time()-t_tok:.1f}s\n")

# ── 9. LOAD MODEL ─────────────────────────────────────────────────────────────
print("=" * 65)
print("Step 3/5: Loading byt5-base (580M params)...")
print("=" * 65)

model = AutoModelForSeq2SeqLM.from_pretrained(
    BYT5_MODEL, torch_dtype=torch.bfloat16
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {BYT5_MODEL}  |  {n_params/1e6:.0f}M params")
if device == "cuda":
    print(f"VRAM after load: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

# ── 10. METRICS ───────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    dec_preds  = tok.batch_decode(preds, skip_special_tokens=True)
    labels_np  = np.where(labels != -100, labels, tok.pad_token_id)
    dec_labels = tok.batch_decode(labels_np, skip_special_tokens=True)

    exact = sum(p.strip() == l.strip()
                for p, l in zip(dec_preds, dec_labels)) / len(dec_preds)

    total_ed, total_len = 0, 0
    for p, l in zip(dec_preds, dec_labels):
        total_ed  += levenshtein(p.strip(), l.strip())
        total_len += max(len(l.strip()), 1)

    return {
        "exact_match": round(exact, 4),
        "cer":         round(total_ed / total_len, 4),
    }

# ── 11. TRAINER ───────────────────────────────────────────────────────────────
print("=" * 65)
print("Step 4/5: Training ByT5-base (~25-35 min on A100)...")
print("=" * 65)

args = Seq2SeqTrainingArguments(
    output_dir=DRIVE_CKPT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BYT5_TRAIN_BATCH,
    per_device_eval_batch_size=BYT5_EVAL_BATCH,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    predict_with_generate=True,
    generation_max_length=MAX_LEN,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    dataloader_num_workers=4,
    logging_steps=100,
    report_to="none",
)

collator = DataCollatorForSeq2Seq(tok, model=model, padding=True, pad_to_multiple_of=8)
trainer  = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

steps = len(tokenized["train"]) // (BYT5_TRAIN_BATCH * max(GRAD_ACCUM_STEPS, 1))
print(f"Train={len(tokenized['train']):,}  Val={len(tokenized['val']):,}")
print(f"Steps/epoch ≈ {steps:,}  |  Total steps ≈ {steps * EPOCHS:,}\n")

t_train = time.time()
result  = trainer.train()
train_time = (time.time() - t_train) / 60
print(f"\nTraining done in {train_time:.1f} min")
print(result.metrics)

# ── 12. SAVE CHECKPOINT ───────────────────────────────────────────────────────
print(f"\nSaving model → {DRIVE_CKPT_DIR}")
trainer.save_model(DRIVE_CKPT_DIR)
tok.save_pretrained(DRIVE_CKPT_DIR)
print("Files:", sorted(os.listdir(DRIVE_CKPT_DIR)))

# ── 13. TEST SET EVAL ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 5/5: Evaluating on held-out test set...")
print("=" * 65)

test_res = trainer.evaluate(tokenized["test"])
em  = test_res.get("eval_exact_match", 0)
cer = test_res.get("eval_cer", 1)

print(f"\n=== Test Results (byt5-base, 50k pairs) ===")
print(f"  Exact Match : {em:.4f}  ({100*em:.1f}% of lines corrected perfectly)")
print(f"  CER         : {cer:.4f}  ({100*cer:.2f}% char error rate after correction)")
print(f"  Loss        : {test_res.get('eval_loss', 0):.4f}")
print()
if   cer < 0.03: print("✓ Excellent — CER < 3%. Ready for PaperMind production.")
elif cer < 0.05: print("~ Good — CER < 5%. Usable; consider +1 epoch.")
else:            print("✗ CER >= 5%. Inspect qualitative errors below.")

# ── 14. QUALITATIVE CHECK — REAL PAPERMIND TrOCR ERRORS ──────────────────────
print("\n=== Qualitative check — real PaperMind OCR errors ===")
real_errors = [
    ("19 ( 0b ) .",                                "1) (b)."),
    ("2 ) False",                                  "2) False"),
    ("3 ) Liskov . principle says objects of a",   "3) Liskov principle says objects of a"),
    ("program . Square extending . i Rectangle",   "program. Square extending Rectangle"),
    ("4 ) Encapsulation bundles data and methods", "4) Encapsulation bundles data and methods"),
    ("lnheritance aIIows a subclass",              "Inheritance allows a subclass"),
    ("0bject-oriented prograrnming",               "Object-oriented programming"),
    ("Q1 . (b)",                                   "Q1. (b)"),
    ("rn ethod overriding",                        "method overriding"),
]

model.eval()
dev = next(model.parameters()).device
exact_q, improved_q, missed_q = 0, 0, 0

for noisy, expected in real_errors:
    inp = tok(noisy, return_tensors="pt", max_length=256, truncation=True)
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256, num_beams=4)
    pred   = tok.decode(out[0], skip_special_tokens=True).strip()
    ed_in  = levenshtein(noisy, expected)
    ed_out = levenshtein(pred,  expected)

    if pred == expected:
        tag = "✓ EXACT   "; exact_q += 1
    elif ed_out < ed_in:
        tag = "~ IMPROVED"; improved_q += 1
    else:
        tag = "✗ MISS    "; missed_q += 1

    print(f"{tag}  \"{noisy}\"")
    if pred != expected:
        print(f"           → \"{pred}\"  (expected \"{expected}\")")
    print(f"           ED {ed_in} → {ed_out}")
    print()

total_q = len(real_errors)
print(f"Result: {exact_q}/{total_q} exact  "
      f"{improved_q}/{total_q} improved  {missed_q}/{total_q} missed")

# ── 15. DONE ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CHECKPOINT READY — integrate into PaperMind backend")
print("=" * 65)
print(f"\nModel  : {BYT5_MODEL}")
print(f"Pairs  : {N_PAIRS:,}")
print(f"Test CER       : {cer:.4f}")
print(f"Test ExactMatch: {em:.4f}")
print(f"\nCheckpoint location in Drive:")
print(f"  {DRIVE_CKPT_DIR}")
print()
print("Integration steps:")
print("  1. Drive → right-click folder → Download as .zip")
print("  2. Extract to C:\\PaperMind\\backend\\models\\byt5-correction\\")
print("     (needs: config.json, tokenizer.json, model.safetensors)")
print("  3. Add to C:\\PaperMind\\backend\\.env:")
print("     CORRECTION_MODEL_PATH=./models/byt5-correction")
print("  4. Restart backend — log should say: seq2seq:./models/byt5-correction")
print("  5. Re-process an answer sheet → verify corrected.json")
print('     "19 ( 0b ) ." should now read "1) (b)."')
