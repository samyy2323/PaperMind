#!/usr/bin/env python3
"""Fine-tune ByT5 to repair TrOCR errors. Works on Colab (A100) and, for smoke tests, on CPU.

    python train.py --data data --out ckpt --model google/byt5-base --epochs 4 --bs 64
    python train.py --data /tmp/smoke --out /tmp/ck --model google/byt5-small --max-steps 20 --bs 8   # smoke test

Design notes
  * NO version pins: it uses whatever transformers/torch Colab already has (pinning old tokenizers/transformers is
    what made the previous notebook hang while compiling Rust wheels). Version differences in Trainer arguments are
    handled below.
  * Weights stay in fp32 and bf16 is used for autocast only. Loading the model in pure bf16 loses small AdamW
    updates and trains noticeably worse.
  * Model selection is by validation CER of generated text, on sentences the model never trained on.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_correction import correct_texts, load_jsonl, score  # noqa: E402
from pm_common import cer  # noqa: E402


def make_portable(ckpt: Path) -> None:
    """Make a checkpoint written by a NEW transformers (Colab) loadable by older ones (the backend's 4.x).

    ByT5 keeps a separate output layer (lm_head.weight), but newer versions write tie_word_embeddings=true, which makes an
    older version overwrite it with the input embeddings -> gibberish output. They also write the extra-token list twice,
    which older tokenizers reject.
    """
    cfg_path, tok_path = Path(ckpt) / "config.json", Path(ckpt) / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("tie_word_embeddings") is not False:
            cfg["tie_word_embeddings"] = False
            cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if tok_path.exists():
        tcfg = json.loads(tok_path.read_text(encoding="utf-8"))
        if isinstance(tcfg.get("extra_special_tokens"), list) and tcfg.get("extra_special_tokens") == tcfg.get("additional_special_tokens"):
            del tcfg["extra_special_tokens"]
            tok_path.write_text(json.dumps(tcfg, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--model", default="google/byt5-base")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1, help="effective batch = bs * grad-accum (use for byt5-large)")
    ap.add_argument("--patience", type=int, default=2, help="stop after this many epochs without validation improvement (0 = off)")
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--max-len", type=int, default=192, help="bytes; lines are <~120 characters")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke test)")
    ap.add_argument("--val-limit", type=int, default=2000, help="validation lines used for model selection")
    ap.add_argument("--seed", type=int, default=42)
    args, _ignored = ap.parse_known_args(argv)  # tolerate options that belong to the data stage

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)

    set_seed(args.seed)
    cuda = torch.cuda.is_available()
    print(f"device: {'cuda ' + torch.cuda.get_device_name(0) if cuda else 'cpu'}")
    if cuda and not torch.cuda.is_bf16_supported():
        print("NOTE: this GPU has no bf16 (T4/P100): training runs in fp32 (fp16 is unstable for ByT5), so use byt5-small or expect it to be slow.")

    data = Path(args.data)
    train_rows, val_rows = load_jsonl(data / "train.jsonl"), load_jsonl(data / "val.jsonl", args.val_limit)
    print(f"train={len(train_rows):,}  val={len(val_rows):,}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(batch):
        enc = tok(batch["input"], max_length=args.max_len, truncation=True)
        enc["labels"] = tok(text_target=batch["target"], max_length=args.max_len, truncation=True)["input_ids"]
        return enc

    def to_ds(rows):
        ds = Dataset.from_dict({"input": [r["input"] for r in rows], "target": [r["target"] for r in rows]})
        return ds.map(encode, batched=True, remove_columns=["input", "target"])

    train_ds, val_ds = to_ds(train_rows), to_ds(val_rows)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)  # fp32 master weights
    print(f"model: {args.model}  {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    def compute_metrics(pred):
        preds, labels = pred
        preds = preds[0] if isinstance(preds, tuple) else preds
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        p = [s.strip() for s in tok.batch_decode(preds, skip_special_tokens=True)]
        r = [s.strip() for s in tok.batch_decode(labels, skip_special_tokens=True)]
        return {"cer": round(cer(p, r), 5), "exact": round(sum(a == b for a, b in zip(p, r)) / len(r), 4)}

    # ---- Trainer argument names differ between transformers releases ----------------------------------
    # (v5 removed warmup_ratio, so compute an explicit number of warmup steps that every version accepts)
    steps_per_epoch = max(1, -(-len(train_ds) // (args.bs * args.grad_accum)))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup))
    print(f"steps/epoch={steps_per_epoch}  total={total_steps}  warmup={warmup_steps}")
    ta = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    tr = inspect.signature(Seq2SeqTrainer.__init__).parameters
    kwargs = dict(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=args.eval_bs, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=warmup_steps, weight_decay=0.01,
        predict_with_generate=True, generation_max_length=args.max_len,
        save_strategy="epoch" if args.max_steps < 0 else "no", save_total_limit=2,
        load_best_model_at_end=args.max_steps < 0, metric_for_best_model="cer", greater_is_better=False,
        bf16=cuda and torch.cuda.is_bf16_supported(), logging_steps=50, report_to="none", seed=args.seed,
        dataloader_num_workers=2 if cuda else 0,
    )
    kwargs["eval_strategy" if "eval_strategy" in ta else "evaluation_strategy"] = "epoch" if args.max_steps < 0 else "no"
    dropped = [k for k in kwargs if k not in ta]  # an argument this transformers version doesn't know: skip, don't crash
    if dropped:
        print(f"note: ignoring unsupported TrainingArguments {dropped}")
        kwargs = {k: v for k, v in kwargs.items() if k in ta}
    trainer_kwargs = dict(model=model, args=Seq2SeqTrainingArguments(**kwargs), train_dataset=train_ds, eval_dataset=val_ds,
                          data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True, pad_to_multiple_of=8),
                          compute_metrics=compute_metrics)
    trainer_kwargs["processing_class" if "processing_class" in tr else "tokenizer"] = tok
    if args.patience > 0 and args.max_steps < 0:
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # Resume automatically if an earlier run in this folder was interrupted (disconnect, quota, closed laptop).
    last_ckpt = None
    if args.max_steps < 0 and Path(args.out).is_dir():
        from transformers.trainer_utils import get_last_checkpoint

        last_ckpt = get_last_checkpoint(args.out)
        if last_ckpt:
            print(f"resuming from {last_ckpt}")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=last_ckpt)
    print(f"\ntrained in {(time.time() - t0) / 60:.1f} min  {result.metrics}")

    out = Path(args.out)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    make_portable(out)

    # ---- held-out test report --------------------------------------------------------------------------
    test_rows = load_jsonl(data / "test.jsonl")
    preds = correct_texts(trainer.model, tok, [r["input"] for r in test_rows], args.eval_bs, beams=1, max_len=args.max_len)
    report = score(test_rows, preds)
    (out / "test_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== TEST (sentences never seen in training) ===")
    print(json.dumps(report, indent=2))
    print(f"\nCheckpoint saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
