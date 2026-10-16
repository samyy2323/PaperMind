# =====================================================================================================
# Unified entry point (appended by bundle.py; not used when the stage scripts are run separately)
# =====================================================================================================
FONT_URLS = {
    "IndieFlower.ttf": "https://github.com/google/fonts/raw/main/ofl/indieflower/IndieFlower-Regular.ttf",
    "PatrickHand.ttf": "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf",
    "Kalam.ttf": "https://github.com/google/fonts/raw/main/ofl/kalam/Kalam-Regular.ttf",
    "ShadowsIntoLight.ttf": "https://github.com/google/fonts/raw/main/ofl/shadowsintolight/ShadowsIntoLight.ttf",
}

# Defaults per size; anything you pass on the command line overrides them.
PRESETS = {
    "smoke": ["--model", "google/byt5-small", "--n-lines", "3000", "--wiki-n", "1500", "--epochs", "1", "--bs", "32",
              "--val-limit", "300"],
    "small": ["--model", "google/byt5-small", "--n-lines", "40000", "--wiki-n", "15000", "--epochs", "5", "--bs", "64",
              "--lr", "3e-4"],
    "base": ["--model", "google/byt5-base", "--n-lines", "120000", "--wiki-n", "40000", "--epochs", "6", "--bs", "64",
             "--lr", "2e-4", "--iam-lines", "10000", "--iam-repeat", "2"],
    "large": ["--model", "google/byt5-large", "--n-lines", "120000", "--wiki-n", "40000", "--epochs", "4", "--bs", "16",
              "--grad-accum", "4", "--eval-bs", "32", "--lr", "1e-4", "--iam-lines", "10000", "--iam-repeat", "2"],
}


def _opt(args: list[str], name: str, default: str) -> str:
    """Last value given for --name in an argument list."""
    value = default
    for i, a in enumerate(args[:-1]):
        if a == name:
            value = args[i + 1]
    return value


def ensure_fonts(dest: str) -> str:
    """Download a few open-licence handwriting fonts (skipped quietly when offline)."""
    import urllib.request

    Path(dest).mkdir(parents=True, exist_ok=True)
    for name, url in FONT_URLS.items():
        target = Path(dest) / name
        if target.exists() and target.stat().st_size > 10_000:
            continue
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not download font {name}: {exc}", file=sys.stderr)
    return dest


def main(argv=None) -> int:
    """
    All-in-one: build data -> train ByT5 -> evaluate.

      main(["--preset", "base", "--out", "/kaggle/working/run1", "--iam-dir", "/kaggle/input/iam-handwriting"])

    --preset   smoke | small | base | large   (smoke = ~5 minute sanity run)
    --stage    all | data | train | eval      (default all; finished stages are skipped when you re-run)
    --out      working folder: data/, ckpt/, fonts/, final_report.json
    Everything else (e.g. --epochs 8 --n-lines 200000 --iam-dir ... --manifest ... --test-manifest ...) is passed to
    the data / training stages and overrides the preset.
    """
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--out", default="papermind_run")
    ap.add_argument("--preset", default="base", choices=sorted(PRESETS))
    ap.add_argument("--stage", default="all", choices=["all", "data", "train", "eval"])
    ap.add_argument("--rebuild", action="store_true", help="rebuild the dataset even if one exists")
    ap.add_argument("--font-dir", default=None)
    known, rest = ap.parse_known_args(argv)
    args = PRESETS[known.preset] + rest

    out = Path(known.out)
    data_dir, ckpt_dir = out / "data", out / "ckpt"
    out.mkdir(parents=True, exist_ok=True)
    print(f"preset={known.preset}  stage={known.stage}  out={out}")

    if known.stage in ("all", "data"):
        if (data_dir / "train.jsonl").exists() and not known.rebuild:
            print(f"dataset already at {data_dir} - skipping (use --rebuild to regenerate)")
        else:
            font_dir = known.font_dir
            if float(_opt(args, "--trocr-frac", "0.35")) > 0 and not font_dir:
                font_dir = ensure_fonts(str(out / "fonts"))
            _main_build(["--out", str(data_dir)] + (["--font-dir", font_dir] if font_dir else []) + args)

    if known.stage in ("all", "train"):
        if not (data_dir / "train.jsonl").exists():
            raise SystemExit(f"no dataset at {data_dir}; run --stage data first")
        _main_train(["--data", str(data_dir), "--out", str(ckpt_dir)] + args)

    if known.stage in ("all", "eval"):
        _main_eval(["--model", str(ckpt_dir), "--data", str(data_dir), "--beams", "4", "--report", str(out / "final_report.json")])

    print(f"\nDone. Model: {ckpt_dir}   Report: {out / 'final_report.json'}")
    return 0


if __name__ == "__main__" and "ipykernel" not in sys.modules:  # in a notebook, call main([...]) yourself
    raise SystemExit(main())
