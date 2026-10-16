"""Tests for the real-handwriting readers (IAM + manifest) used by the correction dataset builder. No models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training" / "correction"))

from PIL import Image  # noqa: E402

from pm_common import iam_detok, read_iam, read_manifest  # noqa: E402
from pm_noise import split_wide  # noqa: E402


def test_iam_detok_restores_spacing():
    assert iam_detok("A|MOVE|to|stop|Mr.|Gaitskell|,|from|(|any|)|peers|.") == "A MOVE to stop Mr. Gaitskell, from (any) peers."


def test_iam_and_manifest_readers(tmp_path):
    (tmp_path / "ascii").mkdir()
    folder = tmp_path / "lines" / "a01" / "a01-000u"
    folder.mkdir(parents=True)
    for name in ("a01-000u-00", "a01-000u-01"):
        Image.new("L", (300, 40), 255).save(folder / f"{name}.png")
    (tmp_path / "ascii" / "lines.txt").write_text(
        "# comment\n"
        "a01-000u-00 ok 154 19 408 746 1661 89 A|MOVE|to|stop|Mr.|Gaitskell|from\n"
        "a01-000u-01 err 154 19 408 746 1661 89 bad|line|here|ok|ok|ok\n",
        encoding="utf-8",
    )
    assert read_iam(str(tmp_path)) == [(str(folder / "a01-000u-00.png"), "A MOVE to stop Mr. Gaitskell from", "a01-000u")]

    (tmp_path / "own.tsv").write_text("lines/a01/a01-000u/a01-000u-00.png\tHello world\nmissing.png\tx\n", encoding="utf-8")
    assert [t for _, t in read_manifest(str(tmp_path / "own.tsv"))] == ["Hello world"]


def test_split_wide_respects_max_aspect():
    pieces = split_wide(Image.new("RGB", (1500, 60), (255, 255, 255)))
    assert len(pieces) >= 3 and all(p.size[0] / p.size[1] <= 6.6 for p in pieces)
