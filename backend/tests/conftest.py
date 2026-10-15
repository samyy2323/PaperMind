"""Test harness: SQLite DB + temp storage + inline pipeline, with deterministic fakes for the heavy models
(TrOCR, sentence-transformers, the LLM). The real models are exercised by scripts/e2e_demo.py."""
from __future__ import annotations

import io
import json
import re
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.core.security import hash_password


# ----------------------------------------------------------------------------- fakes
class FakeEmbedder:
    """Hashed bag-of-words embedding: texts sharing (non-trivial) words are similar, others are ~orthogonal."""

    dim = 512

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z]{4,}", t.lower()):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class FakeLLM:
    """Grades by word overlap with the reference answer, but is deliberately *gullible*: an answer that says
    'ignore previous instructions' gets full marks, like a model that fell for the injection."""

    def __init__(self):
        self.calls = 0
        self.mode = "normal"  # normal | garbage | garbage_once | low_confidence

    def generate(self, system, user):
        self.calls += 1
        if self.mode == "garbage" or (self.mode == "garbage_once" and self.calls == 1):
            return "Sure! The score is high."
        ans = re.search(r"<student_answer>\n(.*?)\n</student_answer>", user, re.S).group(1)
        ref = re.search(r"<reference_answer>(.*?)</reference_answer>", user, re.S).group(1)
        mx = float(re.search(r"<max_marks>(.*?)</max_marks>", user).group(1))
        if "ignore all previous instructions" in ans.lower():
            score, just = mx, "The answer is complete and excellent."
        else:
            rw = set(re.findall(r"[a-z]{4,}", ref.lower()))
            aw = set(re.findall(r"[a-z]{4,}", ans.lower()))
            score = round(mx * (len(rw & aw) / max(len(rw), 1)) * 2) / 2
            just = f"Covers {len(rw & aw)} of {len(rw)} key terms."
        payload = {"score": score, "justification": just}
        payload["confidence"] = 0.4 if self.mode == "low_confidence" else 0.9
        return json.dumps(payload)


class FakeOCR:
    """Reads the lines a test 'wrote' on a page from the PNG's text chunk."""

    def read_image(self, image, page):
        from app.services.ocr import OCRLine

        lines = getattr(image, "fake_lines", [])
        return [OCRLine(page=page, text=t, confidence=0.93) for t in lines]


def fake_load_image(path):
    from PIL import Image as _I

    im = _I.open(path)
    im.load()
    lines = json.loads(im.info.get("fake_ocr", "[]"))
    out = im.convert("RGB")
    out.fake_lines = lines
    return out


def page_png(lines: list[str]) -> bytes:
    """A valid PNG whose metadata carries the handwriting FakeOCR will 'read'."""
    img = Image.new("RGB", (400, 300), "white")
    info = PngInfo()
    info.add_text("fake_ocr", json.dumps(lines))
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("PIPELINE_MODE", "inline")
    monkeypatch.setenv("SECRET_KEY", "test-secret")

    from app.core import config, db
    from app.services.rag import index as faiss_index
    from app.services import storage

    config.get_settings.cache_clear()
    storage._cached.cache_clear()
    db.reset_engine()
    faiss_index.drop_cache()

    import app.models  # noqa: F401

    db.Base.metadata.create_all(db.get_engine())
    yield
    db.reset_engine()


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    embedder = FakeEmbedder()
    from app.services import ocr
    from app.tasks import pipeline

    monkeypatch.setattr(pipeline, "get_llm_client", lambda: llm)
    monkeypatch.setattr(pipeline, "get_embedder", lambda: embedder)
    monkeypatch.setattr("app.services.rag.get_embedder", lambda: embedder)
    monkeypatch.setattr(ocr, "get_ocr_engine", lambda: FakeOCR())
    monkeypatch.setattr(ocr, "load_image", fake_load_image)
    return llm


@pytest.fixture
def client(fake_llm):
    from app.main import create_app

    return TestClient(create_app())


@pytest.fixture
def teacher(client):
    from app.core.db import SessionLocal
    from app.models import User

    with SessionLocal() as db:
        db.add(User(name="Dr. T", email="t@x.edu", role="teacher", password_hash=hash_password("pw")))
        db.commit()
    r = client.post("/api/auth/login", json={"email": "t@x.edu", "password": "pw"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}
