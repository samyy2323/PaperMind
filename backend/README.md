# PaperMind — Backend

FastAPI + PostgreSQL + Celery/Redis backend for PaperMind. It grades handwritten answer sheets with only
open-source, self-hosted models: **TrOCR** (handwriting), **doctr** (line detection), **sentence-transformers +
FAISS** (RAG) and an **open-weight LLM served by Ollama** (or Hugging Face `transformers`). No paid or
proprietary APIs are used anywhere.

```
teacher uploads key PDF ─► questions extracted ─► teacher edits ─► publish
teacher uploads syllabus ─► chunks ─► MiniLM embeddings ─► per-class FAISS index
teacher uploads a student's photos ─► 202 Accepted ─► Celery chain:
    OCR (doctr lines → TrOCR) ─► correction ─► segmentation ─► grading ─► aggregation
                                                           └─ objective: rapidfuzz   subjective: RAG + local LLM
```

## Run it

### Docker (Postgres + Redis + Ollama + API + Celery worker)

```bash
cd backend
docker compose up --build
docker compose exec ollama ollama pull llama3.1:8b     # one-time (≈5 GB)
docker compose exec api python scripts/seed_demo.py    # optional demo teacher/students
```

API: http://localhost:8000 (interactive docs at `/docs`). TrOCR/MiniLM/doctr weights download on first use into the
`models` volume. For GPU inference give the `ollama` service a GPU (see the comment in `docker-compose.yml`) and use a
CUDA build of torch.

### Local, without Docker (SQLite, no Redis)

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate            # source .venv/bin/activate on macOS/Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env                                       # DATABASE_URL=sqlite…, PIPELINE_MODE=thread
alembic upgrade head
python scripts/seed_demo.py
ollama pull llama3.1:8b                                    # or set LLM_MODEL=llama3.2:3b for a faster CPU demo
uvicorn app.main:app --reload
```

`PIPELINE_MODE=thread` runs the same task chain in a background thread (one sheet at a time) so you don't need Redis
for development; production uses `PIPELINE_MODE=celery` with `celery -A app.tasks.celery_app worker --concurrency=1`.

### Point the frontend at it

In the repo root create `.env` with `VITE_USE_MOCK=false` and `VITE_API_BASE_URL=http://localhost:8000/api`, then
`npm run dev`. The demo login uses the accounts from `seed_demo.py` (password `demo1234`). Students created through the
roster can sign in with `<roll>@students.papermind.local` and their roll number as the password.

## Endpoints (`/api`)

| Method & path | Purpose |
|---|---|
| `POST /auth/login` · `GET /auth/me` | JWT login |
| `GET/POST /classes/{id}/roster`, `PATCH/DELETE …/roster/{studentId}` | Class roster CRUD (`POST /classes/ensure` gets-or-creates a class for a subject) |
| `POST /exams` · `GET /exams` · `GET/PATCH /exams/{id}` | Create / list / edit (draft) exams |
| `POST /exams/{id}/answer-key` | Upload the key PDF → returns the parsed questions |
| `PATCH /exams/{id}/questions/{qId}` · `PUT /exams/{id}/questions` | Edit one question / replace the draft's list |
| `POST /exams/{id}/publish` | Lock the exam (validates text, reference answers, marks total) |
| `POST /classes/{id}/knowledge-base` · `GET /knowledge-base` · `DELETE /knowledge-base/{docId}` | RAG source documents |
| `POST /students/{studentId}/answer-sheet` | **Teacher** uploads a student's photos (`examId` + `files[]`) → `202` |
| `GET /answer-sheets/{id}/status` | `processing` stage: `ocr → correction → grading → done` |
| `GET /answer-sheets/{id}/results` | Per-question breakdown (students get a redacted view) |
| `GET /answer-sheets/{id}/reference-pdf` | The photos combined into one PDF (viewing only) |
| `POST /answer-sheets/{id}/reprocess` | Re-run the pipeline on the stored photos |
| `GET /students/{id}/answer-sheets` | A student's results list |
| `GET /exams/{id}/submissions` · `GET /exams/{id}/review-queue` | Review dashboard |
| `PATCH /student-answers/{id}/override` · `POST /student-answers/{id}/review` | Override a score (reason required) / accept the AI score |
| `GET /exams/{id}/analytics` | Average, distribution, flagged vs clean |

JSON is camelCase to match the React app. Students can only read their own, finished sheets; for an answer still
awaiting review the AI score, justification, confidence and flag are withheld.

## How grading works

* **Reference PDF vs images.** The PDF is assembled with Pillow for humans only. The Celery task receives the *image*
  storage keys; the OCR stage never reads the PDF.
* **OCR** (`services/ocr`): doctr finds word boxes → grouped into lines → wide lines split at word gaps (TrOCR
  degrades on very wide crops) → TrOCR per segment. Each line carries a confidence (geometric-mean token probability).
  If doctr can't load, a projection-profile detector is used (logged as a warning).
* **Correction** (`services/correction`): pass-through until a fine-tuned ByT5/BART checkpoint is set in
  `CORRECTION_MODEL_PATH`. `TODO: fine-tune on our OCR error pairs.`
* **Segmentation** (`services/segmentation.py`): uses question numbers the student wrote (`1)`, `Q2.`, `Ans 3:`, tolerating
  `l`/`I` read for `1`); with too few markers it splits lines evenly. Swap it by registering another `Segmenter`.
* **Objective** questions: option letters, true/false and short answers via `rapidfuzz`; near-misses are flagged.
* **Subjective** questions: retrieve top-k chunks from the class's FAISS index → prompt → local LLM → strict JSON
  `{"score", "justification", "confidence"?}` validated in code, rejected and retried once, else flagged `parse_failure`.
* **Prompt-injection defence** (`services/grading/prompt.py`): every untrusted field is XML-escaped; the student's text
  lives only in `<student_answer>`; the system prompt says it is data, never instructions; output is schema-validated.
  Independently, a high score with very low topical similarity is flagged (below).
* **Flags** (`services/grading/flagging.py`; thresholds in `core/config.py`):

  | `flagged_reason` | when |
  |---|---|
  | `injection_suspected` | subjective, score > 80 % of marks **and** embedding similarity to syllabus/reference < 0.30 |
  | `parse_failure` | LLM output invalid twice |
  | `low_confidence` | confidence < 0.6 (LLM-reported, capped by OCR confidence and by a retried/out-of-range reply) |

  Similarity is the max cosine between the student's answer and (retrieved syllabus chunks ∪ the reference answer).
* **Failures** never crash the queue: an unreadable photo, no detectable handwriting, or an unexpected error marks the
  sheet `needs_review` with a message (visible to the teacher, with a *Reprocess* button); later steps skip it. If the LLM
  server is down, only the subjective answers are flagged `parse_failure`.

## Validate grading accuracy

```bash
python scripts/validate_grader.py scripts/sample_eval.csv -v       # or your Mohler-style CSV
```

CSV columns: `question, reference_answer, student_answer, human_score` (optional `max_score`). Prints Pearson r and RMSE.

## Tests

```bash
pytest -q                                   # 47 tests: fakes for OCR/embeddings/LLM, SQLite, inline pipeline
RUN_REAL_MODELS=1 pytest tests/test_real_models.py -q     # smoke-test the real TrOCR / MiniLM / detectors
python scripts/e2e_demo.py                  # whole flow over HTTP with real models (needs the API + Ollama running)
```

## Project layout

```
app/api  app/models  app/schemas  app/core  app/tasks
app/services/{ocr,correction,rag,grading,storage,pdf,ingestion}   segmentation.py  results.py
alembic/  scripts/  tests/  Dockerfile  docker-compose.yml
```

Storage goes through `services/storage` (keys, not paths) — implement `StorageBackend` for S3 and return it from
`get_storage()`.

## Known limits

* `transformers` and `sentence-transformers` are pinned below 5.x / 5.x because transformers 5 cannot load TrOCR's tokenizer.
* Line-level TrOCR on full-page photos is sensitive to lighting and skew; there is no deskew/denoise step yet.
* Local 3B–8B LLMs are inconsistent graders; that is why low confidence, parse failures and the injection signal all
  route answers to a human. Use `validate_grader.py` to measure a model before trusting it.
* Migrations were verified on SQLite; run `alembic upgrade head` against Postgres in your environment (docker-compose does this on start).
