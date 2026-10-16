# PaperMind — Handoff Report

Status as of 2026-10-15. Written so a new session or account can continue without prior context.

## 1. What the project is

PaperMind grades handwritten student answer sheets automatically.

- A teacher uploads a typed answer key (PDF), syllabus material, and photos of each student's answer sheet.
- The backend runs OCR, text correction, question segmentation and grading (rule-based for objective questions, RAG plus a local LLM for subjective ones).
- Low-confidence answers, and answers that look like prompt injection, are routed to the teacher for review.
- Constraint: no paid or proprietary APIs. Every model is open-source and self-hosted.

## 2. Timeline

1. **Frontend prompt.** Built a React 18 + Vite + Tailwind v4 app with teacher and student roles, running on mock data. Views: login, exam editor, knowledge base, review dashboard, submission review, analytics, student results.
2. **Backend prompt.** Built FastAPI, SQLAlchemy and Alembic, Celery tasks, a FAISS-based RAG, a TrOCR-based OCR pipeline, grading, flagging, Docker files and tests. It also carried one frontend change: students no longer upload; the teacher uploads per student.
3. **Frontend rewiring.** Added a roster, a per-student upload page with staged progress, a Reference PDF link, answer-id based overrides, and removed the student upload pages and exam codes.
4. **Real-model debugging.** Ran the full flow with TrOCR and `llama3.2:3b`. Fixed the TrOCR tokenizer (pinned transformers below 5), a pad-token bug, a model-loading race (added `core/singleton.py`), and a single-worker dev queue.
5. **Demo freeze.** Built a snapshot of the frontend for the user's guide and served it on port 4173.

## 3. Current state

### Done and verified

- **Frontend:** all views work on mock data. The production build passes. I clicked through the main teacher and student flows in the browser pane, including the new upload flow, override validation, analytics and the student redaction.
- **Backend tests:** 48 pass with fake OCR, LLM and embedder (4 real-model tests are skipped by default). They cover answer-key parsing, objective grading, the injection-safe prompt, JSON validation and retry, the flag rules, segmentation, the full API flow, roster CRUD, upload validation and the knowledge base lifecycle.
- **Real models, end to end:** the flow ran over HTTP with real TrOCR, MiniLM and `llama3.2:3b` through Ollama. This used images rendered in a handwriting-style font, not real handwriting.
- **Migration:** `alembic upgrade head` and `downgrade base` were verified on SQLite, and `alembic check` found no drift.

### Not done

- **Fine-tuned correction model.** `app/services/correction` is a pass-through stub. The wrapper for a fine-tuned ByT5 or BART checkpoint exists (set `CORRECTION_MODEL_PATH`), but no model has been trained. TrOCR itself is the stock pretrained `microsoft/trocr-base-handwritten`.
- **Grader evaluation.** `backend/scripts/validate_grader.py` computes Pearson and RMSE, but it has never been run. The only data is `scripts/sample_eval.csv` (12 invented rows). No Mohler dataset has been used.

### Not verified

- **Real handwriting:** no photos of real handwriting have been tested.
- **Injection flag with a real model:** the real 3B model resisted the attack, giving the adversarial answer 0/6 with low confidence. The `injection_suspected` flag is only verified with a deliberately gullible fake LLM in tests.
- **Frontend against the real backend:** `VITE_USE_MOCK=false` was never tried in the browser.
- **Docker, Postgres, Redis and Celery:** none run, because this machine has none of them. `PIPELINE_MODE=celery` is untested.
- **Latest segmentation fix:** covered by a unit test, but the real-model demo was not re-run afterwards.
- **Grading quality:** the 3B model sometimes invented details in justifications. `llama3.1:8b` (the config default) has not been tried.
- **Version control:** no git repo; nothing is committed.

## 4. Known issues

1. OCR read "1) (b)" as `19 ( 0b ) .`, so the objective Q1 scored 0 for both demo students. The segmentation fix now keeps the text. Objective matching of `0b` against `(b)` is not handled, so it is wrong but flagged only if confidence is low.
2. Every line of a rendered page took about 50–125 s on CPU; a real sheet with the 8B model will be slow.
3. The first OCR line of a page often has low confidence (0.55 in the demo).
4. In demo mode, `openReferencePdf` shows a message instead of a PDF; real PDFs need the backend.
5. Reprocessing keeps the stored images but re-runs everything; there is no UI progress for it beyond the status poll.

## 5. How to run everything

All paths are on Windows under `C:\PaperMind`.

| What | Command or URL |
|---|---|
| Frontend dev server | `npm run dev` in `C:\PaperMind` → http://localhost:5173 (launch config in `.claude/launch.json`, name `papermind`) |
| Frozen demo (unchanged) | http://localhost:4173, served from `C:\PaperMind_demo\dist` with `npx vite preview --outDir /c/PaperMind_demo/dist --port 4173` |
| Backend | see below; http://localhost:8000, docs at `/docs` |
| Backend tests | `cd backend && .venv/Scripts/python -m pytest -q` |
| Real-model smoke tests | `RUN_REAL_MODELS=1 pytest tests/test_real_models.py -q` |
| Full real-model demo | `python scripts/e2e_demo.py` with the API and Ollama running |

Backend start (local, no Docker, SQLite):

```bash
cd backend
export DATABASE_URL=sqlite:///./demo.db STORAGE_ROOT=./storage_demo PIPELINE_MODE=thread LLM_MODEL=llama3.2:3b HF_HUB_DISABLE_SYMLINKS_WARNING=1
.venv/Scripts/python -m alembic upgrade head
.venv/Scripts/python scripts/seed_demo.py
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

After the environment reset, the dev server (5173) and the demo (4173) were restarted. The backend (8000) was not restarted.

To use the real backend from the frontend, create `C:\PaperMind\.env` with `VITE_USE_MOCK=false` and `VITE_API_BASE_URL=http://localhost:8000/api`.

Demo accounts (from `seed_demo.py`, password `demo1234`): teacher `kulkarni@college.edu`; student `rohan@students.papermind.local`. Students added through the roster sign in with `<roll lowercase>@students.papermind.local` and the roll number as the password.

## 6. Environment on this machine

- Windows 11, 15.7 GB RAM, 12 cores, no GPU. Python 3.10.11, Node 20.19.5.
- No Docker, Postgres or Redis.
- Ollama 0.34.4 with one model pulled: `llama3.2:3b` (2 GB). `llama3.1:8b` is the config default and is not pulled.
- Python venv at `backend/.venv`. Important pins: `transformers>=4.44,<5`, `sentence-transformers>=3.0,<5`, `huggingface_hub<1.0`, `sentencepiece`, `protobuf`. Transformers 5 cannot load TrOCR's tokenizer.
- Hugging Face cache is in `~/.cache/huggingface` (TrOCR, MiniLM). doctr downloads its detection weights on first use.
- Do not install Python packages while the server is running; Windows locks the files.

## 7. Project layout

```
C:\PaperMind
  HANDOFF.md  README.md  package.json  vite.config.js  .env.example
  src/
    api/         config.js, client.js, index.js (one function per endpoint, mock or real),
                 mockApi.js (localStorage key papermind.mockdb.v2), mockData.js (seed + roster)
    components/  ui.jsx, FileUpload, StepIndicator, ProcessingTracker, QuestionBreakdown,
                 ScoreSummary, ReferencePdfLink, ExamPicker, Icon, Logo
    context/     AuthContext
    layouts/     AppShell, TeacherLayout, StudentLayout
    pages/       Login, NotFound, teacher/{ExamsList,ExamEditor,AnswerSheets,KnowledgeBase,
                 ReviewDashboard,SubmissionReview,Analytics}, student/{ResultsList,ResultDetail}
    utils/       grading.js (honours backend flaggedReason / pendingReview), useAsync.js
  backend/
    app/api         auth, classes, exams, knowledge_base, answer_sheets, student_answers, deps
    app/models      user (User, Class, Enrollment), exam, sheet, knowledge
    app/schemas     camelCase request/response models
    app/core        config, db, security, singleton (thread-safe model loading)
    app/services    ocr (detectors, trocr), correction, rag (chunking, embedder, index),
                    grading (objective, prompt, llm, subjective, flagging), storage, pdf,
                    ingestion/answer_key, segmentation, results
    app/tasks       pipeline.py (steps), tasks.py (Celery chain; thread/inline modes), celery_app
    alembic/        0001_initial_schema
    scripts/        validate_grader.py, sample_eval.csv, seed_demo.py, e2e_demo.py
    tests/          conftest (fakes), test_grading_units, test_ingestion, test_e2e_api, test_real_models
    Dockerfile, docker-compose.yml, README.md
C:\PaperMind_demo\dist   frozen frontend snapshot
```

## 8. Key design decisions

- **Reference PDF vs OCR:** the PDF is built with Pillow for humans only. The Celery task is given image storage keys, never the PDF.
- **Prompt-injection defence, in layers:** XML-escaped untrusted fields; the student's text only inside `<student_answer>`; a system prompt that calls it untrusted data; JSON schema validation with one retry; plus an independent flag (below).
- **Flag rules** (`services/grading/flagging.py`): `injection_suspected` when a subjective score is above 80% of marks and similarity is below 0.30; `parse_failure` when the LLM's output is invalid twice; `low_confidence` when confidence is below 0.6. Confidence is capped by OCR confidence, a retried reply, or an out-of-range score. Similarity is the maximum cosine similarity between the answer and the retrieved syllabus chunks or the reference answer.
- **Failures:** an unreadable photo or an unexpected error marks the sheet `needs_review` with a message. Later steps skip it. The teacher sees a Reprocess button.
- **Student privacy:** for answers pending review, students get no AI score, justification, confidence or flag, and the total excludes those marks.
- **Pipeline modes:** `celery` (production), `thread` (one-sheet-at-a-time dev queue, no Redis), `inline` (tests).
- **Storage:** all file access goes through `services/storage` using keys, so an S3 backend is a small change.

## 9. Suggested next steps

1. **Fine-tune the correction model.** Build training pairs of TrOCR output and the true text: run TrOCR over rendered text with varied fonts, or use a public handwriting dataset such as IAM. Fine-tune ByT5-small. CPU-only is slow, so keep it small or use a GPU. Then set `CORRECTION_MODEL_PATH`. Confirm with the user that ByT5 is the model their plan names.
2. **Evaluate the grader.** Get a Mohler-style dataset (or the user's own), pull `llama3.1:8b`, and run `python scripts/validate_grader.py <csv> -v`. Report Pearson and RMSE. Confirm whether "evaluation model" in the plan means this grader or a separate model.
3. Re-run `scripts/e2e_demo.py` to confirm the segmentation fix with real models.
4. Test the frontend against the real backend (`VITE_USE_MOCK=false`) and fix any mismatches.
5. Run the Docker stack on a machine with Docker, including Postgres migrations and a Celery worker.
6. Initialise git and commit.
7. Test with photos of real handwriting; consider deskew and denoise before detection.

## 10. Rules from the user

- The frozen demo on port 4173 must stay untouched.
- Do not use paid or proprietary LLM or OCR APIs.
