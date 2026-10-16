# PaperMind — Frontend

React frontend for PaperMind, which grades handwritten answer sheets with OCR, text correction and RAG-grounded LLM grading. It runs entirely on mock data until you point it at the backend.

## Run

```bash
npm install
npm run dev
```

Open http://localhost:5173 and pick **Teacher** or **Student** on the demo login screen. To restore the seeded data, click **Reset demo data** on the login page. Mock state is saved in `localStorage`.

## Structure

```
src/
  api/          config.js (base URL + mock switch), client.js (axios), index.js (one function per endpoint),
                mockApi.js (in-memory implementation), mockData.js (seed data)
  components/   ui.jsx (Button, Card, Badge, Alert, Tooltip…), FileUpload, StepIndicator,
                QuestionBreakdown, ScoreSummary, ProcessingTracker, ExamPicker, Icon, Logo
  context/      AuthContext (demo role-based auth)
  layouts/      AppShell, TeacherLayout, StudentLayout
  pages/        Login, teacher/*, student/*
  utils/        grading.js (review-flag rules), useAsync.js
```

## Switching to the real backend

Copy `.env.example` to `.env`:

```
VITE_API_BASE_URL=http://localhost:8000/api
VITE_USE_MOCK=false
```

Each function in `src/api/index.js` has this shape:

```js
export const getExams = () =>
  USE_MOCK ? mock.getExams() : data(http.get('/exams'));
```

To switch one endpoint at a time, change that one line. The response shapes the UI expects are the ones returned by `mockApi.js`.

## Review-flag rules (`src/utils/grading.js`)

An answer is sent to the **Needs review** queue when either rule matches:

- **Low confidence:** the LLM's `confidence` is below `0.6`.
- **Relevance mismatch:** the answer is subjective, its embedding similarity to the syllabus (`relevance`) is below `0.35`, **and** it scored at least 70% of the marks. This pattern suggests a prompt-injection or keyword-stuffed answer, so it gets a red shield icon.

An answer leaves the queue when the teacher overrides its score (a reason is required) or accepts the AI score. Until then, students see **Pending teacher review** instead of a mark, and that answer is left out of their total.

## Contributing (getting started)

```bash
git clone <this repo> && cd PaperMind
npm install && npm run dev                      # frontend on http://localhost:5173 (runs on mock data by default)

cd backend
python -m venv .venv && .venv/Scripts/activate  # source .venv/bin/activate on macOS/Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env                            # local settings (never committed)
pytest -q                                       # 59+ tests, no GPU or models needed
```

* Read `HANDOFF.md` first: it records what is done, what is verified, and the known weaknesses.
* Large files are **not** in the repo: the trained correction model (`backend/models/`), datasets (`testdata/`), and the demo
  storage. See `backend/training/README.md` for training the ByT5 correction model and `backend/README.md` for the API.
* Branch per change (`git checkout -b feature/...`), keep tests passing, open a pull request.
