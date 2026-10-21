# PaperMind — Master Execution Plan & Roadmap

**Status:** Phase 1 In Progress  
**Author:** PaperMind Core Engineering  
**Reference Documents:** [`docs/ANALYSIS_AND_ROADMAP.md`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/docs/ANALYSIS_AND_ROADMAP.md), [`rules/PRD.md`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/rules/PRD.md), [`rules/ARCHITECTURE.md`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/rules/ARCHITECTURE.md), [`rules/AI_RULES.md`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/rules/AI_RULES.md)

---

## 1. Executive Summary & Problem Context

Empirical evaluations on the real-world Mendeley examination dataset (3 students, 150 marks, 45 short answers, 60 MCQs) identified the precise failure modes of the automated grading pipeline:

1. **The Silent Error Bottleneck:** Out of 30 answers where the AI score deviated from the teacher's benchmark, **26 errors were completely silent** (unflagged).
2. **Missing Localisation:** 76% of lost short-answer marks (26 of 34 marks) were caused by questions failing to localize student writing (due to OCR-garbled question numbers like `38.` for `33.` or `#` for `34.`), resulting in an empty extracted answer that was silently assigned a score of 0.
3. **Ghost Bleed-Through & Hallucinations:** On blank back pages with faint mirrored bleed-through (e.g. Student 1, Page 4), TrOCR hallucinated 19 lines of fluent nonsense that contaminated subsequent question evaluations.
4. **Lack of Model Ablation Evidence:** While ByT5 text correction is integrated, there has been no measured ablation proving whether sequence-to-sequence correction improves final student scores.

This master plan establishes a rigorous, phased strategy prioritizing high-impact structural fixes before expensive model retraining.

---

## 2. Phased Roadmap Overview

```
Phase 1: Critical Silent-Error & Preprocessing Fixes (P0 - Immediate)
   │── 1.1 Flag Empty/Missing Answers (`answer_not_found`)
   │── 1.2 Blank-Page & Faint Bleed-Through Suppression
   │── 1.3 Text-Correction Ablation Switch (`ENABLE_CORRECTION`)
   └── 1.4 Test Suite Updates & Validation
   
Phase 2: Content-Aware & Geometry-Driven Segmentation (P1)
   │── 2.1 MiniLM Semantic Content Matching for Orphan Blocks
   │── 2.2 Left-Margin Geometry & Indentation Detection
   └── 2.3 Constrained Vocabulary Decoding for Objective Answers
   
Phase 3: Multi-Signal Evaluation & Instructor Calibration (P1)
   │── 3.1 Dual-Signal Scoring (LLM + Semantic Embedding Consensus)
   │── 3.2 Dynamic Teacher Calibration (Linear / Isotonic Mapping)
   │── 3.3 Key-Point Rubric Evaluation Prompting
   └── 3.4 50-Student Mendeley Benchmark Execution
   
Phase 4: Evidence-Rich Review Dashboard & Active Learning (P2)
   │── 4.1 Bounding-Box Persistence & Line Crop Extraction
   │── 4.2 Side-by-Side Visual Review UI (Image Crop + OCR + Justification)
   │── 4.3 One-Click Question Block Re-assignment
   └── 4.4 Active Learning Dataset Logging (Teacher Overrides -> Training Labels)
   
Phase 5: Model Retraining & Structured Answer Booklets (P3)
   │── 5.1 ByT5 v2 Retraining (1–50+ Markers & Protected Digits)
   │── 5.2 TrOCR Cohort Adaptation
   └── 5.3 Institutional Structured Exam Booklets with OMR Bubble Grids
```

---

## 3. Detailed Phase Breakdown

### Phase 1: Critical Silent-Error & Preprocessing Fixes (P0 — Immediate)
*Goal: Eliminate 87% of silent grading errors and suppress TrOCR hallucinations on blank pages.*

- [x] **Task 1.1: Implement `answer_not_found` Review Flag**
  - **Backend:** Update [`app/services/grading/flagging.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/services/grading/flagging.py) and [`app/services/grading/__init__.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/services/grading/__init__.py).
  - When an extracted answer is empty or whitespace-only (both objective and subjective), assign `flagged_reason = "answer_not_found"`.
  - Ensure confidence is capped ($0.0$) and `pending_review = True`.
  - **Frontend:** Update [`src/utils/grading.js`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/src/utils/grading.js) and [`src/components/QuestionBreakdown.jsx`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/src/components/QuestionBreakdown.jsx) with an "Answer not found" badge and review filter.

- [x] **Task 1.2: Image Preprocessing, Blank-Page & Ghost Ink Suppression**
  - **Detector Filtering:** Update [`app/services/ocr/detectors.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/services/ocr/detectors.py).
  - Add `is_blank_page(image)`: Check overall ink density and contrast; skip pages where ink pixel coverage $< 0.15\%$.
  - Add `suppress_bleed_through(gray)`: Apply contrast thresholding to drop pale grey ghost ink from mirrored back pages.
  - Add low-confidence line suppression in [`app/services/ocr/__init__.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/services/ocr/__init__.py): drop OCR lines with confidence $< 0.30$ and near-zero ink.

- [x] **Task 1.3: Text-Correction Ablation Switch**
  - **Configuration:** Add `ENABLE_CORRECTION: bool = True` in [`app/core/config.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/core/config.py).
  - **Service Layer:** In [`app/services/correction/__init__.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/services/correction/__init__.py), return `PassThroughCorrector` if `ENABLE_CORRECTION=False`.
  - Enables systematic ablation comparisons (`with` vs `without` ByT5) on the same test sets.

- [x] **Task 1.4: Unit Test Suite Expansion & Verification**
  - Add test cases in [`tests/test_grading_units.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/tests/test_grading_units.py) asserting that empty answers trigger `answer_not_found`.
  - Add tests verifying blank page skipping and bleed-through suppression.
  - Verify all existing backend unit tests pass.

---

### Phase 2: Content-Aware & Geometry-Driven Segmentation (P1)
*Goal: Recover the 26 marks lost to unassigned handwriting blocks.*

- [x] **Task 2.1: MiniLM Semantic Content Matching**
  - If a handwriting block lacks a recognized question number or has a corrupted pseudo-marker (e.g. `38.` for `33.`, `#`, `Ans:`), embed the block text and compute cosine similarity against the question text and reference answer.
  - Solve 1-to-1 greedy bipartite assignment above threshold $\tau \ge 0.35$.
  - Clean pseudo-markers from the first line of assigned blocks. If match confidence is $< 0.50$, assign `low_confidence` review flag. Unmatched blocks remain attached to the preceding question.
- [x] **Task 2.2: Geometry-Aware Margin & Indentation Analysis**
  - Captured line bounding boxes $(x_0, y_0, x_1, y_1)$ in `DoctrLineDetector` and `ProjectionLineDetector`.
  - Propagated bounding boxes through `OCRLine` $\to$ `ocr.json` $\to$ `corrected.json` $\to$ `SegLine`.
  - Implemented `estimate_page_geometry()`, `is_line_indented()`, and `has_vertical_gap()`.
  - Distinguished indented sub-items/bullets from top-level question markers and detected margin-aligned vertical gaps as unlabelled answer boundaries.
- [ ] **Task 2.3: Constrained Decoding for Objective Answers**
  - Constrain TrOCR decoding at MCQ answer positions to valid option tokens `{"A", "B", "C", "D", "E"}` and `{"True", "False"}`.

---

### Phase 3: Multi-Signal Evaluation & Instructor Calibration (P1)
*Goal: Close the 8-mark gap caused by zero-shot LLM harshness.*

- [ ] **Task 3.1: Dual-Signal Consensus Grading**
  - Combine LLM score with dense embedding / cross-encoder similarity score.
  - If the LLM score and embedding score diverge by $> 30\%$, trigger `model_disagreement` review flag.
- [ ] **Task 3.2: Teacher Calibration Module**
  - Allow teachers to grade the first 15–20 submissions.
  - Fit a per-teacher calibration curve (isotonic or linear regression) to adjust AI marks to the instructor's grading strictness.
- [ ] **Task 3.3: Key-Point Rubric Decomposition**
  - Decompose reference answers into distinct rubric points.
  - Direct the LLM to score presence/absence of each key point rather than holistic guessing.
- [ ] **Task 3.4: Comprehensive 50-Student Mendeley Evaluation**
  - Execute [`backend/scripts/run_mendeley_eval.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/scripts/run_mendeley_eval.py) across all 50 student scripts.
  - Report Pearson correlation ($r$), Quadratic Weighted Kappa ($\kappa$), and Silent Error Rate.

---

### Phase 4: Evidence-Rich Review Dashboard & Active Learning (P2)
*Goal: Reduce teacher review friction from minutes to seconds per paper.*

- [ ] **Task 4.1: Line Bounding-Box Storage**
  - Preserve word and line coordinates from `doctr` through to `StudentAnswer.debug`.
- [ ] **Task 4.2: Side-by-Side Visual Review Interface**
  - Display the high-resolution handwriting image crop directly beside the OCR text, model justification, and reference answer in [`QuestionBreakdown.jsx`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/src/components/QuestionBreakdown.jsx).
- [ ] **Task 4.3: One-Click Block Re-assignment**
  - Enable teachers to drag or select an unassigned block and reassign it to a question.
- [ ] **Task 4.4: Active Learning Dataset Collection**
  - Automatically log all teacher score overrides, text corrections, and block reassignments as fine-tuning pairs.

---

### Phase 5: Model Retraining & Structured Booklets (P3)
*Goal: Long-term institutional scaling and elimination of localization errors by design.*

- [ ] **Task 5.1: ByT5 v2 Fine-Tuning**
  - Retrain ByT5 on synthetic and real handwriting with question markers spanning 1–50+.
- [ ] **Task 5.2: TrOCR Cohort Fine-Tuning**
  - Fine-tune TrOCR encoder-decoder weights on collected institutional handwriting lines.
- [ ] **Task 5.3: Structured Answer Booklets & OMR Integration**
  - Design printable templates with pre-printed question boundaries and OMR bubble grids for MCQs.

---

## 4. Immediate Execution: Phase 1 Checklist

| Step | Component | Status | Target Completion |
|---|---|---|---|
| **1.1** | `answer_not_found` flag in `flagging.py` & `grading/__init__.py` | 🟢 Completed | Verified (35 tests pass) |
| **1.2** | UI support for `answer_not_found` in `grading.js` & `QuestionBreakdown.jsx` | 🟢 Completed | Verified |
| **1.3** | Blank-page & bleed-through suppression in `detectors.py` & `ocr/__init__.py` | 🟢 Completed | Verified (unit tests pass) |
| **1.4** | `ENABLE_CORRECTION` ablation switch in `config.py` & `correction/__init__.py` | 🟢 Completed | Verified (ablation test pass) |
| **1.5** | Unit tests in `test_grading_units.py` & full test suite pass | 🟢 Completed | 68/68 tests passing |
