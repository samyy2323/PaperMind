# PaperMind — Active Context & Updates Log

> **Operational Directive:** This file tracks ONLY the changes, additions, and verified states produced after each successful prompt completion.

---

## Latest Completed Update: Task 2.2 (Geometry-Aware Margin & Indentation Analysis)

**Timestamp:** 2026-10-21 21:44 UTC+05:30  
**Target:** Phase 2: Content-Aware & Geometry-Driven Segmentation (Task 2.2)  
**Status:** Completed & Verified (76/76 backend tests passing)

### 1. Files Modified & Added

| File | Status | Description of Changes |
|---|---|---|
| [`backend/app/services/ocr/detectors.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/services/ocr/detectors.py) | Modified | Added `detect_with_boxes()` returning `(crops, (x0, y0, x1, y1))` to both `ProjectionLineDetector` and `DoctrLineDetector`. Preserved `detect()` as a wrapper returning `[crops for crops, _ in self.detect_with_boxes(image)]` for 100% backward compatibility. |
| [`backend/app/services/ocr/__init__.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/services/ocr/__init__.py) | Modified | Added `box: tuple[float, float, float, float] | None = None` to `OCRLine`. Updated `TrOCREngine.read_image()` to capture bounding boxes from detector and attach them to generated `OCRLine` instances. |
| [`backend/app/tasks/pipeline.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/tasks/pipeline.py) | Modified | Propagated `"box"` from `ocr.json` into `corrected.json` in `step_correction`. Forwarded `box` into `SegLine(..., box=tuple(l["box"]))` in `step_segmentation`. |
| [`backend/app/services/segmentation.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/app/services/segmentation.py) | Modified | Added `box` and coordinate properties (`x0`, `y0`, `x1`, `y1`) to `SegLine`. Implemented `PageGeometry` dataclass, `estimate_page_geometry()`, `is_line_indented()`, and `has_vertical_gap()`. Integrated geometry checks in `MarkerSegmenter`: indented lists inside an active answer are excluded from claiming top-level questions; margin-aligned lines after vertical gaps are detected as answer boundaries for unlabelled questions. |
| [`backend/tests/test_ingestion.py`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/backend/tests/test_ingestion.py) | Modified | Added 3 unit tests: `test_geometry_margin_and_gap_estimation`, `test_indented_numbered_sub_item_does_not_split_or_claim_question`, and `test_margin_aligned_line_with_vertical_gap_recovers_missing_question_number`. All 19 ingestion tests pass. |
| [`rules/PLAN.md`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/rules/PLAN.md) | Modified | Marked Task 2.2 as completed (`[x]`). |

### 2. Architecture & Algorithmic Summary

- **Bounding Box Propagation Pipeline:** Full spatial bounding boxes $(x_0, y_0, x_1, y_1)$ are captured at line detection and persisted through intermediate storage artifacts (`ocr.json`, `corrected.json`) into `SegLine`.
- **Margin & Indentation Modeling:**
  - $X_{\text{margin}}$ estimated via 10th percentile of line starts.
  - $\Delta_{\text{indent}} = \max(25.0, 0.75 \times \text{median\_height})$.
  - Vertical paragraph gap threshold $= \max(18.0, 1.35 \times \text{median\_height})$.
- **Protection Against Nested Sub-Items:** Indented lines matching marker patterns (e.g. `1. Fix:` inside Q3) are recognized as continuation lists, preventing them from stealing top-level exam questions (e.g. Q1).
- **Geometric Answer Boundary Discovery:** Margin-aligned lines following large vertical gaps are identified as answer boundaries even when the student omitted the question number or OCR failed to read it.

### 3. Verification & Test Suite Results

```bash
pytest tests/test_grading_units.py tests/test_ingestion.py tests/test_correction_real_data.py tests/test_correction_training.py tests/test_e2e_api.py -q
............................................................................ [100%]
76 passed in 24.70s
```

---

## Next Scheduled Actions: Phase 2 (P1)

- **Task 2.3: Constrained Vocabulary Decoding for Objective Answers**
  - Constrain TrOCR decoding at MCQ answer positions to valid option tokens `{"A", "B", "C", "D", "E"}` and `{"True", "False"}`.
