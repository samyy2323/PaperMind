# Accuracy Analysis and Roadmap — what is wrong, what is missing, how to fix it

Evidence base: the real-exam test on the public Mendeley dataset (3 completed students, 150 marks, 45 short answers, 60 MCQ),
the ByT5 held-out evaluation, and the unit tests. **Small sample: treat numbers as indicators, not a benchmark.**
Claims about models we have not run here are marked *(to benchmark)*.

---

## 1. Bottom line

1. **The bottleneck is not the models you list (fine-tuned LLM, CNN, RNN).** It is the *structure around* them: finding which handwriting
   answers which question, flagging what we are unsure about, and cleaning the page image before OCR.
2. **The system is silent when it is wrong.** 26 of the 30 wrong answers were *not flagged*; the teacher would never know to check them.
   This is the single most important design flaw, and it is cheap to fix.
3. **Fix order matters.** Cheap, structural fixes first (days), then targeted model upgrades with measured gains. Training new models before
   fixing the structure would mostly polish the wrong thing.

## 2. Where the marks are lost (measured)

| Quantity (3 students) | Teacher | Ours |
|---|---|---|
| MCQ marks (60 questions) | 33 | 27 |
| Short-answer marks (45 questions) | 47 | 13 |

**Short-answer marks lost: 34**
| Cause | Marks lost | Share | Evidence |
|---|---|---|---|
| **Answer not found** (empty extraction → score 0) | **26** | **76 %** | 28 of 45 short answers extracted as empty; 16 of those had marks from the teacher |
| Found, but the LLM scored lower than the teacher (>0.5) | 8 | 24 % | 8 questions; **never** over-scored |
| Found and within 0.5 of the teacher | — | — | 9 questions |

**Silent errors:** of 30 answers that disagreed with the teacher, **only 4 were flagged** (all "low confidence"); **26 were silent**.
Empty extractions are not flagged at all.

**Other things the investigation exposed**
* **OCR invents text.** Student_1's page 4 is a *blank back page* with faint mirrored bleed-through of page 3. TrOCR produced 19 lines of
  fluent nonsense from it (mean confidence 0.43). That text flows into the last answer.
* Two of three students had only 13/20 MCQ letters read correctly (harder handwriting).
* The ByT5 correction stage (v1) corrupted two-digit question numbers (now guarded); we have **never measured whether correction helps
  the final grade** (no ablation).

## 3. Root causes, ranked by impact

| # | Root cause | Effect | Fix difficulty |
|---|---|---|---|
| 1 | Answer localisation relies on OCR-reading the question number; one garbled digit loses the answer (`38.` for `33.`, `#` for `34.`, `all` for `31.`) | 76 % of lost short-answer marks | Medium |
| 2 | No "answer missing / not found" flag; confidence is only about the LLM | 87 % of errors invisible | **Easy** |
| 3 | No image preprocessing (bleed-through, blank pages, ruled lines, margins) → hallucinated or noisy lines | Garbage text appended to answers | Easy–medium |
| 4 | Small zero-shot grader (3B) is harsher than a teacher; no calibration to *this* teacher | 24 % of lost short marks | Easy–medium |
| 5 | MCQ letters: free-text OCR for a 1-character answer from a 4-letter set | up to ~7 of 20 wrong on hard hands | Easy (constrained decoding) |
| 6 | OCR accuracy on difficult handwriting (TrOCR base, not adapted to these students) | MCQ misses, garbled text | Medium–hard |
| 7 | Evaluation gaps: no ablations, no oracle experiments, 3 students | We cannot tell which stage matters | Easy |

## 4. What we are missing (checklist)

* **Image preprocessing**: deskew, contrast normalisation, ruled-line and margin-line removal, **bleed-through suppression**, blank-page detection.
* **Missing-answer detection and "evidence" for the teacher**: line bounding boxes saved per answer so the UI can show the *image crop* beside the OCR text.
* **Content-aware segmentation**: use the question text/reference answer, not only the number.
* **Calibration to the teacher**: a few teacher-graded examples to align the AI's scale with theirs.
* **Ablations and oracle experiments** (§8) so we know what each stage contributes.
* **A feedback loop**: every teacher override and correction is a training label that we currently throw away.
* **An answer-sheet design** that makes the machine's job easy (§7) — a process change, not a model change.
* A second, independent opinion (ensemble/agreement) to turn disagreement into a review flag.

## 5. Component-by-component options

### 5.1 Answer localisation (biggest lever)
| Option | Idea | Pros | Cons | Verdict |
|---|---|---|---|---|
| **Flag empty/missing answers** | An answer with no text → `needs_review: answer_not_found` | Hours of work; makes the 26 silent errors visible | Does not recover marks by itself | **Do now** |
| **Elimination** | If a question has no marker, assign orphan blocks to the unused question numbers | Simple | Ambiguous when several are missing | Do, with content matching |
| **Content matching** | Embed each unassigned block and the *question text + reference answer* (MiniLM already in the project); assign by best similarity under the constraint "one block per question" (Hungarian / DP). Students often echo the question ("Gradient descent is…") | Recovers garbled-number answers; reuses existing models | Needs a similarity threshold; short answers are ambiguous | **Do next** |
| **Geometry** | Use line positions: question numbers sit at the left margin, answers are indented, vertical gaps separate answers | Independent of OCR digits | Needs line bounding boxes (we have them in the detector) | Do with the above |
| **LLM assigns lines to questions** | Give the LLM the question list + OCR lines | Flexible | Slow, can hallucinate, hard to audit | Only for ambiguous leftovers |
| **Document-layout models** (LayoutLMv3, Donut) | Learn the structure | Powerful | Need labelled layouts; heavy | Not now |
| **Structured answer sheets / OMR** | Printed question numbers and boxes; MCQ as bubbles | Removes the problem; OMR is near 100 % | Needs a process change | **Best long-term** (§7) |

### 5.2 OCR (reading handwriting)
| Option | Notes | Verdict |
|---|---|---|
| **Preprocessing first** | Suppress bleed-through (the real ink is strongly coloured/dark, the ghost is pale grey: threshold on darkness/saturation), remove ruling/margin lines, detect and skip blank pages, drop lines with very low confidence or almost no ink | **Do first** — cheap, removes hallucinations |
| **Constrained decoding for MCQ** | At a one-letter answer position restrict TrOCR's output to {A,B,C,D,E} (or True/False). Zero training | **Do** — directly targets MCQ misreads |
| **Fine-tune TrOCR on the target handwriting** | The most likely *model* upgrade that pays off: needs ~1–3k transcribed lines from your own students (the app can produce these from teacher corrections). TrOCR base was tuned on IAM, not on your cohort | **Do after** collecting labels |
| **TrOCR-large-handwritten** | Bigger, slower (~3×), usually more accurate *(to benchmark)* | Benchmark on the Mendeley sheets |
| **Open vision-language models** (e.g. Qwen-VL, Florence-2, InternVL, MiniCPM-V families) | Can read a whole page and return text per question; also give layout. Heavy (GPU), occasional hallucination *(to benchmark)* | Use as a **second opinion**; disagreement with the pipeline = review flag |
| **CRNN (CNN + BiLSTM + CTC)** | Old, small, fast; strong only when trained on a lot of in-domain data | Not as a TrOCR replacement; useful as a tiny **specialist** (§6) |
| Tesseract / EasyOCR / PaddleOCR | Weak on cursive/hand-printed exam writing | Not recommended |
| N-best + rescoring | Keep TrOCR's top hypotheses, pick with a language model/ByT5 | Possible later |

### 5.3 Text correction (ByT5)
* Keep it, but **v2** (two-digit markers, more real handwriting, your own pages) and **keep the runtime guards**.
* **Measure it**: run the whole pipeline *with and without* correction on the same sheets. If grades do not improve, simplify.
* Safer variant: use correction only for *display* and for grading, never for question numbers or digits (already done).

### 5.4 Grading
| Option | Notes | Verdict |
|---|---|---|
| Bigger zero-shot model (8B → 14B/70B) | Better calibration and reasoning; needs more RAM/GPU than a 16 GB laptop (use a GPU box/Colab/another machine) | **Try 8B first**; measure with `validate_grader.py` on Mohler |
| **Rubric/key-point scoring** | Extract key points from the reference answer, ask "is point k present?" per point, sum | More consistent and explainable than one holistic number | **Do** |
| **Calibration to the teacher** | Teacher grades ~20 answers; fit a mapping (linear/isotonic) from AI score to teacher marks | Fixes harshness cheaply; per-teacher | **Do** |
| **Second signal** | Embedding similarity / cross-encoder score; combine with the LLM score; disagreement → flag | Cheap, robust | **Do** |
| **Fine-tune a scorer** | LoRA-tune an 7–8B model, or fine-tune a cross-encoder (e.g. DeBERTa-style) as a regressor, on Mohler (~2.3k) + ASAP-SAS (large, on Kaggle) + the labelled Mendeley answers | Usually beats zero-shot on in-domain data *(to benchmark)* | Do **only if** the gap remains after the above |
| Prompt tweaks | Already: tolerate OCR typos, full marks for the core idea | Done (measured: modest gain) | — |

### 5.5 Flagging and the human in the loop
* Add flag reasons: `answer_not_found`, `low_ocr_confidence`, `low_extraction_confidence` (block assigned by content, not by number),
  `model_disagreement`. Auto-accept only when **all** signals agree.
* **Measure flags like a classifier**: of the wrong answers, what fraction do we flag (recall)? of the flagged, what fraction is wrong
  (precision)? The goal is *coverage vs accuracy*: e.g. "auto-accepts 60 % of answers at 98 % agreement with the teacher, flags the rest".
* **Show evidence**: per answer, the image crop + OCR text + the model's reasoning; let the teacher re-assign a block to a question with one click.
* Log every override/re-assignment/correction as a training example (active learning).

## 6. "Should we use a fine-tuned LLM / CNN / RNN / other models?" — direct answers

| Question | Answer |
|---|---|
| Fine-tuned **LLM** for grading? | **Yes, but later.** First: bigger zero-shot model + rubric scoring + calibration + embedding second-signal, measured on Mohler. Fine-tune (LoRA) if a gap remains. We already have labelled data (Mendeley teacher marks, Mohler, ASAP-SAS) |
| Fine-tuned **OCR** (TrOCR)? | **Probably the best model investment**, once ~1–3k lines of your students' writing are transcribed (app overrides produce them) |
| **CNN**? | As a *specialist*: classify the single MCQ letter/tick or detect blank/ink regions. Not to replace TrOCR (TrOCR already has a vision-transformer encoder) |
| **RNN/LSTM**? | Not needed. Transformers replaced them; a CRNN could be a light ensemble member, nothing more |
| **Vision-language model**? | Worth benchmarking as an independent second reader and for layout; use disagreement as a flag |
| **Layout model** (LayoutLMv3/Donut)? | Not now; structure the answer sheet instead |
| **More correction models**? | No; finish v2 and test whether correction helps at all |
| **Cross-encoder / embeddings**? | Yes as a cheap second scoring signal |

## 7. Should we change the approach? A recommended re-architecture

**"Structure first, then read, then grade with evidence."**

```
1 PREPROCESS   deskew, normalise, remove ruling/margins, suppress bleed-through, drop blank pages
2 LAYOUT       detect lines WITH boxes; find left-margin question numbers (geometry); candidate answer regions
3 LOCALISE     assign regions to questions: number hints + content match (embeddings vs question/reference) + order/elimination
               ── unresolved or low-confidence regions become explicit review items, never silent zeros
4 READ         TrOCR per line; constrained decoding for MCQ/true-false; (optional) VLM second reading
5 CORRECT      ByT5 v2 for body text only (numbers/digits protected)
6 GRADE        objective: rules.  subjective: key-point scoring + LLM + embedding signal, calibrated to the teacher
7 DECIDE       auto-accept only when OCR, localisation and grading signals agree; otherwise flag with the image crop
8 LEARN        every teacher fix is stored as a training label (OCR lines, correction pairs, grader labels)
```
**Process change with the largest payoff:** give students a **structured answer booklet**: numbered boxes per question and an
**OMR bubble grid for the MCQs**. OMR (classical computer vision) is close to 100 % accurate for bubbles, and boxes eliminate
the localisation problem entirely. This is how many universities already handle MCQs.

## 8. How to measure (so we stop guessing)

1. **Run all 50 Mendeley students**; report MCQ accuracy, short-answer Pearson / quadratic-weighted kappa / MAE, total-score error.
2. **Ablations**: with/without preprocessing, with/without ByT5, content matching on/off, 3B vs 8B grader.
3. **Oracle experiments** to find the ceiling of each stage: (a) give the grader *perfect* segmentation (manual) → grader-only error;
   (b) give it *typed* answers for ~10 students → OCR+segmentation share of the error. Tells us where effort pays.
4. **Flag quality**: recall/precision of "needs review", and the *coverage–accuracy curve*.
5. **Your own pages**: 20–30 real pages from your students with typed transcripts (test-only), because IAM and Mendeley are not your cohort.
6. Report **teacher time saved**: minutes per sheet with and without the tool.

## 9. Prioritised plan

| Priority | Task | Effort | Expected effect (hypothesis) |
|---|---|---|---|
| **P0** (hours–1 day) | Flag empty/missing answers (`answer_not_found`) | very low | Makes the 26 silent errors visible; teacher sees exactly what to check |
| **P0** | Blank-page / low-ink / low-confidence line filter; basic bleed-through suppression | low | Removes hallucinated text (Student_1 page 4) |
| **P0** | Ablation switch for the correction stage; run the 50-student evaluation | low | Tells us what really matters |
| **P1** (days) | Content-aware + geometry + elimination segmentation | medium | Targets the 26 marks lost to "not found" (up to ~76 % of the short-answer loss, realistically a part of it) |
| **P1** | Constrained decoding for MCQ letters | low | Fewer MCQ misreads on hard handwriting |
| **P1** | Calibration to the teacher + embedding second signal + key-point scoring | low–medium | Targets the 8 marks lost to harsh grading; better flags |
| **P1** | Try an 8B grader; run `validate_grader.py` on Mohler | low | Quantifies grader quality properly |
| **P2** (1–2 wk) | ByT5 v2 (1–50 markers, real handwriting, own pages) | medium (Colab) | Cleaner text; verify by ablation |
| **P2** | Save line bounding boxes; evidence view in the review UI; one-click re-assign | medium | Turns residual errors into quick checks |
| **P2** | Active-learning log of teacher fixes | low | Free training data from real use |
| **P3** | Fine-tune TrOCR on your cohort; benchmark TrOCR-large and a VLM second reader | medium–high | Better reading on hard handwriting |
| **P3** | LoRA-fine-tuned scorer if still needed | medium | Closer match to teacher marks |
| **P3** | Structured answer booklets + OMR for MCQs | process change | Removes localisation errors by design |

**Upper bound from the data we have:** the 28 empty short answers include some the student genuinely left blank (the teacher gave marks on
16 of them). Even perfect localisation recovers at most those 16 answers (26 marks); the grader's own 8-mark gap and OCR quality
remain. Expect large but not total gains from P0–P1, and use the flags to catch the rest.

## 10. Honest limits of this analysis
* Three students and one exam, one language, one set of handwriting styles.
* The error budget attributes a "not found" to localisation, but part of it may be OCR failure on that line; the oracle experiments (§8.3) separate these.
* Statements about external models (TrOCR-large, VLMs, LoRA scorers) are expectations to *benchmark*, not results we have measured here.
