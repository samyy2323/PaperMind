# Context Logging Rule

## Purpose
Enforces that [`rules/context.md`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/rules/context.md) acts as a living, focused update log rather than static boilerplate.

## Directive
After completing each prompt successfully:
1. Append or update [`rules/context.md`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/rules/context.md) with:
   - **Prompt Goal & Timestamp**
   - **Exact Files Created / Modified** with clickable markdown links
   - **Summary of Logic / Architecture Changes**
   - **Verification Results** (test suite runs, benchmarks, or build checks)
   - **Next Actions / Scheduled Work**
2. Ensure [`rules/context.md`](file:///c:/Users/samee/OneDrive/Desktop/PROJECT_ACHARYA/final_year_projects/PaperMind/PaperMind/rules/context.md) contains **only** information about what has been updated and the current live state, avoiding redundant static project descriptions (which belong in `PRD.md`, `ARCHITECTURE.md`, or `MVP.md`).
