# GenAI Transparency Log — InsightAI

> **Competition:** Data Storm 7.0 — Storming Round
> **Team:** InsightAI
> **Purpose:** Document how, where, and why Generative AI was used during the 36-hour hackathon.

---

## Usage Log

| # | Timestamp | Phase | Tool Used | Purpose | Prompt Summary | Output Used? | Validation Done |
|---|-----------|-------|-----------|---------|----------------|-------------|-----------------|
| 1 | 2026-05-15 22:00 | Planning | Claude (Antigravity) | Repository architecture design | "Plan the repo as a data scientist using Lakehouse architecture" | ✅ Adapted | Reviewed against competition rubric — aligned with Bronze/Silver/Gold requirements |
| 2 | 2026-05-15 22:30 | Scaffolding | Claude (Antigravity) | Code scaffolding | "Create DQ check framework, quarantine system, config loader" | ✅ Modified | Reviewed function signatures, tested imports |
| 3 | 2026-05-30 23:40 | Debugging | GitHub Copilot | Windows UTF-8 logging fix | "Make console logging UTF-8 safe on Windows" | ✅ Modified | Verified with a Unicode logging one-liner |
| 4 | 2026-05-30 23:50 | Validation | GitHub Copilot | Sandbox pipeline rerun | "Rerun full pipeline in sandbox after logger fix" | ✅ Modified | Confirmed exit code 0 and successful artifact generation |
| 5 | 2026-05-30 23:58 | Validation | GitHub Copilot | Pipeline completion review | "Check final pipeline logs for completion and outputs" | ✅ Modified | Confirmed full pipeline completion and expected model artifacts |
| 6 | 2026-05-31 00:00 | Documentation | GitHub Copilot | Project README creation | "Create a README explaining startup, requirements, and folder structure" | ✅ Modified | Checked setup steps and verified links to pipeline artifacts |
| 7 | 2026-05-31 00:10 | Submission Prep | GitHub Copilot | Peak sales submission export | "Create teamname_predictions.csv from Model A peak sales output" | ✅ Modified | Verified generated CSV preview and copied pipeline/README into submissions folder |
| 8 | 2026-05-31 00:20 | Submission Prep | GitHub Copilot | Budget allocation generation | "Create teamname_budget_allocations.csv using Model A opportunity scoring" | ✅ Modified | Ran generator script, verified file header, row count, and 5,000,000 LKR total |
| 9 | 2026-05-31 00:30 | Submission Prep | GitHub Copilot | Western Province filtering | "Derive Western Province outlets from coordinates and limit allocation to that subset" | ✅ Modified | Added bounding-box filter, reran generator, confirmed Western-only output |


---

## Guidelines

- **Every AI interaction** that produces code, analysis, or design decisions should be logged here.
- Mark whether the output was used **as-is**, **adapted**, or **rejected**.
- Document what **validation** was performed on AI-generated output.
- Save significant prompts/responses in `ai_log/prompt_archive/`.
