# 📋 Development Log — InsightAI

> Step-by-step record of what was built, when, and current status.
> Updated as we progress through each phase.

---

## Phase 1: Scaffolding ✅
**Date:** 2026-05-15
**Status:** Complete

### What was done:
1. **Initialized Git repo** with remote `origin` → `github.com/Oxshadha/Data-Storm-7.0-InsightAI`
2. **Created full Lakehouse directory structure:**
   - `data/bronze/` → Raw ingestion (zero transforms)
   - `data/silver/` + `data/silver/rejected/` → Cleaned data + quarantine store
   - `data/gold/` → Feature-engineered, model-ready data
3. **Implemented core utilities (fully working):**
   - `src/utils/config.py` — Config loader from `pipeline_config.yaml`
   - `src/utils/logger.py` — Structured logging (console + file)
   - `src/utils/io.py` — Parquet/CSV read/write with auto-logging
4. **Implemented DQ framework (fully working):**
   - `src/silver/dq_checks.py` — 7 reusable, parameterizable check functions:
     - `check_duplicates()` — Composite key duplicate detection
     - `check_nulls()` — Mandatory field null detection
     - `check_referential_integrity()` — Foreign key validation
     - `check_value_range()` — Min/max boundary check
     - `check_format()` — Regex pattern validation
     - `check_negative_volumes()` — Return/reversal tagging
     - `check_zero_volumes()` — System adjustment tagging
   - `src/silver/quarantine.py` — `QuarantineManager` class (collects, stores, manifests)
5. **Implemented Bronze ingestion (fully working):**
   - `src/bronze/ingest_internal.py` — Reads all 5 CSVs → Parquet as-is
6. **Created pipeline orchestrator:**
   - `run_pipeline.py` — CLI with `--stage bronze|silver|gold|predict`
7. **Created config + tracking files:**
   - `config/pipeline_config.yaml` — All paths, thresholds, parameters
   - `requirements.txt` — Python dependencies
   - `ai_log/genai_transparency_log.md` — GenAI usage tracker
   - `experiments/experiment_log.md` — Model experiment tracker
   - `README.md` — Full architecture documentation

### What's NOT done (stubs only):
- `src/bronze/ingest_poi.py` — POI scraping (Overpass API)
- `src/silver/clean_*.py` — All 6 dataset cleaning scripts
- `src/gold/feature_*.py` — All feature engineering scripts
- `src/modeling/*.py` — Censoring detection + demand estimation
- `notebooks/` — No notebooks created yet

---

## Phase 2: Bronze Ingestion 🔴
**Date:** —
**Status:** Not started

### TODO:
- [ ] Run `ingest_internal.py` to convert all CSVs → Bronze parquet
- [ ] Implement POI scraping from OpenStreetMap (Overpass API)
- [ ] Validate Bronze row counts match raw CSV exactly

---

## Phase 3: Silver — Forensic Cleaning 🔴
**Date:** —
**Status:** Not started

### TODO:
- [ ] `clean_transactions.py` — Handle System Ghosts (negatives, zeros, dupes, outliers)
- [ ] `clean_outlet_master.py` — Fix typos, case, nulls
- [ ] `clean_coordinates.py` — Geo-validation, co-location detection
- [ ] `clean_seasonality.py` — Validate seasonality values
- [ ] `clean_holidays.py` — Parse dates, deduplicate
- [ ] `clean_poi.py` — Standardize POI data
- [ ] Verify quarantine store is populated with documented reasons

---

## Phase 4: Gold — Feature Engineering 🔴
**Date:** —
**Status:** Not started

### TODO:
- [ ] Outlet profile features (size, type, coolers)
- [ ] Transaction behavioral features (trends, variability, patterns)
- [ ] POI density & catchment features
- [ ] Seasonality & holiday features
- [ ] Censoring signal detection (flat volumes, capacity constraints)
- [ ] Join all into `model_input.parquet`

---

## Phase 5: Modeling — Demand Estimation 🔴
**Date:** —
**Status:** Not started

### TODO:
- [ ] Censoring detection (identify constrained outlets)
- [ ] Implement demand estimation (Tobit / Quantile / Bayesian)
- [ ] Generate `insightai_predictions.csv`

---

## Phase 6: Deliverables ✅
**Date:** 2026-05-31
**Status:** Complete

### Completed:
- [x] Final predictions CSV (submissions/InsightAI_predictions.csv)
- [x] 5-page PDF report (assets and report drafts in report_assets/ and report/)
- [x] Final README review (section_A..E updated to reference generated assets)
- [x] GenAI log complete (report_assets/genai_transparency_log.md and prompt archive summary)

Notes: Generated visual evidence (`feature_importance.svg`, `model_metrics.svg`, `budget_distribution.svg`), `top100_allocations.csv`, and copied GenAI transparency log into `report_assets/`.
