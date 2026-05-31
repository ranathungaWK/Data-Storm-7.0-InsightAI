**Data-Storm-7.0-InsightAI Pipeline**

Overview
--------
This repository contains the ETL and modeling pipeline used for the Data-Storm InsightAI competition. The pipeline ingests raw data, builds features (bronze → silver → gold), trains models, and produces prediction and reporting artifacts.

Quick Start
-----------
1. Install Python (3.10+ recommended) and Git.
2. Create and activate a virtual environment from the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# On Linux / macOS: source .venv/bin/activate
```

3. Install python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the full pipeline (examples):

```bash
# Run full pipeline
python run_pipeline.py --stage full

# Run only data cleaning / silver stage
python run_pipeline.py --stage silver

# Run feature engineering / gold stage
python run_pipeline.py --stage gold

# Run model training
python run_pipeline.py --stage models
```

5. View the dashboard (Streamlit):

```bash
python -m streamlit run output/dashboard/app.py
```

Logging
-------
- The project logging is configured in `src/utils/logger.py`. On Windows the logger is configured to use UTF-8 for console output to avoid encoding errors.
- Pipeline logs are written to files such as `sandbox_full_pipeline.log` or `full_pipeline.log` depending on how you run the pipeline.

Main Artifacts
--------------
- Predictions and model outputs: `data/models/`
- Reports and summaries: `output/report/` and `data/models/*.md` or `.csv`
- Dashboard: `output/dashboard/`

Folder Structure
----------------
- **data/**: raw and processed datasets and model artifacts
  - `data/external/` — large external inputs (OSM, etc.)
  - `data/silver/`, `data/gold/`, `data/models/` — processed data at different pipeline stages
- **src/**: code for ETL, feature engineering, and models
  - `src/bronze/`, `src/silver/`, `src/gold/`, `src/models/` — stage-specific modules
  - `src/utils/` — shared utilities (logging, helpers)
- **notebooks/**: Jupyter notebooks for exploration and audits
- **output/**: dashboards and reports (Streamlit app, report scripts)
- **config/**: pipeline configuration files (`pipeline_config.yaml`)
- **logs/**: run logs, rejection manifests, metrics
- **experiments/**: experiment notes and logs
- **cache/**: API/cache artifacts (local cache — safe to delete if you need space)

Development & Tests
-------------------
- Use the virtual environment when running scripts or tests.
- If there are tests, run with `pytest` from the repo root:

```bash
python -m pytest -q
```

Notes & Best Practices
----------------------
- Always use the same timezone/`datetime` normalization in feature engineering. The pipeline normalizes datetimes before joins in `src/gold/feature_seasonality.py`.
- Avoid running full pipeline on a machine with low RAM; pre-aggregate large joins (distributor/monthly) to reduce memory pressure.
- If you see Unicode errors on Windows console, ensure the environment activates the `.venv` and that logging is used via the configured logger.

Contributing
------------
- Create a branch for your change: `git checkout -b fix/your-change`
- Make small, focused commits and open a PR against `main`.

Contact
-------
If you need help running the pipeline or reproducing results, open an issue or ping the repository maintainer.
# Data Storm 7.0 — InsightAI

> **Competition:** Data Storm 7.0 — Storming Round (36-hour hackathon)
> **Team:** InsightAI
> **Objective:** Estimate the latent maximum monthly volume potential (in liters) for ~20,000 retail outlets for January 2026.

---

## 🏗️ Architecture — Lakehouse Pipeline

This repository follows the **Medallion Architecture** (Bronze → Silver → Gold) with a forensic data detective approach.

```
Bronze (Raw)  →  Silver (Cleaned)  →  Gold (Enriched)  →  Predictions
     ↓                  ↓
  As-Is CSV       Quarantined
  → Parquet       Rejected Records
```

### Directory Structure

```
├── config/                     # Pipeline configuration (YAML)
├── data/
│   ├── bronze/                 # Raw ingestion — zero transformations
│   ├── silver/                 # Cleaned + DQ-checked data
│   │   └── rejected/          # Quarantined records with documented reasons
│   └── gold/                   # Feature-engineered, model-ready data
├── src/
│   ├── bronze/                 # Ingestion scripts
│   ├── silver/                 # DQ checks + forensic cleaning
│   ├── gold/                   # Feature engineering
│   ├── modeling/               # Demand estimation models
│   └── utils/                  # Config, logging, I/O helpers
├── notebooks/                  # EDA, forensics, model experiments
├── ai_log/                     # GenAI transparency log (required deliverable)
├── experiments/                # Experiment tracking
├── output/                     # Final predictions + PDF report
├── run_pipeline.py             # Master pipeline orchestrator
└── Refernce Resources/         # Original competition data (read-only)
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Ensure Raw Data

Place `transactions_history_final.csv` (~169MB) in the `Refernce Resources/` directory. This file is excluded from Git due to size limits.

### 3. Run the Pipeline

```bash
# Full pipeline: Bronze → Silver → Gold → Predict
python run_pipeline.py

# Or run individual stages:
python run_pipeline.py --stage bronze
python run_pipeline.py --stage silver
python run_pipeline.py --stage gold
python run_pipeline.py --stage predict
```

---

## 🔬 Data Detective Philosophy

We treat data anomalies as **evidence**, not garbage:

| Ghost Type | Action | NOT This |
|---|---|---|
| Negative Returns | Tag as `RETURN`, aggregate for net volume | ~~Drop~~ |
| Zero-Volume Rows | Tag as `SYSTEM_ADJUSTMENT`, quarantine | ~~Ignore~~ |
| Duplicate Retries | Detect, keep one, quarantine rest with reason | ~~Deduplicate silently~~ |
| Extreme Outliers | Cross-reference with outlet profile | ~~Cap at percentile~~ |

---

## 📊 Deliverables

1. **`output/insightai_predictions.csv`** — Outlet_ID + Maximum_Monthly_Liters for Jan 2026
2. **This repository** — Reproducible codebase with Bronze → Silver → Gold structure
3. **`output/report/InsightAI_Report.pdf`** — 5-page technical summary

---

## 🤖 GenAI Usage

All AI interactions are documented in [`ai_log/genai_transparency_log.md`](ai_log/genai_transparency_log.md).
