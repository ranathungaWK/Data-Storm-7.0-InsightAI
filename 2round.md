# Round 2 — Draft Plan
 # Round 2 — Plan, gaps and implementation mapping

 Purpose: translate the Round-2 requirements into a prioritized, code-aware workplan. For each item I map where the repo already covers it, where it is missing or partial, and concrete tasks + file targets to implement.

 ## 1) Existing assets (what we have right now)
 - Dashboard: output/dashboard/app.py (Streamlit) — reads parquet artifacts and visualizes KPIs, clusters and recommendations.
 - Feature pipeline: src/gold/build_outlet_features.py (transactional, POI, seasonality, holiday features) -> data/gold/outlet_features.parquet
 - Clustering: src/models/clustering.py -> data/models/outlet_clusters.parquet
 - Potential / opportunity scoring: src/models/potential_scoring.py, src/models/opportunity_recommendations.py -> data/models/opportunity_recommendations.parquet
 - Cluster profiles & visualizations: src/models/cluster_profiles.py, src/visualizations/plot_clusters.py, src/visualizations/opportunity_heatmap.py

 ## 2) Gap analysis (user's 9 criteria mapped to code)

 1. Existing Round 1 Architecture
	 - Covered: pipeline (src/silver -> src/gold), artifacts in data/gold and data/models, report in output/report/InsightAI_Report.md
	 - Files: [src/gold/build_outlet_features.py](src/gold/build_outlet_features.py), [src/models/clustering.py](src/models/clustering.py), [src/models/potential_scoring.py](src/models/potential_scoring.py)

 2. Distance decay model (missing / partial)
	 - Current: features include POI local counts and `holiday distance` types but no formal distance-decay weighting for demand or competitor influence.
	 - Change files: update feature builder [src/gold/build_outlet_features.py](src/gold/build_outlet_features.py) or add new module [src/gold/feature_distance_decay.py].
	 - Tasks:
		- Compute distance-weighted POI and competitor influence (e.g., sum w(d) where w(d)=exp(-alpha * d)).
		- Add features: `poi_decay_score`, `competitor_decay_score`, and tune `alpha` via validation.

 3. Competition density features (missing/partial)
	 - Current: POI local features exist (src/gold/feature_poi_local) but explicit competitor density (same-category outlets within catchment) is not present.
	 - Change files: [src/gold/build_outlet_features.py](src/gold/build_outlet_features.py) or new [src/gold/feature_competition.py].
	 - Tasks:
		- Identify competitor categories from POI/outlet_type and compute counts, distance-decayed counts, and nearest-competitor distance.
		- Add `competitor_count_500m`, `nearest_competitor_m`, `competitor_density_decay` to outlet_features.parquet.

 4. Catchment / population / accessibility (partial)
	 - Current: no population or travel-time catchment layers. Coordinates present in data/ and used for heatmap.
	 - Change files: add a catchment module [src/gold/feature_catchment.py] and optional integration with external population raster / OSM isochrones.
	 - Tasks:
		- Add simple radial catchment features (sum transactions / volume within 1km/3km) and optionally integrate isochrone-based accessibility if time allows.

 5. Budget allocation engine (missing)
	 - Current: opportunity_recommendations.parquet contains opportunity scores but there is no budget optimization.
	 - Change files: new module [src/models/budget_allocator.py] + a lightweight API that consumes opportunity_recommendations.
	 - Tasks:
		- Implement knapsack/linear-programming-based allocator to maximize expected incremental volume subject to budget and per-outlet cost.
		- Output `allocated_budget`, `expected_lift` per outlet and aggregated plan.
		- Integrate into dashboard (output/dashboard/app.py) with budget slider and allocation view.

 6. Explainability (XAI) layer (missing)
	 - Current: no SHAP / per-outlet feature attributions recorded.
	 - Change files: add explainability utilities [src/models/explainability.py] and call from potential scoring or a post-hoc step to write attributions into data/models/opportunity_recommendations.parquet (columns like `shap_top_features`).
	 - Tasks:
		- Train a simple surrogate model or use tree-based model to get feature importances and per-row SHAP values (shap package).
		- Save top-N feature attributions per outlet and add UI to dashboard to inspect per-outlet explanations.

 7. Model upgrades (robustness & evaluation)
	 - Current: KMeans clustering and ad-hoc scoring with silhouette ~0.41.
	 - Change files: [src/models/clustering.py], [src/models/potential_scoring.py], tests/notebooks for evaluation.
	 - Tasks:
		- Add clustering stability checks (multiple seeds), silhouette/Calinski, and consider GaussianMixture or HDBSCAN if clusters are noisy.
		- Validate scoring by historical uplift proxy (A/B holdout) or proxy features.

 8. Dashboard improvements & web app features
	 - Current: Streamlit app at [output/dashboard/app.py](output/dashboard/app.py) — loads artifacts and shows KPIs; contains deprecated `use_container_width` warnings.
	 - Change files: update [output/dashboard/app.py](output/dashboard/app.py).
	 - Tasks:
		- Replace deprecated calls (`use_container_width` -> `width`), add per-outlet explainability panel, add budget allocation control and allocation results table, add endpoints to download allocation plan CSV.
		- Add health-check and quick-start docs in README.

 9. Deliverables, reproducibility, and experiments tracking
	 - Current: artifacts saved but no experiment tracking (MLflow) and no consistent experiment metadata.
	 - Change files: add `config/experiments/template.yaml`, small `src/tools/record_experiment.py` to log hyperparams and dataset hashes, or opt for MLflow integration.
	 - Tasks:
		- Save run metadata (git commit, dataset hashes, params) alongside generated artifacts. Add a reproducible `run_pipeline.py --exp` flag.

 ## 3) Prioritized implementation roadmap (short term → medium)

 Phase 1 (High impact, quick wins — 1–2 weeks)
 - Add competition density and distance-decay features into `src/gold/` (feature_competition.py and feature_distance_decay.py). Update build_outlet_features.py to include them and save to data/gold/outlet_features.parquet.
 - Add minimal budget allocator prototype (greedy knapsack) in `src/models/budget_allocator.py` and wire a simple control into Streamlit dashboard.
 - Fix Streamlit deprecation warnings in [output/dashboard/app.py](output/dashboard/app.py) and add per-outlet detail drawer.

 Phase 2 (Explainability + model hardening — 1–3 weeks)
 - Add SHAP-based explainability pipeline `src/models/explainability.py` and store per-outlet attributions.
 - Add clustering stability tests and alternative clustering (GMM/HDBSCAN) experiments.
 - Add experiment metadata saving and a small experiment index (CSV or MLflow).

 Phase 3 (Optional / stretch)
 - Replace radial catchment with isochrones (OSRM or isochrone API) and population layers; build accessibility-based features.
 - Add API layer (FastAPI) to serve allocation & explanations for production dashboards.

 ## 4) Concrete next steps I will take now (if you want me to proceed)
 1. Implement distance-decay + competitor density features (add new modules and update pipeline). This is highest-impact and feeds clustering + scoring.
 2. Add a small budget allocator prototype and integrate basic UI in Streamlit.
 3. Add SHAP explainability skeleton (compute and store top-3 attributions per outlet).

 ## 5) Notes, assumptions and risks
 - Assumption: the existing parquet artifacts are up-to-date and represent latest pipeline outputs in `data/gold` and `data/models`.
 - Risk: fetching POI via Overpass in `feature_poi` can fail due to API limits (logs show 406s). Consider relying on cached POI (`data/overture`) or local OSM extracts.
 - Data privacy: exports with location + budget must be handled cautiously if sharing externally.

 ---
 If you want I'll start by implementing the distance-decay + competitor-density features and update `src/gold/build_outlet_features.py` and add a new `src/gold/feature_competition.py`. Proceed?

## Current status (what we have)
- Data: `data/external/` (OSM PBF), `data/gold/`, `data/silver/` (cleaned/rejected), parquet and CSV artifacts in `src/` and `data/`.
- Pipeline: `run_pipeline.py`, `config/pipeline_config.yaml`.
- Codebase: `src/` (bronze, silver, gold, models, utils), `notebooks/01_data_audit.ipynb`.
- Experiments & outputs: `experiments/` (log), `output/dataset_audit_summary.csv`, `dashboard/` app, `reports/`.
- Environment: `requirements.txt`, virtualenv activated in workspace.
- Logs & cache: `logs/`, `ai_log/`, `cache/` with cached responses.

## Things we need (gaps to fill)
- A clear, single Round-2 objective (confirm with team: metric or deliverable).
- Baseline evaluation numbers and target improvement (current metric values).
- Prioritized experiment list with estimated effort and owners.
- Reproducible experiment config and a lightweight tracking mechanism (CSV or MLflow).
- Standardized dataset snapshots (path + checksum) used for each experiment.
- Short timeline with milestones and acceptance criteria for each deliverable.

## Thoughts & suggestions
- Choose one measurable objective (example: improve competition score by X% or increase F1 by Y points). This keeps experiments focused.
- Limit Round 2 to 2–4 high-impact experiments: focused feature engineering, simple model improvements, and a stable baseline re-run.
- Use `config/` to store versioned experiment configs (YAML files) and `experiments/results.csv` to record runs (config, seed, dataset-hash, metrics, artifact path).
- Prefer reproducibility over complex infra: CSV tracking + artifact folders are enough for now; migrate to MLflow only if we need many parallel runs.
- Add an `evaluation.ipynb` to produce the final dashboard/plots and to generate `InsightAI_Report.md` snippets.

## Proposed immediate plan (next 2 weeks)
1. Define Round-2 objective and metric (1 day) — owner: you / team.
2. Create experiment template & tracking (1 day):
	- `config/experiments/template.yaml` (parameters: dataset, features, model, seed, notes)
	- `experiments/results.csv` (columns: date, config, dataset_hash, seed, metric, notes, artifact_path)
3. Prioritize 3 experiments (2 days): list, estimate effort, set order.
4. Implement & run experiments (7 days): run sequentially, record results, produce interim evaluation.
5. Finalize report & dashboard (2 days): update `reports/InsightAI_Report.md` and `dashboard/` views.

## Suggested prioritized experiments (examples)
- Experiment A — Improved feature set (time-series lags, holiday features, spatial aggregations).
- Experiment B — Alternative model family (e.g., gradient boosting vs neural net) with tuned hyperparams.
- Experiment C — Data augmentation / cleaning improvements and re-run baseline.

## Repro steps (how to run a tracked experiment)
1. Create a `config/experiments/<name>.yaml` from the template.
2. Run:

```bash
python run_pipeline.py --config config/pipeline_config.yaml --experiment config/experiments/<name>.yaml
```

3. Append result row to `experiments/results.csv` with dataset hash and metric.

## Next immediate actions I will take
- (If you agree) create `config/experiments/template.yaml` and an initial `experiments/results.csv` header, and commit the draft of this `2round.md`.

---
_Draft created: please review objective and pick owners for the prioritized experiments so I can flesh out a task timeline and create experiment config files._

