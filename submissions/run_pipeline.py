"""
Master Pipeline Orchestrator — Bronze → Silver → Gold → Predict

Usage:
    python run_pipeline.py                # Run full pipeline
    python run_pipeline.py --stage bronze # Run only bronze stage
"""

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger, setup_logging

logger = get_logger("pipeline")


def run_script(script_path: str) -> None:
    import runpy

    logger.info(f"Running {script_path}")
    runpy.run_path(script_path, run_name="__main__")


def run_required(script_path: str) -> None:
    try:
        run_script(script_path)
    except Exception:
        logger.exception(f"Failed running {script_path}")
        raise


def run_seasonality_features() -> None:
    from src.gold.feature_seasonality import build_seasonality_features

    transactions_path = Path("data/silver/cleaned/transactions_cleaned.parquet")
    holidays_path = Path("data/silver/cleaned/holidays_cleaned.parquet")
    seasonality_path = Path("data/silver/cleaned/seasonality_cleaned.parquet")
    output_path = Path("data/gold/seasonality_features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    txns = pd.read_parquet(transactions_path)
    holidays = pd.read_parquet(holidays_path)
    seasonality = pd.read_parquet(seasonality_path)

    if "date" not in txns.columns:
        txns["date"] = pd.to_datetime(dict(year=txns["year"], month=txns["month"], day=1), errors="coerce")

    seasonality_features = build_seasonality_features(txns, holidays, seasonality)
    seasonality_features.to_parquet(output_path, index=False)
    logger.info(f"Saved seasonality features -> {output_path}")


def package_model_artifacts() -> None:
    root = Path.cwd()
    models_dir = root / "models"
    data_models_dir = root / "data" / "models"

    packages = [
        {
            "folder": "model_a_peak_sales",
            "title": "Model A - Peak Sales",
            "description": "Predicts the highest monthly sales an outlet can achieve using historical demand signals plus location and business context.",
            "feature_rule": "Uses history features plus location and business context. This is the historical peak-sales model.",
            "opportunity_formula": "Opportunity = PeakSales - CurrentSales",
            "source_pkl": models_dir / "peak_sales_rf_model.pkl",
            "source_lgbm": models_dir / "peak_sales_lgbm_model.txt",
            "source_predictions": data_models_dir / "peak_sales_predictions.parquet",
            "source_importance": models_dir / "peak_sales_feature_importance.csv",
            "pred_col": "predicted_value",
            "target_col": "actual_max_monthly",
        },
        {
            "folder": "model_b_store_capacity",
            "title": "Model B - Store Capacity",
            "description": "Estimates store capacity from outlet_size and cooler_count as the capacity proxy.",
            "feature_rule": "Uses only outlet_size and cooler_count.",
            "opportunity_formula": "Opportunity = StoreCapacity - CurrentSales",
            "source_pkl": models_dir / "store_capacity_rf_model.pkl",
            "source_lgbm": models_dir / "store_capacity_lgbm_model.txt",
            "source_predictions": data_models_dir / "store_capacity_predictions.parquet",
            "source_importance": models_dir / "store_capacity_feature_importance.csv",
            "pred_col": "predicted_value",
            "target_col": "actual_max_monthly",
        },
        {
            "folder": "model_b_market_capacity",
            "title": "Model B - Market Capacity",
            "description": "Estimates latent market capacity using location, POI, competition, accessibility, mobility, commercial activity, and outlet static signals.",
            "feature_rule": "Uses location, POI, competition, accessibility, mobility, commercial activity, outlet_size, and cooler_count. This is the broader market-capacity model.",
            "opportunity_formula": "market_gap = predicted_value - current_sales",
            "source_pkl": models_dir / "market_capacity_rf_model.pkl",
            "source_lgbm": models_dir / "market_capacity_lgbm_model.txt",
            "source_predictions": data_models_dir / "market_capacity_predictions.parquet",
            "source_importance": models_dir / "market_capacity_feature_importance.csv",
            "pred_col": "predicted_value",
            "target_col": "actual_max_monthly",
        },
        {
            "folder": "model_c_market_demand",
            "title": "Model C - Market Demand",
            "description": "Predicts latent market demand using only POI, competition, accessibility, mobility, and commercial activity.",
            "feature_rule": "Uses only POI, competition, accessibility, mobility, and commercial activity.",
            "opportunity_formula": "Opportunity = MarketDemand - CurrentSales",
            "source_pkl": models_dir / "model_c_market_demand_rf_model.pkl",
            "source_lgbm": models_dir / "model_c_market_demand_lgbm_model.txt",
            "source_predictions": data_models_dir / "model_c_market_demand_predictions.parquet",
            "source_importance": models_dir / "market_demand_feature_importance.csv",
            "pred_col": "predicted_market_demand",
            "target_col": "actual_max_monthly",
        },
    ]

    for package in packages:
        folder = models_dir / package["folder"]
        folder.mkdir(parents=True, exist_ok=True)

        with open(package["source_pkl"], "rb") as f:
            loaded = pickle.load(f)

        if isinstance(loaded, dict):
            model_object = loaded.get("model_object") or loaded.get("model") or loaded.get("rf_model") or loaded.get("final_model")
        else:
            model_object = loaded

        predictions = pd.read_parquet(package["source_predictions"])
        eval_df = predictions.query("split == 'test'").copy() if "split" in predictions.columns else predictions.copy()
        y_true = eval_df[package["target_col"]].astype(float).to_numpy()
        y_pred = eval_df[package["pred_col"]].astype(float).to_numpy()
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)) if len(y_true) > 1 else float("nan")
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)
        mdape = float(np.median(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)

        bundle = {
            "model_name": package["title"],
            "description": package["description"],
            "model_object": model_object,
            "metrics": {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape": mape,
                "mdape": mdape,
            },
            "current_sales_column": "current_sales",
            "opportunity_formula": package["opportunity_formula"],
            "feature_rule": package["feature_rule"],
            "source_artifacts": {
                "source_pkl": str(package["source_pkl"].relative_to(root)),
                "source_lgbm": str(package["source_lgbm"].relative_to(root)),
                "source_predictions": str(package["source_predictions"].relative_to(root)),
                "source_importance_csv": str(package["source_importance"].relative_to(root)),
            },
        }

        with open(folder / "model.pkl", "wb") as f:
            pickle.dump(bundle, f)

        shutil.copy2(package["source_lgbm"], folder / "lgbm_model.txt")
        shutil.copy2(package["source_predictions"], folder / "predictions.parquet")

        importance_df = pd.read_csv(package["source_importance"])
        top_lines = [
            f"{index + 1}. {row.feature} - {row.importance}"
            for index, row in importance_df.head(20).reset_index(drop=True).iterrows()
        ]
        (folder / "feature_importance.txt").write_text(
            "\n".join([
                package["title"],
                "",
                f"Source: ../{package['source_importance'].name}",
                package["feature_rule"],
                "",
                "Top feature importances:",
                *top_lines,
            ]) + "\n",
            encoding="utf-8",
        )

        readme_lines = [
            f"# {package['title']}",
            "",
            package["description"],
            "",
            "## Feature Rules",
            f"- {package['feature_rule']}",
            "",
            "## Opportunity Definition",
            f"- {package['opportunity_formula']}",
            "",
            "## Metrics",
            f"- RMSE: {rmse:.2f}",
            f"- MAE: {mae:.2f}",
            f"- R2: {r2:.4f}",
            f"- MAPE: {mape:.1f}%",
            f"- MdAPE: {mdape:.1f}%",
            "",
            "## Files",
            "- model.pkl: pickled model bundle and metadata",
            "- lgbm_model.txt: LightGBM booster snapshot",
            "- predictions.parquet: model output parquet file",
            "- feature_importance.txt: plain-text feature importance summary",
            "",
            "## Source Artifacts",
            f"- {package['source_pkl'].relative_to(root)}",
            f"- {package['source_lgbm'].relative_to(root)}",
            f"- {package['source_predictions'].relative_to(root)}",
            f"- {package['source_importance'].relative_to(root)}",
        ]
        (folder / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


def run_full(config: dict) -> None:
    logger.info("=" * 60)
    logger.info("FULL PIPELINE — CLEAN -> FEATURES -> MODELS")
    logger.info("=" * 60)

    # Stage 1: cleaning / preprocessing
    run_required("src/silver/clean_transactions.py")
    run_required("src/silver/clean_coordinates.py")
    run_required("src/silver/clean_outlet_master.py")
    run_required("src/silver/clean_holidays.py")
    run_required("src/silver/clean_poi.py")
    run_required("src/silver/clean_seasonality.py")

    # Stage 2: preprocessing / feature engineering
    run_seasonality_features()
    run_required("src/gold/feature_poi.py")
    run_required("src/gold/build_outlet_features.py")
    run_required("src/models/build_potential_target.py")

    # Stage 3: model outputs
    run_required("src/models/cluster_profiles.py")
    run_required("src/models/opportunity_recommendations.py")
    run_required("src/models/train_dual_potential_models.py")
    run_required("src/models/train_market_demand_model.py")

    # Stage 4: package artifacts into the named model folders
    package_model_artifacts()

    logger.info("FULL PIPELINE COMPLETE")


def run_bronze(config: dict) -> None:
    """Stage 1: Raw ingestion — CSV → Parquet, zero transforms."""
    logger.info("=" * 60)
    logger.info("STAGE 1: BRONZE — Raw Ingestion")
    logger.info("=" * 60)
    logger.warning("Bronze ingestion module is not present in this repo; source inputs are already stored under src/bronze/.")


def run_silver(config: dict) -> None:
    """Stage 2: Forensic cleaning — DQ checks, quarantine."""
    logger.info("=" * 60)
    logger.info("STAGE 2: SILVER — Forensic Cleaning & DQ Checks")
    logger.info("=" * 60)
    # TODO: Wire up silver cleaning scripts
    logger.warning("Silver stage not yet implemented.")


def run_gold(config: dict) -> None:
    """Stage 3: Feature engineering."""
    logger.info("=" * 60)
    logger.info("STAGE 3: GOLD — Feature Engineering")
    logger.info("=" * 60)
    # Run gold scripts in dependency order.
    import runpy

    try:
        logger.info("Running src/gold/feature_poi.py")
        runpy.run_path("src/gold/feature_poi.py", run_name="__main__")
    except Exception:
        logger.exception("Failed running feature_poi.py")

    try:
        logger.info("Running src/gold/build_outlet_features.py")
        runpy.run_path("src/gold/build_outlet_features.py", run_name="__main__")
    except Exception:
        logger.exception("Failed running build_outlet_features.py")

    # compute potential target after features are built
    try:
        logger.info("Running src/models/build_potential_target.py")
        runpy.run_path("src/models/build_potential_target.py", run_name="__main__")
    except Exception:
        logger.exception("Failed running build_potential_target.py")


def run_predict(config: dict) -> None:
    """Stage 4: Demand estimation & prediction."""
    logger.info("=" * 60)
    logger.info("STAGE 4: PREDICT — Demand Estimation")
    logger.info("=" * 60)
    import runpy

    try:
        logger.info("Running src/models/cluster_profiles.py")
        runpy.run_path("src/models/cluster_profiles.py", run_name="__main__")
    except Exception:
        logger.exception("Failed running cluster_profiles.py")

    try:
        logger.info("Running src/models/opportunity_recommendations.py")
        runpy.run_path("src/models/opportunity_recommendations.py", run_name="__main__")
    except Exception:
        logger.exception("Failed running opportunity_recommendations.py")

    try:
        logger.info("Running src/models/train_market_demand_model.py")
        runpy.run_path("src/models/train_market_demand_model.py", run_name="__main__")
    except Exception:
        logger.exception("Failed running train_market_demand_model.py")


STAGES = {
    "full": run_full,
    "bronze": run_bronze,
    "silver": run_silver,
    "gold": run_gold,
    "predict": run_predict,
}


def main():
    parser = argparse.ArgumentParser(description="InsightAI Pipeline Runner")
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        default="full",
        help="Run a specific stage. Default is full end-to-end pipeline.",
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    ensure_dirs(config)

    logger.info("🚀 InsightAI Pipeline — Data Storm 7.0")
    logger.info(f"Team: {config['project']['team_name']}")

    STAGES[args.stage](config)

    logger.info("✅ Pipeline complete.")


if __name__ == "__main__":
    main()
