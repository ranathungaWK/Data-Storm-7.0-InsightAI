"""
Master Pipeline Orchestrator — Bronze → Silver → Gold → Predict

Usage:
    python run_pipeline.py                # Run full pipeline
    python run_pipeline.py --stage bronze # Run only bronze stage
"""

import argparse
import sys

from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger, setup_logging

logger = get_logger("pipeline")


def run_bronze(config: dict) -> None:
    """Stage 1: Raw ingestion — CSV → Parquet, zero transforms."""
    logger.info("=" * 60)
    logger.info("STAGE 1: BRONZE — Raw Ingestion")
    logger.info("=" * 60)
    from src.bronze.ingest_internal import ingest_all
    ingest_all(config)


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
        default=None,
        help="Run a specific stage. If omitted, runs all stages.",
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)
    ensure_dirs(config)

    logger.info("🚀 InsightAI Pipeline — Data Storm 7.0")
    logger.info(f"Team: {config['project']['team_name']}")

    if args.stage:
        STAGES[args.stage](config)
    else:
        for name, fn in STAGES.items():
            fn(config)

    logger.info("✅ Pipeline complete.")


if __name__ == "__main__":
    main()
