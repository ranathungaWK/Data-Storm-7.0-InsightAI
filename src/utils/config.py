"""
Configuration Loader — Single source of truth for all pipeline parameters.

Usage:
    from src.utils.config import load_config
    cfg = load_config()
    bronze_path = cfg["paths"]["bronze"]["root"]
"""

import os
from pathlib import Path
import yaml


# Project root = two levels up from this file (src/utils/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline_config.yaml"


def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load the pipeline configuration YAML.

    Parameters
    ----------
    config_path : str or Path, optional
        Override path to config file. Defaults to config/pipeline_config.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(config_path) if config_path else CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. "
            f"Expected at: {CONFIG_PATH}"
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a relative path from config against the project root.

    Parameters
    ----------
    relative_path : str
        Path relative to project root (e.g., "data/bronze/transactions.parquet").

    Returns
    -------
    Path
        Absolute resolved path.
    """
    return PROJECT_ROOT / relative_path


def ensure_dirs(config: dict) -> None:
    """
    Create all required directories from the config if they don't exist.

    Parameters
    ----------
    config : dict
        Loaded pipeline config.
    """
    paths = config.get("paths", {})
    dirs_to_create = [
        paths.get("bronze", {}).get("root", "data/bronze"),
        paths.get("bronze", {}).get("poi_raw", "data/bronze/poi_raw"),
        paths.get("silver", {}).get("root", paths.get("silver", {}).get("cleaned_dir", "data/silver/cleaned")),
        paths.get("silver", {}).get("rejected_dir", "data/silver/rejected"),
        paths.get("gold", {}).get("root", "data/gold"),
        paths.get("output", {}).get("report", "output/report"),
        "logs",
        "ai_log/prompt_archive",
        "experiments",
    ]

    for d in dirs_to_create:
        full_path = PROJECT_ROOT / d
        full_path.mkdir(parents=True, exist_ok=True)
