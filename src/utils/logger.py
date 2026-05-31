"""
Structured Logger — Consistent logging across the entire pipeline.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger("bronze.ingest")
    logger.info("Ingesting transactions_history_final.csv")
"""

import logging
import sys
from pathlib import Path

from src.utils.config import PROJECT_ROOT, load_config


_initialized = False


def setup_logging(config: dict | None = None) -> None:
    """
    Initialize the root logger with console + file handlers.
    Called once at pipeline start.

    Parameters
    ----------
    config : dict, optional
        Pipeline config. Loaded automatically if not provided.
    """
    global _initialized
    if _initialized:
        return

    if config is None:
        config = load_config()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))
    fmt = log_cfg.get(
        "format",
        "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
    )
    log_file = log_cfg.get("log_file", "logs/pipeline.log")

    # Ensure log directory exists
    log_path = PROJECT_ROOT / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Formatter
    formatter = logging.Formatter(fmt)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Initializes logging if not already done.

    Parameters
    ----------
    name : str
        Logger name (e.g., "bronze.ingest", "silver.dq_checks").

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    setup_logging()
    return logging.getLogger(name)
