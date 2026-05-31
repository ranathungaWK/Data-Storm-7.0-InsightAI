"""
Copy of run_pipeline.py for report assets
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

# NOTE: This is a copy of the main pipeline file included for reproducibility in the report_assets folder.
