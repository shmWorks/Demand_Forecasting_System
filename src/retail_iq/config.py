from pathlib import Path
import os
import random

import numpy as np

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PARQUET_DATA_DIR = DATA_DIR / "parquet"  # Fast Polars I/O — run scripts/convert_to_parquet.py once

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"

# Performance constants
SAMPLE_N_CORR: int = 50_000    # Max rows for correlation heatmap (statistically sufficient)
SAMPLE_N_DIST: int = 100_000   # Max rows for distribution plots

# Ensure directories exist
for _dir in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PARQUET_DATA_DIR,
             OUTPUT_DIR, PLOT_DIR, MODEL_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = 42) -> int:
    """Set global seeds for reproducibility across stdlib and NumPy."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed

