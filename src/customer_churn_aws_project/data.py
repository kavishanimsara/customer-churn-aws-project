"""Data loading and cleaning (notebook 01 + the cleaning cells of notebook 02)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLEANED_DATA_PATH, RAW_DATA_PATH
from .logging_config import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw Telco customer churn CSV."""
    df = pd.read_csv(path)
    logger.info(
        "Loaded raw data from %s (%d rows, %d columns)",
        path,
        df.shape[0],
        df.shape[1],
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe exactly as done in notebook 02.

    * coerce ``MonthlyCharges`` / ``TotalCharges`` to numeric
    * turn whitespace-only strings into ``NaN``
    * drop rows containing missing values
    """
    rows_before = len(df)
    df = df.copy()

    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.replace(r"^\s*$", np.nan, regex=True)

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info(
            "Missing values detected: %s",
            ", ".join(f"{col}={int(n)}" for col, n in missing.items()),
        )

    df = df.dropna()
    rows_after = len(df)
    logger.info(
        "Dropped %d row(s) with missing values (%d -> %d rows)",
        rows_before - rows_after,
        rows_before,
        rows_after,
    )

    return df.reset_index(drop=True)


def save_data(df: pd.DataFrame, path: str | Path = CLEANED_DATA_PATH) -> Path:
    """Persist a cleaned dataframe to CSV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(
        "Saved cleaned dataset to %s (%d rows, %d columns)",
        path,
        df.shape[0],
        df.shape[1],
    )
    return path


def load_cleaned_data(path: str | Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset produced by the preprocess stage."""
    df = pd.read_csv(path)
    logger.info(
        "Loaded cleaned dataset from %s (%d rows, %d columns)",
        path,
        df.shape[0],
        df.shape[1],
    )
    return df

