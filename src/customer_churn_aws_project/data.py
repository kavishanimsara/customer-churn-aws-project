"""Data loading and cleaning (notebook 01 + the cleaning cells of notebook 02)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLEANED_DATA_PATH, RAW_DATA_PATH


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw Telco customer churn CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe exactly as done in notebook 02.

    * coerce ``MonthlyCharges`` / ``TotalCharges`` to numeric
    * turn whitespace-only strings into ``NaN``
    * drop rows containing missing values
    """
    df = df.copy()

    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna()

    return df.reset_index(drop=True)


def save_data(df: pd.DataFrame, path: str | Path = CLEANED_DATA_PATH) -> Path:
    """Persist a cleaned dataframe to CSV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_cleaned_data(path: str | Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset produced by the preprocess stage."""
    return pd.read_csv(path)
