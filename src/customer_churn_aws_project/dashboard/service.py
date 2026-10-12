"""Shared dashboard logic: model loading, feature schema and prediction."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .. import config
from ..logging_config import get_logger

logger = get_logger(__name__)

DECISION_THRESHOLD = 0.5
FIGURE_CATEGORIES = {
    "eda": config.EDA_FIGURES_DIR,
    "model": config.MODEL_FIGURES_DIR,
}


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    """Load the best-model bundle written by the ``train`` stage."""
    logger.info("Loading model bundle from %s", config.MODEL_BUNDLE_PATH)
    return joblib.load(config.MODEL_BUNDLE_PATH)


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    with open(config.METRICS_PATH) as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_feature_options() -> dict[str, list[Any]]:
    """Distinct values per categorical feature, taken from the cleaned dataset."""
    bundle = load_bundle()
    df = pd.read_csv(config.CLEANED_DATA_PATH)
    return {
        col: sorted(df[col].dropna().unique().tolist())
        for col in bundle["categorical_features"]
    }


def get_model_info() -> dict[str, Any]:
    bundle = load_bundle()
    metrics = load_metrics()
    return {
        "model_name": bundle["model_name"],
        "trained_at": bundle["trained_at"],
        "primary_metric": metrics["primary_metric"],
        "numeric_features": bundle["numeric_features"],
        "categorical_features": bundle["categorical_features"],
        "test_metrics": metrics["test"],
    }


def predict_one(payload: dict[str, Any]) -> dict[str, Any]:
    """Score a single customer. ``payload`` must hold every trained feature."""
    bundle = load_bundle()
    numeric = bundle["numeric_features"]
    categorical = bundle["categorical_features"]
    options = load_feature_options()

    missing = [col for col in numeric + categorical if col not in payload]
    if missing:
        raise ValueError(f"Missing feature(s): {', '.join(missing)}")

    for col in categorical:
        if payload[col] not in options[col]:
            raise ValueError(
                f"Invalid value {payload[col]!r} for {col}; "
                f"expected one of {options[col]}"
            )

    row = {col: float(payload[col]) for col in numeric}
    row.update({col: payload[col] for col in categorical})
    frame = pd.DataFrame([row], columns=numeric + categorical)

    probability = float(bundle["model"].predict_proba(frame)[0, 1])
    prediction = "Yes" if probability >= DECISION_THRESHOLD else "No"

    logger.debug("Scored customer: p(churn)=%.4f -> %s", probability, prediction)
    return {
        "model_name": bundle["model_name"],
        "churn_probability": round(probability, 4),
        "prediction": prediction,
        "threshold": DECISION_THRESHOLD,
    }


def list_figures() -> dict[str, list[str]]:
    return {
        category: sorted(p.name for p in directory.glob("*.png"))
        for category, directory in FIGURE_CATEGORIES.items()
    }


def figure_path(category: str, name: str) -> Path:
    """Resolve a figure file, refusing anything outside the figure directories."""
    directory = FIGURE_CATEGORIES.get(category)
    if directory is None:
        raise FileNotFoundError(category)
    path = (directory / name).resolve()
    if path.parent != directory.resolve() or not path.is_file():
        raise FileNotFoundError(name)
    return path
