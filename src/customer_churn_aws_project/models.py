"""Candidate model registry, training, evaluation and persistence."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .logging_config import get_logger

logger = get_logger(__name__)


def _build_estimator(
    kind: str, cfg: dict[str, Any], random_state: int
) -> BaseEstimator:
    """Instantiate a scikit-learn compatible estimator from its config."""
    if kind == "logistic_regression":
        return LogisticRegression(random_state=random_state, **cfg)
    if kind == "random_forest":
        return RandomForestClassifier(random_state=random_state, **cfg)
    if kind == "xgboost":
        return XGBClassifier(random_state=random_state, **cfg)
    if kind == "svm":
        return SVC(random_state=random_state, **cfg)
    if kind == "knn":
        return KNeighborsClassifier(**cfg)
    if kind == "mlp":
        return MLPClassifier(random_state=random_state, **cfg)
    if kind == "catboost":
        from catboost import CatBoostClassifier  # optional dependency

        return CatBoostClassifier(random_seed=random_state, **cfg)

    raise ValueError(f"Unknown estimator kind: {kind!r}")


def get_candidate_models(
    preprocessor: ColumnTransformer,
    models_config: dict[str, dict[str, Any]],
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Build a ``{display_name: Pipeline}`` registry from the params config.

    Models flagged with ``optional: true`` are skipped when their library is
    not installed (e.g. CatBoost on Python 3.14).
    """
    candidates: dict[str, Pipeline] = {}

    for name, raw_cfg in models_config.items():
        cfg = dict(raw_cfg)
        kind = cfg.pop("estimator")
        optional = cfg.pop("optional", False)

        try:
            estimator = _build_estimator(kind, cfg, random_state)
        except ImportError:
            if optional:
                logger.info(
                    "Skipping optional model '%s' (dependency not installed)", name
                )
                continue
            raise

        candidates[name] = Pipeline([("prep", preprocessor), ("clf", estimator)])

    logger.info("Prepared %d candidate model(s)", len(candidates))
    logger.debug("Candidate models: %s", list(candidates))
    return candidates


def train_models(
    candidates: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, Pipeline]:
    """Fit every candidate pipeline in place and return them."""
    for name, pipe in candidates.items():
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        logger.info(
            "Fitted %-35s in %5.2fs", name, time.perf_counter() - start
        )
    return candidates


def evaluate_models(
    candidates: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Evaluate every fitted pipeline on the test set."""
    results = {}
    for name, pipe in candidates.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        row = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        }
        results[name] = row
        logger.info(
            "Evaluated %-35s accuracy=%.4f f1=%.4f roc_auc=%.4f",
            name,
            row["Accuracy"],
            row["F1-Score"],
            row["ROC-AUC"],
        )
    return pd.DataFrame(results).T


def select_best(
    results: pd.DataFrame, primary_metric: str, secondary_metric: str
) -> tuple[str, pd.Series]:
    """Rank by ``primary_metric`` then ``secondary_metric`` (both descending)."""
    ranked = results.sort_values(
        by=[primary_metric, secondary_metric], ascending=[False, False]
    )
    best_name = ranked.index[0]
    logger.debug(
        "Ranked by ['%s', '%s']; winner is '%s'",
        primary_metric,
        secondary_metric,
        best_name,
    )
    return best_name, results.loc[best_name]


def save_model(
    model: BaseEstimator, path: str | Path
) -> Path:
    """Persist a fitted model with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.debug("Persisted model to %s", path)
    return path


def save_model_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    """Persist a model + metadata bundle with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.debug("Persisted model bundle to %s", path)
    return path

