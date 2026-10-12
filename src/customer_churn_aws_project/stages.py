"""DVC pipeline stages: ``preprocess``, ``eda`` and ``train``.

Each stage reads ``params.yaml`` and writes the outputs declared in
``dvc.yaml``. Run a single stage with::

    uv run python -m customer_churn_aws_project.stages <stage>

Logs are written to stdout in a human-readable format (see
``logging_config``). Control verbosity with ``--log-level`` or the
``CHURN_LOG_LEVEL`` environment variable.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from . import config
from .data import clean_data, load_cleaned_data, load_raw_data, save_data
from .features import (
    add_churn_flags,
    build_features,
    build_preprocessor,
    get_encoded_feature_names,
    get_feature_lists,
    split_data,
)
from .logging_config import LOGGER_NAME, configure_logging, get_logger
from .models import (
    evaluate_models,
    get_candidate_models,
    save_model,
    save_model_bundle,
    select_best,
    train_models,
)
from .plots import (
    plot_confusion_matrix,
    plot_eda,
    plot_feature_importance,
    plot_roc_curve,
)

logger = get_logger(f"{LOGGER_NAME}.stages")



def load_params(path: str | Path = config.PARAMS_PATH) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _resolve(path_like: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    path = Path(path_like)
    return path if path.is_absolute() else config.PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def preprocess(params: dict[str, Any]) -> None:
    raw_path = _resolve(params["preprocess"].get("raw_data", config.RAW_DATA_PATH))
    out_path = _resolve(params["preprocess"].get("output", config.CLEANED_DATA_PATH))

    logger.info("Preprocessing %s -> %s", raw_path, out_path)
    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)
    save_data(df_clean, out_path)


def eda(params: dict[str, Any]) -> None:
    logger.info("Generating EDA figures into %s", config.EDA_FIGURES_DIR)
    df = add_churn_flags(load_cleaned_data())
    summary = plot_eda(df, config.EDA_FIGURES_DIR)
    logger.info(
        "Overall churn rate: %.2f%% (%d customers)",
        summary["overall_churn_pct"],
        summary["n_rows"],
    )


def train(params: dict[str, Any]) -> None:
    train_params = params["train"]
    random_state = int(params.get("random_state", 42))
    test_size = float(params["features"]["test_size"])
    primary_metric = train_params["primary_metric"]
    secondary_metric = train_params["secondary_metric"]

    logger.info(
        "Training config: test_size=%.2f random_state=%d primary_metric=%s "
        "secondary_metric=%s",
        test_size,
        random_state,
        primary_metric,
        secondary_metric,
    )

    df = add_churn_flags(load_cleaned_data())
    X, y, customer_ids = build_features(df)
    num_features, cat_features = get_feature_lists(X)
    logger.info(
        "Using %d numerical and %d categorical feature(s)",
        len(num_features),
        len(cat_features),
    )

    X_train, X_test, y_train, y_test, _, _ = split_data(
        X, y, customer_ids, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(num_features, cat_features)
    preprocessor.fit(X_train)
    encoded_names = get_encoded_feature_names(
        preprocessor, num_features, cat_features
    )
    logger.info("Total encoded features: %d", len(encoded_names))

    candidates = get_candidate_models(
        build_preprocessor(num_features, cat_features),
        train_params["models"],
        random_state=random_state,
    )
    train_models(candidates, X_train, y_train)
    results = evaluate_models(candidates, X_test, y_test)

    ranked = results.sort_values(
        by=[primary_metric, secondary_metric], ascending=[False, False]
    )
    logger.info(
        "Model comparison (ranked by %s, then %s):\n%s",
        primary_metric,
        secondary_metric,
        ranked.round(4).to_string(),
    )

    best_name, best_metrics = select_best(results, primary_metric, secondary_metric)
    best_model = candidates[best_name]

    # Persist model + bundle
    model_path = save_model(best_model, config.MODELS_DIR / "best_churn_model.pkl")
    bundle_path = save_model_bundle(
        {
            "model": best_model,
            "model_name": best_name,
            "metrics": best_metrics.to_dict(),
            "numeric_features": num_features,
            "categorical_features": cat_features,
            "trained_at": pd.Timestamp.now().isoformat(),
        },
        config.MODELS_DIR / "best_churn_model_bundle.pkl",
    )

    # Model plots + DVC plot data
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    _, roc_records = plot_roc_curve(y_test, y_proba, config.MODEL_FIGURES_DIR)
    _, cm_records = plot_confusion_matrix(y_test, y_pred, config.MODEL_FIGURES_DIR)
    plot_feature_importance(
        best_model,
        best_model.named_steps["prep"],
        num_features,
        cat_features,
        config.MODEL_FIGURES_DIR,
    )

    config.ROC_CURVE_PATH.write_text(json.dumps(roc_records, indent=2))
    config.CONFUSION_MATRIX_PATH.write_text(json.dumps(cm_records))

    # Metrics
    metrics = {
        "best_model": best_name,
        "primary_metric": primary_metric,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "test": {
            "accuracy": float(best_metrics["Accuracy"]),
            "precision": float(best_metrics["Precision"]),
            "recall": float(best_metrics["Recall"]),
            "f1": float(best_metrics["F1-Score"]),
            "roc_auc": float(best_metrics["ROC-AUC"]),
        },
        "models": {
            name: {
                "accuracy": float(row["Accuracy"]),
                "precision": float(row["Precision"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1-Score"]),
                "roc_auc": float(row["ROC-AUC"]),
            }
            for name, row in results.iterrows()
        },
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    logger.info("Best model: %s", best_name)
    logger.info(
        "Test metrics: %s",
        ", ".join(f"{name}={value:.4f}" for name, value in best_metrics.items()),
    )
    logger.info("Saved model to %s", model_path)
    logger.info("Saved model bundle to %s", bundle_path)
    logger.info("Saved metrics to %s", config.METRICS_PATH)
    logger.info("Saved model figures to %s", config.MODEL_FIGURES_DIR)


STAGES = {"preprocess": preprocess, "eda": eda, "train": train}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Customer churn DVC pipeline stages")
    parser.add_argument("stage", choices=sorted(STAGES), help="stage to run")
    parser.add_argument(
        "--params",
        default=str(config.PARAMS_PATH),
        help="path to params.yaml",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="console log verbosity (default: INFO, or $CHURN_LOG_LEVEL)",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    logger.info("=== Stage '%s' started ===", args.stage)
    start = time.perf_counter()
    params = load_params(args.params)
    try:
        STAGES[args.stage](params)
    except Exception:
        logger.exception("=== Stage '%s' failed ===", args.stage)
        raise
    logger.info(
        "=== Stage '%s' finished in %.2fs ===", args.stage, time.perf_counter() - start
    )


if __name__ == "__main__":
    main()
