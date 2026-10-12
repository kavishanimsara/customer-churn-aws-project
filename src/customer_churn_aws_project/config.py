"""Project-wide configuration: paths, constants and feature definitions.

Everything here is derived from notebooks ``01-preprocess.ipynb`` and
``02-customer_churn_pipeline.ipynb``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved from the repository root, not the current working dir)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_customer_data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EDA_FIGURES_DIR = FIGURES_DIR / "eda"
MODEL_FIGURES_DIR = FIGURES_DIR / "model"

MODEL_BUNDLE_PATH = MODELS_DIR / "best_churn_model_bundle.pkl"

METRICS_PATH = REPORTS_DIR / "metrics.json"
ROC_CURVE_PATH = REPORTS_DIR / "roc_curve.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.json"

PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# ---------------------------------------------------------------------------
# Feature / target definitions
# ---------------------------------------------------------------------------
TARGET_COL = "Churn_binary"
DROP_COLS = ["customerID", "Churn", TARGET_COL, "Tenure_Group", "MonthlyCharges_Group"]

NUM_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

TENURE_BINS = [-1, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["0-12 mo", "13-24 mo", "25-36 mo", "37-48 mo", "49-60 mo", "61-72 mo"]

MONTHLY_BINS = [0, 30, 60, 90, 120, 150]
MONTHLY_LABELS = [
    "Very Low (<$30)",
    "Low ($30-$60)",
    "Medium ($60-$90)",
    "High (>$90)",
    "Very High (>$120)",
]
