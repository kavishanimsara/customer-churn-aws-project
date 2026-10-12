"""Feature engineering and preprocessing (EDA binning + model input pipeline)."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    DROP_COLS,
    MONTHLY_BINS,
    MONTHLY_LABELS,
    NUM_FEATURES,
    TARGET_COL,
    TENURE_BINS,
    TENURE_LABELS,
)
from .logging_config import get_logger

logger = get_logger(__name__)


def add_churn_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add the binary target and the EDA segmentation bins."""
    df = df.copy()

    df[TARGET_COL] = (df["Churn"] == "Yes").astype(int)
    df["Tenure_Group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    )
    df["MonthlyCharges_Group"] = pd.cut(
        df["MonthlyCharges"], bins=MONTHLY_BINS, labels=MONTHLY_LABELS
    )

    logger.debug(
        "Added churn flag and EDA bins; overall churn rate = %.2f%%",
        df[TARGET_COL].mean() * 100,
    )
    return df


def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y, customer_ids)`` from a flagged dataframe."""
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]
    customer_ids = df["customerID"]

    assert not ({"Churn", TARGET_COL} & set(X.columns)), "Target column leaked into X"

    logger.info(
        "Built feature matrix X=%s and target y=%s", X.shape, (len(y),)
    )
    return X, y, customer_ids


def get_feature_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature names into numerical and categorical lists."""
    num_features = [col for col in NUM_FEATURES if col in X.columns]
    cat_features = [col for col in X.columns if col not in num_features]
    logger.debug("Numerical features: %s", num_features)
    logger.debug("Categorical features: %s", cat_features)
    return num_features, cat_features


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    customer_ids: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Stratified train/test split, also returning the customer ids."""
    result = train_test_split(
        X,
        y,
        customer_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_test, y_train, y_test, _, _ = result
    logger.info(
        "Split data (test_size=%.2f, random_state=%d): train=%d rows, test=%d rows",
        test_size,
        random_state,
        len(X_train),
        len(X_test),
    )
    logger.debug(
        "Train churn rate = %.2f%%, test churn rate = %.2f%%",
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    return result


def build_preprocessor(
    num_features: list[str], cat_features: list[str]
) -> ColumnTransformer:
    """Standard-scale numerics and one-hot encode categoricals."""
    logger.debug(
        "Built preprocessor with %d numerical and %d categorical features",
        len(num_features),
        len(cat_features),
    )
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                cat_features,
            ),
        ]
    )


def get_encoded_feature_names(
    preprocessor: ColumnTransformer, num_features: list[str], cat_features: list[str]
) -> list[str]:
    """Human-readable feature names after the column transformer runs."""
    cat_encoder = preprocessor.named_transformers_["cat"]
    names = num_features + list(cat_encoder.get_feature_names_out(cat_features))
    logger.debug("Encoded feature count: %d", len(names))
    return names

