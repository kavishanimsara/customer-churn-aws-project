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

    return df


def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y, customer_ids)`` from a flagged dataframe."""
    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]
    customer_ids = df["customerID"]

    assert not ({"Churn", TARGET_COL} & set(X.columns)), "Target column leaked into X"

    return X, y, customer_ids


def get_feature_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature names into numerical and categorical lists."""
    num_features = [col for col in NUM_FEATURES if col in X.columns]
    cat_features = [col for col in X.columns if col not in num_features]
    return num_features, cat_features


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    customer_ids: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Stratified train/test split, also returning the customer ids."""
    return train_test_split(
        X,
        y,
        customer_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(
    num_features: list[str], cat_features: list[str]
) -> ColumnTransformer:
    """Standard-scale numerics and one-hot encode categoricals."""
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
    return num_features + list(cat_encoder.get_feature_names_out(cat_features))
