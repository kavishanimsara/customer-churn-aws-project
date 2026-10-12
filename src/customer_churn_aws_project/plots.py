"""EDA and model visualisations, saved as PNGs for the DVC pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_curve

from .features import get_encoded_feature_names
from .logging_config import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure: %s", path)
    return path


def plot_numerical_boxplots(df: pd.DataFrame, out_dir: Path) -> Path:
    """Distribution of the numerical columns (notebook 02, cell 11)."""
    numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, col in enumerate(numerical_columns):
        sns.boxplot(data=df, y=col, ax=axes[i], color="lightblue")
        axes[i].set_title(f"Distribution of {col}", fontsize=14, fontweight="bold")
        axes[i].set_ylabel("value")

    plt.tight_layout()
    fig.suptitle("Numerical Columns - Boxplots", fontsize=16, y=1.02)
    return _save(fig, out_dir, "00_numerical_boxplots.png")


def plot_churn_distribution(df: pd.DataFrame, out_dir: Path) -> tuple[Path, float]:
    """Overall churn distribution (notebook 02, Visual 1)."""
    churn_counts = df["Churn"].value_counts()
    overall_churn_pct = df["Churn_binary"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        churn_counts.index,
        churn_counts.values,
        color=["#2b5c8f", "#d95f02"],
        edgecolor="black",
        width=0.45,
    )
    ax.set_title(
        "Overall Customer Churn Distribution", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_ylabel("Number of Customers", fontsize=11)
    ax.set_ylim(0, 6000)
    for b in bars:
        h = b.get_height()
        pct = (h / len(df)) * 100
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + 80,
            f"{h:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    path = _save(fig, out_dir, "01_churn_distribution.png")
    return path, overall_churn_pct


def plot_churn_by_contract(df: pd.DataFrame, overall_churn_pct: float, out_dir: Path) -> Path:
    """Churn rate by contract type (Visual 2)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    contract_cr = df.groupby("Contract")["Churn_binary"].mean() * 100
    bars = ax.bar(
        contract_cr.index,
        contract_cr.values,
        color=["#d95f02", "#386cb0", "#7fc97f"],
        edgecolor="black",
    )
    ax.axhline(
        overall_churn_pct,
        color="red",
        linestyle="--",
        label=f"Overall Avg ({overall_churn_pct:.1f}%)",
    )
    ax.set_title(
        "Customer Churn Rate by Contract Type", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_ylabel("Churn Rate (%)", fontsize=11)
    ax.set_ylim(0, 52)
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0, h + 1.2, f"{h:.1f}%", ha="center", fontweight="bold"
        )
    ax.legend()
    plt.tight_layout()
    return _save(fig, out_dir, "02_churn_by_contract.png")


def plot_churn_by_tenure(df: pd.DataFrame, overall_churn_pct: float, out_dir: Path) -> Path:
    """Churn rate by tenure cohort (Visual 3)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    tenure_cr = df.groupby("Tenure_Group", observed=False)["Churn_binary"].mean() * 100
    bars = ax.bar(
        tenure_cr.index.astype(str),
        tenure_cr.values,
        color=sns.color_palette("Reds_r", len(tenure_cr)),
        edgecolor="black",
    )
    ax.axhline(
        overall_churn_pct,
        color="blue",
        linestyle="--",
        label=f"Overall Avg ({overall_churn_pct:.1f}%)",
    )
    ax.set_title(
        "Churn Rate by Customer Tenure Cohort", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_ylabel("Churn Rate (%)", fontsize=11)
    ax.set_ylim(0, 55)
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2.0, h + 1.2, f"{h:.1f}%", ha="center", fontweight="bold"
        )
    ax.legend()
    plt.tight_layout()
    return _save(fig, out_dir, "03_churn_by_tenure.png")


def plot_churn_by_spending_and_payment(
    df: pd.DataFrame, overall_churn_pct: float, out_dir: Path
) -> Path:
    """Visuals 4 & 5: monthly spending and payment method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    spend_cr = (
        df.groupby("MonthlyCharges_Group", observed=False)["Churn_binary"].mean() * 100
    )
    axes[0].bar(spend_cr.index.astype(str), spend_cr.values, color="#4575b4", edgecolor="black")
    axes[0].axhline(
        overall_churn_pct,
        color="red",
        linestyle="--",
        label=f"Overall Avg ({overall_churn_pct:.1f}%)",
    )
    axes[0].set_title("Churn Rate by Monthly Spending Tier", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)", fontsize=10)
    axes[0].set_ylim(0, 48)
    for i, v in enumerate(spend_cr.values):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[0].legend()

    pm_cr = df.groupby("PaymentMethod")["Churn_binary"].mean() * 100
    axes[1].bar(
        pm_cr.index,
        pm_cr.values,
        color=["#4575b4", "#4575b4", "#d73027", "#4575b4"],
        edgecolor="black",
    )
    axes[1].axhline(
        overall_churn_pct,
        color="black",
        linestyle="--",
        label=f"Overall Avg ({overall_churn_pct:.1f}%)",
    )
    axes[1].set_title("Churn Rate by Payment Method", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)", fontsize=10)
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].set_ylim(0, 52)
    for i, v in enumerate(pm_cr.values):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    return _save(fig, out_dir, "04_churn_by_spending_and_payment.png")


def plot_churn_by_internet_and_support(
    df: pd.DataFrame, overall_churn_pct: float, out_dir: Path
) -> Path:
    """Visual 6: internet service and tech support."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    is_cr = df.groupby("InternetService")["Churn_binary"].mean() * 100
    axes[0].bar(
        is_cr.index,
        is_cr.values,
        color=["#74add1", "#f46d43", "#a6d96a"],
        edgecolor="black",
    )
    axes[0].axhline(
        overall_churn_pct,
        color="red",
        linestyle="--",
        label=f"Overall Avg ({overall_churn_pct:.1f}%)",
    )
    axes[0].set_title("Churn Rate by Internet Service Type", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)", fontsize=10)
    axes[0].set_ylim(0, 50)
    for i, v in enumerate(is_cr.values):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[0].legend()

    ts_cr = df.groupby("TechSupport")["Churn_binary"].mean() * 100
    axes[1].bar(
        ts_cr.index,
        ts_cr.values,
        color=["#f46d43", "#a6d96a", "#74add1"],
        edgecolor="black",
    )
    axes[1].axhline(
        overall_churn_pct,
        color="red",
        linestyle="--",
        label=f"Overall Avg ({overall_churn_pct:.1f}%)",
    )
    axes[1].set_title("Churn Rate by Tech Support Status", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)", fontsize=10)
    axes[1].set_ylim(0, 50)
    for i, v in enumerate(ts_cr.values):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    return _save(fig, out_dir, "05_churn_by_internet_and_support.png")


def plot_correlation_and_tenure(df: pd.DataFrame, out_dir: Path) -> Path:
    """Visuals 7 & 8: correlation heatmap and tenure boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    corr = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn_binary"]].corr()
    sns.heatmap(
        corr, annot=True, cmap="coolwarm", fmt=".3f", ax=axes[0], cbar=True, vmin=-0.5, vmax=1.0
    )
    axes[0].set_title(
        "Correlation Matrix: Numerical Attributes & Churn", fontsize=12, fontweight="bold"
    )

    sns.boxplot(
        x="Churn",
        y="tenure",
        data=df,
        ax=axes[1],
        palette=["#386cb0", "#d95f02"],
        hue="Churn",
        legend=False,
    )
    axes[1].set_title("Tenure Distribution by Churn Status", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Tenure (Months)", fontsize=10)

    plt.tight_layout()
    return _save(fig, out_dir, "06_correlation_and_tenure.png")


def plot_eda(df: pd.DataFrame, out_dir: Path) -> dict[str, float | str]:
    """Generate every EDA figure. Returns summary stats."""
    paths = [plot_numerical_boxplots(df, out_dir)]
    churn_path, overall_churn_pct = plot_churn_distribution(df, out_dir)
    paths.append(churn_path)

    paths.append(plot_churn_by_contract(df, overall_churn_pct, out_dir))
    paths.append(plot_churn_by_tenure(df, overall_churn_pct, out_dir))
    paths.append(plot_churn_by_spending_and_payment(df, overall_churn_pct, out_dir))
    paths.append(plot_churn_by_internet_and_support(df, overall_churn_pct, out_dir))
    paths.append(plot_correlation_and_tenure(df, out_dir))

    logger.info(
        "Saved %d EDA figure(s) to %s", len(paths), out_dir
    )
    return {
        "overall_churn_pct": overall_churn_pct,
        "n_rows": len(df),
        "figures": [str(p) for p in paths],
    }


def plot_roc_curve(
    y_test: pd.Series, y_proba: np.ndarray, out_dir: Path
) -> tuple[Path, list[dict]]:
    """ROC curve for the selected model + records for ``dvc plots``."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2b5c8f", linewidth=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Best Model", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = _save(fig, out_dir, "07_roc_curve.png")

    records = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr)]
    return path, records


def plot_confusion_matrix(
    y_test: pd.Series, y_pred: np.ndarray, out_dir: Path
) -> tuple[Path, list[dict]]:
    """Confusion matrix heatmap + per-sample records for ``dvc plots``."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Best Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = _save(fig, out_dir, "08_confusion_matrix.png")

    records = [
        {"actual": int(a), "predicted": int(p)}
        for a, p in zip(y_test.to_numpy(), np.asarray(y_pred))
    ]
    return path, records


def plot_feature_importance(
    model: BaseEstimator,
    preprocessor: ColumnTransformer,
    num_features: list[str],
    cat_features: list[str],
    out_dir: Path,
    top_n: int = 20,
) -> Path:
    """Top feature importances (or absolute coefficients) of the best model."""
    clf = model.named_steps["clf"]
    names = get_encoded_feature_names(preprocessor, num_features, cat_features)

    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_)
    elif hasattr(clf, "coef_"):
        importances = np.abs(np.asarray(clf.coef_)).ravel()
    else:
        logger.warning(
            "Model '%s' exposes neither feature_importances_ nor coef_; "
            "skipping the feature-importance plot",
            type(clf).__name__,
        )
        return Path()

    series = pd.Series(importances, index=names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(series))))
    sns.barplot(x=series.values, y=series.index, ax=ax, color="#4575b4")
    ax.set_title(f"Top {len(series)} Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return _save(fig, out_dir, "09_feature_importance.png")
