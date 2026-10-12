"""Gradio Blocks UI for the churn dashboard."""

from __future__ import annotations

from typing import Any

import gradio as gr
import pandas as pd

from . import service

NUMERIC_DEFAULTS = {"tenure": 12, "MonthlyCharges": 70.0, "TotalCharges": 1000.0}


def _label(col: str) -> str:
    return col.replace("_", " ")


def _predict_form(*values: Any) -> tuple[str, str, dict[str, Any]]:
    options = service.load_feature_options()
    bundle = service.load_bundle()
    names = bundle["numeric_features"] + bundle["categorical_features"]

    payload: dict[str, Any] = {}
    for name, value in zip(names, values):
        if options.get(name):
            payload[name] = type(options[name][0])(value)
        else:
            payload[name] = float(value)

    try:
        result = service.predict_one(payload)
    except ValueError as exc:
        raise gr.Error(str(exc))

    probability = f"{result['churn_probability']:.1%}"
    verdict = "Likely to churn" if result["prediction"] == "Yes" else "Likely to stay"
    return probability, verdict, result


def _model_summary() -> str:
    info = service.get_model_info()
    test = info["test_metrics"]
    lines = [
        f"**Best model:** {info['model_name']}",
        f"**Selected by:** {info['primary_metric']}",
        f"**Trained at:** {info['trained_at']}",
        "",
        "| Accuracy | Precision | Recall | F1-Score | ROC-AUC |",
        "|---|---|---|---|---|",
        "| "
        + " | ".join(
            f"{test[key]:.3f}"
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
        )
        + " |",
    ]
    return "\n".join(lines)


def _comparison_table() -> pd.DataFrame:
    models = service.load_metrics()["models"]
    frame = pd.DataFrame(models).T.reset_index().rename(columns={"index": "Model"})
    frame.columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ]
    return frame.sort_values("F1-Score", ascending=False).round(4)


def _figure_html() -> str:
    sections = []
    for category, names in service.list_figures().items():
        images = "".join(
            f'<figure><img src="/figures/{category}/{name}" loading="lazy" '
            f'style="width:100%"><figcaption>{name}</figcaption></figure>'
            for name in names
        )
        sections.append(
            f"<h3>{category.upper()} figures</h3>"
            f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px">'
            f"{images}</div>"
        )
    return "".join(sections)


def build_ui() -> gr.Blocks:
    options = service.load_feature_options()
    bundle = service.load_bundle()

    with gr.Blocks(title="Customer Churn Dashboard") as demo:
        gr.Markdown("# Customer Churn Dashboard")

        with gr.Tab("Predict"):
            gr.Markdown("Enter a customer's details to estimate churn probability.")
            inputs: list[gr.components.Component] = []
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Account & charges")
                    for col in bundle["numeric_features"]:
                        inputs.append(
                            gr.Number(
                                label=_label(col),
                                value=NUMERIC_DEFAULTS[col],
                            )
                        )
                with gr.Column():
                    gr.Markdown("### Services & contract")
                    for col in bundle["categorical_features"][:8]:
                        inputs.append(
                            gr.Dropdown(
                                label=_label(col),
                                choices=options[col],
                                value=options[col][0],
                            )
                        )
                with gr.Column():
                    gr.Markdown("&nbsp;")
                    for col in bundle["categorical_features"][8:]:
                        inputs.append(
                            gr.Dropdown(
                                label=_label(col),
                                choices=options[col],
                                value=options[col][0],
                            )
                        )
            predict_btn = gr.Button("Predict churn", variant="primary")
            with gr.Row():
                probability_out = gr.Textbox(label="Churn probability", interactive=False)
                verdict_out = gr.Textbox(label="Prediction", interactive=False)
            details_out = gr.JSON(label="Details")

            predict_btn.click(
                _predict_form,
                inputs=inputs,
                outputs=[probability_out, verdict_out, details_out],
            )

        with gr.Tab("Model performance"):
            gr.Markdown(_model_summary())
            gr.Markdown("### All candidate models (test set)")
            gr.Dataframe(value=_comparison_table(), interactive=False)

        with gr.Tab("Figures"):
            gr.HTML(_figure_html())

    return demo
