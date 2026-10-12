# AWS-Based Customer Churn Prediction & Analytics Solution

## 📌 Project Overview
This project implements a cloud-based Data Science solution on AWS to help a telecommunications company understand customer churn. The solution ingests raw customer data into Amazon S3, automatically validates and processes new uploads via AWS Lambda, performs Exploratory Data Analysis (EDA), trains and compares two machine learning classification models, runs SQL-based analytical queries using Amazon Athena, and delivers actionable business recommendations.

## 👥 Team Members

| Name | Role |
|------|------|
| Supun Gimres | Cloud Infrastructure & Automation  |
| Sachindu Teekshana | Data Science & Machine Learning  |
| Kavisha Nimsara | Data Analytics & Business Insights  |
| Savindi Hewage | Project Manager, Documentation & Integration |

## 🏗️ AWS Architecture

![AWS Architecture Diagram](./architecture/architecture-diagram.png)

**Data Flow:**
1. Raw customer data is uploaded to the **Amazon S3** `raw/` bucket.
2. An **Amazon EventBridge** rule detects the upload and triggers an **AWS Lambda** function.
3. Lambda performs basic data validation (file format, column checks) and logs results to **Amazon CloudWatch**.
4. Validated data is moved to the S3 `processed/` bucket.
5. **Amazon Athena** queries the processed data for SQL-based analytics.
6. **Amazon SageMaker** (or local Jupyter) is used for EDA, preprocessing, and ML model training.
7. Final outputs (model metrics, visualizations, predictions) are stored in the S3 `outputs/` bucket.

## 📁 Repository Structure
```text
customer-churn-aws-project/
├── README.md
├── architecture/
│   ├── architecture-diagram.png
│   └── architecture-diagram.drawio
├── cloud-infrastructure/
│   ├── lambda/
│   │   └── data_validator.py
│   ├── iam/
│   │   └── policies.json
│   └── eventbridge/
│       └── s3-trigger-rule.json
├── data-science/
│   ├── 01_eda_and_visualizations.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training_and_evaluation.ipynb
│   └── requirements.txt
├── sql-queries/
│   ├── 01_churn_rate_by_contract.sql
│   ├── 02_churn_by_payment_method.sql
│   ├── 03_avg_tenure_churned_vs_retained.sql
│   ├── 04_monthly_charges_analysis.sql
│   └── 05_customer_segment_lifetime_value.sql
├── reports/
│   ├── Technical_Report.pdf
│   └── Individual_Contribution_Table.pdf
└── .gitignore
```


## 🚀 How to Run

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible Python dependency management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

*(Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`)*

Verify the install:
```bash
uv --version
```

### 2. Install dependencies

```bash
uv sync
```

This creates a `.venv` and installs everything from `pyproject.toml` / `uv.lock`, including dev dependencies. Activate it with:
```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install DVC (optional)

Only needed if you're pulling/pushing the versioned datasets. Installed as a standalone tool so it doesn't pollute the project's virtual environment.

```bash
uv tool install "dvc[s3]"
```

Then pull the tracked data:
```bash
dvc pull
```

## Dashboard

A FastAPI server with a Gradio UI mounted at `/`. It serves the best model saved by the `train` stage (`models/best_churn_model_bundle.pkl`), so run the pipeline (or `dvc pull`) first.

```bash
uv sync --extra dashboard
uv run churn-dashboard          # http://localhost:8000
```

- UI: `http://localhost:8000/` (Predict, Model performance, Figures tabs)
- API docs: `http://localhost:8000/docs`
- Endpoints: `GET /health`, `GET /api/model/info`, `GET /api/model/options`, `GET /api/metrics`, `GET /api/figures`, `GET /figures/{category}/{name}`, `POST /api/predict`

## Docker

The image contains only what the dashboard needs at runtime. Build from the repository root:

```bash
docker build -f docker/Dockerfile -t churn-dashboard .
docker run --rm -p 8000:8000 churn-dashboard
curl -s localhost:8000/health
```

The image installs only the serving dependencies from `docker/requirements-server.txt` (pinned, compiled from `docker/requirements-server.in`), so the training-only libraries such as xgboost, matplotlib, seaborn and boto3 are not included. Local development still uses the full environment from `uv sync --extra dashboard`.

To update the image's pins after changing `docker/requirements-server.in`:

```bash
uv pip compile docker/requirements-server.in --python-version 3.14 --no-header -o docker/requirements-server.txt
```

The image is built from the bundled model. If the best model is ever retrained to XGBoost, add `xgboost` to `docker/requirements-server.in` and regenerate the lock.

The build context is limited by `docker/Dockerfile.dockerignore` (BuildKit picks it up automatically). Podman users should add `--format docker` so the `HEALTHCHECK` is kept.
