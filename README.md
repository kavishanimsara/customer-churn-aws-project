# AWS-Based Customer Churn Prediction & Analytics Solution

## 📌 Project Overview
This project implements a cloud-based Data Science solution on AWS to help a telecommunications company understand customer churn. The solution ingests raw customer data into Amazon S3, automatically validates and processes new uploads via AWS Lambda, performs Exploratory Data Analysis (EDA), trains and compares two machine learning classification models, runs SQL-based analytical queries using Amazon Athena, and delivers actionable business recommendations.

## 👥 Team Members

| Name | Student ID | Role |
|------|-----------|------|
| Supun Gimres | 22CDS0421 | Cloud Infrastructure & Automation  |
| Sachindu Teekshana | 22CDS0409 | Data Science & Machine Learning  |
| Kavisha Nimsara | 22CDS0425 | Data Analytics & Business Insights  |
| Savindi Hewage | 22CDS0418 | Project Manager, Documentation & Integration |

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


## 🛠️ AWS Services Used

| Service | Purpose | Free Tier Eligible |
|---------|---------|-------------------|
| Amazon S3 | Store raw, processed, and output data | ✅ Yes (5 GB) |
| AWS Lambda | Auto-detect and validate new uploads | ✅ Yes (1M requests/month) |
| Amazon EventBridge | Trigger Lambda on S3 upload events | ✅ Yes |
| Amazon CloudWatch | Monitor logs and troubleshoot workflows | ✅ Yes (5 GB logs) |
| Amazon Athena | Run SQL queries on processed S3 data | ✅ Pay-per-query (~$0.005/GB) |
| Amazon SageMaker | Train and evaluate ML models | ✅ Yes (2 months free, ml.t3.medium) |
| AWS IAM | Manage access control and security | ✅ Yes (Free) |
| AWS Budgets | Monitor costs and set billing alerts | ✅ Yes |

## 🚀 How to Run
- Install uv 
- uv sync
- uv tool install dvc

### Prerequisites
- An AWS account with Free Tier access
- Python 3.9+
- AWS CLI configured (`aws configure`)

### Step 1: Set Up Cloud Infrastructure
1. Create three S3 buckets (or folders within one bucket): `raw/`, `processed/`, `outputs/`.
2. Deploy the Lambda function from `cloud-infrastructure/lambda/data_validator.py`.
3. Configure the EventBridge rule from `cloud-infrastructure/eventbridge/`.
4. Attach the IAM policies from `cloud-infrastructure/iam/`.

### Step 2: Run Data Science Notebooks
```bash
cd data-science
pip install -r requirements.txt
jupyter notebook
