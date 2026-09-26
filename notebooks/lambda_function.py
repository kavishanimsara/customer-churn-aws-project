
"""
lambda_function.py
-------------------
Triggered by S3 ObjectCreated events on the `raw/` prefix of the churn
data bucket. Validates the uploaded CSV and routes it:

    raw/<file>.csv
        -> valid   -> processed/<file>.csv   (copy of the original)
        -> invalid -> raw/rejected/<file>.csv (moved, with a .json report)

All validation steps and outcomes are written to CloudWatch Logs so the
group can demonstrate monitoring/troubleshooting for the assignment.

Environment variables expected on the Lambda function:
    PROCESSED_PREFIX   default "processed/"
    REJECTED_PREFIX    default "raw/rejected/"
    REQUIRED_COLUMNS   comma-separated list, e.g.
                        "customerID,tenure,MonthlyCharges,TotalCharges,Churn"
"""

import csv
import io
import json
import logging
import os
import urllib.parse
from datetime import UTC, datetime

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

PROCESSED_PREFIX = os.environ.get("PROCESSED_PREFIX", "processed/")
REJECTED_PREFIX = os.environ.get("REJECTED_PREFIX", "raw/rejected/")
REQUIRED_COLUMNS = [
    c.strip()
    for c in os.environ.get(
        "REQUIRED_COLUMNS",
        "customerID,tenure,MonthlyCharges,TotalCharges,Churn",
    ).split(",")
    if c.strip()
]

# Columns that must contain numeric values (used for the dtype check)
NUMERIC_COLUMNS = {"tenure", "MonthlyCharges", "TotalCharges"}


def validate_csv(body_bytes):
    """
    Runs basic data-quality checks on a CSV file's raw bytes.
    Returns (is_valid: bool, report: dict)
    """
    report = {
        "checked_at": datetime.now(UTC).isoformat(),
        "row_count": 0,
        "missing_columns": [],
        "missing_value_counts": {},
        "type_errors": {},
        "issues": [],
    }

    try:
        text = body_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        report["issues"].append(f"File is not valid UTF-8 text: {exc}")
        return False, report

    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    report["missing_columns"] = missing_columns
    if missing_columns:
        report["issues"].append(f"Missing required columns: {missing_columns}")

    missing_counts = {c: 0 for c in REQUIRED_COLUMNS if c in fieldnames}
    type_error_counts = {c: 0 for c in NUMERIC_COLUMNS if c in fieldnames}
    row_count = 0

    for row in reader:
        row_count += 1
        for col in missing_counts:
            value = (row.get(col) or "").strip()
            if value == "":
                missing_counts[col] += 1

        for col in type_error_counts:
            value = (row.get(col) or "").strip()
            if value != "":
                try:
                    float(value)
                except ValueError:
                    type_error_counts[col] += 1

    report["row_count"] = row_count
    report["missing_value_counts"] = missing_counts
    report["type_errors"] = type_error_counts

    if row_count == 0:
        report["issues"].append("File contains no data rows.")

    total_missing = sum(missing_counts.values())
    total_type_errors = sum(type_error_counts.values())
    if total_missing > 0:
        report["issues"].append(f"{total_missing} missing values across required columns.")
    if total_type_errors > 0:
        report["issues"].append(f"{total_type_errors} non-numeric values in numeric columns.")

    is_valid = not missing_columns and row_count > 0 and total_type_errors == 0
    return is_valid, report


def lambda_handler(event, context):
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        file_name = key.split("/")[-1]

        logger.info("Processing s3://%s/%s", bucket, key)

        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()

        is_valid, report = validate_csv(body)

        if is_valid:
            dest_key = f"{PROCESSED_PREFIX}{file_name}"
            s3.put_object(Bucket=bucket, Key=dest_key, Body=body)
            logger.info("VALID: copied to s3://%s/%s (%d rows)", bucket, dest_key, report["row_count"])
        else:
            dest_key = f"{REJECTED_PREFIX}{file_name}"
            report_key = f"{REJECTED_PREFIX}{file_name}.report.json"
            s3.put_object(Bucket=bucket, Key=dest_key, Body=body)
            s3.put_object(
                Bucket=bucket,
                Key=report_key,
                Body=json.dumps(report, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            logger.warning("INVALID: %s -> %s | issues=%s", key, dest_key, report["issues"])

        results.append({"source_key": key, "valid": is_valid, "destination": dest_key, "report": report})

    return {"statusCode": 200, "body": json.dumps(results, default=str)}

