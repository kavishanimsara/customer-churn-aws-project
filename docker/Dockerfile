# Build from the repository root:
#   podman build -f docker/Dockerfile -t churn-dashboard .

# ---- dependency builder ----------------------------------------------------
FROM python:3.14-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app
# Pinned serving-only deps; the training-only libs are never installed here.
COPY docker/requirements-server.txt ./
RUN uv venv --python 3.14 /app/.venv && uv pip install --python /app/.venv -r requirements-server.txt

# ---- runtime ---------------------------------------------------------------
FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
COPY models/best_churn_model_bundle.pkl ./models/
COPY reports/metrics.json ./reports/
COPY reports/figures/ ./reports/figures/
COPY data/processed/cleaned_customer_data.csv ./data/processed/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "customer_churn_aws_project.dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
