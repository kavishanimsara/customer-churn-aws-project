"""FastAPI routes for the churn dashboard."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import service

router = APIRouter()


class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/model/info")
def model_info() -> dict[str, Any]:
    return service.get_model_info()


@router.get("/api/model/options")
def model_options() -> dict[str, list[Any]]:
    return service.load_feature_options()


@router.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return service.load_metrics()


@router.get("/api/figures")
def figures() -> dict[str, list[str]]:
    return service.list_figures()


@router.get("/figures/{category}/{name}")
def figure(category: str, name: str) -> FileResponse:
    try:
        return FileResponse(service.figure_path(category, name), media_type="image/png")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Figure not found")


@router.post("/api/predict")
def predict(features: CustomerFeatures) -> dict[str, Any]:
    try:
        return service.predict_one(features.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
