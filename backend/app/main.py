"""FastAPI application for TrafficSense.

Endpoints
---------
GET  /               service information
GET  /api/health     service status + whether a trained model is present
GET  /api/metrics    real saved evaluation metrics and model metadata
POST /api/predict    hourly traffic-volume + congestion-category prediction
POST /api/routes     graph-based route analysis (OSMnx/NetworkX)

The app never fabricates results: when the trained model is missing, the
prediction endpoint returns 503 with a setup message instead of guessing.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .model_store import artifact_paths, load_metadata, load_metrics, load_model_bundle
from .preprocessing import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    classify_volume,
)
from .routing import (
    GraphCache,
    GraphUnavailableError,
    RouteNotFoundError,
    RoutingError,
    build_scenario,
    plan_routes,
    study_area_from_env,
)
from .schemas import (
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
    RouteRequest,
    RouteResponse,
    ServiceInfo,
    Thresholds,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def allowed_origins() -> list[str]:
    """Resolve CORS origins from the environment (never a wildcard by default)."""
    raw = os.getenv("TRAFFICSENSE_CORS_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_DEV_ORIGINS
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or DEFAULT_DEV_ORIGINS


def artifacts_dir() -> Path:
    override = os.getenv("TRAFFICSENSE_ARTIFACTS_DIR")
    return Path(override) if override else artifact_paths()["dir"]


def offline_mode() -> bool:
    """True when network downloads are disabled (cached data only)."""
    return os.getenv("TRAFFICSENSE_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + graph cache once at startup (not per request)."""
    app.state.artifacts_dir = artifacts_dir()
    app.state.model_bundle = load_model_bundle(app.state.artifacts_dir)
    app.state.graph_cache = GraphCache(artifacts_dir=app.state.artifacts_dir)
    app.state.study_area = study_area_from_env()
    yield


app = FastAPI(
    title="TrafficSense API",
    version=__version__,
    description=(
        "Traffic congestion prediction (scikit-learn) and road-network route "
        "analysis (OSMnx/NetworkX) for the TrafficSense B.Tech ML PBL project."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _bundle(request: Request):
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained model is not available. Train it first from the "
                "backend directory: 'python -m app.train' "
                "(add --download if the dataset is missing)."
            ),
        )
    return bundle


# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #


@app.get("/", response_model=ServiceInfo, tags=["service"])
def service_info() -> ServiceInfo:
    return ServiceInfo(
        version=__version__,
        description=(
            "Traffic congestion prediction and route analysis. Predictions come "
            "from a scikit-learn model trained on the UCI Metro Interstate "
            "Traffic Volume dataset (one Minnesota interstate sensor)."
        ),
        documentation="/docs",
        endpoints=[
            "GET /",
            "GET /api/health",
            "GET /api/metrics",
            "POST /api/predict",
            "POST /api/routes",
        ],
    )


@app.get("/api/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        return HealthResponse(
            status="degraded",
            model_available=False,
            detail=(
                "API is running but no trained model was found in the artifacts "
                "directory. Run 'python -m app.train' to create it."
            ),
        )
    metadata = bundle.metadata
    return HealthResponse(
        status="ok",
        model_available=True,
        model_name=metadata.get("selected_model"),
        model_created_utc=metadata.get("created_utc"),
        dataset_rows=metadata.get("dataset_rows_clean"),
        detail="Trained model loaded and ready for predictions.",
    )


@app.get("/api/metrics", response_model=MetricsResponse, tags=["analytics"])
def metrics(request: Request) -> MetricsResponse:
    directory = getattr(request.app.state, "artifacts_dir", None) or artifacts_dir()
    payload = load_metrics(directory)
    metadata = load_metadata(directory)
    if payload is None:
        return MetricsResponse(
            available=False,
            detail=(
                "No metrics file found. Run 'python -m app.train' to train the "
                "model and generate evaluation metrics."
            ),
            metadata=metadata,
        )
    return MetricsResponse(
        available=True,
        detail="Metrics were produced by a real chronological holdout evaluation.",
        model_name=(metadata or {}).get("selected_model"),
        metadata=metadata,
        metrics=payload,
    )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


def _feature_row(payload: PredictionRequest) -> pd.DataFrame:
    """Build a single-row frame in the exact training feature order."""
    hour = payload.hour
    day_of_week = payload.day_of_week
    month = payload.month
    row = {
        "temp": payload.temperature_kelvin(),
        "rain_1h": payload.rainfall_mm,
        "snow_1h": payload.snowfall_mm,
        "clouds_all": payload.cloud_cover,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": int(day_of_week >= 5),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2 * np.pi * day_of_week / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * day_of_week / 7.0)),
        "month_sin": float(np.sin(2 * np.pi * (month - 1) / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * (month - 1) / 12.0)),
        "weather_main": payload.weather_category,
        "holiday": payload.holiday_category,
    }
    return pd.DataFrame([row])[MODEL_FEATURES]


@app.post("/api/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    bundle = _bundle(request)
    frame = _feature_row(payload)

    try:
        prediction = float(bundle.pipeline.predict(frame)[0])
    except Exception as exc:  # noqa: BLE001 - surface a clean 500
        raise HTTPException(
            status_code=500,
            detail=f"The model could not score this input: {exc}",
        ) from exc

    prediction = max(0.0, prediction)
    thresholds = bundle.thresholds
    category = classify_volume(prediction, thresholds)

    return PredictionResponse(
        predicted_traffic_volume=round(prediction, 1),
        congestion_category=category,
        congestion_thresholds=Thresholds(**thresholds),
        thresholds_explanation=(
            "Congestion categories use thresholds derived from the training split "
            f"only: Low <= {thresholds['low_max']:.0f}, Moderate <= "
            f"{thresholds['moderate_max']:.0f}, High <= {thresholds['high_max']:.0f}, "
            "and Severe above that. Units are vehicles per hour."
        ),
        model_name=bundle.model_name,
        model_created_utc=bundle.metadata.get("created_utc"),
        input_echo={
            "hour": payload.hour,
            "day_of_week": payload.day_of_week,
            "month": payload.month,
            "temperature_kelvin": round(payload.temperature_kelvin(), 2),
            "rainfall_mm": payload.rainfall_mm,
            "snowfall_mm": payload.snowfall_mm,
            "cloud_cover": payload.cloud_cover,
            "weather_category": payload.weather_category,
            "holiday_category": payload.holiday_category,
        },
        disclaimer=(
            "This is a historical-data-based estimate from a model trained on one "
            "Minnesota interstate sensor (2012-2018). It is NOT a live traffic "
            "measurement and does not describe any specific road in another city."
        ),
    )


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


@app.post("/api/routes", response_model=RouteResponse, tags=["routing"])
def routes(payload: RouteRequest, request: Request) -> RouteResponse:
    cache: GraphCache = request.app.state.graph_cache
    area = dict(getattr(request.app.state, "study_area", study_area_from_env()))
    if payload.study_area:
        for key in ("latitude", "longitude", "distance_m", "network_type", "name"):
            if key in payload.study_area:
                area[key] = payload.study_area[key]
        try:
            area["latitude"] = float(area["latitude"])
            area["longitude"] = float(area["longitude"])
            area["distance_m"] = int(area["distance_m"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "study_area must contain numeric latitude, longitude and "
                    "distance_m values."
                ),
            ) from exc

    try:
        graph = cache.get(area, allow_download=not offline_mode())
    except GraphUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        scenario = build_scenario(
            payload.scenario,
            severity=payload.severity,
            highway_multipliers=payload.highway_multipliers,
            edge_multipliers=(
                [item.model_dump() for item in payload.edge_multipliers]
                if payload.edge_multipliers
                else None
            ),
        )
    except RoutingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = plan_routes(
            graph,
            (payload.origin.latitude, payload.origin.longitude),
            (payload.destination.latitude, payload.destination.longitude),
            scenario=scenario,
            max_alternatives=payload.max_alternatives,
        )
    except RouteNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RoutingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return RouteResponse(**result)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


@app.exception_handler(GraphUnavailableError)
async def _graph_unavailable(_: Request, exc: GraphUnavailableError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(RoutingError)
async def _routing_error(_: Request, exc: RoutingError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})
