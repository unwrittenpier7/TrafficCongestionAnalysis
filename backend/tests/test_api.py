"""Tests for model training, artifact loading, and the FastAPI endpoints.

These tests use the small synthetic fixture dataset, so they never download the
real 48k-row dataset and never contact OpenStreetMap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.model_store import (
    artifact_paths,
    load_metadata,
    load_metrics,
    load_model_bundle,
)
from app.train import train

# --------------------------------------------------------------------------- #
# Training + artifacts
# --------------------------------------------------------------------------- #


def test_train_produces_all_artifacts(trained_artifacts: Path):
    paths = artifact_paths(trained_artifacts)
    assert paths["model"].is_file()
    assert paths["metadata"].is_file()
    assert paths["metrics"].is_file()
    assert paths["plot"].is_file()


def test_trained_metrics_are_real_and_finite(trained_artifacts: Path):
    metrics = load_metrics(trained_artifacts)
    assert metrics is not None
    for key in ("mae", "rmse", "r2"):
        value = metrics["test"][key]
        assert isinstance(value, float)
        assert value == value  # not NaN
    assert metrics["test"]["mae"] >= 0
    assert metrics["test"]["rmse"] >= 0


def test_metadata_records_thresholds_and_periods(trained_artifacts: Path):
    metadata = load_metadata(trained_artifacts)
    assert metadata is not None
    thresholds = metadata["congestion_thresholds"]
    assert thresholds["low_max"] < thresholds["moderate_max"] < thresholds["high_max"]
    assert metadata["split"]["strategy"].startswith("chronological")
    assert metadata["split"]["periods"]["test"]["rows"] > 0


def test_model_bundle_loads_and_predicts(trained_artifacts: Path):
    bundle = load_model_bundle(trained_artifacts)
    assert bundle is not None
    assert bundle.model_name in {"random_forest", "xgboost", "baseline_mean"}
    assert set(bundle.thresholds) == {"low_max", "moderate_max", "high_max"}


def test_load_model_bundle_returns_none_when_missing(tmp_path: Path):
    assert load_model_bundle(tmp_path / "empty") is None


def test_load_metadata_returns_none_for_corrupt_json(tmp_path: Path):
    paths = artifact_paths(tmp_path)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["metadata"].write_text("{not valid json", encoding="utf-8")
    assert load_metadata(tmp_path) is None


def test_metrics_json_contains_chart_series(trained_artifacts: Path):
    metrics = load_metrics(trained_artifacts)
    assert metrics is not None
    assert len(metrics["actual_vs_predicted"]["actual"]) > 0
    assert len(metrics["actual_vs_predicted"]["actual"]) == len(
        metrics["actual_vs_predicted"]["predicted"]
    )
    assert len(metrics["traffic_by_hour"]) == 24
    assert len(metrics["traffic_by_weekday"]) == 7


# --------------------------------------------------------------------------- #
# API client fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client_with_model(trained_artifacts: Path, monkeypatch):
    monkeypatch.setenv("TRAFFICSENSE_ARTIFACTS_DIR", str(trained_artifacts))
    with TestClient(main_module.app) as client:
        yield client


@pytest.fixture()
def client_without_model(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TRAFFICSENSE_ARTIFACTS_DIR", str(tmp_path / "no-artifacts"))
    with TestClient(main_module.app) as client:
        yield client


# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #


def test_root_returns_service_information(client_with_model):
    response = client_with_model.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "TrafficSense API"
    assert any("/api/predict" in endpoint for endpoint in body["endpoints"])
    assert any("/api/routes" in endpoint for endpoint in body["endpoints"])


def test_health_reports_model_available(client_with_model):
    response = client_with_model.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_available"] is True
    assert body["dataset_rows"] and body["dataset_rows"] > 0


def test_health_degrades_without_model(client_without_model):
    response = client_without_model.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_available"] is False
    assert "train" in body["detail"].lower()


def test_metrics_endpoint_returns_saved_metrics(client_with_model):
    response = client_with_model.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["model_name"]
    assert "test" in body["metrics"]
    assert body["metrics"]["test"]["mae"] >= 0


def test_metrics_endpoint_reports_unavailable_without_model(client_without_model):
    response = client_without_model.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert "train" in body["detail"].lower()


def test_openapi_schema_is_served(client_with_model):
    response = client_with_model.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for endpoint in ("/api/health", "/api/metrics", "/api/predict", "/api/routes"):
        assert endpoint in paths


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


VALID_PAYLOAD = {
    "hour": 8,
    "day_of_week": 1,
    "month": 6,
    "temperature": 295.0,
    "temperature_unit": "kelvin",
    "rainfall_mm": 0.0,
    "snowfall_mm": 0.0,
    "cloud_cover": 30.0,
    "weather_category": "clear",
    "holiday_category": "None",
}


def test_predict_returns_expected_structure(client_with_model):
    response = client_with_model.post("/api/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predicted_traffic_volume"], (int, float))
    assert body["predicted_traffic_volume"] >= 0
    assert body["congestion_category"] in {"Low", "Moderate", "High", "Severe"}
    assert set(body["congestion_thresholds"]) == {"low_max", "moderate_max", "high_max"}
    assert "historical" in body["disclaimer"].lower()
    assert body["input_echo"]["hour"] == 8


def test_predict_accepts_celsius_and_converts(client_with_model):
    payload = dict(VALID_PAYLOAD, temperature=22.0, temperature_unit="celsius")
    response = client_with_model.post("/api/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["input_echo"]["temperature_kelvin"] == pytest.approx(295.15, abs=0.02)


def test_predict_rejects_invalid_hour(client_with_model):
    response = client_with_model.post("/api/predict", json=dict(VALID_PAYLOAD, hour=25))
    assert response.status_code == 422


def test_predict_rejects_invalid_day_of_week(client_with_model):
    response = client_with_model.post(
        "/api/predict", json=dict(VALID_PAYLOAD, day_of_week=9)
    )
    assert response.status_code == 422


def test_predict_rejects_invalid_month(client_with_model):
    response = client_with_model.post("/api/predict", json=dict(VALID_PAYLOAD, month=13))
    assert response.status_code == 422


def test_predict_rejects_negative_rainfall(client_with_model):
    response = client_with_model.post(
        "/api/predict", json=dict(VALID_PAYLOAD, rainfall_mm=-1.0)
    )
    assert response.status_code == 422


def test_predict_rejects_cloud_cover_above_100(client_with_model):
    response = client_with_model.post(
        "/api/predict", json=dict(VALID_PAYLOAD, cloud_cover=150.0)
    )
    assert response.status_code == 422


def test_predict_rejects_unknown_weather_category(client_with_model):
    response = client_with_model.post(
        "/api/predict", json=dict(VALID_PAYLOAD, weather_category="lava")
    )
    assert response.status_code == 422
    assert "weather category" in json.dumps(response.json()).lower()


def test_predict_returns_503_without_model(client_without_model):
    response = client_without_model.post("/api/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503
    assert "train" in response.json()["detail"].lower()


def test_predict_is_deterministic(client_with_model):
    first = client_with_model.post("/api/predict", json=VALID_PAYLOAD).json()
    second = client_with_model.post("/api/predict", json=VALID_PAYLOAD).json()
    assert first["predicted_traffic_volume"] == second["predicted_traffic_volume"]


def test_predict_night_volume_below_peak_volume(client_with_model):
    """A sanity check: 3am should not out-predict 5pm on a weekday."""
    night = client_with_model.post(
        "/api/predict", json=dict(VALID_PAYLOAD, hour=3, day_of_week=2)
    ).json()
    evening = client_with_model.post(
        "/api/predict", json=dict(VALID_PAYLOAD, hour=17, day_of_week=2)
    ).json()
    assert evening["predicted_traffic_volume"] > night["predicted_traffic_volume"]
