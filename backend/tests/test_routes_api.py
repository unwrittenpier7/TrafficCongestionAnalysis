"""API-level tests for /api/routes using a mocked road graph.

The graph is injected into the app state so these tests never download from
OpenStreetMap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.routing import GraphCache

ORIGIN = {"latitude": 17.4400, "longitude": 78.3700}
DESTINATION = {"latitude": 17.4500, "longitude": 78.3800}


@pytest.fixture()
def route_client(tmp_path: Path, tiny_graph, monkeypatch):
    """App with the trained synthetic model and a pre-seeded graph cache."""
    from app.train import train

    artifacts = tmp_path / "artifacts"
    train(
        data_path=_synthetic_csv(tmp_path),
        artifacts_dir=artifacts,
        allow_download=False,
    )
    monkeypatch.setenv("TRAFFICSENSE_ARTIFACTS_DIR", str(artifacts))

    with TestClient(main_module.app) as client:
        # Replace the cache with one that already holds the tiny graph.
        cache = GraphCache(artifacts_dir=artifacts)
        cache._graph = tiny_graph
        cache._key = cache._area_key(main_module.study_area_from_env())
        client.app.state.graph_cache = cache
        yield client


def _synthetic_csv(tmp_path: Path) -> Path:
    from conftest import make_synthetic_frame

    path = tmp_path / "Metro_Interstate_Traffic_Volume.csv"
    make_synthetic_frame().to_csv(path, index=False)
    return path


def test_routes_endpoint_returns_multiple_routes(route_client):
    response = route_client.post(
        "/api/routes",
        json={"origin": ORIGIN, "destination": DESTINATION, "scenario": "free_flow"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["routes"]) >= 2
    first = body["routes"][0]
    assert first["distance_km"] > 0
    assert len(first["geometry"]) >= 2
    assert body["scenario"]["provenance"] in {"free_flow", "simulated", "custom"}


def test_routes_response_documents_limitations(route_client):
    response = route_client.post(
        "/api/routes", json={"origin": ORIGIN, "destination": DESTINATION}
    )
    body = response.json()
    assert body["limitations"]
    assert "OpenStreetMap" in body["attribution"]
    # The default scenario must be flagged as simulated, not measured.
    assert body["scenario"]["provenance"] == "simulated"


def test_routes_rejects_out_of_range_coordinates(route_client):
    response = route_client.post(
        "/api/routes",
        json={"origin": {"latitude": 200.0, "longitude": 78.0},
              "destination": DESTINATION},
    )
    assert response.status_code == 422


def test_routes_rejects_unknown_scenario(route_client):
    response = route_client.post(
        "/api/routes",
        json={"origin": ORIGIN, "destination": DESTINATION, "scenario": "magic"},
    )
    assert response.status_code == 422


def test_routes_returns_404_for_unreachable_destination(route_client, tiny_graph):
    tiny_graph.add_node("ISLAND", y=17.9, x=78.9)
    response = route_client.post(
        "/api/routes",
        json={"origin": ORIGIN, "destination": {"latitude": 17.9, "longitude": 78.9}},
    )
    assert response.status_code == 404
    assert "no road route" in response.json()["detail"].lower()


def test_routes_returns_503_when_graph_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFICSENSE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    with TestClient(main_module.app) as client:
        cache = GraphCache(artifacts_dir=tmp_path / "artifacts")
        client.app.state.graph_cache = cache
        response = client.post(
            "/api/routes", json={"origin": ORIGIN, "destination": DESTINATION}
        )
    assert response.status_code == 503
    assert "road network" in response.json()["detail"].lower()


def test_custom_scenario_edge_multiplier_via_api(route_client):
    response = route_client.post(
        "/api/routes",
        json={
            "origin": ORIGIN,
            "destination": DESTINATION,
            "scenario": "custom",
            "highway_multipliers": {"motorway": 3.0},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["scenario"]["provenance"] == "custom"
    assert all(route["congestion_aware_time_minutes"] >= 0 for route in body["routes"])


def test_routes_offline_mode_refuses_to_download(tmp_path, monkeypatch):
    """With TRAFFICSENSE_OFFLINE=1 an empty cache must fail fast, not download."""
    monkeypatch.setenv("TRAFFICSENSE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("TRAFFICSENSE_OFFLINE", "1")
    with TestClient(main_module.app) as client:
        client.app.state.graph_cache = GraphCache(artifacts_dir=tmp_path / "artifacts")
        response = client.post(
            "/api/routes", json={"origin": ORIGIN, "destination": DESTINATION}
        )
    assert response.status_code == 503
    assert "downloading is disabled" in response.json()["detail"].lower()
