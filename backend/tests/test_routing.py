"""Tests for the graph-based routing module.

All graph tests use a tiny hand-built NetworkX graph so they never depend on a
live OpenStreetMap download.
"""

from __future__ import annotations

import networkx as nx
import pytest

from app.routing import (
    CongestionScenario,
    GraphCache,
    GraphUnavailableError,
    RouteNotFoundError,
    RoutingError,
    build_scenario,
    compute_path_metrics,
    custom_scenario,
    free_flow_scenario,
    nearest_node,
    plan_routes,
    simulated_peak_scenario,
)

ORIGIN = (17.4400, 78.3700)
DESTINATION = (17.4500, 78.3800)


# --------------------------------------------------------------------------- #
# Graph helpers
# --------------------------------------------------------------------------- #


def test_nearest_node_finds_closest_node(tiny_graph):
    node = nearest_node(tiny_graph, *ORIGIN)
    assert node == "A"


def test_nearest_node_raises_for_empty_graph():
    empty = nx.MultiDiGraph()
    with pytest.raises(RoutingError):
        nearest_node(empty, 17.0, 78.0)


def test_edge_annotation_computes_travel_time(tiny_graph):
    data = tiny_graph.get_edge_data("A", "B", 0)
    assert data["length_m"] == pytest.approx(1400.0)
    assert data["travel_time_s"] == pytest.approx(1400.0 / (100000 / 3600), rel=1e-6)


# --------------------------------------------------------------------------- #
# Distance / cost calculations
# --------------------------------------------------------------------------- #


def test_compute_path_metrics_sums_distance(tiny_graph):
    scenario = free_flow_scenario()
    metrics = compute_path_metrics(tiny_graph, ["A", "C", "D"], scenario, label="test")
    assert metrics["distance_km"] == pytest.approx(1.8, rel=1e-6)
    assert metrics["mean_congestion_multiplier"] == pytest.approx(1.0)
    # 900 m at 50 km/h twice == 64.8 s * 2 == 129.6 s == 2.16 minutes
    assert metrics["free_flow_time_minutes"] == pytest.approx(2.16, rel=1e-3)


def test_compute_path_metrics_geometry_has_coordinates(tiny_graph):
    scenario = free_flow_scenario()
    metrics = compute_path_metrics(tiny_graph, ["A", "C", "D"], scenario, label="test")
    assert len(metrics["geometry"]) == 3
    assert metrics["geometry"][0] == pytest.approx([17.44, 78.37], abs=1e-6)


def test_congestion_scenario_increases_cost(tiny_graph):
    free = compute_path_metrics(tiny_graph, ["A", "C", "D"], free_flow_scenario(), label="f")
    peak = compute_path_metrics(
        tiny_graph, ["A", "C", "D"], simulated_peak_scenario(severity=1.0), label="p"
    )
    assert peak["congestion_aware_time_minutes"] > free["congestion_aware_time_minutes"]
    assert peak["mean_congestion_multiplier"] > 1.0


def test_custom_highway_multiplier_is_applied(tiny_graph):
    scenario = custom_scenario(highway_multipliers={"motorway": 2.0})
    metrics = compute_path_metrics(tiny_graph, ["A", "B", "D"], scenario, label="c")
    assert metrics["mean_congestion_multiplier"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Shortest path + alternatives
# --------------------------------------------------------------------------- #


def test_plan_routes_returns_multiple_alternatives(tiny_graph):
    result = plan_routes(tiny_graph, ORIGIN, DESTINATION, scenario=free_flow_scenario())
    labels = {route["label"] for route in result["routes"]}
    assert "shortest_distance" in labels
    assert "fastest_free_flow" in labels
    assert len(result["routes"]) >= 2


def test_shortest_distance_route_is_the_geometrically_shortest(tiny_graph):
    result = plan_routes(tiny_graph, ORIGIN, DESTINATION, scenario=free_flow_scenario())
    shortest = next(r for r in result["routes"] if r["label"] == "shortest_distance")
    assert shortest["distance_km"] == pytest.approx(1.8, rel=1e-3)


def test_congestion_changes_the_optimal_route(tiny_graph):
    """Edge-specific multipliers must be able to change the chosen path.

    Free-flow optimum is the motorway (A-B-D). If motorways become extremely
    congested, the arterial route (A-C-D) must win instead.
    """
    free = plan_routes(tiny_graph, ORIGIN, DESTINATION, scenario=free_flow_scenario())
    free_best = min(free["routes"], key=lambda r: r["free_flow_time_minutes"])

    congested = plan_routes(
        tiny_graph,
        ORIGIN,
        DESTINATION,
        scenario=custom_scenario(highway_multipliers={"motorway": 3.0}),
    )
    congested_best = congested["routes"][0]

    assert free_best["label"] != congested_best["label"]
    assert congested_best["congestion_aware_time_minutes"] <= min(
        r["congestion_aware_time_minutes"] for r in congested["routes"]
    )


def test_uniform_multiplier_does_not_change_ranking(tiny_graph):
    """A uniform multiplier must not change which path is optimal (documented)."""
    free = plan_routes(tiny_graph, ORIGIN, DESTINATION, scenario=free_flow_scenario())
    uniform = plan_routes(
        tiny_graph,
        ORIGIN,
        DESTINATION,
        scenario=custom_scenario(highway_multipliers={
            "motorway": 2.5, "primary": 2.5, "residential": 2.5
        }),
    )
    free_order = [r["label"] for r in free["routes"]]
    uniform_order = [r["label"] for r in uniform["routes"]]
    assert free_order == uniform_order


def test_scenario_metadata_flags_uniform_multiplier():
    assert free_flow_scenario().describe()["is_uniform_multiplier"] is True
    assert simulated_peak_scenario().describe()["is_uniform_multiplier"] is False
    assert simulated_peak_scenario().describe()["provenance"] == "simulated"


def test_plan_routes_includes_limitations_and_attribution(tiny_graph):
    result = plan_routes(tiny_graph, ORIGIN, DESTINATION)
    assert "OpenStreetMap" in result["attribution"]
    assert any("SIMULATED" in text for text in result["limitations"])
    assert result["graph"]["nodes"] == tiny_graph.number_of_nodes()


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_disconnected_route_raises_route_not_found(tiny_graph):
    tiny_graph.add_node("ISLAND", y=17.5, x=78.5)
    with pytest.raises(RouteNotFoundError):
        plan_routes(tiny_graph, (17.4400, 78.3700), (17.5, 78.5))


def test_same_node_origin_and_destination_raises(tiny_graph):
    with pytest.raises(RoutingError):
        plan_routes(tiny_graph, ORIGIN, ORIGIN)


def test_unknown_scenario_name_raises():
    with pytest.raises(RoutingError) as excinfo:
        build_scenario("teleport")
    assert "unknown congestion scenario" in str(excinfo.value).lower()


def test_graph_cache_without_download_raises_when_empty(tmp_path):
    cache = GraphCache(artifacts_dir=tmp_path)
    with pytest.raises(GraphUnavailableError):
        cache.get(
            {"latitude": 0.0, "longitude": 0.0, "distance_m": 100,
             "network_type": "drive", "name": "test"},
            allow_download=False,
        )


def test_graph_cache_uses_pickle_cache(tmp_path, tiny_graph):
    cache = GraphCache(artifacts_dir=tmp_path)
    area = {"latitude": 17.44, "longitude": 78.37, "distance_m": 500,
            "network_type": "drive", "name": "test"}
    # Write a cache entry directly, then confirm it is loaded without download.
    import pickle

    area_key = cache._area_key(area)
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache.cache_dir / f"{area_key}.pkl").open("wb") as handle:
        pickle.dump(tiny_graph, handle)

    loaded = cache.get(area, allow_download=False)
    assert loaded.number_of_nodes() == tiny_graph.number_of_nodes()
