"""Graph-based route analysis for TrafficSense.

This module is deliberately separate from the traffic-volume machine-learning
model. The ML model was trained on a *single* Minnesota interstate sensor; it
cannot tell you how congested an arbitrary road in another city is. Applying
its output to every edge of a road graph would be scientifically dishonest.

Instead we do three clearly-labelled things:

1. **Free-flow routing** - Dijkstra on real OpenStreetMap edge lengths/speeds.
2. **Congestion-aware routing** - Dijkstra on ``travel_time * multiplier``
   where the multiplier is *edge specific*.
3. **Alternative route enumeration** - several distinct simple paths.

Because public per-road traffic counts are not available for arbitrary study
areas, the default congestion multipliers come from an explicitly *simulated*
scenario. The provenance of every multiplier is reported in the response and
shown in the UI. A uniform multiplier is also provided (``free_flow``) and is
documented as NOT changing the ranking of paths - it scales every route by the
same constant.

Attribution: road data (c) OpenStreetMap contributors, ODbL licence.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import networkx as nx

from .model_store import default_artifacts_dir

# --------------------------------------------------------------------------- #
# Study-area configuration
# --------------------------------------------------------------------------- #

#: Small demo study area (Gachibowli, Hyderabad). Override with environment
#: variables or the /api/routes request. Kept small so the graph download is
#: fast and the demo does not hammer the Overpass API.
DEFAULT_STUDY_AREA: dict[str, Any] = {
    "name": "Gachibowli, Hyderabad (demo study area)",
    "latitude": 17.44,
    "longitude": 78.35,
    "distance_m": 600,
    "network_type": "drive",
}

#: Fallback speeds (km/h) by OSM highway class, used when an edge has no speed.
FALLBACK_SPEEDS_KPH: dict[str, float] = {
    "motorway": 100.0,
    "motorway_link": 60.0,
    "trunk": 80.0,
    "trunk_link": 50.0,
    "primary": 60.0,
    "primary_link": 40.0,
    "secondary": 50.0,
    "secondary_link": 35.0,
    "tertiary": 40.0,
    "tertiary_link": 30.0,
    "unclassified": 30.0,
    "residential": 30.0,
    "living_street": 15.0,
    "service": 20.0,
    "road": 30.0,
}

#: Relative congestion weight by highway class for the simulated scenario.
#: Major roads carry more traffic, so they receive larger peak multipliers.
SIMULATED_CLASS_WEIGHT: dict[str, float] = {
    "motorway": 1.75,
    "trunk": 1.60,
    "primary": 1.45,
    "secondary": 1.30,
    "tertiary": 1.15,
    "unclassified": 1.05,
    "residential": 1.10,
    "living_street": 1.00,
    "service": 1.00,
    "road": 1.05,
}


def study_area_from_env() -> dict[str, Any]:
    """Read study-area overrides from the environment, falling back to defaults."""
    area = dict(DEFAULT_STUDY_AREA)
    if os.getenv("TRAFFICSENSE_STUDY_AREA_NAME"):
        area["name"] = os.environ["TRAFFICSENSE_STUDY_AREA_NAME"]
    point = os.getenv("TRAFFICSENSE_STUDY_POINT")
    if point and "," in point:
        try:
            lat, lon = (float(p) for p in point.split(",", 1))
            area["latitude"], area["longitude"] = lat, lon
        except ValueError:
            pass
    if os.getenv("TRAFFICSENSE_STUDY_DIST"):
        try:
            area["distance_m"] = int(float(os.environ["TRAFFICSENSE_STUDY_DIST"]))
        except ValueError:
            pass
    if os.getenv("TRAFFICSENSE_NETWORK_TYPE"):
        area["network_type"] = os.environ["TRAFFICSENSE_NETWORK_TYPE"]
    return area


class RoutingError(RuntimeError):
    """Raised for routing problems that should become HTTP 4xx/5xx responses."""


class RouteNotFoundError(RoutingError):
    """Origin and destination are not connected in the road network."""


class GraphUnavailableError(RoutingError):
    """The road graph could not be obtained (network/dependency failure)."""


# --------------------------------------------------------------------------- #
# Graph loading + caching
# --------------------------------------------------------------------------- #


@dataclass
class GraphCache:
    """Loads an OSMnx drivable graph once and reuses it across requests."""

    artifacts_dir: Path = field(default_factory=default_artifacts_dir)
    _graph: nx.MultiDiGraph | None = None
    _key: str | None = None

    @property
    def cache_dir(self) -> Path:
        directory = self.artifacts_dir / "osm_cache"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _area_key(area: dict[str, Any]) -> str:
        raw = (
            f"{area['latitude']:.5f}_{area['longitude']:.5f}_"
            f"{area['distance_m']}_{area['network_type']}"
        )
        digest = hashlib.sha1(raw.encode()).hexdigest()[:10]
        return f"graph_{digest}"

    def _pickle_path(self, area: dict[str, Any]) -> Path:
        return self.cache_dir / f"{self._area_key(area)}.pkl"

    def clear(self) -> None:
        self._graph = None
        self._key = None

    def get(self, area: dict[str, Any] | None = None, *, allow_download: bool = True):
        """Return a cached/loaded road graph, downloading it only if necessary."""
        area = area or study_area_from_env()
        key = self._area_key(area)
        if self._graph is not None and self._key == key:
            return self._graph

        pickle_path = self._pickle_path(area)
        if pickle_path.is_file():
            try:
                with pickle_path.open("rb") as handle:
                    graph = pickle.load(handle)
                if graph.number_of_nodes() > 0:
                    self._graph, self._key = graph, key
                    return graph
            except Exception:
                # Corrupt cache: fall through and rebuild it.
                pickle_path.unlink(missing_ok=True)

        if not allow_download:
            raise GraphUnavailableError(
                "No cached road graph is available and downloading is disabled."
            )

        graph = self._download(area)
        try:
            with pickle_path.open("wb") as handle:
                pickle.dump(graph, handle)
        except OSError:  # pragma: no cover - cache write is best effort
            pass
        self._graph, self._key = graph, key
        return graph

    @staticmethod
    def _download(area: dict[str, Any]):
        """Fetch the graph from OpenStreetMap via OSMnx/Overpass."""
        try:
            import osmnx as ox
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise GraphUnavailableError(
                "osmnx is not installed; run 'pip install -r requirements.txt'."
            ) from exc

        ox.settings.use_cache = True
        ox.settings.log_console = False
        ox.settings.requests_timeout = 60
        try:
            graph = ox.graph_from_point(
                (area["latitude"], area["longitude"]),
                dist=int(area["distance_m"]),
                network_type=area["network_type"],
                simplify=True,
                retain_all=False,
            )
        except Exception as exc:  # noqa: BLE001 - many possible network errors
            raise GraphUnavailableError(
                "Could not download the road network from OpenStreetMap "
                f"({type(exc).__name__}: {exc}). Check your internet connection or "
                "set TRAFFICSENSE_STUDY_POINT / TRAFFICSENSE_STUDY_DIST to a smaller "
                "area, then retry."
            ) from exc

        if graph is None or graph.number_of_nodes() == 0:
            raise GraphUnavailableError(
                "OpenStreetMap returned an empty graph for this study area."
            )
        _annotate_edges(graph)
        return graph


def _edge_speed_kph(data: dict[str, Any]) -> float:
    """Resolve an edge speed in km/h, preferring OSM data over fallbacks."""
    speed = data.get("speed_kph")
    if speed is not None:
        try:
            value = float(speed)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    highway = data.get("highway")
    if isinstance(highway, list):
        highway = highway[0] if highway else None
    return FALLBACK_SPEEDS_KPH.get(str(highway), 30.0)


def _annotate_edges(graph) -> None:
    """Ensure every edge has ``length_m``, ``speed_kph`` and ``travel_time_s``."""
    for _, _, _, data in graph.edges(keys=True, data=True):
        try:
            length = float(data.get("length", 0.0))
        except (TypeError, ValueError):
            length = 0.0
        if length <= 0:
            length = 1.0  # Avoid zero-length edges breaking Dijkstra.
        data["length_m"] = length

        speed = _edge_speed_kph(data)
        data["speed_kph"] = speed
        data["travel_time_s"] = length / (speed * 1000.0 / 3600.0)


# --------------------------------------------------------------------------- #
# Congestion scenarios
# --------------------------------------------------------------------------- #


@dataclass
class CongestionScenario:
    """An edge-specific congestion multiplier definition.

    ``provenance`` is one of:
      * ``"free_flow"``  - no congestion modelling (multiplier 1.0 everywhere)
      * ``"simulated"``  - deterministic simulated scenario (the default)
      * ``"custom"``     - caller-supplied multipliers
    """

    name: str
    description: str
    provenance: str
    multiplier: Callable[[Any, Any, Any, dict[str, Any]], float]
    is_uniform: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "provenance": self.provenance,
            "is_uniform_multiplier": self.is_uniform,
            "note": (
                "A uniform multiplier scales every route by the same constant and "
                "therefore cannot change which path is optimal."
                if self.is_uniform
                else "Multipliers differ per road segment, so they can change the "
                "optimal path."
            ),
        }


def free_flow_scenario() -> CongestionScenario:
    def multiplier(u, v, k, data):  # noqa: ARG001
        return 1.0

    return CongestionScenario(
        name="free_flow",
        description="Uncongested free-flow travel time (multiplier 1.0 on every edge).",
        provenance="free_flow",
        multiplier=multiplier,
        is_uniform=True,
    )


def simulated_peak_scenario(
    *, severity: float = 1.0, arterial_bias: bool = True
) -> CongestionScenario:
    """Deterministic simulated peak-hour congestion.

    The multiplier for an edge is
    ``1 + severity * (class_weight - 1) * jitter`` where ``jitter`` in
    ``[0.6, 1.4]`` is derived from a hash of the edge identity. It is stable
    across runs and, crucially, *edge specific*.

    This is a SIMULATION for demonstrating congestion-aware routing. It is not
    measured traffic data.
    """

    def multiplier(u, v, k, data):
        highway = data.get("highway")
        if isinstance(highway, list):
            highway = highway[0] if highway else None
        base = SIMULATED_CLASS_WEIGHT.get(str(highway), 1.05)
        if not arterial_bias:
            base = 1.25
        seed = f"{u}-{v}-{k}".encode()
        jitter = 0.6 + (int(hashlib.sha1(seed).hexdigest()[:8], 16) % 1000) / 1000.0 * 0.8
        return float(max(1.0, 1.0 + severity * (base - 1.0) * jitter))

    return CongestionScenario(
        name="simulated_peak",
        description=(
            "Simulated peak-hour congestion. Multiplier per segment is derived "
            "deterministically from the OSM road class and the edge identity. "
            "This is a DEMONSTRATION scenario, not measured traffic."
        ),
        provenance="simulated",
        multiplier=multiplier,
        is_uniform=False,
    )


def custom_scenario(
    highway_multipliers: dict[str, float] | None = None,
    edge_multipliers: Iterable[dict[str, Any]] | None = None,
) -> CongestionScenario:
    """Caller-supplied multipliers, e.g. from a real traffic-count dataset."""
    highway_map = {
        str(k).lower(): float(v) for k, v in (highway_multipliers or {}).items()
    }
    edge_map: dict[tuple, float] = {}
    for item in edge_multipliers or []:
        try:
            edge_map[(item["u"], item["v"], item.get("key", 0))] = float(
                item["multiplier"]
            )
        except (KeyError, TypeError, ValueError):
            continue

    def multiplier(u, v, k, data):
        if (u, v, k) in edge_map:
            return max(1.0, edge_map[(u, v, k)])
        highway = data.get("highway")
        if isinstance(highway, list):
            highway = highway[0] if highway else None
        if str(highway).lower() in highway_map:
            return max(1.0, highway_map[str(highway).lower()])
        return 1.0

    return CongestionScenario(
        name="custom",
        description=(
            "Caller-supplied congestion multipliers applied per road segment "
            "(highway class and/or explicit edge overrides)."
        ),
        provenance="custom",
        multiplier=multiplier,
        is_uniform=False,
    )


def build_scenario(
    name: str,
    *,
    severity: float = 1.0,
    highway_multipliers: dict[str, float] | None = None,
    edge_multipliers: Iterable[dict[str, Any]] | None = None,
) -> CongestionScenario:
    """Factory resolving a scenario name into a :class:`CongestionScenario`."""
    key = (name or "simulated_peak").strip().lower()
    if key in {"free_flow", "free-flow", "none"}:
        return free_flow_scenario()
    if key in {"custom", "observed"}:
        return custom_scenario(highway_multipliers, edge_multipliers)
    if key in {"simulated_peak", "simulated", "peak"}:
        return simulated_peak_scenario(severity=severity)
    raise RoutingError(
        f"Unknown congestion scenario '{name}'. "
        "Supported: free_flow, simulated_peak, custom."
    )


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def nearest_node(graph, latitude: float, longitude: float):
    """Return the graph node nearest to a coordinate.

    Uses a linear scan so no scipy KD-tree is required.
    """
    best_node, best_distance = None, float("inf")
    for node, data in graph.nodes(data=True):
        try:
            d_lat = data["y"] - latitude
            d_lon = data["x"] - longitude
        except KeyError:
            continue
        distance = d_lat * d_lat + d_lon * d_lon
        if distance < best_distance:
            best_node, best_distance = node, distance
    if best_node is None:
        raise RoutingError("The road graph contains no usable nodes.")
    return best_node


def _route_weight(scenario: CongestionScenario):
    def weight(u, v, data):
        # networkx passes the edge attribute dict for MultiDiGraph via the
        # parallel-edge minimum when using a plain function, so we handle keys
        # in the explicit per-edge cost builder below. Here we use the minimum
        # cost across parallel edges.
        return min(
            cost["travel_time_s"] * cost["multiplier"]
            for cost in _edge_costs(data, u, v, scenario)
        )

    return weight


def _edge_costs(data: dict, u, v, scenario: CongestionScenario):
    """Yield the per-key cost dicts for one (u, v) edge bundle."""
    # ``data`` for MultiDiGraph in networkx is a dict keyed by edge key.
    if data and all(isinstance(k, (int, str)) for k in data.keys()):
        candidates = data.items()
    else:  # pragma: no cover - single-graph fallback
        candidates = [(0, data)]
    for key, attrs in candidates:
        if not isinstance(attrs, dict):  # pragma: no cover - defensive
            continue
        length = float(attrs.get("length_m", attrs.get("length", 1.0)) or 1.0)
        speed = float(attrs.get("speed_kph") or _edge_speed_kph(attrs))
        travel_time = length / (speed * 1000.0 / 3600.0)
        multiplier = scenario.multiplier(u, v, key, attrs)
        yield {
            "key": key,
            "length_m": length,
            "speed_kph": speed,
            "travel_time_s": travel_time,
            "multiplier": multiplier,
            "cost_s": travel_time * multiplier,
        }


def _best_edge(data: dict, u, v, scenario: CongestionScenario, weight: str = "cost_s"):
    """Pick the cheapest parallel edge between u and v for the given metric."""
    costs = list(_edge_costs(data, u, v, scenario))
    if not costs:
        return None
    return min(costs, key=lambda c: c[weight])


def compute_path_metrics(
    graph, path: list, scenario: CongestionScenario, *, label: str
) -> dict[str, Any]:
    """Summarise a node path: distance, free-flow time, congestion-aware time."""
    distance_m = 0.0
    free_flow_s = 0.0
    congested_s = 0.0
    multipliers: list[float] = []
    classes: dict[str, int] = {}

    for u, v in zip(path[:-1], path[1:]):
        data = graph.get_edge_data(u, v)
        if not data:
            continue
        best = _best_edge(data, u, v, scenario)
        if best is None:
            continue
        distance_m += best["length_m"]
        free_flow_s += best["travel_time_s"]
        congested_s += best["cost_s"]
        multipliers.append(best["multiplier"])
        highway = _first_highway(data)
        classes[highway] = classes.get(highway, 0) + 1

    geometry = _path_geometry(graph, path)
    return {
        "label": label,
        "node_count": len(path),
        "distance_km": round(distance_m / 1000.0, 3),
        "free_flow_time_minutes": round(free_flow_s / 60.0, 2),
        "congestion_aware_time_minutes": round(congested_s / 60.0, 2),
        "mean_congestion_multiplier": (
            round(sum(multipliers) / len(multipliers), 3) if multipliers else 1.0
        ),
        "max_congestion_multiplier": (
            round(max(multipliers), 3) if multipliers else 1.0
        ),
        "highway_mix": classes,
        "geometry": geometry,
    }


def _first_highway(data: dict) -> str:
    for attrs in data.values():
        if not isinstance(attrs, dict):
            continue
        highway = attrs.get("highway")
        if isinstance(highway, list):
            highway = highway[0] if highway else None
        if highway:
            return str(highway)
    return "unknown"


def _path_geometry(graph, path: list) -> list[list[float]]:
    """Return [[lat, lon], ...] coordinates for a node path."""
    coordinates: list[list[float]] = []
    for node in path:
        data = graph.nodes.get(node, {})
        lat, lon = data.get("y"), data.get("x")
        if lat is None or lon is None:
            continue
        coordinates.append([round(float(lat), 6), round(float(lon), 6)])
    return coordinates


def _shortest_path(graph, source, target, scenario: CongestionScenario, weight: str):
    def edge_weight(u, v, data):
        costs = list(_edge_costs(data, u, v, scenario))
        if not costs:
            return float("inf")
        return min(c[weight] for c in costs)

    try:
        return nx.shortest_path(graph, source, target, weight=edge_weight)
    except nx.NetworkXNoPath as exc:
        raise RouteNotFoundError(
            "No road route connects the selected origin and destination. "
            "Try points closer together or inside the configured study area."
        ) from exc
    except nx.NodeNotFound as exc:  # pragma: no cover - defensive
        raise RoutingError(f"Graph node not found: {exc}") from exc


def _alternative_paths(
    graph, source, target, scenario: CongestionScenario, *, k: int = 4
) -> list[list]:
    """Enumerate up to ``k`` distinct simple paths by congestion-aware cost."""
    def edge_weight(u, v, data):
        costs = list(_edge_costs(data, u, v, scenario))
        if not costs:
            return float("inf")
        return min(c["cost_s"] for c in costs)

    paths: list[list] = []
    try:
        for path in nx.shortest_simple_paths(graph, source, target, weight=edge_weight):
            paths.append(path)
            if len(paths) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return paths
    except Exception:  # pragma: no cover - generator can raise on odd graphs
        return paths
    return paths


def plan_routes(
    graph,
    origin: tuple[float, float],
    destination: tuple[float, float],
    *,
    scenario: CongestionScenario | None = None,
    max_alternatives: int = 3,
) -> dict[str, Any]:
    """Compute distance-optimal, time-optimal and congestion-aware routes.

    ``origin`` and ``destination`` are ``(latitude, longitude)`` pairs.
    """
    scenario = scenario or simulated_peak_scenario()
    source = nearest_node(graph, *origin)
    target = nearest_node(graph, *destination)
    if source == target:
        raise RoutingError(
            "Origin and destination snap to the same road-network node. "
            "Choose points that are further apart."
        )

    candidates: list[dict[str, Any]] = []
    seen: set[tuple] = set()

    def add(path: list, label: str) -> None:
        key = tuple(path)
        if key in seen or len(path) < 2:
            return
        seen.add(key)
        candidates.append(compute_path_metrics(graph, path, scenario, label=label))

    # 1. Shortest distance.
    add(_shortest_path(graph, source, target, scenario, "length_m"), "shortest_distance")
    # 2. Fastest in free-flow conditions.
    add(
        _shortest_path(graph, source, target, scenario, "travel_time_s"),
        "fastest_free_flow",
    )
    # 3. Congestion-aware optimum.
    add(
        _shortest_path(graph, source, target, scenario, "cost_s"),
        "congestion_aware_fastest",
    )
    # 4. Additional distinct alternatives.
    for index, path in enumerate(
        _alternative_paths(graph, source, target, scenario, k=max_alternatives + 3)
    ):
        add(path, f"alternative_{index + 1}")

    # Rank: congestion-aware optimum first, then by congestion-aware time.
    candidates.sort(
        key=lambda c: (
            0 if c["label"] == "congestion_aware_fastest" else 1,
            c["congestion_aware_time_minutes"],
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    origin_node = graph.nodes[source]
    target_node = graph.nodes[target]
    return {
        "scenario": scenario.describe(),
        "origin_snap": {
            "requested": [origin[0], origin[1]],
            "node": [float(origin_node["y"]), float(origin_node["x"])],
            "node_id": int(source) if isinstance(source, int) else str(source),
        },
        "destination_snap": {
            "requested": [destination[0], destination[1]],
            "node": [float(target_node["y"]), float(target_node["x"])],
            "node_id": int(target) if isinstance(target, int) else str(target),
        },
        "routes": candidates,
        "graph": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
        },
        "attribution": "Road data (c) OpenStreetMap contributors, ODbL.",
        "limitations": [
            "Travel times use OpenStreetMap speed limits, which may be missing or "
            "out of date; they are estimates, not measurements.",
            "Congestion multipliers in the default scenario are SIMULATED for "
            "demonstration and are not measured traffic counts.",
            "Origins and destinations snap to the nearest road node, so the route "
            "starts at the closest road rather than the exact point.",
            "The traffic-volume ML model was trained on one Minnesota interstate "
            "sensor and is NOT used to set per-road congestion here.",
        ],
    }
