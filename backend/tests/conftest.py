"""Shared pytest fixtures for the TrafficSense backend.

Tests never touch the live OpenStreetMap download or the real 48k-row dataset:
they use a small deterministic synthetic frame and a tiny hand-built graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def make_synthetic_frame(rows: int = 480, start: str = "2020-01-01 00:00:00") -> pd.DataFrame:
    """Deterministic hourly traffic frame with a realistic daily/weekly shape."""
    stamps = pd.date_range(start=start, periods=rows, freq="h")
    hours = stamps.hour.to_numpy()
    dows = stamps.dayofweek.to_numpy()

    # Two daily peaks on weekdays, flatter at weekends.
    base = 3000.0
    daily = 1500.0 * np.sin(2 * np.pi * (hours - 7) / 24.0)
    weekend = np.where(dows >= 5, -900.0, 0.0)
    noise = np.linspace(-40, 40, rows)
    volume = np.clip(base + daily + weekend + noise, 0, 9000)

    return pd.DataFrame(
        {
            "holiday": ["None"] * rows,
            "temp": 273.15 + 12 * np.sin(2 * np.pi * np.arange(rows) / 24.0),
            "rain_1h": np.zeros(rows),
            "snow_1h": np.zeros(rows),
            "clouds_all": np.full(rows, 40.0),
            "weather_main": ["Clouds"] * rows,
            "weather_description": ["scattered clouds"] * rows,
            "date_time": stamps.astype(str),
            "traffic_volume": volume,
        }
    )


@pytest.fixture()
def synthetic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "Metro_Interstate_Traffic_Volume.csv"
    make_synthetic_frame().to_csv(path, index=False)
    return path


@pytest.fixture()
def tiny_graph():
    """A small directed road graph with three distinct origin->target routes."""
    import networkx as nx

    graph = nx.MultiDiGraph()

    # Nodes laid out as a grid so geometries are meaningful.
    coords = {
        "A": (17.4400, 78.3700),
        "B": (17.4400, 78.3800),
        "C": (17.4500, 78.3700),
        "D": (17.4500, 78.3800),
        "E": (17.4450, 78.3750),
    }
    for node, (lat, lon) in coords.items():
        graph.add_node(node, y=lat, x=lon)

    def add(u, v, length_m, highway, speed_kph, key=0):
        graph.add_edge(
            u,
            v,
            key=key,
            length=length_m,
            length_m=length_m,
            highway=highway,
            speed_kph=speed_kph,
            travel_time_s=length_m / (speed_kph * 1000.0 / 3600.0),
        )

    # Route 1: A -> C -> D  (short but a congested arterial)
    add("A", "C", 900.0, "primary", 50.0)
    add("C", "D", 900.0, "primary", 50.0)
    # Route 2: A -> B -> D  (longer but motorway)
    add("A", "B", 1400.0, "motorway", 100.0)
    add("B", "D", 1400.0, "motorway", 100.0)
    # Route 3: A -> E -> D  (medium residential)
    add("A", "E", 1100.0, "residential", 30.0)
    add("E", "D", 1100.0, "residential", 30.0)

    return graph


@pytest.fixture()
def trained_artifacts(tmp_path: Path, synthetic_csv: Path) -> Path:
    """Train a small real model into a temporary artifacts directory."""
    from app.train import train

    artifacts = tmp_path / "artifacts"
    train(
        data_path=synthetic_csv,
        artifacts_dir=artifacts,
        allow_download=False,
    )
    return artifacts
