"""Artifact paths and helpers for loading the trained TrafficSense model.

Model artifacts are *generated*, never committed. If they are missing the API
must degrade gracefully instead of crashing, which is why every loader here
returns a structured result rather than raising on absence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

MODEL_FILENAME = "traffic_model.joblib"
METADATA_FILENAME = "model_metadata.json"
METRICS_FILENAME = "metrics.json"
PLOT_FILENAME = "actual_vs_predicted.png"


def default_artifacts_dir() -> Path:
    """Return ``backend/artifacts`` regardless of the current directory."""
    return Path(__file__).resolve().parents[1] / "artifacts"


def artifact_paths(artifacts_dir: str | Path | None = None) -> dict[str, Path]:
    directory = Path(artifacts_dir) if artifacts_dir is not None else default_artifacts_dir()
    return {
        "dir": directory,
        "model": directory / MODEL_FILENAME,
        "metadata": directory / METADATA_FILENAME,
        "metrics": directory / METRICS_FILENAME,
        "plot": directory / PLOT_FILENAME,
    }


@dataclass
class ModelBundle:
    """A loaded model pipeline together with its metadata."""

    pipeline: Any
    metadata: dict[str, Any]

    @property
    def thresholds(self) -> dict[str, float]:
        return self.metadata["congestion_thresholds"]

    @property
    def model_name(self) -> str:
        return self.metadata.get("selected_model", "unknown")


def load_metadata(artifacts_dir: str | Path | None = None) -> dict[str, Any] | None:
    path = artifact_paths(artifacts_dir)["metadata"]
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_metrics(artifacts_dir: str | Path | None = None) -> dict[str, Any] | None:
    path = artifact_paths(artifacts_dir)["metrics"]
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_model_bundle(artifacts_dir: str | Path | None = None) -> ModelBundle | None:
    """Load the fitted pipeline and metadata, or return ``None`` when absent."""
    paths = artifact_paths(artifacts_dir)
    if not paths["model"].is_file():
        return None
    metadata = load_metadata(artifacts_dir)
    if metadata is None:
        return None
    try:
        pipeline = joblib.load(paths["model"])
    except Exception:
        return None
    return ModelBundle(pipeline=pipeline, metadata=metadata)
