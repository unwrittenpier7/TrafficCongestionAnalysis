"""Model evaluation helpers for TrafficSense.

Everything in this module computes numbers from *real* predictions. Nothing is
hardcoded or pre-filled: if training has not run, the API reports "no metrics"
rather than inventing values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from .preprocessing import CONGESTION_CATEGORIES, classify_volume

# Matplotlib is imported lazily so the API can start without it installed.
try:  # pragma: no cover - import guard
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE and R^2 for a regression problem."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        raise ValueError("Cannot evaluate on an empty target array.")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_report_from_volumes(
    y_true_volume,
    y_pred_volume,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Classify volumes with the shared thresholds and report P/R/F1 + matrix."""
    y_true = [classify_volume(v, thresholds) for v in np.asarray(y_true_volume, float)]
    y_pred = [classify_volume(v, thresholds) for v in np.asarray(y_pred_volume, float)]

    labels = CONGESTION_CATEGORIES
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    return {
        "labels": labels,
        "per_class": per_class,
        "macro": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "confusion_matrix": matrix,
    }


def actual_vs_predicted_series(
    timestamps, y_true, y_pred, *, max_points: int = 600
) -> dict[str, list]:
    """Build a down-sampled actual-vs-predicted series for the dashboard chart."""
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(list(timestamps)),
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )
    if len(frame) > max_points:
        step = int(np.ceil(len(frame) / max_points))
        frame = frame.iloc[::step]
    return {
        "timestamps": frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "actual": frame["actual"].round(2).tolist(),
        "predicted": frame["predicted"].round(2).tolist(),
    }


def traffic_pattern_by_hour(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Mean/median traffic volume per hour of day, computed from the dataset."""
    summary = (
        frame.groupby("hour")["traffic_volume"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    return [
        {
            "hour": int(row["hour"]),
            "mean": round(float(row["mean"]), 2),
            "median": round(float(row["median"]), 2),
            "count": int(row["count"]),
        }
        for _, row in summary.iterrows()
    ]


def traffic_pattern_by_weekday(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Mean traffic volume per day of week, computed from the dataset."""
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = frame.groupby("day_of_week")["traffic_volume"].agg(["mean", "count"])
    out = []
    for day, row in grouped.iterrows():
        out.append(
            {
                "day_of_week": int(day),
                "day_name": names[int(day)],
                "mean": round(float(row["mean"]), 2),
                "count": int(row["count"]),
            }
        )
    return out


def extract_feature_importance(pipeline, feature_names: list[str]) -> list[dict[str, Any]] | None:
    """Pull feature importances out of a fitted scikit-learn pipeline."""
    model = None
    if hasattr(pipeline, "named_steps"):
        model = pipeline.named_steps.get("model")
    else:  # pragma: no cover - defensive
        model = pipeline

    if model is None or not hasattr(model, "feature_importances_"):
        return None

    # Recover the expanded feature names produced by the ColumnTransformer.
    names: list[str] = []
    if hasattr(pipeline, "named_steps"):
        pre = pipeline.named_steps.get("preprocess")
        if pre is not None and hasattr(pre, "get_feature_names_out"):
            try:
                names = [str(n) for n in pre.get_feature_names_out()]
            except Exception:  # pragma: no cover - defensive
                names = []
    if len(names) != len(model.feature_importances_):
        names = feature_names

    pairs = sorted(
        zip(names, (float(v) for v in model.feature_importances_)),
        key=lambda p: p[1],
        reverse=True,
    )
    return [{"feature": name, "importance": round(value, 6)} for name, value in pairs]


def save_actual_vs_predicted_plot(
    timestamps, y_true, y_pred, output_path: str | Path
) -> str | None:
    """Render an actual-vs-predicted PNG. Returns the path, or None if skipped."""
    if plt is None:  # pragma: no cover - optional dependency
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(list(timestamps)),
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    ).sort_values("timestamp")
    if len(frame) > 800:
        step = int(np.ceil(len(frame) / 800))
        frame = frame.iloc[::step]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=130)
    axes[0].plot(frame["timestamp"], frame["actual"], label="Actual", lw=1.1)
    axes[0].plot(frame["timestamp"], frame["predicted"], label="Predicted", lw=1.1, alpha=0.85)
    axes[0].set_title("Actual vs predicted traffic volume (test period)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Vehicles per hour")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=30, labelsize=7)

    axes[1].scatter(frame["actual"], frame["predicted"], s=6, alpha=0.35, color="#2563eb")
    lims = [
        float(min(frame["actual"].min(), frame["predicted"].min())),
        float(max(frame["actual"].max(), frame["predicted"].max())),
    ]
    axes[1].plot(lims, lims, "r--", lw=1)
    axes[1].set_title("Predicted vs actual (ideal line dashed)")
    axes[1].set_xlabel("Actual volume")
    axes[1].set_ylabel("Predicted volume")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> str:
    """Write metrics to JSON with stable formatting."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=False), encoding="utf-8")
    return str(output)
