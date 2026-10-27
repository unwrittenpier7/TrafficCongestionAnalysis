"""Train and evaluate the TrafficSense traffic-volume models.

Run from the ``backend`` directory:

    python -m app.train                # uses backend/data/Metro_Interstate_Traffic_Volume.csv
    python -m app.train --download     # downloads the dataset from UCI first

The script produces, inside ``backend/artifacts``:
  * ``traffic_model.joblib``  - the fitted scikit-learn pipeline
  * ``model_metadata.json``   - feature order, thresholds, split periods
  * ``metrics.json``          - real MAE / RMSE / R^2 (and classification report)
  * ``actual_vs_predicted.png`` - evaluation figure

All reported numbers come from real predictions on a chronological holdout.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import __version__
from .evaluation import (
    actual_vs_predicted_series,
    classification_report_from_volumes,
    extract_feature_importance,
    regression_metrics,
    save_actual_vs_predicted_plot,
    save_metrics_json,
    traffic_pattern_by_hour,
    traffic_pattern_by_weekday,
)
from .preprocessing import (
    CATEGORICAL_FEATURES,
    DATETIME_COLUMN,
    FEATURE_DEFINITIONS,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    DatasetBundle,
    build_dataset,
    classify_volume,
    derive_congestion_thresholds,
)
from .model_store import artifact_paths

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    """Median-impute numerics, mode-impute + one-hot encode categoricals."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models() -> dict[str, object]:
    """Return the models to compare, including an optional XGBoost entry."""
    models: dict[str, object] = {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "random_forest": RandomForestRegressor(
            n_estimators=220,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    try:  # Optional dependency: only used when it imports cleanly.
        from xgboost import XGBRegressor  # type: ignore

        models["xgboost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.06,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
    except Exception:
        pass
    return models


def _fit_and_score(
    name: str,
    estimator,
    bundle: DatasetBundle,
    thresholds: dict[str, float],
) -> tuple[Pipeline, dict, np.ndarray, float]:
    """Fit one estimator on train and score it on the chronological validation set."""
    pipeline = Pipeline(
        steps=[("preprocess", build_preprocessor()), ("model", estimator)]
    )
    started = time.perf_counter()
    pipeline.fit(bundle.train[MODEL_FEATURES], bundle.train[TARGET_COLUMN])
    fit_seconds = time.perf_counter() - started

    y_val = bundle.validation[TARGET_COLUMN].to_numpy(dtype=float)
    y_val_pred = np.asarray(pipeline.predict(bundle.validation[MODEL_FEATURES]), dtype=float)
    metrics = regression_metrics(y_val, y_val_pred)
    metrics["fit_seconds"] = round(fit_seconds, 3)
    metrics["validation_classification"] = classification_report_from_volumes(
        y_val, y_val_pred, thresholds
    )
    return pipeline, metrics, y_val_pred, fit_seconds


def train(
    *,
    data_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    allow_download: bool = True,
    report_dir: str | Path | None = None,
) -> dict:
    """Execute the full training + evaluation workflow and persist artifacts."""
    paths = artifact_paths(artifacts_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("TrafficSense - training pipeline")
    print("=" * 72)

    bundle = build_dataset(data_path, allow_download=allow_download)
    print(f"Dataset source : {bundle.source}")
    print(f"Rows (clean)   : {len(bundle.frame):,}")
    print(f"Train/Val/Test : {len(bundle.train):,} / {len(bundle.validation):,} / {len(bundle.test):,}")

    # Thresholds come from the TRAINING split only - never from validation/test.
    thresholds = derive_congestion_thresholds(bundle.train[TARGET_COLUMN])
    print(
        "Congestion thresholds (training-derived): "
        f"Low<= {thresholds['low_max']:.0f} < Moderate<= {thresholds['moderate_max']:.0f} "
        f"< High<= {thresholds['high_max']:.0f} < Severe"
    )

    models = candidate_models()
    validation_results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}

    for name, estimator in models.items():
        print(f"\n-- Fitting {name} ...")
        pipeline, metrics, _, _ = _fit_and_score(name, estimator, bundle, thresholds)
        validation_results[name] = metrics
        fitted[name] = pipeline
        print(
            f"   validation MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  "
            f"R2={metrics['r2']:.4f}  ({metrics['fit_seconds']:.1f}s)"
        )

    # Select the best *non-baseline* model by validation MAE.
    selectable = {k: v for k, v in validation_results.items() if k != "baseline_mean"}
    selected_name = min(selectable, key=lambda k: selectable[k]["mae"])
    selected_pipeline = fitted[selected_name]
    print(f"\nSelected model : {selected_name} (lowest validation MAE)")

    # Refit the selected model on train + validation for the final test evaluation.
    train_plus_val = pd.concat([bundle.train, bundle.validation], ignore_index=True)
    final_pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", candidate_models()[selected_name]),
        ]
    )
    final_pipeline.fit(train_plus_val[MODEL_FEATURES], train_plus_val[TARGET_COLUMN])

    y_test = bundle.test[TARGET_COLUMN].to_numpy(dtype=float)
    y_test_pred = np.asarray(
        final_pipeline.predict(bundle.test[MODEL_FEATURES]), dtype=float
    )
    test_metrics = regression_metrics(y_test, y_test_pred)

    # Baseline on the same test rows, for an honest comparison.
    baseline_pipeline = fitted["baseline_mean"]
    y_base_pred = np.asarray(
        baseline_pipeline.predict(bundle.test[MODEL_FEATURES]), dtype=float
    )
    baseline_test_metrics = regression_metrics(y_test, y_base_pred)

    classification = classification_report_from_volumes(y_test, y_test_pred, thresholds)
    baseline_classification = classification_report_from_volumes(
        y_test, y_base_pred, thresholds
    )

    feature_importance = extract_feature_importance(final_pipeline, MODEL_FEATURES)

    print("\nTest-set results (chronological holdout)")
    print(f"  Baseline  MAE={baseline_test_metrics['mae']:.2f}  "
          f"RMSE={baseline_test_metrics['rmse']:.2f}  R2={baseline_test_metrics['r2']:.4f}")
    print(f"  {selected_name:9s} MAE={test_metrics['mae']:.2f}  "
          f"RMSE={test_metrics['rmse']:.2f}  R2={test_metrics['r2']:.4f}")
    print(f"  Macro F1 (categories) = {classification['macro']['f1']:.4f}")

    # ---- persist artifacts -------------------------------------------------
    # Compression keeps the Random Forest artifact manageable (~70 MB vs ~300 MB).
    joblib.dump(final_pipeline, paths["model"], compress=3)

    metadata = {
        "project": "TrafficSense",
        "app_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "dataset_source": bundle.source,
        "dataset_reference": "UCI ML Repository - Metro Interstate Traffic Volume (id 492)",
        "dataset_geography": (
            "Interstate 94, Minneapolis-St. Paul, Minnesota, USA (one sensor). "
            "Not representative of every road or city."
        ),
        "dataset_rows_clean": int(len(bundle.frame)),
        "features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_definitions": FEATURE_DEFINITIONS,
        "target": TARGET_COLUMN,
        "target_unit": "vehicles per hour",
        "congestion_thresholds": thresholds,
        "congestion_categories": ["Low", "Moderate", "High", "Severe"],
        "selected_model": selected_name,
        "candidate_models": list(models.keys()),
        "random_state": RANDOM_STATE,
        "split": {
            "strategy": "chronological (no shuffling)",
            "validation_fraction": round(len(bundle.validation) / len(bundle.frame), 4),
            "test_fraction": round(len(bundle.test) / len(bundle.frame), 4),
            "periods": bundle.periods,
            "refit_on": "train + validation",
        },
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    metrics_payload = {
        "generated_utc": metadata["created_utc"],
        "selected_model": selected_name,
        "test": test_metrics,
        "baseline_test": baseline_test_metrics,
        "validation": {
            name: {
                "mae": res["mae"],
                "rmse": res["rmse"],
                "r2": res["r2"],
                "fit_seconds": res["fit_seconds"],
            }
            for name, res in validation_results.items()
        },
        "classification_test": classification,
        "classification_baseline_test": baseline_classification,
        "feature_importance": feature_importance,
        "actual_vs_predicted": actual_vs_predicted_series(
            bundle.test[DATETIME_COLUMN], y_test, y_test_pred
        ),
        "traffic_by_hour": traffic_pattern_by_hour(bundle.frame),
        "traffic_by_weekday": traffic_pattern_by_weekday(bundle.frame),
        "train_mean_volume": float(bundle.train[TARGET_COLUMN].mean()),
        "test_mean_volume": float(bundle.test[TARGET_COLUMN].mean()),
    }
    save_metrics_json(metrics_payload, paths["metrics"])

    plot_path = save_actual_vs_predicted_plot(
        bundle.test[DATETIME_COLUMN], y_test, y_test_pred, paths["plot"]
    )
    if report_dir is not None and plot_path:
        destination = Path(report_dir) / "figures" / "actual_vs_predicted.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(plot_path).read_bytes())

    print(f"\nArtifacts written to {paths['dir']}")
    for key in ("model", "metadata", "metrics", "plot"):
        exists = "ok" if paths[key].is_file() else "missing"
        print(f"  {paths[key].name:28s} {exists}")
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the TrafficSense models.")
    parser.add_argument("--data", type=str, default=None, help="Path to the dataset CSV.")
    parser.add_argument("--artifacts", type=str, default=None, help="Artifact output dir.")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset from UCI when no local copy is found.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Also copy the evaluation figure into <report-dir>/figures.",
    )
    args = parser.parse_args(argv)

    try:
        train(
            data_path=args.data,
            artifacts_dir=args.artifacts,
            allow_download=args.download,
            report_dir=args.report_dir,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"\nTraining failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
