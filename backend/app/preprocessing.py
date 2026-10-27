"""Data loading, validation, cleaning and feature engineering for TrafficSense.

The module works with the UCI *Metro Interstate Traffic Volume* dataset
(https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume).

Design rules enforced here (see README section "Data leakage"):
  * ``traffic_volume`` is the ONLY label. No lagged/rolling traffic features are
    used, because they would leak future information into a real-time forecast.
  * Every timestamp feature is derived from the observation timestamp itself.
  * Cleaning is deterministic and idempotent so the pipeline is reproducible.
"""

from __future__ import annotations

import gzip
import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TARGET_COLUMN = "traffic_volume"
DATETIME_COLUMN = "date_time"

#: Raw column names expected from the UCI export (already lower snake case).
EXPECTED_RAW_COLUMNS = [
    "holiday",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "weather_main",
    "weather_description",
    DATETIME_COLUMN,
    TARGET_COLUMN,
]

#: Numeric predictors fed to the model.
NUMERIC_FEATURES = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]

#: Categorical predictors fed to the model.
CATEGORICAL_FEATURES = ["weather_main", "holiday"]

#: Column order used for inference. Must never change without retraining.
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Human-readable feature documentation (units + meaning).
FEATURE_DEFINITIONS: dict[str, dict[str, str]] = {
    "temp": {
        "description": "Air temperature at the sensor",
        "unit": "kelvin",
    },
    "rain_1h": {
        "description": "Rainfall accumulated in the preceding hour",
        "unit": "millimetres",
    },
    "snow_1h": {
        "description": "Snowfall accumulated in the preceding hour",
        "unit": "millimetres",
    },
    "clouds_all": {
        "description": "Cloud cover",
        "unit": "percent (0-100)",
    },
    "weather_main": {
        "description": "Short categorical weather label from the source feed",
        "unit": "category",
    },
    "holiday": {
        "description": "US national holiday name, or 'None' for an ordinary day",
        "unit": "category",
    },
    "hour": {"description": "Hour of day extracted from date_time", "unit": "0-23"},
    "day_of_week": {
        "description": "Day of week, Monday=0 ... Sunday=6",
        "unit": "0-6",
    },
    "month": {"description": "Calendar month", "unit": "1-12"},
    "is_weekend": {
        "description": "1 when the observation falls on Saturday or Sunday",
        "unit": "0/1",
    },
    "hour_sin": {
        "description": "Cyclical encoding of the hour of day (sine)",
        "unit": "-1..1",
    },
    "hour_cos": {
        "description": "Cyclical encoding of the hour of day (cosine)",
        "unit": "-1..1",
    },
    "dow_sin": {
        "description": "Cyclical encoding of the day of week (sine)",
        "unit": "-1..1",
    },
    "dow_cos": {
        "description": "Cyclical encoding of the day of week (cosine)",
        "unit": "-1..1",
    },
    "month_sin": {
        "description": "Cyclical encoding of the month (sine)",
        "unit": "-1..1",
    },
    "month_cos": {
        "description": "Cyclical encoding of the month (cosine)",
        "unit": "-1..1",
    },
    TARGET_COLUMN: {
        "description": "Hourly traffic volume recorded by the sensor (label)",
        "unit": "vehicles per hour",
    },
}

#: Accepted weather categories (source vocabulary, lower case).
WEATHER_CATEGORIES = [
    "clear",
    "clouds",
    "drizzle",
    "fog",
    "haze",
    "mist",
    "rain",
    "smoke",
    "snow",
    "squall",
    "thunderstorm",
]

#: Congestion categories, ordered from lightest to heaviest.
CONGESTION_CATEGORIES = ["Low", "Moderate", "High", "Severe"]


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, unreadable or structurally invalid."""


@dataclass(frozen=True)
class DatasetBundle:
    """Container describing a cleaned, feature-engineered dataset."""

    frame: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    source: str

    @property
    def periods(self) -> dict[str, dict[str, str | int]]:
        return {
            "train": _period(self.train),
            "validation": _period(self.validation),
            "test": _period(self.test),
        }


def _period(frame: pd.DataFrame) -> dict[str, str | int]:
    if frame.empty:
        return {"start": None, "end": None, "rows": 0}
    return {
        "start": str(frame[DATETIME_COLUMN].min()),
        "end": str(frame[DATETIME_COLUMN].max()),
        "rows": int(len(frame)),
    }


# --------------------------------------------------------------------------- #
# Locating / downloading the raw file
# --------------------------------------------------------------------------- #

DATASET_FILENAME = "Metro_Interstate_Traffic_Volume.csv"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/492/"
    "metro+interstate+traffic+volume.zip"
)
UCI_ARCHIVE_MEMBER = "Metro_Interstate_Traffic_Volume.csv.gz"


def default_data_dir() -> Path:
    """Return ``backend/data`` regardless of the current working directory."""
    return Path(__file__).resolve().parents[1] / "data"


def find_dataset_file(data_dir: str | os.PathLike | None = None) -> Path | None:
    """Return the first readable dataset file inside ``data_dir``, else ``None``.

    Supported names: the canonical CSV, a ``.csv.gz`` copy, or the raw UCI
    zip archive.
    """
    directory = Path(data_dir) if data_dir is not None else default_data_dir()
    if not directory.exists():
        return None
    candidates = [
        directory / DATASET_FILENAME,
        directory / f"{DATASET_FILENAME}.gz",
        directory / "metro+interstate+traffic+volume.zip",
        directory / "metro.zip",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    # Fall back to any csv in the directory so a student can drop in their copy.
    for candidate in sorted(directory.glob("*.csv")):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def download_dataset(destination: str | os.PathLike | None = None) -> Path:
    """Download the dataset from UCI and store it as an uncompressed CSV.

    Raises :class:`DatasetError` with actionable guidance when the download
    fails, so the API can surface a clear setup message instead of crashing.
    """
    import urllib.error
    import urllib.request

    directory = Path(destination) if destination is not None else default_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / DATASET_FILENAME

    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=60) as response:  # noqa: S310
            payload = response.read()
    except (urllib.error.URLError, OSError, TimeoutError) as exc:  # pragma: no cover
        raise DatasetError(
            "Could not download the dataset from UCI "
            f"({UCI_ZIP_URL}): {exc}. "
            "Download it manually and place "
            f"'{DATASET_FILENAME}' inside {directory}."
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            with archive.open(UCI_ARCHIVE_MEMBER) as member:
                csv_bytes = gzip.decompress(member.read())
    except (zipfile.BadZipFile, KeyError, OSError) as exc:  # pragma: no cover
        raise DatasetError(
            f"The downloaded archive was not readable: {exc}. "
            f"Please place '{DATASET_FILENAME}' inside {directory} manually."
        ) from exc

    target.write_bytes(csv_bytes)
    return target


# --------------------------------------------------------------------------- #
# Loading + validation
# --------------------------------------------------------------------------- #


def _read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = next(
                (n for n in archive.namelist() if n.endswith((".csv", ".csv.gz"))),
                None,
            )
            if member is None:
                raise DatasetError(f"No CSV member found inside {path}.")
            with archive.open(member) as handle:
                raw = handle.read()
        if member.endswith(".gz"):
            raw = gzip.decompress(raw)
        return pd.read_csv(io.BytesIO(raw))
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case, strip and snake_case the column names."""
    renamed = {}
    for column in frame.columns:
        clean = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        renamed[column] = clean
    return frame.rename(columns=renamed)


def validate_columns(frame: pd.DataFrame) -> None:
    """Raise :class:`DatasetError` when mandatory columns are missing."""
    missing = [c for c in EXPECTED_RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise DatasetError(
            "Dataset is missing required columns: "
            f"{', '.join(missing)}. Found: {', '.join(map(str, frame.columns))}. "
            "Expected the UCI 'Metro Interstate Traffic Volume' export."
        )
    if len(frame) == 0:
        raise DatasetError("Dataset contains no rows.")


def load_raw_dataset(
    path: str | os.PathLike | None = None,
    *,
    allow_download: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load the raw dataset from disk (optionally downloading it first).

    Returns the normalised raw frame plus a short description of its origin.
    """
    resolved = find_dataset_file(path) if path is None else Path(path)
    source = "local file"
    if resolved is None:
        if not allow_download:
            raise DatasetError(_missing_dataset_message())
        resolved = download_dataset()
        source = "downloaded from UCI"

    if not resolved.is_file():
        raise DatasetError(
            f"Dataset file not found at {resolved}. {_missing_dataset_message()}"
        )

    frame = normalize_columns(_read_csv(resolved))
    validate_columns(frame)
    return frame, f"{source}: {resolved.name}"


def _missing_dataset_message() -> str:
    directory = default_data_dir()
    return (
        "The traffic dataset is not available. "
        f"Download the UCI 'Metro Interstate Traffic Volume' dataset and save it as "
        f"'{directory / DATASET_FILENAME}', or run:\n"
        f"    python -m app.train --download\n"
        f"Manual source: {UCI_ZIP_URL}"
    )


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse timestamps, preferring the dataset's ISO-like format.

    Falling back to the per-element parser keeps odd exports working while
    avoiding a noisy warning on the common case.
    """
    parsed = pd.to_datetime(series, errors="coerce", format="ISO8601")
    unparsed = parsed.isna() & series.notna()
    if unparsed.any():
        parsed = parsed.where(
            ~unparsed, pd.to_datetime(series[unparsed], errors="coerce")
        )
    return parsed


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalise the raw frame. Deterministic and order independent."""
    df = normalize_columns(frame.copy())
    validate_columns(df)

    # Timestamps -> datetime, dropping rows we cannot place in time.
    df[DATETIME_COLUMN] = _parse_timestamps(df[DATETIME_COLUMN])
    df = df.dropna(subset=[DATETIME_COLUMN])

    # Numeric coercion. Bad strings become NaN and are imputed later by the
    # scikit-learn pipeline (median imputation fitted on the training split).
    for column in ("temp", "rain_1h", "snow_1h", "clouds_all", TARGET_COLUMN):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # The label must exist; predictor NaNs are allowed and handled downstream.
    df = df.dropna(subset=[TARGET_COLUMN])

    # Physical plausibility: the target and weather measures cannot be negative.
    df = df[df[TARGET_COLUMN] >= 0]
    for column in ("rain_1h", "snow_1h"):
        df[column] = df[column].clip(lower=0)
    df["clouds_all"] = df["clouds_all"].clip(lower=0, upper=100)

    # Text normalisation for categorical columns.
    df["weather_main"] = (
        df["weather_main"].astype("string").str.strip().str.lower().fillna("unknown")
    )
    df["weather_description"] = (
        df["weather_description"]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("unknown")
    )
    # The source uses the literal string "None" for ordinary (non-holiday) days.
    df["holiday"] = df["holiday"].astype("string").str.strip().fillna("None")
    df.loc[df["holiday"].str.lower().isin(["nan", "none", ""]), "holiday"] = "None"

    # Remove exact duplicate records, keeping the first chronological copy.
    df = df.sort_values(DATETIME_COLUMN)
    df = df.drop_duplicates()

    # One observation per timestamp keeps the hourly series well defined.
    df = df.drop_duplicates(subset=[DATETIME_COLUMN], keep="first")

    df = df.reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclical features derived from the timestamp."""
    df = frame.copy()
    stamp = df[DATETIME_COLUMN]
    df["hour"] = stamp.dt.hour.astype(int)
    df["day_of_week"] = stamp.dt.dayofweek.astype(int)
    df["month"] = stamp.dt.month.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

    return df


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def chronological_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered frame into train / validation / test by position.

    Random shuffling is deliberately avoided: shuffling a time series leaks
    future information into the training set.
    """
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Split fractions must lie strictly between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1.")

    ordered = frame.sort_values(DATETIME_COLUMN).reset_index(drop=True)
    n_rows = len(ordered)
    if n_rows < 30:
        raise DatasetError(
            f"Need at least 30 chronological rows to train, found {n_rows}."
        )

    n_test = max(1, int(round(n_rows * test_fraction)))
    n_validation = max(1, int(round(n_rows * validation_fraction)))
    n_train = n_rows - n_validation - n_test
    if n_train < 10:
        raise DatasetError(
            "Training split would be too small; reduce the split fractions."
        )

    train = ordered.iloc[:n_train].reset_index(drop=True)
    validation = ordered.iloc[n_train : n_train + n_validation].reset_index(drop=True)
    test = ordered.iloc[n_train + n_validation :].reset_index(drop=True)
    return train, validation, test


def build_dataset(
    path: str | os.PathLike | None = None,
    *,
    allow_download: bool = True,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> DatasetBundle:
    """Full pipeline: load -> clean -> engineer features -> chronological split."""
    raw, source = load_raw_dataset(path, allow_download=allow_download)
    cleaned = clean_dataset(raw)
    featured = engineer_features(cleaned)
    train, validation, test = chronological_split(
        featured,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    return DatasetBundle(
        frame=featured,
        train=train,
        validation=validation,
        test=test,
        source=source,
    )


# --------------------------------------------------------------------------- #
# Congestion thresholds
# --------------------------------------------------------------------------- #


def derive_congestion_thresholds(train_target: Iterable[float]) -> dict[str, float]:
    """Derive Low/Moderate/High/Severe cut-offs from TRAINING data only.

    The three cut-offs are the 25th, 60th and 85th percentiles of the training
    traffic volume. Deriving them from training data (never from the test set)
    keeps the evaluation honest and reproducible.
    """
    values = pd.Series(list(train_target), dtype="float64").dropna()
    if values.empty:
        raise ValueError("Cannot derive congestion thresholds from empty data.")
    q25, q60, q85 = np.percentile(values, [25, 60, 85])
    return {
        "low_max": float(q25),
        "moderate_max": float(q60),
        "high_max": float(q85),
    }


def classify_volume(
    volume: float, thresholds: dict[str, float]
) -> str:
    """Map a volume to a congestion category using training-derived cut-offs."""
    if volume <= thresholds["low_max"]:
        return "Low"
    if volume <= thresholds["moderate_max"]:
        return "Moderate"
    if volume <= thresholds["high_max"]:
        return "High"
    return "Severe"
