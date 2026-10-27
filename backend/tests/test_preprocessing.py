"""Tests for dataset validation, cleaning, feature engineering and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.preprocessing import (
    DATETIME_COLUMN,
    MODEL_FEATURES,
    TARGET_COLUMN,
    DatasetError,
    build_dataset,
    classify_volume,
    clean_dataset,
    derive_congestion_thresholds,
    engineer_features,
    find_dataset_file,
    load_raw_dataset,
    normalize_columns,
    validate_columns,
)
from conftest import make_synthetic_frame


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_normalize_columns_snake_cases_names():
    frame = pd.DataFrame(columns=["Date Time", "Traffic Volume", " rain_1h "])
    result = normalize_columns(frame)
    assert list(result.columns) == ["date_time", "traffic_volume", "rain_1h"]


def test_validate_columns_rejects_missing_columns():
    frame = pd.DataFrame({"traffic_volume": [1, 2]})
    with pytest.raises(DatasetError) as excinfo:
        validate_columns(frame)
    assert "missing required columns" in str(excinfo.value).lower()


def test_validate_columns_rejects_empty_frame():
    frame = pd.DataFrame(columns=["holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
                                  "weather_main", "weather_description", "date_time",
                                  "traffic_volume"])
    with pytest.raises(DatasetError):
        validate_columns(frame)


def test_find_dataset_file_returns_none_for_missing_dir(tmp_path):
    assert find_dataset_file(tmp_path / "does-not-exist") is None


def test_load_raw_dataset_raises_helpful_error_when_missing(tmp_path):
    with pytest.raises(DatasetError) as excinfo:
        load_raw_dataset(tmp_path, allow_download=False)
    message = str(excinfo.value)
    assert "dataset" in message.lower()
    assert "archive.ics.uci.edu" in message


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def test_clean_dataset_parses_timestamps_and_sorts():
    frame = make_synthetic_frame(rows=48)
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    cleaned = clean_dataset(shuffled)
    assert pd.api.types.is_datetime64_any_dtype(cleaned[DATETIME_COLUMN])
    assert cleaned[DATETIME_COLUMN].is_monotonic_increasing


def test_clean_dataset_removes_duplicate_timestamps():
    frame = make_synthetic_frame(rows=24)
    duplicated = pd.concat([frame, frame], ignore_index=True)
    cleaned = clean_dataset(duplicated)
    assert len(cleaned) == len(frame)
    assert cleaned[DATETIME_COLUMN].is_unique


def test_clean_dataset_drops_invalid_timestamps():
    frame = make_synthetic_frame(rows=12)
    frame.loc[0, DATETIME_COLUMN] = "not-a-date"
    cleaned = clean_dataset(frame)
    assert len(cleaned) == 11


def test_clean_dataset_coerces_bad_numeric_values():
    frame = make_synthetic_frame(rows=12)
    frame["temp"] = frame["temp"].astype(object)
    frame.loc[3, "temp"] = "unknown"
    cleaned = clean_dataset(frame)
    assert cleaned["temp"].isna().sum() == 1


def test_clean_dataset_clips_negative_rainfall_and_cloud_cover():
    frame = make_synthetic_frame(rows=6)
    frame.loc[0, "rain_1h"] = -5.0
    frame.loc[1, "clouds_all"] = 250.0
    cleaned = clean_dataset(frame)
    assert cleaned["rain_1h"].min() >= 0
    assert cleaned["clouds_all"].max() <= 100


def test_clean_dataset_normalises_none_holiday_to_literal_none():
    frame = make_synthetic_frame(rows=6)
    frame["holiday"] = [np.nan, "None", "none", "", "Christmas Day", "None"]
    cleaned = clean_dataset(frame)
    assert set(cleaned["holiday"]) <= {"None", "Christmas Day"}


def test_clean_dataset_lowercases_weather_category():
    frame = make_synthetic_frame(rows=4)
    frame["weather_main"] = ["Rain", "RAIN", " drizzle ", "Clear"]
    cleaned = clean_dataset(frame)
    assert set(cleaned["weather_main"]) == {"rain", "drizzle", "clear"}


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def test_engineer_features_adds_expected_columns():
    frame = engineer_features(clean_dataset(make_synthetic_frame(rows=48)))
    for column in ("hour", "day_of_week", "month", "is_weekend",
                   "hour_sin", "hour_cos", "dow_sin", "dow_cos",
                   "month_sin", "month_cos"):
        assert column in frame.columns


def test_engineer_features_hour_matches_timestamp():
    frame = engineer_features(clean_dataset(make_synthetic_frame(rows=48)))
    assert (frame["hour"] == frame[DATETIME_COLUMN].dt.hour).all()
    assert (frame["day_of_week"] == frame[DATETIME_COLUMN].dt.dayofweek).all()
    assert (frame["month"] == frame[DATETIME_COLUMN].dt.month).all()


def test_engineer_features_weekend_flag_is_correct():
    frame = engineer_features(clean_dataset(make_synthetic_frame(rows=240)))
    weekends = frame[frame["is_weekend"] == 1]
    assert set(weekends["day_of_week"]) <= {5, 6}
    weekdays = frame[frame["is_weekend"] == 0]
    assert set(weekdays["day_of_week"]) <= {0, 1, 2, 3, 4}


def test_cyclical_features_are_bounded_and_unit_circle():
    frame = engineer_features(clean_dataset(make_synthetic_frame(rows=240)))
    for pair in (("hour_sin", "hour_cos"), ("dow_sin", "dow_cos"),
                 ("month_sin", "month_cos")):
        magnitude = np.sqrt(frame[pair[0]] ** 2 + frame[pair[1]] ** 2)
        assert np.allclose(magnitude, 1.0, atol=1e-9)


def test_no_lag_or_rolling_traffic_features_present():
    """Guard against future-information leakage via lagged traffic features."""
    frame = engineer_features(clean_dataset(make_synthetic_frame(rows=48)))
    suspicious = [
        column
        for column in frame.columns
        if column != TARGET_COLUMN
        and ("lag" in column or "rolling" in column or "shift" in column)
    ]
    assert suspicious == []


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #


def test_chronological_split_preserves_order_and_sizes(synthetic_csv):
    bundle = build_dataset(synthetic_csv, allow_download=False)
    assert len(bundle.train) + len(bundle.validation) + len(bundle.test) == len(bundle.frame)
    assert bundle.train[DATETIME_COLUMN].max() < bundle.validation[DATETIME_COLUMN].min()
    assert bundle.validation[DATETIME_COLUMN].max() < bundle.test[DATETIME_COLUMN].min()


def test_chronological_split_rejects_invalid_fractions(synthetic_csv):
    bundle = build_dataset(synthetic_csv, allow_download=False)
    from app.preprocessing import chronological_split

    with pytest.raises(ValueError):
        chronological_split(bundle.frame, validation_fraction=0.6, test_fraction=0.6)


def test_build_dataset_reports_periods(synthetic_csv):
    bundle = build_dataset(synthetic_csv, allow_download=False)
    periods = bundle.periods
    assert periods["train"]["rows"] > 0
    assert periods["test"]["start"] > periods["train"]["end"]


# --------------------------------------------------------------------------- #
# Congestion thresholds
# --------------------------------------------------------------------------- #


def test_thresholds_are_ordered_and_derived_from_training_data():
    values = np.linspace(0, 1000, 1000)
    thresholds = derive_congestion_thresholds(values)
    assert thresholds["low_max"] < thresholds["moderate_max"] < thresholds["high_max"]
    # Percentile-based thresholds on a uniform ramp are close to the nominal ones.
    assert 200 < thresholds["low_max"] < 300
    assert 800 < thresholds["high_max"] < 900


def test_classify_volume_boundaries_are_inclusive_at_upper_edge():
    thresholds = {"low_max": 100.0, "moderate_max": 200.0, "high_max": 300.0}
    assert classify_volume(0.0, thresholds) == "Low"
    assert classify_volume(100.0, thresholds) == "Low"
    assert classify_volume(100.1, thresholds) == "Moderate"
    assert classify_volume(200.0, thresholds) == "Moderate"
    assert classify_volume(200.1, thresholds) == "High"
    assert classify_volume(300.0, thresholds) == "High"
    assert classify_volume(300.1, thresholds) == "Severe"


def test_model_features_include_numeric_and_categorical():
    assert "weather_main" in MODEL_FEATURES
    assert "holiday" in MODEL_FEATURES
    assert "hour" in MODEL_FEATURES
