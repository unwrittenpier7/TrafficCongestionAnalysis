"""Pydantic request/response models for the TrafficSense API.

Range validation here is intentionally strict: the API rejects physically
impossible inputs with a 422 and a readable message rather than silently
producing a nonsense prediction.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .preprocessing import WEATHER_CATEGORIES

CongestionCategory = Literal["Low", "Moderate", "High", "Severe"]
ScenarioName = Literal["free_flow", "simulated_peak", "custom"]


# --------------------------------------------------------------------------- #
# Service metadata
# --------------------------------------------------------------------------- #


class ServiceInfo(BaseModel):
    service: str = "TrafficSense API"
    version: str
    description: str
    documentation: str
    endpoints: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_available: bool
    model_name: str | None = None
    model_created_utc: str | None = None
    dataset_rows: int | None = None
    detail: str


class Thresholds(BaseModel):
    low_max: float
    moderate_max: float
    high_max: float


class FeatureDefinition(BaseModel):
    description: str
    unit: str


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #


class PredictionRequest(BaseModel):
    """Hourly traffic-volume prediction request.

    Temperature is accepted in kelvin (the dataset unit). Use the helper field
    ``temperature_unit`` to send Celsius instead; the API converts it.
    """

    hour: int = Field(..., ge=0, le=23, description="Hour of day, 0-23")
    day_of_week: int = Field(
        ..., ge=0, le=6, description="Day of week, Monday=0 ... Sunday=6"
    )
    month: int = Field(..., ge=1, le=12, description="Calendar month, 1-12")
    temperature: float = Field(
        ..., ge=-100, le=400, description="Air temperature (kelvin by default)"
    )
    temperature_unit: Literal["kelvin", "celsius"] = Field(
        default="kelvin", description="Unit of the temperature field"
    )
    rainfall_mm: float = Field(
        default=0.0, ge=0, le=500, description="Rainfall in the last hour (mm)"
    )
    snowfall_mm: float = Field(
        default=0.0, ge=0, le=500, description="Snowfall in the last hour (mm)"
    )
    cloud_cover: float = Field(
        default=0.0, ge=0, le=100, description="Cloud cover percentage (0-100)"
    )
    weather_category: str = Field(
        default="clear",
        description="Weather category, e.g. clear, clouds, rain, snow, mist",
    )
    holiday_category: str = Field(
        default="None",
        description="Holiday name, or 'None' for an ordinary day",
    )

    @field_validator("weather_category")
    @classmethod
    def _validate_weather(cls, value: str) -> str:
        normalised = (value or "").strip().lower()
        if normalised not in WEATHER_CATEGORIES:
            raise ValueError(
                f"Unknown weather category '{value}'. Supported values: "
                + ", ".join(WEATHER_CATEGORIES)
            )
        return normalised

    @field_validator("holiday_category")
    @classmethod
    def _validate_holiday(cls, value: str) -> str:
        text = (value or "None").strip()
        return "None" if text.lower() in {"", "none", "nan"} else text

    def temperature_kelvin(self) -> float:
        if self.temperature_unit == "celsius":
            return self.temperature + 273.15
        return self.temperature


class PredictionResponse(BaseModel):
    predicted_traffic_volume: float = Field(
        ..., description="Predicted hourly traffic volume in vehicles per hour"
    )
    congestion_category: CongestionCategory
    congestion_thresholds: Thresholds
    thresholds_explanation: str
    model_name: str
    model_created_utc: str | None = None
    input_echo: dict[str, Any]
    disclaimer: str


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class MetricsResponse(BaseModel):
    available: bool
    detail: str
    model_name: str | None = None
    metadata: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


class Coordinate(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class EdgeMultiplier(BaseModel):
    u: Any
    v: Any
    key: Any = 0
    multiplier: float = Field(..., ge=0.1, le=10.0)


class RouteRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    scenario: ScenarioName = Field(
        default="simulated_peak",
        description=(
            "free_flow = no congestion model; simulated_peak = DEMONSTRATION "
            "scenario with edge-specific simulated multipliers; custom = supply "
            "your own multipliers."
        ),
    )
    severity: float = Field(
        default=1.0,
        ge=0.0,
        le=3.0,
        description="Simulated congestion severity multiplier (0 = none).",
    )
    max_alternatives: int = Field(
        default=3, ge=0, le=6, description="Number of alternative routes to try."
    )
    highway_multipliers: dict[str, float] | None = Field(
        default=None,
        description="Optional per-highway-class multipliers (scenario='custom').",
    )
    edge_multipliers: list[EdgeMultiplier] | None = Field(
        default=None,
        description="Optional explicit per-edge multipliers (scenario='custom').",
    )
    study_area: dict[str, Any] | None = Field(
        default=None,
        description="Optional study-area override: latitude, longitude, distance_m.",
    )


class RouteGeometry(BaseModel):
    coordinates: list[list[float]]


class RouteOption(BaseModel):
    rank: int
    label: str
    distance_km: float
    free_flow_time_minutes: float
    congestion_aware_time_minutes: float
    mean_congestion_multiplier: float
    max_congestion_multiplier: float
    highway_mix: dict[str, int]
    geometry: list[list[float]]


class RouteSnap(BaseModel):
    requested: list[float]
    node: list[float]
    node_id: Any


class ScenarioInfo(BaseModel):
    name: str
    description: str
    provenance: str
    is_uniform_multiplier: bool
    note: str


class RouteResponse(BaseModel):
    scenario: ScenarioInfo
    origin_snap: RouteSnap
    destination_snap: RouteSnap
    routes: list[RouteOption]
    graph: dict[str, int]
    attribution: str
    limitations: list[str]


class ErrorResponse(BaseModel):
    detail: str
