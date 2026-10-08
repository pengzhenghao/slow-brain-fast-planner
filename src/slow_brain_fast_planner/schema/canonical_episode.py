from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Version of the released canonical episode schema.
CANONICAL_EPISODE_SCHEMA_VERSION = "0.1.0"

Ref = str | dict[str, Any]


class LatLon(BaseModel):
    lat: float
    lon: float
    alt: float | None = None

    model_config = ConfigDict(extra="forbid")


class MapOrigin(LatLon):
    frame: Literal["ENU"] = "ENU"

    model_config = ConfigDict(extra="forbid")


class CalibrationRefs(BaseModel):
    # Spec says "optional ref"; we accept string refs and allow extension fields.
    camera_intrinsics: Ref | None = None
    camera_extrinsics: Ref | None = None

    model_config = ConfigDict(extra="allow")


class StaticContext(BaseModel):
    user_instruction: str | None = None
    goal_description: str | None = None
    goal_latlon: LatLon | None = None
    constraints: list[str] | None = None

    model_config = ConfigDict(extra="allow")


class StreamSpec(BaseModel):
    """Logical stream location.

    Canonical episode storage can either:
    - reference large per-stream records via `records_ref`, or
    - inline small records via `records`.
    """

    records_ref: Ref | None = None
    records: list[Any] | None = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_source(self) -> StreamSpec:
        if self.records_ref is None and self.records is None:
            raise ValueError("Stream must define either `records_ref` or `records`.")
        return self


class Quality(BaseModel):
    missing_streams: list[str] = Field(default_factory=list)
    notes: str | None = None

    model_config = ConfigDict(extra="allow")


class CanonicalEpisode(BaseModel):
    """Top-level metadata for one canonical navigation episode."""

    schema_version: str = Field(..., examples=[CANONICAL_EPISODE_SCHEMA_VERSION])
    episode_id: str

    site: str | None = None
    start_time_utc: str | None = None
    end_time_utc: str | None = None

    map_origin: MapOrigin | None = None
    calibration_refs: CalibrationRefs | None = None
    static_context: StaticContext | None = None

    streams: dict[str, StreamSpec] = Field(default_factory=dict)
    quality: Quality | None = None

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_streams_not_empty(self) -> CanonicalEpisode:
        if not self.streams:
            raise ValueError("Episode must define a non-empty `streams` mapping.")
        return self


# --- Stream record models (v0 minimal) ---


class GPSRecord(BaseModel):
    t: float
    lat: float
    lon: float
    accuracy_m: float | None = None
    source: str | None = None

    model_config = ConfigDict(extra="allow")


class OdomRecord(BaseModel):
    t: float
    x: float
    y: float
    yaw: float
    frame: str = Field(..., description="robot|map_enu (see spec)")

    model_config = ConfigDict(extra="allow")


class RGBRecord(BaseModel):
    t: float
    frame_ref: str
    width: int
    height: int
    camera_id: str | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("width", "height")
    @classmethod
    def _validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("width/height must be > 0")
        return v


class RouteRecord(BaseModel):
    t: float
    provider: str | None = None
    start_latlon: LatLon
    goal_latlon: LatLon
    polyline: str | list[LatLon]
    distance_m: float | None = None
    eta_s: float | None = None
    steps: list[Any] | None = None

    model_config = ConfigDict(extra="allow")


class PlannerCandidate(BaseModel):
    traj_id: str | None = None
    points_xy: list[list[float]]
    score: float

    model_config = ConfigDict(extra="allow")

    @field_validator("points_xy")
    @classmethod
    def _validate_points_xy(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("points_xy must be non-empty")
        for p in v:
            if len(p) != 2:
                raise ValueError("each points_xy entry must be [x, y]")
        return v


class PlannerCandidatesRecord(BaseModel):
    t: float
    frame: str = "robot"
    auto_enabled: bool | None = None
    postprocess: str | None = None
    candidates: list[PlannerCandidate] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class ControlRecord(BaseModel):
    t: float
    v: float
    w: float

    model_config = ConfigDict(extra="allow")


def canonical_episode_json_schema() -> dict[str, Any]:
    """Return the JSON schema for `CanonicalEpisode` (draft-2020-12 style from Pydantic v2)."""

    return CanonicalEpisode.model_json_schema()
