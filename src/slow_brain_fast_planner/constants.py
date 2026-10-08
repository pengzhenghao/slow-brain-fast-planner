from __future__ import annotations

# NOTE: This file is intentionally dependency-free and safe to import anywhere.
# Put "repo-wide defaults" here so scripts and library code stay consistent.

# NMS (Non-Maximum Suppression) settings for trajectory filtering
NMS_MAX_TRAJECTORIES: int = 6
NMS_DISTANCE_THRESHOLD: float = 2.0
PROB_THRESHOLD: float = 0.01

# Dataset / timebase defaults
DEFAULT_DATA_HZ: float = 5.0
DEFAULT_DATA_DT_S: float = 1.0 / DEFAULT_DATA_HZ

# Planner trajectory defaults
DEFAULT_TRAJECTORY_DT_S: float = 0.2  # 20 pts @ 5Hz (commonly logged by the planner)


# VLM defaults
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash-lite"

# Takeover clip dataset defaults (request-intervention benchmark)
DEFAULT_TAKEOVER_CLIP_HISTORY_WINDOW_S: float = 4.0
DEFAULT_TAKEOVER_CLIP_STRIDE_S: float = 2.0
DEFAULT_TAKEOVER_CLIP_HORIZON_S: float = 2.0
DEFAULT_TAKEOVER_CLIP_K_FRAMES: int = 4
DEFAULT_TAKEOVER_CLIP_DEBOUNCE_ON_COUNT: int = 1
DEFAULT_TAKEOVER_CLIP_DEBOUNCE_OFF_COUNT: int = 1
DEFAULT_TAKEOVER_CLIP_POST_WINDOW_S: float = 2.0

# Overlay projection settings
OVERLAY_PROJECTION: str = "fisheye_v1"  # "fisheye_v1" | "simple_xy"

# Fisheye camera parameters (calibrated defaults for the robot's front fisheye camera)
CAMERA_HEIGHT_M: float = 0.41
FISHEYE_K: float = 0.0
FISHEYE_FX: float | None = None
FISHEYE_FY: float | None = None
FISHEYE_CX: float | None = None
FISHEYE_CY: float | None = None

# Overlay visual styles
LINE_WIDTH: int = 4
ALPHA: float = 0.75
LABEL_FONT_SIZE: int = 18

# Padding
PAD_BOTTOM: int = 0
