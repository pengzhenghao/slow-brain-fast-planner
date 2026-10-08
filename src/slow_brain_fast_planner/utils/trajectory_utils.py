from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GoalFilterConfig:
    """Heuristic filter for obviously-bad goals in robot frame.

    This is intentionally conservative and meant to catch clear telemetry/transform bugs like:
      - extremely far goals (hundreds of meters in local robot frame)
      - goals far *behind* the robot (x_forward < 0 and large distance)
      - NaN/inf values

    All thresholds are in robot frame where:
      - x is forward (meters)
      - y is left (meters)
      - bearing is atan2(y, x) in degrees (0=forward, +left)
    """

    enabled: bool = True

    # Reject if dist < min_distance_m (None disables).
    # This is useful for datasets where a "goal" can collapse to ~0 due to stasis / telemetry
    # quirks.
    min_distance_m: float | None = None

    # Reject if dist > max_distance_m (None disables).
    max_distance_m: float | None = 120.0

    # Reject "behind and far" goals.
    # If x_forward < behind_x_threshold_m AND dist >= behind_min_distance_m => reject.
    behind_x_threshold_m: float = 0.0
    behind_min_distance_m: float = 30.0

    # Optional bearing-based back-facing reject:
    # If abs(bearing_deg) >= back_bearing_abs_deg AND dist >= back_bearing_min_distance_m => reject.
    back_bearing_abs_deg: float = 150.0
    back_bearing_min_distance_m: float = 30.0

    # If True, reject when goal is missing (goal_xy is None).
    require_goal: bool = False


@dataclass(frozen=True)
class GtFutureFilterConfig:
    """Heuristic filter for obviously-wrong GT future trajectories (robot local frame).

    Intended to catch cases where the canonical odom/yaw alignment is broken (e.g. GT appears
    to go backward in the robot frame for many steps), or where there are teleports/jumps.
    """

    enabled: bool = True

    # Backward test on x_forward.
    backward_x_threshold_m: float = -0.10  # allow small negative due to noise
    max_backward_frac: float = 0.80  # reject if > this fraction of points are behind threshold
    max_backward_mean_x_m: float = -0.50  # reject if mean x is too negative

    # Teleport/jump sanity.
    max_step_m: float = 10.0  # reject if any consecutive step exceeds this
    max_total_m: float = 50.0  # reject if total path length exceeds this


def gt_future_filter_reason(
    gt_local_xy: list[list[float]] | None,
    *,
    cfg: GtFutureFilterConfig | None,
) -> str | None:
    """Return a short reason string if GT future should be filtered; otherwise None."""
    if cfg is None or (not bool(cfg.enabled)):
        return None
    if not gt_local_xy:
        return "missing_gt_future"

    xs: list[float] = []
    pts: list[tuple[float, float]] = []
    for p in gt_local_xy:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            x = float(p[0])
            y = float(p[1])
        except Exception:
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            return "gt_not_finite"
        xs.append(float(x))
        pts.append((float(x), float(y)))

    if not xs or len(pts) < 2:
        return "missing_gt_future"

    n = float(len(xs))
    back = sum(1 for x in xs if float(x) < float(cfg.backward_x_threshold_m))
    back_frac = float(back) / n if n > 0 else 0.0
    mean_x = float(sum(xs) / n) if n > 0 else 0.0

    if back_frac > float(cfg.max_backward_frac) and mean_x < 0.0:
        return "gt_mostly_backward"
    if mean_x < float(cfg.max_backward_mean_x_m):
        return "gt_mean_backward"

    # Jump checks.
    total = 0.0
    for i in range(1, len(pts)):
        dx = float(pts[i][0] - pts[i - 1][0])
        dy = float(pts[i][1] - pts[i - 1][1])
        d = float((dx * dx + dy * dy) ** 0.5)
        if not math.isfinite(d):
            return "gt_not_finite"
        if d > float(cfg.max_step_m):
            return "gt_step_jump"
        total += d
    if total > float(cfg.max_total_m):
        return "gt_total_jump"

    return None


def goal_filter_reason(
    goal_xy: list[float] | None,
    goal_distance_m: float | None,
    goal_bearing_deg: float | None,
    *,
    cfg: GoalFilterConfig | None,
) -> str | None:
    """Return a short reason string if goal should be filtered; otherwise None."""
    if cfg is None or (not bool(cfg.enabled)):
        return None

    if goal_xy is None or goal_distance_m is None or goal_bearing_deg is None:
        return "missing_goal" if bool(cfg.require_goal) else None

    try:
        gx = float(goal_xy[0])
        gy = float(goal_xy[1])
        d = float(goal_distance_m)
        b = float(goal_bearing_deg)
    except Exception:
        return "bad_goal_type"

    # Finite checks.
    if not (math.isfinite(gx) and math.isfinite(gy) and math.isfinite(d) and math.isfinite(b)):
        return "goal_not_finite"

    # Distance hard cap.
    if cfg.min_distance_m is not None and float(d) < float(cfg.min_distance_m):
        return f"goal_too_close<{float(cfg.min_distance_m):g}m"
    if cfg.max_distance_m is not None and float(d) > float(cfg.max_distance_m):
        return f"goal_too_far>{float(cfg.max_distance_m):g}m"

    # Behind-and-far heuristics.
    if float(gx) < float(cfg.behind_x_threshold_m) and float(d) >= float(cfg.behind_min_distance_m):
        return "goal_far_behind_robot"

    # Bearing-based back-facing heuristic (redundant with x<0, but catches edge cases).
    if abs(float(b)) >= float(cfg.back_bearing_abs_deg) and float(d) >= float(
        cfg.back_bearing_min_distance_m
    ):
        return "goal_far_back_bearing"

    return None


def extract_goal_info(
    record: Any,
) -> tuple[list[float] | None, float | None, float | None]:
    """Extract goal information from a planner record.

    Returns:
        tuple: (goal_xy, goal_distance_m, goal_bearing_deg)
    """
    g = None

    # 1. Try standard attributes
    for key in ("goal_xy", "goal_point_xy", "goal_point", "target_point"):
        if isinstance(record, dict):
            val = record.get(key)
        else:
            val = getattr(record, key, None)
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
            g = val
            break

    # 2. Try Pydantic model_extra
    if g is None:
        try:
            if isinstance(record, dict):
                extra = record.get("model_extra")
            else:
                extra = getattr(record, "model_extra", None)
            if isinstance(extra, dict):
                for key in ("goal_xy", "goal_point_xy", "goal_point"):
                    val = extra.get(key)
                    if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
                        g = val
                        break
        except Exception:
            pass

    try:
        if g is not None:
            gx = float(g[0])
            gy = float(g[1])
            goal_xy = [gx, gy]
            goal_distance_m = float((gx * gx + gy * gy) ** 0.5)
            goal_bearing_deg = (
                float(math.degrees(math.atan2(gy, gx))) if (gx != 0.0 or gy != 0.0) else 0.0
            )
            return goal_xy, goal_distance_m, goal_bearing_deg
    except Exception:
        pass
    return None, None, None
