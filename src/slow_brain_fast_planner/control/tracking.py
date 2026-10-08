from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Limits:
    max_v: float = 1.0  # m/s
    # NOTE: default matches deploy/NavFlow/visualnav_ros/deployment/config/robot.yaml
    max_w: float = 0.5  # rad/s (yaw rate)
    max_a: float = 1.0  # m/s^2
    max_alpha: float = 1.2  # rad/s^2 (yaw accel)
    max_lat_accel: float = 0.8  # m/s^2


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.1
    horizon_s: float = 4.0
    limits: Limits = Limits()


@dataclass(frozen=True)
class PurePursuitConfig:
    lookahead_m: float = 1.0
    lookahead_gain: float = 0.5  # Ld = L0 + k*v
    v_gain: float = 0.8  # base speed gain vs lookahead distance
    eps: float = 1e-6


@dataclass(frozen=True)
class EndpointPDConfig:
    # This mirrors the spirit of
    # deploy/NavFlow/visualnav_ros/deployment/src/pd_controller_ros2_new.py
    dt_nominal: float = 1.0
    eps: float = 1e-8
    stop_eps: float = 1e-3
    # Endpoint -> (v,w) mapping (matches common on-robot heuristics)
    v_div: float = 2.0
    w_div: float = 2.0

    # Rate limits in *command space* (NOT dt-based in the deployed heuristic)
    dv_up: float = 0.3
    dv_down: float = 0.5

    # Minimum forward speed when not stopping (deployed PD often uses a min clamp)
    v_min: float = 0.1

    # Optional slow-down when yaw rate magnitude is large.
    w_slow_thresh: float = 0.2
    w_slow_gain: float = 0.2  # v *= exp(-k*|w|) when |w| > thresh

    # Optional curvature-based speed slow-down (disabled by default to match latest robot PD
    # snippet)
    curvature_speed_gain: float = 0.0  # v *= exp(-k*|curvature|)


DEFAULT_PURE_PURSUIT_CFG = PurePursuitConfig()
DEFAULT_ENDPOINT_PD_CFG = EndpointPDConfig()


def integrate_diff_drive(
    x: float, y: float, yaw: float, v: float, w: float, dt: float
) -> tuple[float, float, float]:
    """Simple unicycle model."""
    x2 = float(x) + float(v) * math.cos(float(yaw)) * float(dt)
    y2 = float(y) + float(v) * math.sin(float(yaw)) * float(dt)
    yaw2 = float(yaw) + float(w) * float(dt)
    # wrap to [-pi, pi]
    yaw2 = (yaw2 + math.pi) % (2 * math.pi) - math.pi
    return x2, y2, yaw2


def world_to_body(points_xy_world: np.ndarray, *, x: float, y: float, yaw: float) -> np.ndarray:
    """Transform world points into body frame (x forward, y left)."""
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    dx = points_xy_world[:, 0] - float(x)
    dy = points_xy_world[:, 1] - float(y)
    xb = c * dx + s * dy
    yb = -s * dx + c * dy
    return np.stack([xb, yb], axis=1)


def body_to_world(points_xy_body: np.ndarray, *, x: float, y: float, yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    X = float(x) + c * points_xy_body[:, 0] - s * points_xy_body[:, 1]
    Y = float(y) + s * points_xy_body[:, 0] + c * points_xy_body[:, 1]
    return np.stack([X, Y], axis=1)


def polyline_length(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    d = points_xy[1:] - points_xy[:-1]
    return float(np.sum(np.linalg.norm(d, axis=1)))


def point_to_polyline_distance(p: np.ndarray, poly: np.ndarray) -> float:
    """Distance from point p (2,) to polyline poly (N,2)."""
    if poly.shape[0] == 0:
        return float("inf")
    if poly.shape[0] == 1:
        return float(np.linalg.norm(p - poly[0]))
    best = float("inf")
    for i in range(poly.shape[0] - 1):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            d = float(np.linalg.norm(p - a))
        else:
            t = float(np.dot(p - a, ab) / denom)
            t = float(min(1.0, max(0.0, t)))
            proj = a + t * ab
            d = float(np.linalg.norm(p - proj))
        if d < best:
            best = d
    return best


def project_point_to_polyline_arclength(p: np.ndarray, poly: np.ndarray) -> tuple[float, float]:
    """Project point p to polyline poly and return (s, d).

    - s: arclength along polyline from start to closest projection point (meters)
    - d: distance from p to polyline (meters)
    """
    if poly.shape[0] == 0:
        return 0.0, float("inf")
    if poly.shape[0] == 1:
        return 0.0, float(np.linalg.norm(p - poly[0]))

    # Precompute segment lengths so we can convert segment-local t to global arclength.
    segs = poly[1:] - poly[:-1]
    seg_lens = np.linalg.norm(segs, axis=1)
    seg_lens = np.asarray(seg_lens, dtype=np.float64)
    prefix = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg_lens)])

    best_d = float("inf")
    best_s = 0.0
    for i in range(poly.shape[0] - 1):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            t = 0.0
            proj = a
        else:
            t = float(np.dot(p - a, ab) / denom)
            t = float(min(1.0, max(0.0, t)))
            proj = a + t * ab

        d = float(np.linalg.norm(p - proj))
        if d < best_d:
            best_d = d
            best_s = float(prefix[i] + t * seg_lens[i])

    return best_s, best_d


def _estimate_curvature_three_points(pts: np.ndarray, idx: int) -> float:
    """Approx curvature estimate from three points on a polyline in body frame."""
    n = int(pts.shape[0])
    if n < 3:
        return 0.0
    i0 = max(0, int(idx) - 1)
    i1 = int(idx)
    i2 = min(n - 1, int(idx) + 1)
    p1 = pts[i0]
    p2 = pts[i1]
    p3 = pts[i2]
    a = float(np.linalg.norm(p2 - p1))
    b = float(np.linalg.norm(p3 - p2))
    c = float(np.linalg.norm(p3 - p1))
    if a * b * c < 1e-8:
        return 0.0
    area2 = float(np.cross(p2 - p1, p3 - p1))
    return float(2.0 * area2 / (a * b * c))


def controller_endpoint_pd(
    path_body_xy: np.ndarray,
    *,
    v_prev: float,
    w_prev: float,
    limits: Limits,
    cfg: EndpointPDConfig = DEFAULT_ENDPOINT_PD_CFG,
) -> tuple[float, float]:
    """Heuristic endpoint-based PD-like controller (matches current deployment style)."""
    if path_body_xy.shape[0] == 0:
        return 0.0, 0.0
    end = path_body_xy[-1]
    dx = float(end[0])
    dy = float(end[1])
    if abs(dx) < float(cfg.stop_eps) and abs(dy) < float(cfg.stop_eps):
        return 0.0, 0.0

    # Basic heading from endpoint.
    if abs(dx) < float(cfg.eps):
        v = 0.0
        w = float(np.sign(dy) * (math.pi / 20.0))
    else:
        v = dx / float(cfg.dt_nominal) / float(max(float(cfg.v_div), 1e-6))
        # Note: on-robot code often uses atan(dy/dx); atan2 is more numerically stable.
        w = math.atan2(dy, dx) / float(cfg.dt_nominal) / float(max(float(cfg.w_div), 1e-6))

    # Optional curvature-based slow down using mid-polyline curvature estimate.
    if float(cfg.curvature_speed_gain) > 0.0:
        mid_idx = int(max(1, path_body_xy.shape[0] // 2))
        curv = abs(_estimate_curvature_three_points(path_body_xy, mid_idx))
        v *= math.exp(-float(cfg.curvature_speed_gain) * float(curv))

    # Rate limits (command-space, matches deployed heuristic style)
    v = float(np.clip(v, float(v_prev) - float(cfg.dv_down), float(v_prev) + float(cfg.dv_up)))

    # Absolute limits
    v_min = float(max(0.0, float(cfg.v_min)))
    v = float(np.clip(v, v_min, float(limits.max_v)))
    w = float(np.clip(w, -float(limits.max_w), float(limits.max_w)))

    # Optional yaw-rate-based slow down.
    # Match on-robot ordering: apply after w saturation, and do NOT re-apply v_min after slowing.
    if float(cfg.w_slow_gain) > 0.0 and abs(float(w)) > float(cfg.w_slow_thresh):
        v *= math.exp(-float(cfg.w_slow_gain) * abs(float(w)))
    return v, w


def controller_pure_pursuit(
    path_body_xy: np.ndarray,
    *,
    v_prev: float,
    w_prev: float,
    limits: Limits,
    cfg: PurePursuitConfig = DEFAULT_PURE_PURSUIT_CFG,
) -> tuple[float, float]:
    """Pure pursuit + curvature-limited speed (diff-drive form: w=v*kappa)."""
    if path_body_xy.shape[0] < 2:
        return 0.0, 0.0

    # Lookahead distance
    Ld = float(cfg.lookahead_m) + float(cfg.lookahead_gain) * float(max(0.0, v_prev))
    Ld = max(float(cfg.lookahead_m), float(Ld))

    # Pick the first point beyond lookahead.
    target = None
    for p in path_body_xy:
        d = float(math.hypot(float(p[0]), float(p[1])))
        if d >= Ld:
            target = p
            break
    if target is None:
        target = path_body_xy[-1]
    tx = float(target[0])
    ty = float(target[1])
    dist = float(math.hypot(tx, ty))
    if dist < float(cfg.eps):
        return 0.0, 0.0

    # Pure pursuit curvature
    kappa = float(2.0 * ty / max(float(cfg.eps), (tx * tx + ty * ty)))

    # Speed planning: curvature-limited and distance-limited
    v_dist = float(cfg.v_gain) * float(dist)
    v_lat = math.sqrt(float(limits.max_lat_accel) / max(float(cfg.eps), abs(float(kappa))))
    v_cmd = float(min(float(limits.max_v), v_dist, v_lat))

    # Rate limit v
    dv_max = (
        float(limits.max_a) * 0.1
    )  # assume controller tick ~0.1s typical; overwritten by sim dt in wrapper
    v = float(np.clip(v_cmd, float(v_prev) - dv_max, float(v_prev) + dv_max))
    v = float(np.clip(v, 0.0, float(limits.max_v)))

    w_cmd = float(v * kappa)
    # Rate limit w
    dw_max = float(limits.max_alpha) * 0.1
    w = float(np.clip(w_cmd, float(w_prev) - dw_max, float(w_prev) + dw_max))
    w = float(np.clip(w, -float(limits.max_w), float(limits.max_w)))
    return v, w


ControllerName = Literal["endpoint_pd", "pure_pursuit"]


def simulate_tracking(
    ref_path_body_xy: np.ndarray,
    *,
    controller: ControllerName,
    sim: SimConfig,
    pp_cfg: PurePursuitConfig = DEFAULT_PURE_PURSUIT_CFG,
    ep_cfg: EndpointPDConfig = DEFAULT_ENDPOINT_PD_CFG,
    return_trace: bool = False,
) -> dict[str, float]:
    """Simulate tracking the given reference path.

    The reference path is given in *initial* body frame (robot at origin, yaw=0).
    We treat it as a world polyline, then at each step transform into current body frame.
    """
    dt = float(sim.dt)
    steps = int(max(1, round(float(sim.horizon_s) / dt)))
    limits = sim.limits

    # Reference polyline in world coordinates (initial body frame == world at t=0)
    ref_world = np.asarray(ref_path_body_xy, dtype=np.float64)
    if ref_world.ndim != 2 or ref_world.shape[1] != 2:
        raise ValueError("ref_path_body_xy must be (N,2)")

    x = 0.0
    y = 0.0
    yaw = 0.0
    v = 0.0
    w = 0.0

    ctes: list[float] = []
    progresses: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    yaws: list[float] = []
    ref_len = polyline_length(ref_world)
    for _k in range(steps):
        # Compute path in current body frame
        path_body = world_to_body(ref_world, x=x, y=y, yaw=yaw)
        if controller == "endpoint_pd":
            v_cmd, w_cmd = controller_endpoint_pd(
                path_body, v_prev=v, w_prev=w, limits=limits, cfg=ep_cfg
            )
        else:
            # correct rate limits for sim dt
            v_cmd, w_cmd = controller_pure_pursuit(
                path_body, v_prev=v, w_prev=w, limits=limits, cfg=pp_cfg
            )
            # re-apply rate limits using dt
            dv = float(limits.max_a) * dt
            dw = float(limits.max_alpha) * dt
            v_cmd = float(np.clip(v_cmd, v - dv, v + dv))
            w_cmd = float(np.clip(w_cmd, w - dw, w + dw))
        v = float(np.clip(v_cmd, 0.0, limits.max_v))
        w = float(np.clip(w_cmd, -limits.max_w, limits.max_w))

        x, y, yaw = integrate_diff_drive(x, y, yaw, v, w, dt)
        s, d = project_point_to_polyline_arclength(np.array([x, y], dtype=np.float64), ref_world)
        ctes.append(float(d))
        progresses.append(float(s))
        if return_trace:
            xs.append(float(x))
            ys.append(float(y))
            yaws.append(float(yaw))

    end_ref = ref_world[-1] if ref_world.shape[0] else np.array([0.0, 0.0])
    end_err = float(np.linalg.norm(np.array([x, y], dtype=np.float64) - end_ref))
    progress_final_m = float(progresses[-1]) if progresses else 0.0
    progress_max_m = float(max(progresses)) if progresses else 0.0
    progress_frac = float(progress_final_m / ref_len) if ref_len > 1e-8 else 0.0

    return {
        "cte_mean": float(np.mean(ctes)) if ctes else float("nan"),
        "cte_p95": float(np.quantile(ctes, 0.95)) if ctes else float("nan"),
        "cte_max": float(np.max(ctes)) if ctes else float("nan"),
        "end_err": float(end_err),
        "progress_final_m": float(progress_final_m),
        "progress_max_m": float(progress_max_m),
        "progress_frac": float(progress_frac),
        "final_x": float(x),
        "final_y": float(y),
        "final_yaw": float(yaw),
        # Only present when requested.
        "trace_x": xs if return_trace else [],
        "trace_y": ys if return_trace else [],
        "trace_yaw": yaws if return_trace else [],
    }
