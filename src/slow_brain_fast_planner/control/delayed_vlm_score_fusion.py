"""Toy closed-loop simulator for delayed VLM trajectory advice + score fusion.

This is intentionally *non-visual* and *dynamics-light* so we can:
1) sanity-check controllers (pure pursuit vs endpoint-based PD-ish)
2) study delayed "S2" (VLM) advice integration strategies without needing a full simulator

Key assumptions (kept consistent with `slow_brain_fast_planner.control.tracking`):
- Robot plant: ideal unicycle / diff-drive (x, y, yaw) with controls (v, w).
- Perfect state estimate (no GPS/odom noise), no actuator delay, no slip/terrain.
- Constraints modeled only via caps: max_v, max_w, max_a, max_alpha, max_lat_accel.
- No obstacles: this evaluates *path tracking* and *selection stability* only.

Terminology:
- "planner": a simple synthetic scorer over a fixed candidate library (motion primitives).
- "dummy VLM": an oracle-like chooser (min cost) but returns after a fixed delay.
- "score fusion": re-rank current candidates using similarity to the last (stale) VLM trajectory.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from slow_brain_fast_planner.control.tracking import (
    EndpointPDConfig,
    Limits,
    PurePursuitConfig,
    body_to_world,
    controller_endpoint_pd,
    controller_pure_pursuit,
    integrate_diff_drive,
    polyline_length,
    project_point_to_polyline_arclength,
    world_to_body,
)

TaskName = Literal["forward", "left_turn", "right_turn"]
ControllerName = Literal["endpoint_pd", "pure_pursuit"]
PolicyName = Literal["local_only", "vlm_hold", "score_fusion", "prob_fusion", "chunk_stitch"]
SimilarityMode = Literal[
    "pointwise_arclen",
    "candidate_to_ref_polyline",
    "symmetric_polyline",
    "body_pointwise_arclen",
    "body_pointwise_horizon_aware",
]


@dataclass(frozen=True)
class ToyTask:
    name: TaskName
    v_ref: float  # m/s
    w_ref: float  # rad/s
    duration_s: float = 12.0


@dataclass(frozen=True)
class PlannerScoreModel:
    # Higher is better.
    noise_std: float = 0.15
    score_scale: float = 1.0
    # Epsilon-greedy: with probability epsilon, pick a random candidate instead of argmax.
    # This simulates a "confused" planner that doesn't always know the right direction.
    epsilon: float = 0.0


@dataclass(frozen=True)
class VlmDelayModel:
    delay_s: float = 1.5
    # If true, the VLM chooses the best candidate by the same oracle objective as we use to define
    # "ideal".
    # This is a *dummy* VLM: it is not vision-based; it's a delayed oracle for systematic testing.
    oracle: bool = True


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = True
    similarity_mode: SimilarityMode = "pointwise_arclen"
    # Additive fusion in the score/logit space:
    #   fused_score = planner_score + lambda_sim * decay(staleness) * similarity(...)
    lambda_sim: float = 1.0
    staleness_tau_s: float = 3.0  # exp(-stale/tau)
    # Normalize similarity by a distance scale so lambda is less brittle.
    dist_scale_m: float = 1.0
    # If true, compare candidates to the *remaining* segment of the stale VLM path by projecting the
    # current robot position onto that stale path (arclength alignment).
    align_to_current_pose: bool = True


@dataclass(frozen=True)
class SimLoopConfig:
    # Control integration (controller tick).
    dt_control: float = 0.1
    # Replan/selection tick (planner tick).
    dt_plan: float = 0.2
    # Candidate horizon used only for the synthetic "ideal" reference construction.
    horizon_s: float = 4.0
    # If true, prepend (0,0) to each candidate polyline (many static sets start at ~0.2m).
    prepend_origin: bool = True
    limits: Limits = Limits()
    # Controller configs.
    pp_cfg: PurePursuitConfig = PurePursuitConfig()
    ep_cfg: EndpointPDConfig = EndpointPDConfig()


@dataclass(frozen=True)
class SimResult:
    # Aggregate metrics
    mean_pos_err_m: float
    p95_pos_err_m: float
    mean_cte_to_ideal_m: float
    p95_cte_to_ideal_m: float
    mean_speed_mps: float
    stopped_frac: float
    mean_abs_dv: float
    mean_abs_dw: float
    # Useful diagnostics
    steps: int
    plan_steps: int
    vlm_updates: int
    chosen_idx_hist: list[int]


@dataclass(frozen=True)
class SimTrace:
    """Full trace for plotting/diagnostics (returned by `run_toy_simulation_with_trace`)."""

    dt_control: float
    dt_plan: float

    # Continuous-time (control tick) traces
    t_s: np.ndarray  # (steps+1,)
    x: np.ndarray  # (steps+1,)
    y: np.ndarray  # (steps+1,)
    yaw: np.ndarray  # (steps+1,)
    v: np.ndarray  # (steps+1,)
    w: np.ndarray  # (steps+1,)
    ideal_world: np.ndarray  # (steps+1,2)
    actual_world: np.ndarray  # (steps+1,2)

    # Discrete-time (planner tick) traces
    plan_t_s: np.ndarray  # (plan_steps,)
    chosen_idx: np.ndarray  # (plan_steps,)
    ref_world: list[np.ndarray]  # len=plan_steps
    vlm_ref_world: list[np.ndarray | None]  # len=plan_steps
    staleness_s: np.ndarray  # (plan_steps,)
    fusion_weight: np.ndarray  # (plan_steps,)


def load_static_candidate_set(path: str | Path, *, prepend_origin: bool) -> list[np.ndarray]:
    p = resolve_static_candidate_set_path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "candidates" not in obj:
        raise ValueError(
            "Expected a task2 static candidate set JSON with top-level key 'candidates'."
        )
    cands = obj.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise ValueError("Static candidate set 'candidates' must be a non-empty list.")
    out: list[np.ndarray] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        pts = c.get("points_xy")
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
            continue
        arr = arr[:, :2].astype(np.float64)
        if prepend_origin:
            arr0 = np.zeros((arr.shape[0] + 1, 2), dtype=np.float64)
            arr0[1:, :] = arr
            arr = arr0
        if not np.all(np.isfinite(arr)):
            continue
        out.append(arr)
    if not out:
        raise ValueError("No valid candidates parsed from static candidate set JSON.")
    return out


def resolve_static_candidate_set_path(path: str | Path) -> Path:
    def _repo_root_from_here() -> Path | None:
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "pyproject.toml").exists():
                return parent
        return None

    p0 = Path(path).expanduser()
    p = p0.resolve()
    if p.exists():
        return p

    tried: list[Path] = [p]

    # Backward-compat: older logs used `assets/task2_static_candidates/...`.
    # The repo now stores these under `assets/trajectory_selection_static_candidates/...`.
    s = str(p).replace("\\", "/")
    if "/assets/task2_static_candidates/" in s:
        s2 = s.replace(
            "/assets/task2_static_candidates/", "/assets/trajectory_selection_static_candidates/"
        )
        p2 = Path(s2)
        tried.append(p2)
        if p2.exists():
            return p2.resolve()
    if "/task2_static_candidates/" in s:
        s2 = s.replace("/task2_static_candidates/", "/trajectory_selection_static_candidates/")
        p2 = Path(s2)
        tried.append(p2)
        if p2.exists():
            return p2.resolve()

    # Repo-local fallback: search in `<repo_root>/assets` (prefer matching subdir names).
    repo_root = _repo_root_from_here()
    if repo_root is not None:
        assets = repo_root / "assets"
        name = p.name
        if assets.exists() and name:
            # If the original path contains a distinctive subdir name, prefer that.
            prefer_sub = None
            for sub in ("takeover_kmeans_medoids", "takeover_kdisk_endpoints", "grid_endpoints"):
                if f"/{sub}/" in s:
                    prefer_sub = sub
                    break
            if prefer_sub is not None:
                candidates = sorted(assets.glob(f"**/{prefer_sub}/{name}"))
            else:
                candidates = sorted(assets.glob(f"**/{name}"))
            for c in candidates:
                tried.append(c)
                if c.exists():
                    return c.resolve()

    tried_s = "\n".join([f"- {x}" for x in tried[:12]])
    raise FileNotFoundError(
        f"Static candidate set JSON not found: {p}\n"
        f"Tried (first {min(12, len(tried))}):\n{tried_s}\n"
        "Fix: pass a valid `--static-candidates-json`, or update old configs from "
        "`assets/task2_static_candidates/...` to "
        "`assets/trajectory_selection_static_candidates/...`."
    )


def _resample_polyline_index(points_xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to exactly n points by linear interpolation on normalized index."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("points_xy must be (N,2) with N>=2")
    if int(n) < 2:
        raise ValueError("n must be >= 2")
    if int(pts.shape[0]) == int(n):
        return pts.copy()
    t_src = np.linspace(0.0, 1.0, int(pts.shape[0]), dtype=np.float64)
    t_dst = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    x = np.interp(t_dst, t_src, pts[:, 0])
    y = np.interp(t_dst, t_src, pts[:, 1])
    return np.stack([x, y], axis=1).astype(np.float64)


def _sample_polyline_by_arclength(points_xy: np.ndarray, *, s_start: float, n: int) -> np.ndarray:
    """Sample n points from polyline starting at arclength s_start (clamped)."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("points_xy must be (N,2) with N>=2")
    if int(n) < 2:
        raise ValueError("n must be >= 2")
    seg = pts[1:] - pts[:-1]
    seg_l = np.linalg.norm(seg, axis=1)
    seg_l = np.asarray(seg_l, dtype=np.float64)
    s = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg_l)])
    total = float(s[-1])
    if not math.isfinite(total) or total <= 1e-9:
        return np.repeat(pts[:1], repeats=int(n), axis=0)
    s0 = float(min(max(float(s_start), 0.0), total))
    # Remaining arclength available:
    remain = float(max(0.0, total - s0))
    # If remain is tiny, return a constant point.
    if remain <= 1e-6:
        return np.repeat(pts[-1:].copy(), repeats=int(n), axis=0)
    targets = s0 + np.linspace(0.0, remain, int(n), dtype=np.float64)
    # Clamp last just in case of rounding.
    targets[-1] = min(targets[-1], total)
    out = np.zeros((int(n), 2), dtype=np.float64)
    # For each target arclength, find segment index i with s[i] <= t <= s[i+1].
    idxs = np.searchsorted(s, targets, side="right") - 1
    idxs = np.clip(idxs, 0, int(pts.shape[0]) - 2)
    for j in range(int(n)):
        i = int(idxs[j])
        s_i = float(s[i])
        s_ip1 = float(s[i + 1])
        if s_ip1 <= s_i + 1e-12:
            out[j] = pts[i]
            continue
        a = float((targets[j] - s_i) / (s_ip1 - s_i))
        out[j] = (1.0 - a) * pts[i] + a * pts[i + 1]
    return out


def _sample_polyline_by_arclength_window(
    points_xy: np.ndarray, *, s_start: float, s_len: float, n: int
) -> np.ndarray:
    """Sample n points from a polyline over arclength window [s_start, s_start+s_len].

    Use this when the *reference* polyline is longer than the candidate horizon. (In this toy sim,
    the task reference can be ~20s long, while candidates are ~4s.)
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("points_xy must be (N,2) with N>=2")
    if int(n) < 2:
        raise ValueError("n must be >= 2")

    seg = pts[1:] - pts[:-1]
    seg_l = np.linalg.norm(seg, axis=1)
    seg_l = np.asarray(seg_l, dtype=np.float64)
    s = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg_l)])
    total = float(s[-1])
    if not math.isfinite(total) or total <= 1e-9:
        return np.repeat(pts[:1], repeats=int(n), axis=0)

    s0 = float(min(max(float(s_start), 0.0), total))
    s_len = float(max(0.0, float(s_len)))
    s1 = float(min(total, s0 + s_len))
    if s1 <= s0 + 1e-6:
        idx = int(np.searchsorted(s, s0, side="right") - 1)
        idx = int(np.clip(idx, 0, int(pts.shape[0]) - 1))
        p = pts[idx].copy()
        return np.repeat(p[None, :], repeats=int(n), axis=0)

    targets = np.linspace(s0, s1, int(n), dtype=np.float64)
    out = np.zeros((int(n), 2), dtype=np.float64)
    idxs = np.searchsorted(s, targets, side="right") - 1
    idxs = np.clip(idxs, 0, int(pts.shape[0]) - 2)
    for j in range(int(n)):
        i = int(idxs[j])
        s_i = float(s[i])
        s_ip1 = float(s[i + 1])
        if s_ip1 <= s_i + 1e-12:
            out[j] = pts[i]
            continue
        a = float((targets[j] - s_i) / (s_ip1 - s_i))
        out[j] = (1.0 - a) * pts[i] + a * pts[i + 1]
    return out


def _chunk_stitch_ref_world(
    *,
    stale_ref_world: np.ndarray,
    robot_pose_world: tuple[float, float, float],
    seg_len: float,
    n: int,
    blend_arclen_m: float = 1.0,
) -> np.ndarray:
    """VLA-RAIL-style chunk stitcher.

    Given the (stale) VLM reference path in world frame and the current robot pose,
    return a reference polyline whose first segment smoothly bridges from the robot's
    current position+heading onto the stale path via a cubic Hermite blend, then
    follows the (remaining) stale path.

    The seam-blend matches position and tangent at the start (robot pose) and at the
    blend endpoint (a point ``blend_arclen_m`` ahead along the stale path). Beyond
    the blend, the reference equals the stale path itself. Total reference covers
    arclength ``seg_len`` along the stale path.
    """
    stale = np.asarray(stale_ref_world, dtype=np.float64)
    if stale.ndim != 2 or stale.shape[0] < 2 or stale.shape[1] < 2:
        return _sample_polyline_by_arclength_window(
            stale, s_start=0.0, s_len=float(seg_len), n=int(n)
        )
    rx, ry, ryaw = (
        float(robot_pose_world[0]),
        float(robot_pose_world[1]),
        float(robot_pose_world[2]),
    )

    # Project robot onto the stale path; slide the reference window forward.
    s0, _d0 = project_point_to_polyline_arclength(np.asarray([rx, ry], dtype=np.float64), stale)
    blend_len = float(max(0.05, min(float(blend_arclen_m), float(seg_len) * 0.5)))

    # Endpoint of the blend on the stale path.
    end_window = _sample_polyline_by_arclength_window(
        stale, s_start=float(s0), s_len=float(blend_len), n=2
    )
    p1 = end_window[-1]
    # Tangent at the blend endpoint: forward difference along stale path.
    tan_window = _sample_polyline_by_arclength_window(
        stale, s_start=float(s0 + max(blend_len - 0.05, 0.0)), s_len=0.1, n=2
    )
    tan_vec = tan_window[-1] - tan_window[0]
    tn = float(np.linalg.norm(tan_vec))
    if tn > 1e-6:
        t1 = (tan_vec / tn) * blend_len
    else:
        t1 = np.array([blend_len, 0.0], dtype=np.float64)
    # Tangent at the robot end matches its current heading, scaled by blend_len.
    t0 = np.array([math.cos(ryaw), math.sin(ryaw)], dtype=np.float64) * blend_len
    p0 = np.array([rx, ry], dtype=np.float64)

    # Number of points to spend on the blend portion vs. the post-blend stale tail.
    n_blend = int(max(2, round(int(n) * (blend_len / max(float(seg_len), blend_len + 1e-6)))))
    n_blend = int(min(n_blend, int(n) - 1))
    n_tail = int(int(n) - n_blend)

    # Cubic Hermite blend: H(u) = h00 p0 + h10 t0 + h01 p1 + h11 t1.
    u = np.linspace(0.0, 1.0, int(n_blend), dtype=np.float64)
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    blend_xy = (
        h00[:, None] * p0[None, :]
        + h10[:, None] * t0[None, :]
        + h01[:, None] * p1[None, :]
        + h11[:, None] * t1[None, :]
    )

    # Tail: stale path from s0+blend_len for the remaining arclength.
    tail_len = float(max(0.0, float(seg_len) - blend_len))
    if int(n_tail) >= 2 and tail_len > 1e-6:
        tail_xy = _sample_polyline_by_arclength_window(
            stale, s_start=float(s0 + blend_len), s_len=float(tail_len), n=int(n_tail)
        )
    else:
        tail_xy = np.repeat(p1[None, :], int(max(0, n_tail)), axis=0)

    out = np.concatenate([blend_xy, tail_xy], axis=0)
    if out.shape[0] != int(n):
        out = _resample_polyline_index(out, int(n))
    return np.asarray(out, dtype=np.float64)


def _ideal_body_path(
    task: ToyTask, *, n_points: int, horizon_s: float, prepend_origin: bool
) -> np.ndarray:
    """Ideal commanded path in robot frame (starting at origin, yaw=0)."""
    n = int(n_points)
    if n < 2:
        raise ValueError("n_points must be >= 2")
    # Keep the "missing origin" convention consistent with candidate sets by matching point count.
    # If candidates have an explicit origin, include it here as well.
    if prepend_origin:
        # First point is exactly (0,0).
        t = np.linspace(0.0, float(horizon_s), n, dtype=np.float64)
    else:
        # Candidates often omit the origin and start at the first dt step.
        t = np.linspace(float(horizon_s) / float(n), float(horizon_s), n, dtype=np.float64)
    v = float(task.v_ref)
    w = float(task.w_ref)
    if abs(w) < 1e-9:
        x = v * t
        y = np.zeros_like(x)
        return np.stack([x, y], axis=1)
    # Constant-curvature arc in body frame.
    k = w / max(1e-9, v)  # curvature (rad/m) with sign
    s = v * t
    x = np.sin(k * s) / k
    y = (1.0 - np.cos(k * s)) / k
    return np.stack([x, y], axis=1)


def _mean_pointwise_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    a = np.asarray(a_xy, dtype=np.float64)
    b = np.asarray(b_xy, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("a_xy and b_xy must be (N,2)")
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return float("inf")
    d = a[:n] - b[:n]
    return float(np.mean(np.linalg.norm(d, axis=1)))


def _mean_candidate_to_ref_polyline_distance(
    candidate_xy: np.ndarray, ref_poly: np.ndarray
) -> float:
    cand = np.asarray(candidate_xy, dtype=np.float64)
    ref = np.asarray(ref_poly, dtype=np.float64)
    if cand.ndim != 2 or cand.shape[1] != 2 or cand.shape[0] < 1:
        return float("inf")
    if ref.ndim != 2 or ref.shape[1] != 2 or ref.shape[0] < 2:
        return float("inf")
    ds: list[float] = []
    for p in cand:
        _s, d = project_point_to_polyline_arclength(p, ref)
        ds.append(float(d))
    return float(np.mean(ds)) if ds else float("inf")


def _select_with_epsilon(scores: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Epsilon-greedy selection: with prob epsilon, pick random; else argmax.

    This simulates a planner that occasionally makes random mistakes,
    giving VLM a chance to correct them.
    """
    if epsilon > 0 and rng.random() < epsilon:
        return int(rng.integers(0, len(scores)))
    return int(np.argmax(scores))


def _corrupt_scores_with_epsilon(
    scores: np.ndarray, epsilon: float, rng: np.random.Generator
) -> np.ndarray:
    """Corrupt scores with epsilon probability to simulate confused planner.

    With probability epsilon, replace scores with random values so that
    argmax becomes effectively random. This corruption propagates through
    fusion, giving VLM a chance to correct the mistake.
    """
    if epsilon > 0 and rng.random() < epsilon:
        # Randomize scores: uniform in [min, max] range so argmax is random
        return rng.uniform(float(scores.min()), float(scores.max()), size=len(scores)).astype(
            np.float64
        )
    return scores


def _softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax with NaN/inf handling."""
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 0:
        return x
    if not np.any(np.isfinite(x)):
        return np.full_like(x, 1.0 / float(len(x)), dtype=np.float64)
    m = float(np.max(x[np.isfinite(x)]))
    x2 = np.where(np.isfinite(x), x, m - 1e9).astype(np.float64)
    z = x2 - float(np.max(x2))
    e = np.exp(z)
    s = float(np.sum(e))
    if not math.isfinite(s) or s <= 0:
        return np.full_like(x2, 1.0 / float(len(x2)), dtype=np.float64)
    return (e / s).astype(np.float64)


def _similarity(
    *,
    candidate_world: np.ndarray,
    stale_ref_world: np.ndarray,
    robot_pose_world: tuple[float, float, float],
    cfg: FusionConfig,
    vlm_remaining_frac: float = 1.0,
) -> float:
    """Return similarity (higher is better) between candidate and stale reference.

    Args:
        candidate_world: Candidate trajectory in world frame (N, 2).
        stale_ref_world: Stale VLM reference trajectory in world frame (M, 2).
        robot_pose_world: Current robot pose (x, y, yaw) in world frame.
        cfg: Fusion configuration.
        vlm_remaining_frac: Fraction of VLM trajectory remaining after robot has moved
            along it. Used by horizon-aware modes to only compare overlapping portions.
            E.g., if VLM was 4s and 1.5s has been consumed, remaining_frac = 0.625.
    """
    cand = np.asarray(candidate_world, dtype=np.float64)
    ref = np.asarray(stale_ref_world, dtype=np.float64)
    if cand.ndim != 2 or cand.shape[1] != 2 or cand.shape[0] < 2:
        return float("-inf")
    if ref.ndim != 2 or ref.shape[1] != 2 or ref.shape[0] < 2:
        return float("-inf")

    x0, y0, yaw0 = (
        float(robot_pose_world[0]),
        float(robot_pose_world[1]),
        float(robot_pose_world[2]),
    )

    dist_scale = float(max(1e-6, float(cfg.dist_scale_m)))
    mode = str(cfg.similarity_mode)

    if mode == "body_pointwise_horizon_aware":
        # Horizon-aware comparison: only compare the overlapping time horizon.
        # If VLM has 60% remaining, only compare first 60% of candidate to 100% of remaining VLM.
        # This avoids the problem of stretching a short leftover VLM trajectory to match a full
        # candidate.
        cand_body = world_to_body(cand, x=x0, y=y0, yaw=yaw0)
        ref_body = world_to_body(ref, x=x0, y=y0, yaw=yaw0)

        # Project current pose onto ref to get remaining segment
        if bool(cfg.align_to_current_pose):
            s0, _d0 = project_point_to_polyline_arclength(
                np.asarray([0.0, 0.0], dtype=np.float64), ref_body
            )
            ref_total_len = polyline_length(ref_body)
            remaining_len = max(0.0, ref_total_len - s0)
            actual_remaining_frac = remaining_len / max(1e-6, ref_total_len)
        else:
            actual_remaining_frac = float(vlm_remaining_frac)

        # Clamp to reasonable range
        frac = float(np.clip(actual_remaining_frac, 0.1, 1.0))

        # Take first frac of candidate points to compare
        n_cand_use = max(2, int(round(cand_body.shape[0] * frac)))
        cand_subset = cand_body[:n_cand_use]

        # Sample the remaining portion of ref (from current position to end)
        if bool(cfg.align_to_current_pose):
            ref_remaining = _sample_polyline_by_arclength(ref_body, s_start=float(s0), n=n_cand_use)
        else:
            ref_remaining = _resample_polyline_index(ref_body, n_cand_use)

        d = _mean_pointwise_distance(cand_subset, ref_remaining)
        return -float(d / dist_scale)

    if mode == "body_pointwise_arclen":
        # Compare in the *current robot frame*.
        # This tends to be more stable than comparing in world space when the stale plan is from an
        # old pose.
        cand_body = world_to_body(cand, x=x0, y=y0, yaw=yaw0)
        ref_body = world_to_body(ref, x=x0, y=y0, yaw=yaw0)
        if bool(cfg.align_to_current_pose):
            s0, _d0 = project_point_to_polyline_arclength(
                np.asarray([0.0, 0.0], dtype=np.float64), ref_body
            )
            ref_aligned = _sample_polyline_by_arclength(
                ref_body, s_start=float(s0), n=int(cand_body.shape[0])
            )
        else:
            ref_aligned = _resample_polyline_index(ref_body, int(cand_body.shape[0]))
        d = _mean_pointwise_distance(cand_body, ref_aligned)
        return -float(d / dist_scale)

    # Default: compare in world, optionally suffix-align by projecting the current pose onto the
    # stale plan.
    if bool(cfg.align_to_current_pose):
        s0, _d0 = project_point_to_polyline_arclength(np.asarray([x0, y0], dtype=np.float64), ref)
        ref_aligned = _sample_polyline_by_arclength(ref, s_start=float(s0), n=int(cand.shape[0]))
    else:
        ref_aligned = _resample_polyline_index(ref, int(cand.shape[0]))

    if mode == "pointwise_arclen":
        d = _mean_pointwise_distance(cand, ref_aligned)
        return -float(d / dist_scale)
    if mode == "candidate_to_ref_polyline":
        d = _mean_candidate_to_ref_polyline_distance(cand, ref_aligned)
        return -float(d / dist_scale)
    if mode == "symmetric_polyline":
        d1 = _mean_candidate_to_ref_polyline_distance(cand, ref_aligned)
        d2 = _mean_candidate_to_ref_polyline_distance(ref_aligned, cand)
        d = 0.5 * (float(d1) + float(d2))
        return -float(d / dist_scale)
    # Fallback: pointwise.
    d = _mean_pointwise_distance(cand, ref_aligned)
    return -float(d / dist_scale)


def _planner_scores(
    *,
    candidates_body: list[np.ndarray],
    ideal_body: np.ndarray,
    score_model: PlannerScoreModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetic planner scores: negative distance-to-ideal + noise."""
    ideal = np.asarray(ideal_body, dtype=np.float64)
    scores: list[float] = []
    for c in candidates_body:
        c2 = _resample_polyline_index(np.asarray(c, dtype=np.float64), int(ideal.shape[0]))
        d = _mean_pointwise_distance(c2, ideal)
        base = -float(d)
        noise = float(rng.normal(loc=0.0, scale=float(score_model.noise_std)))
        scores.append(float(score_model.score_scale) * base + noise)
    return np.asarray(scores, dtype=np.float64)


def _planner_scores_world(
    *,
    candidates_world: list[np.ndarray],
    ideal_seg_world: np.ndarray,
    score_model: PlannerScoreModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetic planner scores: negative ADE-to-ideal-segment + noise (world frame)."""
    ideal = np.asarray(ideal_seg_world, dtype=np.float64)
    scores: list[float] = []
    for c in candidates_world:
        c2 = _resample_polyline_index(np.asarray(c, dtype=np.float64), int(ideal.shape[0]))
        d = _mean_pointwise_distance(c2, ideal)
        base = -float(d)
        noise = float(rng.normal(loc=0.0, scale=float(score_model.noise_std)))
        scores.append(float(score_model.score_scale) * base + noise)
    return np.asarray(scores, dtype=np.float64)


def _vlm_choose_idx_oracle(*, candidates_body: list[np.ndarray], ideal_body: np.ndarray) -> int:
    ideal = np.asarray(ideal_body, dtype=np.float64)
    best_i = 0
    best = float("inf")
    for i, c in enumerate(candidates_body):
        c2 = _resample_polyline_index(np.asarray(c, dtype=np.float64), int(ideal.shape[0]))
        d = _mean_pointwise_distance(c2, ideal)
        if d < best:
            best = float(d)
            best_i = int(i)
    return int(best_i)


def _vlm_choose_idx_oracle_world(
    *, candidates_world: list[np.ndarray], ideal_seg_world: np.ndarray
) -> int:
    """Oracle selector: pick the candidate with minimum ADE to the aligned ideal segment."""
    ideal = np.asarray(ideal_seg_world, dtype=np.float64)
    best_i = 0
    best = float("inf")
    for i, c in enumerate(candidates_world):
        c2 = _resample_polyline_index(np.asarray(c, dtype=np.float64), int(ideal.shape[0]))
        d = _mean_pointwise_distance(c2, ideal)
        if d < best:
            best = float(d)
            best_i = int(i)
    return int(best_i)


def _make_task(task: TaskName, *, v_ref: float, radius_m: float) -> ToyTask:
    name = str(task)
    v = float(v_ref)
    if name == "forward":
        return ToyTask(name="forward", v_ref=v, w_ref=0.0)
    if float(radius_m) <= 1e-6:
        raise ValueError("radius_m must be > 0 for turns")
    w = float(v / float(radius_m))
    if name == "left_turn":
        return ToyTask(name="left_turn", v_ref=v, w_ref=abs(w))
    if name == "right_turn":
        return ToyTask(name="right_turn", v_ref=v, w_ref=-abs(w))
    raise ValueError(f"Unknown task: {task}")


def make_task(task: TaskName, *, v_ref: float, radius_m: float) -> ToyTask:
    """Public wrapper for building the standard toy tasks."""
    return _make_task(task, v_ref=float(v_ref), radius_m=float(radius_m))


def run_toy_simulation(
    *,
    candidates_body: list[np.ndarray],
    task: ToyTask,
    controller: ControllerName,
    policy: PolicyName,
    sim_cfg: SimLoopConfig,
    score_model: PlannerScoreModel,
    vlm_cfg: VlmDelayModel,
    fusion_cfg: FusionConfig,
    seed: int = 0,
) -> SimResult:
    if not candidates_body:
        raise ValueError("candidates_body is empty")
    if float(sim_cfg.dt_control) <= 0 or float(sim_cfg.dt_plan) <= 0:
        raise ValueError("dt_control and dt_plan must be > 0")
    if float(sim_cfg.dt_plan) + 1e-9 < float(sim_cfg.dt_control):
        raise ValueError(
            "Require dt_plan >= dt_control (planner tick should not be faster than control tick)."
        )
    if float(sim_cfg.horizon_s) <= 0:
        raise ValueError("horizon_s must be > 0")

    rng = np.random.default_rng(int(seed))
    limits = sim_cfg.limits

    # Normalize candidate point counts (assume all candidates have same count; enforce by
    # resampling).
    n0 = int(candidates_body[0].shape[0])
    if n0 < 2:
        raise ValueError("candidate must have >=2 points")
    candidates_body2 = [_resample_polyline_index(c, n0) for c in candidates_body]

    # Build ideal trajectory in world for evaluation (ground truth "commanded" path).
    dt = float(sim_cfg.dt_control)
    steps = int(max(1, round(float(task.duration_s) / dt)))
    ideal_world = np.zeros((steps + 1, 2), dtype=np.float64)
    # Ideal starts at origin (same as sim start).
    x_i = 0.0
    y_i = 0.0
    yaw_i = 0.0
    for k in range(steps):
        x_i, y_i, yaw_i = integrate_diff_drive(
            x_i, y_i, yaw_i, float(task.v_ref), float(task.w_ref), dt
        )
        ideal_world[k + 1, 0] = x_i
        ideal_world[k + 1, 1] = y_i

    # Sim state.
    x = 0.0
    y = 0.0
    yaw = 0.0
    v = 0.0
    w = 0.0

    # Plan currently being tracked (world polyline).
    ref_world: np.ndarray | None = None
    chosen_idx_hist: list[int] = []

    # Dummy VLM request/response queue.
    pending: list[tuple[float, int, np.ndarray]] = []  # (deliver_time, request_idx, ref_world_path)
    last_vlm_ref_world: np.ndarray | None = None
    last_vlm_request_t: float | None = None
    vlm_updates = 0

    # Metrics traces.
    pos_errs: list[float] = []
    ctes: list[float] = []
    speeds: list[float] = []
    dvs: list[float] = []
    dws: list[float] = []
    stopped = 0
    plan_steps = 0

    plan_every = int(max(1, round(float(sim_cfg.dt_plan) / dt)))

    for k in range(steps):
        t = float(k) * dt

        # Deliver any ready VLM responses.
        if pending:
            still: list[tuple[float, int, np.ndarray]] = []
            for deliver_t, _req_idx, path_w in pending:
                if float(deliver_t) <= t + 1e-9:
                    last_vlm_ref_world = np.asarray(path_w, dtype=np.float64)
                    last_vlm_request_t = float(deliver_t) - float(vlm_cfg.delay_s)
                    vlm_updates += 1
                else:
                    still.append((float(deliver_t), int(_req_idx), path_w))
            pending = still

        # Replan step.
        if k % plan_every == 0 or ref_world is None:
            plan_steps += 1
            # Candidates in world (frozen at this planning pose).
            cand_world = [body_to_world(c, x=x, y=y, yaw=yaw) for c in candidates_body2]

            # Build the *aligned* ideal segment for this planning pose by projecting the robot
            # position
            # onto the full reference polyline, then taking a horizon-length window.
            s0, _d0 = project_point_to_polyline_arclength(
                np.asarray([x, y], dtype=np.float64), ideal_world
            )
            seg_len = float(abs(float(task.v_ref)) * float(sim_cfg.horizon_s))
            ideal_seg_world = _sample_polyline_by_arclength_window(
                ideal_world, s_start=float(s0), s_len=float(seg_len), n=int(n0)
            )

            # Synthetic planner scores (world-frame ADE to aligned ideal segment).
            scores = _planner_scores_world(
                candidates_world=cand_world,
                ideal_seg_world=ideal_seg_world,
                score_model=score_model,
                rng=rng,
            )

            # Enqueue dummy VLM request each plan tick (to mimic continuous offboard querying).
            if float(vlm_cfg.delay_s) >= 0:
                idx_oracle = _vlm_choose_idx_oracle_world(
                    candidates_world=cand_world, ideal_seg_world=ideal_seg_world
                )
                if float(vlm_cfg.delay_s) <= 1e-9:
                    # Deliver immediately (avoid a 1-tick artifact when delay==0).
                    last_vlm_ref_world = np.asarray(cand_world[int(idx_oracle)], dtype=np.float64)
                    last_vlm_request_t = float(t)
                    vlm_updates += 1
                else:
                    deliver = t + float(vlm_cfg.delay_s)
                    pending.append(
                        (
                            float(deliver),
                            int(idx_oracle),
                            np.asarray(cand_world[int(idx_oracle)], dtype=np.float64),
                        )
                    )

            # Choose executed candidate.
            policy_name = str(policy)
            if policy_name == "vlm_hold":
                if last_vlm_ref_world is not None:
                    # Track the last VLM plan, but only its *remaining* segment from the current
                    # pose.
                    # This avoids pure pursuit locking onto behind-me points when the VLM plan is
                    # stale.
                    stale_ref = np.asarray(last_vlm_ref_world, dtype=np.float64)
                    if bool(fusion_cfg.align_to_current_pose):
                        s_vlm, _d_vlm = project_point_to_polyline_arclength(
                            np.asarray([x, y], dtype=np.float64), stale_ref
                        )
                        ref_world = _sample_polyline_by_arclength_window(
                            stale_ref, s_start=float(s_vlm), s_len=float(seg_len), n=int(n0)
                        )
                    else:
                        ref_world = _resample_polyline_index(stale_ref, int(n0))
                    # Choose an index only for logging (closest current candidate to stale plan).
                    sims = [
                        _similarity(
                            candidate_world=np.asarray(cw, dtype=np.float64),
                            stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                            robot_pose_world=(float(x), float(y), float(yaw)),
                            cfg=fusion_cfg,
                        )
                        for cw in cand_world
                    ]
                    chosen = int(np.argmax(np.asarray(sims, dtype=np.float64)))
                else:
                    # Fallback: corrupt scores with epsilon
                    corrupted_scores = _corrupt_scores_with_epsilon(
                        scores, float(score_model.epsilon), rng
                    )
                    chosen = int(np.argmax(corrupted_scores))
                    ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy_name == "score_fusion":
                # Corrupt planner scores with epsilon probability
                corrupted_scores = _corrupt_scores_with_epsilon(
                    scores, float(score_model.epsilon), rng
                )
                # Apply fusion if lambda > 0 and VLM is available
                if (
                    fusion_cfg.enabled
                    and float(fusion_cfg.lambda_sim) > 0
                    and last_vlm_ref_world is not None
                    and last_vlm_request_t is not None
                ):
                    stale = float(max(0.0, t - float(last_vlm_request_t)))
                    tau = float(max(1e-6, float(fusion_cfg.staleness_tau_s)))
                    weight = math.exp(-stale / tau)
                    sims = np.asarray(
                        [
                            _similarity(
                                candidate_world=np.asarray(cw, dtype=np.float64),
                                stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                                robot_pose_world=(float(x), float(y), float(yaw)),
                                cfg=fusion_cfg,
                            )
                            for cw in cand_world
                        ],
                        dtype=np.float64,
                    )
                    # Fuse CORRUPTED scores with VLM similarity
                    fused = corrupted_scores + float(fusion_cfg.lambda_sim) * float(weight) * sims
                    chosen = int(np.argmax(fused))
                else:
                    # No fusion: use corrupted scores
                    chosen = int(np.argmax(corrupted_scores))
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy_name == "prob_fusion":
                # Probability-space fusion:
                # - planner -> p_planner = softmax(corrupted_scores)
                # - VLM -> p_vlm = softmax(similarity / vlm_temp)
                # - fuse -> p = (1-alpha)*p_planner + alpha*p_vlm
                # where alpha depends on staleness:
                #   alpha = lambda / (lambda + 1) * exp(-stale/tau)
                corrupted_scores = _corrupt_scores_with_epsilon(
                    scores, float(score_model.epsilon), rng
                )
                p_planner = _softmax(corrupted_scores)
                if (
                    fusion_cfg.enabled
                    and float(fusion_cfg.lambda_sim) > 0
                    and last_vlm_ref_world is not None
                    and last_vlm_request_t is not None
                ):
                    stale = float(max(0.0, t - float(last_vlm_request_t)))
                    tau = float(max(1e-6, float(fusion_cfg.staleness_tau_s)))
                    decay = math.exp(-stale / tau)
                    sims = np.asarray(
                        [
                            _similarity(
                                candidate_world=np.asarray(cw, dtype=np.float64),
                                stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                                robot_pose_world=(float(x), float(y), float(yaw)),
                                cfg=fusion_cfg,
                            )
                            for cw in cand_world
                        ],
                        dtype=np.float64,
                    )
                    # Temperature to reduce VLM softmax peakiness
                    vlm_temp = 1.0
                    p_vlm = _softmax(sims / vlm_temp)
                    # More gradual alpha formula
                    lam = float(max(0.0, float(fusion_cfg.lambda_sim)))
                    alpha_base = lam / (lam + 1.0)
                    alpha = float(alpha_base * decay)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    fused_p = (1.0 - alpha) * p_planner + alpha * p_vlm
                    chosen = int(np.argmax(fused_p))
                else:
                    chosen = int(np.argmax(p_planner))
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy_name == "chunk_stitch":
                # VLA-RAIL-style: execute the stale VLM trajectory, smooth the seam to
                # the previously-executed chunk via cubic-Hermite blend on (pos, tangent).
                if last_vlm_ref_world is not None:
                    ref_world = _chunk_stitch_ref_world(
                        stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                        robot_pose_world=(float(x), float(y), float(yaw)),
                        seg_len=float(seg_len),
                        n=int(n0),
                    )
                    sims = [
                        _similarity(
                            candidate_world=np.asarray(cw, dtype=np.float64),
                            stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                            robot_pose_world=(float(x), float(y), float(yaw)),
                            cfg=fusion_cfg,
                        )
                        for cw in cand_world
                    ]
                    chosen = int(np.argmax(np.asarray(sims, dtype=np.float64)))
                else:
                    corrupted_scores = _corrupt_scores_with_epsilon(
                        scores, float(score_model.epsilon), rng
                    )
                    chosen = int(np.argmax(corrupted_scores))
                    ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            else:
                # local_only: corrupt scores with epsilon
                corrupted_scores = _corrupt_scores_with_epsilon(
                    scores, float(score_model.epsilon), rng
                )
                chosen = int(np.argmax(corrupted_scores))
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)

            chosen_idx_hist.append(int(chosen))

        # Controller step: follow current frozen ref_world.
        path_body = world_to_body(np.asarray(ref_world, dtype=np.float64), x=x, y=y, yaw=yaw)
        if controller == "endpoint_pd":
            v_cmd, w_cmd = controller_endpoint_pd(
                path_body, v_prev=v, w_prev=w, limits=limits, cfg=sim_cfg.ep_cfg
            )
        else:
            v_cmd, w_cmd = controller_pure_pursuit(
                path_body, v_prev=v, w_prev=w, limits=limits, cfg=sim_cfg.pp_cfg
            )
            # Apply rate limits using dt (controller_pure_pursuit uses internal ~0.1s defaults).
            dv = float(limits.max_a) * dt
            dw = float(limits.max_alpha) * dt
            v_cmd = float(np.clip(v_cmd, v - dv, v + dv))
            w_cmd = float(np.clip(w_cmd, w - dw, w + dw))

        v_new = float(np.clip(v_cmd, 0.0, float(limits.max_v)))
        w_new = float(np.clip(w_cmd, -float(limits.max_w), float(limits.max_w)))

        dvs.append(abs(v_new - v))
        dws.append(abs(w_new - w))
        v = v_new
        w = w_new
        if v < 1e-3:
            stopped += 1
        speeds.append(float(v))

        x, y, yaw = integrate_diff_drive(x, y, yaw, v, w, dt)

        # Metrics: position error vs ideal world, and CTE to ideal world polyline.
        gt_xy = ideal_world[min(k + 1, ideal_world.shape[0] - 1)]
        pos_errs.append(float(np.linalg.norm(np.asarray([x, y], dtype=np.float64) - gt_xy)))
        _s_ideal, cte = project_point_to_polyline_arclength(
            np.asarray([x, y], dtype=np.float64), ideal_world
        )
        ctes.append(float(cte))

    def _p(x: list[float], q: float) -> float:
        if not x:
            return float("nan")
        return float(np.quantile(np.asarray(x, dtype=np.float64), q))

    return SimResult(
        mean_pos_err_m=float(np.mean(pos_errs)) if pos_errs else float("nan"),
        p95_pos_err_m=_p(pos_errs, 0.95),
        mean_cte_to_ideal_m=float(np.mean(ctes)) if ctes else float("nan"),
        p95_cte_to_ideal_m=_p(ctes, 0.95),
        mean_speed_mps=float(np.mean(speeds)) if speeds else float("nan"),
        stopped_frac=float(stopped) / float(max(1, len(speeds))),
        mean_abs_dv=float(np.mean(dvs)) if dvs else float("nan"),
        mean_abs_dw=float(np.mean(dws)) if dws else float("nan"),
        steps=int(steps),
        plan_steps=int(plan_steps),
        vlm_updates=int(vlm_updates),
        chosen_idx_hist=chosen_idx_hist,
    )


def run_toy_simulation_with_trace(
    *,
    candidates_body: list[np.ndarray],
    task: ToyTask,
    controller: ControllerName,
    policy: PolicyName,
    sim_cfg: SimLoopConfig,
    score_model: PlannerScoreModel,
    vlm_cfg: VlmDelayModel,
    fusion_cfg: FusionConfig,
    seed: int = 0,
) -> tuple[SimResult, SimTrace]:
    """Same as `run_toy_simulation`, but also returns full traces for plotting."""
    if not candidates_body:
        raise ValueError("candidates_body is empty")
    if float(sim_cfg.dt_control) <= 0 or float(sim_cfg.dt_plan) <= 0:
        raise ValueError("dt_control and dt_plan must be > 0")
    if float(sim_cfg.dt_plan) + 1e-9 < float(sim_cfg.dt_control):
        raise ValueError(
            "Require dt_plan >= dt_control (planner tick should not be faster than control tick)."
        )
    if float(sim_cfg.horizon_s) <= 0:
        raise ValueError("horizon_s must be > 0")

    rng = np.random.default_rng(int(seed))
    limits = sim_cfg.limits

    # Normalize candidate point counts (assume all candidates have same count; enforce by
    # resampling).
    n0 = int(candidates_body[0].shape[0])
    if n0 < 2:
        raise ValueError("candidate must have >=2 points")
    candidates_body2 = [_resample_polyline_index(c, n0) for c in candidates_body]

    # Build ideal trajectory in world for evaluation.
    dt = float(sim_cfg.dt_control)
    steps = int(max(1, round(float(task.duration_s) / dt)))
    ideal_world = np.zeros((steps + 1, 2), dtype=np.float64)
    x_i = 0.0
    y_i = 0.0
    yaw_i = 0.0
    for k in range(steps):
        x_i, y_i, yaw_i = integrate_diff_drive(
            x_i, y_i, yaw_i, float(task.v_ref), float(task.w_ref), dt
        )
        ideal_world[k + 1, 0] = x_i
        ideal_world[k + 1, 1] = y_i

    # Sim state.
    x = 0.0
    y = 0.0
    yaw = 0.0
    v = 0.0
    w = 0.0

    # Plan currently being tracked (world polyline).
    ref_world: np.ndarray | None = None
    chosen_idx_hist: list[int] = []

    # Dummy VLM request/response queue.
    pending: list[tuple[float, int, np.ndarray]] = []  # (deliver_time, request_idx, ref_world_path)
    last_vlm_ref_world: np.ndarray | None = None
    last_vlm_request_t: float | None = None
    vlm_updates = 0

    # Metrics traces.
    pos_errs: list[float] = []
    ctes: list[float] = []
    speeds: list[float] = []
    dvs: list[float] = []
    dws: list[float] = []
    stopped = 0
    plan_steps = 0

    # Trace buffers (control tick).
    t_arr = np.linspace(0.0, float(steps) * dt, int(steps) + 1, dtype=np.float64)
    x_arr = np.zeros((steps + 1,), dtype=np.float64)
    y_arr = np.zeros((steps + 1,), dtype=np.float64)
    yaw_arr = np.zeros((steps + 1,), dtype=np.float64)
    v_arr = np.zeros((steps + 1,), dtype=np.float64)
    w_arr = np.zeros((steps + 1,), dtype=np.float64)
    actual_world = np.zeros((steps + 1, 2), dtype=np.float64)
    # initial
    x_arr[0] = float(x)
    y_arr[0] = float(y)
    yaw_arr[0] = float(yaw)
    v_arr[0] = float(v)
    w_arr[0] = float(w)
    actual_world[0, 0] = float(x)
    actual_world[0, 1] = float(y)

    # Trace buffers (planner tick).
    plan_t: list[float] = []
    plan_chosen: list[int] = []
    plan_ref_world: list[np.ndarray] = []
    plan_vlm_world: list[np.ndarray | None] = []
    plan_stale_s: list[float] = []
    plan_weight: list[float] = []

    plan_every = int(max(1, round(float(sim_cfg.dt_plan) / dt)))

    for k in range(steps):
        t = float(k) * dt

        # Deliver any ready VLM responses.
        if pending:
            still: list[tuple[float, int, np.ndarray]] = []
            for deliver_t, _req_idx, path_w in pending:
                if float(deliver_t) <= t + 1e-9:
                    last_vlm_ref_world = np.asarray(path_w, dtype=np.float64)
                    last_vlm_request_t = float(deliver_t) - float(vlm_cfg.delay_s)
                    vlm_updates += 1
                else:
                    still.append((float(deliver_t), int(_req_idx), path_w))
            pending = still

        did_plan = (k % plan_every == 0) or (ref_world is None)
        if did_plan:
            plan_steps += 1

            cand_world = [body_to_world(c, x=x, y=y, yaw=yaw) for c in candidates_body2]
            s0, _d0 = project_point_to_polyline_arclength(
                np.asarray([x, y], dtype=np.float64), ideal_world
            )
            seg_len = float(abs(float(task.v_ref)) * float(sim_cfg.horizon_s))
            ideal_seg_world = _sample_polyline_by_arclength_window(
                ideal_world, s_start=float(s0), s_len=float(seg_len), n=int(n0)
            )
            scores = _planner_scores_world(
                candidates_world=cand_world,
                ideal_seg_world=ideal_seg_world,
                score_model=score_model,
                rng=rng,
            )

            # Enqueue dummy VLM request each plan tick.
            if float(vlm_cfg.delay_s) >= 0:
                idx_oracle = _vlm_choose_idx_oracle_world(
                    candidates_world=cand_world, ideal_seg_world=ideal_seg_world
                )
                if float(vlm_cfg.delay_s) <= 1e-9:
                    last_vlm_ref_world = np.asarray(cand_world[int(idx_oracle)], dtype=np.float64)
                    last_vlm_request_t = float(t)
                    vlm_updates += 1
                else:
                    deliver = t + float(vlm_cfg.delay_s)
                    pending.append(
                        (
                            float(deliver),
                            int(idx_oracle),
                            np.asarray(cand_world[int(idx_oracle)], dtype=np.float64),
                        )
                    )

            policy_name = str(policy)
            staleness = float("nan")
            weight = 0.0

            if policy_name == "vlm_hold":
                if last_vlm_ref_world is not None:
                    stale_ref = np.asarray(last_vlm_ref_world, dtype=np.float64)
                    if bool(fusion_cfg.align_to_current_pose):
                        s_vlm, _d_vlm = project_point_to_polyline_arclength(
                            np.asarray([x, y], dtype=np.float64), stale_ref
                        )
                        ref_world = _sample_polyline_by_arclength_window(
                            stale_ref, s_start=float(s_vlm), s_len=float(seg_len), n=int(n0)
                        )
                    else:
                        ref_world = _resample_polyline_index(stale_ref, int(n0))
                    sims = [
                        _similarity(
                            candidate_world=np.asarray(cw, dtype=np.float64),
                            stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                            robot_pose_world=(float(x), float(y), float(yaw)),
                            cfg=fusion_cfg,
                        )
                        for cw in cand_world
                    ]
                    chosen = int(np.argmax(np.asarray(sims, dtype=np.float64)))
                else:
                    # Fallback: corrupt scores with epsilon
                    corrupted_scores = _corrupt_scores_with_epsilon(
                        scores, float(score_model.epsilon), rng
                    )
                    chosen = int(np.argmax(corrupted_scores))
                    ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy_name == "score_fusion":
                # Corrupt planner scores with epsilon probability
                corrupted_scores = _corrupt_scores_with_epsilon(
                    scores, float(score_model.epsilon), rng
                )
                # Apply fusion if lambda > 0 and VLM is available
                if (
                    fusion_cfg.enabled
                    and float(fusion_cfg.lambda_sim) > 0
                    and last_vlm_ref_world is not None
                    and last_vlm_request_t is not None
                ):
                    staleness = float(max(0.0, t - float(last_vlm_request_t)))
                    tau = float(max(1e-6, float(fusion_cfg.staleness_tau_s)))
                    weight = math.exp(-staleness / tau)
                    sims = np.asarray(
                        [
                            _similarity(
                                candidate_world=np.asarray(cw, dtype=np.float64),
                                stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                                robot_pose_world=(float(x), float(y), float(yaw)),
                                cfg=fusion_cfg,
                            )
                            for cw in cand_world
                        ],
                        dtype=np.float64,
                    )
                    # Fuse CORRUPTED scores with VLM similarity
                    fused = corrupted_scores + float(fusion_cfg.lambda_sim) * float(weight) * sims
                    chosen = int(np.argmax(fused))
                else:
                    # No fusion: use corrupted scores
                    chosen = int(np.argmax(corrupted_scores))
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy_name == "prob_fusion":
                corrupted_scores = _corrupt_scores_with_epsilon(
                    scores, float(score_model.epsilon), rng
                )
                p_planner = _softmax(corrupted_scores)
                if (
                    fusion_cfg.enabled
                    and float(fusion_cfg.lambda_sim) > 0
                    and last_vlm_ref_world is not None
                    and last_vlm_request_t is not None
                ):
                    staleness = float(max(0.0, t - float(last_vlm_request_t)))
                    tau = float(max(1e-6, float(fusion_cfg.staleness_tau_s)))
                    decay = math.exp(-staleness / tau)
                    sims = np.asarray(
                        [
                            _similarity(
                                candidate_world=np.asarray(cw, dtype=np.float64),
                                stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                                robot_pose_world=(float(x), float(y), float(yaw)),
                                cfg=fusion_cfg,
                            )
                            for cw in cand_world
                        ],
                        dtype=np.float64,
                    )
                    # Temperature to reduce VLM softmax peakiness
                    vlm_temp = 1.0
                    p_vlm = _softmax(sims / vlm_temp)
                    # More gradual alpha formula
                    lam = float(max(0.0, float(fusion_cfg.lambda_sim)))
                    alpha_base = lam / (lam + 1.0)
                    alpha = float(alpha_base * decay)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    fused_p = (1.0 - alpha) * p_planner + alpha * p_vlm
                    chosen = int(np.argmax(fused_p))
                else:
                    chosen = int(np.argmax(p_planner))
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy_name == "chunk_stitch":
                if last_vlm_ref_world is not None:
                    ref_world = _chunk_stitch_ref_world(
                        stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                        robot_pose_world=(float(x), float(y), float(yaw)),
                        seg_len=float(seg_len),
                        n=int(n0),
                    )
                    sims = [
                        _similarity(
                            candidate_world=np.asarray(cw, dtype=np.float64),
                            stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                            robot_pose_world=(float(x), float(y), float(yaw)),
                            cfg=fusion_cfg,
                        )
                        for cw in cand_world
                    ]
                    chosen = int(np.argmax(np.asarray(sims, dtype=np.float64)))
                else:
                    corrupted_scores = _corrupt_scores_with_epsilon(
                        scores, float(score_model.epsilon), rng
                    )
                    chosen = int(np.argmax(corrupted_scores))
                    ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            else:
                # local_only: corrupt scores with epsilon
                corrupted_scores = _corrupt_scores_with_epsilon(
                    scores, float(score_model.epsilon), rng
                )
                chosen = int(np.argmax(corrupted_scores))
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)

            chosen_idx_hist.append(int(chosen))
            # Planner-tick trace
            plan_t.append(float(t))
            plan_chosen.append(int(chosen))
            plan_ref_world.append(np.asarray(ref_world, dtype=np.float64).copy())
            plan_vlm_world.append(
                np.asarray(last_vlm_ref_world, dtype=np.float64).copy()
                if last_vlm_ref_world is not None
                else None
            )
            plan_stale_s.append(float(staleness))
            plan_weight.append(float(weight))

        # Controller step: follow current frozen ref_world.
        path_body = world_to_body(np.asarray(ref_world, dtype=np.float64), x=x, y=y, yaw=yaw)
        if controller == "endpoint_pd":
            v_cmd, w_cmd = controller_endpoint_pd(
                path_body, v_prev=v, w_prev=w, limits=limits, cfg=sim_cfg.ep_cfg
            )
        else:
            v_cmd, w_cmd = controller_pure_pursuit(
                path_body, v_prev=v, w_prev=w, limits=limits, cfg=sim_cfg.pp_cfg
            )
            dv = float(limits.max_a) * dt
            dw = float(limits.max_alpha) * dt
            v_cmd = float(np.clip(v_cmd, v - dv, v + dv))
            w_cmd = float(np.clip(w_cmd, w - dw, w + dw))

        v_new = float(np.clip(v_cmd, 0.0, float(limits.max_v)))
        w_new = float(np.clip(w_cmd, -float(limits.max_w), float(limits.max_w)))

        dvs.append(abs(v_new - v))
        dws.append(abs(w_new - w))
        v = v_new
        w = w_new
        if v < 1e-3:
            stopped += 1
        speeds.append(float(v))

        x, y, yaw = integrate_diff_drive(x, y, yaw, v, w, dt)

        # Store control-tick trace at k+1.
        x_arr[k + 1] = float(x)
        y_arr[k + 1] = float(y)
        yaw_arr[k + 1] = float(yaw)
        v_arr[k + 1] = float(v)
        w_arr[k + 1] = float(w)
        actual_world[k + 1, 0] = float(x)
        actual_world[k + 1, 1] = float(y)

        # Metrics: position error vs ideal world, and CTE to ideal world polyline.
        gt_xy = ideal_world[min(k + 1, ideal_world.shape[0] - 1)]
        pos_errs.append(float(np.linalg.norm(np.asarray([x, y], dtype=np.float64) - gt_xy)))
        _s_ideal, cte = project_point_to_polyline_arclength(
            np.asarray([x, y], dtype=np.float64), ideal_world
        )
        ctes.append(float(cte))

    def _p(x: list[float], q: float) -> float:
        if not x:
            return float("nan")
        return float(np.quantile(np.asarray(x, dtype=np.float64), q))

    res = SimResult(
        mean_pos_err_m=float(np.mean(pos_errs)) if pos_errs else float("nan"),
        p95_pos_err_m=_p(pos_errs, 0.95),
        mean_cte_to_ideal_m=float(np.mean(ctes)) if ctes else float("nan"),
        p95_cte_to_ideal_m=_p(ctes, 0.95),
        mean_speed_mps=float(np.mean(speeds)) if speeds else float("nan"),
        stopped_frac=float(stopped) / float(max(1, len(speeds))),
        mean_abs_dv=float(np.mean(dvs)) if dvs else float("nan"),
        mean_abs_dw=float(np.mean(dws)) if dws else float("nan"),
        steps=int(steps),
        plan_steps=int(plan_steps),
        vlm_updates=int(vlm_updates),
        chosen_idx_hist=chosen_idx_hist,
    )

    trace = SimTrace(
        dt_control=float(sim_cfg.dt_control),
        dt_plan=float(sim_cfg.dt_plan),
        t_s=np.asarray(t_arr, dtype=np.float64),
        x=np.asarray(x_arr, dtype=np.float64),
        y=np.asarray(y_arr, dtype=np.float64),
        yaw=np.asarray(yaw_arr, dtype=np.float64),
        v=np.asarray(v_arr, dtype=np.float64),
        w=np.asarray(w_arr, dtype=np.float64),
        ideal_world=np.asarray(ideal_world, dtype=np.float64),
        actual_world=np.asarray(actual_world, dtype=np.float64),
        plan_t_s=np.asarray(plan_t, dtype=np.float64),
        chosen_idx=np.asarray(plan_chosen, dtype=np.int64),
        ref_world=plan_ref_world,
        vlm_ref_world=plan_vlm_world,
        staleness_s=np.asarray(plan_stale_s, dtype=np.float64),
        fusion_weight=np.asarray(plan_weight, dtype=np.float64),
    )

    return res, trace
