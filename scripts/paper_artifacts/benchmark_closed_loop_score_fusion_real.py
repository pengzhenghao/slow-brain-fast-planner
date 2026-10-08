#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files

# Reuse the static candidate JSON format and fusion math from the toy sim.
from slow_brain_fast_planner.control.delayed_vlm_score_fusion import (  # noqa: E402
    _mean_candidate_to_ref_polyline_distance,
    _softmax,
    load_static_candidate_set,
    resolve_static_candidate_set_path,
)
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

PolicyName = Literal[
    "local_only",
    # Immediate oracle planner: argmin distance to current reference segment (no corruption, no
    # delay).
    "planner_oracle",
    # Directly execute the delayed VLM-selected trajectory.
    # Matches `deploy/NavFlow/run_planner.py` semantics for `control.policy`.
    "vlm_hold",
    "vlm_stream",
    # Execute a *current* planner candidate chosen to match the (stale) VLM trajectory.
    "vlm_hold_match",
    "vlm_stream_match",
    "score_fusion",
    "prob_fusion",
    # Stream-query (fixed cadence) + pipelined requests by default (like vlm_stream*),
    # but selection uses the corresponding fusion rule.
    "score_fusion_stream",
    "prob_fusion_stream",
]
SimilarityMode = Literal[
    "body_pointwise_arclen",
    "pointwise_arclen",
    "candidate_to_ref_polyline",
    "symmetric_polyline",
    "body_pointwise_horizon_aware",
]
ControllerName = Literal["endpoint_pd", "pure_pursuit"]
TaskMode = Literal["arclength", "time"]


def _try_import_tqdm():
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _read_robot_yaml_limits(repo_root: Path) -> tuple[float, float]:
    """Read optional deployment limits, otherwise return the public defaults."""
    p = repo_root / "deploy/NavFlow/visualnav_ros/deployment/config/robot.yaml"
    if not p.exists():
        return 1.0, 0.5
    txt = p.read_text(encoding="utf-8")
    mv = None
    mw = None
    for line in txt.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            if k == "max_v":
                mv = float(v)
            elif k == "max_w":
                mw = float(v)
        except Exception:
            continue
    return float(mv if mv is not None else 1.0), float(mw if mw is not None else 0.5)


def _resample_polyline_index(points_xy: np.ndarray, n: int) -> np.ndarray:
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


def _sample_polyline_by_arclength_window(
    points_xy: np.ndarray, *, s_start: float, s_len: float, n: int
) -> np.ndarray:
    """Sample n points from a polyline over arclength window [s_start, s_start+s_len]."""
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


def _mean_pointwise_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    a = np.asarray(a_xy, dtype=np.float64)
    b = np.asarray(b_xy, dtype=np.float64)
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return float("inf")
    d = a[:n] - b[:n]
    return float(np.mean(np.linalg.norm(d, axis=1)))


def _resample_polyline_arclen(points_xy: np.ndarray, n: int) -> np.ndarray:
    """Resample polyline to n points uniformly in arclength.

    This is important for similarity matching: a polyline's native sampling may be
    non-uniform, so comparing it to an arclength-resampled reference can break
    the "self-match" property (candidate vs itself).
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError("points_xy must be (N,2) with N>=2")
    n = int(max(2, int(n)))
    total = float(polyline_length(pts))
    if not math.isfinite(total) or total <= 1e-9:
        return np.repeat(pts[:1], repeats=n, axis=0)
    return _sample_polyline_by_arclength_window(pts, s_start=0.0, s_len=total, n=n)


@dataclass(frozen=True)
class FusionConfig:
    enabled: bool = True
    similarity_mode: SimilarityMode = "body_pointwise_arclen"
    lambda_sim: float = 1.0
    staleness_tau_s: float = 3.0
    dist_scale_m: float = 1.0
    align_to_current_pose: bool = True


@dataclass(frozen=True)
class PlannerScoreModel:
    """Synthetic S1 score model used for closed-loop replay.

    We define the "oracle" label as min ADE to the aligned reference segment.
    The planner score is the same objective + optional Gaussian noise.
    """

    noise_std: float = 0.15
    score_scale: float = 1.0
    # Epsilon: with probability epsilon, corrupt planner scores to simulate confusion.
    epsilon: float = 0.0
    # Temperature: <1 = sharper (more confident), >1 = wider (less confident).
    # Applied as: score = base_score / temperature
    temperature: float = 1.0


@dataclass(frozen=True)
class TaskWindow:
    episode_id: str
    start_t: float
    end_t: float
    # World-frame reference polyline (x,y) and initial yaw.
    ref_xy: np.ndarray  # (N,2)
    init_yaw: float


@dataclass(frozen=True)
class SimLoopConfig:
    dt_control: float = 0.1
    dt_plan: float = 0.2
    horizon_s: float = 4.0
    limits: Limits = Limits()
    ep_cfg: EndpointPDConfig = EndpointPDConfig()
    pp_cfg: PurePursuitConfig = PurePursuitConfig()


@dataclass(frozen=True)
class RunResult:
    task_id: str
    episode_id: str
    start_t: float
    controller: str
    policy: str
    delay_s: float
    seed: int
    # Config knobs
    similarity_mode: str
    lambda_sim: float
    staleness_tau_s: float
    dist_scale_m: float
    align_to_current_pose: bool
    noise_std: float
    epsilon: float
    temperature: float
    # VLM corruption (epsilon-greedy). 0.0 => oracle VLM, 1.0 => random VLM.
    vlm_mistake_prob: float
    # Metrics
    mean_cte_m: float
    p95_cte_m: float
    mean_speed_mps: float
    stopped_frac: float
    chosen_switches: int
    # Closed-loop outcome metrics (often more interpretable than CTE)
    route_completion_max: float
    route_completion_final: float
    goal_dist_final_m: float
    goal_dist_min_m: float
    success: int
    # Reference window stats (helps interpret completion saturation)
    ref_len_m: float
    ref_duration_s: float
    ref_mean_speed_mps: float
    sim_horizon_s: float
    completion_upper_bound: float
    # Behavior / selection agreement metrics (per-plan-tick)
    plan_ticks: int
    # How often we *requested* VLM (can differ from plan_ticks for stream policies)
    vlm_requests: int
    vlm_request_frac: float
    vlm_available_frac: float
    # VLM oracle at request time vs noisy argmax (same plan tick)
    oracle_eq_argmax_ratio: float
    # Delayed VLM "hold" suggestion (argmax similarity to stale VLM) vs current argmax
    vlm_like_eq_argmax_ratio: float
    # What this policy actually chose vs baselines
    chosen_eq_argmax_ratio: float
    chosen_eq_vlm_like_ratio: float
    chosen_eq_vlm_idx_ratio: float
    # How often this policy overrides the noisy argmax
    switch_from_argmax_ratio: float
    # Conditioning on "VLM suggests different than argmax" (only when VLM is available)
    vlm_like_diff_argmax_frac: float
    switch_from_argmax_given_vlm_diff_ratio: float
    switch_to_vlm_like_given_vlm_diff_ratio: float
    # Distribution shift of chosen index vs argmax index (total variation distance)
    tv_dist_chosen_vs_argmax: float


def _similarity(
    *,
    candidate_world: np.ndarray,
    stale_ref_world: np.ndarray,
    robot_pose_world: tuple[float, float, float],
    cfg: FusionConfig,
) -> float:
    """Similarity (higher is better) between current candidate and stale VLM trajectory."""
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
            actual_remaining_frac = 1.0

        # Clamp to reasonable range
        frac = float(np.clip(actual_remaining_frac, 0.1, 1.0))

        # Compare arclength-resampled prefixes to preserve self-match.
        n_cand_use = max(2, int(round(cand_body.shape[0] * frac)))
        cand_total = float(polyline_length(cand_body))
        cand_subset = _sample_polyline_by_arclength_window(
            cand_body,
            s_start=0.0,
            s_len=float(max(0.0, cand_total * frac)),
            n=n_cand_use,
        )

        # Sample the remaining portion of ref (from current position to end), arclength-uniform.
        if bool(cfg.align_to_current_pose):
            ref_remaining = _sample_polyline_by_arclength_window(
                ref_body, s_start=float(s0), s_len=float(remaining_len), n=n_cand_use
            )
        else:
            ref_remaining = _resample_polyline_arclen(ref_body, n_cand_use)

        d = _mean_pointwise_distance(cand_subset, ref_remaining)
        return -float(d / dist_scale)

    if mode == "body_pointwise_arclen":
        cand_body = world_to_body(cand, x=x0, y=y0, yaw=yaw0)
        ref_body = world_to_body(ref, x=x0, y=y0, yaw=yaw0)
        cand_u = _resample_polyline_arclen(cand_body, int(cand_body.shape[0]))
        if bool(cfg.align_to_current_pose):
            s0, _d0 = project_point_to_polyline_arclength(
                np.asarray([0.0, 0.0], dtype=np.float64), ref_body
            )
            ref_aligned = _sample_polyline_by_arclength_window(
                ref_body,
                s_start=float(s0),
                s_len=float(polyline_length(cand_body)),
                n=int(cand_body.shape[0]),
            )
        else:
            ref_aligned = _resample_polyline_arclen(ref_body, int(cand_body.shape[0]))
        d = _mean_pointwise_distance(cand_u, ref_aligned)
        return -float(d / dist_scale)

    # World-space compare, optionally progress-align
    if bool(cfg.align_to_current_pose):
        s0, _d0 = project_point_to_polyline_arclength(np.asarray([x0, y0], dtype=np.float64), ref)
        ref_aligned = _sample_polyline_by_arclength_window(
            ref, s_start=float(s0), s_len=float(polyline_length(cand)), n=int(cand.shape[0])
        )
    else:
        ref_aligned = _resample_polyline_index(ref, int(cand.shape[0]))

    if mode == "pointwise_arclen":
        cand_u = _resample_polyline_arclen(cand, int(cand.shape[0]))
        # ref_aligned is already arclength-uniform when align_to_current_pose=True.
        # When align_to_current_pose=False, it is index-resampled; make it arclength-uniform too.
        if not bool(cfg.align_to_current_pose):
            ref_aligned = _resample_polyline_arclen(ref_aligned, int(cand.shape[0]))
        d = _mean_pointwise_distance(cand_u, ref_aligned)
        return -float(d / dist_scale)

    if mode == "candidate_to_ref_polyline":
        d = _mean_candidate_to_ref_polyline_distance(cand, ref_aligned)
        return -float(d / dist_scale)
    if mode == "symmetric_polyline":
        d1 = _mean_candidate_to_ref_polyline_distance(cand, ref_aligned)
        d2 = _mean_candidate_to_ref_polyline_distance(ref_aligned, cand)
        return -float(0.5 * (d1 + d2) / dist_scale)
    # default: pointwise (index-param)
    d = _mean_pointwise_distance(cand, ref_aligned)
    return -float(d / dist_scale)


def _scores_to_ref_segment(
    *,
    candidates_world: list[np.ndarray],
    ref_seg_world: np.ndarray,
    score_model: PlannerScoreModel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute planner scores for candidates relative to reference segment.

    Temperature controls score sharpness:
    - temp < 1: sharper (more confident, larger score differences)
    - temp = 1: default
    - temp > 1: wider (less confident, smaller score differences)
    """
    ref = np.asarray(ref_seg_world, dtype=np.float64)
    temp = float(max(1e-6, float(score_model.temperature)))
    scores = []
    for c in candidates_world:
        c2 = _resample_polyline_index(np.asarray(c, dtype=np.float64), int(ref.shape[0]))
        d = _mean_pointwise_distance(c2, ref)
        base = -float(d)
        # Apply temperature: divide by temp to control sharpness
        base_scaled = base / temp
        noise = (
            float(rng.normal(0.0, float(score_model.noise_std)))
            if float(score_model.noise_std) > 0
            else 0.0
        )
        scores.append(float(score_model.score_scale) * base_scaled + noise)
    return np.asarray(scores, dtype=np.float64)


def _choose_oracle_idx(*, candidates_world: list[np.ndarray], ref_seg_world: np.ndarray) -> int:
    ref = np.asarray(ref_seg_world, dtype=np.float64)
    best_i = 0
    best = float("inf")
    for i, c in enumerate(candidates_world):
        c2 = _resample_polyline_index(np.asarray(c, dtype=np.float64), int(ref.shape[0]))
        d = _mean_pointwise_distance(c2, ref)
        if d < best:
            best = float(d)
            best_i = int(i)
    return int(best_i)


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
        score_range = float(scores.max() - scores.min()) if len(scores) > 1 else 1.0
        score_range = max(1.0, score_range)
        return rng.uniform(float(scores.min()), float(scores.max()), size=len(scores)).astype(
            np.float64
        )
    return scores


def simulate_closed_loop(
    *,
    task: TaskWindow,
    candidates_body: list[np.ndarray],
    sim_cfg: SimLoopConfig,
    score_model: PlannerScoreModel,
    controller: ControllerName,
    policy: PolicyName,
    delay_s: float,
    vlm_query_hz: float,
    vlm_max_inflight: int,
    # With probability `vlm_mistake_prob`, the VLM returns a random candidate instead of the oracle.
    # This models an imperfect VLM (epsilon-greedy).
    vlm_mistake_prob: float = 0.0,
    fusion_cfg: FusionConfig,
    seed: int,
    # Success/route completion definition
    success_min_completion: float = 0.95,
    success_goal_radius_m: float = 1.0,
    # If true, terminate rollout once success criteria is met.
    # This prevents post-goal wandering from polluting CTE / goal_final metrics.
    stop_on_success: bool = True,
    # Simulation horizon (time budget). If None, uses the reference window duration.
    time_limit_s: float | None = None,
    # If true, return rich trace data (plan history, etc.) for visualization.
    # Keep this False for large sweeps to avoid extra memory/CPU.
    return_trace: bool = False,
) -> tuple[RunResult, dict[str, np.ndarray]]:
    """Run a closed-loop rollout against a real-data reference polyline.

    - Robot starts from the first reference point with initial yaw from odom.
    - At each plan tick (dt_plan), a fixed library of robot-frame candidates is transformed into
    world.
    - The reference segment is the horizon-length arclength window starting at the closest point on
    the ref polyline.
    - \"VLM\" is a delayed oracle: it selects min-ADE candidate w.r.t that reference segment,
      returned after delay_s.
    """
    rng = np.random.default_rng(int(seed))
    dt = float(sim_cfg.dt_control)
    ref_duration_s = float(max(1e-9, float(task.end_t) - float(task.start_t)))
    sim_horizon_s = float(
        ref_duration_s if time_limit_s is None else max(1e-9, float(time_limit_s))
    )
    steps = int(max(1, round(float(sim_horizon_s) / dt)))
    plan_every = int(max(1, round(float(sim_cfg.dt_plan) / dt)))

    # Normalize candidate point counts.
    n0 = int(candidates_body[0].shape[0])
    candidates_body2 = [
        _resample_polyline_index(np.asarray(c, dtype=np.float64), n0) for c in candidates_body
    ]
    cand_len = float(np.mean([polyline_length(c) for c in candidates_body2]))
    cand_len = float(max(1e-3, cand_len))

    ref_xy = np.asarray(task.ref_xy, dtype=np.float64)
    if ref_xy.ndim != 2 or ref_xy.shape[1] != 2 or ref_xy.shape[0] < 2:
        raise ValueError("task.ref_xy must be (N,2) with N>=2")

    # Route completion is measured along the reference polyline arclength.
    # We track both:
    # - completion_final: completion at the end of rollout
    # - completion_max: max completion reached (monotonic progress proxy)
    ref_len = float(polyline_length(ref_xy))
    ref_len_safe = float(max(1e-6, ref_len))
    ref_mean_speed_mps = float(ref_len / ref_duration_s) if ref_len > 1e-9 else float("nan")
    # Upper bound: if we could drive at max_v perfectly aligned for the whole window.
    max_v = float(getattr(sim_cfg.limits, "max_v", 1.0))
    completion_upper_bound = (
        float(min(1.0, max_v * sim_horizon_s / ref_len_safe)) if ref_len > 1e-9 else float("nan")
    )
    s_last = 0.0
    s_max = 0.0

    # Success definition (used both for early stopping and final success flag)
    min_completion = float(np.clip(float(success_min_completion), 0.0, 1.0))
    goal_r = float(max(0.0, float(success_goal_radius_m)))
    goal_xy = np.asarray(ref_xy[-1], dtype=np.float64)

    # Initial pose: first reference point and yaw
    x = float(ref_xy[0, 0])
    y = float(ref_xy[0, 1])
    yaw = float(task.init_yaw)
    v = 0.0
    w = 0.0

    # VLM delay queue: (deliver_t, request_t, oracle_idx, path_world)
    pending: list[tuple[float, float, int, np.ndarray]] = []
    last_vlm_ref_world: np.ndarray | None = None
    last_vlm_idx: int | None = None
    last_vlm_request_t: float | None = None
    next_vlm_request_t: float = 0.0
    try:
        hz = float(vlm_query_hz)
        vlm_period_s = (1.0 / hz) if hz > 1e-6 else 0.0
    except Exception:
        vlm_period_s = 0.0

    # -------------------------------------------------------------------------
    # Request scheduling model (IMPORTANT)
    # -------------------------------------------------------------------------
    # There are TWO independent concepts:
    #  1) request cadence: every plan-tick ("max-rate") vs fixed-Hz ("stream-query")
    #  2) request pipelining: can we have multiple outstanding requests (in-flight)?
    #
    # We control (2) via `vlm_max_inflight`:
    # - `vlm_max_inflight > 0`: force that limit for ALL policies
    # - `vlm_max_inflight <= 0`: use principled defaults:
    #     - stream-query policies (*_stream*): pipelined (unlimited inflight)
    #       (vlm_stream, vlm_stream_match, score_fusion_stream, prob_fusion_stream)
    #     - non-streaming policies that use VLM: single inflight
    #       (vlm_hold, vlm_hold_match, score_fusion, prob_fusion)
    #
    # This keeps semantics clear and avoids ad-hoc per-policy hacks.
    _INF_UNLIMITED = 10**9

    def _is_stream_query_policy(pol: str) -> bool:
        is_stream = str(pol) in (
            "vlm_stream",
            "vlm_stream_match",
            "score_fusion_stream",
            "prob_fusion_stream",
        )
        if not is_stream and "stream" in pol:
            raise ValueError(f"Policy {pol} is not a stream-query policy")
        return is_stream

    def _resolve_inflight_limit(pol: str, max_inflight: int) -> int:
        if int(max_inflight) > 0:
            return int(max_inflight)
        # Defaults: stream-query => pipelined, else => single inflight
        return _INF_UNLIMITED if _is_stream_query_policy(pol) else 1

    inflight_limit = _resolve_inflight_limit(str(policy), int(vlm_max_inflight))

    # Current plan to track (world polyline)
    ref_world: np.ndarray | None = None
    chosen_hist: list[int] = []
    last_selected_plan_world: np.ndarray | None = None
    last_vlm_plan_world: np.ndarray | None = None

    # Optional (for visualization/debug): store per-plan-tick histories.
    plan_t_hist: list[float] | None = [] if bool(return_trace) else None
    plan_chosen_hist: list[int] | None = [] if bool(return_trace) else None
    plan_argmax_hist: list[int] | None = [] if bool(return_trace) else None
    plan_oracle_hist: list[int] | None = [] if bool(return_trace) else None
    plan_vlm_like_hist: list[int] | None = [] if bool(return_trace) else None
    plan_stale_s_hist: list[float] | None = [] if bool(return_trace) else None
    plan_weight_hist: list[float] | None = [] if bool(return_trace) else None
    plan_alpha_hist: list[float] | None = [] if bool(return_trace) else None
    plan_scores_hist: list[np.ndarray] | None = [] if bool(return_trace) else None
    plan_corrupted_scores_hist: list[np.ndarray] | None = [] if bool(return_trace) else None
    plan_sims_hist: list[np.ndarray] | None = [] if bool(return_trace) else None
    plan_selected_world_hist: list[np.ndarray] | None = [] if bool(return_trace) else None
    plan_vlm_aligned_world_hist: list[np.ndarray] | None = [] if bool(return_trace) else None

    # traces
    xs = np.zeros((steps + 1,), dtype=np.float64)
    ys = np.zeros((steps + 1,), dtype=np.float64)
    yaws = np.zeros((steps + 1,), dtype=np.float64)
    vs = np.zeros((steps + 1,), dtype=np.float64)
    ws = np.zeros((steps + 1,), dtype=np.float64)
    t_s = np.linspace(0.0, float(steps) * dt, int(steps) + 1, dtype=np.float64)
    ctes: list[float] = []
    speeds: list[float] = []
    stopped = 0

    # --- behavior metrics (plan-tick based) ---
    n_cand = int(len(candidates_body2))
    hist_argmax = np.zeros((n_cand,), dtype=np.int64)
    hist_chosen = np.zeros((n_cand,), dtype=np.int64)
    vlm_requests = 0
    oracle_eq_argmax = 0
    vlm_like_eq_argmax = 0
    chosen_eq_argmax = 0
    chosen_eq_vlm_like = 0
    chosen_eq_vlm_idx = 0
    switch_from_argmax = 0
    vlm_available = 0
    vlm_like_diff_argmax = 0
    switch_from_argmax_given_vlm_diff = 0
    switch_to_vlm_like_given_vlm_diff = 0

    xs[0] = x
    ys[0] = y
    yaws[0] = yaw
    vs[0] = v
    ws[0] = w

    executed_steps = 0
    for k in range(steps):
        t = float(k) * dt

        # deliver VLM responses
        if pending:
            still: list[tuple[float, float, int, np.ndarray]] = []
            for deliver_t, req_t, idx, p in pending:
                if float(deliver_t) <= t + 1e-9:
                    # Out-of-order safe: only accept if this request is newer than our current VLM
                    # ref.
                    if (
                        last_vlm_request_t is None
                        or float(req_t) > float(last_vlm_request_t) + 1e-9
                    ):
                        last_vlm_ref_world = np.asarray(p, dtype=np.float64)
                        last_vlm_idx = int(idx)
                        last_vlm_request_t = float(req_t)
                else:
                    still.append((float(deliver_t), float(req_t), int(idx), p))
            pending = still

        if (k % plan_every == 0) or (ref_world is None):
            # Candidates in world (frozen at planning pose)
            cand_world = [body_to_world(c, x=x, y=y, yaw=yaw) for c in candidates_body2]

            # Reference segment: align to current pose by projection on ref polyline
            s0, _d0 = project_point_to_polyline_arclength(
                np.asarray([x, y], dtype=np.float64), ref_xy
            )
            ref_seg = _sample_polyline_by_arclength_window(
                ref_xy, s_start=float(s0), s_len=float(cand_len), n=int(n0)
            )

            # Planner scores (noisy oracle)
            scores = _scores_to_ref_segment(
                candidates_world=cand_world, ref_seg_world=ref_seg, score_model=score_model, rng=rng
            )
            idx_argmax = int(np.argmax(scores))
            # Oracle index (noiseless argmin distance to ref segment)
            idx_oracle = _choose_oracle_idx(candidates_world=cand_world, ref_seg_world=ref_seg)

            # Keep a consistent "planner view" for trace/debug (even when a policy doesn't use it
            # directly).
            corrupted_scores = _corrupt_scores_with_epsilon(scores, float(score_model.epsilon), rng)

            # Enqueue a VLM request (oracle).
            #
            # IMPORTANT: streaming request scheduler
            # - Non-stream policies: request at every plan tick ("max-rate"), subject to
            # inflight_limit.
            # - Stream-query policies (vlm_stream, vlm_stream_match):
            #   request at a fixed cadence set by --vlm-query-interval-s / --vlm-query-hz,
            #   independent of VLM latency (subject to inflight_limit).
            # Policies that do not use VLM should not generate VLM traffic.
            do_vlm_request = str(policy) not in ("local_only", "planner_oracle")
            is_stream_query_policy = _is_stream_query_policy(str(policy))
            if is_stream_query_policy and vlm_period_s > 0:
                # Strict cadence: only submit when due (do NOT "pull forward" just because no
                # request is pending).
                do_vlm_request = bool(t + 1e-9 >= float(next_vlm_request_t))
            # Limit outstanding (in-flight) requests (no-streaming baseline if limit==1).
            # For delay_s ~ 0 there is effectively no in-flight queue.
            if inflight_limit < _INF_UNLIMITED and float(delay_s) > 1e-9:
                # pending contains not-yet-delivered requests
                if int(len(pending)) >= int(inflight_limit):
                    do_vlm_request = False
            if do_vlm_request:
                vlm_requests += 1
                oracle_eq_argmax += int(idx_oracle == idx_argmax)
                req_t = float(t)
                # Optionally corrupt the VLM output (epsilon-greedy): sometimes return a random
                # candidate.
                idx_vlm = int(idx_oracle)
                try:
                    p_m = float(vlm_mistake_prob)
                except Exception:
                    p_m = 0.0
                p_m = float(np.clip(p_m, 0.0, 1.0))
                if p_m > 0.0 and rng.random() < p_m:
                    idx_vlm = int(rng.integers(0, int(n_cand)))
                if float(delay_s) <= 1e-9:
                    last_vlm_ref_world = np.asarray(cand_world[int(idx_vlm)], dtype=np.float64)
                    last_vlm_idx = int(idx_vlm)
                    last_vlm_request_t = float(req_t)
                else:
                    pending.append(
                        (
                            float(t + float(delay_s)),
                            float(req_t),
                            int(idx_vlm),
                            np.asarray(cand_world[int(idx_vlm)], dtype=np.float64),
                        )
                    )
                if is_stream_query_policy and vlm_period_s > 0:
                    # Advance schedule by whole periods to avoid drift.
                    if float(next_vlm_request_t) <= 0.0:
                        next_vlm_request_t = float(t + float(vlm_period_s))
                    while float(next_vlm_request_t) <= float(t) + 1e-9:
                        next_vlm_request_t = float(next_vlm_request_t) + float(vlm_period_s)

            # If a delayed VLM is available, compute which candidate best matches it at this tick.
            idx_vlm_like: int | None = None
            sims: np.ndarray | None = None
            if last_vlm_ref_world is not None:
                vlm_available += 1
                sims = np.asarray(
                    [
                        _similarity(
                            candidate_world=np.asarray(cw, dtype=np.float64),
                            stale_ref_world=np.asarray(last_vlm_ref_world, dtype=np.float64),
                            robot_pose_world=(x, y, yaw),
                            cfg=fusion_cfg,
                        )
                        for cw in cand_world
                    ],
                    dtype=np.float64,
                )
                idx_vlm_like = int(np.argmax(sims))
                vlm_like_eq_argmax += int(idx_vlm_like == idx_argmax)
                vlm_like_diff_argmax += int(idx_vlm_like != idx_argmax)

            # selection policy
            if policy == "planner_oracle":
                # Immediate oracle: always take the best-matching candidate to the current
                # reference segment.
                chosen = int(idx_oracle)
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy in ("vlm_hold", "vlm_stream"):
                # DIRECT-APPLY: execute the delayed VLM-selected trajectory (in world frame),
                # but only its remaining segment from the current pose to avoid "targets behind
                # robot".
                if last_vlm_ref_world is not None:
                    stale_ref = np.asarray(last_vlm_ref_world, dtype=np.float64)
                    if bool(fusion_cfg.align_to_current_pose):
                        s_vlm, _d_vlm = project_point_to_polyline_arclength(
                            np.asarray([x, y], dtype=np.float64), stale_ref
                        )
                        ref_world = _sample_polyline_by_arclength_window(
                            stale_ref, s_start=float(s_vlm), s_len=float(cand_len), n=int(n0)
                        )
                    else:
                        ref_world = _resample_polyline_index(stale_ref, int(n0))
                    # Bookkeeping only: "chosen" index for behavior metrics/histograms.
                    if last_vlm_idx is not None:
                        chosen = int(last_vlm_idx)
                    elif idx_vlm_like is not None:
                        chosen = int(idx_vlm_like)
                    else:
                        chosen = int(idx_argmax)
                else:
                    # Fallback when VLM not available: behave like local planner.
                    chosen = int(np.argmax(corrupted_scores))
                    ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy in ("vlm_hold_match", "vlm_stream_match"):
                # MATCH-CANDIDATE: choose the *current* candidate that best matches the stale VLM
                # trajectory.
                if idx_vlm_like is not None:
                    chosen = int(idx_vlm_like)
                else:
                    chosen = int(np.argmax(corrupted_scores))
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy in ("score_fusion", "score_fusion_stream"):
                # Corrupt planner scores with epsilon probability
                # This corruption propagates through fusion, giving VLM a chance to correct
                # Apply fusion if lambda > 0 and VLM is available
                if (
                    fusion_cfg.enabled
                    and float(fusion_cfg.lambda_sim) > 0
                    and last_vlm_ref_world is not None
                    and last_vlm_request_t is not None
                    and sims is not None
                ):
                    stale = float(max(0.0, t - float(last_vlm_request_t)))
                    tau = float(max(1e-6, float(fusion_cfg.staleness_tau_s)))
                    weight = math.exp(-stale / tau)
                    # Fuse CORRUPTED scores with VLM similarity
                    fused = corrupted_scores + float(fusion_cfg.lambda_sim) * float(weight) * sims
                    chosen = int(np.argmax(fused))
                else:
                    # No fusion: just use corrupted scores
                    chosen = int(np.argmax(corrupted_scores))
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            elif policy in ("prob_fusion", "prob_fusion_stream"):
                # Probability-space fusion:
                # - planner -> p_planner = softmax(corrupted_scores)
                # - VLM -> p_vlm = softmax(similarity / vlm_temp)
                # - fuse -> p = (1-alpha)*p_planner + alpha*p_vlm
                # where alpha depends on staleness:
                #   alpha = lambda / (lambda + 1) * exp(-stale/tau)
                #
                # Key fixes vs original:
                # 1. VLM softmax temperature to reduce peakiness (similarity is in meters, so
                # temp=1 is reasonable)
                # 2. Alpha formula changed to lambda/(lambda+1) for more gradual blending
                #    - lambda=1: alpha_base=0.5, lambda=3: alpha_base=0.75,
                #      lambda=10: alpha_base=0.91
                p_planner = _softmax(corrupted_scores)

                if (
                    fusion_cfg.enabled
                    and float(fusion_cfg.lambda_sim) > 0
                    and last_vlm_ref_world is not None
                    and last_vlm_request_t is not None
                    and sims is not None
                ):
                    stale = float(max(0.0, t - float(last_vlm_request_t)))
                    tau = float(max(1e-6, float(fusion_cfg.staleness_tau_s)))
                    decay = math.exp(-stale / tau)

                    # Apply temperature to VLM similarities to reduce peakiness
                    # Similarity is negative distance in meters, so temp=1.0 is reasonable
                    vlm_temp = 1.0
                    p_vlm = _softmax(sims / vlm_temp)

                    # More gradual alpha: lambda/(lambda+1) instead of 1-exp(-lambda)
                    # This gives: lambda=0.5->0.33, lambda=1->0.5, lambda=3->0.75, lambda=10->0.91
                    lam = float(max(0.0, float(fusion_cfg.lambda_sim)))
                    alpha_base = lam / (lam + 1.0)
                    alpha = float(alpha_base * decay)
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    fused_p = (1.0 - alpha) * p_planner + alpha * p_vlm
                    chosen = int(np.argmax(fused_p))
                else:
                    alpha = float("nan")
                    chosen = int(np.argmax(p_planner))

                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)
            else:
                # local_only: corrupt scores with epsilon, then argmax
                chosen = int(np.argmax(corrupted_scores))
                ref_world = np.asarray(cand_world[chosen], dtype=np.float64)

            chosen_hist.append(int(chosen))
            last_selected_plan_world = np.asarray(ref_world, dtype=np.float64)
            last_vlm_plan_world = (
                np.asarray(last_vlm_ref_world, dtype=np.float64)
                if last_vlm_ref_world is not None
                else None
            )

            # Trace (plan-tick): capture what we knew/selected at this planning time.
            if plan_t_hist is not None:
                # VLM staleness/weight are meaningful only when we actually have a delivered ref.
                staleness = float("nan")
                weight = float("nan")
                if last_vlm_request_t is not None:
                    staleness = float(max(0.0, t - float(last_vlm_request_t)))
                    tau = float(max(1e-6, float(fusion_cfg.staleness_tau_s)))
                    weight = float(math.exp(-staleness / tau))

                # For visualization, also store an aligned VLM segment (remaining window) when
                # available.
                vlm_aligned = np.full((int(n0), 2), np.nan, dtype=np.float64)
                if (
                    last_vlm_ref_world is not None
                    and last_vlm_ref_world.ndim == 2
                    and last_vlm_ref_world.shape[1] == 2
                ):
                    try:
                        stale_ref = np.asarray(last_vlm_ref_world, dtype=np.float64)
                        if bool(fusion_cfg.align_to_current_pose):
                            s_vlm, _d_vlm = project_point_to_polyline_arclength(
                                np.asarray([x, y], dtype=np.float64), stale_ref
                            )
                            vlm_aligned = _sample_polyline_by_arclength_window(
                                stale_ref, s_start=float(s_vlm), s_len=float(cand_len), n=int(n0)
                            )
                        else:
                            vlm_aligned = _resample_polyline_index(stale_ref, int(n0))
                    except Exception:
                        pass

                plan_t_hist.append(float(t))
                plan_chosen_hist.append(int(chosen) if chosen is not None else -1)
                plan_argmax_hist.append(int(idx_argmax))
                plan_oracle_hist.append(int(idx_oracle))
                plan_vlm_like_hist.append(int(idx_vlm_like) if idx_vlm_like is not None else -1)
                plan_stale_s_hist.append(float(staleness))
                plan_weight_hist.append(float(weight))
                plan_alpha_hist.append(float(locals().get("alpha", float("nan"))))
                plan_scores_hist.append(np.asarray(scores, dtype=np.float64).copy())
                plan_corrupted_scores_hist.append(
                    np.asarray(corrupted_scores, dtype=np.float64).copy()
                )
                if sims is None:
                    plan_sims_hist.append(np.full((int(n_cand),), np.nan, dtype=np.float64))
                else:
                    plan_sims_hist.append(np.asarray(sims, dtype=np.float64).copy())
                plan_selected_world_hist.append(np.asarray(ref_world, dtype=np.float64).copy())
                plan_vlm_aligned_world_hist.append(np.asarray(vlm_aligned, dtype=np.float64).copy())

            # Update per-plan-tick behavior metrics.
            chosen_eq_argmax += int(chosen == idx_argmax)
            switch_from_argmax += int(chosen != idx_argmax)
            hist_argmax[int(idx_argmax)] += 1
            hist_chosen[int(chosen)] += 1
            if idx_vlm_like is not None:
                chosen_eq_vlm_like += int(chosen == int(idx_vlm_like))
                if last_vlm_idx is not None:
                    chosen_eq_vlm_idx += int(chosen == int(last_vlm_idx))
                if int(idx_vlm_like) != int(idx_argmax):
                    switch_from_argmax_given_vlm_diff += int(chosen != int(idx_argmax))
                    switch_to_vlm_like_given_vlm_diff += int(chosen == int(idx_vlm_like))

        # controller step
        path_body = world_to_body(np.asarray(ref_world, dtype=np.float64), x=x, y=y, yaw=yaw)
        if str(controller) == "pure_pursuit":
            v_cmd, w_cmd = controller_pure_pursuit(
                path_body,
                v_prev=float(v),
                w_prev=float(w),
                limits=sim_cfg.limits,
                cfg=sim_cfg.pp_cfg,
            )
            # Apply acceleration/rate limits per control tick (match toy sim behavior).
            dv = float(sim_cfg.limits.max_a) * dt
            dw = float(sim_cfg.limits.max_alpha) * dt
            v_cmd = float(np.clip(float(v_cmd), float(v) - dv, float(v) + dv))
            w_cmd = float(np.clip(float(w_cmd), float(w) - dw, float(w) + dw))
        else:
            v_cmd, w_cmd = controller_endpoint_pd(
                path_body,
                v_prev=float(v),
                w_prev=float(w),
                limits=sim_cfg.limits,
                cfg=sim_cfg.ep_cfg,
            )
        v = float(np.clip(v_cmd, 0.0, float(sim_cfg.limits.max_v)))
        w = float(np.clip(w_cmd, -float(sim_cfg.limits.max_w), float(sim_cfg.limits.max_w)))

        if v < 1e-3:
            stopped += 1
        speeds.append(float(v))

        x, y, yaw = integrate_diff_drive(x, y, yaw, v, w, dt)
        xs[k + 1] = float(x)
        ys[k + 1] = float(y)
        yaws[k + 1] = float(yaw)
        vs[k + 1] = float(v)
        ws[k + 1] = float(w)

        # Progress + CTE to whole reference polyline
        _s, d = project_point_to_polyline_arclength(np.asarray([x, y], dtype=np.float64), ref_xy)
        try:
            s_last = float(np.clip(float(_s), 0.0, ref_len_safe))
            s_max = float(max(float(s_max), float(s_last)))
        except Exception:
            pass
        ctes.append(float(d))

        executed_steps = int(k + 1)

        # Early stopping: once we "finish" (by success definition), stop the rollout.
        if bool(stop_on_success):
            try:
                completion_now = float(s_last / ref_len_safe) if ref_len > 1e-9 else float("nan")
                goal_dist_now = float(
                    np.linalg.norm(np.asarray([x, y], dtype=np.float64) - goal_xy)
                )
                if (
                    math.isfinite(completion_now)
                    and completion_now >= float(min_completion)
                    and goal_dist_now <= float(goal_r)
                ):
                    break
            except Exception:
                pass

    def _p(x: list[float], q: float) -> float:
        if not x:
            return float("nan")
        return float(np.quantile(np.asarray(x, dtype=np.float64), q))

    # Truncate traces to executed horizon (important when early stopping).
    if int(executed_steps) > 0 and int(executed_steps) < int(steps):
        xs = xs[: int(executed_steps) + 1]
        ys = ys[: int(executed_steps) + 1]
        yaws = yaws[: int(executed_steps) + 1]
        vs = vs[: int(executed_steps) + 1]
        ws = ws[: int(executed_steps) + 1]
        t_s = t_s[: int(executed_steps) + 1]

    chosen = np.asarray(chosen_hist, dtype=np.int64)
    switches = int(np.count_nonzero(np.diff(chosen))) if chosen.size >= 2 else 0

    plan_ticks = int(chosen.size)

    def _ratio(num: int, den: int) -> float:
        if den <= 0:
            return float("nan")
        return float(num) / float(den)

    tv = float("nan")
    if plan_ticks > 0:
        p = hist_chosen.astype(np.float64) / float(plan_ticks)
        q = hist_argmax.astype(np.float64) / float(plan_ticks)
        tv = float(0.5 * float(np.sum(np.abs(p - q))))

    # Route completion + success (endpoint-based)
    route_completion_final = float(s_last / ref_len_safe) if ref_len > 1e-9 else float("nan")
    route_completion_max = float(s_max / ref_len_safe) if ref_len > 1e-9 else float("nan")
    goal_dists: list[float] = []
    # We evaluate goal distance over the *entire rollout*; success should be based on reaching the
    # goal at any time,
    # not necessarily ending close to the goal (the robot can pass the goal and continue).
    for xi, yi in zip(xs.tolist(), ys.tolist(), strict=False):
        goal_dists.append(float(np.linalg.norm(np.asarray([xi, yi], dtype=np.float64) - goal_xy)))
    goal_dist_final_m = (
        float(goal_dists[-1])
        if goal_dists
        else float(np.linalg.norm(np.asarray([x, y], dtype=np.float64) - goal_xy))
    )
    goal_dist_min_m = (
        float(np.min(np.asarray(goal_dists, dtype=np.float64)))
        if goal_dists
        else float(goal_dist_final_m)
    )
    success = (
        int(bool(route_completion_max >= min_completion and goal_dist_min_m <= goal_r))
        if math.isfinite(goal_dist_min_m)
        else 0
    )

    task_id = f"{task.episode_id}@{task.start_t:.2f}"
    res = RunResult(
        task_id=task_id,
        episode_id=str(task.episode_id),
        start_t=float(task.start_t),
        controller=str(controller),
        policy=str(policy),
        delay_s=float(delay_s),
        seed=int(seed),
        similarity_mode=str(fusion_cfg.similarity_mode),
        lambda_sim=float(fusion_cfg.lambda_sim),
        staleness_tau_s=float(fusion_cfg.staleness_tau_s),
        dist_scale_m=float(fusion_cfg.dist_scale_m),
        align_to_current_pose=bool(fusion_cfg.align_to_current_pose),
        noise_std=float(score_model.noise_std),
        epsilon=float(score_model.epsilon),
        temperature=float(score_model.temperature),
        mean_cte_m=float(np.mean(ctes)) if ctes else float("nan"),
        p95_cte_m=_p(ctes, 0.95),
        mean_speed_mps=float(np.mean(speeds)) if speeds else float("nan"),
        stopped_frac=float(stopped) / float(max(1, len(speeds))),
        chosen_switches=int(switches),
        route_completion_max=float(route_completion_max),
        route_completion_final=float(route_completion_final),
        goal_dist_final_m=float(goal_dist_final_m),
        goal_dist_min_m=float(goal_dist_min_m),
        success=int(success),
        ref_len_m=float(ref_len),
        ref_duration_s=float(ref_duration_s),
        ref_mean_speed_mps=float(ref_mean_speed_mps),
        sim_horizon_s=float(sim_horizon_s),
        completion_upper_bound=float(completion_upper_bound),
        plan_ticks=int(plan_ticks),
        vlm_requests=int(vlm_requests),
        vlm_request_frac=_ratio(int(vlm_requests), int(plan_ticks)),
        vlm_available_frac=_ratio(int(vlm_available), int(plan_ticks)),
        oracle_eq_argmax_ratio=_ratio(int(oracle_eq_argmax), int(vlm_requests)),
        vlm_like_eq_argmax_ratio=_ratio(int(vlm_like_eq_argmax), int(vlm_available)),
        chosen_eq_argmax_ratio=_ratio(int(chosen_eq_argmax), int(plan_ticks)),
        chosen_eq_vlm_like_ratio=_ratio(int(chosen_eq_vlm_like), int(vlm_available)),
        chosen_eq_vlm_idx_ratio=_ratio(int(chosen_eq_vlm_idx), int(vlm_available)),
        switch_from_argmax_ratio=_ratio(int(switch_from_argmax), int(plan_ticks)),
        vlm_like_diff_argmax_frac=_ratio(int(vlm_like_diff_argmax), int(vlm_available)),
        switch_from_argmax_given_vlm_diff_ratio=_ratio(
            int(switch_from_argmax_given_vlm_diff), int(vlm_like_diff_argmax)
        ),
        switch_to_vlm_like_given_vlm_diff_ratio=_ratio(
            int(switch_to_vlm_like_given_vlm_diff), int(vlm_like_diff_argmax)
        ),
        tv_dist_chosen_vs_argmax=float(tv),
        vlm_mistake_prob=float(vlm_mistake_prob),
    )

    trace: dict[str, np.ndarray] = {
        "t_s": t_s,
        "x": xs,
        "y": ys,
        "yaw": yaws,
        "v": vs,
        "w": ws,
        "ref_xy": ref_xy,
        "last_selected_plan_world": last_selected_plan_world
        if last_selected_plan_world is not None
        else np.zeros((0, 2), dtype=np.float64),
        "last_vlm_plan_world": last_vlm_plan_world
        if last_vlm_plan_world is not None
        else np.zeros((0, 2), dtype=np.float64),
    }
    if bool(return_trace) and plan_t_hist is not None and plan_selected_world_hist is not None:
        try:
            trace.update(
                {
                    "plan_t_s": np.asarray(plan_t_hist, dtype=np.float64),
                    "plan_idx_chosen": np.asarray(plan_chosen_hist, dtype=np.int64),
                    "plan_idx_argmax": np.asarray(plan_argmax_hist, dtype=np.int64),
                    "plan_idx_oracle": np.asarray(plan_oracle_hist, dtype=np.int64),
                    "plan_idx_vlm_like": np.asarray(plan_vlm_like_hist, dtype=np.int64),
                    "plan_stale_s": np.asarray(plan_stale_s_hist, dtype=np.float64),
                    "plan_weight": np.asarray(plan_weight_hist, dtype=np.float64),
                    "plan_alpha": np.asarray(plan_alpha_hist, dtype=np.float64),
                    "plan_scores": np.stack(plan_scores_hist, axis=0).astype(np.float64)
                    if plan_scores_hist
                    else np.zeros((0, int(n_cand)), dtype=np.float64),
                    "plan_corrupted_scores": np.stack(plan_corrupted_scores_hist, axis=0).astype(
                        np.float64
                    )
                    if plan_corrupted_scores_hist
                    else np.zeros((0, int(n_cand)), dtype=np.float64),
                    "plan_sims": np.stack(plan_sims_hist, axis=0).astype(np.float64)
                    if plan_sims_hist
                    else np.zeros((0, int(n_cand)), dtype=np.float64),
                    "plan_selected_world": np.stack(plan_selected_world_hist, axis=0).astype(
                        np.float64
                    )
                    if plan_selected_world_hist
                    else np.zeros((0, int(n0), 2), dtype=np.float64),
                    "plan_vlm_aligned_world": np.stack(plan_vlm_aligned_world_hist, axis=0).astype(
                        np.float64
                    )
                    if plan_vlm_aligned_world_hist
                    else np.zeros((0, int(n0), 2), dtype=np.float64),
                    "dt_control": np.asarray([float(dt)], dtype=np.float64),
                    "dt_plan": np.asarray([float(sim_cfg.dt_plan)], dtype=np.float64),
                    "plan_every": np.asarray([int(plan_every)], dtype=np.int64),
                }
            )
        except Exception:
            pass
    return res, trace


def build_tasks_from_dataset(
    *,
    dataset_root: Path,
    duration_s: float,
    min_episode_s: float,
    stride_s: float,
    max_episodes: int,
    max_tasks_per_episode: int,
    task_sampling: str = "first",
    seed: int = 0,
) -> list[TaskWindow]:
    meta = find_episode_metadata_files(dataset_root)
    meta = meta[: int(max(0, max_episodes))]
    tasks: list[TaskWindow] = []

    tqdm = _try_import_tqdm()
    it = (
        tqdm(meta, desc="build_tasks:episodes", unit="ep", disable=(not sys.stderr.isatty()))
        if tqdm is not None
        else meta
    )

    # Stats for user-visible progress.
    n_schema_fail = 0
    n_no_odom = 0
    n_too_short = 0
    n_ok = 0

    for ep_meta in it:
        ep_id, ts, xs, ys, yaws = _load_odom_only(dataset_root, ep_meta)
        if ep_id is None or ts is None:
            n_schema_fail += 1
            continue
        if ts.size < 3:
            n_no_odom += 1
            continue
        if ts.size < 3:
            continue
        t0 = float(ts[0])
        t1 = float(ts[-1])
        if (t1 - t0) < float(min_episode_s):
            n_too_short += 1
            continue
        n_ok += 1

        # Candidate window start times on a fixed stride grid.
        starts_all: list[float] = []
        t = float(t0)
        while t + float(duration_s) <= t1 - 1e-6:
            starts_all.append(float(t))
            t += float(stride_s)

        # Choose which windows to use for this episode.
        # Note: windows are only guaranteed non-overlapping if stride_s >= duration_s.
        if int(max_tasks_per_episode) <= 0 or len(starts_all) <= int(max_tasks_per_episode):
            starts = starts_all
        else:
            if str(task_sampling) == "uniform_per_episode":
                # Stable per-episode RNG so postprocess can reproduce the same windows.
                h = hashlib.md5(str(ep_id).encode("utf-8")).digest()
                ep_seed = int.from_bytes(h[:8], "little", signed=False)
                rng_ep = random.Random(int(seed) + int(ep_seed))
                starts = rng_ep.sample(starts_all, k=int(max_tasks_per_episode))
                starts.sort()
            else:
                # Default: take the earliest windows.
                starts = starts_all[: int(max_tasks_per_episode)]

        for st in starts:
            et = float(st + float(duration_s))
            # slice odom for this window
            i0 = int(np.searchsorted(ts, st, side="left"))
            i1 = int(np.searchsorted(ts, et, side="right"))
            if i1 - i0 < 3:
                continue
            ref_xy = np.stack([xs[i0:i1], ys[i0:i1]], axis=1).astype(np.float64)
            init_yaw = float(yaws[i0])
            tasks.append(
                TaskWindow(
                    episode_id=str(ep_id),
                    start_t=float(ts[i0]),
                    end_t=float(ts[i1 - 1]),
                    ref_xy=ref_xy,
                    init_yaw=float(init_yaw),
                )
            )
        if tqdm is not None:
            try:
                it.set_postfix(  # type: ignore[attr-defined]
                    tasks=int(len(tasks)),
                    ok=int(n_ok),
                    schema_fail=int(n_schema_fail),
                    no_odom=int(n_no_odom),
                    too_short=int(n_too_short),
                )
            except Exception:
                pass
    return tasks


def _resolve_records_ref_path(
    records_ref: object, *, episode_dir: Path, dataset_root: Path
) -> Path | None:
    """Resolve a stream records_ref into a local file path (odom only, best-effort)."""
    if isinstance(records_ref, str):
        ref_path = Path(records_ref)
    elif isinstance(records_ref, dict) and isinstance(records_ref.get("path"), str):
        ref_path = Path(records_ref["path"])
    else:
        return None
    if ref_path.is_absolute():
        return ref_path
    cand1 = (episode_dir / ref_path).resolve()
    if cand1.exists():
        return cand1
    cand2 = (dataset_root / ref_path).resolve()
    if cand2.exists():
        return cand2
    return cand1


def _load_odom_only(
    dataset_root: Path, episode_meta_path: Path
) -> tuple[str | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Fast episode loader that only reads odom.jsonl.

    This avoids validating planner_candidates (which can be very slow) and is all we need for
    building real-trajectory reference windows.
    """
    dataset_root = dataset_root.resolve()
    episode_meta_path = episode_meta_path.resolve()
    episode_dir = episode_meta_path.parent

    try:
        ep_obj = json.loads(episode_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None, None, None, None
    if not isinstance(ep_obj, dict):
        return None, None, None, None, None
    ep_id = str(
        ep_obj.get("episode_id")
        or (
            episode_dir.name if episode_meta_path.name == "episode.json" else episode_meta_path.stem
        )
    )

    streams = ep_obj.get("streams", {})
    if not isinstance(streams, dict):
        return None, None, None, None, None
    odom_spec = streams.get("odom")
    if not isinstance(odom_spec, dict):
        return (
            ep_id,
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    # Prefer records_ref; fallback to inline records.
    raw_records = None
    if "records" in odom_spec and isinstance(odom_spec.get("records"), list):
        raw_records = odom_spec.get("records")
    else:
        ref_path = _resolve_records_ref_path(
            odom_spec.get("records_ref"), episode_dir=episode_dir, dataset_root=dataset_root
        )
        if ref_path is None or not ref_path.exists():
            return (
                ep_id,
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
        if ref_path.suffix != ".jsonl":
            return (
                ep_id,
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
        # Stream parse jsonl (fast) without pydantic.
        ts_l: list[float] = []
        xs_l: list[float] = []
        ys_l: list[float] = []
        yaws_l: list[float] = []
        try:
            with ref_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if not isinstance(r, dict):
                        continue
                    # Required keys: t,x,y,yaw
                    try:
                        ts_l.append(float(r["t"]))
                        xs_l.append(float(r["x"]))
                        ys_l.append(float(r["y"]))
                        yaws_l.append(float(r["yaw"]))
                    except Exception:
                        continue
        except Exception:
            return None, None, None, None, None
        ts = np.asarray(ts_l, dtype=np.float64)
        xs = np.asarray(xs_l, dtype=np.float64)
        ys = np.asarray(ys_l, dtype=np.float64)
        yaws = np.asarray(yaws_l, dtype=np.float64)
        if ts.size:
            order = np.argsort(ts)
            ts = ts[order]
            xs = xs[order]
            ys = ys[order]
            yaws = yaws[order]
        return ep_id, ts, xs, ys, yaws

    # Inline records path (rare). Parse similarly.
    ts_l = []
    xs_l = []
    ys_l = []
    yaws_l = []
    for r in raw_records or []:
        if not isinstance(r, dict):
            continue
        try:
            ts_l.append(float(r["t"]))
            xs_l.append(float(r["x"]))
            ys_l.append(float(r["y"]))
            yaws_l.append(float(r["yaw"]))
        except Exception:
            continue
    ts = np.asarray(ts_l, dtype=np.float64)
    xs = np.asarray(xs_l, dtype=np.float64)
    ys = np.asarray(ys_l, dtype=np.float64)
    yaws = np.asarray(yaws_l, dtype=np.float64)
    if ts.size:
        order = np.argsort(ts)
        ts = ts[order]
        xs = xs[order]
        ys = ys[order]
        yaws = yaws[order]
    return ep_id, ts, xs, ys, yaws


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Closed-loop score fusion benchmark on real dataset odom reference."
    )
    p.add_argument(
        "--dataset", type=str, default="", help="Canonical dataset root (contains episodes/)."
    )
    p.add_argument(
        "--static-candidates-json",
        type=str,
        default="assets/trajectory_selection_static_candidates/takeover_kmeans_medoids/static_candidates_k12.json",
        help="Static candidate library JSON (robot-frame XY polylines).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="logs/closed_loop_fusion",
        help="Output directory (run folder created inside).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--postprocess-run-dir",
        type=str,
        default="",
        help="If set, recompute/augment results for an existing run dir and rewrite summary/report "
        "in place.",
    )

    # Task building
    # For the hard split, we treat the extracted clip polyline as a *path to finish*; time is only
    # a simulation budget.
    # So we still slice a reference polyline out of odom using duration/stride, but by default the
    # benchmark
    # runs in arclength mode with an independent simulation time limit.
    p.add_argument("--task-mode", type=str, default="arclength", choices=["arclength", "time"])
    p.add_argument(
        "--duration-s",
        type=float,
        default=18.0,
        help="Reference polyline window length in seconds (used to extract ref_xy from odom).",
    )
    p.add_argument("--min-episode-s", type=float, default=18.0)
    p.add_argument("--stride-s", type=float, default=5.0)
    p.add_argument("--max-episodes", type=int, default=100)
    p.add_argument("--max-tasks-per-episode", type=int, default=1)
    p.add_argument(
        "--time-limit-s",
        type=float,
        default=40.0,
        help="Simulation time budget (seconds). Used when --task-mode=arclength. "
        "If --task-mode=time, the simulation uses the reference window duration instead.",
    )
    p.add_argument(
        "--task-sampling",
        type=str,
        default="uniform_per_episode",
        choices=["first", "uniform_per_episode"],
        help="How to choose task windows within an episode. "
        "'first' takes the earliest windows. "
        "'uniform_per_episode' uniformly samples start times from the stride grid (seeded).",
    )

    # Loop timing
    p.add_argument("--dt-control", type=float, default=0.1)
    p.add_argument("--dt-plan", type=float, default=0.2)  # 5Hz
    p.add_argument("--horizon-s", type=float, default=4.0)
    p.add_argument(
        "--controller", type=str, default="pure_pursuit", choices=["endpoint_pd", "pure_pursuit"]
    )
    p.add_argument(
        "--max-v",
        type=float,
        default=None,
        help="Override linear speed limit max_v (m/s). If unset, use the built-in default.",
    )
    p.add_argument(
        "--max-w",
        type=float,
        default=None,
        help="Override angular speed limit max_w (rad/s). If unset, use the built-in default.",
    )

    # Delay and policies
    p.add_argument("--delay-s", type=float, default=1.5)
    p.add_argument(
        "--policies",
        type=str,
        default="planner_oracle,local_only,vlm_hold,vlm_stream,vlm_hold_match,vlm_stream_match,score_fusion,prob_fusion,score_fusion_stream,prob_fusion_stream",
    )
    p.add_argument(
        "--vlm-query-hz",
        type=float,
        default=1.0,
        help="(Legacy) Used by stream-query policies (vlm_stream, vlm_stream_match, "
        "score_fusion_stream, prob_fusion_stream): VLM request cadence (Hz = queries/sec). "
        "Prefer --vlm-query-interval-s for clarity. "
        "Effective update rate becomes ~vlm_query_hz after warmup, but delayed by --delay-s.",
    )
    p.add_argument(
        "--vlm-query-interval-s",
        type=float,
        default=None,
        help="Preferred over --vlm-query-hz. VLM query interval in seconds (seconds/query). "
        "If set, overrides --vlm-query-hz. For stream-query policies, this directly controls "
        "cadence.",
    )
    p.add_argument(
        "--vlm-max-inflight",
        type=int,
        default=0,
        help="Max number of outstanding VLM requests allowed. "
        "If <=0, uses policy defaults (stream-query *_stream*=pipelined, "
        "non-streaming=single-inflight). "
        'Set to 1 to model "send next query only after the previous returns" (throughput ~= '
        "1/delay).",
    )
    p.add_argument(
        "--vlm-mistake-prob",
        type=float,
        default=0.0,
        help="With this probability, corrupt the VLM output by returning a random candidate "
        "(epsilon-greedy). "
        "0.0 = oracle VLM, 1.0 = fully random VLM.",
    )

    # Planner score model (sweepable)
    p.add_argument(
        "--noise-stds",
        type=str,
        default="1.0",
        help="Comma-separated noise std values for planner scores. Higher = noisier planner.",
    )
    p.add_argument(
        "--epsilons",
        type=str,
        default="0.3",
        help="Comma-separated epsilon values. Probability of planner score corruption.",
    )
    p.add_argument(
        "--score-temps",
        type=str,
        default="1.0",
        help="Comma-separated score temperature values. <1 = sharper (more confident), >1 = wider "
        "(less confident).",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Base seed for reproducibility (task sampling + noise)."
    )

    # Fusion sweep knobs (comma lists)
    p.add_argument(
        "--similarity-modes",
        type=str,
        default="body_pointwise_horizon_aware",
    )
    p.add_argument("--lambda-sims", type=str, default="1.0")
    p.add_argument("--taus", type=str, default="5.0")
    p.add_argument("--dist-scales", type=str, default="1.0")

    # Multiprocessing
    p.add_argument(
        "--num-workers", type=int, default=0, help="0=run in main process; otherwise process pool."
    )

    # Outcome metrics (success / completion)
    p.add_argument(
        "--success-min-completion",
        type=float,
        default=0.95,
        help="Success criterion (part 1): require max route completion >= this fraction (0..1).",
    )
    p.add_argument(
        "--success-goal-radius-m",
        type=float,
        default=3.0,
        help="Success criterion (part 2): require final distance to the reference endpoint <= this "
        "radius (meters).",
    )
    p.add_argument(
        "--stop-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, stop the rollout once the success criteria is met (prevents post-goal "
        "wandering from polluting metrics).",
    )

    # BEV examples
    p.add_argument(
        "--write-examples", action="store_true", help="Write a few BEV example PNGs (slow)."
    )
    p.add_argument("--examples-n", type=int, default=6)
    p.add_argument(
        "--examples-by-episode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When writing examples, pick the earliest (start_t) task window per episode (up to "
        "examples-n episodes).",
    )

    # Practical sweep controls (avoid accidental huge runs)
    p.add_argument("--max-tasks", type=int, default=0, help="If >0, cap number of task windows.")
    p.add_argument(
        "--max-configs", type=int, default=0, help="If >0, cap number of fusion configs."
    )
    p.add_argument(
        "--max-jobs", type=int, default=0, help="If >0, cap number of total (task,cfg) jobs."
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=200, help="Flush partial results every N jobs."
    )
    p.add_argument("--dry-run", action="store_true", help="Print counts and exit (no simulation).")

    return p.parse_args()


def _parse_csv_floats(s: str) -> list[float]:
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _parse_csv_strs(s: str) -> list[str]:
    out: list[str] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(tok)
    return out


def _bev_plot(path: Path, *, trace: dict[str, np.ndarray], title: str) -> None:
    plt = _try_import_matplotlib()
    if plt is None:
        return
    ref = np.asarray(trace["ref_xy"], dtype=np.float64)
    x = np.asarray(trace["x"], dtype=np.float64)
    y = np.asarray(trace["y"], dtype=np.float64)
    last_sel = np.asarray(trace.get("last_selected_plan_world", np.zeros((0, 2))), dtype=np.float64)
    last_vlm = np.asarray(trace.get("last_vlm_plan_world", np.zeros((0, 2))), dtype=np.float64)
    plan_t = np.asarray(trace.get("plan_t_s", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    sel_hist = np.asarray(
        trace.get("plan_selected_world", np.zeros((0, 0, 2), dtype=np.float64)), dtype=np.float64
    )
    vlm_hist = np.asarray(
        trace.get("plan_vlm_aligned_world", np.zeros((0, 0, 2), dtype=np.float64)), dtype=np.float64
    )
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    ax.plot(ref[:, 1], ref[:, 0], color="0.35", lw=3.0, label="ref")
    ax.plot(y, x, color="#1f77b4", lw=2.6, label="executed")

    # Optional: overlay plan snapshots along the rollout (helps debug divergence).
    # Only shown when trace contains per-plan-tick history (written in --write-examples mode).
    try:
        if (
            plan_t.ndim == 1
            and plan_t.size >= 2
            and sel_hist.ndim == 3
            and sel_hist.shape[0] == plan_t.size
        ):
            overlay_every_s = 2.0
            max_overlays = 10
            idxs: list[int] = []
            next_t = float(plan_t[0])
            for i in range(int(plan_t.size)):
                if float(plan_t[i]) + 1e-9 >= float(next_t):
                    idxs.append(int(i))
                    next_t += float(overlay_every_s)
            if len(idxs) > max_overlays:
                # Evenly downsample snapshots to avoid clutter.
                keep = np.linspace(0, len(idxs) - 1, max_overlays).round().astype(int).tolist()
                idxs = [idxs[j] for j in keep]

            for j, i in enumerate(idxs):
                a = 0.18 + 0.42 * (float(j) / max(1.0, float(len(idxs) - 1)))
                sel = np.asarray(sel_hist[int(i)], dtype=np.float64)
                if sel.ndim == 2 and sel.shape[0] >= 2:
                    ax.plot(
                        sel[:, 1],
                        sel[:, 0],
                        color="#ff7f0e",
                        lw=1.2,
                        ls=":",
                        alpha=float(a),
                        label="_nolegend_",
                    )
                if vlm_hist.ndim == 3 and vlm_hist.shape[0] == sel_hist.shape[0]:
                    vv = np.asarray(vlm_hist[int(i)], dtype=np.float64)
                    if vv.ndim == 2 and vv.shape[0] >= 2 and np.any(np.isfinite(vv)):
                        ax.plot(
                            vv[:, 1],
                            vv[:, 0],
                            color="#2ca02c",
                            lw=1.1,
                            ls="--",
                            alpha=float(a * 0.9),
                            label=("VLM plan snapshots (aligned)" if j == 0 else "_nolegend_"),
                        )
    except Exception:
        pass

    if last_sel.ndim == 2 and last_sel.shape[0] >= 2:
        ax.plot(
            last_sel[:, 1],
            last_sel[:, 0],
            color="#ff7f0e",
            lw=2.2,
            ls=":",
            alpha=0.9,
            label="last selected plan",
        )
    if last_vlm.ndim == 2 and last_vlm.shape[0] >= 2:
        ax.plot(
            last_vlm[:, 1],
            last_vlm[:, 0],
            color="#2ca02c",
            lw=2.0,
            ls="--",
            alpha=0.75,
            label="last VLM plan (stale)",
        )
    ax.scatter(
        [y[0]],
        [x[0]],
        s=90,
        facecolors="none",
        edgecolors="k",
        linewidths=2.5,
        zorder=10,
        label="start (executed)",
    )
    ax.scatter(
        [y[-1]],
        [x[-1]],
        s=70,
        marker="x",
        color="#1f77b4",
        linewidths=2.2,
        zorder=10,
        label="end (executed)",
    )
    # Mark the reference endpoint (goal)
    if ref.ndim == 2 and ref.shape[0] >= 1:
        gy, gx = float(ref[-1, 1]), float(ref[-1, 0])
        ax.scatter([gy], [gx], s=90, marker="*", color="#d62728", zorder=11, label="goal (ref end)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("y_left (m)")
    ax.set_ylabel("x_forward (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Always write a PNG (for quick viewing) and a PDF (for paper-quality figures).
    out_png = path if str(path).lower().endswith(".png") else path.with_suffix(".png")
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(str(out_png), dpi=160)
    fig.savefig(str(out_pdf))
    plt.close(fig)


# -----------------------------------------------------------------------------
# Multiprocessing support: use an initializer to avoid pickling nested functions.
# -----------------------------------------------------------------------------

_G_TASKS: list[TaskWindow] | None = None
_G_CONFIGS: list[tuple[FusionConfig, PlannerScoreModel, PolicyName]] | None = None
_G_CANDIDATES: list[np.ndarray] | None = None
_G_SIM_CFG: SimLoopConfig | None = None
_G_DELAY_S: float | None = None
_G_VLM_QUERY_HZ: float | None = None
_G_VLM_MAX_INFLIGHT: int | None = None
_G_VLM_MISTAKE_PROB: float | None = None
_G_BASE_SEED: int | None = None
_G_CONTROLLER: str | None = None
_G_SUCCESS_MIN_COMPLETION: float | None = None
_G_SUCCESS_GOAL_RADIUS_M: float | None = None
_G_STOP_ON_SUCCESS: bool | None = None
_G_TASK_MODE: str | None = None
_G_TIME_LIMIT_S: float | None = None


def _init_worker(
    tasks: list[TaskWindow],
    configs: list[tuple[FusionConfig, PlannerScoreModel, PolicyName]],
    candidates: list[np.ndarray],
    sim_cfg: SimLoopConfig,
    delay_s: float,
    vlm_query_hz: float,
    vlm_max_inflight: int,
    vlm_mistake_prob: float,
    base_seed: int,
    controller: str,
    success_min_completion: float,
    success_goal_radius_m: float,
    stop_on_success: bool,
    task_mode: str,
    time_limit_s: float,
) -> None:
    global \
        _G_TASKS, \
        _G_CONFIGS, \
        _G_CANDIDATES, \
        _G_SIM_CFG, \
        _G_DELAY_S, \
        _G_VLM_QUERY_HZ, \
        _G_VLM_MAX_INFLIGHT, \
        _G_VLM_MISTAKE_PROB, \
        _G_BASE_SEED, \
        _G_CONTROLLER, \
        _G_SUCCESS_MIN_COMPLETION, \
        _G_SUCCESS_GOAL_RADIUS_M, \
        _G_STOP_ON_SUCCESS, \
        _G_TASK_MODE, \
        _G_TIME_LIMIT_S
    _G_TASKS = tasks
    _G_CONFIGS = configs
    _G_CANDIDATES = candidates
    _G_SIM_CFG = sim_cfg
    _G_DELAY_S = float(delay_s)
    _G_VLM_QUERY_HZ = float(vlm_query_hz)
    _G_VLM_MAX_INFLIGHT = int(vlm_max_inflight)
    _G_VLM_MISTAKE_PROB = float(vlm_mistake_prob)
    _G_BASE_SEED = int(base_seed)
    _G_CONTROLLER = str(controller)
    _G_SUCCESS_MIN_COMPLETION = float(success_min_completion)
    _G_SUCCESS_GOAL_RADIUS_M = float(success_goal_radius_m)
    _G_STOP_ON_SUCCESS = bool(stop_on_success)
    _G_TASK_MODE = str(task_mode)
    _G_TIME_LIMIT_S = float(time_limit_s)


def _run_one_job(job: tuple[int, int]) -> RunResult:
    """job=(task_idx, cfg_idx)"""
    assert _G_TASKS is not None and _G_CONFIGS is not None and _G_CANDIDATES is not None
    assert (
        _G_SIM_CFG is not None
        and _G_DELAY_S is not None
        and _G_VLM_QUERY_HZ is not None
        and _G_VLM_MAX_INFLIGHT is not None
        and _G_VLM_MISTAKE_PROB is not None
        and _G_BASE_SEED is not None
        and _G_CONTROLLER is not None
        and _G_SUCCESS_MIN_COMPLETION is not None
        and _G_SUCCESS_GOAL_RADIUS_M is not None
        and _G_STOP_ON_SUCCESS is not None
        and _G_TASK_MODE is not None
        and _G_TIME_LIMIT_S is not None
    )
    ti, ci = int(job[0]), int(job[1])
    task = _G_TASKS[ti]
    fcfg, smodel, pol = _G_CONFIGS[ci]
    seed = int(_G_BASE_SEED) + 1000003 * int(ti) + 1009 * int(ci)
    res, _trace = simulate_closed_loop(
        task=task,
        candidates_body=_G_CANDIDATES,
        sim_cfg=_G_SIM_CFG,
        score_model=smodel,
        controller=str(_G_CONTROLLER),  # type: ignore[arg-type]
        policy=pol,
        delay_s=float(_G_DELAY_S),
        vlm_query_hz=float(_G_VLM_QUERY_HZ),
        vlm_max_inflight=int(_G_VLM_MAX_INFLIGHT),
        vlm_mistake_prob=float(_G_VLM_MISTAKE_PROB),
        fusion_cfg=fcfg,
        seed=int(seed),
        success_min_completion=float(_G_SUCCESS_MIN_COMPLETION),
        success_goal_radius_m=float(_G_SUCCESS_GOAL_RADIUS_M),
        stop_on_success=bool(_G_STOP_ON_SUCCESS),
        time_limit_s=(float(_G_TIME_LIMIT_S) if str(_G_TASK_MODE) == "arclength" else None),
        return_trace=False,
    )
    return res


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(getattr(args, "postprocess_run_dir", "")).strip():
        _postprocess_existing_run(
            run_dir=Path(str(args.postprocess_run_dir)).resolve(), num_workers=int(args.num_workers)
        )
        return

    if not str(args.dataset).strip():
        raise SystemExit("Missing --dataset (or pass --postprocess-run-dir).")

    dataset_root = Path(args.dataset).resolve()

    # Resolve query cadence representation.
    # We keep the internal simulation parameter as Hz, but store BOTH Hz and interval (s) in config
    # so downstream analysis can use the clearer interval units.
    vlm_query_interval_s = getattr(args, "vlm_query_interval_s", None)
    if vlm_query_interval_s is not None:
        try:
            vlm_query_interval_s = float(vlm_query_interval_s)
        except Exception:
            vlm_query_interval_s = None
    if vlm_query_interval_s is not None and float(vlm_query_interval_s) > 1e-9:
        vlm_query_hz = 1.0 / float(vlm_query_interval_s)
    elif vlm_query_interval_s is not None and float(vlm_query_interval_s) <= 1e-9:
        # Treat non-positive interval as "no cadence gating" (equivalent to hz=0 => request every
        # plan tick).
        vlm_query_hz = 0.0
        vlm_query_interval_s = 0.0
    else:
        vlm_query_hz = float(args.vlm_query_hz)
        vlm_query_interval_s = (1.0 / float(vlm_query_hz)) if float(vlm_query_hz) > 1e-9 else 0.0
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"closed_loop_fusion_{run_tag}"
    if run_dir.exists() and any(run_dir.iterdir()) and not bool(args.overwrite):
        raise SystemExit(f"Output exists and not empty: {run_dir} (pass --overwrite).")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Limits from robot.yaml
    max_v, max_w = _read_robot_yaml_limits(repo_root)
    if getattr(args, "max_v", None) is not None:
        max_v = float(args.max_v)
    if getattr(args, "max_w", None) is not None:
        max_w = float(args.max_w)
    limits = Limits(
        max_v=float(max_v), max_w=float(max_w), max_a=1.0, max_alpha=1.2, max_lat_accel=0.8
    )
    sim_cfg = SimLoopConfig(
        dt_control=float(args.dt_control),
        dt_plan=float(args.dt_plan),
        horizon_s=float(args.horizon_s),
        limits=limits,
        ep_cfg=EndpointPDConfig(),
        pp_cfg=PurePursuitConfig(),
    )

    static_candidates_json = resolve_static_candidate_set_path(Path(args.static_candidates_json))
    candidates_body = load_static_candidate_set(static_candidates_json, prepend_origin=True)
    policies = _parse_csv_strs(str(args.policies))
    sim_modes = _parse_csv_strs(str(args.similarity_modes))
    lambda_sims = _parse_csv_floats(str(args.lambda_sims))
    taus = _parse_csv_floats(str(args.taus))
    dist_scales = _parse_csv_floats(str(args.dist_scales))
    # Similarity config is used not only by fusion, but also by "*_match" policies
    # (they pick the candidate with max similarity to the stale VLM trajectory).
    default_sim_mode = str(sim_modes[0]) if sim_modes else "body_pointwise_arclen"
    # Planner model sweeps
    noise_stds = _parse_csv_floats(str(args.noise_stds))
    epsilons = _parse_csv_floats(str(args.epsilons))
    score_temps = _parse_csv_floats(str(args.score_temps))

    # Build tasks
    tasks = build_tasks_from_dataset(
        dataset_root=dataset_root,
        duration_s=float(args.duration_s),
        min_episode_s=float(args.min_episode_s),
        stride_s=float(args.stride_s),
        max_episodes=int(args.max_episodes),
        max_tasks_per_episode=int(args.max_tasks_per_episode),
        task_sampling=str(args.task_sampling),
        seed=int(args.seed),
    )
    if not tasks:
        raise SystemExit("No tasks built. Check dataset path and task filters.")

    # Deterministic task sampling order (for reproducibility)
    rng = random.Random(int(args.seed))
    rng.shuffle(tasks)
    if int(args.max_tasks) > 0:
        tasks = tasks[: int(args.max_tasks)]

    # Build sweep configs
    configs: list[tuple[FusionConfig, PlannerScoreModel, PolicyName]] = []
    for pol in policies:
        if pol not in (
            "local_only",
            "planner_oracle",
            "vlm_hold",
            "vlm_stream",
            "vlm_hold_match",
            "vlm_stream_match",
            "score_fusion",
            "prob_fusion",
            "score_fusion_stream",
            "prob_fusion_stream",
        ):
            continue
        if pol == "planner_oracle":
            # Deterministic, non-corrupted oracle planner (no sweep needed).
            smodel = PlannerScoreModel(noise_std=0.0, score_scale=1.0, epsilon=0.0, temperature=1.0)
            fcfg = FusionConfig(
                enabled=False,
                similarity_mode=default_sim_mode,  # type: ignore[arg-type]
                lambda_sim=0.0,
                staleness_tau_s=3.0,
                dist_scale_m=1.0,
                align_to_current_pose=True,
            )
            configs.append((fcfg, smodel, pol))  # type: ignore[arg-type]
            continue
        # Iterate over planner model parameters for all policies
        for noise_std in noise_stds:
            for eps in epsilons:
                for temp in score_temps:
                    smodel = PlannerScoreModel(
                        noise_std=float(noise_std),
                        score_scale=1.0,
                        epsilon=float(eps),
                        temperature=float(temp),
                    )
                    if pol not in (
                        "score_fusion",
                        "prob_fusion",
                        "score_fusion_stream",
                        "prob_fusion_stream",
                    ):
                        # fusion knobs don't matter; still record one config for reporting
                        # consistency
                        fcfg = FusionConfig(
                            enabled=False,
                            similarity_mode=default_sim_mode,  # type: ignore[arg-type]
                            lambda_sim=0.0,
                            staleness_tau_s=3.0,
                            dist_scale_m=1.0,
                            align_to_current_pose=True,
                        )
                        configs.append((fcfg, smodel, pol))  # type: ignore[arg-type]
                        continue
                    # fusion policies: iterate over fusion parameters
                    for mode in sim_modes:
                        for lam in lambda_sims:
                            for tau in taus:
                                for ds in dist_scales:
                                    fcfg = FusionConfig(
                                        enabled=True,
                                        similarity_mode=mode,  # type: ignore[arg-type]
                                        lambda_sim=float(lam),
                                        staleness_tau_s=float(tau),
                                        dist_scale_m=float(ds),
                                        align_to_current_pose=True,
                                    )
                                    configs.append((fcfg, smodel, pol))  # type: ignore[arg-type]

    if int(args.max_configs) > 0:
        configs = configs[: int(args.max_configs)]

    # Jobs: (task_idx, cfg_idx)
    jobs: list[tuple[int, int]] = []
    for ti in range(len(tasks)):
        for ci in range(len(configs)):
            jobs.append((int(ti), int(ci)))

    if int(args.max_jobs) > 0:
        jobs = jobs[: int(args.max_jobs)]

    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "tasks": int(len(tasks)),
                    "configs": int(len(configs)),
                    "jobs": int(len(jobs)),
                    "policies": policies,
                    "delay_s": float(args.delay_s),
                },
                indent=2,
            )
        )
        return

    tqdm = _try_import_tqdm()
    total = len(jobs)
    pbar = (
        tqdm(total=total, desc="closed_loop_sweep", unit="run", disable=(not sys.stderr.isatty()))
        if tqdm is not None
        else None
    )

    # Stream results to disk so long runs are debuggable / resumable.
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_root),
                "static_candidates_json": str(static_candidates_json),
                "limits": {
                    "max_v": float(max_v),
                    "max_w": float(max_w),
                    "max_a": 1.0,
                    "max_alpha": 1.2,
                    "max_lat_accel": 0.8,
                },
                "sim": asdict(sim_cfg),
                "controller": str(args.controller),
                "tasks": {
                    "task_mode": str(args.task_mode),
                    "duration_s": float(args.duration_s),
                    "min_episode_s": float(args.min_episode_s),
                    "stride_s": float(args.stride_s),
                    "time_limit_s": float(args.time_limit_s),
                    "max_episodes": int(args.max_episodes),
                    "max_tasks_per_episode": int(args.max_tasks_per_episode),
                    "task_sampling": str(args.task_sampling),
                    "seed": int(args.seed),
                    "tasks_built": int(len(tasks)),
                },
                "delay_s": float(args.delay_s),
                "vlm_query_hz": float(vlm_query_hz),
                "vlm_query_interval_s": float(vlm_query_interval_s),
                "vlm_max_inflight": int(args.vlm_max_inflight),
                "vlm_mistake_prob": float(args.vlm_mistake_prob),
                "success": {
                    "min_completion": float(args.success_min_completion),
                    "goal_radius_m": float(args.success_goal_radius_m),
                },
                "stop_on_success": bool(args.stop_on_success),
                "policies": policies,
                "planner_score": {
                    "noise_stds": noise_stds,
                    "epsilons": epsilons,
                    "score_temps": score_temps,
                },
                "fusion_sweep": {
                    "similarity_modes": sim_modes,
                    "lambda_sims": lambda_sims,
                    "taus": taus,
                    "dist_scales": dist_scales,
                },
                "num_workers": int(args.num_workers),
                "caps": {
                    "max_tasks": int(args.max_tasks),
                    "max_configs": int(args.max_configs),
                    "max_jobs": int(args.max_jobs),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    out_csv = run_dir / "results.csv"
    out_csv_f = out_csv.open("w", newline="", encoding="utf-8")
    results_writer = None
    results: list[RunResult] = []
    wrote = 0

    if int(args.num_workers) > 0:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(args.num_workers),
            initializer=_init_worker,
            initargs=(
                tasks,
                configs,
                candidates_body,
                sim_cfg,
                float(args.delay_s),
                float(vlm_query_hz),
                int(args.vlm_max_inflight),
                float(args.vlm_mistake_prob),
                int(args.seed),
                str(args.controller),
                float(args.success_min_completion),
                float(args.success_goal_radius_m),
                bool(args.stop_on_success),
                str(args.task_mode),
                float(args.time_limit_s),
            ),
        ) as ex:
            futs = [ex.submit(_run_one_job, j) for j in jobs]
            for f in concurrent.futures.as_completed(futs):
                res = f.result()
                results.append(res)
                if results_writer is None:
                    results_writer = csv.DictWriter(out_csv_f, fieldnames=list(asdict(res).keys()))
                    results_writer.writeheader()
                results_writer.writerow(asdict(res))
                wrote += 1
                if int(args.checkpoint_every) > 0 and (wrote % int(args.checkpoint_every) == 0):
                    out_csv_f.flush()
                if pbar is not None:
                    pbar.update(1)
    else:
        _init_worker(
            tasks,
            configs,
            candidates_body,
            sim_cfg,
            float(args.delay_s),
            float(vlm_query_hz),
            int(args.vlm_max_inflight),
            float(args.vlm_mistake_prob),
            int(args.seed),
            str(args.controller),
            float(args.success_min_completion),
            float(args.success_goal_radius_m),
            bool(args.stop_on_success),
            str(args.task_mode),
            float(args.time_limit_s),
        )
        for j in jobs:
            res = _run_one_job(j)
            results.append(res)
            if results_writer is None:
                results_writer = csv.DictWriter(out_csv_f, fieldnames=list(asdict(res).keys()))
                results_writer.writeheader()
            results_writer.writerow(asdict(res))
            wrote += 1
            if int(args.checkpoint_every) > 0 and (wrote % int(args.checkpoint_every) == 0):
                out_csv_f.flush()
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()
    out_csv_f.flush()
    out_csv_f.close()

    # Aggregate summary by config
    def _key(r: RunResult) -> tuple:
        return (
            r.controller,
            r.policy,
            r.delay_s,
            r.similarity_mode,
            r.lambda_sim,
            r.staleness_tau_s,
            r.dist_scale_m,
            r.noise_std,
            r.epsilon,
            r.temperature,
            r.vlm_mistake_prob,
        )

    def _nanmean(x: np.ndarray) -> float:
        if x.size <= 0:
            return float("nan")
        if not np.any(np.isfinite(x)):
            return float("nan")
        try:
            return float(np.nanmean(x))
        except Exception:
            return float("nan")

    by = {}
    for r in results:
        by.setdefault(_key(r), []).append(r)
    summary_rows = []
    for k, rs in by.items():
        ctes = np.asarray([float(x.mean_cte_m) for x in rs], dtype=np.float64)
        p95 = np.asarray([float(x.p95_cte_m) for x in rs], dtype=np.float64)
        spd = np.asarray([float(x.mean_speed_mps) for x in rs], dtype=np.float64)
        sw = np.asarray([float(x.chosen_switches) for x in rs], dtype=np.float64)
        comp = np.asarray([float(x.route_completion_max) for x in rs], dtype=np.float64)
        gdist = np.asarray([float(x.goal_dist_final_m) for x in rs], dtype=np.float64)
        gdist_min = np.asarray([float(x.goal_dist_min_m) for x in rs], dtype=np.float64)
        succ = np.asarray([float(x.success) for x in rs], dtype=np.float64)
        ref_len_m = np.asarray([float(x.ref_len_m) for x in rs], dtype=np.float64)
        ref_speed = np.asarray([float(x.ref_mean_speed_mps) for x in rs], dtype=np.float64)
        sim_h = np.asarray([float(x.sim_horizon_s) for x in rs], dtype=np.float64)
        ub = np.asarray([float(x.completion_upper_bound) for x in rs], dtype=np.float64)
        oracle_eq = np.asarray([float(x.oracle_eq_argmax_ratio) for x in rs], dtype=np.float64)
        vlm_like_eq = np.asarray([float(x.vlm_like_eq_argmax_ratio) for x in rs], dtype=np.float64)
        chosen_eq_argmax = np.asarray(
            [float(x.chosen_eq_argmax_ratio) for x in rs], dtype=np.float64
        )
        chosen_eq_vlm = np.asarray(
            [float(x.chosen_eq_vlm_like_ratio) for x in rs], dtype=np.float64
        )
        vlm_diff = np.asarray([float(x.vlm_like_diff_argmax_frac) for x in rs], dtype=np.float64)
        switch_from_argmax = np.asarray(
            [float(x.switch_from_argmax_ratio) for x in rs], dtype=np.float64
        )
        switch_to_vlm_given_diff = np.asarray(
            [float(x.switch_to_vlm_like_given_vlm_diff_ratio) for x in rs], dtype=np.float64
        )
        tv = np.asarray([float(x.tv_dist_chosen_vs_argmax) for x in rs], dtype=np.float64)
        summary_rows.append(
            {
                "controller": str(k[0]),
                "policy": str(k[1]),
                "delay_s": float(k[2]),
                "similarity_mode": str(k[3]),
                "lambda_sim": float(k[4]),
                "staleness_tau_s": float(k[5]),
                "dist_scale_m": float(k[6]),
                "noise_std": float(k[7]),
                "epsilon": float(k[8]),
                "temperature": float(k[9]),
                "n": int(len(rs)),
                "mean_cte_m": float(np.mean(ctes)) if ctes.size else float("nan"),
                "p95_cte_m": float(np.mean(p95)) if p95.size else float("nan"),
                "mean_speed_mps": float(np.mean(spd)) if spd.size else float("nan"),
                "mean_switches": float(np.mean(sw)) if sw.size else float("nan"),
                "mean_route_completion": _nanmean(comp),
                "mean_goal_dist_final_m": _nanmean(gdist),
                "mean_goal_dist_min_m": _nanmean(gdist_min),
                "success_rate": _nanmean(succ),
                "mean_ref_len_m": _nanmean(ref_len_m),
                "mean_ref_speed_mps": _nanmean(ref_speed),
                "mean_sim_horizon_s": _nanmean(sim_h),
                "mean_completion_upper_bound": _nanmean(ub),
                # behavior agreement metrics (mean across tasks)
                "mean_oracle_eq_argmax": _nanmean(oracle_eq),
                "mean_vlm_like_eq_argmax": _nanmean(vlm_like_eq),
                "mean_chosen_eq_argmax": _nanmean(chosen_eq_argmax),
                "mean_chosen_eq_vlm_like": _nanmean(chosen_eq_vlm),
                "mean_vlm_like_diff_argmax": _nanmean(vlm_diff),
                "mean_switch_from_argmax": _nanmean(switch_from_argmax),
                "mean_switch_to_vlm_given_vlm_diff": _nanmean(switch_to_vlm_given_diff),
                "mean_tv_dist_chosen_vs_argmax": _nanmean(tv),
            }
        )

    summary_csv = run_dir / "summary.csv"
    cols2 = list(summary_rows[0].keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols2)
        w.writeheader()
        w.writerows(summary_rows)

    # Best configs by mean_cte
    def _sort_key(r: dict) -> tuple[float, float, float, float]:
        # Prefer "finish the route" over low CTE.
        try:
            succ = float(r.get("success_rate", float("nan")))
        except Exception:
            succ = float("nan")
        try:
            comp = float(r.get("mean_route_completion", float("nan")))
        except Exception:
            comp = float("nan")
        try:
            g = float(r.get("mean_goal_dist_min_m", r.get("mean_goal_dist_final_m", float("nan"))))
        except Exception:
            g = float("nan")
        try:
            cte = float(r.get("mean_cte_m", float("nan")))
        except Exception:
            cte = float("nan")
        # Sort descending success, descending completion, ascending goal distance, then ascending
        # CTE.
        succ = succ if math.isfinite(succ) else -1.0
        comp = comp if math.isfinite(comp) else -1.0
        g = g if math.isfinite(g) else float("inf")
        cte = cte if math.isfinite(cte) else float("inf")
        return (-succ, -comp, g, cte)

    best = sorted(summary_rows, key=_sort_key)[:30]

    # Write a markdown report
    report = run_dir / "report.md"
    lines = []
    lines.append("# Closed-loop score fusion benchmark (real odom reference)\n")
    lines.append("## Run config\n")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "dataset": str(dataset_root),
                "static_candidates_json": str(static_candidates_json),
                "limits": {
                    "max_v": float(max_v),
                    "max_w": float(max_w),
                    "max_a": 1.0,
                    "max_alpha": 1.2,
                    "max_lat_accel": 0.8,
                },
                "sim": asdict(sim_cfg),
                "controller": str(args.controller),
                "tasks": {
                    "task_mode": str(args.task_mode),
                    "duration_s": float(args.duration_s),
                    "min_episode_s": float(args.min_episode_s),
                    "stride_s": float(args.stride_s),
                    "time_limit_s": float(args.time_limit_s),
                    "max_episodes": int(args.max_episodes),
                    "max_tasks_per_episode": int(args.max_tasks_per_episode),
                    "task_sampling": str(args.task_sampling),
                    "seed": int(args.seed),
                    "tasks_built": int(len(tasks)),
                },
                "delay_s": float(args.delay_s),
                "vlm_query_hz": float(vlm_query_hz),
                "vlm_query_interval_s": float(vlm_query_interval_s),
                "vlm_max_inflight": int(args.vlm_max_inflight),
                "success": {
                    "min_completion": float(args.success_min_completion),
                    "goal_radius_m": float(args.success_goal_radius_m),
                },
                "policies": policies,
                "planner_score": {
                    "noise_stds": noise_stds,
                    "epsilons": epsilons,
                    "score_temps": score_temps,
                },
                "fusion_sweep": {
                    "similarity_modes": sim_modes,
                    "lambda_sims": lambda_sims,
                    "taus": taus,
                    "dist_scales": dist_scales,
                },
                "num_workers": int(args.num_workers),
            },
            indent=2,
        )
    )
    lines.append("```\n")
    lines.append("## Top configs (success rate / route completion)\n")
    lines.append("")
    lines.append(
        "Note: completion is bounded by reference speed vs max_v. "
        "See `mean_ref_speed_mps` and `mean_completion_upper_bound` in `summary.csv`."
    )
    lines.append("")
    lines.append(
        "| rank | controller | policy | delay_s | sim | lambda | tau | dist_scale | success | "
        "completion | ub_completion | ref_speed | mean_cte | mean_speed | mean_switches | n |"
    )
    lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(best, start=1):
        lines.append(
            f"| {i} | {r.get('controller', '')} | {r['policy']} | {r['delay_s']:.2f} "
            f"| {r['similarity_mode']} | {r['lambda_sim']:.3g} | {r['staleness_tau_s']:.3g} "
            f"| {r['dist_scale_m']:.3g} "
            f"| {float(r.get('success_rate', float('nan'))):.3f} "
            f"| {float(r.get('mean_route_completion', float('nan'))):.3f} "
            f"| {float(r.get('mean_completion_upper_bound', float('nan'))):.3f} "
            f"| {float(r.get('mean_ref_speed_mps', float('nan'))):.3f} "
            f"| {r['mean_cte_m']:.4f} | {r['mean_speed_mps']:.3f} "
            f"| {r['mean_switches']:.2f} | {r['n']} |"
        )

    # Example BEV images: show how the robot walks.
    if bool(args.write_examples):
        lines.append("\n## BEV examples\n")
        ex_dir = run_dir / "examples_bev"
        ex_dir.mkdir(parents=True, exist_ok=True)

        # Choose which policies to visualize. Prefer a representative fixed set, but only keep
        # those actually enabled.
        prefer_pols = [
            "planner_oracle",
            "local_only",
            "vlm_hold",
            "vlm_stream",
            "vlm_hold_match",
            "vlm_stream_match",
            "score_fusion_stream",
            "prob_fusion_stream",
            "score_fusion",
            "prob_fusion",
        ]
        enabled = [p for p in prefer_pols if p in set(policies)]
        if not enabled:
            enabled = [str(policies[0])] if policies else []

        # Choose example tasks:
        # - default: first N tasks in the current task list
        # - if --examples-by-episode: earliest window per episode (by start_t), then take N episodes
        if bool(args.examples_by_episode):
            best_by_ep: dict[str, int] = {}
            for i, t in enumerate(tasks):
                prev = best_by_ep.get(str(t.episode_id))
                if prev is None or float(t.start_t) < float(tasks[int(prev)].start_t):
                    best_by_ep[str(t.episode_id)] = int(i)
            ex_tasks = [best_by_ep[k] for k in sorted(best_by_ep.keys())]
            ex_tasks = ex_tasks[: int(max(0, int(args.examples_n)))]
        else:
            ex_tasks = list(range(min(int(args.examples_n), len(tasks))))

        base_seed = int(args.seed)
        for i, ti in enumerate(ex_tasks):
            task = tasks[int(ti)]
            for pi, pol in enumerate(enabled):
                # Best-effort: reuse the first sweep settings for examples (not critical for
                # visualization).
                fcfg = FusionConfig(
                    enabled=(
                        str(pol)
                        in (
                            "score_fusion",
                            "prob_fusion",
                            "score_fusion_stream",
                            "prob_fusion_stream",
                        )
                    ),
                    similarity_mode=default_sim_mode,  # type: ignore[arg-type]
                    lambda_sim=float(lambda_sims[0] if lambda_sims else 0.0),
                    staleness_tau_s=float(taus[0] if taus else 3.0),
                    dist_scale_m=float(dist_scales[0] if dist_scales else 1.0),
                    align_to_current_pose=True,
                )
                smodel = PlannerScoreModel(
                    noise_std=float(noise_stds[0] if noise_stds else 0.0),
                    score_scale=1.0,
                    epsilon=float(epsilons[0] if epsilons else 0.0),
                    temperature=float(score_temps[0] if score_temps else 1.0),
                )
                seed = int(base_seed) + 1000003 * int(ti) + 1009 * int(pi)
                rr, tr = simulate_closed_loop(
                    task=task,
                    candidates_body=candidates_body,
                    sim_cfg=sim_cfg,
                    score_model=smodel,
                    controller=str(args.controller),  # type: ignore[arg-type]
                    policy=str(pol),  # type: ignore[arg-type]
                    delay_s=float(args.delay_s),
                    vlm_query_hz=float(vlm_query_hz),
                    vlm_max_inflight=int(args.vlm_max_inflight),
                    vlm_mistake_prob=float(args.vlm_mistake_prob),
                    fusion_cfg=fcfg,
                    seed=int(seed),
                    success_min_completion=float(args.success_min_completion),
                    success_goal_radius_m=float(args.success_goal_radius_m),
                    stop_on_success=bool(args.stop_on_success),
                    time_limit_s=(
                        float(args.time_limit_s) if str(args.task_mode) == "arclength" else None
                    ),
                    return_trace=True,
                )
                p = ex_dir / f"task{i:02d}_{task.episode_id}_{task.start_t:.0f}_{pol}.png"
                title = (
                    f"{rr.task_id} | {pol} delay={rr.delay_s:.1f}s | "
                    f"succ={int(rr.success)} goal_min={rr.goal_dist_min_m:.2f} "
                    f"goal_final={rr.goal_dist_final_m:.2f} "
                    f"cte={rr.mean_cte_m:.2f}"
                )
                _bev_plot(p, trace=tr, title=title)
                # Save trace alongside the image for richer notebook debugging (fusion curves, plan
                # overlays, etc.).
                try:
                    import numpy as _np

                    _np.savez_compressed(str(p.with_suffix(".npz")), **tr)
                except Exception:
                    pass
                lines.append(f"- `{p.relative_to(run_dir)}`")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "results_csv": str(out_csv),
                "summary_csv": str(summary_csv),
                "report": str(report),
            },
            indent=2,
        )
    )


_GP_TASKS: list[TaskWindow] | None = None
_GP_CANDIDATES: list[np.ndarray] | None = None
_GP_SIM_CFG: SimLoopConfig | None = None
_GP_VLM_QUERY_HZ: float | None = None
_GP_VLM_MAX_INFLIGHT: int | None = None
_GP_VLM_MISTAKE_PROB: float | None = None
_GP_SUCCESS_MIN_COMPLETION: float | None = None
_GP_SUCCESS_GOAL_RADIUS_M: float | None = None
_GP_STOP_ON_SUCCESS: bool | None = None
_GP_TASK_MODE: str | None = None
_GP_TIME_LIMIT_S: float | None = None


def _init_postprocess_worker(
    tasks: list[TaskWindow],
    candidates: list[np.ndarray],
    sim_cfg: SimLoopConfig,
    vlm_query_hz: float,
    vlm_max_inflight: int,
    vlm_mistake_prob: float,
    success_min_completion: float,
    success_goal_radius_m: float,
    stop_on_success: bool,
    task_mode: str,
    time_limit_s: float,
) -> None:
    global \
        _GP_TASKS, \
        _GP_CANDIDATES, \
        _GP_SIM_CFG, \
        _GP_VLM_QUERY_HZ, \
        _GP_VLM_MAX_INFLIGHT, \
        _GP_VLM_MISTAKE_PROB, \
        _GP_SUCCESS_MIN_COMPLETION, \
        _GP_SUCCESS_GOAL_RADIUS_M, \
        _GP_STOP_ON_SUCCESS, \
        _GP_TASK_MODE, \
        _GP_TIME_LIMIT_S
    _GP_TASKS = tasks
    _GP_CANDIDATES = candidates
    _GP_SIM_CFG = sim_cfg
    _GP_VLM_QUERY_HZ = float(vlm_query_hz)
    _GP_VLM_MAX_INFLIGHT = int(vlm_max_inflight)
    _GP_VLM_MISTAKE_PROB = float(vlm_mistake_prob)
    _GP_SUCCESS_MIN_COMPLETION = float(success_min_completion)
    _GP_SUCCESS_GOAL_RADIUS_M = float(success_goal_radius_m)
    _GP_STOP_ON_SUCCESS = bool(stop_on_success)
    _GP_TASK_MODE = str(task_mode)
    _GP_TIME_LIMIT_S = float(time_limit_s)


def _run_postprocess_job(
    job: tuple[int, str, str, float, int, str, float, float, float, bool, float, float, float],
) -> RunResult:
    """job=(task_idx, controller, policy, delay_s, seed, similarity_mode, lambda_sim, tau_s,
    dist_scale_m, align_to_current_pose, noise_std, epsilon, temperature)"""
    if (
        _GP_TASKS is None
        or _GP_CANDIDATES is None
        or _GP_SIM_CFG is None
        or _GP_VLM_QUERY_HZ is None
        or _GP_VLM_MAX_INFLIGHT is None
        or _GP_VLM_MISTAKE_PROB is None
        or _GP_SUCCESS_MIN_COMPLETION is None
        or _GP_SUCCESS_GOAL_RADIUS_M is None
        or _GP_STOP_ON_SUCCESS is None
        or _GP_TASK_MODE is None
        or _GP_TIME_LIMIT_S is None
    ):
        raise RuntimeError("postprocess worker not initialized")
    (
        ti,
        controller,
        pol,
        delay_s,
        seed,
        sim_mode,
        lam,
        tau_s,
        dist_scale_m,
        align_to_current_pose,
        noise_std,
        epsilon,
        temperature,
    ) = job
    task = _GP_TASKS[int(ti)]
    fcfg = FusionConfig(
        enabled=(
            str(pol) in ("score_fusion", "prob_fusion", "score_fusion_stream", "prob_fusion_stream")
        ),
        similarity_mode=str(sim_mode),  # type: ignore[arg-type]
        lambda_sim=float(lam),
        staleness_tau_s=float(tau_s),
        dist_scale_m=float(dist_scale_m),
        align_to_current_pose=bool(align_to_current_pose),
    )
    smodel = PlannerScoreModel(
        noise_std=float(noise_std),
        score_scale=1.0,
        epsilon=float(epsilon),
        temperature=float(temperature),
    )
    res, _trace = simulate_closed_loop(
        task=task,
        candidates_body=_GP_CANDIDATES,
        sim_cfg=_GP_SIM_CFG,
        score_model=smodel,
        controller=str(controller),  # type: ignore[arg-type]
        policy=str(pol),  # type: ignore[arg-type]
        delay_s=float(delay_s),
        vlm_query_hz=float(_GP_VLM_QUERY_HZ),
        vlm_max_inflight=int(_GP_VLM_MAX_INFLIGHT),
        vlm_mistake_prob=float(_GP_VLM_MISTAKE_PROB),
        fusion_cfg=fcfg,
        seed=int(seed),
        success_min_completion=float(_GP_SUCCESS_MIN_COMPLETION),
        success_goal_radius_m=float(_GP_SUCCESS_GOAL_RADIUS_M),
        stop_on_success=bool(_GP_STOP_ON_SUCCESS),
        time_limit_s=(float(_GP_TIME_LIMIT_S) if str(_GP_TASK_MODE) == "arclength" else None),
        return_trace=False,
    )
    return res


def _postprocess_existing_run(*, run_dir: Path, num_workers: int) -> None:
    """Augment an existing run_dir by recomputing additional behavior metrics.

    Writes:
    - results_augmented.csv (new results with extra columns)
    - summary.csv (overwritten)
    - report.md (overwritten)
    """
    run_dir = Path(run_dir).resolve()
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config.json: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    success_cfg = cfg.get("success", {}) if isinstance(cfg.get("success", {}), dict) else {}
    try:
        success_min_completion = float(success_cfg.get("min_completion", 0.95))
    except Exception:
        success_min_completion = 0.95
    try:
        success_goal_radius_m = float(success_cfg.get("goal_radius_m", 1.0))
    except Exception:
        success_goal_radius_m = 1.0
    stop_on_success = bool(cfg.get("stop_on_success", True))
    try:
        vlm_mistake_prob = float(cfg.get("vlm_mistake_prob", 0.0))
    except Exception:
        vlm_mistake_prob = 0.0
    # Prefer interval (seconds/query) if present; else fallback to stored Hz.
    try:
        _interval_s = cfg.get("vlm_query_interval_s", None)
        _interval_s = float(_interval_s) if _interval_s is not None else None
    except Exception:
        _interval_s = None
    if _interval_s is not None and float(_interval_s) > 1e-9:
        vlm_query_hz = 1.0 / float(_interval_s)
    else:
        try:
            vlm_query_hz = float(cfg.get("vlm_query_hz", 1.0))
        except Exception:
            vlm_query_hz = 1.0
    try:
        vlm_max_inflight = int(cfg.get("vlm_max_inflight", 0))
    except Exception:
        vlm_max_inflight = 0

    dataset_root = Path(str(cfg.get("dataset", ""))).resolve()
    static_candidates_json = resolve_static_candidate_set_path(
        str(cfg.get("static_candidates_json", ""))
    )
    if not str(dataset_root) or not dataset_root.exists():
        raise SystemExit(f"Bad dataset path in config.json: {dataset_root}")

    # Reconstruct sim config from stored dict (best-effort, tolerant to missing keys).
    sim_d = cfg.get("sim", {}) if isinstance(cfg.get("sim", {}), dict) else {}
    limits_d = sim_d.get("limits", {}) if isinstance(sim_d.get("limits", {}), dict) else {}
    ep_d = sim_d.get("ep_cfg", {}) if isinstance(sim_d.get("ep_cfg", {}), dict) else {}
    pp_d = sim_d.get("pp_cfg", {}) if isinstance(sim_d.get("pp_cfg", {}), dict) else {}
    limits = Limits(
        max_v=float(limits_d.get("max_v", 1.0)),
        max_w=float(limits_d.get("max_w", 0.5)),
        max_a=float(limits_d.get("max_a", 1.0)),
        max_alpha=float(limits_d.get("max_alpha", 1.2)),
        max_lat_accel=float(limits_d.get("max_lat_accel", 0.8)),
    )
    ep_kwargs = {
        k: ep_d[k]
        for k in ep_d.keys()
        if k in getattr(EndpointPDConfig, "__dataclass_fields__", {})
    }
    ep_cfg = EndpointPDConfig(**ep_kwargs) if ep_kwargs else EndpointPDConfig()
    pp_kwargs = {
        k: pp_d[k]
        for k in pp_d.keys()
        if k in getattr(PurePursuitConfig, "__dataclass_fields__", {})
    }
    pp_cfg = PurePursuitConfig(**pp_kwargs) if pp_kwargs else PurePursuitConfig()
    sim_cfg = SimLoopConfig(
        dt_control=float(sim_d.get("dt_control", 0.1)),
        dt_plan=float(sim_d.get("dt_plan", 0.2)),
        horizon_s=float(sim_d.get("horizon_s", 4.0)),
        limits=limits,
        ep_cfg=ep_cfg,
        pp_cfg=pp_cfg,
    )

    candidates_body = load_static_candidate_set(static_candidates_json, prepend_origin=True)

    tasks_d = cfg.get("tasks", {}) if isinstance(cfg.get("tasks", {}), dict) else {}
    task_mode = str(tasks_d.get("task_mode", "time")).strip() or "time"
    try:
        time_limit_s = float(tasks_d.get("time_limit_s", tasks_d.get("duration_s", 20.0)))
    except Exception:
        time_limit_s = float(tasks_d.get("duration_s", 20.0))
    tasks = build_tasks_from_dataset(
        dataset_root=dataset_root,
        duration_s=float(tasks_d.get("duration_s", 20.0)),
        min_episode_s=float(tasks_d.get("min_episode_s", 25.0)),
        stride_s=float(tasks_d.get("stride_s", 5.0)),
        max_episodes=int(tasks_d.get("max_episodes", 32)),
        max_tasks_per_episode=int(tasks_d.get("max_tasks_per_episode", 3)),
        task_sampling=str(tasks_d.get("task_sampling", "first")),
        seed=int(tasks_d.get("seed", 0)),
    )
    if not tasks:
        raise SystemExit("No tasks built during postprocess (dataset path or filters may differ).")
    id_to_idx = {f"{t.episode_id}@{t.start_t:.2f}": i for i, t in enumerate(tasks)}

    in_csv = run_dir / "results.csv"
    if not in_csv.exists():
        raise SystemExit(f"Missing results.csv: {in_csv}")

    default_controller = str(cfg.get("controller", "endpoint_pd"))
    jobs: list[
        tuple[int, str, str, float, int, str, float, float, float, bool, float, float, float]
    ] = []
    planner_cfg = (
        cfg.get("planner_score", {}) if isinstance(cfg.get("planner_score", {}), dict) else {}
    )
    default_noise_std = float(
        planner_cfg.get(
            "noise_std",
            (planner_cfg.get("noise_stds") or [0.15])[0],
        )
    )
    default_epsilon = float(
        planner_cfg.get(
            "epsilon",
            (planner_cfg.get("epsilons") or [0.0])[0],
        )
    )
    default_temp = float(
        planner_cfg.get(
            "temperature",
            (planner_cfg.get("score_temps") or [1.0])[0],
        )
    )
    with in_csv.open("r", newline="", encoding="utf-8") as f:
        rr = csv.DictReader(f)
        for row in rr:
            tid = str(row.get("task_id", "")).strip()
            if not tid:
                continue
            ti = id_to_idx.get(tid)
            if ti is None:
                # fallback match
                ep = str(row.get("episode_id", "")).strip()
                st = float(row.get("start_t", "nan"))
                found = None
                for i, t in enumerate(tasks):
                    if t.episode_id == ep and abs(float(t.start_t) - float(st)) < 1e-6:
                        found = i
                        break
                if found is None:
                    continue
                ti = int(found)

            jobs.append(
                (
                    int(ti),
                    str(row.get("controller", default_controller)),
                    str(row.get("policy", "local_only")),
                    float(row.get("delay_s", cfg.get("delay_s", 1.5))),
                    int(float(row.get("seed", 0))),
                    str(row.get("similarity_mode", "body_pointwise_arclen")),
                    float(row.get("lambda_sim", 0.0)),
                    float(row.get("staleness_tau_s", 3.0)),
                    float(row.get("dist_scale_m", 1.0)),
                    str(row.get("align_to_current_pose", "True")).lower()
                    in ("1", "true", "yes", "y"),
                    float(row.get("noise_std", default_noise_std)),
                    float(row.get("epsilon", default_epsilon)),
                    float(row.get("temperature", default_temp)),
                )
            )

    if not jobs:
        raise SystemExit("No jobs found in results.csv for postprocess.")

    tqdm = _try_import_tqdm()
    pbar = (
        tqdm(total=len(jobs), desc="postprocess", unit="job", disable=(not sys.stderr.isatty()))
        if tqdm is not None
        else None
    )

    out_csv = run_dir / "results_augmented.csv"
    results: list[RunResult] = []
    out_csv_f = out_csv.open("w", newline="", encoding="utf-8")
    writer = None

    import concurrent.futures

    if int(num_workers) > 0:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=int(num_workers),
            initializer=_init_postprocess_worker,
            initargs=(
                tasks,
                candidates_body,
                sim_cfg,
                float(vlm_query_hz),
                int(vlm_max_inflight),
                float(vlm_mistake_prob),
                float(success_min_completion),
                float(success_goal_radius_m),
                bool(stop_on_success),
                str(task_mode),
                float(time_limit_s),
            ),
        ) as ex:
            futs = [ex.submit(_run_postprocess_job, j) for j in jobs]
            for f in concurrent.futures.as_completed(futs):
                res = f.result()
                results.append(res)
                if writer is None:
                    writer = csv.DictWriter(out_csv_f, fieldnames=list(asdict(res).keys()))
                    writer.writeheader()
                writer.writerow(asdict(res))
                if pbar is not None:
                    pbar.update(1)
    else:
        _init_postprocess_worker(
            tasks,
            candidates_body,
            sim_cfg,
            float(vlm_query_hz),
            int(vlm_max_inflight),
            float(vlm_mistake_prob),
            float(success_min_completion),
            float(success_goal_radius_m),
            bool(stop_on_success),
            str(task_mode),
            float(time_limit_s),
        )
        for j in jobs:
            res = _run_postprocess_job(j)
            results.append(res)
            if writer is None:
                writer = csv.DictWriter(out_csv_f, fieldnames=list(asdict(res).keys()))
                writer.writeheader()
            writer.writerow(asdict(res))
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()
    out_csv_f.flush()
    out_csv_f.close()

    # Aggregate summary (same logic as main)
    def _key(r: RunResult) -> tuple:
        return (
            r.controller,
            r.policy,
            r.delay_s,
            r.similarity_mode,
            r.lambda_sim,
            r.staleness_tau_s,
            r.dist_scale_m,
            r.noise_std,
            r.epsilon,
            r.temperature,
            r.vlm_mistake_prob,
        )

    def _nanmean(x: np.ndarray) -> float:
        if x.size <= 0:
            return float("nan")
        if not np.any(np.isfinite(x)):
            return float("nan")
        try:
            return float(np.nanmean(x))
        except Exception:
            return float("nan")

    by = {}
    for r in results:
        by.setdefault(_key(r), []).append(r)

    summary_rows = []
    for k, rs in by.items():
        ctes = np.asarray([float(x.mean_cte_m) for x in rs], dtype=np.float64)
        p95 = np.asarray([float(x.p95_cte_m) for x in rs], dtype=np.float64)
        spd = np.asarray([float(x.mean_speed_mps) for x in rs], dtype=np.float64)
        sw = np.asarray([float(x.chosen_switches) for x in rs], dtype=np.float64)
        comp = np.asarray([float(x.route_completion_max) for x in rs], dtype=np.float64)
        gdist = np.asarray([float(x.goal_dist_final_m) for x in rs], dtype=np.float64)
        gdist_min = np.asarray([float(x.goal_dist_min_m) for x in rs], dtype=np.float64)
        succ = np.asarray([float(x.success) for x in rs], dtype=np.float64)
        ref_len_m = np.asarray([float(x.ref_len_m) for x in rs], dtype=np.float64)
        ref_speed = np.asarray([float(x.ref_mean_speed_mps) for x in rs], dtype=np.float64)
        sim_h = np.asarray([float(x.sim_horizon_s) for x in rs], dtype=np.float64)
        ub = np.asarray([float(x.completion_upper_bound) for x in rs], dtype=np.float64)
        oracle_eq = np.asarray([float(x.oracle_eq_argmax_ratio) for x in rs], dtype=np.float64)
        vlm_like_eq = np.asarray([float(x.vlm_like_eq_argmax_ratio) for x in rs], dtype=np.float64)
        chosen_eq_argmax = np.asarray(
            [float(x.chosen_eq_argmax_ratio) for x in rs], dtype=np.float64
        )
        chosen_eq_vlm = np.asarray(
            [float(x.chosen_eq_vlm_like_ratio) for x in rs], dtype=np.float64
        )
        vlm_diff = np.asarray([float(x.vlm_like_diff_argmax_frac) for x in rs], dtype=np.float64)
        switch_from_argmax = np.asarray(
            [float(x.switch_from_argmax_ratio) for x in rs], dtype=np.float64
        )
        switch_to_vlm_given_diff = np.asarray(
            [float(x.switch_to_vlm_like_given_vlm_diff_ratio) for x in rs], dtype=np.float64
        )
        tv = np.asarray([float(x.tv_dist_chosen_vs_argmax) for x in rs], dtype=np.float64)
        summary_rows.append(
            {
                "controller": str(k[0]),
                "policy": str(k[1]),
                "delay_s": float(k[2]),
                "similarity_mode": str(k[3]),
                "lambda_sim": float(k[4]),
                "staleness_tau_s": float(k[5]),
                "dist_scale_m": float(k[6]),
                "noise_std": float(k[7]),
                "epsilon": float(k[8]),
                "temperature": float(k[9]),
                "n": int(len(rs)),
                "mean_cte_m": float(np.mean(ctes)) if ctes.size else float("nan"),
                "p95_cte_m": float(np.mean(p95)) if p95.size else float("nan"),
                "mean_speed_mps": float(np.mean(spd)) if spd.size else float("nan"),
                "mean_switches": float(np.mean(sw)) if sw.size else float("nan"),
                "mean_route_completion": _nanmean(comp),
                "mean_goal_dist_final_m": _nanmean(gdist),
                "mean_goal_dist_min_m": _nanmean(gdist_min),
                "success_rate": _nanmean(succ),
                "mean_oracle_eq_argmax": _nanmean(oracle_eq),
                "mean_vlm_like_eq_argmax": _nanmean(vlm_like_eq),
                "mean_chosen_eq_argmax": _nanmean(chosen_eq_argmax),
                "mean_chosen_eq_vlm_like": _nanmean(chosen_eq_vlm),
                "mean_vlm_like_diff_argmax": _nanmean(vlm_diff),
                "mean_switch_from_argmax": _nanmean(switch_from_argmax),
                "mean_switch_to_vlm_given_vlm_diff": _nanmean(switch_to_vlm_given_diff),
                "mean_tv_dist_chosen_vs_argmax": _nanmean(tv),
                "mean_ref_len_m": _nanmean(ref_len_m),
                "mean_ref_speed_mps": _nanmean(ref_speed),
                "mean_sim_horizon_s": _nanmean(sim_h),
                "mean_completion_upper_bound": _nanmean(ub),
            }
        )

    summary_csv = run_dir / "summary.csv"
    cols2 = list(summary_rows[0].keys()) if summary_rows else []
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols2)
        w.writeheader()
        w.writerows(summary_rows)

    # Best configs by mean_cte
    def _sort_key(r: dict) -> tuple[float, float, float, float]:
        try:
            succ = float(r.get("success_rate", float("nan")))
        except Exception:
            succ = float("nan")
        try:
            comp = float(r.get("mean_route_completion", float("nan")))
        except Exception:
            comp = float("nan")
        try:
            g = float(r.get("mean_goal_dist_final_m", float("nan")))
        except Exception:
            g = float("nan")
        try:
            cte = float(r.get("mean_cte_m", float("nan")))
        except Exception:
            cte = float("nan")
        succ = succ if math.isfinite(succ) else -1.0
        comp = comp if math.isfinite(comp) else -1.0
        g = g if math.isfinite(g) else float("inf")
        cte = cte if math.isfinite(cte) else float("inf")
        return (-succ, -comp, g, cte)

    best = sorted(summary_rows, key=_sort_key)[:30]

    report = run_dir / "report.md"
    lines = []
    lines.append("# Closed-loop score fusion benchmark (real odom reference)\n")
    lines.append("## Run config\n")
    lines.append("```json")
    lines.append(
        json.dumps(
            {
                "dataset": str(dataset_root),
                "static_candidates_json": str(static_candidates_json),
                "limits": cfg.get("limits", {}),
                "sim": asdict(sim_cfg),
                "tasks": tasks_d,
                "delay_s": float(cfg.get("delay_s", 1.5)),
                "success": cfg.get("success", {}),
                "policies": cfg.get("policies", []),
                "planner_score": cfg.get("planner_score", {}),
                "fusion_sweep": cfg.get("fusion_sweep", {}),
                "num_workers": int(num_workers),
                "postprocess": {"results_augmented_csv": str(out_csv)},
            },
            indent=2,
        )
    )
    lines.append("```\n")
    lines.append("## Top configs (success rate / route completion)\n")
    lines.append("")
    lines.append(
        "| rank | controller | policy | delay_s | sim | lambda | tau | dist_scale | success "
        "| completion | goal_dist | mean_cte | mean_speed | mean_switches "
        "| p(vlm_oracle=argmax) | p(vlm_like=argmax) | p(chosen=argmax) | p(chosen=vlm_like) "
        "| p(switch_to_vlm|vlm_diff) | TV(chosen,argmax) | n |"
    )
    lines.append(
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for i, r in enumerate(best, start=1):
        goal_dist = float(
            r.get("mean_goal_dist_min_m", r.get("mean_goal_dist_final_m", float("nan")))
        )
        lines.append(
            f"| {i} | {r.get('controller', '')} | {r['policy']} | {r['delay_s']:.2f} "
            f"| {r['similarity_mode']} | {r['lambda_sim']:.3g} | {r['staleness_tau_s']:.3g} "
            f"| {r['dist_scale_m']:.3g} "
            f"| {float(r.get('success_rate', float('nan'))):.3f} "
            f"| {float(r.get('mean_route_completion', float('nan'))):.3f} "
            f"| {goal_dist:.3f} "
            f"| {r['mean_cte_m']:.4f} | {r['mean_speed_mps']:.3f} | {r['mean_switches']:.2f} "
            f"| {r.get('mean_oracle_eq_argmax', float('nan')):.3f} "
            f"| {r.get('mean_vlm_like_eq_argmax', float('nan')):.3f} "
            f"| {r.get('mean_chosen_eq_argmax', float('nan')):.3f} "
            f"| {r.get('mean_chosen_eq_vlm_like', float('nan')):.3f} "
            f"| {r.get('mean_switch_to_vlm_given_vlm_diff', float('nan')):.3f} "
            f"| {r.get('mean_tv_dist_chosen_vs_argmax', float('nan')):.3f} | {r['n']} |"
        )

    # List existing BEV images if present.
    ex_dir = run_dir / "examples_bev"
    if ex_dir.exists():
        pngs = sorted([p for p in ex_dir.iterdir() if p.suffix.lower() == ".png"])
        if pngs:
            lines.append("\n## BEV examples\n")
            for p in pngs:
                try:
                    lines.append(f"- `{p.relative_to(run_dir)}`")
                except Exception:
                    lines.append(f"- `{p}`")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "results_augmented_csv": str(out_csv),
                "summary_csv": str(summary_csv),
                "report": str(report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
