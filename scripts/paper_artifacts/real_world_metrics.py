#!/usr/bin/env python3
"""
Real-world NavFlow log metrics.

Given a single run directory like:
  local_logs/real_world_0129/task2_live_2026-01-28_234728_vlm-hold_MED-INT

This script parses:
  - telemetry/telemetry.jsonl
  - config.json (optional; for metadata)
  - final_report.json (optional; for sanity checks / back-compat)

and produces:
  - a one-row summary CSV (default: <run_dir>/real_world_metrics_summary.csv)
  - optional debug CSVs for "top divergence moments" and per-tick series

It is designed to support multiple policies:
  - vlm_hold / vlm_stream (executed path may be stale; measured via publish_waypoint vs current
  argmax)
  - vlm_hold_match / vlm_stream_match (matching discrepancy measured via vlm_sims_k)
  - score_fusion / prob_fusion (+ stream variants) (chosen-vs-argmax and matched-vs-argmax measured
  per planner_decision)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def _yaw_from_quat_xyzw(q_xyzw: Any) -> float:
    """Yaw (rad) from quaternion [x,y,z,w] (ROS convention)."""
    try:
        q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    except Exception:
        return 0.0
    if q.size != 4:
        return 0.0
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _pose_xy_yaw_from_pose_xyzw(pose_xyzw: Any) -> tuple[float, float, float] | None:
    """Pose array is expected as [x,y,z,qx,qy,qz,qw]."""
    try:
        p = np.asarray(pose_xyzw, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if p.size < 7:
        return None
    x = float(p[0])
    y = float(p[1])
    yaw = _yaw_from_quat_xyzw(p[3:7])
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
        return None
    return x, y, yaw


def _transform_points_prev_body_to_cur_body(
    pts_prev_body_xy: np.ndarray, *, prev_pose_xyzw: Any, cur_pose_xyzw: Any
) -> np.ndarray:
    """Transform points expressed in prev body frame -> current body frame (yaw-only SE(2))."""
    prev = _pose_xy_yaw_from_pose_xyzw(prev_pose_xyzw)
    cur = _pose_xy_yaw_from_pose_xyzw(cur_pose_xyzw)
    pts = np.asarray(pts_prev_body_xy, dtype=np.float64).reshape(-1, 2)
    if prev is None or cur is None or pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    xq, yq, yaw_q = prev
    xc, yc, yaw_c = cur
    cq, sq = float(math.cos(yaw_q)), float(math.sin(yaw_q))
    cc, sc = float(math.cos(yaw_c)), float(math.sin(yaw_c))
    # body(prev) -> world
    pw_x = cq * pts[:, 0] - sq * pts[:, 1] + xq
    pw_y = sq * pts[:, 0] + cq * pts[:, 1] + yq
    # world -> body(cur) (inverse rotation)
    dx = pw_x - xc
    dy = pw_y - yc
    pb_x = cc * dx + sc * dy
    pb_y = -sc * dx + cc * dy
    return np.stack([pb_x, pb_y], axis=1)


def _traj_row_to_body_xy(
    traj_row_xy: np.ndarray, *, downsample: int = 2, clip_x: tuple[float, float] = (0.0, 20.0)
) -> np.ndarray:
    """Match NavFlow convention: prepend origin, shift by one, clip x, downsample."""
    xy = np.asarray(traj_row_xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    zeros = np.zeros((1, 2), dtype=np.float64)
    xy2 = np.concatenate([zeros, xy[:-1]], axis=0)
    lo, hi = float(clip_x[0]), float(clip_x[1])
    xy2[:, 0] = np.clip(xy2[:, 0], lo, hi)
    ds = int(max(1, int(downsample)))
    return np.asarray(xy2[::ds], dtype=np.float64)


def _polyline_cumlen(poly_xy: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    seg = poly[1:] - poly[:-1]
    seg_l = np.linalg.norm(seg, axis=1)
    seg_l = np.asarray(seg_l, dtype=np.float64)
    return np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg_l)])


def _sample_polyline_uniform(poly_xy: np.ndarray, *, n: int) -> np.ndarray:
    """Sample n points uniformly in arclength (including endpoints)."""
    poly = np.asarray(poly_xy, dtype=np.float64).reshape(-1, 2)
    n = int(max(2, int(n)))
    if poly.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if poly.shape[0] == 1:
        return np.repeat(poly[:1], repeats=n, axis=0)
    s = _polyline_cumlen(poly)
    total = float(s[-1])
    if (not math.isfinite(total)) or total <= 1e-9:
        return np.repeat(poly[:1], repeats=n, axis=0)
    targets = np.linspace(0.0, total, n, dtype=np.float64)
    out = np.zeros((n, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < s.size and float(s[j + 1]) < float(t):
            j += 1
        if j + 1 >= s.size:
            out[i] = poly[-1]
            continue
        s0 = float(s[j])
        s1 = float(s[j + 1])
        a = poly[j]
        b = poly[j + 1]
        denom = float(max(1e-12, s1 - s0))
        u = float((t - s0) / denom)
        u = float(min(1.0, max(0.0, u)))
        out[i] = (1.0 - u) * a + u * b
    return out


def _ade_polyline(a_xy: np.ndarray, b_xy: np.ndarray, *, n: int = 20) -> float:
    a = np.asarray(a_xy, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b_xy, dtype=np.float64).reshape(-1, 2)
    if a.shape[0] < 2 or b.shape[0] < 2:
        return float("nan")
    aa = _sample_polyline_uniform(a, n=n)
    bb = _sample_polyline_uniform(b, n=n)
    d = aa - bb
    return float(np.mean(np.linalg.norm(d, axis=1)))


def _endpoint_dist(a_xy: np.ndarray, b_xy: np.ndarray, *, n: int = 20) -> float:
    a = np.asarray(a_xy, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b_xy, dtype=np.float64).reshape(-1, 2)
    if a.shape[0] < 1 or b.shape[0] < 1:
        return float("nan")
    aa = (
        _sample_polyline_uniform(a, n=max(2, int(n)))
        if a.shape[0] >= 2
        else np.repeat(a[:1], repeats=max(2, int(n)), axis=0)
    )
    bb = (
        _sample_polyline_uniform(b, n=max(2, int(n)))
        if b.shape[0] >= 2
        else np.repeat(b[:1], repeats=max(2, int(n)), axis=0)
    )
    return float(np.linalg.norm(aa[-1] - bb[-1]))


def _summ(xs: Iterable[float]) -> dict[str, float | int | None]:
    vals: list[float] = []
    for x in xs:
        try:
            v = float(x)
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        vals.append(v)
    if not vals:
        return {
            "count": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


@dataclass(frozen=True)
class PlannerDecision:
    t_wall_s: float
    snap_ms: int
    argmax_idx: int
    chosen_idx: int
    exec_src: str
    vlm_influenced: bool
    traj_k_ego_xy: np.ndarray | None  # [K,T,2]
    sims_k: np.ndarray | None  # [K]


@dataclass(frozen=True)
class PublishWaypoint:
    t_wall_s: float
    snap_ms: int
    source: str
    autonomy_enabled: bool | None
    local_pose_xyzw: list[float] | None
    waypoint_ego_xy: np.ndarray | None  # [N,2]


@dataclass(frozen=True)
class OdomTick:
    t_wall_s: float
    autonomy_enabled: bool | None
    local_pose_xyzw: list[float] | None


@dataclass(frozen=True)
class VLMResult:
    t_wall_s: float
    snap_ms: int
    action: str
    selected_index: int | None  # label shown in overlay; can be None for stop/invalid responses
    snap_base: str | None
    response_path: str | None


@dataclass(frozen=True)
class PlannerRequest:
    t_wall_s: float
    snap_ms: int
    snap_base: str | None
    prompt_path: str | None
    overlay_path: str | None
    request_path: str | None


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "config.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_final_report(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "final_report.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_telemetry(
    run_dir: Path,
) -> tuple[
    list[OdomTick],
    list[PlannerDecision],
    list[PublishWaypoint],
    list[VLMResult],
    list[PlannerRequest],
]:
    tel_path = run_dir / "telemetry" / "telemetry.jsonl"
    if not tel_path.exists():
        raise FileNotFoundError(f"Missing telemetry file: {tel_path}")

    odom: list[OdomTick] = []
    decisions: list[PlannerDecision] = []
    publishes: list[PublishWaypoint] = []
    vlm_results: list[VLMResult] = []
    planner_requests: list[PlannerRequest] = []

    for obj in _iter_jsonl(tel_path):
        ev = str(obj.get("event", "") or "")
        t_wall = _safe_float(obj.get("t_wall_s"))
        if t_wall is None:
            continue

        if ev == "odom_tick":
            autonomy = obj.get("autonomy_enabled", None)
            local_pose = obj.get("local_pose_xyzw", None)
            odom.append(
                OdomTick(
                    t_wall_s=float(t_wall),
                    autonomy_enabled=(bool(autonomy) if autonomy is not None else None),
                    local_pose_xyzw=local_pose,
                )
            )

        elif ev == "planner_decision":
            snap_ms = int(obj.get("snap_ms"))
            argmax_idx = int(obj.get("argmax_idx"))
            chosen_idx = int(obj.get("chosen_idx"))
            exec_src = str(obj.get("exec_src", "") or "")
            vlm_influenced = bool(obj.get("vlm_influenced", False))
            traj = obj.get("planner_traj_k_ego_xy", None)
            sims = obj.get("vlm_sims_k", None)
            traj_k = None
            sims_k = None
            try:
                if traj is not None:
                    traj_k = np.asarray(traj, dtype=np.float64)[..., :2]
                    if traj_k.ndim != 3:
                        traj_k = None
            except Exception:
                traj_k = None
            try:
                if sims is not None:
                    sims_k = np.asarray(sims, dtype=np.float64).reshape(-1)
            except Exception:
                sims_k = None
            decisions.append(
                PlannerDecision(
                    t_wall_s=float(t_wall),
                    snap_ms=int(snap_ms),
                    argmax_idx=int(argmax_idx),
                    chosen_idx=int(chosen_idx),
                    exec_src=exec_src,
                    vlm_influenced=bool(vlm_influenced),
                    traj_k_ego_xy=traj_k,
                    sims_k=sims_k,
                )
            )

        elif ev == "publish_waypoint":
            snap_ms = int(obj.get("snap_ms"))
            source = str(obj.get("source", "") or "")
            autonomy = obj.get("autonomy_enabled", None)
            local_pose = obj.get("local_pose_xyzw", None)
            wp = obj.get("waypoint_ego_xy", None)
            wp_xy = None
            try:
                if wp is not None:
                    wp_xy = np.asarray(wp, dtype=np.float64).reshape(-1, 2)
            except Exception:
                wp_xy = None
            publishes.append(
                PublishWaypoint(
                    t_wall_s=float(t_wall),
                    snap_ms=int(snap_ms),
                    source=source,
                    autonomy_enabled=(bool(autonomy) if autonomy is not None else None),
                    local_pose_xyzw=local_pose,
                    waypoint_ego_xy=wp_xy,
                )
            )

        elif ev == "vlm_result":
            snap_ms = int(obj.get("snap_ms"))
            action = str(obj.get("action", "") or "")
            sel = obj.get("selected_index", None)
            selected_index = None
            try:
                if sel is not None:
                    selected_index = int(sel)
            except Exception:
                selected_index = None
            vlm_results.append(
                VLMResult(
                    t_wall_s=float(t_wall),
                    snap_ms=int(snap_ms),
                    action=action,
                    selected_index=selected_index,
                    snap_base=(
                        str(obj.get("snap_base")) if obj.get("snap_base") is not None else None
                    ),
                    response_path=(
                        str(obj.get("response_path"))
                        if obj.get("response_path") is not None
                        else None
                    ),
                )
            )

        elif ev == "planner_request":
            snap_ms = int(obj.get("snap_ms"))
            planner_requests.append(
                PlannerRequest(
                    t_wall_s=float(t_wall),
                    snap_ms=int(snap_ms),
                    snap_base=(
                        str(obj.get("snap_base")) if obj.get("snap_base") is not None else None
                    ),
                    prompt_path=(
                        str(obj.get("prompt_path")) if obj.get("prompt_path") is not None else None
                    ),
                    overlay_path=(
                        str(obj.get("overlay_path"))
                        if obj.get("overlay_path") is not None
                        else None
                    ),
                    request_path=(
                        str(obj.get("request_path"))
                        if obj.get("request_path") is not None
                        else None
                    ),
                )
            )

    decisions.sort(key=lambda d: d.t_wall_s)
    publishes.sort(key=lambda p: p.t_wall_s)
    odom.sort(key=lambda o: o.t_wall_s)
    vlm_results.sort(key=lambda r: r.t_wall_s)
    planner_requests.sort(key=lambda r: r.t_wall_s)
    return odom, decisions, publishes, vlm_results, planner_requests


def _compute_takeover_metrics(odom: list[OdomTick]) -> dict[str, Any]:
    # Use odom_tick stream (pose + autonomy).
    ts: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    autos: list[bool] = []

    for o in odom:
        if o.local_pose_xyzw is None:
            continue
        pose = _pose_xy_yaw_from_pose_xyzw(o.local_pose_xyzw)
        if pose is None:
            continue
        x, y, _yaw = pose
        if o.autonomy_enabled is None:
            continue
        ts.append(float(o.t_wall_s))
        xs.append(float(x))
        ys.append(float(y))
        autos.append(bool(o.autonomy_enabled))

    if len(ts) < 3:
        return {
            "takeover_trimmed_ticks": 0,
            "takeover_trimmed_time_s": float("nan"),
            "takeover_trimmed_dist_m": float("nan"),
            "takeover_events": 0,
            "takeover_rate_per_1000_ticks": float("nan"),
            "takeover_rate_per_min": float("nan"),
            "takeover_rate_per_100m": float("nan"),
            "longest_autonomy_segment_dist_m": float("nan"),
            "longest_autonomy_segment_time_s": float("nan"),
        }

    t = np.asarray(ts, dtype=np.float64)
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    a = np.asarray(autos, dtype=bool)

    # Dist increments (aligned with t[1:])
    dx = np.diff(x)
    dy = np.diff(y)
    dd = np.sqrt(dx * dx + dy * dy)
    dd = np.asarray(dd, dtype=np.float64)
    dt = np.diff(t)
    dt = np.asarray(dt, dtype=np.float64)

    # Trim away beginning and ending takeover (autonomy==False) segments.
    true_idx = np.where(a)[0]
    if true_idx.size == 0:
        return {
            "takeover_trimmed_ticks": 0,
            "takeover_trimmed_time_s": float("nan"),
            "takeover_trimmed_dist_m": float("nan"),
            "takeover_events": 0,
            "takeover_rate_per_1000_ticks": float("nan"),
            "takeover_rate_per_min": float("nan"),
            "takeover_rate_per_100m": float("nan"),
            "longest_autonomy_segment_dist_m": float("nan"),
            "longest_autonomy_segment_time_s": float("nan"),
        }
    i0 = int(true_idx[0])
    i1 = int(true_idx[-1])
    if i1 <= i0 + 1:
        return {
            "takeover_trimmed_ticks": int(max(0, i1 - i0 + 1)),
            "takeover_trimmed_time_s": float(max(0.0, t[i1] - t[i0])),
            "takeover_trimmed_dist_m": float(np.sum(dd[i0:i1])),
            "takeover_events": 0,
            "takeover_rate_per_1000_ticks": 0.0,
            "takeover_rate_per_min": 0.0,
            "takeover_rate_per_100m": 0.0,
            "longest_autonomy_segment_dist_m": float(np.sum(dd[i0:i1])),
            "longest_autonomy_segment_time_s": float(max(0.0, t[i1] - t[i0])),
        }

    a_tr = a[i0 : i1 + 1]
    t_tr = t[i0 : i1 + 1]
    # Interval arrays between ticks (len = ticks-1) in trimmed window.
    dd_tr = dd[i0:i1]
    dt_tr = dt[i0:i1]

    total_ticks = int(a_tr.size)
    total_time_s = float(max(0.0, float(t_tr[-1] - t_tr[0])))
    total_dist_m = float(np.sum(dd_tr)) if dd_tr.size else 0.0

    # Takeover events: True->False transitions within trimmed window.
    # Count starts of takeover segments that are bounded inside [i0,i1].
    takeover_events = 0
    for i in range(1, int(a_tr.size)):
        if bool(a_tr[i - 1]) is True and bool(a_tr[i]) is False:
            takeover_events += 1

    # Interval attribution rule:
    # For interval (i-1 -> i), attribute distance/time to autonomy state at i (a_tr[i]).
    # This matches an "at this tick we are in autonomy" interpretation and avoids off-by-one
    # when autonomy toggles.
    interval_auto = np.asarray(a_tr[1:], dtype=bool)  # len = total_ticks-1
    auto_dist_m = float(np.sum(dd_tr[interval_auto])) if dd_tr.size else 0.0
    auto_time_s = float(np.sum(dt_tr[interval_auto])) if dt_tr.size else 0.0
    takeover_dist_m = float(max(0.0, total_dist_m - auto_dist_m))
    takeover_time_s = float(max(0.0, total_time_s - auto_time_s))
    takeover_frac_dist = float(takeover_dist_m / max(1e-9, total_dist_m))
    takeover_frac_time = float(takeover_time_s / max(1e-9, total_time_s))
    takeover_frac_ticks = float(np.mean(~interval_auto)) if interval_auto.size else float("nan")

    # Longest autonomy segment distance/time within trimmed window (on intervals).
    best_dist = 0.0
    best_time = 0.0
    cur_dist = 0.0
    cur_time = 0.0
    for i in range(int(interval_auto.size)):
        if bool(interval_auto[i]) is True:
            cur_dist += float(dd_tr[i]) if i < dd_tr.size else 0.0
            cur_time += float(dt_tr[i]) if i < dt_tr.size else 0.0
        else:
            best_dist = max(best_dist, cur_dist)
            best_time = max(best_time, cur_time)
            cur_dist = 0.0
            cur_time = 0.0
    best_dist = max(best_dist, cur_dist)
    best_time = max(best_time, cur_time)

    takeover_rate_per_1000_ticks = (float(takeover_events) / float(max(1, total_ticks))) * 1000.0
    takeover_rate_per_min = (float(takeover_events) / float(max(1e-9, total_time_s))) * 60.0
    takeover_rate_per_100m = (float(takeover_events) / float(max(1e-9, total_dist_m))) * 100.0

    return {
        "takeover_trimmed_ticks": int(total_ticks),
        "takeover_trimmed_time_s": float(total_time_s),
        "takeover_trimmed_dist_m": float(total_dist_m),
        "takeover_time_s": float(takeover_time_s),
        "takeover_dist_m": float(takeover_dist_m),
        "takeover_frac_time": float(takeover_frac_time),
        "takeover_frac_dist": float(takeover_frac_dist),
        "takeover_frac_ticks": float(takeover_frac_ticks),
        "takeover_events": int(takeover_events),
        "takeover_rate_per_1000_ticks": float(takeover_rate_per_1000_ticks),
        "takeover_rate_per_min": float(takeover_rate_per_min),
        "takeover_rate_per_100m": float(takeover_rate_per_100m),
        "longest_autonomy_segment_dist_m": float(best_dist),
        "longest_autonomy_segment_time_s": float(best_time),
    }


def _index_decisions_by_snap(decisions: list[PlannerDecision]) -> dict[int, PlannerDecision]:
    # If duplicates exist, keep the latest by wall time (should be unique per snap).
    out: dict[int, PlannerDecision] = {}
    for d in decisions:
        out[int(d.snap_ms)] = d
    return out


def _compute_planner_decision_discrepancies(decisions: list[PlannerDecision]) -> dict[str, Any]:
    chosen_vs_argmax_ade: list[float] = []
    chosen_vs_argmax_end: list[float] = []
    matched_vs_argmax_ade: list[float] = []
    matched_vs_argmax_end: list[float] = []
    eq_all: list[bool] = []
    eq_with_vlm_ref: list[bool] = []
    vlm_ref_count = 0

    for d in decisions:
        if d.traj_k_ego_xy is None:
            continue
        K = int(d.traj_k_ego_xy.shape[0])
        if K <= 0:
            continue
        ai = int(np.clip(d.argmax_idx, 0, K - 1))
        ci = int(np.clip(d.chosen_idx, 0, K - 1))
        eq_all.append(bool(ci == ai))
        a_traj = d.traj_k_ego_xy[ai]
        c_traj = d.traj_k_ego_xy[ci]
        chosen_vs_argmax_ade.append(_ade_polyline(c_traj, a_traj, n=20))
        chosen_vs_argmax_end.append(_endpoint_dist(c_traj, a_traj, n=20))

        has_vlm_ref = d.sims_k is not None and d.sims_k.size == K and np.isfinite(d.sims_k).any()
        if has_vlm_ref:
            vlm_ref_count += 1
            eq_with_vlm_ref.append(bool(ci == ai))
            mi = int(np.argmax(d.sims_k))
            m_traj = d.traj_k_ego_xy[int(mi)]
            matched_vs_argmax_ade.append(_ade_polyline(m_traj, a_traj, n=20))
            matched_vs_argmax_end.append(_endpoint_dist(m_traj, a_traj, n=20))

    s1 = _summ(chosen_vs_argmax_ade)
    s2 = _summ(chosen_vs_argmax_end)
    s3 = _summ(matched_vs_argmax_ade)
    s4 = _summ(matched_vs_argmax_end)
    eq_all_frac = float(np.mean(eq_all)) if eq_all else float("nan")
    eq_with_vlm_ref_frac = float(np.mean(eq_with_vlm_ref)) if eq_with_vlm_ref else float("nan")
    return {
        # Agreement rates (requested “accuracies”)
        "chosen_equals_argmax_frac_all": eq_all_frac,
        "chosen_equals_argmax_frac_when_vlm_ref": eq_with_vlm_ref_frac,
        "chosen_equals_argmax_count_when_vlm_ref": int(vlm_ref_count),
        "chosen_vs_argmax_ade_m_mean": s1["mean"],
        "chosen_vs_argmax_ade_m_p90": s1["p90"],
        "chosen_vs_argmax_ade_m_max": s1["max"],
        "chosen_vs_argmax_end_m_mean": s2["mean"],
        "chosen_vs_argmax_end_m_p90": s2["p90"],
        "chosen_vs_argmax_end_m_max": s2["max"],
        "vlm_matched_vs_argmax_ade_m_mean": s3["mean"],
        "vlm_matched_vs_argmax_ade_m_p90": s3["p90"],
        "vlm_matched_vs_argmax_ade_m_max": s3["max"],
        "vlm_matched_vs_argmax_end_m_mean": s4["mean"],
        "vlm_matched_vs_argmax_end_m_p90": s4["p90"],
        "vlm_matched_vs_argmax_end_m_max": s4["max"],
    }


def _compute_vlm_vs_argmax(
    run_dir: Path,
    decisions: list[PlannerDecision],
    vlm_results: list[VLMResult],
    planner_requests: list[PlannerRequest],
) -> tuple[dict[str, Any], pd.DataFrame]:
    dec_by_snap = _index_decisions_by_snap(decisions)
    req_by_snap: dict[int, PlannerRequest] = {}
    for r in planner_requests:
        req_by_snap[int(r.snap_ms)] = r

    def _prompt_row_to_label(
        run_dir: Path, pr: PlannerRequest, *, expected_k: int | None
    ) -> tuple[list[int], dict[int, int]]:
        # Parse the candidate label list from prompts/<snap>.json.
        # The prompt explicitly states: "Ordering: index ascending".
        if pr.prompt_path is None:
            return [], {}
        p = Path(pr.prompt_path)
        p = (run_dir / p) if not p.is_absolute() else p
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return [], {}
        user = obj.get("user", "")
        if not isinstance(user, str):
            return [], {}
        labels: list[int] = []
        for line in user.splitlines():
            # Table lines look like: "   33 |        6.1m | (6.2, -0.5)"
            m = re.match(r"^\s*(\d+)\s*\|", line)
            if not m:
                continue
            try:
                labels.append(int(m.group(1)))
            except Exception:
                continue
        if expected_k is not None and labels and int(len(labels)) != int(expected_k):
            # Still return what we got (some prompts may change formatting), but allow caller to
            # skip if needed.
            pass
        label_to_row: dict[int, int] = {int(lbl): int(i) for i, lbl in enumerate(labels)}
        return labels, label_to_row

    prompt_cache: dict[
        int, tuple[list[int], dict[int, int], str | None, str | None, str | None]
    ] = {}

    rows: list[dict[str, Any]] = []
    ades: list[float] = []
    ends: list[float] = []
    lats: list[float] = []
    agree: list[bool] = []
    total_results = 0
    select_results = 0
    stop_results = 0

    for vr in vlm_results:
        total_results += 1
        if str(vr.action) == "stop" or vr.selected_index is None:
            stop_results += 1
            continue
        select_results += 1
        d = dec_by_snap.get(int(vr.snap_ms))
        if d is None or d.traj_k_ego_xy is None:
            continue
        K = int(d.traj_k_ego_xy.shape[0])
        if K <= 0:
            continue
        ai = int(np.clip(d.argmax_idx, 0, K - 1))
        # VLM returns the *label shown in the overlay*, not the 0..K-1 row index.
        pr = req_by_snap.get(int(vr.snap_ms))
        if pr is None:
            continue
        if int(vr.snap_ms) not in prompt_cache:
            labels, label_to_row = _prompt_row_to_label(run_dir=run_dir, pr=pr, expected_k=K)
            prompt_cache[int(vr.snap_ms)] = (
                labels,
                label_to_row,
                pr.prompt_path,
                pr.overlay_path,
                pr.request_path,
            )
        labels, label_to_row, prompt_path, overlay_path, request_path = prompt_cache[
            int(vr.snap_ms)
        ]
        si = label_to_row.get(int(vr.selected_index), None)
        if si is None:
            continue
        si = int(np.clip(int(si), 0, K - 1))
        agree.append(bool(si == ai))
        a_traj = d.traj_k_ego_xy[ai]
        s_traj = d.traj_k_ego_xy[si]
        ade = _ade_polyline(s_traj, a_traj, n=20)
        end = _endpoint_dist(s_traj, a_traj, n=20)
        ades.append(ade)
        ends.append(end)

        lat = float("nan")
        lat = float(max(0.0, float(vr.t_wall_s) - float(pr.t_wall_s)))
        lats.append(lat)

        argmax_label = None
        try:
            if labels and 0 <= ai < len(labels):
                argmax_label = int(labels[ai])
        except Exception:
            argmax_label = None

        rows.append(
            {
                "snap_ms": int(vr.snap_ms),
                "snap_base": vr.snap_base,
                "vlm_action": str(vr.action),
                "vlm_selected_label": int(vr.selected_index),
                "vlm_selected_row": int(si),
                "planner_argmax_row": int(d.argmax_idx),
                "planner_argmax_label": argmax_label,
                "vlm_vs_argmax_ade_m": float(ade),
                "vlm_vs_argmax_end_m": float(end),
                "vlm_latency_s": float(lat),
                "response_path": vr.response_path,
                "prompt_path": prompt_path,
                "overlay_path": overlay_path,
                "request_path": request_path,
            }
        )

    df = pd.DataFrame(rows)
    s1 = _summ(ades)
    s2 = _summ(ends)
    s3 = _summ(lats)
    agree_frac = float(np.mean(agree)) if agree else float("nan")
    return (
        {
            "vlm_results_total": int(total_results),
            "vlm_results_select": int(select_results),
            "vlm_results_stop_or_invalid": int(stop_results),
            "vlm_vs_argmax_count": int(s1["count"]),
            "vlm_selected_equals_argmax_frac": float(agree_frac),
            "vlm_vs_argmax_ade_m_mean": s1["mean"],
            "vlm_vs_argmax_ade_m_p90": s1["p90"],
            "vlm_vs_argmax_ade_m_max": s1["max"],
            "vlm_vs_argmax_end_m_mean": s2["mean"],
            "vlm_vs_argmax_end_m_p90": s2["p90"],
            "vlm_vs_argmax_end_m_max": s2["max"],
            "vlm_latency_s_mean": s3["mean"],
            "vlm_latency_s_p90": s3["p90"],
            "vlm_latency_s_max": s3["max"],
        },
        df,
    )


def _compute_publish_vs_current_argmax(
    decisions: list[PlannerDecision], publishes: list[PublishWaypoint]
) -> tuple[dict[str, Any], pd.DataFrame]:
    # For each publish tick, align to latest planner decision at/before that time.
    if not decisions or not publishes:
        return (
            {
                "publish_stale_s_mean": float("nan"),
                "publish_stale_s_p90": float("nan"),
                "publish_vs_argmax_ade_m_mean": float("nan"),
                "publish_vs_argmax_ade_m_p90": float("nan"),
                "publish_vs_argmax_end_m_mean": float("nan"),
                "publish_vs_argmax_end_m_p90": float("nan"),
            },
            pd.DataFrame([]),
        )

    decs = sorted(decisions, key=lambda d: d.t_wall_s)
    idx = 0
    rows: list[dict[str, Any]] = []
    stales: list[float] = []
    ades: list[float] = []
    ends: list[float] = []

    for p in publishes:
        while idx + 1 < len(decs) and float(decs[idx + 1].t_wall_s) <= float(p.t_wall_s):
            idx += 1
        d = decs[idx]
        if d.traj_k_ego_xy is None or p.waypoint_ego_xy is None or p.waypoint_ego_xy.shape[0] < 2:
            continue
        K = int(d.traj_k_ego_xy.shape[0])
        if K <= 0:
            continue
        ai = int(np.clip(d.argmax_idx, 0, K - 1))
        a_traj_raw = d.traj_k_ego_xy[ai]
        a_body = _traj_row_to_body_xy(a_traj_raw, downsample=2, clip_x=(0.0, 20.0))
        if a_body.shape[0] < 2:
            continue

        ade = _ade_polyline(p.waypoint_ego_xy, a_body, n=10)
        end = _endpoint_dist(p.waypoint_ego_xy, a_body, n=10)
        stale = float(max(0.0, float(p.t_wall_s) - float(p.snap_ms) / 1000.0))
        stales.append(stale)
        ades.append(ade)
        ends.append(end)
        rows.append(
            {
                "t_wall_s": float(p.t_wall_s),
                "publish_snap_ms": int(p.snap_ms),
                "publish_source": str(p.source),
                "autonomy_enabled": (
                    bool(p.autonomy_enabled) if p.autonomy_enabled is not None else None
                ),
                "publish_stale_s": float(stale),
                "planner_snap_ms": int(d.snap_ms),
                "planner_argmax_idx": int(d.argmax_idx),
                "publish_vs_argmax_ade_m": float(ade),
                "publish_vs_argmax_end_m": float(end),
            }
        )

    df = pd.DataFrame(rows)
    s0 = _summ(stales)
    s1 = _summ(ades)
    s2 = _summ(ends)
    return (
        {
            "publish_count_aligned": int(s1["count"]),
            "publish_stale_s_mean": s0["mean"],
            "publish_stale_s_p90": s0["p90"],
            "publish_stale_s_max": s0["max"],
            "publish_vs_argmax_ade_m_mean": s1["mean"],
            "publish_vs_argmax_ade_m_p90": s1["p90"],
            "publish_vs_argmax_ade_m_max": s1["max"],
            "publish_vs_argmax_end_m_mean": s2["mean"],
            "publish_vs_argmax_end_m_p90": s2["p90"],
            "publish_vs_argmax_end_m_max": s2["max"],
        },
        df,
    )


def _compute_temporal_consistency(
    publishes: list[PublishWaypoint],
) -> tuple[dict[str, Any], pd.DataFrame]:
    # Temporal consistency of the *executed* trajectory stream (publish_waypoint).
    # We compare consecutive published waypoint polylines after transforming prev polyline into
    # current body frame.
    ps = [
        p
        for p in publishes
        if p.waypoint_ego_xy is not None
        and p.waypoint_ego_xy.shape[0] >= 2
        and p.local_pose_xyzw is not None
    ]
    if len(ps) < 2:
        return (
            {
                "temporal_consistency_ade_m_mean": float("nan"),
                "temporal_consistency_ade_m_p90": float("nan"),
                "temporal_consistency_end_m_mean": float("nan"),
                "temporal_consistency_end_m_p90": float("nan"),
            },
            pd.DataFrame([]),
        )

    rows: list[dict[str, Any]] = []
    ades: list[float] = []
    ends: list[float] = []

    for prev, cur in zip(ps[:-1], ps[1:], strict=False):
        prev_xy = np.asarray(prev.waypoint_ego_xy, dtype=np.float64).reshape(-1, 2)
        cur_xy = np.asarray(cur.waypoint_ego_xy, dtype=np.float64).reshape(-1, 2)
        prev_in_cur = _transform_points_prev_body_to_cur_body(
            prev_xy, prev_pose_xyzw=prev.local_pose_xyzw, cur_pose_xyzw=cur.local_pose_xyzw
        )
        if prev_in_cur.shape[0] < 2:
            continue
        ade = _ade_polyline(cur_xy, prev_in_cur, n=10)
        end = _endpoint_dist(cur_xy, prev_in_cur, n=10)
        ades.append(ade)
        ends.append(end)
        rows.append(
            {
                "t_wall_s": float(cur.t_wall_s),
                "autonomy_enabled": (
                    bool(cur.autonomy_enabled) if cur.autonomy_enabled is not None else None
                ),
                "temporal_consistency_ade_m": float(ade),
                "temporal_consistency_end_m": float(end),
            }
        )

    df = pd.DataFrame(rows)
    s1 = _summ(ades)
    s2 = _summ(ends)
    return (
        {
            "temporal_consistency_count": int(s1["count"]),
            "temporal_consistency_ade_m_mean": s1["mean"],
            "temporal_consistency_ade_m_p90": s1["p90"],
            "temporal_consistency_ade_m_max": s1["max"],
            "temporal_consistency_end_m_mean": s2["mean"],
            "temporal_consistency_end_m_p90": s2["p90"],
            "temporal_consistency_end_m_max": s2["max"],
        },
        df,
    )


def _write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stable header ordering: sort keys.
    keys = sorted(row.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerow({k: row.get(k, None) for k in keys})


def analyze_run(
    run_dir: Path, *, out_dir: Path | None = None, topk: int = 10, write_debug: bool = True
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    if out_dir is None:
        out_dir = run_dir

    cfg = _load_run_config(run_dir)
    rep = _load_final_report(run_dir)
    odom, decisions, publishes, vlm_results, planner_requests = _parse_telemetry(run_dir)

    # Metadata
    control_cfg = (cfg.get("control") or {}) if isinstance(cfg.get("control"), dict) else {}
    vlm_cfg = (cfg.get("vlm") or {}) if isinstance(cfg.get("vlm"), dict) else {}
    worker_cfg = (cfg.get("vlm_worker") or {}) if isinstance(cfg.get("vlm_worker"), dict) else {}
    logging_cfg = (cfg.get("logging") or {}) if isinstance(cfg.get("logging"), dict) else {}

    policy = str(control_cfg.get("policy") or "")
    vlm_provider = str(vlm_cfg.get("provider") or "")
    exp_name = str(logging_cfg.get("exp_name") or run_dir.name)

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "exp_name": exp_name,
        "policy": policy,
        "vlm_provider": vlm_provider,
        "control_frequency_hz": control_cfg.get("frequency_hz", None),
        "vlm_query_hz_config": control_cfg.get("vlm_query_hz", None),
        "vlm_max_in_flight": worker_cfg.get("max_in_flight", None),
        "fusion_lambda_config": control_cfg.get("fusion_lambda", None),
        "fusion_tau_s_config": control_cfg.get("fusion_tau_s", None),
        "fusion_dist_scale_m_config": control_cfg.get("fusion_dist_scale_m", None),
        "fusion_similarity_mode": control_cfg.get("fusion_similarity_mode", None),
        "planner_ticks_final_report": rep.get("planner_ticks", None),
        "publish_ticks_final_report": (rep.get("counts") or {}).get("publish_ticks", None),
        "travel_distance_m_final_report": rep.get("travel_distance_m", None),
        "total_time_s_final_report": rep.get("total_time_s", None),
        "vlm_calls_submitted_final_report": rep.get("vlm_calls_submitted", None),
        "vlm_results_received_final_report": rep.get("vlm_results_received", None),
        "vlm_avg_dt_s_final_report": rep.get("vlm_avg_dt_s", None),
        "vlm_avg_interval_s_final_report": rep.get("vlm_avg_interval_s", None),
        "auto_distance_m_final_report": (rep.get("autonomy") or {}).get("auto_distance_m", None),
        "auto_distance_frac_final_report": (rep.get("autonomy") or {}).get(
            "auto_distance_frac", None
        ),
        "auto_time_wall_s_final_report": (rep.get("autonomy") or {}).get("auto_time_wall_s", None),
    }

    # Takeover metrics (odom-based; trimmed)
    row.update(_compute_takeover_metrics(odom))

    # (3) chosen vs argmax at same step; (4) matched vs argmax at same step (when sims available)
    row.update(_compute_planner_decision_discrepancies(decisions))

    # (5) VLM selected vs "at that time" argmax (query-time snap)
    vlm_stats, df_vlm = _compute_vlm_vs_argmax(run_dir, decisions, vlm_results, planner_requests)
    row.update(vlm_stats)

    # Execution-vs-current discrepancy (important for vlm_hold / vlm_stream delay story)
    pub_stats, df_pub = _compute_publish_vs_current_argmax(decisions, publishes)
    row.update(pub_stats)

    # (2) temporal consistency of the executed waypoint stream
    tc_stats, df_tc = _compute_temporal_consistency(publishes)
    row.update(tc_stats)

    # "Most divergent moments"
    if write_debug:
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # VLM vs argmax divergence moments (query-time)
        if not df_vlm.empty:
            df_vlm_sorted = df_vlm.sort_values(
                ["vlm_vs_argmax_end_m", "vlm_vs_argmax_ade_m"], ascending=[False, False]
            )
            df_vlm_sorted.head(int(max(1, topk))).to_csv(
                out_dir / "real_world_top_vlm_divergence.csv", index=False
            )
            df_vlm.to_csv(out_dir / "real_world_vlm_vs_argmax_series.csv", index=False)
            try:
                top = df_vlm_sorted.iloc[0].to_dict()
                row["top_vlm_divergence_snap_ms"] = int(top.get("snap_ms"))
                row["top_vlm_divergence_end_m"] = float(top.get("vlm_vs_argmax_end_m"))
                row["top_vlm_divergence_ade_m"] = float(top.get("vlm_vs_argmax_ade_m"))
                row["top_vlm_divergence_snap_base"] = top.get("snap_base")
            except Exception:
                pass

        # Published-vs-current-argmax divergence moments (execution-level)
        if not df_pub.empty:
            df_pub_sorted = df_pub.sort_values(
                ["publish_vs_argmax_end_m", "publish_vs_argmax_ade_m"], ascending=[False, False]
            )
            df_pub_sorted.head(int(max(1, topk))).to_csv(
                out_dir / "real_world_top_publish_divergence.csv", index=False
            )
            df_pub.to_csv(out_dir / "real_world_publish_vs_argmax_series.csv", index=False)
            try:
                top = df_pub_sorted.iloc[0].to_dict()
                row["top_publish_divergence_t_wall_s"] = float(top.get("t_wall_s"))
                row["top_publish_divergence_publish_snap_ms"] = int(top.get("publish_snap_ms"))
                row["top_publish_divergence_end_m"] = float(top.get("publish_vs_argmax_end_m"))
                row["top_publish_divergence_ade_m"] = float(top.get("publish_vs_argmax_ade_m"))
                row["top_publish_divergence_publish_source"] = top.get("publish_source")
                row["top_publish_divergence_planner_snap_ms"] = int(top.get("planner_snap_ms"))
            except Exception:
                pass

        # Optional per-tick series
        if not df_tc.empty:
            df_tc.to_csv(out_dir / "real_world_temporal_consistency_series.csv", index=False)

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "run_dirs",
        nargs="+",
        help=(
            "One or more real-world run directories (each containing telemetry/telemetry.jsonl), "
            "or a root directory (e.g. local_logs/real_world_0129) from which runs will be "
            "discovered."
        ),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. If multiple runs are given, this will be a multi-row CSV. Default: "
        "<first_run>/real_world_metrics_summary.csv",
    )
    ap.add_argument(
        "--topk", type=int, default=10, help="Top-K divergence moments to write per run."
    )
    ap.add_argument("--no-debug", action="store_true", help="Disable writing per-run debug CSVs.")
    args = ap.parse_args()

    def _looks_like_run_dir(p: Path) -> bool:
        return (p / "telemetry" / "telemetry.jsonl").exists()

    def _discover_runs(root: Path) -> list[Path]:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            return []
        tel_paths = list(root.rglob("telemetry.jsonl"))
        runs: set[Path] = set()
        for tp in tel_paths:
            # expected: <run_dir>/telemetry/telemetry.jsonl
            try:
                if tp.parent.name != "telemetry":
                    continue
                run_dir = tp.parent.parent
            except Exception:
                continue
            if _looks_like_run_dir(run_dir):
                runs.add(run_dir.resolve())
        return sorted(runs, key=lambda p: str(p))

    # Expand inputs into actual run directories.
    runs: list[Path] = []
    for raw in args.run_dirs:
        p = Path(raw).expanduser().resolve()
        if _looks_like_run_dir(p):
            runs.append(p)
            continue
        # Treat as root to discover runs.
        discovered = _discover_runs(p)
        runs.extend(discovered)
    # Dedup while preserving order.
    seen: set[Path] = set()
    runs2: list[Path] = []
    for r in runs:
        rr = r.resolve()
        if rr in seen:
            continue
        seen.add(rr)
        runs2.append(rr)
    runs = runs2
    if not runs:
        raise SystemExit("No runs found. Expected telemetry/telemetry.jsonl under the given paths.")

    out = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (runs[0] / "real_world_metrics_summary.csv")
    )

    rows: list[dict[str, Any]] = []
    for r in runs:
        rows.append(
            analyze_run(r, out_dir=r, topk=int(args.topk), write_debug=(not bool(args.no_debug)))
        )

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote: {out} ({len(rows)} row(s))")


if __name__ == "__main__":
    main()
