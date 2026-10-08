from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from slow_brain_fast_planner.benchmarks.dataset import EpisodeLoadResult, load_episode
from slow_brain_fast_planner.utils.trajectory_utils import (
    GoalFilterConfig,
    GtFutureFilterConfig,
    extract_goal_info,
    goal_filter_reason,
    gt_future_filter_reason,
)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def filter_takeover_clips_jsonl_by_goal(
    *,
    dataset_root: Path,
    takeover_clips_jsonl_path: Path,
    cfg: GoalFilterConfig | None = None,
    overwrite: bool = True,
    keep_backup: bool = True,
) -> dict[str, Any]:
    """Filter takeover clips whose t0 has a "weird" goal.

    This post-processes `takeover_clips.jsonl` (written by `build_takeover_clips`) by:
      - loading the episode planner record at t0
      - extracting (goal_xy, dist, bearing)
      - applying GoalFilterConfig heuristics

    If overwrite=True, we replace the file in-place (optionally leaving a .unfiltered backup).
    """
    cfg = cfg or GoalFilterConfig()
    dataset_root = Path(dataset_root).resolve()
    takeover_clips_jsonl_path = Path(takeover_clips_jsonl_path).resolve()
    if not takeover_clips_jsonl_path.exists():
        raise FileNotFoundError(f"takeover_clips.jsonl not found: {takeover_clips_jsonl_path}")

    raw_lines = takeover_clips_jsonl_path.read_text(encoding="utf-8").splitlines()
    clips: list[dict[str, Any]] = []
    parse_errors = 0
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            parse_errors += 1
            continue
        if isinstance(obj, dict):
            clips.append(obj)

    # Episode cache
    ep_cache: dict[str, tuple[EpisodeLoadResult, dict[float, int]]] = {}

    def _get_ep_and_index(eid: str) -> tuple[EpisodeLoadResult, dict[float, int]] | None:
        hit = ep_cache.get(str(eid))
        if hit is not None:
            return hit
        # Find metadata path: assume canonical layout episodes/<eid>/episode.json
        meta = (dataset_root / "episodes" / str(eid) / "episode.json").resolve()
        if not meta.exists():
            # Fallback: episodes/<eid>.json
            meta = (dataset_root / "episodes" / f"{str(eid)}.json").resolve()
        if not meta.exists():
            return None
        ep = load_episode(dataset_root, meta)
        t_to_index: dict[float, int] = {}
        for i, r in enumerate(ep.planner_candidates):
            t_to_index[round(float(r.t), 6)] = int(i)
        ep_cache[str(eid)] = (ep, t_to_index)
        return ep, t_to_index

    kept: list[dict[str, Any]] = []
    filtered = 0
    filtered_reasons: dict[str, int] = {}
    missing_ep = 0
    missing_t0 = 0

    for rec in clips:
        eid = rec.get("episode_id")
        t0 = rec.get("t0")
        if not isinstance(eid, str) or not isinstance(t0, (int, float)):
            # Keep unknown shape (don't destroy data silently).
            kept.append(rec)
            continue

        ep_hit = _get_ep_and_index(str(eid))
        if ep_hit is None:
            missing_ep += 1
            kept.append(rec)
            continue
        ep, t_to_index = ep_hit
        idx = t_to_index.get(round(float(t0), 6))
        if idx is None or (not ep.planner_candidates):
            missing_t0 += 1
            kept.append(rec)
            continue

        pc_rec = ep.planner_candidates[int(idx)]
        goal_xy, goal_distance_m, goal_bearing_deg = extract_goal_info(pc_rec)
        reason = goal_filter_reason(goal_xy, goal_distance_m, goal_bearing_deg, cfg=cfg)
        if reason is not None:
            filtered += 1
            filtered_reasons[str(reason)] = filtered_reasons.get(str(reason), 0) + 1
            continue
        kept.append(rec)

    summary = {
        "enabled": bool(cfg.enabled) if cfg is not None else False,
        "cfg": (cfg.__dict__ if cfg is not None else None),
        "clips_in": int(len(clips)),
        "clips_out": int(len(kept)),
        "clips_filtered": int(filtered),
        "clips_filtered_reasons": {k: int(v) for k, v in sorted(filtered_reasons.items())},
        "clips_parse_errors": int(parse_errors),
        "missing_episode_meta": int(missing_ep),
        "missing_t0_in_planner_candidates": int(missing_t0),
        "path": str(takeover_clips_jsonl_path),
    }

    if not overwrite:
        return summary

    if keep_backup:
        backup = takeover_clips_jsonl_path.with_suffix(
            takeover_clips_jsonl_path.suffix + ".unfiltered"
        )
        if not backup.exists():
            _atomic_write_text(backup, "\n".join(raw_lines) + ("\n" if raw_lines else ""))

    out_text = "\n".join([json.dumps(x, sort_keys=True) for x in kept]) + ("\n" if kept else "")
    _atomic_write_text(takeover_clips_jsonl_path, out_text)
    return summary


def _gt_future_local_from_episode(
    ep: EpisodeLoadResult,
    *,
    t0: float,
    num_points: int,
    traj_dt_s: float,
) -> list[list[float]] | None:
    """Compute GT future in robot local frame at time t0 using canonical odom stream."""
    if not getattr(ep, "odom", None):
        return None
    od = getattr(ep, "odom", []) or []
    try:
        t_arr = [float(r.t) for r in od]
        xy_arr = [(float(r.x), float(r.y)) for r in od]
        yaw_arr = [float(r.yaw) for r in od]
    except Exception:
        return None
    if len(t_arr) < 2 or len(xy_arr) != len(t_arr) or len(yaw_arr) != len(t_arr):
        return None

    def _interp(times: list[float], vals: list[tuple[float, float]] | list[float], tt: float):
        # Linear interpolation (best-effort).
        if not times:
            return None
        if tt <= times[0]:
            return vals[0]
        if tt >= times[-1]:
            return vals[-1]
        import bisect

        i = bisect.bisect_right(times, tt)
        i0 = max(0, i - 1)
        i1 = min(len(times) - 1, i)
        t0i = float(times[i0])
        t1i = float(times[i1])
        if t1i <= t0i + 1e-12:
            return vals[i0]
        a = (float(tt) - t0i) / (t1i - t0i)
        v0 = vals[i0]
        v1 = vals[i1]
        if isinstance(v0, tuple):
            return (
                float(v0[0]) * (1 - a) + float(v1[0]) * a,
                float(v0[1]) * (1 - a) + float(v1[1]) * a,
            )
        return float(v0) * (1 - a) + float(v1) * a

    curr_xy = _interp(t_arr, xy_arr, float(t0))
    curr_yaw = _interp(t_arr, yaw_arr, float(t0))
    if curr_xy is None or curr_yaw is None:
        return None

    c = float(math.cos(float(curr_yaw)))
    s = float(math.sin(float(curr_yaw)))
    out: list[list[float]] = []
    for k in range(1, int(num_points) + 1):
        tt = float(t0) + float(k) * float(traj_dt_s)
        fut_xy = _interp(t_arr, xy_arr, float(tt))
        if fut_xy is None:
            return None
        dx = float(fut_xy[0] - float(curr_xy[0]))
        dy = float(fut_xy[1] - float(curr_xy[1]))
        # world -> robot: x=fwd, y=left
        x = c * dx + s * dy
        y = -s * dx + c * dy
        out.append([float(x), float(y)])
    return out


def filter_takeover_clips_jsonl_by_quality(
    *,
    dataset_root: Path,
    takeover_clips_jsonl_path: Path,
    goal_cfg: GoalFilterConfig | None = None,
    gt_cfg: GtFutureFilterConfig | None = None,
    traj_min_dist_m: float | None = None,
    traj_index: int = 0,
    traj_dt_s: float = 0.2,
    fallback_num_points: int = 10,
    overwrite: bool = True,
    keep_backup: bool = True,
) -> dict[str, Any]:
    """Filter takeover clips by goal sanity and/or GT-future sanity."""
    dataset_root = Path(dataset_root).resolve()
    takeover_clips_jsonl_path = Path(takeover_clips_jsonl_path).resolve()
    if not takeover_clips_jsonl_path.exists():
        raise FileNotFoundError(f"takeover_clips.jsonl not found: {takeover_clips_jsonl_path}")

    raw_lines = takeover_clips_jsonl_path.read_text(encoding="utf-8").splitlines()
    clips: list[dict[str, Any]] = []
    parse_errors = 0
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            parse_errors += 1
            continue
        if isinstance(obj, dict):
            clips.append(obj)

    goal_cfg = goal_cfg or GoalFilterConfig()
    gt_cfg = gt_cfg or GtFutureFilterConfig()

    # Episode cache
    ep_cache: dict[str, tuple[EpisodeLoadResult, dict[float, int]]] = {}

    def _get_ep_and_index(eid: str) -> tuple[EpisodeLoadResult, dict[float, int]] | None:
        hit = ep_cache.get(str(eid))
        if hit is not None:
            return hit
        meta = (dataset_root / "episodes" / str(eid) / "episode.json").resolve()
        if not meta.exists():
            meta = (dataset_root / "episodes" / f"{str(eid)}.json").resolve()
        if not meta.exists():
            return None
        ep = load_episode(dataset_root, meta)
        t_to_index: dict[float, int] = {}
        for i, r in enumerate(ep.planner_candidates):
            t_to_index[round(float(r.t), 6)] = int(i)
        ep_cache[str(eid)] = (ep, t_to_index)
        return ep, t_to_index

    kept: list[dict[str, Any]] = []
    filtered = 0
    filtered_reasons: dict[str, int] = {}
    missing_ep = 0
    missing_t0 = 0

    def _traj_dist_m_from_pc(pc_rec: Any, *, index: int) -> float | None:
        """Best-effort path length from candidate points_xy."""
        try:
            cands = getattr(pc_rec, "candidates", None)
            if cands is None and isinstance(pc_rec, dict):
                cands = pc_rec.get("candidates")
            if not isinstance(cands, list) or not cands:
                return None
            ii = int(index)
            if ii < 0 or ii >= len(cands):
                return None
            cand = cands[ii]
            pts = getattr(cand, "points_xy", None)
            if pts is None and isinstance(cand, dict):
                pts = cand.get("points_xy")
            if not isinstance(pts, list) or len(pts) < 2:
                return 0.0 if isinstance(pts, list) else None
            total = 0.0
            prev = None
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x = float(p[0])
                y = float(p[1])
                if not (math.isfinite(x) and math.isfinite(y)):
                    return None
                if prev is not None:
                    dx = x - prev[0]
                    dy = y - prev[1]
                    total += float((dx * dx + dy * dy) ** 0.5)
                prev = (x, y)
            return float(total)
        except Exception:
            return None

    for rec in clips:
        eid = rec.get("episode_id")
        t0 = rec.get("t0")
        if not isinstance(eid, str) or not isinstance(t0, (int, float)):
            kept.append(rec)
            continue

        ep_hit = _get_ep_and_index(str(eid))
        if ep_hit is None:
            missing_ep += 1
            kept.append(rec)
            continue
        ep, t_to_index = ep_hit
        idx = t_to_index.get(round(float(t0), 6))
        if idx is None or (not ep.planner_candidates):
            missing_t0 += 1
            kept.append(rec)
            continue

        pc_rec = ep.planner_candidates[int(idx)]

        # Goal sanity
        gxy, gd, gb = extract_goal_info(pc_rec)
        r_goal = goal_filter_reason(gxy, gd, gb, cfg=goal_cfg)
        if r_goal is not None:
            filtered += 1
            filtered_reasons[f"goal:{r_goal}"] = filtered_reasons.get(f"goal:{r_goal}", 0) + 1
            continue

        # Trajectory sanity (optional): filter clips whose planner top-trajectory is essentially
        # "stopped".
        if traj_min_dist_m is not None:
            td = _traj_dist_m_from_pc(pc_rec, index=int(traj_index))
            if td is not None and float(td) < float(traj_min_dist_m):
                filtered += 1
                key = f"traj:traj_too_short<{float(traj_min_dist_m):g}m"
                filtered_reasons[key] = filtered_reasons.get(key, 0) + 1
                continue

        # GT future sanity (if enabled + odom exists).
        num_pts = fallback_num_points
        try:
            c0 = getattr(pc_rec, "candidates", None) or []
            if c0 and getattr(c0[0], "points_xy", None):
                num_pts = max(1, int(len(c0[0].points_xy)))
        except Exception:
            num_pts = fallback_num_points
        gt_local = _gt_future_local_from_episode(
            ep, t0=float(t0), num_points=int(num_pts), traj_dt_s=float(traj_dt_s)
        )
        r_gt = gt_future_filter_reason(gt_local, cfg=gt_cfg)
        if r_gt is not None:
            filtered += 1
            filtered_reasons[f"gt:{r_gt}"] = filtered_reasons.get(f"gt:{r_gt}", 0) + 1
            continue

        kept.append(rec)

    summary = {
        "goal_cfg": (goal_cfg.__dict__ if goal_cfg is not None else None),
        "gt_cfg": (gt_cfg.__dict__ if gt_cfg is not None else None),
        "traj_min_dist_m": (None if traj_min_dist_m is None else float(traj_min_dist_m)),
        "traj_index": int(traj_index),
        "traj_dt_s": float(traj_dt_s),
        "fallback_num_points": int(fallback_num_points),
        "clips_in": int(len(clips)),
        "clips_out": int(len(kept)),
        "clips_filtered": int(filtered),
        "clips_filtered_reasons": {k: int(v) for k, v in sorted(filtered_reasons.items())},
        "clips_parse_errors": int(parse_errors),
        "missing_episode_meta": int(missing_ep),
        "missing_t0_in_planner_candidates": int(missing_t0),
        "path": str(takeover_clips_jsonl_path),
    }

    if not overwrite:
        return summary

    if keep_backup:
        backup = takeover_clips_jsonl_path.with_suffix(
            takeover_clips_jsonl_path.suffix + ".unfiltered"
        )
        if not backup.exists():
            _atomic_write_text(backup, "\n".join(raw_lines) + ("\n" if raw_lines else ""))

    out_text = "\n".join([json.dumps(x, sort_keys=True) for x in kept]) + ("\n" if kept else "")
    _atomic_write_text(takeover_clips_jsonl_path, out_text)
    return summary
