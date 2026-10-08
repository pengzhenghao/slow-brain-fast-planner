from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from slow_brain_fast_planner.benchmarks.dataset import EpisodeLoadResult, load_episode
from slow_brain_fast_planner.benchmarks.overlays import (
    OverlayConfig,
    find_nearest_rgb_record,
    select_overlay_candidate_indices_and_probs,
    write_overlay_for_rgb_record,
)
from slow_brain_fast_planner.benchmarks.planner_postprocessing import A0Config, a0_label_index
from slow_brain_fast_planner.benchmarks.rgb_frame_loader import RGBFrameLoader
from slow_brain_fast_planner.schema.canonical_episode import PlannerCandidate
from slow_brain_fast_planner.utils.trajectory_utils import (
    GoalFilterConfig,
    GtFutureFilterConfig,
    extract_goal_info,
    goal_filter_reason,
    gt_future_filter_reason,
)


def _atomic_write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(str(tmp), str(dst))


def _resize_image_to_width(src: Path, *, out_path: Path, width: int) -> None:
    from PIL import Image

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        w, h = im.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image size: {src} {im.size}")
        if int(width) <= 0:
            raise ValueError("width must be > 0")
        if w == int(width):
            _atomic_write_bytes(out_path, src.read_bytes())
            return
        new_h = max(1, int(round(h * (float(width) / float(w)))))
        im2 = im.resize((int(width), int(new_h)), resample=Image.BILINEAR)
        import io

        buf = io.BytesIO()
        im2.save(buf, format="PNG")
        _atomic_write_bytes(out_path, buf.getvalue())


def _softmax_stable(scores: list[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(float(s) - float(m)) for s in scores]
    s = sum(exps)
    if not math.isfinite(s) or s <= 0:
        return [1.0 / float(len(scores)) for _ in scores]
    return [float(e) / float(s) for e in exps]


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def transform_to_local(global_points: np.ndarray, origin_pose: np.ndarray) -> np.ndarray:
    """Transform global points to local frame of origin_pose.

    origin_pose: [x, y, z, qw, qx, qy, qz]
    """
    origin_pos = origin_pose[:3].astype(np.float64)
    origin_rot = quaternion_to_matrix(origin_pose[3:].astype(np.float64))
    points_3d = np.zeros((len(global_points), 3), dtype=np.float64)
    points_3d[:, : min(global_points.shape[1], 3)] = global_points[
        :, : min(global_points.shape[1], 3)
    ]
    rel_pos = points_3d - origin_pos
    local_pos = rel_pos @ origin_rot.T
    return local_pos[:, :2]


def get_interpolated_pose(poses: np.ndarray | None, t: float, max_t: float) -> np.ndarray | None:
    """Get pose at time t, assuming poses span [0, max_t] uniformly."""
    if poses is None or len(poses) == 0:
        return None
    total_duration = float(max_t)
    if total_duration <= 0:
        return poses[0]
    num_poses = int(len(poses))
    idx_float = (float(t) / total_duration) * float(num_poses - 1)
    idx0 = int(np.floor(idx_float))
    idx1 = min(idx0 + 1, num_poses - 1)
    if idx0 >= num_poses:
        idx0 = num_poses - 1
        idx1 = num_poses - 1
    alpha = float(idx_float - idx0)
    p0 = poses[idx0]
    p1 = poses[idx1]
    res = (1 - alpha) * p0 + alpha * p1
    q = res[3:]
    norm = float(np.linalg.norm(q))
    if norm > 1e-6:
        q = q / norm
    res[3:] = q
    return res


def compute_ade(pred_traj: np.ndarray | None, gt_traj: np.ndarray | None) -> float | None:
    if pred_traj is None or gt_traj is None:
        return None
    if len(pred_traj) != len(gt_traj):
        n = min(len(pred_traj), len(gt_traj))
        pred_traj = pred_traj[:n]
        gt_traj = gt_traj[:n]
    if len(pred_traj) == 0:
        return None
    errs = np.linalg.norm(np.asarray(pred_traj) - np.asarray(gt_traj), axis=1)
    return float(np.mean(errs))


def compute_distance_to_polyline(
    points: np.ndarray | None, polyline: np.ndarray | None
) -> float | None:
    if points is None or len(points) == 0:
        return None
    if polyline is None or len(polyline) < 2:
        return None
    dists: list[float] = []
    for p in points:
        min_d = float("inf")
        for i in range(len(polyline) - 1):
            p1 = polyline[i]
            p2 = polyline[i + 1]
            l2 = float(np.sum((p1 - p2) ** 2))
            if l2 == 0:
                d = float(np.linalg.norm(p - p1))
            else:
                t = max(0.0, min(1.0, float(np.dot(p - p1, p2 - p1) / l2)))
                proj = p1 + t * (p2 - p1)
                d = float(np.linalg.norm(p - proj))
            if d < min_d:
                min_d = d
        dists.append(min_d)
    return float(np.mean(dists)) if dists else None


def _interp_linear(times: np.ndarray, values: np.ndarray, t: float) -> np.ndarray | None:
    """Linear interpolation for 1D time series (best-effort)."""
    if times.size == 0 or values.size == 0 or times.shape[0] != values.shape[0]:
        return None
    tt = float(t)
    if tt <= float(times[0]):
        return values[0]
    if tt >= float(times[-1]):
        return values[-1]
    i = int(np.searchsorted(times, tt, side="right"))
    i0 = max(0, i - 1)
    i1 = min(int(times.shape[0] - 1), i)
    t0 = float(times[i0])
    t1 = float(times[i1])
    if t1 <= t0 + 1e-12:
        return values[i0]
    a = (tt - t0) / (t1 - t0)
    return (1.0 - a) * values[i0] + a * values[i1]


def _world_xy_to_robot_xy(
    world_xy: np.ndarray, *, origin_xy: np.ndarray, origin_yaw: float
) -> np.ndarray:
    """World/map XY -> robot frame (x forward, y left) at origin pose."""
    pts = np.asarray(world_xy, dtype=np.float64)
    o = np.asarray(origin_xy, dtype=np.float64).reshape(2)
    d = pts - o.reshape(1, 2)
    c = float(math.cos(origin_yaw))
    s = float(math.sin(origin_yaw))
    x = c * d[:, 0] + s * d[:, 1]
    y = -s * d[:, 0] + c * d[:, 1]
    return np.stack([x, y], axis=1)


@dataclass(frozen=True)
class SnapshotJob:
    dataset_root: str
    out_dir: str
    episode_meta_path: str
    episode_id: str
    t: float
    snapshot_index: int
    # Config
    a0_cfg: A0Config
    overlay_cfg: OverlayConfig
    rgb_time_tolerance_s: float
    require_goal: bool
    write_overlays: bool
    overlay_image_width: int | None
    prompt_history_frames: int
    prompt_image_width: int | None
    compute_gt_metrics: bool
    traj_dt_s: float
    # Optional: write the decoded RGB frame (no overlay) into RUN_DIR for downstream prompt variants
    # that need to re-render alternative overlays (e.g., hierarchical selection).
    write_rgb_frames: bool = False
    # Optional: override per-tick planner candidates with a *static* candidate set.
    # This is useful for baseline candidate sets (clustered prototypes) that are the same for all
    # snapshots.
    static_candidates_json: str | None = None
    # Optional: override candidate *points* with an external library (e.g., raw anchors),
    # while preserving per-timestep planner scores and indices.
    #
    # If set, we replace `record.candidates[i].points_xy` with override_trajs[i] (index-aligned),
    # then compute overlays + GT metrics on the overridden points.
    candidate_points_override_npy: str | None = None
    candidate_points_override_mode: str = "notebook_v1"  # "notebook_v1" | "xy"
    # Optional sanity filtering at runtime (skip_reason="weird_goal:*").
    goal_filter_cfg: GoalFilterConfig | None = None
    # Optional GT-future sanity filtering (skip_reason="bad_gt_future:*").
    gt_filter_cfg: GtFutureFilterConfig | None = None
    gt_filter_fallback_num_points: int = 10


@dataclass(frozen=True)
class SnapshotPrepResult:
    ok: bool
    episode_id: str
    t: float
    snapshot_index: int
    skip_reason: str | None

    # Labels / diagnostics.
    # label is the ORIGINAL planner index of the GT candidate.
    label: int | None
    label_err: str | None
    score_idx: int | None
    num_candidates: int
    auto_enabled: bool | None

    # Prompt artifacts.
    overlay_frame_ref: str | None
    overlay_error: str | None
    prompt_overlay_frame_ref: str | None
    history_frame_refs: list[str] | None
    # Optional base RGB frame (no overlay), stored under RUN_DIR for re-rendering overlays.
    rgb_frame_ref: str | None

    # Observation payload.
    obs: dict[str, Any] | None
    goal_xy: list[float] | None
    goal_distance_m: float | None
    goal_bearing_deg: float | None
    candidate_confidence: list[dict[str, Any]] | None
    candidates_points_xy: list[list[list[float]]] | None

    # Optional GT-based metrics.
    ade: dict[str, Any] | None
    route_dev: dict[str, Any] | None
    gt_local_traj_xy: list[list[float]] | None
    route_local_xy: list[list[float]] | None
    # Raw planner scores aligned with `record.candidates` index space (length == num_candidates).
    # This is useful for post-hoc sweeps (e.g., top-K-by-score oracle analysis) without re-reading
    # planner_candidates.jsonl.
    candidate_scores_raw: list[float] | None = None


_EP_CACHE: dict[str, tuple[EpisodeLoadResult, dict[float, int]]] = {}
_MAX_T_CACHE: dict[str, float] = {}
_RGB_LOADER: RGBFrameLoader | None = None
_STATIC_CANDIDATES_CACHE: dict[str, list[PlannerCandidate]] = {}
_CANDIDATE_POINTS_OVERRIDE_CACHE: dict[str, np.ndarray] = {}


def _load_candidate_points_override_npy(
    path: str | Path,
    *,
    mode: str = "notebook_v1",
) -> np.ndarray:
    """Load an external candidate trajectory library from .npy as (K,T,2) float64.

    This is mainly for the "raw anchors" experiment: keep the planner's *scores/indices* from
    `planner_candidates.jsonl`, but replace each candidate's `points_xy` with a fixed library.

    Supported `mode`:
    - "notebook_v1": replicate `notebooks/trajectory_candidate_study.ipynb` conversion:
        x = x * 0.51
        y = y * 0.32 - 0.16
        then cumulative sum along time (delta->position).
    - "xy": treat values as already being XY positions in meters (no scaling/cumsum).
    """

    p = Path(path).resolve()
    key = f"{p}::mode={str(mode)}"
    hit = _CANDIDATE_POINTS_OVERRIDE_CACHE.get(key)
    if hit is not None:
        return hit

    if not p.exists():
        raise FileNotFoundError(f"candidate_points_override_npy not found: {p}")

    arr = np.load(p)
    if not isinstance(arr, np.ndarray):
        raise ValueError("candidate_points_override_npy did not load as a numpy array")

    if arr.ndim == 3 and int(arr.shape[-1]) == 2:
        trajs = np.asarray(arr, dtype=np.float64)
    elif arr.ndim == 2 and int(arr.shape[1]) % 2 == 0:
        t = int(arr.shape[1]) // 2
        trajs = np.asarray(arr, dtype=np.float64).reshape(int(arr.shape[0]), int(t), 2)
    else:
        raise ValueError(
            f"Unsupported candidate_points_override_npy shape {getattr(arr, 'shape', None)}; "
            "expected (K,T,2) or (K,2T)."
        )

    if str(mode) == "notebook_v1":
        trajs = trajs.copy()
        trajs[:, :, 0] = trajs[:, :, 0] * 0.51
        trajs[:, :, 1] = trajs[:, :, 1] * 0.32 - 0.16
        trajs[:, :, 0] = trajs[:, :, 0].cumsum(axis=1)
        trajs[:, :, 1] = trajs[:, :, 1].cumsum(axis=1)
    elif str(mode) == "xy":
        trajs = np.asarray(trajs, dtype=np.float64)
    else:
        raise ValueError(
            f"Unknown candidate_points_override_mode={mode!r}; expected 'notebook_v1' or 'xy'"
        )

    _CANDIDATE_POINTS_OVERRIDE_CACHE[key] = trajs
    return trajs


def _load_static_candidates(path: str | Path) -> list[PlannerCandidate]:
    """Load a static candidate set JSON (from
    scripts/trajectory_selection/build_static_candidate_set.py)."""
    p = Path(path).resolve()
    key = str(p)
    hit = _STATIC_CANDIDATES_CACHE.get(key)
    if hit is not None:
        return hit
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("static_candidates_json must be a JSON object")
    cands = obj.get("candidates")
    if not isinstance(cands, list) or not cands:
        raise ValueError("static_candidates_json missing non-empty 'candidates' list")
    out: list[PlannerCandidate] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        pts = c.get("points_xy")
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        score = c.get("score", 0.0)
        out.append(
            PlannerCandidate(
                points_xy=[
                    [float(p0[0]), float(p0[1])]
                    for p0 in pts
                    if isinstance(p0, (list, tuple)) and len(p0) >= 2
                ],
                score=float(score) if isinstance(score, (int, float)) else 0.0,
            )
        )
    if not out:
        raise ValueError("static_candidates_json had no valid candidates after parsing")
    _STATIC_CANDIDATES_CACHE[key] = out
    return out


def _get_rgb_loader() -> RGBFrameLoader:
    global _RGB_LOADER
    if _RGB_LOADER is None:
        _RGB_LOADER = RGBFrameLoader(cache_size=128)
    return _RGB_LOADER


def _get_episode_cached(
    dataset_root: Path, episode_meta_path: Path
) -> tuple[EpisodeLoadResult, dict[float, int]]:
    key = str(episode_meta_path)
    hit = _EP_CACHE.get(key)
    if hit is not None:
        return hit
    ep = load_episode(dataset_root, episode_meta_path)
    t_to_index: dict[float, int] = {}
    for i, r in enumerate(ep.planner_candidates):
        t_to_index[round(float(r.t), 6)] = int(i)
    _EP_CACHE[key] = (ep, t_to_index)
    return ep, t_to_index


def _a0_observation(
    *,
    episode_id: str,
    t: float,
    record: Any,
    overlay_frame_ref: str | None,
    candidate_confidence: list[dict[str, Any]] | None,
    goal_xy: list[float] | None,
    goal_distance_m: float | None,
    goal_bearing_deg: float | None,
) -> dict[str, Any]:
    candidates = []
    for i, c in enumerate(getattr(record, "candidates", []) or []):
        end_xy = c.points_xy[-1] if getattr(c, "points_xy", None) else None
        candidates.append({"index": int(i), "score": float(c.score), "endpoint_xy": end_xy})
    return {
        "episode_id": str(episode_id),
        "t": float(t),
        "vision": {"overlay_frame_ref": overlay_frame_ref},
        "goal": {
            "goal_xy": goal_xy,
            "goal_distance_m": goal_distance_m,
            "goal_bearing_deg": goal_bearing_deg,
        },
        "planner": {"candidates": candidates, "candidate_confidence": candidate_confidence},
    }


def prepare_snapshot(job: SnapshotJob) -> SnapshotPrepResult:
    dataset_root = Path(job.dataset_root).resolve()
    out_dir = Path(job.out_dir).resolve()
    meta_path = Path(job.episode_meta_path).resolve()

    ep, t_to_index = _get_episode_cached(dataset_root, meta_path)
    if (not ep.schema_valid) or ep.episode is None or (not ep.planner_candidates):
        return SnapshotPrepResult(
            ok=False,
            episode_id=str(job.episode_id),
            t=float(job.t),
            snapshot_index=int(job.snapshot_index),
            skip_reason="episode_schema_invalid_or_missing_candidates",
            label=None,
            label_err="episode_schema_invalid_or_missing_candidates",
            score_idx=None,
            num_candidates=0,
            auto_enabled=None,
            overlay_frame_ref=None,
            overlay_error="episode_schema_invalid_or_missing_candidates",
            prompt_overlay_frame_ref=None,
            history_frame_refs=None,
            rgb_frame_ref=None,
            obs=None,
            goal_xy=None,
            goal_distance_m=None,
            goal_bearing_deg=None,
            candidate_confidence=None,
            candidates_points_xy=None,
            ade=None,
            route_dev=None,
            gt_local_traj_xy=None,
            route_local_xy=None,
        )

    idx = t_to_index.get(round(float(job.t), 6))
    if idx is None:
        return SnapshotPrepResult(
            ok=False,
            episode_id=str(job.episode_id),
            t=float(job.t),
            snapshot_index=int(job.snapshot_index),
            skip_reason="t_not_found_in_planner_candidates",
            label=None,
            label_err="t_not_found_in_planner_candidates",
            score_idx=None,
            num_candidates=0,
            auto_enabled=None,
            overlay_frame_ref=None,
            overlay_error="t_not_found_in_planner_candidates",
            prompt_overlay_frame_ref=None,
            history_frame_refs=None,
            rgb_frame_ref=None,
            obs=None,
            goal_xy=None,
            goal_distance_m=None,
            goal_bearing_deg=None,
            candidate_confidence=None,
            candidates_points_xy=None,
            ade=None,
            route_dev=None,
            gt_local_traj_xy=None,
            route_local_xy=None,
        )

    record = ep.planner_candidates[int(idx)]

    # Optional override: replace per-tick planner candidates with a static candidate set.
    # Goal/metadata fields remain from the original record.
    if getattr(job, "static_candidates_json", None):
        try:
            static_cands = _load_static_candidates(str(job.static_candidates_json))
            try:
                record2 = record.model_copy(deep=True)  # pydantic v2 BaseModel
            except Exception:
                import copy

                record2 = copy.deepcopy(record)
            try:
                record2.candidates = list(static_cands)
                record = record2
            except Exception:
                # Fall back to mutating record in-place (best-effort).
                record.candidates = list(static_cands)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Optional override: replace candidate *points* (index-aligned) with an external library.
    # This is intended for experiments like:
    # - oracle minADE over raw anchors
    # - score-argmax selection, but evaluating the corresponding anchor trajectory
    if getattr(job, "candidate_points_override_npy", None):
        try:
            override = _load_candidate_points_override_npy(
                str(job.candidate_points_override_npy),
                mode=str(getattr(job, "candidate_points_override_mode", "notebook_v1")),
            )  # (K,T,2)
            if override.ndim != 3 or int(override.shape[-1]) != 2:
                raise ValueError(f"override array must be (K,T,2), got {override.shape}")

            # Deep-copy the record so we don't mutate cached episode objects (best-effort).
            try:
                record2 = record.model_copy(deep=True)  # pydantic v2 BaseModel
            except Exception:
                import copy

                record2 = copy.deepcopy(record)

            # Determine target horizon length from the current record (if available).
            n_points = None
            try:
                if getattr(record2, "candidates", None) and getattr(
                    record2.candidates[0], "points_xy", None
                ):
                    n_points = int(len(record2.candidates[0].points_xy))
            except Exception:
                n_points = None

            if n_points is not None and n_points > 0:
                override2 = np.asarray(override[:, : int(n_points), :], dtype=np.float64)
            else:
                override2 = np.asarray(override, dtype=np.float64)

            n_apply = 0
            try:
                cand_list = list(getattr(record2, "candidates", []) or [])
                if len(cand_list) > int(override2.shape[0]):
                    raise ValueError(
                        f"override library too small: record has {len(cand_list)} candidates "
                        f"but override has {int(override2.shape[0])}"
                    )
                for i in range(min(len(cand_list), int(override2.shape[0]))):
                    pts_i = override2[int(i)]
                    cand_list[int(i)].points_xy = [[float(x), float(y)] for x, y in pts_i.tolist()]  # type: ignore[attr-defined]
                    n_apply += 1
                record2.candidates = cand_list  # type: ignore[attr-defined]
                record = record2
            except Exception as e:
                # Best-effort fallback: try mutating in-place.
                cand_list = list(getattr(record, "candidates", []) or [])
                if len(cand_list) > int(override2.shape[0]):
                    raise ValueError(
                        f"override library too small: record has {len(cand_list)} candidates "
                        f"but override has {int(override2.shape[0])}"
                    ) from e
                for i in range(min(len(cand_list), int(override2.shape[0]))):
                    pts_i = override2[int(i)]
                    cand_list[int(i)].points_xy = [[float(x), float(y)] for x, y in pts_i.tolist()]  # type: ignore[attr-defined]
                    n_apply += 1
                if n_apply <= 0:
                    raise RuntimeError(f"failed_to_apply_candidate_points_override:{e}") from e
        except Exception as e:
            err = f"candidate_points_override_failed:{e}"
            # If override fails, skip this snapshot (fail-soft to avoid crashing multi-worker runs).
            return SnapshotPrepResult(
                ok=False,
                episode_id=str(job.episode_id),
                t=float(job.t),
                snapshot_index=int(job.snapshot_index),
                skip_reason=err,
                label=None,
                label_err=err,
                score_idx=None,
                num_candidates=0,
                auto_enabled=getattr(record, "auto_enabled", None),
                overlay_frame_ref=None,
                overlay_error=err,
                prompt_overlay_frame_ref=None,
                history_frame_refs=None,
                rgb_frame_ref=None,
                obs=None,
                goal_xy=None,
                goal_distance_m=None,
                goal_bearing_deg=None,
                candidate_confidence=None,
                candidates_points_xy=None,
                ade=None,
                route_dev=None,
                gt_local_traj_xy=None,
                route_local_xy=None,
            )
    num_candidates = int(len(getattr(record, "candidates", []) or []))
    candidates_points_xy: list[list[list[float]]] | None = None
    try:
        if num_candidates > 0:
            candidates_points_xy = []
            for c in record.candidates:
                pts = getattr(c, "points_xy", None)
                if isinstance(pts, list):
                    candidates_points_xy.append(
                        [
                            [float(p[0]), float(p[1])]
                            for p in pts
                            if isinstance(p, (list, tuple)) and len(p) >= 2
                        ]
                    )
                else:
                    candidates_points_xy.append([])
    except Exception:
        candidates_points_xy = None

    # Label + score-argmax.
    #
    # NOTE: The "accuracy" metric requires a label. For the usual (planner-derived) candidate set,
    # the label is produced by the A0 rule (see a0_label_index).
    #
    # For *static* candidate sets (clustered prototypes), the A0 label over the original planner
    # set does
    # not apply. We still provide a label so the evaluator counts this snapshot as "evaluated":
    # - If GT metrics are available, we will later overwrite label := min-ADE index (visible
    # oracle).
    # - Otherwise we default to label := 0 (meaningful accuracy is not expected in that mode).
    if getattr(job, "static_candidates_json", None):
        label, label_err = 0, None
    else:
        label, label_err = a0_label_index(record, job.a0_cfg)
    score_idx: int | None = None
    try:
        if num_candidates > 0:
            scores0 = [float(c.score) for c in record.candidates]
            score_idx = int(np.argmax(np.asarray(scores0, dtype=np.float64))) if scores0 else 0
    except Exception:
        score_idx = None

    # Goal info (optional).
    goal_xy: list[float] | None = None
    goal_distance_m: float | None = None
    goal_bearing_deg: float | None = None
    try:
        goal_xy, goal_distance_m, goal_bearing_deg = extract_goal_info(record)
    except Exception:
        goal_xy = None
        goal_distance_m = None
        goal_bearing_deg = None

    # Goal sanity filtering (best-effort).
    try:
        reason = goal_filter_reason(
            goal_xy, goal_distance_m, goal_bearing_deg, cfg=job.goal_filter_cfg
        )
    except Exception:
        reason = None
    if reason is not None:
        return SnapshotPrepResult(
            ok=False,
            episode_id=str(job.episode_id),
            t=float(job.t),
            snapshot_index=int(job.snapshot_index),
            skip_reason=f"weird_goal:{reason}",
            label=label,
            label_err=label_err,
            score_idx=score_idx,
            num_candidates=num_candidates,
            auto_enabled=getattr(record, "auto_enabled", None),
            overlay_frame_ref=None,
            overlay_error=None,
            prompt_overlay_frame_ref=None,
            history_frame_refs=None,
            rgb_frame_ref=None,
            obs=None,
            goal_xy=goal_xy,
            goal_distance_m=goal_distance_m,
            goal_bearing_deg=goal_bearing_deg,
            candidate_confidence=None,
            candidates_points_xy=candidates_points_xy,
            ade=None,
            route_dev=None,
            gt_local_traj_xy=None,
            route_local_xy=None,
        )

    # GT-future sanity filtering (run early to avoid expensive overlay/VLM work).
    gt_local_traj_xy_prefilter: list[list[float]] | None = None
    if bool(job.compute_gt_metrics):
        try:
            if (
                job.gt_filter_cfg is not None
                and bool(job.gt_filter_cfg.enabled)
                and getattr(ep, "odom", None)
            ):
                # Determine number of GT points from candidate horizon when possible.
                n_pts = int(job.gt_filter_fallback_num_points)
                try:
                    if record.candidates:
                        n_pts = max(1, int(len(record.candidates[0].points_xy)))
                except Exception:
                    n_pts = int(job.gt_filter_fallback_num_points)

                od = getattr(ep, "odom", []) or []
                t_arr = np.asarray([float(r.t) for r in od], dtype=np.float64)
                xy_arr = np.asarray([[float(r.x), float(r.y)] for r in od], dtype=np.float64)
                yaw_arr = np.asarray([float(r.yaw) for r in od], dtype=np.float64)
                if (
                    t_arr.size > 1
                    and xy_arr.shape[0] == t_arr.shape[0]
                    and yaw_arr.shape[0] == t_arr.shape[0]
                ):
                    curr_xy = _interp_linear(t_arr, xy_arr, float(job.t))
                    curr_yaw = _interp_linear(t_arr, yaw_arr.reshape(-1, 1), float(job.t))
                    if curr_xy is not None and curr_yaw is not None and int(n_pts) > 0:
                        fut_xy_world: list[list[float]] = []
                        for k in range(1, int(n_pts) + 1):
                            t_k = float(job.t) + float(k) * float(job.traj_dt_s)
                            xy_k = _interp_linear(t_arr, xy_arr, t_k)
                            if xy_k is None:
                                fut_xy_world = []
                                break
                            fut_xy_world.append([float(xy_k[0]), float(xy_k[1])])
                        if fut_xy_world:
                            gt_local = _world_xy_to_robot_xy(
                                np.asarray(fut_xy_world, dtype=np.float64),
                                origin_xy=np.asarray(curr_xy, dtype=np.float64),
                                origin_yaw=float(curr_yaw.reshape(-1)[0]),
                            )
                            gt_local_traj_xy_prefilter = [
                                [float(x), float(y)] for x, y in gt_local.tolist()
                            ]
                            gt_reason = gt_future_filter_reason(
                                gt_local_traj_xy_prefilter, cfg=job.gt_filter_cfg
                            )
                            if gt_reason is not None:
                                return SnapshotPrepResult(
                                    ok=False,
                                    episode_id=str(job.episode_id),
                                    t=float(job.t),
                                    snapshot_index=int(job.snapshot_index),
                                    skip_reason=f"bad_gt_future:{gt_reason}",
                                    label=label,
                                    label_err=label_err,
                                    score_idx=score_idx,
                                    num_candidates=num_candidates,
                                    auto_enabled=getattr(record, "auto_enabled", None),
                                    overlay_frame_ref=None,
                                    overlay_error=None,
                                    prompt_overlay_frame_ref=None,
                                    history_frame_refs=None,
                                    rgb_frame_ref=None,
                                    obs=None,
                                    goal_xy=goal_xy,
                                    goal_distance_m=goal_distance_m,
                                    goal_bearing_deg=goal_bearing_deg,
                                    candidate_confidence=None,
                                    candidates_points_xy=candidates_points_xy,
                                    ade=None,
                                    route_dev=None,
                                    gt_local_traj_xy=gt_local_traj_xy_prefilter,
                                    route_local_xy=None,
                                )
        except Exception:
            gt_local_traj_xy_prefilter = None

    if bool(job.require_goal) and goal_xy is None:
        return SnapshotPrepResult(
            ok=False,
            episode_id=str(job.episode_id),
            t=float(job.t),
            snapshot_index=int(job.snapshot_index),
            skip_reason="missing_goal_xy",
            label=label,
            label_err=label_err,
            score_idx=score_idx,
            num_candidates=num_candidates,
            auto_enabled=getattr(record, "auto_enabled", None),
            overlay_frame_ref=None,
            overlay_error=None,
            prompt_overlay_frame_ref=None,
            history_frame_refs=None,
            rgb_frame_ref=None,
            obs=None,
            goal_xy=None,
            goal_distance_m=None,
            goal_bearing_deg=None,
            candidate_confidence=None,
            candidates_points_xy=candidates_points_xy,
            ade=None,
            route_dev=None,
            gt_local_traj_xy=None,
            route_local_xy=None,
        )

    # Candidate confidence table for prompt.
    # Uses ORIGINAL planner indices (not rank-based).
    # Sorted by set_prob (descending) to show most likely candidates first.
    candidate_confidence: list[dict[str, Any]] | None = None
    raw_scores: list[float] | None = None
    visible_candidate_indices: list[int] = []
    try:
        raw_scores = [float(c.score) for c in record.candidates]
        raw_probs = _softmax_stable(raw_scores)

        # Select candidates exactly as shown in the overlay.
        selected, set_probs = select_overlay_candidate_indices_and_probs(record, job.overlay_cfg)
        visible_candidate_indices = [int(x) for x in (selected or [])]

        pairs: list[tuple[int, float | None]] = []
        if selected:
            if set_probs and len(set_probs) == len(selected):
                pairs = [(int(i), float(p)) for i, p in zip(selected, set_probs, strict=False)]
                pairs.sort(key=lambda x: (-(x[1] if x[1] is not None else -1.0), x[0]))
            else:
                pairs = [(int(i), None) for i in selected]

        # Build table ONLY for candidates shown in overlay.
        rows2: list[dict[str, Any]] = []
        for idx, set_prob in pairs:
            if not (0 <= int(idx) < len(record.candidates)):
                continue
            cand = record.candidates[int(idx)]
            # Compute trajectory distance (path length)
            traj_dist_m: float | None = None
            end_xy: list[float] | None = None
            try:
                pts = cand.points_xy
                if pts and len(pts) >= 2:
                    dist = 0.0
                    for j in range(1, len(pts)):
                        dx = float(pts[j][0]) - float(pts[j - 1][0])
                        dy = float(pts[j][1]) - float(pts[j - 1][1])
                        dist += math.sqrt(dx * dx + dy * dy)
                    traj_dist_m = dist
                    end_xy = [round(float(pts[-1][0]), 1), round(float(pts[-1][1]), 1)]
            except Exception:
                pass

            rows2.append(
                {
                    "index": int(idx),
                    "score": round(float(raw_scores[idx]), 2),
                    "raw_prob": round(float(raw_probs[idx]), 3) if idx < len(raw_probs) else None,
                    # Kept key name for backward compatibility with existing prompts/logs.
                    # Semantics: normalized prob within the *displayed* candidate set.
                    "nms_prob": round(float(set_prob), 3) if set_prob is not None else None,
                    "traj_dist_m": round(traj_dist_m, 1) if traj_dist_m is not None else None,
                    "end_xy": end_xy,
                }
            )

        candidate_confidence = rows2
    except Exception:
        candidate_confidence = None
        raw_scores = None
        visible_candidate_indices = []

    # Overlays + prompt assets.
    overlay_frame_ref = None
    overlay_error = None
    prompt_overlay_frame_ref = None
    history_frame_refs: list[str] | None = None
    rgb_frame_ref: str | None = None

    t_str = f"{float(job.t):.3f}".rstrip("0").rstrip(".")
    if bool(job.write_overlays):
        try:
            rgb_rec = find_nearest_rgb_record(
                ep.rgb, float(job.t), tol_s=float(job.rgb_time_tolerance_s)
            )
            if rgb_rec is None:
                overlay_error = "missing_rgb_near_t"
            else:
                # Optionally write the base RGB (no overlay) into the run directory (for downstream
                # re-rendering).
                if bool(getattr(job, "write_rgb_frames", False)):
                    try:
                        rgb_path = (
                            out_dir
                            / "artifacts"
                            / "rgb_frames"
                            / str(job.episode_id)
                            / f"{t_str}.png"
                        )
                        if not rgb_path.exists():
                            loader = _get_rgb_loader()
                            img = loader.load(
                                rgb_rec, episode_dir=Path(ep.episode_dir), dataset_root=dataset_root
                            )
                            import io

                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            _atomic_write_bytes(rgb_path, buf.getvalue())
                        rgb_frame_ref = str(rgb_path.relative_to(out_dir))
                    except Exception:
                        rgb_frame_ref = None

                overlay_path = (
                    out_dir / "artifacts" / "overlays" / str(job.episode_id) / f"{t_str}.png"
                )
                if not overlay_path.exists():
                    write_overlay_for_rgb_record(
                        rgb_record=rgb_rec,
                        episode_dir=Path(ep.episode_dir),
                        dataset_root=dataset_root,
                        out_path=overlay_path,
                        record=record,
                        cfg=job.overlay_cfg,
                        pred_index=None,
                        label_index=None,
                        rgb_loader=_get_rgb_loader(),
                        base_image_width=(
                            int(job.overlay_image_width) if job.overlay_image_width else None
                        ),
                    )
                overlay_frame_ref = str(overlay_path.relative_to(out_dir))
        except Exception as e:  # noqa: BLE001
            overlay_error = f"overlay_exception:{e}"

    prompt_overlay_frame_ref = overlay_frame_ref
    if overlay_frame_ref is not None and job.prompt_image_width is not None:
        try:
            w = int(job.prompt_image_width)
            if w > 0:
                src = (out_dir / Path(overlay_frame_ref)).resolve()
                outp = (
                    out_dir
                    / "artifacts"
                    / "prompt_images"
                    / str(job.episode_id)
                    / f"{t_str}_w{w}.png"
                )
                if src.exists() and (not outp.exists()):
                    _resize_image_to_width(src, out_path=outp, width=w)
                prompt_overlay_frame_ref = str(outp.relative_to(out_dir))
        except Exception:
            prompt_overlay_frame_ref = overlay_frame_ref

    if overlay_frame_ref is not None and int(job.prompt_history_frames) > 0:
        # History frames: copy N previous RGB frames into RUN_DIR so adapters can resolve them.
        try:
            n_hist = int(job.prompt_history_frames)
        except Exception:
            n_hist = 0
        if n_hist > 0:
            # Use sorted rgb by t, bisect to find nearest.
            times = [float(getattr(r, "t", 0.0)) for r in ep.rgb]
            import bisect

            i2 = bisect.bisect_left(times, float(job.t))
            rgb_idx = None
            best_dist = float("inf")
            for j in (i2 - 1, i2):
                if 0 <= j < len(times):
                    dist = abs(times[j] - float(job.t))
                    if dist < best_dist:
                        best_dist = dist
                        rgb_idx = int(j)
            if rgb_idx is not None and best_dist <= float(job.rgb_time_tolerance_s):
                if rgb_idx < n_hist:
                    # Insufficient history (strict discarding guideline)
                    history_frame_refs = None
                    # We could error out here, or just fail to provide history.
                    # But if the user wants to discard the snapshot, we should mark ok=False or
                    # similar?
                    # The function returns SnapshotPrepResult. Let's return ok=False via a check
                    # below or raise/return early.
                    # Since we are inside a block that sets 'history_frame_refs', let's just not
                    # set it,
                    # and if it is required but missing, maybe we should skip.
                    # BUT, usually if history is requested but not available, that's a skip.
                    return SnapshotPrepResult(
                        ok=False,
                        episode_id=str(job.episode_id),
                        t=float(job.t),
                        snapshot_index=int(job.snapshot_index),
                        skip_reason="insufficient_history_for_prompt",
                        label=label,
                        label_err=label_err,
                        score_idx=score_idx,
                        num_candidates=num_candidates,
                        auto_enabled=getattr(record, "auto_enabled", None),
                        overlay_frame_ref=None,
                        overlay_error="insufficient_history",
                        prompt_overlay_frame_ref=None,
                        history_frame_refs=None,
                        rgb_frame_ref=None,
                        obs=None,
                        goal_xy=None,
                        goal_distance_m=None,
                        goal_bearing_deg=None,
                        candidate_confidence=None,
                        candidates_points_xy=candidates_points_xy,
                        ade=None,
                        route_dev=None,
                        gt_local_traj_xy=None,
                        route_local_xy=None,
                    )

                start = rgb_idx - n_hist
                refs: list[str] = []
                loader = _get_rgb_loader()
                for j in range(start, rgb_idx):
                    rr = ep.rgb[j]
                    dst = (
                        out_dir
                        / "artifacts"
                        / "history_frames"
                        / str(job.episode_id)
                        / t_str
                        / f"hist_{j:06d}.png"
                    )
                    try:
                        if not dst.exists():
                            img = loader.load(
                                rr, episode_dir=Path(ep.episode_dir), dataset_root=dataset_root
                            )
                            import io

                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            _atomic_write_bytes(dst, buf.getvalue())
                        refs.append(str(dst.relative_to(out_dir)))
                    except Exception:
                        continue
                if refs:
                    history_frame_refs = refs

    obs = _a0_observation(
        episode_id=str(job.episode_id),
        t=float(job.t),
        record=record,
        overlay_frame_ref=prompt_overlay_frame_ref,
        candidate_confidence=candidate_confidence,
        goal_xy=goal_xy,
        goal_distance_m=goal_distance_m,
        goal_bearing_deg=goal_bearing_deg,
    )
    # Attach overlay rendering metadata for VLM prompts.
    try:
        if isinstance(obs, dict):
            vision = obs.get("vision")
            if not isinstance(vision, dict):
                vision = {}
                obs["vision"] = vision
            vision["overlay_traj_style"] = str(job.overlay_cfg.traj_style)
            vision["robot_width_m"] = float(job.overlay_cfg.robot_width_m)
    except Exception:
        pass

    # Optional GT metrics (human odom + route).
    ade_out: dict[str, Any] | None = None
    route_out: dict[str, Any] | None = None
    gt_local_traj_xy: list[list[float]] | None = None
    route_local_xy: list[list[float]] | None = None

    if bool(job.compute_gt_metrics) and num_candidates > 0:
        # Ground truth definition for ADE:
        # GT future trajectory is the executed future path in the robot local frame at time t.

        max_t = (
            max([float(rr.t) for rr in ep.planner_candidates])
            if ep.planner_candidates
            else float(job.t)
        )

        # Canonical odom stream (required when GT metrics are enabled).
        if getattr(ep, "odom", None):
            try:
                od = getattr(ep, "odom", []) or []
                # Only support map-frame odom for now; if frame varies, we still try.
                t_arr = np.asarray([float(r.t) for r in od], dtype=np.float64)
                xy_arr = np.asarray([[float(r.x), float(r.y)] for r in od], dtype=np.float64)
                yaw_arr = np.asarray([float(r.yaw) for r in od], dtype=np.float64)
                if t_arr.size > 1 and xy_arr.shape[0] == t_arr.shape[0]:
                    # Cache in a lightweight pose-like shape: [x, y, z(0), qw(1), qx(0), qy(0),
                    # qz(0)]
                    # so existing code paths can still work if needed.
                    # For ADE, we only use x,y,yaw directly below.
                    pass
            except Exception:
                t_arr = np.asarray([], dtype=np.float64)
                xy_arr = np.asarray([], dtype=np.float64).reshape(0, 2)
                yaw_arr = np.asarray([], dtype=np.float64)
        else:
            t_arr = np.asarray([], dtype=np.float64)
            xy_arr = np.asarray([], dtype=np.float64).reshape(0, 2)
            yaw_arr = np.asarray([], dtype=np.float64)

        if t_arr.size == 0 or xy_arr.shape[0] == 0:
            # Some datasets (e.g. lightweight test fixtures) may not include canonical odom.
            # GT-dependent metrics (ADE/FDE/oracles) are optional; if GT is missing, skip them
            # for this snapshot rather than failing the entire run.
            ade_out = None
            route_out = None
            gt_local_traj_xy = gt_local_traj_xy_prefilter
            route_local_xy = None
        else:
            if str(job.episode_id) not in _MAX_T_CACHE:
                _MAX_T_CACHE[str(job.episode_id)] = float(max_t)
            max_t = float(_MAX_T_CACHE[str(job.episode_id)])

            try:
                # ADE against executed GT future in robot local frame.
                num_points = len(record.candidates[0].points_xy) if record.candidates else 0
                traj_dt = float(job.traj_dt_s)

                gt_local: np.ndarray | None = None

                # If canonical odom is available, compute local GT directly from it.
                if (
                    t_arr.size > 1
                    and xy_arr.shape[0] == t_arr.shape[0]
                    and yaw_arr.shape[0] == t_arr.shape[0]
                ):
                    curr_xy = _interp_linear(t_arr, xy_arr, float(job.t))
                    curr_yaw = _interp_linear(t_arr, yaw_arr.reshape(-1, 1), float(job.t))
                    if curr_xy is not None and curr_yaw is not None and int(num_points) > 0:
                        fut_xy_world: list[list[float]] = []
                        for k in range(1, int(num_points) + 1):
                            t_k = float(job.t) + float(k) * traj_dt
                            xy_k = _interp_linear(t_arr, xy_arr, t_k)
                            if xy_k is None:
                                fut_xy_world = []
                                break
                            fut_xy_world.append([float(xy_k[0]), float(xy_k[1])])
                        if fut_xy_world:
                            gt_local = _world_xy_to_robot_xy(
                                np.asarray(fut_xy_world, dtype=np.float64),
                                origin_xy=np.asarray(curr_xy, dtype=np.float64),
                                origin_yaw=float(curr_yaw.reshape(-1)[0]),
                            )

                # Legacy fallback: use pose arrays (with quaternion) if available.
                # No legacy fallback: GT must come from canonical odom.

                gt_for_ade: np.ndarray | None = None
                if gt_local is not None and gt_local.size > 0:
                    gt_for_ade = np.asarray(gt_local, dtype=np.float64)
                    gt_local_traj_xy = [[float(x), float(y)] for x, y in gt_for_ade.tolist()]
                elif gt_local_traj_xy_prefilter is not None:
                    # Prefilter already produced a valid local-frame GT future; reuse it for ADE.
                    gt_for_ade = np.asarray(gt_local_traj_xy_prefilter, dtype=np.float64)
                    gt_local_traj_xy = gt_local_traj_xy_prefilter

                # ADE against executed GT future (both "visible" oracle and global oracle).
                if gt_for_ade is not None and gt_for_ade.size > 0:
                    ades: list[float | None] = []
                    for cand in record.candidates:
                        c_points = np.asarray(cand.points_xy, dtype=np.float64)
                        ades.append(compute_ade(c_points, gt_for_ade))
                    # Keep only valid ADEs; if any candidate is missing, skip ADE dict entirely.
                    if ades and not any(a is None for a in ades):
                        # Global oracle (over all candidates).
                        min_all_idx = int(np.argmin(np.asarray(ades, dtype=np.float64)))
                        min_all_val = float(ades[min_all_idx])

                        # "Visible" oracle: restrict to the post-NMS/prob-filtered candidates
                        # (i.e., what VLM baselines see in the overlay/table).
                        vis = [
                            int(i)
                            for i in (visible_candidate_indices or [])
                            if 0 <= int(i) < len(ades)
                        ]
                        if vis:
                            vis_ades = np.asarray([ades[i] for i in vis], dtype=np.float64)
                            min_vis_idx = int(vis[int(np.argmin(vis_ades))])
                            min_vis_val = float(ades[min_vis_idx])
                        else:
                            # If no visible candidates are available, fall back to global oracle.
                            min_vis_idx = int(min_all_idx)
                            min_vis_val = float(min_all_val)

                        if score_idx is None:
                            score_idx2 = int(
                                np.argmax(np.asarray([float(c.score) for c in record.candidates]))
                            )
                        else:
                            score_idx2 = int(score_idx)
                        score_val = float(ades[score_idx2])

                        ade_out = {
                            "ades": [float(x) for x in ades],
                            # Keep the legacy keys, but redefine them as "visible" oracle for
                            # fairness.
                            "min": float(min_vis_val),
                            "min_idx": int(min_vis_idx),
                            # Debug fields: global oracle + visible set indices.
                            "min_all": float(min_all_val),
                            "min_all_idx": int(min_all_idx),
                            "visible_indices": [int(i) for i in (visible_candidate_indices or [])],
                            "score": float(score_val),
                            "score_idx": int(score_idx2),
                        }
            except Exception:
                ade_out = None

        # NOTE: route deviation metrics currently require a canonical route stream.
        # Legacy raw-route loading has been removed to simplify GT logic.

    # For static candidate sets, overwrite label := visible-min-ADE index when available so
    # "accuracy" becomes "accuracy_vs_min_ade" (and snapshots count as evaluated).
    if (
        getattr(job, "static_candidates_json", None)
        and isinstance(ade_out, dict)
        and ade_out.get("min_idx") is not None
    ):
        try:
            label = int(ade_out.get("min_idx"))  # type: ignore[assignment]
            label_err = None
        except Exception:
            pass

    # Ensure JSON-serializable.
    try:
        json.dumps(obs, sort_keys=True)
    except Exception:
        obs = {"episode_id": str(job.episode_id), "t": float(job.t)}

    return SnapshotPrepResult(
        ok=True,
        episode_id=str(job.episode_id),
        t=float(job.t),
        snapshot_index=int(job.snapshot_index),
        skip_reason=None,
        label=int(label) if label is not None else None,
        label_err=str(label_err) if label_err is not None else None,
        score_idx=int(score_idx) if score_idx is not None else None,
        num_candidates=num_candidates,
        auto_enabled=getattr(record, "auto_enabled", None),
        overlay_frame_ref=overlay_frame_ref,
        overlay_error=overlay_error,
        prompt_overlay_frame_ref=prompt_overlay_frame_ref,
        history_frame_refs=history_frame_refs,
        rgb_frame_ref=rgb_frame_ref,
        obs=obs,
        goal_xy=goal_xy,
        goal_distance_m=goal_distance_m,
        goal_bearing_deg=goal_bearing_deg,
        candidate_confidence=candidate_confidence,
        candidate_scores_raw=raw_scores,
        candidates_points_xy=candidates_points_xy,
        ade=ade_out,
        route_dev=route_out,
        gt_local_traj_xy=gt_local_traj_xy,
        route_local_xy=route_local_xy,
    )
