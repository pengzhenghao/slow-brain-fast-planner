from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files, load_episode
from slow_brain_fast_planner.benchmarks.rgb_frame_loader import RGBFrameLoader
from slow_brain_fast_planner.planner.onnx_s1 import OnnxS1Planner, OnnxS1PlannerConfig
from slow_brain_fast_planner.schema.canonical_episode import CANONICAL_EPISODE_SCHEMA_VERSION
from slow_brain_fast_planner.utils.io import read_json, write_json


@dataclass(frozen=True)
class PlannerCandidatesOnnxReport:
    dataset: str
    episode_id: str
    model: str
    records_written: int
    out_path: str


def _prepare_obs_window(frames_rgb: list[np.ndarray], *, out_h: int, out_w: int) -> np.ndarray:
    """frames_rgb: list of HxWx3 uint8 RGB; returns (1,T,3,H,W) float32 in [0,1]."""

    import cv2  # type: ignore

    resized = []
    for fr in frames_rgb:
        fr2 = cv2.resize(fr, (960, 540), interpolation=cv2.INTER_LINEAR)
        fr3 = cv2.resize(fr2, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        resized.append(fr3)
    x = np.stack(resized, axis=0)  # (T,H,W,3)
    x = x.astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))  # (T,3,H,W)
    return x[np.newaxis, ...]  # (1,T,3,H,W)


def _extract_goal_xy(record: Any) -> list[float] | None:
    """Best-effort goal extraction from a PlannerCandidatesRecord-like object.

    Canonical schema stores goal in extension fields (Pydantic `model_extra`), so we look there.
    """

    for key in ("goal_xy", "goal_point_xy", "goal_point"):
        try:
            if isinstance(record, dict):
                g = record.get(key)
            else:
                extra = getattr(record, "model_extra", {}) or {}
                g = extra.get(key)
                if g is None and hasattr(record, key):
                    g = getattr(record, key)
            if isinstance(g, (list, tuple)) and len(g) >= 2:
                return [float(g[0]), float(g[1])]
        except Exception:
            continue
    return None


def _goal_point_scaled_from_goal_xy(*, goal_xy_robot: list[float]) -> list[float] | None:
    """Convert a robot-frame goal XY vector into the ONNX goal_point encoding.

    Encoding matches the reference COCO testset implementation:
      goal_point_scaled = [clip(||goal||,0..100)/100, cos(atan2(y,x)), sin(atan2(y,x))]
    """

    try:
        if not isinstance(goal_xy_robot, list) or len(goal_xy_robot) < 2:
            return None
        gx = float(goal_xy_robot[0])
        gy = float(goal_xy_robot[1])
        if not np.isfinite(gx) or not np.isfinite(gy):
            return None
        dist = float(np.hypot(gx, gy))
        dist_norm = float(np.clip(dist, 0.0, 100.0) / 100.0)
        ang = float(np.arctan2(gy, gx))
        return [dist_norm, float(np.cos(ang)), float(np.sin(ang))]
    except Exception:
        return None


def _goal_from_future_odom(
    *,
    xy_map: np.ndarray,
    yaw_map: np.ndarray,
    index_5hz: int,
    goal_horizon_steps: int = 50,
) -> tuple[list[float], list[float]]:
    """Derive a point-goal from future odometry (10s ahead at 5Hz).

    - goal_index = min(current_time_stamp + goal_horizon_steps, N-1)
    - goal_xy_robot = rotate(world_delta_xy, -current_yaw)
    - goal_point_scaled = [clip(||goal||,0..100)/100, cos(atan2(y,x)), sin(atan2(y,x))]
    """
    pts = np.asarray(xy_map, dtype=np.float64)
    yaw = np.asarray(yaw_map, dtype=np.float64).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"xy_map must be (N,2), got {pts.shape}")
    if yaw.shape[0] != pts.shape[0]:
        raise ValueError(f"yaw_map length must match xy_map N, got yaw={yaw.shape} xy={pts.shape}")
    N = int(pts.shape[0])
    if N <= 0:
        raise ValueError("empty trajectory")
    i = int(max(0, min(int(index_5hz), N - 1)))
    j = int(min(i + int(goal_horizon_steps), N - 1))

    dx = float(pts[j, 0] - pts[i, 0])
    dy = float(pts[j, 1] - pts[i, 1])
    th = float(yaw[i])
    c = float(np.cos(th))
    s = float(np.sin(th))
    gx = c * dx + s * dy
    gy = -s * dx + c * dy

    dist = float(np.hypot(gx, gy))
    dist_norm = float(np.clip(dist, 0.0, 100.0) / 100.0)
    ang = float(np.arctan2(gy, gx))
    goal_point_scaled = [dist_norm, float(np.cos(ang)), float(np.sin(ang))]
    goal_xy = [float(gx), float(gy)]
    return goal_xy, goal_point_scaled


def write_planner_candidates_from_onnx(
    *,
    dataset: Path,
    episode_id: str | None,
    episode_ids: list[str] | None = None,
    model_path: Path,
    reference_dataset: Path | None = None,
    align_candidate_count_to_reference: bool = True,
    overwrite: bool = False,
    max_frames: int | None = None,
    stride: int = 1,
    max_episodes: int | None = None,
    times_jsonl_path: Path | None = None,
    times_jsonl_time_key: str = "t0",
    times_time_tolerance_s: float = 0.2,
) -> list[PlannerCandidatesOnnxReport]:
    """Generate/overwrite `planner_candidates.jsonl` using the S1 goal-less ONNX planner.

    By default, this writes **raw N=64 candidates per timestep** (matching the prelogged data
    produced
    by `s2e_v2_adapter.py`), and relies on downstream evaluation to apply NMS/top-K.

    If `reference_dataset` is provided and contains prelogged `planner_candidates.jsonl`, we will:
    - copy per-timestep `goal_xy` and `auto_enabled` from it (so overlays/metrics align)
    - optionally align the written candidate count to the reference record candidate count
      (`align_candidate_count_to_reference=True`) to keep candidate indices comparable (0..63).
    """

    dataset_root = dataset.resolve()
    meta = find_episode_metadata_files(dataset_root)
    if not meta:
        raise ValueError(f"No episodes found under: {dataset_root}")

    if episode_id is not None and episode_ids is not None:
        raise ValueError("Pass only one of episode_id or episode_ids (not both).")

    if episode_id is not None:
        wanted = str(episode_id)

        def _epid(pth: Path) -> str:
            return pth.parent.name if pth.name == "episode.json" else pth.stem

        meta = [m for m in meta if _epid(m) == wanted]
        if not meta:
            raise ValueError(f"episode_id not found under {dataset_root}: {wanted}")
    elif episode_ids is not None:
        wanted_set = {str(x) for x in episode_ids if str(x).strip()}

        def _epid(pth: Path) -> str:
            return pth.parent.name if pth.name == "episode.json" else pth.stem

        meta = [m for m in meta if _epid(m) in wanted_set]
        if not meta:
            raise ValueError(
                f"episode_ids not found under {dataset_root}: {sorted(wanted_set)[:5]} ..."
            )
    elif max_episodes is not None:
        meta = meta[: int(max_episodes)]

    planner = OnnxS1Planner(
        OnnxS1PlannerConfig(
            model_path=str(model_path.resolve()),
            max_trajectories=6,
            nms_distance_threshold_m=2.0,
        )
    )
    loader = RGBFrameLoader(cache_size=64)

    reports: list[PlannerCandidatesOnnxReport] = []

    # Optional: restrict inference to specific times (e.g., takeover_clips.jsonl) to save cost.
    # File format: JSONL dict with keys {episode_id: str, <time_key>: float}. Default time_key="t0".
    times_by_episode: dict[str, list[float]] = {}
    if times_jsonl_path is not None:
        tpath = Path(times_jsonl_path).resolve()
        if not tpath.exists():
            raise FileNotFoundError(f"times_jsonl_path not found: {tpath}")
        with tpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                eid = obj.get("episode_id")
                tt = obj.get(times_jsonl_time_key)
                if not isinstance(eid, str) or not isinstance(tt, (int, float)):
                    continue
                times_by_episode.setdefault(str(eid), []).append(float(tt))
        # sort and unique to avoid duplicate work
        for eid in list(times_by_episode.keys()):
            ts = sorted(times_by_episode[eid])
            uniq: list[float] = []
            last = None
            for t0 in ts:
                if last is None or abs(float(t0) - float(last)) > 1e-6:
                    uniq.append(float(t0))
                    last = float(t0)
            times_by_episode[eid] = uniq

    for m in tqdm(meta, desc="Planner episodes", unit="ep", total=len(meta)):
        ep = load_episode(dataset_root, m)
        if not ep.schema_valid:
            raise ValueError(f"Episode schema invalid: {ep.episode_id} errors={ep.schema_errors}")
        if not ep.rgb:
            raise ValueError(f"Episode has no rgb stream: {ep.episode_id}")

        # Optional: load a reference episode (typically the original dataset with prelogged
        # planner_candidates)
        # to copy goal_xy + auto_enabled per timestep.
        ref_by_t: dict[float, Any] = {}
        ref_ts: np.ndarray | None = None
        ref_recs: list[Any] | None = None
        if reference_dataset is not None:
            try:
                ref_root = Path(reference_dataset).resolve()
                ref_meta = find_episode_metadata_files(ref_root)
                wanted = str(ep.episode_id)

                def _epid(pth: Path) -> str:
                    return pth.parent.name if pth.name == "episode.json" else pth.stem

                ref_meta = [mm for mm in ref_meta if _epid(mm) == wanted]
                if ref_meta:
                    ref_ep = load_episode(ref_root, ref_meta[0])
                    if ref_ep.schema_valid and getattr(ref_ep, "planner_candidates", None):
                        tmp: list[tuple[float, Any]] = []
                        for r in ref_ep.planner_candidates:
                            try:
                                tt = float(r.t)
                                ref_by_t[round(tt, 6)] = r
                                tmp.append((tt, r))
                            except Exception:
                                continue
                        if tmp:
                            tmp.sort(key=lambda x: float(x[0]))
                            ref_ts = np.asarray([float(x[0]) for x in tmp], dtype=np.float64)
                            ref_recs = [x[1] for x in tmp]
            except Exception:
                ref_by_t = {}
                ref_ts = None
                ref_recs = None

        def _lookup_ref(
            t: float,
            *,
            ref_by_t: dict[float, Any] = ref_by_t,
            ref_ts: np.ndarray | None = ref_ts,
            ref_recs: list[Any] | None = ref_recs,
        ) -> Any | None:
            # Fast exact match on rounded time (works for dt=0.2 multiples).
            r = ref_by_t.get(round(float(t), 6))
            if r is not None:
                return r
            # Robust fallback for COCO-style floats: nearest neighbor within a tolerance.
            if ref_ts is None or ref_recs is None or ref_ts.size == 0:
                return None
            tt = float(t)
            j = int(np.searchsorted(ref_ts, tt, side="left"))
            cand: list[int] = []
            if 0 <= j < int(ref_ts.size):
                cand.append(j)
            if 0 <= j - 1 < int(ref_ts.size):
                cand.append(j - 1)
            best = None
            best_dt = None
            for k in cand:
                dtt = abs(float(ref_ts[k]) - tt)
                if best_dt is None or dtt < best_dt:
                    best_dt = dtt
                    best = ref_recs[k]
            # tolerance: 50ms is safely below 5Hz dt (0.2s), but handles float noise.
            if best_dt is not None and best_dt <= 0.05:
                return best
            return None

        ep_dir = Path(ep.episode_dir)
        pc_path = ep_dir / "planner_candidates.jsonl"
        if pc_path.exists() and not overwrite:
            raise FileExistsError(f"{pc_path} exists. Pass overwrite=True to replace it.")

        rgb = ep.rgb
        needs_goal_point = "goal_point" in getattr(planner, "_input_names", set())
        odom_xy: np.ndarray | None = None
        odom_yaw: np.ndarray | None = None
        if needs_goal_point and getattr(ep, "odom", None):
            xs = np.asarray([float(r.x) for r in (ep.odom or [])], dtype=np.float64)
            ys = np.asarray([float(r.y) for r in (ep.odom or [])], dtype=np.float64)
            odom_xy = np.stack([xs, ys], axis=1)
            odom_yaw = np.asarray([float(r.yaw) for r in (ep.odom or [])], dtype=np.float64)

        # Model expects a 21-frame observation window.
        T = 21

        # Decide which frame indices/timestamps to evaluate.
        # If `times_jsonl_path` is provided, we evaluate only at those times (per episode) and set
        # record["t"] to the requested time (not necessarily equal to rgb[i].t due to float noise).
        eval_items: list[tuple[int, float]] = []  # (center_rgb_index, requested_time_t)
        if times_by_episode:
            wanted_times = times_by_episode.get(str(ep.episode_id), [])
            if wanted_times:
                rgb_ts = np.asarray([float(r.t) for r in rgb], dtype=np.float64)
                for tt in wanted_times:
                    if rgb_ts.size == 0:
                        continue
                    j = int(np.searchsorted(rgb_ts, float(tt), side="left"))
                    cand = []
                    if 0 <= j < int(rgb_ts.size):
                        cand.append(j)
                    if 0 <= j - 1 < int(rgb_ts.size):
                        cand.append(j - 1)
                    best_i = None
                    best_dt = None
                    for k in cand:
                        dt = abs(float(rgb_ts[k]) - float(tt))
                        if best_dt is None or dt < best_dt:
                            best_dt = dt
                            best_i = int(k)
                    if best_i is None:
                        continue
                    # If the nearest rgb frame is too far, skip (likely mismatched timebase).
                    if best_dt is not None and float(best_dt) > float(times_time_tolerance_s):
                        continue
                    eval_items.append((int(best_i), float(tt)))
                # Stable time order (what downstream expects)
                eval_items.sort(key=lambda x: float(x[1]))

        if not eval_items:
            stride_i = max(1, int(stride))
            # We generate candidates for *all* RGB frames, padding history by repeating the first
            # frame
            # when i < T-1, so timestamps align with prelogged datasets (which include t=0.0,...).
            idxs = list(range(0, len(rgb), stride_i)) if rgb else []
            if max_frames is not None:
                idxs = idxs[: int(max_frames)]
            eval_items = [(int(i), float(rgb[int(i)].t)) for i in idxs]

        lines: list[str] = []
        for i, requested_t in tqdm(
            [(int(ii), float(tt)) for ii, tt in eval_items],
            desc=f"Planner frames {ep.episode_id}",
            unit="frame",
            total=len(eval_items),
            leave=True,
        ):
            # Window indices (length T). If i < T-1, pad by repeating index 0.
            start = int(i) - (T - 1)
            win = [max(0, start + k) for k in range(T)]

            frames_rgb: list[np.ndarray] = []
            for j in win:
                rec = rgb[j]
                # If present, prefer pinhole frames for planner inference while keeping the
                # canonical RGB (`frame_ref`) as the VLM source (typically CP front.mp4).
                extra = getattr(rec, "model_extra", {}) or {}
                pref_ref = extra.get("planner_frame_ref")
                pref_idx = extra.get("planner_frame_index")
                if isinstance(pref_ref, str) and pref_ref.strip() and pref_idx is not None:
                    try:
                        rec = rec.model_copy(
                            update={"frame_ref": str(pref_ref), "frame_index": int(pref_idx)}
                        )
                    except Exception:
                        rec = rgb[j]
                im = loader.load(rec, episode_dir=ep_dir, dataset_root=dataset_root)
                frames_rgb.append(np.asarray(im.convert("RGB"), dtype=np.uint8))

            obs = _prepare_obs_window(
                frames_rgb, out_h=planner.cfg.input_h, out_w=planner.cfg.input_w
            )
            derived_goal_xy: list[float] | None = None
            derived_goal_point_scaled: list[float] | None = None
            metric_spacing = None
            if needs_goal_point:
                # Prefer a goal vector from the reference dataset's planner_candidates record
                # (goal_xy is usually stored as an extension field). This supports datasets without
                # odom.
                t_for_ref = float(requested_t) if requested_t is not None else float(rgb[int(i)].t)
                ref0 = _lookup_ref(t_for_ref)
                goal_xy0 = _extract_goal_xy(ref0) if ref0 is not None else None
                if goal_xy0 is not None:
                    derived_goal_xy = goal_xy0
                    derived_goal_point_scaled = _goal_point_scaled_from_goal_xy(
                        goal_xy_robot=goal_xy0
                    )
                elif odom_xy is not None and odom_yaw is not None:
                    # Fallback: use future executed position (10s ahead at 5Hz) as goal.
                    derived_goal_xy, derived_goal_point_scaled = _goal_from_future_odom(
                        xy_map=odom_xy,
                        yaw_map=odom_yaw,
                        index_5hz=int(i),
                        goal_horizon_steps=50,
                    )
                    metric_spacing = np.asarray([0.0, 0.51, -0.16, 0.16], dtype=np.float32)
                else:
                    raise ValueError(
                        f"Episode {ep.episode_id} is missing both (a) canonical odom stream and "
                        "(b) reference goal_xy in planner_candidates; cannot provide goal_point "
                        "for point-goal ONNX model. Provide odom.jsonl (pose), or pass "
                        "reference_dataset with goal_xy populated, or use a goal-less ONNX model."
                    )

            traj_raw, score_raw = planner.run(
                obs=obs,
                goal_point=(
                    np.asarray(derived_goal_point_scaled, dtype=np.float32)
                    if derived_goal_point_scaled is not None
                    else None
                ),
                metric_spacing=metric_spacing,
            )

            # Write raw candidates (anchor set). Note: some ONNX exports output N=65 instead of
            # N=64.
            # When `reference_dataset` is provided, we align the written candidate count to the
            # reference episode’s candidate count (so indices/plots match your logged dataset).
            traj0 = np.asarray(traj_raw, dtype=np.float64)[0]  # (N,P,2)
            score0 = np.asarray(score_raw, dtype=np.float64)[0].reshape(-1)  # (N,)
            N_model = int(min(traj0.shape[0], score0.shape[0]))

            t = float(requested_t) if requested_t is not None else float(rgb[i].t)
            ref = _lookup_ref(t)
            N_target = N_model
            if bool(align_candidate_count_to_reference):
                try:
                    ref_cands = getattr(ref, "candidates", None) if ref is not None else None
                    # Only align when the reference looks like a *raw anchor set* (e.g. 64),
                    # not when it is a synthetic/top-K set (e.g. 6) used only to carry goal_xy.
                    if ref_cands is not None and len(ref_cands) >= 32:
                        N_target = int(len(ref_cands))
                except Exception:
                    N_target = N_model

            # Alignment behavior:
            # - Many logged datasets store 64 raw candidates (traj_id 0..63).
            # - Some ONNX exports output 65 candidates, where index 0 is effectively a sentinel.
            #   We empirically found that dropping the first candidate (keeping 1..64) matches
            #   the logged raw outputs much better than dropping the last.
            #
            # So when aligning to a reference record and we see (N_model == N_target + 1),
            # we drop the first candidate and reindex to traj_id 0..N_target-1.
            start_k = 0
            # Also: if we see the common ONNX shape N=65, treat index 0 as a sentinel by default,
            # even if no reference match was found.
            if N_model == 65 and N_target >= 64:
                start_k = 1
                N_target = min(int(N_target), 64)
            elif bool(align_candidate_count_to_reference) and N_model == (N_target + 1):
                start_k = 1
            N = int(min(N_model - start_k, N_target))
            candidates: list[dict[str, Any]] = []
            for k in range(N):
                kk = int(start_k + k)
                pts = traj0[kk, :, :2].tolist()
                candidates.append(
                    {"traj_id": str(int(k)), "points_xy": pts, "score": float(score0[kk])}
                )
            goal_xy = _extract_goal_xy(ref) if ref is not None else None
            if goal_xy is None and derived_goal_xy is not None:
                goal_xy = derived_goal_xy
            auto_enabled = getattr(ref, "auto_enabled", None) if ref is not None else None

            rec = {
                "t": t,
                "frame": "robot",
                "auto_enabled": bool(auto_enabled) if auto_enabled is not None else None,
                "postprocess": "raw",
                "candidates": candidates,
                # Debug metadata (extension fields; safe under schema v0).
                "onnx_raw_num_candidates": int(N_model),
                "onnx_written_num_candidates": int(N),
                "onnx_candidate_index_offset": int(start_k),
            }
            if goal_xy is not None:
                rec["goal_xy"] = goal_xy
            if derived_goal_point_scaled is not None:
                rec["goal_point_scaled"] = derived_goal_point_scaled
            lines.append(json.dumps(rec, sort_keys=True))

        pc_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # Ensure episode.json references planner_candidates stream.
        meta_path = ep_dir / "episode.json"
        ep_obj = read_json(meta_path)
        ep_obj.setdefault("schema_version", CANONICAL_EPISODE_SCHEMA_VERSION)
        ep_obj.setdefault("episode_id", ep.episode_id)
        ep_obj.setdefault("streams", {})
        ep_obj["streams"]["planner_candidates"] = {"records_ref": "planner_candidates.jsonl"}
        write_json(meta_path, ep_obj)

        reports.append(
            PlannerCandidatesOnnxReport(
                dataset=str(dataset_root),
                episode_id=str(ep.episode_id),
                model=str(model_path),
                records_written=int(len(lines)),
                out_path=str(pc_path),
            )
        )

    return reports
