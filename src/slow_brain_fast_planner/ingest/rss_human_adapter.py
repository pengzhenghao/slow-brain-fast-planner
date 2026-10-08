from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from slow_brain_fast_planner.schema.canonical_episode import (
    CANONICAL_EPISODE_SCHEMA_VERSION,
    PlannerCandidate,
)
from slow_brain_fast_planner.utils.io import ensure_empty_dir, write_json

_TIME_RE = re.compile(r"^time_(\d+)$")

_SAMPLE_PNG_RE = re.compile(r"^sample_(\d+)\.png$")


def _repo_root() -> Path:
    # This module lives under src/<package>/ingest/.
    return Path(__file__).resolve().parents[3]


def _png_size(path: Path) -> tuple[int, int]:
    """Return (width, height) from a PNG header without extra deps."""
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG with IHDR header: {path}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid PNG dimensions ({width}, {height}) for {path}")
    return width, height


def _resolve_source_png(*, source_root: Path, scenario: str, time_index: int) -> Path | None:
    """Resolve a high-res source frame for (scenario, time_index) in RSS_Human_Data."""
    base = (Path(source_root).resolve() / str(scenario)).resolve()
    if not base.exists():
        return None

    # Most common: sample_<i>.png (no zero pad).
    cand = base / f"sample_{int(time_index)}.png"
    if cand.exists():
        return cand

    # Rare: zero-padded.
    cand2 = base / f"sample_{int(time_index):06d}.png"
    if cand2.exists():
        return cand2

    # Best-effort: other common extensions.
    for ext in (".jpg", ".jpeg", ".webp"):
        c = base / f"sample_{int(time_index)}{ext}"
        if c.exists():
            return c
    return None


@dataclass(frozen=True)
class RSSHumanConversionReport:
    adapter: str
    input_dir: str
    output_dataset_dir: str
    episodes_found: int
    episodes_converted: int
    episodes_skipped: int
    skipped_reasons: dict[str, int]
    dt_s: float
    rgb_frames_per_episode: int
    horizon_points: int
    candidates_k: int
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_segment_dir(d: Path) -> bool:
    """Return True iff d looks like RSS_Human_Data_Processed/{scenario}/time_xxxxx."""
    if not d.is_dir():
        return False
    # Required files (author said unified format).
    req = [
        d / "obs_255_HWC_RGB_seq.npy",
        d / "goal_Xforward_Yleft.npy",
        d / "positive_behavior.npy",
    ]
    return all(p.exists() for p in req)


def iter_rss_segments(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield (episode_id, segment_dir) for an RSS_Human_Data_Processed root or scenario folder.

    Accepted layouts:
    - root/{scenario}/time_xxxxx/{files...}
    - root/time_xxxxx/{files...}   (single scenario)
    """
    root = Path(root).resolve()
    if not root.is_dir():
        return

    # Case A: root is a scenario dir with time_* children
    time_children = [c for c in root.iterdir() if c.is_dir() and _TIME_RE.match(c.name)]
    if time_children:
        scenario = root.name
        for seg in sorted(time_children, key=lambda p: int(_TIME_RE.match(p.name).group(1))):  # type: ignore[union-attr]
            if _is_segment_dir(seg):
                yield f"{scenario}_{seg.name}", seg
        return

    # Case B: root is a dataset root with scenario children
    for scenario_dir in sorted([c for c in root.iterdir() if c.is_dir()], key=lambda p: p.name):
        time_dirs = [c for c in scenario_dir.iterdir() if c.is_dir() and _TIME_RE.match(c.name)]
        if not time_dirs:
            continue
        for seg in sorted(time_dirs, key=lambda p: int(_TIME_RE.match(p.name).group(1))):  # type: ignore[union-attr]
            if _is_segment_dir(seg):
                yield f"{scenario_dir.name}_{seg.name}", seg


def _load_static_candidates(static_candidates_json: Path) -> list[PlannerCandidate]:
    obj = json.loads(static_candidates_json.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("candidates"), list):
        raise ValueError(f"Invalid static candidates JSON: {static_candidates_json}")
    out: list[PlannerCandidate] = []
    for i, c in enumerate(obj["candidates"]):
        if not isinstance(c, dict):
            continue
        pts = c.get("points_xy")
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        score = c.get("score", 0.0)
        out.append(
            PlannerCandidate(
                traj_id=str(c.get("traj_id", i)),
                points_xy=[
                    [float(p[0]), float(p[1])]
                    for p in pts
                    if isinstance(p, (list, tuple)) and len(p) >= 2
                ],
                score=float(score) if isinstance(score, (int, float)) else 0.0,
            )
        )
    if not out:
        raise ValueError(f"No valid candidates parsed from: {static_candidates_json}")
    return out


def convert_rss_human_data_processed(
    *,
    input_root: Path,
    output_dataset_dir: Path,
    dt_s: float = 0.2,
    rgb_frames_per_episode: int = 21,
    static_candidates_json: Path | None = None,
    source_rgb_root: Path | None = None,
    overwrite: bool = False,
    limit_episodes: int | None = None,
) -> RSSHumanConversionReport:
    """Convert RSS_Human_Data_Processed into canonical episodes.

    For each segment:
    - We write a canonical episode with:
      - rgb.jsonl:
        - if `source_rgb_root` is available, reference the high-res PNG
        `RSS_Human_Data/<scenario>/sample_<i>.png`
          (mapped from segment name `time_<i>`), and reuse it for all `rgb_frames_per_episode`
          timestamps.
        - otherwise, fall back to referencing the segment's `obs_255_HWC_RGB_seq.npy` via extension
        field
          `frame_index` (no PNG explosion, but lower resolution).
      - planner_candidates.jsonl: a single record at t = (N-1)*dt_s with a *static* candidate set.
      - odom.jsonl: a synthetic map-frame odom constructed so that the executed future equals
      `positive_behavior.npy`
        (thereby enabling GT trajectory-selection metrics without requiring raw pose logs).
    """
    if float(dt_s) <= 0:
        raise ValueError("dt_s must be > 0")
    if int(rgb_frames_per_episode) <= 0:
        raise ValueError("rgb_frames_per_episode must be > 0")

    input_root = Path(input_root).resolve()
    output_dataset_dir = Path(output_dataset_dir).resolve()
    episodes_dir = output_dataset_dir / "episodes"

    if static_candidates_json is None:
        static_candidates_json = (
            _repo_root()
            / "assets"
            / "trajectory_selection_static_candidates"
            / "takeover_kmeans_medoids"
            / "static_candidates_k24.json"
        )
    static_candidates_json = Path(static_candidates_json).resolve()
    candidates = _load_static_candidates(static_candidates_json)
    K = int(len(candidates))
    candidate_horizon_points = (
        int(len(candidates[0].points_xy)) if candidates and candidates[0].points_xy else 0
    )
    if candidate_horizon_points <= 0:
        raise ValueError("static candidate set has empty points_xy")

    # Optional source root: the "real-world" raw dataset with high-res PNGs.
    if source_rgb_root is not None:
        source_rgb_root = Path(source_rgb_root).resolve()
        if not source_rgb_root.exists():
            raise FileNotFoundError(
                f"source_rgb_root not found: {source_rgb_root}. Provide the RSS_Human_Data root "
                "containing <scenario>/sample_<i>.png frames, or omit it to fall back to the "
                "low-res obs_255_HWC_RGB_seq.npy frames."
            )

    segs = list(iter_rss_segments(input_root))
    if not segs:
        raise ValueError(f"No RSS segments found under: {input_root}")
    if limit_episodes is not None:
        if int(limit_episodes) <= 0:
            raise ValueError("limit_episodes must be > 0 if set")
        segs = segs[: int(limit_episodes)]

    if output_dataset_dir.exists():
        if bool(overwrite):
            shutil.rmtree(output_dataset_dir)
        elif any(output_dataset_dir.iterdir()):
            raise FileExistsError(
                f"Output dataset is not empty: {output_dataset_dir}. "
                "Pass overwrite=True to replace it."
            )
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)

    skipped: dict[str, int] = {}
    converted = 0
    converted_episode_ids: list[str] = []

    for episode_id, seg_dir in tqdm(segs, total=len(segs), desc="rss_human:convert"):
        ep_dir = episodes_dir / str(episode_id)
        try:
            ensure_empty_dir(ep_dir, overwrite=bool(overwrite))
            obs_path = (seg_dir / "obs_255_HWC_RGB_seq.npy").resolve()
            goal_path = (seg_dir / "goal_Xforward_Yleft.npy").resolve()
            pos_path = (seg_dir / "positive_behavior.npy").resolve()

            # Resolve high-res source image from RSS_Human_Data (preferred).
            scenario = str(seg_dir.parent.name)
            m = _TIME_RE.match(str(seg_dir.name))
            time_index = int(m.group(1)) if m else None
            src_png: Path | None = None
            if source_rgb_root is not None and time_index is not None:
                src_png = _resolve_source_png(
                    source_root=Path(source_rgb_root),
                    scenario=scenario,
                    time_index=int(time_index),
                )

            # Decide RGB backing:
            # - If src_png exists, use it and reuse same image for all timestamps.
            # - Else fall back to low-res npy sequence with frame_index.
            use_png = src_png is not None and src_png.exists()
            frame_ref: str
            frame_indices: list[int] | None = None
            if use_png:
                W, H = _png_size(src_png)  # type: ignore[arg-type]
                frame_ref = str(src_png)  # absolute
            else:
                # Load lightweight metadata (shapes) and sanity-check.
                obs = np.load(obs_path, mmap_mode="r")
                if obs.ndim != 4 or obs.shape[-1] != 3:
                    raise ValueError(f"bad_obs_shape:{tuple(obs.shape)}")
                T_total, H, W = int(obs.shape[0]), int(obs.shape[1]), int(obs.shape[2])
                if T_total < int(rgb_frames_per_episode):
                    raise ValueError(
                        f"insufficient_obs_frames:{T_total}<{int(rgb_frames_per_episode)}"
                    )
                # Use the *last* rgb_frames_per_episode frames as the history window.
                start_idx = int(T_total - int(rgb_frames_per_episode))
                frame_indices = list(range(start_idx, start_idx + int(rgb_frames_per_episode)))
                frame_ref = str(obs_path)  # absolute .npy

            goal = np.asarray(np.load(goal_path), dtype=np.float64).reshape(-1)
            if goal.size < 2:
                raise ValueError("bad_goal_shape")
            goal_xy = [float(goal[0]), float(goal[1])]

            expert = np.asarray(np.load(pos_path), dtype=np.float64)
            if expert.ndim != 2 or expert.shape[1] < 2:
                raise ValueError("bad_positive_behavior_shape")
            expert_xy_full = [[float(p[0]), float(p[1])] for p in expert.tolist()]

            # GT extraction uses the candidate horizon length, so we align the expert horizon to
            # that.
            horizon_points = int(candidate_horizon_points)
            if len(expert_xy_full) >= int(horizon_points):
                expert_xy = expert_xy_full[: int(horizon_points)]
            else:
                # Pad with last point (best-effort) if expert is shorter than the candidate horizon.
                last = expert_xy_full[-1] if expert_xy_full else [0.0, 0.0]
                expert_xy = expert_xy_full + [
                    list(last) for _ in range(int(horizon_points) - len(expert_xy_full))
                ]

            # rgb stream:
            # - high-res PNG: reuse same image for all times (fast, matches legacy 960px width)
            # - fallback: reference the .npy via frame_index (extension field).
            rgb_lines: list[str] = []
            for j in range(int(rgb_frames_per_episode)):
                t = float(j) * float(dt_s)
                rec = {
                    "t": float(t),
                    "frame_ref": str(frame_ref),
                    "width": int(W),
                    "height": int(H),
                    "camera_id": "front",
                }
                if not use_png:
                    assert frame_indices is not None
                    rec["frame_index"] = int(frame_indices[int(j)])
                rgb_lines.append(json.dumps(rec, sort_keys=True))
            (ep_dir / "rgb.jsonl").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")

            # planner candidates: one record at the final history time.
            t0 = float(int(rgb_frames_per_episode) - 1) * float(dt_s)
            pc_record = {
                "t": float(t0),
                "frame": "robot",
                "auto_enabled": None,  # unknown for this curated dataset
                "postprocess": "static_candidates",
                "candidates": [
                    {
                        "traj_id": (c.traj_id if c.traj_id is not None else str(i)),
                        "points_xy": c.points_xy,
                        "score": float(c.score),
                    }
                    for i, c in enumerate(candidates)
                ],
                "goal_xy": goal_xy,
                # Adapter extension: store expert GT in local robot frame (x forward, y left).
                "expert_future_xy": expert_xy,
            }
            (ep_dir / "planner_candidates.jsonl").write_text(
                json.dumps(pc_record, sort_keys=True) + "\n", encoding="utf-8"
            )

            # Synthetic odom: make the executed future match expert_future_xy.
            # We build a map_enu odom where:
            # - at time t0, pose is at (0,0) with yaw=0
            # - at times t0 + k*dt, pose is at expert_xy[k-1]
            # This makes the GT local future equal expert_xy under the repo's GT extraction logic.
            odom_lines: list[str] = []
            max_k = int(horizon_points)
            # Provide a small pre-t0 prefix (stationary) to make interpolation robust.
            pre_times = list(range(0, int(rgb_frames_per_episode)))  # 0..N-1 inclusive
            for j in pre_times:
                tt = float(j) * float(dt_s)
                odom_lines.append(
                    json.dumps(
                        {"t": float(tt), "x": 0.0, "y": 0.0, "yaw": 0.0, "frame": "map_enu"},
                        sort_keys=True,
                    )
                )
            for k in range(1, max_k + 1):
                tt = float(t0) + float(k) * float(dt_s)
                xk, yk = expert_xy[k - 1]
                odom_lines.append(
                    json.dumps(
                        {
                            "t": float(tt),
                            "x": float(xk),
                            "y": float(yk),
                            "yaw": 0.0,
                            "frame": "map_enu",
                        },
                        sort_keys=True,
                    )
                )
            (ep_dir / "odom.jsonl").write_text("\n".join(odom_lines) + "\n", encoding="utf-8")

            episode_obj: dict[str, Any] = {
                "schema_version": CANONICAL_EPISODE_SCHEMA_VERSION,
                "episode_id": str(episode_id),
                "streams": {
                    "rgb": {"records_ref": "rgb.jsonl"},
                    "planner_candidates": {"records_ref": "planner_candidates.jsonl"},
                    "odom": {"records_ref": "odom.jsonl"},
                },
                "stats": {
                    "planner_candidates_total": 1,
                    "rgb_frames_total": int(rgb_frames_per_episode),
                    "static_candidates_json": str(static_candidates_json),
                    "static_candidates_k": int(K),
                    "expert_future_points": int(horizon_points),
                    "dt_s": float(dt_s),
                    "rss_segment_dir": str(seg_dir),
                    "source_rgb_root": (
                        str(source_rgb_root) if source_rgb_root is not None else None
                    ),
                    "source_rgb_frame_ref": (
                        str(src_png) if use_png and src_png is not None else None
                    ),
                },
                "quality": {
                    "missing_streams": ["gps", "route", "control"],
                    "notes": (
                        "Converted from RSS_Human_Data_Processed segments. RGB frames prefer "
                        "high-res "
                        "RSS_Human_Data/<scenario>/sample_<i>.png (mapped from time_<i>), "
                        "otherwise fall back "
                        "to obs_255_HWC_RGB_seq.npy with frame_index. Odom is synthetic so that "
                        "the executed "
                        "GT future equals positive_behavior (expert_future_xy)."
                    ),
                },
            }
            (ep_dir / "episode.json").write_text(
                json.dumps(episode_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            converted += 1
            converted_episode_ids.append(str(episode_id))
        except Exception as e:  # noqa: BLE001
            reason = str(e).split("\n")[0][:200]
            skipped[reason] = int(skipped.get(reason, 0)) + 1
            # Best-effort cleanup of partial episode folder.
            try:
                if ep_dir.exists():
                    shutil.rmtree(ep_dir)
            except Exception:
                pass
            continue

    manifest = {
        "dataset_name": "rss_human_data_processed",
        "dataset_version": "2026-01-22",
        "schema_version": CANONICAL_EPISODE_SCHEMA_VERSION,
        "episodes": sorted(set(converted_episode_ids)),
        "adapter": "rss_human_data_processed",
        "input_root": str(input_root),
        "notes": (
            "Per-segment episodes; GT future is derived from positive_behavior.npy "
            "via synthetic odom."
        ),
    }
    write_json(output_dataset_dir / "dataset_manifest.json", manifest)

    return RSSHumanConversionReport(
        adapter="rss_human_data_processed",
        input_dir=str(input_root),
        output_dataset_dir=str(output_dataset_dir),
        episodes_found=int(len(segs)),
        episodes_converted=int(converted),
        episodes_skipped=int(len(segs) - converted),
        skipped_reasons={k: int(skipped[k]) for k in sorted(skipped)},
        dt_s=float(dt_s),
        rgb_frames_per_episode=int(rgb_frames_per_episode),
        horizon_points=int(candidate_horizon_points),
        candidates_k=int(K),
        notes=(
            "RGB frames prefer high-res RSS_Human_Data/<scenario>/sample_<i>.png (mapped from "
            "time_<i>) "
            "when available; otherwise fall back to obs_255_HWC_RGB_seq.npy with frame_index. "
            "No PNGs are written into the processed dataset."
        ),
    )


def write_trajectory_selection_eval_clips_one_per_episode(
    *,
    dataset_root: Path,
    overwrite: bool = False,
    history_window_s: float = 4.0,
    horizon_s: float = 2.0,
    k_frames: int = 4,
) -> dict[str, Any]:
    """Write one label-free trajectory-selection snapshot per episode.

    For datasets without takeover annotations (e.g. RSS_Human_Data_Processed, where each
    micro-episode is a single planner snapshot), this emits ONE clip per episode at the
    episode's first planner_candidates timestamp `t0`, producing:
      <dataset>/trajectory_selection_eval_clips/trajectory_selection_eval_clips.jsonl
      <dataset>/trajectory_selection_eval_clips/summary.json

    The evaluation CLI auto-detects this path. Records intentionally omit takeover labels
    and phases because this source dataset has no takeover annotations.
    """
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    eval_dir = (dataset_root / "trajectory_selection_eval_clips").resolve()
    if eval_dir.exists():
        if bool(overwrite):
            shutil.rmtree(eval_dir)
        elif any(eval_dir.iterdir()):
            raise FileExistsError(
                f"Evaluation clip directory is not empty: {eval_dir}. "
                "Pass overwrite=True to replace it."
            )
    eval_dir.mkdir(parents=True, exist_ok=True)
    clips_path = eval_dir / "trajectory_selection_eval_clips.jsonl"
    summary_path = eval_dir / "summary.json"

    episodes_dir = (dataset_root / "episodes").resolve()
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Missing episodes/ under dataset: {dataset_root}")

    clips_written = 0
    episodes_seen = 0
    skipped: dict[str, int] = {}

    with clips_path.open("w", encoding="utf-8") as f:
        for ep_dir in sorted(
            [p for p in episodes_dir.iterdir() if p.is_dir()], key=lambda p: p.name
        ):
            episodes_seen += 1
            pc_path = ep_dir / "planner_candidates.jsonl"
            if not pc_path.exists():
                skipped["missing_planner_candidates_jsonl"] = (
                    skipped.get("missing_planner_candidates_jsonl", 0) + 1
                )
                continue
            try:
                first = pc_path.read_text(encoding="utf-8").splitlines()[0].strip()
                obj = json.loads(first)
                t0 = float(obj.get("t"))
            except Exception:
                skipped["bad_planner_candidates_jsonl"] = (
                    skipped.get("bad_planner_candidates_jsonl", 0) + 1
                )
                continue

            rec = {
                "episode_id": str(ep_dir.name),
                "t0": float(t0),
                "history_window_s": float(history_window_s),
                "horizon_s": float(horizon_s),
                "k_frames": int(k_frames),
                # Optional; evaluation resolves history frames from rgb.jsonl at t0.
                "frames": [],
            }
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            clips_written += 1

    summary = {
        "dataset": str(dataset_root),
        "out_dir": str(eval_dir),
        "clips_path": str(clips_path),
        "clips_written": int(clips_written),
        "episodes_loaded": int(episodes_seen),
        "skipped_clips": skipped,
        "config": {
            "history_window_s": float(history_window_s),
            "horizon_s": float(horizon_s),
            "k_frames": int(k_frames),
        },
        "notes": "One label-free evaluation clip per episode at its planner snapshot time.",
    }
    write_json(summary_path, summary)
    return summary
