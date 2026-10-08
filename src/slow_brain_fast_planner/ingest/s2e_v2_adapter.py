from __future__ import annotations

import csv
import json
import math
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from slow_brain_fast_planner.schema.canonical_episode import CANONICAL_EPISODE_SCHEMA_VERSION
from slow_brain_fast_planner.utils.io import ensure_empty_dir, write_json

_SAMPLE_RE = re.compile(r"^sample_(\d+)\.png$")


@dataclass(frozen=True)
class S2EV2ConversionReport:
    adapter: str
    input_dir: str
    output_dataset_dir: str
    episode_id: str
    samples_found: int
    samples_converted: int
    samples_skipped: int
    skipped_reasons: dict[str, int]
    missing_streams: list[str]
    streams_written: list[str]
    auto_enabled_true: int = 0
    auto_enabled_false: int = 0
    auto_enabled_ratio: float = 0.0
    takeover_events: int = 0
    distance_m: float = 0.0
    distance_auto_m: float = 0.0
    distance_human_m: float = 0.0
    time_auto_s: float = 0.0
    time_human_s: float = 0.0
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _count_takeover_events(auto_enabled: list[bool]) -> int:
    """Count contiguous False segments in an auto_enabled time series.

    We treat `auto_enabled=False` as "human takeover" and count the number of
    takeover segments (i.e., the number of transitions into takeover).
    """

    n = 0
    prev: bool | None = None
    for a in auto_enabled:
        if (not a) and prev is not False:
            n += 1
        prev = bool(a)
    return int(n)


def _calculate_distance(pose_path: Path) -> float:
    """Calculate total path length from a CSV pose file (x,y,z,...)."""
    if not pose_path.exists():
        return 0.0
    dist = 0.0
    try:
        with pose_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # header
            prev_xyz = None
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    curr_xyz = (float(row[0]), float(row[1]), float(row[2]))
                    if prev_xyz is not None:
                        dx = curr_xyz[0] - prev_xyz[0]
                        dy = curr_xyz[1] - prev_xyz[1]
                        dz = curr_xyz[2] - prev_xyz[2]
                        dist += math.sqrt(dx * dx + dy * dy + dz * dz)
                    prev_xyz = curr_xyz
                except (ValueError, IndexError):
                    continue
    except Exception:
        return 0.0
    return float(dist)


def _quat_wxyz_to_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    """Convert quaternion (w,x,y,z) into yaw (rotation about +Z) in radians."""
    # Standard Tait–Bryan yaw (Z axis) from quaternion.
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    t0 = 2.0 * (float(qw) * float(qz) + float(qx) * float(qy))
    t1 = 1.0 - 2.0 * (float(qy) * float(qy) + float(qz) * float(qz))
    return float(math.atan2(t0, t1))


def convert_s2e_v2_folder(
    *,
    input_dir: Path,
    output_dataset_dir: Path,
    episode_id: str | None = None,
    dt_s: float = 1.0,
    copy_images: bool = True,
    overwrite: bool = False,
    write_manifest: bool = True,
    max_samples: int | None = None,
) -> S2EV2ConversionReport:
    """Convert an s2e_v2 sample folder into a canonical episode.

    Input folder layout:
    - sample_i.png
    - sample_i_traj.npy  shape (1, K, T, 2)
    - sample_i_score.npy shape (1, K)
    - sample_i_auto.npy  scalar bool
    - sample_i_goal.npy  shape (1, 2) optional (raw local-planner goal)

    Output dataset layout:
    - <output_dataset_dir>/dataset_manifest.json
    - <output_dataset_dir>/episodes/<episode_id>/episode.json
    - <output_dataset_dir>/episodes/<episode_id>/rgb.jsonl
    - <output_dataset_dir>/episodes/<episode_id>/planner_candidates.jsonl
    - <output_dataset_dir>/episodes/<episode_id>/assets/rgb/*.png (optional)
    """

    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")

    input_dir = input_dir.resolve()
    output_dataset_dir = output_dataset_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    ep_id = episode_id or input_dir.name
    ep_dir = output_dataset_dir / "episodes" / ep_id
    assets_rgb_dir = ep_dir / "assets" / "rgb"

    ensure_empty_dir(ep_dir, overwrite=overwrite)
    if copy_images:
        assets_rgb_dir.mkdir(parents=True, exist_ok=True)

    rgb_jsonl = ep_dir / "rgb.jsonl"
    pc_jsonl = ep_dir / "planner_candidates.jsonl"
    odom_jsonl = ep_dir / "odom.jsonl"

    missing_streams = ["gps", "odom", "route", "control"]
    streams_written = ["rgb", "planner_candidates"]

    skipped_reasons: Counter[str] = Counter()

    # Find samples deterministically (NUMERIC order, not lexicographic).
    # Lexicographic ordering yields sample_100 before sample_11, which breaks timestamps and
    # distance.
    pngs = [p for p in input_dir.iterdir() if p.is_file() and _SAMPLE_RE.match(p.name)]
    pngs.sort(key=lambda p: int(_SAMPLE_RE.match(p.name).group(1)))  # type: ignore[union-attr]
    if max_samples is not None:
        if int(max_samples) <= 0:
            raise ValueError("max_samples must be > 0 if set")
        pngs = pngs[: int(max_samples)]
    samples_found = len(pngs)

    rgb_lines: list[str] = []
    pc_lines: list[str] = []
    odom_lines: list[str] = []
    auto_enabled_series: list[bool] = []

    # Metrics.
    total_dist = 0.0
    auto_dist = 0.0
    human_dist = 0.0
    auto_time = 0.0
    human_time = 0.0
    prev_xy: tuple[float, float] | None = None

    for png_path in tqdm(pngs, total=len(pngs), desc=f"convert:{ep_id}"):
        m = _SAMPLE_RE.match(png_path.name)
        assert m is not None
        i = int(m.group(1))
        t = float(i) * dt_s

        traj_path = input_dir / f"sample_{i}_traj.npy"
        score_path = input_dir / f"sample_{i}_score.npy"
        auto_path = input_dir / f"sample_{i}_auto.npy"
        goal_path = input_dir / f"sample_{i}_goal.npy"
        pose_path = input_dir / f"sample_{i}_pose.npy"

        missing = [p for p in [traj_path, score_path, auto_path] if not p.exists()]
        if missing:
            skipped_reasons["missing_files"] += 1
            continue

        try:
            width, height = _png_size(png_path)
        except Exception:
            skipped_reasons["invalid_png"] += 1
            continue

        try:
            traj = np.load(traj_path)
            score = np.load(score_path)
            auto = np.load(auto_path)
            goal = np.load(goal_path) if goal_path.exists() else None
            pose = np.load(pose_path) if pose_path.exists() else None
        except Exception:
            skipped_reasons["npy_load_error"] += 1
            continue

        if traj.ndim != 4 or traj.shape[0] != 1 or traj.shape[-1] != 2:
            skipped_reasons["bad_traj_shape"] += 1
            continue
        if score.ndim != 2 or score.shape[0] != 1:
            skipped_reasons["bad_score_shape"] += 1
            continue

        traj_k = traj[0]  # (K, T, 2)
        score_k = score[0]  # (K,)

        if traj_k.shape[0] != score_k.shape[0]:
            skipped_reasons["k_mismatch_traj_vs_score"] += 1
            continue

        # auto.npy is expected to be a scalar bool (but accept numpy scalar / 0-d arrays).
        try:
            auto_enabled = bool(np.asarray(auto).item())
        except Exception:
            skipped_reasons["bad_auto_value"] += 1
            continue

        # Distance and time tracking.
        if auto_enabled:
            auto_time += dt_s
        else:
            human_time += dt_s

        if pose is not None:
            try:
                p = np.asarray(pose).reshape(-1)
                curr_xy = (float(p[0]), float(p[1]))
                # pose format written by NavFlow logger:
                #   [x, y, z, qw, qx, qy, qz]
                if p.size >= 7:
                    yaw = _quat_wxyz_to_yaw(float(p[3]), float(p[4]), float(p[5]), float(p[6]))
                    odom_lines.append(
                        json.dumps(
                            {
                                "t": float(t),
                                "x": float(p[0]),
                                "y": float(p[1]),
                                "yaw": float(yaw),
                                # This is a world-ish local frame (local odom), not robot body
                                # frame.
                                "frame": "map_enu",
                            },
                            sort_keys=True,
                        )
                    )
                if prev_xy is not None:
                    dx = curr_xy[0] - prev_xy[0]
                    dy = curr_xy[1] - prev_xy[1]
                    step_dist = math.sqrt(dx * dx + dy * dy)

                    # Heuristic to filter out large jumps (teleports / resets).
                    # Only accumulate "reasonable" per-step displacement.
                    if step_dist < 10.0:
                        total_dist += step_dist
                        if auto_enabled:
                            auto_dist += step_dist
                        else:
                            human_dist += step_dist
                    else:
                        print(
                            f"Large jump: {step_dist}m in {dt_s}s. Last pose: {prev_xy}, "
                            f"current pose: {curr_xy}, index: {i}, png_path: {png_path}"
                        )
                        skipped_reasons["pose_jump_filtered"] += 1
                prev_xy = curr_xy
            except Exception:
                pass

        # Optional goal: stored in the raw logs as a (1,2) vector, but in a different frame.
        # Convert to the robot frame (x forward, y left):
        #   goal_xy_robot = (goal_y, -goal_x)
        goal_xy: list[float] | None = None
        if goal is not None:
            try:
                g = np.asarray(goal, dtype=np.float64).reshape(-1)
                if g.size >= 2:
                    gx = float(g[0])
                    gy = float(g[1])
                    goal_xy = [float(gy), float(-gx)]
            except Exception:
                # Goal is optional; ignore malformed values.
                goal_xy = None

        # Frame refs: make output self-contained by copying images (default).
        if copy_images:
            out_png_rel = Path("assets/rgb") / png_path.name
            shutil.copy2(png_path, ep_dir / out_png_rel)
            frame_ref = str(out_png_rel)
        else:
            frame_ref = str(png_path)

        rgb_record = {
            "t": t,
            "frame_ref": frame_ref,
            "width": int(width),
            "height": int(height),
            "camera_id": "front",
        }
        rgb_lines.append(json.dumps(rgb_record, sort_keys=True))

        candidates: list[dict[str, Any]] = []
        K = int(traj_k.shape[0])
        for k in range(K):
            points_xy = traj_k[k].tolist()  # (T, 2)
            candidates.append(
                {
                    "traj_id": str(k),
                    "points_xy": points_xy,
                    "score": float(score_k[k]),
                }
            )

        pc_record = {
            "t": t,
            "frame": "robot",
            "auto_enabled": auto_enabled,
            "postprocess": "raw",
            "candidates": candidates,
        }
        if goal_xy is not None:
            # Stored as an extra field (schema allows extension fields).
            pc_record["goal_xy"] = goal_xy
        pc_lines.append(json.dumps(pc_record, sort_keys=True))
        auto_enabled_series.append(bool(auto_enabled))

    rgb_jsonl.write_text("\n".join(rgb_lines) + ("\n" if rgb_lines else ""), encoding="utf-8")
    pc_jsonl.write_text("\n".join(pc_lines) + ("\n" if pc_lines else ""), encoding="utf-8")
    if odom_lines:
        odom_jsonl.write_text("\n".join(odom_lines) + "\n", encoding="utf-8")
        streams_written.append("odom")
        if "odom" in missing_streams:
            missing_streams.remove("odom")

    auto_true = int(sum(1 for a in auto_enabled_series if bool(a)))
    auto_false = int(len(auto_enabled_series) - auto_true)
    auto_ratio = float(auto_true) / float(len(auto_enabled_series)) if auto_enabled_series else 0.0
    takeover_events = _count_takeover_events(auto_enabled_series)

    # Note: we no longer use the global .txt file for distance, as it can contain jumps.
    # We use the per-sample pose tracking calculated during the loop.
    distance_m = float(total_dist)

    episode_obj: dict[str, Any] = {
        "schema_version": CANONICAL_EPISODE_SCHEMA_VERSION,
        "episode_id": ep_id,
        "streams": {
            "rgb": {"records_ref": "rgb.jsonl"},
            "planner_candidates": {"records_ref": "planner_candidates.jsonl"},
            **({"odom": {"records_ref": "odom.jsonl"}} if odom_lines else {}),
        },
        # Extension fields (safe in v0: schema allows extra fields).
        "stats": {
            "planner_candidates_total": int(len(auto_enabled_series)),
            "auto_enabled_true": int(auto_true),
            "auto_enabled_false": int(auto_false),
            "auto_enabled_ratio": float(auto_ratio),
            "takeover_events": int(takeover_events),
            "distance_m": float(distance_m),
            "distance_auto_m": float(auto_dist),
            "distance_human_m": float(human_dist),
            "time_auto_s": float(auto_time),
            "time_human_s": float(human_time),
        },
        "quality": {
            "missing_streams": missing_streams,
            "notes": (
                "Converted from s2e_v2 folder example_data; timestamps are sample_index * dt_s. "
                "If sample_<i>_pose.npy is present, we write canonical odom.jsonl and GT metrics "
                "use executed future odom transformed into robot local frame."
            ),
        },
    }

    (ep_dir / "episode.json").write_text(
        json.dumps(episode_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if write_manifest:
        output_dataset_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dataset_dir / "dataset_manifest.json"

        manifest_obj: dict[str, Any]
        if manifest_path.exists():
            try:
                manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest_obj = {}
        else:
            manifest_obj = {}

        existing_schema_version = manifest_obj.get("schema_version")
        if (
            existing_schema_version is not None
            and existing_schema_version != CANONICAL_EPISODE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Existing dataset_manifest.json schema_version={existing_schema_version!r} "
                "does not match "
                f"expected {CANONICAL_EPISODE_SCHEMA_VERSION!r}"
            )

        episodes = manifest_obj.get("episodes")
        if not isinstance(episodes, list):
            episodes = []
        if ep_id not in episodes:
            episodes.append(ep_id)

        manifest_obj.setdefault("dataset_name", "s2e_v2_converted")
        manifest_obj.setdefault("dataset_version", "0.0.0")
        manifest_obj["schema_version"] = CANONICAL_EPISODE_SCHEMA_VERSION
        manifest_obj["episodes"] = sorted({str(e) for e in episodes})

        write_json(manifest_path, manifest_obj)

    samples_converted = len(rgb_lines)
    samples_skipped = samples_found - samples_converted

    return S2EV2ConversionReport(
        adapter="s2e_v2_folder",
        input_dir=str(input_dir),
        output_dataset_dir=str(output_dataset_dir),
        episode_id=ep_id,
        samples_found=samples_found,
        samples_converted=samples_converted,
        samples_skipped=samples_skipped,
        skipped_reasons={k: skipped_reasons[k] for k in sorted(skipped_reasons)},
        missing_streams=missing_streams,
        streams_written=streams_written,
        auto_enabled_true=int(auto_true),
        auto_enabled_false=int(auto_false),
        auto_enabled_ratio=float(auto_ratio),
        takeover_events=int(takeover_events),
        distance_m=float(distance_m),
        distance_auto_m=float(auto_dist),
        distance_human_m=float(human_dist),
        time_auto_s=float(auto_time),
        time_human_s=float(human_time),
    )
