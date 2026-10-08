"""PyTorch Dataset/DataLoader utilities for trajectory-selection evaluation.

This module provides a lightweight, worker-friendly interface around snapshot evaluation:
- **`SnapshotDataset`**: wraps `SnapshotJob` -> `prepare_snapshot(...)`
- **Sharding samplers**: shard by episode or snapshot (for multi-process/GPU runs)
- **Job builders**: enumerate snapshot times (all ticks or takeover-clip t0s)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Dataset, Sampler

from slow_brain_fast_planner.benchmarks.overlays import OverlayConfig
from slow_brain_fast_planner.benchmarks.planner_postprocessing import A0Config
from slow_brain_fast_planner.benchmarks.sharding import shard_id_for_key
from slow_brain_fast_planner.benchmarks.trajectory_selection_workers import (
    SnapshotJob,
    SnapshotPrepResult,
    prepare_snapshot,
)
from slow_brain_fast_planner.utils.trajectory_utils import (
    GoalFilterConfig,
    GtFutureFilterConfig,
    extract_goal_info,
    goal_filter_reason,
)

# ──────────────────────────────────────────────────────────────────────────────
# Trajectory selection: snapshot dataset
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class SnapshotItem:
    """A single snapshot item for trajectory-selection evaluation."""

    prep: SnapshotPrepResult


class SnapshotDataset(Dataset):
    """PyTorch Dataset for trajectory-selection snapshots.

    Each item corresponds to one snapshot (planner candidate record at time t).
    Data preparation (overlay generation, GT metrics, etc.) happens in workers.
    """

    def __init__(
        self,
        jobs: list[SnapshotJob],
    ):
        self.jobs = jobs

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, idx: int) -> SnapshotItem:
        job = self.jobs[idx]
        result = prepare_snapshot(job)
        return SnapshotItem(prep=result)

    @staticmethod
    def collate_fn(batch: list[SnapshotItem]) -> list[SnapshotItem]:
        """Identity collate - return list of items (no tensor batching)."""
        return batch


# ──────────────────────────────────────────────────────────────────────────────
# Sharding / Distributed Sampler utilities
# ──────────────────────────────────────────────────────────────────────────────


class EpisodeShardedSampler(Sampler):
    """Sampler that shards by episode_id for distributed evaluation.

    Unlike PyTorch's DistributedSampler which shards by index, this sampler
    ensures that all items from the same episode go to the same worker.
    This enables clean per-episode metrics aggregation.
    """

    def __init__(
        self,
        items: list[dict[str, Any]],
        num_shards: int = 1,
        shard_id: int = 0,
        episode_key: str = "episode_id",
        mode: Literal[
            "episode_round_robin", "episode_hash", "episode_greedy"
        ] = "episode_round_robin",
    ):
        self.items = items
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.episode_key = episode_key
        self.mode = mode

        # Group indices by episode
        episode_to_indices: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            eid = str(item.get(episode_key, ""))
            if eid not in episode_to_indices:
                episode_to_indices[eid] = []
            episode_to_indices[eid].append(i)

        # Shard episodes (deterministic)
        all_episodes = sorted(episode_to_indices.keys())
        if int(num_shards) <= 1:
            my_episodes = all_episodes
        elif self.mode == "episode_hash":
            my_episodes = [
                e
                for e in all_episodes
                if shard_id_for_key(str(e), num_shards=int(num_shards)) == int(shard_id)
            ]
        elif self.mode == "episode_greedy":
            # Load-balance by item count while keeping episodes intact (LPT bin packing).
            # Deterministic: sort by (count desc, episode_id asc).
            eps = sorted(
                all_episodes,
                key=lambda e: (-len(episode_to_indices.get(e, [])), str(e)),
            )
            bins: list[list[str]] = [[] for _ in range(int(num_shards))]
            loads: list[int] = [0 for _ in range(int(num_shards))]
            for e in eps:
                k = min(range(int(num_shards)), key=lambda j: (loads[j], j))
                bins[k].append(e)
                loads[k] += int(len(episode_to_indices.get(e, [])))
            my_episodes = bins[int(shard_id) % int(num_shards)]
        else:  # episode_round_robin
            my_episodes = [e for i, e in enumerate(all_episodes) if i % num_shards == shard_id]

        # Flatten to indices
        self.indices: list[int] = []
        for eid in my_episodes:
            self.indices.extend(episode_to_indices[eid])

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class SnapshotShardedSampler(Sampler):
    """Sampler for sharded evaluation.

    Default sharding keeps whole episodes together, but this can be imbalanced when episodes
    contain very different numbers of snapshots. For VLM runs (network-bound), consider:
      - mode="episode_greedy" (balanced, still keeps episodes intact)
      - mode="snapshot_hash"  (best balance, but can split episodes)
    """

    def __init__(
        self,
        jobs: list[SnapshotJob],
        num_shards: int = 1,
        shard_id: int = 0,
        mode: Literal[
            "episode_round_robin", "episode_hash", "episode_greedy", "snapshot_hash"
        ] = "episode_round_robin",
    ):
        self.jobs = jobs
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.mode = mode

        # Snapshot-level sharding (most balanced; does not preserve episode grouping)
        if int(num_shards) <= 1:
            self.indices = list(range(len(jobs)))
            return
        if self.mode == "snapshot_hash":
            self.indices = []
            for i, job in enumerate(jobs):
                key = f"{job.episode_id}:{job.t}:{job.snapshot_index}"
                if shard_id_for_key(key, num_shards=int(num_shards)) == int(shard_id):
                    self.indices.append(int(i))
            return

        # Episode-level sharding (keeps episodes intact)
        episode_to_indices: dict[str, list[int]] = {}
        for i, job in enumerate(jobs):
            eid = str(job.episode_id)
            episode_to_indices.setdefault(eid, []).append(i)

        all_episodes = sorted(episode_to_indices.keys())
        if self.mode == "episode_hash":
            my_episodes = [
                e
                for e in all_episodes
                if shard_id_for_key(str(e), num_shards=int(num_shards)) == int(shard_id)
            ]
        elif self.mode == "episode_greedy":
            eps = sorted(
                all_episodes,
                key=lambda e: (-len(episode_to_indices.get(e, [])), str(e)),
            )
            bins: list[list[str]] = [[] for _ in range(int(num_shards))]
            loads: list[int] = [0 for _ in range(int(num_shards))]
            for e in eps:
                k = min(range(int(num_shards)), key=lambda j: (loads[j], j))
                bins[k].append(e)
                loads[k] += int(len(episode_to_indices.get(e, [])))
            my_episodes = bins[int(shard_id) % int(num_shards)]
        else:  # episode_round_robin
            my_episodes = [e for i, e in enumerate(all_episodes) if i % num_shards == shard_id]

        self.indices = []
        for eid in my_episodes:
            self.indices.extend(episode_to_indices[eid])

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────


def build_trajectory_selection_jobs(
    dataset_path: Path,
    episode_meta_paths: list[Path],
    out_dir: Path,
    *,
    a0_cfg: A0Config,
    overlay_cfg: OverlayConfig,
    static_candidates_json: str | None = None,
    candidate_points_override_npy: str | None = None,
    candidate_points_override_mode: str = "notebook_v1",
    goal_filter_cfg: GoalFilterConfig | None = None,
    gt_filter_cfg: GtFutureFilterConfig | None = None,
    rgb_time_tolerance_s: float = 0.2,
    require_goal: bool = False,
    write_overlays: bool = True,
    overlay_image_width: int | None = None,
    prompt_history_frames: int = 0,
    prompt_image_width: int | None = None,
    compute_gt_metrics: bool = True,
    traj_dt_s: float = 0.2,
    snapshot_stride_s: float | None = None,
    max_snapshots_per_episode: int | None = None,
    max_snapshots_total: int | None = None,
    write_rgb_frames: bool = False,
) -> list[SnapshotJob]:
    """Build list of SnapshotJobs for trajectory-selection evaluation."""
    from slow_brain_fast_planner.benchmarks.planner_postprocessing import subsample_by_time_stride

    def _epid(p: Path) -> str:
        return p.parent.name if p.name == "episode.json" else p.stem

    def _episode_dir_for_meta(dataset_root: Path, meta_path: Path) -> Path:
        if meta_path.name == "episode.json":
            return meta_path.parent
        eid = meta_path.stem
        cand = (dataset_root / "episodes" / eid).resolve()
        return cand if cand.is_dir() else meta_path.parent

    def _read_planner_times(pc_path: Path, *, gf: GoalFilterConfig | None) -> list[float]:
        import math

        out: list[float] = []
        try:
            with pc_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    t = obj.get("t") if isinstance(obj, dict) else None
                    if isinstance(t, (int, float)) and math.isfinite(float(t)):
                        # Optional goal filter at job-build time (cheap, avoids scheduling junk).
                        if gf is not None and bool(gf.enabled) and isinstance(obj, dict):
                            gxy, gd, gb = extract_goal_info(obj)
                            reason = goal_filter_reason(gxy, gd, gb, cfg=gf)
                            if reason is not None:
                                continue
                        out.append(float(t))
        except Exception:
            return []
        out.sort()
        return out

    gf = goal_filter_cfg if goal_filter_cfg is not None else GoalFilterConfig()
    gtf = gt_filter_cfg if gt_filter_cfg is not None else GtFutureFilterConfig()

    jobs: list[SnapshotJob] = []
    for meta_path in episode_meta_paths:
        eid = _epid(meta_path)
        ep_dir = _episode_dir_for_meta(dataset_path, meta_path)
        pc_path = (ep_dir / "planner_candidates.jsonl").resolve()
        times = _read_planner_times(pc_path, gf=gf) if pc_path.exists() else []
        if not times:
            continue
        keep = subsample_by_time_stride(times, stride_s=snapshot_stride_s)
        kept_times = [times[i] for i in keep]
        if max_snapshots_per_episode is not None:
            kept_times = kept_times[: int(max_snapshots_per_episode)]
        for j, tt in enumerate(kept_times):
            jobs.append(
                SnapshotJob(
                    dataset_root=str(dataset_path),
                    out_dir=str(out_dir),
                    episode_meta_path=str(meta_path),
                    episode_id=str(eid),
                    t=float(tt),
                    snapshot_index=int(j),
                    a0_cfg=a0_cfg,
                    overlay_cfg=overlay_cfg,
                    rgb_time_tolerance_s=float(rgb_time_tolerance_s),
                    require_goal=bool(require_goal),
                    write_overlays=bool(write_overlays),
                    overlay_image_width=overlay_image_width,
                    prompt_history_frames=int(prompt_history_frames),
                    prompt_image_width=prompt_image_width,
                    compute_gt_metrics=bool(compute_gt_metrics),
                    traj_dt_s=float(traj_dt_s),
                    static_candidates_json=(
                        str(static_candidates_json) if static_candidates_json is not None else None
                    ),
                    candidate_points_override_npy=(
                        str(candidate_points_override_npy)
                        if candidate_points_override_npy is not None
                        else None
                    ),
                    candidate_points_override_mode=str(candidate_points_override_mode),
                    write_rgb_frames=bool(write_rgb_frames),
                    goal_filter_cfg=gf,
                    gt_filter_cfg=gtf,
                )
            )
    if max_snapshots_total is not None:
        jobs = jobs[: int(max_snapshots_total)]
    return jobs


def build_trajectory_selection_jobs_from_takeover_clips(
    *,
    dataset_path: Path,
    episode_meta_paths: list[Path],
    takeover_clips_path: Path,
    out_dir: Path,
    a0_cfg: A0Config,
    overlay_cfg: OverlayConfig,
    static_candidates_json: str | None = None,
    candidate_points_override_npy: str | None = None,
    candidate_points_override_mode: str = "notebook_v1",
    goal_filter_cfg: GoalFilterConfig | None = None,
    gt_filter_cfg: GtFutureFilterConfig | None = None,
    rgb_time_tolerance_s: float = 0.2,
    require_goal: bool = False,
    write_overlays: bool = True,
    overlay_image_width: int | None = None,
    prompt_history_frames: int = 0,
    prompt_image_width: int | None = None,
    compute_gt_metrics: bool = True,
    traj_dt_s: float = 0.2,
    clip_phase: str | None = None,
    clip_label_filter: str | None = None,
    max_snapshots_per_episode: int | None = None,
    max_snapshots_total: int | None = None,
    write_rgb_frames: bool = False,
) -> list[SnapshotJob]:
    """Build SnapshotJobs from takeover_clips.jsonl (clip times).

    This is useful to reduce evaluation/query load: instead of evaluating every planner tick,
    we evaluate only at clip `t0`s (typically strided, e.g., every 2s).
    """

    def _epid(p: Path) -> str:
        return p.parent.name if p.name == "episode.json" else p.stem

    allowed_eps = {_epid(p) for p in episode_meta_paths}
    meta_by_ep = {_epid(p): p for p in episode_meta_paths}

    takeover_clips_path = Path(takeover_clips_path).resolve()
    if not takeover_clips_path.exists():
        raise FileNotFoundError(f"takeover_clips.jsonl not found: {takeover_clips_path}")

    # Load clips and group by episode (stable time order).
    clips_by_ep: dict[str, list[float]] = {}
    with takeover_clips_path.open("r", encoding="utf-8") as f:
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
            t0 = obj.get("t0")
            ph = obj.get("phase")
            lab = obj.get("label_takeover_request")
            if not isinstance(eid, str) or not isinstance(t0, (int, float)):
                continue
            if eid not in allowed_eps:
                continue
            if clip_phase is not None and isinstance(ph, str) and str(ph) != str(clip_phase):
                continue
            if clip_label_filter is not None and str(clip_label_filter) != "any":
                # Only applies if the label field exists (some takeover_clips.jsonl variants
                # include labels).
                if isinstance(lab, (int, float, bool)):
                    is_takeover_clip = bool(int(lab) == 1)
                    if str(clip_label_filter) == "takeover_only" and (not is_takeover_clip):
                        continue
                    if str(clip_label_filter) == "no_takeover_only" and is_takeover_clip:
                        continue
            clips_by_ep.setdefault(eid, []).append(float(t0))

    gf = goal_filter_cfg if goal_filter_cfg is not None else GoalFilterConfig()
    gtf = gt_filter_cfg if gt_filter_cfg is not None else GtFutureFilterConfig()

    # Optional filtering of clip t0s by goal quality at t0 (reads planner_candidates.jsonl once per
    # episode).
    if gf is not None and bool(gf.enabled):
        for eid in list(clips_by_ep.keys()):
            ep_dir = (dataset_path / "episodes" / str(eid)).resolve()
            pc_path = (ep_dir / "planner_candidates.jsonl").resolve()
            if not pc_path.exists():
                continue
            ok_t: set[float] = set()
            try:
                with pc_path.open("r", encoding="utf-8") as f:
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
                        t = obj.get("t")
                        if not isinstance(t, (int, float)):
                            continue
                        gxy, gd, gb = extract_goal_info(obj)
                        reason = goal_filter_reason(gxy, gd, gb, cfg=gf)
                        if reason is None:
                            ok_t.add(round(float(t), 6))
            except Exception:
                continue

            clips_by_ep[eid] = [
                tt for tt in clips_by_ep.get(eid, []) if round(float(tt), 6) in ok_t
            ]
            if not clips_by_ep[eid]:
                clips_by_ep.pop(eid, None)

    for eid in list(clips_by_ep.keys()):
        clips_by_ep[eid].sort()
        if max_snapshots_per_episode is not None:
            clips_by_ep[eid] = clips_by_ep[eid][: int(max_snapshots_per_episode)]

    jobs: list[SnapshotJob] = []
    for eid in sorted(clips_by_ep.keys()):
        meta = meta_by_ep.get(eid)
        if meta is None:
            continue
        for j, tt in enumerate(clips_by_ep[eid]):
            jobs.append(
                SnapshotJob(
                    dataset_root=str(dataset_path),
                    out_dir=str(out_dir),
                    episode_meta_path=str(meta),
                    episode_id=str(eid),
                    t=float(tt),
                    snapshot_index=int(j),
                    a0_cfg=a0_cfg,
                    overlay_cfg=overlay_cfg,
                    rgb_time_tolerance_s=float(rgb_time_tolerance_s),
                    require_goal=bool(require_goal),
                    write_overlays=bool(write_overlays),
                    overlay_image_width=overlay_image_width,
                    prompt_history_frames=int(prompt_history_frames),
                    prompt_image_width=prompt_image_width,
                    compute_gt_metrics=bool(compute_gt_metrics),
                    traj_dt_s=float(traj_dt_s),
                    static_candidates_json=(
                        str(static_candidates_json) if static_candidates_json is not None else None
                    ),
                    candidate_points_override_npy=(
                        str(candidate_points_override_npy)
                        if candidate_points_override_npy is not None
                        else None
                    ),
                    candidate_points_override_mode=str(candidate_points_override_mode),
                    write_rgb_frames=bool(write_rgb_frames),
                    goal_filter_cfg=gf,
                    gt_filter_cfg=gtf,
                )
            )

    if max_snapshots_total is not None:
        jobs = jobs[: int(max_snapshots_total)]
    return jobs


def get_shard_config_from_env() -> tuple[int, int]:
    """Get (num_shards, shard_id) from environment variables.

    Checks WORLD_SIZE/RANK (PyTorch), SLURM_NTASKS/SLURM_PROCID (SLURM), or returns (1, 0).
    """
    import os

    num_shards = 1
    shard_id = 0

    # Check PyTorch distributed env vars
    if "WORLD_SIZE" in os.environ:
        try:
            num_shards = int(os.environ["WORLD_SIZE"])
        except ValueError:
            pass
    if "RANK" in os.environ:
        try:
            shard_id = int(os.environ["RANK"])
        except ValueError:
            pass

    # Check SLURM env vars (fallback)
    if num_shards == 1 and "SLURM_NTASKS" in os.environ:
        try:
            num_shards = int(os.environ["SLURM_NTASKS"])
        except ValueError:
            pass
    if shard_id == 0 and "SLURM_PROCID" in os.environ:
        try:
            shard_id = int(os.environ["SLURM_PROCID"])
        except ValueError:
            pass

    return num_shards, shard_id
