#!/usr/bin/env python3
"""Regenerate trajectory-selection metrics for an existing run directory.

This script is designed for *post-hoc* metric upgrades (e.g., when we add new metrics like
prefix ADE @ 0.5s/1.0s/2.0s). It replays the saved per-snapshot predictions from the old run
and recomputes metrics using the current evaluator logic, without making any VLM calls.

Inputs (expected in --run-dir):
- config.json
- predictions.jsonl
- (optional) traces/events.jsonl  (used to recover latency_ms if present)

Outputs (default: writes new files alongside the original run artifacts):
- metrics_regenerated.json
- metrics_report_regenerated.txt
- metrics_summary_regenerated.csv
- metrics_summary_simple_regenerated.csv
- metrics_summary_simple_regenerated.txt
- predictions_regenerated.csv

If you pass --inplace, it overwrites:
- metrics.json
- metrics_report.txt
- metrics_summary.csv
- metrics_summary_simple.csv
- metrics_summary_simple.txt
- predictions.csv

Example:
  python scripts/trajectory_selection/regenerate_metrics.py \
    --run-dir logs/<exp>/<trial> \
    --num-workers 4
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from torch.utils.data import DataLoader

from slow_brain_fast_planner import __version__
from slow_brain_fast_planner.benchmarks.data_loading import (
    SnapshotDataset,
    SnapshotItem,
    SnapshotShardedSampler,
    build_trajectory_selection_jobs,
    build_trajectory_selection_jobs_from_takeover_clips,
)
from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files
from slow_brain_fast_planner.benchmarks.overlays import OverlayConfig
from slow_brain_fast_planner.benchmarks.planner_postprocessing import (
    A0Config,
    _trajectory_nms_endpoints,
    safe_div,
    to_jsonable,
)
from slow_brain_fast_planner.benchmarks.trajectory_selection_workers import compute_ade
from slow_brain_fast_planner.cli.trajectory_selection import (
    _ade_prefix_seconds,
    _ap_at_threshold,
    _dcr_tcr_from_corridor,
    _fde_m,
    _fde_m_masked_laststep,
    _format_metrics_report,
    _maoe_deg,
    _moe_deg_posvec,
    _traj_path_length_m,
    _write_metrics_summary_csv,
    _write_metrics_summary_simple_csv,
)
from slow_brain_fast_planner.reporting import generate_run_report_html
from slow_brain_fast_planner.utils.io import utc_now_iso as _utc_now_iso
from slow_brain_fast_planner.utils.io import write_json as _write_json


def _maybe_resolve_run_dir(s: str) -> Path:
    """Accept either a local path or a localhost URL pointing at /logs/..."""
    s = str(s)
    if s.startswith("http://") or s.startswith("https://"):
        u = urlparse(s)
        p = unquote(u.path or "")
        # If user passed http://localhost:.../logs/<...>, map to repo_root/logs/<...>.
        repo_root = Path(__file__).resolve().parents[2]
        if p.startswith("/logs/"):
            return (repo_root / p.lstrip("/")).resolve()
        # Otherwise treat URL path as an absolute path.
        return Path(p).resolve()
    return Path(s).resolve()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _key(ep: str, t: float, snapshot_index: int) -> tuple[str, float, int]:
    # The run format uses float t in JSON; keep 6dp rounding to match dataset/indexing conventions.
    return (str(ep), round(float(t), 6), int(snapshot_index))


def _write_metrics_summary_simple_txt(csv_path: Path, txt_path: Path) -> None:
    """Write a browser-friendly companion for metrics_summary_simple.csv (copy verbatim)."""
    try:
        s = csv_path.read_text(encoding="utf-8")
    except Exception:
        return
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(s if s.endswith("\n") else (s + "\n"), encoding="utf-8")


def _parse_trace_latency_and_usage(
    trace_path: Path,
) -> dict[tuple[str, float, int], dict[str, Any]]:
    """Best-effort: recover provider_meta from traces.

    Some older runs might not have latency_ms; we only return what exists.
    """
    if not trace_path.exists():
        return {}
    out: dict[tuple[str, float, int], dict[str, Any]] = {}
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            if ev.get("event_type") != "model_call":
                continue
            ep = ev.get("episode_id")
            t = ev.get("t")
            snap_idx = ev.get("snapshot_index")
            if ep is None or t is None or snap_idx is None:
                continue
            meta = ev.get("provider_meta")
            if not isinstance(meta, dict):
                continue
            k = _key(str(ep), float(t), int(snap_idx))
            # Keep only the pieces we care about.
            keep: dict[str, Any] = {}
            if meta.get("latency_ms") is not None:
                keep["latency_ms"] = meta.get("latency_ms")
            if isinstance(meta.get("usage"), dict):
                keep["usage"] = meta.get("usage")
            if keep:
                out[k] = keep
    return out


def main() -> None:
    t_start = _dt.datetime.now(tz=_dt.UTC)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir", required=True, help="Existing Task2 run dir (path or localhost URL)."
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Where to write regenerated outputs (default: alongside run-dir).",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite metrics.json / metrics_report.txt / metrics_summary*.csv / predictions.csv "
        "in --run-dir.",
    )
    ap.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "Override dataloader num_workers (default: 4). "
            "This is the number of PyTorch DataLoader worker processes used to prepare snapshots "
            "(loading episode records, computing GT ADE, etc.)."
        ),
    )
    ap.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Override dataloader prefetch_factor (default: config).",
    )
    args = ap.parse_args()

    run_dir = _maybe_resolve_run_dir(str(args.run_dir))
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")

    cfg_path = run_dir / "config.json"
    pred_path = run_dir / "predictions.jsonl"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions.jsonl: {pred_path}")

    cfg = _read_json(cfg_path)
    if str(cfg.get("task")) not in ("task2_trajectory_selection", "trajectory_selection"):
        raise ValueError(f"config.json task is not trajectory_selection: {cfg.get('task')}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load old per-snapshot records (predictions + old skip reasons + overlay refs).
    old_recs = _read_jsonl(pred_path)
    old_by_key: dict[tuple[str, float, int], dict[str, Any]] = {}
    for r in old_recs:
        try:
            ep = str(r.get("episode_id", ""))
            t = float(r.get("t", 0.0))
            snap_idx = int(r.get("snapshot_index", 0))
        except Exception:
            continue
        old_by_key[_key(ep, t, snap_idx)] = r

    trace_meta_by_key = _parse_trace_latency_and_usage(run_dir / "traces" / "events.jsonl")

    dataset_path = Path(str(cfg.get("dataset"))).resolve()
    episode_meta_paths = find_episode_metadata_files(dataset_path)

    # Optional episode filter.
    ep_filter = cfg.get("episode_id")
    if isinstance(ep_filter, str) and ep_filter.strip():
        episode_meta_paths = [
            p for p in episode_meta_paths if (p.parent.name == ep_filter or p.stem == ep_filter)
        ]
    max_eps = cfg.get("max_episodes")
    if max_eps is not None:
        episode_meta_paths = episode_meta_paths[: int(max_eps)]

    # Sharding by episode (match evaluator behavior).
    shard = cfg.get("shard") if isinstance(cfg.get("shard"), dict) else {}
    shard_enabled = bool(shard.get("enabled", False))
    num_shards = int(shard.get("num_shards", 1) or 1)
    shard_id = int(shard.get("shard_id", 0) or 0)
    shard_mode_raw = shard.get("mode", None)
    shard_mode = (
        str(shard_mode_raw).strip()
        if isinstance(shard_mode_raw, str) and shard_mode_raw.strip()
        else None
    )
    # NOTE: we do NOT pre-filter episode_meta_paths for sharding; we shard SnapshotJobs via
    # SnapshotShardedSampler below (matching the main evaluator behavior).
    #
    # IMPORTANT: merged shard runs keep the original shard config (often shard0) in config.json,
    # but `predictions.jsonl` in the merged run dir contains *all shards*. For regen, we should
    # evaluate the full merged set (no sharding), otherwise we'll only score ~1/N of the run.
    shard_merge = cfg.get("shard_merge") if isinstance(cfg.get("shard_merge"), dict) else {}
    if bool(shard_merge.get("enabled", False)) and shard_enabled and num_shards > 1:
        mp = shard_merge.get("merged_predictions")
        if isinstance(mp, str):
            try:
                if Path(mp).resolve() == pred_path.resolve():
                    shard_enabled = False
                    num_shards = 1
                    shard_id = 0
                    shard_mode = None
            except Exception:
                pass

    # Recreate prep config.
    a0 = cfg.get("a0") if isinstance(cfg.get("a0"), dict) else {}
    a0_cfg = A0Config(
        gt_rule=str(a0.get("gt_rule", "raw_argmax")),
        policy="raw_argmax",
        nms_max_trajectories=int(a0.get("nms_max_trajectories", 6)),
        nms_distance_threshold=float(a0.get("nms_distance_threshold", 2.0)),
        prob_threshold=float(a0.get("prob_threshold", 0.01)),
    )

    overlays = cfg.get("overlays") if isinstance(cfg.get("overlays"), dict) else {}
    overlay_cfg = OverlayConfig(
        top_k=int(overlays.get("top_k", 6)),
        min_score=overlays.get("min_score", None),
        projection=str(overlays.get("projection", "fisheye_v1")),
        candidate_set=str(overlays.get("candidate_set", "planner_v2")),
        nms_max_trajectories=int(a0_cfg.nms_max_trajectories),
        nms_distance_threshold=float(a0_cfg.nms_distance_threshold),
        prob_threshold=float(a0_cfg.prob_threshold),
    )

    prompt = cfg.get("prompt") if isinstance(cfg.get("prompt"), dict) else {}

    gt_metrics = cfg.get("gt_metrics") if isinstance(cfg.get("gt_metrics"), dict) else {}
    compute_gt = bool(gt_metrics.get("enabled", True))
    traj_dt_s = float(gt_metrics.get("traj_dt_s", 0.2))

    rgb_time_tolerance_s = float(overlays.get("rgb_time_tolerance_s", 0.2))
    require_goal = bool(cfg.get("require_goal", False))
    max_snapshots_per_episode = cfg.get("max_snapshots_per_episode", None)
    max_snapshots_total = cfg.get("max_snapshots_total", None)

    takeover = cfg.get("takeover_clips") if isinstance(cfg.get("takeover_clips"), dict) else {}
    use_takeover = bool(takeover.get("enabled", False)) and takeover.get("dir") is not None
    takeover_dir = Path(str(takeover.get("dir"))).resolve() if use_takeover else None
    clip_phase = takeover.get("phase", None)
    if isinstance(clip_phase, str) and not clip_phase.strip():
        clip_phase = None

    # IMPORTANT for regen: don't generate overlays again (slow and can perturb skip behavior).
    write_overlays = False

    if use_takeover and takeover_dir is not None:
        takeover_jsonl = takeover_dir / "takeover_clips.jsonl"
        jobs = build_trajectory_selection_jobs_from_takeover_clips(
            dataset_path=dataset_path,
            episode_meta_paths=episode_meta_paths,
            takeover_clips_path=takeover_jsonl,
            out_dir=run_dir,  # irrelevant for regen when write_overlays=False
            a0_cfg=a0_cfg,
            overlay_cfg=overlay_cfg,
            rgb_time_tolerance_s=rgb_time_tolerance_s,
            require_goal=require_goal,
            write_overlays=write_overlays,
            overlay_image_width=None,
            prompt_history_frames=int(prompt.get("history_frames", 0) or 0),
            prompt_image_width=prompt.get("image_width", None),
            compute_gt_metrics=compute_gt,
            traj_dt_s=traj_dt_s,
            clip_phase=clip_phase,
            max_snapshots_per_episode=max_snapshots_per_episode,
            max_snapshots_total=max_snapshots_total,
        )
    else:
        snapshot_stride_s = cfg.get("snapshot_stride_s", None)
        jobs = build_trajectory_selection_jobs(
            dataset_path=dataset_path,
            episode_meta_paths=episode_meta_paths,
            out_dir=run_dir,  # irrelevant for regen when write_overlays=False
            a0_cfg=a0_cfg,
            overlay_cfg=overlay_cfg,
            rgb_time_tolerance_s=rgb_time_tolerance_s,
            require_goal=require_goal,
            write_overlays=write_overlays,
            overlay_image_width=None,
            prompt_history_frames=int(prompt.get("history_frames", 0) or 0),
            prompt_image_width=prompt.get("image_width", None),
            compute_gt_metrics=compute_gt,
            traj_dt_s=traj_dt_s,
            snapshot_stride_s=snapshot_stride_s,
            max_snapshots_per_episode=max_snapshots_per_episode,
            max_snapshots_total=max_snapshots_total,
        )

    dataset = SnapshotDataset(jobs)

    dl_cfg = cfg.get("dataloader") if isinstance(cfg.get("dataloader"), dict) else {}
    num_workers = int(
        args.num_workers if args.num_workers is not None else dl_cfg.get("num_workers", 0)
    )
    prefetch_factor = (
        args.prefetch_factor
        if args.prefetch_factor is not None
        else dl_cfg.get("prefetch_factor", 2)
    )

    sampler = None
    if shard_enabled and num_shards > 1:
        mode0 = shard_mode or "episode_round_robin"
        # Backward-compat: older runs didn't persist shard mode. Try to infer it from predictions.
        if shard_mode is None:
            try:
                pred_eps = {str(k[0]) for k in old_by_key.keys()}
                modes = ["episode_round_robin", "episode_hash", "episode_greedy", "snapshot_hash"]
                best_mode = mode0
                best_score = -1.0
                for m in modes:
                    smp = SnapshotShardedSampler(
                        jobs, num_shards=num_shards, shard_id=shard_id, mode=m
                    )
                    eps_m = {str(jobs[i].episode_id) for i in getattr(smp, "indices", [])}
                    # Jaccard similarity on episode sets.
                    denom = float(len(pred_eps | eps_m)) if (pred_eps or eps_m) else 1.0
                    score = float(len(pred_eps & eps_m)) / denom
                    if score > best_score:
                        best_score = score
                        best_mode = m
                mode0 = best_mode
            except Exception:
                mode0 = mode0
        sampler = SnapshotShardedSampler(jobs, num_shards=num_shards, shard_id=shard_id, mode=mode0)

    dl_kwargs: dict[str, Any] = dict(
        batch_size=1,
        num_workers=num_workers,
        collate_fn=SnapshotDataset.collate_fn,
        sampler=sampler,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)
        dl_kwargs["persistent_workers"] = True

    loader = DataLoader(dataset, **dl_kwargs)

    # Output paths.
    if args.inplace:
        metrics_json_path = out_dir / "metrics.json"
        report_txt_path = out_dir / "metrics_report.txt"
        summary_csv_path = out_dir / "metrics_summary.csv"
        summary_simple_csv_path = out_dir / "metrics_summary_simple.csv"
        summary_simple_txt_path = out_dir / "metrics_summary_simple.txt"
        preds_csv_path = out_dir / "predictions.csv"
        preds_jsonl_path = out_dir / "predictions.jsonl"
        report_html_path = out_dir / "report.html"
    else:
        metrics_json_path = out_dir / "metrics_regenerated.json"
        report_txt_path = out_dir / "metrics_report_regenerated.txt"
        summary_csv_path = out_dir / "metrics_summary_regenerated.csv"
        summary_simple_csv_path = out_dir / "metrics_summary_simple_regenerated.csv"
        summary_simple_txt_path = out_dir / "metrics_summary_simple_regenerated.txt"
        preds_csv_path = out_dir / "predictions_regenerated.csv"
        preds_jsonl_path = out_dir / "predictions_regenerated.jsonl"
        report_html_path = out_dir / "report_regenerated.html"

    csv_cols = [
        "episode_id",
        "t",
        "snapshot_index",
        "prediction",
        "label",
        "correct",
        "skipped",
        "skip_reason",
        "num_candidates",
        "goal_x",
        "goal_y",
        "goal_distance_m",
        "goal_bearing_deg",
        "selected_end_dist_to_goal_m",
        "selected_goal_ang_diff_deg",
        "selected_traj_avg_dist_to_goal_m",
        "selected_goal_progress_m",
        "traj_len_m",
        "maoe_deg",
        "dcr",
        "tcr",
        "compliance_source",
        "fde_selected",
        "fde_min",
        "fde_min_all",
        "fde_score",
        "fde_score_masked_laststep",
        "moe_deg",
        "moe_deg_min",
        "moe_deg_min_all",
        "moe_deg_score",
        "moe_deg_mean_all",
        "ap_fde_2m",
        "ade_selected",
        "ade_selected_0_5s",
        "ade_selected_1_0s",
        "ade_selected_2_0s",
        "ade_min",
        "ade_min_all",
        "ade_min_0_5s",
        "ade_min_1_0s",
        "ade_min_2_0s",
        "ade_score",
        "ade_score_0_5s",
        "ade_score_1_0s",
        "ade_score_2_0s",
        "route_dev_selected",
        "route_dev_min",
        "route_dev_score",
        "overlay_frame_ref",
        "local_plot_frame_ref",
        "auto_enabled",
    ]

    # Aggregates (mirrors slow_brain_fast_planner.cli.trajectory_selection).
    skipped_reasons: dict[str, int] = {}
    total_snapshots = 0
    evaluated = 0
    correct = 0
    ade_min_sum = ade_min_all_sum = ade_score_sum = ade_model_sum = 0.0
    ade_model_0_5s_sum = ade_model_1_0s_sum = ade_model_2_0s_sum = 0.0
    ade_score_0_5s_sum = ade_score_1_0s_sum = ade_score_2_0s_sum = 0.0
    ade_min_0_5s_sum = ade_min_1_0s_sum = ade_min_2_0s_sum = 0.0
    ade_model_0_5s_count = ade_model_1_0s_count = ade_model_2_0s_count = 0
    ade_score_0_5s_count = ade_score_1_0s_count = ade_score_2_0s_count = 0
    ade_min_0_5s_count = ade_min_1_0s_count = ade_min_2_0s_count = 0
    ade_count = 0
    # FDE / MOE / mAP (open-loop trajectory quality).
    fde_model_sum = fde_min_sum = fde_min_all_sum = fde_score_sum = 0.0
    fde_model_count = fde_min_count = fde_min_all_count = fde_score_count = 0
    fde_score_masked_laststep_sum = 0.0
    fde_score_masked_laststep_count = 0
    moe_model_sum = moe_min_sum = moe_min_all_sum = moe_score_sum = moe_mean_all_sum = 0.0
    moe_model_count = moe_min_count = moe_min_all_count = moe_score_count = 0
    moe_mean_all_count = 0
    ap_fde_2m_sum = 0.0
    ap_fde_2m_count = 0
    correct_vs_min_ade = 0
    correct_vs_min_ade_all = 0
    score_agree_count = 0
    correct_vs_score = 0
    route_dev_min_sum = route_dev_score_sum = route_dev_model_sum = 0.0
    route_dev_count = 0
    correct_vs_route_min = 0
    correct_vs_route_score = 0
    stop_count = 0
    invalid_index_count = 0
    auto_true = auto_false = auto_missing = 0
    goal_dist_sum = 0.0
    goal_dist_count = 0
    sel_end_goal_dist_sum = 0.0
    sel_end_goal_dist_count = 0
    sel_goal_ang_diff_sum = 0.0
    sel_goal_ang_diff_count = 0
    sel_traj_avg_goal_dist_sum = 0.0
    sel_traj_avg_goal_dist_count = 0
    sel_goal_progress_sum = 0.0
    sel_goal_progress_count = 0
    maoe_sum = 0.0
    maoe_count = 0
    dcr_sum = 0.0
    dcr_count = 0
    tcr_sum = 0.0
    tcr_count = 0
    model_calls = 0
    prompt_tokens_sum = 0
    cached_prompt_tokens_sum = 0
    output_tokens_sum = 0
    thoughts_tokens_sum = 0
    total_tokens_sum = 0
    latency_ms_list: list[float] = []

    def _bump_skip(reason: str) -> None:
        skipped_reasons[reason] = int(skipped_reasons.get(reason, 0)) + 1

    new_jsonl_recs: list[dict[str, Any]] = []

    with preds_csv_path.open("w", encoding="utf-8", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()

        # Progress bar (best-effort).
        try:
            from tqdm import tqdm  # type: ignore

            loader_iter = tqdm(loader, desc="regen snapshots", total=len(loader))
        except Exception:
            loader_iter = loader

        for batch in loader_iter:
            for item in batch:
                assert isinstance(item, SnapshotItem)
                prep = item.prep
                total_snapshots += 1

                if not prep.ok:
                    _bump_skip(str(prep.skip_reason or "prep_failed"))
                    continue

                ep_id = str(prep.episode_id)
                t = float(prep.t)
                snap_idx = int(prep.snapshot_index)
                k = _key(ep_id, t, snap_idx)
                old = old_by_key.get(k)
                if old is None:
                    _bump_skip("missing_prediction_record")
                    continue

                # Replay the old run decision.
                pred_action = old.get("prediction_action")
                pred_idx = old.get("prediction")
                if pred_action is None:
                    # Legacy: if prediction is an int, treat as select_trajectory.
                    pred_action = "select_trajectory" if pred_idx is not None else "stop"

                # Respect old skip decisions (e.g., overlay failures).
                if bool(old.get("skipped", False)) or old.get("skip_reason") not in (
                    None,
                    "",
                    "null",
                ):
                    _bump_skip(str(old.get("skip_reason") or "skipped_in_original_run"))
                    continue

                label = prep.label
                label_err = prep.label_err

                # Validate prediction.
                pred_err = None
                if pred_action not in ("select_trajectory", "stop"):
                    pred_err = "invalid_action"
                if pred_action == "select_trajectory":
                    if pred_idx is None:
                        pred_err = "missing_selected_index"
                    else:
                        try:
                            pred_idx = int(pred_idx)
                        except Exception:
                            pred_err = "non_int_selected_index"
                elif pred_action == "stop":
                    pred_idx = None

                skip_reason = None
                if label_err:
                    skip_reason = f"label:{label_err}"
                elif pred_err:
                    skip_reason = f"pred:{pred_err}"

                is_eval = skip_reason is None and label is not None
                if not is_eval:
                    _bump_skip(skip_reason or "unknown_skip")
                    continue

                # Correctness.
                if pred_action == "select_trajectory" and pred_idx is not None:
                    is_correct = int(pred_idx) == int(label)
                else:
                    is_correct = False

                # Model output health (mirrors slow_brain_fast_planner.cli.trajectory_selection):
                # - stop_count is counted over evaluated samples
                # - invalid_index_count is counted over evaluated samples (out-of-range indices are
                # not treated as pred errors)
                if pred_action == "stop":
                    stop_count += 1
                elif pred_action == "select_trajectory":
                    try:
                        if pred_idx is not None and not (
                            0 <= int(pred_idx) < int(prep.num_candidates)
                        ):
                            invalid_index_count += 1
                    except Exception:
                        invalid_index_count += 1

                # Goal-relative metrics + ADE prefix metrics (copy evaluator logic).
                selected_end_dist_to_goal_m = None
                selected_goal_ang_diff_deg = None
                selected_traj_avg_dist_to_goal_m = None
                selected_goal_progress_m = None
                selected_ade = None
                selected_ade_0_5s = None
                selected_ade_1_0s = None
                selected_ade_2_0s = None
                selected_fde_m = None
                selected_moe_deg = None
                score_ade_0_5s = None
                score_ade_1_0s = None
                score_ade_2_0s = None
                score_fde_m = None
                score_fde_masked_laststep_m = None
                score_moe_deg = None
                min_ade_0_5s = None
                min_ade_1_0s = None
                min_ade_2_0s = None
                min_fde_m = None
                min_fde_all_m = None
                min_moe_deg = None
                min_moe_all_deg = None
                mean_moe_all_deg = None
                ap_fde_2m = None
                traj_len_m = None
                maoe_deg = None
                dcr = None
                tcr = None
                compliance_source = None

                score_idx = prep.score_idx
                try:
                    pts0 = None
                    if (
                        pred_action == "select_trajectory"
                        and pred_idx is not None
                        and isinstance(prep.candidates_points_xy, list)
                        and 0 <= int(pred_idx) < len(prep.candidates_points_xy)
                    ):
                        pts0 = np.asarray(
                            prep.candidates_points_xy[int(pred_idx)], dtype=np.float64
                        )
                    elif pred_action == "stop":
                        n_pts = 1
                        if (
                            isinstance(prep.gt_local_traj_xy, list)
                            and len(prep.gt_local_traj_xy) > 0
                        ):
                            n_pts = int(len(prep.gt_local_traj_xy))
                        elif (
                            isinstance(prep.candidates_points_xy, list)
                            and prep.candidates_points_xy
                        ):
                            try:
                                n_pts = int(len(prep.candidates_points_xy[0]))
                            except Exception:
                                n_pts = 1
                        pts0 = np.zeros((max(1, n_pts), 2), dtype=np.float64)

                    if (
                        pts0 is not None
                        and pts0.ndim == 2
                        and pts0.shape[0] > 0
                        and pts0.shape[1] >= 2
                    ):
                        goal_xy = None
                        if isinstance(prep.goal_xy, list) and len(prep.goal_xy) >= 2:
                            goal_xy = np.asarray(
                                [float(prep.goal_xy[0]), float(prep.goal_xy[1])], dtype=np.float64
                            )
                        if goal_xy is not None:
                            end_xy = pts0[-1, :2]
                            d_end = float(np.linalg.norm(end_xy - goal_xy))
                            d_all = np.linalg.norm(pts0[:, :2] - goal_xy.reshape(1, 2), axis=1)
                            d_avg = float(np.mean(d_all)) if d_all.size > 0 else None
                            goal_ang = float(
                                math.degrees(math.atan2(float(goal_xy[1]), float(goal_xy[0])))
                            )
                            end_ang = float(
                                math.degrees(math.atan2(float(end_xy[1]), float(end_xy[0])))
                            )
                            diff = (end_ang - goal_ang + 180.0) % 360.0 - 180.0
                            ang_diff = abs(float(diff))

                            selected_end_dist_to_goal_m = d_end if math.isfinite(d_end) else None
                            selected_traj_avg_dist_to_goal_m = (
                                d_avg if (d_avg is not None and math.isfinite(d_avg)) else None
                            )
                            selected_goal_ang_diff_deg = (
                                ang_diff if math.isfinite(ang_diff) else None
                            )

                            if prep.goal_distance_m is not None:
                                gd0 = float(prep.goal_distance_m)
                                if math.isfinite(gd0) and selected_end_dist_to_goal_m is not None:
                                    selected_goal_progress_m = float(gd0) - float(
                                        selected_end_dist_to_goal_m
                                    )

                        # Path length (diagnostic).
                        traj_len_m = _traj_path_length_m(pts0[:, :2])

                        if isinstance(prep.gt_local_traj_xy, list) and prep.gt_local_traj_xy:
                            gt = np.asarray(prep.gt_local_traj_xy, dtype=np.float64)
                            sel = np.asarray(pts0[:, :2], dtype=np.float64)
                            selected_ade = compute_ade(sel, gt)
                            selected_ade_0_5s = _ade_prefix_seconds(
                                sel, gt, dt_s=float(traj_dt_s), seconds=0.5
                            )
                            selected_ade_1_0s = _ade_prefix_seconds(
                                sel, gt, dt_s=float(traj_dt_s), seconds=1.0
                            )
                            selected_ade_2_0s = _ade_prefix_seconds(
                                sel, gt, dt_s=float(traj_dt_s), seconds=2.0
                            )

                            selected_fde_m = _fde_m(sel, gt)
                            selected_moe_deg = _moe_deg_posvec(sel, gt)

                            maoe_deg = _maoe_deg(sel, gt)
                            dcr, tcr = _dcr_tcr_from_corridor(
                                pred_xy=sel,
                                compliant_polyline_xy=gt,
                                dt_s=float(traj_dt_s),
                                corridor_radius_m=1.0,
                            )
                            compliance_source = "gt_corridor"

                            # Candidate baselines + AP@2m (best-effort; requires candidate set).
                            cand_fdes: list[float | None] | None = None
                            cand_moes: list[float | None] | None = None
                            try:
                                if isinstance(prep.candidates_points_xy, list):
                                    cand_fdes = []
                                    cand_moes = []
                                    for i in range(len(prep.candidates_points_xy)):
                                        try:
                                            xy_i = np.asarray(
                                                prep.candidates_points_xy[i], dtype=np.float64
                                            )
                                        except Exception:
                                            xy_i = None
                                        cand_fdes.append(
                                            _fde_m(xy_i, gt) if xy_i is not None else None
                                        )
                                        cand_moes.append(
                                            _moe_deg_posvec(xy_i, gt) if xy_i is not None else None
                                        )
                            except Exception:
                                cand_fdes = None
                                cand_moes = None

                            try:
                                if (
                                    score_idx is not None
                                    and cand_fdes is not None
                                    and cand_moes is not None
                                ):
                                    if 0 <= int(score_idx) < len(cand_fdes):
                                        score_fde_m = cand_fdes[int(score_idx)]
                                        score_moe_deg = cand_moes[int(score_idx)]
                            except Exception:
                                pass
                            try:
                                if score_idx is not None and isinstance(
                                    prep.candidates_points_xy, list
                                ):
                                    si = int(score_idx)
                                    if 0 <= si < len(prep.candidates_points_xy):
                                        sc_xy = np.asarray(
                                            prep.candidates_points_xy[si], dtype=np.float64
                                        )
                                        score_fde_masked_laststep_m = _fde_m_masked_laststep(
                                            sc_xy, gt
                                        )
                            except Exception:
                                score_fde_masked_laststep_m = None

                            try:
                                if cand_fdes is not None:
                                    vals = [
                                        float(v)
                                        for v in cand_fdes
                                        if v is not None and math.isfinite(float(v))
                                    ]
                                    min_fde_all_m = min(vals) if vals else None
                            except Exception:
                                pass
                            try:
                                if cand_moes is not None:
                                    vals = [
                                        float(v)
                                        for v in cand_moes
                                        if v is not None and math.isfinite(float(v))
                                    ]
                                    min_moe_all_deg = min(vals) if vals else None
                                    mean_moe_all_deg = (
                                        float(np.mean(np.asarray(vals, dtype=np.float64)))
                                        if vals
                                        else None
                                    )
                            except Exception:
                                pass

                            try:
                                vis = (
                                    prep.ade.get("visible_indices")
                                    if isinstance(prep.ade, dict)
                                    and isinstance(prep.ade.get("visible_indices"), list)
                                    else None
                                )
                                if vis and cand_fdes is not None and cand_moes is not None:
                                    f_vals = []
                                    m_vals = []
                                    for ii in vis:
                                        try:
                                            i = int(ii)
                                        except Exception:
                                            continue
                                        if not (0 <= i < len(cand_fdes)):
                                            continue
                                        vf = cand_fdes[i]
                                        if vf is not None and math.isfinite(float(vf)):
                                            f_vals.append(float(vf))
                                        vm = cand_moes[i]
                                        if vm is not None and math.isfinite(float(vm)):
                                            m_vals.append(float(vm))
                                    min_fde_m = min(f_vals) if f_vals else None
                                    min_moe_deg = min(m_vals) if m_vals else None
                            except Exception:
                                pass

                            try:
                                if cand_fdes is not None and isinstance(
                                    prep.candidate_scores_raw, list
                                ):
                                    keep = None
                                    try:
                                        if (
                                            isinstance(prep.candidates_points_xy, list)
                                            and len(prep.candidates_points_xy) == len(cand_fdes)
                                            and len(prep.candidate_scores_raw) == len(cand_fdes)
                                        ):
                                            scores_np = np.asarray(
                                                [float(x) for x in prep.candidate_scores_raw],
                                                dtype=np.float64,
                                            )
                                            ep = []
                                            for pts in prep.candidates_points_xy:
                                                if isinstance(pts, list) and pts:
                                                    last = pts[-1]
                                                    ep.append([float(last[0]), float(last[1])])
                                                else:
                                                    ep.append([0.0, 0.0])
                                            endpoints_xy = np.asarray(ep, dtype=np.float64)
                                            if endpoints_xy.ndim == 2 and endpoints_xy.shape == (
                                                len(cand_fdes),
                                                2,
                                            ):
                                                keep = _trajectory_nms_endpoints(
                                                    scores=scores_np,
                                                    endpoints_xy=endpoints_xy,
                                                    max_trajectories=int(
                                                        a0_cfg.nms_max_trajectories
                                                    ),
                                                    distance_threshold=float(
                                                        a0_cfg.nms_distance_threshold
                                                    ),
                                                ).tolist()
                                    except Exception:
                                        keep = None

                                    if keep:
                                        keep_i = [
                                            int(i) for i in keep if 0 <= int(i) < len(cand_fdes)
                                        ]
                                        scores_kept = [
                                            float(prep.candidate_scores_raw[i]) for i in keep_i
                                        ]
                                        fdes_kept = [cand_fdes[i] for i in keep_i]
                                        ap_fde_2m = _ap_at_threshold(
                                            scores=scores_kept, fdes_m=fdes_kept, threshold_m=2.0
                                        )
                                    else:
                                        ap_fde_2m = _ap_at_threshold(
                                            scores=prep.candidate_scores_raw,
                                            fdes_m=cand_fdes,
                                            threshold_m=2.0,
                                        )
                            except Exception:
                                ap_fde_2m = None

                            try:
                                if score_idx is not None and isinstance(
                                    prep.candidates_points_xy, list
                                ):
                                    sc_xy = np.asarray(
                                        prep.candidates_points_xy[int(score_idx)], dtype=np.float64
                                    )
                                    score_ade_0_5s = _ade_prefix_seconds(
                                        sc_xy, gt, dt_s=float(traj_dt_s), seconds=0.5
                                    )
                                    score_ade_1_0s = _ade_prefix_seconds(
                                        sc_xy, gt, dt_s=float(traj_dt_s), seconds=1.0
                                    )
                                    score_ade_2_0s = _ade_prefix_seconds(
                                        sc_xy, gt, dt_s=float(traj_dt_s), seconds=2.0
                                    )
                            except Exception:
                                pass
                            try:
                                if (
                                    isinstance(prep.ade, dict)
                                    and prep.ade.get("min_idx") is not None
                                ):
                                    mi = int(prep.ade.get("min_idx"))
                                    if isinstance(
                                        prep.candidates_points_xy, list
                                    ) and 0 <= mi < len(prep.candidates_points_xy):
                                        mi_xy = np.asarray(
                                            prep.candidates_points_xy[mi], dtype=np.float64
                                        )
                                        min_ade_0_5s = _ade_prefix_seconds(
                                            mi_xy, gt, dt_s=float(traj_dt_s), seconds=0.5
                                        )
                                        min_ade_1_0s = _ade_prefix_seconds(
                                            mi_xy, gt, dt_s=float(traj_dt_s), seconds=1.0
                                        )
                                        min_ade_2_0s = _ade_prefix_seconds(
                                            mi_xy, gt, dt_s=float(traj_dt_s), seconds=2.0
                                        )
                            except Exception:
                                pass
                except Exception:
                    # Best-effort; keep them None.
                    pass

                # Aggregate counters.
                evaluated += 1
                if is_correct:
                    correct += 1

                # Token usage + latency (best-effort): prefer existing predictions.jsonl field,
                # else traces.
                provider_usage = old.get("token_usage")
                if provider_usage is None:
                    meta = trace_meta_by_key.get(k)
                    if isinstance(meta, dict):
                        provider_usage = meta.get("usage")
                if isinstance(provider_usage, dict):
                    try:
                        pt = provider_usage.get("prompt_tokens")
                        cpt = provider_usage.get("cached_prompt_tokens")
                        ot = provider_usage.get("output_tokens")
                        th = provider_usage.get("thoughts_tokens")
                        tt = provider_usage.get("total_tokens")
                        if pt is not None:
                            prompt_tokens_sum += int(pt)
                        if cpt is not None:
                            cached_prompt_tokens_sum += int(cpt)
                        if ot is not None:
                            output_tokens_sum += int(ot)
                        if th is not None:
                            thoughts_tokens_sum += int(th)
                        if tt is not None:
                            total_tokens_sum += int(tt)
                        model_calls += 1
                    except Exception:
                        pass
                meta2 = trace_meta_by_key.get(k)
                if isinstance(meta2, dict) and meta2.get("latency_ms") is not None:
                    try:
                        lm = float(meta2.get("latency_ms"))
                        if math.isfinite(lm) and lm >= 0:
                            latency_ms_list.append(lm)
                    except Exception:
                        pass

                # Auto enabled stats.
                if prep.auto_enabled is True:
                    auto_true += 1
                elif prep.auto_enabled is False:
                    auto_false += 1
                else:
                    auto_missing += 1

                # Goal distance aggregate.
                try:
                    gd = prep.goal_distance_m
                    if gd is not None:
                        gdf = float(gd)
                        if math.isfinite(gdf):
                            goal_dist_sum += gdf
                            goal_dist_count += 1
                except Exception:
                    pass

                # Selected goal-relative aggregates.
                try:
                    if selected_end_dist_to_goal_m is not None:
                        x = float(selected_end_dist_to_goal_m)
                        if math.isfinite(x):
                            sel_end_goal_dist_sum += x
                            sel_end_goal_dist_count += 1
                    if selected_goal_ang_diff_deg is not None:
                        x = float(selected_goal_ang_diff_deg)
                        if math.isfinite(x):
                            sel_goal_ang_diff_sum += x
                            sel_goal_ang_diff_count += 1
                    if selected_traj_avg_dist_to_goal_m is not None:
                        x = float(selected_traj_avg_dist_to_goal_m)
                        if math.isfinite(x):
                            sel_traj_avg_goal_dist_sum += x
                            sel_traj_avg_goal_dist_count += 1
                    if selected_goal_progress_m is not None:
                        x = float(selected_goal_progress_m)
                        if math.isfinite(x):
                            sel_goal_progress_sum += x
                            sel_goal_progress_count += 1
                except Exception:
                    pass

                # Open-loop social metrics (best-effort).
                try:
                    if maoe_deg is not None and math.isfinite(float(maoe_deg)):
                        maoe_sum += float(maoe_deg)
                        maoe_count += 1
                    if dcr is not None and math.isfinite(float(dcr)):
                        dcr_sum += float(dcr)
                        dcr_count += 1
                    if tcr is not None and math.isfinite(float(tcr)):
                        tcr_sum += float(tcr)
                        tcr_count += 1
                except Exception:
                    pass

                # FDE / MOE / mAP aggregates (best-effort; only when GT-derived values exist).
                try:
                    if selected_fde_m is not None and math.isfinite(float(selected_fde_m)):
                        fde_model_sum += float(selected_fde_m)
                        fde_model_count += 1
                    if score_fde_m is not None and math.isfinite(float(score_fde_m)):
                        fde_score_sum += float(score_fde_m)
                        fde_score_count += 1
                    if score_fde_masked_laststep_m is not None and math.isfinite(
                        float(score_fde_masked_laststep_m)
                    ):
                        fde_score_masked_laststep_sum += float(score_fde_masked_laststep_m)
                        fde_score_masked_laststep_count += 1
                    if min_fde_m is not None and math.isfinite(float(min_fde_m)):
                        fde_min_sum += float(min_fde_m)
                        fde_min_count += 1
                    if min_fde_all_m is not None and math.isfinite(float(min_fde_all_m)):
                        fde_min_all_sum += float(min_fde_all_m)
                        fde_min_all_count += 1

                    if selected_moe_deg is not None and math.isfinite(float(selected_moe_deg)):
                        moe_model_sum += float(selected_moe_deg)
                        moe_model_count += 1
                    if score_moe_deg is not None and math.isfinite(float(score_moe_deg)):
                        moe_score_sum += float(score_moe_deg)
                        moe_score_count += 1
                    if min_moe_deg is not None and math.isfinite(float(min_moe_deg)):
                        moe_min_sum += float(min_moe_deg)
                        moe_min_count += 1
                    if min_moe_all_deg is not None and math.isfinite(float(min_moe_all_deg)):
                        moe_min_all_sum += float(min_moe_all_deg)
                        moe_min_all_count += 1
                    if mean_moe_all_deg is not None and math.isfinite(float(mean_moe_all_deg)):
                        moe_mean_all_sum += float(mean_moe_all_deg)
                        moe_mean_all_count += 1

                    if ap_fde_2m is not None and math.isfinite(float(ap_fde_2m)):
                        ap_fde_2m_sum += float(ap_fde_2m)
                        ap_fde_2m_count += 1
                except Exception:
                    pass

                # Score agreement.
                if (
                    pred_idx is not None
                    and score_idx is not None
                    and 0 <= int(pred_idx) < int(prep.num_candidates)
                ):
                    score_agree_count += 1
                    if int(pred_idx) == int(score_idx):
                        correct_vs_score += 1

                # ADE aggregates.
                if isinstance(prep.ade, dict) and pred_idx is not None:
                    ades = prep.ade.get("ades")
                    if isinstance(ades, list) and 0 <= int(pred_idx) < len(ades):
                        ade_count += 1
                        if prep.ade.get("min") is not None:
                            ade_min_sum += float(prep.ade.get("min"))
                        if prep.ade.get("min_all") is not None:
                            ade_min_all_sum += float(prep.ade.get("min_all"))
                        if prep.ade.get("score") is not None:
                            ade_score_sum += float(prep.ade.get("score"))
                        ade_model_sum += float(ades[int(pred_idx)])
                        if prep.ade.get("min_idx") is not None and int(pred_idx) == int(
                            prep.ade.get("min_idx")
                        ):
                            correct_vs_min_ade += 1
                        if prep.ade.get("min_all_idx") is not None and int(pred_idx) == int(
                            prep.ade.get("min_all_idx")
                        ):
                            correct_vs_min_ade_all += 1
                elif (
                    isinstance(prep.ade, dict)
                    and pred_action == "stop"
                    and selected_ade is not None
                ):
                    ade_count += 1
                    if prep.ade.get("min") is not None:
                        ade_min_sum += float(prep.ade.get("min"))
                    if prep.ade.get("min_all") is not None:
                        ade_min_all_sum += float(prep.ade.get("min_all"))
                    if prep.ade.get("score") is not None:
                        ade_score_sum += float(prep.ade.get("score"))
                    ade_model_sum += float(selected_ade)

                # Prefix ADE aggregates (best-effort).
                try:
                    if selected_ade_0_5s is not None and math.isfinite(float(selected_ade_0_5s)):
                        ade_model_0_5s_sum += float(selected_ade_0_5s)
                        ade_model_0_5s_count += 1
                    if selected_ade_1_0s is not None and math.isfinite(float(selected_ade_1_0s)):
                        ade_model_1_0s_sum += float(selected_ade_1_0s)
                        ade_model_1_0s_count += 1
                    if selected_ade_2_0s is not None and math.isfinite(float(selected_ade_2_0s)):
                        ade_model_2_0s_sum += float(selected_ade_2_0s)
                        ade_model_2_0s_count += 1
                    if score_ade_0_5s is not None and math.isfinite(float(score_ade_0_5s)):
                        ade_score_0_5s_sum += float(score_ade_0_5s)
                        ade_score_0_5s_count += 1
                    if score_ade_1_0s is not None and math.isfinite(float(score_ade_1_0s)):
                        ade_score_1_0s_sum += float(score_ade_1_0s)
                        ade_score_1_0s_count += 1
                    if score_ade_2_0s is not None and math.isfinite(float(score_ade_2_0s)):
                        ade_score_2_0s_sum += float(score_ade_2_0s)
                        ade_score_2_0s_count += 1
                    if min_ade_0_5s is not None and math.isfinite(float(min_ade_0_5s)):
                        ade_min_0_5s_sum += float(min_ade_0_5s)
                        ade_min_0_5s_count += 1
                    if min_ade_1_0s is not None and math.isfinite(float(min_ade_1_0s)):
                        ade_min_1_0s_sum += float(min_ade_1_0s)
                        ade_min_1_0s_count += 1
                    if min_ade_2_0s is not None and math.isfinite(float(min_ade_2_0s)):
                        ade_min_2_0s_sum += float(min_ade_2_0s)
                        ade_min_2_0s_count += 1
                except Exception:
                    pass

                # Route deviation aggregates (rarely available).
                if isinstance(prep.route_dev, dict) and pred_idx is not None:
                    rds = prep.route_dev.get("route_devs")
                    if isinstance(rds, list) and 0 <= int(pred_idx) < len(rds):
                        try:
                            route_dev_count += 1
                            if prep.route_dev.get("min") is not None:
                                route_dev_min_sum += float(prep.route_dev.get("min"))
                            if prep.route_dev.get("score") is not None:
                                route_dev_score_sum += float(prep.route_dev.get("score"))
                            route_dev_model_sum += float(rds[int(pred_idx)])
                            if prep.route_dev.get("min_idx") is not None and int(pred_idx) == int(
                                prep.route_dev.get("min_idx")
                            ):
                                correct_vs_route_min += 1
                            if prep.route_dev.get("score_idx") is not None and int(pred_idx) == int(
                                prep.route_dev.get("score_idx")
                            ):
                                correct_vs_route_score += 1
                        except Exception:
                            pass

                # Write regenerated per-snapshot CSV row.
                goal_x = goal_y = None
                if isinstance(prep.goal_xy, list) and len(prep.goal_xy) >= 2:
                    goal_x, goal_y = prep.goal_xy[0], prep.goal_xy[1]

                ade_selected = None
                ade_min = prep.ade.get("min") if isinstance(prep.ade, dict) else None
                ade_min_all = prep.ade.get("min_all") if isinstance(prep.ade, dict) else None
                ade_score = prep.ade.get("score") if isinstance(prep.ade, dict) else None
                if pred_action == "stop":
                    ade_selected = selected_ade
                elif isinstance(prep.ade, dict) and pred_idx is not None:
                    ades = prep.ade.get("ades")
                    if isinstance(ades, list) and 0 <= int(pred_idx) < len(ades):
                        ade_selected = ades[int(pred_idx)]

                w.writerow(
                    {
                        "episode_id": ep_id,
                        "t": float(t),
                        "snapshot_index": int(snap_idx),
                        "prediction": to_jsonable(pred_idx),
                        "label": to_jsonable(label),
                        "correct": bool(is_correct),
                        "skipped": False,
                        "skip_reason": None,
                        "num_candidates": int(prep.num_candidates),
                        "goal_x": to_jsonable(goal_x),
                        "goal_y": to_jsonable(goal_y),
                        "goal_distance_m": to_jsonable(prep.goal_distance_m),
                        "goal_bearing_deg": to_jsonable(prep.goal_bearing_deg),
                        "selected_end_dist_to_goal_m": to_jsonable(selected_end_dist_to_goal_m),
                        "selected_goal_ang_diff_deg": to_jsonable(selected_goal_ang_diff_deg),
                        "selected_traj_avg_dist_to_goal_m": to_jsonable(
                            selected_traj_avg_dist_to_goal_m
                        ),
                        "selected_goal_progress_m": to_jsonable(selected_goal_progress_m),
                        "traj_len_m": to_jsonable(traj_len_m),
                        "maoe_deg": to_jsonable(maoe_deg),
                        "dcr": to_jsonable(dcr),
                        "tcr": to_jsonable(tcr),
                        "compliance_source": to_jsonable(compliance_source),
                        "fde_selected": to_jsonable(selected_fde_m),
                        "fde_min": to_jsonable(min_fde_m),
                        "fde_min_all": to_jsonable(min_fde_all_m),
                        "fde_score": to_jsonable(score_fde_m),
                        "fde_score_masked_laststep": to_jsonable(score_fde_masked_laststep_m),
                        "moe_deg": to_jsonable(selected_moe_deg),
                        "moe_deg_min": to_jsonable(min_moe_deg),
                        "moe_deg_min_all": to_jsonable(min_moe_all_deg),
                        "moe_deg_score": to_jsonable(score_moe_deg),
                        "moe_deg_mean_all": to_jsonable(mean_moe_all_deg),
                        "ap_fde_2m": to_jsonable(ap_fde_2m),
                        "ade_selected": to_jsonable(ade_selected),
                        "ade_selected_0_5s": to_jsonable(selected_ade_0_5s),
                        "ade_selected_1_0s": to_jsonable(selected_ade_1_0s),
                        "ade_selected_2_0s": to_jsonable(selected_ade_2_0s),
                        "ade_min": to_jsonable(ade_min),
                        "ade_min_all": to_jsonable(ade_min_all),
                        "ade_min_0_5s": to_jsonable(min_ade_0_5s),
                        "ade_min_1_0s": to_jsonable(min_ade_1_0s),
                        "ade_min_2_0s": to_jsonable(min_ade_2_0s),
                        "ade_score": to_jsonable(ade_score),
                        "ade_score_0_5s": to_jsonable(score_ade_0_5s),
                        "ade_score_1_0s": to_jsonable(score_ade_1_0s),
                        "ade_score_2_0s": to_jsonable(score_ade_2_0s),
                        "route_dev_selected": None,
                        "route_dev_min": None,
                        "route_dev_score": None,
                        # Prefer old overlay refs (regen does not create overlays).
                        "overlay_frame_ref": old.get("overlay_frame_ref"),
                        "local_plot_frame_ref": old.get("local_plot_frame_ref"),
                        "auto_enabled": to_jsonable(prep.auto_enabled),
                    }
                )

                # Collect for JSONL.
                json_rec = dict(old)
                # Update ADE dict with new metrics.
                new_ade = {
                    "selected": to_jsonable(ade_selected),
                    "min": to_jsonable(ade_min),
                    "min_all": to_jsonable(ade_min_all),
                    "score": to_jsonable(ade_score),
                    "selected_0_5s": to_jsonable(selected_ade_0_5s),
                    "selected_1_0s": to_jsonable(selected_ade_1_0s),
                    "selected_2_0s": to_jsonable(selected_ade_2_0s),
                    "min_0_5s": to_jsonable(min_ade_0_5s),
                    "min_1_0s": to_jsonable(min_ade_1_0s),
                    "min_2_0s": to_jsonable(min_ade_2_0s),
                    "score_0_5s": to_jsonable(score_ade_0_5s),
                    "score_1_0s": to_jsonable(score_ade_1_0s),
                    "score_2_0s": to_jsonable(score_ade_2_0s),
                }
                # Preserve ades list if it exists.
                if isinstance(prep.ade, dict) and "ades" in prep.ade:
                    new_ade["ades"] = to_jsonable(prep.ade["ades"])
                json_rec["ade"] = new_ade

                json_rec["ap_fde_2m"] = to_jsonable(ap_fde_2m)
                json_rec["fde"] = {
                    "selected": to_jsonable(selected_fde_m),
                    "min": to_jsonable(min_fde_m),
                    "min_all": to_jsonable(min_fde_all_m),
                    "score": to_jsonable(score_fde_m),
                    "score_masked_laststep": to_jsonable(score_fde_masked_laststep_m),
                    "score_idx": to_jsonable(score_idx),
                }
                json_rec["moe"] = {
                    "selected": to_jsonable(selected_moe_deg),
                    "min": to_jsonable(min_moe_deg),
                    "min_all": to_jsonable(min_moe_all_deg),
                    "score": to_jsonable(score_moe_deg),
                    "mean_all": to_jsonable(mean_moe_all_deg),
                    "score_idx": to_jsonable(score_idx),
                }

                # Update other recomputed fields.
                json_rec["selected_end_dist_to_goal_m"] = to_jsonable(selected_end_dist_to_goal_m)
                json_rec["selected_goal_ang_diff_deg"] = to_jsonable(selected_goal_ang_diff_deg)
                json_rec["selected_traj_avg_dist_to_goal_m"] = to_jsonable(
                    selected_traj_avg_dist_to_goal_m
                )
                json_rec["selected_goal_progress_m"] = to_jsonable(selected_goal_progress_m)
                json_rec["traj_len_m"] = to_jsonable(traj_len_m)
                json_rec["maoe_deg"] = to_jsonable(maoe_deg)
                json_rec["dcr"] = to_jsonable(dcr)
                json_rec["tcr"] = to_jsonable(tcr)
                json_rec["compliance_source"] = to_jsonable(compliance_source)
                json_rec["correct"] = bool(is_correct)
                json_rec["auto_enabled"] = to_jsonable(prep.auto_enabled)

                new_jsonl_recs.append(json_rec)

    # Write regenerated JSONL.
    with preds_jsonl_path.open("w", encoding="utf-8") as fjl:
        for r in new_jsonl_recs:
            fjl.write(json.dumps(r) + "\n")

    # Metrics dict (largely matches slow_brain_fast_planner.cli.trajectory_selection output).
    pred_error_count = int(
        sum(int(v) for k, v in skipped_reasons.items() if str(k).startswith("pred:"))
    )
    pred_output_total = int(evaluated) + int(pred_error_count)
    # Prefer identifying the run by the actual directory name (robust to old/merged configs).
    exp_name_path = str(run_dir.parent.name) if run_dir.parent is not None else str(run_dir)
    trial_name_path = str(run_dir.name)

    t_end = _dt.datetime.now(tz=_dt.UTC)
    duration_s = (t_end - t_start).total_seconds()

    metrics: dict[str, Any] = {
        "slow_brain_fast_planner_version": __version__,
        "job_started_at_utc": t_start.isoformat(),
        "job_finished_at_utc": t_end.isoformat(),
        "job_duration_s": duration_s,
        "created_at_utc": _utc_now_iso(),
        "task": "trajectory_selection",
        "dataset": str(dataset_path),
        "planner_source": cfg.get("planner_source"),
        "experiment_name": exp_name_path,
        "trial_name": trial_name_path,
        "seed": int(cfg.get("seed", 0) or 0),
        "selector": str(
            ((cfg.get("model") or {}) if isinstance(cfg.get("model"), dict) else {}).get(
                "adapter", "unknown"
            )
        ),
        "regen": {
            "from_run_dir": str(run_dir),
            "config_path": str(cfg_path),
            "predictions_jsonl_path": str(pred_path),
            "wrote_inplace": bool(args.inplace),
            "prep_write_overlays": False,
            "config_experiment_name": cfg.get("experiment_name"),
            "config_trial_name": cfg.get("trial_name"),
            "a0_cfg": asdict(a0_cfg),
            "overlay_cfg": asdict(overlay_cfg),
        },
        "episodes_total": int(len(episode_meta_paths)),
        "episodes_schema_invalid": 0,
        "snapshots_total": int(total_snapshots),
        "snapshots_evaluated": int(evaluated),
        "snapshots_skipped": int(total_snapshots - evaluated),
        "skipped_reasons": {k: int(skipped_reasons[k]) for k in sorted(skipped_reasons)},
        "accuracy": safe_div(correct, evaluated),
        "accuracy_vs_min_ade": safe_div(correct_vs_min_ade, ade_count),
        "accuracy_vs_min_ade_all": safe_div(correct_vs_min_ade_all, ade_count),
        "accuracy_vs_score": safe_div(correct_vs_score, score_agree_count),
        "stop_count": int(stop_count),
        "stop_rate": safe_div(stop_count, evaluated),
        "invalid_index_count": int(invalid_index_count),
        "invalid_index_rate": safe_div(invalid_index_count, evaluated),
        "pred_error_count": int(pred_error_count),
        "pred_error_rate": safe_div(pred_error_count, pred_output_total),
        "ade_min_avg": safe_div(ade_min_sum, ade_count),
        "ade_min_all_avg": safe_div(ade_min_all_sum, ade_count),
        "ade_score_avg": safe_div(ade_score_sum, ade_count),
        "ade_model_avg": safe_div(ade_model_sum, ade_count),
        "ade_model_0_5s_avg": safe_div(ade_model_0_5s_sum, ade_model_0_5s_count),
        "ade_model_1_0s_avg": safe_div(ade_model_1_0s_sum, ade_model_1_0s_count),
        "ade_model_2_0s_avg": safe_div(ade_model_2_0s_sum, ade_model_2_0s_count),
        "ade_score_0_5s_avg": safe_div(ade_score_0_5s_sum, ade_score_0_5s_count),
        "ade_score_1_0s_avg": safe_div(ade_score_1_0s_sum, ade_score_1_0s_count),
        "ade_score_2_0s_avg": safe_div(ade_score_2_0s_sum, ade_score_2_0s_count),
        "ade_min_0_5s_avg": safe_div(ade_min_0_5s_sum, ade_min_0_5s_count),
        "ade_min_1_0s_avg": safe_div(ade_min_1_0s_sum, ade_min_1_0s_count),
        "ade_min_2_0s_avg": safe_div(ade_min_2_0s_sum, ade_min_2_0s_count),
        "ade_count": int(ade_count),
        "fde_model_avg": safe_div(fde_model_sum, fde_model_count),
        "fde_score_avg": safe_div(fde_score_sum, fde_score_count),
        "fde_score_masked_laststep_avg": safe_div(
            fde_score_masked_laststep_sum, fde_score_masked_laststep_count
        ),
        "fde_min_avg": safe_div(fde_min_sum, fde_min_count),
        "fde_min_all_avg": safe_div(fde_min_all_sum, fde_min_all_count),
        "fde_count": int(fde_model_count),
        "fde_score_masked_laststep_count": int(fde_score_masked_laststep_count),
        "moe_deg_avg": safe_div(moe_model_sum, moe_model_count),
        "moe_deg_score_avg": safe_div(moe_score_sum, moe_score_count),
        "moe_deg_min_avg": safe_div(moe_min_sum, moe_min_count),
        "moe_deg_min_all_avg": safe_div(moe_min_all_sum, moe_min_all_count),
        "moe_deg_mean_all_avg": safe_div(moe_mean_all_sum, moe_mean_all_count),
        "moe_deg_count": int(moe_model_count),
        "map_fde_2m_avg": safe_div(ap_fde_2m_sum, ap_fde_2m_count),
        "map_fde_2m_count": int(ap_fde_2m_count),
        "route_dev_min_avg": safe_div(route_dev_min_sum, route_dev_count),
        "route_dev_score_avg": safe_div(route_dev_score_sum, route_dev_count),
        "route_dev_model_avg": safe_div(route_dev_model_sum, route_dev_count),
        "route_dev_count": int(route_dev_count),
        "accuracy_vs_route_min": safe_div(correct_vs_route_min, route_dev_count),
        "accuracy_vs_route_score": safe_div(correct_vs_route_score, route_dev_count),
        "auto_enabled_true_count": int(auto_true),
        "auto_enabled_false_count": int(auto_false),
        "auto_enabled_missing_count": int(auto_missing),
        "auto_enabled_true_frac": None
        if (auto_true + auto_false) == 0
        else float(auto_true) / float(auto_true + auto_false),
        "goal_distance_m_avg": safe_div(goal_dist_sum, goal_dist_count),
        "goal_distance_m_count": int(goal_dist_count),
        "selected_end_dist_to_goal_m_avg": safe_div(sel_end_goal_dist_sum, sel_end_goal_dist_count),
        "selected_end_dist_to_goal_m_count": int(sel_end_goal_dist_count),
        "selected_goal_ang_diff_deg_avg": safe_div(sel_goal_ang_diff_sum, sel_goal_ang_diff_count),
        "selected_goal_ang_diff_deg_count": int(sel_goal_ang_diff_count),
        "selected_traj_avg_dist_to_goal_m_avg": safe_div(
            sel_traj_avg_goal_dist_sum, sel_traj_avg_goal_dist_count
        ),
        "selected_traj_avg_dist_to_goal_m_count": int(sel_traj_avg_goal_dist_count),
        "selected_goal_progress_m_avg": safe_div(sel_goal_progress_sum, sel_goal_progress_count),
        "selected_goal_progress_m_count": int(sel_goal_progress_count),
        "maoe_deg_avg": safe_div(maoe_sum, maoe_count),
        "dcr_avg": safe_div(dcr_sum, dcr_count),
        "tcr_avg": safe_div(tcr_sum, tcr_count),
        "counts": {"correct": int(correct)},
        "model_calls": int(model_calls),
        "prompt_tokens_total": int(prompt_tokens_sum),
        "cached_prompt_tokens_total": int(cached_prompt_tokens_sum),
        "output_tokens_total": int(output_tokens_sum),
        "thoughts_tokens_total": int(thoughts_tokens_sum),
        "total_tokens_total": int(total_tokens_sum),
        "total_tokens_no_thoughts_total": int(prompt_tokens_sum) + int(output_tokens_sum),
        "prompt_tokens_avg": safe_div(prompt_tokens_sum, model_calls),
        "cached_prompt_tokens_avg": safe_div(cached_prompt_tokens_sum, model_calls),
        "output_tokens_avg": safe_div(output_tokens_sum, model_calls),
        "thoughts_tokens_avg": safe_div(thoughts_tokens_sum, model_calls),
        "total_tokens_avg": safe_div(total_tokens_sum, model_calls),
        "total_tokens_no_thoughts_avg": safe_div(
            int(prompt_tokens_sum) + int(output_tokens_sum), model_calls
        ),
        "latency_ms_avg": None,
        "latency_ms_p50": None,
        "latency_ms_p90": None,
    }
    if latency_ms_list:
        lat_sorted = sorted(latency_ms_list)
        n = len(lat_sorted)
        metrics["latency_ms_avg"] = float(sum(lat_sorted) / float(n))
        metrics["latency_ms_p50"] = float(lat_sorted[int(0.50 * (n - 1))])
        metrics["latency_ms_p90"] = float(lat_sorted[int(0.90 * (n - 1))])

    # Write outputs.
    _write_json(metrics_json_path, metrics)
    (report_txt_path).write_text(_format_metrics_report(metrics, run_dir=out_dir), encoding="utf-8")
    _write_metrics_summary_csv(summary_csv_path, metrics)
    _write_metrics_summary_simple_csv(summary_simple_csv_path, metrics)
    _write_metrics_summary_simple_txt(summary_simple_csv_path, summary_simple_txt_path)

    # Generate the updated HTML report.
    try:
        generate_run_report_html(
            run_dir=run_dir,
            out_path=report_html_path,
            predictions_path=preds_jsonl_path,
            embed_images=False,
        )
    except Exception as e:
        print(f"[regen_task2] warning: could not generate report: {e}")

    # Also print a short pointer for CLI usage in logs.
    print(f"[regen_task2] wrote: {metrics_json_path}")
    print(f"[regen_task2] wrote: {summary_simple_csv_path}")
    print(f"[regen_task2] wrote: {preds_csv_path}")
    print(f"[regen_task2] wrote: {preds_jsonl_path}")
    print(f"[regen_task2] wrote: {report_html_path}")


if __name__ == "__main__":
    main()
