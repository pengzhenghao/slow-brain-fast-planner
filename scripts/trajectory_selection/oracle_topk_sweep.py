#!/usr/bin/env python3
"""Trajectory-selection oracle-min-ADE sweeps over candidate-pool variants and K.

Goal: compute "oracle min ADE" when the oracle is restricted to different candidate pools:
- raw_topk (pre-NMS): top-K by raw planner score
- nms_only (post-NMS, no prob threshold)
- planner_v2 (post-NMS + softmax prob threshold)
- all (global oracle over all candidates)

This script runs a *single* dataset pass (loads snapshots once) and reports results in the
same column schema as `metrics_summary_simple.csv` (copy/paste friendly).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from slow_brain_fast_planner import __version__
from slow_brain_fast_planner.benchmarks.data_loading import (
    SnapshotDataset,
    build_trajectory_selection_jobs_from_takeover_clips,
)
from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files
from slow_brain_fast_planner.benchmarks.overlays import OverlayConfig
from slow_brain_fast_planner.benchmarks.planner_postprocessing import (
    A0Config,
    _softmax_stable,
    _trajectory_nms_endpoints,
    safe_div,
)
from slow_brain_fast_planner.cli.trajectory_selection import (
    _ade_prefix_seconds,
    _ap_at_threshold,
    _dcr_tcr_from_corridor,
    _fde_m,
    _fde_m_masked_laststep,
    _maoe_deg,
    _moe_deg_posvec,
)
from slow_brain_fast_planner.utils.io import utc_now_iso as _utc_now_iso

_OUT_COLS = [
    "experiment_name",
    "trial_name",
    "task",
    "dataset",
    "planner_source",
    "selector",
    "seed",
    "snapshots_evaluated",
    "accuracy",
    "accuracy_vs_score",
    "accuracy_vs_min_ade",
    "accuracy_vs_min_ade_all",
    "stop_rate",
    "stop_count",
    "invalid_index_rate",
    "invalid_index_count",
    "pred_error_rate",
    "pred_error_count",
    "ade_model_avg",
    "ade_model_0_5s_avg",
    "ade_model_1_0s_avg",
    "ade_model_2_0s_avg",
    "ade_score_avg",
    "ade_score_0_5s_avg",
    "ade_score_1_0s_avg",
    "ade_score_2_0s_avg",
    "ade_min_avg",
    "ade_min_0_5s_avg",
    "ade_min_1_0s_avg",
    "ade_min_2_0s_avg",
    "ade_min_all_avg",
    "fde_model_avg",
    "fde_score_avg",
    "fde_score_masked_laststep_avg",
    "fde_min_avg",
    "fde_min_all_avg",
    "moe_deg_avg",
    "moe_deg_score_avg",
    "moe_deg_min_avg",
    "moe_deg_min_all_avg",
    "moe_deg_mean_all_avg",
    "map_fde_2m_avg",
    "selected_end_dist_to_goal_m_avg",
    "selected_goal_ang_diff_deg_avg",
    "selected_traj_avg_dist_to_goal_m_avg",
    "selected_goal_progress_m_avg",
    "maoe_deg_avg",
    "dcr_avg",
    "tcr_avg",
    "created_at_utc",
    "slow_brain_fast_planner_version",
    "model_calls",
    "prompt_tokens_total",
    "output_tokens_total",
    "total_tokens_total",
    "total_tokens_avg",
    "latency_ms_avg",
    "latency_ms_p50",
    "latency_ms_p90",
]


@dataclass
class _Agg:
    evaluated: int = 0
    correct: int = 0

    # Agreement counters
    score_agree_count: int = 0
    correct_vs_score: int = 0

    # ADE counters
    ade_count: int = 0
    correct_vs_min_ade: int = 0
    correct_vs_min_ade_all: int = 0

    ade_min_sum: float = 0.0
    ade_min_all_sum: float = 0.0
    ade_score_sum: float = 0.0
    ade_model_sum: float = 0.0

    # Prefix ADE sums + counts (best-effort)
    ade_model_0_5s_sum: float = 0.0
    ade_model_0_5s_count: int = 0
    ade_model_1_0s_sum: float = 0.0
    ade_model_1_0s_count: int = 0
    ade_model_2_0s_sum: float = 0.0
    ade_model_2_0s_count: int = 0

    ade_score_0_5s_sum: float = 0.0
    ade_score_0_5s_count: int = 0
    ade_score_1_0s_sum: float = 0.0
    ade_score_1_0s_count: int = 0
    ade_score_2_0s_sum: float = 0.0
    ade_score_2_0s_count: int = 0

    ade_min_0_5s_sum: float = 0.0
    ade_min_0_5s_count: int = 0
    ade_min_1_0s_sum: float = 0.0
    ade_min_1_0s_count: int = 0
    ade_min_2_0s_sum: float = 0.0
    ade_min_2_0s_count: int = 0

    # Goal-relative metrics
    sel_end_goal_dist_sum: float = 0.0
    sel_end_goal_dist_count: int = 0
    sel_goal_ang_diff_sum: float = 0.0
    sel_goal_ang_diff_count: int = 0
    sel_traj_avg_goal_dist_sum: float = 0.0
    sel_traj_avg_goal_dist_count: int = 0
    sel_goal_progress_sum: float = 0.0
    sel_goal_progress_count: int = 0

    # Open-loop "social" metrics
    maoe_sum: float = 0.0
    maoe_count: int = 0
    dcr_sum: float = 0.0
    dcr_count: int = 0
    tcr_sum: float = 0.0
    tcr_count: int = 0

    # FDE / MOE / mAP (rss-style; best-effort)
    fde_model_sum: float = 0.0
    fde_model_count: int = 0
    fde_score_sum: float = 0.0
    fde_score_count: int = 0
    fde_score_masked_laststep_sum: float = 0.0
    fde_score_masked_laststep_count: int = 0
    fde_min_sum: float = 0.0
    fde_min_count: int = 0
    fde_min_all_sum: float = 0.0
    fde_min_all_count: int = 0

    moe_model_sum: float = 0.0
    moe_model_count: int = 0
    moe_score_sum: float = 0.0
    moe_score_count: int = 0
    moe_min_sum: float = 0.0
    moe_min_count: int = 0
    moe_min_all_sum: float = 0.0
    moe_min_all_count: int = 0
    moe_mean_all_sum: float = 0.0
    moe_mean_all_count: int = 0

    ap_fde_2m_sum: float = 0.0
    ap_fde_2m_count: int = 0

    # Output health (always zero for oracle selector)
    stop_count: int = 0
    invalid_index_count: int = 0
    pred_error_count: int = 0
    pred_output_total: int = 0


def _rank_by_score(scores: np.ndarray) -> list[int]:
    K = int(scores.shape[0])
    return sorted(range(K), key=lambda i: (-float(scores[i]), int(i)))


def _select_visible_indices(
    *,
    pool: str,
    scores: np.ndarray,
    endpoints_xy: np.ndarray,
    top_k: int,
    nms_max_trajectories: int,
    nms_distance_threshold: float,
    prob_threshold: float,
) -> list[int]:
    K = int(scores.shape[0])
    if K <= 0:
        return []

    ranked = _rank_by_score(scores)
    k = int(min(int(top_k), int(K)))
    if k <= 0:
        return []

    if pool == "raw_topk":
        return [int(i) for i in ranked[:k]]

    if pool == "nms_only":
        keep = _trajectory_nms_endpoints(
            scores=scores,
            endpoints_xy=endpoints_xy,
            max_trajectories=int(nms_max_trajectories),
            distance_threshold=float(nms_distance_threshold),
        )
        if keep.size == 0:
            return [int(ranked[0])]
        return [int(i) for i in keep[:k]]

    if pool == "planner_v2":
        keep = _trajectory_nms_endpoints(
            scores=scores,
            endpoints_xy=endpoints_xy,
            max_trajectories=int(nms_max_trajectories),
            distance_threshold=float(nms_distance_threshold),
        )
        if keep.size == 0:
            return [int(ranked[0])]
        probs = _softmax_stable(scores[keep])
        filtered = [
            (int(keep[i]), float(probs[i]))
            for i in range(len(keep))
            if float(probs[i]) >= float(prob_threshold)
        ]
        if not filtered:
            # Match overlay behavior: fall back to raw top-k if filtering removed everything.
            return [int(i) for i in ranked[:k]]
        filtered.sort(key=lambda x: (-x[1], x[0]))
        kept_sorted = [int(i) for i, _p in filtered]
        return [int(i) for i in kept_sorted[:k]]

    raise ValueError(f"Unknown pool: {pool}")


def _fmt(v) -> str:
    # TSV-friendly: empty for None, otherwise str(v)
    if v is None:
        return ""
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        # Match existing CSV-ish style (no forced rounding).
        return repr(float(v))
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        help="Canonical dataset root (e.g. data/slow-brain-fast-planner/hard).",
    )
    ap.add_argument(
        "--planner-source",
        default="prelogged",
        choices=["prelogged"],
        help="For reporting only (oracle sweep reads prelogged candidates).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=None)
    ap.add_argument(
        "--k",
        type=int,
        action="append",
        default=None,
        help="Top-K values to sweep (repeatable). If omitted: 6,12,18,24,30",
    )
    ap.add_argument("--nms-max-trajectories", type=int, default=64)
    ap.add_argument("--nms-distance-threshold", type=float, default=2.0)
    ap.add_argument("--prob-threshold", type=float, default=0.0)
    ap.add_argument("--traj-dt-s", type=float, default=0.2)
    ap.add_argument(
        "--clips-jsonl",
        default=None,
        help=(
            "Optional explicit clips JSONL path. If not provided, we auto-detect common dataset "
            "layouts:\n"
            "- <dataset>/takeover_clips/takeover_clips.jsonl\n"
            "- <dataset>/task2_eval_clips/task2_eval_clips.jsonl"
        ),
    )
    ap.add_argument("--out-tsv", default=None, help="Optional path to also write the TSV output.")
    ap.add_argument("--out-csv", default=None, help="Optional path to also write the CSV output.")
    args = ap.parse_args()

    dataset_path = Path(str(args.dataset)).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"--dataset does not exist: {dataset_path}")

    # Match default benchmark behavior: use takeover clips if available.
    if args.clips_jsonl is not None and str(args.clips_jsonl).strip():
        takeover_clips_path = Path(str(args.clips_jsonl)).resolve()
        if not takeover_clips_path.exists():
            raise SystemExit(f"--clips-jsonl not found: {takeover_clips_path}")
    else:
        candidates = [
            (dataset_path / "takeover_clips" / "takeover_clips.jsonl").resolve(),
            (
                dataset_path
                / "trajectory_selection_eval_clips"
                / "trajectory_selection_eval_clips.jsonl"
            ).resolve(),
            (dataset_path / "task2_eval_clips" / "task2_eval_clips.jsonl").resolve(),
        ]
        takeover_clips_path = None
        for p in candidates:
            if p.exists():
                takeover_clips_path = p
                break
        if takeover_clips_path is None:
            tried = "\n".join([f"- {p}" for p in candidates])
            raise SystemExit(
                f"Expected clips file missing. Tried:\n{tried}\n"
                "(Or pass --clips-jsonl /path/to/*.jsonl)"
            )

    episode_meta_paths = find_episode_metadata_files(dataset_path)

    # Build jobs with overlays disabled (we only need GT + candidates).
    # Note: overlay_cfg here is only used during prep; the sweep recomputes candidate pools
    # independently.
    overlay_cfg = OverlayConfig(candidate_set="planner_v2", top_k=6)
    a0_cfg = A0Config(gt_rule="raw_argmax")

    # Run output dir: only used for optional artifacts; keep it under logs/.
    run_out_dir = (Path("logs") / "tmp_task2_oracle_topk_sweep").resolve()

    jobs = build_trajectory_selection_jobs_from_takeover_clips(
        dataset_path=dataset_path,
        episode_meta_paths=episode_meta_paths,
        takeover_clips_path=takeover_clips_path,
        out_dir=run_out_dir,
        a0_cfg=a0_cfg,
        overlay_cfg=overlay_cfg,
        write_overlays=False,
        compute_gt_metrics=True,
        traj_dt_s=float(args.traj_dt_s),
        clip_label_filter="any",
    )
    dataset = SnapshotDataset(jobs)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        prefetch_factor=(
            int(args.prefetch_factor)
            if args.prefetch_factor is not None and int(args.num_workers) > 0
            else None
        ),
        collate_fn=SnapshotDataset.collate_fn,
    )

    ks_in = args.k if args.k is not None else [6, 12, 18, 24, 30]
    ks = sorted({int(k) for k in ks_in if int(k) > 0})
    pools = ["raw_topk", "nms_only", "planner_v2"]

    # One "global oracle" row (all candidates).
    exp_specs: list[tuple[str, str, int | None]] = []
    exp_specs.append(("oracle_min_ade_all", "all", None))
    for pool in pools:
        for k in ks:
            exp_specs.append(("oracle_min_ade", pool, int(k)))

    aggs: dict[tuple[str, str, int | None], _Agg] = {spec: _Agg() for spec in exp_specs}

    # Progress bar (best-effort).
    try:
        from tqdm import tqdm  # type: ignore

        loader_iter = tqdm(loader, desc="task2 oracle sweep", total=len(loader))
    except Exception:
        loader_iter = loader

    for batch in loader_iter:
        if not batch:
            continue
        item = batch[0]
        prep = getattr(item, "prep", None)
        if prep is None:
            continue

        # Evaluation gating (match CLI notion of "evaluated" as closely as possible).
        if not bool(getattr(prep, "ok", False)):
            continue
        if getattr(prep, "skip_reason", None) is not None:
            continue
        if getattr(prep, "label", None) is None or getattr(prep, "label_err", None) is not None:
            continue

        label = int(prep.label)
        score_idx = int(prep.score_idx) if prep.score_idx is not None else None
        num_cands = int(prep.num_candidates)

        # Candidate scores (needed for top-k selection).
        raw_scores = getattr(prep, "candidate_scores_raw", None)
        if not isinstance(raw_scores, list) or len(raw_scores) != num_cands:
            raw_scores = None

        # Candidate points (needed for endpoints + prefixes).
        cand_xy = getattr(prep, "candidates_points_xy", None)
        if not isinstance(cand_xy, list) or len(cand_xy) != num_cands:
            cand_xy = None

        # ADE list over ALL candidates (global).
        ade_dict = getattr(prep, "ade", None)
        ades_all = None
        if isinstance(ade_dict, dict) and isinstance(ade_dict.get("ades"), list):
            try:
                ades_all = [float(x) for x in ade_dict.get("ades")]  # type: ignore[arg-type]
                if len(ades_all) != num_cands:
                    ades_all = None
            except Exception:
                ades_all = None

        # GT future (robot frame), for prefixes + open-loop metrics.
        gt_xy = None
        if isinstance(getattr(prep, "gt_local_traj_xy", None), list) and prep.gt_local_traj_xy:
            try:
                gt_xy = np.asarray(prep.gt_local_traj_xy, dtype=np.float64)
            except Exception:
                gt_xy = None

        # Compute endpoints (for NMS).
        endpoints_xy = None
        if cand_xy is not None:
            try:
                ep = []
                for pts in cand_xy:
                    if isinstance(pts, list) and pts:
                        last = pts[-1]
                        ep.append([float(last[0]), float(last[1])])
                    else:
                        ep.append([0.0, 0.0])
                endpoints_xy = np.asarray(ep, dtype=np.float64)
                if endpoints_xy.ndim != 2 or endpoints_xy.shape[1] != 2:
                    endpoints_xy = np.zeros((num_cands, 2), dtype=np.float64)
            except Exception:
                endpoints_xy = np.zeros((num_cands, 2), dtype=np.float64)
        else:
            endpoints_xy = np.zeros((num_cands, 2), dtype=np.float64)

        # Precompute rss-style FDE/MOE over ALL candidates once.
        cand_fdes: list[float | None] | None = None
        cand_moes: list[float | None] | None = None
        min_all_fde: float | None = None
        min_all_moe: float | None = None
        mean_all_moe: float | None = None
        ap_fde_2m: float | None = None
        score_fde_masked_laststep_m: float | None = None
        if cand_xy is not None and gt_xy is not None:
            try:
                gt2 = np.asarray(gt_xy, dtype=np.float64)
                if gt2.ndim == 2 and gt2.shape[0] > 0 and gt2.shape[1] >= 2:
                    gt2 = gt2[:, :2]
                else:
                    gt2 = None
            except Exception:
                gt2 = None
            if gt2 is not None:
                cand_fdes = []
                cand_moes = []
                for pts in cand_xy:
                    try:
                        p = np.asarray(pts, dtype=np.float64)
                        if p.ndim == 2 and p.shape[0] > 0 and p.shape[1] >= 2:
                            p2 = p[:, :2]
                        else:
                            p2 = None
                    except Exception:
                        p2 = None
                    cand_fdes.append(_fde_m(p2, gt2))
                    cand_moes.append(_moe_deg_posvec(p2, gt2))

                # Score-argmax endpoint error, gated on having a valid final step (GT covers pred
                # horizon).
                try:
                    if score_idx is not None and 0 <= int(score_idx) < len(cand_xy):
                        p = np.asarray(cand_xy[int(score_idx)], dtype=np.float64)
                        p2 = (
                            p[:, :2]
                            if (p.ndim == 2 and p.shape[0] > 0 and p.shape[1] >= 2)
                            else None
                        )
                        score_fde_masked_laststep_m = _fde_m_masked_laststep(p2, gt2)
                except Exception:
                    score_fde_masked_laststep_m = None

                # Oracle minima over ALL candidates (best-effort).
                try:
                    f_ok = [
                        float(x) for x in cand_fdes if x is not None and math.isfinite(float(x))
                    ]
                    if f_ok:
                        min_all_fde = float(min(f_ok))
                except Exception:
                    min_all_fde = None
                try:
                    m_ok = [
                        float(x) for x in cand_moes if x is not None and math.isfinite(float(x))
                    ]
                    if m_ok:
                        min_all_moe = float(min(m_ok))
                        mean_all_moe = float(sum(m_ok) / float(len(m_ok)))
                except Exception:
                    min_all_moe = None
                    mean_all_moe = None

                # mAP@2m using raw scores as ranking (rss-style AP on per-snapshot labels).
                try:
                    if raw_scores is not None and endpoints_xy is not None:
                        scores_np0 = np.asarray([float(s) for s in raw_scores], dtype=np.float64)
                        keep = _trajectory_nms_endpoints(
                            scores=scores_np0,
                            endpoints_xy=endpoints_xy,
                            max_trajectories=int(args.nms_max_trajectories),
                            distance_threshold=float(args.nms_distance_threshold),
                        ).tolist()
                        keep_i = [int(i) for i in keep if 0 <= int(i) < len(cand_fdes)]
                        ap_fde_2m = _ap_at_threshold(
                            scores=[float(raw_scores[i]) for i in keep_i],
                            fdes_m=[cand_fdes[i] for i in keep_i],
                            threshold_m=2.0,
                        )
                except Exception:
                    ap_fde_2m = None

        scores_np = np.asarray(raw_scores, dtype=np.float64) if raw_scores is not None else None

        # Baseline indices/values for this snapshot (if ADE exists).
        min_all_idx = None
        min_all_val = None
        score_val = None
        if ades_all is not None and ades_all:
            min_all_idx = int(np.argmin(np.asarray(ades_all, dtype=np.float64)))
            min_all_val = float(ades_all[min_all_idx])
            if score_idx is not None and 0 <= int(score_idx) < len(ades_all):
                score_val = float(ades_all[int(score_idx)])

        # "Default visible" oracle baseline (matches default overlay behavior: planner_v2, K=6).
        default_min_vis_idx = None
        default_min_vis_val = None
        if ades_all is not None and min_all_idx is not None and scores_np is not None:
            try:
                default_visible = _select_visible_indices(
                    pool="planner_v2",
                    scores=scores_np,
                    endpoints_xy=endpoints_xy,
                    top_k=6,
                    nms_max_trajectories=6,
                    nms_distance_threshold=float(args.nms_distance_threshold),
                    prob_threshold=float(args.prob_threshold),
                )
            except Exception:
                default_visible = []
            vis0 = [int(i) for i in default_visible if 0 <= int(i) < len(ades_all)]
            if vis0:
                vis_ades0 = np.asarray([ades_all[i] for i in vis0], dtype=np.float64)
                default_min_vis_idx = int(vis0[int(np.argmin(vis_ades0))])
                default_min_vis_val = float(ades_all[default_min_vis_idx])
            else:
                default_min_vis_idx = int(min_all_idx)
                default_min_vis_val = float(min_all_val) if min_all_val is not None else None

        # Helper: compute goal-relative metrics for a selected trajectory.
        def _acc_goal_metrics(agg: _Agg, pts0: np.ndarray | None, *, prep=prep) -> None:
            if pts0 is None or pts0.ndim != 2 or pts0.shape[0] <= 0 or pts0.shape[1] < 2:
                return
            if not (
                isinstance(getattr(prep, "goal_xy", None), list)
                and prep.goal_xy
                and len(prep.goal_xy) >= 2
            ):
                return
            goal_xy = np.asarray([float(prep.goal_xy[0]), float(prep.goal_xy[1])], dtype=np.float64)
            end_xy = pts0[-1, :2]
            d_end = float(np.linalg.norm(end_xy - goal_xy))
            d_all = np.linalg.norm(pts0[:, :2] - goal_xy.reshape(1, 2), axis=1)
            d_avg = float(np.mean(d_all)) if d_all.size > 0 else None

            goal_ang = float(math.degrees(math.atan2(float(goal_xy[1]), float(goal_xy[0]))))
            end_ang = float(math.degrees(math.atan2(float(end_xy[1]), float(end_xy[0]))))
            diff = (end_ang - goal_ang + 180.0) % 360.0 - 180.0
            ang_diff = abs(float(diff))

            if math.isfinite(d_end):
                agg.sel_end_goal_dist_sum += float(d_end)
                agg.sel_end_goal_dist_count += 1
            if d_avg is not None and math.isfinite(float(d_avg)):
                agg.sel_traj_avg_goal_dist_sum += float(d_avg)
                agg.sel_traj_avg_goal_dist_count += 1
            if math.isfinite(ang_diff):
                agg.sel_goal_ang_diff_sum += float(ang_diff)
                agg.sel_goal_ang_diff_count += 1

            if prep.goal_distance_m is not None:
                gd0 = float(prep.goal_distance_m)
                if math.isfinite(gd0) and math.isfinite(d_end):
                    prog = float(gd0) - float(d_end)
                    if math.isfinite(prog):
                        agg.sel_goal_progress_sum += float(prog)
                        agg.sel_goal_progress_count += 1

        def _prefix(pts: np.ndarray | None, seconds: float, *, gt_xy=gt_xy) -> float | None:
            if pts is None or gt_xy is None:
                return None
            return _ade_prefix_seconds(
                pts, gt_xy, dt_s=float(args.traj_dt_s), seconds=float(seconds)
            )

        # Precompute score candidate prefix ADEs once (used in every run row).
        score_pts = None
        if score_idx is not None and cand_xy is not None and 0 <= int(score_idx) < len(cand_xy):
            try:
                score_pts = np.asarray(cand_xy[int(score_idx)], dtype=np.float64)
            except Exception:
                score_pts = None

        score_ade_0_5s = _prefix(score_pts, 0.5)
        score_ade_1_0s = _prefix(score_pts, 1.0)
        score_ade_2_0s = _prefix(score_pts, 2.0)

        # For each experiment spec, compute selected idx and accumulate.
        for (selector, pool, k), agg in aggs.items():
            agg.evaluated += 1
            agg.pred_output_total += 1

            # Determine selected index.
            selected_idx = int(score_idx) if score_idx is not None else 0
            min_vis_idx = None
            min_vis_val = None
            vis: list[int] = []

            if ades_all is not None and min_all_idx is not None:
                if selector == "oracle_min_ade_all":
                    # Prediction: global oracle over ALL candidates.
                    selected_idx = int(min_all_idx)
                    # Baseline "visible oracle" for reporting: default visible set (planner_v2,
                    # K=6).
                    min_vis_idx = (
                        int(default_min_vis_idx)
                        if default_min_vis_idx is not None
                        else int(min_all_idx)
                    )
                    min_vis_val = (
                        float(default_min_vis_val)
                        if default_min_vis_val is not None
                        else float(min_all_val)
                    )
                    try:
                        vis = [
                            int(i) for i in (default_visible or []) if 0 <= int(i) < int(num_cands)
                        ]
                    except Exception:
                        vis = []
                else:
                    visible_indices: list[int] = []
                    if scores_np is not None:
                        try:
                            visible_indices = _select_visible_indices(
                                pool=str(pool),
                                scores=scores_np,
                                endpoints_xy=endpoints_xy,
                                top_k=int(k or 1),
                                nms_max_trajectories=int(args.nms_max_trajectories),
                                nms_distance_threshold=float(args.nms_distance_threshold),
                                prob_threshold=float(args.prob_threshold),
                            )
                        except Exception:
                            visible_indices = []
                    vis = [int(i) for i in (visible_indices or []) if 0 <= int(i) < len(ades_all)]
                    if vis:
                        vis_ades = np.asarray([ades_all[i] for i in vis], dtype=np.float64)
                        min_vis_idx = int(vis[int(np.argmin(vis_ades))])
                        min_vis_val = float(ades_all[min_vis_idx])
                    else:
                        min_vis_idx = int(min_all_idx)
                        min_vis_val = float(min_all_val) if min_all_val is not None else None
                    # Prediction: visible oracle in this pool/K.
                    selected_idx = int(min_vis_idx)

            # Health: invalid index
            if not (0 <= int(selected_idx) < int(num_cands)):
                agg.invalid_index_count += 1
                continue

            if int(selected_idx) == int(label):
                agg.correct += 1

            # accuracy_vs_score
            if score_idx is not None and 0 <= int(score_idx) < int(num_cands):
                agg.score_agree_count += 1
                if int(selected_idx) == int(score_idx):
                    agg.correct_vs_score += 1

            # ADE aggregates (only if we have candidate ADEs).
            if ades_all is not None and 0 <= int(selected_idx) < len(ades_all):
                agg.ade_count += 1
                agg.ade_min_sum += (
                    float(min_vis_val)
                    if min_vis_val is not None
                    else float(ades_all[int(selected_idx)])
                )
                if min_vis_idx is not None and int(selected_idx) == int(min_vis_idx):
                    agg.correct_vs_min_ade += 1

                agg.ade_min_all_sum += (
                    float(min_all_val)
                    if min_all_val is not None
                    else float(ades_all[int(selected_idx)])
                )
                agg.ade_score_sum += float(score_val) if score_val is not None else 0.0
                agg.ade_model_sum += float(ades_all[int(selected_idx)])

                if min_all_idx is not None and int(selected_idx) == int(min_all_idx):
                    agg.correct_vs_min_ade_all += 1

            # Prefix ADE + goal + open-loop metrics (best-effort; require candidate points + GT).
            pts0 = None
            if cand_xy is not None and 0 <= int(selected_idx) < len(cand_xy):
                try:
                    pts0 = np.asarray(cand_xy[int(selected_idx)], dtype=np.float64)
                except Exception:
                    pts0 = None

            _acc_goal_metrics(agg, pts0)

            # FDE / MOE / mAP aggregates (best-effort; require GT + candidate points).
            try:
                if cand_fdes is not None and 0 <= int(selected_idx) < len(cand_fdes):
                    f_sel = cand_fdes[int(selected_idx)]
                    if f_sel is not None and math.isfinite(float(f_sel)):
                        agg.fde_model_sum += float(f_sel)
                        agg.fde_model_count += 1
                if (
                    cand_fdes is not None
                    and score_idx is not None
                    and 0 <= int(score_idx) < len(cand_fdes)
                ):
                    f_score = cand_fdes[int(score_idx)]
                    if f_score is not None and math.isfinite(float(f_score)):
                        agg.fde_score_sum += float(f_score)
                        agg.fde_score_count += 1
                if score_fde_masked_laststep_m is not None and math.isfinite(
                    float(score_fde_masked_laststep_m)
                ):
                    agg.fde_score_masked_laststep_sum += float(score_fde_masked_laststep_m)
                    agg.fde_score_masked_laststep_count += 1
                if min_all_fde is not None and math.isfinite(float(min_all_fde)):
                    agg.fde_min_all_sum += float(min_all_fde)
                    agg.fde_min_all_count += 1
                if cand_fdes is not None and vis:
                    f_vis = [cand_fdes[i] for i in vis if 0 <= int(i) < len(cand_fdes)]
                    f_ok = [float(x) for x in f_vis if x is not None and math.isfinite(float(x))]
                    if f_ok:
                        agg.fde_min_sum += float(min(f_ok))
                        agg.fde_min_count += 1
            except Exception:
                pass

            try:
                if cand_moes is not None and 0 <= int(selected_idx) < len(cand_moes):
                    m_sel = cand_moes[int(selected_idx)]
                    if m_sel is not None and math.isfinite(float(m_sel)):
                        agg.moe_model_sum += float(m_sel)
                        agg.moe_model_count += 1
                if (
                    cand_moes is not None
                    and score_idx is not None
                    and 0 <= int(score_idx) < len(cand_moes)
                ):
                    m_score = cand_moes[int(score_idx)]
                    if m_score is not None and math.isfinite(float(m_score)):
                        agg.moe_score_sum += float(m_score)
                        agg.moe_score_count += 1
                if min_all_moe is not None and math.isfinite(float(min_all_moe)):
                    agg.moe_min_all_sum += float(min_all_moe)
                    agg.moe_min_all_count += 1
                if mean_all_moe is not None and math.isfinite(float(mean_all_moe)):
                    agg.moe_mean_all_sum += float(mean_all_moe)
                    agg.moe_mean_all_count += 1
                if cand_moes is not None and vis:
                    m_vis = [cand_moes[i] for i in vis if 0 <= int(i) < len(cand_moes)]
                    m_ok = [float(x) for x in m_vis if x is not None and math.isfinite(float(x))]
                    if m_ok:
                        agg.moe_min_sum += float(min(m_ok))
                        agg.moe_min_count += 1
            except Exception:
                pass

            try:
                if ap_fde_2m is not None and math.isfinite(float(ap_fde_2m)):
                    agg.ap_fde_2m_sum += float(ap_fde_2m)
                    agg.ap_fde_2m_count += 1
            except Exception:
                pass

            # Prefix ADEs
            sel_ade_0_5s = _prefix(pts0, 0.5)
            sel_ade_1_0s = _prefix(pts0, 1.0)
            sel_ade_2_0s = _prefix(pts0, 2.0)
            if sel_ade_0_5s is not None and math.isfinite(float(sel_ade_0_5s)):
                agg.ade_model_0_5s_sum += float(sel_ade_0_5s)
                agg.ade_model_0_5s_count += 1
            if sel_ade_1_0s is not None and math.isfinite(float(sel_ade_1_0s)):
                agg.ade_model_1_0s_sum += float(sel_ade_1_0s)
                agg.ade_model_1_0s_count += 1
            if sel_ade_2_0s is not None and math.isfinite(float(sel_ade_2_0s)):
                agg.ade_model_2_0s_sum += float(sel_ade_2_0s)
                agg.ade_model_2_0s_count += 1

            if score_ade_0_5s is not None and math.isfinite(float(score_ade_0_5s)):
                agg.ade_score_0_5s_sum += float(score_ade_0_5s)
                agg.ade_score_0_5s_count += 1
            if score_ade_1_0s is not None and math.isfinite(float(score_ade_1_0s)):
                agg.ade_score_1_0s_sum += float(score_ade_1_0s)
                agg.ade_score_1_0s_count += 1
            if score_ade_2_0s is not None and math.isfinite(float(score_ade_2_0s)):
                agg.ade_score_2_0s_sum += float(score_ade_2_0s)
                agg.ade_score_2_0s_count += 1

            # Visible-min prefix ADEs
            min_pts = None
            if (
                min_vis_idx is not None
                and cand_xy is not None
                and 0 <= int(min_vis_idx) < len(cand_xy)
            ):
                try:
                    min_pts = np.asarray(cand_xy[int(min_vis_idx)], dtype=np.float64)
                except Exception:
                    min_pts = None
            min_ade_0_5s = _prefix(min_pts, 0.5)
            min_ade_1_0s = _prefix(min_pts, 1.0)
            min_ade_2_0s = _prefix(min_pts, 2.0)
            if min_ade_0_5s is not None and math.isfinite(float(min_ade_0_5s)):
                agg.ade_min_0_5s_sum += float(min_ade_0_5s)
                agg.ade_min_0_5s_count += 1
            if min_ade_1_0s is not None and math.isfinite(float(min_ade_1_0s)):
                agg.ade_min_1_0s_sum += float(min_ade_1_0s)
                agg.ade_min_1_0s_count += 1
            if min_ade_2_0s is not None and math.isfinite(float(min_ade_2_0s)):
                agg.ade_min_2_0s_sum += float(min_ade_2_0s)
                agg.ade_min_2_0s_count += 1

            # Open-loop MAOE + DCR/TCR vs GT corridor
            try:
                if pts0 is not None and gt_xy is not None:
                    maoe = _maoe_deg(pts0[:, :2], gt_xy[:, :2])
                    if maoe is not None and math.isfinite(float(maoe)):
                        agg.maoe_sum += float(maoe)
                        agg.maoe_count += 1
                    dcr, tcr = _dcr_tcr_from_corridor(
                        pred_xy=pts0[:, :2],
                        compliant_polyline_xy=gt_xy[:, :2],
                        dt_s=float(args.traj_dt_s),
                        corridor_radius_m=1.0,
                    )
                    if dcr is not None and math.isfinite(float(dcr)):
                        agg.dcr_sum += float(dcr)
                        agg.dcr_count += 1
                    if tcr is not None and math.isfinite(float(tcr)):
                        agg.tcr_sum += float(tcr)
                        agg.tcr_count += 1
            except Exception:
                pass

    # Emit TSV
    created_at = _utc_now_iso()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    task = "trajectory_selection"
    dataset_str = str(dataset_path)
    planner_source = str(args.planner_source)

    lines: list[str] = []
    lines.append("\t".join(_OUT_COLS))
    rows: list[dict[str, object]] = []

    # Stable row order: global oracle first, then pool order and K ascending.
    ordered_specs: list[tuple[str, str, int | None]] = []
    ordered_specs.append(("oracle_min_ade_all", "all", None))
    for pool in pools:
        for k in ks:
            ordered_specs.append(("oracle_min_ade", pool, int(k)))

    for selector, pool, k in ordered_specs:
        agg = aggs[(selector, pool, k)]
        # Row metadata
        if selector == "oracle_min_ade_all":
            exp_name = "0115_T2_real_policy=oracle_min_ade_all"
        else:
            exp_name = f"0115_T2_real_policy=oracle_min_ade_pool={pool}_k={int(k)}"
        trial_name = f"{exp_name}_{stamp}"

        row = {
            "experiment_name": exp_name,
            "trial_name": trial_name,
            "task": task,
            "dataset": dataset_str,
            "planner_source": planner_source,
            "selector": selector,
            "seed": int(args.seed),
            "snapshots_evaluated": int(agg.evaluated),
            "accuracy": safe_div(agg.correct, agg.evaluated),
            "accuracy_vs_score": safe_div(agg.correct_vs_score, agg.score_agree_count),
            "accuracy_vs_min_ade": safe_div(agg.correct_vs_min_ade, agg.ade_count),
            "accuracy_vs_min_ade_all": safe_div(agg.correct_vs_min_ade_all, agg.ade_count),
            "stop_rate": safe_div(agg.stop_count, agg.evaluated),
            "stop_count": int(agg.stop_count),
            "invalid_index_rate": safe_div(agg.invalid_index_count, agg.evaluated),
            "invalid_index_count": int(agg.invalid_index_count),
            "pred_error_rate": safe_div(agg.pred_error_count, agg.pred_output_total),
            "pred_error_count": int(agg.pred_error_count),
            "ade_model_avg": safe_div(agg.ade_model_sum, agg.ade_count),
            "ade_model_0_5s_avg": safe_div(agg.ade_model_0_5s_sum, agg.ade_model_0_5s_count),
            "ade_model_1_0s_avg": safe_div(agg.ade_model_1_0s_sum, agg.ade_model_1_0s_count),
            "ade_model_2_0s_avg": safe_div(agg.ade_model_2_0s_sum, agg.ade_model_2_0s_count),
            "ade_score_avg": safe_div(agg.ade_score_sum, agg.ade_count),
            "ade_score_0_5s_avg": safe_div(agg.ade_score_0_5s_sum, agg.ade_score_0_5s_count),
            "ade_score_1_0s_avg": safe_div(agg.ade_score_1_0s_sum, agg.ade_score_1_0s_count),
            "ade_score_2_0s_avg": safe_div(agg.ade_score_2_0s_sum, agg.ade_score_2_0s_count),
            "ade_min_avg": safe_div(agg.ade_min_sum, agg.ade_count),
            "ade_min_0_5s_avg": safe_div(agg.ade_min_0_5s_sum, agg.ade_min_0_5s_count),
            "ade_min_1_0s_avg": safe_div(agg.ade_min_1_0s_sum, agg.ade_min_1_0s_count),
            "ade_min_2_0s_avg": safe_div(agg.ade_min_2_0s_sum, agg.ade_min_2_0s_count),
            "ade_min_all_avg": safe_div(agg.ade_min_all_sum, agg.ade_count),
            "fde_model_avg": safe_div(agg.fde_model_sum, agg.fde_model_count),
            "fde_score_avg": safe_div(agg.fde_score_sum, agg.fde_score_count),
            "fde_score_masked_laststep_avg": safe_div(
                agg.fde_score_masked_laststep_sum, agg.fde_score_masked_laststep_count
            ),
            "fde_min_avg": safe_div(agg.fde_min_sum, agg.fde_min_count),
            "fde_min_all_avg": safe_div(agg.fde_min_all_sum, agg.fde_min_all_count),
            "moe_deg_avg": safe_div(agg.moe_model_sum, agg.moe_model_count),
            "moe_deg_score_avg": safe_div(agg.moe_score_sum, agg.moe_score_count),
            "moe_deg_min_avg": safe_div(agg.moe_min_sum, agg.moe_min_count),
            "moe_deg_min_all_avg": safe_div(agg.moe_min_all_sum, agg.moe_min_all_count),
            "moe_deg_mean_all_avg": safe_div(agg.moe_mean_all_sum, agg.moe_mean_all_count),
            "map_fde_2m_avg": safe_div(agg.ap_fde_2m_sum, agg.ap_fde_2m_count),
            "selected_end_dist_to_goal_m_avg": safe_div(
                agg.sel_end_goal_dist_sum, agg.sel_end_goal_dist_count
            ),
            "selected_goal_ang_diff_deg_avg": safe_div(
                agg.sel_goal_ang_diff_sum, agg.sel_goal_ang_diff_count
            ),
            "selected_traj_avg_dist_to_goal_m_avg": safe_div(
                agg.sel_traj_avg_goal_dist_sum, agg.sel_traj_avg_goal_dist_count
            ),
            "selected_goal_progress_m_avg": safe_div(
                agg.sel_goal_progress_sum, agg.sel_goal_progress_count
            ),
            "maoe_deg_avg": safe_div(agg.maoe_sum, agg.maoe_count),
            "dcr_avg": safe_div(agg.dcr_sum, agg.dcr_count),
            "tcr_avg": safe_div(agg.tcr_sum, agg.tcr_count),
            "created_at_utc": created_at,
            "slow_brain_fast_planner_version": __version__,
            "model_calls": 0,
            "prompt_tokens_total": 0,
            "output_tokens_total": 0,
            "total_tokens_total": 0,
            "total_tokens_avg": None,
            "latency_ms_avg": None,
            "latency_ms_p50": None,
            "latency_ms_p90": None,
        }

        lines.append("\t".join(_fmt(row.get(c)) for c in _OUT_COLS))
        rows.append({c: row.get(c) for c in _OUT_COLS})

    out_text = "\n".join(lines) + "\n"
    print(out_text, end="")
    if args.out_tsv:
        p = Path(str(args.out_tsv)).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(out_text, encoding="utf-8")
    if args.out_csv:
        p = Path(str(args.out_csv)).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_OUT_COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
