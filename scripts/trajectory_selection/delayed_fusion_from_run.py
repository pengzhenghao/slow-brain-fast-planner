#!/usr/bin/env python3
"""Retrospectively compute *delayed* trajectory-selection open-loop metrics from an existing
run directory.

Goal
----
Given an existing trajectory-selection VLM run dir that contains:
  - config.json
  - predictions.jsonl
  - (optional) metrics.json   (for token/latency totals)

we compute metrics for a *delayed feedback* setting:
  - Treat the VLM decision time as t0 (from predictions.jsonl)
  - Evaluate at t1 = t0 + delay_s, using planner candidates + GT future at t1
  - Apply one of:
      - match:          pick candidate at t1 most similar to stale VLM-selected traj from t0
      - score_fusion:   fused_score = s1_score(t1) + lambda * exp(-delay/tau) * sim(candidate(t1),
      stale_vlm(t0)->t1)
      - prob_fusion:    fuse in probability space (same alpha formula as closed-loop benchmark)

Outputs
-------
Creates one or more subfolders under the run dir:
  fusion_result_[delay]_[method]_[params]/
and writes:
  - metrics.json
  - metrics_report.txt
  - metrics_summary.csv
  - metrics_summary_simple.csv (+ .txt)
  - predictions.jsonl
  - predictions.csv

This is intentionally "offline" and makes **no VLM calls**.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from slow_brain_fast_planner import __version__
from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files, load_episode
from slow_brain_fast_planner.benchmarks.overlays import OverlayConfig
from slow_brain_fast_planner.benchmarks.planner_postprocessing import (
    A0Config,
    safe_div,
    to_jsonable,
)
from slow_brain_fast_planner.benchmarks.trajectory_selection_workers import (
    SnapshotJob,
    prepare_snapshot,
)
from slow_brain_fast_planner.cli.trajectory_selection import (
    _ade_prefix_seconds,
    _dcr_tcr_from_corridor,
    _format_metrics_report,
    _maoe_deg,
    _traj_path_length_m,
    _write_metrics_summary_csv,
    _write_metrics_summary_simple_csv,
)
from slow_brain_fast_planner.control.tracking import (
    polyline_length,
    project_point_to_polyline_arclength,
)
from slow_brain_fast_planner.utils.io import utc_now_iso as _utc_now_iso
from slow_brain_fast_planner.utils.io import write_json as _write_json


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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


def _write_metrics_summary_simple_txt(csv_path: Path, txt_path: Path) -> None:
    try:
        s = csv_path.read_text(encoding="utf-8")
    except Exception:
        return
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(s if s.endswith("\n") else (s + "\n"), encoding="utf-8")


def _write_metrics_summary_simple_csv_matching_base(
    *,
    base_csv_path: Path,
    out_csv_path: Path,
    metrics: dict[str, Any],
) -> bool:
    """Write metrics_summary_simple.csv using the *same header* as base_csv_path.

    Returns True if base header was used successfully, else False.
    """
    try:
        header = base_csv_path.read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return False
    cols = [c.strip() for c in header.split(",") if c.strip()]
    if not cols:
        return False
    row = {k: metrics.get(k) for k in cols}
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerow(row)
        return True
    except Exception:
        return False


def _softmax_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size <= 0:
        return x
    if not np.any(np.isfinite(x)):
        return np.full_like(x, 1.0 / float(x.size), dtype=np.float64)
    m = float(np.max(x[np.isfinite(x)]))
    x2 = np.where(np.isfinite(x), x, m - 1e9).astype(np.float64)
    z = x2 - float(np.max(x2))
    e = np.exp(z)
    s = float(np.sum(e))
    if not math.isfinite(s) or s <= 0:
        return np.full_like(x2, 1.0 / float(x2.size), dtype=np.float64)
    return (e / s).astype(np.float64)


def _resample_polyline_arclen(points_xy: np.ndarray, *, n: int) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = int(max(2, int(n)))
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if pts.shape[0] == 1:
        return np.repeat(pts[:1], repeats=n, axis=0)
    # cumulative arclength
    seg = pts[1:] - pts[:-1]
    seg_l = np.linalg.norm(seg, axis=1)
    s = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg_l)])
    total = float(s[-1])
    if not math.isfinite(total) or total <= 1e-9:
        return np.repeat(pts[:1], repeats=n, axis=0)
    targets = np.linspace(0.0, total, n, dtype=np.float64)
    out = np.zeros((n, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < s.size and float(s[j + 1]) < float(t):
            j += 1
        if j + 1 >= s.size:
            out[i] = pts[-1]
            continue
        s0 = float(s[j])
        s1 = float(s[j + 1])
        a = pts[j]
        b = pts[j + 1]
        denom = float(max(1e-12, s1 - s0))
        u = float((t - s0) / denom)
        u = float(min(1.0, max(0.0, u)))
        out[i] = (1.0 - u) * a + u * b
    return out


def _sample_polyline_by_arclength_window(
    points_xy: np.ndarray, *, s_start: float, s_len: float, n: int
) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    n = int(max(2, int(n)))
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if pts.shape[0] == 1:
        return np.repeat(pts[:1], repeats=n, axis=0)
    seg = pts[1:] - pts[:-1]
    seg_l = np.linalg.norm(seg, axis=1)
    s = np.concatenate([np.array([0.0], dtype=np.float64), np.cumsum(seg_l)])
    total = float(s[-1])
    if not math.isfinite(total) or total <= 1e-9:
        return np.repeat(pts[:1], repeats=n, axis=0)
    s0 = float(min(max(float(s_start), 0.0), total))
    s_len = float(max(0.0, float(s_len)))
    s1 = float(min(total, s0 + s_len))
    if s1 <= s0 + 1e-9:
        # choose closest point at s0
        idx = int(np.searchsorted(s, s0, side="right") - 1)
        idx = int(np.clip(idx, 0, int(pts.shape[0]) - 1))
        return np.repeat(pts[idx : idx + 1], repeats=n, axis=0)
    targets = np.linspace(s0, s1, n, dtype=np.float64)
    out = np.zeros((n, 2), dtype=np.float64)
    idxs = np.searchsorted(s, targets, side="right") - 1
    idxs = np.clip(idxs, 0, int(pts.shape[0]) - 2)
    for j in range(n):
        i = int(idxs[j])
        si = float(s[i])
        sj = float(s[i + 1])
        if sj <= si + 1e-12:
            out[j] = pts[i]
            continue
        a = float((targets[j] - si) / (sj - si))
        out[j] = (1.0 - a) * pts[i] + a * pts[i + 1]
    return out


def _mean_pointwise_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    a = np.asarray(a_xy, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b_xy, dtype=np.float64).reshape(-1, 2)
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return float("inf")
    d = a[:n] - b[:n]
    return float(np.mean(np.linalg.norm(d, axis=1)))


def _similarity_body_frame(
    *,
    candidate_xy: np.ndarray,
    stale_ref_xy: np.ndarray,
    align_to_current_pose: bool,
    dist_scale_m: float,
) -> float:
    """Similarity (higher is better) between candidate and stale reference in the same body
    frame."""
    cand = np.asarray(candidate_xy, dtype=np.float64).reshape(-1, 2)
    ref = np.asarray(stale_ref_xy, dtype=np.float64).reshape(-1, 2)
    if cand.shape[0] < 2 or ref.shape[0] < 2:
        return float("-inf")
    dist_scale = float(max(1e-6, float(dist_scale_m)))

    # Resample candidate uniformly by arclength for self-match stability.
    cand_u = _resample_polyline_arclen(cand, n=int(cand.shape[0]))
    seg_len = float(max(1e-6, float(polyline_length(cand_u))))

    if bool(align_to_current_pose):
        s0, _d0 = project_point_to_polyline_arclength(np.asarray([0.0, 0.0], dtype=np.float64), ref)
        ref_aligned = _sample_polyline_by_arclength_window(
            ref, s_start=float(s0), s_len=float(seg_len), n=int(cand_u.shape[0])
        )
    else:
        ref_aligned = _resample_polyline_arclen(ref, n=int(cand_u.shape[0]))

    d = _mean_pointwise_distance(cand_u, ref_aligned)
    return -float(d / dist_scale)


def _transform_points_prev_body_to_cur_body(
    pts_prev_body_xy: np.ndarray,
    *,
    prev_pose_xy_yaw: tuple[float, float, float],
    cur_pose_xy_yaw: tuple[float, float, float],
) -> np.ndarray:
    """Yaw-only SE(2): prev body frame -> current body frame."""
    pts = np.asarray(pts_prev_body_xy, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    xq, yq, yaw_q = (
        float(prev_pose_xy_yaw[0]),
        float(prev_pose_xy_yaw[1]),
        float(prev_pose_xy_yaw[2]),
    )
    xc, yc, yaw_c = float(cur_pose_xy_yaw[0]), float(cur_pose_xy_yaw[1]), float(cur_pose_xy_yaw[2])
    cq, sq = float(math.cos(yaw_q)), float(math.sin(yaw_q))
    cc, sc = float(math.cos(yaw_c)), float(math.sin(yaw_c))
    # body(prev) -> world
    pw_x = cq * pts[:, 0] - sq * pts[:, 1] + xq
    pw_y = sq * pts[:, 0] + cq * pts[:, 1] + yq
    # world -> body(cur)
    dx = pw_x - xc
    dy = pw_y - yc
    pb_x = cc * dx + sc * dy
    pb_y = -sc * dx + cc * dy
    return np.stack([pb_x, pb_y], axis=1)


def _infer_episode_id_from_meta(meta_path: Path) -> str:
    """Best-effort: map episode.json to episode_id used in predictions."""
    guess = meta_path.parent.name if meta_path.name == "episode.json" else meta_path.stem
    try:
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("episode_id"), str)
            and obj["episode_id"].strip()
        ):
            return str(obj["episode_id"])
    except Exception:
        pass
    return str(guess)


def _resolve_planner_time(
    t_to_index: dict[float, int], times_sorted: np.ndarray, *, t_target: float, tol_s: float
) -> float | None:
    """Resolve target time to a planner_candidates time (strict keys + nearest within tol)."""
    t_key = round(float(t_target), 6)
    if t_key in t_to_index:
        return float(t_key)
    if times_sorted.size == 0:
        return None
    i = int(np.searchsorted(times_sorted, float(t_target), side="left"))
    cands: list[float] = []
    for j in (i - 1, i, i + 1):
        if 0 <= j < int(times_sorted.size):
            cands.append(float(times_sorted[j]))
    if not cands:
        return None
    best = min(cands, key=lambda x: abs(float(x) - float(t_target)))
    if abs(float(best) - float(t_target)) <= float(tol_s):
        best_key = round(float(best), 6)
        return float(best_key) if best_key in t_to_index else float(best)
    return None


class _MethodSink:
    """Streaming writers + aggregate counters for one method."""

    def __init__(self, out_dir: Path, *, csv_cols: list[str]) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "predictions.csv"
        self.jsonl_path = self.out_dir / "predictions.jsonl"

        self._fcsv = self.csv_path.open("w", encoding="utf-8", newline="")
        self._wcsv = csv.DictWriter(self._fcsv, fieldnames=csv_cols, extrasaction="ignore")
        self._wcsv.writeheader()

        self._fjl = self.jsonl_path.open("w", encoding="utf-8")

        # Aggregates (mirrors Task2 evaluator + regen script)
        self.skipped_reasons: dict[str, int] = {}
        self.total_snapshots = 0
        self.evaluated = 0
        self.correct = 0
        self.stop_count = 0
        self.invalid_index_count = 0
        self.pred_error_count = 0

        self.ade_count = 0
        self.correct_vs_min_ade = 0
        self.correct_vs_min_ade_all = 0
        self.score_agree_count = 0
        self.correct_vs_score = 0

        self.ade_min_sum = 0.0
        self.ade_min_all_sum = 0.0
        self.ade_score_sum = 0.0
        self.ade_model_sum = 0.0
        self.ade_model_0_5s_sum = 0.0
        self.ade_model_1_0s_sum = 0.0
        self.ade_model_2_0s_sum = 0.0
        self.ade_score_0_5s_sum = 0.0
        self.ade_score_1_0s_sum = 0.0
        self.ade_score_2_0s_sum = 0.0
        self.ade_min_0_5s_sum = 0.0
        self.ade_min_1_0s_sum = 0.0
        self.ade_min_2_0s_sum = 0.0
        self.ade_model_0_5s_count = 0
        self.ade_model_1_0s_count = 0
        self.ade_model_2_0s_count = 0
        self.ade_score_0_5s_count = 0
        self.ade_score_1_0s_count = 0
        self.ade_score_2_0s_count = 0
        self.ade_min_0_5s_count = 0
        self.ade_min_1_0s_count = 0
        self.ade_min_2_0s_count = 0

        self.goal_dist_sum = 0.0
        self.goal_dist_count = 0
        self.sel_end_goal_dist_sum = 0.0
        self.sel_end_goal_dist_count = 0
        self.sel_goal_ang_diff_sum = 0.0
        self.sel_goal_ang_diff_count = 0
        self.sel_traj_avg_goal_dist_sum = 0.0
        self.sel_traj_avg_goal_dist_count = 0
        self.sel_goal_progress_sum = 0.0
        self.sel_goal_progress_count = 0

        self.maoe_sum = 0.0
        self.maoe_count = 0
        self.dcr_sum = 0.0
        self.dcr_count = 0
        self.tcr_sum = 0.0
        self.tcr_count = 0

    def close(self) -> None:
        try:
            self._fcsv.flush()
            self._fcsv.close()
        except Exception:
            pass
        try:
            self._fjl.flush()
            self._fjl.close()
        except Exception:
            pass

    def bump_skip(self, reason: str) -> None:
        self.skipped_reasons[reason] = int(self.skipped_reasons.get(reason, 0)) + 1

    def write_record(self, *, csv_row: dict[str, Any], jsonl_rec: dict[str, Any]) -> None:
        self._wcsv.writerow(csv_row)
        self._fjl.write(json.dumps(jsonl_rec) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Retrospective delayed Task2 metrics + fusion from an existing run dir."
    )
    ap.add_argument(
        "--run-dir",
        required=True,
        type=str,
        help="Existing Task2 run dir (contains config.json + predictions.jsonl).",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        type=str,
        help="Override dataset path (otherwise uses config.json).",
    )
    ap.add_argument(
        "--delay-s", type=float, default=2.0, help="Delay in seconds (t1 = t0 + delay_s)."
    )
    ap.add_argument(
        "--methods",
        type=str,
        default="planner_argmax,match,score_fusion,prob_fusion",
        help="Comma-separated methods: planner_argmax,match,score_fusion,prob_fusion",
    )
    ap.add_argument(
        "--similarity-mode",
        type=str,
        default="body_pointwise_arclen",
        help="(Currently informational; only body_pointwise_arclen is implemented here.)",
    )
    ap.add_argument(
        "--lambda-sim",
        type=float,
        default=1.0,
        help="Fusion strength fallback. If --lambda-score/--lambda-prob are not set, they default "
        "to this value.",
    )
    ap.add_argument(
        "--lambda-score",
        type=float,
        default=None,
        help="Lambda for score_fusion (overrides --lambda-sim).",
    )
    ap.add_argument(
        "--lambda-prob",
        type=float,
        default=None,
        help="Lambda for prob_fusion (overrides --lambda-sim).",
    )
    ap.add_argument(
        "--tau-s", type=float, default=3.0, help="Staleness decay time constant (seconds)."
    )
    ap.add_argument(
        "--dist-scale-m",
        type=float,
        default=1.0,
        help="Distance scale for similarity normalization.",
    )
    ap.add_argument("--align-to-current-pose", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--vlm-temp",
        type=float,
        default=1.0,
        help="Temperature for VLM similarity softmax (prob_fusion).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing fusion_result_* folders if present.",
    )
    ap.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="If >0, cap number of processed snapshots (debug).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg_path = run_dir / "config.json"
    preds_path = run_dir / "predictions.jsonl"
    base_metrics_path = run_dir / "metrics.json"
    base_simple_csv_path = run_dir / "metrics_summary_simple.csv"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json: {cfg_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions.jsonl: {preds_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if str(cfg.get("task")) != "task2_trajectory_selection":
        raise ValueError(f"config.json task is not task2_trajectory_selection: {cfg.get('task')}")

    dataset_root = Path(
        str(args.dataset) if args.dataset else str(cfg.get("dataset", ""))
    ).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_root} (use --dataset to override)"
        )

    delay_s = float(args.delay_s)
    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    allowed = {"planner_argmax", "match", "score_fusion", "prob_fusion"}
    for m in methods:
        if m not in allowed:
            raise ValueError(f"Unknown method {m!r}; expected one of {sorted(allowed)}")

    lam_score = (
        float(args.lambda_score) if args.lambda_score is not None else float(args.lambda_sim)
    )
    lam_prob = float(args.lambda_prob) if args.lambda_prob is not None else float(args.lambda_sim)

    # Build configs (match the original run config style).
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
    gt_metrics = cfg.get("gt_metrics") if isinstance(cfg.get("gt_metrics"), dict) else {}
    traj_dt_s = float(gt_metrics.get("traj_dt_s", 0.2))
    rgb_time_tolerance_s = float(overlays.get("rgb_time_tolerance_s", 0.2))
    require_goal = bool(cfg.get("require_goal", False))

    # Base token/latency totals: keep from base metrics if available (best-effort).
    base_metrics: dict[str, Any] = {}
    if base_metrics_path.exists():
        try:
            obj = json.loads(base_metrics_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                base_metrics = obj
        except Exception:
            base_metrics = {}

    # Episode meta mapping.
    meta_paths = find_episode_metadata_files(dataset_root)
    meta_by_episode: dict[str, Path] = {}
    for mp in meta_paths:
        eid = _infer_episode_id_from_meta(mp)
        meta_by_episode[str(eid)] = mp

    # Writers: keep Task2-style predictions.csv column ordering.
    csv_cols = [
        "episode_id",
        "t",
        "snapshot_index",
        "eval_t",
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
        "overlay_frame_ref",
        "local_plot_frame_ref",
        "auto_enabled",
    ]

    def _folder_name(method: str) -> str:
        d = str(delay_s).rstrip("0").rstrip(".")
        ds = f"{float(args.dist_scale_m):g}"
        if method == "match":
            return f"fusion_result_delay{d}s_match_ds{ds}"
        if method == "planner_argmax":
            return f"fusion_result_delay{d}s_planner_argmax"
        if method == "score_fusion":
            return f"fusion_result_delay{d}s_score_fusion_lam{lam_score:g}_tau{args.tau_s:g}_ds{ds}"
        if method == "prob_fusion":
            return f"fusion_result_delay{d}s_prob_fusion_lam{lam_prob:g}_tau{args.tau_s:g}_ds{ds}"
        return f"fusion_result_delay{d}s_{method}"

    sinks: dict[str, _MethodSink] = {}
    for m in methods:
        out_dir = run_dir / _folder_name(m)
        if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
            raise FileExistsError(f"Output dir exists and not empty: {out_dir} (pass --overwrite)")
        out_dir.mkdir(parents=True, exist_ok=True)
        sinks[m] = _MethodSink(out_dir, csv_cols=csv_cols)

        # Write a config snapshot for this derived run.
        cfg2 = dict(cfg)
        cfg2["created_at_utc"] = _utc_now_iso()
        cfg2["delayed_openloop"] = {
            "base_run_dir": str(run_dir),
            "delay_s": float(delay_s),
            "method": str(m),
            "similarity_mode": str(args.similarity_mode),
            "lambda_sim": float(
                lam_score
                if str(m) == "score_fusion"
                else lam_prob
                if str(m) == "prob_fusion"
                else float(args.lambda_sim)
            ),
            "staleness_tau_s": float(args.tau_s),
            "dist_scale_m": float(args.dist_scale_m),
            "align_to_current_pose": bool(args.align_to_current_pose),
            "vlm_temp": float(args.vlm_temp),
            "notes": (
                "Computed by scripts/task2_delayed_fusion_from_run.py; "
                "evaluates at t1=t0+delay using candidates+GT at t1."
            ),
        }
        (out_dir / "config.json").write_text(json.dumps(cfg2, indent=2) + "\n", encoding="utf-8")

    # Episode caches.
    ep_cache: dict[str, Any] = {}
    odom_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    planner_time_cache: dict[str, tuple[dict[float, int], np.ndarray]] = {}

    def _get_episode(eid: str) -> Any:
        hit = ep_cache.get(eid)
        if hit is not None:
            return hit
        mp = meta_by_episode.get(str(eid))
        if mp is None:
            raise KeyError(f"episode_id not found in dataset: {eid}")
        ep = load_episode(dataset_root, mp)
        ep_cache[str(eid)] = ep
        return ep

    def _get_odom_arrays(eid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hit = odom_cache.get(eid)
        if hit is not None:
            return hit
        ep = _get_episode(eid)
        od = getattr(ep, "odom", None) or []
        t_arr = np.asarray([float(r.t) for r in od], dtype=np.float64)
        xy_arr = (
            np.asarray([[float(r.x), float(r.y)] for r in od], dtype=np.float64)
            if t_arr.size
            else np.zeros((0, 2), dtype=np.float64)
        )
        yaw_arr = (
            np.asarray([float(r.yaw) for r in od], dtype=np.float64)
            if t_arr.size
            else np.zeros((0,), dtype=np.float64)
        )
        if t_arr.size:
            order = np.argsort(t_arr)
            t_arr = t_arr[order]
            xy_arr = xy_arr[order]
            yaw_arr = yaw_arr[order]
        odom_cache[eid] = (t_arr, xy_arr, yaw_arr)
        return t_arr, xy_arr, yaw_arr

    def _pose_xy_yaw(eid: str, t: float) -> tuple[float, float, float] | None:
        t_arr, xy_arr, yaw_arr = _get_odom_arrays(eid)
        if t_arr.size < 2:
            return None

        # local linear helper (avoid importing private helpers)
        def _interp(times: np.ndarray, values: np.ndarray, tt: float) -> np.ndarray | None:
            if times.size == 0 or values.size == 0 or times.shape[0] != values.shape[0]:
                return None
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

        xy = _interp(t_arr, xy_arr, float(t))
        yaw = _interp(t_arr, yaw_arr.reshape(-1, 1), float(t))
        if xy is None or yaw is None:
            return None
        return float(xy.reshape(-1)[0]), float(xy.reshape(-1)[1]), float(yaw.reshape(-1)[0])

    def _get_planner_time_index(eid: str) -> tuple[dict[float, int], np.ndarray]:
        hit = planner_time_cache.get(eid)
        if hit is not None:
            return hit
        ep = _get_episode(eid)
        pcs = getattr(ep, "planner_candidates", None) or []
        t_to_idx: dict[float, int] = {}
        times: list[float] = []
        for i, r in enumerate(pcs):
            tt = round(float(getattr(r, "t", 0.0)), 6)
            t_to_idx[float(tt)] = int(i)
            times.append(float(tt))
        times_sorted = np.asarray(sorted(times), dtype=np.float64)
        planner_time_cache[eid] = (t_to_idx, times_sorted)
        return t_to_idx, times_sorted

    # Main loop (streaming).
    max_n = int(args.max_snapshots)
    tol_s = (
        max(0.25, float(traj_dt_s) * 1.5) if math.isfinite(traj_dt_s) and traj_dt_s > 0 else 0.25
    )

    for i_rec, old in enumerate(_iter_jsonl(preds_path), start=1):
        if max_n > 0 and i_rec > max_n:
            break

        # Count every input record as "total" (mirrors snapshots_total-ish semantics).
        for s in sinks.values():
            s.total_snapshots += 1

        ep_id = str(old.get("episode_id", "") or "")
        if not ep_id:
            for s in sinks.values():
                s.bump_skip("missing_episode_id")
            continue

        try:
            t0 = float(old.get("t"))
            snap_idx = int(old.get("snapshot_index"))
        except Exception:
            for s in sinks.values():
                s.bump_skip("bad_key_fields")
            continue

        # Respect original skips: no VLM selection to replay.
        if bool(old.get("skipped", False)) or old.get("skip_reason") not in (None, "", "null"):
            r = str(old.get("skip_reason") or "skipped_in_original_run")
            for s in sinks.values():
                s.bump_skip(r)
            continue

        pred_action = old.get("prediction_action")
        pred_idx0 = old.get("prediction")
        if pred_action is None:
            pred_action = "select_trajectory" if pred_idx0 is not None else "stop"

        # We only need the stale VLM selection for match/score_fusion/prob_fusion.
        need_stale = any(m in sinks for m in ("match", "score_fusion", "prob_fusion"))
        have_stale = False
        if need_stale and str(pred_action) == "select_trajectory":
            try:
                pred_idx0 = int(pred_idx0) if pred_idx0 is not None else None
            except Exception:
                pred_idx0 = None
            have_stale = pred_idx0 is not None

        # Resolve planner times for t0 and t1 (avoid strict failures in prepare_snapshot).
        try:
            t_to_idx, times_sorted = _get_planner_time_index(ep_id)
        except Exception:
            for s in sinks.values():
                s.bump_skip("missing_episode_meta")
            continue

        t0_res = _resolve_planner_time(
            t_to_idx, times_sorted, t_target=float(t0), tol_s=float(tol_s)
        )
        if t0_res is None:
            for s in sinks.values():
                s.bump_skip("t0_not_found_in_planner_candidates")
            continue

        t1_target = float(t0_res) + float(delay_s)
        t1_res = _resolve_planner_time(
            t_to_idx, times_sorted, t_target=float(t1_target), tol_s=float(tol_s)
        )
        if t1_res is None:
            for s in sinks.values():
                s.bump_skip("t1_not_found_in_planner_candidates")
            continue

        # Build snapshot jobs.
        mp = meta_by_episode.get(ep_id)
        if mp is None:
            for s in sinks.values():
                s.bump_skip("missing_episode_meta")
            continue

        job0 = SnapshotJob(
            dataset_root=str(dataset_root),
            out_dir=str(run_dir),  # irrelevant (write_overlays=False)
            episode_meta_path=str(mp),
            episode_id=str(ep_id),
            t=float(t0_res),
            snapshot_index=int(snap_idx),
            a0_cfg=a0_cfg,
            overlay_cfg=overlay_cfg,
            rgb_time_tolerance_s=float(rgb_time_tolerance_s),
            require_goal=bool(require_goal),
            write_overlays=False,
            overlay_image_width=None,
            prompt_history_frames=0,
            prompt_image_width=None,
            compute_gt_metrics=False,
            traj_dt_s=float(traj_dt_s),
        )
        job1 = SnapshotJob(
            dataset_root=str(dataset_root),
            out_dir=str(run_dir),  # irrelevant (write_overlays=False)
            episode_meta_path=str(mp),
            episode_id=str(ep_id),
            t=float(t1_res),
            snapshot_index=int(snap_idx),
            a0_cfg=a0_cfg,
            overlay_cfg=overlay_cfg,
            rgb_time_tolerance_s=float(rgb_time_tolerance_s),
            require_goal=bool(require_goal),
            write_overlays=False,
            overlay_image_width=None,
            prompt_history_frames=0,
            prompt_image_width=None,
            compute_gt_metrics=True,  # needed for ADE oracle at t1
            traj_dt_s=float(traj_dt_s),
        )

        prep1 = prepare_snapshot(job1)
        if (
            not bool(prep1.ok)
            or prep1.candidates_points_xy is None
            or prep1.ade is None
            or prep1.gt_local_traj_xy is None
        ):
            reason = str(prep1.skip_reason or "prep1_failed")
            for s in sinks.values():
                s.bump_skip(f"prep1:{reason}")
            continue

        # Candidates + planner scores at t1.
        K = int(prep1.num_candidates)
        if K <= 0:
            for s in sinks.values():
                s.bump_skip("no_candidates_t1")
            continue

        cands1 = [np.asarray(x, dtype=np.float64) for x in (prep1.candidates_points_xy or [])]
        if len(cands1) != K:
            for s in sinks.values():
                s.bump_skip("bad_candidates_shape_t1")
            continue

        scores_raw = prep1.candidate_scores_raw
        if not isinstance(scores_raw, list) or len(scores_raw) != K:
            # Fallback: try obs payload; else zeros.
            scores_raw = [0.0 for _ in range(K)]
        s1_scores = np.asarray([float(x) for x in scores_raw], dtype=np.float64)

        # planner_argmax baseline at t1
        chosen_by_method: dict[str, int] = {}
        if "planner_argmax" in sinks:
            chosen_by_method["planner_argmax"] = int(np.argmax(s1_scores)) if s1_scores.size else 0

        # Methods requiring stale VLM selection from t0.
        stale1: np.ndarray | None = None
        sims: np.ndarray | None = None
        if have_stale and need_stale:
            prep0 = prepare_snapshot(job0)
            if not bool(prep0.ok) or prep0.candidates_points_xy is None:
                # Only penalize methods that require stale.
                reason = str(prep0.skip_reason or "prep0_failed")
                for name in ("match", "score_fusion", "prob_fusion"):
                    if name in sinks:
                        sinks[name].bump_skip(f"prep0:{reason}")
                # Still allow planner_argmax
            elif pred_idx0 is not None and (0 <= int(pred_idx0) < int(prep0.num_candidates)):
                stale0 = np.asarray(prep0.candidates_points_xy[int(pred_idx0)], dtype=np.float64)
                if stale0.ndim == 2 and stale0.shape[0] >= 2:
                    pose0 = _pose_xy_yaw(ep_id, float(t0_res))
                    pose1 = _pose_xy_yaw(ep_id, float(t1_res))
                    if pose0 is not None and pose1 is not None:
                        stale1 = _transform_points_prev_body_to_cur_body(
                            stale0, prev_pose_xy_yaw=pose0, cur_pose_xy_yaw=pose1
                        )
            else:
                for name in ("match", "score_fusion", "prob_fusion"):
                    if name in sinks:
                        sinks[name].bump_skip("pred:invalid_index_t0")
                        sinks[name].invalid_index_count += 1

        if stale1 is not None and stale1.shape[0] >= 2 and need_stale:
            sims = np.asarray(
                [
                    _similarity_body_frame(
                        candidate_xy=cands1[j],
                        stale_ref_xy=stale1,
                        align_to_current_pose=bool(args.align_to_current_pose),
                        dist_scale_m=float(args.dist_scale_m),
                    )
                    for j in range(K)
                ],
                dtype=np.float64,
            )

        if sims is not None and sims.size:
            if "match" in sinks:
                chosen_by_method["match"] = int(np.argmax(sims))
            if "score_fusion" in sinks:
                tau = float(max(1e-6, float(args.tau_s)))
                decay = math.exp(-float(delay_s) / float(tau))
                fused = s1_scores + float(lam_score) * float(decay) * sims
                chosen_by_method["score_fusion"] = int(np.argmax(fused)) if fused.size else 0
            if "prob_fusion" in sinks:
                tau = float(max(1e-6, float(args.tau_s)))
                decay = math.exp(-float(delay_s) / float(tau))
                p_planner = _softmax_stable(s1_scores)
                temp = float(max(1e-6, float(args.vlm_temp)))
                p_vlm = _softmax_stable(sims / temp)
                lam = float(max(0.0, float(lam_prob)))
                alpha_base = lam / (lam + 1.0)
                alpha = float(np.clip(alpha_base * decay, 0.0, 1.0))
                fused_p = (1.0 - alpha) * p_planner + alpha * p_vlm
                chosen_by_method["prob_fusion"] = int(np.argmax(fused_p)) if fused_p.size else 0
        else:
            # No stale available => these methods can't run for this snapshot. Mark skips
            # per-method.
            for name in ("match", "score_fusion", "prob_fusion"):
                if name in sinks:
                    reason = (
                        f"pred:{pred_action}"
                        if str(pred_action) != "select_trajectory"
                        else "missing_stale_similarity"
                    )
                    sinks[name].bump_skip(reason)
                    if str(pred_action) == "stop":
                        sinks[name].stop_count += 1

        # Shared per-snapshot values at t1.
        gt = np.asarray(prep1.gt_local_traj_xy, dtype=np.float64)
        goal_xy = prep1.goal_xy if isinstance(prep1.goal_xy, list) else None

        def _goal_metrics(
            sel_xy: np.ndarray,
            *,
            goal_xy=goal_xy,
            prep1=prep1,
        ) -> tuple[float | None, float | None, float | None, float | None]:
            if goal_xy is None or len(goal_xy) < 2 or sel_xy.size == 0:
                return None, None, None, None
            g = np.asarray([float(goal_xy[0]), float(goal_xy[1])], dtype=np.float64)
            end = sel_xy[-1, :2]
            d_end = float(np.linalg.norm(end - g))
            d_all = np.linalg.norm(sel_xy[:, :2] - g.reshape(1, 2), axis=1)
            d_avg = float(np.mean(d_all)) if d_all.size else None
            goal_ang = float(math.degrees(math.atan2(float(g[1]), float(g[0]))))
            end_ang = float(math.degrees(math.atan2(float(end[1]), float(end[0]))))
            diff = (end_ang - goal_ang + 180.0) % 360.0 - 180.0
            ang_diff = abs(float(diff))
            prog = None
            if prep1.goal_distance_m is not None and math.isfinite(float(prep1.goal_distance_m)):
                prog = float(prep1.goal_distance_m) - float(d_end)
            return (
                d_end if math.isfinite(d_end) else None,
                ang_diff if math.isfinite(ang_diff) else None,
                d_avg,
                prog,
            )

        for method, sink in sinks.items():
            chosen = int(chosen_by_method.get(method, 0))
            if chosen < 0 or chosen >= K:
                sink.bump_skip("chosen_invalid_index")
                sink.invalid_index_count += 1
                continue

            label = prep1.label
            if label is None:
                sink.bump_skip("label:missing")
                continue

            sel = np.asarray(cands1[chosen], dtype=np.float64)
            if sel.ndim != 2 or sel.shape[0] < 1:
                sink.bump_skip("bad_selected_traj")
                continue

            is_correct = bool(int(chosen) == int(label))

            # ADE + oracle stats at t1 (from prep1.ade) + prefix ADE recompute (like regen script).
            ade_obj = prep1.ade if isinstance(prep1.ade, dict) else {}
            ades = ade_obj.get("ades") if isinstance(ade_obj.get("ades"), list) else None
            if ades is None or len(ades) != K:
                sink.bump_skip("missing_ade_list_t1")
                continue

            selected_ade = float(ades[int(chosen)])
            selected_ade_0_5s = _ade_prefix_seconds(sel, gt, dt_s=float(traj_dt_s), seconds=0.5)
            selected_ade_1_0s = _ade_prefix_seconds(sel, gt, dt_s=float(traj_dt_s), seconds=1.0)
            selected_ade_2_0s = _ade_prefix_seconds(sel, gt, dt_s=float(traj_dt_s), seconds=2.0)

            # score/min prefixes (best-effort)
            score_idx = ade_obj.get("score_idx")
            min_idx = ade_obj.get("min_idx")
            score_ade_0_5s = score_ade_1_0s = score_ade_2_0s = None
            min_ade_0_5s = min_ade_1_0s = min_ade_2_0s = None
            try:
                if score_idx is not None:
                    sc = np.asarray(cands1[int(score_idx)], dtype=np.float64)
                    score_ade_0_5s = _ade_prefix_seconds(sc, gt, dt_s=float(traj_dt_s), seconds=0.5)
                    score_ade_1_0s = _ade_prefix_seconds(sc, gt, dt_s=float(traj_dt_s), seconds=1.0)
                    score_ade_2_0s = _ade_prefix_seconds(sc, gt, dt_s=float(traj_dt_s), seconds=2.0)
            except Exception:
                pass
            try:
                if min_idx is not None:
                    mi = np.asarray(cands1[int(min_idx)], dtype=np.float64)
                    min_ade_0_5s = _ade_prefix_seconds(mi, gt, dt_s=float(traj_dt_s), seconds=0.5)
                    min_ade_1_0s = _ade_prefix_seconds(mi, gt, dt_s=float(traj_dt_s), seconds=1.0)
                    min_ade_2_0s = _ade_prefix_seconds(mi, gt, dt_s=float(traj_dt_s), seconds=2.0)
            except Exception:
                pass

            # Social metrics vs GT corridor
            maoe = _maoe_deg(sel, gt)
            dcr, tcr = _dcr_tcr_from_corridor(
                pred_xy=sel, compliant_polyline_xy=gt, dt_s=float(traj_dt_s), corridor_radius_m=1.0
            )
            compliance_source = "gt_corridor"

            # Goal-relative metrics
            sel_end_goal, sel_ang_diff, sel_avg_goal, sel_prog = _goal_metrics(sel)
            traj_len_m = _traj_path_length_m(sel)

            # Aggregate counters
            sink.evaluated += 1
            sink.correct += int(is_correct)

            # score agreement at t1 (chosen vs score_idx)
            if score_idx is not None:
                sink.score_agree_count += 1
                if int(chosen) == int(score_idx):
                    sink.correct_vs_score += 1

            # ADE aggregates
            sink.ade_count += 1
            sink.ade_model_sum += float(selected_ade)
            if ade_obj.get("min") is not None:
                sink.ade_min_sum += float(ade_obj.get("min"))
            if ade_obj.get("min_all") is not None:
                sink.ade_min_all_sum += float(ade_obj.get("min_all"))
            if ade_obj.get("score") is not None:
                sink.ade_score_sum += float(ade_obj.get("score"))
            if ade_obj.get("min_idx") is not None and int(chosen) == int(ade_obj.get("min_idx")):
                sink.correct_vs_min_ade += 1
            if ade_obj.get("min_all_idx") is not None and int(chosen) == int(
                ade_obj.get("min_all_idx")
            ):
                sink.correct_vs_min_ade_all += 1

            # Prefix ADE aggregates (best-effort, only when finite)
            def _bump_prefix(v: float | None, *, which: str, sink=sink) -> None:
                if v is None:
                    return
                try:
                    vf = float(v)
                except Exception:
                    return
                if not math.isfinite(vf):
                    return
                if which == "model_0_5":
                    sink.ade_model_0_5s_sum += vf
                    sink.ade_model_0_5s_count += 1
                elif which == "model_1_0":
                    sink.ade_model_1_0s_sum += vf
                    sink.ade_model_1_0s_count += 1
                elif which == "model_2_0":
                    sink.ade_model_2_0s_sum += vf
                    sink.ade_model_2_0s_count += 1
                elif which == "score_0_5":
                    sink.ade_score_0_5s_sum += vf
                    sink.ade_score_0_5s_count += 1
                elif which == "score_1_0":
                    sink.ade_score_1_0s_sum += vf
                    sink.ade_score_1_0s_count += 1
                elif which == "score_2_0":
                    sink.ade_score_2_0s_sum += vf
                    sink.ade_score_2_0s_count += 1
                elif which == "min_0_5":
                    sink.ade_min_0_5s_sum += vf
                    sink.ade_min_0_5s_count += 1
                elif which == "min_1_0":
                    sink.ade_min_1_0s_sum += vf
                    sink.ade_min_1_0s_count += 1
                elif which == "min_2_0":
                    sink.ade_min_2_0s_sum += vf
                    sink.ade_min_2_0s_count += 1

            _bump_prefix(selected_ade_0_5s, which="model_0_5")
            _bump_prefix(selected_ade_1_0s, which="model_1_0")
            _bump_prefix(selected_ade_2_0s, which="model_2_0")
            _bump_prefix(score_ade_0_5s, which="score_0_5")
            _bump_prefix(score_ade_1_0s, which="score_1_0")
            _bump_prefix(score_ade_2_0s, which="score_2_0")
            _bump_prefix(min_ade_0_5s, which="min_0_5")
            _bump_prefix(min_ade_1_0s, which="min_1_0")
            _bump_prefix(min_ade_2_0s, which="min_2_0")

            # Goal aggregates
            if prep1.goal_distance_m is not None and math.isfinite(float(prep1.goal_distance_m)):
                sink.goal_dist_sum += float(prep1.goal_distance_m)
                sink.goal_dist_count += 1
            for val, acc_sum, _acc_cnt in [
                (sel_end_goal, "sel_end_goal_dist_sum", "sel_end_goal_dist_count"),
                (sel_ang_diff, "sel_goal_ang_diff_sum", "sel_goal_ang_diff_count"),
                (sel_avg_goal, "sel_traj_avg_goal_dist_sum", "sel_traj_avg_goal_dist_count"),
                (sel_prog, "sel_goal_progress_sum", "sel_goal_progress_count"),
            ]:
                if val is None:
                    continue
                if not math.isfinite(float(val)):
                    continue
                if acc_sum == "sel_end_goal_dist_sum":
                    sink.sel_end_goal_dist_sum += float(val)
                    sink.sel_end_goal_dist_count += 1
                elif acc_sum == "sel_goal_ang_diff_sum":
                    sink.sel_goal_ang_diff_sum += float(val)
                    sink.sel_goal_ang_diff_count += 1
                elif acc_sum == "sel_traj_avg_goal_dist_sum":
                    sink.sel_traj_avg_goal_dist_sum += float(val)
                    sink.sel_traj_avg_goal_dist_count += 1
                elif acc_sum == "sel_goal_progress_sum":
                    sink.sel_goal_progress_sum += float(val)
                    sink.sel_goal_progress_count += 1

            # Social aggregates
            if maoe is not None and math.isfinite(float(maoe)):
                sink.maoe_sum += float(maoe)
                sink.maoe_count += 1
            if dcr is not None and math.isfinite(float(dcr)):
                sink.dcr_sum += float(dcr)
                sink.dcr_count += 1
            if tcr is not None and math.isfinite(float(tcr)):
                sink.tcr_sum += float(tcr)
                sink.tcr_count += 1

            # Row outputs
            gx = gy = None
            if goal_xy is not None and len(goal_xy) >= 2:
                gx, gy = goal_xy[0], goal_xy[1]

            csv_row = {
                "episode_id": ep_id,
                "t": float(t0),
                "snapshot_index": int(snap_idx),
                "eval_t": float(t1_res),
                "prediction": int(chosen),
                "label": int(label),
                "correct": bool(is_correct),
                "skipped": False,
                "skip_reason": None,
                "num_candidates": int(K),
                "goal_x": to_jsonable(gx),
                "goal_y": to_jsonable(gy),
                "goal_distance_m": to_jsonable(prep1.goal_distance_m),
                "goal_bearing_deg": to_jsonable(prep1.goal_bearing_deg),
                "selected_end_dist_to_goal_m": to_jsonable(sel_end_goal),
                "selected_goal_ang_diff_deg": to_jsonable(sel_ang_diff),
                "selected_traj_avg_dist_to_goal_m": to_jsonable(sel_avg_goal),
                "selected_goal_progress_m": to_jsonable(sel_prog),
                "traj_len_m": to_jsonable(traj_len_m),
                "maoe_deg": to_jsonable(maoe),
                "dcr": to_jsonable(dcr),
                "tcr": to_jsonable(tcr),
                "compliance_source": to_jsonable(compliance_source),
                "ade_selected": to_jsonable(selected_ade),
                "ade_selected_0_5s": to_jsonable(selected_ade_0_5s),
                "ade_selected_1_0s": to_jsonable(selected_ade_1_0s),
                "ade_selected_2_0s": to_jsonable(selected_ade_2_0s),
                "ade_min": to_jsonable(ade_obj.get("min")),
                "ade_min_all": to_jsonable(ade_obj.get("min_all")),
                "ade_min_0_5s": to_jsonable(min_ade_0_5s),
                "ade_min_1_0s": to_jsonable(min_ade_1_0s),
                "ade_min_2_0s": to_jsonable(min_ade_2_0s),
                "ade_score": to_jsonable(ade_obj.get("score")),
                "ade_score_0_5s": to_jsonable(score_ade_0_5s),
                "ade_score_1_0s": to_jsonable(score_ade_1_0s),
                "ade_score_2_0s": to_jsonable(score_ade_2_0s),
                "overlay_frame_ref": None,
                "local_plot_frame_ref": None,
                "auto_enabled": to_jsonable(prep1.auto_enabled),
            }

            json_rec = {
                "task": "task2_trajectory_selection",
                "episode_id": ep_id,
                "t": float(t0),
                "eval_t": float(t1_res),
                "snapshot_index": int(snap_idx),
                "num_candidates": int(K),
                "prediction": int(chosen),
                "prediction_action": "select_trajectory",
                "label": int(label),
                "correct": bool(is_correct),
                "skipped": False,
                "skip_reason": None,
                "goal_xy": to_jsonable(prep1.goal_xy),
                "goal_distance_m": to_jsonable(prep1.goal_distance_m),
                "goal_bearing_deg": to_jsonable(prep1.goal_bearing_deg),
                "selected_end_dist_to_goal_m": to_jsonable(sel_end_goal),
                "selected_goal_ang_diff_deg": to_jsonable(sel_ang_diff),
                "selected_traj_avg_dist_to_goal_m": to_jsonable(sel_avg_goal),
                "selected_goal_progress_m": to_jsonable(sel_prog),
                "traj_len_m": to_jsonable(traj_len_m),
                "maoe_deg": to_jsonable(maoe),
                "dcr": to_jsonable(dcr),
                "tcr": to_jsonable(tcr),
                "compliance_source": to_jsonable(compliance_source),
                "auto_enabled": to_jsonable(prep1.auto_enabled),
                "ade": {
                    "ades": to_jsonable(ades),
                    "min": to_jsonable(ade_obj.get("min")),
                    "min_all": to_jsonable(ade_obj.get("min_all")),
                    "score": to_jsonable(ade_obj.get("score")),
                    "selected": to_jsonable(selected_ade),
                    "selected_0_5s": to_jsonable(selected_ade_0_5s),
                    "selected_1_0s": to_jsonable(selected_ade_1_0s),
                    "selected_2_0s": to_jsonable(selected_ade_2_0s),
                    "score_0_5s": to_jsonable(score_ade_0_5s),
                    "score_1_0s": to_jsonable(score_ade_1_0s),
                    "score_2_0s": to_jsonable(score_ade_2_0s),
                    "min_0_5s": to_jsonable(min_ade_0_5s),
                    "min_1_0s": to_jsonable(min_ade_1_0s),
                    "min_2_0s": to_jsonable(min_ade_2_0s),
                    "min_idx": to_jsonable(ade_obj.get("min_idx")),
                    "min_all_idx": to_jsonable(ade_obj.get("min_all_idx")),
                    "score_idx": to_jsonable(ade_obj.get("score_idx")),
                    "visible_indices": to_jsonable(ade_obj.get("visible_indices")),
                },
                "delayed_openloop": {
                    "delay_s": float(delay_s),
                    "method": str(method),
                    "lambda_sim": float(
                        lam_score
                        if str(method) == "score_fusion"
                        else lam_prob
                        if str(method) == "prob_fusion"
                        else float(args.lambda_sim)
                    ),
                    "staleness_tau_s": float(args.tau_s),
                    "dist_scale_m": float(args.dist_scale_m),
                    "align_to_current_pose": bool(args.align_to_current_pose),
                    "vlm_temp": float(args.vlm_temp),
                },
            }
            sink.write_record(csv_row=csv_row, jsonl_rec=json_rec)

    # Finalize metrics per method.
    for _method, sink in sinks.items():
        sink.close()

        out_dir = sink.out_dir
        # Derive counts
        pred_output_total = int(sink.evaluated) + int(sink.pred_error_count)

        # Prefer identifying the run by the actual directory name (robust to merged configs).
        exp_name_path = str(run_dir.parent.name) if run_dir.parent is not None else str(run_dir)
        trial_name_path = str(out_dir.name)

        metrics: dict[str, Any] = {
            "slow_brain_fast_planner_version": __version__,
            "created_at_utc": _utc_now_iso(),
            "task": "task2_trajectory_selection",
            "dataset": str(dataset_root),
            "planner_source": cfg.get("planner_source"),
            "experiment_name": exp_name_path,
            "trial_name": trial_name_path,
            "seed": int(cfg.get("seed", 0) or 0),
            "selector": str(
                ((cfg.get("model") or {}) if isinstance(cfg.get("model"), dict) else {}).get(
                    "adapter", "unknown"
                )
            ),
            "delayed_openloop": (
                json.loads((out_dir / "config.json").read_text(encoding="utf-8")).get(
                    "delayed_openloop"
                )
                if (out_dir / "config.json").exists()
                else {}
            ),
            "snapshots_total": int(sink.total_snapshots),
            "snapshots_evaluated": int(sink.evaluated),
            "snapshots_skipped": int(sink.total_snapshots - sink.evaluated),
            "skipped_reasons": {
                k: int(sink.skipped_reasons[k]) for k in sorted(sink.skipped_reasons)
            },
            "accuracy": safe_div(int(sink.correct), int(sink.evaluated)),
            "accuracy_vs_min_ade": safe_div(int(sink.correct_vs_min_ade), int(sink.ade_count)),
            "accuracy_vs_min_ade_all": safe_div(
                int(sink.correct_vs_min_ade_all), int(sink.ade_count)
            ),
            "accuracy_vs_score": safe_div(int(sink.correct_vs_score), int(sink.score_agree_count)),
            "stop_count": int(sink.stop_count),
            "stop_rate": safe_div(int(sink.stop_count), int(sink.evaluated)),
            "invalid_index_count": int(sink.invalid_index_count),
            "invalid_index_rate": safe_div(int(sink.invalid_index_count), int(sink.evaluated)),
            "pred_error_count": int(sink.pred_error_count),
            "pred_error_rate": safe_div(int(sink.pred_error_count), int(pred_output_total)),
            "ade_count": int(sink.ade_count),
            "ade_min_avg": safe_div(float(sink.ade_min_sum), int(sink.ade_count)),
            "ade_min_all_avg": safe_div(float(sink.ade_min_all_sum), int(sink.ade_count)),
            "ade_score_avg": safe_div(float(sink.ade_score_sum), int(sink.ade_count)),
            "ade_model_avg": safe_div(float(sink.ade_model_sum), int(sink.ade_count)),
            "ade_model_0_5s_avg": safe_div(
                float(sink.ade_model_0_5s_sum), int(sink.ade_model_0_5s_count)
            ),
            "ade_model_1_0s_avg": safe_div(
                float(sink.ade_model_1_0s_sum), int(sink.ade_model_1_0s_count)
            ),
            "ade_model_2_0s_avg": safe_div(
                float(sink.ade_model_2_0s_sum), int(sink.ade_model_2_0s_count)
            ),
            "ade_score_0_5s_avg": safe_div(
                float(sink.ade_score_0_5s_sum), int(sink.ade_score_0_5s_count)
            ),
            "ade_score_1_0s_avg": safe_div(
                float(sink.ade_score_1_0s_sum), int(sink.ade_score_1_0s_count)
            ),
            "ade_score_2_0s_avg": safe_div(
                float(sink.ade_score_2_0s_sum), int(sink.ade_score_2_0s_count)
            ),
            "ade_min_0_5s_avg": safe_div(
                float(sink.ade_min_0_5s_sum), int(sink.ade_min_0_5s_count)
            ),
            "ade_min_1_0s_avg": safe_div(
                float(sink.ade_min_1_0s_sum), int(sink.ade_min_1_0s_count)
            ),
            "ade_min_2_0s_avg": safe_div(
                float(sink.ade_min_2_0s_sum), int(sink.ade_min_2_0s_count)
            ),
            "goal_distance_m_avg": safe_div(float(sink.goal_dist_sum), int(sink.goal_dist_count)),
            "goal_distance_m_count": int(sink.goal_dist_count),
            "selected_end_dist_to_goal_m_avg": safe_div(
                float(sink.sel_end_goal_dist_sum), int(sink.sel_end_goal_dist_count)
            ),
            "selected_end_dist_to_goal_m_count": int(sink.sel_end_goal_dist_count),
            "selected_goal_ang_diff_deg_avg": safe_div(
                float(sink.sel_goal_ang_diff_sum), int(sink.sel_goal_ang_diff_count)
            ),
            "selected_goal_ang_diff_deg_count": int(sink.sel_goal_ang_diff_count),
            "selected_traj_avg_dist_to_goal_m_avg": safe_div(
                float(sink.sel_traj_avg_goal_dist_sum), int(sink.sel_traj_avg_goal_dist_count)
            ),
            "selected_traj_avg_dist_to_goal_m_count": int(sink.sel_traj_avg_goal_dist_count),
            "selected_goal_progress_m_avg": safe_div(
                float(sink.sel_goal_progress_sum), int(sink.sel_goal_progress_count)
            ),
            "selected_goal_progress_m_count": int(sink.sel_goal_progress_count),
            "maoe_deg_avg": safe_div(float(sink.maoe_sum), int(sink.maoe_count)),
            "dcr_avg": safe_div(float(sink.dcr_sum), int(sink.dcr_count)),
            "tcr_avg": safe_div(float(sink.tcr_sum), int(sink.tcr_count)),
        }

        # Keep token/latency totals from the base run (best-effort; these are properties of the VLM
        # run, not the derived eval).
        for k in [
            "model_calls",
            "prompt_tokens_total",
            "cached_prompt_tokens_total",
            "output_tokens_total",
            "thoughts_tokens_total",
            "total_tokens_total",
            "total_tokens_no_thoughts_total",
            "total_tokens_avg",
            "total_tokens_no_thoughts_avg",
            "latency_ms_avg",
            "latency_ms_p50",
            "latency_ms_p90",
        ]:
            if k in base_metrics:
                metrics[k] = base_metrics.get(k)

        # Write artifacts.
        _write_json(out_dir / "metrics.json", metrics)
        (out_dir / "metrics_report.txt").write_text(
            _format_metrics_report(metrics, run_dir=out_dir), encoding="utf-8"
        )
        _write_metrics_summary_csv(out_dir / "metrics_summary.csv", metrics)
        ms_simple_csv = out_dir / "metrics_summary_simple.csv"
        wrote_base = False
        if base_simple_csv_path.exists():
            wrote_base = _write_metrics_summary_simple_csv_matching_base(
                base_csv_path=base_simple_csv_path, out_csv_path=ms_simple_csv, metrics=metrics
            )
        if not wrote_base:
            _write_metrics_summary_simple_csv(ms_simple_csv, metrics)
        _write_metrics_summary_simple_txt(ms_simple_csv, out_dir / "metrics_summary_simple.txt")

    print(
        json.dumps(
            {
                "base_run_dir": str(run_dir),
                "outputs": {m: str(sinks[m].out_dir) for m in sinks},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
