"""Trajectory Selection Benchmark.

This is the main entry point for trajectory selection evaluation.
Supports both heuristic baselines and VLM-based selectors.

Usage:
    python -m slow_brain_fast_planner.cli.trajectory_selection --dataset data/processed

Multi-GPU:
    # Set WORLD_SIZE and RANK (or use --num-shards/--shard-id)
    WORLD_SIZE=4 RANK=0 python -m slow_brain_fast_planner.cli.trajectory_selection --help
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from slow_brain_fast_planner import __version__, constants
from slow_brain_fast_planner.agents.vlm_advisors import (
    VQAHierarchicalTrajectorySelectionAdvisor,
    VQATrajectorySelectionAdvisor,
)
from slow_brain_fast_planner.benchmarks.data_loading import (
    SnapshotDataset,
    SnapshotItem,
    SnapshotShardedSampler,
    build_trajectory_selection_jobs,
    build_trajectory_selection_jobs_from_takeover_clips,
    get_shard_config_from_env,
)
from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files
from slow_brain_fast_planner.benchmarks.model_adapters import (
    DummyTrajectoryModelAdapter,
    GeminiGenAIAdapter,
    GeminiGenAIConfig,
    OpenAIChatCompletionsConfig,
    OpenAIChatCompletionsHttpAdapter,
)
from slow_brain_fast_planner.benchmarks.overlays import OverlayConfig
from slow_brain_fast_planner.benchmarks.planner_postprocessing import (
    A0Config,
    _trajectory_nms_endpoints,
    safe_div,
    to_jsonable,
)
from slow_brain_fast_planner.benchmarks.vqa_trajectory import VQATrajectoryPromptConfig
from slow_brain_fast_planner.reporting import generate_run_report_html
from slow_brain_fast_planner.utils.io import utc_now_iso, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# CLI model names that select the hierarchical VLM advisor.
# "gemini_genai_chain_of_planning" is the paper-facing alias of "gemini_genai_hierarchical";
# both names run the exact same selector.
HIERARCHICAL_MODELS = ("gemini_genai_hierarchical", "gemini_genai_chain_of_planning")


def _traj_path_length_m(xy: np.ndarray | None) -> float | None:
    """Sum of segment lengths along a polyline in meters (robot local frame)."""
    if xy is None:
        return None
    try:
        pts = np.asarray(xy, dtype=np.float64)
    except Exception:
        return None
    if pts.ndim != 2 or pts.shape[0] <= 0 or pts.shape[1] < 2:
        return None
    if pts.shape[0] == 1:
        return 0.0
    diffs = pts[1:, :2] - pts[:-1, :2]
    seg = np.linalg.norm(diffs, axis=1)
    if seg.size == 0:
        return 0.0
    s = float(np.sum(seg))
    return s if math.isfinite(s) and s >= 0 else None


def _maoe_deg(pred_xy: np.ndarray | None, gt_xy: np.ndarray | None) -> float | None:
    """Maximum Average Orientation Error (MAOE) in degrees.

    Implemented as:
    - heading angles from step displacement vectors (origin->p0, p0->p1, ...)
    - per-step absolute wrapped angle error vs GT
    - max over prefix-average errors
    """
    if pred_xy is None or gt_xy is None:
        return None
    p = np.asarray(pred_xy, dtype=np.float64)
    g = np.asarray(gt_xy, dtype=np.float64)
    if p.ndim != 2 or g.ndim != 2 or p.shape[1] < 2 or g.shape[1] < 2:
        return None
    n = min(int(p.shape[0]), int(g.shape[0]))
    if n <= 0:
        return None
    p = p[:n, :2]
    g = g[:n, :2]

    def headings(xy: np.ndarray) -> np.ndarray:
        v0 = xy[0:1, :]
        if xy.shape[0] >= 2:
            v = np.concatenate([v0, xy[1:, :] - xy[:-1, :]], axis=0)
        else:
            v = v0
        return np.arctan2(v[:, 1], v[:, 0])

    hp = headings(p)
    hg = headings(g)
    if hp.shape[0] != hg.shape[0] or hp.size == 0:
        return None
    d = hp - hg
    d = (d + np.pi) % (2.0 * np.pi) - np.pi
    ae = np.abs(d)
    csum = np.cumsum(ae)
    denom = np.arange(1, int(ae.shape[0]) + 1, dtype=np.float64)
    pref = csum / denom
    m = float(np.max(pref)) if pref.size else None
    if m is None or (not math.isfinite(m)):
        return None
    return float(m * (180.0 / math.pi))


def _fde_m(pred_xy: np.ndarray | None, gt_xy: np.ndarray | None) -> float | None:
    """Final Displacement Error (FDE) in meters.

    Defined as the L2 distance between the final predicted point and final GT point, after aligning
    horizon length to min(len(pred), len(gt)).
    """
    if pred_xy is None or gt_xy is None:
        return None
    p = np.asarray(pred_xy, dtype=np.float64)
    g = np.asarray(gt_xy, dtype=np.float64)
    if p.ndim != 2 or g.ndim != 2 or p.shape[1] < 2 or g.shape[1] < 2:
        return None
    n = min(int(p.shape[0]), int(g.shape[0]))
    if n <= 0:
        return None
    end_p = p[n - 1, :2]
    end_g = g[n - 1, :2]
    d = float(np.linalg.norm(end_p - end_g))
    return d if math.isfinite(d) and d >= 0 else None


def _fde_m_masked_laststep(pred_xy: np.ndarray | None, gt_xy: np.ndarray | None) -> float | None:
    """FDE (meters) but only if GT covers the prediction's final step.

    This mirrors scripts that gate endpoint error by a "future_mask[..., -1]" (i.e. the final
    step is valid). We approximate that by requiring len(gt) >= len(pred), and we compare:
      pred[-1] vs gt[len(pred)-1]
    """
    if pred_xy is None or gt_xy is None:
        return None
    p = np.asarray(pred_xy, dtype=np.float64)
    g = np.asarray(gt_xy, dtype=np.float64)
    if p.ndim != 2 or g.ndim != 2 or p.shape[1] < 2 or g.shape[1] < 2:
        return None
    n_pred = int(p.shape[0])
    n_gt = int(g.shape[0])
    if n_pred <= 0 or n_gt <= 0:
        return None
    if n_gt < n_pred:
        return None
    end_p = p[n_pred - 1, :2]
    end_g = g[n_pred - 1, :2]
    d = float(np.linalg.norm(end_p - end_g))
    return d if math.isfinite(d) and d >= 0 else None


def _moe_deg_posvec(pred_xy: np.ndarray | None, gt_xy: np.ndarray | None) -> float | None:
    """Maximum Orientation Error (MOE) in degrees, matching `rss/eval_point_goal.py`.

    We treat each timestep point as a 2D vector from the origin (robot frame), compute the angle
    between predicted and GT vectors, and return the maximum angle over time.
    Timesteps where either vector has near-zero norm are ignored.
    """
    if pred_xy is None or gt_xy is None:
        return None
    p = np.asarray(pred_xy, dtype=np.float64)
    g = np.asarray(gt_xy, dtype=np.float64)
    if p.ndim != 2 or g.ndim != 2 or p.shape[1] < 2 or g.shape[1] < 2:
        return None
    n = min(int(p.shape[0]), int(g.shape[0]))
    if n <= 0:
        return None
    p = p[:n, :2]
    g = g[:n, :2]
    pn = np.linalg.norm(p, axis=1)
    gn = np.linalg.norm(g, axis=1)
    eps = 1e-9
    mask = (pn > eps) & (gn > eps)
    if not bool(np.any(mask)):
        return None
    dot = np.sum(p[mask] * g[mask], axis=1)
    denom = pn[mask] * gn[mask]
    cos = dot / denom
    cos = np.clip(cos, -1.0 + 1e-7, 1.0 - 1e-7)
    ang = np.degrees(np.arccos(cos))
    if ang.size <= 0:
        return None
    m = float(np.max(ang))
    return m if math.isfinite(m) and m >= 0 else None


def _average_precision_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """Calculate AP from a precision-recall curve (monotonic precision envelope)."""
    p = np.asarray(precision, dtype=np.float64).reshape(-1)
    r = np.asarray(recall, dtype=np.float64).reshape(-1)
    if p.size == 0 or r.size == 0 or p.size != r.size:
        return 0.0
    r = np.concatenate(([0.0], r, [1.0]))
    p = np.concatenate(([0.0], p, [0.0]))
    for i in range(int(p.size) - 2, -1, -1):
        p[i] = max(float(p[i]), float(p[i + 1]))
    idx = np.where(r[1:] != r[:-1])[0] + 1
    ap = float(np.sum((r[idx] - r[idx - 1]) * p[idx]))
    return ap if math.isfinite(ap) and ap >= 0 else 0.0


def _ap_at_threshold(
    *,
    scores: list[float] | None,
    fdes_m: list[float | None] | None,
    threshold_m: float = 2.0,
) -> float | None:
    """Per-snapshot AP over candidates, using candidate scores and FDE<threshold as positives.

    Returns 0.0 when there are no positives, and None when inputs are missing.
    """
    if scores is None or fdes_m is None:
        return None
    if len(scores) <= 0 or len(scores) != len(fdes_m):
        return None
    K = int(len(scores))
    s = np.asarray([float(x) for x in scores], dtype=np.float64)
    s = np.where(np.isfinite(s), s, -float("inf"))
    m = np.zeros((K,), dtype=np.float64)
    for i in range(K):
        f = fdes_m[i]
        if f is None:
            m[i] = 0.0
        else:
            try:
                m[i] = 1.0 if float(f) < float(threshold_m) else 0.0
            except Exception:
                m[i] = 0.0
    total_pos = float(np.sum(m))
    if total_pos <= 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    m_sorted = m[order]
    tp = np.cumsum(m_sorted)
    fp = np.cumsum(1.0 - m_sorted)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / total_pos
    return _average_precision_from_pr(precision, recall)


def _openloop_success_rc_spl(
    *,
    goal_distance_m: float | None,
    end_dist_to_goal_m: float | None,
    path_length_m: float | None,
    success_threshold_m: float = 3.0,
) -> tuple[bool | None, float | None, float | None]:
    """Open-loop approximations of SR/RC/SPL from snapshot geometry."""
    if end_dist_to_goal_m is None or (not math.isfinite(float(end_dist_to_goal_m))):
        return None, None, None
    s = bool(float(end_dist_to_goal_m) <= float(success_threshold_m))

    rc = None
    spl = None
    try:
        if (
            goal_distance_m is not None
            and math.isfinite(float(goal_distance_m))
            and float(goal_distance_m) > 1e-9
        ):
            rc0 = (float(goal_distance_m) - float(end_dist_to_goal_m)) / float(goal_distance_m)
            rc = float(max(0.0, min(1.0, rc0))) if math.isfinite(rc0) else None
            if (
                path_length_m is not None
                and math.isfinite(float(path_length_m))
                and float(path_length_m) >= 0
            ):
                denom = max(float(path_length_m), float(goal_distance_m))
                if denom > 1e-9:
                    spl = float(float(goal_distance_m) / denom) if s else 0.0
    except Exception:
        rc = None
        spl = None
    return s, rc, spl


def _dcr_tcr_from_corridor(
    *,
    pred_xy: np.ndarray | None,
    compliant_polyline_xy: np.ndarray | None,
    dt_s: float,
    corridor_radius_m: float = 1.0,
) -> tuple[float | None, float | None]:
    """Distance/Time compliance ratio vs a corridor around a reference polyline.

    NOTE: This is intentionally **un-gated** (computed regardless of "success") because trajectory
    selection is
    open-loop. If you want SocialNav-style gating (Eq. 6), gate the returned values elsewhere.
    """
    if pred_xy is None or compliant_polyline_xy is None:
        return None, None
    p = np.asarray(pred_xy, dtype=np.float64)
    ref = np.asarray(compliant_polyline_xy, dtype=np.float64)
    if p.ndim != 2 or ref.ndim != 2 or p.shape[1] < 2 or ref.shape[1] < 2:
        return None, None
    if int(p.shape[0]) <= 1:
        return 1.0, 1.0

    seg = p[1:, :2] - p[:-1, :2]
    seg_len = np.linalg.norm(seg, axis=1)
    mid = 0.5 * (p[1:, :2] + p[:-1, :2])
    n_seg = int(seg_len.shape[0])
    if n_seg <= 0:
        return 1.0, 1.0

    def point_to_polyline_dist(pt: np.ndarray, poly: np.ndarray) -> float:
        m = float("inf")
        for i in range(int(poly.shape[0]) - 1):
            a = poly[i, :2]
            b = poly[i + 1, :2]
            ab = b - a
            l2 = float(np.dot(ab, ab))
            if l2 <= 1e-12:
                d = float(np.linalg.norm(pt - a))
            else:
                t = float(np.dot(pt - a, ab) / l2)
                t = max(0.0, min(1.0, t))
                proj = a + t * ab
                d = float(np.linalg.norm(pt - proj))
            if d < m:
                m = d
        return m

    r = float(corridor_radius_m)
    compliant = np.zeros((n_seg,), dtype=bool)
    for i in range(n_seg):
        try:
            compliant[i] = bool(point_to_polyline_dist(mid[i, :2], ref) <= r)
        except Exception:
            compliant[i] = False

    d_actual = float(np.sum(seg_len)) if seg_len.size else 0.0
    if not (math.isfinite(d_actual) and d_actual >= 0):
        return None, None
    if d_actual <= 1e-9:
        return 1.0, 1.0
    d_comp = float(np.sum(seg_len[compliant])) if seg_len.size else 0.0
    dcr = float(d_comp / d_actual)
    # Time: ratio over segments (dt cancels, keep semantic for clarity).
    _ = float(dt_s)
    tcr = float(float(np.sum(compliant)) / float(n_seg)) if n_seg > 0 else 1.0
    return dcr, tcr


def _ade_prefix_seconds(
    pred_xy: np.ndarray | None,
    gt_xy: np.ndarray | None,
    *,
    dt_s: float,
    seconds: float,
) -> float | None:
    """ADE over the first `seconds` of the trajectory (prefix), using `dt_s` to map seconds->points.

    Note: in this repo, trajectories are represented as a sequence of future points sampled at dt_s.
    For a prefix duration T, we include points at dt, 2dt, ..., floor(T/dt)*dt.
    """
    if pred_xy is None or gt_xy is None:
        return None
    if pred_xy.ndim != 2 or gt_xy.ndim != 2 or pred_xy.shape[1] < 2 or gt_xy.shape[1] < 2:
        return None
    if not (dt_s > 0) or not math.isfinite(dt_s):
        return None
    if seconds <= 0 or not math.isfinite(seconds):
        return None
    n = int(math.floor(float(seconds) / float(dt_s) + 1e-9))
    if n <= 0:
        return None
    m = min(int(len(pred_xy)), int(len(gt_xy)), int(n))
    if m <= 0:
        return None
    errs = np.linalg.norm(np.asarray(pred_xy[:m, :2]) - np.asarray(gt_xy[:m, :2]), axis=1)
    return float(np.mean(errs)) if errs.size > 0 else None


def _ensure_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out_dir.iterdir()]
    if existing and not overwrite:
        raise FileExistsError(
            f"--out already exists and is non-empty: {out_dir}. Remove it or pass --overwrite."
        )


def _open_jsonl(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def _open_csv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else (text + "\n"), encoding="utf-8")


def _write_metrics_summary_simple_txt(csv_path: Path, txt_path: Path) -> None:
    """Write a browser-friendly companion for metrics_summary_simple.csv.

    Per user request: copy the CSV contents verbatim into a .txt file (same two-line format).
    """
    try:
        s = csv_path.read_text(encoding="utf-8")
    except Exception:
        return
    _write_text(txt_path, s)


def _write_metrics_summary_csv(path: Path, metrics: dict[str, Any]) -> None:
    """Write a header + exactly one row, where each top-level metrics key becomes a column.

    Nested dict/list values are JSON-stringified so this stays one-row and copy/paste-friendly.
    """
    cols = sorted([str(k) for k in metrics.keys()])
    row: dict[str, Any] = {}
    for k in cols:
        v = metrics.get(k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[k] = v
        else:
            try:
                row[k] = json.dumps(v, sort_keys=True)
            except Exception:
                row[k] = str(v)
    with _open_csv(path) as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerow(row)


def _write_metrics_summary_simple_csv(path: Path, metrics: dict[str, Any]) -> None:
    """Write a header + exactly one row, with only core identifiers + key trajectory-selection
    metrics."""

    cols = [
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
    ]
    row = {k: metrics.get(k) for k in cols}
    with _open_csv(path) as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerow(row)


def _format_metrics_report(metrics: dict[str, Any], *, run_dir: Path) -> str:
    def fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            if not math.isfinite(v):
                return str(v)
            # Keep reasonable precision for copy/paste.
            return f"{v:.6f}".rstrip("0").rstrip(".")
        return str(v)

    # Group keys for readability.
    sections: list[tuple[str, list[str]]] = [
        (
            "Overview",
            [
                "task",
                "slow_brain_fast_planner_version",
                "job_started_at_utc",
                "job_finished_at_utc",
                "job_duration_s",
                "created_at_utc",
                "dataset",
                "planner_source",
                "selector",
                "episodes_total",
                "snapshots_total",
                "snapshots_evaluated",
                "snapshots_skipped",
                "accuracy",
            ],
        ),
        (
            "Model output health",
            [
                "stop_rate",
                "stop_count",
                "invalid_index_rate",
                "invalid_index_count",
                "pred_error_rate",
                "pred_error_count",
            ],
        ),
        (
            "Token usage (best-effort)",
            [
                "model_calls",
                "prompt_tokens_total",
                "cached_prompt_tokens_total",
                "output_tokens_total",
                "thoughts_tokens_total",
                "total_tokens_total",
                "total_tokens_no_thoughts_total",
                "prompt_tokens_avg",
                "cached_prompt_tokens_avg",
                "output_tokens_avg",
                "thoughts_tokens_avg",
                "total_tokens_avg",
                "total_tokens_no_thoughts_avg",
                "latency_ms_avg",
                "latency_ms_p50",
                "latency_ms_p90",
            ],
        ),
        (
            "ADE (selected vs baselines)",
            [
                "ade_model_avg",
                "ade_model_0_5s_avg",
                "ade_model_1_0s_avg",
                "ade_model_2_0s_avg",
                "ade_min_avg",
                "ade_min_all_avg",
                "ade_score_avg",
                "ade_score_0_5s_avg",
                "ade_score_1_0s_avg",
                "ade_score_2_0s_avg",
                "ade_min_0_5s_avg",
                "ade_min_1_0s_avg",
                "ade_min_2_0s_avg",
                "ade_count",
                "accuracy_vs_min_ade",
                "accuracy_vs_min_ade_all",
                "accuracy_vs_score",
            ],
        ),
        (
            "FDE / MOE / mAP (open-loop)",
            [
                "fde_model_avg",
                "fde_score_avg",
                "fde_score_masked_laststep_avg",
                "fde_min_avg",
                "fde_min_all_avg",
                "fde_count",
                "fde_score_masked_laststep_count",
                "moe_deg_avg",
                "moe_deg_score_avg",
                "moe_deg_min_avg",
                "moe_deg_min_all_avg",
                "moe_deg_mean_all_avg",
                "moe_deg_count",
                "map_fde_2m_avg",
                "map_fde_2m_count",
            ],
        ),
        (
            "Goal-relative (selected trajectory)",
            [
                "selected_end_dist_to_goal_m_avg",
                "selected_goal_ang_diff_deg_avg",
                "selected_traj_avg_dist_to_goal_m_avg",
                "selected_goal_progress_m_avg",
                "selected_end_dist_to_goal_m_count",
                "selected_goal_ang_diff_deg_count",
                "selected_traj_avg_dist_to_goal_m_count",
                "selected_goal_progress_m_count",
            ],
        ),
        (
            "Open-loop social metrics",
            [
                "maoe_deg_avg",
                "dcr_avg",
                "tcr_avg",
            ],
        ),
        (
            "Goal stats (snapshot)",
            [
                "goal_distance_m_avg",
                "goal_distance_m_count",
            ],
        ),
        (
            "Route deviation (if available)",
            [
                "route_dev_model_avg",
                "route_dev_min_avg",
                "route_dev_score_avg",
                "route_dev_count",
                "accuracy_vs_route_min",
                "accuracy_vs_route_score",
            ],
        ),
        (
            "Auto-enabled breakdown",
            [
                "auto_enabled_true_frac",
                "auto_enabled_true_count",
                "auto_enabled_false_count",
                "auto_enabled_missing_count",
            ],
        ),
    ]

    # Render.
    lines: list[str] = []
    lines.append("Slow Brain, Fast Planner Trajectory Selection Metrics Report")
    lines.append(f"Run dir: {run_dir}")
    lines.append("")

    # Include file pointers for convenience.
    lines.append("Artifacts:")
    lines.append(f"- metrics.json: {run_dir / 'metrics.json'}")
    lines.append(f"- predictions.jsonl: {run_dir / 'predictions.jsonl'}")
    lines.append(f"- predictions.csv: {run_dir / 'predictions.csv'}")
    lines.append(f"- report.html: {run_dir / 'report.html'}")
    lines.append("")

    for title, keys in sections:
        present = [(k, metrics.get(k)) for k in keys if k in metrics]
        if not present:
            continue
        lines.append(title)
        lines.append("-" * len(title))
        w = max(len(k) for k, _ in present)
        for k, v in present:
            lines.append(f"{k.ljust(w)} : {fmt(v)}")
        lines.append("")

    # Always include skipped reasons if any.
    sr = metrics.get("skipped_reasons")
    if isinstance(sr, dict) and sr:
        lines.append("Skipped reasons")
        lines.append("--------------")
        for k in sorted(sr):
            lines.append(f"{k}: {sr[k]}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _log_progress(msg: str) -> None:
    try:
        from tqdm import tqdm as _tqdm

        _tqdm.write(str(msg))
    except Exception:
        print(str(msg))


def _epid(p: Path) -> str:
    return p.parent.name if p.name == "episode.json" else p.stem


def main(argv: list[str] | None = None) -> int:
    t_start = _dt.datetime.now(tz=_dt.UTC)
    parser = argparse.ArgumentParser(
        description="Trajectory-selection benchmark with PyTorch DataLoader parallelism.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset", default="data/processed", help="Canonical dataset root.")
    parser.add_argument(
        "--planner-source-report",
        default=None,
        choices=["prelogged", "onnx"],
        help="Planner candidate source (metadata only; written to metrics/reporting).",
    )
    parser.add_argument(
        "--out", default=None, help="Run directory output path (overrides --output-dir/--exp-name)."
    )
    parser.add_argument("--output-dir", default="logs", help="Base output directory.")
    parser.add_argument(
        "--exp-name", default="trajectory_selection", help="Experiment name prefix."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite --out if it exists.")

    parser.add_argument(
        "--episode-id",
        action="append",
        default=None,
        help="Only evaluate a specific episode_id (repeatable).",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="Only evaluate first N episodes."
    )
    parser.add_argument(
        "--require-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, skip snapshots that do not contain goal_xy.",
    )
    parser.add_argument("--max-snapshots-per-episode", type=int, default=None)
    parser.add_argument("--max-snapshots-total", type=int, default=None)
    parser.add_argument("--snapshot-stride-s", type=float, default=None)
    parser.add_argument(
        "--use-takeover-clips",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "If true, evaluate only at clip t0 times from takeover_clips.jsonl "
            "(reduces VLM queries vs per-tick snapshots). "
            "Default: auto (enabled if takeover_clips.jsonl exists)."
        ),
    )
    parser.add_argument(
        "--takeover-clips-dir",
        default=None,
        help="Directory containing takeover_clips.jsonl + summary.json (default: "
        "<dataset>/takeover_clips).",
    )
    parser.add_argument(
        "--takeover-clips-phase",
        default=None,
        choices=["pre", "center", "post", "none"],
        help="Optional phase filter when using takeover clips (default: all).",
    )
    parser.add_argument(
        "--takeover-clips-label-filter",
        default="any",
        choices=["any", "takeover_only", "no_takeover_only"],
        help=(
            "When using takeover_clips.jsonl, optionally filter by clip label_takeover_request.\n"
            "  any: evaluate all clips (default)\n"
            "  takeover_only: only clips with label_takeover_request==1\n"
            "  no_takeover_only: only clips with label_takeover_request==0\n"
            "Note: this filter is ignored for clip-times files that do not include labels."
        ),
    )
    parser.add_argument("--traj-dt-s", type=float, default=constants.DEFAULT_TRAJECTORY_DT_S)
    parser.add_argument("--seed", type=int, default=0)

    # A0 label config
    parser.add_argument(
        "--a0-gt-rule",
        choices=[
            "raw_argmax",
            "planner_v1_nms_softmax_argmax",
            "planner_v2_nms_softmax_threshold_argmax",
        ],
        default="raw_argmax",
    )
    parser.add_argument("--nms-max-trajectories", type=int, default=constants.NMS_MAX_TRAJECTORIES)
    parser.add_argument(
        "--nms-distance-threshold", type=float, default=constants.NMS_DISTANCE_THRESHOLD
    )
    parser.add_argument("--a0-prob-threshold", type=float, default=constants.PROB_THRESHOLD)

    # Overlays + prompting
    parser.add_argument(
        "--write-overlays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write overlay images under RUN_DIR/artifacts/overlays (needed for VLM models).",
    )
    parser.add_argument("--overlay-image-width", type=int, default=None)
    parser.add_argument(
        "--overlay-candidate-set",
        choices=["planner_v2", "raw_topk", "kcenter_endpoints", "nms_only"],
        default="planner_v2",
    )
    parser.add_argument("--overlay-topk", type=int, default=6)
    parser.add_argument("--overlay-min-score", type=float, default=None)
    parser.add_argument(
        "--static-candidates-json",
        default=None,
        help=(
            "Optional JSON file containing a *static* candidate set (same K trajectories for all "
            "snapshots).\n"
            "Create one via scripts/trajectory_selection/build_static_candidate_set.py.\n"
            "If set, the benchmark will ignore per-tick planner candidates and use this static set "
            "for overlays/ADE."
        ),
    )
    parser.add_argument(
        "--candidate-points-override-npy",
        default=None,
        help=(
            "Optional .npy file containing an external candidate trajectory library (e.g., the raw "
            "64-anchor set).\n"
            "If set, the benchmark will KEEP per-timestep planner scores/indices from "
            "planner_candidates.jsonl, but\n"
            "REPLACE each candidate's points_xy with the corresponding trajectory from this "
            "library (index-aligned).\n"
            "This enables experiments like oracle minADE over anchors, and score-argmax over "
            "anchors."
        ),
    )
    parser.add_argument(
        "--candidate-points-override-mode",
        choices=["notebook_v1", "xy"],
        default="notebook_v1",
        help=(
            "How to interpret --candidate-points-override-npy.\n"
            "- notebook_v1: apply the notebook conversion (x*=0.51, y*=0.32-0.16, then cumsum over "
            "time)\n"
            "- xy: treat values as already being (x,y) in meters (no scaling/cumsum)."
        ),
    )
    parser.add_argument(
        "--overlay-projection",
        choices=["fisheye_v1", "simple_xy"],
        default=constants.OVERLAY_PROJECTION,
    )
    parser.add_argument("--overlay-camera-height-m", type=float, default=0.41)
    parser.add_argument("--overlay-fisheye-k", type=float, default=0.0)
    parser.add_argument("--overlay-fisheye-fx", type=float, default=None)
    parser.add_argument("--overlay-fisheye-fy", type=float, default=None)
    parser.add_argument("--overlay-fisheye-cx", type=float, default=None)
    parser.add_argument("--overlay-fisheye-cy", type=float, default=None)
    parser.add_argument("--overlay-px-per-meter", type=float, default=25.0)
    parser.add_argument("--overlay-origin-u", type=float, default=0.5)
    parser.add_argument("--overlay-origin-v", type=float, default=0.9)
    parser.add_argument("--overlay-line-width", type=int, default=constants.LINE_WIDTH)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--overlay-point-radius", type=int, default=2)
    parser.add_argument("--overlay-point-every", type=int, default=2)
    parser.add_argument("--overlay-label-font-size", type=int, default=constants.LABEL_FONT_SIZE)
    # Visual prompting: draw robot-width corridors (polygon) instead of centerline trajectories.
    parser.add_argument(
        "--overlay-traj-style",
        choices=["line", "corridor"],
        default="line",
        help='Trajectory rendering style in overlays: "line" (legacy) or "corridor" (wide '
        "footprint).",
    )
    parser.add_argument(
        "--overlay-robot-width-m",
        type=float,
        default=0.8,
        help="Assumed robot width in meters for corridor rendering (default: 0.8m).",
    )
    parser.add_argument(
        "--overlay-corridor-alpha",
        type=float,
        default=0.35,
        help="Alpha for corridor fill in [0,1] (default: 0.35).",
    )
    parser.add_argument(
        "--overlay-corridor-outline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, draw left/right corridor boundary lines on top of the filled corridor.",
    )
    parser.add_argument(
        "--overlay-goal-direction-arrow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, draw a floating goal-direction arrow on the overlay image (VLM-friendly).",
    )
    parser.add_argument(
        "--overlay-goal-marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, draw the legacy ground-projected goal marker in the overlay image (magenta). "
            "This is independent of --overlay-goal-direction-arrow."
        ),
    )
    parser.add_argument(
        "--overlay-goal-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, draw a goal banner at the top of the overlay image (text or goal_xy "
        "fallback).",
    )
    parser.add_argument(
        "--overlay-goal-projection-marker",
        choices=["none", "dot"],
        default="none",
        help="Ground-projection marker style for the goal when using the raised goal arrow "
        "(default: none).",
    )
    parser.add_argument("--rgb-time-tolerance-s", type=float, default=constants.DEFAULT_DATA_DT_S)

    parser.add_argument("--prompt-history-frames", type=int, default=0)
    parser.add_argument("--prompt-image-width", type=int, default=None)
    parser.add_argument(
        "--prompt-show-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, include planner confidence columns (score/probabilities) in the candidate "
            "table text. "
            "If false, hide planner confidence but keep geometry (end_xy/traj_dist/goal-relative "
            "metrics)."
        ),
    )
    parser.add_argument(
        "--prompt-hide-scores",
        action="store_true",
        help="Alias for --no-prompt-show-scores.",
    )
    parser.add_argument(
        "--prompt-goal-info",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, include per-snapshot goal numeric info in the prompt text "
        "(vector/bearing/distance).",
    )
    parser.add_argument(
        "--prompt-goal-geometry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, include goal-relative geometry columns in the candidate table "
            "(end_goal_dist_m, goal_ang_diff_deg, progress_m)."
        ),
    )

    # Model adapter
    parser.add_argument(
        "--model",
        choices=[
            "dummy_argmax",
            "dummy_always0",
            "dummy_random",
            "dummy_random_all",
            "dummy_random_topk",
            "dummy_sample_nms_prob",
            "oracle_min_ade",
            "oracle_min_ade_all",
            "openai_chat_completions",
            "gemini_genai",
            *HIERARCHICAL_MODELS,
        ],
        default="dummy_argmax",
    )
    parser.add_argument(
        "--dummy-topk", type=int, default=1, help="K for dummy_random_topk baseline."
    )
    parser.add_argument("--gemini-model", default=constants.DEFAULT_GEMINI_MODEL)
    parser.add_argument("--gemini-temperature", type=float, default=0.0)
    parser.add_argument(
        "--gemini-include-thoughts", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--gemini-thinking-budget", type=int, default=None)
    parser.add_argument("--gemini-thinking-level", default=None)
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument(
        "--openai-response-format-json", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--openai-timeout-s", type=float, default=60.0)

    # Hierarchical selector knobs (only used when --model=gemini_genai_hierarchical or
    # --model=gemini_genai_chain_of_planning)
    parser.add_argument(
        "--hier-branch",
        type=int,
        default=4,
        help="Branching factor: #cluster representatives shown per rep stage.",
    )
    parser.add_argument(
        "--hier-max-levels",
        type=int,
        default=3,
        help="Max selection rounds (levels), INCLUDING the final leaf-pick round.",
    )
    parser.add_argument(
        "--hier-max-leaf",
        type=int,
        default=16,
        help="Leaf threshold: if current pool size <= this, do a final pick; otherwise keep "
        "clustering.",
    )
    parser.add_argument(
        "--hier-final-topk",
        type=int,
        default=6,
        help="Hard cap for the final pick set (downselect if leaf pool is larger).",
    )
    parser.add_argument(
        "--hier-rep-retries",
        type=int,
        default=1,
        help="Retries for rep selection when model returns invalid index (not in allowed reps "
        "set).",
    )
    parser.add_argument(
        "--hier-use-chat-history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, run hierarchical stages as a single multi-turn chat (reduces repeated "
        "system prompt).",
    )
    parser.add_argument(
        "--hier-cluster-space",
        choices=["traj", "endpoint"],
        default="endpoint",
        help="Clustering feature space: 'traj' uses subsampled trajectory shape; 'endpoint' uses "
        "endpoint only.",
    )
    parser.add_argument(
        "--hier-traj-feature-points",
        type=int,
        default=6,
        help="When --hier-cluster-space=traj, number of subsampled points used in the feature "
        "vector.",
    )
    parser.add_argument(
        "--hier-balance-clusters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, enforce roughly-equal cluster sizes during assignment (reduces imbalance).",
    )
    parser.add_argument(
        "--hier-overlay-mode",
        choices=["single", "grid"],
        default="grid",
        help=(
            "How to visualize clusters in rep-selection stages. "
            "'single' draws reps on one image; 'grid' renders a 2x2 panel grid (one cluster per "
            "panel)."
        ),
    )

    # DataLoader parallelism
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers for parallel data preparation (default: 0 = main "
        "process).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor (default: 2).",
    )

    # Sharding (for multi-GPU/multi-node)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument(
        "--shard-mode",
        default="episode_round_robin",
        choices=["episode_round_robin", "episode_hash", "episode_greedy", "snapshot_hash"],
        help=(
            "How to partition work across shards. "
            "'episode_round_robin' keeps whole episodes but can be imbalanced; "
            "'episode_greedy' balances snapshot counts while keeping episodes intact (recommended "
            "for VLM); "
            "'snapshot_hash' gives the best balance but may split episodes across shards."
        ),
    )
    parser.add_argument("--no-gt-metrics", action="store_true")
    parser.add_argument(
        "--skip-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If true, skip the expensive per-episode odom/schema validation step.",
    )

    # Report
    parser.add_argument("--write-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--write-local-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, write per-snapshot local-frame plots under RUN_DIR/artifacts/local_plots. "
            "This requires matplotlib and is best-effort. Default: false (keeps CLI runnable in "
            "minimal envs)."
        ),
    )

    args = parser.parse_args(argv)

    # Sharding config
    if args.num_shards is not None or args.shard_id is not None:
        num_shards = int(args.num_shards or 1)
        shard_id = int(args.shard_id or 0)
    else:
        num_shards, shard_id = get_shard_config_from_env()
    shard_enabled = num_shards > 1

    dataset_path = Path(args.dataset).resolve()
    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = str(args.exp_name)
        if shard_enabled:
            exp_name = f"{exp_name}_shard{shard_id}of{num_shards}"
        out_dir = (Path(args.output_dir) / exp_name / f"{exp_name}_{timestamp}").resolve()
    _ensure_out_dir(out_dir, overwrite=bool(args.overwrite))
    experiment_name = str(out_dir.parent.name) if out_dir.parent is not None else str(out_dir)
    trial_name = str(out_dir.name)

    # Episode selection
    episode_meta_paths = find_episode_metadata_files(dataset_path)
    if args.episode_id:
        wanted = {str(x) for x in args.episode_id if str(x).strip()}
        avail = {_epid(p) for p in episode_meta_paths}
        missing = sorted([e for e in wanted if e not in avail])
        if missing:
            raise SystemExit(f"--episode-id not found in dataset: {missing}")
        episode_meta_paths = [p for p in episode_meta_paths if _epid(p) in wanted]
    if args.max_episodes is not None:
        episode_meta_paths = episode_meta_paths[: int(args.max_episodes)]

    # GT strictness: if GT metrics are enabled, require canonical odom in the dataset.
    if not bool(args.no_gt_metrics):
        skip_val = bool(args.skip_validation)
        if args.skip_validation is None:
            # Auto-skip if we see evidence of prior validation or successful processing.
            if (dataset_path / ".validated_trajectory_selection").exists() or (
                dataset_path / ".validated_task2"
            ).exists():
                skip_val = True
            elif (dataset_path / "process_summary.json").exists():
                # process_summary.json implies it went through scripts/process_data.py
                skip_val = True

        if skip_val:
            logger.info(
                "Skipping per-episode validation (dataset marked as validated or processed)."
            )
        else:
            from tqdm import tqdm

            from slow_brain_fast_planner.benchmarks.dataset import load_episode

            missing_odom: list[str] = []
            for meta in tqdm(episode_meta_paths, desc="Validating episodes (odom/planner)"):
                ep = load_episode(dataset_path, meta)
                if (not ep.schema_valid) or (not getattr(ep, "odom", None)):
                    missing_odom.append(_epid(meta))
            if missing_odom:
                missing_odom = sorted(set(missing_odom))
                raise SystemExit(
                    "GT metrics are enabled but canonical odom stream is missing for episodes: "
                    f"{missing_odom}. Either (a) include odom in the canonical dataset, "
                    "(b) run with --no-gt-metrics."
                )

            # Success! Drop a token to skip next time.
            try:
                (dataset_path / ".validated_trajectory_selection").touch()
            except Exception:
                pass  # Read-only dataset, no big deal.

    a0_cfg = A0Config(
        gt_rule=args.a0_gt_rule,
        nms_max_trajectories=int(args.nms_max_trajectories),
        nms_distance_threshold=float(args.nms_distance_threshold),
        prob_threshold=float(args.a0_prob_threshold),
    )

    overlay_cfg = OverlayConfig(
        top_k=int(args.overlay_topk),
        min_score=args.overlay_min_score,
        projection=str(args.overlay_projection),
        candidate_set=str(args.overlay_candidate_set),
        nms_max_trajectories=int(args.nms_max_trajectories),
        nms_distance_threshold=float(args.nms_distance_threshold),
        prob_threshold=float(args.a0_prob_threshold),
        camera_height_m=float(args.overlay_camera_height_m),
        fisheye_k=float(args.overlay_fisheye_k),
        fisheye_fx=args.overlay_fisheye_fx,
        fisheye_fy=args.overlay_fisheye_fy,
        fisheye_cx=args.overlay_fisheye_cx,
        fisheye_cy=args.overlay_fisheye_cy,
        px_per_meter=float(args.overlay_px_per_meter),
        origin_u=float(args.overlay_origin_u),
        origin_v=float(args.overlay_origin_v),
        line_width=int(args.overlay_line_width),
        alpha=float(args.overlay_alpha),
        draw_points=True,
        point_radius=int(args.overlay_point_radius),
        point_every=int(args.overlay_point_every),
        highlight_pred=False,
        highlight_label=False,
        label_indices=True,
        label_font_size=int(args.overlay_label_font_size),
        extend_start_to_bottom=True,
        traj_style=str(args.overlay_traj_style),
        robot_width_m=float(args.overlay_robot_width_m),
        corridor_alpha=float(args.overlay_corridor_alpha),
        corridor_outline=bool(args.overlay_corridor_outline),
        draw_goal_direction_arrow=bool(args.overlay_goal_direction_arrow),
        draw_goal_marker=bool(args.overlay_goal_marker),
        draw_goal_text=bool(args.overlay_goal_text),
        goal_projection_marker=str(args.overlay_goal_projection_marker),
    )

    # Build jobs
    # Default behavior: use takeover clips if they exist, otherwise fall back to per-tick snapshots.
    # If the user explicitly enables takeover clips, require the file to exist (fail-fast).
    default_takeover_clips_dir = (
        Path(args.takeover_clips_dir).resolve()
        if args.takeover_clips_dir
        else (dataset_path / "takeover_clips")
    )
    auto_clips_path = (default_takeover_clips_dir / "takeover_clips.jsonl").resolve()
    # Some datasets (e.g. COCO) ship a lightweight clip-times file (same shape as takeover_clips:
    # episode_id + t0),
    # but without the takeover labels.
    default_ts_eval_clips_dir = (dataset_path / "trajectory_selection_eval_clips").resolve()
    auto_ts_clips_path = (
        default_ts_eval_clips_dir / "trajectory_selection_eval_clips.jsonl"
    ).resolve()
    # Backward-compat (older datasets).
    default_task2_eval_clips_dir = (dataset_path / "task2_eval_clips").resolve()
    auto_task2_clips_path = (default_task2_eval_clips_dir / "task2_eval_clips.jsonl").resolve()
    if args.use_takeover_clips is None:
        if auto_clips_path.exists():
            use_takeover_clips = True
            takeover_clips_dir = default_takeover_clips_dir
            clips_path = auto_clips_path
        elif auto_ts_clips_path.exists():
            use_takeover_clips = True
            takeover_clips_dir = default_ts_eval_clips_dir
            clips_path = auto_ts_clips_path
        elif auto_task2_clips_path.exists():
            use_takeover_clips = True
            takeover_clips_dir = default_task2_eval_clips_dir
            clips_path = auto_task2_clips_path
        else:
            use_takeover_clips = False
            takeover_clips_dir = None
    else:
        use_takeover_clips = bool(args.use_takeover_clips)
        takeover_clips_dir = default_takeover_clips_dir if use_takeover_clips else None

    takeover_clips_summary: dict[str, Any] | None = None
    if takeover_clips_dir is not None:
        # If user explicitly enabled takeover clips, require takeover_clips.jsonl.
        if args.use_takeover_clips is not None:
            clips_path = (takeover_clips_dir / "takeover_clips.jsonl").resolve()
        summary_path = (takeover_clips_dir / "summary.json").resolve()
        if not clips_path.exists():
            raise SystemExit(f"--use-takeover-clips enabled but missing: {clips_path}")
        if summary_path.exists():
            try:
                takeover_clips_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                takeover_clips_summary = None
        jobs = build_trajectory_selection_jobs_from_takeover_clips(
            dataset_path=dataset_path,
            episode_meta_paths=episode_meta_paths,
            takeover_clips_path=clips_path,
            out_dir=out_dir,
            a0_cfg=a0_cfg,
            overlay_cfg=overlay_cfg,
            static_candidates_json=(
                str(args.static_candidates_json) if args.static_candidates_json else None
            ),
            candidate_points_override_npy=(
                str(args.candidate_points_override_npy)
                if args.candidate_points_override_npy
                else None
            ),
            candidate_points_override_mode=str(args.candidate_points_override_mode),
            rgb_time_tolerance_s=float(args.rgb_time_tolerance_s),
            require_goal=bool(args.require_goal),
            write_overlays=bool(args.write_overlays),
            overlay_image_width=(
                int(args.overlay_image_width) if args.overlay_image_width else None
            ),
            prompt_history_frames=int(args.prompt_history_frames),
            prompt_image_width=(
                int(args.prompt_image_width) if args.prompt_image_width is not None else None
            ),
            compute_gt_metrics=(not bool(args.no_gt_metrics)),
            traj_dt_s=float(args.traj_dt_s),
            clip_phase=(
                str(args.takeover_clips_phase) if args.takeover_clips_phase is not None else None
            ),
            clip_label_filter=str(args.takeover_clips_label_filter),
            max_snapshots_per_episode=args.max_snapshots_per_episode,
            max_snapshots_total=args.max_snapshots_total,
            write_rgb_frames=(str(args.model) in HIERARCHICAL_MODELS),
        )
    else:
        jobs = build_trajectory_selection_jobs(
            dataset_path=dataset_path,
            episode_meta_paths=episode_meta_paths,
            out_dir=out_dir,
            a0_cfg=a0_cfg,
            overlay_cfg=overlay_cfg,
            static_candidates_json=(
                str(args.static_candidates_json) if args.static_candidates_json else None
            ),
            candidate_points_override_npy=(
                str(args.candidate_points_override_npy)
                if args.candidate_points_override_npy
                else None
            ),
            candidate_points_override_mode=str(args.candidate_points_override_mode),
            rgb_time_tolerance_s=float(args.rgb_time_tolerance_s),
            require_goal=bool(args.require_goal),
            write_overlays=bool(args.write_overlays),
            overlay_image_width=(
                int(args.overlay_image_width) if args.overlay_image_width else None
            ),
            prompt_history_frames=int(args.prompt_history_frames),
            prompt_image_width=(
                int(args.prompt_image_width) if args.prompt_image_width is not None else None
            ),
            compute_gt_metrics=(not bool(args.no_gt_metrics)),
            traj_dt_s=float(args.traj_dt_s),
            snapshot_stride_s=args.snapshot_stride_s,
            max_snapshots_per_episode=args.max_snapshots_per_episode,
            max_snapshots_total=args.max_snapshots_total,
            write_rgb_frames=(str(args.model) in HIERARCHICAL_MODELS),
        )

    if not jobs:
        logger.warning("No snapshots found to evaluate.")
        return 0

    # Loud run banner (helps catch missing shell line continuations / wrong args).
    model_desc = str(args.model)
    if str(args.model) in ("gemini_genai", *HIERARCHICAL_MODELS):
        model_desc = f"gemini_genai:{args.gemini_model}"
    elif str(args.model) == "openai_chat_completions":
        model_desc = f"openai_chat_completions:{args.openai_model}@{args.openai_base_url}"
    planner_src = (
        str(args.planner_source_report) if args.planner_source_report is not None else "unknown"
    )
    logger.info("=" * 88)
    logger.info(
        "Trajectory selection | selector=%s | planner_source=%s | dataset=%s",
        model_desc,
        planner_src,
        str(dataset_path),
    )
    logger.info(
        "Planned samples=%d (%s) | episodes=%d | num_workers=%d | exp=%s | trial=%s",
        int(len(jobs)),
        "clips" if takeover_clips_dir is not None else "snapshots",
        int(len(episode_meta_paths)),
        int(args.num_workers),
        experiment_name,
        trial_name,
    )
    if takeover_clips_dir is not None:
        logger.info("Using clip file: %s", str(clips_path))
    logger.info("=" * 88)

    # Create dataset and dataloader
    dataset = SnapshotDataset(jobs)

    # Apply sharding via sampler
    if shard_enabled:
        sampler = SnapshotShardedSampler(
            jobs,
            num_shards=num_shards,
            shard_id=shard_id,
            mode=str(args.shard_mode),
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor) if args.num_workers > 0 else None,
            collate_fn=SnapshotDataset.collate_fn,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor) if args.num_workers > 0 else None,
            collate_fn=SnapshotDataset.collate_fn,
        )

    # Model adapter init
    model_mode = {
        "dummy_argmax": "argmax_score",
        "dummy_always0": "always_0",
        "dummy_random": "random",
        "dummy_random_all": "random_all",
        "dummy_random_topk": "random_topk",
        "dummy_sample_nms_prob": "sample_nms_prob",
    }.get(str(args.model))
    if str(args.model) == "oracle_min_ade":
        model = None
    elif str(args.model) == "oracle_min_ade_all":
        model = None
    elif str(args.model).startswith("dummy_"):
        assert model_mode is not None
        model = DummyTrajectoryModelAdapter(
            mode=model_mode, seed=int(args.seed), topk=int(args.dummy_topk)
        )
    elif str(args.model) == "openai_chat_completions":
        if args.openai_model is None:
            raise SystemExit("--openai-model is required when --model=openai_chat_completions")
        model = OpenAIChatCompletionsHttpAdapter(
            OpenAIChatCompletionsConfig(
                model=str(args.openai_model),
                base_url=str(args.openai_base_url),
                timeout_s=float(args.openai_timeout_s),
                response_format_json=bool(args.openai_response_format_json),
                image_base_dir=out_dir,
            )
        )
    else:
        model = GeminiGenAIAdapter(
            GeminiGenAIConfig(
                model=str(args.gemini_model),
                temperature=float(args.gemini_temperature),
                image_base_dir=out_dir,
                include_thoughts=bool(args.gemini_include_thoughts),
                thinking_budget=args.gemini_thinking_budget,
                thinking_level=args.gemini_thinking_level,
            )
        )

    # Benchmark identifier (human-facing).
    benchmark_name = "trajectory_selection"
    model_id = str(args.model)
    decoding_config: dict[str, Any] = {}
    if str(args.model).startswith("dummy_"):
        decoding_config = {
            "adapter": str(args.model),
            "seed": int(args.seed),
            "topk": int(args.dummy_topk),
        }
    elif str(args.model) == "openai_chat_completions":
        model_id = f"openai_chat_completions:{args.openai_model}@{args.openai_base_url}"
        decoding_config = {
            "temperature": 0.0,
            "max_tokens": None,
            "response_format_json": bool(args.openai_response_format_json),
            "timeout_s": float(args.openai_timeout_s),
        }
    elif str(args.model) in ("gemini_genai", *HIERARCHICAL_MODELS):
        model_id = f"gemini_genai:{args.gemini_model}"
        decoding_config = {
            "temperature": float(args.gemini_temperature),
            "include_thoughts": bool(getattr(args, "gemini_include_thoughts", False)),
            "thinking_budget": getattr(args, "gemini_thinking_budget", None),
            "thinking_level": getattr(args, "gemini_thinking_level", None),
        }
        if str(args.model) in HIERARCHICAL_MODELS:
            decoding_config = dict(decoding_config)
            decoding_config["hierarchical"] = {
                "branch": int(args.hier_branch),
                "max_levels": int(args.hier_max_levels),
                "max_leaf": int(args.hier_max_leaf),
                "final_topk": int(args.hier_final_topk),
                "overlay_mode": str(args.hier_overlay_mode),
                "use_chat_history": bool(args.hier_use_chat_history),
                "cluster_space": str(args.hier_cluster_space),
                "traj_feature_points": int(args.hier_traj_feature_points),
                "balance_clusters": bool(args.hier_balance_clusters),
            }
    elif str(args.model) == "oracle_min_ade":
        model_id = "oracle_min_ade"
        decoding_config = {"baseline": "oracle_min_ade"}
    elif str(args.model) == "oracle_min_ade_all":
        model_id = "oracle_min_ade_all"
        decoding_config = {"baseline": "oracle_min_ade_all"}

    prompt_show_scores = bool(args.prompt_show_scores) and (
        not bool(getattr(args, "prompt_hide_scores", False))
    )
    prompt_cfg = VQATrajectoryPromptConfig(
        task_name=benchmark_name,
        task_description=None,
        include_candidate_score_table=True,
        include_planner_confidence=bool(prompt_show_scores),
        include_goal_hint=bool(args.prompt_goal_info),
        include_goal_geometry_columns=bool(args.prompt_goal_geometry),
    )
    prompt_version = (
        VQAHierarchicalTrajectorySelectionAdvisor.PROMPT_VERSION
        if str(args.model) in HIERARCHICAL_MODELS
        else VQATrajectorySelectionAdvisor.PROMPT_VERSION
    )
    advisor = None
    if str(args.model) not in ("oracle_min_ade", "oracle_min_ade_all"):
        assert model is not None
        if str(args.model) in HIERARCHICAL_MODELS:
            advisor = VQAHierarchicalTrajectorySelectionAdvisor(
                model=model,
                prompt_cfg=prompt_cfg,
                overlay_cfg=overlay_cfg,
                out_dir=out_dir,
                branch=int(args.hier_branch),
                max_leaf=int(args.hier_max_leaf),
                final_topk=int(args.hier_final_topk),
                max_levels=int(args.hier_max_levels),
                rep_prompt_retries=int(args.hier_rep_retries),
                use_chat_history=bool(args.hier_use_chat_history),
                hier_overlay_mode=str(args.hier_overlay_mode),
                cluster_space=str(args.hier_cluster_space),
                traj_feature_points=int(args.hier_traj_feature_points),
                balance_clusters=bool(args.hier_balance_clusters),
            )
        else:
            advisor = VQATrajectorySelectionAdvisor(model=model, prompt_cfg=prompt_cfg)

    # Write config
    write_json(
        out_dir / "config.json",
        {
            "slow_brain_fast_planner_version": __version__,
            "created_at_utc": utc_now_iso(),
            "dataset": str(dataset_path),
            "planner_source": str(args.planner_source_report)
            if args.planner_source_report is not None
            else None,
            "experiment_name": experiment_name,
            "trial_name": trial_name,
            "task": benchmark_name,
            "shard": {
                "enabled": shard_enabled,
                "num_shards": num_shards,
                "shard_id": shard_id,
                "mode": str(args.shard_mode),
            },
            "dataloader": {
                "num_workers": int(args.num_workers),
                "prefetch_factor": int(args.prefetch_factor),
            },
            "gt_metrics": {
                "enabled": (not bool(args.no_gt_metrics)),
                "traj_dt_s": float(args.traj_dt_s),
            },
            "gt_sources": {
                "canonical_odom": True,
            },
            "episode_id": args.episode_id,
            "max_episodes": args.max_episodes,
            "require_goal": bool(args.require_goal),
            "max_snapshots_per_episode": args.max_snapshots_per_episode,
            "max_snapshots_total": args.max_snapshots_total,
            "snapshot_stride_s": args.snapshot_stride_s,
            "seed": int(args.seed),
            "takeover_clips": {
                "enabled": bool(takeover_clips_dir is not None),
                "dir": str(takeover_clips_dir) if takeover_clips_dir is not None else None,
                "phase": str(args.takeover_clips_phase)
                if args.takeover_clips_phase is not None
                else None,
                "summary": takeover_clips_summary,
            },
            "a0": {
                "gt_rule": args.a0_gt_rule,
                "nms_max_trajectories": int(args.nms_max_trajectories),
                "nms_distance_threshold": float(args.nms_distance_threshold),
                "prob_threshold": float(args.a0_prob_threshold),
            },
            "candidates": {
                "static_candidates_json": (
                    str(args.static_candidates_json) if args.static_candidates_json else None
                ),
                "candidate_points_override_npy": (
                    str(args.candidate_points_override_npy)
                    if args.candidate_points_override_npy
                    else None
                ),
                "candidate_points_override_mode": str(args.candidate_points_override_mode),
            },
            "model": {
                "adapter": str(args.model),
                "model_id": str(model_id),
                "decoding_config": decoding_config,
            },
            "overlays": {
                "enabled": bool(args.write_overlays),
                "candidate_set": str(args.overlay_candidate_set),
                "top_k": int(args.overlay_topk),
                "min_score": args.overlay_min_score,
                "projection": str(args.overlay_projection),
                "traj_style": str(args.overlay_traj_style),
                "robot_width_m": float(args.overlay_robot_width_m),
                "corridor_alpha": float(args.overlay_corridor_alpha),
                "corridor_outline": bool(args.overlay_corridor_outline),
                "goal_direction_arrow": bool(args.overlay_goal_direction_arrow),
                "rgb_time_tolerance_s": float(args.rgb_time_tolerance_s),
            },
            "prompt": {
                "history_frames": int(args.prompt_history_frames),
                "image_width": args.prompt_image_width,
                "show_scores": bool(prompt_show_scores),
            },
            "traces": {
                "files": ["traces/events.jsonl"],
                "event_types": ["obs", "model_call", "action", "metric"],
            },
        },
    )

    # Evaluation loop
    skipped_reasons: Counter[str] = Counter()
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
    fde_min_sum = fde_min_all_sum = fde_score_sum = fde_model_sum = 0.0
    fde_count = 0
    fde_min_count = 0
    fde_min_all_count = 0
    fde_score_count = 0
    fde_score_masked_laststep_sum = 0.0
    fde_score_masked_laststep_count = 0
    moe_deg_min_sum = moe_deg_min_all_sum = moe_deg_score_sum = moe_deg_model_sum = 0.0
    moe_deg_mean_all_sum = 0.0
    moe_deg_count = 0
    moe_deg_min_count = 0
    moe_deg_min_all_count = 0
    moe_deg_score_count = 0
    moe_deg_mean_all_count = 0
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
    stop_error_count = 0
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
    latency_ms: list[float] = []
    pred_output_error_count = 0
    pred_output_error_reasons: Counter[str] = Counter()

    pred_path = out_dir / "predictions.jsonl"
    pred_csv_path = out_dir / "predictions.csv"
    trace_path = out_dir / "traces" / "events.jsonl"
    event_id = 0

    try:
        from tqdm import tqdm

        desc = "clips" if takeover_clips_dir is not None else "snapshots"
        loader_iter = tqdm(loader, desc=desc, total=len(loader))
    except ImportError:
        loader_iter = loader

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
        "parse_note",
        "model_output_error",
    ]

    with (
        _open_jsonl(pred_path) as pred_f,
        _open_csv(pred_csv_path) as pred_csv_f,
        _open_jsonl(trace_path) as trace_f,
    ):
        csv_writer = csv.DictWriter(pred_csv_f, fieldnames=csv_cols, extrasaction="ignore")
        csv_writer.writeheader()
        for batch in loader_iter:
            for item in batch:
                assert isinstance(item, SnapshotItem)
                prep = item.prep
                total_snapshots += 1

                if not prep.ok:
                    skipped_reasons[str(prep.skip_reason or "prep_failed")] += 1
                    continue

                ep_id = str(prep.episode_id)
                t = float(prep.t)
                snap_idx = int(prep.snapshot_index)
                label = prep.label
                label_err = prep.label_err
                score_idx = prep.score_idx

                if prep.auto_enabled is True:
                    auto_true += 1
                elif prep.auto_enabled is False:
                    auto_false += 1
                else:
                    auto_missing += 1

                if (
                    str(args.model)
                    in ("gemini_genai", *HIERARCHICAL_MODELS, "openai_chat_completions")
                    and not prep.prompt_overlay_frame_ref
                ):
                    skipped_reasons["missing_overlay_for_vlm"] += 1
                    continue

                obs = prep.obs or {"episode_id": ep_id, "t": float(t)}

                event_id += 1
                trace_f.write(
                    json.dumps(
                        {
                            "event_type": "obs",
                            "event_id": event_id,
                            "time_utc": utc_now_iso(),
                            "task": benchmark_name,
                            "episode_id": ep_id,
                            "t": float(t),
                            "snapshot_index": int(snap_idx),
                            "observation": obs,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

                query_bundle: dict[str, Any] = {
                    "task": benchmark_name,
                    "episode_id": ep_id,
                    "t": float(t),
                    "snapshot_index": int(snap_idx),
                    "overlay_frame_ref": prep.prompt_overlay_frame_ref,
                    "overlay_frame_ref_report": prep.overlay_frame_ref,
                    "history_frame_refs": prep.history_frame_refs,
                    "num_candidates": int(prep.num_candidates),
                    "goal_text": None,
                    "goal_xy": prep.goal_xy,
                    "goal_distance_m": prep.goal_distance_m,
                    "goal_bearing_deg": prep.goal_bearing_deg,
                    "candidate_scores": prep.candidate_confidence,
                    "image_info": {
                        "overlay_traj_style": str(args.overlay_traj_style),
                        "robot_width_m": float(args.overlay_robot_width_m),
                        "goal_direction_arrow": bool(args.overlay_goal_direction_arrow),
                        "goal_projection_marker": str(args.overlay_goal_projection_marker),
                    },
                    "obs": obs,
                }
                # Hierarchical selector needs base RGB + full candidate point arrays for
                # re-rendering custom overlays.
                # Keep these out of traces/reports by namespacing them under "__*".
                if str(args.model) in HIERARCHICAL_MODELS:
                    query_bundle["__rgb_frame_ref"] = prep.rgb_frame_ref
                    query_bundle["__candidates_points_xy"] = prep.candidates_points_xy

                messages: list[dict[str, Any]] | None = None
                raw: str | None = None
                thoughts: list[str] | None = None
                pred_err: str | None = None
                advice: dict[str, Any] | None = None
                provider_meta: dict[str, Any] | None = None

                if str(args.model) in ("oracle_min_ade", "oracle_min_ade_all"):
                    oracle_idx = None
                    if isinstance(prep.ade, dict):
                        if str(args.model) == "oracle_min_ade_all":
                            oracle_idx = prep.ade.get("min_all_idx")
                        else:
                            oracle_idx = prep.ade.get("min_idx")
                    if oracle_idx is None and isinstance(prep.ade, dict):
                        oracle_idx = prep.ade.get("score_idx")
                    if oracle_idx is None:
                        oracle_idx = score_idx
                    if oracle_idx is None:
                        oracle_idx = 0
                    advice = {
                        "type": "select_trajectory",
                        "selected_index": int(oracle_idx),
                        "confidence": 1.0,
                        "debug": {
                            "rationale": (
                                "oracle_min_ade_all (global-min ADE over all candidates; "
                                "falls back to score argmax/0 if unavailable)"
                                if str(args.model) == "oracle_min_ade_all"
                                else "oracle_min_ade (visible-min ADE; falls back to score "
                                "argmax/0 if unavailable)"
                            )
                        },
                    }
                    raw = json.dumps(
                        {"action": "select_trajectory", "selected_index": int(oracle_idx)}
                    )
                else:
                    if advisor is None:
                        raise RuntimeError("advisor was not initialized (unexpected)")
                    try:
                        advice = advisor.advise(query_bundle)
                    except Exception as e:
                        # This benchmark does not support ask_for_help; fall back to stop.
                        # We also catch RuntimeError from exhausted retries here.
                        logger.error(f"Model failed for {ep_id} t={t} snap={snap_idx}: {e}")
                        advice = {
                            "type": "stop",
                            "confidence": 0.0,
                            "debug": {"rationale": f"model_exception:{e}"},
                        }
                    raw = getattr(advisor, "last_raw_response", None)
                    messages = getattr(advisor, "last_messages", None)
                    thoughts = getattr(advisor, "last_thoughts", None)
                    provider_meta = getattr(advisor, "last_provider_meta", None)

                if isinstance(advice, dict) and isinstance(advice.get("error"), str):
                    pred_err = str(advice.get("error"))

                event_id += 1
                trace_f.write(
                    json.dumps(
                        {
                            "event_type": "model_call",
                            "event_id": event_id,
                            "time_utc": utc_now_iso(),
                            "task": benchmark_name,
                            "episode_id": ep_id,
                            "t": float(t),
                            "snapshot_index": int(snap_idx),
                            "model": {
                                "adapter": str(args.model),
                                "model_id": str(model_id),
                                "prompt_version": str(prompt_version),
                                "decoding_config": decoding_config,
                            },
                            "query_bundle": {
                                k: v for k, v in query_bundle.items() if not str(k).startswith("__")
                            },
                            "messages": messages,
                            "raw_response": raw,
                            "thoughts": thoughts,
                            "parsed_advice": advice,
                            "provider_meta": provider_meta,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

                pred_action = advice.get("type") if isinstance(advice, dict) else None
                pred_idx = None
                if isinstance(advice, dict) and pred_action == "select_trajectory":
                    try:
                        pred_idx = int(advice.get("selected_index"))
                    except Exception:
                        pred_idx = None
                        pred_err = pred_err or "selected_index_not_int"

                pred_rationale: str | None = None
                parse_note: str | None = None
                if isinstance(advice, dict):
                    dbg = advice.get("debug")
                    if isinstance(dbg, dict):
                        pn = dbg.get("parse_note")
                        if isinstance(pn, str) and pn.strip():
                            parse_note = pn.strip()
                        rat2 = dbg.get("rationale")
                        if isinstance(rat2, str) and rat2.strip():
                            pred_rationale = rat2.strip()

                # Distinguish "real stop" vs "we stopped because the model failed".
                # This run previously under-reported errors because failures were folded into stop.
                model_output_error: str | None = None
                try:
                    if pred_err:
                        model_output_error = f"pred:{pred_err}"
                    elif isinstance(pred_rationale, str) and pred_rationale.startswith(
                        "model_exception:"
                    ):
                        model_output_error = pred_rationale
                    elif isinstance(parse_note, str) and parse_note in (
                        "empty_response",
                        "parse_failed",
                    ):
                        model_output_error = f"parse_note:{parse_note}"
                    elif raw is None:
                        model_output_error = "raw_response:null"
                    elif isinstance(raw, str) and (
                        raw.strip() == "" or raw.strip().lower() == "none"
                    ):
                        model_output_error = "raw_response:empty"
                except Exception:
                    model_output_error = model_output_error or None

                skip_reason = None
                if label_err:
                    skip_reason = f"label:{label_err}"
                elif pred_err:
                    skip_reason = f"pred:{pred_err}"

                is_eval = skip_reason is None and label is not None and advice is not None
                if not is_eval:
                    is_correct = None
                elif pred_action == "select_trajectory" and pred_idx is not None:
                    is_correct = int(pred_idx) == int(label)
                else:
                    is_correct = False

                # Model output health (counted over evaluated samples).
                if is_eval:
                    if model_output_error:
                        pred_output_error_count += 1
                        pred_output_error_reasons[str(model_output_error)] += 1
                        if pred_action == "stop":
                            stop_error_count += 1
                    elif pred_action == "stop":
                        stop_count += 1
                    if pred_action == "select_trajectory":
                        try:
                            if pred_idx is not None and not (
                                0 <= int(pred_idx) < int(prep.num_candidates)
                            ):
                                invalid_index_count += 1
                        except Exception:
                            invalid_index_count += 1

                # Goal-relative trajectory metrics for the *selected* trajectory.
                # These are evaluator-side diagnostics of the selector (independent of ADE/oracle
                # labels).
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
                min_fde_idx = None
                min_fde_all_m = None
                min_fde_all_idx = None
                min_moe_deg = None
                min_moe_idx = None
                min_moe_all_deg = None
                min_moe_all_idx = None
                mean_moe_all_deg = None
                ap_fde_2m = None
                traj_len_m = None
                maoe_deg = None
                dcr = None
                tcr = None
                compliance_source = None
                try:
                    # Select trajectory points:
                    # - select_trajectory: use candidate points
                    # - stop: treat as a static trajectory at (0,0) in robot frame
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
                        # Length: match GT length if available, else match candidate[0] length if
                        # available, else 1.
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
                        # Goal-relative metrics.
                        if isinstance(prep.goal_xy, list) and len(prep.goal_xy) >= 2:
                            goal_xy = np.asarray(
                                [float(prep.goal_xy[0]), float(prep.goal_xy[1])], dtype=np.float64
                            )
                        if pts0.ndim == 2 and pts0.shape[0] > 0 and pts0.shape[1] >= 2:
                            end_xy = pts0[-1, :2]
                            d_end = float(np.linalg.norm(end_xy - goal_xy))
                            d_all = np.linalg.norm(pts0[:, :2] - goal_xy.reshape(1, 2), axis=1)
                            d_avg = float(np.mean(d_all)) if d_all.size > 0 else None

                            # Angular difference between the endpoint direction and the goal
                            # direction (robot frame).
                            goal_ang = float(
                                math.degrees(math.atan2(float(goal_xy[1]), float(goal_xy[0])))
                            )
                            end_ang = float(
                                math.degrees(math.atan2(float(end_xy[1]), float(end_xy[0])))
                            )
                            # Wrap to [-180, 180] then take abs.
                            diff = (end_ang - goal_ang + 180.0) % 360.0 - 180.0
                            ang_diff = abs(float(diff))

                            selected_end_dist_to_goal_m = d_end if math.isfinite(d_end) else None
                            selected_traj_avg_dist_to_goal_m = (
                                d_avg if (d_avg is not None and math.isfinite(d_avg)) else None
                            )
                            selected_goal_ang_diff_deg = (
                                ang_diff if math.isfinite(ang_diff) else None
                            )

                            # Progress toward goal (positive means the trajectory endpoint is
                            # closer to goal than start).
                            if prep.goal_distance_m is not None:
                                gd0 = float(prep.goal_distance_m)
                                if math.isfinite(gd0) and selected_end_dist_to_goal_m is not None:
                                    selected_goal_progress_m = float(gd0) - float(
                                        selected_end_dist_to_goal_m
                                    )

                        # Path length (diagnostic).
                        traj_len_m = _traj_path_length_m(pts0[:, :2])

                        # ADE for selected trajectory.
                        if isinstance(prep.gt_local_traj_xy, list) and prep.gt_local_traj_xy:
                            gt = np.asarray(prep.gt_local_traj_xy, dtype=np.float64)
                            sel = np.asarray(pts0[:, :2], dtype=np.float64)
                            # Align lengths (compute_ade does this too, but we keep it local).
                            n = min(len(gt), len(sel))
                            if n > 0:
                                errs = np.linalg.norm(sel[:n] - gt[:n], axis=1)
                                selected_ade = float(np.mean(errs))
                            # Prefix ADEs.
                            selected_ade_0_5s = _ade_prefix_seconds(
                                sel, gt, dt_s=float(args.traj_dt_s), seconds=0.5
                            )
                            selected_ade_1_0s = _ade_prefix_seconds(
                                sel, gt, dt_s=float(args.traj_dt_s), seconds=1.0
                            )
                            selected_ade_2_0s = _ade_prefix_seconds(
                                sel, gt, dt_s=float(args.traj_dt_s), seconds=2.0
                            )

                            # FDE / MOE (rss-style) for the selected trajectory.
                            selected_fde_m = _fde_m(sel, gt)
                            selected_moe_deg = _moe_deg_posvec(sel, gt)

                            # MAOE vs executed GT future trajectory (open-loop).
                            maoe_deg = _maoe_deg(sel, gt)

                            # DCR/TCR: treat a corridor around GT as "socially compliant region".
                            dcr, tcr = _dcr_tcr_from_corridor(
                                pred_xy=sel,
                                compliant_polyline_xy=gt,
                                dt_s=float(args.traj_dt_s),
                                corridor_radius_m=1.0,
                            )
                            compliance_source = "gt_corridor"

                            # Candidate-level FDE/MOE baselines (min/min_all/score) and AP@2m.
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

                            # Score baseline (argmax score index from the dataset prep).
                            score_fde_masked_laststep_m = None
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

                            # Min over all candidates.
                            try:
                                if cand_fdes is not None:
                                    best_v = None
                                    best_i = None
                                    for i, v in enumerate(cand_fdes):
                                        if v is None:
                                            continue
                                        vf = float(v)
                                        if not math.isfinite(vf):
                                            continue
                                        if best_v is None or vf < float(best_v):
                                            best_v = vf
                                            best_i = int(i)
                                    min_fde_all_m = best_v
                                    min_fde_all_idx = best_i
                            except Exception:
                                pass
                            try:
                                if cand_moes is not None:
                                    best_v = None
                                    best_i = None
                                    vals = []
                                    for i, v in enumerate(cand_moes):
                                        if v is None:
                                            continue
                                        vf = float(v)
                                        if not math.isfinite(vf):
                                            continue
                                        vals.append(vf)
                                        if best_v is None or vf < float(best_v):
                                            best_v = vf
                                            best_i = int(i)
                                    min_moe_all_deg = best_v
                                    min_moe_all_idx = best_i
                                    mean_moe_all_deg = (
                                        float(np.mean(np.asarray(vals, dtype=np.float64)))
                                        if vals
                                        else None
                                    )
                            except Exception:
                                pass

                            # Min over "visible" candidate set (if available from ADE prep; else
                            # None).
                            try:
                                vis = (
                                    prep.ade.get("visible_indices")
                                    if isinstance(prep.ade, dict)
                                    and isinstance(prep.ade.get("visible_indices"), list)
                                    else None
                                )
                                if vis and cand_fdes is not None and cand_moes is not None:
                                    best_f = None
                                    best_fi = None
                                    best_m = None
                                    best_mi = None
                                    for ii in vis:
                                        try:
                                            i = int(ii)
                                        except Exception:
                                            continue
                                        if not (0 <= i < len(cand_fdes)):
                                            continue
                                        vf = cand_fdes[i]
                                        if vf is not None:
                                            f = float(vf)
                                            if math.isfinite(f) and (
                                                best_f is None or f < float(best_f)
                                            ):
                                                best_f = f
                                                best_fi = int(i)
                                        vm = cand_moes[i]
                                        if vm is not None:
                                            m0 = float(vm)
                                            if math.isfinite(m0) and (
                                                best_m is None or m0 < float(best_m)
                                            ):
                                                best_m = m0
                                                best_mi = int(i)
                                    min_fde_m = best_f
                                    min_fde_idx = best_fi
                                    min_moe_deg = best_m
                                    min_moe_idx = best_mi
                            except Exception:
                                pass

                            # AP over candidates (scores vs FDE<thr).
                            try:
                                if cand_fdes is not None and isinstance(
                                    prep.candidate_scores_raw, list
                                ):
                                    # Match `rss/eval_point_goal.py`: compute AP on the post-NMS
                                    # pool
                                    # (endpoint-distance NMS, then pad up to max_trajectories).
                                    # This makes AP comparable to scripts that compute metrics
                                    # after NMS.
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
                                                    max_trajectories=int(args.nms_max_trajectories),
                                                    distance_threshold=float(
                                                        args.nms_distance_threshold
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

                            # Baseline prefix ADEs (score/min) for comparison (only if indices
                            # available).
                            try:
                                if score_idx is not None and isinstance(
                                    prep.candidates_points_xy, list
                                ):
                                    sc_xy = np.asarray(
                                        prep.candidates_points_xy[int(score_idx)], dtype=np.float64
                                    )
                                    score_ade_0_5s = _ade_prefix_seconds(
                                        sc_xy, gt, dt_s=float(args.traj_dt_s), seconds=0.5
                                    )
                                    score_ade_1_0s = _ade_prefix_seconds(
                                        sc_xy, gt, dt_s=float(args.traj_dt_s), seconds=1.0
                                    )
                                    score_ade_2_0s = _ade_prefix_seconds(
                                        sc_xy, gt, dt_s=float(args.traj_dt_s), seconds=2.0
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
                                            mi_xy, gt, dt_s=float(args.traj_dt_s), seconds=0.5
                                        )
                                        min_ade_1_0s = _ade_prefix_seconds(
                                            mi_xy, gt, dt_s=float(args.traj_dt_s), seconds=1.0
                                        )
                                        min_ade_2_0s = _ade_prefix_seconds(
                                            mi_xy, gt, dt_s=float(args.traj_dt_s), seconds=2.0
                                        )
                            except Exception:
                                pass
                except Exception:
                    selected_end_dist_to_goal_m = None
                    selected_goal_ang_diff_deg = None
                    selected_traj_avg_dist_to_goal_m = None
                    selected_goal_progress_m = None
                    selected_ade = None
                    selected_ade_0_5s = None
                    selected_ade_1_0s = None
                    selected_ade_2_0s = None
                    score_ade_0_5s = None
                    score_ade_1_0s = None
                    score_ade_2_0s = None
                    min_ade_0_5s = None
                    min_ade_1_0s = None
                    min_ade_2_0s = None
                    traj_len_m = None
                    maoe_deg = None
                    dcr = None
                    tcr = None
                    compliance_source = None

                if not is_eval:
                    skipped_reasons[skip_reason or "unknown_skip"] += 1
                else:
                    evaluated += 1
                    if is_correct:
                        correct += 1
                    # Token accounting (best-effort, VLM models only).
                    if isinstance(provider_meta, dict) and isinstance(
                        provider_meta.get("usage"), dict
                    ):
                        u = provider_meta.get("usage") or {}
                        try:
                            pt = u.get("prompt_tokens")
                            cpt = u.get("cached_prompt_tokens")
                            ot = u.get("output_tokens")
                            th = u.get("thoughts_tokens")
                            tt = u.get("total_tokens")
                            calls = u.get("calls", 1)
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
                            model_calls += int(calls) if calls is not None else 1
                        except Exception:
                            pass
                    # Latency accounting (best-effort, VLM models only).
                    try:
                        if (
                            isinstance(provider_meta, dict)
                            and provider_meta.get("latency_ms") is not None
                        ):
                            lm = float(provider_meta.get("latency_ms"))
                            if math.isfinite(lm) and lm >= 0:
                                latency_ms.append(lm)
                    except Exception:
                        pass
                    try:
                        gd = prep.goal_distance_m
                        if gd is not None:
                            gdf = float(gd)
                            if math.isfinite(gdf):
                                goal_dist_sum += gdf
                                goal_dist_count += 1
                    except Exception:
                        pass
                    # Aggregates for goal-relative selected-trajectory metrics.
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

                    # FDE / MOE / mAP aggregates (best-effort; only when GT is available).
                    try:
                        if selected_fde_m is not None and math.isfinite(float(selected_fde_m)):
                            fde_model_sum += float(selected_fde_m)
                            fde_count += 1
                        if min_fde_m is not None and math.isfinite(float(min_fde_m)):
                            fde_min_sum += float(min_fde_m)
                            fde_min_count += 1
                        if min_fde_all_m is not None and math.isfinite(float(min_fde_all_m)):
                            fde_min_all_sum += float(min_fde_all_m)
                            fde_min_all_count += 1
                        if score_fde_m is not None and math.isfinite(float(score_fde_m)):
                            fde_score_sum += float(score_fde_m)
                            fde_score_count += 1
                        if score_fde_masked_laststep_m is not None and math.isfinite(
                            float(score_fde_masked_laststep_m)
                        ):
                            fde_score_masked_laststep_sum += float(score_fde_masked_laststep_m)
                            fde_score_masked_laststep_count += 1

                        if selected_moe_deg is not None and math.isfinite(float(selected_moe_deg)):
                            moe_deg_model_sum += float(selected_moe_deg)
                            moe_deg_count += 1
                        if min_moe_deg is not None and math.isfinite(float(min_moe_deg)):
                            moe_deg_min_sum += float(min_moe_deg)
                            moe_deg_min_count += 1
                        if min_moe_all_deg is not None and math.isfinite(float(min_moe_all_deg)):
                            moe_deg_min_all_sum += float(min_moe_all_deg)
                            moe_deg_min_all_count += 1
                        if score_moe_deg is not None and math.isfinite(float(score_moe_deg)):
                            moe_deg_score_sum += float(score_moe_deg)
                            moe_deg_score_count += 1
                        if mean_moe_all_deg is not None and math.isfinite(float(mean_moe_all_deg)):
                            moe_deg_mean_all_sum += float(mean_moe_all_deg)
                            moe_deg_mean_all_count += 1

                        if ap_fde_2m is not None and math.isfinite(float(ap_fde_2m)):
                            ap_fde_2m_sum += float(ap_fde_2m)
                            ap_fde_2m_count += 1
                    except Exception:
                        pass
                    if (
                        pred_idx is not None
                        and score_idx is not None
                        and 0 <= int(pred_idx) < int(prep.num_candidates)
                    ):
                        score_agree_count += 1
                        if int(pred_idx) == int(score_idx):
                            correct_vs_score += 1

                    if isinstance(prep.ade, dict) and pred_idx is not None:
                        ades = prep.ade.get("ades")
                        if isinstance(ades, list) and 0 <= int(pred_idx) < len(ades):
                            ade_count += 1
                            ade_min_sum += (
                                float(prep.ade.get("min"))
                                if prep.ade.get("min") is not None
                                else 0.0
                            )
                            ade_min_all_sum += (
                                float(prep.ade.get("min_all"))
                                if prep.ade.get("min_all") is not None
                                else 0.0
                            )
                            ade_score_sum += (
                                float(prep.ade.get("score"))
                                if prep.ade.get("score") is not None
                                else 0.0
                            )
                            ade_model_sum += float(ades[int(pred_idx)])
                            if int(pred_idx) == int(prep.ade.get("min_idx")):
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
                        # Stop trajectory: compute ADE against GT directly (not a candidate index).
                        ade_count += 1
                        ade_min_sum += (
                            float(prep.ade.get("min")) if prep.ade.get("min") is not None else 0.0
                        )
                        ade_min_all_sum += (
                            float(prep.ade.get("min_all"))
                            if prep.ade.get("min_all") is not None
                            else 0.0
                        )
                        ade_score_sum += (
                            float(prep.ade.get("score"))
                            if prep.ade.get("score") is not None
                            else 0.0
                        )
                        ade_model_sum += float(selected_ade)

                    # Prefix ADE aggregates (best-effort; based on computed prefix values).
                    try:
                        if selected_ade_0_5s is not None and math.isfinite(
                            float(selected_ade_0_5s)
                        ):
                            ade_model_0_5s_sum += float(selected_ade_0_5s)
                            ade_model_0_5s_count += 1
                        if selected_ade_1_0s is not None and math.isfinite(
                            float(selected_ade_1_0s)
                        ):
                            ade_model_1_0s_sum += float(selected_ade_1_0s)
                            ade_model_1_0s_count += 1
                        if selected_ade_2_0s is not None and math.isfinite(
                            float(selected_ade_2_0s)
                        ):
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

                    if isinstance(prep.route_dev, dict) and pred_idx is not None:
                        devs = prep.route_dev.get("route_devs")
                        if isinstance(devs, list) and 0 <= int(pred_idx) < len(devs):
                            route_dev_count += 1
                            route_dev_min_sum += (
                                float(prep.route_dev.get("min"))
                                if prep.route_dev.get("min") is not None
                                else 0.0
                            )
                            route_dev_score_sum += (
                                float(prep.route_dev.get("score"))
                                if prep.route_dev.get("score") is not None
                                else 0.0
                            )
                            route_dev_model_sum += float(devs[int(pred_idx)])
                            if int(pred_idx) == int(prep.route_dev.get("min_idx")):
                                correct_vs_route_min += 1
                            if int(pred_idx) == int(prep.route_dev.get("score_idx")):
                                correct_vs_route_score += 1

                if evaluated > 0 and (evaluated % 200 == 0):
                    _log_progress(
                        f"[progress] evaluated={evaluated} total={total_snapshots} "
                        f"acc={float(correct) / float(evaluated):.4f} "
                        f"skipped={total_snapshots - evaluated}"
                    )

                event_id += 1
                trace_f.write(
                    json.dumps(
                        {
                            "event_type": "action",
                            "event_id": event_id,
                            "time_utc": utc_now_iso(),
                            "task": benchmark_name,
                            "episode_id": ep_id,
                            "t": float(t),
                            "snapshot_index": int(snap_idx),
                            "action": {
                                "type": to_jsonable(pred_action),
                                "framework_action": to_jsonable(pred_action),
                                "selected_index": to_jsonable(pred_idx),
                                "rationale": pred_rationale,
                                "parse_error": pred_err,
                                "parse_note": parse_note,
                                "model_output_error": model_output_error,
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

                event_id += 1
                trace_f.write(
                    json.dumps(
                        {
                            "event_type": "metric",
                            "event_id": event_id,
                            "time_utc": utc_now_iso(),
                            "task": benchmark_name,
                            "episode_id": ep_id,
                            "t": float(t),
                            "snapshot_index": int(snap_idx),
                            "metric": {
                                "correct": is_correct,
                                "skipped": (not is_eval),
                                "skip_reason": skip_reason,
                                "label": to_jsonable(label),
                                "prediction": to_jsonable(pred_idx),
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

                rec = {
                    "task": benchmark_name,
                    "episode_id": ep_id,
                    "t": float(t),
                    "snapshot_index": int(snap_idx),
                    "num_candidates": int(prep.num_candidates),
                    "goal_xy": to_jsonable(prep.goal_xy),
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
                    "ap_fde_2m": to_jsonable(ap_fde_2m),
                    "fde": {
                        "selected": to_jsonable(selected_fde_m),
                        "min": to_jsonable(min_fde_m),
                        "min_all": to_jsonable(min_fde_all_m),
                        "score": to_jsonable(score_fde_m),
                        "score_masked_laststep": to_jsonable(score_fde_masked_laststep_m),
                        "min_idx": to_jsonable(min_fde_idx),
                        "min_all_idx": to_jsonable(min_fde_all_idx),
                        "score_idx": to_jsonable(score_idx),
                    },
                    "moe": {
                        "selected": to_jsonable(selected_moe_deg),
                        "min": to_jsonable(min_moe_deg),
                        "min_all": to_jsonable(min_moe_all_deg),
                        "score": to_jsonable(score_moe_deg),
                        "mean_all": to_jsonable(mean_moe_all_deg),
                        "min_idx": to_jsonable(min_moe_idx),
                        "min_all_idx": to_jsonable(min_moe_all_idx),
                        "score_idx": to_jsonable(score_idx),
                    },
                    "prediction": to_jsonable(pred_idx),
                    "prediction_action": to_jsonable(pred_action),
                    "rationale": pred_rationale,
                    "parse_note": parse_note,
                    "model_output_error": model_output_error,
                    "label": to_jsonable(label),
                    "correct": is_correct,
                    "skipped": (not is_eval),
                    "skip_reason": skip_reason,
                    "overlay_frame_ref": prep.overlay_frame_ref,
                    "overlay_error": prep.overlay_error,
                    "local_plot_frame_ref": None,
                    "local_plot_error": None,
                    "auto_enabled": prep.auto_enabled,
                    "ade": {
                        "min": to_jsonable(
                            prep.ade.get("min") if isinstance(prep.ade, dict) else None
                        ),
                        "min_all": to_jsonable(
                            prep.ade.get("min_all") if isinstance(prep.ade, dict) else None
                        ),
                        "score": to_jsonable(
                            prep.ade.get("score") if isinstance(prep.ade, dict) else None
                        ),
                        "selected": to_jsonable(
                            selected_ade
                            if pred_action == "stop"
                            else (prep.ade.get("ades") or [])[pred_idx]
                            if (
                                isinstance(prep.ade, dict)
                                and pred_idx is not None
                                and isinstance(prep.ade.get("ades"), list)
                                and 0 <= int(pred_idx) < len(prep.ade.get("ades"))
                            )
                            else None
                        ),
                        "selected_0_5s": to_jsonable(selected_ade_0_5s),
                        "selected_1_0s": to_jsonable(selected_ade_1_0s),
                        "selected_2_0s": to_jsonable(selected_ade_2_0s),
                        "score_0_5s": to_jsonable(score_ade_0_5s),
                        "score_1_0s": to_jsonable(score_ade_1_0s),
                        "score_2_0s": to_jsonable(score_ade_2_0s),
                        "min_0_5s": to_jsonable(min_ade_0_5s),
                        "min_1_0s": to_jsonable(min_ade_1_0s),
                        "min_2_0s": to_jsonable(min_ade_2_0s),
                        "min_idx": to_jsonable(
                            prep.ade.get("min_idx") if isinstance(prep.ade, dict) else None
                        ),
                        "min_all_idx": to_jsonable(
                            prep.ade.get("min_all_idx") if isinstance(prep.ade, dict) else None
                        ),
                        "score_idx": to_jsonable(
                            prep.ade.get("score_idx") if isinstance(prep.ade, dict) else None
                        ),
                        "visible_indices": to_jsonable(
                            prep.ade.get("visible_indices") if isinstance(prep.ade, dict) else None
                        ),
                    },
                    "route_dev": {
                        "min": to_jsonable(
                            prep.route_dev.get("min") if isinstance(prep.route_dev, dict) else None
                        ),
                        "score": to_jsonable(
                            prep.route_dev.get("score")
                            if isinstance(prep.route_dev, dict)
                            else None
                        ),
                        "selected": to_jsonable(
                            (prep.route_dev.get("route_devs") or [])[pred_idx]
                            if (
                                isinstance(prep.route_dev, dict)
                                and pred_idx is not None
                                and isinstance(prep.route_dev.get("route_devs"), list)
                                and 0 <= int(pred_idx) < len(prep.route_dev.get("route_devs"))
                            )
                            else None
                        ),
                        "min_idx": to_jsonable(
                            prep.route_dev.get("min_idx")
                            if isinstance(prep.route_dev, dict)
                            else None
                        ),
                        "score_idx": to_jsonable(
                            prep.route_dev.get("score_idx")
                            if isinstance(prep.route_dev, dict)
                            else None
                        ),
                    },
                    "token_usage": to_jsonable(
                        provider_meta.get("usage") if isinstance(provider_meta, dict) else None
                    ),
                }

                # Per-snapshot local-frame plot (selected vs GT vs route) (optional).
                if bool(getattr(args, "write_local_plots", False)):
                    try:
                        from slow_brain_fast_planner.benchmarks.trajectory_selection_plots import (
                            write_local_frame_plot,
                        )

                        selected_traj = None
                        all_cands = None
                        if (
                            pred_idx is not None
                            and isinstance(prep.candidates_points_xy, list)
                            and 0 <= int(pred_idx) < len(prep.candidates_points_xy)
                        ):
                            try:
                                selected_traj = np.asarray(
                                    prep.candidates_points_xy[int(pred_idx)], dtype=np.float64
                                )
                            except Exception:
                                selected_traj = None
                        elif pred_action == "stop":
                            try:
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
                                    n_pts = int(len(prep.candidates_points_xy[0]))
                                selected_traj = np.zeros((max(1, n_pts), 2), dtype=np.float64)
                            except Exception:
                                selected_traj = None
                        if (
                            isinstance(prep.candidates_points_xy, list)
                            and prep.candidates_points_xy
                        ):
                            try:
                                all_cands = [
                                    np.asarray(x, dtype=np.float64)
                                    for x in prep.candidates_points_xy
                                ]
                            except Exception:
                                all_cands = None

                        gt_traj = (
                            np.asarray(prep.gt_local_traj_xy, dtype=np.float64)
                            if isinstance(prep.gt_local_traj_xy, list)
                            else None
                        )
                        route_xy = (
                            np.asarray(prep.route_local_xy, dtype=np.float64)
                            if isinstance(prep.route_local_xy, list)
                            else None
                        )

                        if selected_traj is not None or gt_traj is not None or route_xy is not None:
                            t_str = f"{float(t):.3f}".rstrip("0").rstrip(".")
                            local_plot_path = (
                                out_dir / "artifacts" / "local_plots" / str(ep_id) / f"{t_str}.png"
                            ).resolve()
                            write_local_frame_plot(
                                out_path=local_plot_path,
                                episode_id=str(ep_id),
                                t=float(t),
                                selected_traj_xy=selected_traj,
                                gt_human_traj_xy=gt_traj,
                                gt_route_xy=route_xy,
                                all_candidates_xy=all_cands,
                                candidate_scores=None,
                                selected_index=(int(pred_idx) if pred_idx is not None else None),
                                label_index=(int(label) if label is not None else None),
                                score_index=(
                                    int(prep.ade.get("score_idx"))
                                    if isinstance(prep.ade, dict)
                                    and prep.ade.get("score_idx") is not None
                                    else None
                                ),
                                min_ade_index=(
                                    int(prep.ade.get("min_idx"))
                                    if isinstance(prep.ade, dict)
                                    and prep.ade.get("min_idx") is not None
                                    else None
                                ),
                                min_ade_all_index=(
                                    int(prep.ade.get("min_all_idx"))
                                    if isinstance(prep.ade, dict)
                                    and prep.ade.get("min_all_idx") is not None
                                    else None
                                ),
                                goal_xy=(
                                    np.asarray(prep.goal_xy, dtype=np.float64)
                                    if isinstance(prep.goal_xy, list) and len(prep.goal_xy) >= 2
                                    else None
                                ),
                                title_suffix=None,
                            )
                            rec["local_plot_frame_ref"] = str(local_plot_path.relative_to(out_dir))
                    except Exception as e:
                        rec["local_plot_error"] = f"local_plot_exception:{e}"

                pred_f.write(json.dumps(rec, sort_keys=True) + "\n")
                # Also write a flat CSV row for quick copy/paste.
                try:
                    gx = gy = None
                    if isinstance(prep.goal_xy, list) and len(prep.goal_xy) >= 2:
                        gx = prep.goal_xy[0]
                        gy = prep.goal_xy[1]
                    ade_sel = ade_min = ade_min_all = ade_score = None
                    ade_sel_0_5s = ade_sel_1_0s = ade_sel_2_0s = None
                    ade_min_0_5s = ade_min_1_0s = ade_min_2_0s = None
                    ade_score_0_5s = ade_score_1_0s = ade_score_2_0s = None
                    if isinstance(rec.get("ade"), dict):
                        ade_sel = rec["ade"].get("selected")
                        ade_min = rec["ade"].get("min")
                        ade_min_all = rec["ade"].get("min_all")
                        ade_score = rec["ade"].get("score")
                        ade_sel_0_5s = rec["ade"].get("selected_0_5s")
                        ade_sel_1_0s = rec["ade"].get("selected_1_0s")
                        ade_sel_2_0s = rec["ade"].get("selected_2_0s")
                        ade_min_0_5s = rec["ade"].get("min_0_5s")
                        ade_min_1_0s = rec["ade"].get("min_1_0s")
                        ade_min_2_0s = rec["ade"].get("min_2_0s")
                        ade_score_0_5s = rec["ade"].get("score_0_5s")
                        ade_score_1_0s = rec["ade"].get("score_1_0s")
                        ade_score_2_0s = rec["ade"].get("score_2_0s")

                    fde_sel = fde_min2 = fde_min_all2 = fde_score2 = None
                    fde_score_masked_laststep2 = None
                    if isinstance(rec.get("fde"), dict):
                        fde_sel = rec["fde"].get("selected")
                        fde_min2 = rec["fde"].get("min")
                        fde_min_all2 = rec["fde"].get("min_all")
                        fde_score2 = rec["fde"].get("score")
                        fde_score_masked_laststep2 = rec["fde"].get("score_masked_laststep")

                    moe_sel = moe_min2 = moe_min_all2 = moe_score2 = moe_mean_all2 = None
                    if isinstance(rec.get("moe"), dict):
                        moe_sel = rec["moe"].get("selected")
                        moe_min2 = rec["moe"].get("min")
                        moe_min_all2 = rec["moe"].get("min_all")
                        moe_score2 = rec["moe"].get("score")
                        moe_mean_all2 = rec["moe"].get("mean_all")
                    ap_fde_2m2 = rec.get("ap_fde_2m")
                    dev_sel = dev_min = dev_score = None
                    if isinstance(rec.get("route_dev"), dict):
                        dev_sel = rec["route_dev"].get("selected")
                        dev_min = rec["route_dev"].get("min")
                        dev_score = rec["route_dev"].get("score")

                    csv_writer.writerow(
                        {
                            "episode_id": ep_id,
                            "t": float(t),
                            "snapshot_index": int(snap_idx),
                            "prediction": rec.get("prediction"),
                            "label": rec.get("label"),
                            "correct": rec.get("correct"),
                            "skipped": rec.get("skipped"),
                            "skip_reason": rec.get("skip_reason"),
                            "num_candidates": rec.get("num_candidates"),
                            "goal_x": gx,
                            "goal_y": gy,
                            "goal_distance_m": rec.get("goal_distance_m"),
                            "goal_bearing_deg": rec.get("goal_bearing_deg"),
                            "selected_end_dist_to_goal_m": rec.get("selected_end_dist_to_goal_m"),
                            "selected_goal_ang_diff_deg": rec.get("selected_goal_ang_diff_deg"),
                            "selected_traj_avg_dist_to_goal_m": rec.get(
                                "selected_traj_avg_dist_to_goal_m"
                            ),
                            "selected_goal_progress_m": rec.get("selected_goal_progress_m"),
                            "traj_len_m": rec.get("traj_len_m"),
                            "maoe_deg": rec.get("maoe_deg"),
                            "dcr": rec.get("dcr"),
                            "tcr": rec.get("tcr"),
                            "compliance_source": rec.get("compliance_source"),
                            "fde_selected": fde_sel,
                            "fde_min": fde_min2,
                            "fde_min_all": fde_min_all2,
                            "fde_score": fde_score2,
                            "fde_score_masked_laststep": fde_score_masked_laststep2,
                            "moe_deg": moe_sel,
                            "moe_deg_min": moe_min2,
                            "moe_deg_min_all": moe_min_all2,
                            "moe_deg_score": moe_score2,
                            "moe_deg_mean_all": moe_mean_all2,
                            "ap_fde_2m": ap_fde_2m2,
                            "ade_selected": ade_sel,
                            "ade_selected_0_5s": ade_sel_0_5s,
                            "ade_selected_1_0s": ade_sel_1_0s,
                            "ade_selected_2_0s": ade_sel_2_0s,
                            "ade_min": ade_min,
                            "ade_min_all": ade_min_all,
                            "ade_min_0_5s": ade_min_0_5s,
                            "ade_min_1_0s": ade_min_1_0s,
                            "ade_min_2_0s": ade_min_2_0s,
                            "ade_score": ade_score,
                            "ade_score_0_5s": ade_score_0_5s,
                            "ade_score_1_0s": ade_score_1_0s,
                            "ade_score_2_0s": ade_score_2_0s,
                            "route_dev_selected": dev_sel,
                            "route_dev_min": dev_min,
                            "route_dev_score": dev_score,
                            "overlay_frame_ref": rec.get("overlay_frame_ref"),
                            "local_plot_frame_ref": rec.get("local_plot_frame_ref"),
                            "auto_enabled": rec.get("auto_enabled"),
                            "parse_note": rec.get("parse_note"),
                            "model_output_error": rec.get("model_output_error"),
                        }
                    )
                except Exception:
                    pass

    # Model output health (best-effort):
    # - pred_output_error_count: evaluated snapshots where the model/output was unhealthy (e.g.
    # empty response),
    #   even if we fell back to stop.
    # - pred_skip_error_count: snapshots skipped due to pred parsing/adapter errors ("pred:*").
    # - pred_error_count: total (pred_output_error_count + pred_skip_error_count)
    pred_skip_error_count = int(
        sum(int(v) for k, v in skipped_reasons.items() if str(k).startswith("pred:"))
    )
    pred_error_count = int(pred_output_error_count) + int(pred_skip_error_count)
    pred_output_total = int(evaluated) + int(pred_skip_error_count)

    t_end = _dt.datetime.now(tz=_dt.UTC)
    duration_s = (t_end - t_start).total_seconds()

    metrics: dict[str, Any] = {
        "task": benchmark_name,
        "slow_brain_fast_planner_version": __version__,
        "job_started_at_utc": t_start.isoformat(),
        "job_finished_at_utc": t_end.isoformat(),
        "job_duration_s": duration_s,
        "created_at_utc": utc_now_iso(),
        "dataset": str(dataset_path),
        "planner_source": str(args.planner_source_report)
        if args.planner_source_report is not None
        else None,
        "selector": str(args.model),
        "experiment_name": experiment_name,
        "trial_name": trial_name,
        "seed": int(args.seed),
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
        "stop_error_count": int(stop_error_count),
        "stop_error_rate": safe_div(stop_error_count, evaluated),
        "invalid_index_count": int(invalid_index_count),
        "invalid_index_rate": safe_div(invalid_index_count, evaluated),
        "pred_output_error_count": int(pred_output_error_count),
        "pred_output_error_reasons": {
            k: int(pred_output_error_reasons[k]) for k in sorted(pred_output_error_reasons)
        },
        "pred_error_count": int(pred_error_count),
        # Error rate among "model outputs" (evaluated + pred-error skipped). This excludes
        # label/prep skips.
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
        "fde_model_avg": safe_div(fde_model_sum, fde_count),
        "fde_score_avg": safe_div(fde_score_sum, fde_score_count),
        "fde_score_masked_laststep_avg": safe_div(
            fde_score_masked_laststep_sum, fde_score_masked_laststep_count
        ),
        "fde_min_avg": safe_div(fde_min_sum, fde_min_count),
        "fde_min_all_avg": safe_div(fde_min_all_sum, fde_min_all_count),
        "fde_count": int(fde_count),
        "fde_score_masked_laststep_count": int(fde_score_masked_laststep_count),
        "moe_deg_avg": safe_div(moe_deg_model_sum, moe_deg_count),
        "moe_deg_score_avg": safe_div(moe_deg_score_sum, moe_deg_score_count),
        "moe_deg_min_avg": safe_div(moe_deg_min_sum, moe_deg_min_count),
        "moe_deg_min_all_avg": safe_div(moe_deg_min_all_sum, moe_deg_min_all_count),
        "moe_deg_mean_all_avg": safe_div(moe_deg_mean_all_sum, moe_deg_mean_all_count),
        "moe_deg_count": int(moe_deg_count),
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
        # Open-loop social metrics.
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
        # Useful when interpreting Gemini "thinking" models: input+output only (no thoughts).
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
        "shard": {
            "enabled": shard_enabled,
            "num_shards": num_shards,
            "shard_id": shard_id,
            "mode": str(args.shard_mode),
        },
        "dataloader": {"num_workers": int(args.num_workers)},
    }
    if latency_ms:
        lat_sorted = sorted(latency_ms)
        n = len(lat_sorted)
        metrics["latency_ms_avg"] = float(sum(lat_sorted) / float(n))
        metrics["latency_ms_p50"] = float(lat_sorted[int(0.50 * (n - 1))])
        metrics["latency_ms_p90"] = float(lat_sorted[int(0.90 * (n - 1))])
    if ade_count == 0:
        metrics["ade_note"] = (
            "ADE metrics unavailable (ade_count=0). This happens when GT pose files are missing "
            "or cannot be aligned."
        )
    if route_dev_count == 0:
        metrics["route_dev_note"] = (
            "Route deviation metrics unavailable (route_dev_count=0). This happens when route "
            "files are missing "
            "or cannot be aligned."
        )

    write_json(out_dir / "metrics.json", metrics)
    _write_text(out_dir / "metrics_report.txt", _format_metrics_report(metrics, run_dir=out_dir))
    _write_metrics_summary_csv(out_dir / "metrics_summary.csv", metrics)
    ms_simple_csv = out_dir / "metrics_summary_simple.csv"
    _write_metrics_summary_simple_csv(ms_simple_csv, metrics)
    _write_metrics_summary_simple_txt(ms_simple_csv, out_dir / "metrics_summary_simple.txt")

    # Print a human-friendly summary to stdout (so users don't need to open metrics.json).
    try:
        print("")
        print(_format_metrics_report(metrics, run_dir=out_dir))
        print(f"[done] metrics_report.txt: {out_dir / 'metrics_report.txt'}")
        print(f"[done] metrics_summary.csv: {out_dir / 'metrics_summary.csv'}")
        print(f"[done] metrics_summary_simple.csv: {out_dir / 'metrics_summary_simple.csv'}")
        print(f"[done] metrics_summary_simple.txt: {out_dir / 'metrics_summary_simple.txt'}")
    except Exception:
        # Fall back silently; files are already written.
        pass

    if bool(args.write_report):
        generate_run_report_html(
            run_dir=out_dir, out_path=out_dir / "report.html", embed_images=False
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
