#!/usr/bin/env python3
"""Merge trajectory-selection shard outputs into a single run directory.

This repo supports episode-level sharding via WORLD_SIZE/RANK (or --num-shards/--shard-id),
but shard runs naturally write into separate output directories. For VLM runs this makes
parallelization easy, but browsing + reporting is nicer with a single merged run directory.

This script expects shard directories that each contain:
  - predictions.jsonl
  - metrics.json (optional but recommended)
  - config.json (optional)
  - traces/*.jsonl (optional)
  - artifacts/... (optional, referenced by overlay_frame_ref/local_plot_frame_ref)

It writes into OUT_DIR:
  - predictions.jsonl (merged; with asset paths rewritten to include shards/<name>/ prefix)
  - traces/events.jsonl (merged, best-effort)
  - metrics.json (merged, best-effort)
  - config.json (merged, best-effort)
  - report.html (generated; optional)

Typical layout (recommended by the companion bash launcher):
  RUN_DIR/
    shards/
      shard0/  (a normal per-shard run dir)
      shard1/
      ...
    <merged outputs live at RUN_DIR/>

Convenience mode (recommended for quick checks):
  Pass RUN_DIR directly and the script will infer:
  - SHARDS_DIR = RUN_DIR/shards
  - OUT_DIR    = RUN_DIR/quickcheck_merged
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _safe_div(a: float | int, b: float | int) -> float | None:
    try:
        bf = float(b)
        if bf == 0:
            return None
        return float(a) / bf
    except Exception:
        return None


def _snapshot_key(rec: dict[str, Any]) -> tuple[str, float, int] | None:
    try:
        eid = str(rec.get("episode_id"))
        t = float(rec.get("t"))
        si = int(rec.get("snapshot_index"))
        return (eid, t, si)
    except Exception:
        return None


def _rewrite_asset_ref(ref: Any, *, shard_name: str) -> Any:
    """Rewrite a relative asset ref to live under shards/<shard_name>/..."""
    if not isinstance(ref, str) or not ref.strip():
        return ref
    # Avoid double-prefixing when called multiple times.
    if ref.startswith("shards/"):
        return ref
    p = Path(ref)
    if p.is_absolute():
        # Keep absolute refs unchanged (rare; but safe).
        return ref
    return str(Path("shards") / shard_name / p)


_TRACE_REF_PREFIXES: tuple[str, ...] = (
    # These are the common run-relative directories written by Slow Brain, Fast Planner CLIs.
    "artifacts/",
    "prompts/",
    "requests/",
    "responses/",
)


def _rewrite_trace_ref(ref: Any, *, shard_name: str) -> Any:
    """Rewrite known run-relative refs inside traces to include shard prefix."""
    if not isinstance(ref, str) or not ref.strip():
        return ref
    if ref.startswith("shards/"):
        return ref
    p = Path(ref)
    if p.is_absolute():
        return ref
    s = ref.strip()
    if any(s.startswith(pref) for pref in _TRACE_REF_PREFIXES):
        return str(Path("shards") / shard_name / p)
    return ref


def _rewrite_trace_refs_inplace(obj: Any, *, shard_name: str) -> Any:
    """In-place rewrite of trace payload refs (best-effort)."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, str):
                obj[k] = _rewrite_trace_ref(v, shard_name=shard_name)
            else:
                _rewrite_trace_refs_inplace(v, shard_name=shard_name)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                obj[i] = _rewrite_trace_ref(v, shard_name=shard_name)
            else:
                _rewrite_trace_refs_inplace(v, shard_name=shard_name)
        return obj
    return obj


def _rewrite_trace_event_assets(ev: dict[str, Any], *, shard_name: str) -> dict[str, Any]:
    """Best-effort rewrite of asset paths inside trace events.

    This makes VLM trace browsing consistent in the merged run dir.
    """
    if not isinstance(ev, dict):
        return ev
    # Historically we only rewrote a few query_bundle fields for model_call events.
    # However, other parts of the trace payload can also carry run-relative refs:
    # - obs.observation.vision.overlay_frame_ref
    # - model_call.messages[*].content[*].image_ref
    # - model_call.query_bundle.obs.vision.overlay_frame_ref
    #
    # The report.html generator will surface these refs, so rewrite them too.
    _rewrite_trace_refs_inplace(ev, shard_name=shard_name)
    return ev


def _write_predictions_csv_from_jsonl(out_csv: Path, preds: list[dict[str, Any]]) -> None:
    """Write predictions.csv in the same column style as the Task2 evaluator."""
    import csv

    cols = [
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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in preds:
            gx = gy = None
            goal_xy = rec.get("goal_xy")
            if isinstance(goal_xy, list) and len(goal_xy) >= 2:
                gx, gy = goal_xy[0], goal_xy[1]

            ade = rec.get("ade") if isinstance(rec.get("ade"), dict) else None
            fde = rec.get("fde") if isinstance(rec.get("fde"), dict) else None
            moe = rec.get("moe") if isinstance(rec.get("moe"), dict) else None
            dev = rec.get("route_dev") if isinstance(rec.get("route_dev"), dict) else None
            row = {
                "episode_id": rec.get("episode_id"),
                "t": rec.get("t"),
                "snapshot_index": rec.get("snapshot_index"),
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
                "selected_traj_avg_dist_to_goal_m": rec.get("selected_traj_avg_dist_to_goal_m"),
                "selected_goal_progress_m": rec.get("selected_goal_progress_m"),
                "traj_len_m": rec.get("traj_len_m"),
                "maoe_deg": rec.get("maoe_deg"),
                "dcr": rec.get("dcr"),
                "tcr": rec.get("tcr"),
                "compliance_source": rec.get("compliance_source"),
                "fde_selected": (fde.get("selected") if isinstance(fde, dict) else None),
                "fde_min": (fde.get("min") if isinstance(fde, dict) else None),
                "fde_min_all": (fde.get("min_all") if isinstance(fde, dict) else None),
                "fde_score": (fde.get("score") if isinstance(fde, dict) else None),
                "moe_deg": (moe.get("selected") if isinstance(moe, dict) else None),
                "moe_deg_min": (moe.get("min") if isinstance(moe, dict) else None),
                "moe_deg_min_all": (moe.get("min_all") if isinstance(moe, dict) else None),
                "moe_deg_score": (moe.get("score") if isinstance(moe, dict) else None),
                "moe_deg_mean_all": (moe.get("mean_all") if isinstance(moe, dict) else None),
                "ap_fde_2m": rec.get("ap_fde_2m"),
                "ade_selected": (ade.get("selected") if isinstance(ade, dict) else None),
                "ade_selected_0_5s": (ade.get("selected_0_5s") if isinstance(ade, dict) else None),
                "ade_selected_1_0s": (ade.get("selected_1_0s") if isinstance(ade, dict) else None),
                "ade_selected_2_0s": (ade.get("selected_2_0s") if isinstance(ade, dict) else None),
                "ade_min": (ade.get("min") if isinstance(ade, dict) else None),
                "ade_min_all": (ade.get("min_all") if isinstance(ade, dict) else None),
                "ade_min_0_5s": (ade.get("min_0_5s") if isinstance(ade, dict) else None),
                "ade_min_1_0s": (ade.get("min_1_0s") if isinstance(ade, dict) else None),
                "ade_min_2_0s": (ade.get("min_2_0s") if isinstance(ade, dict) else None),
                "ade_score": (ade.get("score") if isinstance(ade, dict) else None),
                "ade_score_0_5s": (ade.get("score_0_5s") if isinstance(ade, dict) else None),
                "ade_score_1_0s": (ade.get("score_1_0s") if isinstance(ade, dict) else None),
                "ade_score_2_0s": (ade.get("score_2_0s") if isinstance(ade, dict) else None),
                "route_dev_selected": (dev.get("selected") if isinstance(dev, dict) else None),
                "route_dev_min": (dev.get("min") if isinstance(dev, dict) else None),
                "route_dev_score": (dev.get("score") if isinstance(dev, dict) else None),
                "overlay_frame_ref": rec.get("overlay_frame_ref"),
                "local_plot_frame_ref": rec.get("local_plot_frame_ref"),
                "auto_enabled": rec.get("auto_enabled"),
            }
            w.writerow(row)


def _write_shards_index_html(out_path: Path, *, shard_dirs: list[Path], shards_root: Path) -> None:
    """Small convenience index for browsing per-shard outputs."""
    lines: list[str] = []
    lines.append("<!doctype html>")
    lines.append("<html><head><meta charset='utf-8'/>")
    lines.append("<title>Task2 Shards</title>")
    lines.append(
        "<style>body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;padding:16px}"
        "code{background:#f5f5f5;padding:2px 4px;border-radius:4px}"
        "a{color:#0645ad;text-decoration:none} a:hover{text-decoration:underline}"
        ".muted{color:#666} ul{line-height:1.6}</style>"
    )
    lines.append("</head><body>")
    lines.append("<h2>Task 2 shard runs</h2>")
    lines.append(f"<div class='muted'>Root: <code>{shards_root}</code></div>")
    lines.append("<ul>")
    for sd in shard_dirs:
        name = sd.name
        # Links are relative to OUT_DIR, which contains shards/<name>/...
        lines.append("<li>")
        lines.append(f"<b>{name}</b>: ")
        lines.append(f"<a href='shards/{name}/report.html'>report</a>")
        lines.append(" | ")
        lines.append(f"<a href='shards/{name}/metrics.json'>metrics.json</a>")
        lines.append(" | ")
        lines.append(f"<a href='shards/{name}/predictions.jsonl'>predictions.jsonl</a>")
        lines.append("</li>")
    lines.append("</ul>")
    lines.append("</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _merge_metrics_from_predictions(
    preds: list[dict[str, Any]],
    *,
    base_metrics: dict[str, Any] | None,
    shard_count_hint: int | None = None,
) -> dict[str, Any]:
    """Best-effort merged metrics matching the per-run schema in
    slow_brain_fast_planner.cli.trajectory_selection."""

    evaluated = 0
    correct = 0
    ade_count = 0
    correct_vs_min_ade = 0
    correct_vs_min_ade_all = 0
    score_agree_count = 0
    correct_vs_score = 0
    route_dev_count = 0
    correct_vs_route_min = 0
    correct_vs_route_score = 0
    auto_true = auto_false = auto_missing = 0

    ade_min_sum = ade_min_all_sum = ade_score_sum = ade_model_sum = 0.0
    ade_model_0_5s_sum = ade_model_1_0s_sum = ade_model_2_0s_sum = 0.0
    ade_score_0_5s_sum = ade_score_1_0s_sum = ade_score_2_0s_sum = 0.0
    ade_min_0_5s_sum = ade_min_1_0s_sum = ade_min_2_0s_sum = 0.0
    ade_model_0_5s_count = ade_model_1_0s_count = ade_model_2_0s_count = 0
    ade_score_0_5s_count = ade_score_1_0s_count = ade_score_2_0s_count = 0
    ade_min_0_5s_count = ade_min_1_0s_count = ade_min_2_0s_count = 0

    route_dev_min_sum = route_dev_score_sum = route_dev_model_sum = 0.0

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

    # Open-loop social metrics (present in newer Task2 outputs).
    maoe_sum = 0.0
    maoe_count = 0
    dcr_sum = 0.0
    dcr_count = 0
    tcr_sum = 0.0
    tcr_count = 0

    # Open-loop trajectory metrics (FDE/MOE/mAP; optional in older runs).
    fde_model_sum = fde_min_sum = fde_min_all_sum = fde_score_sum = 0.0
    fde_model_count = fde_min_count = fde_min_all_count = fde_score_count = 0
    fde_score_masked_laststep_sum = 0.0
    fde_score_masked_laststep_count = 0
    moe_model_sum = moe_min_sum = moe_min_all_sum = moe_score_sum = moe_mean_all_sum = 0.0
    moe_model_count = moe_min_count = moe_min_all_count = moe_score_count = 0
    moe_mean_all_count = 0
    ap_fde_2m_sum = 0.0
    ap_fde_2m_count = 0

    # Latency (best-effort): many records don't include it; we also try traces separately.
    latency_ms: list[float] = []
    stop_count = 0
    invalid_index_count = 0
    pred_error_count = 0

    for rec in preds:
        # Count pred-error skips (these don't contribute to evaluated/correct).
        if bool(rec.get("skipped")):
            sr = rec.get("skip_reason")
            if isinstance(sr, str) and sr.startswith("pred:"):
                pred_error_count += 1
            continue

        # Auto-enabled counts
        ae = rec.get("auto_enabled")
        if ae is True:
            auto_true += 1
        elif ae is False:
            auto_false += 1
        else:
            auto_missing += 1

        evaluated += 1
        if rec.get("correct") is True:
            correct += 1

        # Open-loop social metrics (best-effort; only when present and finite).
        try:
            v = rec.get("maoe_deg")
            if v is not None:
                vf = float(v)
                if math.isfinite(vf):
                    maoe_sum += vf
                    maoe_count += 1
        except Exception:
            pass
        try:
            v = rec.get("dcr")
            if v is not None:
                vf = float(v)
                if math.isfinite(vf):
                    dcr_sum += vf
                    dcr_count += 1
        except Exception:
            pass
        try:
            v = rec.get("tcr")
            if v is not None:
                vf = float(v)
                if math.isfinite(vf):
                    tcr_sum += vf
                    tcr_count += 1
        except Exception:
            pass

        # FDE / MOE / mAP (optional; best-effort).
        try:
            fde = rec.get("fde") if isinstance(rec.get("fde"), dict) else None
            if isinstance(fde, dict):
                for k, sum_acc, _cnt_acc in [
                    ("selected", "fde_model_sum", "fde_model_count"),
                    ("min", "fde_min_sum", "fde_min_count"),
                    ("min_all", "fde_min_all_sum", "fde_min_all_count"),
                    ("score", "fde_score_sum", "fde_score_count"),
                    (
                        "score_masked_laststep",
                        "fde_score_masked_laststep_sum",
                        "fde_score_masked_laststep_count",
                    ),
                ]:
                    v = fde.get(k)
                    if v is None:
                        continue
                    vf = float(v)
                    if not math.isfinite(vf):
                        continue
                    if sum_acc == "fde_model_sum":
                        fde_model_sum += vf
                        fde_model_count += 1
                    elif sum_acc == "fde_min_sum":
                        fde_min_sum += vf
                        fde_min_count += 1
                    elif sum_acc == "fde_min_all_sum":
                        fde_min_all_sum += vf
                        fde_min_all_count += 1
                    elif sum_acc == "fde_score_masked_laststep_sum":
                        fde_score_masked_laststep_sum += vf
                        fde_score_masked_laststep_count += 1
                    else:
                        fde_score_sum += vf
                        fde_score_count += 1
        except Exception:
            pass
        try:
            moe = rec.get("moe") if isinstance(rec.get("moe"), dict) else None
            if isinstance(moe, dict):
                for k, sum_acc, _cnt_acc in [
                    ("selected", "moe_model_sum", "moe_model_count"),
                    ("min", "moe_min_sum", "moe_min_count"),
                    ("min_all", "moe_min_all_sum", "moe_min_all_count"),
                    ("score", "moe_score_sum", "moe_score_count"),
                ]:
                    v = moe.get(k)
                    if v is None:
                        continue
                    vf = float(v)
                    if not math.isfinite(vf):
                        continue
                    if sum_acc == "moe_model_sum":
                        moe_model_sum += vf
                        moe_model_count += 1
                    elif sum_acc == "moe_min_sum":
                        moe_min_sum += vf
                        moe_min_count += 1
                    elif sum_acc == "moe_min_all_sum":
                        moe_min_all_sum += vf
                        moe_min_all_count += 1
                    else:
                        moe_score_sum += vf
                        moe_score_count += 1
                v = moe.get("mean_all")
                if v is not None:
                    vf = float(v)
                    if math.isfinite(vf):
                        moe_mean_all_sum += vf
                        moe_mean_all_count += 1
        except Exception:
            pass
        try:
            v = rec.get("ap_fde_2m")
            if v is not None:
                vf = float(v)
                if math.isfinite(vf):
                    ap_fde_2m_sum += vf
                    ap_fde_2m_count += 1
        except Exception:
            pass

        # Model output health (best-effort; based on predictions.jsonl fields).
        pred_action = rec.get("prediction_action")
        if pred_action == "stop":
            stop_count += 1
        if pred_action == "select_trajectory":
            try:
                pred_idx = rec.get("prediction")
                nc = rec.get("num_candidates")
                if pred_idx is not None and nc is not None:
                    if not (0 <= int(pred_idx) < int(nc)):
                        invalid_index_count += 1
            except Exception:
                invalid_index_count += 1

        # Token usage / latency are not reliably stored per-rec; leave to base_metrics or traces.

        # Goal-relative metrics for selected trajectory.
        try:
            gd = rec.get("goal_distance_m")
            if gd is not None:
                gdf = float(gd)
                if math.isfinite(gdf):
                    goal_dist_sum += gdf
                    goal_dist_count += 1
        except Exception:
            pass
        for k_sum, _k_cnt, key in [
            ("sel_end_goal_dist_sum", "sel_end_goal_dist_count", "selected_end_dist_to_goal_m"),
            ("sel_goal_ang_diff_sum", "sel_goal_ang_diff_count", "selected_goal_ang_diff_deg"),
            (
                "sel_traj_avg_goal_dist_sum",
                "sel_traj_avg_goal_dist_count",
                "selected_traj_avg_dist_to_goal_m",
            ),
            ("sel_goal_progress_sum", "sel_goal_progress_count", "selected_goal_progress_m"),
        ]:
            try:
                v = rec.get(key)
                if v is None:
                    continue
                vf = float(v)
                if not math.isfinite(vf):
                    continue
                if k_sum == "sel_end_goal_dist_sum":
                    sel_end_goal_dist_sum += vf
                    sel_end_goal_dist_count += 1
                elif k_sum == "sel_goal_ang_diff_sum":
                    sel_goal_ang_diff_sum += vf
                    sel_goal_ang_diff_count += 1
                elif k_sum == "sel_traj_avg_goal_dist_sum":
                    sel_traj_avg_goal_dist_sum += vf
                    sel_traj_avg_goal_dist_count += 1
                elif k_sum == "sel_goal_progress_sum":
                    sel_goal_progress_sum += vf
                    sel_goal_progress_count += 1
            except Exception:
                pass

        # accuracy_vs_score: compare prediction to score_idx (if present)
        pred_idx = rec.get("prediction")
        try:
            if pred_idx is not None:
                pred_idx_int = int(pred_idx)
            else:
                pred_idx_int = None
        except Exception:
            pred_idx_int = None

        try:
            ade_obj = rec.get("ade") if isinstance(rec.get("ade"), dict) else None
            if ade_obj is not None:
                # In the main evaluator, ade_count increments even for "stop" when ade.selected is
                # present.
                sel = ade_obj.get("selected")
                if sel is not None:
                    sf = float(sel)
                    if math.isfinite(sf):
                        ade_count += 1
                        # min/min_all/score are baseline aggregates (best-effort)
                        for key, acc in [
                            ("min", "ade_min_sum"),
                            ("min_all", "ade_min_all_sum"),
                            ("score", "ade_score_sum"),
                        ]:
                            v = ade_obj.get(key)
                            if v is None:
                                continue
                            vf = float(v)
                            if math.isfinite(vf):
                                if acc == "ade_min_sum":
                                    ade_min_sum += vf
                                elif acc == "ade_min_all_sum":
                                    ade_min_all_sum += vf
                                else:
                                    ade_score_sum += vf

                        ade_model_sum += sf

                        # Prefix ADEs
                        for key, sum_acc, _cnt_acc in [
                            ("selected_0_5s", "ade_model_0_5s_sum", "ade_model_0_5s_count"),
                            ("selected_1_0s", "ade_model_1_0s_sum", "ade_model_1_0s_count"),
                            ("selected_2_0s", "ade_model_2_0s_sum", "ade_model_2_0s_count"),
                            ("score_0_5s", "ade_score_0_5s_sum", "ade_score_0_5s_count"),
                            ("score_1_0s", "ade_score_1_0s_sum", "ade_score_1_0s_count"),
                            ("score_2_0s", "ade_score_2_0s_sum", "ade_score_2_0s_count"),
                            ("min_0_5s", "ade_min_0_5s_sum", "ade_min_0_5s_count"),
                            ("min_1_0s", "ade_min_1_0s_sum", "ade_min_1_0s_count"),
                            ("min_2_0s", "ade_min_2_0s_sum", "ade_min_2_0s_count"),
                        ]:
                            v = ade_obj.get(key)
                            if v is None:
                                continue
                            vf = float(v)
                            if not math.isfinite(vf):
                                continue
                            if sum_acc == "ade_model_0_5s_sum":
                                ade_model_0_5s_sum += vf
                                ade_model_0_5s_count += 1
                            elif sum_acc == "ade_model_1_0s_sum":
                                ade_model_1_0s_sum += vf
                                ade_model_1_0s_count += 1
                            elif sum_acc == "ade_model_2_0s_sum":
                                ade_model_2_0s_sum += vf
                                ade_model_2_0s_count += 1
                            elif sum_acc == "ade_score_0_5s_sum":
                                ade_score_0_5s_sum += vf
                                ade_score_0_5s_count += 1
                            elif sum_acc == "ade_score_1_0s_sum":
                                ade_score_1_0s_sum += vf
                                ade_score_1_0s_count += 1
                            elif sum_acc == "ade_score_2_0s_sum":
                                ade_score_2_0s_sum += vf
                                ade_score_2_0s_count += 1
                            elif sum_acc == "ade_min_0_5s_sum":
                                ade_min_0_5s_sum += vf
                                ade_min_0_5s_count += 1
                            elif sum_acc == "ade_min_1_0s_sum":
                                ade_min_1_0s_sum += vf
                                ade_min_1_0s_count += 1
                            elif sum_acc == "ade_min_2_0s_sum":
                                ade_min_2_0s_sum += vf
                                ade_min_2_0s_count += 1
                        # Compare to min indices when prediction is an index.
                        if pred_idx_int is not None:
                            try:
                                mi = ade_obj.get("min_idx")
                                if mi is not None and int(pred_idx_int) == int(mi):
                                    correct_vs_min_ade += 1
                            except Exception:
                                pass
                            try:
                                mia = ade_obj.get("min_all_idx")
                                if mia is not None and int(pred_idx_int) == int(mia):
                                    correct_vs_min_ade_all += 1
                            except Exception:
                                pass

                        # Score agreement for accuracy_vs_score
                        try:
                            si = ade_obj.get("score_idx")
                            nc = rec.get("num_candidates")
                            if pred_idx_int is not None and si is not None and nc is not None:
                                if 0 <= int(pred_idx_int) < int(nc):
                                    score_agree_count += 1
                                    if int(pred_idx_int) == int(si):
                                        correct_vs_score += 1
                        except Exception:
                            pass
        except Exception:
            pass

        # Route deviation aggregates (best-effort)
        try:
            dev_obj = rec.get("route_dev") if isinstance(rec.get("route_dev"), dict) else None
            if dev_obj is not None:
                sel = dev_obj.get("selected")
                if sel is not None:
                    sf = float(sel)
                    if math.isfinite(sf):
                        route_dev_count += 1
                        vmin = dev_obj.get("min")
                        vsc = dev_obj.get("score")
                        if vmin is not None:
                            vf = float(vmin)
                            if math.isfinite(vf):
                                route_dev_min_sum += vf
                        if vsc is not None:
                            vf = float(vsc)
                            if math.isfinite(vf):
                                route_dev_score_sum += vf
                        route_dev_model_sum += sf

                        if pred_idx_int is not None:
                            try:
                                mi = dev_obj.get("min_idx")
                                if mi is not None and int(pred_idx_int) == int(mi):
                                    correct_vs_route_min += 1
                            except Exception:
                                pass
                            try:
                                si = dev_obj.get("score_idx")
                                if si is not None and int(pred_idx_int) == int(si):
                                    correct_vs_route_score += 1
                            except Exception:
                                pass
        except Exception:
            pass

        # Per-record latency (rare, but keep if present)
        try:
            pm = rec.get("provider_meta")
            if isinstance(pm, dict) and pm.get("latency_ms") is not None:
                lm = float(pm.get("latency_ms"))
                if math.isfinite(lm) and lm >= 0:
                    latency_ms.append(lm)
        except Exception:
            pass

    metrics: dict[str, Any] = {}
    if isinstance(base_metrics, dict):
        # Carry through useful identifiers, but override computed aggregates below.
        for k in [
            "task",
            "slow_brain_fast_planner_version",
            "dataset",
            "planner_source",
            "selector",
            "experiment_name",
            "trial_name",
            "seed",
        ]:
            if k in base_metrics:
                metrics[k] = base_metrics.get(k)
    metrics["created_at_utc"] = _utc_now_iso()

    # Shard metadata (best-effort)
    metrics["shard"] = {
        "enabled": True,
        "num_shards": int(shard_count_hint) if shard_count_hint is not None else None,
        "shard_id": None,
        "merged": True,
    }

    metrics["snapshots_evaluated"] = int(evaluated)
    metrics["counts"] = {"correct": int(correct)}
    metrics["accuracy"] = _safe_div(correct, evaluated)
    pred_output_total = int(evaluated) + int(pred_error_count)
    metrics["stop_count"] = int(stop_count)
    metrics["stop_rate"] = _safe_div(stop_count, evaluated)
    metrics["invalid_index_count"] = int(invalid_index_count)
    metrics["invalid_index_rate"] = _safe_div(invalid_index_count, evaluated)
    metrics["pred_error_count"] = int(pred_error_count)
    metrics["pred_error_rate"] = _safe_div(pred_error_count, pred_output_total)

    metrics["ade_count"] = int(ade_count)
    metrics["accuracy_vs_min_ade"] = _safe_div(correct_vs_min_ade, ade_count)
    metrics["accuracy_vs_min_ade_all"] = _safe_div(correct_vs_min_ade_all, ade_count)
    metrics["ade_min_avg"] = _safe_div(ade_min_sum, ade_count)
    metrics["ade_min_all_avg"] = _safe_div(ade_min_all_sum, ade_count)
    metrics["ade_score_avg"] = _safe_div(ade_score_sum, ade_count)
    metrics["ade_model_avg"] = _safe_div(ade_model_sum, ade_count)
    metrics["ade_model_0_5s_avg"] = _safe_div(ade_model_0_5s_sum, ade_model_0_5s_count)
    metrics["ade_model_1_0s_avg"] = _safe_div(ade_model_1_0s_sum, ade_model_1_0s_count)
    metrics["ade_model_2_0s_avg"] = _safe_div(ade_model_2_0s_sum, ade_model_2_0s_count)
    metrics["ade_score_0_5s_avg"] = _safe_div(ade_score_0_5s_sum, ade_score_0_5s_count)
    metrics["ade_score_1_0s_avg"] = _safe_div(ade_score_1_0s_sum, ade_score_1_0s_count)
    metrics["ade_score_2_0s_avg"] = _safe_div(ade_score_2_0s_sum, ade_score_2_0s_count)
    metrics["ade_min_0_5s_avg"] = _safe_div(ade_min_0_5s_sum, ade_min_0_5s_count)
    metrics["ade_min_1_0s_avg"] = _safe_div(ade_min_1_0s_sum, ade_min_1_0s_count)
    metrics["ade_min_2_0s_avg"] = _safe_div(ade_min_2_0s_sum, ade_min_2_0s_count)

    metrics["accuracy_vs_score"] = _safe_div(correct_vs_score, score_agree_count)
    metrics["route_dev_count"] = int(route_dev_count)
    metrics["route_dev_min_avg"] = _safe_div(route_dev_min_sum, route_dev_count)
    metrics["route_dev_score_avg"] = _safe_div(route_dev_score_sum, route_dev_count)
    metrics["route_dev_model_avg"] = _safe_div(route_dev_model_sum, route_dev_count)
    metrics["accuracy_vs_route_min"] = _safe_div(correct_vs_route_min, route_dev_count)
    metrics["accuracy_vs_route_score"] = _safe_div(correct_vs_route_score, route_dev_count)

    metrics["auto_enabled_true_count"] = int(auto_true)
    metrics["auto_enabled_false_count"] = int(auto_false)
    metrics["auto_enabled_missing_count"] = int(auto_missing)
    metrics["auto_enabled_true_frac"] = (
        None if (auto_true + auto_false) == 0 else float(auto_true) / float(auto_true + auto_false)
    )

    metrics["goal_distance_m_avg"] = _safe_div(goal_dist_sum, goal_dist_count)
    metrics["goal_distance_m_count"] = int(goal_dist_count)
    metrics["selected_end_dist_to_goal_m_avg"] = _safe_div(
        sel_end_goal_dist_sum, sel_end_goal_dist_count
    )
    metrics["selected_end_dist_to_goal_m_count"] = int(sel_end_goal_dist_count)
    metrics["selected_goal_ang_diff_deg_avg"] = _safe_div(
        sel_goal_ang_diff_sum, sel_goal_ang_diff_count
    )
    metrics["selected_goal_ang_diff_deg_count"] = int(sel_goal_ang_diff_count)
    metrics["selected_traj_avg_dist_to_goal_m_avg"] = _safe_div(
        sel_traj_avg_goal_dist_sum, sel_traj_avg_goal_dist_count
    )
    metrics["selected_traj_avg_dist_to_goal_m_count"] = int(sel_traj_avg_goal_dist_count)
    metrics["selected_goal_progress_m_avg"] = _safe_div(
        sel_goal_progress_sum, sel_goal_progress_count
    )
    metrics["selected_goal_progress_m_count"] = int(sel_goal_progress_count)

    # Open-loop social metrics (aggregated over samples where the per-rec value exists).
    metrics["maoe_deg_avg"] = _safe_div(maoe_sum, maoe_count)
    metrics["dcr_avg"] = _safe_div(dcr_sum, dcr_count)
    metrics["tcr_avg"] = _safe_div(tcr_sum, tcr_count)

    # Open-loop trajectory metrics (FDE/MOE/mAP; aggregated when present).
    metrics["fde_model_avg"] = _safe_div(fde_model_sum, fde_model_count)
    metrics["fde_score_avg"] = _safe_div(fde_score_sum, fde_score_count)
    metrics["fde_score_masked_laststep_avg"] = _safe_div(
        fde_score_masked_laststep_sum, fde_score_masked_laststep_count
    )
    metrics["fde_min_avg"] = _safe_div(fde_min_sum, fde_min_count)
    metrics["fde_min_all_avg"] = _safe_div(fde_min_all_sum, fde_min_all_count)
    metrics["fde_count"] = int(fde_model_count)
    metrics["fde_score_masked_laststep_count"] = int(fde_score_masked_laststep_count)
    metrics["moe_deg_avg"] = _safe_div(moe_model_sum, moe_model_count)
    metrics["moe_deg_score_avg"] = _safe_div(moe_score_sum, moe_score_count)
    metrics["moe_deg_min_avg"] = _safe_div(moe_min_sum, moe_min_count)
    metrics["moe_deg_min_all_avg"] = _safe_div(moe_min_all_sum, moe_min_all_count)
    metrics["moe_deg_mean_all_avg"] = _safe_div(moe_mean_all_sum, moe_mean_all_count)
    metrics["moe_deg_count"] = int(moe_model_count)
    metrics["map_fde_2m_avg"] = _safe_div(ap_fde_2m_sum, ap_fde_2m_count)
    metrics["map_fde_2m_count"] = int(ap_fde_2m_count)

    # Latency summary (best-effort)
    metrics["latency_ms_avg"] = None
    metrics["latency_ms_p50"] = None
    metrics["latency_ms_p90"] = None
    if latency_ms:
        lat_sorted = sorted(latency_ms)
        n = len(lat_sorted)
        metrics["latency_ms_avg"] = float(sum(lat_sorted) / float(n))
        metrics["latency_ms_p50"] = float(lat_sorted[int(0.50 * (n - 1))])
        metrics["latency_ms_p90"] = float(lat_sorted[int(0.90 * (n - 1))])

    return metrics


def _merge_latency_from_traces(shard_dirs: list[Path]) -> list[float]:
    lat: list[float] = []
    for sd in shard_dirs:
        tdir = sd / "traces"
        if not tdir.is_dir():
            continue
        for p in sorted(tdir.glob("*.jsonl")):
            try:
                for ev in _iter_jsonl(p):
                    if ev.get("event_type") != "model_call":
                        continue
                    pm = ev.get("provider_meta")
                    if isinstance(pm, dict) and pm.get("latency_ms") is not None:
                        lm = float(pm.get("latency_ms"))
                        if math.isfinite(lm) and lm >= 0:
                            lat.append(lm)
            except Exception:
                continue
    return lat


def main(argv: list[str] | None = None) -> int:
    t_start = _dt.datetime.now(tz=_dt.UTC)
    ap = argparse.ArgumentParser(
        description="Merge trajectory-selection shard run directories into a single run dir."
    )
    ap.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help=(
            "Convenience positional argument: a RUN_DIR containing shards/. "
            "If provided, defaults to --shards-dir RUN_DIR/shards and --out "
            "RUN_DIR/quickcheck_merged."
        ),
    )
    ap.add_argument(
        "--shards-dir",
        default=None,
        help="Directory containing shard run dirs (e.g., RUN_DIR/shards/shard0, shard1, ...).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Output merged run dir. If omitted, defaults to <RUN_DIR>/quickcheck_merged "
            "(or if only --shards-dir is given, to <SHARDS_DIR>/../quickcheck_merged)."
        ),
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite merged outputs if they exist."
    )
    ap.add_argument(
        "--write-report", action="store_true", help="Generate report.html in merged OUT_DIR."
    )
    args = ap.parse_args(argv)

    run_dir = Path(str(args.run_dir)).resolve() if args.run_dir is not None else None
    shards_root = Path(str(args.shards_dir)).resolve() if args.shards_dir is not None else None
    out_dir = Path(str(args.out)).resolve() if args.out is not None else None

    # Infer shards/out for quick-check usage.
    if shards_root is None:
        if run_dir is None:
            raise SystemExit("Must provide either RUN_DIR (positional) or --shards-dir.")
        shards_root = (run_dir / "shards").resolve()
    if out_dir is None:
        # Default to ../quickcheck_merged relative to shards root.
        out_dir = (shards_root.parent / "quickcheck_merged").resolve()

    if not shards_root.is_dir():
        raise SystemExit(f"--shards-dir is not a directory: {shards_root}")

    shard_dirs = [p for p in sorted(shards_root.iterdir()) if p.is_dir()]
    shard_dirs = [p for p in shard_dirs if (p / "predictions.jsonl").exists()]
    if not shard_dirs:
        raise SystemExit(f"No shard dirs with predictions.jsonl found under: {shards_root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    preds_out = out_dir / "predictions.jsonl"
    preds_csv_out = out_dir / "predictions.csv"
    traces_out = out_dir / "traces" / "events.jsonl"
    metrics_out = out_dir / "metrics.json"
    cfg_out = out_dir / "config.json"
    shards_index_out = out_dir / "shards_index.html"

    # Basic overwrite guard
    for p in (preds_out, traces_out, metrics_out):
        if p.exists() and not bool(args.overwrite):
            raise SystemExit(f"Refusing to overwrite existing file without --overwrite: {p}")

    # Ensure shards are reachable from OUT_DIR (symlink shards/<name> -> original shard_dir)
    (out_dir / "shards").mkdir(parents=True, exist_ok=True)
    shard_name_by_dir: dict[Path, str] = {}
    for sd in shard_dirs:
        name = sd.name
        shard_name_by_dir[sd] = name
        link = out_dir / "shards" / name
        # If the shard dir already lives at OUT_DIR/shards/<name>, do nothing.
        # (This is the recommended layout produced by bash/task2_vlm_sharded.sh.)
        try:
            if link.resolve() == sd.resolve():
                continue
        except Exception:
            # Fall back to string compare if resolve fails for some reason.
            if str(link) == str(sd):
                continue
        if link.exists() or link.is_symlink():
            # If it's already correct, keep it; else overwrite when allowed.
            try:
                if link.is_symlink() and link.resolve() == sd:
                    continue
            except Exception:
                pass
            if not bool(args.overwrite):
                raise SystemExit(f"shards link exists; pass --overwrite to replace: {link}")
            try:
                if link.is_dir() and not link.is_symlink():
                    raise SystemExit(f"Refusing to replace real directory at: {link}")
                link.unlink()
            except Exception:
                pass
        link.symlink_to(sd)

    # Merge predictions (dedupe by snapshot key)
    seen: set[tuple[str, float, int]] = set()
    merged_preds: list[dict[str, Any]] = []
    for sd in shard_dirs:
        shard_name = shard_name_by_dir[sd]
        for rec in _iter_jsonl(sd / "predictions.jsonl"):
            k = _snapshot_key(rec)
            if k is None:
                continue
            if k in seen:
                continue
            seen.add(k)
            rec2 = dict(rec)
            # Rewrite asset refs so report.html can resolve them from OUT_DIR.
            rec2["overlay_frame_ref"] = _rewrite_asset_ref(
                rec2.get("overlay_frame_ref"), shard_name=shard_name
            )
            rec2["local_plot_frame_ref"] = _rewrite_asset_ref(
                rec2.get("local_plot_frame_ref"), shard_name=shard_name
            )
            merged_preds.append(rec2)

    # Stable ordering: episode_id, then t, then snapshot_index
    merged_preds.sort(
        key=lambda r: (
            str(r.get("episode_id")),
            float(r.get("t", 0.0)),
            int(r.get("snapshot_index", 0)),
        )
    )
    _write_jsonl(preds_out, merged_preds)

    # Merge traces (best-effort; no dedupe)
    merged_traces: list[dict[str, Any]] = []
    for sd in shard_dirs:
        shard_name = shard_name_by_dir[sd]
        tdir = sd / "traces"
        if not tdir.is_dir():
            continue
        for tp in sorted(tdir.glob("*.jsonl")):
            try:
                for ev in _iter_jsonl(tp):
                    merged_traces.append(_rewrite_trace_event_assets(ev, shard_name=shard_name))
            except Exception:
                continue
    if merged_traces:
        # Sort loosely by time_utc if present, else keep append order.
        def _tkey(ev: dict[str, Any]) -> str:
            v = ev.get("time_utc")
            return str(v) if isinstance(v, str) else ""

        merged_traces.sort(key=_tkey)
        _write_jsonl(traces_out, merged_traces)

    # Merge config/metrics: use a base from the first shard as a template
    base_cfg = None
    base_metrics = None
    first = shard_dirs[0]
    if (first / "config.json").exists():
        try:
            obj = _read_json(first / "config.json")
            if isinstance(obj, dict):
                base_cfg = obj
        except Exception:
            base_cfg = None
    if (first / "metrics.json").exists():
        try:
            obj = _read_json(first / "metrics.json")
            if isinstance(obj, dict):
                base_metrics = obj
        except Exception:
            base_metrics = None

    # Sum shard-level totals where possible (best-effort)
    totals: dict[str, Any] = {
        "episodes_total": 0,
        "episodes_schema_invalid": 0,
        "snapshots_total": 0,
        "snapshots_skipped": 0,
        "model_calls": 0,
        "prompt_tokens_total": 0,
        "cached_prompt_tokens_total": 0,
        "output_tokens_total": 0,
        "thoughts_tokens_total": 0,
        "total_tokens_total": 0,
        "total_tokens_no_thoughts_total": 0,
        "auto_enabled_true_count": 0,
        "auto_enabled_false_count": 0,
        "auto_enabled_missing_count": 0,
        "skipped_reasons": {},
    }
    skipped_reasons: dict[str, int] = {}
    for sd in shard_dirs:
        mp = sd / "metrics.json"
        if not mp.exists():
            continue
        try:
            m = _read_json(mp)
        except Exception:
            continue
        if not isinstance(m, dict):
            continue
        for k in [
            "episodes_total",
            "episodes_schema_invalid",
            "snapshots_total",
            "snapshots_skipped",
        ]:
            try:
                totals[k] = int(totals.get(k, 0)) + int(m.get(k, 0))
            except Exception:
                pass
        for k in [
            "model_calls",
            "prompt_tokens_total",
            "cached_prompt_tokens_total",
            "output_tokens_total",
            "thoughts_tokens_total",
            "total_tokens_total",
            "total_tokens_no_thoughts_total",
        ]:
            try:
                totals[k] = int(totals.get(k, 0)) + int(m.get(k, 0))
            except Exception:
                pass
        for k in [
            "auto_enabled_true_count",
            "auto_enabled_false_count",
            "auto_enabled_missing_count",
        ]:
            try:
                totals[k] = int(totals.get(k, 0)) + int(m.get(k, 0))
            except Exception:
                pass
        sr = m.get("skipped_reasons")
        if isinstance(sr, dict):
            for kk, vv in sr.items():
                try:
                    skipped_reasons[str(kk)] = int(skipped_reasons.get(str(kk), 0)) + int(vv)
                except Exception:
                    continue
    totals["skipped_reasons"] = {k: int(skipped_reasons[k]) for k in sorted(skipped_reasons)}

    merged_metrics = _merge_metrics_from_predictions(
        merged_preds,
        base_metrics=base_metrics,
        shard_count_hint=len(shard_dirs),
    )
    t_end = _dt.datetime.now(tz=_dt.UTC)
    duration_s = (t_end - t_start).total_seconds()
    merged_metrics["job_started_at_utc"] = t_start.isoformat()
    merged_metrics["job_finished_at_utc"] = t_end.isoformat()
    merged_metrics["job_duration_s"] = duration_s

    # Match Task2 evaluator convention:
    #   experiment_name = out_dir.parent.name
    #   trial_name      = out_dir.name
    # (Shard runs naturally have experiment_name="shards", trial_name="shardK";
    #  merged run should reflect the actual run directory.)
    try:
        merged_metrics["experiment_name"] = (
            str(out_dir.parent.name) if out_dir.parent is not None else str(out_dir)
        )
        merged_metrics["trial_name"] = str(out_dir.name)
    except Exception:
        pass
    # Add shard-summed totals (these include prep-fail skips that aren't recorded in
    # predictions.jsonl)
    for k in [
        "episodes_total",
        "episodes_schema_invalid",
        "snapshots_total",
        "snapshots_skipped",
        "skipped_reasons",
        "model_calls",
        "prompt_tokens_total",
        "cached_prompt_tokens_total",
        "output_tokens_total",
        "thoughts_tokens_total",
        "total_tokens_total",
        "total_tokens_no_thoughts_total",
        "auto_enabled_true_count",
        "auto_enabled_false_count",
        "auto_enabled_missing_count",
    ]:
        if k in totals and totals.get(k) not in (None, {}, []):
            merged_metrics[k] = totals.get(k)

    # Token averages (if totals present)
    mc = merged_metrics.get("model_calls")
    try:
        if mc is not None and int(mc) > 0:
            merged_metrics["prompt_tokens_avg"] = _safe_div(
                int(merged_metrics.get("prompt_tokens_total") or 0), int(mc)
            )
            merged_metrics["cached_prompt_tokens_avg"] = _safe_div(
                int(merged_metrics.get("cached_prompt_tokens_total") or 0), int(mc)
            )
            merged_metrics["output_tokens_avg"] = _safe_div(
                int(merged_metrics.get("output_tokens_total") or 0), int(mc)
            )
            merged_metrics["thoughts_tokens_avg"] = _safe_div(
                int(merged_metrics.get("thoughts_tokens_total") or 0), int(mc)
            )
            merged_metrics["total_tokens_avg"] = _safe_div(
                int(merged_metrics.get("total_tokens_total") or 0), int(mc)
            )
            merged_metrics["total_tokens_no_thoughts_avg"] = _safe_div(
                int(merged_metrics.get("total_tokens_no_thoughts_total") or 0), int(mc)
            )
    except Exception:
        pass

    # Latency percentiles from traces if available
    lat = _merge_latency_from_traces(shard_dirs)
    if lat:
        lat_sorted = sorted(lat)
        n = len(lat_sorted)
        merged_metrics["latency_ms_avg"] = float(sum(lat_sorted) / float(n))
        merged_metrics["latency_ms_p50"] = float(lat_sorted[int(0.50 * (n - 1))])
        merged_metrics["latency_ms_p90"] = float(lat_sorted[int(0.90 * (n - 1))])

    _write_json(metrics_out, merged_metrics)

    # Config: base + merge metadata
    merged_cfg: dict[str, Any] = {}
    if isinstance(base_cfg, dict):
        merged_cfg.update(base_cfg)
    merged_cfg["created_at_utc"] = _utc_now_iso()
    merged_cfg["shard_merge"] = {
        "enabled": True,
        "shards_dir": str(shards_root),
        "num_shards_found": int(len(shard_dirs)),
        "shard_names": [shard_name_by_dir[sd] for sd in shard_dirs],
        "merged_predictions": str(preds_out),
    }
    _write_json(cfg_out, merged_cfg)

    # Generate the same companion artifacts as the main evaluator (best-effort).
    try:
        from slow_brain_fast_planner.cli.trajectory_selection import (
            _format_metrics_report,
            _write_metrics_summary_csv,
            _write_metrics_summary_simple_csv,
            _write_metrics_summary_simple_txt,
        )

        (out_dir / "metrics_report.txt").write_text(
            _format_metrics_report(merged_metrics, run_dir=out_dir) + "\n",
            encoding="utf-8",
        )
        _write_metrics_summary_csv(out_dir / "metrics_summary.csv", merged_metrics)
        ms_simple_csv = out_dir / "metrics_summary_simple.csv"
        _write_metrics_summary_simple_csv(ms_simple_csv, merged_metrics)
        _write_metrics_summary_simple_txt(ms_simple_csv, out_dir / "metrics_summary_simple.txt")
    except Exception:
        # If imports fail for some reason, don't block merging.
        pass

    # Write merged predictions.csv for quick inspection.
    try:
        _write_predictions_csv_from_jsonl(preds_csv_out, merged_preds)
    except Exception:
        pass

    # Write a small index page to quickly browse per-shard artifacts.
    try:
        _write_shards_index_html(shards_index_out, shard_dirs=shard_dirs, shards_root=shards_root)
    except Exception:
        pass

    if bool(args.write_report):
        from slow_brain_fast_planner.reporting import generate_run_report_html

        generate_run_report_html(
            run_dir=out_dir, out_path=out_dir / "report.html", embed_images=False
        )

    print(f"[merge_task2_shards] merged run: {out_dir}")
    print(f"[merge_task2_shards] predictions: {preds_out}")
    print(f"[merge_task2_shards] predictions.csv: {preds_csv_out}")
    print(f"[merge_task2_shards] metrics: {metrics_out}")
    print(f"[merge_task2_shards] shard index: {shards_index_out}")
    if bool(args.write_report):
        print(f"[merge_task2_shards] report: {out_dir / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
