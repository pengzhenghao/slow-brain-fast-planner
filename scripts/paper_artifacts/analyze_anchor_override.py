#!/usr/bin/env python3
"""
Analyze "anchor override" experiments.

This script is designed to answer:
1) Top score-argmax index histogram over evaluated snapshots.
2) How different the prelogged candidate trajectories are vs an external anchor library
   (endpoint delta + trajectory delta), evaluated at the chosen indices.
3) Sanity check for "index 0 sentinel" concerns: how often would score-argmax change
   if we dropped candidate 0 and reindexed (i.e., argmax over 1..K-1)?

Typical usage:
  python scripts/analyze_anchor_override.py --run-dir <RUN_DIR>

It will auto-read:
  - <RUN_DIR>/predictions.jsonl
  - <RUN_DIR>/config.json  (to find dataset + anchor npy + mode)

Outputs:
  <out_dir>/per_snapshot.csv
  <out_dir>/score_idx_hist.csv
  <out_dir>/summary.json
  <out_dir>/score_idx_hist.png
  <out_dir>/delta_end_cdf.png
  <out_dir>/delta_traj_cdf.png
  <out_dir>/delta_end_by_idx.png
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_anchor_library(path: Path, *, mode: str) -> np.ndarray:
    """Return (K,T,2) float64."""
    arr = np.load(path)
    if arr.ndim == 3 and int(arr.shape[-1]) == 2:
        trajs = np.asarray(arr, dtype=np.float64)
    elif arr.ndim == 2 and int(arr.shape[1]) % 2 == 0:
        t = int(arr.shape[1]) // 2
        trajs = np.asarray(arr, dtype=np.float64).reshape(int(arr.shape[0]), int(t), 2)
    else:
        raise ValueError(f"Unsupported anchor npy shape: {getattr(arr, 'shape', None)}")

    if str(mode) == "notebook_v1":
        trajs = trajs.copy()
        trajs[:, :, 0] = trajs[:, :, 0] * 0.51
        trajs[:, :, 1] = trajs[:, :, 1] * 0.32 - 0.16
        trajs[:, :, 0] = trajs[:, :, 0].cumsum(axis=1)
        trajs[:, :, 1] = trajs[:, :, 1].cumsum(axis=1)
    elif str(mode) == "xy":
        trajs = np.asarray(trajs, dtype=np.float64)
    else:
        raise ValueError(f"Unknown mode: {mode} (expected notebook_v1|xy)")
    return trajs


def _traj_delta_metrics(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (mean_L2_over_time, max_L2_over_time)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = int(min(a.shape[0], b.shape[0]))
    if n <= 0:
        return float("nan"), float("nan")
    d = np.linalg.norm(a[:n, :2] - b[:n, :2], axis=1)
    return float(np.mean(d)), float(np.max(d))


def _cdf_plot(ax, values: np.ndarray, *, label: str) -> None:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    v = np.sort(v)
    y = np.linspace(0.0, 1.0, num=int(v.size), endpoint=True)
    ax.plot(v, y, lw=2.0, label=label)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Merged Task2 run dir (contains predictions.jsonl, config.json).",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <run_dir>/anchor_override_analysis)",
    )
    ap.add_argument(
        "--dataset", default=None, help="Override dataset root (default: from config.json)."
    )
    ap.add_argument(
        "--anchor-npy", default=None, help="Override anchor npy path (default: from config.json)."
    )
    ap.add_argument(
        "--anchor-mode", default=None, choices=["notebook_v1", "xy"], help="Override anchor mode."
    )
    ap.add_argument("--max-snapshots", type=int, default=None, help="Optional cap for debugging.")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    cfg_path = run_dir / "config.json"
    pred_path = run_dir / "predictions.jsonl"
    if not cfg_path.exists():
        raise SystemExit(f"Missing config.json: {cfg_path}")
    if not pred_path.exists():
        raise SystemExit(f"Missing predictions.jsonl: {pred_path}")

    cfg = _read_json(cfg_path)
    dataset_root = Path(args.dataset or cfg.get("dataset") or "").resolve()
    if not dataset_root.exists():
        raise SystemExit(f"Dataset not found: {dataset_root}")

    # Anchor library info (optional: the run may not be an override run; we still allow analysis).
    cand_cfg = (cfg.get("candidates") or {}) if isinstance(cfg, dict) else {}
    anchor_npy = args.anchor_npy or (
        cand_cfg.get("candidate_points_override_npy") if isinstance(cand_cfg, dict) else None
    )
    anchor_mode = args.anchor_mode or (
        cand_cfg.get("candidate_points_override_mode") if isinstance(cand_cfg, dict) else None
    )
    anchor_mode = str(anchor_mode or "notebook_v1")

    anchor_trajs = None
    if anchor_npy:
        anchor_path = Path(str(anchor_npy)).resolve()
        if not anchor_path.exists():
            raise SystemExit(f"Anchor npy not found: {anchor_path}")
        anchor_trajs = _load_anchor_library(anchor_path, mode=anchor_mode)

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (run_dir / "anchor_override_analysis").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read evaluated snapshots from predictions (this matches the exact set used by the run).
    snaps: list[tuple[str, float, dict[str, Any]]] = []
    for obj in _iter_jsonl(pred_path):
        if not isinstance(obj, dict):
            continue
        if obj.get("skipped") is True or obj.get("skip_reason") is not None:
            continue
        eid = obj.get("episode_id")
        tt = obj.get("t")
        if not isinstance(eid, str) or not isinstance(tt, (int, float)):
            continue
        snaps.append((str(eid), float(tt), obj))
        if args.max_snapshots is not None and len(snaps) >= int(args.max_snapshots):
            break
    if not snaps:
        raise SystemExit("No evaluated snapshots found in predictions.jsonl")

    # Group needed times by episode for efficient lookup.
    need_times_by_ep: dict[str, set[float]] = defaultdict(set)
    for eid, tt, _ in snaps:
        need_times_by_ep[str(eid)].add(round(float(tt), 6))

    # Load only needed planner_candidates records.
    # We only need per time: candidate scores + candidate points for selected indices.
    pc_cache: dict[tuple[str, float], dict[str, Any]] = {}
    for eid, times in sorted(need_times_by_ep.items(), key=lambda x: x[0]):
        pc_path = (dataset_root / "episodes" / str(eid) / "planner_candidates.jsonl").resolve()
        if not pc_path.exists():
            continue
        wanted = set(times)
        for rec in _iter_jsonl(pc_path):
            if not isinstance(rec, dict):
                continue
            t0 = rec.get("t")
            if not isinstance(t0, (int, float)):
                continue
            tr = round(float(t0), 6)
            if tr not in wanted:
                continue
            pc_cache[(str(eid), float(tr))] = rec
            if len(pc_cache) >= len(snaps):
                # best-effort early exit; ok even if missing some episodes
                pass

    # Compute diagnostics.
    score_idx_hist: Counter[int] = Counter()
    pred_idx_hist: Counter[int] = Counter()
    argmax_drop0_changed = 0
    argmax_drop0_total = 0

    rows: list[dict[str, Any]] = []

    for eid, tt, pred_obj in snaps:
        tr = round(float(tt), 6)
        rec = pc_cache.get((str(eid), float(tr)))
        if not isinstance(rec, dict):
            continue
        cands = rec.get("candidates") or []
        if not isinstance(cands, list) or not cands:
            continue
        scores = []
        for c in cands:
            try:
                scores.append(float((c or {}).get("score")))
            except Exception:
                scores.append(float("nan"))
        K = len(scores)
        if K <= 0 or not np.all(np.isfinite(np.asarray(scores, dtype=np.float64))):
            continue
        score_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        score_idx_hist[score_idx] += 1

        # If we "drop index 0 sentinel", argmax is computed over scores[1:] then shifted back by +1.
        if K >= 2:
            argmax_drop0 = int(np.argmax(np.asarray(scores[1:], dtype=np.float64))) + 1
            argmax_drop0_total += 1
            if int(argmax_drop0) != int(score_idx):
                argmax_drop0_changed += 1

        pred_idx = pred_obj.get("prediction")
        pred_idx = int(pred_idx) if isinstance(pred_idx, (int, float)) else None
        if pred_idx is not None:
            pred_idx_hist[int(pred_idx)] += 1

        # Endpoint/trajectory deltas vs anchor library (if provided).
        end_delta_m = None
        traj_delta_mean_m = None
        traj_delta_max_m = None
        if anchor_trajs is not None and 0 <= int(score_idx) < int(anchor_trajs.shape[0]):
            pts = (cands[int(score_idx)] or {}).get("points_xy") or []
            if isinstance(pts, list) and len(pts) >= 2:
                a = np.asarray(
                    [
                        [float(p[0]), float(p[1])]
                        for p in pts
                        if isinstance(p, (list, tuple)) and len(p) >= 2
                    ],
                    dtype=np.float64,
                )
                b = np.asarray(anchor_trajs[int(score_idx)], dtype=np.float64)
                if a.size and b.size:
                    # Align horizon length
                    n = int(min(a.shape[0], b.shape[0]))
                    a2 = a[:n, :2]
                    b2 = b[:n, :2]
                    end_delta_m = float(np.linalg.norm(a2[-1] - b2[-1]))
                    traj_delta_mean_m, traj_delta_max_m = _traj_delta_metrics(a2, b2)

        ade = pred_obj.get("ade") if isinstance(pred_obj, dict) else None
        ade_min_all = None
        ade_score = None
        ade_selected = None
        if isinstance(ade, dict):
            for k, var in (
                ("min_all", "ade_min_all"),
                ("score", "ade_score"),
                ("selected", "ade_selected"),
            ):
                v = ade.get(k)
                if isinstance(v, (int, float)):
                    if var == "ade_min_all":
                        ade_min_all = float(v)
                    elif var == "ade_score":
                        ade_score = float(v)
                    else:
                        ade_selected = float(v)

        rows.append(
            {
                "episode_id": str(eid),
                "t": float(tt),
                "num_candidates": int(K),
                "score_idx": int(score_idx),
                "pred_idx": (int(pred_idx) if pred_idx is not None else ""),
                "score_val": float(scores[int(score_idx)]),
                "argmax_drop0_idx": (int(np.argmax(np.asarray(scores[1:], dtype=np.float64))) + 1)
                if K >= 2
                else "",
                "argmax_drop0_changed": (
                    1
                    if (
                        K >= 2
                        and int(score_idx)
                        != (int(np.argmax(np.asarray(scores[1:], dtype=np.float64))) + 1)
                    )
                    else 0
                ),
                "end_delta_m": (end_delta_m if end_delta_m is not None else ""),
                "traj_delta_mean_m": (traj_delta_mean_m if traj_delta_mean_m is not None else ""),
                "traj_delta_max_m": (traj_delta_max_m if traj_delta_max_m is not None else ""),
                "ade_min_all": (ade_min_all if ade_min_all is not None else ""),
                "ade_score": (ade_score if ade_score is not None else ""),
                "ade_selected": (ade_selected if ade_selected is not None else ""),
            }
        )

    if not rows:
        raise SystemExit("No rows produced (missing planner_candidates lookups?)")

    # Write per-snapshot CSV.
    csv_path = out_dir / "per_snapshot.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write histogram CSV.
    hist_path = out_dir / "score_idx_hist.csv"
    with hist_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "count"])
        w.writeheader()
        for idx, cnt in score_idx_hist.most_common():
            w.writerow({"index": int(idx), "count": int(cnt)})

    # Build numpy arrays for plots.
    end_d = np.asarray(
        [float(r["end_delta_m"]) for r in rows if str(r.get("end_delta_m", "")).strip() != ""],
        dtype=np.float64,
    )
    traj_d = np.asarray(
        [
            float(r["traj_delta_mean_m"])
            for r in rows
            if str(r.get("traj_delta_mean_m", "")).strip() != ""
        ],
        dtype=np.float64,
    )

    # Summary JSON.
    summary = {
        "run_dir": str(run_dir),
        "dataset": str(dataset_root),
        "num_snapshots": int(len(rows)),
        "anchor_npy": str(anchor_npy) if anchor_npy else None,
        "anchor_mode": str(anchor_mode),
        "score_idx_top10": [(int(i), int(c)) for i, c in score_idx_hist.most_common(10)],
        "pred_idx_top10": [(int(i), int(c)) for i, c in pred_idx_hist.most_common(10)],
        "argmax_drop0_changed": int(argmax_drop0_changed),
        "argmax_drop0_total": int(argmax_drop0_total),
        "argmax_drop0_change_rate": (float(argmax_drop0_changed) / float(argmax_drop0_total))
        if argmax_drop0_total
        else None,
        "end_delta_m": {
            "count": int(end_d.size),
            "mean": float(np.mean(end_d)) if end_d.size else None,
            "p50": float(np.quantile(end_d, 0.5)) if end_d.size else None,
            "p90": float(np.quantile(end_d, 0.9)) if end_d.size else None,
            "max": float(np.max(end_d)) if end_d.size else None,
        },
        "traj_delta_mean_m": {
            "count": int(traj_d.size),
            "mean": float(np.mean(traj_d)) if traj_d.size else None,
            "p50": float(np.quantile(traj_d, 0.5)) if traj_d.size else None,
            "p90": float(np.quantile(traj_d, 0.9)) if traj_d.size else None,
            "max": float(np.max(traj_d)) if traj_d.size else None,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Plots (matplotlib is an optional dependency in many envs, but present in this repo).
    try:
        import matplotlib.pyplot as plt

        # 1) Score-idx histogram (top 20 + other).
        topn = 20
        items = score_idx_hist.most_common(topn)
        idxs = [i for i, _ in items]
        counts = [c for _, c in items]
        other = int(sum(score_idx_hist.values()) - sum(counts))
        if other > 0:
            idxs = idxs + [-1]
            counts = counts + [other]
        labels = [str(i) if i >= 0 else "other" for i in idxs]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(counts)), counts, color="#4C78A8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_xlabel("score argmax index (top20) / other")
        ax.set_ylabel("count")
        ax.set_title("Task2: score-argmax index histogram (evaluated snapshots)")
        fig.tight_layout()
        fig.savefig(out_dir / "score_idx_hist.png", dpi=160)
        plt.close(fig)

        # 2) CDFs for endpoint delta + trajectory delta (mean).
        fig, ax = plt.subplots(figsize=(6, 4))
        _cdf_plot(ax, end_d, label="endpoint delta (m)")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("delta (m)")
        ax.set_ylabel("CDF")
        ax.set_title("prelogged vs anchor: endpoint delta CDF")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / "delta_end_cdf.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        _cdf_plot(ax, traj_d, label="trajectory delta mean (m)")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("delta (m)")
        ax.set_ylabel("CDF")
        ax.set_title("prelogged vs anchor: trajectory mean delta CDF")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(out_dir / "delta_traj_cdf.png", dpi=160)
        plt.close(fig)

        # 3) Mean endpoint delta by index (top 30 indices by frequency).
        by_idx: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            if str(r.get("end_delta_m", "")).strip() == "":
                continue
            by_idx[int(r["score_idx"])].append(float(r["end_delta_m"]))
        top = [i for i, _ in score_idx_hist.most_common(30)]
        means = [
            float(np.mean(np.asarray(by_idx[i], dtype=np.float64)))
            if by_idx.get(i)
            else float("nan")
            for i in top
        ]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(top)), means, color="#F58518")
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels([str(i) for i in top], rotation=0)
        ax.set_xlabel("score argmax index (top30)")
        ax.set_ylabel("mean endpoint delta (m)")
        ax.set_title("prelogged vs anchor: mean endpoint delta by frequent indices")
        fig.tight_layout()
        fig.savefig(out_dir / "delta_end_by_idx.png", dpi=160)
        plt.close(fig)
    except Exception:
        # Best-effort: analysis CSV/JSON still exist.
        pass

    print(f"[ok] wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
