#!/usr/bin/env python3
"""
Sweep hard-limit thresholds for Task1 gating and compute budget-ADE curve.

This script evaluates the "planner confidence" baseline for Task1 gating:
- Vary top1_prob_max and top1_top2_margin_max thresholds
- For each threshold, compute query_rate and resulting wrapper ADE
- Output a CSV with (threshold params, query_rate, mean_ade) for plotting

The key insight: the hard-limit gate triggers when:
  (top1_prob <= top1_prob_max) OR (top1_prob - top2_prob <= top1_top2_margin_max)

Lower thresholds = more selective (fewer queries), higher thresholds = more queries.

Usage:
    python scripts/sweep_hard_limit_budget_ade.py \
        --task2-predictions logs/0120_T2_real_policy=gemini-3-flash.../predictions.jsonl \
        --dataset data_ssd/2026-01-15_human_data_processed \
        --out logs/hard_limit_sweep/budget_curve.csv

    # Or with pre-merged joint analysis CSV:
    python scripts/sweep_hard_limit_budget_ade.py \
        --merged-csv logs/.../joint_analysis/merged_task1_task2.csv \
        --dataset data_ssd/2026-01-15_human_data_processed \
        --out logs/hard_limit_sweep/budget_curve.csv

Author: Auto-generated for budget-ADE baseline analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Load a CSV file as list of dicts."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def compute_hard_limit_hit(
    candidate_probs: list[float],
    *,
    top1_prob_max: float,
    top1_top2_margin_max: float,
    min_candidates: int = 2,
) -> bool:
    """
    Deterministic hard query trigger based on candidate probability peakedness.

    Triggers request if:
      (top1_prob <= top1_prob_max) OR (margin <= top1_top2_margin_max)
    """
    if len(candidate_probs) < min_candidates:
        return False

    probs = sorted(candidate_probs, reverse=True)
    top1 = probs[0]
    top2 = probs[1] if len(probs) > 1 else 0.0
    margin = top1 - top2

    return (top1 <= top1_prob_max) or (margin <= top1_top2_margin_max)


def load_all_planner_candidates_for_episodes(
    dataset_root: Path,
    episode_ids: set[str],
    *,
    topk: int = 12,
    temperature: float = 1.0,
) -> dict[tuple[str, float], list[float]]:
    """
    Batch-load candidate probabilities from planner_candidates.jsonl for all episodes.

    Returns: dict mapping (episode_id, t_rounded) -> probs
    """
    result: dict[tuple[str, float], list[float]] = {}

    for eid in sorted(episode_ids):
        pc_path = dataset_root / "episodes" / eid / "planner_candidates.jsonl"
        if not pc_path.exists():
            continue

        with pc_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    r = json.loads(s)
                except Exception:
                    continue

                t = r.get("t")
                if t is None:
                    continue

                candidates = r.get("candidates", [])
                if not candidates:
                    continue

                # Extract scores
                scores = []
                for c in candidates:
                    if isinstance(c, dict):
                        sc = c.get("score")
                        if sc is not None:
                            try:
                                scores.append(float(sc))
                            except Exception:
                                pass

                if len(scores) < 2:
                    continue

                # Sort by score descending and take top-k
                scores_sorted = sorted(scores, reverse=True)[:topk]

                # Softmax
                scores_arr = np.array(scores_sorted) / temperature
                probs = softmax(scores_arr)

                t_key = round(float(t), 2)
                result[(eid, t_key)] = probs.tolist()

    return result


def load_task2_predictions(task2_path: Path) -> list[dict[str, Any]]:
    """Load Task2 predictions and extract ADEs."""
    records = _read_jsonl(task2_path)

    rows: list[dict[str, Any]] = []
    for r in records:
        episode_id = r.get("episode_id")
        t = r.get("t")
        ade = r.get("ade", {})

        if episode_id is None or t is None:
            continue

        rows.append(
            {
                "episode_id": str(episode_id),
                "t": float(t),
                "ade_score": float(ade.get("score", np.nan)),  # Planner argmax ADE
                "ade_selected": float(ade.get("selected", np.nan)),  # VLM selected ADE
                "ade_min": float(ade.get("min", np.nan)),  # Oracle
            }
        )

    return rows


def load_from_merged_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load from a pre-merged joint analysis CSV."""
    records = _read_csv(csv_path)

    rows: list[dict[str, Any]] = []
    for r in records:
        episode_id = r.get("episode_id")
        t = r.get("t")

        if episode_id is None or t is None:
            continue

        try:
            rows.append(
                {
                    "episode_id": str(episode_id),
                    "t": float(t),
                    "ade_score": float(r.get("ade_score", np.nan)),
                    "ade_selected": float(r.get("ade_selected", np.nan)),
                    "ade_min": float(r.get("ade_min", np.nan)),
                }
            )
        except Exception:
            continue

    return rows


def sweep_hard_limit(
    rows: list[dict[str, Any]],
    *,
    top1_prob_values: list[float],
    margin_values: list[float],
) -> list[dict[str, Any]]:
    """
    Sweep hard-limit thresholds and compute (query_rate, mean_ade) for each.

    Returns a list of dicts with sweep results.
    """
    results: list[dict[str, Any]] = []

    for p_max in top1_prob_values:
        for m_max in margin_values:
            # For each snapshot, determine if we'd query
            gated_ade: list[float] = []
            n_query = 0
            n_valid = 0

            for r in rows:
                probs = r.get("candidate_probs", [])
                ade_score = r.get("ade_score", np.nan)
                ade_selected = r.get("ade_selected", np.nan)

                if len(probs) < 2:
                    continue

                n_valid += 1
                hit = compute_hard_limit_hit(
                    probs,
                    top1_prob_max=p_max,
                    top1_top2_margin_max=m_max,
                )

                if hit:
                    n_query += 1
                    gated_ade.append(float(ade_selected))
                else:
                    gated_ade.append(float(ade_score))

            if n_valid == 0:
                continue

            query_rate = float(n_query) / float(n_valid)
            mean_ade = float(np.nanmean(gated_ade)) if gated_ade else float("nan")

            results.append(
                {
                    "top1_prob_max": p_max,
                    "margin_max": m_max,
                    "query_rate": query_rate,
                    "n_query": n_query,
                    "n_valid": n_valid,
                    "mean_ade": mean_ade,
                }
            )

    return results


def compute_reference_points(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute reference ADE values (planner-only, VLM-only, oracle)."""
    ade_scores = [r["ade_score"] for r in rows if np.isfinite(r.get("ade_score", np.nan))]
    ade_selected = [r["ade_selected"] for r in rows if np.isfinite(r.get("ade_selected", np.nan))]
    ade_min = [r["ade_min"] for r in rows if np.isfinite(r.get("ade_min", np.nan))]

    return {
        "planner_argmax_ade": float(np.mean(ade_scores)) if ade_scores else float("nan"),
        "vlm_100pct_ade": float(np.mean(ade_selected)) if ade_selected else float("nan"),
        "oracle_ade": float(np.mean(ade_min)) if ade_min else float("nan"),
        "n_snapshots": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep hard-limit thresholds for budget-ADE curve")
    parser.add_argument(
        "--task2-predictions",
        type=Path,
        default=None,
        help="Path to Task2 predictions.jsonl",
    )
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=None,
        help="Path to merged_task1_task2.csv (alternative to --task2-predictions)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Dataset root (for loading planner_candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: stdout)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=12,
        help="Top-k candidates to consider for probability calculation (default: 12)",
    )
    parser.add_argument(
        "--top1-prob-values",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
        help="Comma-separated top1_prob_max thresholds to sweep",
    )
    parser.add_argument(
        "--margin-values",
        type=str,
        default="0.00,0.02,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.35,0.40,0.50",
        help="Comma-separated margin thresholds to sweep",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a budget curve plot (PNG)",
    )

    args = parser.parse_args()

    if args.merged_csv is None and args.task2_predictions is None:
        print("ERROR: Must provide either --merged-csv or --task2-predictions", flush=True)
        return 1

    # Parse threshold values
    p_values = [float(x.strip()) for x in args.top1_prob_values.split(",") if x.strip()]
    m_values = [float(x.strip()) for x in args.margin_values.split(",") if x.strip()]

    # Load Task2 predictions or merged CSV
    if args.merged_csv:
        print(f"Loading from merged CSV: {args.merged_csv}", flush=True)
        rows = load_from_merged_csv(args.merged_csv)
    else:
        print(f"Loading Task2 predictions from: {args.task2_predictions}", flush=True)
        rows = load_task2_predictions(args.task2_predictions)
    print(f"  Loaded {len(rows)} snapshots", flush=True)

    # Load candidate probabilities from dataset (batch by episode for efficiency)
    print(f"Loading candidate probabilities from: {args.dataset}", flush=True)
    episode_ids = {r["episode_id"] for r in rows}
    print(f"  Found {len(episode_ids)} unique episodes", flush=True)

    probs_cache = load_all_planner_candidates_for_episodes(
        args.dataset,
        episode_ids,
        topk=args.topk,
    )
    print(f"  Loaded {len(probs_cache)} timestep probabilities", flush=True)

    n_valid_probs = 0
    for r in rows:
        t_key = round(float(r["t"]), 2)
        probs = probs_cache.get((r["episode_id"], t_key), [])
        r["candidate_probs"] = probs
        if len(probs) >= 2:
            n_valid_probs += 1

    print(
        f"  {n_valid_probs}/{len(rows)} snapshots matched with candidate probabilities", flush=True
    )

    if n_valid_probs == 0:
        print("ERROR: No snapshots with candidate probabilities found!", flush=True)
        return 1

    # Compute reference points
    refs = compute_reference_points(rows)
    print("\nReference points:", flush=True)
    print(f"  Planner argmax (0% query): {refs['planner_argmax_ade']:.4f}m", flush=True)
    print(f"  VLM (100% query): {refs['vlm_100pct_ade']:.4f}m", flush=True)
    print(f"  Oracle: {refs['oracle_ade']:.4f}m", flush=True)

    # Sweep
    print(
        f"\nSweeping {len(p_values)} x {len(m_values)} = "
        f"{len(p_values) * len(m_values)} threshold combinations...",
        flush=True,
    )
    results = sweep_hard_limit(rows, top1_prob_values=p_values, margin_values=m_values)
    print(f"  Generated {len(results)} result rows", flush=True)

    # Sort by query_rate for easier plotting
    results.sort(key=lambda r: (r["query_rate"], r["mean_ade"]))

    # Dedupe to get Pareto-optimal frontier (for each unique query_rate, keep lowest ADE)
    frontier: dict[float, dict] = {}
    for r in results:
        qr = round(r["query_rate"], 4)
        if qr not in frontier or r["mean_ade"] < frontier[qr]["mean_ade"]:
            frontier[qr] = r
    frontier_rows = sorted(frontier.values(), key=lambda r: r["query_rate"])

    # Write CSV
    cols = ["query_rate", "n_query", "n_valid", "mean_ade", "top1_prob_max", "margin_max"]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in frontier_rows:
                w.writerow(r)
        print(f"\nWrote frontier points to: {args.out}", flush=True)

        # Also write full sweep
        full_out = args.out.parent / f"{args.out.stem}_full.csv"
        with full_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in results:
                w.writerow(r)
        print(f"Wrote all sweep points to: {full_out}", flush=True)
    else:
        import sys

        w = csv.DictWriter(sys.stdout, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in frontier_rows:
            w.writerow(r)

    # Plot if requested
    if args.plot and args.out:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            qrates = [r["query_rate"] * 100 for r in frontier_rows]
            ades = [r["mean_ade"] for r in frontier_rows]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                qrates,
                ades,
                "o-",
                linewidth=2,
                markersize=6,
                color="#ff00aa",
                label="Hard-limit baseline (Pareto frontier)",
            )

            # Reference lines
            ax.axhline(
                y=refs["planner_argmax_ade"],
                color="red",
                linestyle="--",
                label=f"Planner Argmax: {refs['planner_argmax_ade']:.3f}m",
            )
            ax.axhline(
                y=refs["vlm_100pct_ade"],
                color="blue",
                linestyle="--",
                label=f"VLM (100%): {refs['vlm_100pct_ade']:.3f}m",
            )
            ax.axhline(
                y=refs["oracle_ade"],
                color="green",
                linestyle="--",
                label=f"Oracle: {refs['oracle_ade']:.3f}m",
            )

            ax.set_xlabel("Query Rate (%)", fontsize=12)
            ax.set_ylabel("Mean ADE (m)", fontsize=12)
            ax.set_title("Hard-Limit Baseline: Budget-ADE Curve", fontsize=14)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-5, 105)

            plot_path = args.out.parent / f"{args.out.stem}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved plot to: {plot_path}", flush=True)
        except ImportError:
            print("WARNING: matplotlib not available, skipping plot", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
