#!/usr/bin/env python3
"""Generate paper figures for Section 5: Closed-Loop Simulation.

This script reads results from the section5_closed_loop experiments and generates:
1. Figure A: Policy comparison bar chart (main result)
2. Figure B: Delay robustness curves
3. Figure C: Performance vs trajectory switches scatter (smoothness trade-off)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(run_dir: Path) -> pd.DataFrame:
    """Load results.csv from a run directory."""
    run_dir = Path(run_dir)

    # Case 1: direct run dir
    results_csv = run_dir / "results.csv"
    if results_csv.exists():
        return pd.read_csv(results_csv)

    # Case 2: common layout: <parent>/<closed_loop_fusion_*/results.csv>
    candidates = [p.parent for p in run_dir.glob("closed_loop_fusion_*/results.csv")]

    # Case 3: step folder layout: recurse for closed_loop_fusion_*/results.csv
    if not candidates:
        candidates = [p.parent for p in run_dir.rglob("closed_loop_fusion_*/results.csv")]

    if not candidates:
        raise FileNotFoundError(f"No results.csv found under {run_dir}")

    latest_by_parent = {}
    for candidate in candidates:
        parent = candidate.parent
        current = latest_by_parent.get(parent)
        if current is None or candidate.name > current.name:
            latest_by_parent[parent] = candidate

    frames = []
    for candidate in sorted(latest_by_parent.values(), key=lambda p: str(p)):
        frame = pd.read_csv(candidate / "results.csv")
        frame["source_run_dir"] = str(candidate)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate_results(df: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """Aggregate results by policy and relevant parameters."""
    agg_dict = {
        "mean_cte_m": ["mean", "std"],
        "p95_cte_m": ["mean"],
        "mean_speed_mps": ["mean"],
        "chosen_switches": ["mean"],
    }

    # Add count
    grouped = df.groupby(groupby_cols).agg(agg_dict)
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    grouped["n"] = df.groupby(groupby_cols).size().values

    return grouped


def plot_policy_comparison(df: pd.DataFrame, out_dir: Path):
    """Figure A: Bar chart comparing policies at fixed delay."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot_policy_comparison")
        return

    # Find best lambda for each policy (lowest CTE)
    best_per_policy = []
    for policy in df["policy"].unique():
        policy_df = df[df["policy"] == policy]
        if "lambda_sim" in policy_df.columns:
            # For fusion policies, find best lambda
            agg = policy_df.groupby("lambda_sim")["mean_cte_m"].mean()
            best_lambda = agg.idxmin()
            best_row = policy_df[policy_df["lambda_sim"] == best_lambda].copy()
        else:
            best_row = policy_df.copy()

        best_row = (
            best_row.groupby("policy")
            .agg(
                {
                    "mean_cte_m": ["mean", "std"],
                    "chosen_switches": "mean",
                }
            )
            .reset_index()
        )
        best_row.columns = ["policy", "mean_cte", "std_cte", "switches"]
        best_per_policy.append(best_row)

    summary = pd.concat(best_per_policy, ignore_index=True)

    # Order policies logically
    policy_order = [
        "planner_oracle",
        "local_only",
        "vlm_hold",
        "vlm_stream",
        "vlm_hold_match",
        "vlm_stream_match",
        "score_fusion",
        "prob_fusion",
        "score_fusion_stream",
        "prob_fusion_stream",
    ]
    policy_label_map = {
        "planner_oracle": "Planner Oracle",
        "local_only": "Local Only",
        "vlm_hold": "VLM Hold (direct)",
        "vlm_stream": "VLM Stream (direct)",
        "vlm_hold_match": "VLM Hold Match",
        "vlm_stream_match": "VLM Stream Match",
        "score_fusion": "Score Fusion (hold)",
        "prob_fusion": "Prob Fusion (hold)",
        "score_fusion_stream": "Score Fusion (stream)",
        "prob_fusion_stream": "Prob Fusion (stream)",
    }
    summary["policy_order"] = summary["policy"].map({p: i for i, p in enumerate(policy_order)})
    summary = summary.sort_values("policy_order")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(summary))
    color_map = {
        "planner_oracle": "#4d4d4d",
        "local_only": "#d62728",
        "vlm_hold": "#ff7f0e",
        "vlm_stream": "#2ca02c",
        "vlm_hold_match": "#8c564b",
        "vlm_stream_match": "#17becf",
        "score_fusion": "#1f77b4",
        "prob_fusion": "#9467bd",
        "score_fusion_stream": "#aec7e8",
        "prob_fusion_stream": "#c5b0d5",
    }
    bars = ax.bar(
        x,
        summary["mean_cte"],
        yerr=summary["std_cte"],
        capsize=4,
        color=[color_map.get(p, "gray") for p in summary["policy"]],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [policy_label_map.get(p, str(p)) for p in summary["policy"]], rotation=15, ha="right"
    )
    ax.set_ylabel("Mean Cross-Track Error (m)")
    delay_values = (
        sorted(pd.to_numeric(df["delay_s"], errors="coerce").dropna().unique().tolist())
        if "delay_s" in df.columns
        else []
    )
    delay_label = f"{delay_values[0]:g}s" if len(delay_values) == 1 else "fixed delay"
    ax.set_title(f"Policy Comparison (delay = {delay_label})")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, summary["mean_cte"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"figA_policy_comparison.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved: figA_policy_comparison.pdf/png")

    # Save summary table
    summary.to_csv(out_dir / "policy_comparison_summary.csv", index=False)


def plot_delay_curves(step2_dir: Path, out_dir: Path):
    """Figure B: CTE vs delay for each policy."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot_delay_curves")
        return

    # Collect results from all delay subdirectories
    all_results = []
    for delay_dir in sorted(step2_dir.glob("delay_*")):
        delay_str = delay_dir.name.replace("delay_", "")
        try:
            delay_val = float(delay_str)
            df = load_results(delay_dir)
            df["delay_sweep"] = delay_val
            all_results.append(df)
        except Exception as e:
            print(f"  Warning: Could not load {delay_dir}: {e}")

    if not all_results:
        print("  No delay sweep results found, skipping plot_delay_curves")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # For fusion policies, use best lambda (or lambda=1.0)
    def get_best_cte(group):
        if "lambda_sim" in group.columns and group["lambda_sim"].nunique() > 1:
            # Use lambda=1.0 or 2.0 for consistency
            subset = group[group["lambda_sim"].isin([1.0, 2.0])]
            if len(subset) > 0:
                return subset.groupby("delay_sweep")["mean_cte_m"].mean().min()
        return group["mean_cte_m"].mean()

    # Aggregate by policy and delay
    agg = (
        combined.groupby(["policy", "delay_sweep"])
        .agg(
            {
                "mean_cte_m": "mean",
                "chosen_switches": "mean",
            }
        )
        .reset_index()
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    policy_colors = {
        "planner_oracle": "#4d4d4d",
        "local_only": "#d62728",
        "vlm_hold": "#ff7f0e",
        "vlm_stream": "#2ca02c",
        "vlm_hold_match": "#8c564b",
        "vlm_stream_match": "#17becf",
        "score_fusion": "#1f77b4",
        "prob_fusion": "#9467bd",
        "score_fusion_stream": "#aec7e8",
        "prob_fusion_stream": "#c5b0d5",
    }
    policy_labels = {
        "planner_oracle": "Planner Oracle",
        "local_only": "Local Only",
        "vlm_hold": "VLM Hold (direct)",
        "vlm_stream": "VLM Stream (direct)",
        "vlm_hold_match": "VLM Hold Match",
        "vlm_stream_match": "VLM Stream Match",
        "score_fusion": "Score Fusion (hold)",
        "prob_fusion": "Prob Fusion (hold)",
        "score_fusion_stream": "Score Fusion (stream)",
        "prob_fusion_stream": "Prob Fusion (stream)",
    }

    for policy in [
        "planner_oracle",
        "local_only",
        "vlm_hold",
        "vlm_stream",
        "vlm_hold_match",
        "vlm_stream_match",
        "score_fusion",
        "prob_fusion",
        "score_fusion_stream",
        "prob_fusion_stream",
    ]:
        policy_df = agg[agg["policy"] == policy].sort_values("delay_sweep")
        if len(policy_df) == 0:
            continue

        ax1.plot(
            policy_df["delay_sweep"],
            policy_df["mean_cte_m"],
            "o-",
            label=policy_labels.get(policy, policy),
            color=policy_colors.get(policy, "gray"),
            linewidth=2,
            markersize=6,
        )

        ax2.plot(
            policy_df["delay_sweep"],
            policy_df["chosen_switches"],
            "o-",
            label=policy_labels.get(policy, policy),
            color=policy_colors.get(policy, "gray"),
            linewidth=2,
            markersize=6,
        )

    ax1.set_xlabel("VLM Delay (s)")
    ax1.set_ylabel("Mean Cross-Track Error (m)")
    ax1.set_title("(A) Tracking Accuracy vs Delay")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("VLM Delay (s)")
    ax2.set_ylabel("Trajectory Switches per Episode")
    ax2.set_title("(B) Trajectory Smoothness vs Delay")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"figB_delay_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved: figB_delay_curves.pdf/png")


def plot_cte_vs_switches(df: pd.DataFrame, out_dir: Path):
    """Figure C: CTE vs trajectory switches scatter (Pareto front)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot_cte_vs_switches")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    policy_colors = {
        "planner_oracle": "#4d4d4d",
        "local_only": "#d62728",
        "vlm_hold": "#ff7f0e",
        "vlm_stream": "#2ca02c",
        "vlm_hold_match": "#8c564b",
        "vlm_stream_match": "#17becf",
        "score_fusion": "#1f77b4",
        "prob_fusion": "#9467bd",
        "score_fusion_stream": "#aec7e8",
        "prob_fusion_stream": "#c5b0d5",
    }
    policy_markers = {
        "planner_oracle": "o",
        "local_only": "o",
        "vlm_hold": "s",
        "vlm_stream": "^",
        "vlm_hold_match": "X",
        "vlm_stream_match": "*",
        "score_fusion": "D",
        "prob_fusion": "p",
        "score_fusion_stream": "d",
        "prob_fusion_stream": "P",
    }

    for policy in df["policy"].unique():
        policy_df = df[df["policy"] == policy]
        ax.scatter(
            policy_df["chosen_switches"],
            policy_df["mean_cte_m"],
            c=policy_colors.get(policy, "gray"),
            marker=policy_markers.get(policy, "o"),
            label=policy,
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Trajectory Switches per Episode")
    ax.set_ylabel("Mean Cross-Track Error (m)")
    ax.set_title("Tracking Accuracy vs Smoothness Trade-off")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Add annotation for ideal region
    ax.annotate(
        "Better",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        fontsize=12,
        color="green",
        fontweight="bold",
    )

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"figC_cte_vs_switches.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    print("  Saved: figC_cte_vs_switches.pdf/png")


def generate_summary_table(df: pd.DataFrame, out_dir: Path):
    """Generate a summary table for the paper."""
    # Find best config per policy
    summary_rows = []

    for policy in df["policy"].unique():
        policy_df = df[df["policy"] == policy]

        if "lambda_sim" in policy_df.columns:
            # For fusion policies, group by lambda and find best
            agg = policy_df.groupby("lambda_sim").agg(
                {
                    "mean_cte_m": "mean",
                    "chosen_switches": "mean",
                }
            )
            best_lambda = agg["mean_cte_m"].idxmin()
            best_cte = agg.loc[best_lambda, "mean_cte_m"]
            best_switches = agg.loc[best_lambda, "chosen_switches"]
        else:
            best_lambda = None
            best_cte = policy_df["mean_cte_m"].mean()
            best_switches = policy_df["chosen_switches"].mean()

        summary_rows.append(
            {
                "Policy": policy,
                "Best λ": best_lambda if best_lambda else "-",
                "Mean CTE (m)": f"{best_cte:.3f}",
                "Switches": f"{best_switches:.1f}",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_table.csv", index=False)

    # Also write as markdown
    with open(out_dir / "summary_table.md", "w") as f:
        f.write("# Section 5: Closed-Loop Simulation Results\n\n")
        columns = [str(column) for column in summary_df.columns]
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in summary_df.itertuples(index=False, name=None):
            f.write("| " + " | ".join(str(value) for value in row) + " |\n")
        f.write("\n")

    print("  Saved: summary_table.csv, summary_table.md")


def main():
    parser = argparse.ArgumentParser(description="Generate Section 5 paper figures")
    parser.add_argument(
        "--step1-dir", type=Path, required=True, help="Path to step1 policy comparison results"
    )
    parser.add_argument(
        "--step2-dir", type=Path, default=None, help="Path to step2 delay sweep results"
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for figures")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading Step 1 results...")
    df1 = load_results(args.step1_dir)
    print(f"  Loaded {len(df1)} rows")

    print("\nGenerating figures...")

    # Figure A: Policy comparison
    plot_policy_comparison(df1, args.out)

    # Figure B: Delay curves (if available)
    if args.step2_dir and args.step2_dir.exists():
        print("\nLoading Step 2 results for delay curves...")
        plot_delay_curves(args.step2_dir, args.out)

    # Figure C: CTE vs switches
    plot_cte_vs_switches(df1, args.out)

    # Summary table
    generate_summary_table(df1, args.out)

    print(f"\nAll figures saved to: {args.out}")


if __name__ == "__main__":
    main()
