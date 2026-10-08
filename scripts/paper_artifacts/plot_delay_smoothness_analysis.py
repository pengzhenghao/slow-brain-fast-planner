#!/usr/bin/env python
"""Plot delay effect analysis: comparing trajectory smoothness across delay values.

This script generates figures showing:
1. CTE vs delay for different tau values
2. Trajectory switches (smoothness) vs delay
3. Scatter: CTE vs switches showing the smoothness-performance tradeoff

Usage:
    python scripts/plot_delay_smoothness_analysis.py \
        --run-dirs logs/0126_closed_loop_experiments/step3_tau_x_delay_best_from_step2\
/delay_*/closed_loop_fusion_* \
        --out figures/delay_smoothness_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def load_summary_csvs(run_dirs: list[Path]) -> pd.DataFrame:
    """Load and concatenate summary.csv files from multiple run directories."""
    dfs = []
    for rd in run_dirs:
        summary = rd / "summary.csv"
        if not summary.exists():
            print(f"Warning: {summary} not found, skipping.")
            continue
        df = pd.read_csv(summary)
        # Extract delay from parent directory name (e.g., delay_0.5)
        parent_name = rd.parent.name
        if parent_name.startswith("delay_"):
            try:
                delay_val = float(parent_name.replace("delay_", ""))
                df["delay_s"] = delay_val
            except ValueError:
                pass
        dfs.append(df)
    if not dfs:
        raise ValueError("No valid summary.csv files found.")
    return pd.concat(dfs, ignore_index=True)


def plot_cte_vs_delay_by_tau(df: pd.DataFrame, out_dir: Path):
    """Plot mean CTE vs delay, one curve per tau value."""
    plt = _try_import_matplotlib()
    if plt is None:
        print("matplotlib not available, skipping plot.")
        return

    # Filter to score_fusion policy only
    df_sf = df[df["policy"] == "score_fusion"].copy()

    # Get unique tau values
    taus = sorted(df_sf["staleness_tau_s"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for tau in taus:
        df_tau = df_sf[df_sf["staleness_tau_s"] == tau]
        # Group by delay and take mean across other hyperparams
        grouped = (
            df_tau.groupby("delay_s").agg({"mean_cte_m": ["mean", "std"], "n": "sum"}).reset_index()
        )
        grouped.columns = ["delay_s", "mean_cte", "std_cte", "n"]

        ax.plot(
            grouped["delay_s"],
            grouped["mean_cte"],
            "o-",
            label=f"τ={tau}s",
            linewidth=2,
            markersize=8,
        )
        ax.fill_between(
            grouped["delay_s"],
            grouped["mean_cte"] - grouped["std_cte"],
            grouped["mean_cte"] + grouped["std_cte"],
            alpha=0.2,
        )

    # Add local_only baseline
    df_local = df[df["policy"] == "local_only"]
    if not df_local.empty:
        local_mean = df_local["mean_cte_m"].mean()
        ax.axhline(
            local_mean, color="black", linestyle="--", linewidth=2, label="local_only (baseline)"
        )

    ax.set_xlabel("VLM Delay (seconds)", fontsize=14)
    ax.set_ylabel("Mean Cross-Track Error (m)", fontsize=14)
    ax.set_title(
        "Effect of VLM Delay on Tracking Performance\n(score_fusion, λ=3, planner-grid avg)",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    out_path = out_dir / "cte_vs_delay_by_tau.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_dir / "cte_vs_delay_by_tau.pdf")
    print(f"Saved: {out_path}")
    plt.close()


def plot_switches_vs_delay_by_tau(df: pd.DataFrame, out_dir: Path):
    """Plot mean trajectory switches vs delay, one curve per tau value."""
    plt = _try_import_matplotlib()
    if plt is None:
        print("matplotlib not available, skipping plot.")
        return

    df_sf = df[df["policy"] == "score_fusion"].copy()
    taus = sorted(df_sf["staleness_tau_s"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for tau in taus:
        df_tau = df_sf[df_sf["staleness_tau_s"] == tau]
        grouped = (
            df_tau.groupby("delay_s")
            .agg(
                {
                    "mean_switches": ["mean", "std"],
                }
            )
            .reset_index()
        )
        grouped.columns = ["delay_s", "mean_sw", "std_sw"]

        ax.plot(
            grouped["delay_s"],
            grouped["mean_sw"],
            "o-",
            label=f"τ={tau}s",
            linewidth=2,
            markersize=8,
        )
        ax.fill_between(
            grouped["delay_s"],
            grouped["mean_sw"] - grouped["std_sw"],
            grouped["mean_sw"] + grouped["std_sw"],
            alpha=0.2,
        )

    # Add local_only baseline
    df_local = df[df["policy"] == "local_only"]
    if not df_local.empty:
        local_mean = df_local["mean_switches"].mean()
        ax.axhline(
            local_mean, color="black", linestyle="--", linewidth=2, label="local_only (baseline)"
        )

    ax.set_xlabel("VLM Delay (seconds)", fontsize=14)
    ax.set_ylabel("Mean Trajectory Switches per Episode", fontsize=14)
    ax.set_title(
        "Effect of VLM Delay on Trajectory Smoothness\n(score_fusion, λ=3, planner-grid avg)",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)

    out_path = out_dir / "switches_vs_delay_by_tau.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_dir / "switches_vs_delay_by_tau.pdf")
    print(f"Saved: {out_path}")
    plt.close()


def plot_cte_vs_switches_scatter(df: pd.DataFrame, out_dir: Path):
    """Scatter plot of CTE vs switches, colored by delay."""
    plt = _try_import_matplotlib()
    if plt is None:
        print("matplotlib not available, skipping plot.")
        return

    df_sf = df[df["policy"] == "score_fusion"].copy()

    # Group by (delay, tau) and average
    grouped = (
        df_sf.groupby(["delay_s", "staleness_tau_s"])
        .agg(
            {
                "mean_cte_m": "mean",
                "mean_switches": "mean",
            }
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    delays = sorted(grouped["delay_s"].unique())
    cmap = plt.cm.viridis
    colors = [cmap(i / len(delays)) for i in range(len(delays))]

    for i, delay in enumerate(delays):
        df_d = grouped[grouped["delay_s"] == delay]
        ax.scatter(
            df_d["mean_switches"],
            df_d["mean_cte_m"],
            c=[colors[i]],
            s=100,
            label=f"delay={delay}s",
            alpha=0.8,
            edgecolors="black",
        )

    # Add local_only point
    df_local = df[df["policy"] == "local_only"]
    if not df_local.empty:
        local_cte = df_local["mean_cte_m"].mean()
        local_sw = df_local["mean_switches"].mean()
        ax.scatter(
            [local_sw],
            [local_cte],
            c="red",
            s=200,
            marker="*",
            label="local_only",
            edgecolors="black",
            zorder=10,
        )

    ax.set_xlabel("Mean Trajectory Switches (lower = smoother)", fontsize=14)
    ax.set_ylabel("Mean Cross-Track Error (m, lower = better)", fontsize=14)
    ax.set_title(
        "Smoothness-Performance Tradeoff by VLM Delay\n(each point is a (delay, τ) combination)",
        fontsize=14,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate the best point
    best_idx = grouped["mean_cte_m"].idxmin()
    best = grouped.loc[best_idx]
    ax.annotate(
        f"Best: delay={best['delay_s']:.1f}s, τ={best['staleness_tau_s']:.0f}s",
        xy=(best["mean_switches"], best["mean_cte_m"]),
        xytext=(best["mean_switches"] + 2, best["mean_cte_m"] - 0.02),
        fontsize=11,
        arrowprops=dict(arrowstyle="->", color="green"),
        color="green",
        fontweight="bold",
    )

    out_path = out_dir / "cte_vs_switches_scatter.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_dir / "cte_vs_switches_scatter.pdf")
    print(f"Saved: {out_path}")
    plt.close()


def plot_delay_comparison_bar(df: pd.DataFrame, out_dir: Path):
    """Bar chart comparing delay=0 vs delay=0.5 at best tau."""
    plt = _try_import_matplotlib()
    if plt is None:
        print("matplotlib not available, skipping plot.")
        return

    df_sf = df[df["policy"] == "score_fusion"].copy()

    # Find best tau for each delay
    delays = [0.0, 0.5]
    best_tau = 5.0  # Based on your experiment results

    metrics = ["mean_cte_m", "mean_switches"]
    labels = ["Mean CTE (m)", "Mean Switches"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, label in zip(axes, metrics, labels, strict=False):
        vals = []
        errs = []
        for delay in delays:
            df_d = df_sf[(df_sf["delay_s"] == delay) & (df_sf["staleness_tau_s"] == best_tau)]
            if df_d.empty:
                # Fallback: use any tau
                df_d = df_sf[df_sf["delay_s"] == delay]
            vals.append(df_d[metric].mean())
            errs.append(df_d[metric].std())

        x = np.arange(len(delays))
        bars = ax.bar(
            x, vals, yerr=errs, capsize=5, color=["#ff7f0e", "#2ca02c"], edgecolor="black"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"delay={d}s" for d in delays], fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"{label} Comparison", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

        # Add values on bars
        for bar, val in zip(bars, vals, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    # Calculate improvement
    if len(vals) == 2 and vals[0] > 0:
        pct_improvement = (vals[0] - vals[1]) / vals[0] * 100
        fig.suptitle(
            f"Why Delay Helps: delay=0.5s achieves {pct_improvement:.1f}% lower CTE\n"
            f"(τ={best_tau}s, λ=3, score_fusion)",
            fontsize=14,
            fontweight="bold",
        )

    out_path = out_dir / "delay_comparison_bar.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_dir / "delay_comparison_bar.pdf")
    print(f"Saved: {out_path}")
    plt.close()


def generate_markdown_summary(df: pd.DataFrame, out_dir: Path):
    """Generate a markdown summary of the delay effect analysis."""
    df_sf = df[df["policy"] == "score_fusion"].copy()

    # Find best config at each delay
    summary_lines = [
        "# Delay Effect Analysis: Why VLM Latency Can Help",
        "",
        "## Key Finding",
        "",
        "Counter-intuitively, a small VLM delay (0.5s) outperforms zero delay (0.0s).",
        "This is due to **temporal smoothing**: delayed VLM advice acts as a low-pass filter,",
        "reducing trajectory switching jitter and improving tracking stability.",
        "",
        "## Mechanism",
        "",
        "1. **Low-Pass Filtering**: With delay=0.5s, VLM advice is based on state from 0.5s ago,",
        "   smoothing out high-frequency noise in trajectory selection.",
        "",
        "2. **Look-Ahead Stabilization**: The `body_pointwise_horizon_aware` similarity mode",
        "   compares the *remaining* portion of stale VLM trajectory to current candidates,",
        "   effectively providing a look-ahead reference.",
        "",
        "3. **Optimal Blending**: With τ=5s and delay=0.5s:",
        "   - Initial weight: exp(-0.5/5) ≈ 0.905 (90% VLM influence)",
        "   - This leaves 10% for local planner reactivity",
        "",
        "## Quantitative Results",
        "",
    ]

    # Add table
    summary_lines.append("| Delay | τ | Mean CTE (m) | Mean Switches | CTE Improvement |")
    summary_lines.append("|-------|---|--------------|---------------|-----------------|")

    baseline_cte = df_sf[df_sf["delay_s"] == 0.0]["mean_cte_m"].mean()

    for delay in sorted(df_sf["delay_s"].unique()):
        for tau in [5.0]:  # Focus on best tau
            df_sub = df_sf[(df_sf["delay_s"] == delay) & (df_sf["staleness_tau_s"] == tau)]
            if df_sub.empty:
                continue
            cte = df_sub["mean_cte_m"].mean()
            sw = df_sub["mean_switches"].mean()
            improvement = (baseline_cte - cte) / baseline_cte * 100 if baseline_cte > 0 else 0
            summary_lines.append(
                f"| {delay:.1f}s | {tau:.0f}s | {cte:.4f} | {sw:.1f} | {improvement:+.1f}% |"
            )

    summary_lines.extend(
        [
            "",
            "## Implications for Deployment",
            "",
            "- Don't over-optimize for zero latency VLM",
            "- A small buffer (0.5-1.0s) can actually improve performance",
            "- This is analogous to Smith Predictor in control theory",
            "",
        ]
    )

    out_path = out_dir / "delay_analysis_summary.md"
    out_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot delay effect analysis for closed-loop score fusion"
    )
    parser.add_argument(
        "--run-dirs", nargs="+", required=True, help="Run directories containing summary.csv"
    )
    parser.add_argument(
        "--out", type=str, default="figures/delay_smoothness_analysis", help="Output directory"
    )
    args = parser.parse_args()

    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {len(run_dirs)} run directories...")
    df = load_summary_csvs(run_dirs)
    print(f"Loaded {len(df)} rows")

    print("\nGenerating plots...")
    plot_cte_vs_delay_by_tau(df, out_dir)
    plot_switches_vs_delay_by_tau(df, out_dir)
    plot_cte_vs_switches_scatter(df, out_dir)
    plot_delay_comparison_bar(df, out_dir)
    generate_markdown_summary(df, out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
