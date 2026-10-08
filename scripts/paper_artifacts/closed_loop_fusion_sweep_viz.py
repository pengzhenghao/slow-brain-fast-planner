#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _format_float_for_path(x: float) -> str:
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if abs(xf - round(xf)) <= 1e-9:
        return f"{xf:.1f}"
    return str(xf)


def _find_latest_run(parent: Path) -> Path | None:
    if not parent.exists():
        return None
    runs = sorted(
        [
            p
            for p in parent.rglob("closed_loop_fusion_*")
            if p.is_dir() and p.name.startswith("closed_loop_fusion_")
        ]
    )
    return runs[-1] if runs else None


def _find_latest_delay_run(parent: Path, delay_tag: str) -> Path | None:
    runs = []
    for delay_dir in parent.glob(f"delay_{delay_tag}*"):
        if not delay_dir.is_dir():
            continue
        run = _find_latest_run(delay_dir)
        if run is not None:
            runs.append(run)
    return sorted(runs, key=lambda path: str(path))[-1] if runs else None


def _load_summary(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary.csv: {p}")
    return pd.read_csv(p)


def _safe_float(x: object, default: float = math.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _best_row_sort_key(r: pd.Series) -> tuple[float, float, float, float]:
    # Match the benchmark script's reporting preference:
    #   maximize success_rate, maximize route completion, minimize goal distance, minimize mean CTE.
    succ = _safe_float(r.get("success_rate", math.nan), default=-1.0)
    comp = _safe_float(r.get("mean_route_completion", math.nan), default=-1.0)
    g = _safe_float(
        r.get("mean_goal_dist_min_m", r.get("mean_goal_dist_final_m", math.nan)), default=math.inf
    )
    cte = _safe_float(r.get("mean_cte_m", math.nan), default=math.inf)
    return (-succ, -comp, g, cte)


def _select_best_per_param(df: pd.DataFrame, *, param_col: str) -> pd.DataFrame:
    need = {"policy", param_col}
    missing = [c for c in sorted(need) if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns for sweep selection: {missing}. Columns: {sorted(df.columns)}"
        )

    parts: list[pd.DataFrame] = []
    for (_policy, _pval), g in df.groupby(["policy", param_col], as_index=False, dropna=False):
        gg = g.copy()
        # Stable order: choose best row by our composite key.
        gg["_sort_key"] = gg.apply(_best_row_sort_key, axis=1)
        gg = gg.sort_values("_sort_key", ascending=True).drop(columns=["_sort_key"])
        parts.append(gg.iloc[:1])
    out = pd.concat(parts, ignore_index=True)
    out[param_col] = pd.to_numeric(out[param_col], errors="coerce")
    return out.sort_values(["policy", param_col]).reset_index(drop=True)


def _plot_sweep(
    df_best: pd.DataFrame,
    *,
    param_col: str,
    title: str,
    out_pdf: Path,
    out_png: Path,
    policies: list[str] | None = None,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(f"matplotlib is required for plotting: {e}") from e

    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    d = df_best.copy()
    if policies is not None and len(policies) > 0:
        d = d[d["policy"].isin(set(policies))].copy()

    if d.empty:
        raise ValueError("No rows to plot after filtering.")

    metrics = []
    if "success_rate" in d.columns:
        metrics.append(("success_rate", "Success rate"))
    metrics.append(("mean_cte_m", "Mean CTE (m)"))

    nrows = len(metrics)
    fig, axes = plt.subplots(nrows, 1, figsize=(7.6, 2.7 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    color_map = {
        "score_fusion": "#1f77b4",
        "prob_fusion": "#9467bd",
        "score_fusion_stream": "#aec7e8",
        "prob_fusion_stream": "#c5b0d5",
        "local_only": "#d62728",
        "vlm_hold": "#ff7f0e",
        "vlm_stream": "#2ca02c",
        "vlm_hold_match": "#8c564b",
        "vlm_stream_match": "#17becf",
        "planner_oracle": "0.25",
    }

    for ax, (mcol, ylabel) in zip(axes, metrics, strict=False):
        if mcol not in d.columns:
            continue
        for pol, g in d.groupby("policy", as_index=False):
            x = pd.to_numeric(g[param_col], errors="coerce").to_numpy()
            y = pd.to_numeric(g[mcol], errors="coerce").to_numpy()
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            ax.plot(
                x,
                y,
                marker="o",
                lw=2.2,
                ms=5.5,
                color=color_map.get(str(pol), None),
                label=str(pol),
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    axes[0].set_title(title)
    axes[-1].set_xlabel(param_col)

    # De-duplicate legend entries.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_pdf), bbox_inches="tight")
    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Make appendix-ready PDFs for lambda/tau sweeps from closed-loop fusion runs "
        "(summary.csv)."
    )
    p.add_argument(
        "--section5-root", type=str, default="", help="Logs root (e.g. logs/section5_closed_loop)."
    )
    p.add_argument(
        "--delay-s",
        type=float,
        default=2.0,
        help="Used for auto-discovery when run dirs are not provided.",
    )

    p.add_argument(
        "--lambda-run-dir",
        type=str,
        default="",
        help="Explicit closed_loop_fusion_* run dir for the lambda sweep.",
    )
    p.add_argument(
        "--tau-run-dir",
        type=str,
        default="",
        help="Explicit closed_loop_fusion_* run dir for the tau sweep.",
    )

    p.add_argument(
        "--out-dir", type=str, default="latex/figs", help="Output directory for PDF/PNG figures."
    )
    p.add_argument(
        "--policies",
        type=str,
        default="",
        help="Optional: comma-separated policy filter (e.g. score_fusion,prob_fusion).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    section5_root = (
        Path(args.section5_root).expanduser().resolve()
        if str(args.section5_root).strip()
        else (repo_root / "logs/section5_closed_loop").resolve()
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    policies = [p.strip() for p in str(args.policies).split(",") if p.strip()]

    delay_tag = _format_float_for_path(float(args.delay_s))

    # Resolve run dirs
    lam_run = (
        Path(args.lambda_run_dir).expanduser().resolve()
        if str(args.lambda_run_dir).strip()
        else None
    )
    if lam_run is None:
        lam_run = _find_latest_delay_run(
            section5_root / "step3_lambda_sweep_fusions",
            delay_tag,
        )

    tau_run = (
        Path(args.tau_run_dir).expanduser().resolve() if str(args.tau_run_dir).strip() else None
    )
    if tau_run is None:
        tau_parent = section5_root / "step5_tau_sweep_fusions" / f"delay_{delay_tag}"
        tau_run = _find_latest_run(tau_parent)

    outputs: dict[str, str] = {}

    if lam_run is not None and lam_run.exists():
        df = _load_summary(lam_run)
        df_best = _select_best_per_param(df, param_col="lambda_sim")
        out_pdf = out_dir / "closed_loop_lambda_sweep.pdf"
        out_png = out_dir / "closed_loop_lambda_sweep.png"
        _plot_sweep(
            df_best,
            param_col="lambda_sim",
            title=f"Closed-loop fusion sensitivity: lambda (delay={float(args.delay_s):g}s)",
            out_pdf=out_pdf,
            out_png=out_png,
            policies=policies if policies else None,
        )
        df_best.to_csv(out_dir / "closed_loop_lambda_sweep_best.csv", index=False)
        outputs["lambda_run_dir"] = str(lam_run)
        outputs["lambda_fig_pdf"] = str(out_pdf)

    if tau_run is not None and tau_run.exists():
        df = _load_summary(tau_run)
        df_best = _select_best_per_param(df, param_col="staleness_tau_s")
        out_pdf = out_dir / "closed_loop_tau_sweep.pdf"
        out_png = out_dir / "closed_loop_tau_sweep.png"
        _plot_sweep(
            df_best,
            param_col="staleness_tau_s",
            title=f"Closed-loop fusion sensitivity: tau (delay={float(args.delay_s):g}s)",
            out_pdf=out_pdf,
            out_png=out_png,
            policies=policies if policies else None,
        )
        df_best.to_csv(out_dir / "closed_loop_tau_sweep_best.csv", index=False)
        outputs["tau_run_dir"] = str(tau_run)
        outputs["tau_fig_pdf"] = str(out_pdf)

    if not outputs:
        raise SystemExit(
            "No runs found to plot.\n"
            f"- Tried lambda: {section5_root}/step3_lambda_sweep_fusions/delay_{delay_tag}\n"
            f"- Tried tau   : {section5_root}/step5_tau_sweep_fusions/delay_{delay_tag}\n"
            "Pass --lambda-run-dir/--tau-run-dir to point to a specific closed_loop_fusion_* "
            "directory."
        )

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
