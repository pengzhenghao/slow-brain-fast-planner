#!/usr/bin/env python3
"""
Analyze VLM latency from a Task2 real-world run directory.

Expected run_dir layout (as produced by Task2LiveNode):
  - config.json
  - final_report.json  (contains events_vlm with t_submit/t_result/dt/interval)
  - telemetry/telemetry.jsonl (optional; not required here)

Outputs (default: run_dir):
  - vlm_latency_events.csv
  - vlm_latency_summary.txt
  - vlm_latency_analysis.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _p(xs: np.ndarray, q: float) -> float:
    if xs.size == 0:
        return float("nan")
    return float(np.percentile(xs, q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory, e.g. local_logs/real_world/task2_live_....",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: run-dir)",
    )
    ap.add_argument(
        "--long-s",
        type=float,
        default=5.0,
        help="Threshold (seconds) considered a 'long' VLM delay.",
    )
    ap.add_argument(
        "--near-timeout-margin-s",
        type=float,
        default=2.0,
        help="Mark calls with dt >= (worker_timeout_s - margin) as 'near-timeout'.",
    )
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    out_dir: Path = args.out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    report_path = run_dir / "final_report.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing final_report.json: {report_path}")

    config = _load_json(config_path)
    report = _load_json(report_path)

    worker_timeout_s = float(config.get("vlm_worker", {}).get("timeout_s", float("nan")))
    provider = str(config.get("vlm", {}).get("provider", "unknown"))
    model = str(config.get("vlm", {}).get(provider, {}).get("model", "unknown"))

    events = report.get("events_vlm", [])
    if not isinstance(events, list) or len(events) == 0:
        raise RuntimeError("final_report.json has no events_vlm to analyze.")

    t_submit = np.array([e.get("t_submit", np.nan) for e in events], dtype=float)
    t_result = np.array([e.get("t_result", np.nan) for e in events], dtype=float)
    dt = np.array([e.get("dt", np.nan) for e in events], dtype=float)
    interval = np.array(
        [
            np.nan if (e.get("interval", None) is None) else e.get("interval", np.nan)
            for e in events
        ],
        dtype=float,
    )

    t0 = float(np.nanmin(t_submit))
    t_rel = t_submit - t0

    # Heuristic: dt near worker timeout is likely a stall (even if it didn't cross the timeout).
    near_timeout_like = np.zeros_like(dt, dtype=bool)
    if np.isfinite(worker_timeout_s):
        margin = float(max(0.0, args.near_timeout_margin_s))
        near_timeout_like = dt >= (worker_timeout_s - margin)

    long_like = dt >= float(args.long_s)

    # Write CSV.
    csv_path = out_dir / "vlm_latency_events.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "i",
                "t_submit",
                "t_result",
                "t_rel_submit_s",
                "dt_s",
                "interval_s",
                "is_long",
                "is_near_timeout",
            ]
        )
        for i in range(len(events)):
            w.writerow(
                [
                    i,
                    t_submit[i],
                    t_result[i],
                    t_rel[i],
                    dt[i],
                    interval[i],
                    bool(long_like[i]),
                    bool(near_timeout_like[i]),
                ]
            )

    # Summary stats.
    finite_dt = dt[np.isfinite(dt)]
    finite_interval = interval[np.isfinite(interval)]

    summary_lines: list[str] = []
    summary_lines.append(f"run_dir: {run_dir}")
    summary_lines.append(f"provider: {provider}")
    summary_lines.append(f"model: {model}")
    summary_lines.append(f"worker_timeout_s: {worker_timeout_s}")
    summary_lines.append(f"events_vlm: {len(events)}")
    summary_lines.append("")
    summary_lines.append("dt_s stats (VLM submit->result wall time):")
    summary_lines.append(f"  mean:   {float(np.mean(finite_dt)):.3f}")
    summary_lines.append(f"  median: {_p(finite_dt, 50):.3f}")
    summary_lines.append(f"  p90:    {_p(finite_dt, 90):.3f}")
    summary_lines.append(f"  p95:    {_p(finite_dt, 95):.3f}")
    summary_lines.append(f"  p99:    {_p(finite_dt, 99):.3f}")
    summary_lines.append(f"  max:    {float(np.max(finite_dt)):.3f}")
    summary_lines.append(f"  >= {args.long_s:.1f}s: {int(np.sum(long_like & np.isfinite(dt)))}")
    if np.isfinite(worker_timeout_s):
        margin = float(max(0.0, args.near_timeout_margin_s))
        summary_lines.append(
            f"  >= {worker_timeout_s - margin:.1f}s (near-timeout): "
            f"{int(np.sum(near_timeout_like & np.isfinite(dt)))}"
        )
    summary_lines.append("")
    summary_lines.append("submit interval_s stats (time between consecutive submits):")
    if finite_interval.size > 0:
        summary_lines.append(f"  mean:   {float(np.mean(finite_interval)):.3f}")
        summary_lines.append(f"  median: {_p(finite_interval, 50):.3f}")
        summary_lines.append(f"  p90:    {_p(finite_interval, 90):.3f}")
        summary_lines.append(f"  p95:    {_p(finite_interval, 95):.3f}")
        summary_lines.append(f"  max:    {float(np.max(finite_interval)):.3f}")
    else:
        summary_lines.append("  (no interval data)")

    summary_path = out_dir / "vlm_latency_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Plot.
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.2, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0])

    # dt time series
    ax0.plot(t_rel, dt, linewidth=1.0, color="#4C78A8", alpha=0.6)
    ax0.scatter(
        t_rel[~near_timeout_like],
        dt[~near_timeout_like],
        s=18,
        color="#4C78A8",
        alpha=0.8,
        label="dt",
    )
    if np.any(near_timeout_like):
        ax0.scatter(
            t_rel[near_timeout_like],
            dt[near_timeout_like],
            s=28,
            color="#E45756",
            alpha=0.95,
            label="near-timeout",
        )
    ax0.axhline(args.long_s, color="#F58518", linestyle="--", linewidth=1, alpha=0.8)
    if np.isfinite(worker_timeout_s):
        ax0.axhline(worker_timeout_s, color="#E45756", linestyle=":", linewidth=1, alpha=0.9)
    ax0.set_ylabel("VLM latency dt (s)")
    ax0.set_title("VLM latency over time (submit->result)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    # interval time series
    ax1.plot(t_rel, interval, linewidth=1.0, color="#72B7B2", alpha=0.8)
    ax1.scatter(t_rel, interval, s=14, color="#72B7B2", alpha=0.8)
    ax1.set_ylabel("submit interval (s)")
    ax1.set_xlabel("time since first submit (s)")
    ax1.grid(True, alpha=0.25)

    # histogram of dt
    bins = np.linspace(0, max(float(np.nanmax(dt)), args.long_s, 1.0), 60)
    ax2.hist(finite_dt, bins=bins, color="#4C78A8", alpha=0.85)
    ax2.axvline(args.long_s, color="#F58518", linestyle="--", linewidth=1, alpha=0.8)
    if np.isfinite(worker_timeout_s):
        ax2.axvline(worker_timeout_s, color="#E45756", linestyle=":", linewidth=1, alpha=0.9)
    ax2.set_xlabel("VLM latency dt (s)")
    ax2.set_ylabel("count")
    ax2.set_title("Latency distribution")
    ax2.grid(True, alpha=0.25)

    plot_path = out_dir / "vlm_latency_analysis.png"
    fig.suptitle(f"{run_dir.name} | {provider}:{model}", fontsize=12)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
