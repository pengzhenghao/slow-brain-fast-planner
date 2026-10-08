#!/usr/bin/env python3
"""
Create a compact policy-level table for the paper from `real_world_metrics_all.csv`.

Outputs:
- CSV: one row per policy (mean across runs + N)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POLICY_ORDER = [
    "local_only",
    "vlm_hold",
    "vlm_stream",
    "vlm_hold_match",
    "vlm_stream_match",
    "score_fusion_stream",
    "prob_fusion_stream",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", dest="in_csv", required=True, type=str)
    ap.add_argument("--out-csv", dest="out_csv", required=True, type=str)
    ap.add_argument(
        "--drop-n",
        action="store_true",
        help="If set, omit the n_runs column (useful for compact paper tables).",
    )
    args = ap.parse_args()

    in_csv = Path(args.in_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    df = pd.read_csv(in_csv)

    # Columns to include in the paper table (policy-level means).
    cols = [
        "takeover_rate_per_100m",
        "takeover_frac_time",
        "publish_stale_s_p90",
        "publish_vs_argmax_end_m_p90",
        "vlm_latency_s_p90",
        "vlm_vs_argmax_end_m_p90",
        "temporal_consistency_ade_m_p90",
        "temporal_consistency_end_m_p90",
        "longest_autonomy_segment_dist_m",
    ]
    cols = [c for c in cols if c in df.columns]

    rows = []
    for pol in POLICY_ORDER:
        sub = df[df["policy"] == pol]
        if sub.empty:
            continue
        d = {"policy": pol, "n_runs": int(len(sub))}
        for c in cols:
            d[c] = float(sub[c].mean(skipna=True)) if np.isfinite(sub[c]).any() else np.nan
        rows.append(d)

    out = pd.DataFrame(rows)
    if bool(args.drop_n) and "n_runs" in out.columns:
        out = out.drop(columns=["n_runs"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
