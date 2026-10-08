#!/usr/bin/env python3
"""
Batch-generate BEV trajectory figures for all real-world runs in a log root.

Outputs:
  - singles/<direction>/*.png: one BEV per run
  - head_to_head/<direction>/*.png: all pairwise head-to-head comparisons per direction
  - index.csv: metadata for all generated images

Direction is inferred from the run label (typically ends with _INT-MED or _MED-INT).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path (works when invoked as "python scripts/xxx.py").
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.paper_artifacts import plot_real_world_bev_head_to_head as h2h  # noqa: E402

_DIR_RE = re.compile(r"(?:^|_)(INT|MED)-(INT|MED)$")


def _sanitize_filename(s: str, *, max_len: int = 160) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._+=-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("._-")
    if not s:
        s = "x"
    if len(s) > int(max_len):
        s = s[: int(max_len)].rstrip("._-")
    return s


def _infer_direction(label_or_name: str) -> str:
    m = _DIR_RE.search(str(label_or_name).strip())
    if not m:
        return "UNKNOWN"
    return f"{m.group(1)}-{m.group(2)}"


def _strip_direction_suffix(label: str) -> str:
    s = str(label).strip()
    m = _DIR_RE.search(s)
    if not m:
        return s
    # Remove the trailing "_INT-MED" or "_MED-INT"
    d = f"_{m.group(1)}-{m.group(2)}"
    if s.endswith(d):
        return s[: -len(d)]
    return s


def _infer_timestamp_from_dirname(run_dir: Path) -> str:
    # run dir names look like: task2_live_2026-01-29_005229_vlm-stream-match_INT-MED
    parts = run_dir.name.split("_")
    if len(parts) >= 4 and parts[0] == "task2" and parts[1] == "live":
        date = parts[2]
        time = parts[3]
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) and re.fullmatch(r"\d{6}", time):
            return f"{date}_{time}"
    # fallback
    return run_dir.name


@dataclass(frozen=True)
class RunInfo:
    run_dir: str
    label: str
    policy_base: str
    direction: str
    timestamp: str
    total_time_s: float
    total_dist_m: float
    takeovers: int
    takeover_time_s: float
    takeover_dist_m: float
    takeover_frac_time: float
    takeover_frac_dist: float


def _load_and_prepare_series(
    run_dir: Path, *, trim: bool, align_yaw: bool, stride: int
) -> tuple[h2h.OdomSeries, h2h.TakeoverStats]:
    s = h2h._load_odom_series(run_dir)
    if trim:
        s = h2h._trim_to_first_last_autonomy(s)
    if int(stride) > 1:
        s = h2h._downsample(s, stride=int(stride))
    s = h2h._align_xy_yaw(s, align_yaw=bool(align_yaw))
    stats = h2h._compute_takeover_stats(s)
    return s, stats


def _set_square_limits(ax, *, x: np.ndarray, y: np.ndarray, pad_m: float) -> None:
    pad = float(max(0.0, float(pad_m)))
    if x.size == 0 or y.size == 0:
        return
    xmin = float(np.min(x)) - pad
    xmax = float(np.max(x)) + pad
    ymin = float(np.min(y)) - pad
    ymax = float(np.max(y)) + pad
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    r = 0.5 * max((xmax - xmin), (ymax - ymin))
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)


def _render_single(
    *,
    run_dir: Path,
    out_path: Path,
    label: str,
    trim: bool,
    align_yaw: bool,
    stride: int,
    dpi: int,
    fig_w: float,
    fig_h: float,
    pad_m: float,
) -> RunInfo:
    plt = h2h._try_import_matplotlib_pyplot()
    if plt is None:
        raise SystemExit("matplotlib is required to plot figures (failed to import).")

    series, stats = _load_and_prepare_series(run_dir, trim=trim, align_yaw=align_yaw, stride=stride)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        1, 1, figsize=(float(fig_w), float(fig_h)), dpi=int(dpi), constrained_layout=True
    )
    h2h._plot_run(ax, series=series, title=label, stats=stats)
    _set_square_limits(ax, x=np.asarray(series.x_m), y=np.asarray(series.y_m), pad_m=float(pad_m))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    direction = _infer_direction(label)
    policy_base = _strip_direction_suffix(label)
    ts = _infer_timestamp_from_dirname(run_dir)
    return RunInfo(
        run_dir=str(run_dir),
        label=str(label),
        policy_base=str(policy_base),
        direction=str(direction),
        timestamp=str(ts),
        total_time_s=float(stats.total_time_s),
        total_dist_m=float(stats.total_dist_m),
        takeovers=int(stats.takeover_events),
        takeover_time_s=float(stats.takeover_time_s),
        takeover_dist_m=float(stats.takeover_dist_m),
        takeover_frac_time=float(stats.takeover_frac_time),
        takeover_frac_dist=float(stats.takeover_frac_dist),
    )


def _render_pair(
    *,
    run_a: Path,
    run_b: Path,
    out_path: Path,
    label_a: str,
    label_b: str,
    trim: bool,
    align_yaw: bool,
    stride: int,
    dpi: int,
    fig_w: float,
    fig_h: float,
    pad_m: float,
) -> None:
    plt = h2h._try_import_matplotlib_pyplot()
    if plt is None:
        raise SystemExit("matplotlib is required to plot figures (failed to import).")

    sa, sta = _load_and_prepare_series(run_a, trim=trim, align_yaw=align_yaw, stride=stride)
    sb, stb = _load_and_prepare_series(run_b, trim=trim, align_yaw=align_yaw, stride=stride)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(
        1, 2, figsize=(float(fig_w), float(fig_h)), dpi=int(dpi), constrained_layout=True
    )
    ax_a, ax_b = axs[0], axs[1]
    h2h._plot_run(ax_a, series=sa, title=label_a, stats=sta)
    h2h._plot_run(ax_b, series=sb, title=label_b, stats=stb)

    xs = np.concatenate([np.asarray(sa.x_m), np.asarray(sb.x_m)], axis=0)
    ys = np.concatenate([np.asarray(sa.y_m), np.asarray(sb.y_m)], axis=0)
    for ax in (ax_a, ax_b):
        _set_square_limits(ax, x=xs, y=ys, pad_m=float(pad_m))

    fig.suptitle(
        "Real-world BEV trajectory (local odom) with human takeovers highlighted", fontsize=12
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate BEV figures for all real_world runs under a log root."
    )
    p.add_argument(
        "--root",
        type=str,
        default="local_logs/real_world_0129",
        help="Root directory containing task2_live_* runs.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <root>/bev_generated_all).",
    )
    p.add_argument(
        "--stride", type=int, default=2, help="Downsample odom ticks by this stride (>=1)."
    )
    p.add_argument(
        "--trim",
        action="store_true",
        help="Trim leading/trailing manual segment (to first/last autonomy tick).",
    )
    p.add_argument("--no-trim", dest="trim", action="store_false", help="Do not trim.")
    p.set_defaults(trim=True)
    p.add_argument(
        "--align-yaw", action="store_true", help="Rotate so the first heading points +x."
    )
    p.add_argument(
        "--no-align-yaw",
        dest="align_yaw",
        action="store_false",
        help="Do not rotate; only translate.",
    )
    p.set_defaults(align_yaw=True)
    p.add_argument("--pad-m", type=float, default=0.5, help="Axis padding (meters).")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--single-fig-w", type=float, default=5.2)
    p.add_argument("--single-fig-h", type=float, default=4.8)
    p.add_argument("--pair-fig-w", type=float, default=10.5)
    p.add_argument("--pair-fig-h", type=float, default=4.8)
    p.add_argument(
        "--pairs", action="store_true", help="Generate all head-to-head pairs per direction."
    )
    p.add_argument(
        "--no-pairs", dest="pairs", action="store_false", help="Disable head-to-head generation."
    )
    p.set_defaults(pairs=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")
    out_dir = (
        Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "bev_generated_all")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tel_paths = sorted(root.glob("task2_live_*/telemetry/telemetry.jsonl"))
    run_dirs = sorted({p.parent.parent for p in tel_paths})
    if not run_dirs:
        raise SystemExit(
            f"No runs found under {root} (expected task2_live_*/telemetry/telemetry.jsonl)"
        )

    singles_dir = out_dir / "singles"
    pairs_dir = out_dir / "head_to_head"

    infos: list[RunInfo] = []
    by_direction: dict[str, list[Path]] = {}
    labels: dict[str, str] = {}

    # Singles
    for run_dir in run_dirs:
        label = h2h._load_run_label(run_dir)
        direction = _infer_direction(label)
        ts = _infer_timestamp_from_dirname(run_dir)
        fname = _sanitize_filename(f"{ts}__{label}.png")
        out_path = singles_dir / direction / fname
        info = _render_single(
            run_dir=run_dir,
            out_path=out_path,
            label=label,
            trim=bool(args.trim),
            align_yaw=bool(args.align_yaw),
            stride=int(args.stride),
            dpi=int(args.dpi),
            fig_w=float(args.single_fig_w),
            fig_h=float(args.single_fig_h),
            pad_m=float(args.pad_m),
        )
        infos.append(info)
        by_direction.setdefault(direction, []).append(run_dir)
        labels[str(run_dir)] = str(label)

    # Pairs
    pair_rows: list[dict[str, str]] = []
    if bool(args.pairs):
        for direction, runs in sorted(by_direction.items()):
            runs_sorted = sorted(runs, key=lambda p: _infer_timestamp_from_dirname(p))
            n = len(runs_sorted)
            for i in range(n):
                for j in range(i + 1, n):
                    ra = runs_sorted[i]
                    rb = runs_sorted[j]
                    la = labels.get(str(ra), ra.name)
                    lb = labels.get(str(rb), rb.name)
                    out_name = _sanitize_filename(
                        f"{_infer_timestamp_from_dirname(ra)}__{la}__vs__{_infer_timestamp_from_dirname(rb)}__{lb}.png",
                        max_len=200,
                    )
                    out_path = pairs_dir / direction / out_name
                    _render_pair(
                        run_a=ra,
                        run_b=rb,
                        out_path=out_path,
                        label_a=la,
                        label_b=lb,
                        trim=bool(args.trim),
                        align_yaw=bool(args.align_yaw),
                        stride=int(args.stride),
                        dpi=int(args.dpi),
                        fig_w=float(args.pair_fig_w),
                        fig_h=float(args.pair_fig_h),
                        pad_m=float(args.pad_m),
                    )
                    pair_rows.append(
                        {
                            "direction": direction,
                            "run_a": str(ra),
                            "run_b": str(rb),
                            "out_png": str(out_path),
                        }
                    )

    # index.csv
    index_path = out_dir / "index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "direction",
                "timestamp",
                "policy_base",
                "label",
                "run_dir",
                "total_time_s",
                "total_dist_m",
                "takeovers",
                "takeover_time_s",
                "takeover_dist_m",
                "takeover_frac_time",
                "takeover_frac_dist",
                "out_png",
                "run_a",
                "run_b",
            ],
        )
        w.writeheader()
        for info in infos:
            out_png = str(
                singles_dir
                / info.direction
                / _sanitize_filename(f"{info.timestamp}__{info.label}.png")
            )
            row = {
                "kind": "single",
                "direction": info.direction,
                "timestamp": info.timestamp,
                "policy_base": info.policy_base,
                "label": info.label,
                "run_dir": info.run_dir,
                "total_time_s": f"{info.total_time_s:.6f}",
                "total_dist_m": f"{info.total_dist_m:.6f}",
                "takeovers": str(info.takeovers),
                "takeover_time_s": f"{info.takeover_time_s:.6f}",
                "takeover_dist_m": f"{info.takeover_dist_m:.6f}",
                "takeover_frac_time": f"{info.takeover_frac_time:.6f}",
                "takeover_frac_dist": f"{info.takeover_frac_dist:.6f}",
                "out_png": out_png,
                "run_a": "",
                "run_b": "",
            }
            w.writerow(row)
        for pr in pair_rows:
            w.writerow(
                {
                    "kind": "pair",
                    "direction": pr["direction"],
                    "timestamp": "",
                    "policy_base": "",
                    "label": "",
                    "run_dir": "",
                    "total_time_s": "",
                    "total_dist_m": "",
                    "takeovers": "",
                    "takeover_time_s": "",
                    "takeover_dist_m": "",
                    "takeover_frac_time": "",
                    "takeover_frac_dist": "",
                    "out_png": pr["out_png"],
                    "run_a": pr["run_a"],
                    "run_b": pr["run_b"],
                }
            )

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
