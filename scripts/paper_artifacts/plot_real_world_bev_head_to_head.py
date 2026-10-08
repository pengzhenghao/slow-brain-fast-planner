#!/usr/bin/env python3
"""
Head-to-head BEV trajectory plots for real-world runs.

This script reads:
  <run_dir>/telemetry/telemetry.jsonl

and uses the *local odom* pose stream:
  event == "odom_tick"
  local_pose_xyzw == [x, y, z, qx, qy, qz, qw]
  autonomy_enabled == bool

to render a bird's-eye trajectory, highlighting human takeover segments
where autonomy_enabled == False.

Example:
  python scripts/plot_real_world_bev_head_to_head.py \
    --a local_logs/real_world_0129/task2_live_2026-01-29_004047_vlm-stream-match_INT-MED \
    --b local_logs/real_world_0129/task2_live_2026-01-29_005229_vlm-score-fusion-stream_MED-INT \
    --out local_logs/real_world_0129/bev_head_to_head.png
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _try_import_matplotlib_pyplot():
    """Import matplotlib.pyplot safely (Agg backend)."""
    import contextlib
    import io

    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            import matplotlib  # type: ignore

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _yaw_from_quat_xyzw(q_xyzw: Any) -> float:
    """Yaw (rad) from quaternion [x,y,z,w] (ROS convention)."""
    try:
        q = np.asarray(q_xyzw, dtype=np.float64).reshape(-1)
    except Exception:
        return 0.0
    if q.size != 4:
        return 0.0
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _pose_xy_yaw_from_pose_xyzw(pose_xyzw: Any) -> tuple[float, float, float] | None:
    """Pose array is expected as [x,y,z,qx,qy,qz,qw]."""
    try:
        p = np.asarray(pose_xyzw, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if p.size < 7:
        return None
    x = float(p[0])
    y = float(p[1])
    yaw = _yaw_from_quat_xyzw(p[3:7])
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
        return None
    return x, y, yaw


@dataclass(frozen=True)
class OdomSeries:
    t_wall_s: np.ndarray  # [N]
    x_m: np.ndarray  # [N]
    y_m: np.ndarray  # [N]
    yaw_rad: np.ndarray  # [N]
    autonomy: np.ndarray  # [N] bool


def _load_run_label(run_dir: Path) -> str:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            exp_name = (cfg.get("logging", {}) or {}).get("exp_name", None)
            if isinstance(exp_name, str) and exp_name.strip():
                return exp_name.strip()
            policy = (cfg.get("control", {}) or {}).get("policy", None)
            if isinstance(policy, str) and policy.strip():
                return policy.strip()
        except Exception:
            pass
    return run_dir.name


def _load_odom_series(run_dir: Path, *, require_autonomy: bool = True) -> OdomSeries:
    tel_path = run_dir / "telemetry" / "telemetry.jsonl"
    if not tel_path.exists():
        raise FileNotFoundError(f"Missing telemetry file: {tel_path}")

    ts: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    yaws: list[float] = []
    autos: list[bool] = []

    for obj in _iter_jsonl(tel_path):
        if str(obj.get("event", "") or "") != "odom_tick":
            continue
        t = obj.get("t_wall_s", None)
        try:
            t_wall = float(t)
        except Exception:
            continue
        pose = obj.get("local_pose_xyzw", None)
        if pose is None:
            continue
        parsed = _pose_xy_yaw_from_pose_xyzw(pose)
        if parsed is None:
            continue
        x, y, yaw = parsed
        autonomy = obj.get("autonomy_enabled", None)
        if autonomy is None and require_autonomy:
            continue
        ts.append(float(t_wall))
        xs.append(float(x))
        ys.append(float(y))
        yaws.append(float(yaw))
        autos.append(bool(autonomy) if autonomy is not None else False)

    if len(ts) < 2:
        raise RuntimeError(f"Not enough odom_tick points in {tel_path} (got {len(ts)})")

    t_arr = np.asarray(ts, dtype=np.float64)
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    yaw_arr = np.asarray(yaws, dtype=np.float64)
    a_arr = np.asarray(autos, dtype=bool)

    order = np.argsort(t_arr)
    return OdomSeries(
        t_wall_s=t_arr[order],
        x_m=x_arr[order],
        y_m=y_arr[order],
        yaw_rad=yaw_arr[order],
        autonomy=a_arr[order],
    )


def _trim_to_first_last_autonomy(s: OdomSeries) -> OdomSeries:
    idx = np.where(s.autonomy)[0]
    if idx.size == 0:
        return s
    i0 = int(idx[0])
    i1 = int(idx[-1])
    if i1 <= i0:
        return s
    sl = slice(i0, i1 + 1)
    return OdomSeries(
        t_wall_s=s.t_wall_s[sl],
        x_m=s.x_m[sl],
        y_m=s.y_m[sl],
        yaw_rad=s.yaw_rad[sl],
        autonomy=s.autonomy[sl],
    )


def _downsample(s: OdomSeries, *, stride: int) -> OdomSeries:
    stride = int(max(1, int(stride)))
    sl = slice(None, None, stride)
    return OdomSeries(
        t_wall_s=s.t_wall_s[sl],
        x_m=s.x_m[sl],
        y_m=s.y_m[sl],
        yaw_rad=s.yaw_rad[sl],
        autonomy=s.autonomy[sl],
    )


def _align_xy_yaw(s: OdomSeries, *, align_yaw: bool) -> OdomSeries:
    x0 = float(s.x_m[0])
    y0 = float(s.y_m[0])
    dx = s.x_m - x0
    dy = s.y_m - y0
    if not align_yaw:
        return OdomSeries(
            t_wall_s=s.t_wall_s,
            x_m=dx,
            y_m=dy,
            yaw_rad=s.yaw_rad,
            autonomy=s.autonomy,
        )
    yaw0 = float(s.yaw_rad[0])
    c = float(math.cos(-yaw0))
    si = float(math.sin(-yaw0))
    xr = c * dx - si * dy
    yr = si * dx + c * dy
    return OdomSeries(
        t_wall_s=s.t_wall_s,
        x_m=xr,
        y_m=yr,
        yaw_rad=(s.yaw_rad - yaw0),
        autonomy=s.autonomy,
    )


@dataclass(frozen=True)
class TakeoverStats:
    total_time_s: float
    total_dist_m: float
    takeover_events: int
    takeover_time_s: float
    takeover_dist_m: float

    @property
    def takeover_frac_time(self) -> float:
        return float(self.takeover_time_s / max(1e-9, self.total_time_s))

    @property
    def takeover_frac_dist(self) -> float:
        return float(self.takeover_dist_m / max(1e-9, self.total_dist_m))


def _compute_takeover_stats(s: OdomSeries) -> TakeoverStats:
    t = np.asarray(s.t_wall_s, dtype=np.float64)
    x = np.asarray(s.x_m, dtype=np.float64)
    y = np.asarray(s.y_m, dtype=np.float64)
    a = np.asarray(s.autonomy, dtype=bool)

    if t.size < 2:
        return TakeoverStats(0.0, 0.0, 0, 0.0, 0.0)

    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)
    dd = np.sqrt(dx * dx + dy * dy)
    dt = np.asarray(dt, dtype=np.float64)
    dd = np.asarray(dd, dtype=np.float64)

    total_time_s = float(max(0.0, float(t[-1] - t[0])))
    total_dist_m = float(np.sum(dd)) if dd.size else 0.0

    takeover_events = 0
    for i in range(1, int(a.size)):
        if bool(a[i - 1]) is True and bool(a[i]) is False:
            takeover_events += 1

    # Attribute interval (i-1->i) to state at i (matches scripts/real_world_metrics.py).
    interval_auto = np.asarray(a[1:], dtype=bool)
    auto_dist_m = float(np.sum(dd[interval_auto])) if dd.size else 0.0
    auto_time_s = float(np.sum(dt[interval_auto])) if dt.size else 0.0
    takeover_dist_m = float(max(0.0, total_dist_m - auto_dist_m))
    takeover_time_s = float(max(0.0, total_time_s - auto_time_s))
    return TakeoverStats(
        total_time_s=total_time_s,
        total_dist_m=total_dist_m,
        takeover_events=int(takeover_events),
        takeover_time_s=takeover_time_s,
        takeover_dist_m=takeover_dist_m,
    )


def _iter_state_segments(state: np.ndarray) -> Iterator[tuple[int, int, bool]]:
    s = np.asarray(state, dtype=bool).reshape(-1)
    n = int(s.size)
    if n == 0:
        return
    start = 0
    for i in range(1, n):
        if bool(s[i]) != bool(s[i - 1]):
            yield start, i, bool(s[i - 1])
            start = i
    yield start, n, bool(s[-1])


def _plot_run(ax, *, series: OdomSeries, title: str, stats: TakeoverStats) -> None:
    x = np.asarray(series.x_m, dtype=np.float64)
    y = np.asarray(series.y_m, dtype=np.float64)
    a = np.asarray(series.autonomy, dtype=bool)

    # Draw per-state segments so we can color takeover vs autonomy.
    for i0, i1, is_auto in _iter_state_segments(a):
        if i1 - i0 < 2:
            continue
        if is_auto:
            color = "tab:blue"
            lw = 1.8
            alpha = 0.95
            z = 2
        else:
            color = "tab:red"
            lw = 2.8
            alpha = 0.95
            z = 3
        ax.plot(x[i0:i1], y[i0:i1], color=color, linewidth=lw, alpha=alpha, zorder=z)

    # If takeovers are very short (e.g., a single tick), segment plotting may not show red.
    # Overlay takeover ticks as faint points so "manual moments" are still visible.
    takeover_idx = np.where(~a)[0]
    if takeover_idx.size:
        ax.scatter(
            x[takeover_idx],
            y[takeover_idx],
            s=6,
            c="tab:red",
            alpha=0.35,
            linewidths=0.0,
            zorder=1,
        )

    # Mark takeover starts (True->False).
    takeover_starts = []
    for i in range(1, int(a.size)):
        if bool(a[i - 1]) is True and bool(a[i]) is False:
            takeover_starts.append(i)
    if takeover_starts:
        ax.scatter(
            x[np.asarray(takeover_starts, dtype=int)],
            y[np.asarray(takeover_starts, dtype=int)],
            s=18,
            c="tab:red",
            edgecolors="white",
            linewidths=0.5,
            zorder=4,
        )

    # Start/end markers.
    ax.scatter([x[0]], [y[0]], s=35, c="tab:green", edgecolors="black", linewidths=0.6, zorder=5)
    ax.scatter([x[-1]], [y[-1]], s=35, c="black", marker="X", zorder=5)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    txt = (
        f"dist: {stats.total_dist_m:.1f} m\n"
        f"time: {stats.total_time_s:.1f} s\n"
        f"takeovers: {stats.takeover_events}\n"
        f"takeover time: {stats.takeover_time_s:.1f} s ({100.0 * stats.takeover_frac_time:.1f}%)\n"
        f"takeover dist: {stats.takeover_dist_m:.1f} m ({100.0 * stats.takeover_frac_dist:.1f}%)"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "#dddddd",
        },
        zorder=10,
    )

    # Compact in-axis legend (avoids a global legend overlapping labels).
    try:
        from matplotlib.lines import Line2D  # type: ignore

        handles = [
            Line2D([0], [0], color="tab:blue", lw=2.0, label="autonomy"),
            Line2D([0], [0], color="tab:red", lw=3.0, label="manual (takeover)"),
            Line2D(
                [0], [0], marker="o", markersize=5, color="tab:red", lw=0.0, label="takeover start"
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=5,
                markerfacecolor="tab:green",
                markeredgecolor="black",
                color="none",
                lw=0.0,
                label="start",
            ),
            Line2D([0], [0], marker="X", markersize=6, color="black", lw=0.0, label="end"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.85, fontsize=8)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot head-to-head BEV trajectories (local odom) with takeover highlighting."
    )
    p.add_argument(
        "--a", type=str, required=True, help="Run dir A (contains telemetry/telemetry.jsonl)."
    )
    p.add_argument(
        "--b", type=str, required=True, help="Run dir B (contains telemetry/telemetry.jsonl)."
    )
    p.add_argument("--out", type=str, required=True, help="Output PNG path.")
    p.add_argument("--label-a", type=str, default=None, help="Optional label override for run A.")
    p.add_argument("--label-b", type=str, default=None, help="Optional label override for run B.")
    p.add_argument(
        "--trim",
        action="store_true",
        help="Trim leading/trailing manual segment (to first/last autonomy tick).",
    )
    p.add_argument(
        "--no-trim", dest="trim", action="store_false", help="Do not trim; plot full log window."
    )
    p.set_defaults(trim=True)
    p.add_argument(
        "--align-yaw", action="store_true", help="Rotate so the first heading points +x."
    )
    p.add_argument(
        "--no-align-yaw",
        dest="align_yaw",
        action="store_false",
        help="Do not rotate; only translate to origin.",
    )
    p.set_defaults(align_yaw=True)
    p.add_argument(
        "--stride", type=int, default=1, help="Downsample odom ticks by this stride (>=1)."
    )
    p.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    p.add_argument("--fig-w", type=float, default=10.5, help="Figure width (inches).")
    p.add_argument("--fig-h", type=float, default=4.8, help="Figure height (inches).")
    p.add_argument("--pad-m", type=float, default=0.5, help="Axis padding (meters).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plt = _try_import_matplotlib_pyplot()
    if plt is None:
        raise SystemExit("matplotlib is required to plot figures (failed to import).")

    run_a = Path(args.a).expanduser().resolve()
    run_b = Path(args.b).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_a = str(args.label_a).strip() if args.label_a else _load_run_label(run_a)
    label_b = str(args.label_b).strip() if args.label_b else _load_run_label(run_b)

    sa = _load_odom_series(run_a)
    sb = _load_odom_series(run_b)
    if bool(args.trim):
        sa = _trim_to_first_last_autonomy(sa)
        sb = _trim_to_first_last_autonomy(sb)
    if int(args.stride) > 1:
        sa = _downsample(sa, stride=int(args.stride))
        sb = _downsample(sb, stride=int(args.stride))

    # Always translate to origin; optionally rotate.
    sa = _align_xy_yaw(sa, align_yaw=bool(args.align_yaw))
    sb = _align_xy_yaw(sb, align_yaw=bool(args.align_yaw))

    sta = _compute_takeover_stats(sa)
    stb = _compute_takeover_stats(sb)

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(float(args.fig_w), float(args.fig_h)),
        dpi=int(args.dpi),
        constrained_layout=True,
    )
    ax_a, ax_b = axs[0], axs[1]
    _plot_run(ax_a, series=sa, title=label_a, stats=sta)
    _plot_run(ax_b, series=sb, title=label_b, stats=stb)

    # Shared axis limits for a fair visual comparison.
    pad = float(max(0.0, float(args.pad_m)))
    xs = np.concatenate([sa.x_m, sb.x_m], axis=0)
    ys = np.concatenate([sa.y_m, sb.y_m], axis=0)
    if xs.size and ys.size:
        xmin = float(np.min(xs)) - pad
        xmax = float(np.max(xs)) + pad
        ymin = float(np.min(ys)) - pad
        ymax = float(np.max(ys)) + pad
        # Make the view square-ish (keep equal aspect but same extents on both axes).
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        r = 0.5 * max((xmax - xmin), (ymax - ymin))
        for ax in (ax_a, ax_b):
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)

    fig.suptitle(
        "Real-world BEV trajectory (local odom) with human takeovers highlighted", fontsize=12
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
