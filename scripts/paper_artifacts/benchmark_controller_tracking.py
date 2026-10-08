from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files, load_episode
from slow_brain_fast_planner.control.tracking import (
    EndpointPDConfig,
    Limits,
    PurePursuitConfig,
    SimConfig,
    body_to_world,
    controller_endpoint_pd,
    controller_pure_pursuit,
    integrate_diff_drive,
    polyline_length,
    project_point_to_polyline_arclength,
    simulate_tracking,
    world_to_body,
)


def _try_import_tqdm():
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _try_import_matplotlib_pyplot():
    """Import matplotlib.pyplot safely.

    Some environments have a NumPy ABI mismatch; matplotlib import can spew large errors.
    We suppress stderr during import and return None on failure.
    """
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline controller tracking benchmark (endpoint-PD vs pure pursuit)."
    )
    p.add_argument(
        "--dataset", type=str, required=True, help="Canonical dataset root (contains episodes/)."
    )
    p.add_argument(
        "--out", type=str, default="logs/controller_benchmark.csv", help="Output CSV path."
    )
    p.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="Optional path to write a human-readable summary .txt alongside the CSV.",
    )
    p.add_argument(
        "--fig-dir",
        type=str,
        default=None,
        help="Optional directory to write figures (png). If unset, no figures are generated.",
    )
    p.add_argument("--no-progress", action="store_true", help="Disable progress printing.")
    p.add_argument(
        "--progress-interval-s", type=float, default=1.0, help="Progress print interval in seconds."
    )
    p.add_argument(
        "--trace-n",
        type=int,
        default=12,
        help="How many representative bird's-eye trajectory PNGs to render (0 disables).",
    )
    p.add_argument(
        "--trace-seed", type=int, default=0, help="Seed for selecting random representative traces."
    )
    p.add_argument("--max-episodes", type=int, default=10)
    p.add_argument("--max-records-per-episode", type=int, default=200)
    p.add_argument("--stride", type=int, default=5, help="Use every Nth planner_candidates record.")

    # Benchmark mode
    p.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "replan"],
        help="single: evaluate one frozen path per snapshot; replan: simulate receding-horizon "
        "replanning across snapshots in each episode.",
    )

    # Candidate selection
    p.add_argument("--candidate", type=str, default="argmax_score", choices=["argmax_score"])

    # Simulation
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--horizon-s", type=float, default=4.0)
    p.add_argument("--max-v", type=float, default=1.0)
    # Default matches deploy/NavFlow/visualnav_ros/deployment/config/robot.yaml
    p.add_argument("--max-w", type=float, default=0.5)
    p.add_argument("--max-a", type=float, default=1.0)
    p.add_argument("--max-alpha", type=float, default=1.2)
    p.add_argument("--max-lat-accel", type=float, default=0.8)

    # Pure pursuit knobs
    p.add_argument("--pp-lookahead-m", type=float, default=1.0)
    p.add_argument("--pp-lookahead-gain", type=float, default=0.5)
    p.add_argument("--pp-v-gain", type=float, default=0.8)

    # Endpoint PD knobs
    p.add_argument("--ep-dt-nominal", type=float, default=1.0)
    # Match latest on-robot PD heuristic (curvature slow-down disabled by default).
    p.add_argument("--ep-curvature-speed-gain", type=float, default=0.0)

    return p.parse_args()


def pick_candidate_points_xy(record) -> np.ndarray | None:
    # record: PlannerCandidatesRecord
    if not record.candidates:
        return None
    scores = []
    for c in record.candidates:
        try:
            scores.append(float(c.score))
        except Exception:
            scores.append(float("-inf"))
    idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    pts = record.candidates[idx].points_xy
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        return None
    return arr


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out = Path(args.summary_out).resolve() if args.summary_out else None
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.fig_dir).resolve() if args.fig_dir else None
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    meta_files = find_episode_metadata_files(dataset_root)
    meta_files = meta_files[: int(max(0, args.max_episodes))]

    limits = Limits(
        max_v=float(args.max_v),
        max_w=float(args.max_w),
        max_a=float(args.max_a),
        max_alpha=float(args.max_alpha),
        max_lat_accel=float(args.max_lat_accel),
    )
    sim = SimConfig(dt=float(args.dt), horizon_s=float(args.horizon_s), limits=limits)
    pp_cfg = PurePursuitConfig(
        lookahead_m=float(args.pp_lookahead_m),
        lookahead_gain=float(args.pp_lookahead_gain),
        v_gain=float(args.pp_v_gain),
    )
    ep_cfg = EndpointPDConfig(
        dt_nominal=float(args.ep_dt_nominal),
        curvature_speed_gain=float(args.ep_curvature_speed_gain),
    )

    rows: list[dict[str, object]] = []
    paths_xy: list[np.ndarray] = []
    mode = str(getattr(args, "mode", "single")).lower().strip()

    show_progress = not bool(args.no_progress)
    eps_total = int(len(meta_files))
    eps_done = 0
    rec_scanned = 0
    rec_used = 0

    tqdm = _try_import_tqdm()
    ep_iter = meta_files
    tbar = None
    if show_progress and tqdm is not None:
        tbar = tqdm(meta_files, total=eps_total, desc="episodes", dynamic_ncols=True)
        ep_iter = tbar

    last_print_wall = 0.0
    progress_interval_s = float(max(0.1, float(args.progress_interval_s)))

    def _fallback_progress(done: bool = False) -> None:
        nonlocal last_print_wall
        if not show_progress:
            return
        if tbar is not None:
            try:
                tbar.set_postfix(scanned=int(rec_scanned), used=int(rec_used))
            except Exception:
                pass
            return
        # lightweight line progress when tqdm isn't available
        now = time.time()
        if (not done) and (now - last_print_wall) < progress_interval_s:
            return
        last_print_wall = now
        pct = 100.0 * float(eps_done) / float(max(1, eps_total))
        print(
            f"progress: {pct:5.1f}% eps={eps_done}/{eps_total} scanned={rec_scanned} used={rec_used}"
        )

    def _simulate_episode_replan(
        *,
        times_s: list[float],
        paths_body: list[np.ndarray],
        controller: str,
    ) -> dict[str, float]:
        """Simulate a receding-horizon replanning loop over an episode.

        At each record time, we "replan" by freezing the current local path (robot frame)
        into world coordinates at the robot's *current simulated pose*, then track it
        until the next record time.
        """
        if not times_s or not paths_body:
            return {
                "cte_mean": float("nan"),
                "cte_p95": float("nan"),
                "cte_max": float("nan"),
                "end_err": float("nan"),
                "progress_final_m": float("nan"),
                "progress_max_m": float("nan"),
                "progress_frac": float("nan"),
            }

        dt = float(sim.dt)
        t0 = float(times_s[0])
        times_rel = [float(t) - t0 for t in times_s]
        # Simulate until the last available replan time + a short tail to execute the last plan.
        total_s = float(max(0.0, times_rel[-1])) + float(sim.horizon_s)
        steps = int(max(1, round(total_s / dt)))

        x = 0.0
        y = 0.0
        yaw = 0.0
        v = 0.0
        w = 0.0

        idx = 0
        ref_world = body_to_world(np.asarray(paths_body[0], dtype=np.float64), x=x, y=y, yaw=yaw)
        ctes: list[float] = []
        prog_fracs: list[float] = []
        prog_ms: list[float] = []

        for k in range(steps):
            t = float(k) * dt
            while (idx + 1) < len(times_rel) and t >= float(times_rel[idx + 1]):
                idx += 1
                ref_world = body_to_world(
                    np.asarray(paths_body[idx], dtype=np.float64), x=x, y=y, yaw=yaw
                )

            # Current path in body frame (for controller), computed from frozen ref_world
            path_body = world_to_body(ref_world, x=x, y=y, yaw=yaw)

            if controller == "endpoint_pd":
                v_cmd, w_cmd = controller_endpoint_pd(
                    path_body, v_prev=v, w_prev=w, limits=sim.limits, cfg=ep_cfg
                )
            else:
                v_cmd, w_cmd = controller_pure_pursuit(
                    path_body, v_prev=v, w_prev=w, limits=sim.limits, cfg=pp_cfg
                )
                # Apply proper rate limits using dt (same idea as simulate_tracking)
                dv = float(sim.limits.max_a) * dt
                dw = float(sim.limits.max_alpha) * dt
                v_cmd = float(np.clip(v_cmd, v - dv, v + dv))
                w_cmd = float(np.clip(w_cmd, w - dw, w + dw))

            v = float(np.clip(v_cmd, 0.0, float(sim.limits.max_v)))
            w = float(np.clip(w_cmd, -float(sim.limits.max_w), float(sim.limits.max_w)))
            x, y, yaw = integrate_diff_drive(x, y, yaw, v, w, dt)

            s, d = project_point_to_polyline_arclength(
                np.array([x, y], dtype=np.float64), ref_world
            )
            ref_len = polyline_length(ref_world)
            ctes.append(float(d))
            prog_ms.append(float(s))
            prog_fracs.append(float(s / ref_len) if ref_len > 1e-8 else 0.0)

        # end_err: distance to last frozen plan endpoint in world
        end_ref = (
            np.asarray(ref_world[-1], dtype=np.float64)
            if ref_world.shape[0]
            else np.array([0.0, 0.0])
        )
        end_err = float(np.linalg.norm(np.array([x, y], dtype=np.float64) - end_ref))

        def _q(xs: list[float], q: float) -> float:
            if not xs:
                return float("nan")
            return float(np.quantile(np.asarray(xs, dtype=np.float64), q))

        return {
            "cte_mean": float(np.mean(ctes)) if ctes else float("nan"),
            "cte_p95": _q(ctes, 0.95),
            "cte_max": float(np.max(ctes)) if ctes else float("nan"),
            "end_err": float(end_err),
            "progress_final_m": float(prog_ms[-1]) if prog_ms else 0.0,
            "progress_max_m": float(max(prog_ms)) if prog_ms else 0.0,
            "progress_frac": float(np.mean(prog_fracs)) if prog_fracs else float("nan"),
        }

    for ep_meta in ep_iter:
        ep = load_episode(dataset_root, ep_meta)
        if not ep.schema_valid:
            eps_done += 1
            _fallback_progress()
            continue
        pcs = ep.planner_candidates
        if not pcs:
            eps_done += 1
            _fallback_progress()
            continue
        pcs = pcs[: int(args.max_records_per_episode)]
        stride = max(1, int(args.stride))
        if mode == "replan":
            times_s: list[float] = []
            paths_body: list[np.ndarray] = []
            frame = "robot"
            for j, rec in enumerate(pcs):
                rec_scanned += 1
                if j % stride != 0:
                    continue
                pts = pick_candidate_points_xy(rec)
                if pts is None:
                    continue
                if getattr(rec, "frame", "robot") != "robot":
                    frame = str(getattr(rec, "frame", "robot"))
                times_s.append(float(rec.t))
                paths_body.append(pts)
                rec_used += 1

            if len(paths_body) >= 1:
                res_ep = _simulate_episode_replan(
                    times_s=times_s, paths_body=paths_body, controller="endpoint_pd"
                )
                res_pp = _simulate_episode_replan(
                    times_s=times_s, paths_body=paths_body, controller="pure_pursuit"
                )
                row = {
                    "episode_id": ep.episode_id,
                    "t": float(times_s[0]) if times_s else 0.0,
                    "frame": frame,
                    "path_len_m": float(np.mean([polyline_length(p) for p in paths_body]))
                    if paths_body
                    else 0.0,
                    "endpoint_pd_cte_mean": res_ep["cte_mean"],
                    "endpoint_pd_cte_p95": res_ep["cte_p95"],
                    "endpoint_pd_cte_max": res_ep["cte_max"],
                    "endpoint_pd_end_err": res_ep["end_err"],
                    "endpoint_pd_progress_final_m": res_ep.get("progress_final_m"),
                    "endpoint_pd_progress_max_m": res_ep.get("progress_max_m"),
                    "endpoint_pd_progress_frac": res_ep.get("progress_frac"),
                    "pure_pursuit_cte_mean": res_pp["cte_mean"],
                    "pure_pursuit_cte_p95": res_pp["cte_p95"],
                    "pure_pursuit_cte_max": res_pp["cte_max"],
                    "pure_pursuit_end_err": res_pp["end_err"],
                    "pure_pursuit_progress_final_m": res_pp.get("progress_final_m"),
                    "pure_pursuit_progress_max_m": res_pp.get("progress_max_m"),
                    "pure_pursuit_progress_frac": res_pp.get("progress_frac"),
                }
                rows.append(row)
        else:
            for j, rec in enumerate(pcs):
                rec_scanned += 1
                if j % stride != 0:
                    continue
                pts = pick_candidate_points_xy(rec)
                if pts is None:
                    continue
                # Enforce robot-frame convention (expected by this benchmark)
                if getattr(rec, "frame", "robot") != "robot":
                    # Still run, but annotate.
                    frame = str(getattr(rec, "frame", "robot"))
                else:
                    frame = "robot"

                res_ep = simulate_tracking(
                    pts, controller="endpoint_pd", sim=sim, pp_cfg=pp_cfg, ep_cfg=ep_cfg
                )
                res_pp = simulate_tracking(
                    pts, controller="pure_pursuit", sim=sim, pp_cfg=pp_cfg, ep_cfg=ep_cfg
                )

                row = {
                    "episode_id": ep.episode_id,
                    "t": float(rec.t),
                    "frame": frame,
                    "path_len_m": float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))),
                    "endpoint_pd_cte_mean": res_ep["cte_mean"],
                    "endpoint_pd_cte_p95": res_ep["cte_p95"],
                    "endpoint_pd_cte_max": res_ep["cte_max"],
                    "endpoint_pd_end_err": res_ep["end_err"],
                    "endpoint_pd_progress_final_m": res_ep.get("progress_final_m"),
                    "endpoint_pd_progress_max_m": res_ep.get("progress_max_m"),
                    "endpoint_pd_progress_frac": res_ep.get("progress_frac"),
                    "pure_pursuit_cte_mean": res_pp["cte_mean"],
                    "pure_pursuit_cte_p95": res_pp["cte_p95"],
                    "pure_pursuit_cte_max": res_pp["cte_max"],
                    "pure_pursuit_end_err": res_pp["end_err"],
                    "pure_pursuit_progress_final_m": res_pp.get("progress_final_m"),
                    "pure_pursuit_progress_max_m": res_pp.get("progress_max_m"),
                    "pure_pursuit_progress_frac": res_pp.get("progress_frac"),
                }
                rows.append(row)
                paths_xy.append(pts)
                rec_used += 1
        eps_done += 1
        _fallback_progress()

    if not rows:
        raise SystemExit("No records evaluated. Check dataset path and streams.")
    if tbar is not None:
        try:
            tbar.set_postfix(scanned=int(rec_scanned), used=int(rec_used))
            tbar.close()
        except Exception:
            pass
    else:
        _fallback_progress(done=True)

    # Write CSV
    cols = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # Print a summary to stdout (and optionally write a human-readable report).
    def _col(col: str) -> np.ndarray:
        xs = [float(r[col]) for r in rows if r.get(col) is not None]
        return np.asarray(xs, dtype=np.float64)

    def _stats(col: str) -> dict[str, float]:
        x = _col(col)
        if x.size == 0:
            return {
                "mean": float("nan"),
                "median": float("nan"),
                "p90": float("nan"),
                "p95": float("nan"),
            }
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "p90": float(np.quantile(x, 0.90)),
            "p95": float(np.quantile(x, 0.95)),
        }

    summary = {
        "n": len(rows),
        "dataset": str(dataset_root),
        "out_csv": str(out_path),
        "config": {
            "max_episodes": int(args.max_episodes),
            "max_records_per_episode": int(args.max_records_per_episode),
            "stride": int(args.stride),
            "dt": float(args.dt),
            "horizon_s": float(args.horizon_s),
            "limits": {
                "max_v": float(args.max_v),
                "max_w": float(args.max_w),
                "max_a": float(args.max_a),
                "max_alpha": float(args.max_alpha),
                "max_lat_accel": float(args.max_lat_accel),
            },
            "pure_pursuit": {
                "lookahead_m": float(args.pp_lookahead_m),
                "lookahead_gain": float(args.pp_lookahead_gain),
                "v_gain": float(args.pp_v_gain),
            },
            "endpoint_pd": {
                "dt_nominal": float(args.ep_dt_nominal),
                "curvature_speed_gain": float(args.ep_curvature_speed_gain),
            },
        },
        "endpoint_pd": {
            "cte_mean": _stats("endpoint_pd_cte_mean"),
            "cte_p95": _stats("endpoint_pd_cte_p95"),
            "cte_max": _stats("endpoint_pd_cte_max"),
            "end_err": _stats("endpoint_pd_end_err"),
            "progress_final_m": _stats("endpoint_pd_progress_final_m"),
            "progress_max_m": _stats("endpoint_pd_progress_max_m"),
            "progress_frac": _stats("endpoint_pd_progress_frac"),
        },
        "pure_pursuit": {
            "cte_mean": _stats("pure_pursuit_cte_mean"),
            "cte_p95": _stats("pure_pursuit_cte_p95"),
            "cte_max": _stats("pure_pursuit_cte_max"),
            "end_err": _stats("pure_pursuit_end_err"),
            "progress_final_m": _stats("pure_pursuit_progress_final_m"),
            "progress_max_m": _stats("pure_pursuit_progress_max_m"),
            "progress_frac": _stats("pure_pursuit_progress_frac"),
        },
    }
    print(json.dumps(summary, indent=2))

    if summary_out is not None:
        lines: list[str] = []
        lines.append("Controller tracking benchmark (offline simulation)")
        lines.append("")
        lines.append(f"Dataset: {dataset_root}")
        lines.append(f"Output CSV: {out_path}")
        lines.append(f"Evaluated snapshots (rows): {len(rows)}")
        lines.append("")
        lines.append("What this benchmark does")
        lines.append(
            "- For each snapshot, it reads planner_candidates.jsonl and selects the argmax-score "
            "candidate trajectory."
        )
        lines.append(
            "- It treats that polyline (points_xy in robot frame at that snapshot) as a *reference "
            "path*."
        )
        lines.append(
            "- It then simulates a skid-steer robot tracking that reference path for the horizon."
        )
        lines.append("- Controllers compared:")
        lines.append(
            "  - endpoint_pd: endpoint-based heuristic (similar spirit to current deployment PD "
            "mapping)"
        )
        lines.append("  - pure_pursuit: pure pursuit path tracking + curvature-limited speed")
        lines.append("")
        lines.append("Robot / dynamics assumptions (important)")
        lines.append("- Plant model: ideal unicycle (x,y,yaw) with control (v, w).")
        lines.append(
            "- Perfect state estimate, no actuator delay, no wheel slip, no terrain effects."
        )
        lines.append(
            "- This is NOT a safety/obstacle benchmark; it only measures *path tracking quality* "
            "given a planned polyline."
        )
        lines.append("")
        lines.append("What is NOT modeled (so robot width/height/mass are not used)")
        lines.append(
            "- Robot footprint (width/length) and any collision checking / obstacle clearance."
        )
        lines.append(
            "- Mass / inertia explicitly. The only proxy is via caps like max_a (linear accel) and "
            "max_alpha (yaw accel)."
        )
        lines.append(
            "- Actuation latency, controller update rate mismatch, wheel slip, saturation "
            "dynamics, low-level PID behavior."
        )
        lines.append(
            "- Localization error (odom drift, GPS noise), time sync issues, or frame misalignment."
        )
        lines.append("")
        lines.append('Why the "hyperparameters" matter (and what to tune in real world)')
        lines.append(
            "- dt: controller tick used by the simulator. Set to your real control loop period "
            "(e.g., 0.05 for 20Hz)."
        )
        lines.append(
            "- horizon_s: how long we simulate tracking per snapshot (e.g., 2-4s). Longer punishes "
            "instability more."
        )
        lines.append(
            "- max_v (m/s): hard cap on commanded forward speed. Set to a safe max for your "
            "platform + environment."
        )
        lines.append(
            "- max_w (rad/s): hard cap on yaw rate. Too high can cause overshoot/slip; too low "
            "makes turns sluggish."
        )
        lines.append(
            "- max_a (m/s^2): cap on how fast v can change (comfort + traction). Too low -> can't "
            "correct; too high -> jerky/slip."
        )
        lines.append(
            '- max_alpha (rad/s^2): cap on how fast w can change. Controls "snappiness" of heading '
            "corrections."
        )
        lines.append(
            "- max_lat_accel (m/s^2): limits speed on curves via a_lat = v^2 * |kappa|. Critical "
            "to prevent cornering slip/tip."
        )
        lines.append("")
        lines.append("Controller-specific knobs")
        lines.append("- Pure pursuit:")
        lines.append(
            "  - pp_lookahead_m: base lookahead distance. Bigger = smoother but cuts corners; "
            "smaller = tighter but can oscillate."
        )
        lines.append(
            "  - pp_lookahead_gain: how lookahead grows with speed (Ld = L0 + k*v). Helps "
            "stability at higher speeds."
        )
        lines.append(
            "  - pp_v_gain: how aggressively speed increases with target distance. Too high = "
            "overshoot; too low = slow progress."
        )
        lines.append("- Endpoint-PD baseline:")
        lines.append(
            "  - ep_dt_nominal: scales v and w from endpoint geometry (heuristic). Larger -> less "
            "aggressive."
        )
        lines.append(
            "  - ep_curvature_speed_gain: slows down in curved segments (v *= exp(-k*curvature)). "
            "Higher = more cautious in turns."
        )
        lines.append("")
        lines.append("Where the default values came from")
        lines.append(
            '- They are conservative "reasonable defaults" for a small-to-medium diff-drive robot '
            "indoors."
        )
        lines.append(
            "- They are NOT estimated from your robot logs (unless you explicitly set them via "
            "env/CLI)."
        )
        lines.append("")
        lines.append("Real-world tuning checklist (practical)")
        lines.append(
            "- Measure/confirm max_v and max_w your base can actually achieve without wheel slip."
        )
        lines.append(
            "- Set max_a/max_alpha to avoid jerky motion; start low and increase until tracking "
            "improves without oscillation."
        )
        lines.append(
            "- Set max_lat_accel conservatively if the robot is tall/heavy (tip risk) or floors "
            "are slippery."
        )
        lines.append(
            "- Tune pure pursuit lookahead against speed: at higher speeds you usually need larger "
            "lookahead."
        )
        lines.append(
            "- Verify coordinate conventions: your planner points must be in robot frame (x "
            "forward, y left) and time-aligned with odom."
        )
        lines.append("")
        lines.append("Metrics (per snapshot)")
        lines.append(
            "- cte_mean: mean cross-track error (meters) over the simulated horizon; lower = "
            "tracks path more closely."
        )
        lines.append(
            "- cte_p95 : 95th percentile cross-track error (meters) over the horizon; lower = "
            "fewer large deviations."
        )
        lines.append(
            "- cte_max : maximum cross-track error (meters) over the horizon; lower = worst-case "
            "better."
        )
        lines.append(
            "- end_err : distance (meters) between simulated final position and the reference "
            "endpoint; lower = reaches endpoint better."
        )
        lines.append(
            "- progress_final_m: arclength (meters) advanced along the reference path at the end "
            "of the horizon; higher = makes more progress."
        )
        lines.append(
            "- progress_max_m: maximum arclength achieved during the horizon (helps detect "
            "oscillation/backtracking)."
        )
        lines.append(
            "- progress_frac: progress_final_m / path_len_m (unitless 0..1+); higher = reaches "
            "further fraction of the planned path."
        )
        lines.append("")
        lines.append("Interpretation note")
        lines.append(
            "- A controller can have low cte_mean but still large end_err if it stays near the "
            "path but makes slow progress."
        )
        lines.append("")
        lines.append("Aggregate results (across all evaluated snapshots)")

        def fmt_stat(d: dict[str, float]) -> str:
            return f"mean={d['mean']:.4f}  median={d['median']:.4f}  p90={d['p90']:.4f}  p95={d['p95']:.4f}"

        lines.append("")
        lines.append("endpoint_pd:")
        lines.append(f"  cte_mean: {fmt_stat(summary['endpoint_pd']['cte_mean'])}")
        lines.append(f"  cte_p95 : {fmt_stat(summary['endpoint_pd']['cte_p95'])}")
        lines.append(f"  cte_max : {fmt_stat(summary['endpoint_pd']['cte_max'])}")
        lines.append(f"  end_err : {fmt_stat(summary['endpoint_pd']['end_err'])}")
        lines.append(f"  progress_final_m: {fmt_stat(summary['endpoint_pd']['progress_final_m'])}")
        lines.append(f"  progress_max_m  : {fmt_stat(summary['endpoint_pd']['progress_max_m'])}")
        lines.append(f"  progress_frac   : {fmt_stat(summary['endpoint_pd']['progress_frac'])}")
        lines.append("")
        lines.append("pure_pursuit:")
        lines.append(f"  cte_mean: {fmt_stat(summary['pure_pursuit']['cte_mean'])}")
        lines.append(f"  cte_p95 : {fmt_stat(summary['pure_pursuit']['cte_p95'])}")
        lines.append(f"  cte_max : {fmt_stat(summary['pure_pursuit']['cte_max'])}")
        lines.append(f"  end_err : {fmt_stat(summary['pure_pursuit']['end_err'])}")
        lines.append(f"  progress_final_m: {fmt_stat(summary['pure_pursuit']['progress_final_m'])}")
        lines.append(f"  progress_max_m  : {fmt_stat(summary['pure_pursuit']['progress_max_m'])}")
        lines.append(f"  progress_frac   : {fmt_stat(summary['pure_pursuit']['progress_frac'])}")
        lines.append("")
        lines.append("Caveats / next steps")
        lines.append(
            "- If you want a stronger proxy for real-world performance, add slip/noise/delay "
            "models and control smoothness metrics."
        )
        lines.append(
            "- If your dataset contains logged control (v,w), you can compare simulated commands "
            "vs logs for realism."
        )
        lines.append("")
        summary_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Optional figures
    if fig_dir is not None:
        # Prefer standard matplotlib PNGs. If matplotlib can't import (usually NumPy ABI mismatch),
        # we fall back to pure-Python SVG + PIL-based trajectory PNGs.
        plt = _try_import_matplotlib_pyplot()
        use_svg_fallback = plt is None
        wrote: list[str] = []
        if plt is not None:
            # Metric plots as PNG (histograms + scatter)
            def _mpl_hist_pair(
                col_a: str, col_b: str, *, title: str, xlabel: str, out_name: str, bins: int = 60
            ) -> None:
                a = _col(col_a)
                b = _col(col_b)
                if a.size == 0 or b.size == 0:
                    return
                fig = plt.figure(figsize=(7.2, 4.6))
                ax = fig.add_subplot(1, 1, 1)
                ax.hist(a, bins=bins, alpha=0.55, label="endpoint_pd", color="#1f77b4")
                ax.hist(b, bins=bins, alpha=0.55, label="pure_pursuit", color="#ff7f0e")
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("count")
                ax.grid(True, alpha=0.25)
                ax.legend()
                fig.tight_layout()
                outp = fig_dir / out_name
                fig.savefig(str(outp), dpi=160)
                plt.close(fig)
                wrote.append(str(outp))

            def _mpl_scatter_pair(
                x_col: str,
                y_a: str,
                y_b: str,
                *,
                title: str,
                xlabel: str,
                ylabel: str,
                out_name: str,
            ) -> None:
                x = _col(x_col)
                a = _col(y_a)
                b = _col(y_b)
                if x.size == 0 or a.size == 0 or b.size == 0:
                    return
                n = int(min(x.size, a.size, b.size))
                x = x[:n]
                a = a[:n]
                b = b[:n]
                # subsample for size
                max_points = 6000
                if n > max_points:
                    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
                    x = x[idx]
                    a = a[idx]
                    b = b[idx]
                fig = plt.figure(figsize=(7.2, 4.8))
                ax = fig.add_subplot(1, 1, 1)
                ax.scatter(x, a, s=10, alpha=0.35, label="endpoint_pd", color="#1f77b4")
                ax.scatter(x, b, s=10, alpha=0.35, label="pure_pursuit", color="#ff7f0e")
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
                ax.legend()
                fig.tight_layout()
                outp = fig_dir / out_name
                fig.savefig(str(outp), dpi=160)
                plt.close(fig)
                wrote.append(str(outp))

            _mpl_hist_pair(
                "endpoint_pd_cte_mean",
                "pure_pursuit_cte_mean",
                title="CTE mean distribution",
                xlabel="cte_mean (m)",
                out_name="cte_mean_hist.png",
            )
            _mpl_hist_pair(
                "endpoint_pd_end_err",
                "pure_pursuit_end_err",
                title="Endpoint error distribution",
                xlabel="end_err (m)",
                out_name="end_err_hist.png",
            )
            _mpl_hist_pair(
                "endpoint_pd_progress_frac",
                "pure_pursuit_progress_frac",
                title="Progress fraction distribution",
                xlabel="progress_frac (0..1)",
                out_name="progress_frac_hist.png",
            )
            _mpl_scatter_pair(
                "path_len_m",
                "endpoint_pd_end_err",
                "pure_pursuit_end_err",
                title="Endpoint error vs path length",
                xlabel="path_len_m (m)",
                ylabel="end_err (m)",
                out_name="end_err_vs_path_len.png",
            )
            _mpl_scatter_pair(
                "path_len_m",
                "endpoint_pd_progress_final_m",
                "pure_pursuit_progress_final_m",
                title="Progress vs path length",
                xlabel="path_len_m (m)",
                ylabel="progress_final_m (m)",
                out_name="progress_vs_path_len.png",
            )

        # Always also write SVGs (quick to view in a browser) if matplotlib isn't available.
        if use_svg_fallback:

            def _svg_escape(s: str) -> str:
                return (
                    str(s)
                    .replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&apos;")
                )

            def _write_svg(path: Path, *, w: int, h: int, body: str) -> None:
                svg = (
                    f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' "
                    f"viewBox='0 0 {w} {h}' font-family='DejaVu Sans, Arial, sans-serif'>\n"
                    f"{body}\n</svg>\n"
                )
                path.write_text(svg, encoding="utf-8")
                wrote.append(str(path))

            def _svg_axes(*, w: int, h: int, m: int) -> tuple[int, int, int, int]:
                # plot box: [x0,x1] x [y0,y1]
                x0 = m
                x1 = w - m
                y0 = m
                y1 = h - m
                return x0, x1, y0, y1

            def _svg_hist_pair(
                col_a: str,
                col_b: str,
                *,
                title: str,
                xlabel: str,
                out_name: str,
                bins: int = 60,
            ) -> None:
                a = _col(col_a)
                b = _col(col_b)
                if a.size == 0 or b.size == 0:
                    return
                lo = float(min(float(np.min(a)), float(np.min(b))))
                hi = float(max(float(np.max(a)), float(np.max(b))))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    return
                edges = np.linspace(lo, hi, int(bins) + 1, dtype=np.float64)
                ha, _ = np.histogram(a, bins=edges)
                hb, _ = np.histogram(b, bins=edges)
                ymax = int(max(int(np.max(ha)), int(np.max(hb)), 1))

                W, H, M = 900, 520, 70
                x0, x1, y0, y1 = _svg_axes(w=W, h=H, m=M)
                pw = x1 - x0
                ph = y1 - y0
                nb = len(edges) - 1
                bw = pw / float(nb)

                def x_at(i: int) -> float:
                    return float(x0) + float(i) * bw

                def y_at(v: float) -> float:
                    # v in [0,ymax]
                    return float(y1) - (float(v) / float(ymax)) * float(ph)

                body = []
                body.append(
                    f"<text x='{W / 2:.1f}' y='30' text-anchor='middle' font-size='18'>{_svg_escape(title)}</text>"
                )
                body.append(
                    f"<text x='{W / 2:.1f}' y='{H - 10}' text-anchor='middle' font-size='14'>{_svg_escape(xlabel)}</text>"
                )
                body.append(
                    f"<line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#222' stroke-width='2' />"
                )
                body.append(
                    f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#222' stroke-width='2' />"
                )

                # ticks
                for t in range(6):
                    xv = float(x0) + float(t) * (pw / 5.0)
                    val = lo + float(t) * (hi - lo) / 5.0
                    body.append(
                        f"<line x1='{xv:.1f}' y1='{y1}' x2='{xv:.1f}' y2='{y1 + 6}' stroke='#222' />"
                    )
                    body.append(
                        f"<text x='{xv:.1f}' y='{y1 + 22}' text-anchor='middle' font-size='12'>{val:.3g}</text>"
                    )
                for t in range(6):
                    yv = float(y1) - float(t) * (ph / 5.0)
                    val = float(t) * float(ymax) / 5.0
                    body.append(
                        f"<line x1='{x0 - 6}' y1='{yv:.1f}' x2='{x0}' y2='{yv:.1f}' stroke='#222' />"
                    )
                    body.append(
                        f"<text x='{x0 - 10}' y='{yv + 4:.1f}' text-anchor='end' font-size='12'>{val:.0f}</text>"
                    )

                # bars (overlay with alpha)
                for i in range(nb):
                    xa = x_at(i)
                    # endpoint_pd
                    ya = y_at(float(ha[i]))
                    hb_i = float(hb[i])
                    yb = y_at(hb_i)
                    # draw endpoint_pd first, then pure_pursuit
                    body.append(
                        f"<rect x='{xa:.2f}' y='{ya:.2f}' width='{bw:.2f}' height='{float(y1) - ya:.2f}' "
                        f"fill='#1f77b4' fill-opacity='0.55' stroke='none' />"
                    )
                    body.append(
                        f"<rect x='{xa:.2f}' y='{yb:.2f}' width='{bw:.2f}' height='{float(y1) - yb:.2f}' "
                        f"fill='#ff7f0e' fill-opacity='0.55' stroke='none' />"
                    )

                # legend
                lx, ly = x1 - 240, y0 + 12
                body.append(
                    f"<rect x='{lx}' y='{ly}' width='230' height='54' fill='white' stroke='#ccc' />"
                )
                body.append(
                    f"<rect x='{lx + 10}' y='{ly + 12}' width='18' height='10' fill='#1f77b4' fill-opacity='0.55' />"
                )
                body.append(f"<text x='{lx + 36}' y='{ly + 22}' font-size='12'>endpoint_pd</text>")
                body.append(
                    f"<rect x='{lx + 10}' y='{ly + 32}' width='18' height='10' fill='#ff7f0e' fill-opacity='0.55' />"
                )
                body.append(f"<text x='{lx + 36}' y='{ly + 42}' font-size='12'>pure_pursuit</text>")

                _write_svg(fig_dir / out_name, w=W, h=H, body="\n".join(body))

            def _svg_scatter_pair(
                x_col: str,
                y_a: str,
                y_b: str,
                *,
                title: str,
                xlabel: str,
                ylabel: str,
                out_name: str,
            ) -> None:
                x = _col(x_col)
                a = _col(y_a)
                b = _col(y_b)
                if x.size == 0 or a.size == 0 or b.size == 0:
                    return
                n = int(min(x.size, a.size, b.size))
                x = x[:n]
                a = a[:n]
                b = b[:n]
                # subsample for file size
                max_points = 4000
                if n > max_points:
                    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
                    x = x[idx]
                    a = a[idx]
                    b = b[idx]
                    n = int(x.size)

                xmin = float(np.min(x))
                xmax = float(np.max(x))
                ymin = float(min(float(np.min(a)), float(np.min(b))))
                ymax = float(max(float(np.max(a)), float(np.max(b))))
                if not (
                    np.isfinite(xmin)
                    and np.isfinite(xmax)
                    and np.isfinite(ymin)
                    and np.isfinite(ymax)
                ):
                    return
                if xmax <= xmin:
                    xmax = xmin + 1.0
                if ymax <= ymin:
                    ymax = ymin + 1.0

                W, H, M = 900, 560, 80
                x0, x1, y0, y1 = _svg_axes(w=W, h=H, m=M)
                pw = x1 - x0
                ph = y1 - y0

                def xp(v: float) -> float:
                    return float(x0) + (float(v) - xmin) / (xmax - xmin) * float(pw)

                def yp(v: float) -> float:
                    return float(y1) - (float(v) - ymin) / (ymax - ymin) * float(ph)

                body = []
                body.append(
                    f"<text x='{W / 2:.1f}' y='30' text-anchor='middle' font-size='18'>{_svg_escape(title)}</text>"
                )
                body.append(
                    f"<text x='{W / 2:.1f}' y='{H - 10}' text-anchor='middle' font-size='14'>{_svg_escape(xlabel)}</text>"
                )
                body.append(
                    f"<text x='18' y='{H / 2:.1f}' text-anchor='middle' font-size='14' transform='rotate(-90 18 {H / 2:.1f})'>{_svg_escape(ylabel)}</text>"
                )
                body.append(
                    f"<line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#222' stroke-width='2' />"
                )
                body.append(
                    f"<line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#222' stroke-width='2' />"
                )

                # ticks
                for t in range(6):
                    xv = float(x0) + float(t) * (pw / 5.0)
                    val = xmin + float(t) * (xmax - xmin) / 5.0
                    body.append(
                        f"<line x1='{xv:.1f}' y1='{y1}' x2='{xv:.1f}' y2='{y1 + 6}' stroke='#222' />"
                    )
                    body.append(
                        f"<text x='{xv:.1f}' y='{y1 + 22}' text-anchor='middle' font-size='12'>{val:.3g}</text>"
                    )
                for t in range(6):
                    yv = float(y1) - float(t) * (ph / 5.0)
                    val = ymin + float(t) * (ymax - ymin) / 5.0
                    body.append(
                        f"<line x1='{x0 - 6}' y1='{yv:.1f}' x2='{x0}' y2='{yv:.1f}' stroke='#222' />"
                    )
                    body.append(
                        f"<text x='{x0 - 10}' y='{yv + 4:.1f}' text-anchor='end' font-size='12'>{val:.3g}</text>"
                    )

                # points
                for i in range(n):
                    body.append(
                        f"<circle cx='{xp(float(x[i])):.2f}' cy='{yp(float(a[i])):.2f}' r='2.1' fill='#1f77b4' fill-opacity='0.28' />"
                    )
                for i in range(n):
                    body.append(
                        f"<circle cx='{xp(float(x[i])):.2f}' cy='{yp(float(b[i])):.2f}' r='2.1' fill='#ff7f0e' fill-opacity='0.28' />"
                    )

                # legend
                lx, ly = x1 - 240, y0 + 12
                body.append(
                    f"<rect x='{lx}' y='{ly}' width='230' height='54' fill='white' stroke='#ccc' />"
                )
                body.append(
                    f"<circle cx='{lx + 20}' cy='{ly + 20}' r='4' fill='#1f77b4' fill-opacity='0.7' />"
                )
                body.append(f"<text x='{lx + 36}' y='{ly + 24}' font-size='12'>endpoint_pd</text>")
                body.append(
                    f"<circle cx='{lx + 20}' cy='{ly + 40}' r='4' fill='#ff7f0e' fill-opacity='0.7' />"
                )
                body.append(f"<text x='{lx + 36}' y='{ly + 44}' font-size='12'>pure_pursuit</text>")

                _write_svg(fig_dir / out_name, w=W, h=H, body="\n".join(body))

            _svg_hist_pair(
                "endpoint_pd_cte_mean",
                "pure_pursuit_cte_mean",
                title="CTE mean distribution",
                xlabel="cte_mean (m)",
                out_name="cte_mean_hist.svg",
            )
            _svg_hist_pair(
                "endpoint_pd_end_err",
                "pure_pursuit_end_err",
                title="Endpoint error distribution",
                xlabel="end_err (m)",
                out_name="end_err_hist.svg",
            )
            _svg_hist_pair(
                "endpoint_pd_progress_frac",
                "pure_pursuit_progress_frac",
                title="Progress fraction distribution",
                xlabel="progress_frac (0..1)",
                out_name="progress_frac_hist.svg",
            )
            _svg_scatter_pair(
                "path_len_m",
                "endpoint_pd_end_err",
                "pure_pursuit_end_err",
                title="Endpoint error vs path length",
                xlabel="path_len_m (m)",
                ylabel="end_err (m)",
                out_name="end_err_vs_path_len.svg",
            )
            _svg_scatter_pair(
                "path_len_m",
                "endpoint_pd_progress_final_m",
                "pure_pursuit_progress_final_m",
                title="Progress vs path length",
                xlabel="path_len_m (m)",
                ylabel="progress_final_m (m)",
                out_name="progress_vs_path_len.svg",
            )

        if summary_out is not None:
            with summary_out.open("a", encoding="utf-8") as f:
                f.write("\nFigures:\n")
                f.write(f"- Output directory: {fig_dir}\n")
                for p in wrote:
                    f.write(f"- {p}\n")

        # Bird's-eye trajectory plots (PNG): reference vs simulated tracking traces.
        # Only meaningful in "single" mode (one path per snapshot).
        if mode != "single":
            trace_n = 0
        trace_n = int(max(0, int(getattr(args, "trace_n", 0) or 0)))
        if trace_n > 0 and rows and paths_xy:
            traj_dir = fig_dir / "trajectories"
            traj_dir.mkdir(parents=True, exist_ok=True)

            def _safe_float(v: object, *, default: float) -> float:
                try:
                    x = float(v)  # type: ignore[arg-type]
                    return x if np.isfinite(x) else default
                except Exception:
                    return default

            def _rank_indices(col: str, k: int, *, largest: bool) -> list[int]:
                vals = []
                for i in range(len(rows)):
                    default = -1e30 if largest else 1e30
                    vals.append((_safe_float(rows[i].get(col), default=default), i))  # type: ignore[arg-type]
                vals.sort(key=lambda t: t[0], reverse=bool(largest))
                out = []
                for _v, idx in vals:
                    out.append(int(idx))
                    if len(out) >= int(k):
                        break
                return out

            rng = random.Random(int(getattr(args, "trace_seed", 0) or 0))
            chosen: list[int] = []
            chosen_set: set[int] = set()

            def _add_many(ixs: list[int]) -> None:
                nonlocal chosen
                for i in ixs:
                    if len(chosen) >= trace_n:
                        return
                    if int(i) in chosen_set:
                        continue
                    chosen.append(int(i))
                    chosen_set.add(int(i))

            # Representative mix: worst-case / best-case / random.
            _add_many(_rank_indices("endpoint_pd_cte_mean", 2, largest=True))
            _add_many(_rank_indices("pure_pursuit_cte_mean", 2, largest=True))
            _add_many(_rank_indices("endpoint_pd_end_err", 2, largest=True))
            _add_many(_rank_indices("pure_pursuit_end_err", 2, largest=True))
            _add_many(
                _rank_indices("endpoint_pd_progress_frac", 1, largest=False)
            )  # slow progress example
            _add_many(
                _rank_indices("pure_pursuit_progress_frac", 1, largest=False)
            )  # slow progress example

            # Fill the rest with random samples.
            if len(chosen) < trace_n:
                all_ix = list(range(len(rows)))
                rng.shuffle(all_ix)
                _add_many(all_ix)

            manifest: list[dict[str, object]] = []
            if plt is not None:
                # matplotlib-based bird's-eye plots (much easier to read)
                for rank, i in enumerate(chosen):
                    ref = np.asarray(paths_xy[i], dtype=np.float64)
                    res_ep = simulate_tracking(
                        ref,
                        controller="endpoint_pd",
                        sim=sim,
                        pp_cfg=pp_cfg,
                        ep_cfg=ep_cfg,
                        return_trace=True,
                    )
                    res_pp = simulate_tracking(
                        ref,
                        controller="pure_pursuit",
                        sim=sim,
                        pp_cfg=pp_cfg,
                        ep_cfg=ep_cfg,
                        return_trace=True,
                    )

                    ep_xy = np.stack(
                        [
                            np.asarray(res_ep["trace_x"], dtype=np.float64),
                            np.asarray(res_ep["trace_y"], dtype=np.float64),
                        ],
                        axis=1,
                    )
                    pp_xy = np.stack(
                        [
                            np.asarray(res_pp["trace_x"], dtype=np.float64),
                            np.asarray(res_pp["trace_y"], dtype=np.float64),
                        ],
                        axis=1,
                    )

                    # Plot in (y_left, x_forward) so x-forward is up.
                    fig = plt.figure(figsize=(7.6, 6.0))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.plot(ref[:, 1], ref[:, 0], color="0.45", linewidth=3, label="ref")
                    ax.plot(
                        ep_xy[:, 1],
                        ep_xy[:, 0],
                        color="#1f77b4",
                        linewidth=2.5,
                        label="endpoint_pd",
                    )
                    ax.plot(
                        pp_xy[:, 1],
                        pp_xy[:, 0],
                        color="#ff7f0e",
                        linewidth=2.5,
                        label="pure_pursuit",
                    )
                    ax.scatter([0.0], [0.0], s=40, c="k", marker="o", label="start")
                    ax.scatter(
                        [ref[-1, 1]], [ref[-1, 0]], s=40, c="0.45", marker="o", label="ref_end"
                    )
                    # heading arrow (initial heading is +x)
                    ax.arrow(
                        0.0,
                        0.0,
                        0.0,
                        0.6,
                        head_width=0.15,
                        head_length=0.18,
                        fc="k",
                        ec="k",
                        alpha=0.7,
                    )
                    ax.set_aspect("equal", adjustable="box")
                    ax.grid(True, alpha=0.25)
                    ax.set_xlabel("y_left (m)")
                    ax.set_ylabel("x_forward (m)")

                    ep_id = str(rows[i].get("episode_id", ""))
                    t_val = _safe_float(rows[i].get("t"), default=float("nan"))
                    ax.set_title(f"idx={i} ep={ep_id} t={t_val:.2f}s")
                    ax.legend(loc="best", fontsize=9)

                    # Tight bounds with margin, equal scale
                    all_pts = np.concatenate([ref, ep_xy, pp_xy], axis=0)
                    x_min = float(np.min(all_pts[:, 0]))
                    x_max = float(np.max(all_pts[:, 0]))
                    y_min = float(np.min(all_pts[:, 1]))
                    y_max = float(np.max(all_pts[:, 1]))
                    cx = 0.5 * (x_min + x_max)
                    cy = 0.5 * (y_min + y_max)
                    span = max(x_max - x_min, y_max - y_min, 1.0)
                    span = span * 0.65 + 0.8
                    ax.set_ylim(cx - span, cx + span)
                    ax.set_xlim(cy - span, cy + span)

                    # metrics text
                    ax.text(
                        0.02,
                        0.02,
                        (
                            f"endpoint_pd: cte_mean={_safe_float(rows[i].get('endpoint_pd_cte_mean'), default=float('nan')):.3f} "
                            f"end_err={_safe_float(rows[i].get('endpoint_pd_end_err'), default=float('nan')):.3f} "
                            f"prog_frac={_safe_float(rows[i].get('endpoint_pd_progress_frac'), default=float('nan')):.3f}\n"
                            f"pure_pursuit: cte_mean={_safe_float(rows[i].get('pure_pursuit_cte_mean'), default=float('nan')):.3f} "
                            f"end_err={_safe_float(rows[i].get('pure_pursuit_end_err'), default=float('nan')):.3f} "
                            f"prog_frac={_safe_float(rows[i].get('pure_pursuit_progress_frac'), default=float('nan')):.3f}"
                        ),
                        transform=ax.transAxes,
                        fontsize=9,
                        va="bottom",
                        ha="left",
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="0.8"),
                    )
                    fig.tight_layout()
                    out_path_png = traj_dir / f"traj_{rank:02d}_idx{i}.png"
                    fig.savefig(str(out_path_png), dpi=170)
                    plt.close(fig)

                    manifest.append(
                        {
                            "rank": int(rank),
                            "row_idx": int(i),
                            "episode_id": ep_id,
                            "t": t_val,
                            "png": str(out_path_png),
                        }
                    )
            else:
                # PIL fallback (used when matplotlib can't import)
                try:
                    from PIL import Image, ImageDraw, ImageFont  # type: ignore

                    font = ImageFont.load_default()
                except Exception as e:  # noqa: BLE001
                    if summary_out is not None:
                        with summary_out.open("a", encoding="utf-8") as f:
                            f.write("\nTrajectory PNGs:\n")
                            f.write(f"- Skipped (PIL import failed): {type(e).__name__}: {e}\n")
                    return

                def _to_px(
                    x: float,
                    y: float,
                    *,
                    x_min: float,
                    x_max: float,
                    y_min: float,
                    y_max: float,
                    W: int,
                    H: int,
                    pad: int,
                ) -> tuple[int, int]:
                    # Map (x forward, y left) to image pixels with +x up.
                    xr = max(1e-6, float(x_max - x_min))
                    yr = max(1e-6, float(y_max - y_min))
                    # y left -> image x left (invert so positive y goes left)
                    u = float(pad) + (float(y_max) - float(y)) / yr * float(W - 2 * pad)
                    # x forward -> image y up (invert so larger x is higher)
                    v = float(pad) + (float(x_max) - float(x)) / xr * float(H - 2 * pad)
                    return int(round(u)), int(round(v))

                def _draw_poly(
                    draw: ImageDraw.ImageDraw,
                    pts: np.ndarray,
                    *,
                    color: tuple[int, int, int],
                    width: int,
                ) -> None:
                    if pts.shape[0] < 2:
                        return
                    xy = [tuple(map(int, pts[i])) for i in range(pts.shape[0])]
                    draw.line(xy, fill=color, width=int(width), joint="curve")

                for rank, i in enumerate(chosen):
                    ref = np.asarray(paths_xy[i], dtype=np.float64)
                    res_ep = simulate_tracking(
                        ref,
                        controller="endpoint_pd",
                        sim=sim,
                        pp_cfg=pp_cfg,
                        ep_cfg=ep_cfg,
                        return_trace=True,
                    )
                    res_pp = simulate_tracking(
                        ref,
                        controller="pure_pursuit",
                        sim=sim,
                        pp_cfg=pp_cfg,
                        ep_cfg=ep_cfg,
                        return_trace=True,
                    )

                    ep_xy = np.stack(
                        [
                            np.asarray(res_ep["trace_x"], dtype=np.float64),
                            np.asarray(res_ep["trace_y"], dtype=np.float64),
                        ],
                        axis=1,
                    )
                    pp_xy = np.stack(
                        [
                            np.asarray(res_pp["trace_x"], dtype=np.float64),
                            np.asarray(res_pp["trace_y"], dtype=np.float64),
                        ],
                        axis=1,
                    )

                    all_pts = np.concatenate([ref, ep_xy, pp_xy], axis=0)
                    x_min = float(np.min(all_pts[:, 0]))
                    x_max = float(np.max(all_pts[:, 0]))
                    y_min = float(np.min(all_pts[:, 1]))
                    y_max = float(np.max(all_pts[:, 1]))
                    cx = 0.5 * (x_min + x_max)
                    cy = 0.5 * (y_min + y_max)
                    span = max(x_max - x_min, y_max - y_min, 1.0)
                    span = span * 0.65 + 0.8
                    x_min, x_max = cx - span, cx + span
                    y_min, y_max = cy - span, cy + span

                    W, H, pad = 1100, 820, 90
                    img = Image.new("RGB", (W, H), (255, 255, 255))
                    d = ImageDraw.Draw(img)

                    # Border
                    d.rectangle([pad, pad, W - pad, H - pad], outline=(180, 180, 180), width=2)

                    def _map_pts(
                        pts_xy: np.ndarray,
                        *,
                        x_min: float = x_min,
                        x_max: float = x_max,
                        y_min: float = y_min,
                        y_max: float = y_max,
                        W: int = W,
                        H: int = H,
                        pad: int = pad,
                    ) -> np.ndarray:
                        out = np.zeros((pts_xy.shape[0], 2), dtype=np.int32)
                        for j in range(pts_xy.shape[0]):
                            u, v = _to_px(
                                float(pts_xy[j, 0]),
                                float(pts_xy[j, 1]),
                                x_min=x_min,
                                x_max=x_max,
                                y_min=y_min,
                                y_max=y_max,
                                W=W,
                                H=H,
                                pad=pad,
                            )
                            out[j, 0] = int(u)
                            out[j, 1] = int(v)
                        return out

                    ref_px = _map_pts(ref)
                    ep_px = _map_pts(ep_xy)
                    pp_px = _map_pts(pp_xy)
                    _draw_poly(d, ref_px, color=(120, 120, 120), width=5)
                    _draw_poly(d, ep_px, color=(31, 119, 180), width=4)
                    _draw_poly(d, pp_px, color=(255, 127, 14), width=4)

                    ep_id = str(rows[i].get("episode_id", ""))
                    t_val = _safe_float(rows[i].get("t"), default=float("nan"))
                    d.text(
                        (12, 10), f"idx={i} ep={ep_id} t={t_val:.2f}s", fill=(0, 0, 0), font=font
                    )
                    d.text(
                        (12, 30),
                        "gray=ref  blue=endpoint_pd  orange=pure_pursuit",
                        fill=(0, 0, 0),
                        font=font,
                    )

                    out_path_png = traj_dir / f"traj_{rank:02d}_idx{i}.png"
                    img.save(str(out_path_png))
                    manifest.append(
                        {
                            "rank": int(rank),
                            "row_idx": int(i),
                            "episode_id": ep_id,
                            "t": t_val,
                            "png": str(out_path_png),
                        }
                    )

            # Write manifest for easy browsing
            (traj_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            if summary_out is not None:
                with summary_out.open("a", encoding="utf-8") as f:
                    f.write("\nTrajectory PNGs (bird's-eye):\n")
                    f.write(f"- Directory: {traj_dir}\n")
                    f.write(f"- Wrote {len(manifest)} PNGs + manifest.json\n")


if __name__ == "__main__":
    main()
