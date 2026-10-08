#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from slow_brain_fast_planner.control.delayed_vlm_score_fusion import (
    FusionConfig,
    PlannerScoreModel,
    SimLoopConfig,
    VlmDelayModel,
    load_static_candidate_set,
    make_task,
    run_toy_simulation,
)
from slow_brain_fast_planner.control.tracking import EndpointPDConfig, Limits, PurePursuitConfig


def _parse_csv_floats(s: str) -> list[float]:
    out: list[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _parse_csv_ints(s: str) -> list[int]:
    out: list[int] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Toy sim: delayed dummy VLM + score fusion over a static trajectory library."
    )
    p.add_argument(
        "--candidates-json",
        type=str,
        default="assets/trajectory_selection_static_candidates/takeover_kmeans_medoids/static_candidates_k12.json",
        help="Path to a static trajectory-selection candidate set JSON (robot-frame XY polylines).",
    )
    p.add_argument(
        "--out", type=str, default="logs/toy_delayed_vlm_fusion.csv", help="Output CSV path."
    )
    p.add_argument("--summary-out", type=str, default=None, help="Optional JSON summary path.")

    # Sweeps
    p.add_argument(
        "--tasks",
        type=str,
        default="forward,left_turn,right_turn",
        help="Comma-separated tasks: forward,left_turn,right_turn.",
    )
    p.add_argument(
        "--controllers",
        type=str,
        default="endpoint_pd,pure_pursuit",
        help="Comma-separated controllers.",
    )
    p.add_argument(
        "--policies",
        type=str,
        default="local_only,vlm_hold,score_fusion,prob_fusion",
        help="Comma-separated policies.",
    )
    p.add_argument(
        "--delays-s",
        type=str,
        default="0.0,0.5,1.5,2.0",
        help="Comma-separated VLM delays (seconds).",
    )
    p.add_argument(
        "--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds (noise seeds)."
    )

    # Task parameters
    p.add_argument(
        "--v-ref", type=float, default=1.3, help="Reference forward speed for the toy task (m/s)."
    )
    p.add_argument(
        "--radius-m", type=float, default=2.5, help="Turn radius for left/right tasks (m)."
    )
    p.add_argument("--duration-s", type=float, default=12.0, help="Episode duration (seconds).")

    # Loop timing
    p.add_argument("--dt-control", type=float, default=0.1, help="Controller integration tick (s).")
    p.add_argument("--dt-plan", type=float, default=0.2, help="Replan / selection tick (s).")
    p.add_argument(
        "--horizon-s",
        type=float,
        default=4.0,
        help="Candidate horizon (used for ideal reference construction).",
    )
    p.add_argument("--prepend-origin", action=argparse.BooleanOptionalAction, default=True)

    # Limits (plant constraints)
    # Defaults match deploy/NavFlow/visualnav_ros/deployment/config/robot.yaml
    p.add_argument("--max-v", type=float, default=1.0)
    p.add_argument("--max-w", type=float, default=0.5)
    p.add_argument("--max-a", type=float, default=1.0)
    p.add_argument("--max-alpha", type=float, default=1.2)
    # Match deploy/NavFlow pure pursuit defaults unless overridden.
    p.add_argument("--max-lat-accel", type=float, default=0.8)

    # Controller knobs (PP)
    p.add_argument("--pp-lookahead-m", type=float, default=1.0)
    p.add_argument("--pp-lookahead-gain", type=float, default=0.5)
    p.add_argument("--pp-v-gain", type=float, default=0.8)
    # Controller knobs (endpoint PD)
    p.add_argument("--ep-dt-nominal", type=float, default=1.0)
    # Match latest on-robot PD heuristic (curvature slow-down disabled by default).
    p.add_argument("--ep-curvature-speed-gain", type=float, default=0.0)

    # Planner score model (synthetic)
    p.add_argument("--noise-std", type=float, default=0.15)
    p.add_argument("--score-scale", type=float, default=1.0)
    p.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon-greedy: probability of picking random candidate instead of argmax. "
        "Simulates a confused planner that VLM can correct.",
    )

    # Fusion knobs
    p.add_argument("--lambda-sim", type=float, default=1.0)
    p.add_argument("--staleness-tau-s", type=float, default=3.0)
    p.add_argument("--dist-scale-m", type=float, default=1.0)
    p.add_argument(
        "--similarity-mode",
        type=str,
        default="pointwise_arclen",
        choices=[
            "pointwise_arclen",
            "candidate_to_ref_polyline",
            "symmetric_polyline",
            "body_pointwise_arclen",
            "body_pointwise_horizon_aware",
        ],
    )
    p.add_argument("--align-to-current-pose", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out = Path(args.summary_out).resolve() if args.summary_out else None
    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)

    candidates = load_static_candidate_set(
        args.candidates_json, prepend_origin=bool(args.prepend_origin)
    )
    n_cand = int(len(candidates))
    n_pts = int(candidates[0].shape[0])

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    controllers = [c.strip() for c in str(args.controllers).split(",") if c.strip()]
    policies = [m.strip() for m in str(args.policies).split(",") if m.strip()]
    delays = _parse_csv_floats(str(args.delays_s))
    seeds = _parse_csv_ints(str(args.seeds))

    limits = Limits(
        max_v=float(args.max_v),
        max_w=float(args.max_w),
        max_a=float(args.max_a),
        max_alpha=float(args.max_alpha),
        max_lat_accel=float(args.max_lat_accel),
    )
    sim_cfg = SimLoopConfig(
        dt_control=float(args.dt_control),
        dt_plan=float(args.dt_plan),
        horizon_s=float(args.horizon_s),
        prepend_origin=bool(args.prepend_origin),
        limits=limits,
        pp_cfg=PurePursuitConfig(
            lookahead_m=float(args.pp_lookahead_m),
            lookahead_gain=float(args.pp_lookahead_gain),
            v_gain=float(args.pp_v_gain),
        ),
        ep_cfg=EndpointPDConfig(
            dt_nominal=float(args.ep_dt_nominal),
            curvature_speed_gain=float(args.ep_curvature_speed_gain),
        ),
    )
    score_model = PlannerScoreModel(
        noise_std=float(args.noise_std),
        score_scale=float(args.score_scale),
        epsilon=float(args.epsilon),
    )
    fusion_cfg = FusionConfig(
        enabled=True,
        similarity_mode=str(args.similarity_mode),  # type: ignore[arg-type]
        lambda_sim=float(args.lambda_sim),
        staleness_tau_s=float(args.staleness_tau_s),
        dist_scale_m=float(args.dist_scale_m),
        align_to_current_pose=bool(args.align_to_current_pose),
    )

    rows: list[dict[str, object]] = []
    for task_name in tasks:
        for controller in controllers:
            for policy in policies:
                for delay_s in delays:
                    for seed in seeds:
                        task = make_task(
                            task_name, v_ref=float(args.v_ref), radius_m=float(args.radius_m)
                        )
                        # Override duration for this run.
                        task = type(task)(
                            name=task.name,
                            v_ref=task.v_ref,
                            w_ref=task.w_ref,
                            duration_s=float(args.duration_s),
                        )
                        vlm = VlmDelayModel(delay_s=float(delay_s), oracle=True)
                        res = run_toy_simulation(
                            candidates_body=candidates,
                            task=task,
                            controller=controller,  # type: ignore[arg-type]
                            policy=policy,  # type: ignore[arg-type]
                            sim_cfg=sim_cfg,
                            score_model=score_model,
                            vlm_cfg=vlm,
                            fusion_cfg=fusion_cfg,
                            seed=int(seed),
                        )
                        rows.append(
                            {
                                "task": task.name,
                                "v_ref": float(task.v_ref),
                                "w_ref": float(task.w_ref),
                                "radius_m": float(args.radius_m),
                                "controller": str(controller),
                                "policy": str(policy),
                                "delay_s": float(delay_s),
                                "seed": int(seed),
                                "num_candidates": int(n_cand),
                                "traj_points": int(n_pts),
                                "dt_control": float(sim_cfg.dt_control),
                                "dt_plan": float(sim_cfg.dt_plan),
                                "horizon_s": float(sim_cfg.horizon_s),
                                "prepend_origin": bool(sim_cfg.prepend_origin),
                                "noise_std": float(score_model.noise_std),
                                "score_scale": float(score_model.score_scale),
                                "epsilon": float(score_model.epsilon),
                                "lambda_sim": float(fusion_cfg.lambda_sim),
                                "staleness_tau_s": float(fusion_cfg.staleness_tau_s),
                                "dist_scale_m": float(fusion_cfg.dist_scale_m),
                                "similarity_mode": str(fusion_cfg.similarity_mode),
                                "align_to_current_pose": bool(fusion_cfg.align_to_current_pose),
                                "mean_pos_err_m": float(res.mean_pos_err_m),
                                "p95_pos_err_m": float(res.p95_pos_err_m),
                                "mean_cte_to_ideal_m": float(res.mean_cte_to_ideal_m),
                                "p95_cte_to_ideal_m": float(res.p95_cte_to_ideal_m),
                                "mean_speed_mps": float(res.mean_speed_mps),
                                "stopped_frac": float(res.stopped_frac),
                                "mean_abs_dv": float(res.mean_abs_dv),
                                "mean_abs_dw": float(res.mean_abs_dw),
                                "steps": int(res.steps),
                                "plan_steps": int(res.plan_steps),
                                "vlm_updates": int(res.vlm_updates),
                            }
                        )

    if not rows:
        raise SystemExit("No rows produced. Check sweep arguments.")

    cols = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "out_csv": str(out_path),
        "candidates_json": str(Path(args.candidates_json).resolve()),
        "num_candidates": int(n_cand),
        "traj_points": int(n_pts),
        "sweeps": {
            "tasks": tasks,
            "controllers": controllers,
            "policies": policies,
            "delays_s": delays,
            "seeds": seeds,
        },
        "config": {
            "task": {
                "v_ref": float(args.v_ref),
                "radius_m": float(args.radius_m),
                "duration_s": float(args.duration_s),
            },
            "loop": {
                "dt_control": float(args.dt_control),
                "dt_plan": float(args.dt_plan),
                "horizon_s": float(args.horizon_s),
            },
            "limits": {
                "max_v": float(args.max_v),
                "max_w": float(args.max_w),
                "max_a": float(args.max_a),
                "max_alpha": float(args.max_alpha),
                "max_lat_accel": float(args.max_lat_accel),
            },
            "planner_score": {
                "noise_std": float(args.noise_std),
                "score_scale": float(args.score_scale),
                "epsilon": float(args.epsilon),
            },
            "fusion": {
                "lambda_sim": float(args.lambda_sim),
                "staleness_tau_s": float(args.staleness_tau_s),
                "dist_scale_m": float(args.dist_scale_m),
                "similarity_mode": str(args.similarity_mode),
                "align_to_current_pose": bool(args.align_to_current_pose),
            },
        },
        "rows": int(len(rows)),
    }
    print(json.dumps(summary, indent=2))
    if summary_out is not None:
        summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
