#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


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


def _parse_csv_floats(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_csv_strs(s: str) -> list[str]:
    out: list[str] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(str(part))
    return out


def _try_find_delay_dir(step_dir: Path, delay_s: float) -> Path | None:
    if not step_dir.exists():
        return None
    for p in step_dir.glob("delay_*"):
        if not p.is_dir():
            continue
        try:
            v = float(p.name.split("delay_", 1)[1])
        except Exception:
            continue
        if abs(float(v) - float(delay_s)) <= 1e-9:
            return p
    return None


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _find_latest_run(section5_root: Path, *, delay_s: float) -> Path | None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from slow_brain_fast_planner.benchmarks.closed_loop_results import (
            find_latest_run,  # noqa: E402
        )
    except Exception:
        find_latest_run = None  # type: ignore[assignment]

    def _latest(parent: Path) -> Path | None:
        if parent is None or not parent.exists():
            return None
        if find_latest_run is not None:
            return find_latest_run(parent, recursive=True)
        runs = sorted(
            [
                p
                for p in parent.rglob("closed_loop_fusion_*")
                if p.is_dir() and p.name.startswith("closed_loop_fusion_")
            ]
        )
        return runs[-1] if runs else None

    # Prefer a step4 run at the same delay (matches typical "default config" used for section5).
    step4 = section5_root / "step4_main_results"
    ddir = _try_find_delay_dir(step4, delay_s)
    if ddir is not None:
        # Prefer streaming runs when available (matches our default fusion policy variants).
        for group in (
            "score_fusion_stream",
            "prob_fusion_stream",
            "score_fusion",
            "prob_fusion",
            "baselines",
        ):
            r = _latest(ddir / group)
            if r is not None:
                return r

    # Next: any explicit tune folder that matches the delay.
    tune = section5_root / f"tune_delay_{_format_float_for_path(delay_s)}"
    r = _latest(tune)
    if r is not None:
        return r

    # Fallback: newest run anywhere under the root.
    return _latest(section5_root)


def _cfg_to_benchmark_args(cfg: dict) -> list[str]:
    args: list[str] = []
    args += ["--dataset", str(cfg["dataset"])]
    args += ["--static-candidates-json", str(cfg["static_candidates_json"])]

    sim = cfg.get("sim", {}) if isinstance(cfg.get("sim", {}), dict) else {}
    args += ["--dt-control", str(sim.get("dt_control", 0.1))]
    args += ["--dt-plan", str(sim.get("dt_plan", 0.2))]
    args += ["--horizon-s", str(sim.get("horizon_s", 4.0))]

    if "controller" in cfg:
        args += ["--controller", str(cfg["controller"])]

    limits = cfg.get("limits", {}) if isinstance(cfg.get("limits", {}), dict) else {}
    if "max_v" in limits and limits["max_v"] is not None:
        args += ["--max-v", str(limits["max_v"])]
    if "max_w" in limits and limits["max_w"] is not None:
        args += ["--max-w", str(limits["max_w"])]

    tasks = cfg.get("tasks", {}) if isinstance(cfg.get("tasks", {}), dict) else {}
    if "task_mode" in tasks:
        args += ["--task-mode", str(tasks["task_mode"])]
    for k, flag in [
        ("duration_s", "--duration-s"),
        ("min_episode_s", "--min-episode-s"),
        ("stride_s", "--stride-s"),
        ("time_limit_s", "--time-limit-s"),
        ("max_episodes", "--max-episodes"),
        ("max_tasks_per_episode", "--max-tasks-per-episode"),
        ("task_sampling", "--task-sampling"),
        ("seed", "--seed"),
    ]:
        if k in tasks and tasks[k] is not None:
            args += [flag, str(tasks[k])]

    if "delay_s" in cfg:
        args += ["--delay-s", str(cfg["delay_s"])]

    # Query cadence
    if "vlm_query_interval_s" in cfg and cfg["vlm_query_interval_s"] is not None:
        args += ["--vlm-query-interval-s", str(cfg["vlm_query_interval_s"])]
    elif "vlm_query_hz" in cfg and cfg["vlm_query_hz"] is not None:
        args += ["--vlm-query-hz", str(cfg["vlm_query_hz"])]

    if "vlm_max_inflight" in cfg and cfg["vlm_max_inflight"] is not None:
        args += ["--vlm-max-inflight", str(cfg["vlm_max_inflight"])]
    if "vlm_mistake_prob" in cfg and cfg["vlm_mistake_prob"] is not None:
        args += ["--vlm-mistake-prob", str(cfg["vlm_mistake_prob"])]

    success = cfg.get("success", {}) if isinstance(cfg.get("success", {}), dict) else {}
    if "min_completion" in success and success["min_completion"] is not None:
        args += ["--success-min-completion", str(success["min_completion"])]
    if "goal_radius_m" in success and success["goal_radius_m"] is not None:
        args += ["--success-goal-radius-m", str(success["goal_radius_m"])]

    # BooleanOptionalAction
    if "stop_on_success" in cfg:
        args += ["--stop-on-success" if bool(cfg["stop_on_success"]) else "--no-stop-on-success"]

    if "policies" in cfg and cfg["policies"] is not None:
        args += ["--policies", ",".join([str(x) for x in cfg["policies"]])]

    planner = cfg.get("planner_score", {}) if isinstance(cfg.get("planner_score", {}), dict) else {}
    for k, flag in [
        ("noise_stds", "--noise-stds"),
        ("epsilons", "--epsilons"),
        ("score_temps", "--score-temps"),
    ]:
        v = planner.get(k, None)
        if isinstance(v, list) and v:
            args += [flag, ",".join(str(x) for x in v)]

    fusion = cfg.get("fusion_sweep", {}) if isinstance(cfg.get("fusion_sweep", {}), dict) else {}
    for k, flag in [
        ("similarity_modes", "--similarity-modes"),
        ("lambda_sims", "--lambda-sims"),
        ("taus", "--taus"),
        ("dist_scales", "--dist-scales"),
    ]:
        v = fusion.get(k, None)
        if isinstance(v, list) and v:
            args += [flag, ",".join(str(x) for x in v)]

    if "num_workers" in cfg and cfg["num_workers"] is not None:
        args += ["--num-workers", str(cfg["num_workers"])]

    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep lambda_sim for closed-loop score/prob fusion (wrapper around "
        "benchmark_closed_loop_score_fusion_real.py)."
    )
    p.add_argument(
        "--section5-root", type=str, default="", help="Logs root (e.g. logs/section5_closed_loop)."
    )
    p.add_argument(
        "--delay-s",
        type=float,
        default=2.0,
        help="Delay (seconds) used for folder selection + benchmark --delay-s.",
    )

    p.add_argument(
        "--base-run-dir",
        type=str,
        default="",
        help="Optional: explicit run_dir containing config.json to use as the default config "
        "template.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Override dataset path (if omitted, uses base config).",
    )
    p.add_argument(
        "--static-candidates-json",
        type=str,
        default="",
        help="Override static candidate library JSON (if omitted, uses base config).",
    )

    p.add_argument(
        "--policies",
        type=str,
        default="score_fusion_stream,prob_fusion_stream",
        help="Comma-separated policies to evaluate (default: "
        "score_fusion_stream,prob_fusion_stream).",
    )
    p.add_argument(
        "--similarity-mode", type=str, default="", help="Override similarity mode (single value)."
    )
    p.add_argument(
        "--dist-scale", type=float, default=None, help="Override dist_scale_m (single value)."
    )

    p.add_argument(
        "--lambda-sims",
        type=str,
        default="",
        help="Comma-separated lambda values to sweep. If omitted, uses base config's lambda_sims.",
    )
    p.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Hold tau fixed at this value. If omitted, uses the first base config tau.",
    )

    p.add_argument(
        "--out-root",
        type=str,
        default="",
        help="Override output root; default uses "
        "<section5-root>/step3_lambda_sweep_fusions/delay_<delay>.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num-workers (defaults to base config).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the benchmark command and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    section5_root = (
        Path(args.section5_root).expanduser().resolve()
        if str(args.section5_root).strip()
        else (repo_root / "logs/section5_closed_loop").resolve()
    )

    base_run = (
        Path(args.base_run_dir).expanduser().resolve()
        if str(args.base_run_dir).strip()
        else _find_latest_run(section5_root, delay_s=float(args.delay_s))
    )
    if base_run is None:
        raise SystemExit(
            "Could not find a base run. Pass --base-run-dir or set --section5-root to a folder "
            "containing a closed_loop_fusion_* run."
        )
    cfg_path = Path(base_run) / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Base run missing config.json: {cfg_path}")
    base_cfg = _load_json(cfg_path)

    # Apply explicit overrides.
    if str(args.dataset).strip():
        base_cfg["dataset"] = str(Path(args.dataset).expanduser().resolve())
    if str(args.static_candidates_json).strip():
        base_cfg["static_candidates_json"] = str(
            Path(args.static_candidates_json).expanduser().resolve()
        )
    base_cfg["delay_s"] = float(args.delay_s)
    base_cfg["policies"] = _parse_csv_strs(str(args.policies))

    fusion = base_cfg.get("fusion_sweep", {})
    if not isinstance(fusion, dict):
        fusion = {}
        base_cfg["fusion_sweep"] = fusion

    if str(args.similarity_mode).strip():
        fusion["similarity_modes"] = [str(args.similarity_mode).strip()]
    else:
        sims = fusion.get("similarity_modes", None)
        if isinstance(sims, list) and sims:
            fusion["similarity_modes"] = [str(sims[0])]

    if args.dist_scale is not None:
        fusion["dist_scales"] = [float(args.dist_scale)]
    else:
        dss = fusion.get("dist_scales", None)
        if isinstance(dss, list) and dss:
            fusion["dist_scales"] = [float(dss[0])]

    if str(args.lambda_sims).strip():
        fusion["lambda_sims"] = _parse_csv_floats(str(args.lambda_sims))

    # Hold tau fixed for the lambda sweep.
    if args.tau is not None:
        fusion["taus"] = [float(args.tau)]
    else:
        taus = fusion.get("taus", None)
        if isinstance(taus, list) and taus:
            fusion["taus"] = [float(taus[0])]

    if (
        "lambda_sims" not in fusion
        or not isinstance(fusion.get("lambda_sims"), list)
        or not fusion.get("lambda_sims")
    ):
        raise SystemExit(
            "Missing lambda sweep values. Pass --lambda-sims or ensure base config has "
            "fusion_sweep.lambda_sims."
        )

    # Output root (parent folder that benchmark will create closed_loop_fusion_*/ in).
    if str(args.out_root).strip():
        out_root = Path(args.out_root).expanduser().resolve()
    else:
        out_root = (
            section5_root
            / "step3_lambda_sweep_fusions"
            / f"delay_{_format_float_for_path(float(args.delay_s))}"
        )

    # Compose benchmark command.
    bench = repo_root / "scripts/paper_artifacts/benchmark_closed_loop_score_fusion_real.py"
    cmd = [sys.executable, str(bench), "--out", str(out_root)]
    cmd += _cfg_to_benchmark_args(base_cfg)

    if args.num_workers is not None:
        cmd += ["--num-workers", str(int(args.num_workers))]

    meta = {
        "base_run_dir": str(base_run),
        "section5_root": str(section5_root),
        "out_root": str(out_root),
        "cmd": cmd,
        "sweep": {
            "param": "lambda_sim",
            "lambda_sims": fusion.get("lambda_sims", []),
            "tau_fixed": fusion.get("taus", []),
        },
    }
    if bool(args.dry_run):
        print(json.dumps(meta, indent=2))
        return

    # Best-effort: record the invoked command next to the sweep folder.
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "sweep_lambda_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as e:
        print(f"Warning: could not write sweep metadata under {out_root}: {e}", file=sys.stderr)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
