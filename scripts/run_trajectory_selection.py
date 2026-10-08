#!/usr/bin/env python
"""Trajectory Selection Benchmark.

This script supports two modes:
1. Run on an existing canonical dataset with prelogged planner candidates
2. Run on an existing canonical dataset and generate planner candidates via ONNX

Raw logs are converted into a canonical dataset beforehand via scripts/process_data.py
(slow_brain_fast_planner.cli.process_data).

Usage:
    # Prelogged candidates
    python scripts/run_trajectory_selection.py --dataset data/processed --model dummy_argmax

    # Generate candidates via ONNX
    python scripts/run_trajectory_selection.py --dataset data/processed --planner-source onnx

Multi-GPU (episode-level sharding):
    WORLD_SIZE=4 RANK=0 python scripts/run_trajectory_selection.py --dataset data/processed
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Workflow step 3/3: run the trajectory-selection benchmark.\n\n"
            "This evaluates a trajectory selector (baseline/oracle/VLM) on a canonical dataset.\n\n"
            "For multi-GPU sharding, use --num-shards/--shard-id (or set WORLD_SIZE/RANK)."
        ),
        add_help=False,  # We'll pass through to the CLI
    )
    parser.add_argument("--dataset", default="data/processed", help="Canonical dataset root.")
    parser.add_argument("--episode-id", action="append", default=None)
    parser.add_argument(
        "--planner-source",
        choices=["prelogged", "onnx"],
        required=True,
    )
    parser.add_argument(
        "--planner-onnx",
        default="assets/planner.onnx",
    )
    parser.add_argument("--planner-overwrite", action="store_true")
    parser.add_argument("--planner-max-frames", type=int, default=None)
    parser.add_argument("--planner-stride", type=int, default=1)
    parser.add_argument(
        "--planner-align-to-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, align ONNX-written candidate count to reference prelogged candidate count "
        "(e.g., 64).",
    )
    parser.add_argument("--planner-output-dataset", default=None)
    parser.add_argument(
        "--planner-times-jsonl-path",
        default=None,
        help=(
            "Optional: only run ONNX planner at specific times from a JSONL file.\n"
            "Each line must contain {episode_id: str, <time_key>: float}. "
            "Useful to cache planner outputs only at clip t0s (much faster/smaller than per-tick)."
        ),
    )
    parser.add_argument("--planner-times-jsonl-time-key", default="t0")
    parser.add_argument("--planner-times-time-tolerance-s", type=float, default=0.2)
    parser.add_argument(
        "--planner-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true and --planner-source=onnx, generate planner candidates then exit (skip "
        "evaluation).",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-snapshots-per-episode", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-validation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if "-h" in raw_argv or "--help" in raw_argv:
        parser.print_help()
        print(
            "\nAll other flags are forwarded to slow_brain_fast_planner.cli.trajectory_selection; "
            "run `python -m slow_brain_fast_planner.cli.trajectory_selection --help` "
            "for the full list."
        )
        return 0

    args, unknown = parser.parse_known_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    dataset_arg = str(args.dataset)
    dataset_path = Path(dataset_arg).resolve()

    def _get_unknown_arg_value(flag: str) -> str | None:
        for i, a in enumerate(unknown):
            if a == flag and i + 1 < len(unknown):
                return str(unknown[i + 1])
        return None

    def _get_unknown_int(flag: str) -> int | None:
        v = _get_unknown_arg_value(flag)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    # Compute the run output directory up-front so ONNX planner outputs can be written there.
    # This avoids ever touching the source dataset.
    out_arg = _get_unknown_arg_value("--out")
    if out_arg:
        run_out_dir = Path(out_arg).resolve()
    else:
        output_dir = _get_unknown_arg_value("--output-dir") or "logs"
        exp_name = _get_unknown_arg_value("--exp-name") or "trajectory_selection"
        # Mirror CLI sharding naming to keep shard runs separated.
        num_shards = _get_unknown_arg_value("--num-shards")
        shard_id = _get_unknown_arg_value("--shard-id")
        if num_shards is not None or shard_id is not None:
            ns = int(num_shards or 1)
            sid = int(shard_id or 0)
        else:
            from slow_brain_fast_planner.benchmarks.data_loading import get_shard_config_from_env

            ns, sid = get_shard_config_from_env()
        if int(ns) > 1:
            exp_name = f"{exp_name}_shard{int(sid)}of{int(ns)}"
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_out_dir = (Path(output_dir) / exp_name / f"{exp_name}_{timestamp}").resolve()

    planner_dataset_path = dataset_path

    # Stage 2: Generate planner candidates via ONNX if requested
    if str(args.planner_source) == "onnx":
        import json

        from slow_brain_fast_planner.benchmarks.dataset import (
            find_episode_metadata_files,
            load_episode,
        )
        from slow_brain_fast_planner.planner import write_planner_candidates_from_onnx

        logger.info(f"Stage 2/3: Generating planner candidates via ONNX: {args.planner_onnx}")

        # If the user is only evaluating a small number of snapshots, don't run the ONNX planner
        # for the entire episode. We infer a reasonable max_frames cap from eval limits.
        #
        # Priority:
        # 1) explicit --planner-max-frames
        # 2) wrapper --max-snapshots-per-episode (if set)
        # 3) CLI --max-snapshots-total (only safe to use as per-episode cap when evaluating 1
        # episode)
        eval_max_total = _get_unknown_int("--max-snapshots-total")
        eval_max_per_ep = args.max_snapshots_per_episode
        cap_frames: int | None = None
        if args.planner_max_frames is not None:
            cap_frames = int(args.planner_max_frames)
        elif eval_max_per_ep is not None:
            cap_frames = int(eval_max_per_ep)
        elif eval_max_total is not None and int(eval_max_total) > 0:
            single_ep = False
            if args.episode_id and len(args.episode_id) == 1:
                single_ep = True
            if args.max_episodes is not None and int(args.max_episodes) == 1:
                single_ep = True
            if single_ep:
                cap_frames = int(eval_max_total)
        if cap_frames is not None and cap_frames <= 0:
            cap_frames = None

        # NEVER write to source dataset.
        # - If user explicitly provides --planner-output-dataset, write there.
        # - Otherwise, write into the evaluation run output directory.
        if args.planner_output_dataset is not None:
            out_root = Path(str(args.planner_output_dataset)).resolve()
        else:
            out_root = (run_out_dir / "planner_dataset").resolve()
        logger.info(f"Planner output dataset (no source writes): {out_root}")

        # Proceed with copying dataset and generating candidates
        def _resolve_ref(ref: str, *, episode_dir: Path, dataset_root: Path) -> str:
            p = Path(ref)
            if p.is_absolute():
                return str(p)
            cand1 = (episode_dir / p).resolve()
            if cand1.exists():
                return str(cand1)
            cand2 = (dataset_root / p).resolve()
            return str(cand2) if cand2.exists() else str(cand1)

        (out_root / "episodes").mkdir(parents=True, exist_ok=True)

        meta = find_episode_metadata_files(dataset_path)
        if args.episode_id:
            wanted = set(str(x) for x in args.episode_id)

            def _epid(pth: Path) -> str:
                return pth.parent.name if pth.name == "episode.json" else pth.stem

            meta = [m for m in meta if _epid(m) in wanted]
        elif args.max_episodes is not None:
            meta = meta[: int(args.max_episodes)]
        else:
            logger.warning(
                "No --max-episodes (or --episode-id) specified for --planner-source=onnx; "
                "defaulting to 1 episode for safety. Pass --max-episodes to run more."
            )
            meta = meta[:1]

        from tqdm import tqdm

        for m in tqdm(meta, desc="Preparing episode folders"):
            ep = load_episode(dataset_path, m)
            if not ep.schema_valid or ep.episode is None:
                raise SystemExit(
                    f"Episode schema invalid: {ep.episode_id} errors={ep.schema_errors}"
                )
            ep_out_dir = out_root / "episodes" / ep.episode_id
            ep_out_dir.mkdir(parents=True, exist_ok=True)

            rgb_lines = []
            for r in ep.rgb:
                d = r.model_dump()
                d["frame_ref"] = _resolve_ref(
                    str(r.frame_ref),
                    episode_dir=Path(ep.episode_dir),
                    dataset_root=dataset_path,
                )
                # Preserve optional planner-specific RGB refs (e.g., pinhole) if present.
                if (
                    isinstance(d.get("planner_frame_ref"), str)
                    and str(d.get("planner_frame_ref")).strip()
                ):
                    d["planner_frame_ref"] = _resolve_ref(
                        str(d["planner_frame_ref"]),
                        episode_dir=Path(ep.episode_dir),
                        dataset_root=dataset_path,
                    )
                rgb_lines.append(json.dumps(d, sort_keys=True))
            (ep_out_dir / "rgb.jsonl").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")

            # Preserve canonical odom stream if available (needed for GT/oracle metrics).
            has_odom = bool(getattr(ep, "odom", None))
            if has_odom:
                odom_lines = [json.dumps(r.model_dump(), sort_keys=True) for r in (ep.odom or [])]
                (ep_out_dir / "odom.jsonl").write_text(
                    "\n".join(odom_lines) + ("\n" if odom_lines else ""), encoding="utf-8"
                )

            (ep_out_dir / "episode.json").write_text(
                json.dumps(
                    {
                        "schema_version": ep.episode.schema_version,
                        "episode_id": ep.episode_id,
                        "streams": {
                            "rgb": {"records_ref": "rgb.jsonl"},
                            **({"odom": {"records_ref": "odom.jsonl"}} if has_odom else {}),
                            "planner_candidates": {"records_ref": "planner_candidates.jsonl"},
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        planner_dataset_path = out_root

        # IMPORTANT: If the user provided many --episode-id values (sharded execution),
        # run planner caching in ONE call so the progress bar shows correct episode totals
        # (instead of printing 0/1 repeatedly).
        #
        # Note: Stage 2 already copied ONLY the selected episodes into planner_dataset_path,
        # so calling with episode_id=None will operate on exactly that subset.
        write_planner_candidates_from_onnx(
            dataset=planner_dataset_path,
            episode_id=None,
            episode_ids=[str(x) for x in (args.episode_id or [])] if args.episode_id else None,
            model_path=Path(args.planner_onnx),
            reference_dataset=dataset_path,
            align_candidate_count_to_reference=bool(args.planner_align_to_reference),
            overwrite=bool(args.planner_overwrite),
            max_frames=cap_frames,
            stride=int(args.planner_stride),
            # If args.max_episodes was provided, Stage 2 already applied it to the copied subset.
            # Keep this None so we process the whole subset (for sharded runs, that's this rank's
            # episodes).
            max_episodes=None,
            times_jsonl_path=(
                Path(args.planner_times_jsonl_path).resolve()
                if args.planner_times_jsonl_path
                else None
            ),
            times_jsonl_time_key=str(args.planner_times_jsonl_time_key),
            times_time_tolerance_s=float(args.planner_times_time_tolerance_s),
        )
        logger.info("Planner candidate generation complete.")
        logger.info(f"Planner dataset ready: {planner_dataset_path}")
        if bool(args.planner_only):
            return 0

    # Stage 3: Run evaluation via the CLI module
    logger.info("Stage 3/3: Running evaluation...")
    from slow_brain_fast_planner.cli.trajectory_selection import main as cli_main

    eval_argv: list[str] = ["--dataset", str(planner_dataset_path)]
    # Force the run output directory we computed (so planner + metrics + report live together).
    eval_argv += ["--out", str(run_out_dir)]
    # Pass through planner source for reporting/metadata.
    eval_argv += ["--planner-source-report", str(args.planner_source)]
    if args.episode_id:
        for eid in args.episode_id:
            eval_argv += ["--episode-id", str(eid)]
    if args.max_episodes is not None:
        eval_argv += ["--max-episodes", str(args.max_episodes)]
    if args.max_snapshots_per_episode is not None:
        eval_argv += ["--max-snapshots-per-episode", str(args.max_snapshots_per_episode)]

    # Always append --overwrite since we already "own" this directory (Stage 2 might have written
    # there)
    eval_argv.append("--overwrite")

    if args.skip_validation is not None:
        eval_argv += ["--skip-validation" if args.skip_validation else "--no-skip-validation"]

    eval_argv += unknown

    # Preserve prior behavior: run from repo root
    os.chdir(str(repo_root))
    return cli_main(eval_argv)


if __name__ == "__main__":
    raise SystemExit(main())
