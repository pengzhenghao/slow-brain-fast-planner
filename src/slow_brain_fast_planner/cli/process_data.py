from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slow_brain_fast_planner import constants
from slow_brain_fast_planner.benchmarks.goal_filtering import filter_takeover_clips_jsonl_by_quality
from slow_brain_fast_planner.benchmarks.takeover_clips import (
    TakeoverClipsConfig,
    build_takeover_clips,
)
from slow_brain_fast_planner.benchmarks.takeover_split import build_takeover_split
from slow_brain_fast_planner.ingest import convert_rss_human_data_processed, convert_s2e_v2_folder
from slow_brain_fast_planner.ingest.rss_human_adapter import (
    iter_rss_segments,
    write_trajectory_selection_eval_clips_one_per_episode,
)
from slow_brain_fast_planner.utils.io import write_json
from slow_brain_fast_planner.utils.trajectory_utils import GoalFilterConfig, GtFutureFilterConfig

_SAMPLE_RE = re.compile(r"^sample_(\d+)\.png$")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _looks_like_episode_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        for p in path.iterdir():
            if p.is_file() and _SAMPLE_RE.match(p.name):
                return True
    except FileNotFoundError:
        return False
    return False


def _find_episode_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for child in root.iterdir():
        if child.is_dir() and _looks_like_episode_dir(child):
            out.append(child)
    out.sort(key=lambda p: p.name)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Process raw s2e_v2 folder-style logs into a canonical dataset:\n"
            "  raw episode folders -> canonical episodes -> takeover_split -> takeover_clips.\n\n"
            "This is the recommended data processing entrypoint for the repo."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--raw",
        default="data/raw",
        help="Input raw directory (episode folder or root of episode folders).",
    )
    parser.add_argument(
        "--processed",
        default="data/processed",
        help=(
            "Output canonical dataset directory (will contain dataset_manifest.json and episodes/)."
        ),
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=constants.DEFAULT_DATA_DT_S,
        help=(
            "Sample time step in seconds. For 5Hz data, use 0.2. "
            "If you set --hz, it will override this."
        ),
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=None,
        help=(
            "If set, override dt via dt_s = 1/hz "
            f"(e.g., {constants.DEFAULT_DATA_HZ:g} -> {constants.DEFAULT_DATA_DT_S:g})."
        ),
    )
    parser.add_argument(
        "--copy-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy PNG frames into episode assets/ (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the processed dataset directory if it exists.",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Only convert the first N episode folders (sorted by name).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only convert the first N samples per episode (sorted by sample index).",
    )
    parser.add_argument(
        "--min-converted-samples",
        type=int,
        default=200,
        help=(
            "Skip episodes with fewer than this many converted samples "
            "(default: 200 ~ 40s @ 5Hz). Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--max-skip-frac",
        type=float,
        default=None,
        help="Skip episodes where samples_skipped / samples_found exceeds this fraction.",
    )

    # RSS_Human_Data_Processed adapter knobs (new curated format).
    parser.add_argument(
        "--rss-static-candidates-json",
        default=None,
        help=(
            "If --raw points at RSS_Human_Data_Processed, use this static trajectory-selection "
            "candidate set JSON (default: assets/trajectory_selection_static_candidates"
            "/takeover_kmeans_medoids/static_candidates_k24.json)."
        ),
    )
    parser.add_argument(
        "--rss-source-rgb-root",
        default=None,
        help=(
            "If --raw points at RSS_Human_Data_Processed, optionally provide the original high-res "
            "RSS_Human_Data root (containing <scenario>/sample_<i>.png). If set, the converter "
            "uses "
            "those frames instead of the low-res obs_255 npy frames; the path must exist."
        ),
    )
    parser.add_argument(
        "--rss-rgb-frames",
        type=int,
        default=21,
        help="For RSS segments, how many history frames to expose via rgb.jsonl (default: 21).",
    )

    # Takeover split knobs.
    parser.add_argument(
        "--split-out",
        default=None,
        help="Split output directory (default: <processed>/takeover_split).",
    )
    parser.add_argument(
        "--auto-threshold",
        type=float,
        default=0.8,
        help="Mostly-auto threshold (default: 0.8).",
    )
    parser.add_argument(
        "--min-t",
        type=float,
        default=21.0,
        help="Min t (seconds) to include in split (default: 21.0).",
    )
    parser.add_argument(
        "--min-rgb-history-window",
        type=int,
        default=21,
        help="Require at least this many RGB frames at times <= t (default: 21).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for balanced sampling (default: 0).",
    )

    # Takeover clip knobs.
    parser.add_argument(
        "--clips-out",
        default=None,
        help="Clips output directory (default: <processed>/takeover_clips).",
    )
    parser.add_argument(
        "--clip-history-window-s",
        type=float,
        default=constants.DEFAULT_TAKEOVER_CLIP_HISTORY_WINDOW_S,
        help="Clip history L (default: 4.0)",
    )
    parser.add_argument(
        "--clip-stride-s",
        type=float,
        default=constants.DEFAULT_TAKEOVER_CLIP_STRIDE_S,
        help="Clip stride S (default: 2.0)",
    )
    parser.add_argument(
        "--clip-horizon-s",
        type=float,
        default=constants.DEFAULT_TAKEOVER_CLIP_HORIZON_S,
        help="Clip prediction horizon H (default: 2.0)",
    )
    parser.add_argument(
        "--clip-k-frames",
        type=int,
        default=constants.DEFAULT_TAKEOVER_CLIP_K_FRAMES,
        help="Frames per clip K (default: 4)",
    )
    parser.add_argument(
        "--clip-debounce-on-count",
        type=int,
        default=constants.DEFAULT_TAKEOVER_CLIP_DEBOUNCE_ON_COUNT,
        help="Enter takeover after this many consecutive auto=False (default: 1)",
    )
    parser.add_argument(
        "--clip-debounce-off-count",
        type=int,
        default=constants.DEFAULT_TAKEOVER_CLIP_DEBOUNCE_OFF_COUNT,
        help="Leave takeover after this many consecutive auto=True (default: 1)",
    )
    parser.add_argument(
        "--clip-post-window-s",
        type=float,
        default=constants.DEFAULT_TAKEOVER_CLIP_POST_WINDOW_S,
        help="Label phase=post if within this time after takeover ends (default: 4.0)",
    )
    parser.add_argument(
        "--clip-min-rgb-history-window",
        type=int,
        default=21,
        help=(
            "For takeover clips, require at least this many RGB frames at times <= t0 "
            "(planner-history safety filter). (default: 21)"
        ),
    )
    # Goal sanity filtering (clips only).
    parser.add_argument(
        "--goal-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, filter takeover_clips whose t0 has a clearly-bad goal (default: true).",
    )
    parser.add_argument("--goal-filter-max-distance-m", type=float, default=120.0)
    parser.add_argument("--goal-filter-behind-min-distance-m", type=float, default=30.0)
    parser.add_argument("--goal-filter-behind-x-threshold-m", type=float, default=0.0)
    parser.add_argument("--goal-filter-back-bearing-abs-deg", type=float, default=150.0)
    parser.add_argument("--goal-filter-back-bearing-min-distance-m", type=float, default=30.0)
    parser.add_argument(
        "--goal-filter-require-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, also filter clips with missing goal info at t0 (default: false).",
    )

    # GT-future sanity filtering (clips only; uses canonical odom).
    parser.add_argument(
        "--gt-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, filter takeover_clips whose GT future (odom) looks wrong at t0 (default: "
        "true).",
    )
    parser.add_argument("--gt-filter-backward-x-threshold-m", type=float, default=-0.10)
    parser.add_argument("--gt-filter-max-backward-frac", type=float, default=0.80)
    parser.add_argument("--gt-filter-max-backward-mean-x-m", type=float, default=-0.50)
    parser.add_argument("--gt-filter-max-step-m", type=float, default=10.0)
    parser.add_argument("--gt-filter-max-total-m", type=float, default=50.0)
    parser.add_argument(
        "--gt-filter-fallback-num-points",
        type=int,
        default=10,
        help="If candidates are missing, use this many GT points for filtering (default: 10).",
    )

    args = parser.parse_args(argv)
    if args.hz is not None:
        hz = float(args.hz)
        if hz <= 0:
            raise SystemExit("--hz must be > 0 if set")
        args.dt_s = 1.0 / hz

    raw = Path(args.raw).resolve()
    processed = Path(args.processed).resolve()
    split_out = Path(args.split_out).resolve() if args.split_out else (processed / "takeover_split")
    clips_out = Path(args.clips_out).resolve() if args.clips_out else (processed / "takeover_clips")

    if not raw.exists():
        raise SystemExit(f"--raw path not found: {raw}")

    if bool(args.overwrite) and processed.exists():
        shutil.rmtree(processed)
    processed.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Adapter auto-detect: RSS_Human_Data_Processed/{scenario}/time_xxxxx/
    #
    # This curated dataset is already segmented into 21-frame windows with:
    #   obs_255_HWC_RGB_seq.npy, goal_Xforward_Yleft.npy, positive_behavior.npy
    #
    # We convert each segment into a *single-snapshot* canonical episode. The dataset
    # has no takeover annotations, so instead of the debounced takeover_split /
    # takeover_clips pipeline we emit one evaluation clip per episode at its snapshot time.
    # ----------------------------------------------------------------------
    is_rss = False
    try:
        for _eid, _seg in iter_rss_segments(raw):
            is_rss = True
            break
    except Exception:
        is_rss = False

    if is_rss:
        rss_source_rgb_root = (
            Path(args.rss_source_rgb_root).resolve() if args.rss_source_rgb_root else None
        )
        if rss_source_rgb_root is not None and not rss_source_rgb_root.exists():
            raise SystemExit(f"--rss-source-rgb-root path not found: {rss_source_rgb_root}")
        report = convert_rss_human_data_processed(
            input_root=raw,
            output_dataset_dir=processed,
            dt_s=float(args.dt_s),
            rgb_frames_per_episode=int(args.rss_rgb_frames),
            static_candidates_json=(
                Path(args.rss_static_candidates_json) if args.rss_static_candidates_json else None
            ),
            source_rgb_root=rss_source_rgb_root,
            overwrite=bool(args.overwrite),
            limit_episodes=(int(args.limit_episodes) if args.limit_episodes is not None else None),
        )

        # One label-free evaluation clip per micro-episode at its planner snapshot time.
        snapshot_clips_summary = write_trajectory_selection_eval_clips_one_per_episode(
            dataset_root=processed,
            overwrite=bool(args.overwrite),
        )

        conversion_summary = report.to_dict()
        write_json(processed / "conversion_report.json", conversion_summary)
        out_obj = {
            "conversion": conversion_summary,
            "takeover_split": None,
            "takeover_clips": None,
            "trajectory_selection_eval_clips": snapshot_clips_summary,
            "takeover_clips_quality_filter": None,
            "takeover_clips_report": None,
        }
        write_json(processed / "process_summary.json", out_obj)
        print(json.dumps(out_obj, indent=2, sort_keys=True))
        logger.info("[process_data] RSS adapter complete. Output: %s", str(processed))
        return 0

    if _looks_like_episode_dir(raw):
        episode_dirs = [raw]
    else:
        episode_dirs = _find_episode_dirs(raw)

    if args.limit_episodes is not None:
        if int(args.limit_episodes) <= 0:
            raise SystemExit("--limit-episodes must be > 0 if set")
        episode_dirs = episode_dirs[: int(args.limit_episodes)]

    if not episode_dirs:
        raise SystemExit(f"No episode folders found under: {raw}")

    conversion_reports: list[dict[str, Any]] = []
    skipped_episodes: list[dict[str, Any]] = []

    min_converted_samples = (
        None if int(args.min_converted_samples) <= 0 else int(args.min_converted_samples)
    )
    if min_converted_samples is not None:
        logger.info(f"[process_data] min_converted_samples enabled: {min_converted_samples}")

    for ep_dir in tqdm(episode_dirs, total=len(episode_dirs), desc="process:convert_episodes"):
        report = convert_s2e_v2_folder(
            input_dir=ep_dir,
            output_dataset_dir=processed,
            episode_id=ep_dir.name,
            dt_s=float(args.dt_s),
            copy_images=bool(args.copy_images),
            overwrite=True,  # always rebuild episode dir when processing
            write_manifest=True,
            max_samples=args.max_samples,
        )

        # Optional episode-level filters.
        if min_converted_samples is not None and report.samples_converted < int(
            min_converted_samples
        ):
            logger.info(
                f"[process_data] skip episode_id={report.episode_id} "
                f"reason=min_converted_samples samples_converted={int(report.samples_converted)} "
                f"min_converted_samples={int(min_converted_samples)}"
            )
            skipped_episodes.append(
                {
                    "episode_id": report.episode_id,
                    "reason": "min_converted_samples",
                    "samples_converted": int(report.samples_converted),
                    "min_converted_samples": int(min_converted_samples),
                }
            )
            continue

        if args.max_skip_frac is not None and report.samples_found > 0:
            skip_frac = float(report.samples_skipped) / float(report.samples_found)
            if skip_frac > float(args.max_skip_frac):
                logger.info(
                    f"[process_data] skip episode_id={report.episode_id} "
                    f"reason=max_skip_frac samples_found={int(report.samples_found)} "
                    f"samples_skipped={int(report.samples_skipped)} "
                    f"skip_frac={float(skip_frac):.4f} max_skip_frac={float(args.max_skip_frac)}"
                )
                skipped_episodes.append(
                    {
                        "episode_id": report.episode_id,
                        "reason": "max_skip_frac",
                        "samples_found": int(report.samples_found),
                        "samples_skipped": int(report.samples_skipped),
                        "skip_frac": float(skip_frac),
                        "max_skip_frac": float(args.max_skip_frac),
                    }
                )
                continue

        conversion_reports.append(report.to_dict())

    limit_episodes_val = None if args.limit_episodes is None else int(args.limit_episodes)
    max_samples_val = None if args.max_samples is None else int(args.max_samples)

    if skipped_episodes:
        from collections import Counter

        reasons = Counter([str(s.get("reason", "unknown")) for s in skipped_episodes])
        logger.info(
            f"[process_data] episodes_skipped={len(skipped_episodes)} reasons={dict(reasons)}"
        )

    conversion_summary = {
        "adapter": "s2e_v2_curation_pipeline",
        "raw": str(raw),
        "processed": str(processed),
        "dt_s": float(args.dt_s),
        "copy_images": bool(args.copy_images),
        "overwrite": bool(args.overwrite),
        "limit_episodes": limit_episodes_val,
        "max_samples": max_samples_val,
        "episodes_found": int(len(episode_dirs)),
        "episodes_converted": int(len(conversion_reports)),
        "episodes_skipped": int(len(skipped_episodes)),
        "samples_found_total": int(sum(r["samples_found"] for r in conversion_reports)),
        "samples_converted_total": int(sum(r["samples_converted"] for r in conversion_reports)),
        "samples_skipped_total": int(sum(r["samples_skipped"] for r in conversion_reports)),
        "steps_auto_total": int(sum(r.get("auto_enabled_true", 0) for r in conversion_reports)),
        "steps_human_total": int(sum(r.get("auto_enabled_false", 0) for r in conversion_reports)),
        "takeover_events_total": int(sum(r.get("takeover_events", 0) for r in conversion_reports)),
        "distance_m_total": float(sum(r.get("distance_m", 0.0) for r in conversion_reports)),
        "distance_auto_m_total": float(
            sum(r.get("distance_auto_m", 0.0) for r in conversion_reports)
        ),
        "distance_human_m_total": float(
            sum(r.get("distance_human_m", 0.0) for r in conversion_reports)
        ),
        "time_s_total": float(sum(r["samples_converted"] for r in conversion_reports))
        * float(args.dt_s),
        "time_auto_s_total": float(sum(r.get("time_auto_s", 0.0) for r in conversion_reports)),
        "time_human_s_total": float(sum(r.get("time_human_s", 0.0) for r in conversion_reports)),
        "episodes": conversion_reports,
        "skipped": skipped_episodes,
    }
    write_json(processed / "conversion_report.json", conversion_summary)

    # Build takeover split.
    split_summary = build_takeover_split(
        dataset_root=processed,
        out_dir=split_out,
        auto_threshold=float(args.auto_threshold),
        min_t=float(args.min_t),
        min_rgb_history_window=int(args.min_rgb_history_window),
        seed=int(args.seed),
        overwrite=True,
    )

    # Build takeover clips.
    clips_cfg = TakeoverClipsConfig(
        history_window_s=float(args.clip_history_window_s),
        stride_s=float(args.clip_stride_s),
        horizon_s=float(args.clip_horizon_s),
        k_frames=int(args.clip_k_frames),
        debounce_on_count=int(args.clip_debounce_on_count),
        debounce_off_count=int(args.clip_debounce_off_count),
        post_window_s=float(args.clip_post_window_s),
        min_rgb_history_window=(
            None
            if int(args.clip_min_rgb_history_window) <= 0
            else int(args.clip_min_rgb_history_window)
        ),
    )
    clips_summary = build_takeover_clips(
        dataset_root=processed,
        out_dir=clips_out,
        config=clips_cfg,
        overwrite=True,
    )

    clips_quality_filter_summary = None
    if bool(args.goal_filter) or bool(args.gt_filter):
        try:
            gf_cfg = GoalFilterConfig(
                enabled=bool(args.goal_filter),
                max_distance_m=(
                    None
                    if args.goal_filter_max_distance_m is None
                    else float(args.goal_filter_max_distance_m)
                ),
                behind_x_threshold_m=float(args.goal_filter_behind_x_threshold_m),
                behind_min_distance_m=float(args.goal_filter_behind_min_distance_m),
                back_bearing_abs_deg=float(args.goal_filter_back_bearing_abs_deg),
                back_bearing_min_distance_m=float(args.goal_filter_back_bearing_min_distance_m),
                require_goal=bool(args.goal_filter_require_goal),
            )
            gt_cfg = GtFutureFilterConfig(
                enabled=bool(args.gt_filter),
                backward_x_threshold_m=float(args.gt_filter_backward_x_threshold_m),
                max_backward_frac=float(args.gt_filter_max_backward_frac),
                max_backward_mean_x_m=float(args.gt_filter_max_backward_mean_x_m),
                max_step_m=float(args.gt_filter_max_step_m),
                max_total_m=float(args.gt_filter_max_total_m),
            )
            clips_quality_filter_summary = filter_takeover_clips_jsonl_by_quality(
                dataset_root=processed,
                takeover_clips_jsonl_path=(clips_out / "takeover_clips.jsonl"),
                goal_cfg=gf_cfg,
                gt_cfg=gt_cfg,
                traj_dt_s=float(args.traj_dt_s),
                fallback_num_points=int(args.gt_filter_fallback_num_points),
                overwrite=True,
                keep_backup=True,
            )
            logger.info(
                "[process_data] takeover_clips quality-filter summary: %s",
                clips_quality_filter_summary,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[process_data] takeover_clips quality-filter failed (continuing): %s", str(e)
            )

    out_obj = {
        "conversion": conversion_summary,
        "takeover_split": split_summary,
        "takeover_clips": clips_summary,
        "takeover_clips_quality_filter": clips_quality_filter_summary,
        "takeover_clips_report": None,
    }
    write_json(processed / "process_summary.json", out_obj)

    logger.info("=" * 40)
    logger.info("DATASET PROCESS SUMMARY")
    logger.info("-" * 40)
    logger.info(f"Episodes converted: {conversion_summary['episodes_converted']}")
    logger.info(f"Total samples:     {conversion_summary['samples_converted_total']}")
    logger.info(f"Total distance:    {conversion_summary['distance_m_total']:.2f} m")
    logger.info(f"  - Auto:          {conversion_summary['distance_auto_m_total']:.2f} m")
    logger.info(f"  - Human:         {conversion_summary['distance_human_m_total']:.2f} m")
    logger.info(
        f"Total time:        {conversion_summary['time_s_total']:.1f} s "
        f"(~{conversion_summary['time_s_total'] / 60:.1f} min)"
    )
    logger.info(f"  - Auto:          {conversion_summary['time_auto_s_total']:.1f} s")
    logger.info(f"  - Human:         {conversion_summary['time_human_s_total']:.1f} s")
    logger.info(f"Total steps:       {conversion_summary['samples_converted_total']}")
    logger.info(f"  - Auto:          {conversion_summary['steps_auto_total']}")
    logger.info(f"  - Human:         {conversion_summary['steps_human_total']}")
    logger.info(f"Takeover events:   {conversion_summary['takeover_events_total']}")
    if clips_summary:
        logger.info(f"Total clips:       {clips_summary.get('clips_written', 0)}")
    logger.info("-" * 40)
    logger.info(f"Output: {processed}")
    logger.info("=" * 40)

    print(json.dumps(out_obj, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
