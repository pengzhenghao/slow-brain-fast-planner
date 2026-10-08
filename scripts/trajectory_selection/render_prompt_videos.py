from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from slow_brain_fast_planner import constants
from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files, load_episode
from slow_brain_fast_planner.benchmarks.overlays import (
    OverlayConfig,
    find_nearest_rgb_record,
    render_overlay_pil,
)
from slow_brain_fast_planner.benchmarks.planner_postprocessing import subsample_by_time_stride
from slow_brain_fast_planner.benchmarks.rgb_frame_loader import RGBFrameLoader


def _get_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "This script requires OpenCV for MP4 encode. Install opencv-python-headless. "
            f"Import error: {e}"
        ) from e


def _nearest_rgb_index(rgb_records: list[Any], t: float, tol_s: float) -> int | None:
    """Return nearest RGB record index to time t within tolerance (records sorted by t)."""
    if not rgb_records:
        return None
    times = [float(getattr(r, "t", 0.0)) for r in rgb_records]
    import bisect

    i = bisect.bisect_left(times, float(t))
    best_idx: int | None = None
    best_dist = float("inf")
    for j in (i - 1, i):
        if 0 <= j < len(times):
            dist = abs(times[j] - float(t))
            if dist < best_dist:
                best_dist = dist
                best_idx = int(j)
    if best_idx is None or best_dist > float(tol_s):
        return None
    return best_idx


def _resize_to_height(im: Image.Image, height: int) -> Image.Image:
    w, h = im.size
    if h <= 0 or w <= 0:
        return im
    if int(height) <= 0 or h == int(height):
        return im
    new_w = max(1, int(round(float(w) * (float(height) / float(h)))))
    return im.resize((int(new_w), int(height)), resample=Image.BILINEAR)


def _resize_to_width(im: Image.Image, width: int) -> Image.Image:
    w, h = im.size
    if h <= 0 or w <= 0:
        return im
    if int(width) <= 0 or w == int(width):
        return im
    new_h = max(1, int(round(float(h) * (float(width) / float(w)))))
    return im.resize((int(width), int(new_h)), resample=Image.BILINEAR)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=int(size))
    except Exception:
        return ImageFont.load_default()


def _make_banner(text: str, width: int, height: int = 28) -> Image.Image:
    banner = Image.new("RGB", (int(width), int(height)), color=(0, 0, 0))
    draw = ImageDraw.Draw(banner)
    font = _load_font(16)
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return banner


def _extract_goal_xy(record: Any) -> tuple[float, float] | None:
    """Best-effort extraction of goal_xy from a planner_candidates record (supports extension
    fields)."""
    for key in ("goal_xy", "goal_point_xy", "goal_point"):
        v = getattr(record, key, None)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                return float(v[0]), float(v[1])
            except Exception:
                pass
    # Pydantic v2 extras.
    try:
        extra = getattr(record, "model_extra", None)
        if isinstance(extra, dict):
            for key in ("goal_xy", "goal_point_xy", "goal_point"):
                v = extra.get(key)
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        return float(v[0]), float(v[1])
                    except Exception:
                        pass
    except Exception:
        pass
    return None


def _compose_prompt_frame(
    *,
    episode_id: str,
    t: float,
    snap_idx: int,
    goal_distance_m: float | None,
    overlay_img: Image.Image,
    history_imgs: list[Image.Image],
    pad_px: int = 8,
) -> Image.Image:
    # Normalize all panels to the same height for easy concatenation.
    target_h = int(overlay_img.size[1])
    hist = [_resize_to_height(im.convert("RGB"), target_h) for im in history_imgs]
    overlay2 = overlay_img.convert("RGB")

    panels = hist + [overlay2]
    if not panels:
        panels = [overlay2]

    widths = [im.size[0] for im in panels]
    total_w = int(sum(widths) + pad_px * (len(panels) - 1))
    total_h = int(target_h)

    if goal_distance_m is None or not math.isfinite(float(goal_distance_m)):
        goal_str = "goal_dist_m=NA"
    else:
        goal_str = f"goal_dist_m={float(goal_distance_m):.2f}"
    banner_text = (
        f"{episode_id}  clip={snap_idx}  t={float(t):.3f}s  {goal_str}  "
        f"prompt=[{len(hist)} hist + overlay]"
    )
    banner = _make_banner(banner_text, total_w, height=28)

    out = Image.new("RGB", (int(total_w), int(total_h + banner.size[1])), color=(20, 20, 20))

    x = 0
    y0 = 0
    for im in panels:
        out.paste(im, (int(x), int(y0)))
        x += int(im.size[0]) + int(pad_px)

    out.paste(banner, (0, total_h))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Task 2 visualization: render the exact VLM prompt image sequence and stitch to "
            "MP4.\n\n"
            "For each evaluated snapshot, the prompt consists of:\n"
            "  [history RGB frames] + [overlay image]\n\n"
            "This script produces one MP4 per episode under --out."
        )
    )
    p.add_argument("--dataset", default="data/processed", help="Canonical dataset root.")
    p.add_argument("--out", default="logs/task2_prompt_videos", help="Output directory.")
    p.add_argument(
        "--episode-id",
        action="append",
        default=None,
        help="Only render specific episode_id(s) (repeatable).",
    )
    p.add_argument("--max-episodes", type=int, default=None, help="Only render first N episodes.")
    p.add_argument(
        "--max-snapshots-per-episode", type=int, default=None, help="Cap snapshots per episode."
    )
    p.add_argument(
        "--max-snapshots-total", type=int, default=None, help="Stop after N snapshots total."
    )
    p.add_argument(
        "--snapshot-stride-s",
        type=float,
        default=None,
        help="If set, subsample snapshots at this stride (seconds).",
    )
    p.add_argument(
        "--rgb-time-tolerance-s",
        type=float,
        default=constants.DEFAULT_DATA_DT_S,
        help="Time tolerance to match rgb frame to snapshot time.",
    )
    p.add_argument("--fps", type=float, default=None, help="Output video FPS (default: inferred).")

    # Prompt knobs.
    p.add_argument(
        "--prompt-history-frames", type=int, default=0, help="Number of history RGB frames."
    )
    p.add_argument(
        "--prompt-image-width", type=int, default=None, help="Resize overlay image to width (px)."
    )

    # Overlay knobs (match the benchmark defaults).
    p.add_argument(
        "--overlay-candidate-set", choices=["planner_v2", "raw_topk"], default="planner_v2"
    )
    p.add_argument("--overlay-topk", type=int, default=6)
    p.add_argument(
        "--overlay-projection",
        choices=["fisheye_v1", "simple_xy"],
        default=constants.OVERLAY_PROJECTION,
    )
    p.add_argument("--nms-max-trajectories", type=int, default=constants.NMS_MAX_TRAJECTORIES)
    p.add_argument("--nms-distance-threshold", type=float, default=constants.NMS_DISTANCE_THRESHOLD)
    p.add_argument("--a0-prob-threshold", type=float, default=constants.PROB_THRESHOLD)

    args = p.parse_args(argv)

    dataset_root = Path(args.dataset).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    meta = find_episode_metadata_files(dataset_root)
    if not meta:
        raise SystemExit(f"No episodes found under: {dataset_root}")

    if args.episode_id:
        wanted = {str(x) for x in args.episode_id if str(x).strip()}

        def _epid(pth: Path) -> str:
            return pth.parent.name if pth.name == "episode.json" else pth.stem

        meta = [m for m in meta if _epid(m) in wanted]
        if not meta:
            raise SystemExit(f"--episode-id not found in dataset: {sorted(wanted)}")

    if args.max_episodes is not None:
        meta = meta[: int(args.max_episodes)]

    # Prompt overlay config: do NOT highlight pred/label (avoid leakage), but keep index labels.
    overlay_cfg = OverlayConfig(
        top_k=int(args.overlay_topk),
        projection=str(args.overlay_projection),
        candidate_set=str(args.overlay_candidate_set),
        nms_max_trajectories=int(args.nms_max_trajectories),
        nms_distance_threshold=float(args.nms_distance_threshold),
        prob_threshold=float(args.a0_prob_threshold),
        highlight_pred=False,
        highlight_label=False,
        label_indices=True,
    )

    cv2 = _get_cv2()
    rgb_loader = RGBFrameLoader(cache_size=128)

    total_done = 0
    pbar_ep = tqdm(meta, desc="Episodes")
    for meta_path in pbar_ep:
        if args.max_snapshots_total is not None and total_done >= int(args.max_snapshots_total):
            break
        ep = load_episode(dataset_root, meta_path)
        if not ep.schema_valid or ep.episode is None:
            continue
        if not ep.rgb or not ep.planner_candidates:
            continue

        # Subsample snapshots using the same logic as the benchmark.
        times = [float(r.t) for r in ep.planner_candidates]
        keep = subsample_by_time_stride(times, stride_s=args.snapshot_stride_s)
        records = [ep.planner_candidates[i] for i in keep]
        if args.max_snapshots_per_episode is not None:
            records = records[: int(args.max_snapshots_per_episode)]

        # Infer fps if not set.
        fps = float(args.fps) if args.fps is not None else None
        if fps is None:
            if args.snapshot_stride_s is not None and float(args.snapshot_stride_s) > 0:
                fps = 1.0 / float(args.snapshot_stride_s)
            else:
                # Default to dataset timebase.
                fps = float(constants.DEFAULT_DATA_HZ)
        fps = max(1e-3, float(fps))

        out_dir = out_root / ep.episode_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_video = out_dir / "task2_prompt.mp4"

        writer = None
        writer_size: tuple[int, int] | None = None

        def _hist_frames_for_t(t: float, *, ep=ep) -> list[Any]:
            n = int(args.prompt_history_frames)
            if n <= 0:
                return []
            idx = _nearest_rgb_index(ep.rgb, float(t), tol_s=float(args.rgb_time_tolerance_s))
            if idx is None or idx <= 0:
                return []
            start = max(0, int(idx) - int(n))
            return [ep.rgb[j] for j in range(start, int(idx))]

        pbar_ep.set_description(f"Episode {ep.episode_id}")
        pbar_frames = tqdm(records, desc="Frames", leave=False)
        for snap_idx, r in enumerate(pbar_frames):
            if args.max_snapshots_total is not None and total_done >= int(args.max_snapshots_total):
                break

            rgb_rec = find_nearest_rgb_record(
                ep.rgb, float(r.t), tol_s=float(args.rgb_time_tolerance_s)
            )
            if rgb_rec is None:
                continue

            # Render overlay from the nearest RGB record.
            base = rgb_loader.load(
                rgb_rec, episode_dir=Path(ep.episode_dir), dataset_root=dataset_root
            )
            overlay = render_overlay_pil(base_image=base, record=r, cfg=overlay_cfg)
            if args.prompt_image_width is not None:
                overlay = _resize_to_width(overlay, int(args.prompt_image_width))

            # Load history frames (raw RGB), then compose the prompt montage frame.
            hist_recs = _hist_frames_for_t(float(r.t))
            hist_imgs: list[Image.Image] = []
            for hr in hist_recs:
                try:
                    hist_imgs.append(
                        rgb_loader.load(
                            hr, episode_dir=Path(ep.episode_dir), dataset_root=dataset_root
                        ).convert("RGB")
                    )
                except Exception:
                    continue

            goal_xy = _extract_goal_xy(r)
            goal_distance_m = None
            if goal_xy is not None:
                gx, gy = goal_xy
                try:
                    goal_distance_m = float(math.hypot(float(gx), float(gy)))
                except Exception:
                    goal_distance_m = None

            frame = _compose_prompt_frame(
                episode_id=str(ep.episode_id),
                t=float(r.t),
                snap_idx=int(snap_idx),
                goal_distance_m=goal_distance_m,
                overlay_img=overlay,
                history_imgs=hist_imgs,
            )

            # Initialize writer on first frame.
            if writer is None:
                w, h = frame.size
                if w <= 0 or h <= 0:
                    continue
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video), fourcc, float(fps), (int(w), int(h)))
                if not writer.isOpened():
                    raise SystemExit(f"Failed to open video writer: {out_video}")
                writer_size = (int(w), int(h))

            # Ensure consistent size.
            assert writer_size is not None
            if frame.size != writer_size:
                frame = frame.resize(writer_size, resample=Image.BILINEAR)

            arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            total_done += 1

        if writer is not None:
            writer.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
