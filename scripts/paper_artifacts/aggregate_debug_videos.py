#!/usr/bin/env python3
"""
Aggregate multiple per-run videos into one 1080p debug mosaic.

Example:
  python scripts/aggregate_debug_videos.py /path/to/task2_live_* --out-dir local_logs/_aggregated

Inputs (looked up under <run_dir>/videos by default):
  - rerender_odom.mp4 (or newest rerender_odom_*.mp4)
    - also accepts rerender.mp4 (older naming)
  - overlays.mp4 (optional; if missing we fill with white)
  - bev_thin.mp4
  - vlm_trace.mp4 (optional; if missing we fill with white)

Output:
  - <out_dir>/<run_name>__agg.mp4

Design goal:
  Keep a synchronized timeline even if one video is shorter by freezing (tpad clone) its last frame.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _run_ffmpeg_with_progress(*, cmd: list[str], total_duration_s: float | None) -> int:
    """Run ffmpeg and display a lightweight progress indicator."""
    total_s = float(total_duration_s) if isinstance(total_duration_s, (int, float)) else None
    if total_s is not None and total_s <= 1e-6:
        total_s = None

    # We rely on ffmpeg's machine-readable progress on stdout.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    out_time_s: float | None = None
    frame: int | None = None
    speed: str | None = None
    last_print = 0.0

    def _parse_hhmmss(s: str) -> float | None:
        # "HH:MM:SS.micro"
        try:
            parts = s.strip().split(":")
            if len(parts) != 3:
                return None
            hh = float(parts[0])
            mm = float(parts[1])
            ss = float(parts[2])
            return float(hh * 3600.0 + mm * 60.0 + ss)
        except Exception:
            return None

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = (line or "").strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k == "out_time_ms":
                try:
                    out_time_s = float(v) / 1e6
                except Exception:
                    pass
            elif k == "out_time_us":
                try:
                    out_time_s = float(v) / 1e6
                except Exception:
                    pass
            elif k == "out_time":
                t = _parse_hhmmss(v)
                if t is not None:
                    out_time_s = float(t)
            elif k == "frame":
                try:
                    frame = int(v)
                except Exception:
                    pass
            elif k == "speed":
                speed = v
            elif k == "progress" and v == "end":
                break

            now = time.time()
            if now - last_print < 0.2:
                continue
            last_print = now

            if total_s is not None and out_time_s is not None:
                pct = max(0.0, min(100.0, 100.0 * float(out_time_s) / float(total_s)))
                msg = f"\rencoding: {pct:5.1f}%  t={out_time_s:6.1f}/{total_s:6.1f}s"
            elif frame is not None:
                msg = f"\rencoding: frame={int(frame)}"
            else:
                msg = "\rencoding: ..."
            if speed:
                msg += f"  speed={speed}"
            print(msg, file=sys.stderr, end="", flush=True)
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass

    rc = int(proc.wait())
    # End the progress line cleanly.
    try:
        print("", file=sys.stderr)
    except Exception:
        pass
    return rc


def _human_bytes(n: int) -> str:
    try:
        x = float(max(0, int(n)))
    except Exception:
        return str(n)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    u = 0
    while x >= 1024.0 and u < len(units) - 1:
        x /= 1024.0
        u += 1
    if u == 0:
        return f"{int(round(x))} {units[u]}"
    return f"{x:.2f} {units[u]}"


def _ffprobe_json(path: Path) -> dict:
    if shutil.which("ffprobe") is None:
        return {}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-of",
        "json",
        str(path),
    ]
    p = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if int(p.returncode) != 0:
        return {}
    try:
        return json.loads(p.stdout or "{}")
    except Exception:
        return {}


def _video_info(path: Path) -> dict[str, object]:
    obj = _ffprobe_json(path)
    out: dict[str, object] = {}
    try:
        fmt = obj.get("format") if isinstance(obj, dict) else None
        if isinstance(fmt, dict) and fmt.get("duration") is not None:
            out["duration_s"] = float(fmt.get("duration"))
    except Exception:
        pass
    try:
        streams = obj.get("streams") if isinstance(obj, dict) else None
        s0 = streams[0] if isinstance(streams, list) and streams else {}
        if isinstance(s0, dict):
            if s0.get("width") is not None:
                out["width"] = int(s0.get("width"))
            if s0.get("height") is not None:
                out["height"] = int(s0.get("height"))
            afr = str(s0.get("avg_frame_rate") or "")
            if "/" in afr:
                a, b = afr.split("/", 1)
                fa = float(a)
                fb = float(b)
                if fb != 0.0:
                    out["fps"] = fa / fb
            elif afr.strip():
                out["fps"] = float(afr)
            fr = s0.get("nb_read_frames")
            if fr is not None:
                out["frames"] = int(fr)
    except Exception:
        pass
    return out


def _print_summary(*, run_dir: Path, videos_dir: Path, out_mp4: Path) -> None:
    try:
        size_b = int(out_mp4.stat().st_size) if out_mp4.exists() else 0
    except Exception:
        size_b = 0
    info = _video_info(out_mp4)
    try:
        uri = out_mp4.resolve().as_uri()
    except Exception:
        uri = str(out_mp4)

    print("", file=sys.stderr)
    print("=== Video generated ===", file=sys.stderr)
    print(f"Run folder:   {run_dir}", file=sys.stderr)
    print(f"Videos folder:{videos_dir}", file=sys.stderr)
    print(f"Video file:   {out_mp4}", file=sys.stderr)
    print(f"Video link:   {uri}", file=sys.stderr)
    print(f"Size:         {_human_bytes(size_b)} ({size_b} bytes)", file=sys.stderr)
    if isinstance(info.get("width"), int) and isinstance(info.get("height"), int):
        print(f"Resolution:   {int(info['width'])}x{int(info['height'])}", file=sys.stderr)
    if isinstance(info.get("fps"), (int, float)):
        print(f"FPS:          {float(info['fps']):.3f}", file=sys.stderr)
    if isinstance(info.get("frames"), int):
        print(f"Frames:       {int(info['frames'])}", file=sys.stderr)
    print("=======================", file=sys.stderr)


def _pick_video(videos_dir: Path, name: str, globs: list[str]) -> Path | None:
    # Prefer exact name, else newest matching glob.
    p0 = videos_dir / name
    if p0.exists():
        return p0
    cands: list[Path] = []
    for g in globs:
        try:
            cands.extend(sorted(videos_dir.glob(g)))
        except Exception:
            continue
    cands = [p for p in cands if p.is_file()]
    if not cands:
        return None
    # newest mtime
    cands.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return cands[0]


def _layout_filter(
    *,
    w: int,
    h: int,
    fps: float,
    pad_s: list[float],
    stretch: list[float],
    layout: str,
    left_frac: float,
    top_frac: float,
    draw_titles: bool,
    titles: list[str] | None,
    crop_overlays_square: bool,
    crop_bev_square: bool,
    right_frac: float,
    border_px: int,
    border_rgba: str,
) -> str:
    """Return ffmpeg filter_complex for 4 inputs.

    Input indices:
      0 rerender_odom
      1 overlays
      2 bev_thin
      3 vlm_trace
    """

    def _even(x: int) -> int:
        y = int(x)
        return y + (y % 2)

    def _esc_drawtext(s: str) -> str:
        # Minimal escaping for ffmpeg drawtext 'text='...'' strings.
        return str(s).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    def chain(i: int, tw: int, th: int, pad: float, st: float) -> str:
        # Normalize timestamps + fps, scale with aspect preserved, pad to exact target, freeze tail
        # if needed.
        parts: list[str] = [
            # Some inputs (notably overlays.mp4) are not truly 5Hz over the full run; retime if
            # requested.
            f"[{i}:v]setpts=(PTS-STARTPTS)*{float(max(1e-6, st)):.9f}",
            f"fps={float(fps):g}",
        ]
        if int(i) == 1 and bool(crop_overlays_square):
            # Center-crop to square for a cleaner "VLM Input" panel.
            parts.append("crop='min(iw,ih)':'min(iw,ih)':(iw-ow)/2:(ih-oh)/2")
        if int(i) == 2 and bool(crop_bev_square):
            # Keep BEV legend at the top: crop height to width from the top (when portrait).
            parts.append("crop='min(iw,ih)':'min(iw,ih)':(iw-ow)/2:0")
        parts.append(f"scale={tw}:{th}:force_original_aspect_ratio=decrease")
        parts.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=white")
        if int(border_px) > 0:
            parts.append(f"drawbox=x=0:y=0:w=iw:h=ih:color={str(border_rgba)}:t={int(border_px)}")
        if bool(draw_titles):
            title = None
            try:
                if isinstance(titles, list) and 0 <= int(i) < len(titles):
                    title = titles[int(i)]
            except Exception:
                title = None
            if isinstance(title, str) and title.strip():
                t2 = _esc_drawtext(title.strip())
                if int(i) in (1, 2, 3):
                    # Titles without box (avoid covering content; BEV title can overlap the gray
                    # line).
                    parts.append(
                        "drawtext="
                        f"text='{t2}':x=w-tw-10:y=8:fontsize=18:"
                        "fontcolor=black:box=0:shadowcolor=white:shadowx=1:shadowy=1"
                    )
                else:
                    parts.append(
                        "drawtext="
                        f"text='{t2}':x=w-tw-10:y=8:fontsize=18:"
                        "fontcolor=black:box=1:boxcolor=white@0.65:boxborderw=6"
                    )
        if pad and pad > 1e-3:
            parts.append(f"tpad=stop_mode=clone:stop_duration={pad:.3f}")
        return ",".join(parts) + f"[v{i}]"

    def chain_vlm_landscape(i: int, tw: int, th: int, pad: float, st: float) -> str:
        # Make VLM trace fill a landscape box and crop from the top to reduce bottom whitespace.
        parts: list[str] = [
            f"[{i}:v]setpts=(PTS-STARTPTS)*{float(max(1e-6, st)):.9f}",
            f"fps={float(fps):g}",
            f"scale={tw}:{th}:force_original_aspect_ratio=increase",
            f"crop={tw}:{th}:(iw-ow)/2:0",
        ]
        if int(border_px) > 0:
            parts.append(f"drawbox=x=0:y=0:w=iw:h=ih:color={str(border_rgba)}:t={int(border_px)}")
        if bool(draw_titles):
            title = None
            try:
                if isinstance(titles, list) and 0 <= int(i) < len(titles):
                    title = titles[int(i)]
            except Exception:
                title = None
            if isinstance(title, str) and title.strip():
                t2 = _esc_drawtext(title.strip())
                parts.append(
                    "drawtext="
                    f"text='{t2}':x=w-tw-10:y=8:fontsize=18:"
                    "fontcolor=black:box=0:shadowcolor=white:shadowx=1:shadowy=1"
                )
        if pad and pad > 1e-3:
            parts.append(f"tpad=stop_mode=clone:stop_duration={pad:.3f}")
        return ",".join(parts) + f"[v{i}]"

    # Aesthetic default: left column = real-world views (large), right column = BEV + text
    # (smaller).
    # Total: 1920x1080
    if str(layout) == "grid2x2":
        # 2x2 grid: each cell 960x540
        cells = [(960, 540), (960, 540), (960, 540), (960, 540)]
        f0 = chain(0, cells[0][0], cells[0][1], pad_s[0], stretch[0])
        f1 = chain(1, cells[1][0], cells[1][1], pad_s[1], stretch[1])
        f2 = chain(2, cells[2][0], cells[2][1], pad_s[2], stretch[2])
        f3 = chain(3, cells[3][0], cells[3][1], pad_s[3], stretch[3])
        return (
            f"{f0};{f1};{f2};{f3};"
            "[v0][v1][v2][v3]"
            "xstack=inputs=4:layout=0_0|960_0|0_540|960_540:fill=white[vout]"
        )

    if str(layout) == "rerender_right3":
        # Left: huge rerender (no title). Right: 3 rows:
        #   - overlays (square, optional crop)      -> "VLM INPUT"
        #   - vlm_trace (landscape 2:1)            -> "VLM REASONING"
        #   - bev_thin (square, optional crop)     -> "BEV TRAJECTORY"
        #
        # Choose right column width as a fraction of the total width.
        rf = float(right_frac)
        if not (0.15 <= rf <= 0.70):
            rf = 0.40
        right_w = _even(int(round(float(w) * rf)))
        right_w = max(2, min(int(w - 2), int(right_w)))
        left_w = _even(int(w - right_w))
        # Middle row ~2:1 (width:height). Top row targets ~16:9 to reduce letterboxing for overlays.
        row2_h = _even(int(round(float(right_w) * 0.5)))
        row2_h = max(2, min(int(h - 4), int(row2_h)))
        row1_h = _even(int(round(float(right_w) * (9.0 / 16.0))))
        row1_h = max(2, min(int(h - row2_h - 2), int(row1_h)))
        row3_h = _even(int(h - row1_h - row2_h))
        if row3_h < 2:
            row3_h = 2
            row1_h = _even(int(h - row2_h - row3_h))

        f0 = chain(0, left_w, h, pad_s[0], stretch[0])
        f1 = chain(1, right_w, row1_h, pad_s[1], stretch[1])
        f3 = chain_vlm_landscape(3, right_w, row2_h, pad_s[3], stretch[3])
        f2 = chain(2, right_w, row3_h, pad_s[2], stretch[2])
        return (
            f"{f0};{f1};{f2};{f3};"
            "[v1][v3][v2]vstack=inputs=3[right];"
            "[v0][right]hstack=inputs=2[vout]"
        )

    # default: "main_column"
    lf = float(left_frac)
    if not (0.1 <= lf <= 0.9):
        lf = 0.5
    left_w = _even(int(round(float(w) * lf)))
    left_w = max(2, min(int(w - 2), int(left_w)))
    right_w = _even(int(w - left_w))
    tf = float(top_frac)
    if not (0.2 <= tf <= 0.8):
        tf = 0.5
    top_h = _even(int(round(float(h) * tf)))
    top_h = max(2, min(int(h - 2), int(top_h)))
    bot_h = _even(int(h - top_h))

    f0 = chain(0, left_w, top_h, pad_s[0], stretch[0])
    f1 = chain(1, left_w, bot_h, pad_s[1], stretch[1])
    f2 = chain(2, right_w, top_h, pad_s[2], stretch[2])
    f3 = chain(3, right_w, bot_h, pad_s[3], stretch[3])
    return (
        f"{f0};{f1};{f2};{f3};"
        "[v0][v1]vstack=inputs=2[left];"
        "[v2][v3]vstack=inputs=2[right];"
        "[left][right]hstack=inputs=2[vout]"
    )


def _aggregate_one(
    *, run_dir: Path, videos_dir: Path, out_dir: Path, args: argparse.Namespace
) -> int:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH.", file=sys.stderr)
        return 3

    rerender = _pick_video(
        videos_dir,
        "rerender_odom.mp4",
        [
            "rerender_odom_*.mp4",
            "rerender_odom*.mp4",
            "rerender.mp4",
            "rerender_*.mp4",
            "camera_waypoints.mp4",
        ],
    )
    overlays = _pick_video(videos_dir, "overlays.mp4", ["overlays*.mp4"])
    bev = _pick_video(videos_dir, "bev_thin.mp4", ["bev_thin*.mp4"])
    vlm = _pick_video(videos_dir, "vlm_trace.mp4", ["vlm_trace*.mp4"])

    missing_required = [n for n, p in [("rerender_odom", rerender), ("bev_thin", bev)] if p is None]
    if missing_required:
        print(
            f"ERROR: missing required videos in {videos_dir}: {', '.join(missing_required)}",
            file=sys.stderr,
        )
        return 2

    missing_optional = [n for n, p in [("overlays", overlays), ("vlm_trace", vlm)] if p is None]
    if missing_optional:
        print(
            f"WARNING: missing optional videos in {videos_dir}: {', '.join(missing_optional)} "
            "(will fill with white).",
            file=sys.stderr,
        )

    rerender = rerender.resolve()
    bev = bev.resolve()
    overlays = overlays.resolve() if overlays is not None else None
    vlm = vlm.resolve() if vlm is not None else None

    # Determine max duration and pad shorter streams by freezing the last frame.
    present_inputs: list[tuple[str, Path]] = [("rerender_odom", rerender), ("bev_thin", bev)]
    if overlays is not None:
        present_inputs.append(("overlays", overlays))
    if vlm is not None:
        present_inputs.append(("vlm_trace", vlm))
    present_infos = [_video_info(p) for _nm, p in present_inputs]
    durs: list[float] = []
    for inf in present_infos:
        d = inf.get("duration_s")
        durs.append(float(d) if isinstance(d, (int, float)) and d is not None else 0.0)
    target_d = max(durs) if durs else 0.0

    # Optional: retime overlays to span the full run duration (legacy workaround).
    stretch = [1.0, 1.0, 1.0, 1.0]
    if bool(args.retime_overlays) and overlays is not None and target_d > 0:
        # Only retime overlays (index 1) if it appears significantly shorter.
        # NOTE: overlays is input index 1 in our filter graph.
        d1 = _video_info(overlays).get("duration_s") if overlays is not None else None
        d1 = float(d1) if isinstance(d1, (int, float)) else 0.0
        if d1 > 1e-6 and (target_d - d1) > 0.5:
            stretch[1] = float(target_d) / float(d1)

    def _blank_lavfi(*, fps: float, duration_s: float) -> list[str]:
        # Small source; we scale/pad later. Make it finite to avoid hanging encodes.
        d = float(duration_s) if duration_s > 1e-6 else 1.0
        return ["-f", "lavfi", "-i", f"color=white:s=16x16:r={float(fps):g}:d={float(d):.6f}"]

    # Build canonical 4-input list expected by _layout_filter:
    #   0 rerender, 1 overlays, 2 bev, 3 vlm_trace
    inputs4: list[tuple[str, Path | None]] = [
        ("rerender_odom", rerender),
        ("overlays", overlays),
        ("bev_thin", bev),
        ("vlm_trace", vlm),
    ]

    infos4: list[dict[str, object]] = []
    durs4: list[float] = []
    for _nm, p in inputs4:
        if p is None:
            infos4.append({})
            durs4.append(float(target_d))
        else:
            inf = _video_info(p)
            infos4.append(inf)
            d = inf.get("duration_s")
            durs4.append(float(d) if isinstance(d, (int, float)) and d is not None else 0.0)
    pad_s = [max(0.0, float(target_d) - float(d)) for d in durs4]

    print("Aggregating videos:", file=sys.stderr)
    for (nm, p), inf, pad in zip(inputs4, infos4, pad_s, strict=False):
        wh = ""
        if (
            p is not None
            and isinstance(inf.get("width"), int)
            and isinstance(inf.get("height"), int)
        ):
            wh = f"{int(inf['width'])}x{int(inf['height'])}"
        dur = inf.get("duration_s") if p is not None else float(target_d)
        dur_s = (
            f"{float(dur):.2f}s"
            if isinstance(dur, (int, float))
            else ("?" if p is not None else f"{float(target_d):.2f}s")
        )
        pad_s2 = f"{float(pad):.2f}s" if float(pad) > 1e-6 else "0"
        st = stretch[1] if nm == "overlays" else 1.0
        st_s = f"  retime_x{float(st):.3f}" if abs(float(st) - 1.0) > 1e-3 else ""
        p_s = str(p) if p is not None else "<white>"
        print(
            f"- {nm:12s}: {p_s}  ({wh or '?:?'}  dur={dur_s}  pad={pad_s2}){st_s}", file=sys.stderr
        )
    print(f"Target duration: {float(target_d):.2f}s", file=sys.stderr)

    out_name = (str(args.name).strip() if args.name else run_dir.name).strip()
    if not out_name:
        out_name = "run"
    out_mp4 = (out_dir / f"{out_name}__agg.mp4").resolve()

    w, h = 1920, 1080
    if str(args.layout) == "rerender_right3":
        titles = ["", "VLM INPUT", "BEV TRAJECTORY", "VLM REASONING"]
    else:
        titles = ["RERENDER", "OVERLAY", "BEV_THIN", "VLM_TRACE"]
    fc = _layout_filter(
        w=w,
        h=h,
        fps=float(args.fps),
        pad_s=pad_s,
        stretch=stretch,
        layout=str(args.layout),
        left_frac=float(args.left_frac),
        top_frac=float(args.top_frac),
        draw_titles=bool(args.titles),
        titles=titles,
        crop_overlays_square=bool(args.crop_overlays_square),
        crop_bev_square=bool(args.crop_bev_square),
        right_frac=float(args.right_frac),
        border_px=int(args.border_px),
        border_rgba=str(args.border_rgba),
    )

    # Use ffmpeg progress output for a clean progress indicator.
    cmd: list[str] = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-loglevel",
        "error",
        "-nostats",
        "-progress",
        "pipe:1",
    ]
    # Inputs in order expected by filter graph.
    cmd += ["-i", str(rerender)]
    if overlays is not None:
        cmd += ["-i", str(overlays)]
    else:
        cmd += _blank_lavfi(fps=float(args.fps), duration_s=float(target_d))
    cmd += ["-i", str(bev)]
    if vlm is not None:
        cmd += ["-i", str(vlm)]
    else:
        cmd += _blank_lavfi(fps=float(args.fps), duration_s=float(target_d))
    cmd += [
        "-filter_complex",
        fc,
        "-map",
        "[vout]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        str(args.preset),
        "-crf",
        str(int(args.crf)),
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]

    print(str(out_mp4))
    rc = _run_ffmpeg_with_progress(
        cmd=cmd, total_duration_s=float(target_d) if target_d > 0 else None
    )
    if rc != 0:
        print(f"ERROR: ffmpeg failed (rc={rc})", file=sys.stderr)
        return int(rc)

    _print_summary(run_dir=run_dir, videos_dir=videos_dir, out_mp4=out_mp4)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "run_dirs", type=str, nargs="+", help="One or more run folders (shell globs ok)."
    )
    ap.add_argument("--videos-dir", type=str, default=None, help="Defaults to <run_dir>/videos")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="local_logs/_aggregated",
        help="Output directory (outside run folder).",
    )
    ap.add_argument(
        "--layout",
        type=str,
        default="rerender_right3",
        choices=["rerender_right3", "main_column", "grid2x2"],
    )
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", type=str, default="veryfast")
    ap.add_argument(
        "--name", type=str, default=None, help="Output name (defaults to run_dir name)."
    )
    ap.add_argument(
        "--left-frac",
        type=float,
        default=0.5,
        help="When layout=main_column, fraction of width for the left column (default: 0.5).",
    )
    ap.add_argument(
        "--top-frac",
        type=float,
        default=0.5,
        help="When layout=main_column, fraction of height for the top row (default: 0.5).",
    )
    ap.add_argument(
        "--right-frac",
        type=float,
        default=0.40,
        help="When layout=rerender_right3, fraction of width for the right column (default: 0.40).",
    )
    ap.add_argument(
        "--no-titles",
        dest="titles",
        action="store_false",
        default=True,
        help="Disable panel titles (default: titles enabled).",
    )
    ap.add_argument(
        "--crop-overlays-square",
        dest="crop_overlays_square",
        action="store_true",
        default=False,
        help="Enable square center-crop for overlays (default: disabled).",
    )
    ap.add_argument(
        "--no-crop-overlays-square",
        dest="crop_overlays_square",
        action="store_false",
        help="Disable square center-crop for overlays.",
    )
    ap.add_argument(
        "--crop-bev-square",
        dest="crop_bev_square",
        action="store_true",
        default=False,
        help="Enable square crop for bev_thin (default: disabled).",
    )
    ap.add_argument(
        "--no-crop-bev-square",
        dest="crop_bev_square",
        action="store_false",
        help="Disable square crop for bev_thin.",
    )
    ap.add_argument(
        "--border-px",
        type=int,
        default=4,
        help="Panel border thickness in pixels (default: 4; set 0 to disable).",
    )
    ap.add_argument(
        "--border-rgba",
        type=str,
        default="0xE6E6E6@1.0",
        help="Panel border color (ffmpeg syntax, default: 0xE6E6E6@1.0).",
    )
    ap.add_argument(
        "--retime-overlays",
        action="store_true",
        default=False,
        help=(
            "If true, retime overlays.mp4 to span the same duration as rerender_odom/vlm_trace "
            "(default: False). "
            "This helps when overlays were captured at a lower effective rate but encoded at 5fps."
        ),
    )
    ap.add_argument("--no-retime-overlays", dest="retime_overlays", action="store_false")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rc_all = 0
    for run_dir_s in list(args.run_dirs or []):
        run_dir = Path(str(run_dir_s)).expanduser().resolve()
        if not run_dir.exists():
            print(f"WARNING: run_dir does not exist: {run_dir}", file=sys.stderr)
            rc_all = max(rc_all, 2)
            continue
        videos_dir = (
            Path(args.videos_dir).expanduser().resolve()
            if args.videos_dir
            else (run_dir / "videos")
        )
        rc = _aggregate_one(run_dir=run_dir, videos_dir=videos_dir, out_dir=out_dir, args=args)
        rc_all = max(int(rc_all), int(rc))
    return int(rc_all)


if __name__ == "__main__":
    raise SystemExit(main())
