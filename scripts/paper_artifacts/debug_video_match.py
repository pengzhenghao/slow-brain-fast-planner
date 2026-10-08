from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int


def _read_odom_times(odom_jsonl: Path) -> np.ndarray:
    ts: list[float] = []
    with odom_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("t", None)
            if isinstance(t, (int, float)) and math.isfinite(float(t)):
                ts.append(float(t))
    if not ts:
        raise RuntimeError(f"No timestamps parsed from {odom_jsonl}")
    return np.asarray(ts, dtype=np.float64)


def _infer_dt_s(times_s: np.ndarray) -> float:
    if times_s.size < 2:
        return 0.2
    d = np.diff(times_s)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 0.2
    return float(np.median(d))


def _probe_video(path: Path) -> _VideoInfo:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required for this debug script.") from e

    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
    w0 = int(round(float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)))
    h0 = int(round(float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)))
    cap.release()
    if fps <= 0:
        fps = 0.0
    return _VideoInfo(path=path, fps=fps, frame_count=frame_count, width=w0, height=h0)


def _read_frame_bgr(path: Path, frame_index: int) -> np.ndarray | None:
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required for this debug script.") from e

    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _resize_bgr(frame: np.ndarray, *, w: int, h: int) -> np.ndarray:
    import cv2  # type: ignore

    if frame.shape[0] == h and frame.shape[1] == w:
        return frame
    return cv2.resize(frame, (int(w), int(h)), interpolation=cv2.INTER_AREA)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    d = aa - bb
    return float(np.mean(d * d))


def _psnr_from_mse(mse: float, *, peak: float = 255.0) -> float:
    if mse <= 0 or not math.isfinite(mse):
        return float("inf")
    return float(10.0 * math.log10((peak * peak) / mse))


def _best_shift_by_mse(
    cp_frames: list[np.ndarray],
    scene_frames: list[np.ndarray],
    *,
    max_shift: int = 10,
) -> tuple[int, dict[int, float]]:
    """
    Search integer shifts in 5Hz-frame units.
    shift = +k means compare cp[i+k] vs scene[i].
    """
    assert len(cp_frames) == len(scene_frames)
    n = len(cp_frames)
    shift_to_mse: dict[int, float] = {}
    for s in range(-int(max_shift), int(max_shift) + 1):
        mses: list[float] = []
        for i in range(n):
            j = i + s
            if j < 0 or j >= n:
                continue
            mses.append(_mse(cp_frames[j], scene_frames[i]))
        shift_to_mse[int(s)] = float(np.mean(mses)) if mses else float("inf")
    best = min(shift_to_mse.items(), key=lambda kv: kv[1])[0]
    return int(best), shift_to_mse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Debug whether an episode's CP 1080p video aligns with the scene low-res video.\n\n"
            "This script expects an episode directory that contains:\n"
            "  episodes/<episode_id>/assets/rgb/front.mp4            (CP 1080p if present)\n"
            "  episodes/<episode_id>/assets/rgb/front_pinhole.mp4    (scene video)\n\n"
            "It extracts a ~20s window at the 5Hz odom cadence:\n"
            "  - CP frames sampled using the same time->frame_index mapping used by our adapter\n"
            "  - scene frames sampled by time as well\n"
            "and writes side-by-side PNGs plus a numeric similarity report."
        )
    )
    p.add_argument("--dataset", required=True, help="Processed dataset dir (contains episodes/).")
    p.add_argument(
        "--episode-id",
        default=None,
        help="Episode id under <dataset>/episodes/. If omitted, uses first.",
    )
    p.add_argument(
        "--out", default="data/_debug_video_match", help="Output directory for extracted PNGs."
    )
    p.add_argument(
        "--start-t", type=float, default=0.0, help="Start time in seconds (default 0.0)."
    )
    p.add_argument(
        "--seconds", type=float, default=20.0, help="Window length in seconds (default 20.0)."
    )
    p.add_argument(
        "--max-shift", type=int, default=10, help="Search +-this many 5Hz frames for best offset."
    )
    args = p.parse_args(argv)

    ds = Path(args.dataset).resolve()
    if not ds.is_dir():
        raise SystemExit(f"--dataset not found: {ds}")
    eps_dir = (ds / "episodes").resolve()
    if not eps_dir.is_dir():
        raise SystemExit(f"episodes/ not found under --dataset: {eps_dir}")

    episode_id = args.episode_id
    if episode_id is None:
        eps = sorted([p for p in eps_dir.iterdir() if p.is_dir()])
        if not eps:
            raise SystemExit(f"No episodes found under: {eps_dir}")
        episode_id = eps[0].name

    ep = (eps_dir / str(episode_id)).resolve()
    if not ep.is_dir():
        raise SystemExit(f"Episode dir not found: {ep}")

    odom = (ep / "odom.jsonl").resolve()
    if not odom.is_file():
        raise SystemExit(f"Missing odom.jsonl: {odom}")

    assets_rgb = (ep / "assets" / "rgb").resolve()
    if not assets_rgb.is_dir():
        raise SystemExit(f"Missing assets/rgb dir: {assets_rgb}")

    cp_mp4 = (assets_rgb / "front.mp4").resolve()
    if not cp_mp4.is_file():
        raise SystemExit(f"Missing CP mp4 (front.mp4): {cp_mp4}")

    scene_mp4 = None
    for cand in ("front_pinhole.mp4", "front_scene.mp4"):
        pth = (assets_rgb / cand).resolve()
        if pth.is_file() and pth != cp_mp4:
            scene_mp4 = pth
            break
    if scene_mp4 is None:
        # Fall back: any other mp4 under assets/rgb
        mp4s = sorted(
            [p for p in assets_rgb.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
        )
        mp4s = [p for p in mp4s if p.resolve() != cp_mp4.resolve()]
        scene_mp4 = mp4s[0].resolve() if mp4s else None
    if scene_mp4 is None or not scene_mp4.is_file():
        raise SystemExit(f"Could not find scene video mp4 under: {assets_rgb}")

    times = _read_odom_times(odom)
    dt_s = _infer_dt_s(times)
    if dt_s <= 0:
        dt_s = 0.2

    cp_info = _probe_video(cp_mp4)
    scene_info = _probe_video(scene_mp4)

    out_root = Path(args.out).resolve() / str(episode_id)
    out_cp = (out_root / "cp_resized").resolve()
    out_scene = (out_root / "scene").resolve()
    out_pair = (out_root / "pair").resolve()
    out_cp.mkdir(parents=True, exist_ok=True)
    out_scene.mkdir(parents=True, exist_ok=True)
    out_pair.mkdir(parents=True, exist_ok=True)

    start_t = float(args.start_t)
    win_s = float(args.seconds)
    if win_s <= 0:
        raise SystemExit("--seconds must be > 0")
    start_i = int(round(start_t / dt_s))
    start_i = max(0, min(start_i, int(times.size) - 1))
    n_steps = int(round(win_s / dt_s))
    n_steps = max(1, min(n_steps, int(times.size) - start_i))

    # Derive output frame size from scene frame (fallback to video metadata).
    scene0 = _read_frame_bgr(scene_info.path, 0)
    if scene0 is None:
        raise SystemExit(f"Could not read frame 0 from scene video: {scene_info.path}")
    out_h, out_w = int(scene0.shape[0]), int(scene0.shape[1])

    # If fps is missing, fall back to 1/dt.
    fps_cp = float(cp_info.fps) if cp_info.fps > 0 else float(1.0 / dt_s)
    fps_scene = float(scene_info.fps) if scene_info.fps > 0 else float(1.0 / dt_s)

    cp_frames_resized: list[np.ndarray] = []
    scene_frames: list[np.ndarray] = []
    mses: list[float] = []
    psnrs: list[float] = []

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required for this debug script.") from e

    # Extract per-5Hz-step frames.
    for k in range(n_steps):
        i = int(start_i + k)
        t = float(times[i])
        cp_idx = int(round((t - float(times[0])) * fps_cp))
        scene_idx = int(round((t - float(times[0])) * fps_scene))
        if cp_info.frame_count > 0:
            cp_idx = max(0, min(cp_idx, int(cp_info.frame_count) - 1))
        if scene_info.frame_count > 0:
            scene_idx = max(0, min(scene_idx, int(scene_info.frame_count) - 1))

        cp_bgr = _read_frame_bgr(cp_info.path, cp_idx)
        sc_bgr = _read_frame_bgr(scene_info.path, scene_idx)
        if cp_bgr is None or sc_bgr is None:
            # Stop early if we run out of frames.
            break

        cp_r = _resize_bgr(cp_bgr, w=out_w, h=out_h)
        sc_r = sc_bgr

        cp_frames_resized.append(cp_r)
        scene_frames.append(sc_r)

        mse = _mse(cp_r, sc_r)
        mses.append(mse)
        psnrs.append(_psnr_from_mse(mse))

        fn = f"{k:04d}_t={t:07.3f}_i={i:04d}_cp={cp_idx:06d}_scene={scene_idx:06d}.png"
        cv2.imwrite(str((out_cp / fn).resolve()), cp_r)
        cv2.imwrite(str((out_scene / fn).resolve()), sc_r)
        pair = np.concatenate([cp_r, sc_r], axis=1)
        cv2.imwrite(str((out_pair / fn).resolve()), pair)

    if not cp_frames_resized or not scene_frames:
        raise SystemExit("No frames extracted; check video paths and indices.")

    best_shift, shift_to_mse = _best_shift_by_mse(
        cp_frames_resized,
        scene_frames,
        max_shift=int(args.max_shift),
    )

    report = {
        "dataset": str(ds),
        "episode_id": str(episode_id),
        "odom_dt_s_median": float(dt_s),
        "start_t_s": float(start_t),
        "seconds_requested": float(win_s),
        "steps_extracted": int(len(cp_frames_resized)),
        "cp_video": {
            "path": str(cp_info.path),
            "fps": float(cp_info.fps),
            "frames": int(cp_info.frame_count),
            "w": int(cp_info.width),
            "h": int(cp_info.height),
        },
        "scene_video": {
            "path": str(scene_info.path),
            "fps": float(scene_info.fps),
            "frames": int(scene_info.frame_count),
            "w": int(scene_info.width),
            "h": int(scene_info.height),
        },
        "resized_to": {"w": int(out_w), "h": int(out_h)},
        "mse_mean": float(np.mean(mses)) if mses else None,
        "psnr_db_mean": float(np.mean(psnrs)) if psnrs else None,
        "best_shift_5hz_frames": int(best_shift),
        "shift_to_mse": {
            str(k): float(v) for k, v in sorted(shift_to_mse.items(), key=lambda kv: kv[0])
        },
        "out_dir": str(out_root),
    }
    (out_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
