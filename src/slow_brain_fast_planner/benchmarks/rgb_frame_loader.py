from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from slow_brain_fast_planner.schema.canonical_episode import RGBRecord


def _resolve_records_ref_path(ref: str, *, episode_dir: Path, dataset_root: Path) -> Path:
    """Resolve a local path relative to episode_dir or dataset_root."""

    p = Path(ref)
    if p.is_absolute():
        return p

    cand1 = (episode_dir / p).resolve()
    if cand1.exists():
        return cand1

    cand2 = (dataset_root / p).resolve()
    if cand2.exists():
        return cand2

    # Fall back to episode-relative (even if missing) for clearer error messages upstream.
    return cand1


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _is_video_path(path: Path) -> bool:
    return path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _is_npy_sequence_path(path: Path) -> bool:
    """Return True iff path points to a numpy RGB sequence (.npy).

    Expected shape: (T, H, W, 3) uint8 (or numeric convertible to uint8).
    Decoding uses `frame_index` extension field on RGBRecord.
    """
    return path.suffix.lower() == ".npy"


@dataclass(frozen=True)
class RGBFrameKey:
    path: str
    frame_index: int | None


class RGBFrameLoader:
    """Load RGB frames for overlays/prompts.

    Supports:
    - image-backed frames: `RGBRecord.frame_ref` points to a PNG/JPEG/etc
    - video-backed frames: `RGBRecord.frame_ref` points to an mp4 and record has `frame_index`

    This stays benchmark-friendly:
    - decoding is dependency-optional (OpenCV is used if available)
    - a small LRU cache avoids re-decoding frames in tight loops
    """

    def __init__(self, *, cache_size: int = 128) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be > 0")
        self._cache_size = int(cache_size)
        self._frame_cache: OrderedDict[RGBFrameKey, Image.Image] = OrderedDict()
        self._video_cache: OrderedDict[str, Any] = OrderedDict()  # mp4_path -> cv2.VideoCapture

    def _cache_get(self, key: RGBFrameKey) -> Image.Image | None:
        img = self._frame_cache.get(key)
        if img is None:
            return None
        # LRU: move to end
        self._frame_cache.move_to_end(key)
        return img

    def _cache_put(self, key: RGBFrameKey, img: Image.Image) -> None:
        self._frame_cache[key] = img
        self._frame_cache.move_to_end(key)
        while len(self._frame_cache) > self._cache_size:
            self._frame_cache.popitem(last=False)

    def _get_cv2(self):
        try:
            import cv2  # type: ignore

            return cv2
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Video-backed RGB requires OpenCV (opencv-python-headless). "
                "Install it or convert your frames to images. "
                f"Import error: {e}"
            ) from e

    def _get_capture(self, mp4_path: str):
        # LRU keep a few open handles (helps sequential decoding).
        if mp4_path in self._video_cache:
            cap = self._video_cache[mp4_path]
            self._video_cache.move_to_end(mp4_path)
            return cap
        cv2 = self._get_cv2()
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video: {mp4_path}")
        self._video_cache[mp4_path] = cap
        self._video_cache.move_to_end(mp4_path)
        # Keep at most 4 videos open (tunable).
        while len(self._video_cache) > 4:
            _k, old = self._video_cache.popitem(last=False)
            try:
                old.release()
            except Exception:
                pass
        return cap

    def load(self, rgb: RGBRecord, *, episode_dir: Path, dataset_root: Path) -> Image.Image:
        """Return a PIL RGB image for the given `RGBRecord`."""

        ref_path = _resolve_records_ref_path(
            str(rgb.frame_ref), episode_dir=episode_dir, dataset_root=dataset_root
        )
        if _is_image_path(ref_path):
            key = RGBFrameKey(path=str(ref_path), frame_index=None)
            cached = self._cache_get(key)
            if cached is not None:
                return cached.copy()
            with Image.open(ref_path) as im:
                img = im.convert("RGB")
            self._cache_put(key, img)
            return img.copy()

        if _is_npy_sequence_path(ref_path):
            frame_index = getattr(rgb, "frame_index", None)
            if frame_index is None:
                raise ValueError(
                    "RGBRecord for .npy ref must include frame_index (extension field). "
                    f"frame_ref={rgb.frame_ref!r}"
                )
            idx = int(frame_index)
            key = RGBFrameKey(path=str(ref_path), frame_index=idx)
            cached = self._cache_get(key)
            if cached is not None:
                return cached.copy()

            arr = np.load(str(ref_path), mmap_mode="r")
            if arr.ndim != 4 or arr.shape[-1] != 3:
                raise ValueError(f"Invalid .npy RGB seq shape {tuple(arr.shape)} for {ref_path}")
            T = int(arr.shape[0])
            if idx < 0 or idx >= T:
                raise IndexError(f"frame_index out of range: {idx} not in [0,{T}) for {ref_path}")
            frame = np.asarray(arr[idx], dtype=np.uint8)
            img = Image.fromarray(frame, mode="RGB")
            self._cache_put(key, img)
            return img.copy()

        if _is_video_path(ref_path):
            frame_index = getattr(rgb, "frame_index", None)
            if frame_index is None:
                raise ValueError(
                    "RGBRecord for video ref must include frame_index (extension field). "
                    f"frame_ref={rgb.frame_ref!r}"
                )
            idx = int(frame_index)
            key = RGBFrameKey(path=str(ref_path), frame_index=idx)
            cached = self._cache_get(key)
            if cached is not None:
                return cached.copy()

            cap = self._get_capture(str(ref_path))
            cv2 = self._get_cv2()

            # Random access via CAP_PROP_POS_FRAMES (best-effort).
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Failed to decode frame_index={idx} from video {ref_path}")

            # OpenCV gives BGR uint8.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8), mode="RGB")
            self._cache_put(key, img)
            return img.copy()

        raise ValueError(f"Unsupported RGB frame_ref (not image/video): {rgb.frame_ref!r}")
