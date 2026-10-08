from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files, load_episode
from slow_brain_fast_planner.utils.io import write_json

ClipPhase = Literal["pre", "center", "post", "none"]


@dataclass(frozen=True)
class TakeoverClipsConfig:
    # Clip / query definition.
    history_window_s: float = 4.0
    stride_s: float = 2.0
    horizon_s: float = 2.0  # prediction horizon
    k_frames: int = 4  # frames fed to VLM (uniform from history)

    # Conservative debouncing (hysteresis) for auto_enabled.
    # - ON: switch to takeover quickly (1 false frame) to avoid filtering out brief takeovers
    # - OFF: switch back to auto quickly by default (1 true frame) to preserve "post" regions;
    #        increase this if you want stronger suppression of chatter when giving back control.
    debounce_on_count: int = 1
    debounce_off_count: int = 1

    # Phase labeling.
    post_window_s: float = 2.0  # consider "post" if within this time after a takeover ends

    # Filtering.
    min_t0_s: float | None = None  # optional floor on query times
    # Planner-safety filtering.
    # Require at least this many RGB frames at times <= t0 (matches the ONNX planner's
    # contiguous history-window requirement).
    # If set, we will also store the resolved contiguous RGB window in each clip record.
    min_rgb_history_window: int | None = None

    def __post_init__(self) -> None:
        if self.history_window_s <= 0:
            raise ValueError("history_window_s must be > 0")
        if self.stride_s <= 0:
            raise ValueError("stride_s must be > 0")
        if self.horizon_s <= 0:
            raise ValueError("horizon_s must be > 0")
        if self.k_frames <= 0:
            raise ValueError("k_frames must be > 0")
        if self.debounce_on_count <= 0:
            raise ValueError("debounce_on_count must be > 0")
        if self.debounce_off_count <= 0:
            raise ValueError("debounce_off_count must be > 0")
        if self.post_window_s < 0:
            raise ValueError("post_window_s must be >= 0")
        if self.min_rgb_history_window is not None and int(self.min_rgb_history_window) <= 0:
            raise ValueError("min_rgb_history_window must be > 0 if set")


def _ensure_empty_dir(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _subsample_times(times: list[float], *, stride_s: float) -> list[int]:
    """Return indices to keep from a sorted times list (deterministic)."""
    if not times:
        return []
    keep: list[int] = []
    next_t = float(times[0])
    for i, t in enumerate(times):
        tt = float(t)
        if tt + 1e-12 >= next_t:
            keep.append(i)
            next_t = tt + float(stride_s)
    return keep


def _debounce_hysteresis(
    *,
    auto_enabled: list[bool],
    on_count: int,
    off_count: int,
) -> list[bool]:
    """Debounce a boolean auto_enabled series using hysteresis.

    - state True  = auto-enabled
    - state False = takeover

    Conservative choice: enter takeover quickly (small on_count). For leaving takeover,
    a larger off_count can suppress chatter; default keeps timing faithful.
    """

    if not auto_enabled:
        return []

    state = bool(auto_enabled[0])
    false_run = 0
    true_run = 0
    out: list[bool] = []
    for v in auto_enabled:
        vv = bool(v)
        if not vv:
            false_run += 1
            true_run = 0
        else:
            true_run += 1
            false_run = 0

        if state is True:
            # Switch into takeover (False) quickly when seeing a short false run.
            if false_run >= int(on_count):
                state = False
                # reset runs so we require a fresh true_run to come back.
                true_run = 0
        else:
            # Switch back into auto (True) more slowly.
            if true_run >= int(off_count):
                state = True
                false_run = 0

        out.append(bool(state))
    return out


def _segments_from_takeover_state(
    times: list[float], is_auto: list[bool]
) -> list[dict[str, float]]:
    """Return takeover segments as [{'start_t':..., 'end_t':...}, ...].

    A segment is a contiguous interval where is_auto==False.
    end_t is the time of the first auto==True after the segment (best-effort).
    """
    if len(times) != len(is_auto):
        raise ValueError("times and is_auto must have same length")
    segs: list[dict[str, float]] = []
    in_seg = False
    start_t: float | None = None
    for t, a in zip(times, is_auto, strict=False):
        tt = float(t)
        aa = bool(a)
        if (not aa) and (not in_seg):
            in_seg = True
            start_t = tt
        elif aa and in_seg:
            segs.append({"start_t": float(start_t if start_t is not None else tt), "end_t": tt})
            in_seg = False
            start_t = None
    if in_seg:
        # Open-ended segment; end at last timestamp (best-effort).
        segs.append(
            {
                "start_t": float(start_t if start_t is not None else float(times[-1])),
                "end_t": float(times[-1]),
            }
        )
    return segs


def _any_takeover_in_window(
    *,
    times: list[float],
    is_auto: list[bool],
    t0: float,
    horizon_s: float,
) -> bool | None:
    """Return True iff takeover occurs in [t0, t0+horizon].

    Returns None if there is insufficient future coverage (times[-1] < t0+horizon).
    """
    if not times:
        return None
    t0 = float(t0)
    t1 = float(t0) + float(horizon_s)
    if float(times[-1]) + 1e-9 < t1:
        return None

    from bisect import bisect_left, bisect_right

    lo = bisect_left(times, t0 - 1e-12)
    hi = bisect_right(times, t1 + 1e-12)
    for i in range(lo, hi):
        if not bool(is_auto[i]):
            return True
    return False


def _nearest_record_at_or_before(times: list[float], *, t: float) -> int:
    """Return index of last time <= t; if none, return 0."""
    from bisect import bisect_right

    if not times:
        return -1
    idx = bisect_right(times, float(t)) - 1
    return idx if idx >= 0 else 0


def _rgb_index_at_or_before(rgb_times: list[float], *, t: float) -> int | None:
    """Return index of last RGB record at time <= t (None if none)."""
    from bisect import bisect_right

    if not rgb_times:
        return None
    idx = bisect_right(rgb_times, float(t)) - 1
    return int(idx) if idx >= 0 else None


def _has_rgb_history_window(*, rgb_times: list[float], t: float, window: int) -> bool:
    """Return True iff there are at least `window` RGB records at times <= t.

    This intentionally matches the ONNX planner's history windowing: it finds the last
    RGB record with time <= t, then requires a contiguous index window of length `window`.
    """
    if int(window) <= 0:
        return True
    idx = _rgb_index_at_or_before(rgb_times, t=float(t))
    if idx is None:
        return False
    start = int(idx) - int(window) + 1
    return start >= 0


def _uniform_history_rgb(
    *,
    rgb_times: list[float],
    rgb_frame_refs: list[str],
    t0: float,
    history_window_s: float,
    k_frames: int,
) -> list[dict[str, Any]] | None:
    if not rgb_times or not rgb_frame_refs or len(rgb_times) != len(rgb_frame_refs):
        return None
    if int(k_frames) <= 0:
        return None

    t0 = float(t0)
    start = float(t0) - float(history_window_s)
    if int(k_frames) == 1:
        targets = [t0]
    else:
        step = float(history_window_s) / float(int(k_frames) - 1)
        targets = [start + i * step for i in range(int(k_frames))]

    out: list[dict[str, Any]] = []
    for tt in targets:
        idx = _nearest_record_at_or_before(rgb_times, t=tt)
        if idx < 0:
            return None
        out.append({"t": float(rgb_times[idx]), "frame_ref": str(rgb_frame_refs[idx])})
    return out


def _phase_for_t0(
    *,
    t0: float,
    is_auto_at_t0: bool,
    takeover_segments: list[dict[str, float]],
    horizon_s: float,
    post_window_s: float,
) -> ClipPhase:
    t0 = float(t0)
    horizon_s = float(horizon_s)
    post_window_s = float(post_window_s)

    if not is_auto_at_t0:
        return "center"

    # Upcoming takeover starts within horizon -> pre.
    next_start: float | None = None
    for seg in takeover_segments:
        st = float(seg["start_t"])
        if st > t0 + 1e-12:
            next_start = st
            break
    if next_start is not None and next_start <= t0 + horizon_s + 1e-12:
        return "pre"

    # Recently ended takeover -> post.
    last_end: float | None = None
    for seg in takeover_segments:
        et = float(seg["end_t"])
        if et <= t0 + 1e-12:
            last_end = et
        else:
            break
    if last_end is not None and (t0 - last_end) <= post_window_s + 1e-12:
        return "post"

    return "none"


def build_takeover_clips(
    *,
    dataset_root: Path,
    out_dir: Path,
    config: TakeoverClipsConfig | None = None,
    overwrite: bool = False,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    """Build predictive takeover clips (history-only inputs) from a canonical dataset.

    Output files:
    - takeover_clips.jsonl
    - summary.json
    """
    config = config or TakeoverClipsConfig()
    dataset_root = dataset_root.resolve()
    out_dir = out_dir.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    _ensure_empty_dir(out_dir, overwrite=bool(overwrite))

    metas = find_episode_metadata_files(dataset_root)
    if max_episodes is not None:
        if int(max_episodes) <= 0:
            raise ValueError("max_episodes must be > 0 if set")
        metas = metas[: int(max_episodes)]
    if not metas:
        raise ValueError(f"No episodes found under dataset: {dataset_root}")

    clips_path = out_dir / "takeover_clips.jsonl"
    total_clips = 0
    written_clips = 0
    written_label_takeover = 0
    written_label_no_takeover = 0
    written_phase_counts: dict[str, int] = {}
    written_auto_debounced_counts: dict[str, int] = {"auto": 0, "human": 0}
    written_auto_raw_counts: dict[str, int] = {"auto": 0, "human": 0}

    episodes_loaded = 0
    episodes_with_auto = 0
    episodes_with_rgb = 0

    skipped_episodes: list[dict[str, Any]] = []
    skipped_clips: dict[str, int] = {}

    with clips_path.open("w", encoding="utf-8") as f:
        for meta in tqdm(metas, total=len(metas), desc="takeover_clips:episodes"):
            ep = load_episode(dataset_root, meta)
            if not ep.schema_valid or ep.episode is None:
                skipped_episodes.append(
                    {
                        "episode_id": ep.episode_id,
                        "episode_meta_path": ep.episode_meta_path,
                        "reason": "schema_invalid",
                        "details": ep.schema_errors,
                    }
                )
                continue
            episodes_loaded += 1

            pc = [r for r in ep.planner_candidates if r.auto_enabled is not None]
            if not pc:
                skipped_episodes.append(
                    {
                        "episode_id": ep.episode_id,
                        "episode_meta_path": ep.episode_meta_path,
                        "reason": "missing_auto_enabled",
                        "details": ep.planner_candidates_errors,
                    }
                )
                continue
            episodes_with_auto += 1

            if not ep.rgb:
                skipped_episodes.append(
                    {
                        "episode_id": ep.episode_id,
                        "episode_meta_path": ep.episode_meta_path,
                        "reason": "missing_rgb",
                        "details": ep.rgb_errors,
                    }
                )
                continue
            episodes_with_rgb += 1

            pc_times = [float(r.t) for r in pc]
            pc_auto_raw = [bool(r.auto_enabled) for r in pc]
            pc_auto_debounced = _debounce_hysteresis(
                auto_enabled=pc_auto_raw,
                on_count=int(config.debounce_on_count),
                off_count=int(config.debounce_off_count),
            )

            takeover_segments = _segments_from_takeover_state(pc_times, pc_auto_debounced)

            # Choose query times by striding over planner_candidates timeline.
            keep_idx = _subsample_times(pc_times, stride_s=float(config.stride_s))
            for i in keep_idx:
                t0 = float(pc_times[i])
                if config.min_t0_s is not None and t0 < float(config.min_t0_s):
                    continue

                total_clips += 1

                label = _any_takeover_in_window(
                    times=pc_times,
                    is_auto=pc_auto_debounced,
                    t0=t0,
                    horizon_s=float(config.horizon_s),
                )
                if label is None:
                    skipped_clips["insufficient_future"] = (
                        skipped_clips.get("insufficient_future", 0) + 1
                    )
                    continue

                # Build uniform history frame bundle from RGB.
                rgb_times = [float(r.t) for r in ep.rgb]
                rgb_refs = [str(r.frame_ref) for r in ep.rgb]

                # Optional planner-safety filtering: ensure planner-history window exists at t0.
                planner_rgb_window: list[dict[str, Any]] | None = None
                rgb_index_at_or_before_t0: int | None = None
                if config.min_rgb_history_window is not None:
                    w = int(config.min_rgb_history_window)
                    if not _has_rgb_history_window(rgb_times=rgb_times, t=t0, window=w):
                        skipped_clips["insufficient_rgb_history_for_planner"] = (
                            skipped_clips.get("insufficient_rgb_history_for_planner", 0) + 1
                        )
                        continue
                    rgb_index_at_or_before_t0 = _rgb_index_at_or_before(rgb_times, t=t0)
                    if rgb_index_at_or_before_t0 is None:
                        skipped_clips["missing_rgb_at_or_before_t0"] = (
                            skipped_clips.get("missing_rgb_at_or_before_t0", 0) + 1
                        )
                        continue
                    start_idx = int(rgb_index_at_or_before_t0) - w + 1
                    planner_rgb_window = [
                        {"t": float(rgb_times[j]), "frame_ref": str(rgb_refs[j])}
                        for j in range(start_idx, int(rgb_index_at_or_before_t0) + 1)
                    ]
                frames = _uniform_history_rgb(
                    rgb_times=rgb_times,
                    rgb_frame_refs=rgb_refs,
                    t0=t0,
                    history_window_s=float(config.history_window_s),
                    k_frames=int(config.k_frames),
                )
                if frames is None:
                    skipped_clips["missing_rgb_history"] = (
                        skipped_clips.get("missing_rgb_history", 0) + 1
                    )
                    continue

                is_auto_at_t0 = bool(pc_auto_debounced[i])
                phase = _phase_for_t0(
                    t0=t0,
                    is_auto_at_t0=is_auto_at_t0,
                    takeover_segments=takeover_segments,
                    horizon_s=float(config.horizon_s),
                    post_window_s=float(config.post_window_s),
                )

                rec = {
                    "episode_id": ep.episode_id,
                    "t0": float(t0),
                    "history_window_s": float(config.history_window_s),
                    "horizon_s": float(config.horizon_s),
                    "stride_s": float(config.stride_s),
                    "k_frames": int(config.k_frames),
                    "frames": frames,  # list[{t, frame_ref}] uniformly sampled from history
                    # Planner-history window (contiguous) at/before t0 (optional but recommended
                    # for safety).
                    "min_rgb_history_window": (
                        None
                        if config.min_rgb_history_window is None
                        else int(config.min_rgb_history_window)
                    ),
                    "rgb_index_at_or_before_t0": rgb_index_at_or_before_t0,
                    "planner_rgb_window": planner_rgb_window,
                    "label_takeover_request": 1 if bool(label) else 0,
                    "phase": str(phase),
                    "is_pre": bool(phase == "pre"),
                    "is_center": bool(phase == "center"),
                    "is_post": bool(phase == "post"),
                    # Helpful debug fields (no evaluation assumptions).
                    "auto_enabled_raw_at_t0": bool(pc_auto_raw[i]),
                    "auto_enabled_debounced_at_t0": bool(is_auto_at_t0),
                    "debounce": {
                        "on_count": int(config.debounce_on_count),
                        "off_count": int(config.debounce_off_count),
                    },
                }
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                written_clips += 1
                if bool(label):
                    written_label_takeover += 1
                else:
                    written_label_no_takeover += 1
                written_phase_counts[str(phase)] = written_phase_counts.get(str(phase), 0) + 1
                written_auto_debounced_counts["auto" if bool(is_auto_at_t0) else "human"] += 1
                written_auto_raw_counts["auto" if bool(pc_auto_raw[i]) else "human"] += 1

    summary = {
        "dataset": str(dataset_root),
        "out_dir": str(out_dir),
        "config": {
            "history_window_s": float(config.history_window_s),
            "stride_s": float(config.stride_s),
            "horizon_s": float(config.horizon_s),
            "k_frames": int(config.k_frames),
            "debounce_on_count": int(config.debounce_on_count),
            "debounce_off_count": int(config.debounce_off_count),
            "post_window_s": float(config.post_window_s),
            "min_t0_s": config.min_t0_s if config.min_t0_s is None else float(config.min_t0_s),
            "min_rgb_history_window": (
                None
                if config.min_rgb_history_window is None
                else int(config.min_rgb_history_window)
            ),
        },
        "episodes_total": int(len(metas)),
        "episodes_loaded": int(episodes_loaded),
        "episodes_with_auto_enabled": int(episodes_with_auto),
        "episodes_with_rgb": int(episodes_with_rgb),
        "clips_total_considered": int(total_clips),
        "clips_written": int(written_clips),
        "clips_written_label_takeover_request": int(written_label_takeover),
        "clips_written_label_no_request": int(written_label_no_takeover),
        "clips_written_phase_counts": {k: int(v) for k, v in sorted(written_phase_counts.items())},
        "clips_written_auto_enabled_debounced_at_t0_counts": {
            "auto": int(written_auto_debounced_counts.get("auto", 0)),
            "human": int(written_auto_debounced_counts.get("human", 0)),
        },
        "clips_written_auto_enabled_raw_at_t0_counts": {
            "auto": int(written_auto_raw_counts.get("auto", 0)),
            "human": int(written_auto_raw_counts.get("human", 0)),
        },
        "clips_skipped": {k: int(v) for k, v in sorted(skipped_clips.items())},
        "skipped_episodes": skipped_episodes,
        "clips_path": str(clips_path),
    }
    write_json(out_dir / "summary.json", summary)
    return summary
