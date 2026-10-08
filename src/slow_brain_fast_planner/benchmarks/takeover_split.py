from __future__ import annotations

import json
import random
import shutil
from bisect import bisect_right
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files, load_episode
from slow_brain_fast_planner.utils.io import write_json


def _ensure_empty_dir(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _sample_balanced(
    positives: list[dict[str, Any]],
    negatives: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(int(seed))
    n = min(len(positives), len(negatives))
    if n <= 0:
        return [], []
    pos_sample = rng.sample(positives, k=n)
    neg_sample = rng.sample(negatives, k=n)
    return pos_sample, neg_sample


def _has_rgb_history(*, rgb_times: list[float], t: float, window: int) -> bool:
    """Return True iff there are at least `window` RGB records at times <= t.

    This matches the ONNX planner's history windowing: it finds the last RGB record
    with time <= t, then requires a contiguous index window of length `window` ending there.
    """

    if window <= 0:
        return True
    if not rgb_times:
        return False
    idx = bisect_right(rgb_times, float(t)) - 1
    if idx < 0:
        return False
    start = idx - int(window) + 1
    return start >= 0


def build_takeover_split(
    *,
    dataset_root: Path,
    out_dir: Path,
    auto_threshold: float = 0.8,
    min_t: float = 21.0,
    min_rgb_history_window: int = 21,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a PoC takeover split using `planner_candidates.auto_enabled` as proxy labels.

    Positives: auto_enabled=False snapshots in mostly-auto episodes.
    Negatives: auto_enabled=True snapshots.

    Writes:
    - takeover_split.jsonl: balanced 50/50 (sampled deterministically by seed)
    - takeover_split_all.jsonl: unbalanced / natural prevalence (all eligible snapshots)
    - summary.json
    """

    dataset_root = Path(dataset_root).resolve()
    out_dir = Path(out_dir).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    _ensure_empty_dir(out_dir, overwrite=bool(overwrite))

    metas = find_episode_metadata_files(dataset_root)
    if not metas:
        raise ValueError(f"No episodes found under dataset: {dataset_root}")

    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    episodes_loaded = 0
    episodes_with_auto = 0
    episodes_selected = 0

    for meta in tqdm(metas, total=len(metas), desc="takeover_split:episodes"):
        ep = load_episode(dataset_root, meta)
        if not ep.schema_valid or ep.episode is None:
            skipped.append(
                {
                    "episode_id": ep.episode_id,
                    "episode_meta_path": ep.episode_meta_path,
                    "reason": "schema_invalid",
                    "details": ep.schema_errors,
                }
            )
            continue

        episodes_loaded += 1
        auto_records_all = [r for r in ep.planner_candidates if r.auto_enabled is not None]
        if not auto_records_all:
            skipped.append(
                {
                    "episode_id": ep.episode_id,
                    "episode_meta_path": ep.episode_meta_path,
                    "reason": "missing_auto_enabled",
                    "details": ep.planner_candidates_errors,
                }
            )
            continue

        auto_true = sum(1 for r in auto_records_all if bool(r.auto_enabled))
        auto_false = len(auto_records_all) - auto_true
        auto_ratio = float(auto_true) / float(len(auto_records_all)) if auto_records_all else 0.0
        episodes_with_auto += 1

        selected = auto_ratio >= float(auto_threshold)

        # Filter snapshots used for split construction.
        # (Episode selection remains based on the full episode ratio.)
        rgb_times = [float(rr.t) for rr in ep.rgb] if ep.rgb else []
        eligible_auto_records = []
        filtered_min_t = 0
        filtered_rgb_history = 0
        for r in auto_records_all:
            tt = float(r.t)
            if tt < float(min_t):
                filtered_min_t += 1
                continue
            if not _has_rgb_history(rgb_times=rgb_times, t=tt, window=int(min_rgb_history_window)):
                filtered_rgb_history += 1
                continue
            eligible_auto_records.append(r)

        if selected:
            episodes_selected += 1
            for r in eligible_auto_records:
                rec = {"episode_id": ep.episode_id, "t": float(r.t)}
                if bool(r.auto_enabled):
                    negatives.append(rec)
                else:
                    positives.append(rec)

        episode_stats.append(
            {
                "episode_id": ep.episode_id,
                "episode_meta_path": ep.episode_meta_path,
                "auto_total": int(len(auto_records_all)),
                "auto_true": int(auto_true),
                "auto_false": int(auto_false),
                "auto_ratio": float(auto_ratio),
                "selected": bool(selected),
                "eligible_total": int(len(eligible_auto_records)),
                "filtered_min_t": int(filtered_min_t),
                "filtered_rgb_history": int(filtered_rgb_history),
                "planner_candidates_errors": ep.planner_candidates_errors,
            }
        )

    pos_sample, neg_sample = _sample_balanced(positives, negatives, seed=int(seed))
    for rec in pos_sample:
        rec["label"] = 1
    for rec in neg_sample:
        rec["label"] = 0

    labeled = sorted(pos_sample + neg_sample, key=lambda r: (r["episode_id"], r["t"], r["label"]))

    # Balanced split (50/50).
    split_balanced_path = out_dir / "takeover_split.jsonl"
    with split_balanced_path.open("w", encoding="utf-8") as f:
        for rec in labeled:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    # Unbalanced split (natural prevalence): all eligible snapshots from selected episodes.
    all_labeled: list[dict[str, Any]] = []
    for rec in positives:
        all_labeled.append({"episode_id": rec["episode_id"], "t": float(rec["t"]), "label": 1})
    for rec in negatives:
        all_labeled.append({"episode_id": rec["episode_id"], "t": float(rec["t"]), "label": 0})
    all_labeled.sort(key=lambda r: (r["episode_id"], r["t"], r["label"]))
    split_all_path = out_dir / "takeover_split_all.jsonl"
    with split_all_path.open("w", encoding="utf-8") as f:
        for rec in all_labeled:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    summary = {
        "dataset": str(dataset_root),
        "out_dir": str(out_dir),
        "auto_threshold": float(auto_threshold),
        "min_t": float(min_t),
        "min_rgb_history_window": int(min_rgb_history_window),
        "seed": int(seed),
        "episodes_total": int(len(metas)),
        "episodes_loaded": int(episodes_loaded),
        "episodes_with_auto_enabled": int(episodes_with_auto),
        "episodes_selected": int(episodes_selected),
        "positives_total": int(len(positives)),
        "negatives_total": int(len(negatives)),
        "positives_sampled": int(len(pos_sample)),
        "negatives_sampled": int(len(neg_sample)),
        "balanced_split_path": str(split_balanced_path),
        "unbalanced_split_path": str(split_all_path),
        "unbalanced_total": int(len(all_labeled)),
        "label_mapping": {"takeover": 1, "auto": 0},
        "episodes": episode_stats,
        "skipped": skipped,
    }
    write_json(out_dir / "summary.json", summary)
    return summary
