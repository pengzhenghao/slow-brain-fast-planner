from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slow_brain_fast_planner.benchmarks.data_loading import (
    build_trajectory_selection_jobs_from_takeover_clips,
)
from slow_brain_fast_planner.benchmarks.dataset import find_episode_metadata_files
from slow_brain_fast_planner.benchmarks.overlays import OverlayConfig
from slow_brain_fast_planner.benchmarks.planner_postprocessing import A0Config
from slow_brain_fast_planner.ingest.rss_human_adapter import (
    convert_rss_human_data_processed,
    write_trajectory_selection_eval_clips_one_per_episode,
)


def _make_rss_segment(root: Path) -> Path:
    segment = root / "scenario" / "time_1"
    segment.mkdir(parents=True)
    np.save(segment / "obs_255_HWC_RGB_seq.npy", np.zeros((21, 8, 8, 3), dtype=np.uint8))
    np.save(segment / "goal_Xforward_Yleft.npy", np.array([5.0, 0.0], dtype=np.float32))
    np.save(
        segment / "positive_behavior.npy",
        np.stack([np.linspace(0.2, 4.0, 20), np.zeros(20)], axis=1).astype(np.float32),
    )
    return segment


def test_rss_conversion_refuses_to_overwrite_existing_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _make_rss_segment(raw)
    out = tmp_path / "processed"

    report = convert_rss_human_data_processed(
        input_root=raw,
        output_dataset_dir=out,
        overwrite=True,
    )
    assert report.episodes_converted == 1

    sentinel = out / "episodes" / "scenario_time_1" / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        convert_rss_human_data_processed(
            input_root=raw,
            output_dataset_dir=out,
            overwrite=False,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_rss_eval_clips_are_label_free_and_honor_overwrite(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    episode = dataset / "episodes" / "ep"
    episode.mkdir(parents=True)
    (episode / "planner_candidates.jsonl").write_text('{"t": 4.0}\n', encoding="utf-8")

    summary = write_trajectory_selection_eval_clips_one_per_episode(
        dataset_root=dataset,
        overwrite=True,
    )
    clips_path = Path(summary["clips_path"])
    record = json.loads(clips_path.read_text(encoding="utf-8"))
    assert record["episode_id"] == "ep"
    assert "label_takeover_request" not in record
    assert "phase" not in record

    (episode / "episode.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "episode_id": "ep",
                "streams": {
                    "planner_candidates": {"records_ref": "planner_candidates.jsonl"},
                },
            }
        ),
        encoding="utf-8",
    )
    jobs = build_trajectory_selection_jobs_from_takeover_clips(
        dataset_path=dataset,
        episode_meta_paths=find_episode_metadata_files(dataset),
        takeover_clips_path=clips_path,
        out_dir=tmp_path / "out",
        a0_cfg=A0Config(),
        overlay_cfg=OverlayConfig(),
        clip_label_filter="takeover_only",
        write_overlays=False,
        compute_gt_metrics=False,
    )
    assert len(jobs) == 1

    original = clips_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_trajectory_selection_eval_clips_one_per_episode(
            dataset_root=dataset,
            overwrite=False,
        )
    assert clips_path.read_text(encoding="utf-8") == original
