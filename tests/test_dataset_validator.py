from __future__ import annotations

from pathlib import Path

from slow_brain_fast_planner.validation import DatasetValidationConfig, validate_dataset


def test_validate_minidataset_counts(minidataset_path: Path) -> None:
    report = validate_dataset(
        minidataset_path,
        config=DatasetValidationConfig(
            horizon_s=5.0,
            time_tolerance_s=0.2,
            include_episode_details=False,
        ),
    )

    assert report["episode_count"] == 2

    # Stream completeness.
    assert report["missing_stream_counts"]["gps"] == 1
    assert report["missing_stream_counts"]["odom"] == 0
    assert report["missing_stream_counts"]["planner_candidates"] == 0
    assert report["missing_stream_counts"]["rgb"] == 0

    # Trajectory selection: 3 (ep_ok) + 2 (ep_partial)
    ts = report["tasks"]["trajectory_selection"]
    assert ts["snapshots_total"] == 5
    assert ts["snapshots_valid"] == 5
