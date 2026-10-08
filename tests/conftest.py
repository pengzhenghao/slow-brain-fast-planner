from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.fixture
def minidataset_path(tmp_path: Path) -> Path:
    dataset = tmp_path / "minidataset"
    episodes = dataset / "episodes"

    for episode_id, planner_times, include_gps in (
        ("ep_ok", [0.0, 1.0, 2.0], True),
        ("ep_partial", [0.0, 1.0], False),
    ):
        episode_dir = episodes / episode_id
        assets_dir = episode_dir / "assets"
        assets_dir.mkdir(parents=True)

        rgb_records = []
        for index, t in enumerate(planner_times):
            image_path = assets_dir / f"frame_{index}.png"
            Image.new("RGB", (64, 48), color=(20 + index, 40, 60)).save(image_path)
            rgb_records.append(
                {
                    "t": t,
                    "frame_ref": str(image_path.relative_to(episode_dir)),
                    "width": 64,
                    "height": 48,
                    "camera_id": "front",
                }
            )

        odom_records = [
            {"t": float(t), "x": 0.2 * float(t), "y": 0.0, "yaw": 0.0, "frame": "map_enu"}
            for t in range(9)
        ]
        candidates = [
            {
                "traj_id": str(candidate_index),
                "points_xy": [
                    [0.2 * float(step + 1), 0.1 * float(candidate_index)] for step in range(20)
                ],
                "score": float(1 - candidate_index),
            }
            for candidate_index in range(2)
        ]
        planner_records = [
            {
                "t": t,
                "frame": "robot",
                "auto_enabled": True,
                "candidates": candidates,
                "goal_xy": [5.0, 0.0],
            }
            for t in planner_times
        ]

        _write_jsonl(episode_dir / "rgb.jsonl", rgb_records)
        _write_jsonl(episode_dir / "odom.jsonl", odom_records)
        _write_jsonl(episode_dir / "planner_candidates.jsonl", planner_records)
        streams = {
            "rgb": {"records_ref": "rgb.jsonl"},
            "odom": {"records_ref": "odom.jsonl"},
            "planner_candidates": {"records_ref": "planner_candidates.jsonl"},
        }
        if include_gps:
            _write_jsonl(
                episode_dir / "gps.jsonl",
                [{"t": 0.0, "lat": 0.0, "lon": 0.0, "accuracy_m": 1.0}],
            )
            streams["gps"] = {"records_ref": "gps.jsonl"}

        (episode_dir / "episode.json").write_text(
            json.dumps(
                {
                    "schema_version": "0.1.0",
                    "episode_id": episode_id,
                    "streams": streams,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return dataset
