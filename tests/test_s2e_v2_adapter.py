from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from slow_brain_fast_planner.ingest import convert_s2e_v2_folder
from slow_brain_fast_planner.validation import DatasetValidationConfig, validate_dataset


def _write_minimal_png(path: Path, *, width: int = 64, height: int = 48) -> None:
    # The adapter reads only the first 24 bytes and checks the PNG signature + IHDR marker.
    sig = b"\x89PNG\r\n\x1a\n"
    length = (13).to_bytes(4, "big")  # IHDR chunk length (not validated by adapter)
    ihdr = b"IHDR"
    w = int(width).to_bytes(4, "big")
    h = int(height).to_bytes(4, "big")
    path.write_bytes(sig + length + ihdr + w + h)


def _make_s2e_v2_episode_dir(
    root: Path,
    *,
    episode_name: str,
    n_samples: int,
    K: int = 3,
    T: int = 5,
    auto_series: list[bool] | None = None,
) -> Path:
    ep = root / episode_name
    ep.mkdir(parents=True, exist_ok=True)
    if auto_series is None:
        auto_series = [True] * n_samples
    assert len(auto_series) == n_samples

    for i in range(n_samples):
        _write_minimal_png(ep / f"sample_{i}.png", width=64, height=48)

        traj = np.zeros((1, K, T, 2), dtype=np.float32)
        # Make trajectories non-trivial but deterministic.
        traj[0, :, :, 0] = float(i)
        traj[0, :, :, 1] = np.linspace(0.0, 1.0, T, dtype=np.float32)
        np.save(ep / f"sample_{i}_traj.npy", traj)

        score = np.linspace(0.0, 1.0, K, dtype=np.float32).reshape(1, K)
        np.save(ep / f"sample_{i}_score.npy", score)

        np.save(ep / f"sample_{i}_auto.npy", np.array(bool(auto_series[i])))

        # Optional goal (1,2); presence shouldn't break.
        goal = np.array([[1.0, 2.0]], dtype=np.float32)
        np.save(ep / f"sample_{i}_goal.npy", goal)

    return ep


def test_convert_s2e_v2_folder_smoke_and_takeover_stats(tmp_path: Path) -> None:
    # Create a tiny synthetic episode (keeps repo tests hermetic).
    # auto series: T, F, F, T, F => false frames=3, takeover segments=2
    input_dir = _make_s2e_v2_episode_dir(
        tmp_path,
        episode_name="s2e_v2_test_ep",
        n_samples=5,
        auto_series=[True, False, False, True, False],
    )

    out_dataset = tmp_path / "canonical_dataset"
    report = convert_s2e_v2_folder(
        input_dir=input_dir,
        output_dataset_dir=out_dataset,
        dt_s=1.0,
        copy_images=False,
        overwrite=True,
    )

    assert report.samples_found == 5
    assert report.samples_converted == 5
    assert report.auto_enabled_true == 2
    assert report.auto_enabled_false == 3
    assert report.takeover_events == 2

    # Validate the produced dataset (schema allows extension fields).
    v = validate_dataset(out_dataset, config=DatasetValidationConfig(include_episode_details=True))
    assert v["episode_count"] == 1
    assert v["missing_stream_counts"]["rgb"] == 0
    assert v["missing_stream_counts"]["planner_candidates"] == 0

    # Ensure episode.json contains the stats block.
    ep_json = out_dataset / "episodes" / "s2e_v2_test_ep" / "episode.json"
    obj = json.loads(ep_json.read_text(encoding="utf-8"))
    assert obj["stats"]["auto_enabled_true"] == 2
    assert obj["stats"]["auto_enabled_false"] == 3
    assert obj["stats"]["takeover_events"] == 2


def test_convert_example_s2e_v2_root_folder(tmp_path: Path) -> None:
    root = tmp_path / "raw_root"
    ep1 = _make_s2e_v2_episode_dir(root, episode_name="s2e_v2_ep1", n_samples=3)
    ep2 = _make_s2e_v2_episode_dir(root, episode_name="s2e_v2_ep2", n_samples=4)

    episode_dirs = sorted([ep1, ep2], key=lambda p: p.name)
    out_dataset = tmp_path / "canonical_dataset"

    for ep_dir in episode_dirs:
        convert_s2e_v2_folder(
            input_dir=ep_dir,
            output_dataset_dir=out_dataset,
            dt_s=1.0,
            copy_images=False,
            overwrite=True,
            write_manifest=True,
        )

    v = validate_dataset(out_dataset, config=DatasetValidationConfig(include_episode_details=False))
    assert v["episode_count"] == len(episode_dirs)
