from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from slow_brain_fast_planner.benchmarks.takeover_clips import (
    TakeoverClipsConfig,
    build_takeover_clips,
)
from slow_brain_fast_planner.ingest import convert_s2e_v2_folder


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
    auto_series: list[bool],
    K: int = 3,
    T: int = 5,
) -> Path:
    ep = root / episode_name
    ep.mkdir(parents=True, exist_ok=True)
    assert len(auto_series) == n_samples

    for i in range(n_samples):
        _write_minimal_png(ep / f"sample_{i}.png", width=64, height=48)

        traj = np.zeros((1, K, T, 2), dtype=np.float32)
        traj[0, :, :, 0] = float(i)
        traj[0, :, :, 1] = np.linspace(0.0, 1.0, T, dtype=np.float32)
        np.save(ep / f"sample_{i}_traj.npy", traj)

        score = np.linspace(0.0, 1.0, K, dtype=np.float32).reshape(1, K)
        np.save(ep / f"sample_{i}_score.npy", score)

        np.save(ep / f"sample_{i}_auto.npy", np.array(bool(auto_series[i])))
    return ep


def test_build_takeover_clips_predictive_horizon_2s(tmp_path: Path) -> None:
    # Timeline at dt=1s:
    # t=0,1 auto; t=2,3 takeover; t=4,5,6 auto
    # Query stride=2s -> t0 in {0,2,4,6} but last is skipped (insufficient future for 2s horizon).
    auto = [True, True, False, False, True, True, True]
    raw_ep = _make_s2e_v2_episode_dir(
        tmp_path, episode_name="ep", n_samples=len(auto), auto_series=auto
    )

    dataset = tmp_path / "canonical_dataset"
    convert_s2e_v2_folder(
        input_dir=raw_ep,
        output_dataset_dir=dataset,
        dt_s=1.0,
        copy_images=False,
        overwrite=True,
    )

    out_dir = tmp_path / "clips"
    cfg = TakeoverClipsConfig(
        history_window_s=4.0,
        stride_s=2.0,
        horizon_s=2.0,
        k_frames=4,
        debounce_on_count=1,
        debounce_off_count=1,
        post_window_s=4.0,
    )
    summary = build_takeover_clips(
        dataset_root=dataset, out_dir=out_dir, config=cfg, overwrite=True
    )
    assert summary["clips_written"] == 3  # t0=0,2,4 (t0=6 skipped)

    clips = [
        json.loads(line)
        for line in (out_dir / "takeover_clips.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    clips.sort(key=lambda r: float(r["t0"]))

    # Predictive horizon=2s:
    # - t0=0: window [0,2] includes takeover at t=2 -> True; phase pre
    # - t0=2: window [2,4] includes takeover at t=2,3 -> True; phase center
    # - t0=4: window [4,6] all auto -> False; phase post (recently ended at t=4)
    assert clips[0]["t0"] == 0.0
    assert clips[0]["label_takeover_request"] == 1
    assert clips[0]["phase"] == "pre"

    assert clips[1]["t0"] == 2.0
    assert clips[1]["label_takeover_request"] == 1
    assert clips[1]["phase"] == "center"

    assert clips[2]["t0"] == 4.0
    assert clips[2]["label_takeover_request"] == 0
    assert clips[2]["phase"] == "post"

    # Frames are history-only (<= t0) and K=4.
    for rec in clips:
        assert len(rec["frames"]) == 4
        assert all(float(fr["t"]) <= float(rec["t0"]) + 1e-9 for fr in rec["frames"])
