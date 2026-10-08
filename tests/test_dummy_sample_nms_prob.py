import json

import numpy as np

from slow_brain_fast_planner.benchmarks.model_adapters import DummyTrajectoryModelAdapter


def _call(adapter: DummyTrajectoryModelAdapter, obs: dict) -> dict:
    raw = adapter.call(messages=[], obs=obs)
    return json.loads(raw)


def test_dummy_sample_nms_prob_matches_numpy_choice() -> None:
    # Build a fake "planner" observation with 6 candidates and an NMS-kept subset.
    K = 6
    obs = {
        "planner": {
            "candidates": [{"score": float(i)} for i in range(K)],
            # NMS-kept candidates: original indices 4, 1, 0 with given probs.
            "candidate_confidence": [
                {"index": 4, "nms_prob": 0.7},
                {"index": 1, "nms_prob": 0.2},
                {"index": 0, "nms_prob": 0.1},
            ],
        }
    }

    seed = 123
    adapter = DummyTrajectoryModelAdapter(mode="sample_nms_prob", seed=seed)
    out = _call(adapter, obs)

    idxs = np.asarray([4, 1, 0], dtype=np.int64)
    p = np.asarray([0.7, 0.2, 0.1], dtype=np.float64)
    p = p / float(np.sum(p))
    expected = int(np.random.default_rng(seed).choice(idxs, p=p))

    assert out["action"] == "select_trajectory"
    assert int(out["selected_index"]) == expected


def test_dummy_sample_nms_prob_falls_back_to_uniform_when_missing_table() -> None:
    K = 5
    obs = {"planner": {"candidates": [{"score": 0.0} for _ in range(K)]}}

    seed = 7
    adapter = DummyTrajectoryModelAdapter(mode="sample_nms_prob", seed=seed)
    out = _call(adapter, obs)

    expected = int(np.random.default_rng(seed).integers(0, K))
    assert out["action"] == "select_trajectory"
    assert int(out["selected_index"]) == expected
