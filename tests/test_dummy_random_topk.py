import json

from slow_brain_fast_planner.benchmarks.model_adapters import DummyTrajectoryModelAdapter


def _call(adapter: DummyTrajectoryModelAdapter, obs: dict) -> dict:
    raw = adapter.call(messages=[], obs=obs)
    return json.loads(raw)


def test_dummy_random_topk() -> None:
    # 10 candidates with increasing scores.
    K = 10
    obs = {"planner": {"candidates": [{"score": float(i)} for i in range(K)]}}

    seed = 42
    topk = 3
    # Top 3 indices are 9, 8, 7.
    adapter = DummyTrajectoryModelAdapter(mode="random_topk", seed=seed, topk=topk)

    # Run a few times to see if it always picks from {7, 8, 9}.
    for _ in range(10):
        out = _call(adapter, obs)
        assert out["action"] == "select_trajectory"
        assert out["selected_index"] in [7, 8, 9]


def test_dummy_random_topk_fallback() -> None:
    # Test with NaN scores.
    K = 5
    obs = {"planner": {"candidates": [{"score": float("nan")} for _ in range(K)]}}
    adapter = DummyTrajectoryModelAdapter(mode="random_topk", seed=0, topk=2)
    out = _call(adapter, obs)
    assert out["action"] == "select_trajectory"
    assert out["selected_index"] == 0
