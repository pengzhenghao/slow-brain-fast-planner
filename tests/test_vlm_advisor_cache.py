from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from slow_brain_fast_planner.agents.vlm_advisors import (
    CachedVlmAdvisor,
    VlmAdviceCache,
    VlmCacheMissError,
    query_bundle_hash,
)


@dataclass
class _DummyBackend:
    calls: int = 0
    last_raw_response: str | None = None
    last_messages: list[dict[str, Any]] | None = None
    last_thoughts: list[str] | None = None

    def advise(self, query_bundle: Mapping[str, Any]) -> dict[str, Any]:
        self.calls += 1
        # Deterministic "advice" for tests.
        idx = int(query_bundle.get("num_candidates") or 1) - 1
        if idx < 0:
            idx = 0
        self.last_messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        self.last_raw_response = f"selected_index={idx}"
        return {"type": "select_trajectory", "selected_index": idx, "confidence": 0.0}


def test_query_bundle_hash_is_order_invariant(tmp_path: Path) -> None:
    qb1 = {"a": 1, "b": {"x": 2, "y": 3}}
    qb2 = {"b": {"y": 3, "x": 2}, "a": 1}
    h1 = query_bundle_hash(qb1, base_dir=tmp_path)
    h2 = query_bundle_hash(qb2, base_dir=tmp_path)
    assert h1 == h2


def test_query_bundle_hash_includes_image_bytes(tmp_path: Path) -> None:
    img = tmp_path / "overlay.png"
    img.write_bytes(b"abc")
    qb = {"overlay_frame_ref": "overlay.png", "num_candidates": 3}

    h1 = query_bundle_hash(qb, base_dir=tmp_path)
    img.write_bytes(b"abcd")  # change content, keep ref
    h2 = query_bundle_hash(qb, base_dir=tmp_path)
    assert h1 != h2


def test_cache_first_writes_and_replays(tmp_path: Path) -> None:
    cache_path = tmp_path / "vlm_cache.jsonl"
    cache = VlmAdviceCache(cache_path)
    backend = _DummyBackend()

    advisor = CachedVlmAdvisor(
        backend=backend,
        cache=cache,
        mode="cache_first",
        model_id="dummy",
        prompt_version="p1",
        decoding_config={"temperature": 0.0},
        base_dir=tmp_path,
    )

    img = tmp_path / "overlay.png"
    img.write_bytes(b"img")
    qb = {"overlay_frame_ref": "overlay.png", "num_candidates": 4}

    a1 = advisor.advise(qb)
    assert a1["type"] == "select_trajectory"
    assert backend.calls == 1
    assert advisor.last_cache_hit is False
    assert cache_path.exists()

    a2 = advisor.advise(qb)
    assert a2 == a1
    assert backend.calls == 1  # replayed
    assert advisor.last_cache_hit is True


def test_replay_only_requires_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "vlm_cache.jsonl"
    cache = VlmAdviceCache(cache_path)
    backend = _DummyBackend()

    # Prime cache with one entry via cache_first.
    prime = CachedVlmAdvisor(
        backend=backend,
        cache=cache,
        mode="cache_first",
        model_id="dummy",
        prompt_version="p1",
        decoding_config={"temperature": 0.0},
        base_dir=tmp_path,
    )
    (tmp_path / "overlay.png").write_bytes(b"img")
    qb = {"overlay_frame_ref": "overlay.png", "num_candidates": 2}
    _ = prime.advise(qb)

    # Replay should not call backend.
    replay_backend = _DummyBackend()
    replay = CachedVlmAdvisor(
        backend=replay_backend,
        cache=cache,
        mode="replay_only",
        model_id="dummy",
        prompt_version="p1",
        decoding_config={"temperature": 0.0},
        base_dir=tmp_path,
    )
    out = replay.advise(qb)
    assert out["type"] == "select_trajectory"
    assert replay_backend.calls == 0
    assert replay.last_cache_hit is True

    # Cache miss should raise.
    with pytest.raises(VlmCacheMissError):
        replay.advise({"overlay_frame_ref": "overlay.png", "num_candidates": 3})
