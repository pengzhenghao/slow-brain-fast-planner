from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_generator():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "trajectory_selection"
        / "build_static_candidate_set.py"
    )
    spec = importlib.util.spec_from_file_location("build_static_candidate_set", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = _load_generator()


def test_portable_provenance_path_drops_machine_prefix(tmp_path: Path) -> None:
    dataset = tmp_path / "hard"
    clips = dataset / "takeover_clips" / "takeover_clips.jsonl"
    clips.parent.mkdir(parents=True)
    clips.touch()

    assert generator._portable_provenance_path(clips, dataset_root=dataset) == (
        "hard/takeover_clips/takeover_clips.jsonl"
    )


def test_checked_in_candidate_assets_have_portable_provenance() -> None:
    root = (
        Path(__file__).resolve().parent.parent / "assets" / "trajectory_selection_static_candidates"
    )
    for path in root.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert not Path(str(payload["dataset"])).is_absolute()
        takeover_clips = payload.get("takeover_clips")
        if takeover_clips is not None:
            assert not Path(str(takeover_clips)).is_absolute()
