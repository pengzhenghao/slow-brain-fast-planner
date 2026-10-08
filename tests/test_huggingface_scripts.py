from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub


def _load_script(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


download_dataset = _load_script("download_dataset")
upload_to_huggingface = _load_script("upload_to_huggingface")


def test_available_splits_reads_only_top_level(monkeypatch) -> None:
    calls = []

    class FakeApi:
        def list_repo_tree(self, repo_id, **kwargs):
            calls.append((repo_id, kwargs))
            return [
                SimpleNamespace(path="hard"),
                SimpleNamespace(path="mini"),
                SimpleNamespace(path="README.md"),
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: FakeApi())

    assert download_dataset._available_splits("owner/repo", None, "token") == ["mini", "hard"]
    assert calls == [
        (
            "owner/repo",
            {"repo_type": "dataset", "revision": None, "recursive": False},
        )
    ]


def test_large_upload_stage_rebuilds_when_source_changes(tmp_path: Path) -> None:
    src = tmp_path / "source"
    src.mkdir()
    (src / "a.txt").write_text("a", encoding="utf-8")

    stage = upload_to_huggingface._prepare_large_upload_stage(src, "hard", "owner/repo-a")
    assert (stage / "hard" / "a.txt").read_text(encoding="utf-8") == "a"

    (src / "b.txt").write_text("b", encoding="utf-8")
    stage = upload_to_huggingface._prepare_large_upload_stage(src, "hard", "owner/repo-a")
    assert sorted(path.name for path in (stage / "hard").iterdir()) == ["a.txt", "b.txt"]

    (src / "a.txt").unlink()
    stage = upload_to_huggingface._prepare_large_upload_stage(src, "hard", "owner/repo-a")
    assert sorted(path.name for path in (stage / "hard").iterdir()) == ["b.txt"]

    replacement = src / "replacement.txt"
    replacement.write_text("c", encoding="utf-8")
    replacement.replace(src / "b.txt")
    stage = upload_to_huggingface._prepare_large_upload_stage(src, "hard", "owner/repo-a")
    assert (stage / "hard" / "b.txt").read_text(encoding="utf-8") == "c"

    other_stage = upload_to_huggingface._prepare_large_upload_stage(
        src,
        "hard",
        "owner/repo-b",
    )
    assert other_stage != stage


def test_make_public_dry_run_does_not_call_api(monkeypatch) -> None:
    def fail(_repo_id: str) -> None:
        raise AssertionError("make_public should not run during a dry run")

    monkeypatch.setattr(upload_to_huggingface, "make_public", fail)
    assert upload_to_huggingface.main(["--make-public", "--dry-run"]) == 0
