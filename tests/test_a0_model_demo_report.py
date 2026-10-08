from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def test_eval_a0_model_demo_writes_traces_and_report(
    tmp_path: Path, minidataset_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_base = tmp_path / "runs"
    exp_name = "run_a0_model_demo"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_trajectory_selection.py",
            "--dataset",
            str(minidataset_path),
            "--planner-source",
            "prelogged",
            "--model",
            "dummy_argmax",
            "--output-dir",
            str(out_base),
            "--exp-name",
            exp_name,
            "--max-episodes",
            "1",
            "--max-snapshots-total",
            "8",
            "--write-report",
            "--overwrite",
        ],
        check=True,
        cwd=str(repo_root),
    )

    # Find the run directory created under out_base/exp_name/.
    runs_root = out_base / exp_name
    assert runs_root.exists()
    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()])
    assert run_dirs, "expected at least one run directory"
    out_dir = run_dirs[-1]

    assert (out_dir / "config.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "predictions.jsonl").exists()
    assert (out_dir / "traces" / "events.jsonl").exists()
    assert (out_dir / "report.html").exists()

    preds = _read_jsonl(out_dir / "predictions.jsonl")
    assert preds, "expected at least one prediction record"

    # Overlay paths (if present) should be relative to RUN_DIR and should exist.
    #
    # Note: overlays are not strictly required for non-VLM models, so missing RGB near t can produce
    # overlay_frame_ref=null while still evaluating the snapshot.
    overlay_refs = [p.get("overlay_frame_ref") for p in preds if not p.get("skipped")]
    overlay_refs2 = [ref for ref in overlay_refs if isinstance(ref, str)]
    for ref in overlay_refs2:
        assert (out_dir / Path(ref)).exists()

    events = _read_jsonl(out_dir / "traces" / "events.jsonl")
    types = {e.get("event_type") for e in events}
    assert {"obs", "model_call", "action", "metric"}.issubset(types)

    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "trajectory_selection" in html
    assert "Model raw output" in html
