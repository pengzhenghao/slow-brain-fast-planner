from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "relative_path",
    [
        "scripts/trajectory_selection/delayed_fusion_from_run.py",
        "scripts/trajectory_selection/oracle_topk_sweep.py",
        "scripts/trajectory_selection/regenerate_metrics.py",
        "scripts/trajectory_selection/render_prompt_videos.py",
    ],
)
def test_script_help_starts(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        [sys.executable, relative_path, "--help"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
