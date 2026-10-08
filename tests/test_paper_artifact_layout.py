from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / "paper_artifacts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plot_section5_figures = _load_script("plot_section5_figures")
closed_loop_fusion_sweep_viz = _load_script("closed_loop_fusion_sweep_viz")


def _write_results(run_dir: Path, value: float) -> None:
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "policy": "local_only",
                "mean_cte_m": value,
                "p95_cte_m": value,
                "mean_speed_mps": 1.0,
                "chosen_switches": 0,
            }
        ]
    ).to_csv(run_dir / "results.csv", index=False)


def test_plot_loader_uses_latest_run_per_group(tmp_path: Path) -> None:
    _write_results(tmp_path / "group_a" / "closed_loop_fusion_20260101_000000", 1.0)
    _write_results(tmp_path / "group_a" / "closed_loop_fusion_20260102_000000", 2.0)
    _write_results(tmp_path / "group_b" / "closed_loop_fusion_20260101_000000", 3.0)

    frame = plot_section5_figures.load_results(tmp_path)
    assert sorted(frame["mean_cte_m"].tolist()) == [2.0, 3.0]


def test_lambda_discovery_accepts_interval_suffix(tmp_path: Path) -> None:
    run = (
        tmp_path
        / "step3_lambda_sweep_fusions"
        / "delay_2.0_interval_s_1.0"
        / "closed_loop_fusion_20260101_000000"
    )
    run.mkdir(parents=True)

    assert (
        closed_loop_fusion_sweep_viz._find_latest_delay_run(
            tmp_path / "step3_lambda_sweep_fusions",
            "2.0",
        )
        == run
    )
