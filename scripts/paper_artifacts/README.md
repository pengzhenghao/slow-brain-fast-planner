# Paper artifact scripts

These scripts reproduce specific figures, tables, and sweeps from the paper. They are **not** part of the public benchmark workflow and generally require inputs that are not shipped with the released dataset (real-robot telemetry, sweep logs, internal CSVs with machine-specific paths).

They are preserved here for provenance. Treat them as brittle: paths, column names, and log-directory conventions reflect the original authors' environment at the time the paper was submitted.

## Contents

Real-robot deployment (Section 5 / Table V):
- `real_world_metrics.py` — compute per-run metrics from on-robot telemetry logs.
- `real_world_make_paper_table.py` — aggregate per-run metrics into the main-paper policy-level table.
- `plot_real_world_bev_generate_all.py`, `plot_real_world_bev_head_to_head.py` — BEV visualizations for qualitative comparison.
- `analyze_realworld_vlm_latency.py` — VLM latency distribution plots.
- `benchmark_closed_loop_score_fusion_real.py` — replay the fusion policy offline against real-robot logs.

Closed-loop sim sweeps (Section 4):
- `sweep_closed_loop_fusion_lambda.py`, `sweep_closed_loop_fusion_tau.py` — hyperparameter sweeps over the score-fusion coefficients.
- `sweep_hard_limit_budget_ade.py` — ADE vs. compute-budget sweep.
- `closed_loop_fusion_sweep_viz.py` — plotting helpers for the sweep outputs.
- `plot_delay_smoothness_analysis.py`, `make_closed_loop_delay_table.py`, `plot_section5_figures.py` — figure generators.
- `benchmark_controller_tracking.py` — tracking-error sanity benchmark for the pure-pursuit controller.

Legacy debug / one-off:
- `analyze_anchor_override.py`, `aggregate_debug_videos.py`, `debug_video_match.py`.

## Supported workflow

The supported, reproducible workflow lives in the top-level `scripts/` directory and is documented in `README.md`:
- `scripts/run_trajectory_selection.py` — open-loop trajectory selection benchmark.
- `scripts/closed_loop_sim/sim_delayed_score_fusion.py` — closed-loop score-fusion simulator.
- `scripts/trajectory_selection/` — sharded eval launchers and shard merging.
- `scripts/process_data.py` — raw logs → canonical dataset ingestion.
