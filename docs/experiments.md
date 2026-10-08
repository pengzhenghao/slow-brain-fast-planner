# Experiment reference

See the [README](../README.md) for installation and quickstart. Run commands below from the repository root.

## Canonical dataset format (overview)

A dataset root contains episode folders:

```
<dataset_root>/
  episodes/<episode_id>/
    episode.json
    rgb.jsonl
    planner_candidates.jsonl
    odom.jsonl                  # optional (enables GT/oracle metrics like ADE/FDE)
    route.jsonl                 # optional (for route metrics, if present)
```

To build datasets from raw RSS-style episode folders:
- **Real logs → canonical**: `python scripts/process_data.py --raw ... --processed ...`

## Trajectory selection benchmark

### Single-process

If your dataset already contains `planner_candidates.jsonl`, you can run the CLI directly:

```bash
python -m slow_brain_fast_planner.cli.trajectory_selection \
  --dataset data/slow-brain-fast-planner/mini \
  --model dummy_argmax \
  --write-report
```

Common selector options:
- `dummy_argmax` (baseline)
- `oracle_min_ade` (oracle; requires `odom.jsonl`)
- `gemini_genai` (VLM; requires `GEMINI_API_KEY`)
- `gemini_genai_hierarchical` / `gemini_genai_chain_of_planning` (the same hierarchical selector under two names)

To reproduce the hard-split planner baseline from the paper without creating overlays:

```bash
python scripts/download_dataset.py --split hard
python -m slow_brain_fast_planner.cli.trajectory_selection \
  --dataset data/slow-brain-fast-planner/hard \
  --model dummy_argmax \
  --planner-source-report prelogged \
  --no-write-overlays \
  --write-report
```

Expected evaluation count: 1,412. The paper baseline is approximately `ade_score_avg=1.643778`, with `ade_min_avg=1.309979` after display-candidate filtering and `ade_min_all_avg=0.389871` over all 64 raw candidates.

A full hard-split Gemini run makes roughly 1,412 API calls. Start with `mini/`, then use sharding for the full run.

The paper’s best open-loop configuration used the historical `gemini-3-flash` endpoint, the 18 highest-scored raw candidates, no history frames, and no goal or planner-score cues. The historical endpoint was unavailable at the last reproduction check; the example below uses `gemini-3-flash-preview`. Check your provider for current model availability:

```bash
python scripts/run_trajectory_selection.py \
  --dataset data/slow-brain-fast-planner/hard \
  --planner-source prelogged \
  --model gemini_genai \
  --gemini-model gemini-3-flash-preview \
  --overlay-candidate-set raw_topk \
  --overlay-topk 18 \
  --no-overlay-goal-direction-arrow \
  --no-overlay-goal-marker \
  --no-overlay-goal-text \
  --no-prompt-show-scores \
  --no-prompt-goal-info \
  --no-prompt-goal-geometry \
  --prompt-history-frames 0 \
  --write-report
```

Use the sharded launcher for this full run. The preview endpoint is useful for rerunning the method, but it is not bit-for-bit equivalent to the historical model used for the submitted numbers.

If you need to generate candidates via an ONNX planner first (requires planner weights, which are not shipped with this repo — every dataset split already includes `planner_candidates.jsonl`, so `--planner-source prelogged` covers all shipped evaluations), use the wrapper:

```bash
python scripts/run_trajectory_selection.py \
  --dataset data/processed \
  --planner-source onnx \
  --model dummy_argmax
```

### Sharded (multi-process) runs

The benchmark supports **episode-level sharding**. For a single machine:

```bash
DATASET=data/slow-brain-fast-planner/hard SHARDS=8 OUT_DIR=logs EXP_NAME=trajsel_debug MERGE=1 \
  scripts/trajectory_selection/run_eval_sharded_local.sh -- \
    --model dummy_argmax --num-workers 4 --write-report
```

This creates:
- `logs/trajsel_debug/trajsel_debug_<timestamp>/shards/shard0/ ... shardN/` (per-shard run dirs)
- `logs/trajsel_debug/trajsel_debug_<timestamp>/merged/` (optional merged run dir + report)

To merge later:

```bash
python scripts/trajectory_selection/merge_shards.py logs/trajsel_debug/trajsel_debug_<timestamp> \
  --out logs/trajsel_debug/trajsel_debug_<timestamp>/merged --overwrite --write-report
```

For multi-GPU launches (one process per GPU), see `scripts/launch_multi_gpu_sharded_eval.sh`.

### Static candidate sets / prompt visualization

- Build a static candidate set JSON:

```bash
python scripts/trajectory_selection/build_static_candidate_set.py --help
```

- Render VLM prompt videos:

```bash
python scripts/trajectory_selection/render_prompt_videos.py --help
```

## Closed-loop simulation

### Small synthetic smoke test

This path uses a fixed candidate library, a synthetic noisy scorer, and a delayed oracle. It does not call Gemini and does not simulate perception or obstacles.

```bash
scripts/closed_loop_sim/run_sim_delayed_score_fusion.sh \
  --tasks forward,left_turn \
  --controllers pure_pursuit \
  --policies local_only,score_fusion,prob_fusion \
  --delays-s 0,2 \
  --seeds 0 \
  --duration-s 8 \
  --out logs/toy_smoke.csv \
  --summary-out logs/toy_smoke.json
```

Valid toy task names are `forward`, `left_turn`, and `right_turn`.

### Paper closed-loop replay

The paper replay extracts reference paths from the hard split’s odometry, then evaluates a synthetic corrupted planner and delayed oracle under the published defaults: K=12, epsilon=0.3, score noise=1.0, 18-second reference windows, and at most one task per episode.

```bash
python scripts/download_dataset.py --split hard
bash scripts/closed_loop_sim/section5_closed_loop_experiments.sh all
bash scripts/closed_loop_sim/section5_closed_loop_experiments.sh figures
```

Step `1` runs the fixed-delay lambda sweep; step `2` runs the 0–5 second delay sweep. Results land under `logs/section5_closed_loop/`. The launcher accepts environment overrides listed by `--help`; for a quick plumbing check, use:

```bash
DATASET=data/slow-brain-fast-planner/mini \
NUM_WORKERS=1 MAX_EPISODES=1 MAX_TASKS_PER_EPISODE=1 \
STEP1_LAMBDAS=1.0 STEP1_DELAY=0.0 \
OUT_BASE=logs/section5_smoke \
bash scripts/closed_loop_sim/section5_closed_loop_experiments.sh 1
```

`notebooks/closed_loop_fusion_viz.ipynb` and `scripts/paper_artifacts/plot_section5_figures.py` consume the generated layout. The fusion implementation is `src/slow_brain_fast_planner/control/delayed_vlm_score_fusion.py`.

## Output structure

A single run directory contains:

```
<run_dir>/
  config.json
  metrics.json
  predictions.jsonl
  predictions.csv
  traces/events.jsonl
  report.html
  artifacts/               # overlays, prompt images, etc. (if enabled)
```

## Implementation map

### Trajectory selection

- `scripts/run_trajectory_selection.py` — public wrapper; optionally generates ONNX candidates before evaluation.
- `src/slow_brain_fast_planner/cli/trajectory_selection.py` — main benchmark CLI, metrics aggregation, traces, and run metadata.
- `src/slow_brain_fast_planner/benchmarks/data_loading.py` — clip/snapshot job construction and sharding.
- `src/slow_brain_fast_planner/benchmarks/trajectory_selection_workers.py` — per-snapshot preparation and evaluation.
- `src/slow_brain_fast_planner/benchmarks/model_adapters.py` — Gemini and OpenAI-compatible model clients.
- `src/slow_brain_fast_planner/agents/vlm_advisors.py` and `src/slow_brain_fast_planner/benchmarks/vqa_trajectory.py` — prompt construction, hierarchical selection, and response parsing.
- `src/slow_brain_fast_planner/benchmarks/planner_postprocessing.py` — top-k, NMS, and score/probability filtering.

Important metrics include `accuracy` (dataset label), `accuracy_vs_score` (planner argmax), `ade_model_avg`, `ade_score_avg`, `ade_min_avg` (best displayed candidate), and `ade_min_all_avg` (best of all raw candidates).

### Closed-loop fusion

The fusion implementation is in `src/slow_brain_fast_planner/control/delayed_vlm_score_fusion.py`, with tracking helpers in `src/slow_brain_fast_planner/control/tracking.py`:

```text
fused_score = planner_score
              + lambda * exp(-staleness / tau)
              * similarity(current_candidate, stale_vlm_choice)
```

The simulator intentionally uses a delayed oracle rather than a real VLM so it can isolate latency and fusion behavior without conflating perception errors.

### Data format and ingestion

- Schema: `src/slow_brain_fast_planner/schema/canonical_episode.py` (`0.1.0`).
- Validation: `src/slow_brain_fast_planner/validation/dataset_validator.py`.
- Ingestion: `scripts/process_data.py` and adapters under `src/slow_brain_fast_planner/ingest/`.
- Canonical datasets store episodes under `<root>/episodes/<episode_id>/`.

## Other directories

- `notebooks/` — Jupyter notebooks for analysis and paper figures (fusion visualizations, trajectory candidate studies).
- `vllm/` — a separate uv project that serves a Qwen2.5-VL model as an OpenAI-compatible endpoint for the OpenAI-compatible selectors (see `vllm/README.md`).
- `scripts/paper_artifacts/` — provenance scripts for paper-only figures and real-robot logs. Some require inputs not distributed with this release; read that directory’s README before use.

## Development checks

```bash
pytest -q
ruff check .
ruff format --check .
```

Development conventions:

- Invoke package CLIs as `python -m slow_brain_fast_planner.cli.<name>` or use the wrappers under `scripts/`.
- Keep repository-wide constants such as data rate, NMS defaults, camera parameters, and the default Gemini model in `src/slow_brain_fast_planner/constants.py`.
- A new selector must be added to the CLI choices, implemented as a model/advisor adapter, and included in run metadata construction.
- Tests create hermetic datasets under pytest `tmp_path`; the suite must not depend on untracked local fixtures.
- Ruff uses line length 100 and rules `E,F,I,UP,B`; prompt-heavy agent code has a targeted exclusion.
