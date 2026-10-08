# Slow Brain, Fast Planner

Official implementation of **[Slow Brain, Fast Planner: Latency-Resilient VLM-Augmented Urban Navigation](https://arxiv.org/abs/2606.20458)**.

Zhenghao “Mark” Peng, Honglin He, Quanyi Li, Yukai Ma, Bolei Zhou

[**Paper**](https://arxiv.org/abs/2606.20458) |
[**Code**](https://github.com/pengzhenghao/slow-brain-fast-planner) |
[**Dataset**](https://huggingface.co/datasets/pengzhenghao97/slow-brain-fast-planner-dataset) |
[**Webpage**](https://pengzhenghao.github.io/slow-brain-fast-planner/)

A slow vision-language model selects among a fast local planner’s candidate trajectories.
Score Fusion turns delayed VLM advice into a decaying score bonus on fresh candidates,
allowing the planner to keep running between VLM responses.

## Installation

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/pengzhenghao/slow-brain-fast-planner.git
cd slow-brain-fast-planner
uv sync
source .venv/bin/activate
```

Supports Python 3.11–3.13, Linux x86-64 with CUDA 12, and macOS arm64 with CPU inference.
The Python package is `slow_brain_fast_planner`.

## Quickstart

Run a small synthetic reference-tracking simulation without a dataset or API key:

```bash
bash scripts/closed_loop_sim/run_sim_delayed_score_fusion.sh \
  --tasks forward,left_turn \
  --controllers pure_pursuit \
  --policies local_only,score_fusion,prob_fusion \
  --delays-s 0,2 --seeds 0 --duration-s 8 \
  --out logs/toy_smoke.csv \
  --summary-out logs/toy_smoke.json
```

This example uses synthetic planner scoring and delayed-oracle advice to isolate
latency and fusion behavior. It does not simulate visual perception or obstacles.

## Dataset and trajectory selection

The [dataset](https://huggingface.co/datasets/pengzhenghao97/slow-brain-fast-planner-dataset)
contains the `mini` and `hard` splits below. The partner-fleet routine-scenario
pool used in the paper is not included.

| Split | Content | Size |
|---|---|---|
| `mini` | 1 episode, 3 clips | ~125 MB |
| `hard` | 32 episodes, 1,414 stored clips; 1,412 after GT-quality filtering | ~10.9 GB |

Download the mini split and run the planner baseline:

```bash
python scripts/download_dataset.py --split mini
python scripts/run_trajectory_selection.py \
  --dataset data/slow-brain-fast-planner/mini \
  --planner-source prelogged \
  --model dummy_argmax \
  --write-report
```

The run evaluates three clips and writes metrics, predictions, and `report.html`
under `logs/trajectory_selection/`.

To run the VLM selector, set `GEMINI_API_KEY` and replace `--model dummy_argmax`
with `--model gemini_genai`. The default is `gemini-2.5-flash-lite`;
`--gemini-model` selects another endpoint. This makes paid API calls.
OpenAI-compatible selectors use `OPENAI_API_KEY` and `--openai-model`.

The release includes offline trajectory selection, delayed-fusion simulation,
and experiment reporting. It excludes the on-robot stack, partner-fleet routine-scenario
data, and ONNX planner weights. Dataset evaluations use prelogged planner candidates.

See the [experiment reference](docs/experiments.md) for full-split evaluation,
sharding, simulation sweeps, data formats, and the implementation map.

## Development

```bash
pytest -q
ruff check .
ruff format --check .
```

When regenerating the lockfile on a machine with a custom uv package index,
use `UV_CONFIG_FILE=/dev/null uv lock`.

## Citation

```bibtex
@misc{peng2026slowbrainfastplanner,
  title={Slow Brain, Fast Planner: Latency-Resilient VLM-Augmented Urban Navigation},
  author={Zhenghao Peng and Honglin He and Quanyi Li and Yukai Ma and Bolei Zhou},
  year={2026},
  eprint={2606.20458},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2606.20458}
}
```
