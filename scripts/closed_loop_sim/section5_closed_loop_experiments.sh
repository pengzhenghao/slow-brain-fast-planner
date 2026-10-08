#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Section 5: Closed-Loop Simulation Experiments
# =============================================================================
#
# Purpose: Validate VLM fusion mechanisms under realistic latency conditions.
#
# Minimal workflow (what we actually run):
#   1. FIXED DELAY (delay=2s): sweep λ for fusion-based methods (4 policies)
#   2. DELAY SWEEP: using chosen best λ values, sweep delay for all 9 methods
#
# Key Design Choices (fixed for main experiments):
#   - Controller: pure_pursuit (user preference)
#   - Similarity: body_pointwise_horizon_aware (best from prior analysis)
#   - τ (tau): 5.0s (final)
#   - Similarity distance scale: 0.3 (final)
#   - Static candidates: K=12 k-means medoids
#   - Planner noise: epsilon=0.3, noise_std=1.0, temp=1.0
#   - Reference window: 18s, at most one task per episode
#
# Policies (9 methods):
#   - local_only: Noisy planner argmax (no VLM)
#   - vlm_hold: Directly execute delayed VLM-selected trajectory (hold: single in-flight)
#   - vlm_stream: Directly execute delayed VLM-selected trajectory (stream: fixed-cadence querying)
#   - vlm_hold_match: Execute current planner candidate closest to delayed VLM trajectory (hold querying)
#   - vlm_stream_match: Same as above, but stream-query (fixed-cadence querying)
#   - score_fusion: Fuse planner score + VLM similarity (hold querying)
#   - prob_fusion: Fuse in probability space (hold querying)
#   - score_fusion_stream: Score fusion + stream-query (fixed cadence + pipelined by default)
#   - prob_fusion_stream: Prob fusion + stream-query (fixed cadence + pipelined by default)
#
# =============================================================================

cd "$(dirname "$0")/../.."

STEP="${1:-help}"

# -----------------------------------------------------------------------------
# Common Configuration
# -----------------------------------------------------------------------------
DATASET="${DATASET:-data/slow-brain-fast-planner/hard}"
CANDIDATES="${CANDIDATES:-assets/trajectory_selection_static_candidates/takeover_kmeans_medoids/static_candidates_k12.json}"
OUT_BASE="${OUT_BASE:-logs/section5_closed_loop}"

# Fixed settings
CONTROLLER="${CONTROLLER:-pure_pursuit}"
SIMILARITY_MODE="${SIMILARITY_MODE:-body_pointwise_horizon_aware}"
TAU="${TAU:-5.0}"
DIST_SCALE_M="${DIST_SCALE_M:-0.3}"
# VLM query cadence (preferred): seconds per query.
# Backward-compat: if VLM_QUERY_INTERVAL_S is not set but VLM_QUERY_HZ is set, we convert Hz -> seconds.
VLM_QUERY_INTERVAL_S="${VLM_QUERY_INTERVAL_S:-}"
if [[ -z "${VLM_QUERY_INTERVAL_S}" ]]; then
  if [[ -n "${VLM_QUERY_HZ:-}" ]]; then
    VLM_QUERY_INTERVAL_S="$(python - <<'PY'
import os
hz = float(os.environ.get("VLM_QUERY_HZ", "1.0"))
print(1.0 / hz if hz > 1e-9 else 0.0)
PY
)"
  else
    VLM_QUERY_INTERVAL_S="1.0"
  fi
fi
# IMPORTANT: VLM request pipeline model
# - DO NOT force this to 1 for `vlm_stream*` experiments, otherwise your effective update rate becomes
#   delay-limited (~ 1/delay) and the query-interval sweep no longer measures "streaming frequency".
# - Default (=0): use per-policy defaults implemented in the simulator:
#     - non-stream: single in-flight (request again only after the previous returns)
#       (vlm_hold, vlm_hold_match, score_fusion, prob_fusion, local_only)
#     - stream: pipelined (unlimited in-flight)
#       (vlm_stream, vlm_stream_match, score_fusion_stream, prob_fusion_stream)
# - If you want to emulate a system that cannot pipeline requests, set VLM_MAX_INFLIGHT=1 explicitly.
VLM_MAX_INFLIGHT="${VLM_MAX_INFLIGHT:-0}"
VLM_MISTAKE_PROB="${VLM_MISTAKE_PROB:-0.0}"

# Planner noise model (single representative setting)
EPSILON="${EPSILON:-0.3}"
NOISE_STD="${NOISE_STD:-1.0}"
TEMPERATURE="${TEMPERATURE:-1.0}"

# Task sampling
TASK_SAMPLING="${TASK_SAMPLING:-uniform_per_episode}"
NUM_WORKERS="${NUM_WORKERS:-16}"
TASK_DURATION_S="${TASK_DURATION_S:-18.0}"
MIN_EPISODE_S="${MIN_EPISODE_S:-18.0}"
MAX_EPISODES="${MAX_EPISODES:-100}"
MAX_TASKS_PER_EPISODE="${MAX_TASKS_PER_EPISODE:-1}"

# Example visualization
WRITE_EXAMPLES="${WRITE_EXAMPLES:-0}"
EXAMPLES_N="${EXAMPLES_N:-6}"
EXAMPLE_ARGS=""
if [[ "${WRITE_EXAMPLES}" == "1" ]]; then
  EXAMPLE_ARGS="--write-examples --examples-n ${EXAMPLES_N} --examples-by-episode"
fi

# -----------------------------------------------------------------------------
# Helper function
# -----------------------------------------------------------------------------
run_experiment() {
  local name="$1"
  local policies="$2"
  local delays="$3"
  local lambdas="$4"
  local extra_args="${5:-}"
  
  echo ""
  echo "========================================"
  echo "Running: ${name}"
  echo "  Policies: ${policies}"
  echo "  Delays: ${delays}"
  echo "  Lambdas: ${lambdas}"
  echo "========================================"
  echo ""
  
  python scripts/paper_artifacts/benchmark_closed_loop_score_fusion_real.py \
    --dataset "${DATASET}" \
    --static-candidates-json "${CANDIDATES}" \
    --controller "${CONTROLLER}" \
    --vlm-query-interval-s "${VLM_QUERY_INTERVAL_S}" \
    --vlm-max-inflight "${VLM_MAX_INFLIGHT}" \
    --vlm-mistake-prob "${VLM_MISTAKE_PROB}" \
    --task-sampling "${TASK_SAMPLING}" \
    --duration-s "${TASK_DURATION_S}" \
    --min-episode-s "${MIN_EPISODE_S}" \
    --max-episodes "${MAX_EPISODES}" \
    --epsilons "${EPSILON}" \
    --noise-stds "${NOISE_STD}" \
    --score-temps "${TEMPERATURE}" \
    --policies "${policies}" \
    --similarity-modes "${SIMILARITY_MODE}" \
    --lambda-sims "${lambdas}" \
    --taus "${TAU}" \
    --delay-s "${delays}" \
    --dist-scales "${DIST_SCALE_M}" \
    --num-workers "${NUM_WORKERS}" \
    --max-tasks-per-episode "${MAX_TASKS_PER_EPISODE}" \
    ${EXAMPLE_ARGS} \
    ${extra_args} \
    --out "${OUT_BASE}/${name}"
}

# -----------------------------------------------------------------------------
# STEP 1: Lambda sweep at fixed delay (fusion-only, 4 methods)
# -----------------------------------------------------------------------------
# Fix delay=2s, sweep lambda for fusion-based methods:
#   score_fusion, prob_fusion, score_fusion_stream, prob_fusion_stream
#
# Output layout is kept under step3_* for compatibility with existing analysis tooling.

if [[ "${STEP}" == "1" || "${STEP}" == "all" ]]; then
  echo "=== STEP 1: Lambda sweep @ fixed delay (fusion-only, 4 methods) ==="
  STEP1_DELAY="${STEP1_DELAY:-2.0}"
  # Include the chosen finals by default: λ_score=1, λ_prob=3.
  STEP1_LAMBDAS="${STEP1_LAMBDAS:-0.1,0.5,1.0,2.0,3.0,5.0,10.0}"
  run_experiment \
    "step3_lambda_sweep_fusions/delay_${STEP1_DELAY}_interval_s_${VLM_QUERY_INTERVAL_S}" \
    "score_fusion,prob_fusion,score_fusion_stream,prob_fusion_stream" \
    "${STEP1_DELAY}" \
    "${STEP1_LAMBDAS}"
fi

# -----------------------------------------------------------------------------
# STEP 2: Delay sweep @ best lambda (all 9 methods)
# -----------------------------------------------------------------------------
# Choose best lambdas from STEP 1, then sweep delay over all methods.
# NOTE: `benchmark_closed_loop_score_fusion_real.py` cannot take per-policy lambdas in one call,
# so we run baselines together + each fusion policy separately (still under step4 layout).

if [[ "${STEP}" == "2" || "${STEP}" == "all" ]]; then
  echo "=== STEP 2: Delay sweep @ best lambdas (all 9 methods) ==="
  # Final decisions:
  BEST_LAMBDA_SCORE_FUSION="${BEST_LAMBDA_SCORE_FUSION:-1.0}"
  BEST_LAMBDA_PROB_FUSION="${BEST_LAMBDA_PROB_FUSION:-3.0}"
  BEST_LAMBDA_SCORE_FUSION_STREAM="${BEST_LAMBDA_SCORE_FUSION_STREAM:-${BEST_LAMBDA_SCORE_FUSION}}"
  BEST_LAMBDA_PROB_FUSION_STREAM="${BEST_LAMBDA_PROB_FUSION_STREAM:-${BEST_LAMBDA_PROB_FUSION}}"
  # Single query cadence for all stream-query policies during the delay sweep.
  BEST_VLM_QUERY_INTERVAL_S="${BEST_VLM_QUERY_INTERVAL_S:-${VLM_QUERY_INTERVAL_S}}"
  STEP2_DELAYS="${STEP2_DELAYS:-0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0}"

  echo "  Using knobs:"
  echo "    TAU=${TAU}"
  echo "    DIST_SCALE_M=${DIST_SCALE_M}"
  echo "    BEST_VLM_QUERY_INTERVAL_S=${BEST_VLM_QUERY_INTERVAL_S}"
  echo "    BEST_LAMBDA_SCORE_FUSION=${BEST_LAMBDA_SCORE_FUSION}"
  echo "    BEST_LAMBDA_PROB_FUSION=${BEST_LAMBDA_PROB_FUSION}"
  echo "    BEST_LAMBDA_SCORE_FUSION_STREAM=${BEST_LAMBDA_SCORE_FUSION_STREAM}"
  echo "    BEST_LAMBDA_PROB_FUSION_STREAM=${BEST_LAMBDA_PROB_FUSION_STREAM}"

  for delay in ${STEP2_DELAYS}; do
    # Baselines in one run (no lambdas)
    run_experiment \
      "step4_main_results/delay_${delay}/baselines" \
      "planner_oracle,local_only,vlm_hold,vlm_stream,vlm_hold_match,vlm_stream_match" \
      "${delay}" \
      "0.0" \
      "--vlm-query-interval-s ${BEST_VLM_QUERY_INTERVAL_S}"

    # Fusion policies run separately so each can use its own chosen lambda
    run_experiment \
      "step4_main_results/delay_${delay}/score_fusion" \
      "score_fusion" \
      "${delay}" \
      "${BEST_LAMBDA_SCORE_FUSION}"

    run_experiment \
      "step4_main_results/delay_${delay}/prob_fusion" \
      "prob_fusion" \
      "${delay}" \
      "${BEST_LAMBDA_PROB_FUSION}"

    run_experiment \
      "step4_main_results/delay_${delay}/score_fusion_stream" \
      "score_fusion_stream" \
      "${delay}" \
      "${BEST_LAMBDA_SCORE_FUSION_STREAM}" \
      "--vlm-query-interval-s ${BEST_VLM_QUERY_INTERVAL_S}"

    run_experiment \
      "step4_main_results/delay_${delay}/prob_fusion_stream" \
      "prob_fusion_stream" \
      "${delay}" \
      "${BEST_LAMBDA_PROB_FUSION_STREAM}" \
      "--vlm-query-interval-s ${BEST_VLM_QUERY_INTERVAL_S}"
  done
fi

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
if [[ "${STEP}" == "figures" ]]; then
  echo "=== Figures: Generate plots ==="
  FIGURE_DELAY="${FIGURE_DELAY:-2.0}"
  # NOTE: Prefer the notebook (`notebooks/closed_loop_fusion_viz.ipynb`) for paper-ready figures.
  python scripts/paper_artifacts/plot_section5_figures.py \
    --step1-dir "${OUT_BASE}/step4_main_results/delay_${FIGURE_DELAY}" \
    --step2-dir "${OUT_BASE}/step4_main_results" \
    --out "${OUT_BASE}/paper_figures"
  
  echo ""
  echo "Figures saved to: ${OUT_BASE}/paper_figures/"
fi

# -----------------------------------------------------------------------------
# Help
# -----------------------------------------------------------------------------
if [[ "${STEP}" == "help" || "${STEP}" == "-h" || "${STEP}" == "--help" ]]; then
  cat << 'HELP'
Section 5: Closed-Loop Simulation Experiments

Usage:
  bash scripts/closed_loop_sim/section5_closed_loop_experiments.sh <step>

Steps:
  1       - Lambda sweep at fixed delay (fusion-only)
  2       - Delay sweep using chosen best lambdas (main results)
  all     - Run (1) then (2)
  figures - Convenience plotting script (prefer the notebook)

Final decisions (baked in by default):
  TAU=5.0
  DIST_SCALE_M=0.3
  BEST_LAMBDA_SCORE_FUSION=1.0
  BEST_LAMBDA_PROB_FUSION=3.0

Key env vars (optional overrides):
  DATASET=data/slow-brain-fast-planner/hard
  CANDIDATES=assets/.../static_candidates_k12.json
  OUT_BASE=logs/section5_closed_loop
  EPSILON=0.3
  NOISE_STD=1.0
  VLM_MISTAKE_PROB=0.0
  VLM_QUERY_INTERVAL_S=1.0            seconds/query (preferred)
  VLM_QUERY_HZ=<legacy>              only used if VLM_QUERY_INTERVAL_S is unset
  VLM_MAX_INFLIGHT=0                 0 => policy defaults
  TASK_DURATION_S=18.0
  MIN_EPISODE_S=18.0
  MAX_EPISODES=100
  MAX_TASKS_PER_EPISODE=1
  STEP2_DELAYS="0.0 0.5 1.0 2.0"      override delay sweep
  FIGURE_DELAY=2.0                      fixed-delay policy comparison

HELP
fi
