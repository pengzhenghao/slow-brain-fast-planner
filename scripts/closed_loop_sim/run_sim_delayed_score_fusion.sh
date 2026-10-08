#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the toy closed-loop simulation sweep.
#
# Example:
#   scripts/closed_loop_sim/run_sim_delayed_score_fusion.sh --out-csv logs/sim.csv

python scripts/closed_loop_sim/sim_delayed_score_fusion.py "$@"

