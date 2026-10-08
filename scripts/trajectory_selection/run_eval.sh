#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper around the trajectory-selection CLI.
#
# Example:
#   scripts/trajectory_selection/run_eval.sh --dataset data/processed --model dummy_argmax

python -m slow_brain_fast_planner.cli.trajectory_selection "$@"

