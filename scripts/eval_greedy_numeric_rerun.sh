#!/bin/bash
# Submit greedy baseline re-run on numeric subset only
# Usage: bash scripts/eval_greedy_numeric_rerun.sh

cd /data/DERI-Gong/jh015/VideoZoomer

sbatch \
  --export=ALL,\
DATA_PATH=/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_28_missing.yaml,\
OUTPUT_DIR=/data/DERI-Gong/jh015/VideoZoomer/infer_results/greedy_numeric_rerun \
  scripts/eval_greedy_baseline.sh
