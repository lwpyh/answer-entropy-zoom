#!/bin/bash
# Submit branch baseline re-run on numeric subset only
# Usage: bash scripts/eval_branch_numeric_rerun.sh

cd /data/DERI-Gong/jh015/VideoZoomer

sbatch \
  --export=ALL,\
DATA_PATH=/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_28_missing.yaml,\
OUTPUT_DIR=/data/DERI-Gong/jh015/VideoZoomer/infer_results/branch_numeric_rerun \
  scripts/eval_branch_baseline.sh
