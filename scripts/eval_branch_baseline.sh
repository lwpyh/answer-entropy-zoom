#!/bin/bash
# =============================================================================
# eval_branch_baseline.sh
#
# Step-level uncertainty branching inference on LongVideoReasoning.
#
# Pass-1: greedy (T=0) with logprobs -> detect highest-entropy step k
# Pass-2: if max window entropy > ent_threshold, branch n times from step k
#         and apply decision logic (tool-trigger / answer vote / main)
#
# Output: infer_results/branch_baseline/results_branch.jsonl
#
# Usage:
#   sbatch scripts/eval_branch_baseline.sh
#   ENT_THRESHOLD=1.0 N_BRANCHES=3 sbatch scripts/eval_branch_baseline.sh
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G
#SBATCH --exclude=sbg2,ddg1,ddg2

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0
module load openssl/3.3.0-gcc-12.2.0

mamba activate VideoZoomer

export HUGGING_FACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export HF_HOME="/data/home/acw652/.cache/huggingface"
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0,1

set -euo pipefail
set -x

# ── Config (override via env vars) ────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/branch_baseline}"

# Branching hyperparams
# ent_threshold: start at 0.8; inspect branch_entropy in output JSONL to tune
# n_branches=2 cheapest; 3 gives more stable vote
ENT_THRESHOLD="${ENT_THRESHOLD:-0.8}"
N_BRANCHES="${N_BRANCHES:-2}"
BRANCH_TEMP="${BRANCH_TEMP:-0.5}"
WINDOW_SIZE="${WINDOW_SIZE:-16}"
LOGPROBS_K="${LOGPROBS_K:-20}"          # vLLM v1 hard cap: max 20

echo "============================================================"
echo "  DATA          : ${DATA_PATH}"
echo "  MODEL         : ${MODEL_PATH}"
echo "  OUTPUT        : ${OUTPUT_DIR}"
echo "  ent_threshold : ${ENT_THRESHOLD}  n_branches: ${N_BRANCHES}"
echo "  branch_temp   : ${BRANCH_TEMP}    window: ${WINDOW_SIZE}"
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_branch.py \
    --data_path                  "${DATA_PATH}"          \
    --model_path                 "${MODEL_PATH}"         \
    --video_root                 "${VIDEO_ROOT}"         \
    --output_dir                 "${OUTPUT_DIR}"         \
    \
    --gpu_memory_utilization     0.7                     \
    --tensor_parallel_size       2                       \
    --max_model_len              32768                   \
    --max_pixels                 100352                  \
    --min_pixels                 25088                   \
    \
    --fps                        0.5                     \
    --frames_upbound             64                      \
    --max_tokens                 4096                    \
    --tool_limit_mm              128                     \
    --tool_max_frames_per_call   16                      \
    --tool_workers               8                       \
    --max_rounds                 5                       \
    \
    --ent_threshold              "${ENT_THRESHOLD}"      \
    --n_branches                 "${N_BRANCHES}"         \
    --branch_temp                "${BRANCH_TEMP}"        \
    --window_size                "${WINDOW_SIZE}"        \
    --logprobs_k                 "${LOGPROBS_K}"         \
    \
    --batch_size                 64
