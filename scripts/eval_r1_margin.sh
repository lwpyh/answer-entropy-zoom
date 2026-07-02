#!/bin/bash
# =============================================================================
# eval_r1_margin.sh
#
# Round-1 margin-gated zoom inference on LongVideoReasoning.
#
# Round 1: notool-style (stop=</answer> only), greedy + logprobs.
#          Model always produces a full answer. Compute answer-token margin.
#   margin >= θ  →  done (single pass)
#   margin <  θ  →  append Round 1 output, inject reconsider turn → zoom rounds
#
# Round 2+: zoom-enabled continuation (NOT a restart).
#           margin checked after each answer; stops when confident or max_rounds.
#
# Output: infer_results/r1_margin/results_r1margin.jsonl
#
# Usage:
#   sbatch scripts/eval_r1_margin.sh
#   MARGIN_THRESHOLD=1.5 sbatch scripts/eval_r1_margin.sh
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

MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/r1_margin}"

MARGIN_THRESHOLD="${MARGIN_THRESHOLD:-2.0}"
LOGPROBS_K="${LOGPROBS_K:-5}"

echo "============================================================"
echo "  DATA             : ${DATA_PATH}"
echo "  MODEL            : ${MODEL_PATH}"
echo "  OUTPUT           : ${OUTPUT_DIR}"
echo "  margin_threshold : ${MARGIN_THRESHOLD}"
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_r1_margin.py \
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
    --margin_threshold           "${MARGIN_THRESHOLD}"   \
    --logprobs_k                 "${LOGPROBS_K}"         \
    \
    --batch_size                 64
