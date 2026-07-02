#!/bin/bash
# =============================================================================
# eval_answer_token_entropy_zoom.sh — single-pass answer-distribution entropy
#
# Design: read p(A)/p(B)/p(C)/p(D) from top-20 logprobs at the answer token
# position of the greedy R1 output → H(A,B,C,D) → score = -H.
#
# Single-pass proxy for answer_entropy (k=5 sampling), ~5x faster:
#   answer_entropy      : 5 forward passes (T=0.7 sampling)
#   answer_token_entropy: 1 forward pass  (greedy, top-20 logprobs at answer pos)
#
# Threshold guide (score = -H, range ≈ [-log(4), 0]):
#   threshold = -0.10 → skip ~5%  (only when model is near-certain)
#   threshold = -0.30 → skip ~10-15% (comparable to answer_entropy default)
#   threshold = -0.50 → skip ~20%
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G
#SBATCH --exclude=sbg2,ddg1,ddg2

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0
module load openssl/3.3.0-gcc-12.2.0

mamba activate VideoZoomer

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"   # set your token here or via env
export HF_HOME="/data/home/acw652/.cache/huggingface"
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail
set -x

MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/answer_token_entropy_zoom}"

ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--0.30}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  score_mode         : answer_token_entropy"
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}"
echo "  Single-pass proxy for answer_entropy (~5x faster, no k-sample overhead)"
echo "  H(p(A),p(B),p(C),p(D)) from top-20 logprobs at answer token position"
echo "  score=-H > threshold → skip zoom (confident); else → execute zoom"
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_hmm_zoom.py \
    --data_path                  "${DATA_PATH}"              \
    --model_path                 "${MODEL_PATH}"             \
    --video_root                 "${VIDEO_ROOT}"             \
    --output_dir                 "${OUTPUT_DIR}"             \
    \
    --gpu_memory_utilization     0.7                         \
    --tensor_parallel_size       1                           \
    --max_model_len              32768                       \
    --max_pixels                 100352                      \
    --min_pixels                 25088                       \
    \
    --fps                        0.5                         \
    --frames_upbound             64                          \
    --max_tokens                 4096                        \
    --tool_limit_mm              128                         \
    --tool_max_frames_per_call   16                          \
    --tool_workers               8                           \
    --max_rounds                 5                           \
    \
    --score_mode                 answer_token_entropy        \
    --entropy_threshold          "${ENTROPY_THRESHOLD}"      \
    --iter_ate                                               \
    \
    --batch_size                 32
