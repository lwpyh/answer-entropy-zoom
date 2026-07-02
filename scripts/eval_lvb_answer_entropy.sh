#!/bin/bash
# =============================================================================
# eval_lvb_answer_entropy.sh — Answer entropy zoom trigger on LongVideoBench-Val
#
# R1: draw k=5 samples (T=0.7) → H_answer → score=-H
#     score > threshold (-0.30) → skip zoom, use majority-vote answer
#     score ≤ threshold          → execute zoom → R2+
# 1337 questions, 4-5 option (A-E), multiple categories
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G
#SBATCH --job-name=lvb_entropy
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
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

set -euo pipefail
set -x

MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_lvb_val.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/DERI-Gong/jh015/VideoZoomer/infer_results/lvb_answer_entropy}"

ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--0.30}"
ANSWER_K="${ANSWER_K:-5}"
ANSWER_TEMP="${ANSWER_TEMP:-0.7}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  MODEL              : ${MODEL_PATH}"
echo "  TASK               : LongVideoBench-Val (1337 samples, A-E 4-5 option)"
echo "  score_mode         : answer_entropy"
echo "  answer_k           : ${ANSWER_K}"
echo "  answer_temperature : ${ANSWER_TEMP}"
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}"
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_hmm_zoom_lvb.py \
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
    --score_mode                 answer_entropy              \
    --entropy_threshold          "${ENTROPY_THRESHOLD}"      \
    --answer_k                   "${ANSWER_K}"               \
    --answer_temperature         "${ANSWER_TEMP}"            \
    \
    --batch_size                 32
