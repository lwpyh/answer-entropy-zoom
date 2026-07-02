#!/bin/bash
# =============================================================================
# eval_iter_ae_zoom.sh — Iterative Answer Entropy (iter_ae)
#
# R1 trigger: answer_entropy (threshold=-0.30)
# R2+ gating: same answer_entropy check before EVERY zoom execution
#
# At each round, before executing a zoom call:
#   Draw k=5 force-answer samples (T=0.7) → compute H(P(A/B/C/D))
#   H < |threshold| → already confident → skip this zoom, return majority answer
#   H ≥ |threshold| → still uncertain → execute zoom → continue
#
# Key difference from recursive_ae / Wald:
#   Those methods check AFTER getting zoom results (stopping criteria).
#   iter_ae checks BEFORE executing zoom (pre-emptive gate, same as R1).
#   Uniform decision mechanism across ALL rounds.
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
OUTPUT_DIR="${OUTPUT_DIR:-/data/DERI-Gong/jh015/VideoZoomer/infer_results/iter_ae_zoom}"

ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--0.30}"
ANSWER_K="${ANSWER_K:-5}"
ANSWER_TEMP="${ANSWER_TEMP:-0.7}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  === R1 trigger (answer_entropy) ==="
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}"
echo "  answer_k           : ${ANSWER_K}"
echo "  answer_temperature : ${ANSWER_TEMP}"
echo "  === R2+ gating (iter_ae) ==="
echo "  Same answer_entropy check before every zoom in R2+"
echo "  H < |threshold| → skip zoom → majority answer"
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_hmm_zoom.py \
    --data_path                  "${DATA_PATH}"              \
    --model_path                 "${MODEL_PATH}"             \
    --video_root                 "${VIDEO_ROOT}"             \
    --output_dir                 "${OUTPUT_DIR}"             \
    \
    --gpu_memory_utilization     0.7                         \
    --tensor_parallel_size       2                           \
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
    --iter_ae                                                \
    \
    --batch_size                 32
