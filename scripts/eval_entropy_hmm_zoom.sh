#!/bin/bash
# =============================================================================
# eval_entropy_hmm_zoom.sh
#
# Step 1: fit_entropy_hmm.py — fit GaussianHMM(K=2) on entropy_zoom_v2 sequences
# Step 2: run inference with entropy_hmm score mode
#
# Score = -1.5*last_H - 0.6*trend + 0.8*C_ratio + 0.4*ends_in_C + 0.5*last_q_C
# Threshold ~ p85 of score distribution (skip ~15%, same as hmm_zoom_v2)
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

cd /data/DERI-Gong/jh015/VideoZoomer

MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/entropy_hmm_zoom}"
HMM_MODEL="$(pwd)/entropy_hmm_artifacts/hmm_k2.pkl"
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--1.8}"

# ── Step 1: Fit HMM (CPU-only, fast) ────────────────────────────────────── #
echo "============================================================"
echo "  Step 1: Fitting GaussianHMM(K=2) on entropy sequences ..."
echo "============================================================"
python fit_entropy_hmm.py

# ── Step 2: Inference ────────────────────────────────────────────────────── #
echo "============================================================"
echo "  Step 2: entropy_hmm inference"
echo "  OUTPUT    : ${OUTPUT_DIR}"
echo "  HMM_MODEL : ${HMM_MODEL}"
echo "  threshold : ${ENTROPY_THRESHOLD}"
echo "  (hmm_score > threshold → skip zoom)"
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
    --score_mode                 entropy_hmm                 \
    --entropy_hmm_model          "${HMM_MODEL}"              \
    --entropy_threshold          "${ENTROPY_THRESHOLD}"      \
    \
    --batch_size                 32
