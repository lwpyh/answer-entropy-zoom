#!/bin/bash
# =============================================================================
# eval_question_prior_zoom.sh — Question-prior zoom trigger
#
# Key insight: entropy methods re-discover the model's own zoom decision
# (low entropy at end = model is generating zoom-request boilerplate).
# This method uses PRIOR features independent of the think chain:
#   (1) Question text keywords: detail/motion/fast → need zoom
#                               overall/summary/general → skip zoom
#   (2) Video duration:         >5min → more need for zoom
#                               <1min → less need for zoom
#   (3) Think chain length:     short think → model is confident → skip
#
# score = 0.5*kw_score + 0.8*dur_score + 0.4*think_score  (no logprobs)
# Higher score = prior suggests confident/easy → skip zoom.
#
# Threshold guide (based on entropy_zoom_v2 distribution):
#   threshold=0.7734  → skip  5%
#   threshold=0.6817  → skip 10% (comparable to hmm_zoom_v2)
#   threshold=0.3479  → skip 15%
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
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/question_prior_zoom}"

# threshold=0.6817 → skip ~10% (comparable to hmm_zoom_v2)
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-0.6817}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  score_mode         : question_prior"
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}"
echo "  (uses question keywords + video duration + think length)"
echo "  (NO logprobs — purely prior-based decision)"
echo "  (score > threshold → skip zoom)"
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
    --score_mode                 question_prior              \
    --entropy_threshold          "${ENTROPY_THRESHOLD}"      \
    \
    --batch_size                 32
