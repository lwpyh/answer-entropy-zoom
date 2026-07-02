#!/bin/bash
# =============================================================================
# eval_combined_zoom.sh — Combined zoom trigger (Direction 1)
#
# Combines three complementary signals via z-normalization:
#   keyword score     (weight=0.5): hmm_zoom_v2 rules
#   early_entropy     (weight=0.3): reasoning-part entropy (first 75% tokens)
#   question_prior    (weight=0.2): video duration + question keywords
#
# combined = 0.5*z(keyword) + 0.3*z(early_entropy) + 0.2*z(question_prior)
# Normalization constants from 1000-sample calibration:
#   keyword:       mean=-2.268, std=2.861
#   early_entropy: mean=-1.192, std=0.683
#   question_prior:mean=-0.271, std=0.648
#
# threshold=1.5 → skip ~10% (comparable to hmm_zoom_v2)
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
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/combined_zoom}"

# threshold=0.8374 → skip ~10% (comparable to hmm_zoom_v2)
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:-0.8374}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  score_mode         : combined"
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}  (→ skip ~10%)"
echo "  combined = 0.5*z(keyword) + 0.3*z(early_entropy) + 0.2*z(question_prior)"
echo "  score > threshold → skip zoom"
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
    --score_mode                 combined                    \
    --entropy_threshold          "${ENTROPY_THRESHOLD}"      \
    \
    --batch_size                 32
