#!/bin/bash
# =============================================================================
# eval_iter_entropy_zoom.sh — Post-zoom iteration entropy control (Direction 2)
#
# Combines:
#   (1) R1 zoom trigger: keyword mode (hmm_zoom_v2, best single method 0.774)
#   (2) R2+ iteration gate: if entropy doesn't decrease between rounds,
#       zoom is not converging → force finalize immediately.
#
# Stopping rule:
#   delta_H = H_t - H_{t-1}
#   If delta_H >= iter_threshold → zoom not helping → stop
#   iter_threshold=0.0 = stop if entropy doesn't strictly decrease
#
# This is purely unsupervised and requires no GT labels.
# Works with any score_mode; here uses keyword for R1 trigger.
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
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/iter_entropy_zoom}"

HMM_THRESHOLD="${HMM_THRESHOLD:-1.7}"     # R1 trigger threshold (keyword mode)
ITER_THRESHOLD="${ITER_THRESHOLD:-0.0}"   # delta_H threshold for post-zoom stopping

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  score_mode         : keyword  (R1 zoom trigger)"
echo "  hmm_threshold      : ${HMM_THRESHOLD}"
echo "  iter_entropy       : enabled"
echo "  iter_threshold     : ${ITER_THRESHOLD}  (stop if H_t - H_{t-1} >= threshold)"
echo "  (entropy not decreasing = zoom not converging = force stop)"
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
    --score_mode                 keyword                     \
    --hmm_threshold              "${HMM_THRESHOLD}"          \
    --iter_entropy                                           \
    --iter_threshold             "${ITER_THRESHOLD}"         \
    \
    --batch_size                 32
