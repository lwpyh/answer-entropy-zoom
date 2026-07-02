#!/bin/bash
# =============================================================================
# eval_videomme_answer_entropy_128f.sh — Answer entropy zoom on VideoMME, 128 frames
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
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_videomme.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/DERI-Gong/jh015/VideoZoomer/infer_results/videomme_answer_entropy_128f}"

ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--0.30}"
ANSWER_K="${ANSWER_K:-5}"
ANSWER_TEMP="${ANSWER_TEMP:-0.7}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  score_mode         : answer_entropy"
echo "  answer_k           : ${ANSWER_K}"
echo "  answer_temperature : ${ANSWER_TEMP}"
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}"
echo "  frames_upbound     : 128"
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
    --frames_upbound             128                         \
    --max_tokens                 4096                        \
    --tool_limit_mm              256                         \
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
