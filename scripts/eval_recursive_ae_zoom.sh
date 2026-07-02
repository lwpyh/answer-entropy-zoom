#!/bin/bash
# =============================================================================
# eval_recursive_ae_zoom.sh — Recursive Answer Entropy (R1 + R2+ stopping)
#
# R1 trigger: answer_entropy (threshold=-0.30)
# R2+ stopping: Recursive Answer Entropy
#   After each zoom round, draw rae_k=3 samples (T=0.7) → compute H_t
#   Stop if:
#     (1) H_t < rae_low_threshold (0.15) → confident enough → stop
#     (2) H_t >= H_{t-1}                 → zoom not reducing uncertainty → stop
#
# More principled than Wald:
#   Wald stops when two rounds are "consistent" (even if consistently wrong)
#   RAE stops when the model is genuinely confident OR when zoom stops helping
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
OUTPUT_DIR="${OUTPUT_DIR:-/data/DERI-Gong/jh015/VideoZoomer/infer_results/recursive_ae_zoom}"

ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--0.30}"   # R1 trigger
ANSWER_K="${ANSWER_K:-5}"
ANSWER_TEMP="${ANSWER_TEMP:-0.7}"
RAE_K="${RAE_K:-3}"
RAE_LOW_THRESHOLD="${RAE_LOW_THRESHOLD:-0.15}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  === R1 trigger (answer_entropy) ==="
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}"
echo "  answer_k           : ${ANSWER_K}"
echo "  === R2+ stopping (recursive_ae) ==="
echo "  rae_k              : ${RAE_K}"
echo "  rae_low_threshold  : ${RAE_LOW_THRESHOLD}  (stop if H_t < threshold)"
echo "  also stop if H_t >= H_{t-1}  (zoom not reducing uncertainty)"
echo "  answer_temperature : ${ANSWER_TEMP}"
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
    --recursive_ae                                           \
    --rae_k                      "${RAE_K}"                  \
    --rae_low_threshold          "${RAE_LOW_THRESHOLD}"      \
    \
    --batch_size                 32
