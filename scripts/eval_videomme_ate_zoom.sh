#!/bin/bash
# =============================================================================
# eval_videomme_ate_zoom.sh — answer_token_entropy + iter_ate on VideoMME-Val
#
# Pipeline (mirrors job 12563232 on LongVideoReason, acc=78.6%, zoom=31.1%):
#   R1: TOOL_SYS greedy → stops at zoom call
#   ATE pass: R1 context + _FORCE_ANS_TURN → measure H(A,B,C,D) from logprobs
#   H low (score > -0.30) → use ATE answer, skip zoom  (2 passes)
#   H high                → execute zoom → R2+          (3+ passes)
#   iter_ate: re-check H after each subsequent zoom round
#   H=inf (model still calling zoom in ATE) → strongest uncertainty → always zoom
# 2700 samples, A-D standard options
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G
#SBATCH --job-name=vmme_ate
#SBATCH --exclude=sbg2,ddg1,ddg2

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0
module load openssl/3.3.0-gcc-12.2.0

mamba activate VideoZoomer

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_HOME="/data/home/acw652/.cache/huggingface"
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0,1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4

set -euo pipefail
set -x

MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_videomme.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/videomme_ate_zoom}"

ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--0.30}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  TASK               : VideoMME-Val (2700 samples, A-D)"
echo "  score_mode         : answer_token_entropy"
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}"
echo "  Pipeline           : R1 → ATE forced-answer (H gate) → zoom if uncertain"
echo "  iter_ate           : re-check H after each zoom round"
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
    --score_mode                 answer_token_entropy        \
    --entropy_threshold          "${ENTROPY_THRESHOLD}"      \
    --iter_ate                                               \
    \
    --batch_size                 32
