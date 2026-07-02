#!/bin/bash
# =============================================================================
# eval_ppl_zoom.sh  ─  Token-PPL zoom trigger (zero-shot, no GT needed)
#
# Design: TOOL_SYS R1 generates think chain with logprobs → compute mean
# log-prob of think tokens → high logprob = confident → skip zoom
#
# Threshold guide (mean logprob, roughly):
#   threshold=-1.5  → very selective skip (few samples skipped)
#   threshold=-1.8  → ~15% skip rate (default)
#   threshold=-2.2  → ~30% skip rate (aggressive)
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
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/ppl_zoom}"

PPL_THRESHOLD="${PPL_THRESHOLD:--1.8}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  score_mode         : ppl"
echo "  ppl_threshold      : ${PPL_THRESHOLD}"
echo "  (mean think logprob > threshold → skip zoom)"
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
    --score_mode                 ppl                         \
    --ppl_threshold              "${PPL_THRESHOLD}"          \
    \
    --batch_size                 32
