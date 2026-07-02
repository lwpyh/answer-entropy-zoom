#!/bin/bash
# =============================================================================
# eval_qwen35_tool.sh — Answer-entropy-gated Qwen3.5-4B tool
#
# Architecture (generalises the VideoZoomer zoom-gate to an LM tool):
#   R1 : main VLM (TOOL_SYS, greedy) → direct answer OR zoom signal
#   If zoom signal → answer_entropy gate (k=5 samples):
#     score > threshold → SKIP tool, majority-vote answer
#     score ≤ threshold → CALL Qwen3.5-4B (video + temporal analysis prompt)
#   R2 : main VLM with Qwen3.5-4B analysis in <tool_response> → final answer
#
# GPU budget:
#   vLLM  @ 0.60 utilisation × 2×A40 (80 GB) ≈ 48 GB
#   Qwen3.5-4B bf16                           ≈  8 GB
#   Remaining for activations / CUDA kernels  ≈ 24 GB  ✓
#
# Baseline for comparison: slurm-6913190.out (answer_entropy_zoom, acc=0.800)
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
QWEN35_PATH="${QWEN35_PATH:-/data/home/acw652/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/DERI-Gong/infer_results/qwen35_tool}"

ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--0.30}"
ANSWER_K="${ANSWER_K:-5}"
ANSWER_TEMP="${ANSWER_TEMP:-0.7}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  main model         : ${MODEL_PATH}"
echo "  tool model         : ${QWEN35_PATH}"
echo "  score_mode         : answer_entropy → Qwen3.5-4B tool"
echo "  answer_k           : ${ANSWER_K}"
echo "  answer_temperature : ${ANSWER_TEMP}"
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}"
echo "  Baseline: answer_entropy_zoom acc=0.800 (slurm-6913190.out)"
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_qwen35_tool.py \
    --data_path                  "${DATA_PATH}"              \
    --model_path                 "${MODEL_PATH}"             \
    --video_root                 "${VIDEO_ROOT}"             \
    --output_dir                 "${OUTPUT_DIR}"             \
    --qwen35_path                "${QWEN35_PATH}"            \
    \
    --gpu_memory_utilization     0.60                        \
    --tensor_parallel_size       2                           \
    --max_model_len              32768                       \
    --max_pixels                 100352                      \
    --min_pixels                 25088                       \
    \
    --fps                        0.5                         \
    --frames_upbound             64                          \
    --max_tokens                 4096                        \
    \
    --entropy_threshold          "${ENTROPY_THRESHOLD}"      \
    --answer_k                   "${ANSWER_K}"               \
    --answer_temperature         "${ANSWER_TEMP}"            \
    --qwen35_max_tokens          512                         \
    \
    --batch_size                 16
