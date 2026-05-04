#!/bin/bash
# =============================================================================
# eval_qwen35_lvr_toolcall.sh — Qwen3.5-4B pure tool-call (no entropy gate)
#
# Replicates lmms-eval run_lvr_tool.sh (job 7976340) using vLLM batch
# inference for ~30× speedup (batch_size=32 vs batch_size=1).
#
# Parameters aligned to lmms-eval:
#   max_pixels=602112  min_pixels=200704  max_frames=128  enable_thinking
#   entropy_gate_k=0  (disabled — always execute tool)
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=qwen35_lvr_tool
#SBATCH --exclude=sbg2,ddg1,ddg2,rdg8,rdg9,xlg1

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

MODEL_PATH="${MODEL_PATH:-/data/home/acw652/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/LongVideoReason_test_fixed.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/DERI-Gong/infer_results/qwen35_lvr_toolcall}"

echo "============================================================"
echo "  Model   : ${MODEL_PATH}"
echo "  Data    : ${DATA_PATH}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  Mode    : pure tool-call (no entropy gate)"
echo "  Pixels  : max=602112  min=200704  frames=128  thinking=True"
echo "  Matches : lmms-eval run_lvr_tool.sh (job 7976340)"
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_qwen35_lvr.py \
    --data_path                 "${DATA_PATH}"          \
    --model_path                "${MODEL_PATH}"         \
    --video_root                "${VIDEO_ROOT}"         \
    --output_dir                "${OUTPUT_DIR}"         \
    \
    --gpu_memory_utilization    0.85                    \
    --tensor_parallel_size      1                       \
    --max_model_len             131072                   \
    \
    --max_pixels                602112                  \
    --min_pixels                200704                  \
    --max_frames                128                     \
    --fps                       0.5                     \
    --max_tokens                2048                    \
    --enable_thinking                                   \
    \
    --entropy_gate_k            0                       \
    \
    --batch_size                16

echo "=== Done: $(date) ==="
