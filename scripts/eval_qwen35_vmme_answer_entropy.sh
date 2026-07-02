#!/bin/bash
# =============================================================================
# eval_qwen35_vmme_answer_entropy.sh — Qwen3.5-4B entropy-gated on VideoMME
# Mirrors eval_qwen35_lvr_answer_entropy.sh but on VideoMME (2700 samples)
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 108:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=qwen35_vmme_ent
#SBATCH --exclude=sbg2,ddg1,ddg2,rdg8,rdg9,xlg1

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0
module load openssl/3.3.0-gcc-12.2.0
mamba activate VideoZoomer

export HF_HOME="/data/home/acw652/.cache/huggingface"
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail
set -x

MODEL_PATH="/data/home/acw652/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATA_PATH="/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/VideoMME_val_fixed.json"
OUTPUT_DIR="/data/DERI-Gong/infer_results/qwen35_vmme_answer_entropy"

echo "============================================================"
echo "  Task    : VideoMME — entropy-gated tool-call"
echo "  Model   : Qwen3.5-4B"
echo "  Gate    : k=3, T=0.7, threshold=-0.30"
echo "  Data    : ${DATA_PATH}"
echo "  Output  : ${OUTPUT_DIR}"
echo "  Pixels  : max=602112  min=200704  frames=128  thinking=True"
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_qwen35_lvr.py \
    --data_path                 "${DATA_PATH}"   \
    --model_path                "${MODEL_PATH}"  \
    --output_dir                "${OUTPUT_DIR}"  \
    --max_model_len             131072           \
    --max_pixels                602112           \
    --min_pixels                200704           \
    --max_frames                128              \
    --max_tokens                2048             \
    --batch_size                32               \
    --entropy_gate_k            3               \
    --entropy_gate_threshold    -0.30            \
    --entropy_gate_temperature  0.7              \
    --gpu_memory_utilization    0.85

echo "=== Done | node=$(hostname) | exit=$? | $(date) ==="
