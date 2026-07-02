#!/bin/bash
#SBATCH -p gpushort
#SBATCH -A pilot_sae_gpu
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=qwen35_lvr_dbg
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
FULL_DATA="/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/LongVideoReason_test_fixed.json"
DEBUG_DATA="/tmp/lvr_debug_5.json"
OUTPUT_DIR="/data/DERI-Gong/infer_results/qwen35_lvr_debug"

# Create 5-sample subset
python -c "import json; d=json.load(open('$FULL_DATA')); json.dump(d[:5], open('$DEBUG_DATA','w'))"

echo "=== Qwen3.5 LVR debug (5 samples) | $(date) ==="

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_qwen35_lvr.py \
    --data_path     "$DEBUG_DATA"   \
    --model_path    "$MODEL_PATH"   \
    --output_dir    "$OUTPUT_DIR"   \
    --max_model_len 131072          \
    --max_pixels    602112          \
    --max_frames    128             \
    --max_tokens    2048            \
    --batch_size    5               \
    --entropy_gate_k 0              \
    --gpu_memory_utilization 0.85

echo "=== Done | node=$(hostname) | exit=$? | $(date) ==="
