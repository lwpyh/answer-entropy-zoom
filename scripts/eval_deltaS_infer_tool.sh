#!/bin/bash
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 1:0:0
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:2
#SBATCH --exclude=sbg2,ddg1,ddg2  # exclude V100 nodes (CC 7.0, 16 GB)

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

DATA_PATH="/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_deltaS.yaml"
MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/eval_deltaS_tool}"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_tool.py \
    --data_path              "${DATA_PATH}" \
    --model_path             "${MODEL_PATH}" \
    --video_root             "${VIDEO_ROOT}" \
    --output_dir             "${OUTPUT_DIR}" \
    --gpu_memory_utilization 0.7 \
    --tensor_parallel_size   2 \
    --max_model_len          32768 \
    --max_tokens             4096 \
    --temperature            0.0 \
    --max_pixels             100352 \
    --min_pixels             25088 \
    --fps                    0.5 \
    --frames_upbound         64 \
    --max_generation_round   5 \
    --tool_call_max_frames   16 \
    --limit_mm_per_prompt    128 \
    --tool_workers           8 \
    --batch_size             8
