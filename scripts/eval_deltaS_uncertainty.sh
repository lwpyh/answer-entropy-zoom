#!/bin/bash
# The working directory for the job is
# the current directory by default in
# Slurm
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 1:0:0  # Request 1 hour runtime
#SBATCH --cpus-per-gpu=8     # 8 cores per GPU
#SBATCH --mem-per-cpu=12G  # 12G×8cpu=96GB; vLLM sleep mode offloads ~31GB to CPU + FSDP ~14GB + Ray 20GB = ~65GB peak
#SBATCH --gres=gpu:1 # request 1 GPU
#SBATCH --exclude=sbg2,ddg1,ddg2  # exclude V100 nodes (CC 7.0, 16 GB) - incompatible with flash-attn 2.8.2 + OOM

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
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail
set -x

DATA_PATH="/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_deltaS.yaml"
MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/uncertainty}"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_uncertainty.py \
    --data_path              "${DATA_PATH}" \
    --model_path             "${MODEL_PATH}" \
    --video_root             "${VIDEO_ROOT}" \
    --output_dir             "${OUTPUT_DIR}" \
    --gpu_memory_utilization 0.8 \
    --tensor_parallel_size   1 \
    --max_model_len          32768 \
    --max_tokens             2048 \
    --max_pixels             100352 \
    --min_pixels             12544 \
    --fps                    0.2 \
    --frames_upbound         120 \
    --n_samples              4 \
    --sample_temp            0.7 \
    --logprobs_k             20 \
    --batch_size             4
