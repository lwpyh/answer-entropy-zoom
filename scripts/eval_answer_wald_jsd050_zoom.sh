#!/bin/bash
# answer_entropy + Wald JSD stopping, wald_threshold=0.50
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

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_hmm_zoom.py \
    --data_path        /data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml \
    --model_path       zsgvivo/videozoomer \
    --video_root       /data/DERI-Gong/jh015/VideoZoomer \
    --output_dir       /data/DERI-Gong/jh015/VideoZoomer/infer_results/answer_wald_jsd050_zoom \
    --gpu_memory_utilization 0.7 --tensor_parallel_size 2 \
    --max_model_len 32768 --max_pixels 100352 --min_pixels 25088 \
    --fps 0.5 --frames_upbound 64 --max_tokens 4096 \
    --tool_limit_mm 128 --tool_max_frames_per_call 16 \
    --tool_workers 8 --max_rounds 5 \
    --score_mode answer_entropy \
    --entropy_threshold -0.30 \
    --answer_k 5 --answer_temperature 0.7 \
    --answer_wald \
    --wald_k 3 --wald_threshold 0.50 \
    --batch_size 32
