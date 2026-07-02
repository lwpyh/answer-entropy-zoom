#!/bin/bash
# The working directory for the job is
# the current directory by default in
# Slurm
#SBATCH -n 8 # (or --ntasks=1) # Request 8 core
#SBATCH -t 1:0:0  # Request 1 hour runtime
#SBATCH --mem-per-cpu=1G   # Request 1GB RAM

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0 
module load cuda/12.2.2-gcc-12.2.0
module load openssl/3.3.0-gcc-12.2.0

# cd verltool
# git submodule update --init --recursive
# mamba create --name VideoZoomer python=3.11 -y

python -m pip install --user -U huggingface_hub
export PATH=$HOME/.local/bin:$PATH

mamba activate VideoZoomer
huggingface-cli login --token YOUR_HF_TOKEN_HERE
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export DECORD_EOF_RETRY_MAX=20480 
export HF_HOME="/data/home/acw652/.cache/huggingface"
export NCCL_DEBUG=INFO  # 启用详细的NCCL调试信息
export CUDA_LAUNCH_BLOCKING=1
set -x

# cd VideoZoomer
# pip3 install -r requirements.txt
# pip3 install -e .
# pip3 install httpx==0.23.3
# pip install -U huggingface_hub
# hf download LongVideo-Reason/longvideo-reason \
#     --repo-type dataset \
#     --local-dir ./longvideo-reason
