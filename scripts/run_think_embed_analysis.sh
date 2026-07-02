#!/bin/bash
#SBATCH --job-name=think_embed_hmm
#SBATCH -p gpushort
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/think_embed_%j.out
#SBATCH --error=logs/think_embed_%j.err

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0

set -e
cd /data/DERI-Gong/jh015/VideoZoomer
mkdir -p logs think_embed_artifacts

mamba activate VideoZoomer
pip install -q scikit-learn hmmlearn scipy

echo "[job] Starting think-chain embedding HMM analysis"
echo "[job] Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
date

python3 analyze_r1_think_embed.py \
    --model_path   zsgvivo/videozoomer \
    --tv_results   infer_results/temp_vote/results_tvote.jsonl \
    --greedy_results infer_results/greedy_baseline/results_greedy.jsonl \
    --output_dir   think_embed_artifacts \
    --embed_cache  think_seg_embeddings.npz \
    --n_states     4 \
    --pca_dim      64 \
    --batch_size   64 \
    --max_seg_len  256 \
    --min_seg      2 \
    --device       cuda \
    --no_model

echo "[job] Done"
date
