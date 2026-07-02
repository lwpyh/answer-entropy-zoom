#!/bin/bash
#SBATCH -p gpushort
#SBATCH -t 1:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0

set -x

MODEL_PATH="zsgvivo/videozoomer"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
EMBED_CACHE="$(pwd)/input_decoder_multilayer_embeddings.npz"
ANALYSIS_OUT="$(pwd)/analysis_input_hmm_multilayer.json"

mamba activate VideoZoomer
pip install -q scikit-learn hmmlearn scipy

cd /data/DERI-Gong/jh015/VideoZoomer

if [ -f "$EMBED_CACHE" ]; then
    NO_MODEL_FLAG="--no_model"
else
    NO_MODEL_FLAG=""
fi

python3 analyze_input_hidden_states.py \
    --model_path     "$MODEL_PATH"     \
    --video_root     "$VIDEO_ROOT"     \
    --tv_results     "$(pwd)/infer_results/temp_vote/results_tvote.jsonl"       \
    --greedy_results "$(pwd)/infer_results/greedy_baseline/results_greedy.jsonl" \
    --output_path    "$ANALYSIS_OUT"   \
    --embed_cache    "$EMBED_CACHE"    \
    --n_components   4                 \
    --pca_dim        64                \
    --device         cuda              \
    --max_duration   3600              \
    $NO_MODEL_FLAG
