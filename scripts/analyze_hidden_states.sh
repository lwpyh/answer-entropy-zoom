#!/bin/bash
# =============================================================================
# analyze_hidden_states.sh  ─  Extract main-model hidden states for HMM analysis
#
# Uses the main VideoZoomer model's last-layer hidden states to:
#   1. Embed all segments of 551 NOTOOL_SYS think chains
#   2. K-means cluster into K=4 data-driven states
#   3. Estimate T_correct / T_incorrect transition matrices
#   4. Compare embedding-based vs keyword-based matrices
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 4:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G
#SBATCH --exclude=sbg2,ddg1,ddg2

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0

set -x

MODEL_PATH="zsgvivo/videozoomer"
TV_RESULTS="$(pwd)/infer_results/temp_vote/results_tvote.jsonl"
GREEDY_RESULTS="$(pwd)/infer_results/greedy_baseline/results_greedy.jsonl"
OUTPUT_PATH="$(pwd)/analysis_hmm_hidden_states.json"
EMBED_CACHE="$(pwd)/r1_segment_embeddings.npz"

echo "============================================================"
echo "  MODEL     : $MODEL_PATH"
echo "  OUTPUT    : $OUTPUT_PATH"
echo "  CACHE     : $EMBED_CACHE"
echo "============================================================"

mamba activate VideoZoomer

pip install -q scikit-learn

cd /data/DERI-Gong/jh015/VideoZoomer

python3 analyze_r1_hidden_states.py \
    --model_path    "$MODEL_PATH"     \
    --tv_results    "$TV_RESULTS"     \
    --greedy_results "$GREEDY_RESULTS" \
    --output_path   "$OUTPUT_PATH"    \
    --embed_cache   "$EMBED_CACHE"    \
    --n_states      4                 \
    --batch_size    32                \
    --max_seg_len   128               \
    --device        cuda

echo "============================================================"
echo "  Done. Results at $OUTPUT_PATH"
echo "============================================================"
