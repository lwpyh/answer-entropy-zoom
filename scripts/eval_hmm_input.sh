#!/bin/bash
# =============================================================================
# eval_hmm_input.sh  ─  Input hidden state HMM zoom trigger (two-phase)
#
# Phase 1: Extract per-frame visual embeddings from vision encoder (1 GPU)
#          Fit two GaussianHMMs (zoom vs nozoom), save artifacts
# Phase 2: Run inference with HMM pre-generation zoom gate (2 GPU)
# =============================================================================

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

set -x

MODEL_PATH="zsgvivo/videozoomer"
DATA_PATH="/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
EMBED_CACHE="$(pwd)/input_decoder_multilayer_embeddings.npz"
ANALYSIS_OUT="$(pwd)/analysis_input_hmm_multilayer.json"
OUTPUT_DIR="$(pwd)/infer_results/hmm_input_multilayer"

echo "============================================================"
echo "  DATA       : $DATA_PATH"
echo "  OUTPUT     : $OUTPUT_DIR"
echo "  EMBED CACHE: $EMBED_CACHE"
echo "============================================================"

mamba activate VideoZoomer

pip install -q scikit-learn hmmlearn scipy

cd /data/DERI-Gong/jh015/VideoZoomer

# ── Phase 1: Extract frame embeddings + fit HMMs ──────────────────────────── #
echo "=== Phase 1: Vision encoder embedding + HMM fitting ==="

# Check if cache already exists
if [ -f "$EMBED_CACHE" ]; then
    echo "  Embedding cache found, skipping vision encoder step (--no_model)"
    NO_MODEL_FLAG="--no_model"
else
    echo "  No cache found, running vision encoder on GPU"
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

echo "=== Phase 1 done. HMM artifacts at $ANALYSIS_OUT ==="

# ── Phase 2: Inference with input-HMM zoom gate ──────────────────────────── #
echo "=== Phase 2: HMM-input inference ==="
python3 main_infer_hmm_input.py \
    --data_path    "$DATA_PATH"      \
    --video_root   "$VIDEO_ROOT"     \
    --model_path   "$MODEL_PATH"     \
    --output_dir   "$OUTPUT_DIR"     \
    --hmm_analysis "$ANALYSIS_OUT"   \
    --gpu_memory_utilization 0.45    \
    --tensor_parallel_size   2       \
    --embed_device cuda:0            \
    --fps          0.5               \
    --batch_size   32

echo "============================================================"
echo "  Done. Results at $OUTPUT_DIR"
echo "============================================================"
