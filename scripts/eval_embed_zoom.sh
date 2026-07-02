#!/bin/bash
# =============================================================================
# eval_embed_zoom.sh  ─  Embedding-based zoom trigger (two-phase)
#
# Phase 1: Train LR classifier from cached segment embeddings
#          (uses analysis_hmm_hidden_states_lr_weights.json)
# Phase 2: Run inference with embedding-based zoom gate
#          (vLLM at gpu_util=0.45, embed model on cuda:0 with remaining memory)
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
OUTPUT_DIR="$(pwd)/infer_results/embed_zoom"
EMBED_CACHE="$(pwd)/r1_segment_embeddings.npz"
HS_OUTPUT="$(pwd)/analysis_hmm_hidden_states.json"
LR_WEIGHTS="$(pwd)/analysis_hmm_hidden_states_lr_weights.json"

echo "============================================================"
echo "  DATA     : $DATA_PATH"
echo "  OUTPUT   : $OUTPUT_DIR"
echo "  LR WTS   : $LR_WEIGHTS"
echo "============================================================"

mamba activate VideoZoomer

pip install -q scikit-learn

cd /data/DERI-Gong/jh015/VideoZoomer

# ── Phase 1: Train LR (reuse cached embeddings, no GPU needed) ─────────── #
echo "=== Phase 1: Training LR classifier ==="
python3 analyze_r1_hidden_states.py \
    --model_path     "$MODEL_PATH"  \
    --tv_results     "$(pwd)/infer_results/temp_vote/results_tvote.jsonl" \
    --greedy_results "$(pwd)/infer_results/greedy_baseline/results_greedy.jsonl" \
    --output_path    "$HS_OUTPUT"   \
    --embed_cache    "$EMBED_CACHE" \
    --n_states       4              \
    --no_model

echo "=== Phase 1 done. LR weights at $LR_WEIGHTS ==="

# ── Phase 2: Embedding-based zoom inference ─────────────────────────────── #
echo "=== Phase 2: Embedding-zoom inference ==="
python3 main_infer_embed_zoom.py \
    --data_path   "$DATA_PATH"   \
    --video_root  "$VIDEO_ROOT"  \
    --model_path  "$MODEL_PATH"  \
    --output_dir  "$OUTPUT_DIR"  \
    --lr_weights  "$LR_WEIGHTS"  \
    --gpu_memory_utilization 0.45 \
    --tensor_parallel_size   2    \
    --embed_device cuda:0         \
    --embed_batch  16             \
    --embed_max_len 256           \
    --fps          0.5            \
    --batch_size   32

echo "============================================================"
echo "  Done. Results at $OUTPUT_DIR"
echo "============================================================"
