#!/bin/bash
# =============================================================================
# eval_error_detection.sh
#
# Run first-error detection signal analysis (v2).
#
# Modes:
#   MODE=full    (default) – all 4 phases: greedy, diverse, align-check,
#                            self-verify.  Best for signal discovery.
#   MODE=fast    – skip Phase 1c (U_align) and 1d (U_flip); only
#                  greedy+diverse.  Useful for quick iteration.
#   MODE=analyze – skip inference entirely; re-analyse existing JSONL.
#
# Examples:
#   sbatch eval_error_detection.sh               # full run
#   MODE=fast   sbatch eval_error_detection.sh   # no align/flip passes
#   MODE=analyze sbatch eval_error_detection.sh  # analysis-only
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 10:00:00
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

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_deltaS_v2.yaml"
MODE="${MODE:-full}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/error_detection_${MODE}}"

echo "============================================================"
echo "  DATA   : ${DATA_PATH}"
echo "  MODEL  : ${MODEL_PATH}"
echo "  OUTPUT : ${OUTPUT_DIR}"
echo "  MODE   : ${MODE}"
echo "============================================================"

# ── Build extra flags based on mode ────────────────────────────────────────
EXTRA_FLAGS=""
if [ "${MODE}" = "fast" ]; then
    EXTRA_FLAGS="--skip_align --skip_flip"
elif [ "${MODE}" = "analyze" ]; then
    EXTRA_FLAGS="--analyze_only"
fi

# ── Run ────────────────────────────────────────────────────────────────────
python /data/DERI-Gong/jh015/VideoZoomer/main_infer_error_detection.py \
    --data_path                 "${DATA_PATH}"        \
    --model_path                "${MODEL_PATH}"       \
    --video_root                "${VIDEO_ROOT}"       \
    --output_dir                "${OUTPUT_DIR}"       \
    \
    --gpu_memory_utilization    0.7                   \
    --tensor_parallel_size      2                     \
    --max_model_len             32768                 \
    --max_pixels                100352                \
    \
    --fps                       0.2                   \
    --min_pixels                12544                 \
    --frames_upbound            120                   \
    --max_tokens                2048                  \
    \
    --n_samples                 5                     \
    --sample_temp               0.7                   \
    --logprobs_k                20                    \
    --batch_size                16                    \
    ${EXTRA_FLAGS}
