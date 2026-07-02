#!/bin/bash
# =============================================================================
# eval_adaptive_zoom.sh
#
# U_beam_ent-gated adaptive tool inference:
#   Round 1: no-tool beam (all samples), params aligned to eval_videommlu.sh
#   Round 2+: uncertain samples only → single tool beam (entropy+zoom in one pass) → gate
#   Max 5 outer rounds (default)
#
# Modes:
#   MODE=full    (default) – inference + analysis
#   MODE=analyze – skip inference; re-analyse existing JSONL (no GPU needed)
#
# Threshold sweep: run once at θ=0.5, then re-analyse at other thresholds:
#   MODE=analyze ENT_THRESHOLD=0.3 sbatch eval_adaptive_zoom.sh
#
# Examples:
#   sbatch scripts/eval_adaptive_zoom.sh                          # full run
#   MODE=analyze sbatch scripts/eval_adaptive_zoom.sh             # re-analyse
#   ENT_THRESHOLD=0.3 sbatch scripts/eval_adaptive_zoom.sh        # lower gate
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
module load openssl/3.3.0-gcc-12.2.0

mamba activate VideoZoomer

export HUGGING_FACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export HF_HOME="/data/home/acw652/.cache/huggingface"
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0,1

set -euo pipefail
set -x
# pip install vllm==0.9.2

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml}"
REF_JSONL="${REF_JSONL:-/data/DERI-Gong/jh015/VideoZoomer/infer_results/error_detection_full/results_error_detection.jsonl}"
MODE="${MODE:-full}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/adaptive_zoom}"
ENT_THRESHOLD="${ENT_THRESHOLD:-0.3}"   # U_beam_ent gate: samples ≥ θ get tool round
MAX_ROUNDS="${MAX_ROUNDS:-5}"

echo "============================================================"
echo "  DATA          : ${DATA_PATH}"
echo "  MODEL         : ${MODEL_PATH}"
echo "  REF           : ${REF_JSONL}"
echo "  OUTPUT        : ${OUTPUT_DIR}"
echo "  MODE          : ${MODE}"
echo "  ENT_THRESHOLD : ${ENT_THRESHOLD}  (θ — gate for tool activation)"
echo "  MAX_ROUNDS    : ${MAX_ROUNDS}"
echo "============================================================"

EXTRA_FLAGS=""
if [ "${MODE}" = "analyze" ]; then
    EXTRA_FLAGS="--analyze_only"
fi
# Add --r2_continue_from_r1 to seed Round-2 context with Round-1 reasoning.
# Compare:
#   default (R2 fresh start):      sbatch eval_adaptive_zoom.sh
#   R2 continues from R1:          R2_CONTINUE_FROM_R1=1 sbatch eval_adaptive_zoom.sh
if [ "${R2_CONTINUE_FROM_R1:-0}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --r2_continue_from_r1"
fi
if [ "${R1_TOOL_SYS:-0}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --r1_tool_sys"
fi

# ── Run ───────────────────────────────────────────────────────────────────────
python /data/DERI-Gong/jh015/VideoZoomer/main_infer_adaptive_zoom.py \
    --data_path                  "${DATA_PATH}"        \
    --model_path                 "${MODEL_PATH}"       \
    --video_root                 "${VIDEO_ROOT}"       \
    --output_dir                 "${OUTPUT_DIR}"       \
    --ref_jsonl                  "${REF_JSONL}"        \
    \
    --gpu_memory_utilization     0.7                   \
    --tensor_parallel_size       2                     \
    --max_model_len              32768                 \
    --max_pixels                 100352                \
    \
    --notool_fps                 0.5                   \
    --notool_min_pixels          25088                 \
    --notool_frames_upbound      64                    \
    --notool_max_tokens          4096                  \
    \
    --tool_fps                   0.5                   \
    --tool_min_pixels            25088                 \
    --tool_frames_upbound        64                    \
    --tool_max_tokens            4096                  \
    --tool_limit_mm              128                   \
    --tool_max_frames_per_call   16                    \
    --tool_workers               8                     \
    \
    --n_paths                    5                     \
    --sample_temp                0.4                   \
    --ent_threshold              "${ENT_THRESHOLD}"    \
    --max_rounds                 "${MAX_ROUNDS}"       \
    --batch_size                 32                    \
    ${EXTRA_FLAGS}
