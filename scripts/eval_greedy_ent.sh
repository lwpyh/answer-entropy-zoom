#!/bin/bash
# =============================================================================
# eval_greedy_ent.sh
#
# Greedy-R1 + beam-entropy-gated adaptive tool inference:
#   Round 1: greedy (T=0, n=1) for prediction  +  beam (n=B, T) for entropy gate
#   Round 2+: uncertain samples only → greedy tool turn → beam re-eval → gate
#
# Key difference from eval_adaptive_zoom.sh:
#   --greedy_r1   acc_notool = true greedy answer (T=0), not beam majority
#
# Examples:
#   sbatch scripts/eval_greedy_ent.sh
#   MODE=analyze sbatch scripts/eval_greedy_ent.sh
#   ENT_THRESHOLD=0.5 sbatch scripts/eval_greedy_ent.sh
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

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml}"
REF_JSONL="${REF_JSONL:-/data/DERI-Gong/jh015/VideoZoomer/infer_results/error_detection_full/results_error_detection.jsonl}"
MODE="${MODE:-full}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/greedy_ent}"
ENT_THRESHOLD="${ENT_THRESHOLD:-0.3}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"

echo "============================================================"
echo "  DATA          : ${DATA_PATH}"
echo "  MODEL         : ${MODEL_PATH}"
echo "  REF           : ${REF_JSONL}"
echo "  OUTPUT        : ${OUTPUT_DIR}"
echo "  MODE          : ${MODE}"
echo "  ENT_THRESHOLD : ${ENT_THRESHOLD}  (θ — gate for tool activation)"
echo "  MAX_ROUNDS    : ${MAX_ROUNDS}"
echo "  GREEDY_R1     : true  (T=0 prediction; beam for entropy only)"
echo "============================================================"

EXTRA_FLAGS="--greedy_r1"
if [ "${MODE}" = "analyze" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --analyze_only"
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
    --notool_fps                 0.2                   \
    --notool_min_pixels          12544                 \
    --notool_frames_upbound      120                    \
    --notool_max_tokens          2048                  \
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
    --batch_size                 64                    \
    ${EXTRA_FLAGS}
