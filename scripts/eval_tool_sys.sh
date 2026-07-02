#!/bin/bash
# =============================================================================
# eval_tool_sys.sh
#
# Unified majority-vote inference (all stages beam, no greedy):
#   (1) R1: TOOL_SYS prompt, beam(n=5, T=1.0), stop at </video_zoom>
#       → zoom detected in any path → execute → STEP B
#       → no zoom in all paths      → graduate (majority answer)
#   (2) STEP B: beam(n=5, T=1.0) on post-zoom context (is_last=True)
#       → low entropy / last round  → graduate
#       → high entropy              → STEP A' (beam zoom detection)
#   (3) STEP A': beam(n=5, T=1.0), stop at </video_zoom> on continuation
#       → zoom found → execute → back to STEP B
#       → no zoom   → graduate (majority from last STEP B)
#   Video params aligned to training: fps=0.5, min_pixels=25088, frames_upbound=64
#   Temperature aligned to training rollout default: T=1.0
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
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/adaptive_zoom_aligned}"
ENT_THRESHOLD="${ENT_THRESHOLD:-0.3}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"

echo "============================================================"
echo "  DATA          : ${DATA_PATH}"
echo "  MODEL         : ${MODEL_PATH}"
echo "  OUTPUT        : ${OUTPUT_DIR}"
echo "  ENT_THRESHOLD : ${ENT_THRESHOLD}"
echo "  STRATEGY      : unified majority-vote (beam n=5, T=1.0, all stages)"
echo "============================================================"

EXTRA_FLAGS=""
if [ "${MODE}" = "analyze" ]; then
    EXTRA_FLAGS="--analyze_only"
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
    --sample_temp                1.0                   \
    --ent_threshold              "${ENT_THRESHOLD}"    \
    --max_rounds                 "${MAX_ROUNDS}"       \
    --batch_size                 64                    \
    ${EXTRA_FLAGS}
