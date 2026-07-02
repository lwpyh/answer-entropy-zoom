#!/bin/bash
# =============================================================================
# eval_notool_gate_zoom.sh — NOTOOL-first gate + iter_ate multi-round zoom
#
# Redesigned pipeline (logically correct order):
#   Phase 0: NOTOOL pass (1 inference) → measure H(A,B,C,D)
#            H < 0.50 → confident → answer directly  (1 pass total)
#            H ≥ 0.50 → uncertain → proceed to zoom
#   R1     : TOOL_SYS greedy → stops at <video_zoom>
#   Zoom   : execute video clip extraction
#   R2+    : continue with zoom context → answer or another zoom call
#            iter_ate: after each R2+ zoom call, re-check H via ATE pass
#            confident or not improving → stop; else → next zoom round
#   max_rounds=5 → up to 4 zoom executions after R1
#
# Threshold -0.50 rationale (vs previous -0.30):
#   - Previous ATE (threshold=-0.30): zoom 31.1% of samples
#   - iter_ae k=5 (threshold=-0.30): zoom 21.9%, acc=79.2% (best result)
#   - In clean NOTOOL context (no R1 prior), H is noisier for mid-range samples
#   - -0.50 = "zoom only if H≥0.50", matches iter_ae's effective cutoff
#     (k=5 discrete: min non-zero H = 0.50 for 4/1 split)
#   - Targets genuinely uncertain samples, avoids zooming mild mid-range noise
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G
#SBATCH --exclude=sbg2,ddg1,ddg2

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0
module load openssl/3.3.0-gcc-12.2.0

mamba activate VideoZoomer

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"
export HF_HOME="/data/home/acw652/.cache/huggingface"
export DECORD_EOF_RETRY_MAX=20480
export CUDA_VISIBLE_DEVICES=0

set -euo pipefail
set -x

MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/notool_gate_zoom}"

ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--0.50}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  Pipeline           : NOTOOL gate → TOOL_SYS R1 → zoom → R2+ (iter_ate)"
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}"
echo "  Phase 0: NOTOOL pass → H < 0.50 → skip zoom (1 pass only)"
echo "  Phase 0: H ≥ 0.50 → TOOL_SYS → zoom → R2+ with iter_ate re-check"
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_hmm_zoom.py \
    --data_path                  "${DATA_PATH}"              \
    --model_path                 "${MODEL_PATH}"             \
    --video_root                 "${VIDEO_ROOT}"             \
    --output_dir                 "${OUTPUT_DIR}"             \
    \
    --gpu_memory_utilization     0.7                         \
    --tensor_parallel_size       1                           \
    --max_model_len              32768                       \
    --max_pixels                 100352                      \
    --min_pixels                 25088                       \
    \
    --fps                        0.5                         \
    --frames_upbound             64                          \
    --max_tokens                 4096                        \
    --tool_limit_mm              128                         \
    --tool_max_frames_per_call   16                          \
    --tool_workers               8                           \
    --max_rounds                 5                           \
    \
    --notool_gate                                            \
    --iter_ate                                               \
    --entropy_threshold          "${ENTROPY_THRESHOLD}"      \
    \
    --batch_size                 32
