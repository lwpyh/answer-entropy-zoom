#!/bin/bash
# =============================================================================
# eval_answer_wald_zoom.sh — Full answer-distribution pipeline (NeurIPS)
#
# R1 trigger: answer_entropy  (k=5 samples → H(P(A/B/C/D)), no keywords)
# R2+ stopping: Wald JSD test on answer distributions
#
# Complete pipeline — zero human priors, zero logprobs:
#
#   R1: draw k=5 samples (T=0.7) → H_answer → score=-H
#       score > threshold → SKIP zoom (use majority-vote answer directly)
#       score ≤ threshold → EXECUTE zoom → R2+
#
#   R2+: after each zoom round, draw wald_k=3 samples (T=0.7)
#       Compute D_t = P(A/B/C/D), compare JSD(D_t, D_{t-1})
#       JSD < wald_threshold → answer converged → STOP, use majority answer
#       JSD ≥ wald_threshold → still changing → continue zooming
#
# Why this is principled:
#   - R1 trigger measures answer uncertainty directly (not token uncertainty)
#   - R2+ stopping compares same-type distributions (answers vs answers)
#     → no text-length confound (fixes iter_entropy failure)
#   - Both signals are purely observational — no keywords, weights, or logprobs
#
# JSD: 0 = identical, log(2)≈0.69 = maximally different
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

MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"
VIDEO_ROOT="/data/DERI-Gong/jh015/VideoZoomer"
DATA_PATH="${DATA_PATH:-/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/answer_wald_zoom}"

# R1 answer_entropy trigger
ENTROPY_THRESHOLD="${ENTROPY_THRESHOLD:--0.30}"  # score=-H > threshold → skip zoom
ANSWER_K="${ANSWER_K:-5}"
ANSWER_TEMP="${ANSWER_TEMP:-0.7}"

# R2+ Wald stopping
WALD_THRESHOLD="${WALD_THRESHOLD:-0.05}"     # JSD < threshold → answer converged → stop
WALD_K="${WALD_K:-3}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  === R1 trigger ==="
echo "  score_mode         : answer_entropy  (no keywords, no logprobs)"
echo "  answer_k           : ${ANSWER_K}  (temperature samples)"
echo "  entropy_threshold  : ${ENTROPY_THRESHOLD}  (score=-H > threshold → skip)"
echo "  === R2+ stopping ==="
echo "  answer_wald        : enabled"
echo "  wald_k             : ${WALD_K}  (samples per round)"
echo "  wald_threshold     : ${WALD_THRESHOLD}  (JSD < threshold → converged → stop)"
echo "  answer_temperature : ${ANSWER_TEMP}"
echo "  Full pipeline: zero human priors, zero logprobs."
echo "============================================================"

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_hmm_zoom.py \
    --data_path                  "${DATA_PATH}"              \
    --model_path                 "${MODEL_PATH}"             \
    --video_root                 "${VIDEO_ROOT}"             \
    --output_dir                 "${OUTPUT_DIR}"             \
    \
    --gpu_memory_utilization     0.7                         \
    --tensor_parallel_size       2                           \
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
    --score_mode                 answer_entropy              \
    --entropy_threshold          "${ENTROPY_THRESHOLD}"      \
    --answer_k                   "${ANSWER_K}"               \
    --answer_temperature         "${ANSWER_TEMP}"            \
    --answer_wald                                            \
    --wald_k                     "${WALD_K}"                 \
    --wald_threshold             "${WALD_THRESHOLD}"         \
    \
    --batch_size                 32
