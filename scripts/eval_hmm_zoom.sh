#!/bin/bash
# =============================================================================
# eval_hmm_zoom.sh  ─  Single-pass HMM-based zoom trigger inference
#
# Design: TOOL_SYS R1 → extract think chain → HMM score → zoom/skip decision
# Advantages over temp_vote:
#   - No NOTOOL_SYS degradation (always TOOL_SYS)
#   - Single forward pass (no 5× temperature sampling)
#   - ~14% fewer zoom executions at default threshold
#   - Expected acc ≥ greedy baseline at threshold ≥ +1.7
#
# Threshold guide (calibrated on 551 LVR samples):
#   threshold=+1.7  → zoom ~86%,  acc ≈ +0.7pp vs greedy
#   threshold=+1.0  → zoom ~80%,  acc ≈ +0.3pp vs greedy
#   threshold=-6.0  → zoom ~40%,  acc ≈ -2.6pp vs greedy (save 60% compute)
#   threshold=-inf  → zoom 100%,  acc = greedy baseline (degenerate case)
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
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/infer_results/hmm_zoom_v2}"

# HMM threshold: +1.7 = skip top 14% most confident samples
# Set to -999 to degrade to always-zoom (≡ greedy baseline)
HMM_THRESHOLD="${HMM_THRESHOLD:-1.7}"

# Optional: path to analysis_hmm_transitions.json for calibrated weights
HMM_WEIGHTS="${HMM_WEIGHTS:-$(pwd)/analysis_hmm_transitions.json}"

echo "============================================================"
echo "  DATA               : ${DATA_PATH}"
echo "  OUTPUT             : ${OUTPUT_DIR}"
echo "  hmm_threshold      : ${HMM_THRESHOLD}"
echo "  hmm_weights        : ${HMM_WEIGHTS}"
echo "  Expected zoom rate : ~86% at threshold=1.7"
echo "  Expected acc       : ≈ greedy + 0.7pp"
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
    --hmm_threshold              "${HMM_THRESHOLD}"          \
    --hmm_weights_path           "${HMM_WEIGHTS}"            \
    \
    --batch_size                 32
