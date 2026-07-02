#!/bin/bash
# =============================================================================
# convert_webm_mp4.sh
#
# Converts remaining webm/mkv videos to H264 mp4 so decord can load them
# without hanging (VP9 pix_fmt=None causes decord to hang on some nodes).
#
# Strategy:
#   - Tries GPU path: vp9_cuvid (nvdec) decode → h264_nvenc encode  [fast]
#   - Falls back to CPU: -skip_frame nokey decode → libx264 ultrafast [slower]
#   - Output: 1 fps, max 640px wide — enough for max_pixels=100352
#   - Runs NJOBS conversions in parallel
#
# Usage:
#   sbatch scripts/convert_webm_mp4.sh
# =============================================================================

#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=8G
#SBATCH --exclude=sbg2,ddg1,ddg2

module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0
module load openssl/3.3.0-gcc-12.2.0

mamba activate VideoZoomer
pip install av
set -euo pipefail
set -x

# Pure-Python conversion: no ffmpeg CLI needed, uses PyAV + libx264 directly.
python /data/DERI-Gong/jh015/VideoZoomer/scripts/convert_webm_mp4.py

echo "==================================================================="
echo "  Conversion complete."
echo "  mp4 files written next to originals in longvila_videos/"
echo "  Now re-submit eval_adaptive_zoom.sh — inference will prefer mp4."
echo "==================================================================="
