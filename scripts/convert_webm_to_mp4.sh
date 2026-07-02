#!/bin/bash
#SBATCH --job-name=webm2mp4
#SBATCH --output=/dev/null
#SBATCH -p compute
#SBATCH -A pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH --array=0-130

module load miniforge/24.7.1
source /share/apps/rocky9/general/apps/miniforge/24.7.1/etc/profile.d/conda.sh
conda activate r1_video_v4

WEBM_LIST=/data/DERI-Gong/jh015/VideoZoomer/scripts/webm_list.txt
WEBM_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$WEBM_LIST")

if [ -z "$WEBM_FILE" ]; then
    echo "No file for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

MP4_FILE="${WEBM_FILE%.webm}.mp4"

# Skip only if mp4 exists AND is large enough to be valid (>5MB)
if [ -f "$MP4_FILE" ] && [ "$(stat -c%s "$MP4_FILE")" -gt 5000000 ]; then
    echo "Already exists and valid, skipping: $MP4_FILE"
    exit 0
fi

echo "Converting: $WEBM_FILE -> $MP4_FILE"
ffmpeg -y \
    -i "$WEBM_FILE" \
    -c:v libx264 \
    -preset fast \
    -crf 18 \
    -c:a aac \
    -b:a 128k \
    "$MP4_FILE"

if [ $? -eq 0 ]; then
    echo "Done: $MP4_FILE"
else
    echo "FAILED: $WEBM_FILE"
    rm -f "$MP4_FILE"
fi
