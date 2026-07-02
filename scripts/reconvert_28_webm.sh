#!/bin/bash
#SBATCH --job-name=webm2mp4_fix
#SBATCH --output=/dev/null
#SBATCH -p compute
#SBATCH -A pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --array=0-27

module load miniforge/24.7.1
source /share/apps/rocky9/general/apps/miniforge/24.7.1/etc/profile.d/conda.sh
conda activate r1_video_v4

WEBM_LIST=/data/DERI-Gong/jh015/VideoZoomer/scripts/webm_reconv_list.txt
WEBM_FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$WEBM_LIST")

if [ -z "$WEBM_FILE" ]; then
    echo "No file for task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

MP4_FILE="${WEBM_FILE%.webm}.mp4"
echo "Converting: $(basename $WEBM_FILE)"

# -movflags +faststart writes moov atom at the start, safe against mid-write cancellation
ffmpeg -y \
    -i "$WEBM_FILE" \
    -c:v libx264 \
    -preset fast \
    -crf 18 \
    -movflags +faststart \
    -c:a aac \
    -b:a 128k \
    "$MP4_FILE"

if [ $? -eq 0 ]; then
    echo "Done: $(basename $MP4_FILE)"
else
    echo "FAILED: $(basename $WEBM_FILE)"
    rm -f "$MP4_FILE"
fi
