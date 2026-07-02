#!/bin/bash
# Fresh run with fixed R1: TOOL_SYS (no "Do not zoom"), tool_beam_sp, strict entropy

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

python /data/DERI-Gong/jh015/VideoZoomer/main_infer_adaptive_zoom.py \
    --data_path                  /data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_longvideoreason.yaml \
    --model_path                 zsgvivo/videozoomer                  \
    --video_root                 /data/DERI-Gong/jh015/VideoZoomer   \
    --output_dir                 /data/DERI-Gong/jh015/VideoZoomer/infer_results/adaptive_zoom_fix_r1 \
    --ref_jsonl                  /data/DERI-Gong/jh015/VideoZoomer/infer_results/error_detection_full/results_error_detection.jsonl \
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
    --ent_threshold              0.5                   \
    --max_rounds                 5                     \
    --batch_size                 32
