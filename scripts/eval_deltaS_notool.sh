#!/bin/bash
# The working directory for the job is
# the current directory by default in
# Slurm
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -t 1:0:0  # Request 1 hour runtime
#SBATCH --cpus-per-gpu=8     # 8 cores per GPU
#SBATCH --mem-per-cpu=12G  # 12G×8cpu=96GB; vLLM sleep mode offloads ~31GB to CPU + FSDP ~14GB + Ray 20GB = ~65GB peak
#SBATCH --gres=gpu:1 # request 1 GPU
#SBATCH --exclude=sbg2,ddg1,ddg2  # exclude V100 nodes (CC 7.0, 16 GB) - incompatible with flash-attn 2.8.2 + OOM
module load miniforge/24.7.1
module load gcc/12.2.0
module load cmake/3.27.9-gcc-12.2.0
module load cuda/12.4.0-gcc-12.2.0
module load openssl/3.3.0-gcc-12.2.0

# Initialize conda so that 'mamba activate' works in non-interactive Slurm shells
# mamba create -n VideoZoomer python=3.11 -y
mamba activate VideoZoomer

# mamba install -c conda-forge dbus-python
# pip install -r requirements.txt --no-build-isolation
# pip3 install -e .
# pip3 install httpx==0.23.3
# pip install flash-attn==2.6.3 --no-build-isolation
# pip install math-verify==0.8.0
export HUGGING_FACE_HUB_TOKEN="YOUR_HF_TOKEN_HERE"
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
export DECORD_EOF_RETRY_MAX=20480
export HF_HOME="/data/home/acw652/.cache/huggingface"
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
# Make CUDA runtime libs visible to vLLM (fixes libcuda.so.1 not found)
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

set -euo pipefail
set -x
# python -m pip install wandb accelerate codetiming datasets hydra-core peft pybind11 pylatexenc torchdata transformers==4.52.0 vllm==0.6.3 tensordict==0.9.1
# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH="/data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_deltaS.yaml"
MODEL_PATH="${MODEL_PATH:-zsgvivo/videozoomer}"   # HuggingFace model; override via env var
LOG_ROOT="${LOG_ROOT:-$(pwd)/eval_logs}"
PROJECT_NAME="${PROJECT_NAME:-videozoomer_eval}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_deltaS_notool}"

# Align with requirements.txt: vllm==0.9.2 + torch==2.7.0 + flash_attn==2.8.2
# vllm 0.6.3: rope_type="default" assert fails; vllm 0.7.3: KeyError in Qwen2.5-VL weight loading.
# vllm 0.9.2 is what requirements.txt targets and fixes both issues.
# pip install vllm==0.9.2
# # Reinstall flash-attn for the torch 2.7.0 that vllm 0.9.2 pulls in
# pip install flash-attn==2.8.2

# pip install math_verify


# Patch: transformers 4.52.0 bug - ALL_PARALLEL_STYLES is None for torch<2.5
# (check is >=2.3 but ALL_PARALLEL_STYLES only set for >=2.5, causing TypeError)
python -c "
import re, pathlib
p = pathlib.Path('/data/home/acw652/.local/lib/python3.12/site-packages/transformers/modeling_utils.py')
txt = p.read_text()
old = 'if v not in ALL_PARALLEL_STYLES:'
new = 'if ALL_PARALLEL_STYLES is not None and v not in ALL_PARALLEL_STYLES:'
if old in txt:
    p.write_text(txt.replace(old, new))
    print('Patch applied: ALL_PARALLEL_STYLES None guard')
else:
    print('Patch already applied or not needed')
"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_PATH}" \
    data.val_files="${DATA_PATH}" \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.image_key=videos \
    data.video_fps=0.2 \
    data.max_pixels=65536 \
    data.min_pixels=12544 \
    data.frames_upbound=120 \
    data.storage_system=local \
    data.pass_video_as_frames=False \
    'data.system_prompt="You are a helpful assistant."' \
    reward_model.reward_manager=naive_multithreads \
    reward_model.val_reward_manager=naive_multithreads \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler=constant \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm_video_multiturn \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.tool_call=null \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.max_total_response_length=4096 \
    actor_rollout_ref.rollout.max_generation_round=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1000000 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    trainer.val_only=True \
    trainer.val_generations_to_log_to_wandb=128 \
    reward_model.log_rewards_separately=True \
    reward_model.acc_reward_weight=1.0 \
    reward_model.format_reward_weight=0.0 \
    reward_model.tool_call_penalty=0.0 \
    "trainer.validation_data_dir=${LOG_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/val"

