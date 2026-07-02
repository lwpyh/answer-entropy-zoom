#!/usr/bin/env python3
"""
Standalone inference script for VideoZoomer evaluation.

Bypasses VERL / Ray / FSDP entirely.  Just vLLM (single GPU) + video
frame loading + openr1 multiple-choice scoring.

Memory footprint:
  GPU : ~16 GB (model weights) + KV-cache (controlled by gpu_memory_utilization)
  CPU : ~3-5 GB (no FSDP, no Ray object-store)

Usage
-----
  python main_infer.py \
      --data_path  /data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_deltaS.yaml \
      --video_root /data/DERI-Gong/jh015 \
      --output_dir ./infer_results/eval_deltaS_notool

Optional overrides (defaults match the VERL eval_deltaS_notool.sh settings):
  --model_path               zsgvivo/videozoomer
  --gpu_memory_utilization   0.8
  --tensor_parallel_size     1
  --max_pixels               65536
  --fps                      0.2
  --frames_upbound           120
  --max_model_len            32768
  --max_tokens               2048
  --temperature              0.0
  --batch_size               4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

# ── project root on path so we can import VERL reward scorers ─────────────── #
sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   required=True,
                   help="YAML dataset config (same format as VERL training data)")
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer",
                   help="Base directory for relative video paths in the JSON")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results")

    # vLLM settings
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_tokens",             type=int,   default=2048)
    p.add_argument("--temperature",            type=float, default=0.0)

    # Video settings
    p.add_argument("--max_pixels",    type=int,   default=65536)
    p.add_argument("--min_pixels",    type=int,   default=12544)
    p.add_argument("--fps",           type=float, default=0.2)
    p.add_argument("--frames_upbound",type=int,   default=120)

    # Inference settings
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--system_prompt", default="You are a helpful assistant.")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(yaml_path: str):
    """Load all samples from a VERL-style YAML dataset config."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    samples = []
    for ds in cfg.get("datasets", []):
        with open(ds["json_path"]) as f:
            samples.extend(json.load(f))
    return samples


def resolve_video_path(video_rel: str, video_root: str) -> str:
    """Turn a relative video path (starts with './' or similar) into absolute."""
    p = Path(video_rel)
    if p.is_absolute():
        return str(p)
    # Strip leading './'
    parts = p.parts
    if parts and parts[0] in (".", ".."):
        rel = Path(*parts[1:])
    else:
        rel = p
    return str(Path(video_root) / rel)


# ═══════════════════════════════════════════════════════════════════════════════
# Video preprocessing  (mirrors multimodal_dataset.py logic)
# ═══════════════════════════════════════════════════════════════════════════════

def load_video_frames(
    video_path: str,
    fps: float = 0.2,
    max_pixels: int = 65536,
    min_pixels: int = 12544,
    frames_upbound: int = 120,
):
    """
    Load video, sample frames at `fps`, resize to pixel constraints.

    Returns
    -------
    frame_times : list[float]   – timestamp of each frame (seconds)
    frames      : list[PIL.Image]
    """
    from decord import VideoReader, cpu
    from PIL import Image
    import numpy as np

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    total = len(vr)
    avg_fps = vr.get_avg_fps()

    # Build frame index list
    stride = max(1, int(avg_fps / fps)) if fps > 0 else 1
    indices = list(range(0, total, stride))

    # Apply upbound
    if len(indices) > frames_upbound:
        sel = np.linspace(0, len(indices) - 1, frames_upbound, dtype=int)
        indices = [indices[i] for i in sel]

    frame_times = [idx / avg_fps for idx in indices]
    raw = vr.get_batch(indices).asnumpy()   # (N, H, W, 3)

    frames = []
    for arr in raw:
        img = Image.fromarray(arr.astype("uint8"), "RGB")
        h, w = img.height, img.width
        pixels = h * w
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            img = img.resize(
                (max(2, int(w * scale / 2) * 2), max(2, int(h * scale / 2) * 2)),
                Image.LANCZOS,
            )
        elif pixels < min_pixels:
            scale = (min_pixels / pixels) ** 0.5
            img = img.resize(
                (max(2, int(w * scale / 2) * 2), max(2, int(h * scale / 2) * 2)),
                Image.LANCZOS,
            )
        frames.append(img)

    return frame_times, frames


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════════════

def build_vllm_input(
    question: str,
    frame_times: list,
    frames: list,
    processor,
    system_prompt: str,
):
    """
    Build the vLLM input dict:
      { 'prompt': str, 'multi_modal_data': {'image': [...]} }

    We pass a TEXT prompt (not pre-tokenised token_ids) so that vLLM's own
    input_processor handles the vision-token expansion.  Pre-expanding manually
    causes a mismatch because vLLM independently recomputes token counts from
    the images using its internal smart_resize.

    The prompt contains ONE single <|image_pad|> placeholder per frame (wrapped
    in <|vision_start|>…<|vision_end|>).  vLLM's Qwen2.5-VL input_processor
    then expands each single placeholder to the correct number of tokens.
    """
    # Build user text: frame timestamps interleaved with single-token placeholders
    if "<image>" in question:
        # Replace the single <image> tag with all frame placeholders
        frame_tokens = "".join(
            f"<frame{i}_time{frame_times[i]:.2f}s>"
            f"<|vision_start|><|image_pad|><|vision_end|>"
            for i in range(len(frames))
        )
        user_content = question.replace("<image>", frame_tokens, 1)
    else:
        user_content = "".join(
            f"<frame{i}_time{frame_times[i]:.2f}s>"
            f"<|vision_start|><|image_pad|><|vision_end|>"
            for i in range(len(frames))
        ) + "\n" + question

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Pass prompt as TEXT — vLLM tokenises it and expands the single
    # <|image_pad|> placeholders to the correct count for each image.
    return {
        "prompt": prompt_text,
        "multi_modal_data": {"image": frames},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Reward scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_response(response_text: str, ground_truth: str) -> float:
    """
    Score a model response against ground truth using judge_multi_choice
    (the same rule-based scorer used by VERL's no-tool eval).
    Handles ground truth wrapped in <answer>…</answer> tags.
    """
    # Extract letter from ground truth
    import re
    gt_match = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
    gt = gt_match.group(1).strip() if gt_match else ground_truth.strip()

    try:
        from verl.utils.reward_score.openr1 import judge_multi_choice
        return float(judge_multi_choice(response_text, gt))
    except Exception as e:
        # Fallback: simple substring match
        pred_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
        pred = pred_match.group(1).strip() if pred_match else response_text.strip()
        return 1.0 if pred == gt else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results.jsonl")
    summary_json = os.path.join(args.output_dir, "summary.json")

    # ── Load dataset ──────────────────────────────────────────────────────── #
    print(f"[data] Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data] {len(samples)} samples")

    # ── Load processor ────────────────────────────────────────────────────── #
    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # ── Build vLLM engine ─────────────────────────────────────────────────── #
    from vllm import LLM, SamplingParams

    print(
        f"[vllm] Initialising {args.model_path}  "
        f"(tp={args.tensor_parallel_size}, "
        f"util={args.gpu_memory_utilization}, "
        f"max_model_len={args.max_model_len})"
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        mm_processor_kwargs={
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
        },
        limit_mm_per_prompt={"image": args.frames_upbound},
        enforce_eager=False,
        enable_prefix_caching=False,
    )

    sampling_params = SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        detokenize=True,
    )

    # ── Inference loop ────────────────────────────────────────────────────── #
    results = []
    total_correct = 0.0
    total_scored  = 0

    with open(output_jsonl, "w") as out_f:
        for batch_start in tqdm(range(0, len(samples), args.batch_size),
                                desc="Batches"):
            batch = samples[batch_start: batch_start + args.batch_size]

            # Prepare vLLM inputs for the batch
            vllm_inputs   = []
            valid_samples = []

            for sample in batch:
                try:
                    video_rel  = sample["videos"][0]
                    video_path = resolve_video_path(video_rel, args.video_root)
                    frame_times, frames = load_video_frames(
                        video_path,
                        fps=args.fps,
                        max_pixels=args.max_pixels,
                        min_pixels=args.min_pixels,
                        frames_upbound=args.frames_upbound,
                    )
                    vllm_inp = build_vllm_input(
                        question=sample["problem"],
                        frame_times=frame_times,
                        frames=frames,
                        processor=processor,
                        system_prompt=args.system_prompt,
                    )
                    vllm_inputs.append(vllm_inp)
                    valid_samples.append(sample)
                except Exception as e:
                    pid = sample.get("problem_id", "?")
                    print(f"\n[prep] skip sample {pid}: {e}")

            if not vllm_inputs:
                continue

            # Run inference
            outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

            # Score and save
            for sample, out in zip(valid_samples, outputs):
                response_text = out.outputs[0].text
                ground_truth  = sample.get("solution", "")
                reward        = score_response(response_text, ground_truth)

                record = {
                    "problem_id":  sample.get("problem_id"),
                    "data_source": sample.get("data_source"),
                    "problem":     sample["problem"],
                    "solution":    ground_truth,
                    "response":    response_text,
                    "reward":      reward,
                    "delta_s":     sample.get("delta_s"),
                    "video":       sample.get("videos", [None])[0],
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                total_correct += reward
                total_scored  += 1

    # ── Summary ───────────────────────────────────────────────────────────── #
    accuracy = total_correct / total_scored if total_scored > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Samples:  {len(results)}")
    print(f"Scored:   {total_scored}")
    print(f"Accuracy: {accuracy:.4f}  ({total_correct:.0f} / {total_scored})")
    print(f"Results:  {output_jsonl}")

    summary = {
        "accuracy":   accuracy,
        "correct":    total_correct,
        "scored":     total_scored,
        "total":      len(results),
        "model":      args.model_path,
        "data_path":  args.data_path,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:  {summary_json}")


if __name__ == "__main__":
    main()
