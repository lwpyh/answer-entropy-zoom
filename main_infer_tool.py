#!/usr/bin/env python3
"""
Standalone tool-use inference script for VideoZoomer evaluation.

Supports multi-turn <video_zoom> tool calls (up to max_generation_round rounds).
Bypasses VERL / Ray / FSDP entirely — just vLLM (single GPU) + decord + openr1.

Parameters are aligned with scripts/eval_video.sh:
  fps=0.5, max_pixels=100352, min_pixels=25088, frames_upbound=64,
  max_tokens=4096, max_generation_round=5, tool_call_max_frames=16

Multi-turn loop (mirrors vllm_rollout_spmd_video.py):
  Round k: generate with stop="</video_zoom>"
           → parse <video_zoom> JSON → extract_video_clip (parallel)
           → append <tool_response> turn to prompt
           → repeat until max_generation_round or no tool call

Usage
-----
  python main_infer_tool.py \\
      --data_path  /data/DERI-Gong/jh015/VideoZoomer/longvideo-reason/eval_deltaS.yaml \\
      --video_root /data/DERI-Gong/jh015 \\
      --output_dir ./infer_results/eval_deltaS_tool
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

# ── System prompt (verbatim from scripts/eval_video.sh) ────────────────────── #
TOOL_SYSTEM_PROMPT = (
    'You are a helpful assistant. You will receive a low-frame-rate video and '
    'related questions. You can analyze the video content to answer the question '
    'and trigger high-frame-rate inspections when finer temporal resolution is '
    'needed. When you detect ambiguous motion/objects that require closer '
    'inspection, wrap your request in <video_zoom></video_zoom> tags and provide '
    'the exact time segment and target frame rate in JSON format: '
    '<video_zoom> {"segment": [start_sec, end_sec], "fps": n} </video_zoom>, '
    'it will return the video clip at the target fps to help you better answer '
    'the question. Note that the total frames num of the request clip cannot '
    'exceed 16 (e.g., (end_sec - start_sec) * fps \u2264 16) and DO NOT include '
    '<answer> tags in this round. \n'
    ' Example usage: <video_zoom> {"segment": [4.0, 6.0], "fps": 2} </video_zoom>. '
    'If the initial tool response does not provide sufficient information to '
    'answer the question, you may continue to request additional video zoom '
    'inspections as needed, until you either (1) gather enough information to '
    'form a complete answer, or (2) are explicitly instructed to stop using the '
    'tool. Output the thinking process within <think> </think> tags, once you '
    'confirm your final answer, place the final answer in \\boxed{} inside '
    '<answer> and </answer>.'
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   required=True,
                   help="YAML dataset config (VERL format)")
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015",
                   help="Base directory for relative video paths")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results")

    # vLLM settings (from eval_video.sh)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_tokens",             type=int,   default=4096)
    p.add_argument("--temperature",            type=float, default=0.0)

    # Video settings (from eval_video.sh)
    p.add_argument("--max_pixels",     type=int,   default=100352)
    p.add_argument("--min_pixels",     type=int,   default=25088)
    p.add_argument("--fps",            type=float, default=0.5)
    p.add_argument("--frames_upbound", type=int,   default=64)

    # Tool-use settings (from eval_video.sh)
    p.add_argument("--max_generation_round", type=int, default=5,
                   help="Max multi-turn rounds (including final answer round)")
    p.add_argument("--tool_call_max_frames", type=int, default=16,
                   help="Max frames per zoom clip (tool_call_max_frames in eval_video.sh)")
    p.add_argument("--limit_mm_per_prompt",  type=int, default=128,
                   help="Max total images per prompt across all turns")
    p.add_argument("--tool_workers",         type=int, default=8,
                   help="Parallel threads for tool execution")

    # Inference
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--system_prompt", default=TOOL_SYSTEM_PROMPT)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading  (same format as main_infer.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(yaml_path: str):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    samples = []
    for ds in cfg.get("datasets", []):
        with open(ds["json_path"]) as f:
            samples.extend(json.load(f))
    return samples


def resolve_video_path(video_rel: str, video_root: str) -> str:
    p = Path(video_rel)
    if p.is_absolute():
        return str(p)
    parts = p.parts
    if parts and parts[0] in (".", ".."):
        rel = Path(*parts[1:])
    else:
        rel = p
    return str(Path(video_root) / rel)


# ═══════════════════════════════════════════════════════════════════════════════
# Video preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def load_video_frames(video_path, fps, max_pixels, min_pixels, frames_upbound):
    from decord import VideoReader, cpu
    from PIL import Image
    import numpy as np

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    total   = len(vr)
    avg_fps = vr.get_avg_fps()

    stride  = max(1, int(avg_fps / fps)) if fps > 0 else 1
    indices = list(range(0, total, stride))
    if len(indices) > frames_upbound:
        sel     = np.linspace(0, len(indices) - 1, frames_upbound, dtype=int)
        indices = [indices[s] for s in sel]

    frame_times = [idx / avg_fps for idx in indices]
    raw         = vr.get_batch(indices).asnumpy()

    frames = []
    for arr in raw:
        img = Image.fromarray(arr.astype("uint8"), "RGB")
        h, w    = img.height, img.width
        pixels  = h * w
        if pixels > max_pixels:
            scale = (max_pixels / pixels) ** 0.5
            img   = img.resize(
                (max(2, int(w * scale / 2) * 2), max(2, int(h * scale / 2) * 2)),
                Image.LANCZOS,
            )
        elif pixels < min_pixels:
            scale = (min_pixels / pixels) ** 0.5
            img   = img.resize(
                (max(2, int(w * scale / 2) * 2), max(2, int(h * scale / 2) * 2)),
                Image.LANCZOS,
            )
        frames.append(img)

    return frame_times, frames


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════════════

def _frame_vision_tokens(frame_times):
    """One <|image_pad|> placeholder per frame; vLLM expands to correct count."""
    return "".join(
        f"<frame{i}_time{t:.2f}s><|vision_start|><|image_pad|><|vision_end|>"
        for i, t in enumerate(frame_times)
    )


def build_initial_prompt(question, frame_times, processor, system_prompt):
    """
    Build turn-1 prompt as TEXT.  Passing a text prompt (not pre-tokenised
    token_ids) lets vLLM's own input_processor handle vision-token expansion,
    avoiding the placeholder-count mismatch from manual pre-expansion.
    """
    vision_str = _frame_vision_tokens(frame_times)
    if "<image>" in question:
        user_content = question.replace("<image>", vision_str, 1)
    else:
        user_content = vision_str + "\n" + question

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_tool_response_turn(frame_times, is_last_round: bool) -> str:
    """
    Build the user turn that carries zoom results.
    Mirrors vllm_rollout_spmd_video.py tool_call_prompt_message exactly.
    """
    vision_pad = _frame_vision_tokens(frame_times)
    msg = (
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<tool_response>\n"
        "The frames of the video clip are shown below:\n"
        + vision_pad
        + "\n</tool_response>\n"
        "continue your reasoning process inside <think> and </think> "
        "and then write your final answer inside <answer> and </answer>"
    )
    if is_last_round:
        msg += (
            "Do not call <video_zoom> in this round, "
            "give a final answer based on information above."
        )
    msg += "<|im_end|>\n<|im_start|>assistant\n"
    return msg


def build_tool_error_turn(error_str: str, is_last_round: bool) -> str:
    msg = (
        "<|im_end|>\n<|im_start|>user\n"
        + error_str
        + "\nPlease analyze the error information obtained from the function "
        "tool and adjust your response. "
        "Continue your reasoning process inside <think> and </think>."
    )
    if is_last_round:
        msg += (
            " Do not call <video_zoom> in this round, "
            "give a final answer based on information above."
        )
    msg += "<|im_end|>\n<|im_start|>assistant\n"
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
# Tool-call parsing  (mirrors function_tools.prepare_tool_call_inputs_video)
# ═══════════════════════════════════════════════════════════════════════════════

_ZOOM_RE = re.compile(r"<video_zoom>(.*?)</video_zoom>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_tool_call(response_text: str):
    """
    Returns (start_time, end_time, fps) if a valid <video_zoom> call is found,
    else None.
    """
    m = _ZOOM_RE.search(response_text)
    if not m:
        return None
    jsons = _JSON_RE.findall(m.group(1))
    if not jsons:
        return None
    try:
        obj = json.loads(jsons[0])
        return float(obj["segment"][0]), float(obj["segment"][1]), float(obj["fps"])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Reward scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_response(all_responses, ground_truth: str) -> float:
    """
    Score the full multi-turn response using judge_multi_choice.
    Concatenates all response strings (same as VERL's ' '.join(predict_str)).
    Ground truth may be wrapped in <answer>…</answer>.
    """
    gt_m = re.search(r"<answer>(.*?)</answer>", ground_truth, re.DOTALL)
    gt   = gt_m.group(1).strip() if gt_m else ground_truth.strip()
    full = " ".join(all_responses)
    try:
        from verl.utils.reward_score.openr1 import judge_multi_choice
        return float(judge_multi_choice(full, gt))
    except Exception:
        pred_m = re.search(r"<answer>(.*?)</answer>", full, re.DOTALL)
        pred   = pred_m.group(1).strip() if pred_m else ""
        return 1.0 if pred == gt else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results.jsonl")
    summary_json = os.path.join(args.output_dir, "summary.json")

    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    print(f"[vllm]  Initialising {args.model_path}  "
          f"(tp={args.tensor_parallel_size}, util={args.gpu_memory_utilization}, "
          f"max_model_len={args.max_model_len})")
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
        limit_mm_per_prompt={"image": args.limit_mm_per_prompt},
        enforce_eager=False,
        enable_prefix_caching=False,
    )

    # Sampling params: stop at </video_zoom> so we can intercept tool calls
    tool_sp = SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        stop=["</video_zoom>"],
        include_stop_str_in_output=True,
        detokenize=True,
    )
    # Last round: no stop string, let the model produce a full answer
    final_sp = SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        detokenize=True,
    )

    # Import tool implementation (same code the VERL rollout uses)
    from verl.workers.rollout.vllm_rollout.function_tools import extract_video_clip

    results       = []
    total_correct = 0.0
    total_scored  = 0

    with open(output_jsonl, "w") as out_f:
        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Batches"):
            batch = samples[batch_start: batch_start + args.batch_size]

            # ── Build initial inputs ────────────────────────────────────────── #
            accum_prompts  = []   # accumulated prompt text per valid sample
            accum_images   = []   # accumulated image list per valid sample
            valid_samples  = []

            for sample in batch:
                try:
                    video_rel  = sample["videos"][0]
                    video_path = resolve_video_path(video_rel, args.video_root)
                    ft, frames = load_video_frames(
                        video_path,
                        fps=args.fps,
                        max_pixels=args.max_pixels,
                        min_pixels=args.min_pixels,
                        frames_upbound=args.frames_upbound,
                    )
                    prompt = build_initial_prompt(
                        sample["problem"], ft, processor, args.system_prompt
                    )
                    accum_prompts.append(prompt)
                    accum_images.append(list(frames))
                    valid_samples.append(sample)
                except Exception as e:
                    print(f"\n[prep] skip {sample.get('problem_id', '?')}: {e}")

            if not valid_samples:
                continue

            # ── Multi-turn state ────────────────────────────────────────────── #
            N             = len(valid_samples)
            all_responses = [[] for _ in range(N)]   # response strings per sample
            active        = list(range(N))            # indices still in the loop

            # ── Generation loop ─────────────────────────────────────────────── #
            for round_idx in range(args.max_generation_round):
                if not active:
                    break

                is_last_round = (round_idx == args.max_generation_round - 1)
                sp = final_sp if is_last_round else tool_sp

                vllm_inputs = [
                    {
                        "prompt":           accum_prompts[i],
                        "multi_modal_data": {"image": accum_images[i]},
                    }
                    for i in active
                ]
                outputs = llm.generate(vllm_inputs, sp)

                # Collect responses and detect tool calls
                tool_queue  = {}   # local_j → (global_i, (start, end, fps))
                next_active = []

                for local_j, (i, out) in enumerate(zip(active, outputs)):
                    resp = out.outputs[0].text
                    all_responses[i].append(resp)

                    tool_call = None if is_last_round else parse_tool_call(resp)
                    if tool_call:
                        tool_queue[local_j] = (i, tool_call)
                        next_active.append(i)
                        # Append this response to the accumulated prompt
                        # (the tool response turn is appended after execution below)
                        accum_prompts[i] += resp

                # ── Execute tool calls in parallel ──────────────────────────── #
                if tool_queue and not is_last_round:
                    is_next_last = (round_idx == args.max_generation_round - 2)

                    with ThreadPoolExecutor(max_workers=args.tool_workers) as ex:
                        futures = {}
                        for local_j, (i, (start, end, fps_z)) in tool_queue.items():
                            video_path = resolve_video_path(
                                valid_samples[i]["videos"][0], args.video_root
                            )
                            f = ex.submit(
                                extract_video_clip,
                                video_path=video_path,
                                start_time=start,
                                end_time=end,
                                fps=fps_z,
                                max_pixels=args.max_pixels,
                                min_pixels=args.min_pixels,
                                max_frames=args.tool_call_max_frames,
                                storage_system="local",
                            )
                            futures[f] = (local_j, i)

                        for f in as_completed(futures):
                            local_j, i = futures[f]
                            result = f.result()

                            if isinstance(result, dict):
                                zoom_times  = result["frame_time"]
                                zoom_frames = result["frames"]
                                turn_text   = build_tool_response_turn(
                                    zoom_times, is_next_last
                                )
                                accum_prompts[i] += turn_text
                                accum_images[i].extend(zoom_frames)
                            else:
                                # extract_video_clip returned an error string
                                turn_text = build_tool_error_turn(
                                    str(result), is_next_last
                                )
                                accum_prompts[i] += turn_text

                active = next_active

            # ── Score and save ──────────────────────────────────────────────── #
            for i, sample in enumerate(valid_samples):
                gt     = sample.get("solution", "")
                reward = score_response(all_responses[i], gt)
                n_tool_calls = sum(
                    1 for r in all_responses[i] if parse_tool_call(r) is not None
                )

                record = {
                    "problem_id":      sample.get("problem_id"),
                    "data_source":     sample.get("data_source"),
                    "problem":         sample["problem"],
                    "solution":        gt,
                    "responses":       all_responses[i],
                    "reward":          reward,
                    "tool_call_count": n_tool_calls,
                    "n_rounds":        len(all_responses[i]),
                    "delta_s":         sample.get("delta_s"),
                    "video":           sample.get("videos", [None])[0],
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                total_correct += reward
                total_scored  += 1

    # ── Summary ─────────────────────────────────────────────────────────────── #
    accuracy       = total_correct / total_scored if total_scored else 0.0
    avg_tool_calls = (
        sum(r["tool_call_count"] for r in results) / len(results) if results else 0.0
    )
    avg_rounds = (
        sum(r["n_rounds"] for r in results) / len(results) if results else 0.0
    )

    print(f"\n{'='*60}")
    print(f"Samples:        {len(results)}")
    print(f"Accuracy:       {accuracy:.4f}  ({total_correct:.0f} / {total_scored})")
    print(f"Avg tool calls: {avg_tool_calls:.2f}")
    print(f"Avg rounds:     {avg_rounds:.2f}")
    print(f"Results:        {output_jsonl}")

    summary = {
        "accuracy":       accuracy,
        "correct":        total_correct,
        "scored":         total_scored,
        "total":          len(results),
        "avg_tool_calls": avg_tool_calls,
        "avg_rounds":     avg_rounds,
        "model":          args.model_path,
        "data_path":      args.data_path,
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:        {summary_json}")


if __name__ == "__main__":
    main()
