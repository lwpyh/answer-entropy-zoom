#!/usr/bin/env python3
"""
main_infer_qwen35_lvr.py — Qwen3.5-4B LongVideoReason vLLM Inference

vLLM batch inference for Qwen3.5-4B on LongVideoReason.
Matches lmms-eval format (native Qwen function-calling: FrameAt / VideoClip).

Two experiments controlled by --entropy_gate_k:
  --entropy_gate_k 0   → pure tool-call (always execute tool)  [job 7976340]
  --entropy_gate_k 5   → answer-entropy gate                    [job 7976341]

Parameters aligned to lmms-eval run_lvr_tool*.sh:
  max_pixels=602112  min_pixels=200704  max_frames=128  enable_thinking=True
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from main_infer_adaptive_zoom import (
    load_video_frames,
    resolve_video_path,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Prompt constants  (identical to lmms-eval longvideoreason/utils.py)
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_BLOCK = """\
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "FrameAt", "description": "Get a single frame at a specific time from the video.", "parameters": {"type": "object", "properties": {"time": {"type": "number", "description": "Time (in seconds) of the frame to extract."}}, "required": ["time"]}}}
{"type": "function", "function": {"name": "VideoClip", "description": "Extract a video clip between start and end times.", "parameters": {"type": "object", "properties": {"t_start": {"type": "number", "description": "Start time (in seconds) of the clip."}, "t_end": {"type": "number", "description": "End time (in seconds) of the clip."}}, "required": ["t_start", "t_end"]}}}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

ANSWER_FORMAT = (
    "\n\nAfter your analysis, give your final answer as "
    "<answer>A</answer>, <answer>B</answer>, <answer>C</answer>, or <answer>D</answer>."
)

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER_RE    = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
BOXED_RE     = re.compile(r"\\boxed\{([A-D])\}", re.IGNORECASE)

MAX_TOOL_CALLS = 3   # matches lmms-eval max_agentic_steps=3


# ═══════════════════════════════════════════════════════════════════════════════
# Tool dispatch  (mirrors lmms-eval utils.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _open_video(video_path):
    import decord
    decord.bridge.set_bridge("native")
    vr  = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    return vr, fps, len(vr)


def _to_pil(frame_np, max_pixels=602112):
    from PIL import Image
    img = Image.fromarray(frame_np)
    if img.width * img.height > max_pixels:
        scale = math.sqrt(max_pixels / (img.width * img.height))
        img = img.resize(
            (max(2, int(img.width * scale)), max(2, int(img.height * scale))),
            Image.LANCZOS,
        )
    return img


def tool_frame_at(video_path, time_sec, max_pixels=602112):
    try:
        vr, fps, total = _open_video(video_path)
        idx   = max(0, min(int(time_sec * fps), total - 1))
        frame = vr[idx].asnumpy()
        return [_to_pil(frame, max_pixels)], f"Frame at t={time_sec:.2f}s"
    except Exception as e:
        print(f"[FrameAt] {e}")
        return [], f"FrameAt failed: {e}"


def tool_video_clip(video_path, t_start, t_end, max_frames=8, max_pixels=602112):
    try:
        vr, fps, total = _open_video(video_path)
        start_f = max(0, int(t_start * fps))
        end_f   = min(total - 1, int(t_end * fps))
        if start_f >= end_f:
            end_f = min(start_f + int(fps * 2), total - 1)
        n = end_f - start_f + 1
        if n <= max_frames:
            indices = list(range(start_f, end_f + 1))
        else:
            step    = n / max_frames
            indices = [int(start_f + i * step) for i in range(max_frames)]
        raw = vr.get_batch(indices).asnumpy()
        return [_to_pil(f, max_pixels) for f in raw], f"Clip [{t_start:.2f}s–{t_end:.2f}s], {len(indices)} frames"
    except Exception as e:
        print(f"[VideoClip] {e}")
        return [], f"VideoClip failed: {e}"


def dispatch_tool(tc, video_path, max_pixels=602112):
    name = tc.get("name", "")
    args = tc.get("arguments", {})
    if name == "FrameAt":
        t = float(args.get("time", 0))
        return tool_frame_at(video_path, t, max_pixels)
    if name == "VideoClip":
        t0 = float(args.get("t_start", 0))
        t1 = float(args.get("t_end", t0 + 30))
        return tool_video_clip(video_path, t0, t1, max_pixels=max_pixels)
    return [], f"Unknown tool '{name}'"


# ═══════════════════════════════════════════════════════════════════════════════
# Answer extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_answer(text):
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    m = BOXED_RE.search(text)
    if m:
        return m.group(1).upper()
    # last standalone letter in final 300 chars
    m = re.search(r"\b([A-D])\b(?=[^A-D]*$)", text[-300:])
    if m:
        return m.group(1).upper()
    return None


def parse_tool_call(text):
    """Return last parsed tool-call dict or None."""
    matches = TOOL_CALL_RE.findall(text)
    for raw in reversed(matches):
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Entropy gate
# ═══════════════════════════════════════════════════════════════════════════════

def compute_answer_entropy(answers):
    """H(P(A/B/C/D)) from a list of answer letters."""
    if not answers:
        return float("inf")
    from collections import Counter
    counts = Counter(answers)
    total  = len(answers)
    probs  = np.array([counts.get(c, 0) / total for c in "ABCD"], dtype=np.float64) + 1e-12
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs)))


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════════════

def build_initial_prompt(question, frames, processor, enable_thinking=True):
    """Initial multi-modal prompt: all video frames + TOOL_BLOCK + question."""
    question_clean = question.replace("<image>", "").strip()
    content = []
    for _ in frames:
        content.append({"type": "image"})
    content.append({"type": "text", "text": TOOL_BLOCK + "\n\n" + question_clean + ANSWER_FORMAT})

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": content},
    ]
    try:
        prompt = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
    except TypeError:
        # older transformers: no chat_template_kwargs
        prompt = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    return prompt


def build_continuation_prompt(
    initial_prompt, r1_output, tool_frames, tool_desc,
    question, is_last, processor
):
    """Append tool call + tool response turn to existing prompt text."""
    # Close assistant turn, open user turn with tool response
    n_imgs = len(tool_frames)
    img_tokens = "".join(["<image>" for _ in tool_frames])  # placeholder; replaced by vLLM

    tool_response = (
        f"<|im_end|>\n<|im_start|>user\n"
        f"<tool_response>\n{tool_desc} — frames shown above.\n</tool_response>\n"
    )

    follow_up = "Based on these frames and the full video context, "
    if is_last:
        follow_up += "give your final answer now (no more tool calls allowed)."
    else:
        follow_up += "continue your analysis."
    follow_up += ANSWER_FORMAT

    continuation = tool_response + follow_up + "<|im_end|>\n<|im_start|>assistant\n"

    # We need to inject the image tokens at the right place in the prompt.
    # The tool response images come right after the tool_response header.
    # We'll embed <|vision_start|><|image_pad|><|vision_end|> placeholders
    # via the processor by building a mini-message for the user turn.
    # Simpler: just embed image tokens as raw text; vLLM maps them via multi_modal_data.

    # Build the raw text prompt: initial_prompt + r1_output + continuation
    # Images in tool_response are referenced by position in multi_modal_data["image"]
    # For simplicity, we embed image-count-aware placeholder using processor vocab.
    img_placeholder = processor.image_token if hasattr(processor, "image_token") else "<|image_pad|>"
    vision_s = "<|vision_start|>"
    vision_e = "<|vision_end|>"

    tool_imgs_str = "".join([f"{vision_s}{img_placeholder}{vision_e}" for _ in tool_frames])

    # Insert image tokens into tool_response text
    tool_response_with_imgs = (
        f"<|im_end|>\n<|im_start|>user\n"
        f"<tool_response>\n{tool_imgs_str}{tool_desc} — frames shown above.\n</tool_response>\n"
        + follow_up
        + "<|im_end|>\n<|im_start|>assistant\n"
    )

    full_prompt = initial_prompt + r1_output + tool_response_with_imgs
    return full_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(data_path):
    if data_path.endswith(".yaml"):
        with open(data_path) as f:
            cfg = yaml.safe_load(f)
        samples = []
        for ds in cfg.get("datasets", []):
            with open(ds["json_path"]) as f:
                samples.extend(json.load(f))
        return samples
    elif data_path.endswith(".json"):
        with open(data_path) as f:
            return json.load(f)
    elif data_path.endswith(".jsonl"):
        samples = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    else:
        raise ValueError(f"Unsupported data format: {data_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sample state
# ═══════════════════════════════════════════════════════════════════════════════

class SampleState:
    def __init__(self, pid, gt, video_path, question, initial_prompt, frames):
        self.pid            = pid
        self.gt             = gt
        self.video_path     = video_path
        self.question       = question
        self.initial_prompt = initial_prompt
        self.frames         = list(frames)   # all frames for current prompt (grows with tool use)
        self.initial_frames = list(frames)   # original video frames (for entropy k-sampling)
        self.current_prompt = initial_prompt
        self.tool_calls     = 0
        self.answer         = None
        self.gate_skipped   = False
        self.gate_answer    = None


# ═══════════════════════════════════════════════════════════════════════════════
# Main inference
# ═══════════════════════════════════════════════════════════════════════════════

def run(args):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_qwen35_lvr.jsonl")
    acc_path     = os.path.join(args.output_dir, "accuracy.txt")

    samples = load_dataset(args.data_path)
    print(f"[data] {len(samples)} samples from {args.data_path}")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Allow initial frames + up to MAX_TOOL_CALLS × 8 tool frames
    max_mm_images = args.max_frames + MAX_TOOL_CALLS * 8 + 4
    print(f"[vLLM] init model={args.model_path}, tp={args.tensor_parallel_size}, "
          f"max_pixels={args.max_pixels}, min_pixels={args.min_pixels}, "
          f"max_frames={args.max_frames}, max_mm_images={max_mm_images}")

    llm = LLM(
        model                  = args.model_path,
        tensor_parallel_size   = args.tensor_parallel_size,
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_model_len          = args.max_model_len,
        dtype                  = "bfloat16",
        trust_remote_code      = True,
        mm_processor_kwargs    = {
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
        },
        limit_mm_per_prompt    = {"image": max_mm_images},
        enforce_eager          = False,
        enable_prefix_caching  = False,
    )

    greedy_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
        include_stop_str_in_output=False,
        detokenize=True,
    )
    sample_sp = SamplingParams(
        n=args.entropy_gate_k,
        temperature=args.entropy_gate_temperature,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
        include_stop_str_in_output=False,
        detokenize=True,
    ) if args.entropy_gate_k > 0 else None

    # ── Resume ────────────────────────────────────────────────────────────────
    done_pids, file_mode = set(), "w"
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                try:
                    done_pids.add(str(json.loads(line)["problem_id"]))
                except Exception:
                    pass
        if done_pids:
            print(f"[resume] {len(done_pids)} done, skipping")
            file_mode = "a"

    pending = [
        s for s in samples
        if str(s.get("problem_id", s.get("extra_info", {}).get("problem_id", ""))) not in done_pids
    ]
    print(f"[run] {len(pending)} pending, batch_size={args.batch_size}, "
          f"entropy_gate_k={args.entropy_gate_k} "
          f"({'answer-entropy gate' if args.entropy_gate_k > 0 else 'pure tool-call'})")

    n_correct = n_total = n_gate_skipped = n_tool_calls = 0

    with open(output_jsonl, file_mode) as out_f:
        for batch_start in tqdm(
            range(0, len(pending), args.batch_size),
            desc="batches",
            total=(len(pending) + args.batch_size - 1) // args.batch_size,
        ):
            batch   = pending[batch_start: batch_start + args.batch_size]
            active  = []

            # ── Preprocess ────────────────────────────────────────────────── #
            for samp in batch:
                pid   = str(samp.get("problem_id", samp.get("extra_info", {}).get("problem_id", "?")))
                gt    = samp.get("solution", "")
                q     = samp.get("problem", "")
                vrel  = (samp.get("videos") or [""])[0]
                vpath = resolve_video_path(vrel, args.video_root)
                try:
                    _, frames = load_video_frames(
                        vpath, args.fps, args.max_pixels, args.min_pixels, args.max_frames
                    )
                    prompt = build_initial_prompt(q, frames, processor, args.enable_thinking)
                    state  = SampleState(pid, gt, vpath, q, prompt, frames)
                    active.append(state)
                except Exception as e:
                    print(f"\n[prep] skip {pid}: {e}")

            if not active:
                continue

            # ══════════════════════════════════════════════════════════════════
            # Agentic loop  (≤ MAX_TOOL_CALLS rounds)
            # ══════════════════════════════════════════════════════════════════
            from collections import Counter
            done = []

            for _round in range(MAX_TOOL_CALLS + 1):
                if not active:
                    break

                inputs = [
                    {"prompt": s.current_prompt, "multi_modal_data": {"image": list(s.frames)}}
                    for s in active
                ]
                r1_outputs = llm.generate(inputs, greedy_sp)

                direct_done  = []    # (state, text)  — answered or no tool
                gate_pending = []    # (state, text, tc)  — need entropy gate
                tool_pending = []    # (state, text, tc)  — gate disabled, go straight to tool

                for state, out in zip(active, r1_outputs):
                    text = out.outputs[0].text
                    ans  = extract_answer(text)

                    if ans or _round == MAX_TOOL_CALLS:
                        state.answer = ans or "?"
                        direct_done.append((state, text))
                        continue

                    tc = parse_tool_call(text)
                    if tc is None:
                        state.answer = "?"
                        direct_done.append((state, text))
                        continue

                    if args.entropy_gate_k > 0:
                        gate_pending.append((state, text, tc))
                    else:
                        tool_pending.append((state, text, tc))

                # ── Batch entropy gate checks ───────────────────────────── #
                if gate_pending:
                    gate_inputs = [
                        {"prompt": s.current_prompt, "multi_modal_data": {"image": list(s.frames)}}
                        for s, _, _ in gate_pending
                    ]
                    gate_outs = llm.generate(gate_inputs, sample_sp)

                    for (state, r1_text, tc), gout in zip(gate_pending, gate_outs):
                        k_texts   = [o.text for o in gout.outputs]
                        answers_k = [a for t in k_texts if (a := extract_answer(t)) is not None]
                        H         = compute_answer_entropy(answers_k)
                        score     = -H

                        if score > args.entropy_gate_threshold and answers_k:
                            majority = Counter(answers_k).most_common(1)[0][0]
                            state.answer       = majority
                            state.gate_skipped = True
                            state.gate_answer   = majority
                            direct_done.append((state, r1_text))
                        else:
                            tool_pending.append((state, r1_text, tc))

                done.extend(direct_done)

                # ── Dispatch tools ──────────────────────────────────────── #
                next_active = []
                for state, r1_text, tc in tool_pending:
                    tool_frames, tool_desc = dispatch_tool(tc, state.video_path, args.max_pixels)
                    state.tool_calls += 1
                    is_last = (state.tool_calls >= MAX_TOOL_CALLS)

                    if not tool_frames:
                        state.answer = "?"
                        done.append((state, r1_text))
                        continue

                    new_prompt = build_continuation_prompt(
                        state.current_prompt, r1_text, tool_frames,
                        tool_desc, state.question, is_last, processor,
                    )
                    state.current_prompt = new_prompt
                    state.frames         = state.frames + tool_frames
                    next_active.append(state)

                active = next_active

            # Flush remaining active as done
            for state in active:
                state.answer = state.answer or "?"
                done.append((state, ""))

            # ── Score + write ─────────────────────────────────────────────── #
            for state, _ in done:
                gt_letter = extract_answer(state.gt) or state.gt.strip()
                correct   = (state.answer == gt_letter)
                n_correct    += int(correct)
                n_total      += 1
                n_gate_skipped += int(state.gate_skipped)
                n_tool_calls   += state.tool_calls

                rec = {
                    "problem_id":   state.pid,
                    "answer":       state.answer,
                    "gt":           state.gt,
                    "correct":      correct,
                    "tool_calls":   state.tool_calls,
                    "gate_skipped": state.gate_skipped,
                }
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()

    # ── Final summary ─────────────────────────────────────────────────────── #
    acc = n_correct / max(n_total, 1)
    avg_tc = n_tool_calls / max(n_total, 1)
    gate_pct = n_gate_skipped / max(n_total, 1) * 100

    summary = (
        f"accuracy={acc:.4f}  ({n_correct}/{n_total})\n"
        f"avg_tool_calls={avg_tc:.3f}\n"
        f"gate_skipped={n_gate_skipped}/{n_total} ({gate_pct:.1f}%)\n"
    )
    print(summary)
    with open(acc_path, "w") as f:
        f.write(summary)

    return acc


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="Qwen/Qwen3.5-4B")
    p.add_argument("--output_dir",  default="./infer_results/qwen35_lvr")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--max_model_len",          type=int,   default=32768)

    # Video params — aligned to lmms-eval run_lvr_tool*.sh
    p.add_argument("--max_pixels",  type=int,   default=602112)
    p.add_argument("--min_pixels",  type=int,   default=200704)
    p.add_argument("--max_frames",  type=int,   default=128)
    p.add_argument("--fps",         type=float, default=0.5)
    p.add_argument("--max_tokens",  type=int,   default=2048)

    # Thinking
    p.add_argument("--enable_thinking", action="store_true", default=True)
    p.add_argument("--no_thinking",     dest="enable_thinking", action="store_false")

    # Entropy gate — set k=0 to disable (pure tool-call)
    p.add_argument("--entropy_gate_k",           type=int,   default=5,
                   help="k temperature samples for entropy gate. 0 = disabled (pure tool-call).")
    p.add_argument("--entropy_gate_threshold",   type=float, default=-0.30,
                   help="score=-H_answer; score>threshold → confident → skip tool.")
    p.add_argument("--entropy_gate_temperature", type=float, default=0.7)

    p.add_argument("--batch_size", type=int, default=32)

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
