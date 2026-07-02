#!/usr/bin/env python3
"""
main_infer_qwen35_tool.py  —  Answer-entropy-gated Qwen3.5-4B tool

Architecture (generalises the VideoZoomer zoom-gate to an LM tool):

  Round 1  [TOOL_SYS, greedy, T=0]
    - Main VLM sees video + question.
    - If model gives a direct <answer> → done (confident, no tool needed).
    - If model calls <video_zoom> → uncertain signal → run answer_entropy gate.

  Answer-entropy gate  (k temperature samples)
    - Build k force-answer prompts: "zoom unavailable, answer now".
    - Extract MC answer (A/B/C/D) from each chain.
    - H = -sum p_i * log(p_i)  over A/B/C/D distribution.
    - score = -H   (higher = more confident).
    - score > threshold  →  SKIP tool  → majority-vote answer, done.
    - score ≤ threshold  →  CALL Qwen3.5-4B tool.

  Qwen3.5-4B tool call  (lazy singleton, bfloat16)
    - Receives original video frames + question.
    - Generates temporal analysis using internal CoT thinking.
    - Returns structured analysis text (think tokens stripped).

  Round 2  [TOOL_SYS, greedy, T=0]
    - Main VLM sees R1 context + Qwen3.5-4B analysis as <tool_response>.
    - Produces final <answer>.

Key properties:
  ✓ Entropy gate identical to answer_entropy_zoom baseline → direct comparison.
  ✓ Tool is a multimodal LM (not a pixel-manipulation function) → new modality.
  ✓ No MCP server / external API — Qwen3.5-4B loaded in-process via transformers.
  ✓ GPU budget: vLLM @ 0.7 utilisation on 2×A40 ≈ 56 GB reserved,
               Qwen3.5-4B @ bf16 ≈ 8 GB weights → fits in remaining ~24 GB.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from main_infer_adaptive_zoom import (
    TOOL_SYS,
    load_dataset,
    resolve_video_path,
    load_video_frames,
    build_tool_initial_prompt,
    extract_mc_answer,
    parse_zoom_call,
    score_answer,
    _frame_tokens,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3.5-4B tool — lazy singleton loader + inference helper
# ═══════════════════════════════════════════════════════════════════════════════

_QWEN35_MODEL     = None
_QWEN35_PROCESSOR = None

QWEN35_ANALYSIS_PROMPT = (
    "You are a specialist in temporal video analysis. "
    "A video-question-answering system is uncertain about this question:\n\n"
    "Question: {question}\n\n"
    "Watch the video carefully, then:\n"
    "1. Describe the key temporal events you observe and their approximate timestamps.\n"
    "2. Note any object interactions, state changes, or motion patterns that are relevant.\n"
    "3. Based solely on what you see, reason through which answer choice is correct.\n\n"
    "Be precise and concise."
)

QWEN35_DEFAULT_PATH = "Qwen/Qwen3.5-4B"


def _load_qwen35(model_path: str):
    """Load Qwen3.5-4B once (in-process singleton, after vLLM is initialised)."""
    global _QWEN35_MODEL, _QWEN35_PROCESSOR
    if _QWEN35_MODEL is None:
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info  # noqa: F401 (import to verify available)
        print(f"[qwen35] Loading {model_path} …")
        _QWEN35_PROCESSOR = AutoProcessor.from_pretrained(model_path)
        _QWEN35_MODEL = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        print("[qwen35] Model ready.")
    return _QWEN35_MODEL, _QWEN35_PROCESSOR


def call_qwen35_analysis(
    frames,          # list[PIL.Image]
    question: str,
    model_path: str,
    max_new_tokens: int = 512,
) -> str:
    """
    Call Qwen3.5-4B with video frames + question → return analysis text.

    Thinking tokens are stripped; only the final analysis is returned.
    Returns empty string on failure (caller falls back to force-answer path).
    """
    from qwen_vl_utils import process_vision_info
    try:
        model, processor = _load_qwen35(model_path)

        content = [{"type": "image", "image": f} for f in frames]
        content.append({
            "type": "text",
            "text": QWEN35_ANALYSIS_PROMPT.format(question=question),
        })
        messages = [
            {"role": "system", "content": "You are a helpful video analysis assistant."},
            {"role": "user",   "content": content},
        ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        # Frames are PIL images, no video files — video_inputs will be None
        image_inputs, _, _ = process_vision_info(
            messages, return_video_kwargs=True, image_patch_size=16, return_video_metadata=True
        )

        inputs = processor(
            text=text_input,
            images=image_inputs,
            do_resize=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs.input_ids.shape[1]:]
        decoded = processor.tokenizer.decode(generated, skip_special_tokens=True)

        # Strip thinking block
        if "</think>" in decoded:
            decoded = decoded.split("</think>", 1)[1].strip()

        return decoded.strip()

    except Exception as e:
        print(f"[qwen35] call failed: {e}")
        return ""


def build_qwen35_tool_response_turn(analysis: str) -> str:
    """
    User turn that delivers Qwen3.5-4B analysis and asks for final answer.
    Mirrors build_tool_response_turn but with text analysis instead of image frames.
    """
    return (
        "\n<|im_end|>\n<|im_start|>user\n"
        "<tool_response>\n"
        "Temporal analysis from a specialised video reasoning model:\n"
        f"{analysis}\n"
        "</tool_response>\n"
        "Based on the above analysis, continue your reasoning inside <think> and </think>, "
        "then write your final answer inside <answer> and </answer>. "
        "Do not call <video_zoom> in this round."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Answer distribution entropy (identical to main_infer_hmm_zoom.py)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_answer_dist_entropy(texts: list) -> dict:
    """k temperature-sample texts → H(P(A/B/C/D)), majority answer, n_valid."""
    answers = [extract_mc_answer(t) for t in texts]
    answers = [a for a in answers if a and a in "ABCD"]

    if not answers:
        return {
            "answer_entropy_score": -np.log(4),
            "H_answer":             np.log(4),
            "answer_dist":          {c: 0.25 for c in "ABCD"},
            "majority_answer":      None,
            "n_valid":              0,
        }

    counts  = Counter(answers)
    total   = len(answers)
    dist    = {c: counts.get(c, 0) / total for c in "ABCD"}
    H       = -sum(p * np.log(p) for p in dist.values() if p > 0)
    majority = counts.most_common(1)[0][0]

    return {
        "answer_entropy_score": -H,
        "H_answer":              H,
        "answer_dist":           dist,
        "majority_answer":       majority,
        "n_valid":               total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════════

class SampleState:
    def __init__(self, pid, gt, video_path, prompt, images, question=""):
        self.pid            = pid
        self.gt             = gt
        self.video_path     = video_path
        self.prompt         = prompt
        self.images         = images
        self.question       = question
        self.n_tool_calls   = 0
        self.n_rounds       = 0
        self.final_answer   = None
        self.acc_final      = None
        self.raw_output     = ""
        # gate outcome flags
        self.tool_skipped   = False   # entropy gate → skip Qwen3.5-4B
        self.tool_triggered = False   # entropy gate → call Qwen3.5-4B
        self.direct_answer  = False   # R1 gave direct answer, no zoom signal
        self.hmm_score      = None
        self.hmm_features   = None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--model_path",  required=True)
    p.add_argument("--video_root",  required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--qwen35_path", default=QWEN35_DEFAULT_PATH,
                   help="HuggingFace model path/ID for Qwen3.5-4B tool.")

    # vLLM engine
    p.add_argument("--gpu_memory_utilization", type=float, default=0.6,
                   help="Lower than zoom scripts to leave room for Qwen3.5-4B (recommend ≤0.65).")
    p.add_argument("--tensor_parallel_size",   type=int,   default=2)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=100352)
    p.add_argument("--min_pixels",             type=int,   default=25088)

    # Video sampling
    p.add_argument("--fps",            type=float, default=0.5)
    p.add_argument("--frames_upbound", type=int,   default=64)
    p.add_argument("--max_tokens",     type=int,   default=4096)

    # Answer entropy gate
    p.add_argument("--entropy_threshold",  type=float, default=-0.30,
                   help="score=-H_answer > threshold → skip Qwen3.5-4B tool.")
    p.add_argument("--answer_k",           type=int,   default=5,
                   help="Number of temperature samples for entropy estimation.")
    p.add_argument("--answer_temperature", type=float, default=0.7,
                   help="Sampling temperature for entropy estimation.")

    # Qwen3.5-4B tool
    p.add_argument("--qwen35_max_tokens", type=int, default=512,
                   help="Max tokens for Qwen3.5-4B analysis output.")

    p.add_argument("--batch_size", type=int, default=16,
                   help="Batch size for vLLM main-model calls.")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_qwen35_tool.jsonl")

    # ── vLLM engine ──────────────────────────────────────────────────────── #
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )

    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        limit_mm_per_prompt={"image": 128},
        dtype="bfloat16",
    )

    # Sampling params
    greedy_sp = SamplingParams(
        n=1, temperature=0, max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
    )
    answer_sp = SamplingParams(
        n=args.answer_k,
        temperature=args.answer_temperature,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
    )
    force_ans_sp = SamplingParams(
        n=1, temperature=0, max_tokens=512,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
    )

    _FORCE_ANS_TURN = (
        "\n<|im_end|>\n<|im_start|>user\n"
        "<tool_response>\n"
        "The zoom tool is unavailable. Based on the video frames "
        "already shown, give your best final answer now.\n"
        "</tool_response>\n"
        "Do not call <video_zoom>. "
        "Write your final answer inside <answer> and </answer>."
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    # ── Dataset ──────────────────────────────────────────────────────────── #
    samples = load_dataset(args.data_path)
    print(f"[run]  {len(samples)} samples, answer_entropy gate → Qwen3.5-4B tool")
    print(f"[run]  threshold={args.entropy_threshold}, k={args.answer_k}, "
          f"T={args.answer_temperature}")
    print(f"[run]  qwen35_path={args.qwen35_path}")

    # ── Resume ───────────────────────────────────────────────────────────── #
    done_pids, file_mode = set(), "w"
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                try:
                    done_pids.add(json.loads(line)["problem_id"])
                except Exception:
                    pass
        if done_pids:
            print(f"[resume] {len(done_pids)} samples already done")
            file_mode = "a"

    pending = [
        s for s in samples
        if str(s.get("problem_id", "")) not in done_pids
        and str(s.get("extra_info", {}).get("problem_id", "")) not in done_pids
    ]

    all_records = []
    n_correct = n_tool_triggered = n_tool_skipped = n_direct = 0

    with open(output_jsonl, file_mode) as out_f:
        for batch_start in tqdm(
            range(0, len(pending), args.batch_size),
            desc="Batches",
            total=(len(pending) + args.batch_size - 1) // args.batch_size,
        ):
            batch = pending[batch_start: batch_start + args.batch_size]

            # ── Preprocess ───────────────────────────────────────────────── #
            active = []
            for sample in batch:
                pid       = (sample.get("problem_id") or
                             sample.get("extra_info", {}).get("problem_id", "?"))
                gt        = sample.get("solution", "")
                question  = sample.get("problem", "")
                video_rel = (sample.get("videos") or [""])[0]
                video_path = resolve_video_path(video_rel, args.video_root)
                try:
                    frame_times, frames = load_video_frames(
                        video_path, args.fps, args.max_pixels, args.min_pixels,
                        args.frames_upbound,
                    )
                    prompt = build_tool_initial_prompt(question, frame_times, processor)
                    state  = SampleState(pid, gt, video_path, prompt, list(frames),
                                         question=question)
                    active.append(state)
                except Exception as e:
                    print(f"\n[prep] skip {pid}: {e}")

            if not active:
                continue

            done = []

            # ── R1: greedy main-model ────────────────────────────────────── #
            r1_inputs = [
                {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                for s in active
            ]
            r1_outputs = llm.generate(r1_inputs, greedy_sp)

            # Deferred: zoom-signal samples waiting for entropy gate
            ae_deferred = []   # (s, r1_text)

            for s, out in zip(active, r1_outputs):
                text = out.outputs[0].text
                s.raw_output = text
                s.n_rounds   = 1

                zoom_call = parse_zoom_call(text)
                if zoom_call is None:
                    # Model gave direct answer — done
                    s.direct_answer = True
                    s.final_answer  = extract_mc_answer(text)
                    s.acc_final     = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                    done.append(s)
                else:
                    ae_deferred.append((s, text))

            # ── Answer-entropy gate (batch) ───────────────────────────────── #
            if ae_deferred:
                # Build force-answer prompts for all zoom-signal samples
                fa_inputs = []
                for s, text in ae_deferred:
                    end_pos  = text.find("</video_zoom>") + len("</video_zoom>")
                    fa_prompt = s.prompt + text[:end_pos] + _FORCE_ANS_TURN
                    fa_inputs.append({
                        "prompt": fa_prompt,
                        "multi_modal_data": {"image": list(s.images)},
                    })

                fa_outputs = llm.generate(fa_inputs, answer_sp)

                qwen35_queue = []   # (s, r1_text) samples that need Qwen3.5-4B

                for (s, r1_text), fa_out in zip(ae_deferred, fa_outputs):
                    chain_texts = [c.text for c in fa_out.outputs]
                    feats = compute_answer_dist_entropy(chain_texts)
                    score = feats["answer_entropy_score"]

                    s.hmm_score    = score
                    s.hmm_features = feats

                    if score > args.entropy_threshold:
                        # Confident → skip Qwen3.5-4B, use majority vote
                        s.tool_skipped = True
                        majority = feats["majority_answer"]
                        if majority:
                            s.final_answer = majority
                            s.acc_final    = score_answer(majority, s.gt)
                            done.append(s)
                        else:
                            # All k chains truncated → fallback force-answer
                            end_pos   = r1_text.find("</video_zoom>") + len("</video_zoom>")
                            s.prompt += r1_text[:end_pos] + _FORCE_ANS_TURN
                            fa_single = llm.generate(
                                [{"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}],
                                force_ans_sp,
                            )
                            fa_text       = fa_single[0].outputs[0].text
                            s.raw_output  = fa_text
                            s.n_rounds    = 2
                            s.final_answer = extract_mc_answer(fa_text)
                            s.acc_final   = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                            done.append(s)
                    else:
                        # Uncertain → enqueue for Qwen3.5-4B tool call
                        qwen35_queue.append((s, r1_text))

                # ── Qwen3.5-4B tool calls (sequential, CPU↔GPU) ──────────── #
                if qwen35_queue:
                    print(f"\n[qwen35] calling tool for {len(qwen35_queue)} samples …")
                    # Lazy-load Qwen3.5-4B now that vLLM R1 is done
                    _load_qwen35(args.qwen35_path)

                    r2_inputs = []
                    r2_states = []
                    for s, r1_text in qwen35_queue:
                        analysis = call_qwen35_analysis(
                            list(s.images), s.question,
                            args.qwen35_path, args.qwen35_max_tokens,
                        )
                        end_pos   = r1_text.find("</video_zoom>") + len("</video_zoom>")
                        if analysis:
                            tool_turn = build_qwen35_tool_response_turn(analysis)
                        else:
                            # Qwen3.5-4B failed → degrade to force-answer
                            tool_turn = _FORCE_ANS_TURN

                        s.prompt    += r1_text[:end_pos] + tool_turn
                        s.tool_triggered = True
                        s.n_tool_calls   = 1
                        r2_inputs.append({
                            "prompt": s.prompt,
                            "multi_modal_data": {"image": list(s.images)},
                        })
                        r2_states.append(s)

                    # ── R2: main model with Qwen3.5-4B analysis ─────────── #
                    r2_outputs = llm.generate(r2_inputs, greedy_sp)
                    for s, r2_out in zip(r2_states, r2_outputs):
                        r2_text        = r2_out.outputs[0].text
                        s.raw_output   = r2_text
                        s.n_rounds     = 2
                        s.final_answer = extract_mc_answer(r2_text)
                        s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

            # ── Record results ───────────────────────────────────────────── #
            for s in done:
                if s.acc_final:
                    n_correct += 1
                if s.tool_triggered:
                    n_tool_triggered += 1
                if s.tool_skipped:
                    n_tool_skipped += 1
                if s.direct_answer:
                    n_direct += 1

                record = {
                    "problem_id":     s.pid,
                    "gt":             s.gt,
                    "final_answer":   s.final_answer,
                    "acc_final":      s.acc_final,
                    "n_rounds":       s.n_rounds,
                    "n_tool_calls":   s.n_tool_calls,
                    "direct_answer":  s.direct_answer,
                    "tool_skipped":   s.tool_skipped,
                    "tool_triggered": s.tool_triggered,
                    "hmm_score":      s.hmm_score,
                    "hmm_features":   s.hmm_features,
                }
                all_records.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────── #
    total = len(all_records)
    print("\n" + "=" * 60)
    print(f"Qwen3.5-4B Tool Inference ({total} samples)")
    print(f"  Accuracy        : {n_correct}/{total} = {n_correct/max(total,1):.4f}")
    print(f"  Direct answer   : {n_direct}/{total}  ({100*n_direct/max(total,1):.1f}%)")
    print(f"  Tool skipped    : {n_tool_skipped}/{total}  ({100*n_tool_skipped/max(total,1):.1f}%)")
    print(f"  Tool called     : {n_tool_triggered}/{total}  ({100*n_tool_triggered/max(total,1):.1f}%)")
    avg_calls = sum(r["n_tool_calls"] for r in all_records) / max(total, 1)
    print(f"  Avg Qwen35 calls: {avg_calls:.3f}")
    print("=" * 60)
    print(f"Results → {output_jsonl}")


if __name__ == "__main__":
    main()
