#!/usr/bin/env python3
"""
main_infer_margin_zoom.py  ─  Answer-token margin gated zoom inference

Single greedy pass (T=0) with logprobs.
At the answer token (A/B/C/D position), compute:
    margin = logp(rank1) - logp(rank2)

Decision:
  - model already called zoom  →  execute zoom, continue normally
  - margin >= threshold        →  confident, output as-is  (single pass)
  - margin <  threshold        →  inject "reconsider" turn, re-infer greedy
                                  (model may now call zoom with more context)

No temperature sampling, no branching pass.
Equivalent to majority-voting signal but derived from a single forward pass.
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from main_infer_adaptive_zoom import (
    load_dataset,
    resolve_video_path,
    load_video_frames,
    build_tool_initial_prompt,
    build_tool_response_turn,
    extract_mc_answer,
    parse_zoom_call,
    score_answer,
)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results/margin_zoom")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=2)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=100352)
    p.add_argument("--min_pixels",             type=int,   default=25088)

    # Video & tool
    p.add_argument("--fps",                      type=float, default=0.5)
    p.add_argument("--frames_upbound",           type=int,   default=64)
    p.add_argument("--max_tokens",               type=int,   default=4096)
    p.add_argument("--tool_limit_mm",            type=int,   default=128)
    p.add_argument("--tool_max_frames_per_call", type=int,   default=16)
    p.add_argument("--tool_workers",             type=int,   default=8)
    p.add_argument("--max_rounds",               type=int,   default=5)

    # Margin threshold
    p.add_argument("--margin_threshold", type=float, default=2.0,
                   help="log-prob margin logp(rank1)-logp(rank2) at answer token. "
                        "Below this → inject reconsider turn. "
                        "margin=2.0 means rank1 is ~7x more likely than rank2.")
    p.add_argument("--logprobs_k", type=int, default=5,
                   help="Top-K logprobs per token (only need 2, 5 for safety)")

    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Answer-token margin
# ═══════════════════════════════════════════════════════════════════════════

def get_answer_token_margin(logprobs_list, expected_answer):
    """
    Scan logprobs from the end to find the last position where the generated
    token is the expected answer letter (A/B/C/D).
    Returns (margin, position_idx) or (None, None) if not found.

    margin = logp(rank1) - logp(rank2)
    High margin → confident.  Low margin → uncertain.
    """
    if not logprobs_list or expected_answer is None:
        return None, None

    for i in reversed(range(len(logprobs_list))):
        lp_dict = logprobs_list[i]
        if not lp_dict:
            continue
        # Sort by logprob descending
        items = sorted(lp_dict.values(), key=lambda x: x.logprob, reverse=True)
        if not items:
            continue
        top_tok = (items[0].decoded_token or "").strip()
        if top_tok == expected_answer:
            if len(items) >= 2:
                margin = items[0].logprob - items[1].logprob
            else:
                margin = 10.0   # only one candidate → very confident
            return float(margin), i

    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Reconsider turn (injected when margin < threshold)
# ═══════════════════════════════════════════════════════════════════════════

def build_reconsider_turn(is_last: bool) -> str:
    """
    Appended after a low-confidence Pass-1 answer.
    Closes the assistant turn, opens a user turn prompting reconsideration,
    then opens a new assistant turn.
    """
    msg = (
        "<|im_end|>\n<|im_start|>user\n"
        "Your answer seems uncertain. Please re-examine the video carefully. "
        "You may use <video_zoom> to zoom into key segments if needed, "
        "then provide your final answer inside <answer> and </answer>."
    )
    if is_last:
        msg += " Do not call <video_zoom> in this round."
    msg += "<|im_end|>\n<|im_start|>assistant\n"
    return msg


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════

class SampleState:
    def __init__(self, pid, gt, video_path, prompt, images):
        self.pid            = pid
        self.gt             = gt
        self.video_path     = video_path
        self.prompt         = prompt
        self.images         = images
        self.n_tool_calls   = 0
        self.n_rounds       = 0
        self.final_answer   = None
        self.acc_final      = None
        self.raw_output     = ""
        # Margin metadata
        self.zoom_triggered = False     # margin < threshold triggered reconsider
        self.answer_margin  = None      # margin at Pass-1 answer token
        self.margin_checked = False     # have we already done margin check this sample?


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_margin.jsonl")

    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout.function_tools import extract_video_clip

    max_mm = max(args.frames_upbound, args.tool_limit_mm)
    print(f"[vLLM]  tp={args.tensor_parallel_size}, util={args.gpu_memory_utilization}, "
          f"limit_mm={max_mm}")
    llm = LLM(
        model                  = args.model_path,
        tensor_parallel_size   = args.tensor_parallel_size,
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_model_len          = args.max_model_len,
        dtype                  = "bfloat16",
        trust_remote_code      = True,
        mm_processor_kwargs    = {"max_pixels": args.max_pixels, "min_pixels": args.min_pixels},
        limit_mm_per_prompt    = {"image": max_mm},
        enforce_eager          = False,
        enable_prefix_caching  = False,
    )

    # Greedy + logprobs (Pass-1 and all subsequent passes)
    greedy_sp = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=args.logprobs_k,
    )

    # Greedy without logprobs for rounds after zoom (save memory, no margin check needed)
    greedy_nolog_sp = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
    )

    print(f"[run]   margin_threshold={args.margin_threshold}, "
          f"logprobs_k={args.logprobs_k}, batch_size={args.batch_size}")

    # Resume
    done_pids = set()
    file_mode = "w"
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                try:
                    done_pids.add(json.loads(line)["problem_id"])
                except Exception:
                    pass
        if done_pids:
            print(f"[resume] {len(done_pids)} samples already done, skipping")
            file_mode = "a"

    pending = [
        s for s in samples
        if str(s.get("problem_id", "")) not in done_pids
        and str(s.get("extra_info", {}).get("problem_id", "")) not in done_pids
    ]

    all_records     = []
    n_correct       = 0
    n_zoom_triggered = 0

    with open(output_jsonl, file_mode) as out_f:
        for batch_start in tqdm(
            range(0, len(pending), args.batch_size),
            desc="Batches",
            total=(len(pending) + args.batch_size - 1) // args.batch_size,
        ):
            batch = pending[batch_start: batch_start + args.batch_size]

            # ── Preprocess ────────────────────────────────────────────── #
            active = []
            for sample in batch:
                pid = (sample.get("problem_id") or
                       sample.get("extra_info", {}).get("problem_id", "?"))
                gt        = sample.get("solution", "")
                question  = sample.get("problem", "")
                video_rel = (sample.get("videos") or [""])[0]
                video_path = resolve_video_path(video_rel, args.video_root)
                try:
                    frame_times, frames = load_video_frames(
                        video_path, args.fps, args.max_pixels, args.min_pixels,
                        args.frames_upbound)
                    prompt = build_tool_initial_prompt(question, frame_times, processor)
                    active.append(SampleState(pid, gt, video_path, prompt, list(frames)))
                except Exception as e:
                    print(f"\n[prep] skip {pid}: {e}")

            if not active:
                continue

            done = []

            # ── Multi-round inference ──────────────────────────────────── #
            for round_idx in range(1, args.max_rounds + 1):
                is_last = (round_idx == args.max_rounds)

                # Use logprobs only when we still need to do margin check
                need_logprobs = [not s.margin_checked for s in active]
                sp = greedy_sp if any(need_logprobs) else greedy_nolog_sp

                inputs = [
                    {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                    for s in active
                ]
                outputs = llm.generate(inputs, sp)

                zoom_queue  = {}
                next_active = []

                for idx, (s, out) in enumerate(zip(active, outputs)):
                    text     = out.outputs[0].text
                    logprobs = out.outputs[0].logprobs or []
                    s.n_rounds   = round_idx
                    s.raw_output = text

                    zoom_call = parse_zoom_call(text)

                    # ── Margin check (only once per sample, before any zoom) ──
                    if not s.margin_checked and zoom_call is None and not is_last:
                        answer = extract_mc_answer(text)
                        margin, _ = get_answer_token_margin(logprobs, answer)
                        s.answer_margin  = margin
                        s.margin_checked = True

                        if margin is not None and margin < args.margin_threshold:
                            # Low confidence → inject reconsider turn, re-queue
                            s.zoom_triggered = True
                            s.prompt += text + build_reconsider_turn(is_last=False)
                            next_active.append(s)
                            continue   # don't route to done yet

                    s.margin_checked = True   # skip margin check in later rounds

                    if zoom_call is not None and not is_last:
                        end_pos   = text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += text[:end_pos]
                        zoom_queue[idx] = (s, zoom_call)
                    else:
                        s.final_answer = extract_mc_answer(text)
                        s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

                # ── Execute zoom clips ─────────────────────────────────── #
                if zoom_queue:
                    with ThreadPoolExecutor(max_workers=args.tool_workers) as ex:
                        futures = {}
                        for idx, (s, (s_t, e_t, fps_z)) in zoom_queue.items():
                            f = ex.submit(
                                extract_video_clip,
                                video_path     = s.video_path,
                                start_time     = s_t,
                                end_time       = e_t,
                                fps            = fps_z,
                                max_pixels     = args.max_pixels,
                                min_pixels     = args.min_pixels,
                                max_frames     = args.tool_max_frames_per_call,
                                storage_system = "local",
                            )
                            futures[f] = idx
                        zoom_results = {futures[f]: f.result() for f in as_completed(futures)}

                    for idx, (s, _) in zoom_queue.items():
                        result = zoom_results.get(idx)
                        if isinstance(result, dict):
                            times, frames = result["frame_time"], result["frames"]
                            s.prompt  += build_tool_response_turn(times, is_last=False)
                            s.images  += list(frames)
                            s.n_tool_calls += 1
                            next_active.append(s)
                        else:
                            s.final_answer = extract_mc_answer(s.raw_output)
                            s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                            done.append(s)

                active = next_active
                if not active:
                    break

            # Force-finalize remaining
            for s in active:
                s.final_answer = extract_mc_answer(s.raw_output)
                s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                done.append(s)

            # ── Save ──────────────────────────────────────────────────── #
            for s in done:
                n_correct += int(s.acc_final or 0)
                if s.zoom_triggered:
                    n_zoom_triggered += 1
                rec = {
                    "problem_id":     s.pid,
                    "gt":             s.gt,
                    "acc_final":      s.acc_final,
                    "final_answer":   s.final_answer,
                    "n_rounds":       s.n_rounds,
                    "n_tool_calls":   s.n_tool_calls,
                    "raw_output":     s.raw_output,
                    "zoom_triggered": s.zoom_triggered,
                    "answer_margin":  s.answer_margin,
                    # Schema compat
                    "branched":        s.zoom_triggered,
                    "branch_entropy":  s.answer_margin,
                    "branch_decision": "reconsider" if s.zoom_triggered else "main",
                    "acc_notool":      None,
                    "r1_majority_ans": s.final_answer,
                    "graduated_round": s.n_rounds,
                }
                all_records.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    # ── Summary ───────────────────────────────────────────────────────────── #
    done_this_run = len(all_records)
    print(f"\n{'='*60}")
    print(f"  Margin-zoom inference ({done_this_run} new samples)")
    if done_this_run:
        print(f"  Accuracy (this run): {n_correct}/{done_this_run} = "
              f"{n_correct/done_this_run:.4f}")
        print(f"  Zoom triggered: {n_zoom_triggered}/{done_this_run} "
              f"({n_zoom_triggered/done_this_run*100:.1f}%)")

    with open(output_jsonl) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    accs = [float(r["acc_final"]) for r in all_recs if r.get("acc_final") is not None]
    if accs:
        avg_rounds = np.mean([r.get("n_rounds", 1) for r in all_recs])
        avg_tools  = np.mean([r.get("n_tool_calls", 0) for r in all_recs])
        margins    = [r["answer_margin"] for r in all_recs if r.get("answer_margin") is not None]
        n_triggered = sum(1 for r in all_recs if r.get("zoom_triggered"))
        print(f"  Accuracy (full file, {len(accs)} samples): {np.mean(accs):.4f}")
        print(f"  Avg rounds: {avg_rounds:.2f},  Avg tool calls: {avg_tools:.2f}")
        print(f"  Zoom triggered total: {n_triggered}")
        if margins:
            print(f"  Answer margin stats: mean={np.mean(margins):.2f}  "
                  f"median={np.median(margins):.2f}  "
                  f"<threshold: {sum(1 for m in margins if m < args.margin_threshold)}")
    print(f"  Output: {output_jsonl}")
    print(f"{'='*60}")

    summary = {
        "n_samples":        len(accs),
        "accuracy":         float(np.mean(accs)) if accs else None,
        "mode":             "margin_zoom",
        "margin_threshold": args.margin_threshold,
        "logprobs_k":       args.logprobs_k,
        "fps":              args.fps,
        "frames_upbound":   args.frames_upbound,
        "max_rounds":       args.max_rounds,
        "model":            args.model_path,
    }
    with open(os.path.join(args.output_dir, "margin_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {args.output_dir}/margin_summary.json")


if __name__ == "__main__":
    main()
