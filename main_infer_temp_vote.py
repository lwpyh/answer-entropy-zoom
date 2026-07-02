#!/usr/bin/env python3
"""
main_infer_temp_vote.py  ─  Greedy answer + temperature-voting uncertainty gate

Round 1  [two batched vLLM calls, NOTOOL_SYS, stop=["</answer>"]]
  ① Greedy call (T=0, n=1)  → high-quality answer
  ② Vote call   (T=vote_temp, n=n_vote) → diversity for uncertainty signal
     agreement = #{vote samples matching greedy answer} / n_vote_valid
     agreement ≥ θ  →  confident, output greedy answer  (no zoom)
     agreement < θ  →  uncertain, trigger zoom rounds

Round 2+  [TOOL_SYS, greedy, stop=["</video_zoom>","</answer>"]]
  - Append Round 1 output (zoom tags stripped) + reconsider turn
  - Execute zoom clips, continue until answer or max_rounds

Key design
──────────
• Greedy answer preserves T=0 accuracy (no plurality degradation)
• Temperature diversity gives genuine uncertainty signal
• Default threshold 0.8 = at least 3/4 vote samples must agree with greedy
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from main_infer_adaptive_zoom import (
    TOOL_SYS,
    NOTOOL_SYS,
    load_dataset,
    resolve_video_path,
    load_video_frames,
    build_tool_response_turn,
    extract_mc_answer,
    parse_zoom_call,
    score_answer,
    _frame_tokens,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_agreement(answers: list):
    """
    Return (plurality_answer, agreement_fraction) from a list of answers.
    None values are ignored.  Returns (None, 1.0) if no valid answers.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 1.0
    top_ans, top_cnt = Counter(valid).most_common(1)[0]
    return top_ans, top_cnt / len(valid)


def build_r1_prompt(question: str, frame_times, processor) -> str:
    """Round 1: NOTOOL_SYS so model answers without calling zoom."""
    msgs = [
        {"role": "system",  "content": NOTOOL_SYS},
        {"role": "user",    "content": _frame_tokens(frame_times) + "\n" + question},
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_r2_prompt(question: str, frame_times, r1_text: str, processor, is_last: bool) -> str:
    """
    Round 2+: TOOL_SYS + prior R1 output (stripped of zoom tags) as assistant turn,
    then a reconsider user turn.
    """
    reconsider = (
        "Your previous answer may be uncertain. "
        "Please look more carefully at the video — use <video_zoom> if needed — "
        "and give your final answer in <answer>\\boxed{X}</answer>."
        if is_last else
        "Your previous answer may be uncertain. "
        "Use <video_zoom> to examine relevant segments and reconsider."
    )
    msgs = [
        {"role": "system",    "content": TOOL_SYS},
        {"role": "user",      "content": _frame_tokens(frame_times) + "\n" + question},
        {"role": "assistant", "content": r1_text},
        {"role": "user",      "content": reconsider},
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════

class SampleState:
    def __init__(self, pid, gt, video_path, r1_prompt, images, question="", frame_times=None):
        self.pid           = pid
        self.gt            = gt
        self.video_path    = video_path
        self.prompt        = r1_prompt
        self.images        = images
        self.question      = question
        self.frame_times   = frame_times or []
        self.n_tool_calls  = 0
        self.n_rounds      = 0
        self.final_answer  = None
        self.acc_final     = None
        self.raw_output    = ""
        self.zoom_triggered = False
        self.r1_agreement  = None
        self.r1_plurality  = None   # plurality answer from R1 vote


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",  required=True)
    p.add_argument("--video_root", default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path", default="zsgvivo/videozoomer")
    p.add_argument("--output_dir", default="./infer_results/temp_vote")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=2)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=200704)
    p.add_argument("--min_pixels",             type=int,   default=3136)

    # Video
    p.add_argument("--fps",           type=float, default=0.5)
    p.add_argument("--frames_upbound",type=int,   default=32)
    p.add_argument("--max_tokens",    type=int,   default=2048)

    # Tool
    p.add_argument("--max_rounds",           type=int,   default=3)
    p.add_argument("--tool_max_frames_per_call", type=int, default=8)
    p.add_argument("--tool_limit_mm",        type=int,   default=64)
    p.add_argument("--tool_workers",         type=int,   default=8)

    # Temperature voting
    p.add_argument("--n_vote",               type=int,   default=4,
                   help="Number of temperature samples used ONLY for agreement signal")
    p.add_argument("--vote_temp",            type=float, default=0.7,
                   help="Sampling temperature for vote samples")
    p.add_argument("--agreement_threshold",  type=float, default=0.8,
                   help="Min fraction of vote samples matching greedy (0.8=3/4, 1.0=all)")

    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_tvote.jsonl")

    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout.function_tools import extract_video_clip

    max_mm = max(args.frames_upbound, args.tool_limit_mm)
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

    # Round 1a: greedy answer (T=0, n=1) — used as final answer
    r1_greedy_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens   = args.max_tokens,
        stop         = ["</answer>"],
        include_stop_str_in_output = True,
        detokenize   = True,
    )

    # Round 1b: vote samples (T>0, n=n_vote) — used ONLY for agreement signal
    r1_vote_sp = SamplingParams(
        n            = args.n_vote,
        temperature  = args.vote_temp,
        max_tokens   = args.max_tokens,
        stop         = ["</answer>"],
        include_stop_str_in_output = True,
        detokenize   = True,
    )

    # Round 2+: greedy, TOOL_SYS
    tool_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens   = args.max_tokens,
        stop         = ["</video_zoom>", "</answer>"],
        include_stop_str_in_output = True,
        detokenize   = True,
    )

    print(f"[cfg]   n_vote={args.n_vote}  vote_temp={args.vote_temp}  "
          f"agreement_threshold={args.agreement_threshold}  batch_size={args.batch_size}")

    # Resume
    done_pids, file_mode = set(), "w"
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                try:   done_pids.add(json.loads(line)["problem_id"])
                except Exception: pass
        if done_pids:
            print(f"[resume] {len(done_pids)} done, skipping")
            file_mode = "a"

    pending = [
        s for s in samples
        if str(s.get("problem_id", "")) not in done_pids
        and str(s.get("extra_info", {}).get("problem_id", "")) not in done_pids
    ]

    all_records = []
    n_correct = n_zoom_triggered = 0

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
                pid      = (sample.get("problem_id") or
                            sample.get("extra_info", {}).get("problem_id", "?"))
                gt       = sample.get("solution", "")
                question = sample.get("problem", "")
                video_rel = (sample.get("videos") or [""])[0]
                video_path = resolve_video_path(video_rel, args.video_root)
                try:
                    frame_times, frames = load_video_frames(
                        video_path, args.fps, args.max_pixels, args.min_pixels,
                        args.frames_upbound)
                    r1_prompt = build_r1_prompt(question, frame_times, processor)
                    active.append(SampleState(pid, gt, video_path, r1_prompt,
                                              list(frames), question=question,
                                              frame_times=frame_times))
                except Exception as e:
                    print(f"\n[prep] skip {pid}: {e}")

            if not active:
                continue

            done = []

            # ══════════════════════════════════════════════════════════════
            # Round 1: greedy answer + temperature vote for uncertainty
            # ══════════════════════════════════════════════════════════════
            r1_inputs = [
                {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                for s in active
            ]
            # Two batched calls: greedy for answer quality, temperature for diversity
            r1_greedy_outputs = llm.generate(r1_inputs, r1_greedy_sp)
            r1_vote_outputs   = llm.generate(r1_inputs, r1_vote_sp)

            zoom_pending = []
            for s, greedy_out, vote_out in zip(active, r1_greedy_outputs, r1_vote_outputs):
                greedy_text   = greedy_out.outputs[0].text
                greedy_answer = extract_mc_answer(greedy_text)

                # Agreement = fraction of (greedy + vote_samples) matching greedy
                # Greedy counts as 1 vote; total pool = n_vote + 1
                vote_answers = [extract_mc_answer(o.text) for o in vote_out.outputs]
                if greedy_answer is not None:
                    all_answers = [greedy_answer] + [a for a in vote_answers if a is not None]
                    matching  = sum(1 for a in all_answers if a == greedy_answer)
                    agreement = matching / len(all_answers)  # greedy always matches itself
                else:
                    # Greedy produced no answer (called zoom) → use plurality of votes
                    greedy_answer, agreement = compute_agreement(vote_answers)
                    agreement = agreement if agreement is not None else 0.0

                s.raw_output   = greedy_text
                s.n_rounds     = 1
                s.r1_agreement = agreement
                s.r1_plurality = greedy_answer   # greedy answer

                if agreement >= args.agreement_threshold:
                    # Confident: greedy answer confirmed by vote
                    s.final_answer = greedy_answer
                    s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                    done.append(s)
                else:
                    # Uncertain: rebuild with TOOL_SYS for zoom rounds
                    clean_text = re.sub(
                        r'<video_zoom>.*?</video_zoom>', '', greedy_text, flags=re.DOTALL
                    ).strip()
                    s.prompt = build_r2_prompt(
                        s.question, s.frame_times, clean_text, processor, is_last=False
                    )
                    s.zoom_triggered = True
                    zoom_pending.append(s)

            # ══════════════════════════════════════════════════════════════
            # Round 2+: zoom-enabled greedy continuation
            # ══════════════════════════════════════════════════════════════
            active = zoom_pending

            for round_idx in range(2, args.max_rounds + 1):
                is_last = (round_idx == args.max_rounds)
                if not active:
                    break

                inputs = [
                    {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                    for s in active
                ]
                outputs = llm.generate(inputs, tool_sp)

                zoom_queue  = {}
                next_active = []

                for idx, (s, out) in enumerate(zip(active, outputs)):
                    text = out.outputs[0].text
                    s.n_rounds   = round_idx
                    s.raw_output = text

                    zoom_call = parse_zoom_call(text)
                    if zoom_call is not None and not is_last:
                        end_pos = text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += text[:end_pos]
                        zoom_queue[idx] = (s, zoom_call)
                    else:
                        s.final_answer = extract_mc_answer(text)
                        s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

                # Execute zoom clips (parallel)
                if zoom_queue:
                    with ThreadPoolExecutor(max_workers=args.tool_workers) as ex:
                        futures = {
                            ex.submit(
                                extract_video_clip,
                                video_path   = s.video_path,
                                start_time   = s_t,
                                end_time     = e_t,
                                fps          = fps_z,
                                max_pixels   = args.max_pixels,
                                min_pixels   = args.min_pixels,
                                max_frames   = args.tool_max_frames_per_call,
                                storage_system = "local",
                            ): idx
                            for idx, (s, (s_t, e_t, fps_z)) in zoom_queue.items()
                        }
                        zoom_results = {futures[f]: f.result() for f in as_completed(futures)}

                    for idx, (s, _) in zoom_queue.items():
                        result = zoom_results.get(idx)
                        if isinstance(result, dict):
                            times, frames = result["frame_time"], result["frames"]
                            s.prompt += build_tool_response_turn(times, is_last=False)
                            s.images += list(frames)
                            s.n_tool_calls += 1
                            next_active.append(s)
                        else:
                            s.final_answer = extract_mc_answer(s.raw_output)
                            s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                            done.append(s)

                active = next_active

            # Timeout: force-finish remaining
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
                    "r1_agreement":   s.r1_agreement,
                    "r1_plurality":   s.r1_plurality,
                }
                all_records.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    # ── Summary ─────────────────────────────────────────────────────────── #
    n_this = len(all_records)
    print(f"\n{'='*60}")
    print(f"  Temp-vote inference ({n_this} new samples)")
    if n_this:
        print(f"  Accuracy (this run):   {n_correct}/{n_this} = {n_correct/n_this:.4f}")
        print(f"  Zoom triggered:        {n_zoom_triggered}/{n_this} "
              f"({n_zoom_triggered/n_this*100:.1f}%)")

    with open(output_jsonl) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    accs = [float(r["acc_final"]) for r in all_recs if r.get("acc_final") is not None]
    if accs:
        n_zoom = sum(1 for r in all_recs if r.get("zoom_triggered"))
        n_conf = len(all_recs) - n_zoom
        conf_acc = np.mean([r["acc_final"] for r in all_recs if not r.get("zoom_triggered")])
        zoom_acc = (np.mean([r["acc_final"] for r in all_recs if r.get("zoom_triggered")])
                    if n_zoom else 0)
        agreements = [r["r1_agreement"] for r in all_recs if r.get("r1_agreement") is not None]
        print(f"  Accuracy (full {len(accs)}):  {np.mean(accs):.4f}")
        print(f"  Confident (no zoom): {n_conf:4d}  acc={conf_acc:.4f}")
        print(f"  Zoom-triggered:      {n_zoom:4d}  acc={zoom_acc:.4f}")
        if agreements:
            below = sum(1 for a in agreements if a < args.agreement_threshold)
            print(f"  R1 agreement: mean={np.mean(agreements):.3f}  "
                  f"below_θ={below}/{len(agreements)}")
        avg_r = np.mean([r.get("n_rounds", 1) for r in all_recs])
        avg_t = np.mean([r.get("n_tool_calls", 0) for r in all_recs])
        print(f"  Avg rounds: {avg_r:.2f}  Avg tool calls: {avg_t:.2f}")
    print(f"  Output: {output_jsonl}")
    print(f"{'='*60}")

    with open(os.path.join(args.output_dir, "tvote_summary.json"), "w") as f:
        json.dump({
            "n_samples": len(accs), "accuracy": float(np.mean(accs)) if accs else None,
            "mode": "greedy_answer_temp_vote_uncertainty",
            "n_vote": args.n_vote, "vote_temp": args.vote_temp,
            "agreement_threshold": args.agreement_threshold,
            "fps": args.fps, "frames_upbound": args.frames_upbound,
            "max_rounds": args.max_rounds, "model": args.model_path,
        }, f, indent=2)
    print(f"  Summary saved to {args.output_dir}/tvote_summary.json")


if __name__ == "__main__":
    main()
