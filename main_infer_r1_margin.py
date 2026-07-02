#!/usr/bin/env python3
"""
main_infer_r1_margin.py  ─  Round-1 margin-gated zoom inference

Flow
────
Round 1  [notool-style, same TOOL_SYS, stop=["</answer>"] only]
  - Greedy + logprobs.  Model ALWAYS produces a full answer (no zoom execution).
  - User turn appends: "Do not call <video_zoom> in this response."
  - Find answer token (A/B/C/D) in logprobs → compute margin = logp(r1) - logp(r2).
  - margin ≥ θ  →  done  (single pass, high confidence — equiv. majority voting agreement)
  - margin < θ  →  append Round 1 output to context, inject reconsider turn  →  Round 2

Round 2+  [zoom-enabled, same conversation context]
  - Round 1 assistant output stays in context (not a restart).
  - stop=["</video_zoom>", "</answer>"]
  - Model calls zoom → execute clip → append tool_response → continue
  - Model gives answer → check margin again
    - margin ≥ θ  →  stop
    - margin < θ  AND  rounds < max  →  inject another reconsider turn  (iterative zoom)
    - rounds == max  →  force output

Key fixes vs margin_zoom.py
───────────────────────────
- Round 1 never stops at </video_zoom>, so answer token is ALWAYS available
- More robust answer token detection (handles BPE space prefixes)
- margin is checked after EVERY answer, not just the first
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
    TOOL_SYS,
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
# Prompt builders
# ═══════════════════════════════════════════════════════════════════════════

def build_r1_prompt(question: str, frame_times, processor) -> str:
    """
    Round 1 prompt: TOOL_SYS (identical to tool rounds), but user turn
    explicitly asks for a direct answer without zoom.
    This keeps system prompts consistent while preventing zoom in Round 1.
    """
    vis  = _frame_tokens(frame_times)
    user = question.replace("<image>", vis, 1) if "<image>" in question else vis + "\n" + question
    user += (
        "\n\nPlease reason step by step inside <think> </think> and provide "
        "your initial answer inside <answer> </answer>. "
        "Do not call <video_zoom> in this response."
    )
    msgs = [{"role": "system", "content": TOOL_SYS},
            {"role": "user",   "content": user}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_reconsider_turn(is_last: bool) -> str:
    """
    Appended after a low-confidence answer.
    Closes the current assistant turn, opens a user turn asking for
    reconsideration with zoom available, then opens a new assistant turn.
    """
    msg = (
        "<|im_end|>\n<|im_start|>user\n"
        "Your answer seems uncertain. Please re-examine the video carefully "
        "and refine your reasoning. You may now use <video_zoom> to zoom into "
        "key segments if needed, then provide your final answer inside "
        "<answer> and </answer>."
    )
    if is_last:
        msg += " Do not call <video_zoom> in this round."
    msg += "<|im_end|>\n<|im_start|>assistant\n"
    return msg


# ═══════════════════════════════════════════════════════════════════════════
# Answer-token margin
# ═══════════════════════════════════════════════════════════════════════════

# BPE / sentencepiece space prefixes that may appear before a token
_BPE_PREFIXES = ('\u0120', '\u2581', '\u0116')   # Ġ (GPT-2), ▁ (SP), Ė (some tiktoken)

def _clean_token(tok: str) -> str:
    """Strip whitespace and common BPE space prefixes."""
    return tok.strip().lstrip(''.join(_BPE_PREFIXES)).strip()


def get_answer_token_margin(logprobs_list, expected_answer: str):
    """
    Scan logprobs from the END to find the last position where the
    generated (rank-1) token is the expected answer letter (A/B/C/D).

    Returns (margin, token_idx) or (None, None).
    margin = logp(rank1) - logp(rank2)  — high means confident.

    Robust matching: handles leading BPE space chars, surrounding punctuation
    (e.g. '{A}' neighbours), and short compound tokens like ' A'.
    """
    if not logprobs_list or expected_answer is None:
        return None, None

    for i in reversed(range(len(logprobs_list))):
        lp_dict = logprobs_list[i]
        if not lp_dict:
            continue
        items = sorted(lp_dict.values(), key=lambda x: x.logprob, reverse=True)
        if not items:
            continue

        raw = items[0].decoded_token or ""
        clean = _clean_token(raw)

        # Primary match: clean token is exactly the answer letter
        if clean == expected_answer and len(clean) == 1:
            margin = (items[0].logprob - items[1].logprob) if len(items) >= 2 else 10.0
            return float(margin), i

        # Fallback: token contains the answer letter and is short (e.g. 'A.', 'A)')
        if (len(clean) <= 2
                and clean.startswith(expected_answer)
                and expected_answer in 'ABCD'):
            margin = (items[0].logprob - items[1].logprob) if len(items) >= 2 else 10.0
            return float(margin), i

    return None, None


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════

class SampleState:
    def __init__(self, pid, gt, video_path, r1_prompt, images):
        self.pid            = pid
        self.gt             = gt
        self.video_path     = video_path
        self.prompt         = r1_prompt     # mutates as conversation grows
        self.images         = images
        self.n_tool_calls   = 0
        self.n_rounds       = 0
        self.final_answer   = None
        self.acc_final      = None
        self.raw_output     = ""
        # Margin / zoom metadata
        self.r1_margin      = None     # margin from Round 1 answer token
        self.last_margin    = None     # margin from most recent answer token
        self.zoom_triggered = False    # did we ever enter zoom rounds?
        self.r1_done        = False    # has Round 1 been appended to context?


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results/r1_margin")

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

    # Margin
    p.add_argument("--margin_threshold", type=float, default=2.0,
                   help="logp(r1)-logp(r2) at answer token. "
                        "≥ θ → confident (done). < θ → trigger zoom rounds.")
    p.add_argument("--logprobs_k", type=int, default=5)

    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_r1margin.jsonl")

    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
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

    # Round 1: stop ONLY at </answer>, always get a complete answer
    r1_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=args.logprobs_k,
    )

    # Round 2+: zoom-enabled, check margin after each answer
    tool_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=args.logprobs_k,   # keep logprobs to check margin after zoom
    )

    print(f"[run]   margin_threshold={args.margin_threshold}  "
          f"logprobs_k={args.logprobs_k}  batch_size={args.batch_size}")

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
                    r1_prompt = build_r1_prompt(question, frame_times, processor)
                    active.append(SampleState(pid, gt, video_path, r1_prompt, list(frames)))
                except Exception as e:
                    print(f"\n[prep] skip {pid}: {e}")

            if not active:
                continue

            done = []

            # ── Round 1: notool greedy + logprobs ─────────────────────── #
            r1_inputs = [
                {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                for s in active
            ]
            r1_outputs = llm.generate(r1_inputs, r1_sp)

            # Classify: confident → done, uncertain → zoom_queue
            zoom_pending = []   # samples that need zoom rounds
            for s, r1_out in zip(active, r1_outputs):
                text     = r1_out.outputs[0].text
                logprobs = r1_out.outputs[0].logprobs or []
                s.n_rounds   = 1
                s.raw_output = text

                answer = extract_mc_answer(text)
                margin, _ = get_answer_token_margin(logprobs, answer)
                s.r1_margin   = margin
                s.last_margin = margin

                if margin is not None and margin < args.margin_threshold:
                    # Uncertain: strip any accidental zoom tags, append to context
                    clean_text = re.sub(
                        r'<video_zoom>.*?</video_zoom>', '', text, flags=re.DOTALL
                    ).strip()
                    # Append Round 1 assistant output + reconsider turn
                    s.prompt += clean_text + build_reconsider_turn(is_last=False)
                    s.zoom_triggered = True
                    s.r1_done = True
                    zoom_pending.append(s)
                else:
                    # Confident (or margin unknown): take Round 1 answer
                    s.final_answer = answer
                    s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                    done.append(s)

            # ── Round 2+ zoom loop ─────────────────────────────────────── #
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
                    text     = out.outputs[0].text
                    logprobs = out.outputs[0].logprobs or []
                    s.n_rounds   = round_idx
                    s.raw_output = text

                    zoom_call = parse_zoom_call(text)

                    if zoom_call is not None and not is_last:
                        # Zoom call: append to prompt, execute clip
                        end_pos   = text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += text[:end_pos]
                        zoom_queue[idx] = (s, zoom_call)
                    else:
                        # Answer: check margin, decide stop or continue
                        answer = extract_mc_answer(text)
                        margin, _ = get_answer_token_margin(logprobs, answer)
                        s.last_margin = margin

                        confident = (margin is None or margin >= args.margin_threshold)

                        if confident or is_last:
                            s.final_answer = answer
                            s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                            done.append(s)
                        else:
                            # Still uncertain: inject another reconsider turn
                            s.prompt += text + build_reconsider_turn(is_last=False)
                            next_active.append(s)

                # ── Execute zoom clips ─────────────────────────────────── #
                if zoom_queue:
                    with ThreadPoolExecutor(max_workers=args.tool_workers) as ex:
                        futures = {
                            ex.submit(
                                extract_video_clip,
                                video_path     = s.video_path,
                                start_time     = s_t,
                                end_time       = e_t,
                                fps            = fps_z,
                                max_pixels     = args.max_pixels,
                                min_pixels     = args.min_pixels,
                                max_frames     = args.tool_max_frames_per_call,
                                storage_system = "local",
                            ): idx
                            for idx, (s, (s_t, e_t, fps_z)) in zoom_queue.items()
                        }
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
                    "r1_margin":      s.r1_margin,
                    "last_margin":    s.last_margin,
                    # Schema compat
                    "branched":        s.zoom_triggered,
                    "branch_entropy":  s.r1_margin,
                    "branch_decision": "zoom" if s.zoom_triggered else "main",
                    "acc_notool":      None,
                    "r1_majority_ans": s.final_answer,
                    "graduated_round": s.n_rounds,
                }
                all_records.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    # ── Summary ─────────────────────────────────────────────────────────── #
    n_this = len(all_records)
    print(f"\n{'='*60}")
    print(f"  R1-margin inference ({n_this} new samples)")
    if n_this:
        print(f"  Accuracy (this run):   {n_correct}/{n_this} = {n_correct/n_this:.4f}")
        print(f"  Zoom triggered:        {n_zoom_triggered}/{n_this} "
              f"({n_zoom_triggered/n_this*100:.1f}%)")

    with open(output_jsonl) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    accs = [float(r["acc_final"]) for r in all_recs if r.get("acc_final") is not None]
    if accs:
        r1_margins  = [r["r1_margin"]   for r in all_recs if r.get("r1_margin")   is not None]
        n_triggered = sum(1 for r in all_recs if r.get("zoom_triggered"))
        n_single    = sum(1 for r in all_recs if not r.get("zoom_triggered"))

        print(f"  Accuracy (full, {len(accs)}):  {np.mean(accs):.4f}")
        print(f"  Single-pass (no zoom):  {n_single}  "
              f"acc={np.mean([r['acc_final'] for r in all_recs if not r.get('zoom_triggered')]):.4f}")
        print(f"  Zoom-triggered:         {n_triggered}  "
              f"acc={np.mean([r['acc_final'] for r in all_recs if r.get('zoom_triggered')]):.4f}"
              if n_triggered else "  Zoom-triggered: 0")
        if r1_margins:
            print(f"  R1 margin: mean={np.mean(r1_margins):.2f}  "
                  f"median={np.median(r1_margins):.2f}  "
                  f"below_θ={sum(1 for m in r1_margins if m < args.margin_threshold)}"
                  f"/{len(r1_margins)}")
        avg_rounds = np.mean([r.get("n_rounds", 1) for r in all_recs])
        avg_tools  = np.mean([r.get("n_tool_calls", 0) for r in all_recs])
        print(f"  Avg rounds: {avg_rounds:.2f}  Avg tool calls: {avg_tools:.2f}")
    print(f"  Output: {output_jsonl}")
    print(f"{'='*60}")

    summary = {
        "n_samples":        len(accs),
        "accuracy":         float(np.mean(accs)) if accs else None,
        "mode":             "r1_margin_zoom",
        "margin_threshold": args.margin_threshold,
        "logprobs_k":       args.logprobs_k,
        "fps":              args.fps,
        "frames_upbound":   args.frames_upbound,
        "max_rounds":       args.max_rounds,
        "model":            args.model_path,
    }
    with open(os.path.join(args.output_dir, "r1margin_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {args.output_dir}/r1margin_summary.json")


if __name__ == "__main__":
    main()
