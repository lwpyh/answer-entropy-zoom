#!/usr/bin/env python3
"""
main_infer_partial_branch.py  ─  Partial-Branching uncertainty-gated zoom inference

Round 1  [greedy T=0, stop=["</answer>"] only, logprobs]
  ① Find peak-entropy window in reasoning → split token index
  ② At split point: take top-K alternative tokens (rank-2 … rank-K+1)
     from the logprob dict  →  deterministic, no temperature needed
  ③ Construct K branch prompts: prompt + prefix_text + alt_token_k
     Roll out short greedy continuations (max 512 tok, stop=["</answer>"])
  ④ Collect answers: main + K branches
     Agreement = #{answer == majority} / (K+1)
     agreement ≥ θ  →  confident, output main answer  (single pass + cheap branches)
     agreement < θ  →  uncertain, trigger zoom rounds

Round 2+  [same conversation context, NOT a restart]
  - Append Round 1 main output + reconsider turn
  - stop=["</video_zoom>","</answer>"], execute zoom clips
  - After each answer re-check agreement via same partial-branch mechanism
    agreement ≥ θ  →  stop
    agreement < θ  AND rounds < max  →  inject another reconsider turn
    rounds == max  →  force output

Key properties
──────────────
• Branching is for MEASUREMENT only — we never swap in a branch answer
• Branches are short (max 512 tok) and start from mid-reasoning → cheap
• Fully deterministic (T=0 everywhere), no temperature tuning needed
• Replaces margin (single logit gap) with multi-path agreement (richer signal)
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
from main_infer_branch import (
    token_entropy,
    get_rank1_token,
    find_split_point,
)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builders
# ═══════════════════════════════════════════════════════════════════════════

def build_r1_prompt(question: str, frame_times, processor) -> str:
    """
    Round 1: NOTOOL_SYS — no zoom instructions, model answers directly.
    Using TOOL_SYS here causes the model to call zoom regardless of the
    user-turn instruction, leaving an un-executed tool call and no <answer>.
    """
    vis  = _frame_tokens(frame_times)
    user = question.replace("<image>", vis, 1) if "<image>" in question else vis + "\n" + question
    user += (
        "\n\nPlease reason step by step inside <think> </think> and provide "
        "your answer inside <answer> </answer>."
    )
    msgs = [{"role": "system", "content": NOTOOL_SYS},
            {"role": "user",   "content": user}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_r2_prompt(question: str, frame_times, r1_clean_output: str,
                    processor, is_last: bool = False) -> str:
    """
    Round 2+: Rebuild with TOOL_SYS so the model can call zoom.
    Carries R1 output as a prior assistant turn for continuity.
    """
    vis  = _frame_tokens(frame_times)
    user = question.replace("<image>", vis, 1) if "<image>" in question else vis + "\n" + question
    reconsider = (
        "Your answer seems uncertain. Please re-examine the video carefully "
        "and refine your reasoning. You may now use <video_zoom> to zoom into "
        "key segments if needed, then provide your final answer inside "
        "<answer> and </answer>."
    )
    if is_last:
        reconsider += " Do not call <video_zoom> in this round."
    msgs = [
        {"role": "system",    "content": TOOL_SYS},
        {"role": "user",      "content": user},
        {"role": "assistant", "content": r1_clean_output},
        {"role": "user",      "content": reconsider},
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_reconsider_turn(is_last: bool) -> str:
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
# Partial-branch uncertainty
# ═══════════════════════════════════════════════════════════════════════════

def get_branch_tokens(logprobs_list, split_idx: int, n_branches: int):
    """
    Return up to n_branches alternative decoded tokens at logprobs_list[split_idx].
    Skips rank-1 (already taken by main path) and blank / whitespace-only tokens.
    """
    lp_dict = logprobs_list[split_idx]
    if not lp_dict:
        return []
    items = sorted(lp_dict.values(), key=lambda x: x.logprob, reverse=True)
    alternatives = []
    for item in items[1:]:          # skip rank-1
        tok = item.decoded_token or ""
        if tok.strip():             # skip empty/whitespace-only
            alternatives.append(tok)
        if len(alternatives) >= n_branches:
            break
    return alternatives


def compute_agreement(main_answer, branch_answers):
    """
    Agreement = fraction of answers (main + branches) that match the plurality.
    Returns float in [0, 1].
    Returns 0.0 if main_answer is None (model failed to produce answer → uncertain).
    Returns 1.0 if main_answer is valid but no branches produced answers.
    """
    if main_answer is None:
        return 0.0      # no main answer → treat as maximally uncertain
    all_ans = [a for a in [main_answer] + branch_answers if a is not None]
    if len(all_ans) <= 1:
        return 1.0      # only main, treat as confident (no branches to compare)
    top_count = Counter(all_ans).most_common(1)[0][1]
    return top_count / len(all_ans)


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════

class SampleState:
    def __init__(self, pid, gt, video_path, r1_prompt, images, question="", frame_times=None):
        self.pid             = pid
        self.gt              = gt
        self.video_path      = video_path
        self.prompt          = r1_prompt
        self.images          = images
        self.question        = question      # stored to rebuild R2 with TOOL_SYS
        self.frame_times     = frame_times or []
        self.n_tool_calls    = 0
        self.n_rounds        = 0
        self.final_answer    = None
        self.acc_final       = None
        self.raw_output      = ""
        self.zoom_triggered  = False
        self.r1_agreement    = None   # agreement from Round 1 branches
        self.last_agreement  = None   # most recent agreement check
        self.peak_entropy    = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results/partial_branch")

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
    p.add_argument("--branch_max_tokens",        type=int,   default=512,
                   help="Max tokens for branch rollout (short, just enough for answer)")
    p.add_argument("--tool_limit_mm",            type=int,   default=128)
    p.add_argument("--tool_max_frames_per_call", type=int,   default=16)
    p.add_argument("--tool_workers",             type=int,   default=8)
    p.add_argument("--max_rounds",               type=int,   default=5)

    # Partial-branch uncertainty
    p.add_argument("--n_branches",       type=int,   default=4,
                   help="Number of alternative tokens to branch (top-2..top-K+1)")
    p.add_argument("--ent_threshold",    type=float, default=0.5,
                   help="Peak window entropy to trigger branching at all. "
                        "Below this → treat as confident without branching.")
    p.add_argument("--window_size",      type=int,   default=16)
    p.add_argument("--agreement_threshold", type=float, default=0.6,
                   help="Min fraction of (main+branches) agreeing to be confident. "
                        "n_branches=3: 0.75=3/4, 0.5=2/4. "
                        "n_branches=5: 0.67=4/6, 0.5=3/6.")
    p.add_argument("--logprobs_k",       type=int,   default=20,
                   help="Top-K logprobs per token (need enough for n_branches+1)")

    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_pbranch.jsonl")

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

    # Round 1: stop only at </answer>, always get a full answer; needs logprobs
    r1_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=args.logprobs_k,
    )

    # Branch rollouts: short greedy, stop at </answer>
    branch_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.branch_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
    )

    # Tool rounds 2+: zoom-enabled; keep logprobs for re-checking agreement
    tool_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=args.logprobs_k,
    )

    print(f"[run]   n_branches={args.n_branches}  ent_threshold={args.ent_threshold}  "
          f"agreement_threshold={args.agreement_threshold}  window={args.window_size}  "
          f"branch_max_tokens={args.branch_max_tokens}  batch_size={args.batch_size}")

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
    n_correct = n_zoom_triggered = n_branched_at_all = 0

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
                    active.append(SampleState(pid, gt, video_path, r1_prompt, list(frames),
                                              question=question, frame_times=frame_times))
                except Exception as e:
                    print(f"\n[prep] skip {pid}: {e}")

            if not active:
                continue

            done = []

            # ══════════════════════════════════════════════════════════════
            # Round 1: greedy + logprobs, no zoom execution
            # ══════════════════════════════════════════════════════════════
            r1_inputs = [
                {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                for s in active
            ]
            r1_outputs = llm.generate(r1_inputs, r1_sp)

            # ── Partial-branch uncertainty for each sample ─────────────── #
            # Collect branch inputs across ALL samples in batch (batched together)
            branch_inputs  = []   # flat list of vLLM inputs
            branch_meta    = []   # (sample_idx, branch_idx) for un-flattening
            sample_branch_counts = []  # how many branches per sample

            for s_idx, (s, r1_out) in enumerate(zip(active, r1_outputs)):
                text     = r1_out.outputs[0].text
                logprobs = r1_out.outputs[0].logprobs or []
                s.n_rounds   = 1
                s.raw_output = text

                # Find split point (reuse existing find_split_point)
                prefix_text, peak_ent = find_split_point(
                    text, logprobs, args.window_size, args.ent_threshold
                )
                s.peak_entropy = peak_ent

                if prefix_text is None:
                    # Entropy below threshold → confident without branching
                    sample_branch_counts.append(0)
                    continue

                # Map prefix_text to approximate token index in logprobs
                # find_split_point already computed approx_prefix_len internally;
                # we replicate it here to get best_i (split token index)
                entropies = [token_entropy(d) for d in logprobs]
                n = len(logprobs)
                best_i = 0
                best_e = 0.0
                for i in range(n - args.window_size):
                    e = float(np.mean(entropies[i:i + args.window_size]))
                    if e > best_e:
                        best_e, best_i = e, i

                alt_tokens = get_branch_tokens(logprobs, best_i, args.n_branches)
                if not alt_tokens:
                    sample_branch_counts.append(0)
                    continue

                n_branched_at_all += 1
                sample_branch_counts.append(len(alt_tokens))
                for b_idx, alt_tok in enumerate(alt_tokens):
                    branch_prompt = s.prompt + prefix_text + alt_tok
                    branch_inputs.append({
                        "prompt":            branch_prompt,
                        "multi_modal_data":  {"image": list(s.images)},
                    })
                    branch_meta.append((s_idx, b_idx))

            # ── Run all branches in one batched call ───────────────────── #
            branch_answers_per_sample = {i: [] for i in range(len(active))}
            if branch_inputs:
                branch_outputs = llm.generate(branch_inputs, branch_sp)
                for (s_idx, _), b_out in zip(branch_meta, branch_outputs):
                    ans = extract_mc_answer(b_out.outputs[0].text)
                    branch_answers_per_sample[s_idx].append(ans)

            # ── Agreement check → route to done or zoom ────────────────── #
            zoom_pending = []
            for s_idx, (s, r1_out) in enumerate(zip(active, r1_outputs)):
                main_answer  = extract_mc_answer(s.raw_output)
                branch_ans   = branch_answers_per_sample[s_idx]
                agreement    = compute_agreement(main_answer, branch_ans)
                s.r1_agreement   = agreement
                s.last_agreement = agreement

                if agreement >= args.agreement_threshold:
                    # Confident: use main answer
                    s.final_answer = main_answer
                    s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                    done.append(s)
                else:
                    # Uncertain: rebuild prompt with TOOL_SYS so model can call zoom.
                    # Carry R1 output (zoom tags stripped) as prior assistant turn.
                    clean_text = re.sub(
                        r'<video_zoom>.*?</video_zoom>', '', s.raw_output, flags=re.DOTALL
                    ).strip()
                    s.prompt = build_r2_prompt(
                        s.question, s.frame_times, clean_text, processor, is_last=False
                    )
                    s.zoom_triggered = True
                    zoom_pending.append(s)

            # ══════════════════════════════════════════════════════════════
            # Round 2+: zoom-enabled continuation
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
                    text     = out.outputs[0].text
                    logprobs = out.outputs[0].logprobs or []
                    s.n_rounds   = round_idx
                    s.raw_output = text

                    zoom_call = parse_zoom_call(text)

                    if zoom_call is not None and not is_last:
                        end_pos   = text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += text[:end_pos]
                        zoom_queue[idx] = (s, zoom_call)
                    else:
                        # Answer produced: re-check uncertainty via partial branches
                        main_answer = extract_mc_answer(text)
                        prefix_text2, peak_ent2 = find_split_point(
                            text, logprobs, args.window_size, args.ent_threshold
                        )

                        if prefix_text2 is not None and not is_last:
                            # Re-branch to measure agreement after zoom
                            entropies2 = [token_entropy(d) for d in logprobs]
                            n2 = len(logprobs)
                            best_i2 = max(
                                range(n2 - args.window_size),
                                key=lambda i: float(np.mean(entropies2[i:i + args.window_size]))
                            ) if n2 > args.window_size else 0
                            alt_toks2 = get_branch_tokens(logprobs, best_i2, args.n_branches)
                            b_answers2 = []
                            if alt_toks2:
                                b_inputs2 = [
                                    {"prompt": s.prompt + prefix_text2 + alt_t,
                                     "multi_modal_data": {"image": list(s.images)}}
                                    for alt_t in alt_toks2
                                ]
                                b_outs2 = llm.generate(b_inputs2, branch_sp)
                                b_answers2 = [extract_mc_answer(o.outputs[0].text) for o in b_outs2]

                            agreement2 = compute_agreement(main_answer, b_answers2)
                            s.last_agreement = agreement2

                            if agreement2 >= args.agreement_threshold or is_last:
                                s.final_answer = main_answer
                                s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                                done.append(s)
                            else:
                                s.prompt += text + build_reconsider_turn(is_last=False)
                                next_active.append(s)
                        else:
                            # Low entropy or last round: done
                            s.final_answer = main_answer
                            s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                            done.append(s)

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
                    "problem_id":      s.pid,
                    "gt":              s.gt,
                    "acc_final":       s.acc_final,
                    "final_answer":    s.final_answer,
                    "n_rounds":        s.n_rounds,
                    "n_tool_calls":    s.n_tool_calls,
                    "raw_output":      s.raw_output,
                    "zoom_triggered":  s.zoom_triggered,
                    "r1_agreement":    s.r1_agreement,
                    "last_agreement":  s.last_agreement,
                    "peak_entropy":    s.peak_entropy,
                    # Schema compat
                    "branched":        s.zoom_triggered,
                    "branch_entropy":  s.peak_entropy,
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
    print(f"  Partial-branch inference ({n_this} new samples)")
    if n_this:
        print(f"  Accuracy (this run):   {n_correct}/{n_this} = {n_correct/n_this:.4f}")
        print(f"  Branched at all:       {n_branched_at_all}/{n_this} "
              f"({n_branched_at_all/n_this*100:.1f}%)")
        print(f"  Zoom triggered:        {n_zoom_triggered}/{n_this} "
              f"({n_zoom_triggered/n_this*100:.1f}%)")

    with open(output_jsonl) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    accs = [float(r["acc_final"]) for r in all_recs if r.get("acc_final") is not None]
    if accs:
        n_zoom = sum(1 for r in all_recs if r.get("zoom_triggered"))
        n_conf = len(all_recs) - n_zoom
        conf_acc = np.mean([r["acc_final"] for r in all_recs if not r.get("zoom_triggered")])
        zoom_acc = np.mean([r["acc_final"] for r in all_recs if r.get("zoom_triggered")]) if n_zoom else 0
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

    with open(os.path.join(args.output_dir, "pbranch_summary.json"), "w") as f:
        json.dump({
            "n_samples": len(accs), "accuracy": float(np.mean(accs)) if accs else None,
            "mode": "partial_branch_zoom",
            "n_branches": args.n_branches, "ent_threshold": args.ent_threshold,
            "agreement_threshold": args.agreement_threshold,
            "window_size": args.window_size, "branch_max_tokens": args.branch_max_tokens,
            "fps": args.fps, "frames_upbound": args.frames_upbound,
            "max_rounds": args.max_rounds, "model": args.model_path,
        }, f, indent=2)
    print(f"  Summary saved to {args.output_dir}/pbranch_summary.json")


if __name__ == "__main__":
    main()
