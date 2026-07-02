#!/usr/bin/env python3
"""
main_infer_branch.py  ─  Step-level uncertainty branching inference

Pass-1  greedy (T=0) with logprobs  →  detect highest-entropy step in output
Pass-2  branch from that step (n_branches, T=branch_temp) if entropy > threshold
Decision logic:
  a. Majority of branches trigger tool, main didn't  →  trigger tool
  b. Branches consistently disagree with main answer →  use branch answer
  c. Default                                         →  keep main path
Tool execution identical to main_infer_greedy.py
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
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
    p.add_argument("--output_dir",  default="./infer_results/branch_baseline")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=2)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=100352)
    p.add_argument("--min_pixels",             type=int,   default=25088)

    # Video & tool  (align to greedy baseline)
    p.add_argument("--fps",                      type=float, default=0.5)
    p.add_argument("--frames_upbound",           type=int,   default=64)
    p.add_argument("--max_tokens",               type=int,   default=4096)
    p.add_argument("--tool_limit_mm",            type=int,   default=128)
    p.add_argument("--tool_max_frames_per_call", type=int,   default=16)
    p.add_argument("--tool_workers",             type=int,   default=8)
    p.add_argument("--max_rounds",               type=int,   default=5)

    # Branching hyperparams
    p.add_argument("--ent_threshold", type=float, default=0.8,
                   help="Mean-window entropy (nats) to trigger branching. "
                        "Tune by inspecting branch_entropy distribution.")
    p.add_argument("--n_branches",    type=int,   default=2,
                   help="Number of branch samples (2 or 3)")
    p.add_argument("--branch_temp",   type=float, default=0.5,
                   help="Sampling temperature for branch generation")
    p.add_argument("--logprobs_k",    type=int,   default=20,
                   help="Top-K logprobs per token in Pass-1 (vLLM v1 max=20)")
    p.add_argument("--window_size",   type=int,   default=16,
                   help="Token window size for sliding-window entropy")

    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Entropy / logprob utilities
# ═══════════════════════════════════════════════════════════════════════════

def token_entropy(logprob_dict):
    """Approximate entropy (nats) from vLLM top-K logprob dict."""
    if not logprob_dict:
        return 0.0
    logps = np.array([v.logprob for v in logprob_dict.values()], dtype=np.float64)
    ps = np.exp(logps)
    ps /= ps.sum()          # normalise top-K to sum to 1
    return float(-np.sum(ps * np.log(ps + 1e-12)))


def get_rank1_token(logprob_dict):
    """Decoded text of the actually-generated token (greedy → rank=1)."""
    if not logprob_dict:
        return ""
    for v in logprob_dict.values():
        if getattr(v, "rank", None) == 1:
            return v.decoded_token or ""
    return max(logprob_dict.values(), key=lambda x: x.logprob).decoded_token or ""


def mean_logprob(branch_output):
    """Mean logprob of the tokens in one branch CompletionOutput."""
    if not branch_output.logprobs:
        return -999.0
    lps = [
        max(d.values(), key=lambda x: x.logprob).logprob
        for d in branch_output.logprobs if d
    ]
    return float(np.mean(lps)) if lps else -999.0


# ═══════════════════════════════════════════════════════════════════════════
# Split point detection
# ═══════════════════════════════════════════════════════════════════════════

def find_split_point(output_text, logprobs_list, window, ent_threshold):
    """
    Identify where to split the greedy output for branching.

    Algorithm:
      1. Compute per-token entropy from top-K logprob dicts.
      2. Sliding window of `window` tokens → find peak mean-entropy window.
      3. If peak < ent_threshold  →  no branching needed.
      4. Reconstruct approximate char position via decoded_token lengths.
      5. Walk back to nearest '\\n' (or '. ') for a clean sentence boundary.

    Returns:
      (prefix_text, peak_entropy)  on success
      (None, peak_entropy)         if entropy below threshold or text too short
    """
    n = len(logprobs_list)
    if n < window * 2 + 1:
        return None, 0.0

    entropies = [token_entropy(d) for d in logprobs_list]

    best_i, best_e = 0, 0.0
    for i in range(n - window):
        e = float(np.mean(entropies[i:i + window]))
        if e > best_e:
            best_e, best_i = e, i

    if best_e < ent_threshold:
        return None, best_e

    # Reconstruct approximate char length of prefix up to best_i
    approx_prefix_len = sum(
        len(get_rank1_token(logprobs_list[i])) for i in range(best_i)
    )

    # Find nearest sentence boundary before (or slightly after) that position
    search_end = min(approx_prefix_len + 30, len(output_text))
    nl_pos = output_text.rfind('\n', 0, search_end)
    if nl_pos < 5:
        dot_pos = output_text.rfind('. ', 0, search_end)
        nl_pos = dot_pos + 1 if dot_pos > 5 else approx_prefix_len

    prefix = output_text[:nl_pos + 1]

    # Require a non-trivial prefix and a non-trivial suffix
    if len(prefix) < 10 or len(output_text) - len(prefix) < 5:
        return None, best_e

    return prefix, best_e


# ═══════════════════════════════════════════════════════════════════════════
# Decision logic
# ═══════════════════════════════════════════════════════════════════════════

def decide(main_text, branch_outputs, prefix_text):
    """
    Compare Pass-1 main output against Pass-2 branches and choose the best.

    Returns (decision_tag, chosen_text)
      "main"        – keep main path unchanged
      "branch_tool" – a branch confidently triggers tool; use that branch
      "branch_ans"  – branches consistently disagree with main answer; switch
    """
    full_branch_texts = [prefix_text + b.text for b in branch_outputs]

    main_answer   = extract_mc_answer(main_text)
    main_has_tool = parse_zoom_call(main_text) is not None

    branch_answers = [extract_mc_answer(t) for t in full_branch_texts]
    branch_tools   = [parse_zoom_call(t) is not None for t in full_branch_texts]
    branch_lps     = [mean_logprob(b) for b in branch_outputs]

    n = len(branch_outputs)

    # Rule 1: majority of branches trigger tool, main does not
    n_tool = sum(branch_tools)
    if n_tool > n / 2 and not main_has_tool:
        best_idx = max(
            (i for i, t in enumerate(branch_tools) if t),
            key=lambda i: branch_lps[i],
        )
        return "branch_tool", full_branch_texts[best_idx]

    # Rule 2: branches consistently produce an answer that differs from main
    valid_answers = [a for a in branch_answers if a]
    if valid_answers:
        counts = Counter(valid_answers)
        top_ans, cnt = counts.most_common(1)[0]
        # Require strict majority AND disagreement with main
        if cnt >= max(2, math.ceil(n * 0.6)) and top_ans != main_answer:
            best_idx = max(
                (i for i, a in enumerate(branch_answers) if a == top_ans),
                key=lambda i: branch_lps[i],
            )
            return "branch_ans", full_branch_texts[best_idx]

    return "main", main_text


# ═══════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════

class SampleState:
    def __init__(self, pid, gt, video_path, prompt, images):
        self.pid           = pid
        self.gt            = gt
        self.video_path    = video_path
        self.prompt        = prompt
        self.images        = images
        self.n_tool_calls  = 0
        self.n_rounds      = 0
        self.final_answer  = None
        self.acc_final     = None
        self.raw_output    = ""
        # Branching metadata (recorded per-round; last branch wins)
        self.branched        = False
        self.branch_entropy  = 0.0
        self.branch_decision = "none"


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_branch.jsonl")

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

    # Pass-1 sampling params: greedy + logprobs for entropy
    pass1_sp = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
        logprobs=args.logprobs_k,
    )

    # Pass-2 sampling params: branching from uncertain prefix
    pass2_sp = SamplingParams(
        n=args.n_branches,
        temperature=args.branch_temp,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
    )

    print(f"[run]   ent_threshold={args.ent_threshold}, n_branches={args.n_branches}, "
          f"branch_temp={args.branch_temp}, logprobs_k={args.logprobs_k}, "
          f"window={args.window_size}")

    # Resume: skip already-done problem_ids
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

    all_records = []
    n_correct   = 0
    n_branched  = 0

    with open(output_jsonl, file_mode) as out_f:
        for batch_start in tqdm(
            range(0, len(pending), args.batch_size),
            desc="Batches",
            total=(len(pending) + args.batch_size - 1) // args.batch_size,
        ):
            batch  = pending[batch_start: batch_start + args.batch_size]

            # ── Preprocess batch ───────────────────────────────────────── #
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

                # Pass-1: greedy with logprobs
                p1_inputs = [
                    {"prompt": s.prompt, "multi_modal_data": {"image": list(s.images)}}
                    for s in active
                ]
                p1_outputs = llm.generate(p1_inputs, pass1_sp)

                # Analyse each sample: does it need branching?
                needs_branch  = []        # list of (list_idx, s, prefix, entropy)
                direct        = {}        # list_idx → chosen_text (no branch)

                for idx, (s, p1_out) in enumerate(zip(active, p1_outputs)):
                    text     = p1_out.outputs[0].text
                    logprobs = p1_out.outputs[0].logprobs or []
                    s.n_rounds   = round_idx
                    s.raw_output = text

                    prefix_text, max_ent = find_split_point(
                        text, logprobs, args.window_size, args.ent_threshold
                    )

                    # Don't branch on last round (no point) or if entropy is low
                    if prefix_text is not None and not is_last:
                        needs_branch.append((idx, s, prefix_text, max_ent))
                    else:
                        direct[idx] = text

                # Pass-2: branch for uncertain samples (batched)
                if needs_branch:
                    p2_inputs = [
                        {
                            "prompt": s.prompt + prefix_text,
                            "multi_modal_data": {"image": list(s.images)},
                        }
                        for _, s, prefix_text, _ in needs_branch
                    ]
                    p2_outputs = llm.generate(p2_inputs, pass2_sp)

                    for (idx, s, prefix_text, max_ent), p2_out in zip(needs_branch, p2_outputs):
                        main_text  = s.raw_output
                        decision, chosen_text = decide(main_text, p2_out.outputs, prefix_text)

                        s.branched        = True
                        s.branch_entropy  = round(max_ent, 4)
                        s.branch_decision = decision
                        s.raw_output      = chosen_text
                        n_branched       += 1

                        direct[idx] = chosen_text

                # Route each sample: tool call → zoom_queue, otherwise → done
                zoom_queue  = {}
                next_active = []

                for idx, s in enumerate(active):
                    chosen_text = direct.get(idx, s.raw_output)
                    zoom_call   = parse_zoom_call(chosen_text)

                    if zoom_call is not None and not is_last:
                        end_pos   = chosen_text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += chosen_text[:end_pos]
                        zoom_queue[idx] = (s, zoom_call)
                    else:
                        s.final_answer = extract_mc_answer(chosen_text)
                        s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

                if not zoom_queue:
                    active = []
                    break

                # Execute zoom clips in parallel
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

            # Force-finalize samples that hit max_rounds
            for s in active:
                s.final_answer = extract_mc_answer(s.raw_output)
                s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                done.append(s)

            # ── Save batch results ─────────────────────────────────────── #
            for s in done:
                n_correct += int(s.acc_final or 0)
                rec = {
                    "problem_id":      s.pid,
                    "gt":              s.gt,
                    "acc_final":       s.acc_final,
                    "final_answer":    s.final_answer,
                    "n_rounds":        s.n_rounds,
                    "n_tool_calls":    s.n_tool_calls,
                    "raw_output":      s.raw_output,
                    "branched":        s.branched,
                    "branch_entropy":  s.branch_entropy,
                    "branch_decision": s.branch_decision,
                    # Compatibility with adaptive zoom JSONL schema
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
    print(f"  Branch inference ({done_this_run} new samples)")
    if done_this_run:
        print(f"  Accuracy (this run): {n_correct}/{done_this_run} = "
              f"{n_correct/done_this_run:.4f}")
        print(f"  Branched: {n_branched} / {done_this_run} "
              f"({n_branched/done_this_run*100:.1f}%)")

    with open(output_jsonl) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    accs = [float(r["acc_final"]) for r in all_recs if r.get("acc_final") is not None]
    if accs:
        avg_rounds = np.mean([r.get("n_rounds", 1) for r in all_recs])
        avg_tools  = np.mean([r.get("n_tool_calls", 0) for r in all_recs])
        avg_ent    = np.mean([r.get("branch_entropy", 0) for r in all_recs
                              if r.get("branched")] or [0])
        b_count    = sum(1 for r in all_recs if r.get("branched"))
        print(f"  Accuracy (full file, {len(accs)} samples): {np.mean(accs):.4f}")
        print(f"  Avg rounds: {avg_rounds:.2f},  Avg tool calls: {avg_tools:.2f}")
        print(f"  Branched total: {b_count},  avg branch entropy: {avg_ent:.3f}")
        # Decision breakdown
        decisions = Counter(r.get("branch_decision") for r in all_recs if r.get("branched"))
        for tag, cnt in decisions.most_common():
            print(f"    {tag}: {cnt}")
    print(f"  Output: {output_jsonl}")
    print(f"{'='*60}")

    summary = {
        "n_samples":      len(accs),
        "accuracy":       float(np.mean(accs)) if accs else None,
        "mode":           "branch_uncertainty",
        "ent_threshold":  args.ent_threshold,
        "n_branches":     args.n_branches,
        "branch_temp":    args.branch_temp,
        "window_size":    args.window_size,
        "logprobs_k":     args.logprobs_k,
        "fps":            args.fps,
        "frames_upbound": args.frames_upbound,
        "max_rounds":     args.max_rounds,
        "model":          args.model_path,
    }
    with open(os.path.join(args.output_dir, "branch_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {args.output_dir}/branch_summary.json")


if __name__ == "__main__":
    main()
