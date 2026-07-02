#!/usr/bin/env python3
"""
main_infer_greedy.py  ──  Greedy (T=0, n=1) inference with natural tool use

Mirrors the vllm_video_multiturn training rollout (eval_video.sh):
  - TOOL_SYS system prompt, no explicit "Do not call zoom" anywhere
  - Model freely calls <video_zoom> as it sees fit
  - Zoom is executed and fed back; continues until model gives <answer>
  - Hard safety limit: max_rounds (same as max_generation_round in training)
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
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


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   required=True)
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results/greedy_baseline")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=2)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=100352)
    p.add_argument("--min_pixels",             type=int,   default=25088)

    # Video & tool  (aligned to eval_video.sh training config)
    p.add_argument("--fps",                      type=float, default=0.5)
    p.add_argument("--frames_upbound",           type=int,   default=64)
    p.add_argument("--max_tokens",               type=int,   default=4096)
    p.add_argument("--tool_limit_mm",            type=int,   default=128,
                   help="limit_mm_per_prompt in training config")
    p.add_argument("--tool_max_frames_per_call", type=int,   default=16,
                   help="tool_call_max_frames in training config")
    p.add_argument("--tool_workers",             type=int,   default=8)
    p.add_argument("--max_rounds",               type=int,   default=5,
                   help="Hard limit; same as max_generation_round in training")

    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════════

class SampleState:
    def __init__(self, pid, gt, video_path, prompt, images):
        self.pid          = pid
        self.gt           = gt
        self.video_path   = video_path
        self.prompt       = prompt   # accumulated prompt text
        self.images       = images   # accumulated image list
        self.n_tool_calls = 0
        self.n_rounds     = 0
        self.final_answer = None
        self.acc_final    = None
        self.raw_output   = ""       # last generated text


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_greedy.jsonl")

    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout.function_tools import extract_video_clip

    max_mm = max(args.frames_upbound, args.tool_limit_mm)
    print(f"[vLLM]  Initialising (tp={args.tensor_parallel_size}, "
          f"util={args.gpu_memory_utilization}, limit_mm={max_mm})")
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

    # T=0 greedy, n=1. Stop at </video_zoom> (tool call) or </answer> (done).
    greedy_sp = SamplingParams(
        n=1, temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</video_zoom>", "</answer>"],
        include_stop_str_in_output=True,
        detokenize=True,
    )

    print(f"[run]   T=0 greedy, n=1, natural tool use, "
          f"fps={args.fps}, frames≤{args.frames_upbound}, max_rounds={args.max_rounds}")

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

    pending = [s for s in samples
               if str(s.get("problem_id", "")) not in done_pids
               and str(s.get("extra_info", {}).get("problem_id", "")) not in done_pids]

    all_records = []
    n_correct   = 0

    with open(output_jsonl, file_mode) as out_f:
        for batch_start in tqdm(range(0, len(pending), args.batch_size),
                                desc="Batches",
                                total=(len(pending) + args.batch_size - 1) // args.batch_size):
            batch = pending[batch_start: batch_start + args.batch_size]

            # ── Preprocess ────────────────────────────────────────────── #
            active = []
            for sample in batch:
                pid = (sample.get("problem_id") or
                       sample.get("extra_info", {}).get("problem_id", "?"))
                gt         = sample.get("solution", "")
                question   = sample.get("problem", "")
                video_rel  = (sample.get("videos") or [""])[0]
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

            # ── Multi-round greedy inference ───────────────────────────── #
            for round_idx in range(1, args.max_rounds + 1):
                is_last = (round_idx == args.max_rounds)

                inputs = [
                    {"prompt": s.prompt,
                     "multi_modal_data": {"image": list(s.images)}}
                    for s in active
                ]
                outputs = llm.generate(inputs, greedy_sp)

                zoom_queue  = {}   # idx → (state, zoom_call_tuple)
                next_active = []

                for idx, (s, out) in enumerate(zip(active, outputs)):
                    text = out.outputs[0].text
                    s.n_rounds   = round_idx
                    s.raw_output = text

                    zoom_call = parse_zoom_call(text)

                    if zoom_call is not None and not is_last:
                        # Model called zoom — queue for execution
                        end_pos = text.find("</video_zoom>") + len("</video_zoom>")
                        s.prompt += text[:end_pos]
                        zoom_queue[idx] = (s, zoom_call)
                    else:
                        # Model gave answer, or we hit the hard round limit
                        s.final_answer = extract_mc_answer(text)
                        s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

                if not zoom_queue:
                    active = []  # all samples already in done, skip force-finalize
                    break

                # ── Execute zoom clips (parallel) ──────────────────────── #
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
                    zoom_results = {}
                    for f in as_completed(futures):
                        zoom_results[futures[f]] = f.result()

                # ── Deliver zoom results and continue ──────────────────── #
                for idx, (s, _) in zoom_queue.items():
                    result = zoom_results.get(idx)
                    if isinstance(result, dict):
                        times, frames = result["frame_time"], result["frames"]
                        s.prompt  += build_tool_response_turn(times, is_last=False)
                        s.images  += list(frames)
                        s.n_tool_calls += 1
                        next_active.append(s)
                    else:
                        # Zoom execution failed — finalize with what we have
                        s.final_answer = extract_mc_answer(s.raw_output)
                        s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

                active = next_active

            # Force-finalize any remaining active samples (hit max_rounds)
            for s in active:
                s.final_answer = extract_mc_answer(s.raw_output)
                s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                done.append(s)

            # ── Save records ───────────────────────────────────────────── #
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
                    # Compatible field names with adaptive zoom JSONL
                    "acc_notool":      None,
                    "r1_majority_ans": s.final_answer,
                    "graduated_round": s.n_rounds,
                }
                all_records.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    done_this_run = len(all_records)
    print(f"\n{'='*60}")
    print(f"  Greedy+tool results ({done_this_run} new samples)")
    if done_this_run:
        print(f"  Accuracy (this run): {n_correct}/{done_this_run} = "
              f"{n_correct/done_this_run:.4f}")

    # Full file stats
    with open(output_jsonl) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    accs = [float(r["acc_final"]) for r in all_recs if r.get("acc_final") is not None]
    if accs:
        avg_rounds = np.mean([r.get("n_rounds", 1) for r in all_recs])
        avg_tools  = np.mean([r.get("n_tool_calls", 0) for r in all_recs])
        print(f"  Accuracy (full file, {len(accs)} samples): {np.mean(accs):.4f}")
        print(f"  Avg rounds: {avg_rounds:.2f},  Avg tool calls: {avg_tools:.2f}")
    print(f"  Output: {output_jsonl}")
    print(f"{'='*60}")

    summary = {
        "n_samples":      len(accs),
        "accuracy":       float(np.mean(accs)) if accs else None,
        "mode":           "greedy_T0_natural_tools",
        "fps":            args.fps,
        "frames_upbound": args.frames_upbound,
        "max_rounds":     args.max_rounds,
        "model":          args.model_path,
    }
    with open(os.path.join(args.output_dir, "greedy_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {args.output_dir}/greedy_summary.json")


if __name__ == "__main__":
    main()
