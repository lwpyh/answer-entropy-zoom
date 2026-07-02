#!/usr/bin/env python3
"""
main_infer_embed_zoom.py  ─  Embedding-based zoom trigger

Design:
  Round 1  [TOOL_SYS, greedy, T=0]
    - Model generates <think>...</think><video_zoom>...  OR  <answer>...
    - If direct answer → done
    - If zoom call → extract <think> text → embed with main model's last hidden layer
    - LR classifier: logit = emb @ coef + intercept
      logit < logit_threshold → SKIP zoom (force-answer pass, R1.5)
      logit ≥ logit_threshold → EXECUTE zoom → Round 2+

  Force-answer pass (R1.5) for skipped samples:
    Append "zoom unavailable" user turn → generate <answer> without zoom

  Round 2+  [same as greedy_baseline]

Advantages over HMM keyword approach:
  ✓ Uses full 3584-dim hidden state (vs 4 discrete keyword states)
  ✓ Data-driven classifier (LR trained on 548 paired samples)
  ✓ No hand-crafted regex patterns
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    build_tool_response_turn,
    extract_mc_answer,
    parse_zoom_call,
    score_answer,
    _frame_tokens,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding model helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_embed_model(model_path: str, device: str = 'cuda:0'):
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    print(f'[embed] Loading tokenizer ...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f'[embed] Loading model (bf16) on {device} ...')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f'[embed] Loaded. hidden_size={model.config.hidden_size}')
    return tokenizer, model


@torch.no_grad()
def embed_texts(texts: list, tokenizer, embed_model, device: str,
                max_length: int = 256) -> np.ndarray:
    """
    Embed a list of text strings using the model's last hidden layer.
    Returns L2-normalised float32 numpy array [N, H].
    """
    enc = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    out = embed_model.model(
        input_ids      = enc['input_ids'],
        attention_mask = enc['attention_mask'],
        output_hidden_states = True,
        return_dict    = True,
    )
    last_h = out.hidden_states[-1].float()               # [B, T, H]
    mask   = enc['attention_mask'].unsqueeze(-1).float() # [B, T, 1]
    emb    = (last_h * mask).sum(1) / mask.sum(1).clamp(min=1e-8)  # [B, H]
    emb    = emb.cpu().numpy()
    norms  = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
    return emb / norms


def load_lr_weights(path: str):
    with open(path) as f:
        w = json.load(f)
    coef      = np.array(w['coef'], dtype=np.float32)      # [H]
    intercept = float(w['intercept'])
    threshold = float(w['logit_threshold'])
    print(f'[lr] Loaded weights from {path}')
    print(f'     cv_auc={w.get("cv_auc", "?"):.4f}  '
          f'threshold={threshold:.4f}  skip_rate={w.get("skip_rate", "?")}')
    return coef, intercept, threshold


def extract_think_text(raw_r1: str) -> str:
    m = re.search(r'<think>(.*?)</think>', raw_r1, re.DOTALL)
    if m:
        return m.group(1).strip()
    return re.sub(r'<(?:video_zoom|answer)[^>]*>.*', '', raw_r1,
                  flags=re.DOTALL).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════════

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
        self.raw_output    = ''
        self.zoom_skipped  = False
        self.zoom_triggered = False
        self.embed_logit   = None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_path',   required=True)
    p.add_argument('--video_root',  default='/data/DERI-Gong/jh015/VideoZoomer')
    p.add_argument('--model_path',  default='zsgvivo/videozoomer')
    p.add_argument('--output_dir',  default='./infer_results/embed_zoom')
    p.add_argument('--lr_weights',  required=True,
                   help='Path to analysis_hmm_hidden_states_lr_weights.json')

    # vLLM
    p.add_argument('--gpu_memory_utilization', type=float, default=0.45)
    p.add_argument('--tensor_parallel_size',   type=int,   default=2)
    p.add_argument('--max_model_len',          type=int,   default=32768)
    p.add_argument('--max_pixels',             type=int,   default=100352)
    p.add_argument('--min_pixels',             type=int,   default=25088)

    # Video & tool
    p.add_argument('--fps',                      type=float, default=0.5)
    p.add_argument('--frames_upbound',           type=int,   default=64)
    p.add_argument('--max_tokens',               type=int,   default=4096)
    p.add_argument('--tool_limit_mm',            type=int,   default=128)
    p.add_argument('--tool_max_frames_per_call', type=int,   default=16)
    p.add_argument('--tool_workers',             type=int,   default=8)
    p.add_argument('--max_rounds',               type=int,   default=5)

    # Embedding device (alongside vLLM)
    p.add_argument('--embed_device',  default='cuda:0',
                   help='Device for the embedding forward pass (separate from vLLM)')
    p.add_argument('--embed_batch',   type=int, default=16,
                   help='Batch size for embedding inference')
    p.add_argument('--embed_max_len', type=int, default=256,
                   help='Max token length per think chain for embedding')

    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, 'results_embed_zoom.jsonl')

    # ── LR weights ────────────────────────────────────────────────────────── #
    coef, intercept, logit_threshold = load_lr_weights(args.lr_weights)

    # ── Data ─────────────────────────────────────────────────────────────── #
    print(f'[data] Loading {args.data_path}')
    samples = load_dataset(args.data_path)
    print(f'[data] {len(samples)} samples')

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # ── Load vLLM first ───────────────────────────────────────────────────── #
    from vllm import LLM, SamplingParams
    from verl.workers.rollout.vllm_rollout.function_tools import extract_video_clip

    max_mm = max(args.frames_upbound, args.tool_limit_mm)
    print(f'[vLLM] Initialising (tp={args.tensor_parallel_size}, '
          f'gpu_util={args.gpu_memory_utilization}) ...')
    llm = LLM(
        model                  = args.model_path,
        tensor_parallel_size   = args.tensor_parallel_size,
        gpu_memory_utilization = args.gpu_memory_utilization,
        max_model_len          = args.max_model_len,
        dtype                  = 'bfloat16',
        trust_remote_code      = True,
        mm_processor_kwargs    = {'max_pixels': args.max_pixels,
                                  'min_pixels': args.min_pixels},
        limit_mm_per_prompt    = {'image': max_mm},
        enforce_eager          = False,
        enable_prefix_caching  = False,
    )

    # ── Load embedding model into remaining GPU memory ────────────────────── #
    em_tok, em_model = load_embed_model(args.model_path, args.embed_device)

    greedy_sp = SamplingParams(
        n=1, temperature=0.0, max_tokens=args.max_tokens,
        stop=['</video_zoom>', '</answer>'],
        include_stop_str_in_output=True, detokenize=True,
    )
    force_ans_sp = SamplingParams(
        n=1, temperature=0.0, max_tokens=args.max_tokens,
        stop=['</answer>'],
        include_stop_str_in_output=True, detokenize=True,
    )

    print(f'[run] Embedding-based zoom gate  '
          f'fps={args.fps}  logit_thr={logit_threshold:.3f}')

    # ── Resume ────────────────────────────────────────────────────────────── #
    done_pids, file_mode = set(), 'w'
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                try: done_pids.add(json.loads(line)['problem_id'])
                except: pass
        if done_pids:
            print(f'[resume] {len(done_pids)} done, skipping')
            file_mode = 'a'

    pending = [
        s for s in samples
        if str(s.get('problem_id', '')) not in done_pids
        and str(s.get('extra_info', {}).get('problem_id', '')) not in done_pids
    ]

    all_records = []
    n_correct = n_zoom_triggered = n_zoom_skipped = n_direct = 0

    with open(output_jsonl, file_mode) as out_f:
        for batch_start in tqdm(
            range(0, len(pending), args.batch_size),
            desc='Batches',
            total=(len(pending) + args.batch_size - 1) // args.batch_size,
        ):
            batch = pending[batch_start: batch_start + args.batch_size]

            # ── Preprocess ────────────────────────────────────────────────── #
            active = []
            for sample in batch:
                pid      = (sample.get('problem_id') or
                            sample.get('extra_info', {}).get('problem_id', '?'))
                gt       = sample.get('solution', '')
                question = sample.get('problem', '')
                video_rel = (sample.get('videos') or [''])[0]
                video_path = resolve_video_path(video_rel, args.video_root)
                try:
                    frame_times, frames = load_video_frames(
                        video_path, args.fps, args.max_pixels, args.min_pixels,
                        args.frames_upbound)
                    prompt = build_tool_initial_prompt(question, frame_times, processor)
                    active.append(SampleState(pid, gt, video_path, prompt, list(frames)))
                except Exception as e:
                    print(f'\n[prep] skip {pid}: {e}')

            if not active:
                continue

            done = []

            # ══════════════════════════════════════════════════════════════════
            # Round 1: TOOL_SYS greedy
            # ══════════════════════════════════════════════════════════════════
            inputs  = [{'prompt': s.prompt,
                        'multi_modal_data': {'image': list(s.images)}}
                       for s in active]
            outputs = llm.generate(inputs, greedy_sp)

            zoom_pending      = []   # (state, think_text) — waiting for embed decision
            force_answer_queue = {}  # idx → state (skip zoom)
            zoom_queue        = {}   # idx → (state, zoom_call)
            next_active       = []

            for s, out in zip(active, outputs):
                text      = out.outputs[0].text
                s.n_rounds   = 1
                s.raw_output = text
                zoom_call = parse_zoom_call(text)

                if zoom_call is None:
                    # Direct answer, no zoom
                    s.final_answer = extract_mc_answer(text)
                    s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                    done.append(s)
                else:
                    think_text = extract_think_text(text)
                    zoom_pending.append((s, text, think_text, zoom_call))

            # ── Batch embed all think chains that called zoom ──────────────── #
            if zoom_pending:
                think_texts = [tp[2] if tp[2] else tp[1][:512]
                               for tp in zoom_pending]

                # Batch embedding
                all_logits = []
                for i in range(0, len(think_texts), args.embed_batch):
                    batch_t = think_texts[i: i + args.embed_batch]
                    embs    = embed_texts(batch_t, em_tok, em_model,
                                         args.embed_device, args.embed_max_len)
                    logits  = embs @ coef + intercept
                    all_logits.extend(logits.tolist())

                for batch_idx, ((s, text, think_text, zoom_call), logit) \
                        in enumerate(zip(zoom_pending, all_logits)):
                    s.embed_logit = logit
                    end_pos = text.find('</video_zoom>') + len('</video_zoom>')

                    if logit < logit_threshold:
                        # ── Confident no-zoom → force-answer pass ─────────── #
                        s.prompt += text[:end_pos]
                        s.prompt += (
                            '\n<|im_end|>\n<|im_start|>user\n'
                            '<tool_response>\n'
                            'The zoom tool is unavailable. Based on the video frames '
                            'already shown, give your best final answer now.\n'
                            '</tool_response>\n'
                            'Do not call <video_zoom>. '
                            'Write your final answer inside <answer> and </answer>.'
                            '<|im_end|>\n<|im_start|>assistant\n'
                        )
                        s.zoom_skipped = True
                        force_answer_queue[batch_idx] = s
                    else:
                        # ── Uncertain → execute zoom ───────────────────────── #
                        s.prompt += text[:end_pos]
                        s.zoom_triggered = True
                        zoom_queue[batch_idx] = (s, zoom_call)

            # ── Force-answer pass ─────────────────────────────────────────── #
            if force_answer_queue:
                fa_inputs = [
                    {'prompt': s.prompt, 'multi_modal_data': {'image': list(s.images)}}
                    for s in force_answer_queue.values()
                ]
                fa_outputs = llm.generate(fa_inputs, force_ans_sp)
                for s, out in zip(force_answer_queue.values(), fa_outputs):
                    fa_text = out.outputs[0].text
                    s.raw_output   = fa_text
                    s.n_rounds     = 2
                    s.final_answer = extract_mc_answer(fa_text)
                    s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                    done.append(s)

            # ── Execute zoom clips (parallel) ─────────────────────────────── #
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
                            storage_system = 'local',
                        ): idx
                        for idx, (s, (s_t, e_t, fps_z)) in zoom_queue.items()
                    }
                    zoom_results = {futures[f]: f.result() for f in as_completed(futures)}

                for idx, (s, _) in zoom_queue.items():
                    result = zoom_results.get(idx)
                    if isinstance(result, dict):
                        times, frames = result['frame_time'], result['frames']
                        s.prompt  += build_tool_response_turn(times, is_last=False)
                        s.images  += list(frames)
                        s.n_tool_calls += 1
                        next_active.append(s)
                    else:
                        s.final_answer = extract_mc_answer(s.raw_output)
                        s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

            active = next_active

            # ══════════════════════════════════════════════════════════════════
            # Round 2+: standard greedy continuation
            # ══════════════════════════════════════════════════════════════════
            for round_idx in range(2, args.max_rounds + 1):
                is_last = (round_idx == args.max_rounds)
                if not active:
                    break

                inputs  = [{'prompt': s.prompt,
                            'multi_modal_data': {'image': list(s.images)}}
                           for s in active]
                outputs = llm.generate(inputs, greedy_sp)

                zoom_queue2  = {}
                next_active2 = []

                for idx, (s, out) in enumerate(zip(active, outputs)):
                    text      = out.outputs[0].text
                    s.n_rounds   = round_idx
                    s.raw_output = text
                    zoom_call = parse_zoom_call(text)
                    if zoom_call is not None and not is_last:
                        end_pos  = text.find('</video_zoom>') + len('</video_zoom>')
                        s.prompt += text[:end_pos]
                        zoom_queue2[idx] = (s, zoom_call)
                    else:
                        s.final_answer = extract_mc_answer(text)
                        s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                        done.append(s)

                if zoom_queue2:
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
                                storage_system = 'local',
                            ): idx
                            for idx, (s, (s_t, e_t, fps_z)) in zoom_queue2.items()
                        }
                        zoom_results2 = {futures[f]: f.result()
                                         for f in as_completed(futures)}

                    for idx, (s, _) in zoom_queue2.items():
                        result = zoom_results2.get(idx)
                        if isinstance(result, dict):
                            times, frames = result['frame_time'], result['frames']
                            s.prompt  += build_tool_response_turn(times, is_last=False)
                            s.images  += list(frames)
                            s.n_tool_calls += 1
                            next_active2.append(s)
                        else:
                            s.final_answer = extract_mc_answer(s.raw_output)
                            s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                            done.append(s)

                active = next_active2

            for s in active:
                s.final_answer = extract_mc_answer(s.raw_output)
                s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                done.append(s)

            # ── Save ──────────────────────────────────────────────────────── #
            for s in done:
                n_correct += int(s.acc_final or 0)
                if s.zoom_triggered:  n_zoom_triggered += 1
                if s.zoom_skipped:    n_zoom_skipped   += 1
                if not s.zoom_triggered and not s.zoom_skipped:
                    n_direct += 1

                rec = {
                    'problem_id':     s.pid,
                    'gt':             s.gt,
                    'acc_final':      s.acc_final,
                    'final_answer':   s.final_answer,
                    'n_rounds':       s.n_rounds,
                    'n_tool_calls':   s.n_tool_calls,
                    'raw_output':     s.raw_output,
                    'zoom_triggered': s.zoom_triggered,
                    'zoom_skipped':   s.zoom_skipped,
                    'embed_logit':    s.embed_logit,
                }
                all_records.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out_f.flush()

    # ── Summary ───────────────────────────────────────────────────────────── #
    with open(output_jsonl) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]

    accs = [float(r['acc_final']) for r in all_recs if r.get('acc_final') is not None]
    n_zoom  = sum(1 for r in all_recs if r.get('zoom_triggered'))
    n_skip  = sum(1 for r in all_recs if r.get('zoom_skipped'))
    n_dir   = sum(1 for r in all_recs if not r.get('zoom_triggered')
                                      and not r.get('zoom_skipped'))

    def grp_acc(fn):
        sub = [float(r['acc_final']) for r in all_recs if fn(r)
               and r.get('acc_final') is not None]
        return np.mean(sub) if sub else float('nan')

    print(f'\n{"="*60}')
    print(f'  Embedding-Zoom inference ({len(accs)} samples)')
    print(f'  Overall acc = {np.mean(accs):.4f}')
    print(f'  ├─ Direct answer  ({n_dir:4d}):  {grp_acc(lambda r: not r.get("zoom_triggered") and not r.get("zoom_skipped")):.4f}')
    print(f'  ├─ Embed skip     ({n_skip:4d}):  {grp_acc(lambda r: r.get("zoom_skipped")):.4f}')
    print(f'  └─ Zoom executed  ({n_zoom:4d}):  {grp_acc(lambda r: r.get("zoom_triggered")):.4f}')
    avg_tools = np.mean([r.get('n_tool_calls', 0) for r in all_recs])
    print(f'  Avg tool calls: {avg_tools:.3f}  '
          f'(greedy baseline ~0.943)')
    print(f'  Output: {output_jsonl}')
    print(f'{"="*60}')

    summary = {
        'n_samples':      len(accs),
        'accuracy':       float(np.mean(accs)) if accs else None,
        'mode':           'embed_zoom',
        'logit_threshold': logit_threshold,
        'zoom_triggered': n_zoom,
        'zoom_skipped':   n_skip,
        'direct_answer':  n_dir,
        'fps':            args.fps,
        'model':          args.model_path,
    }
    with open(os.path.join(args.output_dir, 'embed_zoom_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
