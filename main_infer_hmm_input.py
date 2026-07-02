#!/usr/bin/env python3
"""
main_infer_hmm_input.py  ─  Input-hidden-state HMM zoom trigger

Design (pre-generation decision, no distribution shift):
  For each sample, BEFORE any generation:
    1. Load video frames
    2. Run vision encoder → per-frame embeddings [N_frames, 3584]
    3. PCA reduce → [N_frames, 32]
    4. log_ratio = HMM_zoom.score(seq) − HMM_nozoom.score(seq)
    5. log_ratio < threshold → skip zoom → NOTOOL_SYS direct answer
       log_ratio ≥ threshold → execute zoom → TOOL_SYS R1 + R2

Key advantage: decision uses video content representation, not generated text.
No TOOL_SYS/NOTOOL_SYS distribution shift.
"""

import argparse
import json
import os
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from main_infer_adaptive_zoom import (
    TOOL_SYS, NOTOOL_SYS,
    load_dataset, resolve_video_path, load_video_frames,
    build_tool_initial_prompt, build_tool_response_turn,
    build_notool_prompt,
    extract_mc_answer, parse_zoom_call, score_answer, _frame_tokens,
)


IMAGE_TOKEN_ID = 151655   # Qwen2.5-VL <|image_pad|>
VIDEO_TOKEN_ID = 151656   # Qwen2.5-VL <|video_pad|>


# ─── Decoder prefill helpers ──────────────────────────────────────────────── #

def load_decoder_model(model_path: str, device: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc  = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    return proc, model


@torch.no_grad()
def get_frame_hidden_states(question, frames, proc, model, device):
    """
    Decoder prefill on (video+query) → multi-layer hidden states at image tokens.
    Returns [N_frames, 4*hidden_size] float32 numpy array.
    Hidden states from 4 representative layers (L//4, L//2, 3L//4, L),
    mean-pooled over image tokens per frame then concatenated across layers.
    """
    try:
        messages = [
            {"role": "system", "content": NOTOOL_SYS},
            {"role": "user",   "content": [
                {"type": "video", "video": frames},
                {"type": "text",  "text": question},
            ]},
        ]
        text   = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=None, videos=[frames],
                      return_tensors='pt', padding=True).to(device)
    except Exception:
        messages[1]["content"] = (
            [{"type": "image"}] * len(frames) + [{"type": "text", "text": question}]
        )
        text   = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=list(frames),
                      return_tensors='pt', padding=True).to(device)

    out      = model(**inputs, output_hidden_states=True, return_dict=True)
    hs       = out.hidden_states               # tuple: (n_layers+1) x [1, seq_len, H]
    ids      = inputs['input_ids'][0]
    img_mask = (ids == IMAGE_TOKEN_ID) | (ids == VIDEO_TOKEN_ID)

    # Select 4 representative layers: L//4, L//2, 3L//4, L (last)
    n_hs = len(hs)
    layer_idx = [n_hs // 4, n_hs // 2, 3 * n_hs // 4, n_hs - 1]
    layers_img = torch.stack(
        [hs[i][0].float()[img_mask] for i in layer_idx], dim=0
    )  # [4, n_img_tokens, H]

    if layers_img.shape[1] == 0:
        return np.zeros((1, 4 * model.config.hidden_size), dtype=np.float32)

    thw_list = inputs.get('video_grid_thw') or inputs.get('image_grid_thw')
    if thw_list is None:
        vec = layers_img.mean(dim=1).flatten().cpu().numpy()
        return vec.reshape(1, -1)

    embs, idx = [], 0
    for thw in thw_list:                  # iterate over all clips/images
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
        # t = temporal tokens; each temporal slot has h*w spatial tokens
        for _ in range(t):
            chunk = layers_img[:, idx: idx + h * w, :]         # [4, h*w, H]
            embs.append(chunk.mean(dim=1).flatten().cpu().numpy())  # [4H]
            idx += h * w
    return np.array(embs, dtype=np.float32)                    # [N_frames, 4H]


def load_hmm_artifacts(analysis_path: str):
    with open(analysis_path) as f:
        meta = json.load(f)
    with open(meta['hmm_path'], 'rb') as f: hmm_model = pickle.load(f)
    with open(meta['pca_path'], 'rb') as f: pca_data  = pickle.load(f)
    uncertain_state = meta['uncertain_state']
    threshold       = meta['default_threshold']   # uncertain_frac threshold
    print(f'[hmm] Loaded. K={meta["n_components"]}  pca_dim={meta["pca_dim"]}  '
          f'uncertain_state=S{uncertain_state}  thr={threshold:.3f}  '
          f'auc={meta.get("auc", "?"):.4f}')
    return hmm_model, uncertain_state, pca_data['pca'], pca_data['scaler'], threshold


def uncertain_fraction(seq_pca, hmm_model, uncertain_state):
    """Proportion of frames assigned to the 'uncertain' HMM state."""
    try:
        states = hmm_model.predict(seq_pca)
        return float((states == uncertain_state).mean())
    except Exception:
        return 0.5   # tie → execute zoom


# ─── Per-sample state ─────────────────────────────────────────────────────── #

class SampleState:
    def __init__(self, pid, gt, video_path, images, frame_times):
        self.pid          = pid
        self.gt           = gt
        self.video_path   = video_path
        self.images       = images
        self.frame_times  = frame_times
        self.n_tool_calls = 0
        self.n_rounds     = 0
        self.final_answer = None
        self.acc_final    = None
        self.raw_output   = ''
        self.zoom_skipped  = False
        self.zoom_triggered = False
        self.log_ratio    = None
        self.prompt       = None


# ─── Args ─────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data_path',    required=True)
    p.add_argument('--video_root',   default='/data/DERI-Gong/jh015/VideoZoomer')
    p.add_argument('--model_path',   default='zsgvivo/videozoomer')
    p.add_argument('--output_dir',   default='./infer_results/hmm_input')
    p.add_argument('--hmm_analysis', required=True,
                   help='Path to analysis_input_hmm.json')

    # vLLM
    p.add_argument('--gpu_memory_utilization', type=float, default=0.45)
    p.add_argument('--tensor_parallel_size',   type=int,   default=2)
    p.add_argument('--max_model_len',          type=int,   default=32768)
    p.add_argument('--max_pixels',             type=int,   default=100352)
    p.add_argument('--min_pixels',             type=int,   default=25088)

    # Video
    p.add_argument('--fps',                      type=float, default=0.5)
    p.add_argument('--frames_upbound',           type=int,   default=64)
    p.add_argument('--max_tokens',               type=int,   default=4096)
    p.add_argument('--tool_limit_mm',            type=int,   default=128)
    p.add_argument('--tool_max_frames_per_call', type=int,   default=16)
    p.add_argument('--tool_workers',             type=int,   default=8)
    p.add_argument('--max_rounds',               type=int,   default=5)

    p.add_argument('--embed_device', default='cuda:0')
    p.add_argument('--batch_size',   type=int, default=32)
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, 'results_hmm_input.jsonl')

    # Load HMM artifacts
    hmm_model, uncertain_state, pca, scaler, unc_threshold = \
        load_hmm_artifacts(args.hmm_analysis)

    # Load data
    print(f'[data] Loading {args.data_path}')
    samples = load_dataset(args.data_path)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Load vLLM (generation)
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

    # Load decoder model into remaining GPU memory (for prefill)
    dec_proc, dec_model = load_decoder_model(args.model_path, args.embed_device)

    greedy_sp = SamplingParams(
        n=1, temperature=0.0, max_tokens=args.max_tokens,
        stop=['</video_zoom>', '</answer>'],
        include_stop_str_in_output=True, detokenize=True,
    )
    notool_sp = SamplingParams(
        n=1, temperature=0.0, max_tokens=args.max_tokens,
        stop=['</answer>'],
        include_stop_str_in_output=True, detokenize=True,
    )

    print(f'[run] Input-HMM zoom gate  uncertain_state=S{uncertain_state}  thr={unc_threshold:.3f}')

    # Resume
    done_pids, file_mode = set(), 'w'
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as f:
            for line in f:
                try: done_pids.add(json.loads(line)['problem_id'])
                except: pass
        if done_pids:
            print(f'[resume] {len(done_pids)} done')
            file_mode = 'a'

    pending = [
        s for s in samples
        if str(s.get('problem_id', '')) not in done_pids
        and str(s.get('extra_info', {}).get('problem_id', '')) not in done_pids
    ]

    all_records = []
    n_correct = n_zoom_triggered = n_zoom_skipped = 0

    with open(output_jsonl, file_mode) as out_f:
        for batch_start in tqdm(
            range(0, len(pending), args.batch_size),
            desc='Batches',
            total=(len(pending) + args.batch_size - 1) // args.batch_size,
        ):
            batch = pending[batch_start: batch_start + args.batch_size]

            # ── Preprocess + HMM gate (before any generation) ─────────────── #
            skip_batch  = []   # NOTOOL_SYS direct answer
            zoom_batch  = []   # TOOL_SYS with zoom

            for sample in batch:
                pid      = (sample.get('problem_id') or
                            sample.get('extra_info', {}).get('problem_id', '?'))
                gt       = sample.get('solution', '')
                video_rel = (sample.get('videos') or [''])[0]
                video_path = resolve_video_path(video_rel, args.video_root)
                try:
                    frame_times, frames = load_video_frames(
                        video_path, args.fps, args.max_pixels, args.min_pixels,
                        args.frames_upbound)
                    s = SampleState(pid, gt, video_path, list(frames), frame_times)

                    # Decoder prefill → HMM uncertain-state fraction → decision
                    embs    = get_frame_hidden_states(
                        sample.get('problem', ''), frames,
                        dec_proc, dec_model, args.embed_device)
                    seq_pca = pca.transform(scaler.transform(embs))
                    unc_frac = uncertain_fraction(seq_pca, hmm_model, uncertain_state)
                    s.log_ratio = unc_frac   # reuse field for output

                    if unc_frac < unc_threshold:   # low uncertain fraction → skip zoom
                        # Confident no-zoom → NOTOOL_SYS
                        s.zoom_skipped = True
                        s.prompt = build_notool_prompt(
                            sample.get('problem', ''), frame_times, processor)
                        skip_batch.append(s)
                    else:
                        # Needs zoom → TOOL_SYS
                        s.zoom_triggered = True
                        s.prompt = build_tool_initial_prompt(
                            sample.get('problem', ''), frame_times, processor)
                        zoom_batch.append(s)

                except Exception as e:
                    print(f'\n[prep] skip {pid}: {e}')

            done = []

            # ══════════════════════════════════════════════════════════════════
            # Path A: NOTOOL_SYS direct answer (zoom skipped)
            # ══════════════════════════════════════════════════════════════════
            if skip_batch:
                inputs = [{'prompt': s.prompt,
                           'multi_modal_data': {'image': list(s.images)}}
                          for s in skip_batch]
                outputs = llm.generate(inputs, notool_sp)
                for s, out in zip(skip_batch, outputs):
                    text = out.outputs[0].text
                    s.n_rounds     = 1
                    s.raw_output   = text
                    s.final_answer = extract_mc_answer(text)
                    s.acc_final    = score_answer(s.final_answer, s.gt) if s.final_answer else 0.0
                    done.append(s)

            # ══════════════════════════════════════════════════════════════════
            # Path B: TOOL_SYS with zoom → R1 + zoom execution + R2+
            # ══════════════════════════════════════════════════════════════════
            active = zoom_batch

            for round_idx in range(1, args.max_rounds + 1):
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
                    text = out.outputs[0].text
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
                                start_time     = s_t, end_time = e_t, fps = fps_z,
                                max_pixels     = args.max_pixels,
                                min_pixels     = args.min_pixels,
                                max_frames     = args.tool_max_frames_per_call,
                                storage_system = 'local',
                            ): idx
                            for idx, (s, (s_t, e_t, fps_z)) in zoom_queue2.items()
                        }
                        results = {futures[f]: f.result() for f in as_completed(futures)}

                    for idx, (s, _) in zoom_queue2.items():
                        res = results.get(idx)
                        if isinstance(res, dict):
                            s.prompt  += build_tool_response_turn(res['frame_time'], is_last=False)
                            s.images  += list(res['frames'])
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
                    'log_ratio':      s.log_ratio,
                }
                all_records.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out_f.flush()

    # ── Summary ───────────────────────────────────────────────────────────── #
    with open(output_jsonl) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]

    accs   = [float(r['acc_final']) for r in all_recs if r.get('acc_final') is not None]
    n_zoom = sum(1 for r in all_recs if r.get('zoom_triggered'))
    n_skip = sum(1 for r in all_recs if r.get('zoom_skipped'))

    def grp(fn):
        sub = [float(r['acc_final']) for r in all_recs if fn(r) and r.get('acc_final') is not None]
        return np.mean(sub) if sub else float('nan')

    print(f'\n{"="*60}')
    print(f'  Input-HMM Zoom ({len(accs)} samples)  acc={np.mean(accs):.4f}')
    print(f'  ├─ Zoom skip (NOTOOL_SYS) ({n_skip:4d}):  {grp(lambda r: r.get("zoom_skipped")):.4f}')
    print(f'  └─ Zoom exec (TOOL_SYS)   ({n_zoom:4d}):  {grp(lambda r: r.get("zoom_triggered")):.4f}')
    avg_tc = np.mean([r.get('n_tool_calls', 0) for r in all_recs])
    print(f'  Avg tool calls: {avg_tc:.3f}  (greedy ~0.943)')
    print(f'  Output: {output_jsonl}')
    print(f'{"="*60}')

    summary = {
        'n_samples': len(accs), 'accuracy': float(np.mean(accs)) if accs else None,
        'mode': 'hmm_input', 'uncertain_threshold': unc_threshold,
        'zoom_triggered': n_zoom, 'zoom_skipped': n_skip,
        'fps': args.fps, 'model': args.model_path,
    }
    with open(os.path.join(args.output_dir, 'hmm_input_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
