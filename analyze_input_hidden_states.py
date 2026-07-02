#!/usr/bin/env python3
"""
analyze_input_hidden_states.py

Build an unsupervised GaussianHMM on decoder hidden states
after full video+query prefill (not just vision encoder).

Design:
  For each sample (video + question):
    1. Build prompt = system + image_tokens + question
    2. Run decoder prefill (forward pass, no generation)
    3. Extract last-layer hidden states at image token positions
    4. Mean-pool per frame → [N_frames, 3584]

  Fit ONE unsupervised GaussianHMM (K states, no labels).
  HMM discovers structure: e.g. "clear understanding" vs "uncertain" states.

  At inference time:
    - Run prefill → frame hidden states → HMM Viterbi decode
    - Proportion of "uncertain" states (high emission variance) > threshold → ZOOM
    - Otherwise → SKIP zoom (answer directly with NOTOOL_SYS)

  No zoom_benefit labels required.

Usage:
  python3 analyze_input_hidden_states.py \
      --model_path zsgvivo/videozoomer \
      --tv_results  infer_results/temp_vote/results_tvote.jsonl \
      --greedy_results infer_results/greedy_baseline/results_greedy.jsonl \
      --embed_cache input_decoder_embeddings.npz \
      --output_path analysis_input_hmm.json
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from main_infer_adaptive_zoom import (
    load_dataset, resolve_video_path, load_video_frames, NOTOOL_SYS,
)

IMAGE_TOKEN_ID = 151655   # Qwen2.5-VL <|image_pad|>
VIDEO_TOKEN_ID = 151656   # Qwen2.5-VL <|video_pad|>


# ─── Decoder prefill + hidden state extraction ───────────────────────────── #

def load_model(model_path, device):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f'[model] Loading processor ...')
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f'[model] Loading model (bf16) ...')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f'[model] Ready. hidden_size={model.config.hidden_size}')
    return proc, model


@torch.no_grad()
def get_frame_hidden_states(question, frames, proc, model, device):
    """
    Run decoder prefill on (video frames + question).
    Returns [N_frames, 4*hidden_size] float32 numpy array:
      hidden states from 4 representative layers (L//4, L//2, 3L//4, L)
      at image token positions, mean-pooled per frame then concatenated.
    """
    # Build minimal chat prompt (no tool system prompt — we want clean visual alignment)
    messages = [
        {"role": "system",  "content": NOTOOL_SYS},
        {"role": "user",    "content": [
            {"type": "video", "video": frames},   # list of PIL images treated as video
            {"type": "text",  "text": question},
        ]},
    ]
    try:
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(
            text=[text], images=None, videos=[frames],
            return_tensors='pt', padding=True,
        ).to(device)
    except Exception:
        # Fallback: pass frames as images
        messages[1]["content"] = (
            [{"type": "image"}] * len(frames) + [{"type": "text", "text": question}]
        )
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(
            text=[text], images=list(frames),
            return_tensors='pt', padding=True,
        ).to(device)

    # Decoder prefill — extract hidden states from all layers
    out = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    hs        = out.hidden_states              # tuple: (n_layers+1) x [1, seq_len, H]
    input_ids = inputs['input_ids'][0]         # [seq_len]
    # video input uses <|video_pad|>=151656; image fallback uses <|image_pad|>=151655
    img_mask  = (input_ids == IMAGE_TOKEN_ID) | (input_ids == VIDEO_TOKEN_ID)

    # Select 4 representative layers: L//4, L//2, 3L//4, L (last)
    n_hs = len(hs)
    layer_idx = [n_hs // 4, n_hs // 2, 3 * n_hs // 4, n_hs - 1]
    # Extract image-token rows from each selected layer → [4, n_img_tokens, H]
    layers_img = torch.stack(
        [hs[i][0].float()[img_mask] for i in layer_idx], dim=0
    )  # [4, n_img_tokens, H]

    if layers_img.shape[1] == 0:
        return np.zeros((1, 4 * model.config.hidden_size), dtype=np.float32)

    # Group by frame using grid_thw (video_grid_thw for video input, image_grid_thw for images)
    _dbg_keys = [k for k in inputs.keys() if 'thw' in k or 'grid' in k]
    print(f'[DBG] thw keys={_dbg_keys}  '
          f'video_grid_thw={inputs.get("video_grid_thw", "MISSING").__class__.__name__ if inputs.get("video_grid_thw") is not None else "None"}  '
          f'img={inputs.get("image_grid_thw", "MISSING").__class__.__name__ if inputs.get("image_grid_thw") is not None else "None"}',
          flush=True)
    thw_list = inputs.get('video_grid_thw') if inputs.get('video_grid_thw') is not None else inputs.get('image_grid_thw')
    if thw_list is None:
        # Fallback: treat all image tokens as one frame
        vec = layers_img.mean(dim=1).flatten().cpu().numpy()   # [4H]
        return vec.reshape(1, -1)

    frame_embs, idx = [], 0
    for thw in thw_list:                  # iterate over all clips/images
        t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
        # t = temporal tokens; each temporal slot has h*w spatial tokens
        for _ in range(t):
            chunk = layers_img[:, idx: idx + h * w, :]         # [4, h*w, H]
            vec   = chunk.mean(dim=1).flatten().cpu().numpy()  # [4H]
            frame_embs.append(vec)
            idx += h * w

    return np.array(frame_embs, dtype=np.float32)  # [N_frames, 4H]


# ─── HMM helpers ─────────────────────────────────────────────────────────── #

def fit_hmm(sequences, n_components, n_iter=300, random_state=42):
    from hmmlearn import hmm as hmmlib
    X       = np.vstack(sequences)
    lengths = [len(s) for s in sequences]
    m = hmmlib.GaussianHMM(
        n_components=n_components, covariance_type='diag',
        n_iter=n_iter, random_state=random_state,
    )
    m.fit(X, lengths)
    return m


def state_uncertainty(hmm_model):
    """
    For each HMM state, compute mean diagonal variance (emission uncertainty).
    Higher = the model's hidden states are more spread out in this state = more uncertain.
    """
    covars = np.asarray(hmm_model.covars_)
    print(f'[DBG] covars_ shape={covars.shape}', flush=True)
    # spherical: [K]; diag: [K, n_features]; full: [K, n, n]
    if covars.ndim == 1:
        return covars                                    # [K]
    return covars.reshape(covars.shape[0], -1).mean(axis=1)  # [K]


def print_transition(T, title):
    K = T.shape[0]
    print(f'\n  {title}')
    header = '         ' + '  '.join(f'  S{j}' for j in range(K))
    print(f'  {header}')
    for i in range(K):
        row = '  '.join(f'{T[i,j]:+.4f}' for j in range(K))
        print(f'  S{i}   {row}')


# ─── Main ─────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',     default='zsgvivo/videozoomer')
    p.add_argument('--video_root',     default='/data/DERI-Gong/jh015/VideoZoomer')
    p.add_argument('--tv_results',     default='infer_results/temp_vote/results_tvote.jsonl')
    p.add_argument('--greedy_results', default='infer_results/greedy_baseline/results_greedy.jsonl')
    p.add_argument('--output_path',    default='analysis_input_hmm.json')
    p.add_argument('--embed_cache',    default='input_decoder_embeddings.npz')
    p.add_argument('--no_model',       action='store_true')
    p.add_argument('--device',         default='cuda')
    p.add_argument('--fps',            type=float, default=0.5)
    p.add_argument('--frames_upbound', type=int,   default=64)
    p.add_argument('--max_pixels',     type=int,   default=100352)
    p.add_argument('--min_pixels',     type=int,   default=25088)
    p.add_argument('--n_components',   type=int,   default=4)
    p.add_argument('--pca_dim',        type=int,   default=32)
    p.add_argument('--max_duration',   type=float, default=None,
                   help='Skip videos longer than this many seconds (e.g. 3600 for 1h)')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load paired samples (for calibrating threshold after HMM fitting) ─── #
    print('[data] Loading paired samples ...')
    tv_recs   = [json.loads(l) for l in open(args.tv_results)]
    grdy_recs = [json.loads(l) for l in open(args.greedy_results)]
    grdy_acc  = {r['problem_id']: float(r['acc_final']) for r in grdy_recs}

    all_samples = load_dataset('longvideo-reason/eval_longvideoreason.yaml')
    pid_to_sample = {
        str(s.get('problem_id') or s.get('extra_info', {}).get('problem_id', '')): s
        for s in all_samples
    }

    tv_no_zoom = [r for r in tv_recs if not r.get('zoom_triggered')]
    paired = []
    for r in tv_no_zoom:
        pid = str(r['problem_id'])
        if pid not in grdy_acc or pid not in pid_to_sample:
            continue
        label = int(grdy_acc[pid] > float(r['acc_final']))
        paired.append({'pid': pid, 'label': label, 'sample': pid_to_sample[pid]})

    n_zoom = sum(p['label'] for p in paired)
    print(f'  paired={len(paired)}  needs_zoom={n_zoom}  skip_zoom={len(paired)-n_zoom}')

    # ── Extract or load decoder hidden states ─────────────────────────────── #
    partial_cache = args.embed_cache.replace('.npz', '_partial.npz')

    if args.no_model and os.path.exists(args.embed_cache):
        print(f'[embed] Loading from cache {args.embed_cache}')
        cache      = np.load(args.embed_cache, allow_pickle=True)
        all_embs   = list(cache['embeddings'])
        all_labels = list(cache['labels'].astype(int))
        all_pids   = list(cache.get('pids', np.array([])))
        print(f'  Loaded {len(all_embs)} samples.')
    else:
        # Resume from partial cache if available
        all_embs, all_labels, all_pids = [], [], []
        done_pids = set()
        if os.path.exists(partial_cache):
            try:
                pc = np.load(partial_cache, allow_pickle=True)
                all_embs   = list(pc['embeddings'])
                all_labels = list(pc['labels'].astype(int))
                all_pids   = list(pc['pids'])
                done_pids  = set(all_pids)
                print(f'[embed] Resuming from partial cache: {len(done_pids)} done')
            except Exception as e:
                print(f'[embed] Could not load partial cache: {e}')

        proc, model = load_model(args.model_path, args.device)

        for p in tqdm(paired, desc='Decoder prefill'):
            if p['pid'] in done_pids:
                continue
            s          = p['sample']
            video_rel  = (s.get('videos') or [''])[0]
            video_path = resolve_video_path(video_rel, args.video_root)
            question   = s.get('problem', '')
            # Duration filter for debug runs
            if args.max_duration is not None and video_path and os.path.exists(video_path):
                try:
                    import subprocess
                    r = subprocess.run(
                        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                         '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
                        capture_output=True, text=True, timeout=10)
                    dur = float(r.stdout.strip()) if r.stdout.strip() else 0.0
                    if dur > args.max_duration:
                        continue
                except Exception:
                    pass
            try:
                _, frames = load_video_frames(
                    video_path, args.fps, args.max_pixels, args.min_pixels,
                    args.frames_upbound)
                embs = get_frame_hidden_states(question, frames, proc, model, args.device)
            except Exception as e:
                print(f'\n  skip {p["pid"]}: {e}')
                continue
            all_embs.append(embs)
            all_labels.append(p['label'])
            all_pids.append(p['pid'])
            # Save partial checkpoint every 20 samples
            if len(all_pids) % 20 == 0:
                np.savez(partial_cache,
                         embeddings=np.array(all_embs, dtype=object),
                         labels=np.array(all_labels),
                         pids=np.array(all_pids))

        print(f'\n[embed] {len(all_embs)} samples extracted.')
        np.savez(args.embed_cache,
                 embeddings=np.array(all_embs, dtype=object),
                 labels=np.array(all_labels),
                 pids=np.array(all_pids))
        print(f'[embed] Saved to {args.embed_cache}')
        # Remove partial cache after successful completion
        if os.path.exists(partial_cache):
            os.remove(partial_cache)
        del model; torch.cuda.empty_cache()

    labels = np.array(all_labels)

    # ── PCA ──────────────────────────────────────────────────────────────── #
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    lengths = [len(e) for e in all_embs]
    print(f'\n[stats] seq lengths: mean={np.mean(lengths):.1f}  '
          f'min={np.min(lengths)}  max={np.max(lengths)}')

    X_all   = np.vstack(all_embs)
    nan_mask = ~np.isfinite(X_all).all(axis=1)
    if nan_mask.any():
        print(f'[warn] {nan_mask.sum()} frame rows contain NaN/Inf — replacing with 0')
        X_all[nan_mask] = 0.0
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_all)
    # StandardScaler can still produce NaN if a feature has zero variance; clip them
    X_sc    = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)
    pca     = PCA(n_components=args.pca_dim, random_state=42)
    X_pca   = pca.fit_transform(X_sc)
    print(f'[pca]  dim={args.pca_dim}  cum_var={pca.explained_variance_ratio_.cumsum()[-1]:.3f}')

    seqs_pca, idx = [], 0
    for e in all_embs:
        n = len(e)
        seqs_pca.append(X_pca[idx: idx + n])
        idx += n

    # ── Fit ONE unsupervised GaussianHMM (no labels) ──────────────────────── #
    K = args.n_components
    print(f'\n[hmm] Fitting unsupervised GaussianHMM(K={K}) on all {len(seqs_pca)} sequences ...')
    hmm_model = fit_hmm(seqs_pca, K)
    print(f'[hmm] Converged: {hmm_model.monitor_.converged}')

    # ── Interpret states: rank by emission variance ────────────────────────── #
    uncert = state_uncertainty(hmm_model)           # [K] mean diag variance
    state_rank = np.argsort(uncert)[::-1]           # descending: most uncertain first
    print(f'\n[states] Emission uncertainty (mean diag variance) per state:')
    for rank, s in enumerate(state_rank):
        print(f'  S{s}: uncertainty={uncert[s]:.4f}  {"← UNCERTAIN (zoom)" if rank==0 else ""}')

    uncertain_state = int(state_rank[0])            # highest variance = most uncertain

    # ── Transition matrix ─────────────────────────────────────────────────── #
    T = hmm_model.transmat_
    print(f'\n{"="*60}')
    print(f'  UNSUPERVISED HMM TRANSITION MATRIX (K={K})')
    print(f'  States ranked by uncertainty: {[f"S{s}" for s in state_rank]}')
    print(f'{"="*60}')
    print_transition(T, 'T (single HMM, all samples):')
    print(f'\n  Start probs: {hmm_model.startprob_.round(4)}')

    # ── Calibration: proportion of uncertain state → threshold ────────────── #
    print(f'\n[calib] Computing per-sample uncertain-state proportion ...')
    uncertain_fracs = []
    for seq in seqs_pca:
        try:
            viterbi = hmm_model.predict(seq)
            frac    = float((viterbi == uncertain_state).mean())
        except Exception:
            frac = 0.5
        uncertain_fracs.append(frac)
    uncertain_fracs = np.array(uncertain_fracs)

    from scipy.stats import pointbiserialr
    from sklearn.metrics import roc_auc_score
    r, pval = pointbiserialr(labels, uncertain_fracs)
    try:
        auc = roc_auc_score(labels, uncertain_fracs)
    except Exception:
        auc = float('nan')
    print(f'  Correlation (uncertain_frac vs zoom_benefit): r={r:.4f}  p={pval:.4f}')
    print(f'  AUC: {auc:.4f}')

    print(f'\n  Threshold calibration (zoom if uncertain_frac > thr):')
    print(f'  {"skip_rate":>10}  {"thr":>8}  {"skip_zoom_benefit":>20}  {"zoom_zoom_benefit":>20}')
    for sr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        thr = np.quantile(uncertain_fracs, 1.0 - sr)  # top sr% → zoom
        zm  = uncertain_fracs >= thr
        sk  = ~zm
        zb  = labels[zm].mean()  if zm.sum()  > 0 else float('nan')
        sb  = labels[sk].mean()  if sk.sum()  > 0 else float('nan')
        print(f'  {sr:>10.2f}  {thr:>8.3f}  {sb:>20.3f}  {zb:>20.3f}')

    # Default: zoom top 85%
    default_thr = float(np.quantile(uncertain_fracs, 0.15))

    # ── Save ──────────────────────────────────────────────────────────────── #
    hmm_path = args.output_path.replace('.json', '_hmm.pkl')
    pca_path = args.output_path.replace('.json', '_pca.pkl')

    with open(hmm_path, 'wb') as f: pickle.dump(hmm_model, f)
    with open(pca_path, 'wb') as f: pickle.dump({'pca': pca, 'scaler': scaler}, f)

    out = {
        'n_paired':          len(all_embs),
        'n_zoom':            int(labels.sum()),
        'n_nozoom':          int((1 - labels).sum()),
        'n_components':      K,
        'pca_dim':           args.pca_dim,
        'uncertain_state':   uncertain_state,
        'state_uncertainty': uncert.tolist(),
        'state_rank':        state_rank.tolist(),
        'T':                 T.tolist(),
        'startprob':         hmm_model.startprob_.tolist(),
        'auc':               float(auc),
        'correlation':       float(r),
        'pval':              float(pval),
        'default_threshold': default_thr,
        'hmm_path':          hmm_path,
        'pca_path':          pca_path,
    }
    with open(args.output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[save] → {args.output_path}')
    print(f'        → {hmm_path}')
    print(f'        → {pca_path}')


if __name__ == '__main__':
    main()
