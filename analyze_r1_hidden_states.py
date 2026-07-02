#!/usr/bin/env python3
"""
analyze_r1_hidden_states.py

Use the main model's last-layer hidden states to:
  1. Embed each segment of the 551 NOTOOL_SYS think chains
  2. Cluster segments into K data-driven states (K-means, K=4)
  3. Estimate T_correct / T_incorrect transition matrices
     (correct = "zoom helps this sample", incorrect = "zoom not needed")
  4. Output final transition matrices and compare with keyword-based approach
  5. Save centroids + matrices for online HMM inference

Usage (offline, needs 1 GPU):
  python3 analyze_r1_hidden_states.py \
      --model_path zsgvivo/videozoomer \
      --tv_results infer_results/temp_vote/results_tvote.jsonl \
      --greedy_results infer_results/greedy_baseline/results_greedy.jsonl \
      --output_path analysis_hmm_hidden_states.json \
      --n_states 4 --batch_size 16
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm

# ─── Text segmentation (same as keyword-based) ───────────────────────────── #

def segment_text(text: str) -> list:
    lines = text.strip().split('\n')
    segs, buf = [], []
    for line in lines:
        line = line.strip()
        if not line:
            if buf:
                segs.append(' '.join(buf)); buf = []
            continue
        if line.startswith(('-', '•', '*', '·')):
            buf.append(line.lstrip('-•*· '))
        else:
            if buf:
                segs.append(' '.join(buf)); buf = []
            for sent in re.split(r'(?<=[.!?])\s+', line):
                if len(sent.strip()) > 5:
                    buf.append(sent.strip())
    if buf:
        segs.append(' '.join(buf))
    return [s for s in segs if len(s.strip()) > 5]


def extract_think_text(raw: str) -> str:
    m = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return re.sub(r'<(?:video_zoom|answer)[^>]*>.*', '', raw, flags=re.DOTALL).strip()


# ─── Keyword-based classifier (reference) ────────────────────────────────── #

FINAL_PATS   = [r'therefore', r'thus', r'so the answer', r'final answer', r'answer is',
                r'correct answer', r'the answer', r'in conclusion', r'to summarize',
                r'</answer>', r'i would (?:choose|select|say)', r'option [A-D] is correct', r'my answer']
VERIFY_PATS  = [r'\bwait\b', r'let me (?:check|verify|reconsider|re-examine|look again)',
                r'double.?check', r'\bconfirm\b', r'\bhmm\b', r'but wait', r'however',
                r'on the other hand', r"i.m not (?:sure|certain)",
                r"it.s (?:unclear|ambiguous|hard to tell)", r'need to (?:check|verify|reconsider)',
                r'let me re', r'but (?:looking|considering|thinking)']
ANALYSIS_PATS = [r'calculat', r'comput', r'\d+\s*[+\-×÷*/]\s*\d+', r'because', r'since\s+\w',
                 r'given that', r'this (?:means|suggests|indicates|implies|shows)',
                 r'compar', r'contrast', r'analyz', r'reason', r'deduc',
                 r'based on', r'from (?:this|the|these)', r'percentage', r'ratio', r'proportion']
SETUP_PATS   = [r'the video (?:shows|depicts|begins|starts|features)', r'i can (?:see|observe|notice|identify)',
                r'in the (?:clip|video|frame|footage)', r'at \d+\.?\d*\s*s', r'the question (?:asks|is about)']

def keyword_classify(text: str) -> str:
    t = text.lower()
    for p in FINAL_PATS:
        if re.search(p, t): return 'F'
    for p in VERIFY_PATS:
        if re.search(p, t): return 'V'
    for p in ANALYSIS_PATS:
        if re.search(p, t): return 'A'
    return 'S'

STATE_LABELS = ['S', 'A', 'V', 'F']


# ─── Embedding extraction ─────────────────────────────────────────────────── #

def load_model(model_path: str, device: str):
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    print(f"[model] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[model] Loading model (bf16) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"[model] Loaded. hidden_size={model.config.hidden_size}")
    return tokenizer, model


@torch.no_grad()
def embed_batch(texts: list, tokenizer, model, device: str, max_length: int = 256) -> np.ndarray:
    """
    Embed a batch of text strings using the model's last transformer layer.
    Returns float32 numpy array of shape [len(texts), hidden_size].
    """
    enc = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    out = model.model(               # language backbone only (no vision encoder)
        input_ids      = enc['input_ids'],
        attention_mask = enc['attention_mask'],
        output_hidden_states = True,
        return_dict    = True,
    )
    # Last layer hidden states: [B, seq_len, H]
    last_h = out.hidden_states[-1].float()
    mask   = enc['attention_mask'].unsqueeze(-1).float()  # [B, seq_len, 1]

    # Mean pool over non-padding tokens
    embeddings = (last_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
    return embeddings.cpu().numpy()


# ─── Transition matrix estimation ─────────────────────────────────────────── #

def estimate_transition(trajectories: list, n_states: int) -> np.ndarray:
    """
    Count transitions in a list of state-label sequences.
    Returns row-normalized transition matrix [n_states, n_states].
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for traj in trajectories:
        for a, b in zip(traj[:-1], traj[1:]):
            counts[a, b] += 1
    # Add tiny smoothing to avoid zero rows
    counts += 1e-6
    return counts / counts.sum(axis=1, keepdims=True)


def print_matrix(T: np.ndarray, state_names: list, title: str):
    n = len(state_names)
    header = '       ' + '   '.join(f'{s:>6}' for s in state_names)
    print(f'\n{title}')
    print(header)
    for i, s in enumerate(state_names):
        row = '   '.join(f'{T[i,j]:+.4f}' for j in range(n))
        print(f'  {s}    {row}')


# ─── Main ─────────────────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',     default='zsgvivo/videozoomer')
    p.add_argument('--tv_results',     default='infer_results/temp_vote/results_tvote.jsonl')
    p.add_argument('--greedy_results', default='infer_results/greedy_baseline/results_greedy.jsonl')
    p.add_argument('--output_path',    default='analysis_hmm_hidden_states.json')
    p.add_argument('--n_states',       type=int, default=4)
    p.add_argument('--batch_size',     type=int, default=32)
    p.add_argument('--max_seg_len',    type=int, default=128,  help='Max tokens per segment')
    p.add_argument('--device',         default='cuda')
    p.add_argument('--kmeans_seed',    type=int, default=42)
    p.add_argument('--no_model',       action='store_true',
                   help='Skip embedding step (reload embeddings from --embed_cache)')
    p.add_argument('--embed_cache',    default='r1_segment_embeddings.npz')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load paired data ──────────────────────────────────────────────────── #
    print('[data] Loading temp_vote no-zoom samples ...')
    tv_recs  = [json.loads(l) for l in open(args.tv_results)]
    grdy_recs = [json.loads(l) for l in open(args.greedy_results)]

    grdy_acc  = {r['problem_id']: float(r['acc_final']) for r in grdy_recs}
    tv_no_zoom = [r for r in tv_recs if not r.get('zoom_triggered')]
    print(f'  temp_vote no-zoom: {len(tv_no_zoom)}')

    # zoom_benefit_label: greedy acc > 0 AND greedy acc > tv acc
    paired = []
    for r in tv_no_zoom:
        pid  = r['problem_id']
        if pid not in grdy_acc:
            continue
        tv_acc   = float(r['acc_final'])
        zoom_acc = grdy_acc[pid]
        think    = extract_think_text(r.get('raw_output', ''))
        segs     = segment_text(think)
        if not segs:
            continue
        # zoom helps if greedy (with zoom) is correct and no-zoom was wrong
        label = int(zoom_acc > tv_acc)
        paired.append({
            'pid':   pid,
            'label': label,        # 1=needs zoom, 0=skip zoom
            'segs':  segs,
            'think': think,
        })

    n_zoom = sum(p['label'] for p in paired)
    print(f'  paired: {len(paired)}  needs_zoom={n_zoom}  skip_zoom={len(paired)-n_zoom}')

    # Flatten all segments
    all_segs   = []
    seg_offsets = []   # (start_idx, end_idx) for each sample
    for p in paired:
        start = len(all_segs)
        all_segs.extend(p['segs'])
        seg_offsets.append((start, len(all_segs)))

    print(f'  total segments: {len(all_segs)}')

    # ── Embed all segments ────────────────────────────────────────────────── #
    if args.no_model and os.path.exists(args.embed_cache):
        print(f'[embed] Loading from cache {args.embed_cache}')
        cache = np.load(args.embed_cache)
        all_embeds = cache['embeddings']
    else:
        tokenizer, model = load_model(args.model_path, args.device)
        all_embeds = []
        for i in tqdm(range(0, len(all_segs), args.batch_size), desc='Embedding segments'):
            batch = all_segs[i: i + args.batch_size]
            emb   = embed_batch(batch, tokenizer, model, args.device, args.max_seg_len)
            all_embeds.append(emb)
        all_embeds = np.vstack(all_embeds)   # [N_segs, H]
        print(f'[embed] Done. shape={all_embeds.shape}')
        np.savez(args.embed_cache, embeddings=all_embeds)
        print(f'[embed] Saved to {args.embed_cache}')
        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # L2-normalise for cosine-based K-means
    all_embeds_norm = normalize(all_embeds, norm='l2')

    # ── K-means clustering ────────────────────────────────────────────────── #
    K = args.n_states
    print(f'[kmeans] Fitting K={K} on {len(all_embeds_norm)} segments ...')
    km = KMeans(n_clusters=K, random_state=args.kmeans_seed, n_init=20, max_iter=500)
    km.fit(all_embeds_norm)
    cluster_labels = km.labels_   # [N_segs]
    print(f'[kmeans] Cluster distribution: {Counter(cluster_labels.tolist())}')

    # ── Label clusters with keyword-based majority vote ────────────────────── #
    cluster_keyword_votes = [Counter() for _ in range(K)]
    for seg, cl in zip(all_segs, cluster_labels):
        cluster_keyword_votes[cl][keyword_classify(seg)] += 1

    cluster_name = []
    for k in range(K):
        top = cluster_keyword_votes[k].most_common(1)[0][0]
        cluster_name.append(top)
    print(f'[kmeans] Cluster→keyword mapping: {dict(enumerate(cluster_name))}')

    # Build per-sample trajectory in cluster-space
    for p, (s0, s1) in zip(paired, seg_offsets):
        p['cluster_traj'] = cluster_labels[s0:s1].tolist()
        p['keyword_traj'] = [keyword_classify(seg) for seg in p['segs']]

    # ── Estimate transition matrices (embedding-based) ─────────────────────── #
    traj_zoom   = [p['cluster_traj'] for p in paired if p['label'] == 1]
    traj_nozoom = [p['cluster_traj'] for p in paired if p['label'] == 0]

    T_correct_emb   = estimate_transition(traj_zoom,   K)
    T_incorrect_emb = estimate_transition(traj_nozoom, K)
    dT_emb          = T_correct_emb - T_incorrect_emb

    # ── Estimate transition matrices (keyword-based, for comparison) ──────── #
    KW = 4
    kw_idx = {s: i for i, s in enumerate(STATE_LABELS)}
    traj_zoom_kw   = [[kw_idx[c] for c in p['keyword_traj']] for p in paired if p['label'] == 1]
    traj_nozoom_kw = [[kw_idx[c] for c in p['keyword_traj']] for p in paired if p['label'] == 0]
    T_correct_kw   = estimate_transition(traj_zoom_kw,   KW)
    T_incorrect_kw = estimate_transition(traj_nozoom_kw, KW)
    dT_kw          = T_correct_kw - T_incorrect_kw

    # ── Print results ──────────────────────────────────────────────────────── #
    state_names_emb = [f'C{k}({cluster_name[k]})' for k in range(K)]

    print('\n' + '='*70)
    print('  EMBEDDING-BASED TRANSITION MATRICES')
    print('='*70)
    print_matrix(T_correct_emb,   state_names_emb, 'T_correct   (zoom helps, zoom_benefit=1):')
    print_matrix(T_incorrect_emb, state_names_emb, 'T_incorrect (skip zoom,  zoom_benefit=0):')
    print_matrix(dT_emb,          state_names_emb, 'ΔT = T_correct − T_incorrect:')

    print('\n' + '='*70)
    print('  KEYWORD-BASED TRANSITION MATRICES (reference)')
    print('='*70)
    print_matrix(T_correct_kw,   STATE_LABELS, 'T_correct   (zoom helps):')
    print_matrix(T_incorrect_kw, STATE_LABELS, 'T_incorrect (skip zoom):')
    print_matrix(dT_kw,          STATE_LABELS, 'ΔT = T_correct − T_incorrect:')

    # ── Trajectory length stats ────────────────────────────────────────────── #
    traj_lens = [len(p['cluster_traj']) for p in paired]
    print(f'\n  Trajectory lengths: mean={np.mean(traj_lens):.1f}  '
          f'median={np.median(traj_lens):.0f}  '
          f'single-step={sum(l==1 for l in traj_lens)/len(traj_lens):.1%}')

    # ── Cluster-level accuracy ─────────────────────────────────────────────── #
    print('\n  Cluster size & zoom-benefit rate:')
    for k in range(K):
        members = [p for p in paired if k in p['cluster_traj']]
        zoom_rate = np.mean([p['label'] for p in members]) if members else float('nan')
        print(f'    C{k}({cluster_name[k]}): n_samples={len(members)}  zoom_benefit_rate={zoom_rate:.3f}')

    # ── Dominant ΔT transitions ────────────────────────────────────────────── #
    print('\n  Top ΔT transitions (embedding-based):')
    flat = [(f'C{i}({cluster_name[i]})→C{j}({cluster_name[j]})', dT_emb[i,j])
            for i in range(K) for j in range(K)]
    for name, val in sorted(flat, key=lambda x: abs(x[1]), reverse=True)[:8]:
        bar = '+' * int(abs(val) * 20) if val > 0 else '-' * int(abs(val) * 20)
        print(f'    {name:30s}: {val:+.4f}  {bar}')

    print('\n  Top ΔT transitions (keyword-based):')
    flat_kw = [(f'{STATE_LABELS[i]}→{STATE_LABELS[j]}', dT_kw[i,j])
               for i in range(KW) for j in range(KW)]
    for name, val in sorted(flat_kw, key=lambda x: abs(x[1]), reverse=True)[:8]:
        bar = '+' * int(abs(val) * 20) if val > 0 else '-' * int(abs(val) * 20)
        print(f'    {name:10s}: {val:+.4f}  {bar}')

    # ── Logistic Regression on per-sample mean-pooled embeddings ─────────── #
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import normalize as sk_normalize

    # Build per-sample embedding: mean over that sample's segments
    X = np.vstack([
        sk_normalize(all_embeds[s0:s1].mean(axis=0, keepdims=True), norm='l2')
        for (s0, s1) in seg_offsets
    ])   # [N, H]
    y = np.array([p['label'] for p in paired])   # [N]  1=zoom helps

    print(f'\n[lr] Training LogisticRegression on X={X.shape}, y={y.shape}  '
          f'pos_rate={y.mean():.3f}')

    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced',
                            solver='lbfgs', random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(lr, X, y, cv=cv, scoring='roc_auc')
    print(f'[lr] 5-fold CV AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}')

    lr.fit(X, y)
    logits = lr.decision_function(X)
    probs  = 1 / (1 + np.exp(-logits))
    train_auc = cross_val_score(lr, X, y, cv=5, scoring='roc_auc').mean()

    # Calibrate threshold: find logit threshold that skips the top P% most confident
    # "skip zoom" = prob < threshold → label=0 predicted
    skip_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    print('\n[lr] Threshold calibration (skip_rate = fraction of samples where zoom skipped):')
    print(f'  {"skip_rate":>10}  {"logit_thr":>10}  {"zoom_benefit_among_skipped":>28}')
    for sr in skip_rates:
        thr = np.quantile(logits, sr)       # skip the bottom sr% (most "no-zoom" confident)
        skip_mask = logits < thr
        if skip_mask.sum() > 0:
            skip_benefit = y[skip_mask].mean()
        else:
            skip_benefit = float('nan')
        print(f'  {sr:>10.2f}  {thr:>+10.3f}  {skip_benefit:>28.3f}')

    # Default: skip bottom 15% (equiv to zoom ~85%)
    default_skip_rate = 0.15
    default_logit_thr = float(np.quantile(logits, default_skip_rate))
    print(f'\n[lr] Default logit threshold: {default_logit_thr:.4f}  '
          f'(skip_rate={default_skip_rate})')

    # Save classifier
    lr_weights = {
        'coef':       lr.coef_[0].tolist(),      # [H]
        'intercept':  float(lr.intercept_[0]),
        'logit_threshold': default_logit_thr,    # score < thr → skip zoom
        'skip_rate':  default_skip_rate,
        'cv_auc':     float(auc_scores.mean()),
        'n_train':    int(len(y)),
        'pos_rate':   float(y.mean()),
        'hidden_size': int(X.shape[1]),
    }
    lr_path = args.output_path.replace('.json', '_lr_weights.json')
    with open(lr_path, 'w') as f:
        json.dump(lr_weights, f)
    print(f'[lr] Saved LR weights to {lr_path}')

    # ── Save ──────────────────────────────────────────────────────────────── #
    out = {
        'n_states': K,
        'n_paired': len(paired),
        'n_zoom': int(n_zoom),
        'n_skip': len(paired) - int(n_zoom),

        # Embedding-based matrices (list-of-lists, row-normalised)
        'embedding': {
            'state_names': state_names_emb,
            'cluster_keyword_map': cluster_name,
            'T_correct':   T_correct_emb.tolist(),
            'T_incorrect': T_incorrect_emb.tolist(),
            'delta_T':     dT_emb.tolist(),
            'centroids':   km.cluster_centers_.tolist(),   # [K, H] L2-normalised
        },

        # Keyword-based matrices (for comparison)
        'keyword': {
            'state_names': STATE_LABELS,
            'T_correct':   T_correct_kw.tolist(),
            'T_incorrect': T_incorrect_kw.tolist(),
            'delta_T':     dT_kw.tolist(),
        },

        # Per-cluster stats
        'cluster_stats': [
            {
                'cluster': k,
                'name': cluster_name[k],
                'keyword_vote': dict(cluster_keyword_votes[k]),
                'size': int((cluster_labels == k).sum()),
            }
            for k in range(K)
        ],
    }

    with open(args.output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[save] Written to {args.output_path}')


if __name__ == '__main__':
    main()
