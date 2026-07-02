#!/usr/bin/env python3
"""
analyze_r1_think_embed.py

Extract last-layer hidden states from R1 think chain segments,
fit GaussianHMM(K=4), compute T_correct/T_incorrect/delta_T,
and save artifacts for on-the-fly HMM inference.

Pipeline (EIRL-inspired, implicit stage, single layer):
  1. Load temp_vote NOTOOL_SYS think chains + greedy labels
  2. Segment each think chain → [S1, S2, ..., S_T]
  3. For each segment: last-layer last-token hidden state → [H]
  4. StandardScaler + PCA(64) → [T, 64] per sample
  5. GaussianHMM(K=4) on all sequences → state trajectories
  6. Estimate T_correct / T_incorrect from zoom_benefit labels
  7. Logistic regression on Viterbi state proportions → AUC
  8. Save: hmm.pkl, pca.pkl, scaler.pkl, analysis_think_embed.json

On-the-fly use (main_infer_hmm_zoom.py):
  After Round 1 generates <think>...</think><video_zoom/>:
    embed_think_chain(think_text) → [T, 64] → hmm.predict → zoom_score
    zoom_score < threshold → execute zoom
    zoom_score > threshold → skip zoom

Usage:
  python3 analyze_r1_think_embed.py \\
      --model_path zsgvivo/videozoomer \\
      --tv_results infer_results/temp_vote/results_tvote.jsonl \\
      --greedy_results infer_results/greedy_baseline/results_greedy.jsonl \\
      --output_dir think_embed_artifacts/

  # Skip embedding (reload from cache):
  python3 analyze_r1_think_embed.py --no_model --embed_cache think_seg_embeddings.npz
"""

import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ─── Text segmentation (shared with main_infer_hmm_zoom) ─────────────────── #

def _segment(text: str) -> list:
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


# ─── Keyword classifier (for reference labels on HMM states) ─────────────── #

FINAL_PATS   = [r'therefore', r'thus', r'so the answer', r'final answer', r'answer is',
                r'correct answer', r'the answer', r'in conclusion', r'to summarize',
                r'i would (?:choose|select|say)', r'option [A-D] is correct']
VERIFY_PATS  = [r'\bwait\b', r'let me (?:check|verify|reconsider|re-examine|look again)',
                r'double.?check', r'\bconfirm\b', r'\bhmm\b', r'but wait', r'however',
                r'on the other hand', r"i.m not (?:sure|certain)",
                r"it.s (?:unclear|ambiguous|hard to tell)",
                r'need to (?:check|verify|reconsider)', r'let me re']
ANALYSIS_PATS = [r'calculat', r'comput', r'\d+\s*[+\-×÷*/]\s*\d+', r'because',
                 r'since\s+\w', r'given that',
                 r'this (?:means|suggests|indicates|implies|shows)',
                 r'compar', r'contrast', r'analyz', r'reason', r'deduc',
                 r'based on', r'from (?:this|the|these)']

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


# ─── Model loading ────────────────────────────────────────────────────────── #

def load_model(model_path: str, device: str):
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    print(f"[model] Loading tokenizer from {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[model] Loading model (bf16) ...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    H = model.config.hidden_size
    print(f"[model] Loaded. hidden_size={H}", flush=True)
    return tokenizer, model


@torch.no_grad()
def embed_segments_batch(texts: list, tokenizer, model, device: str,
                          max_length: int = 256) -> np.ndarray:
    """
    Encode a batch of text segments.
    Returns float32 numpy array [B, hidden_size].
    Uses last-layer hidden state at the LAST token position (decoder-style summary).
    """
    enc = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    ).to(device)

    out = model.model(                   # language backbone (no vision encoder)
        input_ids      = enc['input_ids'],
        attention_mask = enc['attention_mask'],
        output_hidden_states = True,
        return_dict    = True,
    )
    # Last layer: [B, seq_len, H]
    last_h = out.hidden_states[-1].float()  # [B, L, H]

    # Last NON-PADDING token position for each item
    # attention_mask: 1 = real token, 0 = pad
    lengths = enc['attention_mask'].sum(dim=1) - 1  # 0-indexed position of last real token
    embeddings = last_h[torch.arange(len(texts), device=device), lengths]  # [B, H]
    return embeddings.cpu().numpy()


# ─── Transition matrix estimation ─────────────────────────────────────────── #

def estimate_transition(trajectories: list, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=float)
    for traj in trajectories:
        for a, b in zip(traj[:-1], traj[1:]):
            counts[a, b] += 1
    counts += 1e-6   # Laplace smoothing
    return counts / counts.sum(axis=1, keepdims=True)


def print_matrix(T: np.ndarray, state_names: list, title: str):
    header = '       ' + '   '.join(f'{s:>7}' for s in state_names)
    print(f'\n{title}')
    print(header)
    for i, s in enumerate(state_names):
        row = '   '.join(f'{T[i,j]:+.4f}' for j in range(len(state_names)))
        print(f'  {s:>6}  {row}')


# ─── Argument parsing ─────────────────────────────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model_path',     default='zsgvivo/videozoomer')
    p.add_argument('--tv_results',     default='infer_results/temp_vote/results_tvote.jsonl')
    p.add_argument('--greedy_results', default='infer_results/greedy_baseline/results_greedy.jsonl')
    p.add_argument('--output_dir',     default='think_embed_artifacts')
    p.add_argument('--n_states',       type=int, default=4)
    p.add_argument('--pca_dim',        type=int, default=64,
                   help='PCA dimensionality before HMM fitting.')
    p.add_argument('--batch_size',     type=int, default=64)
    p.add_argument('--max_seg_len',    type=int, default=256,
                   help='Max tokens per segment for embedding.')
    p.add_argument('--device',         default='cuda')
    p.add_argument('--no_model',       action='store_true',
                   help='Skip embedding; load from --embed_cache.')
    p.add_argument('--embed_cache',    default='think_seg_embeddings.npz',
                   help='NPZ file with keys: embeddings [N,H], labels [N], offsets [M,2].')
    p.add_argument('--hmm_seed',       type=int, default=42)
    p.add_argument('--min_seg',        type=int, default=2,
                   help='Skip samples with fewer than this many segments.')
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────── #

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────── #
    print('[data] Loading results ...', flush=True)
    tv_recs   = [json.loads(l) for l in open(args.tv_results)]
    grdy_recs = [json.loads(l) for l in open(args.greedy_results)]

    grdy_acc = {str(r['problem_id']): float(r['acc_final']) for r in grdy_recs}

    # Use temp_vote no-zoom samples (single-pass NOTOOL_SYS → raw_output = full think chain)
    tv_nozoom = [r for r in tv_recs if not r.get('zoom_triggered')]
    print(f'  temp_vote no-zoom: {len(tv_nozoom)}', flush=True)

    paired = []
    for r in tv_nozoom:
        pid      = str(r['problem_id'])
        tv_acc   = float(r['acc_final'])
        ga       = grdy_acc.get(pid)
        if ga is None:
            continue
        think = extract_think_text(r.get('raw_output', ''))
        segs  = _segment(think)
        if len(segs) < args.min_seg:
            continue
        # zoom_benefit=1 if zoom (greedy) was correct and no-zoom (tv) was wrong
        label = int(ga > tv_acc)
        paired.append({
            'pid':   pid,
            'label': label,
            'segs':  segs,
            'think': think,
        })

    n_zoom = sum(p['label'] for p in paired)
    print(f'  paired: {len(paired)}  zoom_benefit=1: {n_zoom}  zoom_benefit=0: {len(paired)-n_zoom}',
          flush=True)

    # Flatten all segments
    all_segs    = []
    seg_offsets = []   # (start_idx, end_idx) per sample
    for p in paired:
        s0 = len(all_segs)
        all_segs.extend(p['segs'])
        seg_offsets.append((s0, len(all_segs)))

    print(f'  total segments: {len(all_segs)}', flush=True)

    # ── Embed all segments ─────────────────────────────────────────────────── #
    cache_path = os.path.join(args.output_dir, args.embed_cache)
    if args.no_model and os.path.exists(cache_path):
        print(f'[embed] Loading from {cache_path}', flush=True)
        cache        = np.load(cache_path)
        all_embeds   = cache['embeddings']
        # Reload offsets/labels from cache if stored; otherwise recompute
        if 'offsets' in cache:
            seg_offsets = cache['offsets'].tolist()
        if 'labels' in cache:
            labels_arr = cache['labels']
            for i, p in enumerate(paired):
                p['label'] = int(labels_arr[i])
    else:
        tokenizer, model = load_model(args.model_path, args.device)
        all_embeds = []
        for i in tqdm(range(0, len(all_segs), args.batch_size),
                      desc='Embedding segments', total=(len(all_segs)+args.batch_size-1)//args.batch_size):
            batch = all_segs[i: i + args.batch_size]
            emb   = embed_segments_batch(batch, tokenizer, model, args.device, args.max_seg_len)
            all_embeds.append(emb)
        all_embeds = np.vstack(all_embeds).astype(np.float32)   # [N_segs, H]
        print(f'[embed] Done. shape={all_embeds.shape}', flush=True)
        np.savez(
            cache_path,
            embeddings = all_embeds,
            offsets    = np.array(seg_offsets, dtype=np.int32),
            labels     = np.array([p['label'] for p in paired], dtype=np.int32),
        )
        print(f'[embed] Saved to {cache_path}', flush=True)
        del model
        torch.cuda.empty_cache()

    # ── StandardScaler + PCA ──────────────────────────────────────────────── #
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Replace any NaN/Inf from earlier embedding issues
    nan_mask = ~np.isfinite(all_embeds)
    if nan_mask.any():
        print(f'[pca] WARNING: {nan_mask.sum()} NaN/Inf values → replaced with 0', flush=True)
        all_embeds[nan_mask] = 0.0

    print(f'[pca] Fitting StandardScaler ...', flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_embeds)   # [N_segs, H]

    pca_dim = min(args.pca_dim, X_scaled.shape[1], X_scaled.shape[0] - 1)
    print(f'[pca] Fitting PCA({pca_dim}) on {X_scaled.shape} ...', flush=True)
    pca = PCA(n_components=pca_dim, random_state=args.hmm_seed, whiten=False)
    X_pca = pca.fit_transform(X_scaled).astype(np.float64)  # [N_segs, pca_dim]
    print(f'[pca] Explained variance ratio (first 10): '
          f'{pca.explained_variance_ratio_[:10].round(3).tolist()}', flush=True)
    print(f'[pca] Cumulative (first {pca_dim}): '
          f'{pca.explained_variance_ratio_.sum():.3f}', flush=True)

    # ── GaussianHMM ───────────────────────────────────────────────────────── #
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print('[hmm] hmmlearn not found. Install: pip install hmmlearn', flush=True)
        sys.exit(1)

    K = args.n_states

    # Build per-sample sequence list for hmmlearn (concatenate + lengths)
    sample_seqs   = []   # list of [T_i, pca_dim]
    sample_labels = []   # list of int
    for p, (s0, s1) in zip(paired, seg_offsets):
        seq = X_pca[s0:s1]                   # [T_i, pca_dim]
        sample_seqs.append(seq)
        sample_labels.append(p['label'])

    sample_labels = np.array(sample_labels)

    X_concat = np.vstack(sample_seqs)              # [N_total_steps, pca_dim]
    lengths   = np.array([len(s) for s in sample_seqs])  # [N_samples]

    print(f'[hmm] Fitting GaussianHMM(n_components={K}) on '
          f'{X_concat.shape[0]} steps from {len(sample_seqs)} samples ...', flush=True)
    print(f'  Trajectory length stats: mean={lengths.mean():.1f}  '
          f'median={np.median(lengths):.0f}  '
          f'single-step={( lengths==1).mean():.1%}', flush=True)

    hmm = GaussianHMM(
        n_components    = K,
        covariance_type = 'diag',
        n_iter          = 200,
        tol             = 1e-4,
        random_state    = args.hmm_seed,
    )
    hmm.fit(X_concat, lengths)
    print(f'[hmm] Converged: {hmm.monitor_.converged}  '
          f'log-likelihood: {hmm.monitor_.history[-1]:.2f}', flush=True)

    # ── Viterbi decoding → per-sample state trajectories ─────────────────── #
    viterbi_trajs = []
    for seq in sample_seqs:
        if len(seq) < 1:
            viterbi_trajs.append([])
            continue
        try:
            _, state_seq = hmm.decode(seq, algorithm='viterbi')
        except Exception as e:
            print(f'  [warn] Viterbi failed for a sample: {e}', flush=True)
            state_seq = np.zeros(len(seq), dtype=int)
        viterbi_trajs.append(state_seq.tolist())

    # ── Transition matrices ───────────────────────────────────────────────── #
    traj_zoom   = [t for t, l in zip(viterbi_trajs, sample_labels) if l == 1]
    traj_nozoom = [t for t, l in zip(viterbi_trajs, sample_labels) if l == 0]

    T_correct   = estimate_transition(traj_zoom,   K)
    T_incorrect = estimate_transition(traj_nozoom, K)
    dT          = T_correct - T_incorrect

    # ── Label HMM states with keyword majority vote ──────────────────────── #
    from collections import Counter
    state_kw_votes = [Counter() for _ in range(K)]
    for p, (s0, s1), vtraj in zip(paired, seg_offsets, viterbi_trajs):
        segs_sample = all_segs[s0:s1]
        for seg, st in zip(segs_sample, vtraj):
            state_kw_votes[st][keyword_classify(seg)] += 1

    state_names = []
    for k in range(K):
        top = state_kw_votes[k].most_common(1)
        name = top[0][0] if top else '?'
        state_names.append(f'H{k}({name})')

    # ── Print results ─────────────────────────────────────────────────────── #
    print('\n' + '='*72, flush=True)
    print('  THINK-CHAIN EMBEDDING HMM  (last-layer, last-token, PCA+GaussHMM)')
    print('='*72, flush=True)
    print(f'  K={K}  PCA_dim={pca_dim}  n_samples={len(paired)}', flush=True)
    print(f'  State names (HMM → keyword majority): {state_names}', flush=True)

    print('\n  State usage:', flush=True)
    for k in range(K):
        n_occ = sum(t.count(k) for t in viterbi_trajs)
        zoom_rate = np.mean([p['label'] for p, t in zip(paired, viterbi_trajs)
                             if k in t]) if any(k in t for t in viterbi_trajs) else float('nan')
        print(f'    {state_names[k]:12s}  occurrences={n_occ:5d}  zoom_rate={zoom_rate:.3f}',
              flush=True)

    print_matrix(T_correct,   state_names, 'T_correct   (zoom_benefit=1):')
    print_matrix(T_incorrect, state_names, 'T_incorrect (zoom_benefit=0):')
    print_matrix(dT,          state_names, 'ΔT = T_correct − T_incorrect:')

    # ── Top ΔT transitions ─────────────────────────────────────────────────── #
    print('\n  Top ΔT transitions:', flush=True)
    flat = [(f'{state_names[i]}→{state_names[j]}', dT[i, j])
            for i in range(K) for j in range(K)]
    for name, val in sorted(flat, key=lambda x: abs(x[1]), reverse=True)[:10]:
        bar = ('+' if val > 0 else '-') * min(int(abs(val) * 30), 30)
        print(f'    {name:30s}: {val:+.4f}  {bar}', flush=True)

    # ── Logistic regression on Viterbi trajectory features → AUC ─────────── #
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    def traj_to_features(traj: list, K: int) -> np.ndarray:
        """Convert Viterbi state sequence to feature vector."""
        n = max(len(traj), 1)
        # State proportions [K]
        cnt = np.zeros(K)
        for s in traj:
            cnt[s] += 1
        props = cnt / n
        # Transition counts (normalized) [K*K]
        trans = np.zeros(K * K)
        for a, b in zip(traj[:-1], traj[1:]):
            trans[a * K + b] += 1
        if len(traj) > 1:
            trans /= (len(traj) - 1)
        # Last state one-hot [K]
        last = np.zeros(K)
        if traj:
            last[traj[-1]] = 1
        return np.concatenate([props, trans, last])

    feat_dim = K + K * K + K
    X_feat = np.vstack([traj_to_features(t, K) for t in viterbi_trajs])   # [N, feat_dim]
    y      = sample_labels

    print(f'\n[lr] LogisticRegression on Viterbi trajectory features '
          f'X={X_feat.shape}  pos_rate={y.mean():.3f}', flush=True)
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced', solver='lbfgs',
                            random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(lr, X_feat, y, cv=cv, scoring='roc_auc')
    print(f'[lr] 5-fold CV AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}', flush=True)
    lr.fit(X_feat, y)

    # ── Keyword-based reference transition matrices ──────────────────────── #
    kw_idx = {s: i for i, s in enumerate(STATE_LABELS)}
    traj_zoom_kw   = [[kw_idx[keyword_classify(s)] for s in all_segs[so:s1]]
                      for p, (so, s1) in zip(paired, seg_offsets) if p['label'] == 1]
    traj_nozoom_kw = [[kw_idx[keyword_classify(s)] for s in all_segs[so:s1]]
                      for p, (so, s1) in zip(paired, seg_offsets) if p['label'] == 0]
    T_correct_kw   = estimate_transition(traj_zoom_kw,   len(STATE_LABELS))
    T_incorrect_kw = estimate_transition(traj_nozoom_kw, len(STATE_LABELS))
    dT_kw          = T_correct_kw - T_incorrect_kw

    print('\n' + '='*72, flush=True)
    print('  KEYWORD-BASED (S/A/V/F) REFERENCE TRANSITION MATRICES')
    print('='*72, flush=True)
    print_matrix(T_correct_kw,   STATE_LABELS, 'T_correct   (zoom_benefit=1):')
    print_matrix(T_incorrect_kw, STATE_LABELS, 'T_incorrect (zoom_benefit=0):')
    print_matrix(dT_kw,          STATE_LABELS, 'ΔT (keyword):')

    # ── Save artifacts ─────────────────────────────────────────────────────── #
    hmm_path    = os.path.join(args.output_dir, 'hmm_think_embed.pkl')
    scaler_path = os.path.join(args.output_dir, 'scaler_think_embed.pkl')
    pca_path    = os.path.join(args.output_dir, 'pca_think_embed.pkl')
    analysis_path = os.path.join(args.output_dir, 'analysis_think_embed.json')

    with open(hmm_path, 'wb')    as f: pickle.dump(hmm, f)
    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
    with open(pca_path, 'wb')    as f: pickle.dump(pca, f)
    print(f'\n[save] HMM    → {hmm_path}', flush=True)
    print(f'[save] Scaler → {scaler_path}', flush=True)
    print(f'[save] PCA    → {pca_path}', flush=True)

    result = {
        'n_states':     K,
        'pca_dim':      pca_dim,
        'hidden_size':  int(all_embeds.shape[1]),
        'n_samples':    len(paired),
        'n_zoom':       int(n_zoom),
        'n_nozoom':     len(paired) - int(n_zoom),
        'hmm_converged': bool(hmm.monitor_.converged),
        'hmm_loglik':   float(hmm.monitor_.history[-1]),
        'lr_auc_mean':  float(auc_scores.mean()),
        'lr_auc_std':   float(auc_scores.std()),
        'state_names':  state_names,
        'embedding': {
            'T_correct':   T_correct.tolist(),
            'T_incorrect': T_incorrect.tolist(),
            'delta_T':     dT.tolist(),
        },
        'keyword': {
            'state_names': STATE_LABELS,
            'T_correct':   T_correct_kw.tolist(),
            'T_incorrect': T_incorrect_kw.tolist(),
            'delta_T':     dT_kw.tolist(),
        },
        'state_stats': [
            {
                'state': k,
                'name': state_names[k],
                'keyword_votes': dict(state_kw_votes[k]),
                'total_occurrences': int(sum(t.count(k) for t in viterbi_trajs)),
            }
            for k in range(K)
        ],
        'artifacts': {
            'hmm':    hmm_path,
            'scaler': scaler_path,
            'pca':    pca_path,
        },
    }

    with open(analysis_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[save] Analysis → {analysis_path}', flush=True)

    # ── On-the-fly inference stub (printed for copy-paste) ─────────────────── #
    print('\n' + '='*72, flush=True)
    print('  ON-THE-FLY INFERENCE (add to main_infer_hmm_zoom.py)')
    print('='*72, flush=True)
    print("""
import pickle, numpy as np
from analyze_r1_think_embed import embed_segments_batch, _segment, extract_think_text

# Load artifacts once at startup:
hmm    = pickle.load(open('think_embed_artifacts/hmm_think_embed.pkl',    'rb'))
scaler = pickle.load(open('think_embed_artifacts/scaler_think_embed.pkl', 'rb'))
pca    = pickle.load(open('think_embed_artifacts/pca_think_embed.pkl',    'rb'))

def hmm_embed_score(think_text, tokenizer, model, device):
    segs = _segment(think_text)
    if not segs:
        return None   # fallback to keyword score
    embs = embed_segments_batch(segs, tokenizer, model, device)   # [T, H]
    embs_pca = pca.transform(scaler.transform(embs)).astype(np.float64)
    try:
        log_prob, states = hmm.decode(embs_pca, algorithm='viterbi')
    except Exception:
        return None
    # Feature: fraction of time in each state
    K = hmm.n_components
    props = np.bincount(states, minlength=K) / max(len(states), 1)
    return log_prob, states.tolist(), props.tolist()
""", flush=True)


if __name__ == '__main__':
    main()
