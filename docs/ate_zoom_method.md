# Answer Token Entropy with Iterative ATE (ATE + iter_ate)

## Overview

A confidence-gated zoom trigger for video QA. The model runs a forced-answer pass after its
first zoom call and reads the probability distribution over answer options from a single set of
logprobs. High entropy → uncertain → execute zoom; low entropy → confident → skip zoom and
use the forced answer directly.

**Key results on LongVideoReason (n=1000):**

| Method | Acc | Zoom% | Passes/sample |
|---|---|---|---|
| Greedy (no zoom) | 77.7% | 0% | 1 |
| Full zoom (always) | 76.9% | 100% | 2–3 |
| **ATE + iter_ate** | **79.2%** | 21.9% | 1.22 avg |
| iter_ae k=5 (oracle-like) | 79.9% | 21.6% | 6.3 avg |

---

## Pipeline

### Pass 1 — R1 (TOOL_SYS greedy)

```
Input : [TOOL_SYS]  +  [video frames + question]
Params: T=0, greedy, stop=["</video_zoom>", "</answer>"]

Output A: <think>…</think><answer>X</answer>   →  done, no zoom needed
Output B: <think>…</think><video_zoom>…</video_zoom>  →  proceed to ATE pass
```

The model runs once with its normal tool-use system prompt. If it answers directly, the
pipeline ends. If it calls zoom, the output (including the full think chain and zoom tag)
is kept as context for the next pass.

---

### Pass 2 — ATE Forced-Answer Pass

```
Input : [R1 output up to </video_zoom>]  +  _FORCE_ANS_TURN
Params: T=0, logprobs=20, stop=["</video_zoom>", "</answer>"]
```

`_FORCE_ANS_TURN` is injected as a user turn:

```
<tool_response>
The zoom tool is unavailable. Based on the video frames already shown,
give your best final answer now.
</tool_response>
Do not call <video_zoom>. Write your final answer inside <answer> and </answer>.
```

The model must answer based on its R1 reasoning chain without executing zoom.

**Why this context matters:** The model has already reasoned through the problem in R1.
The forced-answer pass asks it to commit to an answer from that partial reasoning.
A model that has thought through the problem confidently will produce a sharp distribution;
a genuinely uncertain model will either spread probability mass or ignore the instruction
and call zoom again.

**Entropy computation:**

At the token position of the answer letter inside `<answer>X</answer>`, the top-20 logprobs
give log-probabilities for all vocabulary tokens. We extract log p(A), log p(B), log p(C),
log p(D) (and p(E), p(F) for benchmarks with more options), normalise to a probability
distribution, and compute Shannon entropy:

```
H = -Σ p(c) · log p(c)    for c ∈ {A, B, C, D}

score = -H          (higher = more confident)

H = 0        →  all mass on one option  →  very confident
H = log(4)   →  uniform over A/B/C/D   →  maximally uncertain
```

**Special case — H = ∞:**  If the model ignores `_FORCE_ANS_TURN` and generates
`<video_zoom>` again, the stop string fires before `<answer>` appears. No answer token is
found (`found_answer = False`), so H = ∞, score = −∞. This is treated as the strongest
uncertainty signal and always triggers zoom. It is a feature, not a bug: a model that
refuses a "no zoom" instruction has implicitly signalled it cannot answer without more
visual information.

---

### Decision Gate

```
score > threshold (−0.30)  →  use ATE answer X, skip zoom   [2 passes total]
score ≤ threshold           →  execute zoom → R2+            [3+ passes total]
H = ∞                       →  always execute zoom           [3+ passes total]
```

The threshold −0.30 corresponds to entropy H < 0.30 nats, meaning the model assigns
roughly 74%+ probability to its top choice.

---

### Pass 3+ — iter_ate Loop (max_rounds = 5)

After each zoom execution the pipeline can run another ATE pass to re-evaluate confidence:

```
zoom executed  →  R2: [full context + zoom frames]  →  answer OR another zoom call

if another zoom call:
    run ATE pass on updated context
    score > threshold  OR  score not improving  →  stop, use current ATE answer
    score ≤ threshold                           →  execute another zoom  →  repeat
```

This allows early termination when the model becomes confident mid-sequence and avoids
wasting zoom rounds on questions where zoom is no longer helping.

---

## Implementation

**Script:** `main_infer_hmm_zoom.py`  
**Launch:** `scripts/eval_answer_token_entropy_zoom.sh`

Key `SamplingParams` objects:

```python
# R1: greedy, no logprobs needed
greedy_sp = SamplingParams(n=1, temperature=0.0, max_tokens=4096,
                           stop=["</video_zoom>", "</answer>"],
                           include_stop_str_in_output=True)

# ATE pass: greedy + top-20 logprobs to read answer distribution
ate_sp = SamplingParams(n=1, temperature=0.0, max_tokens=4096,
                        stop=["</video_zoom>", "</answer>"],
                        include_stop_str_in_output=True,
                        logprobs=20)
```

**ATE prompt construction (zoom samples):**

```python
end_pos  = r1_text.find("</video_zoom>") + len("</video_zoom>")
fa_prompt = base_prompt + r1_text[:end_pos] + _FORCE_ANS_TURN
```

**Entropy calculation** (`compute_answer_token_entropy`):

1. Regex-search for `<answer>([A-F])` in ATE output to find the answer token position.
2. Count tokens in the prefix before that position → index into `token_logprobs`.
3. For each option letter, look up its token ID in the top-20 logprob dict.
4. Normalise found log-probabilities to a probability simplex.
5. Compute H and return `score = -H`.

**Benchmark wrappers** (for non-standard option sets):

| Benchmark | Options | Script |
|---|---|---|
| LongVideoReason, VideoMME | A–D | `main_infer_hmm_zoom.py` |
| LongVideoBench | A–E | `main_infer_hmm_zoom_lvb.py` |
| MLVU | A–F | `main_infer_hmm_zoom_mlvu.py` |

The wrappers patch `extract_mc_answer` before invoking the main script.
`compute_answer_token_entropy` uses `[A-F]` regex and iterates over `ABCDEF`
in all cases; options absent from the top-20 logprobs are simply omitted from
the normalisation.

---

## Why Not NOTOOL-first?

An alternative design runs a clean NOTOOL pass before R1 to pre-screen confident samples.
This fails for VideoZoomer: the model is so deeply fine-tuned to call `<video_zoom>` that
it does so even when explicitly instructed not to, regardless of whether the zoom tool is
declared in the system prompt. 100% of samples produce H = ∞ in the pre-screen pass,
making the gate useless.

The ATE pass succeeds because the model already has its own R1 reasoning chain as context.
The forced-answer prompt asks it to commit based on thinking it has already done, rather
than demanding a cold-start answer. This is the critical distinction.

---

## Result Files

| Benchmark | Path |
|---|---|
| LongVideoReason | `infer_results/iter_ae_zoom/results_hmm_zoom_v2.jsonl` |
| VideoMME | `infer_results/videomme_ate_zoom/results_hmm_zoom_v2.jsonl` |
| LongVideoBench | `infer_results/lvb_ate_zoom/results_hmm_zoom_v2.jsonl` |
| MLVU | `infer_results/mlvu_ate_zoom/results_hmm_zoom_v2.jsonl` |

Each record contains `hmm_score`, `hmm_features` (including `H_answer`, `majority_answer`,
`answer_probs`), `zoom_skipped`, `zoom_triggered`, `n_rounds`, and `acc_final`.
