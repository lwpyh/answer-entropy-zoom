#!/usr/bin/env python3
"""
main_infer_adaptive_zoom.py  ──  U_beam_ent-Gated Adaptive Tool Inference

Pipeline (≤ max_rounds outer rounds):

  Round 1  [all N samples — no-tool beam]
    beam(n=B, T, notool_prompt, no stop tokens)  →  U_1 = answer entropy of all B paths
      ├── U_1 < θ  →  graduate R1  (answer = beam majority)
      └── U_1 ≥ θ  →  active (proceed to tool rounds)

  Round r = 2 … max_rounds  [active samples only — tool beam, stop at </video_zoom>]
    beam(n=B, T, tool_context, stop=</video_zoom>)
      → zoom-calling paths stop early (no <answer>); answering paths run to completion
      → U_r = answer entropy of answering paths only (zoom paths → None → excluded)
    Gate using U_r:
      ├── U_r < θ  OR  r == max_rounds  →  graduate (majority of answering paths)
      └── U_r ≥ θ  →  still uncertain → continue to r+1:
            if any path proposed zoom → execute clip → commit → r+1 (with new evidence)
            if no zoom proposed      → continue to r+1 (same context, retry)

Key properties:
  - R1 is pure no-tool: clean entropy signal, no stop-token bias
  - R2+: zoom is OPTIONAL EVIDENCE inside the "continue" branch; entropy alone gates stop/go
  - Samples graduate as soon as answering paths agree (U_r < θ), regardless of zoom activity
  - All params aligned to eval_videommlu.sh / eval_adaptive_zoom.sh training setup
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))


# ═══════════════════════════════════════════════════════════════════════════════
# System prompts  (identical to main_infer_tool_uncertainty.py)
# ═══════════════════════════════════════════════════════════════════════════════

NOTOOL_SYS = "You are a helpful assistant."

TOOL_SYS = (
    'You are a helpful assistant. You will receive a low-frame-rate video and '
    'related questions. You can analyze the video content to answer the question '
    'and trigger high-frame-rate inspections when finer temporal resolution is '
    'needed. When you detect ambiguous motion/objects that require closer '
    'inspection, wrap your request in <video_zoom></video_zoom> tags and provide '
    'the exact time segment and target frame rate in JSON format: '
    '<video_zoom> {"segment": [start_sec, end_sec], "fps": n} </video_zoom>, '
    'it will return the video clip at the target fps to help you better answer '
    'the question. Note that the total frames num of the request clip cannot '
    'exceed 16 (e.g., (end_sec - start_sec) * fps \u2264 16) and DO NOT include '
    '<answer> tags in this round. \n'
    ' Example usage: <video_zoom> {"segment": [4.0, 6.0], "fps": 2} </video_zoom>. '
    'If the initial tool response does not provide sufficient information to '
    'answer the question, you may continue to request additional video zoom '
    'inspections as needed, until you either (1) gather enough information to '
    'form a complete answer, or (2) are explicitly instructed to stop using the '
    'tool. Output the thinking process within <think> </think> tags, once you '
    'confirm your final answer, place the final answer in \\boxed{} inside '
    '<answer> and </answer>.'
)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path",   default=None,
                   help="YAML dataset config (required unless --analyze_only)")
    p.add_argument("--video_root",  default="/data/DERI-Gong/jh015/VideoZoomer")
    p.add_argument("--model_path",  default="zsgvivo/videozoomer")
    p.add_argument("--output_dir",  default="./infer_results/adaptive_zoom")
    p.add_argument("--ref_jsonl",   default=None,
                   help="Optional: existing error_detection JSONL for label join")

    # vLLM
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--max_model_len",          type=int,   default=32768)
    p.add_argument("--max_pixels",             type=int,   default=100352)

    # Round-1 no-tool video params  (aligned to eval_videommlu.sh training: same as tool params)
    p.add_argument("--notool_fps",            type=float, default=0.5)
    p.add_argument("--notool_min_pixels",     type=int,   default=25088)
    p.add_argument("--notool_frames_upbound", type=int,   default=64)

    # Round 2+ tool video params  (aligned to eval_videommlu.sh training)
    p.add_argument("--tool_fps",              type=float, default=0.5)
    p.add_argument("--tool_min_pixels",       type=int,   default=25088)
    p.add_argument("--tool_frames_upbound",   type=int,   default=64)
    p.add_argument("--tool_limit_mm",         type=int,   default=128,
                   help="Max images per prompt (covers tool frames + zoom frames)")
    p.add_argument("--tool_max_frames_per_call", type=int, default=16)
    p.add_argument("--tool_workers",          type=int,   default=8)

    # Beam / uncertainty
    p.add_argument("--n_paths",       type=int,   default=5,
                   help="Beam paths B for uncertainty measurement")
    p.add_argument("--sample_temp",   type=float, default=0.4)
    p.add_argument("--ent_threshold", type=float, default=0.5,
                   help="U_beam_ent threshold θ: samples ≥ θ get another tool round")
    p.add_argument("--max_rounds",    type=int,   default=5,
                   help="Max outer rounds including round 1 (no-tool)")
    p.add_argument("--r1_inner_rounds", type=int, default=None,
                   help="Max zoom rounds inside R1 per path (default: same as max_rounds). "
                        "Mirrors eval_videommlu.sh max_generation_round.")

    # Tokens
    p.add_argument("--notool_max_tokens", type=int, default=4096)  # eval_videommlu.sh: max_response_length=4096
    p.add_argument("--tool_max_tokens",   type=int, default=4096)

    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip inference; re-analyse saved JSONL")
    p.add_argument("--r1_tool_sys", action="store_true",
                   help="R1: use TOOL_SYS (training-aligned) instead of NOTOOL_SYS; "
                        "appends 'Do not call <video_zoom>' to prevent zoom in R1")
    p.add_argument("--r2_continue_from_r1", action="store_true",
                   help="R2+: seed tool_base with R1's majority-path output so the model "
                        "continues its own Round-1 reasoning instead of starting fresh. "
                        "Adds a user continuation turn after the R1 assistant output.")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(yaml_path: str):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    samples = []
    for ds in cfg.get("datasets", []):
        with open(ds["json_path"]) as f:
            samples.extend(json.load(f))
    return samples


def resolve_video_path(video_rel: str, video_root: str) -> str:
    p = Path(video_rel)
    if p.is_absolute():
        return str(p)
    parts = p.parts
    rel = Path(*parts[1:]) if parts and parts[0] in (".", "..") else p
    return str(Path(video_root) / rel)


# ═══════════════════════════════════════════════════════════════════════════════
# Video loading
# ═══════════════════════════════════════════════════════════════════════════════

def _resize_frame(img, max_pixels, min_pixels):
    from PIL import Image
    h, w = img.height, img.width
    px   = h * w
    if px > max_pixels:
        s = (max_pixels / px) ** 0.5
        img = img.resize((max(2, int(w*s/2)*2), max(2, int(h*s/2)*2)), Image.LANCZOS)
    elif px < min_pixels:
        s = (min_pixels / px) ** 0.5
        img = img.resize((max(2, int(w*s/2)*2), max(2, int(h*s/2)*2)), Image.LANCZOS)
    return img


def _av_frame_to_pil(f):
    """Convert a PyAV VideoFrame to PIL RGB Image without using filtergraph or swscale.
    Directly accesses raw plane data to avoid pix_fmt=-1 hang on certain VP9/H264 streams."""
    from PIL import Image
    w, h = f.width, f.height
    fmt = f.format.name if f.format else ''

    if fmt in ('yuv420p', 'yuvj420p'):
        y = np.frombuffer(bytes(f.planes[0]), dtype=np.uint8).reshape(h,  f.planes[0].line_size)[:h, :w]
        u = np.frombuffer(bytes(f.planes[1]), dtype=np.uint8).reshape(h//2, f.planes[1].line_size)[:h//2, :w//2]
        v = np.frombuffer(bytes(f.planes[2]), dtype=np.uint8).reshape(h//2, f.planes[2].line_size)[:h//2, :w//2]
        y_img = Image.fromarray(y, 'L')
        u_img = Image.fromarray(u, 'L').resize((w, h), Image.BILINEAR)
        v_img = Image.fromarray(v, 'L').resize((w, h), Image.BILINEAR)
        return Image.merge('YCbCr', [y_img, u_img, v_img]).convert('RGB')

    elif fmt == 'yuv420p10le':
        stride0 = f.planes[0].line_size  # bytes per row
        stride1 = f.planes[1].line_size
        stride2 = f.planes[2].line_size
        y = np.frombuffer(bytes(f.planes[0]), dtype='<u2').reshape(h,   stride0 // 2)[:h,    :w]
        u = np.frombuffer(bytes(f.planes[1]), dtype='<u2').reshape(h//2, stride1 // 2)[:h//2, :w//2]
        v = np.frombuffer(bytes(f.planes[2]), dtype='<u2').reshape(h//2, stride2 // 2)[:h//2, :w//2]
        y_img = Image.fromarray((y >> 2).astype(np.uint8), 'L')
        u_img = Image.fromarray((u >> 2).astype(np.uint8), 'L').resize((w, h), Image.BILINEAR)
        v_img = Image.fromarray((v >> 2).astype(np.uint8), 'L').resize((w, h), Image.BILINEAR)
        return Image.merge('YCbCr', [y_img, u_img, v_img]).convert('RGB')

    else:
        # Other formats: try standard conversion (no known hang issue)
        return Image.fromarray(f.to_ndarray(format='rgb24'), 'RGB')


def _load_via_av(video_path, fps, max_pixels, min_pixels, frames_upbound):
    """Fallback loader using PyAV — handles AV1/webm that decord cannot decode.
    Uses seek-based sampling to avoid decoding all frames for long/high-res videos."""
    import av
    from PIL import Image

    container = av.open(video_path)
    stream    = next(s for s in container.streams if s.type == "video")
    avg_fps   = float(stream.average_rate) if stream.average_rate else (fps if fps > 0 else 24.0)

    # Prefer stream duration (more reliable than container.duration for webm)
    if stream.duration is not None and stream.duration > 0:
        duration = float(stream.duration * stream.time_base)
    else:
        duration = None

    target_fps = fps if fps > 0 else 1.0

    if duration is not None and duration > 0:
        # Seek-based sampling: compute n evenly-spaced timestamps, seek + decode 1 frame each
        n = min(frames_upbound, max(1, int(duration * target_fps)))
        timestamps = np.linspace(0.0, duration * 0.999, n)

        all_frames = []
        for t in timestamps:
            seek_pts = int(t / float(stream.time_base))
            try:
                container.seek(seek_pts, stream=stream)
            except Exception:
                continue
            got = False
            for packet in container.demux(stream):
                if packet.size == 0:
                    break
                try:
                    for f in packet.decode():
                        all_frames.append((t, _av_frame_to_pil(f)))
                        got = True
                        break
                except Exception:
                    pass
                if got:
                    break
    else:
        # No duration info — sequential fallback with early exit
        stride = max(1, int(avg_fps / target_fps))
        all_frames, frame_idx = [], 0
        for packet in container.demux(stream):
            if packet.size == 0:
                break
            try:
                for f in packet.decode():
                    if frame_idx % stride == 0:
                        try:
                            all_frames.append((frame_idx / avg_fps, _av_frame_to_pil(f)))
                        except Exception:
                            pass
                    frame_idx += 1
            except Exception:
                pass
            if len(all_frames) >= frames_upbound:
                break

    container.close()

    if not all_frames:
        raise ValueError(f"No frames decoded from {video_path}")

    frame_times = [t for t, _ in all_frames]
    frames      = [_resize_frame(img, max_pixels, min_pixels) for _, img in all_frames]
    return frame_times, frames


def load_video_frames(video_path, fps, max_pixels, min_pixels, frames_upbound):
    import os
    # Use pre-converted H264 mp4 if available — decord handles H264 without hanging.
    if video_path.lower().endswith(('.webm', '.mkv')):
        mp4_alt = os.path.splitext(video_path)[0] + '.mp4'
        if os.path.exists(mp4_alt):
            print(f"[mp4-alt] {os.path.basename(mp4_alt)}", flush=True)
            video_path = mp4_alt

    from decord import VideoReader, cpu
    import av as _av

    # Pre-probe: detect VP9/AV1 streams where pix_fmt is None before first decode.
    # On some cluster nodes (xlg1), decord HANGS (instead of raising) on these files.
    # Reading stream.codec_context.pix_fmt via av.open() is safe — no decode, no filtergraph.
    try:
        with _av.open(video_path) as _probe:
            _vs = next((s for s in _probe.streams if s.type == 'video'), None)
            _pix_fmt = _vs.codec_context.pix_fmt if _vs else 'known'
    except Exception:
        _pix_fmt = 'known'

    if _pix_fmt is None:
        # VP9/AV1: pix_fmt only known after first decoded frame — decord hangs on these.
        print(f"[av-fallback] {video_path}: pix_fmt=None (VP9/AV1), bypassing decord")
        return _load_via_av(video_path, fps, max_pixels, min_pixels, frames_upbound)

    try:
        vr      = VideoReader(video_path, ctx=cpu(0), num_threads=8)
        total   = len(vr)
        avg_fps = vr.get_avg_fps()
        stride  = max(1, int(avg_fps / fps)) if fps > 0 else 1
        indices = list(range(0, total, stride))
        if len(indices) > frames_upbound:
            sel     = np.linspace(0, len(indices) - 1, frames_upbound, dtype=int)
            indices = [indices[s] for s in sel]

        frame_times = [idx / avg_fps for idx in indices]
        raw         = vr.get_batch(indices).asnumpy()
        from PIL import Image
        frames = [_resize_frame(Image.fromarray(arr.astype("uint8"), "RGB"),
                                max_pixels, min_pixels) for arr in raw]
        return frame_times, frames

    except Exception:
        # decord failed (e.g. AV1/webm) — fall back to PyAV
        return _load_via_av(video_path, fps, max_pixels, min_pixels, frames_upbound)


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt building
# ═══════════════════════════════════════════════════════════════════════════════

def _frame_tokens(frame_times: list) -> str:
    return "".join(
        f"<frame{i}_time{t:.2f}s><|vision_start|><|image_pad|><|vision_end|>"
        for i, t in enumerate(frame_times)
    )


def build_notool_prompt(question: str, frame_times, processor,
                        use_tool_sys: bool = False) -> str:
    vis = _frame_tokens(frame_times)
    user = question.replace("<image>", vis, 1) if "<image>" in question else vis + "\n" + question
    if use_tool_sys:
        # Align R1 system prompt with training distribution (TOOL_SYS was used for all rounds).
        # Explicitly forbid video_zoom so entropy computation stays clean.
        user += ("\nDo not call <video_zoom> in this round, "
                 "give a final answer based on the video frames provided.")
        sys = TOOL_SYS
    else:
        sys = NOTOOL_SYS
    msgs = [{"role": "system", "content": sys},
            {"role": "user",   "content": user}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_tool_initial_prompt(question: str, frame_times, processor) -> str:
    """Returns the raw prompt text (no images dict yet) for tool-use conversation."""
    vis  = _frame_tokens(frame_times)
    user = question.replace("<image>", vis, 1) if "<image>" in question else vis + "\n" + question
    msgs = [{"role": "system", "content": TOOL_SYS},
            {"role": "user",   "content": user}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_tool_response_turn(frame_times: list, is_last: bool) -> str:
    """User turn that delivers zoom results and asks for next action."""
    pad = _frame_tokens(frame_times)
    msg = (
        "<|im_end|>\n<|im_start|>user\n"
        "<tool_response>\nThe frames of the video clip are shown below:\n"
        + pad + "\n</tool_response>\n"
        "continue your reasoning process inside <think> and </think> "
        "and then write your final answer inside <answer> and </answer>"
    )
    if is_last:
        msg += (" Do not call <video_zoom> in this round, "
                "give a final answer based on information above.")
    msg += "<|im_end|>\n<|im_start|>assistant\n"
    return msg


def build_r1_continuation_turn(is_last: bool) -> str:
    """User turn appended after R1 assistant output to open Round 2 tool-enabled generation.
    Closes the R1 assistant turn and starts a new user turn asking the model to continue
    its reasoning with the zoom tool now available."""
    msg = (
        "<|im_end|>\n<|im_start|>user\n"
        "continue your reasoning process inside <think> and </think> "
        "and then write your final answer inside <answer> and </answer>"
    )
    if is_last:
        msg += (" Do not call <video_zoom> in this round, "
                "give a final answer based on information above.")
    msg += "<|im_end|>\n<|im_start|>assistant\n"
    return msg


def build_nozoom_turn() -> str:
    """User turn injected before the last-round beam generation to forbid zoom calls.
    tool_base always ends with an open <|im_start|>assistant\\n; this function closes
    that turn and appends a user instruction that prohibits <video_zoom>."""
    return (
        "<|im_end|>\n<|im_start|>user\n"
        "Do not call <video_zoom> in this round, "
        "give a final answer based on information above."
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def _pick_r1_output(outputs, maj: str) -> str:
    """Return the beam path text that matches the majority answer (fallback: first path).
    Strips trailing <|im_end|> so build_r1_continuation_turn can close the turn cleanly."""
    text = outputs[0].text if outputs else ""
    for out in outputs:
        if extract_mc_answer(out.text) == maj:
            text = out.text
            break
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")]
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Answer extraction / scoring / uncertainty
# ═══════════════════════════════════════════════════════════════════════════════

_ZOOM_RE = re.compile(r"<video_zoom>(.*?)</video_zoom>", re.DOTALL)
_JSON_RE = re.compile(r"\{.*?\}",                       re.DOTALL)


def extract_mc_answer(text: str):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    frag = m.group(1).strip() if m else text
    hits = re.findall(r"\b([A-D])\b", frag)
    if hits:
        return hits[-1]
    if frag and frag[0] in "ABCD":
        return frag[0]
    hits = re.findall(r"([A-D])\.", text)
    return hits[-1] if hits else None


def extract_mc_answer_strict(text: str):
    """Tag-only extraction: only read from <answer>…</answer>; return None if absent.
    Use this in tool rounds where zoom-path text may accidentally contain A/B/C/D letters."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        return None
    frag = m.group(1).strip()
    hits = re.findall(r"\b([A-D])\b", frag)
    if hits:
        return hits[-1]
    if frag and frag[0] in "ABCD":
        return frag[0]
    return None


def parse_zoom_call(text: str):
    m = _ZOOM_RE.search(text)
    if not m:
        return None
    jsons = _JSON_RE.findall(m.group(1))
    if not jsons:
        return None
    try:
        obj = json.loads(jsons[0])
        return float(obj["segment"][0]), float(obj["segment"][1]), float(obj["fps"])
    except Exception:
        return None


def score_answer(pred: str, gt: str) -> float:
    gt_m = re.search(r"<answer>(.*?)</answer>", gt, re.DOTALL)
    gt_c = gt_m.group(1).strip() if gt_m else gt.strip()
    gt_l = extract_mc_answer(gt_c) or gt_c.strip()
    return 1.0 if pred == gt_l else 0.0


def majority_answer(answers: list):
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def _best_answer(ans_history: list):
    """Return majority from the most recent round that has at least one valid answer.
    Needed because in R2+ tool rounds some (or all) paths may have called zoom,
    leaving ans_history[-1] with all-None entries."""
    for round_ans in reversed(ans_history):
        maj = majority_answer(round_ans)
        if maj is not None:
            return maj
    return None


def compute_U_beam_ent(comp_outputs, strict: bool = False) -> tuple:
    """
    Returns (U_beam_ent: float, answers: list[str|None], majority: str|None).
    U_beam_ent = H(empirical answer distribution) across B paths.

    strict=True: use tag-only extraction (tool rounds) so zoom-path text with
    stray A/B/C/D letters doesn't corrupt the answer distribution.
    """
    extractor = extract_mc_answer_strict if strict else extract_mc_answer
    answers = [extractor(o.text) for o in comp_outputs]
    valid   = [a for a in answers if a is not None]
    if not valid:
        return float(np.log(4)), answers, None   # max entropy

    cnt   = Counter(valid)
    total = sum(cnt.values())
    probs = np.array([v / total for v in cnt.values()])
    ent   = float(-np.sum(probs * np.log(probs + 1e-12)))
    maj   = cnt.most_common(1)[0][0]
    return ent, answers, maj


def compute_ent_from_answer_list(answers: list) -> tuple:
    """
    Compute entropy from a plain list of answer strings (str | None).
    Same semantics as compute_U_beam_ent but takes answers directly
    (used for R1 multi-turn where each path has its own context).
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return float(np.log(4)), answers, None
    cnt   = Counter(valid)
    total = sum(cnt.values())
    probs = np.array([v / total for v in cnt.values()])
    ent   = float(-np.sum(probs * np.log(probs + 1e-12)))
    maj   = cnt.most_common(1)[0][0]
    return ent, answers, maj


def extract_best_logprob_answer(comp_outputs):
    """
    Single-pass greedy proxy: from B beam paths, pick the answer of the path
    with the highest mean chosen-token log-probability.  Requires logprobs≥1
    in SamplingParams.  Falls back to majority if logprobs are unavailable.
    """
    best_ans    = None
    best_mean   = -float("inf")
    for out in comp_outputs:
        if out.logprobs:
            lps = [
                lp_dict[tok_id].logprob
                for tok_id, lp_dict in zip(out.token_ids, out.logprobs)
                if tok_id in lp_dict
            ]
            mean_lp = float(np.mean(lps)) if lps else -float("inf")
        else:
            mean_lp = -float("inf")
        if mean_lp > best_mean:
            best_mean = mean_lp
            best_ans  = extract_mc_answer(out.text)
    # fallback to majority if logprobs unavailable
    if best_mean == -float("inf"):
        answers = [extract_mc_answer(o.text) for o in comp_outputs]
        valid   = [a for a in answers if a is not None]
        best_ans = Counter(valid).most_common(1)[0][0] if valid else None
    return best_ans


# ═══════════════════════════════════════════════════════════════════════════════
# Per-sample state
# ═══════════════════════════════════════════════════════════════════════════════

class SampleState:
    """
    Tracks all mutable state for one sample across outer rounds.

    tool_base   : raw prompt text accumulated up to and including the last
                  greedy-generated text (which may end with a zoom call).
    zoom_pending: (zoom_times, zoom_frames) from the most recent executed zoom,
                  or None if no zoom has been executed yet in this outer round.

    After executing a zoom, build the eval / continuation prompts:
      eval_prompt       = tool_base + build_tool_response_turn(times, is_last=True)
      continuation_prompt = tool_base + build_tool_response_turn(times, is_last=False)
    Both end with <|im_start|>assistant\\n (ready for model generation).
    """

    def __init__(self, pid, gt, video_path, question,
                 notool_prompt, notool_images,
                 tool_initial_prompt, tool_initial_images):
        self.pid        = pid
        self.gt         = gt
        self.video_path = video_path
        self.question   = question

        # Round-1 no-tool context (fixed)
        self.notool_prompt = notool_prompt
        self.notool_images = notool_images

        # Tool conversation (built incrementally across outer rounds)
        self.tool_initial_prompt = tool_initial_prompt   # fixed reference for Round 1
        self.tool_initial_images = list(tool_initial_images)
        self.tool_base    = tool_initial_prompt   # updated per round
        self.tool_images  = list(tool_initial_images)
        self.zoom_pending = None                  # (times, frames) | None

        # Outcomes
        self.U_ent_history  = []    # one entry per outer round
        self.ans_history    = []    # one entry (list[str|None]) per outer round
        self.n_tool_calls   = 0
        self.n_rounds       = 0
        self.graduated      = False
        self.grad_round     = None
        self.final_answer   = None
        self.r1_majority_ans = None  # round-1 beam majority answer
        self.acc_r1          = None  # round-1 beam majority accuracy
        self.acc_final      = None  # final round accuracy

    # ── Prompt helpers ────────────────────────────────────────────────── #

    def eval_prompt(self) -> str:
        """Tool context ready for beam evaluation (no-zoom allowed in responses)."""
        if self.zoom_pending is None:
            return self.tool_base
        times, _ = self.zoom_pending
        return self.tool_base + build_tool_response_turn(times, is_last=True)

    def eval_images(self) -> list:
        if self.zoom_pending is None:
            return list(self.tool_images)
        _, frames = self.zoom_pending
        return list(self.tool_images) + list(frames)

    def continuation_prompt(self) -> str:
        """Tool context for next greedy turn (zoom still allowed)."""
        if self.zoom_pending is None:
            return self.tool_base
        times, _ = self.zoom_pending
        return self.tool_base + build_tool_response_turn(times, is_last=False)

    def continuation_images(self) -> list:
        return self.eval_images()

    def commit_continuation(self):
        """
        After beam evaluation decides to continue, commit the pending zoom turn
        into tool_base so the next greedy turn has the full context.
        """
        if self.zoom_pending is None:
            return
        times, frames = self.zoom_pending
        self.tool_base  += build_tool_response_turn(times, is_last=False)
        self.tool_images += list(frames)
        self.zoom_pending = None

    # ── Graduation ────────────────────────────────────────────────────── #

    def graduate(self, answer, acc, round_idx):
        self.graduated    = True
        self.grad_round   = round_idx
        self.final_answer = answer
        self.acc_final    = acc

    def to_record(self) -> dict:
        r = {
            "problem_id":      self.pid,
            "gt":              self.gt,             # ground truth (for post-hoc analysis)
            "acc_notool":      self.acc_r1,         # R1 beam-majority accuracy
            "r1_majority_ans": self.r1_majority_ans,  # R1 beam majority answer
            "acc_final":       self.acc_final,
            "n_rounds":        self.n_rounds,
            "n_tool_calls":    self.n_tool_calls,
            "graduated_round": self.grad_round,
            "final_answer":    self.final_answer,
        }
        for i, (ent, ans) in enumerate(zip(self.U_ent_history, self.ans_history), start=1):
            r[f"U_beam_ent_r{i}"] = ent
            r[f"answers_r{i}"]    = ans
        return r


# ═══════════════════════════════════════════════════════════════════════════════
# Main inference loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(args, samples, processor, llm):
    from vllm import SamplingParams
    from verl.workers.rollout.vllm_rollout.function_tools import extract_video_clip

    # R1: pure no-tool beam — no stop tokens, all B paths generate complete answers
    notool_beam_sp = SamplingParams(
        n=args.n_paths, temperature=args.sample_temp,
        max_tokens=args.notool_max_tokens,
        detokenize=True,
    )
    # R2+: tool beam — stops at </video_zoom> for zoom detection, or at </answer> for answering paths.
    # Both stop strings are included in output (include_stop_str_in_output=True) so the regex
    # <answer>(.*?)</answer> in extract_mc_answer_strict still matches correctly.
    # Stopping answering paths early (at </answer>) prevents super-long sequences that trigger
    # a vLLM V1 engine crash when the last few prompts approach max_tokens.
    tool_beam_sp = SamplingParams(
        n=args.n_paths, temperature=args.sample_temp,
        max_tokens=args.tool_max_tokens,
        stop=["</video_zoom>", "</answer>"], include_stop_str_in_output=True,
        detokenize=True,
    )

    output_jsonl = os.path.join(args.output_dir, "results_adaptive_zoom.jsonl")
    all_records  = []
    done_pids    = set()

    # Resume: load already-completed records so we can skip them
    if os.path.exists(output_jsonl):
        with open(output_jsonl) as _f:
            for _line in _f:
                if _line.strip():
                    _rec = json.loads(_line)
                    all_records.append(_rec)
                    if _rec.get("problem_id") is not None:
                        done_pids.add(_rec["problem_id"])
        print(f"[resume] {len(done_pids)} samples already done, skipping them")

    file_mode = "a" if done_pids else "w"
    with open(output_jsonl, file_mode) as out_f:
        for batch_start in tqdm(range(0, len(samples), args.batch_size), desc="Batches"):
            batch = [s for s in samples[batch_start: batch_start + args.batch_size]
                     if s.get("problem_id") not in done_pids]
            if not batch:
                continue
            states = []

            # ── Preprocess ──────────────────────────────────────────────── #
            for sample in batch:
                try:
                    vp = resolve_video_path(sample["videos"][0], args.video_root)

                    ft1, fr1 = load_video_frames(
                        vp, args.notool_fps, args.max_pixels,
                        args.notool_min_pixels, args.notool_frames_upbound)
                    ft2, fr2 = load_video_frames(
                        vp, args.tool_fps, args.max_pixels,
                        args.tool_min_pixels, args.tool_frames_upbound)

                    notool_inp  = build_notool_prompt(sample["problem"], ft1, processor,
                                                     use_tool_sys=args.r1_tool_sys)
                    tool_init   = build_tool_initial_prompt(sample["problem"], ft2, processor)

                    states.append(SampleState(
                        pid             = sample.get("problem_id"),
                        gt              = sample.get("solution", ""),
                        video_path      = vp,
                        question        = sample.get("problem", ""),
                        notool_prompt   = notool_inp,
                        notool_images   = list(fr1),
                        tool_initial_prompt = tool_init,
                        tool_initial_images = list(fr2),
                    ))
                except Exception as e:
                    print(f"\n[prep] skip {sample.get('problem_id','?')}: {e}")

            if not states:
                continue

            # ── Round 1: no-tool beam (entropy gate) ──────────────────────── #
            # Uses TOOL_SYS (training-aligned) WITHOUT any "Do not call zoom"
            # appended instruction, removing the contradictory-prompt confusion.
            # tool_beam_sp stops zoom-calling paths at </video_zoom> so they
            # contribute answer=None and are cleanly excluded from entropy
            # (strict=True).  No zoom is ever executed: R1 is a pure no-tool pass.
            r1_inputs = [{"prompt": s.tool_initial_prompt,
                          "multi_modal_data": {"image": s.tool_initial_images}}
                         for s in states]
            beam_outs = llm.generate(r1_inputs, tool_beam_sp)

            active = []
            for s, b_out in zip(states, beam_outs):
                ent, ans, maj = compute_U_beam_ent(b_out.outputs, strict=True)
                s.r1_majority_ans = maj
                s.U_ent_history.append(ent)
                s.ans_history.append(ans)
                s.n_rounds = 1
                s.acc_r1   = score_answer(maj, s.gt) if maj else 0.0

                if ent < args.ent_threshold:
                    s.graduate(maj, s.acc_r1, round_idx=1)
                else:
                    active.append(s)

            # ── Rounds 2 … max_rounds ────────────────────────────────────── #
            # Tool beam: stop at </video_zoom>.
            # Zoom-calling paths stop early (no <answer>); answering paths run to completion.
            # U_r = answer entropy of answering paths only (zoom paths contribute None → excluded).
            # Gate: U_r < θ → graduate; U_r ≥ θ → continue to r+1.
            #   Inside the "continue" branch: if any path proposed zoom → execute it (add evidence).
            #   If no zoom proposed → continue anyway (same context, retry).
            # tool_base accumulates context: initial prompt + committed zoom turns.
            for outer_r in range(2, args.max_rounds + 1):
                if not active:
                    break

                is_last_outer = (outer_r == args.max_rounds)

                # On the last round, append a "no zoom" user turn so the model cannot
                # call <video_zoom> even if no tool_response was injected this round.
                # We modify the prompt only (not s.tool_base) since there are no more rounds.
                nozoom_suffix = build_nozoom_turn() if is_last_outer else ""
                beam_inputs = [
                    {"prompt":           s.tool_base + nozoom_suffix,
                     "multi_modal_data": {"image": list(s.tool_images)}}
                    for s in active
                ]
                beam_outs = llm.generate(beam_inputs, tool_beam_sp)

                zoom_queue = {}         # idx → (state, zoom_call): need zoom execution
                to_continue_direct = [] # states that are uncertain but proposed no zoom

                for idx, (s, b_out) in enumerate(zip(active, beam_outs)):
                    # U_r from answering paths only; zoom paths → None → excluded from entropy.
                    # strict=True: only read <answer> tags so stray letters in zoom JSON/text
                    # don't corrupt the answer distribution.
                    ent, ans, maj = compute_U_beam_ent(b_out.outputs, strict=True)
                    s.U_ent_history.append(ent)
                    s.ans_history.append(ans)
                    s.n_rounds = outer_r
                    acc = score_answer(maj, s.gt) if maj else 0.0

                    if ent < args.ent_threshold or is_last_outer:
                        # Answering paths agree (low entropy) OR last round → graduate.
                        # If current round yields no valid answer (model disobeyed nozoom or
                        # produced malformed output), fall back to the best answer from any
                        # previous round rather than graduating with None.
                        if maj is None:
                            maj = _best_answer(s.ans_history)
                            acc = score_answer(maj, s.gt) if maj else 0.0
                        s.graduate(maj, acc, outer_r)
                    else:
                        # Still uncertain → continue to r+1
                        # Pick zoom from first path that proposed one
                        zoom_call, zoom_text = None, None
                        for output in b_out.outputs:
                            z = parse_zoom_call(output.text)
                            if z is not None:
                                end_pos = output.text.find("</video_zoom>") + len("</video_zoom>")
                                zoom_call, zoom_text = z, output.text[:end_pos]
                                break

                        if zoom_call is not None:
                            # Zoom proposed → queue for execution (adds evidence before r+1)
                            s.tool_base += zoom_text
                            zoom_queue[idx] = (s, zoom_call)
                        else:
                            # No zoom proposed → continue without new evidence
                            to_continue_direct.append(s)

                if not zoom_queue:
                    # No zoom to execute; carry forward uncertain-no-zoom samples
                    active = list(to_continue_direct)
                    continue

                # Execute zoom clips (parallel) ─────────────────────────────── #
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
                            min_pixels     = args.tool_min_pixels,
                            max_frames     = args.tool_max_frames_per_call,
                            storage_system = "local",
                        )
                        futures[f] = idx
                    zoom_results = {}
                    for f in as_completed(futures):
                        zoom_results[futures[f]] = f.result()

                # Rebuild active: uncertain-no-zoom samples + zoom-completed samples
                active = list(to_continue_direct)
                for idx, (s, _) in zoom_queue.items():
                    result = zoom_results.get(idx)
                    if isinstance(result, dict):
                        times, frames = result["frame_time"], result["frames"]
                        is_last_delivery = (outer_r + 1 == args.max_rounds)
                        s.tool_base  += build_tool_response_turn(times, is_last=is_last_delivery)
                        s.tool_images += list(frames)
                        s.n_tool_calls += 1
                        active.append(s)
                    else:
                        # Zoom execution failed → graduate with best available answer
                        ans = _best_answer(s.ans_history)
                        acc = score_answer(ans, s.gt) if ans else 0.0
                        s.graduate(ans, acc, outer_r)

            # Force-graduate any remaining active samples (rare: only R4/R5 overflow)
            for s in active:
                best_ans = _best_answer(s.ans_history)
                acc      = score_answer(best_ans, s.gt) if best_ans else 0.0
                s.graduate(best_ans, acc, round_idx=s.n_rounds)

            # ── Save records ─────────────────────────────────────────────── #
            for s in states:
                rec = s.to_record()
                all_records.append(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()

    return all_records, output_jsonl


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(results: list, output_dir: str, ent_threshold: float,
            max_rounds: int, ref_jsonl: str = None):

    # ── Optional: join ref JSONL for delta_s ─────────────────────────── #
    ref_lookup = {}
    if ref_jsonl and os.path.exists(ref_jsonl):
        with open(ref_jsonl) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("problem_id") is not None:
                        ref_lookup[r["problem_id"]] = r
        for rec in results:
            pid = rec.get("problem_id")
            if pid in ref_lookup:
                rec.setdefault("delta_s", ref_lookup[pid].get("delta_s"))

    valid    = [r for r in results if r.get("acc_r1") is not None or r.get("acc_notool") is not None]
    if not valid:
        print("[analysis] No valid records.")
        return

    # Normalise field name: acc_r1 → acc_notool
    for r in valid:
        if "acc_notool" not in r and "acc_r1" in r:
            r["acc_notool"] = r["acc_r1"]

    acc_nt   = np.array([float(r["acc_notool"]) for r in valid])
    acc_fin  = np.array([float(r.get("acc_final", r["acc_notool"])) for r in valid])
    n_rounds = np.array([int(r.get("n_rounds", 1))  for r in valid])
    n_tools  = np.array([int(r.get("n_tool_calls", 0)) for r in valid])
    n        = len(valid)

    print(f"\n{'='*82}")
    print(f"ADAPTIVE ZOOM RESULTS  (n={n},  θ={ent_threshold:.3f},  max_rounds={max_rounds})")
    print(f"{'='*82}")
    print(f"  acc_notool  (R1 beam, no tool)  : {acc_nt.mean():.4f}")
    print(f"  acc_final   (adaptive pipeline) : {acc_fin.mean():.4f}")
    print(f"  Δacc                            : {(acc_fin - acc_nt).mean():+.4f}")
    print(f"  Mean rounds used                : {n_rounds.mean():.2f}")
    print(f"  Mean tool calls per sample      : {n_tools.mean():.2f}")
    print(f"  Samples needing >1 round        : {(n_rounds > 1).sum()} / {n}  "
          f"({100*(n_rounds > 1).mean():.1f}%)")
    print(f"{'='*82}")

    # ── Per-round graduation breakdown ───────────────────────────────── #
    print(f"\n{'─'*82}")
    print(f"PER-ROUND GRADUATION BREAKDOWN")
    print(f"{'─'*82}")
    print(f"  {'Round':>6}  {'n_grad':>7}  {'%total':>7}  "
          f"{'acc_notool':>10}  {'acc_final':>9}  {'Δacc':>7}  "
          f"{'U_ent_mean':>10}  {'n_tools':>7}")
    print(f"  {'-'*80}")
    for r in range(1, max_rounds + 1):
        grp = [rec for rec in valid if rec.get("graduated_round") == r]
        if not grp:
            continue
        an   = np.mean([rec["acc_notool"] for rec in grp])
        af   = np.mean([rec.get("acc_final", rec["acc_notool"]) for rec in grp])
        pct  = 100 * len(grp) / n
        # U_beam_ent at the round they graduated
        ents = [rec.get(f"U_beam_ent_r{r}") for rec in grp
                if rec.get(f"U_beam_ent_r{r}") is not None]
        ent_mean = np.mean(ents) if ents else float("nan")
        tc   = np.mean([rec.get("n_tool_calls", 0) for rec in grp])
        print(f"  {r:>6}  {len(grp):>7}  {pct:>6.1f}%  "
              f"{an:>10.4f}  {af:>9.4f}  {af-an:>+7.4f}  "
              f"{ent_mean:>10.4f}  {tc:>7.2f}")

    # ── Compute efficiency ───────────────────────────────────────────── #
    print(f"\n{'─'*82}")
    print("COMPUTE EFFICIENCY  (accuracy vs. avg beam paths per sample)")
    print(f"{'─'*82}")
    print(f"  {'Scenario':35}  {'acc':>7}  {'avg_paths':>9}  {'paths_saved%':>12}")
    B        = len(valid[0].get("answers_r1", [1] * 5))   # n_paths
    max_paths = max_rounds * B   # upper bound (all rounds for all samples)
    # estimate actual paths used: round 1 all samples (B), later rounds only active
    for scenario_label, scenario_rounds in [
            ("R1 only (θ=∞)",       1),
            ("R1+R2 if uncertain",   2),
            ("Actual (θ={:.2f})".format(ent_threshold), None),
            (f"All {max_rounds} rounds (θ=0)", max_rounds),
    ]:
        if scenario_rounds is not None and scenario_rounds != max_rounds:
            # Simulate stopping after scenario_rounds outer rounds
            acc_s = []
            paths_s = []
            for rec in valid:
                nr = int(rec.get("n_rounds", 1))
                # Accuracy: if n_rounds ≤ scenario_rounds use acc_final else acc_notool
                if nr <= scenario_rounds:
                    acc_s.append(float(rec.get("acc_final", rec["acc_notool"])))
                    paths_s.append(nr * B)
                else:
                    # Would have stopped at scenario_rounds with that round's beam answer
                    r_ans_key = f"answers_r{min(scenario_rounds, nr)}"
                    r_ans = rec.get(r_ans_key)
                    maj   = majority_answer(r_ans) if r_ans else None
                    acc   = score_answer(maj, rec.get("gt", "")) if maj else 0.0
                    acc_s.append(acc)
                    paths_s.append(scenario_rounds * B)
            acc_v = np.mean(acc_s)
            avg_p = np.mean(paths_s)
        elif scenario_rounds is None:
            # Actual run
            acc_v = float(acc_fin.mean())
            avg_p = float(n_rounds.mean()) * B
        else:
            # All max_rounds rounds
            acc_v = float(acc_fin.mean())  # same accuracy
            avg_p = max_rounds * B
        saved = 100 * (1 - avg_p / (max_rounds * B))
        print(f"  {scenario_label:35}  {acc_v:.4f}  {avg_p:>9.1f}  {saved:>11.1f}%")

    # ── State flow: notool → final ───────────────────────────────────── #
    print(f"\n{'─'*82}")
    print("STATE FLOW  notool → final  (correct / wrong)")
    print(f"{'─'*82}")
    print(f"  {'group':30}  {'0→0':>6}  {'0→1 ✓fix':>10}  {'1→0 ✗brk':>10}  {'1→1':>6}  {'n':>5}")
    print(f"  {'-'*70}")

    def flow_row(grp_label, grp):
        c = Counter((int(rec["acc_notool"]), int(rec.get("acc_final", rec["acc_notool"])))
                    for rec in grp)
        ng = len(grp)
        print(f"  {grp_label:30}  "
              f"{c[(0,0)]:>4} ({100*c[(0,0)]/ng:4.1f}%)  "
              f"{c[(0,1)]:>4} ({100*c[(0,1)]/ng:4.1f}%)  "
              f"{c[(1,0)]:>4} ({100*c[(1,0)]/ng:4.1f}%)  "
              f"{c[(1,1)]:>4} ({100*c[(1,1)]/ng:4.1f}%)  "
              f"{ng:>5}")

    flow_row("All samples", valid)
    r1_grp  = [r for r in valid if r.get("graduated_round") == 1]
    r2p_grp = [r for r in valid if r.get("graduated_round", 1) > 1]
    if r1_grp:
        flow_row("Graduated R1 (no tool)", r1_grp)
    if r2p_grp:
        flow_row("Used tool (R2+)", r2p_grp)

    # ── Post-hoc threshold sweep on R1 U_beam_ent ────────────────────── #
    ents_r1 = np.array([float(r.get("U_beam_ent_r1", np.nan)) for r in valid])
    if np.isfinite(ents_r1).sum() >= 10:
        print(f"\n{'─'*82}")
        print("POST-HOC R1 THRESHOLD SWEEP  (if we applied different θ to R1 gate)")
        print("  (acc_final only available for samples that actually used tools; "
              "others assumed same as R1)")
        print(f"{'─'*82}")
        print(f"  {'θ':>6}  {'active%':>8}  {'acc_R1_only':>12}  {'acc_active_grp':>14}  {'estimated_full':>14}")
        print(f"  {'-'*60}")

        # For samples that actually used tools → we know acc_final
        # For samples not activated → acc would be acc_notool
        for theta in [0.1, 0.2, 0.3, 0.5, 0.693, 0.9, 1.2, 999.0]:
            would_activate = ents_r1 >= theta
            pct_active = 100 * would_activate.mean()
            acc_r1_only = acc_nt.mean()   # if θ=∞: keep all R1 answers

            # Estimate: activated → use acc_final (if they were actually activated)
            #           not activated → use acc_notool
            est_acc = []
            for i, rec in enumerate(valid):
                if would_activate[i]:
                    # Sample would be activated: use its actual acc_final if available
                    est_acc.append(float(rec.get("acc_final", rec["acc_notool"])))
                else:
                    est_acc.append(float(rec["acc_notool"]))
            est_full = np.mean(est_acc)

            # acc_active_grp: mean acc_final for samples that ARE activated at this θ
            active_accs = [float(valid[i].get("acc_final", valid[i]["acc_notool"]))
                           for i in range(n) if would_activate[i]]
            acc_act = np.mean(active_accs) if active_accs else float("nan")

            print(f"  {theta:>6.3f}  {pct_active:>7.1f}%  {acc_r1_only:>12.4f}  "
                  f"{acc_act:>14.4f}  {est_full:>14.4f}")

    # ── U_beam_ent trajectory for tool-use samples ────────────────────── #
    tool_samples = [r for r in valid if int(r.get("n_rounds", 1)) > 1]
    if tool_samples:
        max_r_found = max(r.get("n_rounds", 1) for r in tool_samples)
        print(f"\n{'─'*82}")
        print("U_beam_ent TRAJECTORY  (mean over samples that used tools)")
        print(f"{'─'*82}")
        header = f"  {'Round':>6}"
        for r in range(1, max_r_found + 1):
            header += f"  {'R'+str(r):>8}"
        print(header)
        print(f"  {'-'*60}")

        for label, grp in [("mean U_ent",     tool_samples),
                           ("std  U_ent",     tool_samples)]:
            row = f"  {label:>6}"
            for r in range(1, max_r_found + 1):
                vals = [float(rec[f"U_beam_ent_r{r}"]) for rec in grp
                        if rec.get(f"U_beam_ent_r{r}") is not None]
                if vals:
                    v = np.mean(vals) if "mean" in label else np.std(vals)
                    row += f"  {v:>8.4f}"
                else:
                    row += f"  {'—':>8}"
            print(row)

    # ── Save ─────────────────────────────────────────────────────────── #
    summary = {
        "n":             n,
        "ent_threshold": ent_threshold,
        "max_rounds":    max_rounds,
        "acc_notool":    float(acc_nt.mean()),
        "acc_final":     float(acc_fin.mean()),
        "delta_acc":     float((acc_fin - acc_nt).mean()),
        "mean_rounds":   float(n_rounds.mean()),
        "mean_tool_calls": float(n_tools.mean()),
        "pct_used_tools": float(100 * (n_rounds > 1).mean()),
        "per_round": {
            str(r): {
                "n":         len([rec for rec in valid if rec.get("graduated_round") == r]),
                "acc_notool": float(np.mean([rec["acc_notool"] for rec in valid
                                             if rec.get("graduated_round") == r]) or 0),
                "acc_final":  float(np.mean([rec.get("acc_final", rec["acc_notool"])
                                             for rec in valid
                                             if rec.get("graduated_round") == r]) or 0),
            }
            for r in range(1, max_rounds + 1)
            if any(rec.get("graduated_round") == r for rec in valid)
        },
    }
    path = os.path.join(output_dir, "adaptive_zoom_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results_adaptive_zoom.jsonl")

    # ── Analysis-only ────────────────────────────────────────────────── #
    if args.analyze_only:
        print(f"[analysis] Loading {output_jsonl}")
        with open(output_jsonl) as f:
            results = [json.loads(l) for l in f if l.strip()]
        print(f"[analysis] {len(results)} records")
        analyze(results, args.output_dir,
                ent_threshold=args.ent_threshold,
                max_rounds=args.max_rounds,
                ref_jsonl=args.ref_jsonl)
        return

    if not args.data_path:
        raise ValueError("--data_path required unless --analyze_only is set")

    print(f"[data]  Loading {args.data_path}")
    samples = load_dataset(args.data_path)
    print(f"[data]  {len(samples)} samples")

    from transformers import AutoProcessor
    print(f"[model] Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    from vllm import LLM
    max_mm = max(args.notool_frames_upbound, args.tool_limit_mm)
    print(f"[vLLM]  Initialising (tp={args.tensor_parallel_size}, "
          f"util={args.gpu_memory_utilization}, limit_mm={max_mm})")
    llm = LLM(
        model                 = args.model_path,
        tensor_parallel_size  = args.tensor_parallel_size,
        gpu_memory_utilization= args.gpu_memory_utilization,
        max_model_len         = args.max_model_len,
        dtype                 = "bfloat16",
        trust_remote_code     = True,
        mm_processor_kwargs   = {
            "max_pixels": args.max_pixels,
            "min_pixels": args.tool_min_pixels,
        },
        limit_mm_per_prompt   = {"image": max_mm},
        enforce_eager         = False,
        enable_prefix_caching = False,
    )

    print(f"[run]   θ={args.ent_threshold:.3f},  B={args.n_paths},  "
          f"T={args.sample_temp},  max_rounds={args.max_rounds},  "
          f"R1=no-tool/TOOL_SYS/strict-ent")

    results, jsonl_path = run_inference(args, samples, processor, llm)
    print(f"\nResults JSONL: {jsonl_path}  ({len(results)} samples)")

    analyze(results, args.output_dir,
            ent_threshold=args.ent_threshold,
            max_rounds=args.max_rounds,
            ref_jsonl=args.ref_jsonl)


if __name__ == "__main__":
    main()
