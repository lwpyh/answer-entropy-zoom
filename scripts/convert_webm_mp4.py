#!/usr/bin/env python3
"""
convert_webm_mp4.py

Converts remaining webm/mkv videos to H264 mp4 using PyAV only
(no ffmpeg CLI required).  All frames are preserved at the original
frame rate — only width is capped at MAX_W pixels.

Safe on nodes where VP9 pix_fmt=None causes decord/filtergraph hangs:
- Sequential demux (no seeking) avoids keyframe-seek overhead
- f.reformat(format='yuv420p') uses libswscale only (no libavfilter)
- Existing 1-fps mp4 files are overwritten with full-fps versions
"""

import av
import json
import multiprocessing as mp
import os
import shutil
import sys

VIDEO_ROOT = "/data/DERI-Gong/jh015/VideoZoomer"
JSONL      = f"{VIDEO_ROOT}/infer_results/adaptive_zoom_lvr/results_adaptive_zoom.jsonl"
JSON_DATA  = f"{VIDEO_ROOT}/longvideo-reason/LongVideoReason_test_fixed.json"
NJOBS      = 8     # parallel workers
MAX_W      = 640   # max output width (resolution cap; inference uses max_pixels=100352)


# ── Per-file conversion ────────────────────────────────────────────────────────

def convert_one(src):
    dst     = os.path.splitext(src)[0] + '.mp4'
    # Write to local /tmp (non-NFS) to avoid mp4 muxer seek-back EINVAL on Lustre/NFS.
    # The mp4 container must seek back to update the mdat size header on close();
    # NFS rejects this seek with EINVAL.  A local temp file sidesteps the issue.
    tmp_dst = f'/tmp/vc_{os.getpid()}_{os.path.basename(dst)}'
    name    = os.path.basename(src)

    # ── Step 1: open source ───────────────────────────────────────────────────
    try:
        in_c = av.open(src)
        in_s = next(s for s in in_c.streams if s.type == 'video')
    except Exception as e:
        return f'[FAIL] {name}: open-source failed: {e}'

    # ── Step 2: read metadata ─────────────────────────────────────────────────
    try:
        fps = in_s.average_rate if in_s.average_rate else 25
        fps_f = float(fps)
        if not (1 <= fps_f <= 240):          # clamp insane values
            fps = 25
        w, h = in_s.width, in_s.height
        if w <= 0 or h <= 0:
            in_c.close()
            return f'[FAIL] {name}: bad dimensions {w}x{h}'
        if w > MAX_W:
            ow = MAX_W
            oh = max(2, int(h * MAX_W / w) & ~1)
        else:
            ow, oh = (w & ~1) or 2, (h & ~1) or 2
    except Exception as e:
        in_c.close()
        return f'[FAIL] {name}: read-metadata failed: {e}'

    print(f'[info] {name}: fps={fps} w={w} h={h} → {ow}x{oh}', flush=True)

    # ── Step 3: open output ───────────────────────────────────────────────────
    try:
        out_c = av.open(tmp_dst, 'w', format='mp4')
    except Exception as e:
        in_c.close()
        return f'[FAIL] {name}: open-output failed (fps={fps} {w}x{h}): {e}'

    # ── Step 4: configure stream ──────────────────────────────────────────────
    try:
        out_s = out_c.add_stream('libx264', rate=fps)
        out_s.width   = ow
        out_s.height  = oh
        out_s.pix_fmt = 'yuv420p'
        out_s.options = {'crf': '23', 'preset': 'ultrafast', 'tune': 'zerolatency'}
    except Exception as e:
        out_c.close()
        in_c.close()
        return f'[FAIL] {name}: add-stream failed (fps={fps} {ow}x{oh}): {e}'

    # ── Step 5: encode ────────────────────────────────────────────────────────
    written = 0
    demux_gen = in_c.demux(in_s)
    while True:
        # Wrap each next() call so a corrupt packet doesn't abort the whole file
        try:
            pkt = next(demux_gen)
        except StopIteration:
            break
        except Exception as e:
            print(f'[warn] {name}: demux error at frame {written}: {e}', flush=True)
            break
        if pkt.size == 0:
            continue
        try:
            for f in pkt.decode():
                # reformat uses libswscale: resize + pix_fmt convert, no filtergraph
                vf = f.reformat(width=ow, height=oh, format='yuv420p')
                vf.pts = written   # sequential frame index
                for p in out_s.encode(vf):
                    out_c.mux(p)
                written += 1
        except Exception:
            pass

    in_c.close()

    # ── Step 5b: flush encoder ────────────────────────────────────────────────
    try:
        for p in out_s.encode():
            out_c.mux(p)
    except Exception as e:
        try:
            out_c.close()
        except Exception:
            pass
        try:
            if os.path.exists(tmp_dst):
                os.remove(tmp_dst)
        except Exception:
            pass
        return f'[FAIL] {name}: flush-encoder failed at frame {written}: {e}'

    # ── Step 5c: close container ──────────────────────────────────────────────
    try:
        out_c.close()
    except Exception as e:
        try:
            if os.path.exists(tmp_dst):
                os.remove(tmp_dst)
        except Exception:
            pass
        return f'[FAIL] {name}: close-container failed at frame {written}: {e}'

    if written == 0:
        try:
            os.remove(tmp_dst)
        except Exception:
            pass
        return f'[FAIL] {name}: 0 frames decoded'

    shutil.move(tmp_dst, dst)  # copy from local /tmp to NFS dst (cross-fs safe)
    return f'[done] {os.path.basename(dst)} ({written} frames, {ow}×{oh}, {float(fps):.3f}fps)'


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    with open(JSON_DATA) as f:
        samples = json.load(f)

    done_pids: set = set()
    if os.path.exists(JSONL):
        with open(JSONL) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get('problem_id') is not None:
                        done_pids.add(rec['problem_id'])

    # Include all webm/mkv for samples not yet inferred; overwrite existing mp4s
    todo = sorted(set(
        os.path.join(VIDEO_ROOT, v)
        for s in samples
        if s.get('problem_id') not in done_pids
        for v in s.get('videos', [])
        if v.lower().endswith(('.webm', '.mkv'))
    ))

    print(f"Files to convert : {len(todo)}")
    print(f"Parallel workers : {NJOBS}")
    sys.stdout.flush()

    if not todo:
        print("Nothing to do.")
        return

    with mp.Pool(NJOBS) as pool:
        for result in pool.imap_unordered(convert_one, todo):
            print(result, flush=True)

    print("=== All done ===")


if __name__ == '__main__':
    main()
