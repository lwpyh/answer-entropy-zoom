#!/usr/bin/env python3
"""
Wrapper: answer-entropy zoom for MLVU (6-option A-F).
Patches extract_mc_answer to handle A-F before running main_infer_hmm_zoom.
"""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Patch A-F answer extraction BEFORE main_infer_hmm_zoom imports it
import main_infer_adaptive_zoom as _az


def _extract_mc_answer_af(text: str):
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    frag = m.group(1) if m else text[-200:]
    hits = re.findall(r"\b([A-F])\b", frag)
    if hits:
        return hits[-1]
    hits = re.findall(r"([A-F])\.", text)
    return hits[-1] if hits else None


_az.extract_mc_answer = _extract_mc_answer_af

# Run main_infer_hmm_zoom in __main__ context (picks up the patched module)
import runpy
runpy.run_path(
    os.path.join(os.path.dirname(__file__), "main_infer_hmm_zoom.py"),
    run_name="__main__",
)
