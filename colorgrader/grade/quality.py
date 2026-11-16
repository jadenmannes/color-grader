"""The quality rules. A grade fails when it breaks one of them.

Each check returns {"id", "name", "status": pass | warn | fail, "detail"}.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis.neutral import estimate_cast
from ..color import working as W
from ..color.ycbcr import rgb_to_ycbcr
from . import pipeline
from .skin import offset_from_line, skin_on_display

F32 = np.float32


def _rule(rid: str, name: str, status: str, detail: str) -> dict[str, str]:
    return {"id": rid, "name": name, "status": status, "detail": detail}


def tone_step(idt, shot: dict[str, Any], look: dict[str, Any], odt, cam_sat: float, corrections, bit_depth: int,
              lo_code: float, hi_code: float) -> float:
    """Largest display step (in 8-bit display codes) produced by one code step of the source, on the gray axis."""
    n = 256 if bit_depth <= 8 else 1024
    ramp = np.linspace(0.0, 1.0, n, dtype=F32)
    keep = (ramp >= lo_code - 0.01) & (ramp <= hi_code + 0.01)
    if keep.sum() < 8:
        return 0.0
    sig = np.repeat(ramp[keep][:, None], 3, axis=1)
    if idt.curve.code_values == "absolute":
        from ..color.ycbcr import absolute_to_signal

        sig = absolute_to_signal(sig, idt.full_range, idt.bit_depth)
    lin = pipeline.camera_stage(sig, idt, corrections)
    disp = odt(pipeline.look_stage(pipeline.shot_stage(lin, shot, cam_sat), look))
    y = disp @ np.array([0.2126, 0.7152, 0.0722], dtype=F32)
    return float(np.max(np.abs(np.diff(y))) * 255.0)


def check_shot(sample_lin: np.ndarray, ypos: np.ndarray | None, skin_lin: np.ndarray | None, shot: dict[str, Any],
               look: dict[str, Any], odt, cam_sat: float, src_clip: dict[str, float] | None, ref_flags: dict[str, Any] | None,
               wb_confidence: float, step: float = 0.0, bit_depth: int = 10, grain_on: bool = False) -> list[dict[str, str]]:
    """Run the per-shot rules on a pooled sample."""
    ref_flags = ref_flags or {}
    src_clip = src_clip or {}
    di_b = pipeline.shot_stage(sample_lin, shot, cam_sat)
    disp_b = np.clip(odt(di_b), 0, 1)
    disp_g = np.clip(odt(pipeline.look_stage(di_b, look)), 0, 1)
    rules = []
    # 1 and 2: clipped highlights, crushed shadows
    white_b, white_g = float((disp_b.min(axis=1) >= 0.995).mean()), float((disp_g.min(axis=1) >= 0.995).mean())
    black_b, black_g = float((disp_b.max(axis=1) <= 0.004).mean()), float((disp_g.max(axis=1) <= 0.004).mean())
    allow_hi = max(white_b, src_clip.get("high", 0.0) / 100.0) + 0.004 + (ref_flags.get("clipped", 0.0) if ref_flags.get("clipped", 0.0) >= 0.015 else 0.0)
    allow_lo = max(black_b, src_clip.get("low", 0.0) / 100.0) + 0.004 + (ref_flags.get("crushed", 0.0) if ref_flags.get("crushed", 0.0) >= 0.02 else 0.0)
    rules.append(_rule("highlights", "No new clipped highlights", "fail" if white_g > allow_hi else "pass",
                       f"{white_g * 100:.2f}% of pixels at white after the grade, {max(white_b, src_clip.get('high', 0.0) / 100.0) * 100:.2f}% before."))
    rules.append(_rule("shadows", "No new crushed shadows", "fail" if black_g > allow_lo else "pass",
                       f"{black_g * 100:.2f}% of pixels at black after the grade, {black_b * 100:.2f}% before."))
    # a single channel driven to its limit turns bright colors into flat patches
    top_b, top_g = float((disp_b.max(axis=1) >= 0.995).mean()), float((disp_g.max(axis=1) >= 0.995).mean())
    new_top = top_g - max(top_b, white_g, src_clip.get("high", 0.0) / 100.0)
    rules.append(_rule("color_clip", "No clipped colors", "fail" if new_top > 0.005 else "pass",
                       f"{top_g * 100:.2f}% of pixels hold a channel at its limit after the grade, {top_b * 100:.2f}% before."))
    # 3: skin
    if skin_lin is not None and len(skin_lin) >= 30:
        skin_b = pipeline.shot_stage(skin_lin, shot, cam_sat)
        ang_b, _ = skin_on_display(skin_b, odt)
        ang, sat = skin_on_display(pipeline.look_stage(skin_b, look), odt)
        off, off_b = offset_from_line(ang), offset_from_line(ang_b)
        mono = look.get("saturation", 50.0) < 20.0
        side = "on the line" if abs(off) <= 4 else (f"{abs(off):.0f} degrees toward yellow-green" if off > 0 else f"{abs(off):.0f} degrees toward red-magenta")
        detail = f"Skin sits {side}, saturation {sat:.0f}%."
        pushed = abs(off) - abs(off_b)   # how far the look moved skin away from the line
        if mono or (abs(off) <= 8.0 and 5.0 <= sat <= 40.0):
            status = "pass"
        elif abs(off) > 14.0 or pushed > 5.0 or not (3.0 <= sat <= 46.0):
            status = "fail"
        else:
            status = "warn"
            detail += " The light in this scene tints the skin. A stronger correction would break it into patches."
        rules.append(_rule("skin", "Skin on the skin tone line", status, detail))
    else:
        rules.append(_rule("skin", "Skin on the skin tone line", "pass", "No faces in this shot."))
    # 4: neutrals after balance
    if wb_confidence >= 0.3:
        wb = estimate_cast(W.di_decode(di_b), ypos, odt, None)
        worst = max(abs(wb["temperature"]), abs(wb["tint"]))
        rules.append(_rule("neutral", "Whites and grays neutral after balance", "pass" if worst <= 0.08 else "fail",
                           f"Residual cast after balance: {wb['text']}."))
    else:
        rules.append(_rule("neutral", "Whites and grays neutral after balance", "pass",
                           "No firm neutral anchor in this shot. The camera's white balance stays, with small trims."))
    # 5: saturation control
    y, cb, cr = rgb_to_ycbcr(disp_g)
    sat = np.hypot(cb, cr) * 200.0
    mids = (y > 0.25) & (y < 0.65)
    deep, bright = y < 0.06, y > 0.95
    mid_sat = float(sat[mids].mean()) if mids.sum() > 50 else float(sat.mean())
    deep_sat = float(sat[deep].mean()) if deep.sum() > 50 else 0.0
    bright_sat = float(sat[bright].mean()) if bright.sum() > 50 else 0.0
    peak = float(np.percentile(sat, 99.5))
    bad = []
    if deep_sat > max(9.0, 1.3 * mid_sat):
        bad.append(f"deep shadows carry {deep_sat:.0f}% against {mid_sat:.0f}% in the midtones")
    if bright_sat > max(16.0, 1.5 * mid_sat):
        bad.append(f"bright highlights carry {bright_sat:.0f}% against {mid_sat:.0f}% in the midtones")
    if peak > 85.0:
        bad.append(f"peak saturation reaches {peak:.0f}% of the scope")
    rules.append(_rule("saturation", "Saturation controlled", "fail" if bad else "pass",
                       "; ".join(bad) + "." if bad else f"Midtones {mid_sat:.0f}%, deep shadows {deep_sat:.0f}%, bright highlights {bright_sat:.0f}%, peak {peak:.0f}%."))
    # 6: banding risk
    limit_warn, limit_fail = (2.5, 4.0)
    if step > limit_fail and not grain_on:
        rules.append(_rule("banding", "Smooth gradients", "fail", f"One {bit_depth}-bit source step becomes {step:.1f} display steps. Skies and walls risk banding."))
    elif step > limit_warn:
        rules.append(_rule("banding", "Smooth gradients", "warn" if not grain_on else "pass",
                           f"One {bit_depth}-bit source step becomes {step:.1f} display steps." + (" Fine grain hides the steps." if grain_on else " Fine grain or less contrast lowers the risk.")))
    else:
        rules.append(_rule("banding", "Smooth gradients", "pass", f"One {bit_depth}-bit source step becomes {step:.1f} display steps at most."))
    return rules


def subtlety(look: dict[str, Any]) -> dict[str, str]:
    notes = []
    if look["contrast"] > 1.5:
        notes.append(f"contrast {look['contrast']:.2f}")
    if look["saturation"] > 62 or (20 < look["saturation"] < 36):
        notes.append(f"saturation {look['saturation']:.0f}")
    if max(abs(v) for v in look["lift"]) > 0.06:
        notes.append("strong lift")
    if max(abs(v - 1.0) for v in look["gain"]) > 0.08:
        notes.append("strong highlight color")
    return _rule("subtle", "Grade stays subtle", "warn" if notes else "pass",
                 ("Strong settings: " + ", ".join(notes) + ". Lower the strength slider for a gentler grade.") if notes else "All look settings sit in the gentle range.")


def summarize(rules: list[dict[str, str]]) -> dict[str, int]:
    return {"failed": sum(r["status"] == "fail" for r in rules), "warned": sum(r["status"] == "warn" for r in rules),
            "passed": sum(r["status"] == "pass" for r in rules)}
