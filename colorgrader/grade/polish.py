"""Finishing pass on the tone of a look: the difference between a corrected picture and a finished one.

A picture straight out of the input conversion fills only part of the display range. Blacks sit
in dark gray, the brightest tones stop short of white, and the midtones lie close together. The
pass sets the black point, opens the top and adds midtone contrast with the four master controls
of the look: contrast, lift, gamma and gain. The midtone level stays where the look put it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from . import pipeline

F32 = np.float32
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=F32)
BLACK, WHITE = 0.022, 0.94        # display levels of a finished picture: first and 99th percentile
BLACK_MIN = 0.010                 # a first percentile under this level reads as crushed
SKIN_TOP = 0.76                   # bright skin above this display level loses its shape and color
FLAT_IQR, FULL_IQR = 0.22, 0.34   # midtone spread of a flat picture and of a picture with full contrast
MID_GAIN = 0.22                   # extra distance of the quartiles from the median at full amount
LIMITS = {"contrast": (0.85, 1.55), "lift": (-0.05, 0.08), "gamma": (-0.12, 0.12), "gain": (0.85, 1.30)}


def _apply(look: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    k = dict(look)
    k["contrast"] = float(np.clip(look["contrast"] * (1.0 + 0.3 * x[0]), *LIMITS["contrast"]))
    k["lift"] = [float(v + 0.03 * x[1]) for v in look["lift"]]
    k["gamma"] = [float(v + 0.06 * x[2]) for v in look["gamma"]]
    k["gain"] = [float(v * (1.0 + 0.1 * x[3])) for v in look["gain"]]
    return k


def _stats(parts: list[np.ndarray], look: dict[str, Any], odt) -> dict[str, float]:
    lows, highs, all_y = [], [], []
    for part in parts:
        y = np.clip(odt(pipeline.look_stage(part, look)), 0.0, 1.0) @ LUMA
        lows.append(float(np.percentile(y, 1.0)))
        highs.append(float(np.percentile(y, 99.0)))
        all_y.append(y)
    q = np.percentile(np.concatenate(all_y), [25, 50, 75])
    # the darkest shot sets the black point and the brightest sets the top: no shot crushes or clips
    return {"black": min(lows), "white": max(highs), "q25": float(q[0]), "median": float(q[1]), "q75": float(q[2])}


def _skin_top(skin_parts: list[np.ndarray], look: dict[str, Any], odt) -> float:
    """Level of the brightest skin over all shots: 90th percentile of display luma."""
    tops = [float(np.percentile(np.clip(odt(pipeline.look_stage(sk, look)), 0.0, 1.0) @ LUMA, 90.0)) for sk in skin_parts]
    return max(tops) if tops else 0.0


def polish_tone(look: dict[str, Any], parts: list[np.ndarray], odt, amount: float,
                skin_parts: list[np.ndarray] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the look with finished tone, plus what changed.

    parts: balanced samples per shot (DaVinci Intermediate). skin_parts: skin samples per shot with faces.
    """
    a = float(np.clip(amount, 0.0, 1.0))
    # blank frames (black slates, fades) say nothing about the tonal range
    live = [p for p in parts if float(np.median(np.clip(odt(p), 0, 1) @ LUMA)) > 0.05]
    if a <= 0.0 or not live:
        return look, {"changed": False}
    live = [p[:: max(1, len(p) // 3000)] for p in live]
    skin_parts = [sk[:: max(1, len(sk) // 1500)] for sk in (skin_parts or []) if sk is not None and len(sk) >= 30]
    s0 = _stats(live, look, odt)
    skin0 = _skin_top(skin_parts, look, odt)
    # a flat picture takes the full midtone contrast, a picture with full contrast takes none
    mid = MID_GAIN * a * float(np.clip((FULL_IQR - (s0["q75"] - s0["q25"])) / (FULL_IQR - FLAT_IQR), 0.0, 1.0))
    black_goal = float(np.clip(s0["black"], BLACK_MIN, BLACK))        # deepen gray blacks, open crushed ones
    white_goal = min(max(s0["white"], WHITE), s0["white"] + 0.25)
    # skin that glows sets the midtone level: the face matters more than the average of the frame
    over = max(0.0, skin0 - SKIN_TOP)
    median_goal = s0["median"] - 0.9 * over
    target = {
        "black": s0["black"] + a * (black_goal - s0["black"]),
        "white": s0["white"] + a * (white_goal - s0["white"]),
        "median": median_goal,
        "q25": median_goal - (s0["median"] - s0["q25"]) * (1.0 + mid),
        "q75": median_goal + (s0["q75"] - s0["median"]) * (1.0 + mid),
    }
    weights = {"black": 3.0, "white": 2.0, "median": 3.0, "q25": 1.0, "q75": 1.0}

    def resid(x: np.ndarray) -> np.ndarray:
        k = _apply(look, x)
        s = _stats(live, k, odt)
        r = [weights[key] * (s[key] - target[key]) for key in weights]
        if skin_parts:
            r.append(6.0 * max(0.0, _skin_top(skin_parts, k, odt) - SKIN_TOP))
        return np.array(r + list(0.004 * x))

    lo = np.array([(LIMITS["contrast"][0] / look["contrast"] - 1.0) / 0.3, (LIMITS["lift"][0] - min(look["lift"])) / 0.03,
                   (LIMITS["gamma"][0] - min(look["gamma"])) / 0.06, (LIMITS["gain"][0] / max(look["gain"]) - 1.0) / 0.1])
    hi = np.array([(LIMITS["contrast"][1] / look["contrast"] - 1.0) / 0.3, (LIMITS["lift"][1] - max(look["lift"])) / 0.03,
                   (LIMITS["gamma"][1] - max(look["gamma"])) / 0.06, (LIMITS["gain"][1] / min(look["gain"]) - 1.0) / 0.1])
    lo, hi = np.minimum(lo, -1e-3), np.maximum(hi, 1e-3)
    # the solver takes its difference steps relative to x: it runs on x + 1, so a start at zero still moves
    sol = least_squares(lambda u: resid(u - 1.0), np.ones(4), bounds=(lo + 1.0, hi + 1.0), diff_step=0.03, max_nfev=60)
    x = sol.x - 1.0
    out = _apply(look, x)
    s1 = _stats(live, out, odt)
    info = {"changed": True, "before": s0, "after": s1, "target": target, "amount": a,
            "skin_before": skin0, "skin_after": _skin_top(skin_parts, out, odt)}
    return out, info


def describe(info: dict[str, Any]) -> str:
    if not info.get("changed"):
        return ""
    b, a = info["before"], info["after"]
    return (f"Finishing pass: blacks set from {b['black'] * 100:.0f}% to {a['black'] * 100:.0f}%, "
            f"brightest tones from {b['white'] * 100:.0f}% to {a['white'] * 100:.0f}%, "
            f"midtone contrast {((a['q75'] - a['q25']) / max(b['q75'] - b['q25'], 1e-4) - 1) * 100:+.0f}%."
            + (f" Midtones lowered so the brightest skin sits at {info['skin_after'] * 100:.0f}% (it reached {info['skin_before'] * 100:.0f}%)."
               if info.get("skin_before", 0) > SKIN_TOP + 0.01 else " Midtone level unchanged."))
