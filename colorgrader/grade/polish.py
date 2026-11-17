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
CURVE_GAIN = 0.50                 # extra midtone slope of the film curve at full amount on flat footage
CURVE_MIN = 0.30                  # share of the film curve a picture with full contrast still takes
DENSITY = 0.08                    # luminance taken from the most saturated colors at full amount
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
        signed = odt.signed(pipeline.look_stage(part, look))
        y = np.clip(signed, 0.0, 1.0) @ LUMA
        # below black the signed value keeps counting: a crushed shot reads as negative, never as a plain zero
        lows.append(float(np.percentile(np.clip(signed, -0.25, 1.0) @ LUMA, 1.0)))
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
    live = [p[:: max(1, len(p) // 1200)] for p in live]
    skin_parts = [sk[:: max(1, len(sk) // 600)] for sk in (skin_parts or []) if sk is not None and len(sk) >= 30]
    s0 = _stats(live, look, odt)
    skin0 = _skin_top(skin_parts, look, odt)
    black_goal = float(np.clip(s0["black"], BLACK_MIN, BLACK))        # deepen gray blacks, open crushed ones
    white_goal = min(max(s0["white"], WHITE), s0["white"] + 0.25)
    # skin that glows sets the midtone level: the face matters more than the average of the frame
    over = max(0.0, skin0 - SKIN_TOP)
    median_goal = s0["median"] - 0.9 * over
    target = {
        # gray blacks deepen by the amount. Crushed blacks open in full: lost shadow detail is never a style here
        "black": black_goal if s0["black"] < BLACK_MIN else s0["black"] + a * (black_goal - s0["black"]),
        "white": s0["white"] + a * (white_goal - s0["white"]),
        "median": median_goal,
    }
    weights = {"black": 3.0, "white": 2.0, "median": 3.0}

    def solve(base: dict[str, Any]) -> dict[str, Any]:
        def resid(x: np.ndarray) -> np.ndarray:
            k = _apply(base, x)
            st = _stats(live, k, odt)
            r = [weights[key] * (st[key] - target[key]) for key in weights]
            if skin_parts:
                r.append(6.0 * max(0.0, _skin_top(skin_parts, k, odt) - SKIN_TOP))
            return np.array(r + list(0.004 * x))

        lo = np.array([(LIMITS["contrast"][0] / base["contrast"] - 1.0) / 0.3, (LIMITS["lift"][0] - min(base["lift"])) / 0.03,
                       (LIMITS["gamma"][0] - min(base["gamma"])) / 0.06, (LIMITS["gain"][0] / max(base["gain"]) - 1.0) / 0.1])
        hi = np.array([(LIMITS["contrast"][1] / base["contrast"] - 1.0) / 0.3, (LIMITS["lift"][1] - max(base["lift"])) / 0.03,
                       (LIMITS["gamma"][1] - max(base["gamma"])) / 0.06, (LIMITS["gain"][1] / min(base["gain"]) - 1.0) / 0.1])
        lo, hi = np.minimum(lo, -1e-3), np.maximum(hi, 1e-3)
        # the solver takes its difference steps relative to x: it runs on x + 1, so a start at zero still moves
        sol = least_squares(lambda u: resid(u - 1.0), np.ones(4), bounds=(lo + 1.0, hi + 1.0), diff_step=0.03, max_nfev=30)
        return _apply(base, sol.x - 1.0)

    out = solve(look)
    for _ in range(2):   # display levels near black react sharply: refine from the first answer
        if abs(_stats(live, out, odt)["black"] - target["black"]) < 0.012:
            break
        out = solve(out)
    # a flat picture takes the full film curve, a picture with full contrast a part of it
    flat = float(np.clip((FULL_IQR - (s0["q75"] - s0["q25"])) / (FULL_IQR - FLAT_IQR), CURVE_MIN, 1.0))
    out["tone_curve"] = film_curve(out, live, skin_parts, odt, 1.0 + CURVE_GAIN * a * flat)
    out["sat_vs_lum"] = density_points(DENSITY * a) if out.get("saturation", 50.0) > 5.0 else []
    s1 = _stats(live, out, odt)
    info = {"changed": True, "before": s0, "after": s1, "target": target, "amount": a,
            "skin_before": skin0, "skin_after": _skin_top(skin_parts, out, odt)}
    return out, info


def _luma_di(parts: list[np.ndarray], look: dict[str, Any]) -> np.ndarray:
    return np.concatenate([pipeline.look_stage(p, look).mean(axis=1) for p in parts])


def curve_points(black: float, mid: float, white: float, slope_lo: float, slope_hi: float) -> list[list[float]]:
    """S-curve in DaVinci Intermediate: black, middle and white stay put, the midtones around `mid` get steeper.

    The curve eases into black and white the way a film print does: more contrast where the
    picture lives, soft ends, nothing newly clipped or crushed.
    """
    d_lo, d_hi = 0.35 * (mid - black), 0.35 * (white - mid)
    pts = [[black, black], [mid - d_lo, mid - slope_lo * d_lo], [mid, mid], [mid + d_hi, mid + slope_hi * d_hi], [white, white]]
    if black > 0.03:
        pts.insert(0, [0.0, 0.0])
    if white < 0.97:
        pts.append([1.0, 1.0])
    return [[float(a), float(b)] for a, b in pts]


def film_curve(look: dict[str, Any], parts: list[np.ndarray], skin_parts: list[np.ndarray], odt, slope: float) -> list[list[float]]:
    """The film-style contrast curve for this footage under this look. Empty when the slope adds nothing."""
    if slope <= 1.02:
        return []
    base = dict(look, tone_curve=[])
    y = _luma_di(parts, base)
    black, mid, white = (float(v) for v in np.percentile(y, [1.0, 50.0, 99.0]))
    black, white = max(black, 0.0), min(white, 1.0)
    if not (black < mid - 0.05 and mid + 0.05 < white):
        return []
    hi = slope
    pts = curve_points(black, mid, white, slope, hi)
    for _ in range(8):   # the upper half pushes bright skin up: soften it until faces keep their shape
        if not skin_parts or _skin_top(skin_parts, dict(base, tone_curve=pts), odt) <= max(SKIN_TOP, _skin_top(skin_parts, base, odt)) + 0.005:
            break
        hi = 1.0 + (hi - 1.0) * 0.7
        pts = curve_points(black, mid, white, slope, hi)
    return pts


def density_points(depth: float) -> list[list[float]]:
    """Sat vs Lum curve: strong colors get darker and richer, the way dye on film takes light away.

    Skin and other gentle colors (saturation under 0.08) stay untouched.
    """
    if depth <= 0.004:
        return []
    return [[0.0, 1.0], [0.08, 1.0], [0.16, 1.0 - 0.6 * depth], [0.26, 1.0 - depth], [1.0, 1.0 - depth]]


def describe(info: dict[str, Any]) -> str:
    if not info.get("changed"):
        return ""
    b, a = info["before"], info["after"]
    low = lambda v: "under 0%" if v < -0.002 else f"{max(v, 0.0) * 100:.0f}%"
    return (f"Finishing pass: blacks set from {low(b['black'])} to {low(a['black'])}, "
            f"brightest tones from {b['white'] * 100:.0f}% to {a['white'] * 100:.0f}%, "
            f"midtone contrast {((a['q75'] - a['q25']) / max(b['q75'] - b['q25'], 1e-4) - 1) * 100:+.0f}% with a film-style curve that eases into black and white."
            + (f" Midtones lowered so the brightest skin sits at {info['skin_after'] * 100:.0f}% (it reached {info['skin_before'] * 100:.0f}%)."
               if info.get("skin_before", 0) > SKIN_TOP + 0.01 else " Midtone level unchanged."))
