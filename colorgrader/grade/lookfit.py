"""Step 4 of the grade: fit the shared look to a reference.

The look is a small set of standard controls: contrast, lift/gamma/gain with
color, saturation, a saturation-by-luma curve and hue curves. The fit finds
the control values that give the footage the reference's statistics:

  tone shape    luma quantiles around the median (exposure stays the footage's own)
  black level   the floor of the reference (lifted or deep blacks)
  tints         color of near-neutral tones in shadows, midtones and highlights
  saturation    chroma level in shadows, midtones and highlights
  palette       saturation and hue of each color family

No pixel-by-pixel transfer: every result is a smooth, rebuildable control.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..color import ops
from . import model, pipeline

F32 = np.float32
U = np.array([0.02, 0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.95, 0.98])
ZONES = ((0.02, 0.25), (0.35, 0.65), (0.78, 0.98))
ZONE_NAMES = ("shadows", "midtones", "highlights")
SIGMA = 0.035       # chroma width of "near neutral", DaVinci Intermediate units
SECTORS = 12
MID_NEAR, MID_FAR = 0.025, 0.09   # midtone distance (DaVinci Intermediate): about 0.34 and 1.2 stops

# parameter scales: x = 1 is a clear but ordinary move
S_CONTRAST, S_LIFT, S_GAMMA = 0.30, 0.03, 0.06
S_TL, S_TG, S_TH = 0.010, 0.020, 0.030
S_SAT, S_LVS = 0.25, 0.40
X_LO = np.array([-1.33, -1.67, -2.0, -3, -3, -3, -3, -3, -3, -3.2, -1.25, -1.25])
X_HI = np.array([2.0, 3.33, 2.0, 3, 3, 3, 3, 3, 3, 1.5, 0.0, 0.0])   # shadows and highlights never gain saturation
# the two saturation-by-luma parameters act on few pixels: a firmer pull to neutral keeps them honest
REG_WEIGHT = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6], dtype=np.float64)
NAMES = ["contrast", "lift", "gamma", "lift tint a", "lift tint b", "gamma tint a", "gamma tint b", "gain tint a", "gain tint b",
         "saturation", "shadow saturation", "highlight saturation"]


def _tint_rgb(c1: float, c2: float) -> np.ndarray:
    """A color offset with zero luma, from two chroma coordinates."""
    return ops.hsl_join(np.zeros(1, dtype=F32), np.array([c1], dtype=F32), np.array([c2], dtype=F32))[0].astype(np.float64)


def lum_vs_sat_points(s_shadow: float, s_high: float, dead_zone: float = 0.02) -> list[list[float]]:
    """Lum vs Sat curve from a shadow and a highlight multiplier. Tiny moves collapse to no curve."""
    if abs(s_shadow - 1.0) < dead_zone and abs(s_high - 1.0) < dead_zone:
        return []
    # anchors in DaVinci Intermediate luma: shadows fade in under 19% on the display, highlights over 75%
    return [[0.0, s_shadow], [0.04, 0.5 * (s_shadow + 1.0)], [0.12, 1.0], [0.50, 1.0], [0.62, 0.5 * (s_high + 1.0)], [0.74, s_high]]


def saturation_limiter(start: float = 0.16, ceiling: float = 0.26) -> list[list[float]]:
    """Sat vs Sat curve that leaves normal colors alone and bends extreme ones under a ceiling.

    Saturation here is Resolve's HSL-curve saturation of the working-space signal. Real-world
    colors stay under about 0.15. The Rec.709 red primary sits near 0.2.
    """
    span = ceiling - start
    pts = [[0.0, 1.0], [start, 1.0]]
    for s_in in (0.20, 0.25, 0.30, 0.40, 0.55, 0.75, 1.0):
        if s_in <= start:
            continue
        out = start + span * float(np.tanh((s_in - start) / span))
        pts.append([s_in, out / s_in])
    return pts


def params_to_look(x: np.ndarray, base: dict[str, Any] | None = None, final: bool = True) -> dict[str, Any]:
    """Fit vector -> look dict in Resolve units (not yet rounded). final=False keeps tiny curve moves (for fitting)."""
    look = model.neutral_look() if base is None else dict(base)
    look["contrast"] = 1.0 + S_CONTRAST * x[0]
    look["pivot"] = model.PIVOT
    look["lift"] = list(S_LIFT * x[1] + _tint_rgb(S_TL * x[3], S_TL * x[4]))
    look["gamma"] = list(S_GAMMA * x[2] + _tint_rgb(S_TG * x[5], S_TG * x[6]))
    look["gain"] = list(1.0 + _tint_rgb(S_TH * x[7], S_TH * x[8]))
    look["saturation"] = 50.0 * float(np.exp(S_SAT * x[9]))
    s_sh, s_hi = 1.0 + S_LVS * x[10], 1.0 + S_LVS * x[11]
    if final:  # moves under 6% are noise: drop them one by one
        s_sh = 1.0 if abs(s_sh - 1.0) < 0.06 else s_sh
        s_hi = 1.0 if abs(s_hi - 1.0) < 0.06 else s_hi
    look["lum_vs_sat"] = lum_vs_sat_points(s_sh, s_hi, 0.02 if final else 0.0)
    return look


def look_to_params(look: dict[str, Any]) -> np.ndarray:
    """Inverse of params_to_look for looks in canonical form (used by tests and saved looks)."""
    x = np.zeros(12)
    x[0] = (look["contrast"] - 1.0) / S_CONTRAST
    w = ops.HSL_WEIGHTS.astype(np.float64)

    def split(vals, neutral):
        v = np.asarray(vals, dtype=np.float64) - neutral
        m = float(v @ w)
        _, c1, c2, _, _ = ops.hsl_split((v - m)[None, :].astype(F32))
        return m, float(c1[0]), float(c2[0])

    m, a, b = split(look["lift"], 0.0)
    x[1], x[3], x[4] = m / S_LIFT, a / S_TL, b / S_TL
    m, a, b = split(look["gamma"], 0.0)
    x[2], x[5], x[6] = m / S_GAMMA, a / S_TG, b / S_TG
    _, a, b = split(look["gain"], 1.0)
    x[7], x[8] = a / S_TH, b / S_TH
    x[9] = np.log(look["saturation"] / 50.0) / S_SAT
    lvs = look.get("lum_vs_sat") or []
    if lvs:
        x[10] = (lvs[0][1] - 1.0) / S_LVS
        x[11] = (lvs[-1][1] - 1.0) / S_LVS
    return x


def features(di: np.ndarray) -> dict[str, Any]:
    """Look statistics of one picture: a working-space sample (N, 3)."""
    lum, c1, c2, hue, chroma = ops.hsl_split(di)
    order = np.argsort(lum, kind="stable")
    n = len(lum)
    rank = np.empty(n, dtype=F32)
    rank[order] = np.linspace(0.0, 1.0, n, dtype=F32)
    srt = lum[order]
    q = srt[np.clip((U * (n - 1)).astype(int), 0, n - 1)]
    med = float(srt[n // 2])
    at = lambda u: float(srt[int(u * (n - 1))])
    spread = np.array([at(0.75) - at(0.25), at(0.90) - at(0.10), at(0.95) - at(0.05)])
    w = np.exp(-(chroma / SIGMA) ** 2)
    tints, sats = [], []
    for lo, hi in ZONES:
        sel = (rank >= lo) & (rank < hi)
        ww = w[sel]
        tot = max(float(ww.sum()), 1e-6)
        tints += [float((ww * c1[sel]).sum() / tot), float((ww * c2[sel]).sum() / tot)]
        # colorfulness of the colored things in this zone: gray areas do not dilute it
        sats.append(float(np.percentile(chroma[sel], 90)) if sel.any() else 1e-4)
    return {"tone": (q - med).astype(np.float64), "spread": spread, "median": med,
            "floor": float(srt[int(0.005 * (n - 1))]), "ceil": float(srt[int(0.995 * (n - 1))]),
            "tints": np.array(tints), "sats": np.maximum(np.array(sats), 1e-4)}


def features_multi(di: np.ndarray, bounds: list[tuple[int, int]]) -> dict[str, Any]:
    """Average the statistics of several shots. Each shot is measured on its own, so exposure
    differences between shots never read as contrast."""
    if len(bounds) == 1:
        return features(di[bounds[0][0]:bounds[0][1]])
    fs = [features(di[a:b]) for a, b in bounds]
    out = {k: np.mean([f[k] for f in fs], axis=0) for k in ("tone", "tints", "spread")}
    out["sats"] = np.exp(np.mean([np.log(f["sats"]) for f in fs], axis=0))
    for k in ("median", "floor", "ceil"):
        out[k] = float(np.mean([f[k] for f in fs]))
    return out


def _solve(resid, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, iters: int = 14, h: float = 0.05) -> tuple[np.ndarray, float]:
    """Damped Gauss-Newton with box limits. The fit is small, smooth and well conditioned."""
    x = np.clip(x0.astype(np.float64), lo, hi)
    r = resid(x)
    cost = float(r @ r)
    lam = 1e-3
    n = len(x)
    for _ in range(iters):
        jac = np.empty((len(r), n))
        for i in range(n):
            step = h if x[i] + h <= hi[i] else -h
            xi = x.copy()
            xi[i] += step
            jac[:, i] = (resid(xi) - r) / step
        a, g = jac.T @ jac, jac.T @ r
        improved = False
        for _try in range(8):
            dx = np.linalg.solve(a + lam * np.diag(np.diag(a)) + 1e-9 * np.eye(n), -g)
            xn = np.clip(x + dx, lo, hi)
            rn = resid(xn)
            cn = float(rn @ rn)
            if cn < cost:
                gain = cost - cn
                x, r, cost = xn, rn, cn
                lam = max(lam / 3.0, 1e-6)
                improved = True
                break
            lam *= 4.0
        if not improved or gain < 1e-7 * max(cost, 1e-9):
            break
    return x, cost


# How far the footage statistics move toward the reference when the two pictures differ.
# Each number reflects how much of a statistic belongs to the grade and how much to the scene.
CARRY = {"spread": 0.5, "floor_up": 0.85, "floor_down": 0.4, "tints": 0.8, "sats": 0.5, "median": 0.7}
# limits for different-content references, in fit units (contrast 0.82..1.45, lift -0.02..0.06, gamma +-0.06 ...)
B_LO = np.array([-0.6, -0.67, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.3, -1.25, -1.25])
B_HI = np.array([1.5, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.9, 0.0, 0.0])


def fit(footage_di: np.ndarray | list[np.ndarray], ref: dict[str, Any], ref_info: dict[str, Any] | None = None,
        reg: float = 0.03, carry_override: dict[str, float] | None = None,
        limits: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find the look that gives the footage the reference's statistics.

    footage_di: balanced footage, one sample (N, 3) per shot. ref: features() of the reference.
    ref_info: {"capped": bool, "monochrome": bool, "same_content": bool}.

    Same picture (the reference is one of the shots, graded): the statistics match in full and
    the known grade comes back. Different picture: scene content shapes the statistics too, so
    the footage moves part of the way and the controls stay inside gentle limits.
    """
    ref_info = ref_info or {}
    parts = footage_di if isinstance(footage_di, list) else [footage_di]
    bounds, start = [], 0
    for part in parts:
        bounds.append((start, start + len(part)))
        start += len(part)
    x_sample = np.concatenate(parts).astype(F32)
    feats = lambda arr: features_multi(arr, bounds)
    f0 = feats(x_sample)
    same = bool(ref_info.get("same_content"))
    mono = bool(ref_info.get("monochrome"))
    d_med = ref["median"] - f0["median"]
    # Exposure stays the footage's own. A reference whose midtones sit close to the footage
    # passes its midtone level on. A night scene on a day shot passes on nothing.
    carry = 1.0 if same else CARRY["median"] * float(np.clip(1.0 - (abs(d_med) - MID_NEAR) / (MID_FAR - MID_NEAR), 0.0, 1.0))
    k = {key: 1.0 for key in CARRY} if same else (carry_override or CARRY)   # designed targets (look suggestions) carry in full
    target = {
        "spread": f0["spread"] * (ref["spread"] / np.maximum(f0["spread"], 1e-4)) ** k["spread"],
        "floor": f0["floor"] + (k["floor_up"] if ref["floor"] >= f0["floor"] else k["floor_down"]) * (ref["floor"] - f0["floor"]),
        "tints": f0["tints"] + k["tints"] * (ref["tints"] - f0["tints"]),
        "sats": f0["sats"] * (ref["sats"] / f0["sats"]) ** k["sats"],
        "median": f0["median"] + carry * d_med,
        "ceil": ref["ceil"] if ref_info.get("capped") else None,
    }

    def resid(x: np.ndarray) -> np.ndarray:
        g = pipeline.look_stage(x_sample, params_to_look(x, final=False))
        f = feats(g)
        # contrast comes from the width of the tone distribution, never from its shape:
        # shape belongs to the picture content, width to the grade
        r = [40.0 * (f["spread"] - target["spread"]), [20.0 * (f["median"] - target["median"])], [25.0 * (f["floor"] - target["floor"])]]
        if target["ceil"] is not None:
            r.append([15.0 * (f["ceil"] - target["ceil"])])
        r.append(60.0 * (f["tints"] - target["tints"]))
        if not mono:
            r.append(1.5 * (np.log(f["sats"]) - np.log(target["sats"])))
        r.append(reg * REG_WEIGHT * x)
        return np.concatenate([np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in r])

    lo, hi = (X_LO.copy(), X_HI.copy()) if same else (B_LO.copy(), B_HI.copy())
    if limits is not None:
        lo, hi = np.asarray(limits[0], dtype=np.float64).copy(), np.asarray(limits[1], dtype=np.float64).copy()
    if mono:
        lo[9], hi[9] = -1e-6, 1e-6
    x, cost = _solve(resid, np.zeros(12), lo, hi)
    look = params_to_look(x)
    if mono:
        look["saturation"] = 0.0
    at_limit = [NAMES[i] for i in range(10) if (x[i] <= lo[i] + 1e-3 or x[i] >= hi[i] - 1e-3) and not (mono and i == 9)]
    g = pipeline.look_stage(x_sample, look)
    info = {"x": [round(float(v), 4) for v in x], "cost": cost, "at_limit": at_limit, "mode": "same picture" if same else "different picture",
            "footage": f0, "result": feats(g), "reference": ref, "floor_target": target["floor"],
            "exposure_gap_stops": round(float(d_med) / 0.0733, 2), "midtone_carry": round(carry, 2)}
    return look, info


def refine_paired(x0: np.ndarray, footage_frame_di: np.ndarray, ref_frame_di: np.ndarray, ref_display: np.ndarray,
                  reg: float = 0.01) -> tuple[np.ndarray, dict[str, Any]]:
    """Sharpen a same-picture fit by comparing the two frames cell by cell.

    Both frames shrink to a coarse grid, so small framing differences average out. Cells where
    the reference clips carry no information and drop out. The refined values stand only when
    they explain the reference better than the starting values.
    """
    import cv2

    grid = (64, 36)
    a = cv2.resize(footage_frame_di.astype(F32), grid, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    b = cv2.resize(ref_frame_di.astype(F32), grid, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    d = cv2.resize(np.clip(ref_display, 0, 1).astype(F32), grid, interpolation=cv2.INTER_AREA).reshape(-1, 3)
    keep = (d.min(axis=1) > 0.02) & (d.max(axis=1) < 0.98)
    if keep.sum() < 200:
        return x0, {"used": False, "why": "too few usable cells"}
    a, b = a[keep], b[keep]

    def raw(x: np.ndarray) -> np.ndarray:
        return (pipeline.look_stage(a, params_to_look(x, final=False)) - b).astype(np.float64).ravel()

    def resid(x: np.ndarray) -> np.ndarray:
        r = raw(x) * 40.0
        k = 0.6  # Huber: cells that moved in the frame count less
        r = np.where(np.abs(r) <= k, r, np.sign(r) * np.sqrt(np.maximum(k * (2 * np.abs(r) - k), 0.0)))
        return np.concatenate([r, reg * REG_WEIGHT * (x - x0), 0.03 * REG_WEIGHT * x * (REG_WEIGHT > 1)])

    before = float(np.median(np.abs(raw(x0))))
    x1, _ = _solve(resid, x0.copy(), X_LO, X_HI, iters=8)
    after = float(np.median(np.abs(raw(x1))))
    ok = after < before and after < 0.012
    return (x1 if ok else x0), {"used": bool(ok), "before": round(before, 5), "after": round(after, 5), "cells": int(keep.sum())}


# ------------------------------------------------------------------ palette
def hue_profile(di: np.ndarray) -> dict[str, np.ndarray]:
    """Per hue sector: share of chroma mass, mean chroma, mean hue."""
    lum, c1, c2, hue, chroma = ops.hsl_split(di)
    sel = (chroma > 0.012) & (lum > 0.08) & (lum < 0.75)
    hue, chroma, c1, c2 = hue[sel], chroma[sel], c1[sel], c2[sel]
    width = 360.0 / SECTORS
    idx = (((hue + width / 2) % 360.0) // width).astype(int) % SECTORS
    mass = np.bincount(idx, weights=chroma, minlength=SECTORS)
    count = np.bincount(idx, minlength=SECTORS)
    mean_c = mass / np.maximum(count, 1)
    s1 = np.bincount(idx, weights=c1, minlength=SECTORS)
    s2 = np.bincount(idx, weights=c2, minlength=SECTORS)
    mean_h = np.degrees(np.arctan2(s2, s1)) % 360.0
    total = max(float(mass.sum()), 1e-9)
    return {"mass": mass / total, "chroma": mean_c, "hue": mean_h, "count": count,
            "colored_share": float(sel.mean()) if len(sel) else 0.0}


def _wrap(d: np.ndarray | float):
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def fit_palette(graded_di: np.ndarray, ref_di: np.ndarray, skin_hue: float | None = None,
                strength: float = 0.7) -> tuple[list[list[float]], list[list[float]], list[str]]:
    """Hue vs Sat and Hue vs Hue points that bring each color family toward the reference.

    graded_di: footage after the fitted look. skin_hue: center of the skin band, left untouched.
    """
    pf, pr = hue_profile(graded_di), hue_profile(ref_di)
    width = 360.0 / SECTORS
    centers = np.arange(SECTORS) * width
    both = (pf["mass"] >= 0.03) & (pr["mass"] >= 0.03)
    mult = np.ones(SECTORS)
    notes: list[str] = []
    if both.sum() >= 2:
        ratio = pr["chroma"][both] / np.maximum(pf["chroma"][both], 1e-5)
        norm = float(np.median(ratio))
        mult[both] = np.clip((ratio / norm) ** strength, 0.6, 1.15)
    absent = (pf["mass"] >= 0.05) & (pr["mass"] < 0.01)
    mult[absent] = 0.88
    shift = np.zeros(SECTORS)
    strong = (pf["mass"] >= 0.05) & (pr["mass"] >= 0.05)
    d = _wrap(pr["hue"] - pf["hue"])
    shift[strong] = np.clip(0.5 * d[strong], -10.0, 10.0)
    shift[np.abs(shift) < 2.0] = 0.0
    if skin_hue is not None:
        near = np.abs(_wrap(centers - skin_hue)) <= 28.0
        mult[near] = 1.0
        shift[near] = 0.0
    mult = 0.25 * np.roll(mult, 1) + 0.5 * mult + 0.25 * np.roll(mult, -1)
    mult[np.abs(mult - 1.0) < 0.04] = 1.0
    hvs = [[float(c), float(m)] for c, m in zip(centers, mult)] if np.any(mult != 1.0) else []
    hvh = [[float(c), float(s)] for c, s in zip(centers, shift)] if np.any(shift != 0.0) else []
    from ..analysis.names import hue_name

    for c, m in zip(centers, mult):
        if m <= 0.85:
            notes.append(f"{hue_name(c)} muted")
        elif m >= 1.15:
            notes.append(f"{hue_name(c)} richer")
    return hvs, hvh, sorted(set(notes))
