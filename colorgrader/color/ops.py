"""Grading operators with the same math as DaVinci Resolve's controls.

Every function takes the numbers a colorist types into Resolve and applies
the math Resolve applies. The recipe then lists the same numbers.

Sources for the math: measured recreations of Resolve's Primaries, Hue/Sat
and HSL curve tools by Thatcher Freeman (utility-dctls, MIT license).
Lift/Gamma/Gain/Offset math holds for Lum Mix = 0. Contrast is the linear
form, used when "Use S-curve for contrast" is off in Project Settings.

Arrays are float32, shape (..., 3).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

F32 = np.float32

OFFSET_UNIT = 7.5 / 1023.0  # one Offset point: 7.5 ten-bit code values (a printer light point)
OFFSET_BASE = 25.0
SAT_WEIGHTS = np.array([0.21, 0.70, 0.09], dtype=F32)        # Primaries Sat knob
HSL_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=F32)  # HSL curves
HSL_K = 0.86666666


# ---------------------------------------------------------------- primaries
def gamma_ui_to_exponent(g):
    g = np.asarray(g, dtype=np.float64)
    return np.where(g <= 0, 1.0 - 4.0 * g, 1.0 / (4.0 * np.maximum(g, 0) + 1.0))


def gamma_exponent_to_ui(e):
    e = np.asarray(e, dtype=np.float64)
    return np.where(e >= 1.0, (e - 1.0) / -4.0, (1.0 / e - 1.0) / 4.0)


def lggo(x: np.ndarray, lift=(0.0, 0.0, 0.0), gamma=(0.0, 0.0, 0.0), gain=(1.0, 1.0, 1.0),
         offset=(25.0, 25.0, 25.0)) -> np.ndarray:
    """Lift, Gamma, Gain, Offset as shown in the Primaries palette (R, G, B values)."""
    lift = np.asarray(lift, dtype=np.float64) * 2.0
    gain = np.asarray(gain, dtype=np.float64)
    off = (np.asarray(offset, dtype=np.float64) - OFFSET_BASE) * OFFSET_UNIT
    black = 1.0 - 1.0 / (1.0 - lift)
    white = 1.0 / gain
    slope = 1.0 / (white - black)
    ge = gamma_ui_to_exponent(gamma)
    y = (x - black.astype(F32)) * slope.astype(F32) + off.astype(F32)
    if np.allclose(ge, 1.0):
        return y.astype(F32, copy=False)
    return (np.sign(y) * np.power(np.abs(y), ge.astype(F32))).astype(F32, copy=False)


def lggo_affine(lift, gain, offset):
    """Slope and intercept per channel of the linear part of lggo()."""
    lift = np.asarray(lift, dtype=np.float64) * 2.0
    gain = np.asarray(gain, dtype=np.float64)
    off = (np.asarray(offset, dtype=np.float64) - OFFSET_BASE) * OFFSET_UNIT
    black = 1.0 - 1.0 / (1.0 - lift)
    slope = 1.0 / (1.0 / gain - black)
    return slope, off - slope * black


def contrast(x: np.ndarray, amount: float, pivot: float) -> np.ndarray:
    """Linear contrast around a pivot (S-curve option off)."""
    if amount == 1.0:
        return x
    return ((x - F32(pivot)) * F32(amount) + F32(pivot)).astype(F32, copy=False)


def saturation(x: np.ndarray, sat_ui: float) -> np.ndarray:
    """Primaries Sat knob. 50 is neutral, 0 is monochrome, 100 doubles."""
    if sat_ui == 50.0:
        return x
    k = F32(sat_ui / 50.0)
    lum = (x @ SAT_WEIGHTS)[..., None]
    return (lum + (x - lum) * k).astype(F32, copy=False)


def rgb_mixer(x: np.ndarray, m) -> np.ndarray:
    m = np.asarray(m, dtype=F32).reshape(3, 3)
    return (x.reshape(-1, 3) @ m.T).reshape(x.shape)


# -------------------------------------------------------------- curve tables
def _hashable(points):
    return tuple((float(a), float(b)) for a, b in points)


@lru_cache(maxsize=256)
def _periodic_table(points: tuple, default: float, n: int = 1440) -> np.ndarray:
    """Smooth periodic curve through (hue_degrees, value) points, sampled on n steps."""
    if not points:
        return np.full(n, default, dtype=F32)
    from scipy.interpolate import PchipInterpolator

    pts = sorted(((h % 360.0), v) for h, v in points)
    hs = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts])
    keep = np.concatenate([[True], np.diff(hs) > 1e-6])
    hs, vs = hs[keep], vs[keep]
    if len(hs) == 1:
        return np.full(n, vs[0], dtype=F32)
    xs = np.concatenate([hs - 360.0, hs, hs + 360.0])
    ys = np.concatenate([vs, vs, vs])
    f = PchipInterpolator(xs, ys)
    return f(np.arange(n) * (360.0 / n)).astype(F32)


@lru_cache(maxsize=256)
def _unit_table(points: tuple, default: float, n: int = 1024) -> np.ndarray:
    """Smooth curve through (input 0..1, value) points, flat beyond the end points."""
    if not points:
        return np.full(n, default, dtype=F32)
    from scipy.interpolate import PchipInterpolator

    pts = sorted(points)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    keep = np.concatenate([[True], np.diff(xs) > 1e-6])
    xs, ys = xs[keep], ys[keep]
    grid = np.linspace(0.0, 1.0, n)
    if len(xs) == 1:
        return np.full(n, ys[0], dtype=F32)
    out = PchipInterpolator(xs, ys)(np.clip(grid, xs[0], xs[-1]))
    return out.astype(F32)


def eval_periodic(points, hue_deg: np.ndarray, default: float) -> np.ndarray:
    table = _periodic_table(_hashable(points), float(default))
    n = len(table)
    pos = (np.asarray(hue_deg, dtype=F32) % F32(360.0)) * F32(n / 360.0)
    i0 = np.floor(pos).astype(np.int32) % n
    frac = pos - np.floor(pos)
    return table[i0] * (F32(1.0) - frac) + table[(i0 + 1) % n] * frac


def eval_unit(points, x: np.ndarray, default: float) -> np.ndarray:
    table = _unit_table(_hashable(points), float(default))
    n = len(table)
    pos = np.clip(np.asarray(x, dtype=F32), 0.0, 1.0) * F32(n - 1)
    i0 = np.minimum(np.floor(pos).astype(np.int32), n - 2)
    frac = pos - i0
    return table[i0] * (F32(1.0) - frac) + table[i0 + 1] * frac


# ---------------------------------------------------------------- HSL curves
def hsl_split(x: np.ndarray):
    """Luma, two chroma axes, hue in degrees (red 0, yellow 60, green 120, blue 240) and chroma size."""
    lum = x @ HSL_WEIGHTS
    c1 = x[..., 0] - F32(0.5) * (x[..., 1] + x[..., 2])
    c2 = F32(HSL_K) * (x[..., 1] - x[..., 2])
    hue = np.degrees(np.arctan2(c2, c1)) % F32(360.0)
    return lum, c1, c2, hue, np.hypot(c1, c2)


def hsl_join(lum: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    # inverse of [[wr, wg, wb], [1, -.5, -.5], [0, k, -k]]
    wr, wg, wb = (float(v) for v in HSL_WEIGHTS)
    k = HSL_K
    r = lum + F32(wg + wb) * c1 + F32((wb - wg) / (2.0 * k)) * c2
    g = lum - F32(wr) * c1 + F32((wr + 2.0 * wb) / (2.0 * k)) * c2
    b = lum - F32(wr) * c1 - F32((wr + 2.0 * wg) / (2.0 * k)) * c2
    return np.stack([r, g, b], axis=-1).astype(F32, copy=False)


def hsv_saturation(x: np.ndarray, gain: float = 1.0, gamma: float = 0.0) -> np.ndarray:
    """A node in HSV color space with channels 1 and 3 off: Gain and Gamma act on saturation alone.

    Hue and value (the largest channel) stay. More saturation lowers the other channels, so colors
    get denser and darker as they saturate, the way film dye behaves. Gain lifts strong colors most,
    Gamma lifts gentle colors (skin, walls) most.
    """
    if gain == 1.0 and gamma == 0.0:
        return x
    v = x.max(axis=-1, keepdims=True)
    mn = x.min(axis=-1, keepdims=True)
    ok = v > F32(1e-6)
    s = np.where(ok, (v - mn) / np.where(ok, v, F32(1.0)), F32(0.0))
    s2 = np.maximum(s * F32(gain), F32(0.0))
    e = float(gamma_ui_to_exponent(gamma))
    if e != 1.0:
        s2 = np.power(s2, F32(e))
    k = np.where(s > F32(1e-6), s2 / np.where(s > F32(1e-6), s, F32(1.0)), F32(1.0))
    return np.where(ok, v - (v - x) * k, x).astype(F32, copy=False)


def tone_curve(x: np.ndarray, points) -> np.ndarray:
    """Custom curve, all channels ganged: one smooth monotone curve through (input, output) points.

    Values outside the first and last point keep their distance to the curve ends, so nothing clips here.
    """
    if not points:
        return x
    lo, hi = min(p[0] for p in points), max(p[0] for p in points)
    xc = np.clip(x, F32(max(lo, 0.0)), F32(min(hi, 1.0)))
    return (eval_unit(points, xc, 0.0) + (x - xc)).astype(F32, copy=False)


def channel_curves(x: np.ndarray, red=None, green=None, blue=None) -> np.ndarray:
    """Custom curves with gang off: one curve per channel. Same extension rule as tone_curve()."""
    if not (red or green or blue):
        return x
    out = np.array(x, dtype=F32, copy=True)
    for c, points in enumerate((red, green, blue)):
        if points:
            out[..., c] = tone_curve(out[..., c], points)
    return out


HUE_LUM_CHROMA = 0.05   # a color counts in full for Hue vs Lum from this chroma up; grays have no hue to key on


def hsl_curves(x: np.ndarray, hue_vs_hue=None, hue_vs_sat=None, lum_vs_sat=None, sat_vs_sat=None, sat_vs_lum=None,
               hue_vs_lum=None) -> np.ndarray:
    """Hue vs Hue (degrees of rotation), Hue vs Sat, Lum vs Sat, Sat vs Sat, Sat vs Lum and Hue vs Lum (multipliers).

    Points are (input, value). Hue inputs are degrees on the usual color wheel. Hue vs Lum scales the luminance
    of a color family, in full for clear colors and fading out toward gray.
    """
    if not (hue_vs_hue or hue_vs_sat or lum_vs_sat or sat_vs_sat or sat_vs_lum or hue_vs_lum):
        return x
    lum, c1, c2, hue, chroma = hsl_split(x)
    if hue_vs_lum:
        w = np.clip(chroma / F32(HUE_LUM_CHROMA), 0.0, 1.0)
        lum = lum * (F32(1.0) + (eval_periodic(hue_vs_lum, hue, 1.0) - F32(1.0)) * w)
    if sat_vs_lum:
        lum = lum * eval_unit(sat_vs_lum, chroma, 1.0)
    scale = np.ones_like(lum)
    if hue_vs_sat:
        scale = scale * eval_periodic(hue_vs_sat, hue, 1.0)
    if lum_vs_sat:
        scale = scale * eval_unit(lum_vs_sat, lum, 1.0)
    if sat_vs_sat:
        scale = scale * eval_unit(sat_vs_sat, chroma, 1.0)
    if hue_vs_hue:
        rot = np.radians(eval_periodic(hue_vs_hue, hue, 0.0))
        cs, sn = np.cos(rot), np.sin(rot)
        c1, c2 = c1 * cs - c2 * sn, c1 * sn + c2 * cs
    return hsl_join(lum, c1 * scale, c2 * scale)
