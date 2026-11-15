"""White balance estimate from the picture.

Finds the color cast as two numbers in stops: red and blue relative to green,
in linear working-space light. Evidence, strongest first:
  1. neutral objects: a cluster of near-gray pixels close to the camera's own balance
  2. skin: a detected face must land on the skin tone line
  3. sky and foliage: their hue must stay inside the natural range
The camera's own white balance is the prior. Weak evidence means a small correction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..color import working as W
from ..color.ycbcr import rgb_to_ycbcr

F32 = np.float32
SKIN_LINE = 123.0   # degrees on the vectorscope
SKIN_DIRECTION = -28.7  # degrees, hue direction of skin in linear Rec.709 log chroma (ColorChecker skin patches)
SKY_HUE = (198.0, 226.0)
FOLIAGE_HUE = (72.0, 132.0)
GRID = 0.02         # stops per cell
SPAN = 0.8          # search range in stops, each side


def vector_angle(display_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorscope angle in degrees (Cb right, Cr up) and chroma size, for display R'G'B'."""
    _, cb, cr = rgb_to_ycbcr(display_rgb)
    return np.degrees(np.arctan2(cr, cb)) % 360.0, np.hypot(cb, cr)


def _hsv_hue_sat_val(rgb: np.ndarray):
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    d = mx - mn
    safe = np.where(d > 1e-6, d, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mx == r, ((g - b) / safe) % 6.0, np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) * 60.0
    h = np.where(d > 1e-6, h, 0.0)
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


def gains_from_cast(dr: float, db: float) -> np.ndarray:
    """Linear RGB gains removing a cast, normalized to keep the luminance of gray."""
    g = np.array([2.0 ** (-dr), 1.0, 2.0 ** (-db)], dtype=np.float64)
    wy = np.array([0.27411851, 0.87363190, -0.14775041])  # DaVinci Wide Gamut luminance weights
    return (g / float(wy @ g)).astype(F32)


def _circ_mean(deg: np.ndarray, weights: np.ndarray | None = None) -> float:
    rad = np.radians(deg)
    w = np.ones_like(rad) if weights is None else weights
    return float(np.degrees(np.arctan2((np.sin(rad) * w).sum(), (np.cos(rad) * w).sum())) % 360.0)


def estimate_cast(lin: np.ndarray, ypos: np.ndarray | None, odt, skin_lin: np.ndarray | None = None) -> dict[str, Any]:
    """Estimate the color cast of a shot.

    lin: (N, 3) linear DaVinci Wide Gamut samples. ypos: (N,) vertical position 0 (top) to 1.
    skin_lin: skin samples from detected faces. odt: OutputTransform, for hue checks on the display.
    """
    lin = np.asarray(lin, dtype=F32)
    lum = lin @ np.array([0.27411851, 0.87363190, -0.14775041], dtype=F32)
    valid = (lin.min(axis=1) > 1e-4) & (lum > 0.025) & (lin.max(axis=1) < 6.0)
    px = lin[valid]
    notes: list[str] = []
    n_cells = int(round(2 * SPAN / GRID)) + 1
    axis = np.linspace(-SPAN, SPAN, n_cells)
    DR, DB = np.meshgrid(axis, axis, indexing="ij")
    T = (DR - DB) / np.sqrt(2.0)   # warm (+) or cool (-)
    M = (DR + DB) / np.sqrt(2.0)   # magenta (+) or green (-)
    prior = T ** 2 / (2 * 0.30 ** 2) + M ** 2 / (2 * 0.17 ** 2)
    cost = prior.copy()
    info: dict[str, Any] = {"neutral_share": 0.0, "skin": None, "sky": None, "foliage": None}

    # classify sky and foliage on the picture as shot: the camera's balance is the prior
    sky = np.zeros(len(px), dtype=bool)
    fol = np.zeros(len(px), dtype=bool)
    if ypos is not None and len(px) > 500:
        hue, sat, val = _hsv_hue_sat_val(np.clip(odt(W.di_encode(px)), 0, 1))
        yp = ypos[valid]
        sky = (yp < 0.55) & (hue > 170) & (hue < 265) & (sat > 0.10) & (val > 0.40)
        fol = (hue > 60) & (hue < 165) & (sat > 0.22) & (val > 0.08) & (val < 0.85)

    # 1. neutral objects. Sky is left out: pale sky looks like a gray object and is none.
    evidence = np.zeros((n_cells, n_cells), dtype=np.float64)
    if len(px) > 200:
        dr = np.log2(px[:, 0] / px[:, 1])
        db = np.log2(px[:, 2] / px[:, 1])
        wgt = np.clip(lum[valid] / 0.18, 0.2, 2.0)
        total = max(float(wgt.sum()), 1e-6)
        wgt = np.where(sky, 0.0, wgt)
        inside = (np.abs(dr) < SPAN) & (np.abs(db) < SPAN)
        hist, _, _ = np.histogram2d(dr[inside], db[inside], bins=n_cells, range=[[-SPAN - GRID / 2, SPAN + GRID / 2]] * 2,
                                    weights=wgt[inside])
        import cv2

        evidence = cv2.GaussianBlur(hist, (0, 0), 1.6) * (2 * np.pi * 1.6 ** 2) / total
        cost -= 55.0 * evidence

    def corrected(mean_lin: np.ndarray) -> np.ndarray:
        g = np.stack([2.0 ** (-DR), np.ones_like(DR), 2.0 ** (-DB)], axis=-1)
        g = g / (g @ np.array([0.27411851, 0.87363190, -0.14775041]))[..., None]
        return (mean_lin[None, None, :] * g).astype(F32)

    # 2. skin: its hue direction in linear light is the same for light and dark skin
    if skin_lin is not None and len(skin_lin) >= 30:
        from ..color.gamuts import apply_matrix, conversion_matrix
        from .. import library

        lib = library.get()
        to709 = conversion_matrix(lib.gamut(W.WORKING_GAMUT), lib.gamut("rec709"))
        s709 = np.maximum(apply_matrix(corrected(np.median(skin_lin, axis=0).astype(np.float64)), to709), 1e-5)
        theta = np.degrees(np.arctan2(np.log2(s709[..., 2] / s709[..., 1]), np.log2(s709[..., 0] / s709[..., 1])))
        err = (theta - SKIN_DIRECTION + 180.0) % 360.0 - 180.0
        cost += 0.6 * (np.maximum(np.abs(err) - 10.0, 0.0) / 12.0) ** 2
        info["skin"] = {"samples": int(len(skin_lin))}

    # 3. sky and foliage must keep a natural hue
    for key, mask, rng, weight in (("sky", sky, SKY_HUE, 0.7), ("foliage", fol, FOLIAGE_HUE, 0.35)):
        share = float(mask.mean()) if len(mask) else 0.0
        if share < 0.04:
            continue
        hh, ss, _ = _hsv_hue_sat_val(np.clip(odt(W.di_encode(corrected(np.mean(px[mask], axis=0).astype(np.float64)))), 0, 1))
        lo, hi = rng
        d = np.minimum(np.abs(hh - lo), np.abs(hh - hi))
        pen = np.where((hh >= lo) & (hh <= hi), 0.0, (d / 10.0) ** 2)
        if key == "sky":
            pen = np.where(ss < 0.06, np.maximum(pen, 1.0), pen)  # a blue sky must stay blue
        cost += weight * min(1.0, share / 0.15) * np.minimum(pen, 4.0)
        info[key] = {"share": round(share, 3)}
    idx = np.unravel_index(int(np.argmin(cost)), cost.shape)
    # sub-cell refinement: weighted centroid of the 5x5 neighborhood
    i, j = idx
    sl = (slice(max(0, i - 2), i + 3), slice(max(0, j - 2), j + 3))
    wloc = np.exp(-(cost[sl] - cost[idx]) * 4.0)
    dr_hat = float((DR[sl] * wloc).sum() / wloc.sum())
    db_hat = float((DB[sl] * wloc).sum() / wloc.sum())
    share = float(evidence[idx]) if evidence.size else 0.0
    info["neutral_share"] = round(share, 4)
    conf_neutral = float(np.clip((share - 0.01) / 0.08, 0.0, 1.0))
    conf = conf_neutral
    if info["skin"]:
        conf = max(conf, 0.6)
    if info["sky"] or info["foliage"]:
        conf = max(conf, 0.35)
    t = (dr_hat - db_hat) / np.sqrt(2.0)
    m = (dr_hat + db_hat) / np.sqrt(2.0)
    anchors = []
    if conf_neutral > 0.15:
        anchors.append("neutral objects")
    if info["skin"]:
        anchors.append("skin")
    if info["sky"]:
        anchors.append("sky")
    if info["foliage"]:
        anchors.append("foliage")
    if not anchors:
        notes.append("No neutral object, face, sky or foliage found. The camera's white balance stays.")
    return {"dr": round(dr_hat, 4), "db": round(db_hat, 4), "temperature": round(float(t), 4), "tint": round(float(m), 4),
            "confidence": round(conf, 3), "anchors": anchors, "detail": info, "notes": notes,
            "text": describe_cast(float(t), float(m))}


def describe_cast(t: float, m: float) -> str:
    parts = []
    if abs(t) >= 0.04:
        parts.append(f"{abs(t):.2f} stops {'warm' if t > 0 else 'cool'}")
    if abs(m) >= 0.03:
        parts.append(f"{abs(m):.2f} stops {'magenta' if m > 0 else 'green'}")
    return "neutral" if not parts else ", ".join(parts)
