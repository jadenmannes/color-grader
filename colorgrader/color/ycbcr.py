"""Y'CbCr <-> R'G'B' in float32, with explicit matrix and range handling."""
from __future__ import annotations

import numpy as np

F32 = np.float32

# (Kr, Kb)
KR_KB = {
    "bt709": (0.2126, 0.0722),
    "bt2020nc": (0.2627, 0.0593),
    "bt2020c": (0.2627, 0.0593),
    "bt601": (0.299, 0.114),
    "smpte170m": (0.299, 0.114),
    "bt470bg": (0.299, 0.114),
}


def pick_matrix(tag: str | None, width: int, height: int) -> str:
    """Matrix name from the file tag, else the usual default for the frame size."""
    if tag in KR_KB:
        return tag
    return "bt709" if (width >= 1280 or height > 576) else "bt601"


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray, matrix: str, full_range: bool,
                 bit_depth: int) -> np.ndarray:
    """Planes hold code values in source bit-depth units (float32). Returns (H, W, 3) R'G'B'.

    Video-range input maps 16..235 (8-bit scale) to 0..1. Values above 1 (super
    whites) and below 0 stay in the float result.
    """
    kr, kb = KR_KB[matrix]
    kg = 1.0 - kr - kb
    s = float(1 << (bit_depth - 8))
    if full_range:
        peak = float((1 << bit_depth) - 1)
        yn = y.astype(F32) / F32(peak)
        cbn = (cb.astype(F32) - F32(128.0 * s)) / F32(peak)
        crn = (cr.astype(F32) - F32(128.0 * s)) / F32(peak)
    else:
        yn = (y.astype(F32) - F32(16.0 * s)) / F32(219.0 * s)
        cbn = (cb.astype(F32) - F32(128.0 * s)) / F32(224.0 * s)
        crn = (cr.astype(F32) - F32(128.0 * s)) / F32(224.0 * s)
    r = yn + F32(2.0 * (1.0 - kr)) * crn
    b = yn + F32(2.0 * (1.0 - kb)) * cbn
    g = (yn - F32(kr) * r - F32(kb) * b) / F32(kg)
    return np.stack([r, g, b], axis=-1)


def rgb_to_ycbcr(rgb: np.ndarray, matrix: str = "bt709"):
    """R'G'B' (0..1) to normalized Y' (0..1), Cb, Cr (-0.5..0.5)."""
    kr, kb = KR_KB[matrix]
    kg = 1.0 - kr - kb
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = F32(kr) * r + F32(kg) * g + F32(kb) * b
    cb = (b - y) / F32(2.0 * (1.0 - kb))
    cr = (r - y) / F32(2.0 * (1.0 - kr))
    return y, cb, cr


def signal_to_absolute(v: np.ndarray, full_range: bool, bit_depth: int) -> np.ndarray:
    """Range-normalized signal to full-scale code values (code / max code).

    Published log curves such as F-Log, S-Log3 and V-Log are written against
    absolute code values, whatever range flag the file carries.
    """
    if full_range:
        return v
    s = float(1 << (bit_depth - 8))
    peak = float((1 << bit_depth) - 1)
    return v * F32(219.0 * s / peak) + F32(16.0 * s / peak)


def absolute_to_signal(v: np.ndarray, full_range: bool, bit_depth: int) -> np.ndarray:
    if full_range:
        return v
    s = float(1 << (bit_depth - 8))
    peak = float((1 << bit_depth) - 1)
    return (v - F32(16.0 * s / peak)) / F32(219.0 * s / peak)
