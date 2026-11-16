"""Step 5 of the grade: keep skin natural while the look pushes the rest of the frame.

A hue-band correction (Hue vs Hue, Hue vs Sat) limited to skin hues. It is a
function of color only, so it lives in the shared look and in the LUT.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis.neutral import SKIN_LINE, vector_angle
from ..color import ops
from . import pipeline

F32 = np.float32
KEEP = 4.0      # degrees: skin within this distance of the line counts as on the line
BAND_IN, BAND_OUT = 22.0, 55.0   # full strength inside, fading out to the outer edge
MAX_ROTATION = 7.0                # degrees: larger hue moves break up skin into patches


def skin_on_display(skin_di: np.ndarray, odt) -> tuple[float, float]:
    """Vectorscope angle (degrees) and saturation (percent of the scope radius) of a skin sample."""
    disp = np.clip(odt(skin_di), 0.0, 1.0)
    ang, mag = vector_angle(np.median(disp, axis=0)[None, :])
    return float(ang[0]), float(mag[0]) * 200.0


def offset_from_line(angle: float) -> float:
    return float((angle - SKIN_LINE + 180.0) % 360.0 - 180.0)


def _band(center: float, value: float, neutral: float) -> list[list[float]]:
    pts = [[center - BAND_OUT, neutral], [center - BAND_IN, value], [center + BAND_IN, value], [center + BAND_OUT, neutral]]
    return [[p[0] % 360.0, p[1]] for p in pts]


def skin_fix(skin_di: np.ndarray, look: dict[str, Any], odt, keep: float = KEEP) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find the skin-band correction for a look.

    skin_di: balanced skin samples (N, 3) in the working space. keep: how far from the line skin
    stays before a correction starts, in degrees. Returns (skin dict, info for the report).
    """
    empty = {"hue_vs_hue": [], "hue_vs_sat": []}
    if skin_di is None or len(skin_di) < 30:
        return empty, {"present": False}
    plain = dict(look, skin=empty)
    a0, s0 = skin_on_display(skin_di, odt)
    graded = pipeline.look_stage(skin_di, plain)
    a1, s1 = skin_on_display(graded, odt)
    target = SKIN_LINE + float(np.clip(offset_from_line(a0), -keep, keep))
    info = {"present": True, "before_offset": round(offset_from_line(a0), 1), "look_offset": round(offset_from_line(a1), 1),
            "before_sat": round(s0, 1), "look_sat": round(s1, 1), "rotation": 0.0, "sat_scale": 1.0}
    _, c1, c2, _, _ = ops.hsl_split(graded)
    center = float(np.degrees(np.arctan2(np.median(c2), np.median(c1))) % 360.0)
    info["band_center"] = round(center, 1)
    rot = 0.0
    if abs(a1 - target) > 1.5:
        def err(r: float) -> float:
            out = ops.hsl_curves(graded, hue_vs_hue=_band(center, r, 0.0))
            return skin_on_display(out, odt)[0] - target

        lo, hi = -MAX_ROTATION, MAX_ROTATION
        e_lo, e_hi = err(lo), err(hi)
        if e_lo * e_hi < 0:
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                e_mid = err(mid)
                if e_lo * e_mid <= 0:
                    hi, e_hi = mid, e_mid
                else:
                    lo, e_lo = mid, e_mid
            rot = 0.5 * (lo + hi)
        else:
            rot = lo if abs(e_lo) < abs(e_hi) else hi
            info["limited"] = True
    out = ops.hsl_curves(graded, hue_vs_hue=_band(center, rot, 0.0)) if rot else graded
    a2, s2 = skin_on_display(out, odt)
    scale = 1.0
    if s2 > 1.25 * s0 and s2 > 20.0:
        scale = max(0.85, 1.25 * s0 / s2)
    elif s2 < 0.7 * s0 and s2 < 7.0 and look.get("saturation", 50.0) > 20.0:
        scale = min(1.12, 0.8 * s0 / max(s2, 1e-3))
    skin = {"hue_vs_hue": _band(center, rot, 0.0) if abs(rot) >= 0.3 else [],
            "hue_vs_sat": _band(center, scale, 1.0) if abs(scale - 1.0) >= 0.03 else []}
    final = pipeline.look_stage(skin_di, dict(look, skin=skin))
    a3, s3 = skin_on_display(final, odt)
    info.update({"rotation": round(rot, 1), "sat_scale": round(scale, 3), "after_offset": round(offset_from_line(a3), 1), "after_sat": round(s3, 1)})
    return skin, info
