"""Step 2 of the grade: balance one shot so it starts neutral.

Works on linear light: white balance and exposure are gains, black point is
an offset. In Resolve this is one node with its gamma set to Linear.

Rules follow a colorist's caution. The camera's own choices stay unless the
picture gives clear evidence against them, and every correction is capped.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis.neutral import gains_from_cast
from ..color import ops
from . import model

F32 = np.float32
DWG_Y = np.array([0.27411851, 0.87363190, -0.14775041], dtype=F32)

MID_LOW, MID_HIGH = -2.0, 0.5        # normal range of the median, stops from middle gray
BLACK_TARGET = 0.0004                # linear light, about 4% on the display
BLACK_TRIGGER = 0.0012               # about 6% on the display


def _stops(lum: np.ndarray) -> np.ndarray:
    return np.log2(np.maximum(lum, 0.18 * 2.0 ** -8) / 0.18)


def exposure_decision(lin: np.ndarray, skin_lin: np.ndarray | None, finished_picture: bool) -> tuple[float, list[str]]:
    """Exposure change in stops for a white-balanced linear sample, with reasons."""
    lum = lin @ DWG_Y
    med = float(np.median(_stops(lum)))
    p995 = float(np.percentile(lum, 99.5))
    why: list[str] = []
    e = 0.0
    if med > MID_HIGH:
        e = -0.7 * (med - MID_HIGH)
        why.append(f"midtones sat {med - MID_HIGH:.1f} stops above the normal range")
    elif med < MID_LOW:
        e = 0.7 * (MID_LOW - med)
        if p995 > 1.0:
            e *= 0.4
            why.append("dark frame with bright highlights reads as low-key on purpose, so the lift stays small")
        else:
            why.append(f"midtones sat {MID_LOW - med:.1f} stops below the normal range")
    if e >= 0.0 and p995 < 0.45:
        e_wp = min(1.0, 0.6 * float(np.log2(0.7 / max(p995, 1e-4))))
        e_wp = max(0.0, min(e_wp, 0.2 - med))
        if e_wp > e + 0.05:
            e = e_wp
            why.append("brightest tones sat far below white")
    if skin_lin is not None and len(skin_lin) >= 30:
        ls = float(np.median(skin_lin @ DWG_Y)) * 2.0 ** e
        if ls > 0.55:
            d = -0.7 * float(np.log2(ls / 0.55))
            e += d
            why.append("skin sat too bright")
        elif ls < 0.07:
            d = 0.7 * float(np.log2(0.07 / max(ls, 1e-4)))
            e += d
            why.append("skin sat too dark")
    if finished_picture:
        e *= 0.5  # the camera finished this picture: half-strength corrections
    if e > 0.0 and p995 * 2.0 ** e > 8.0:
        e = max(0.0, float(np.log2(8.0 / p995)))
        why.append("lift limited to protect the highlights")
    e = float(np.clip(e, -2.5, 2.0))
    if abs(e) < 0.05:
        return 0.0, ["exposure sits in the normal range"]
    return e, why


def black_decision(lin: np.ndarray, finished_picture: bool) -> tuple[np.ndarray, list[str]]:
    """Linear offset per channel that sets and neutralizes the black point."""
    lum = lin @ DWG_Y
    n = max(20, int(len(lum) * 0.005))
    dark = lin[np.argpartition(lum, n)[:n]]
    b_rgb = dark.mean(axis=0).astype(np.float64)
    b_lum = float(b_rgb @ DWG_Y.astype(np.float64))
    why: list[str] = []
    sub = 0.0
    trigger = 0.004 if finished_picture else BLACK_TRIGGER
    if b_lum > trigger:
        k = 0.5 if (finished_picture or b_lum > 0.03) else 0.8
        sub = min(k * (b_lum - BLACK_TARGET), 0.012)
        why.append(f"blacks sat lifted at {b_lum * 100:.2f}% of scene white" + (", haze kept in part" if b_lum > 0.03 else ""))
    imbalance = np.clip(b_rgb - b_lum, -0.004, 0.004)
    tint = np.zeros(3)
    if float(np.max(np.abs(imbalance))) > 0.0003 and not finished_picture:
        tint = -0.8 * imbalance
        why.append("darkest tones carried a color cast")
    off = tint - sub
    if not why:
        why.append("black point sits where it belongs")
    return off, why


def balance_shot(sample_lin: np.ndarray, skin_lin: np.ndarray | None, wb: dict[str, Any], finished_picture: bool,
                 cast: tuple[float, float] | None = None, exposure_trim: float = 0.0) -> dict[str, Any]:
    """Build the balance part of a shot grade.

    wb: white balance estimate from the analysis. cast: override (dr, db) from the scene consensus.
    """
    conf = float(wb.get("confidence", 0.0))
    dr, db = cast if cast is not None else (float(wb.get("dr", 0.0)), float(wb.get("db", 0.0)))
    k = 0.6 + 0.35 * conf
    if finished_picture:
        k *= 0.8
    g_wb = gains_from_cast(k * dr, k * db).astype(np.float64)
    lin1 = sample_lin * g_wb.astype(F32)
    skin1 = skin_lin * g_wb.astype(F32) if (skin_lin is not None and len(skin_lin)) else None
    e, why_e = exposure_decision(lin1, skin1, finished_picture)
    e += exposure_trim
    gain = g_wb * 2.0 ** e
    off, why_b = black_decision(sample_lin * gain.astype(F32), finished_picture)
    t = (dr - db) / np.sqrt(2.0) * k
    m = (dr + db) / np.sqrt(2.0) * k
    parts = []
    if abs(t) >= 0.03:
        parts.append(f"{abs(t):.2f} stops {'cooler' if t > 0 else 'warmer'}")
    if abs(m) >= 0.03:
        parts.append(f"{abs(m):.2f} stops toward {'green' if m > 0 else 'magenta'}")
    anchors = ", ".join(wb.get("anchors") or []) or "no anchor found, camera balance kept"
    why_wb = (f"White balance moved {' and '.join(parts)}. Anchors: {anchors}." if parts
              else f"White balance stays. Anchors: {anchors}.")
    grade = model.neutral_shot()
    grade["balance"]["gain"] = [float(v) for v in gain]
    grade["balance"]["offset"] = [float(ops.OFFSET_BASE + v / ops.OFFSET_UNIT) for v in off]
    grade["reasons"] = {
        "gain": f"Exposure {e:+.2f} stops: {'; '.join(why_e)}. {why_wb}",
        "offset": f"Black point: {'; '.join(why_b)}.",
    }
    grade["info"] = {"exposure": round(e, 3), "cast_applied": [round(k * dr, 4), round(k * db, 4)], "wb_strength": round(k, 3)}
    return model.quantize_shot(grade) | {"reasons": grade["reasons"], "info": grade["info"]}
