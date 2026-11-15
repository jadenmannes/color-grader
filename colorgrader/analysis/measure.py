"""Per-shot measurements, the same for footage and for the reference.

Input is a pooled pixel sample in the working space (DaVinci Wide Gamut /
Intermediate). Display numbers come from the standard output transform, so
footage and reference read on one scale.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..color import working as W
from ..color.ycbcr import rgb_to_ycbcr
from . import names
from .neutral import SKIN_LINE, estimate_cast, vector_angle

F32 = np.float32
DWG_Y = np.array([0.27411851, 0.87363190, -0.14775041], dtype=F32)
ZONES = (("shadows", 0.0, 0.25), ("midtones", 0.25, 0.65), ("highlights", 0.65, 1.01))


def srgb_to_oklab(rgb_lin: np.ndarray) -> np.ndarray:
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]], dtype=F32)
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]], dtype=F32)
    lms = np.cbrt(np.maximum(rgb_lin @ m1.T, 0.0))
    return lms @ m2.T


def palette(display_rgb: np.ndarray, k: int = 6, seed: int = 5) -> list[dict[str, Any]]:
    """Main colors of the frame: k-means in Oklab on display pixels."""
    import cv2

    px = np.clip(display_rgb.reshape(-1, 3), 0, 1).astype(F32)
    if len(px) > 20000:
        px = px[np.random.default_rng(seed).choice(len(px), 20000, replace=False)]
    lab = srgb_to_oklab(np.power(px, F32(2.2))).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    _, labels, _ = cv2.kmeans(lab, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    out = []
    for i in range(k):
        sel = labels == i
        share = float(sel.mean())
        if share < 0.02:
            continue
        col = np.median(px[sel], axis=0)
        out.append({"hex": names.hex_color(col), "name": names.color_name(col), "share": round(share, 3),
                    "rgb": [round(float(c), 4) for c in col]})
    out.sort(key=lambda c: -c["share"])
    return out


def measure(di: np.ndarray, odt, ypos: np.ndarray | None = None, skin_di: np.ndarray | None = None,
            source_clip: dict[str, float] | None = None, with_cast: bool = True) -> dict[str, Any]:
    """All shot measurements from a working-space pixel sample (N, 3)."""
    di = np.asarray(di, dtype=F32).reshape(-1, 3)
    lin = W.di_decode(di)
    lum = np.maximum(lin @ DWG_Y, 0.18 * 2.0 ** -8)  # floor: 8 stops under middle gray
    stops = np.log2(lum / 0.18)
    disp = np.clip(odt(di), 0.0, 1.0)
    yp, cb, cr = rgb_to_ycbcr(disp)
    sat = np.hypot(cb, cr) * 200.0  # percent of the vectorscope radius
    q = lambda a, p: [round(float(v), 4) for v in np.percentile(a, p)]
    s_q = q(stops, [1, 5, 25, 50, 75, 95, 99])
    y_q = q(yp * 100.0, [0.1, 1, 10, 50, 90, 99, 99.9])
    ch_lo = [round(float(np.percentile(disp[:, c], 0.1)) * 100, 2) for c in range(3)]
    ch_hi = [round(float(np.percentile(disp[:, c], 99.9)) * 100, 2) for c in range(3)]
    out: dict[str, Any] = {
        "exposure": {"median_stops": s_q[3], "key_stops": round(float(stops.mean()), 3), "p5_stops": s_q[1], "p95_stops": s_q[5],
                     "midtone_level": y_q[3]},
        "black_point": {"level": y_q[0], "rgb": ch_lo},
        "white_point": {"level": y_q[6], "rgb": ch_hi},
        "clipping": {"highlights": round(float((disp.max(axis=1) >= 0.995).mean()) * 100, 3),
                     "shadows": round(float((disp.max(axis=1) <= 0.004).mean()) * 100, 3)},
        "contrast": {"scene_range_stops": round(s_q[5] - s_q[1], 3), "display_spread": round(y_q[4] - y_q[2], 2),
                     "midtone_level": y_q[3]},
    }
    if source_clip:
        out["clipping"].update({"source_highlights": source_clip.get("high", 0.0), "source_shadows": source_clip.get("low", 0.0)})
    zones = {}
    for name, lo, hi in ZONES:
        sel = (yp >= lo) & (yp < hi)
        zones[name] = {"share": round(float(sel.mean()) * 100, 1),
                       "saturation": round(float(sat[sel].mean()), 2) if sel.any() else 0.0}
    out["saturation"] = {"overall": round(float(sat.mean()), 2), **{k: v["saturation"] for k, v in zones.items()},
                         "zone_share": {k: v["share"] for k, v in zones.items()}}
    out["palette"] = palette(disp)
    if skin_di is not None and len(skin_di) >= 30:
        sdisp = np.clip(odt(np.asarray(skin_di, dtype=F32)), 0, 1)
        ang, mag = vector_angle(np.median(sdisp, axis=0)[None, :])
        sy = float(np.median(rgb_to_ycbcr(sdisp)[0]))
        off = float((ang[0] - SKIN_LINE + 180.0) % 360.0 - 180.0)
        out["skin"] = {"present": True, "angle": round(float(ang[0]), 1), "offset_deg": round(off, 1),
                       "saturation": round(float(mag[0]) * 200.0, 1), "level": round(sy * 100, 1),
                       "text": skin_text(off, float(mag[0]) * 200.0)}
    else:
        out["skin"] = {"present": False, "text": "No faces found."}
    if with_cast:
        skin_lin = W.di_decode(np.asarray(skin_di, dtype=F32)) if (skin_di is not None and len(skin_di) >= 30) else None
        out["white_balance"] = estimate_cast(lin, ypos, odt, skin_lin)
    return out


def skin_text(offset_deg: float, sat_pct: float) -> str:
    if abs(offset_deg) <= 5:
        pos = "on the skin tone line"
    elif offset_deg > 0:
        pos = f"{abs(offset_deg):.0f} degrees toward yellow-green"
    else:
        pos = f"{abs(offset_deg):.0f} degrees toward red-magenta"
    if sat_pct < 8:
        s = "pale"
    elif sat_pct > 34:
        s = "heavy saturation"
    else:
        s = "healthy saturation"
    return f"Skin sits {pos}, {s}."


def summarize(m: dict[str, Any]) -> list[str]:
    """Short readable lines for the analysis panel."""
    e, c, s = m["exposure"], m["contrast"], m["saturation"]
    lines = [
        f"Exposure: midtones at {e['midtone_level']:.0f}%, median {e['median_stops']:+.1f} stops from middle gray.",
        f"Black point {m['black_point']['level']:.1f}%, white point {m['white_point']['level']:.1f}%.",
        f"Contrast: {c['scene_range_stops']:.1f} stops between the 5th and 95th percentile.",
        f"Saturation: shadows {s['shadows']:.0f}%, midtones {s['midtones']:.0f}%, highlights {s['highlights']:.0f}%.",
    ]
    clip = m["clipping"]
    hi = clip.get("source_highlights", clip["highlights"])
    lo = clip.get("source_shadows", clip["shadows"])
    lines.append(f"Clipping: {hi:.2f}% of pixels in the highlights, {lo:.2f}% in the shadows.")
    if "white_balance" in m:
        wb = m["white_balance"]
        anchors = ", ".join(wb["anchors"]) or "none"
        lines.append(f"White balance: {wb['text']}. Anchors: {anchors}.")
    lines.append(m["skin"]["text"])
    return lines
