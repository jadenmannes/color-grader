"""The color controls: every adjustment a colorist reaches for, as data and as math.

One control set serves two places:

  look   shared by every shot (model.neutral_look() holds all of it)
  trim   one shot's own adjustments, stored in the shot grade as a partial set

The chain below runs in a fixed order, the order a node tree in DaVinci Resolve would have. Each control
names its Resolve tool in SPEC. Where Resolve's math is known (Lift/Gamma/Gain/Offset, Contrast, Sat, the
HSL curves, the RGB mixer, a Gain on a linear node) the recipe values reproduce the app exactly. Where it is
not published (HDR zones, Log wheels, Hue vs Lum, qualifier softness) the app uses its own math, the recipe
marks the node as set by eye, and the LUT and export carry the exact result.

Units, all chosen so an assistant and a person read them the same way:

  stops      exposure, temp, tint, shadows, highlights, HDR zone exposure: 1 = one camera stop
  degrees    hue rotations and hue positions: red 0, yellow 60, green 120, cyan 180, blue 240, magenta 300
  Resolve    contrast, pivot, lift, gamma, gain, offset, saturation: the numbers Resolve shows
  DI         curve inputs and qualifier lum: DaVinci Intermediate, black 0, middle gray 0.336, scene white about 0.52
  chroma     saturation inputs of curves and qualifiers: HSL chroma of the working space,
             gray 0, skin about 0.03 to 0.06, a clear sky about 0.06 to 0.12, strong colors 0.15 and up
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ..color import ops
from ..color import working as W

F32 = np.float32
PIVOT = round(W.MID_GRAY, 3)
DWG_Y = np.array([0.27411851, 0.87363190, -0.14775041], dtype=F32)   # luminance of DaVinci Wide Gamut
ZONES = ("dark", "shadow", "light", "highlight", "specular")
ZONE_NEUTRAL = {"exposure": 0.0, "saturation": 1.0, "temp": 0.0, "tint": 0.0}
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
LOG_LOW, LOG_HIGH = 0.333, 0.55        # Resolve's default Low Range and High Range of the Log wheels
HUE_SIX = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)


def neutral() -> dict[str, Any]:
    """The controls beyond the classic primaries, at neutral. model.neutral_look() adds these."""
    return {
        "temp": 0.0, "tint": 0.0,
        "hdr": {z: dict(ZONE_NEUTRAL) for z in ZONES},
        "rgb_mixer": copy.deepcopy(IDENTITY),
        "offset": [25.0, 25.0, 25.0],
        "log": {"shadow": [0.0, 0.0, 0.0], "midtone": [0.0, 0.0, 0.0], "highlight": [0.0, 0.0, 0.0],
                "low_range": LOG_LOW, "high_range": LOG_HIGH},
        "shadows": 0.0, "highlights": 0.0,
        "curves_rgb": {"r": [], "g": [], "b": []},
        "hue": 0.0, "color_boost": 0.0,
        "hue_vs_lum": [],
        "secondaries": [],
    }


BASE_NEUTRAL = {"exposure": 0.0, "contrast": 1.0, "pivot": PIVOT, "lift": [0.0, 0.0, 0.0], "gamma": [0.0, 0.0, 0.0],
                "gain": [1.0, 1.0, 1.0], "saturation": 50.0, "hsv_sat": {"gain": 1.0, "gamma": 0.0},
                "tone_curve": [], "lum_vs_sat": [], "hue_vs_sat": [], "hue_vs_hue": [], "sat_vs_lum": [], "sat_vs_sat": []}


def full_neutral() -> dict[str, Any]:
    """Every control at neutral: the classic primaries and curves plus neutral()."""
    return copy.deepcopy(BASE_NEUTRAL) | neutral()


# --------------------------------------------------------------- shapes
def _smooth(a: float, b: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(F32)


def _hann(x: np.ndarray, a: float, b: float) -> np.ndarray:
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return (np.sin(np.pi * t) ** 2).astype(F32)


def zone_weights(stops: np.ndarray) -> dict[str, np.ndarray]:
    """Weight of each HDR zone by exposure in stops around middle gray (0 = 18% gray).

    dark below about -3.5 stops, shadow around -2, light around +0.75 (skin and midtones above gray),
    highlight around +2.5, specular above +4.5. Neighbors overlap so moves blend smoothly.
    """
    return {"dark": (1.0 - _smooth(-4.5, -2.5, stops)).astype(F32),
            "shadow": _hann(stops, -4.0, 0.0),
            "light": _hann(stops, -1.0, 2.5),
            "highlight": (_smooth(1.0, 3.0, stops) * (1.0 - _smooth(4.0, 6.0, stops))).astype(F32),
            "specular": _smooth(3.5, 5.5, stops)}


def wb_gains(exposure: float = 0.0, temp: float = 0.0, tint: float = 0.0) -> list[float]:
    """Gains of a linear node for exposure and white balance, rounded as the recipe prints them.

    temp +1 = one stop warmer (red up half a stop, blue down half a stop). tint +1 = one stop toward magenta (green down).
    """
    e, t, ti = float(exposure or 0.0), float(temp or 0.0), float(tint or 0.0)
    return [round(2.0 ** (e + t / 2.0), 3), round(2.0 ** (e - ti), 3), round(2.0 ** (e - t / 2.0), 3)]


def shadows_highlights_points(shadows: float, highlights: float) -> list[list[float]]:
    """The Shadows and Highlights sliders as a Custom curve (all channels), so Resolve can take the same points.

    shadows: stops at the center of the shadows (DI 0.18), fading to nothing at black and at middle gray (0.336).
    highlights: stops at the center of the highlights (DI 0.57), fading to nothing at middle gray and at DI 0.80.
    Black, middle gray and white stay where they are. Negative highlights recover bright skies and windows,
    positive shadows open dark areas.
    """
    s, h = float(np.clip(shadows, -1.5, 1.5)), float(np.clip(highlights, -1.5, 1.5))
    xs = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, PIVOT, 0.40, 0.46, 0.52, 0.57, 0.62, 0.68, 0.74, 0.80, 1.0]
    out, last = [], -1.0
    for x in xs:
        bs = float(np.sin(np.pi * np.clip((x - 0.02) / (PIVOT - 0.02), 0.0, 1.0)) ** 2)
        bh = float(np.sin(np.pi * np.clip((x - PIVOT) / (0.80 - PIVOT), 0.0, 1.0)) ** 2)
        y = x + W.STOP * (s * bs + h * bh)
        y = max(y, last + 1e-3)                 # the curve may flatten, never fold back
        out.append([round(x, 3), round(y, 4)])
        last = y
    return out


def color_boost_points(boost: float) -> list[list[float]]:
    """Color Boost as a Sat vs Sat curve: gentle colors gain the most, strong colors stay. boost 1 = gentle colors doubled."""
    b = float(np.clip(boost, -1.0, 1.5))
    return [[0.0, round(1.0 + b, 4)], [0.04, round(1.0 + 0.9 * b, 4)], [0.10, round(1.0 + 0.55 * b, 4)],
            [0.18, round(1.0 + 0.2 * b, 4)], [0.28, 1.0], [1.0, 1.0]]


def hue_points(points, hue: float) -> list[list[float]]:
    """Hue vs Hue points with the global Hue rotation added to every point (a flat curve when there are none)."""
    hue = float(hue or 0.0)
    if not hue:
        return points or []
    if not points:
        return [[h, round(hue, 2)] for h in HUE_SIX]
    return [[h, round(v + hue, 2)] for h, v in points]


def _is_identity(m) -> bool:
    return m is None or np.allclose(np.asarray(m, dtype=np.float64), np.eye(3))


def _hdr_active(hdr) -> bool:
    if not hdr:
        return False
    return any(abs(float((hdr.get(z) or {}).get(k, ZONE_NEUTRAL[k])) - ZONE_NEUTRAL[k]) > 1e-9
               for z in ZONES for k in ZONE_NEUTRAL)


def apply_hdr(lin: np.ndarray, hdr: dict[str, Any]) -> np.ndarray:
    """HDR zones on linear light: each zone moves exposure, color and saturation of its own band of brightness."""
    y = lin @ DWG_Y
    stops = np.log2(np.maximum(y, F32(1e-6)) / F32(0.18))
    wts = zone_weights(stops)
    e = np.zeros_like(y)
    t = np.zeros_like(y)
    ti = np.zeros_like(y)
    s = np.ones_like(y)
    for z in ZONES:
        zs = hdr.get(z) or {}
        w = wts[z]
        e += w * F32(float(zs.get("exposure", 0.0)))
        t += w * F32(float(zs.get("temp", 0.0)))
        ti += w * F32(float(zs.get("tint", 0.0)))
        s += w * F32(float(zs.get("saturation", 1.0)) - 1.0)
    gains = np.stack([np.exp2(e + t / 2), np.exp2(e - ti), np.exp2(e - t / 2)], axis=-1).astype(F32)
    out = lin * gains
    if not np.allclose(s, 1.0):
        y2 = (out @ DWG_Y)[..., None]
        out = y2 + (out - y2) * s[..., None]
    return out.astype(F32)


def _log_active(lg) -> bool:
    return bool(lg) and any(abs(float(v)) > 1e-9 for k in ("shadow", "midtone", "highlight") for v in (lg.get(k) or [0, 0, 0]))


def apply_log(di: np.ndarray, lg: dict[str, Any]) -> np.ndarray:
    """Log wheels: Shadow, Midtone and Highlight offsets, each over its own range of brightness (DI)."""
    v = di @ ops.HSL_WEIGHTS
    lo, hi = float(lg.get("low_range", LOG_LOW)), float(lg.get("high_range", LOG_HIGH))
    ws = 1.0 - _smooth(lo - 0.1, lo + 0.1, v)
    wh = _smooth(hi - 0.1, hi + 0.1, v)
    wm = np.clip(1.0 - ws - wh, 0.0, 1.0)
    out = di + ws[..., None] * np.asarray(lg.get("shadow", [0, 0, 0]), dtype=F32) \
        + wm[..., None] * np.asarray(lg.get("midtone", [0, 0, 0]), dtype=F32) \
        + wh[..., None] * np.asarray(lg.get("highlight", [0, 0, 0]), dtype=F32)
    return out.astype(F32)


def secondary_key(di: np.ndarray, sec: dict[str, Any]) -> np.ndarray:
    """The qualifier matte of a secondary: 1 where the correction applies in full, 0 where it leaves the picture alone."""
    lum, _, _, hue, chroma = ops.hsl_split(di)
    k = np.ones_like(lum)
    q = sec.get("hue")
    if q:
        center, width, soft = (float(v) for v in q)
        d = np.abs((hue - F32(center) + F32(180.0)) % F32(360.0) - F32(180.0))
        k = k * (1.0 - _smooth(width / 2.0, width / 2.0 + max(soft, 1e-3), d))
        k = k * _smooth(0.004, 0.015, chroma)          # grays carry no hue to key on
    q = sec.get("sat")
    if q:
        lo, hi, soft = (float(v) for v in q)
        soft = max(soft, 1e-4)
        k = k * _smooth(lo - soft, lo, chroma) * (1.0 - _smooth(hi, hi + soft, chroma))
    q = sec.get("lum")
    if q:
        lo, hi, soft = (float(v) for v in q)
        soft = max(soft, 1e-4)
        k = k * _smooth(lo - soft, lo, lum) * (1.0 - _smooth(hi, hi + soft, lum))
    if sec.get("invert"):
        k = 1.0 - k
    return np.clip(k, 0.0, 1.0).astype(F32)


SEC_ADJUST_NEUTRAL = {"exposure": 0.0, "temp": 0.0, "tint": 0.0, "saturation": 1.0, "hue": 0.0}


def apply_secondary(di: np.ndarray, sec: dict[str, Any]) -> np.ndarray:
    """One qualifier secondary: key the picture, correct inside the key, blend by the key."""
    if not secondary_active(sec):
        return di
    adj = {**SEC_ADJUST_NEUTRAL, **(sec.get("adjust") or {})}
    k = secondary_key(di, sec)
    if float(k.max()) <= 0.0:
        return di
    y = di
    gains = wb_gains(adj["exposure"], adj["temp"], adj["tint"])
    if gains != [1.0, 1.0, 1.0]:
        y = W.di_encode(W.di_decode(y) * np.asarray(gains, dtype=F32))
    if abs(float(adj["saturation"]) - 1.0) > 1e-9:
        y = ops.saturation(y, 50.0 * float(adj["saturation"]))
    if abs(float(adj["hue"])) > 1e-9:
        y = ops.hsl_curves(y, hue_vs_hue=[[0.0, float(adj["hue"])]])
    return (di + k[..., None] * (y - di)).astype(F32)


# ---------------------------------------------------------------- chain
def apply(di: np.ndarray, c: dict[str, Any]) -> np.ndarray:
    """Every control in node order, on DaVinci Intermediate values. Missing keys count as neutral."""
    g = c.get
    gains = wb_gains(g("exposure", 0.0), g("temp", 0.0), g("tint", 0.0))
    hdr = g("hdr")
    if gains != [1.0, 1.0, 1.0] or _hdr_active(hdr):
        lin = W.di_decode(di)
        if gains != [1.0, 1.0, 1.0]:   # a node in linear gamma: exposure and white balance the way a camera sees them
            lin = lin * np.asarray(gains, dtype=F32)
        if _hdr_active(hdr):
            lin = apply_hdr(lin, hdr)
        di = W.di_encode(lin)
    if not _is_identity(g("rgb_mixer")):
        di = ops.rgb_mixer(di, g("rgb_mixer"))
    di = ops.contrast(di, float(g("contrast", 1.0)), float(g("pivot", PIVOT)))
    lift, gamma, gain, offset = g("lift", [0.0] * 3), g("gamma", [0.0] * 3), g("gain", [1.0] * 3), g("offset", [25.0] * 3)
    if any(abs(v) > 1e-12 for v in lift) or any(abs(v) > 1e-12 for v in gamma) or any(abs(v - 1.0) > 1e-12 for v in gain) \
            or any(abs(v - 25.0) > 1e-12 for v in offset):
        di = ops.lggo(di, lift, gamma, gain, offset)
    if _log_active(g("log")):
        di = apply_log(di, g("log"))
    sh, hl = float(g("shadows", 0.0) or 0.0), float(g("highlights", 0.0) or 0.0)
    if sh or hl:
        di = ops.tone_curve(di, shadows_highlights_points(sh, hl))
    di = ops.tone_curve(di, g("tone_curve"))
    crv = g("curves_rgb") or {}
    di = ops.channel_curves(di, crv.get("r"), crv.get("g"), crv.get("b"))
    di = ops.saturation(di, float(g("saturation", 50.0)))
    di = ops.hsl_curves(di, lum_vs_sat=g("lum_vs_sat"))
    cb = float(g("color_boost", 0.0) or 0.0)
    if cb:
        di = ops.hsl_curves(di, sat_vs_sat=color_boost_points(cb))
    hs = g("hsv_sat") or {}
    di = ops.hsv_saturation(di, float(hs.get("gain", 1.0)), float(hs.get("gamma", 0.0)))
    di = ops.hsl_curves(di, hue_vs_sat=g("hue_vs_sat"), hue_vs_hue=hue_points(g("hue_vs_hue"), g("hue", 0.0)))
    di = ops.hsl_curves(di, hue_vs_lum=g("hue_vs_lum"))
    skin = g("skin") or {}
    di = ops.hsl_curves(di, hue_vs_sat=skin.get("hue_vs_sat"), hue_vs_hue=skin.get("hue_vs_hue"))
    di = ops.hsl_curves(di, sat_vs_lum=g("sat_vs_lum"))
    di = ops.hsl_curves(di, sat_vs_sat=g("sat_vs_sat"))
    for sec in g("secondaries") or []:
        di = apply_secondary(di, sec)
    return di


# --------------------------------------------------- strength, rounding, neutral
def _pts(points, kin: int, kval: int):
    return [[round(float(a), kin), round(float(b), kval)] for a, b in (points or [])]


def quantize(c: dict[str, Any]) -> dict[str, Any]:
    """Round the controls of neutral() the way the recipe prints them. Only keys present are touched."""
    k = copy.deepcopy(c)
    for key, d in (("temp", 2), ("tint", 2), ("hue", 1), ("color_boost", 2), ("shadows", 2), ("highlights", 2)):
        if key in k:
            k[key] = round(float(k[key] or 0.0), d)
    if "offset" in k:
        k["offset"] = [round(float(v), 2) for v in k["offset"]]
    if "hdr" in k:
        k["hdr"] = {z: {f: round(float((k["hdr"].get(z) or {}).get(f, ZONE_NEUTRAL[f])), 2) for f in ZONE_NEUTRAL} for z in ZONES}
    if "rgb_mixer" in k:
        k["rgb_mixer"] = [[round(float(v), 3) for v in row] for row in (k["rgb_mixer"] or IDENTITY)]
    if "log" in k:
        lg = k["log"] or {}
        k["log"] = {w: [round(float(v), 3) for v in (lg.get(w) or [0.0, 0.0, 0.0])] for w in ("shadow", "midtone", "highlight")}
        k["log"]["low_range"] = round(float(lg.get("low_range", LOG_LOW)), 3)
        k["log"]["high_range"] = round(float(lg.get("high_range", LOG_HIGH)), 3)
    if "curves_rgb" in k:
        k["curves_rgb"] = {ch: _pts((k["curves_rgb"] or {}).get(ch), 3, 3) for ch in ("r", "g", "b")}
    if "hue_vs_lum" in k:
        k["hue_vs_lum"] = _pts(k["hue_vs_lum"], 1, 3)
    if "secondaries" in k:
        k["secondaries"] = [quantize_secondary(s) for s in k["secondaries"] or []]
    return k


def quantize_secondary(sec: dict[str, Any]) -> dict[str, Any]:
    out = {"name": str(sec.get("name") or "Secondary"), "enabled": bool(sec.get("enabled", True)), "invert": bool(sec.get("invert", False))}
    for q, d in (("hue", 1), ("sat", 3), ("lum", 3)):
        if sec.get(q):
            out[q] = [round(float(v), d) for v in sec[q]]
    adj = {**SEC_ADJUST_NEUTRAL, **(sec.get("adjust") or {})}
    out["adjust"] = {k: round(float(adj[k]), 2 if k != "hue" else 1) for k in SEC_ADJUST_NEUTRAL}
    return out


def scale(c: dict[str, Any], a: float) -> dict[str, Any]:
    """The controls of neutral() moved toward neutral: a = 1 keeps them, 0 removes them."""
    k = copy.deepcopy(c)
    for key in ("temp", "tint", "hue", "color_boost", "shadows", "highlights"):
        if key in k:
            k[key] = a * float(k[key] or 0.0)
    if "offset" in k:
        k["offset"] = [25.0 + a * (float(v) - 25.0) for v in k["offset"]]
    if "hdr" in k:
        k["hdr"] = {z: {"exposure": a * float((k["hdr"].get(z) or {}).get("exposure", 0.0)),
                        "temp": a * float((k["hdr"].get(z) or {}).get("temp", 0.0)),
                        "tint": a * float((k["hdr"].get(z) or {}).get("tint", 0.0)),
                        "saturation": 1.0 + a * (float((k["hdr"].get(z) or {}).get("saturation", 1.0)) - 1.0)} for z in ZONES}
    if "rgb_mixer" in k and k["rgb_mixer"] is not None:
        m = np.asarray(k["rgb_mixer"], dtype=np.float64)
        k["rgb_mixer"] = (np.eye(3) + a * (m - np.eye(3))).tolist()
    if "log" in k and k["log"]:
        for w in ("shadow", "midtone", "highlight"):
            k["log"][w] = [a * float(v) for v in (k["log"].get(w) or [0.0, 0.0, 0.0])]
    if "curves_rgb" in k:
        k["curves_rgb"] = {ch: [[x, x + a * (v - x)] for x, v in ((k["curves_rgb"] or {}).get(ch) or [])] if a > 0 else []
                           for ch in ("r", "g", "b")}
    if "hue_vs_lum" in k:
        k["hue_vs_lum"] = [[x, 1.0 + a * (v - 1.0)] for x, v in k["hue_vs_lum"] or []]
    if "secondaries" in k:
        out = []
        for s in k["secondaries"] or []:
            s = copy.deepcopy(s)
            adj = {**SEC_ADJUST_NEUTRAL, **(s.get("adjust") or {})}
            s["adjust"] = {"exposure": a * adj["exposure"], "temp": a * adj["temp"], "tint": a * adj["tint"],
                           "hue": a * adj["hue"], "saturation": 1.0 + a * (adj["saturation"] - 1.0)}
            out.append(s)
        k["secondaries"] = out
    return k


def is_neutral(c: dict[str, Any]) -> bool:
    """True when none of the controls of neutral() changes the picture."""
    if any(abs(float(c.get(k, 0.0) or 0.0)) > 1e-9 for k in ("temp", "tint", "hue", "color_boost", "shadows", "highlights")):
        return False
    if any(abs(float(v) - 25.0) > 1e-9 for v in c.get("offset", [25.0] * 3)):
        return False
    if _hdr_active(c.get("hdr")) or _log_active(c.get("log")) or not _is_identity(c.get("rgb_mixer")):
        return False
    if any(abs(v - x) > 1e-9 for ch in ("r", "g", "b") for x, v in ((c.get("curves_rgb") or {}).get(ch) or [])):
        return False
    if any(abs(v - 1.0) > 1e-9 for _, v in c.get("hue_vs_lum") or []):
        return False
    return not any(secondary_active(s) for s in c.get("secondaries") or [])


def secondary_active(sec: dict[str, Any]) -> bool:
    """A secondary changes the picture: switched on, with a correction that does something."""
    if not sec.get("enabled", True):
        return False
    adj = {**SEC_ADJUST_NEUTRAL, **(sec.get("adjust") or {})}
    return any(abs(float(adj[k]) - SEC_ADJUST_NEUTRAL[k]) > 1e-9 for k in SEC_ADJUST_NEUTRAL)


def merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial change to a control set. Nested groups (hdr zones, log wheels, rgb curves, hsv) merge field by field."""
    out = copy.deepcopy(base)
    for key, val in changes.items():
        if key in ("hdr",) and isinstance(val, dict):
            cur = out.get("hdr") or {z: dict(ZONE_NEUTRAL) for z in ZONES}
            for z, fields in val.items():
                if z in ZONES and isinstance(fields, dict):
                    cur[z] = {**ZONE_NEUTRAL, **(cur.get(z) or {}), **{f: float(v) for f, v in fields.items() if f in ZONE_NEUTRAL}}
            out["hdr"] = cur
        elif key in ("log", "curves_rgb", "hsv_sat") and isinstance(val, dict):
            out[key] = {**(out.get(key) or (full_neutral()[key])), **val}
        else:
            out[key] = copy.deepcopy(val)
    return out


# ------------------------------------------------------------------ spec
def _n(key, label, group, neutral_value, lo, hi, step, unit, help_text, resolve, exact=True, kind="number"):
    return {"key": key, "label": label, "group": group, "type": kind, "neutral": neutral_value, "min": lo, "max": hi,
            "step": step, "unit": unit, "help": help_text, "resolve": resolve, "exact": exact}


SPEC: list[dict[str, Any]] = [
    _n("exposure", "Exposure", "Exposure and white balance", 0.0, -3.0, 3.0, 0.05, "stops",
       "Brighter or darker on linear light, the way the camera's exposure works. Black stays black.",
       "Gain on a node with Gamma set to Linear (like HDR Global exposure)"),
    _n("temp", "Temperature", "Exposure and white balance", 0.0, -1.5, 1.5, 0.02, "stops",
       "Warmer (+) or cooler (-) on linear light: red and blue move half a stop each per stop.", "Gain R and B on the same linear node"),
    _n("tint", "Tint", "Exposure and white balance", 0.0, -1.0, 1.0, 0.02, "stops",
       "Toward magenta (+) or green (-) on linear light.", "Gain G on the same linear node"),
    {**_n("hdr", "HDR zones", "HDR zones", None, None, None, None, "",
          "Five zones by brightness: dark (under -3.5 stops), shadow (about -2), light (about +0.75, skin and upper mids), "
          "highlight (about +2.5), specular (+4.5 and up). Each has exposure (stops), saturation (1 = unchanged), temp and tint (stops).",
          "HDR palette zone wheels (Exp, Sat, color). Set by eye: Resolve's zone math is unpublished", exact=False, kind="zones"),
     "fields": {"exposure": [-2.0, 2.0, 0.05], "saturation": [0.0, 2.0, 0.02], "temp": [-1.0, 1.0, 0.02], "tint": [-1.0, 1.0, 0.02]},
     "zones": list(ZONES)},
    {**_n("rgb_mixer", "RGB mixer", "RGB mixer", IDENTITY, -1.0, 2.0, 0.01, "",
          "Each output channel as a mix of the input channels. Rows: red, green, blue output. Identity = unchanged.",
          "RGB Mixer palette, Preserve Luminance off", kind="mixer")},
    _n("contrast", "Contrast", "Primaries", 1.0, 0.5, 2.0, 0.005, "", "Spreads tones away from the pivot.", "Primaries, Contrast (S-curve off)"),
    _n("pivot", "Pivot", "Primaries", PIVOT, 0.1, 0.7, 0.005, "DI", "The tone contrast turns around. 0.336 is middle gray.", "Primaries, Pivot"),
    _n("lift", "Lift", "Primaries", [0.0, 0.0, 0.0], -0.2, 0.2, 0.002, "", "Shadow level and shadow color, per channel R G B.",
       "Primaries, Lift wheel (Lum Mix 0)", kind="rgb"),
    _n("gamma", "Gamma", "Primaries", [0.0, 0.0, 0.0], -0.2, 0.2, 0.002, "", "Midtone level and color, per channel.",
       "Primaries, Gamma wheel", kind="rgb"),
    _n("gain", "Gain", "Primaries", [1.0, 1.0, 1.0], 0.5, 1.8, 0.005, "", "Highlight level and color, per channel.",
       "Primaries, Gain wheel", kind="rgb"),
    _n("offset", "Offset", "Primaries", [25.0, 25.0, 25.0], 15.0, 35.0, 0.05, "points",
       "Moves every tone equally, per channel. 10 points = one stop in DaVinci Intermediate.", "Primaries, Offset wheel", kind="rgb"),
    _n("saturation", "Saturation", "Primaries", 50.0, 0.0, 100.0, 0.5, "", "50 = unchanged, 0 = black and white.", "Primaries, Sat"),
    _n("hue", "Hue", "Primaries", 0.0, -180.0, 180.0, 1.0, "degrees", "Turns every color around the color wheel.",
       "Hue vs Hue with every point at the same rotation (Primaries Hue does the same by eye)"),
    _n("color_boost", "Color boost", "Primaries", 0.0, -1.0, 1.5, 0.02, "",
       "Saturates gentle colors and leaves strong ones alone (vibrance). 1 = gentle colors doubled.",
       "Sat vs Sat curve (the app's exact form of Color Boost)"),
    _n("shadows", "Shadows", "Primaries", 0.0, -1.5, 1.5, 0.02, "stops",
       "Opens (+) or deepens (-) the shadows without moving black or mid gray.", "Custom curve, all channels (the app's exact form of the Shadows slider)"),
    _n("highlights", "Highlights", "Primaries", 0.0, -1.5, 1.5, 0.02, "stops",
       "Pulls down (-) or lifts (+) bright areas such as skies and windows without moving mid gray.",
       "Custom curve, all channels (the app's exact form of the Highlights slider)"),
    {**_n("log", "Log wheels", "Log wheels", None, -0.1, 0.1, 0.001, "DI",
          "Shadow, Midtone and Highlight offsets, each working only inside its range of brightness; Low Range and High Range set the borders.",
          "Primaries, Log mode. Set by eye: the range falloff is the app's own", exact=False, kind="log"),
     "ranges": {"low_range": [0.05, 0.6, 0.005], "high_range": [0.3, 0.9, 0.005]}},
    _n("hsv_sat", "HSV saturation", "Saturation", {"gain": 1.0, "gamma": 0.0}, None, None, None, "",
       "A node in HSV with channels 1 and 3 off: Gain lifts strong colors, Gamma lifts gentle colors; colors get denser as they saturate.",
       "Node Color Space HSV, Channels 1 and 3 off, Primaries Gain and Gamma", kind="hsv"),
    _n("tone_curve", "Custom curve", "Curves", [], 0.0, 1.0, None, "DI",
       "One curve on all channels: input to output in DaVinci Intermediate.", "Curves, Custom, ganged", kind="curve"),
    {**_n("curves_rgb", "RGB curves", "Curves", {"r": [], "g": [], "b": []}, 0.0, 1.0, None, "DI",
          "A curve per channel (gang off): bend red, green or blue on its own.", "Curves, Custom, gang off", kind="rgb_curves")},
    _n("lum_vs_sat", "Lum vs Sat", "Curves", [], 0.0, 1.0, None, "multiplier", "Saturation by brightness.", "Curves, Lum vs Sat", kind="curve"),
    _n("hue_vs_hue", "Hue vs Hue", "Curves", [], 0.0, 360.0, None, "degrees", "Turns one color family toward another.",
       "Curves, Hue vs Hue", kind="hue_curve"),
    _n("hue_vs_sat", "Hue vs Sat", "Curves", [], 0.0, 360.0, None, "multiplier", "Saturation of one color family.",
       "Curves, Hue vs Sat", kind="hue_curve"),
    _n("hue_vs_lum", "Hue vs Lum", "Curves", [], 0.0, 360.0, None, "multiplier", "Brightness of one color family; grays stay.",
       "Curves, Hue vs Lum. Set by eye: Resolve's form differs slightly", exact=False, kind="hue_curve"),
    _n("sat_vs_sat", "Sat vs Sat", "Curves", [], 0.0, 0.6, None, "multiplier", "Saturation by saturation: limit strong colors, lift gentle ones.",
       "Curves, Sat vs Sat", kind="curve"),
    _n("sat_vs_lum", "Sat vs Lum", "Curves", [], 0.0, 0.6, None, "multiplier", "Brightness by saturation: denser strong colors.",
       "Curves, Sat vs Lum. Resolve's form is unpublished, check by eye", exact=False, kind="curve"),
    _n("secondaries", "Secondaries", "Secondaries", [], None, None, None, "",
       "Qualifier corrections: key a hue, saturation and brightness range (for example the sky or the skin), then change exposure, "
       "temp, tint, saturation or hue only there. Hue in degrees, sat in chroma (skin about 0.03 to 0.06), lum in DI.",
       "Qualifier (HSL) in its own node. Ranges set by eye with Highlight mode on", exact=False, kind="secondaries"),
]

FINISH_SPEC: list[dict[str, Any]] = [
    _n("relight", "Relight", "Finish", None, 0.0, 1.5, 0.05, "stops", "Background darker around tracked faces and shoulders.",
       "Windows, Tracker and an Outside node", exact=False, kind="finish"),
    _n("vignette", "Vignette", "Finish", None, 0.0, 0.8, 0.05, "stops", "Corners darker.", "Window, Invert, Offset", exact=False, kind="finish"),
    _n("grain", "Grain", "Finish", None, 0.0, 1.2, 0.05, "%", "Fine film grain.", "Film Grain (Studio)", exact=False, kind="finish"),
    _n("denoise", "Noise reduction", "Finish", 0, 0, 100, 5, "", "Cleans camera noise before the grade.", "Motion Effects NR (Studio)",
       exact=False, kind="finish"),
    _n("sharpen", "Sharpen", "Finish", 0, 0, 100, 5, "", "Fine edge contrast.", "Blur palette, Radius under 0.50", exact=False, kind="finish"),
    _n("clarity", "Mid/Detail (clarity)", "Finish", 0, 0, 100, 5, "", "Wide local contrast for faces and textures.",
       "Primaries, Midtone Detail (MD)", exact=False, kind="finish"),
]
