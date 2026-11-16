"""The grade as data: numbers a colorist types into DaVinci Resolve.

Every value is stored in Resolve's own units and rounded to the precision the
recipe prints. The app renders with these rounded numbers, so preview, LUT,
export and recipe agree.
"""
from __future__ import annotations

import copy
from typing import Any

from ..color import working as W

PIVOT = round(W.MID_GRAY, 3)  # 0.336, middle gray in DaVinci Intermediate

# decimals printed in the recipe
PRECISION = {"gain": 3, "lift": 3, "gamma": 3, "offset": 2, "contrast": 3, "pivot": 3, "saturation": 1,
             "curve_in": 1, "curve_value": 3, "lum_in": 3}


def r(value: float, kind: str) -> float:
    return round(float(value), PRECISION[kind])


def neutral_shot() -> dict[str, Any]:
    """Per-shot correction: balance in linear light, then match trims in log."""
    return {
        "balance": {"gain": [1.0, 1.0, 1.0], "offset": [25.0, 25.0, 25.0]},
        "match": {"contrast": 1.0, "pivot": PIVOT, "saturation": 50.0, "hue_vs_hue": [], "hue_vs_sat": []},
        "reasons": {},
    }


def neutral_look() -> dict[str, Any]:
    """The shared look. Neutral values change nothing."""
    return {
        "name": "None",
        "description": "",
        "source": "none",
        "contrast": 1.0,
        "pivot": PIVOT,
        "lift": [0.0, 0.0, 0.0],
        "gamma": [0.0, 0.0, 0.0],
        "gain": [1.0, 1.0, 1.0],
        "saturation": 50.0,
        "lum_vs_sat": [],   # [[input luma 0..1, saturation multiplier], ...]
        "hue_vs_sat": [],   # [[hue degrees, saturation multiplier], ...]
        "hue_vs_hue": [],   # [[hue degrees, rotation in degrees], ...]
        "skin": {"hue_vs_hue": [], "hue_vs_sat": []},
        "sat_vs_sat": [],   # [[input saturation 0..1, saturation multiplier], ...] limits extreme colors
        "tone_curve": [],   # [[input, output], ...] custom curve on all channels: the film-style contrast curve
        "sat_vs_lum": [],   # [[input saturation 0..1, luminance multiplier], ...] color density
        "reasons": {},
    }


def neutral_finish() -> dict[str, Any]:
    return {
        "vignette": {"enabled": False, "amount": 0.25, "start": 0.45},  # amount: stops darker in the corners
        "grain": {"enabled": False, "amount": 0.35, "size": 1.0},     # amount: percent of signal
        "detail": {"denoise": 0, "sharpen": 0, "clarity": 0},          # each 0..100, 0 = off
    }


def default_detail(bit_depth: int, log_source: bool) -> dict[str, int]:
    """Starting values for a finished picture. Log footage carries more noise, 8-bit log the most."""
    denoise = (35 if bit_depth <= 8 else 20) if log_source else 0
    return {"denoise": denoise, "sharpen": 30, "clarity": 25}


def quantize_shot(shot: dict[str, Any]) -> dict[str, Any]:
    s = copy.deepcopy(shot)
    s["balance"]["gain"] = [r(v, "gain") for v in s["balance"]["gain"]]
    s["balance"]["offset"] = [r(v, "offset") for v in s["balance"]["offset"]]
    s["match"]["contrast"] = r(s["match"]["contrast"], "contrast")
    s["match"]["pivot"] = r(s["match"].get("pivot", PIVOT), "pivot")
    s["match"]["saturation"] = r(s["match"]["saturation"], "saturation")
    s["match"]["hue_vs_hue"] = [[r(a, "curve_in"), r(b, "curve_in")] for a, b in (s["match"].get("hue_vs_hue") or [])]
    s["match"]["hue_vs_sat"] = [[r(a, "curve_in"), r(b, "curve_value")] for a, b in (s["match"].get("hue_vs_sat") or [])]
    return s


def _q_points(points, kin: str, kval: str = "curve_value"):
    return [[r(a, kin), r(b, kval)] for a, b in (points or [])]


def quantize_look(look: dict[str, Any]) -> dict[str, Any]:
    k = copy.deepcopy(look)
    k["contrast"] = r(k["contrast"], "contrast")
    k["pivot"] = r(k.get("pivot", PIVOT), "pivot")
    for key in ("lift", "gamma", "gain"):
        k[key] = [r(v, key) for v in k[key]]
    k["saturation"] = r(k["saturation"], "saturation")
    k["lum_vs_sat"] = _q_points(k.get("lum_vs_sat"), "lum_in")
    k["hue_vs_sat"] = _q_points(k.get("hue_vs_sat"), "curve_in")
    k["hue_vs_hue"] = _q_points(k.get("hue_vs_hue"), "curve_in", "curve_in")
    k["sat_vs_sat"] = _q_points(k.get("sat_vs_sat"), "lum_in")
    k["tone_curve"] = _q_points(k.get("tone_curve"), "lum_in")
    k["sat_vs_lum"] = _q_points(k.get("sat_vs_lum"), "lum_in")
    skin = k.get("skin") or {}
    k["skin"] = {"hue_vs_hue": _q_points(skin.get("hue_vs_hue"), "curve_in", "curve_in"),
                 "hue_vs_sat": _q_points(skin.get("hue_vs_sat"), "curve_in")}
    return k


def at_strength(look: dict[str, Any], strength: float) -> dict[str, Any]:
    """Blend a look toward neutral. strength 1.0 keeps the full look, 0.0 removes it."""
    a = float(min(1.0, max(0.0, strength)))
    k = copy.deepcopy(look)
    k["contrast"] = 1.0 + a * (look["contrast"] - 1.0)
    k["lift"] = [a * v for v in look["lift"]]
    k["gamma"] = [a * v for v in look["gamma"]]
    k["gain"] = [1.0 + a * (v - 1.0) for v in look["gain"]]
    k["saturation"] = 50.0 + a * (look["saturation"] - 50.0)
    k["lum_vs_sat"] = [[x, 1.0 + a * (v - 1.0)] for x, v in look.get("lum_vs_sat", [])]
    k["hue_vs_sat"] = [[x, 1.0 + a * (v - 1.0)] for x, v in look.get("hue_vs_sat", [])]
    k["hue_vs_hue"] = [[x, a * v] for x, v in look.get("hue_vs_hue", [])]
    k["sat_vs_sat"] = [[x, 1.0 + a * (v - 1.0)] for x, v in look.get("sat_vs_sat", [])] if a > 0 else []
    k["tone_curve"] = [[x, x + a * (v - x)] for x, v in look.get("tone_curve") or []] if a > 0 else []
    k["sat_vs_lum"] = [[x, 1.0 + a * (v - 1.0)] for x, v in look.get("sat_vs_lum") or []] if a > 0 else []
    skin = look.get("skin") or {}
    k["skin"] = {"hue_vs_hue": [[x, a * v] for x, v in skin.get("hue_vs_hue", [])],
                 "hue_vs_sat": [[x, 1.0 + a * (v - 1.0)] for x, v in skin.get("hue_vs_sat", [])]}
    return quantize_look(k)


def is_neutral_look(look: dict[str, Any]) -> bool:
    n = neutral_look()
    keys = ("contrast", "lift", "gamma", "gain", "saturation")
    flat = all(look.get(k) == n[k] for k in keys)
    curves = (all(abs(v - 1.0) < 1e-9 for _, v in look.get("lum_vs_sat", []) + look.get("hue_vs_sat", []) + look.get("sat_vs_sat", [])
                  + (look.get("sat_vs_lum") or []))
              and all(abs(v - x) < 1e-9 for x, v in look.get("tone_curve") or []))
    hue = all(abs(v) < 1e-9 for _, v in look.get("hue_vs_hue", []))
    skin = look.get("skin") or {}
    skin_ok = all(abs(v) < 1e-9 for _, v in skin.get("hue_vs_hue", [])) and all(abs(v - 1.0) < 1e-9 for _, v in skin.get("hue_vs_sat", []))
    return flat and curves and hue and skin_ok
