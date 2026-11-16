"""Looks for footage with no reference: three suggestions that suit the footage.

The templates live in data (data/looks/suggestions.json, plus an optional
suggestions.json in the user's looks folder). A template says what a look does
to a picture: contrast, black level, tints per tonal zone, saturation per zone.

For each project the app
  1. reads the footage: brightness, contrast, color level, faces, sky, foliage, warm or cool palette
  2. scores every template against these traits and keeps the best of three different families
  3. fits each look to the footage with the same solver the reference match uses,
     so the look lands on standard controls and keeps the footage's exposure
  4. adds skin protection and runs the quality rules with their fixes
  5. renders a preview on the lead shot
"""
from __future__ import annotations

import copy
import json
import time
from typing import Any, Callable

import numpy as np

from .. import paths
from ..analysis import names
from ..color import ops
from . import auto, lookfit, model, pipeline

F32 = np.float32
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=F32)

TRAIT_WORDS = {
    "skin": "faces in the shots", "sky": "open sky", "foliage": "greenery", "low_key": "dark, low-key scenes",
    "high_key": "bright, airy scenes", "colorful": "rich color in the scene", "muted": "quiet color in the scene",
    "flat": "soft contrast in the footage", "contrasty": "strong contrast in the footage", "warm": "warm tones in the scene",
    "cool": "cool tones in the scene", "daylight": "daylight",
}


def load_templates() -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in (paths.SHIPPED_DATA / "looks" / "suggestions.json", paths.sub("looks") / "suggestions.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for t in data.get("templates", []):
            if isinstance(t, dict) and t.get("id") and t.get("name"):
                out[t["id"]] = t
    return list(out.values())


def _ramp(v: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi, straight in between. lo > hi gives a falling ramp."""
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def scene_profile(b: auto.Builder) -> dict[str, float]:
    """Traits of the balanced footage, each from 0 to 1."""
    total = sum(it["duration"] for it in b.items) or 1.0
    key = sky = foliage = skin = 0.0
    parts = b.pooled_footage(12000)
    for it, part in zip(b.items, parts):
        w = it["duration"] / total
        key += w * float(np.median(np.clip(b.odt(part), 0.0, 1.0) @ LUMA))
        detail = (it["wb"].get("detail") or {})
        sky += w * float((detail.get("sky") or {}).get("share") or 0.0)
        foliage += w * float((detail.get("foliage") or {}).get("share") or 0.0)
        skin += w * (1.0 if it["skin"] is not None else 0.0)
    bounds, start = [], 0
    for part in parts:
        bounds.append((start, start + len(part)))
        start += len(part)
    pooled = np.concatenate(parts)
    f0 = lookfit.features_multi(pooled, bounds)
    prof = lookfit.hue_profile(pooled)
    mass = prof["mass"]                                   # 12 sectors of 30 degrees, red at 0
    warm = float(mass[0] + mass[1] + mass[2])             # red, orange, yellow
    cool = float(mass[6] + mass[7] + mass[8])             # cyan to blue
    colored = prof["colored_share"]
    low_key = _ramp(key, 0.34, 0.20)
    return {
        "key": key, "spread": float(f0["spread"][1]), "chroma": float(f0["sats"][1]),
        "low_key": low_key, "high_key": _ramp(key, 0.50, 0.62),
        "flat": _ramp(f0["spread"][1], 0.36, 0.24), "contrasty": _ramp(f0["spread"][1], 0.44, 0.56),
        "colorful": _ramp(f0["sats"][1], 0.085, 0.13), "muted": _ramp(f0["sats"][1], 0.065, 0.035),
        "skin": skin, "sky": _ramp(sky, 0.03, 0.20), "foliage": _ramp(foliage, 0.03, 0.25),
        "warm": _ramp(warm - cool, 0.10, 0.45) * _ramp(colored, 0.05, 0.25),
        "cool": _ramp(cool - warm, 0.05, 0.40) * _ramp(colored, 0.05, 0.25),
        "daylight": (1.0 - low_key) * max(_ramp(sky, 0.03, 0.15), _ramp(foliage, 0.03, 0.15)),
    }


def score(template: dict[str, Any], traits: dict[str, float]) -> tuple[float, list[str]]:
    """How well a template suits the footage, and the traits that speak for it."""
    total = float(template.get("base", 0.3))
    pros = []
    for trait, weight in (template.get("suits") or {}).items():
        gain = float(weight) * float(traits.get(trait, 0.0))
        total += gain
        if gain >= 0.04:
            pros.append((gain, trait))
    return total, [t for _, t in sorted(pros, reverse=True)[:3]]


def _tint(hue_deg: float, amount: float) -> tuple[float, float]:
    a = np.radians(hue_deg)
    return float(amount * np.cos(a)), float(amount * np.sin(a))


def _hue_curve(points, neutral: float, skin_hue: float | None) -> list[list[float]]:
    """Template points on the 12 standard hues. The skin band stays untouched."""
    if not points:
        return []
    vals = np.full(lookfit.SECTORS, neutral, dtype=np.float64)
    width = 360.0 / lookfit.SECTORS
    for h, v in points:
        vals[int(round((h % 360.0) / width)) % lookfit.SECTORS] = v
    centers = np.arange(lookfit.SECTORS) * width
    if skin_hue is not None:
        vals[np.abs((centers - skin_hue + 180.0) % 360.0 - 180.0) <= 28.0] = neutral
    if np.all(vals == neutral):
        return []
    return [[float(c), float(v)] for c, v in zip(centers, vals)]


def fit_template(template: dict[str, Any], parts: list[np.ndarray], skin: np.ndarray | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Turn a template into a look for this footage. Returns (look, fit info)."""
    bounds, start = [], 0
    for part in parts:
        bounds.append((start, start + len(part)))
        start += len(part)
    f0 = lookfit.features_multi(np.concatenate(parts), bounds)
    spread = float(f0["spread"][1])
    ratio = float(template.get("contrast", 1.0))
    if ratio >= 1.0:   # footage with strong contrast takes less on top, flat footage takes more
        ratio = 1.0 + (ratio - 1.0) * float(np.clip((0.52 - spread) / 0.16, 0.25, 1.3))
    else:              # flat footage takes less softening
        ratio = 1.0 - (1.0 - ratio) * float(np.clip((spread - 0.22) / 0.16, 0.3, 1.2))
    tints = f0["tints"].copy()
    for i, zone in enumerate(lookfit.ZONE_NAMES):
        hue, amount = (template.get("tints") or {}).get(zone, [0, 0.0])
        a, b_ = _tint(hue, amount)
        tints[2 * i] += a
        tints[2 * i + 1] += b_
    level = float(template.get("saturation", 1.0))
    roll = template.get("rolloff") or {}
    floor = template.get("floor")
    ref = {"spread": f0["spread"] * ratio, "floor": f0["floor"] if floor is None else float(floor), "tints": tints,
           "sats": f0["sats"] * level, "median": f0["median"], "ceil": f0["ceil"], "tone": f0["tone"]}
    mono = bool(template.get("mono"))
    # The solver sets contrast, levels, tints and the saturation level. The roll-off in deep shadows
    # and bright highlights comes straight from the template, so its two fit parameters stay at zero.
    lo, hi = lookfit.B_LO.copy(), lookfit.B_HI.copy()
    lo[10:], hi[10:] = -1e-6, 1e-6
    look, info = lookfit.fit(parts, ref, {"monochrome": mono, "same_content": False},
                             carry_override={"spread": 1.0, "floor_up": 0.8, "floor_down": 0.6, "tints": 1.0, "sats": 1.0, "median": 0.0},
                             limits=(lo, hi))
    look["lum_vs_sat"] = [] if mono else lookfit.lum_vs_sat_points(float(roll.get("shadows", 1.0)), float(roll.get("highlights", 1.0)))
    skin_hue = None
    if skin is not None and not mono:
        _, c1, c2, _, _ = ops.hsl_split(pipeline.look_stage(skin, model.quantize_look(look)))
        skin_hue = float(np.degrees(np.arctan2(np.median(c2), np.median(c1))) % 360.0)
    if not mono:
        look["hue_vs_sat"] = _hue_curve(template.get("hue_vs_sat"), 1.0, skin_hue)
        look["hue_vs_hue"] = _hue_curve(template.get("hue_vs_hue"), 0.0, skin_hue)
        look["sat_vs_sat"] = lookfit.saturation_limiter()
    look["name"] = template["name"]
    look["description"] = template.get("mood", "")
    look["source"] = "suggestion"
    look["reasons"] = _reasons(template, look, ratio, mono)
    info["contrast_ratio"] = ratio
    return look, info


def _reasons(template: dict[str, Any], look: dict[str, Any], ratio: float, mono: bool) -> dict[str, str]:
    name = template["name"]
    tints = template.get("tints") or {}
    tone = lambda zone: names.hue_name(tints[zone][0]) if tints.get(zone) and tints[zone][1] > 0 else None
    sat_level = float(template.get("saturation", 1.0))
    out = {"contrast": f"{name}: tonal spread {(ratio - 1.0) * 100:+.0f}% against your balanced footage. Your exposure stays."}
    sh, mid, hi = tone("shadows"), tone("midtones"), tone("highlights")
    floor = template.get("floor")
    level = "black level kept" if floor is None else ("lifted blacks" if floor >= 0.05 else "soft blacks" if floor >= 0.03 else "deep blacks")
    out["lift"] = f"{name}: {level}" + (f", {sh} in the shadows." if sh else ".")
    out["gamma"] = f"{name}: midtones hold their level" + (f", with a {mid} lean." if mid else ".")
    out["gain"] = f"{name}: {hi} highlights." if hi else f"{name}: highlights stay neutral."
    if mono:
        out["saturation"] = f"{name}: all color removed."
    else:
        out["saturation"] = f"{name}: color at {sat_level * 100:.0f}% of your balanced footage."
        out["lum_vs_sat"] = f"{name}: less color in deep shadows and bright highlights, the way film behaves."
        out["hue_vs_sat"] = f"{name}: palette trim of single color families."
        out["hue_vs_hue"] = f"{name}: small hue move of single color families."
        out["sat_vs_sat"] = "Safety limit: bends extreme colors under the display gamut. Normal colors pass untouched."
    return out


def _preview_shot(b: auto.Builder) -> dict[str, Any]:
    """The shot the previews show: the longest shot with faces, else the longest shot."""
    with_skin = [it for it in b.items if it["skin"] is not None]
    pool = with_skin or b.items
    return max(pool, key=lambda it: it["duration"])["shot"]


def thumb_path(suggestion_id: str):
    d = paths.sub("cache") / "suggestions"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{paths.safe_id(suggestion_id)}.jpg"


def suggest(proj, progress: Callable[[float, str], None] | None = None, count: int = 3) -> list[dict[str, Any]]:
    """Find `count` looks that suit the footage. Stores them in the project and returns them."""
    say = progress or (lambda f, m: None)
    if not proj.state.get("graded"):   # the looks sit on balanced, matched shots
        auto.build(proj, lambda f, m: say(0.3 * f, m), refit=False)
    say(0.32, "Reading the footage")
    b = auto.prepared_builder(proj)
    traits = scene_profile(b)
    ranked = sorted(((score(t, traits), t) for t in load_templates()), key=lambda r: -r[0][0])
    parts = b.pooled_footage(12000)
    skin = b.pooled_skin()
    finish = copy.deepcopy(proj.state.get("finish") or model.neutral_finish())
    strength = float(proj.state.get("strength", 80)) / 100.0
    snap = b.snapshot()
    lead = _preview_shot(b)
    t_mid = lead["times"][len(lead["times"]) // 2] if lead.get("times") else 0.5 * (lead["start"] + lead["end"])
    chosen: list[dict[str, Any]] = []
    families: set[str] = set()
    stamp = int(time.time())
    for (points, pros), tpl in ranked:
        if len(chosen) >= count:
            break
        if tpl.get("family") in families:
            continue
        say(0.35 + 0.6 * len(chosen) / count, f"Fitting {tpl['name']}")
        look, _ = fit_template(tpl, parts, skin)
        look = b.apply_skin(look)
        look, reports = auto.enforce_rules(b, look, finish, 1.0 if tpl.get("mono") else strength)
        failed = sum(r["status"] == "fail" for rules in reports.values() for r in rules)
        fixes = list(b.fixes)
        b.restore(snap)
        if failed:     # this look breaks the footage even after the fixes: next candidate
            continue
        look = model.quantize_look(look) | {"reasons": look.get("reasons", {})}
        why = ("Suits this footage: " + ", ".join(TRAIT_WORDS.get(p, p) for p in pros) + ".") if pros else "A safe choice for any footage."
        use = 1.0 if tpl.get("mono") else strength   # black and white means no color left: full strength
        try:
            img = proj.render_frame(lead["clip_id"], t_mid, "graded", 480, lead["id"], look=model.at_strength(look, use))
            thumb_path(tpl["id"]).write_bytes(pipeline.to_jpeg(img, 86))
        except Exception:
            thumb_path(tpl["id"]).unlink(missing_ok=True)
        chosen.append({"id": tpl["id"], "name": tpl["name"], "mood": tpl.get("mood", ""), "family": tpl.get("family", ""),
                       "why": why, "score": round(points, 3), "look": look, "fixes": fixes, "stamp": stamp,
                       "strength": int(round(use * 100))})
        families.add(tpl.get("family"))
    with proj.lock:
        proj.state["suggestions"] = chosen
        proj.state["scene_profile"] = {k: round(float(v), 3) for k, v in traits.items()}
        proj.save()
    say(1.0, "Looks ready")
    return chosen


def apply(proj, suggestion_id: str) -> dict[str, Any]:
    """Make a suggestion the project look. The caller rebuilds the grade (no refit)."""
    for s in proj.state.get("suggestions") or []:
        if s["id"] == suggestion_id:
            proj.set_look(copy.deepcopy(s["look"]))
            if s.get("strength", 0) > proj.state.get("strength", 80):   # a monochrome look at part strength keeps color
                proj.set_strength(s["strength"])
            return s
    raise KeyError("No such suggestion. Ask for suggestions again.")
