"""Three looks designed from the shot itself, for interviews shot outdoors.

No preset list. The app reads the scene of the chosen shot (the light it was shot in, the palette,
the faces, the sky, the contrast), decides which creative directions suit it, and computes every
number of each look from those measurements. Each look follows the colorist workflow:

  1 White balance   per shot, a Gain on a linear node, anchored on skin and neutral objects
  2 Exposure        per shot on a linear node (black stays black), plus the look's own bias
  3 Contrast        Contrast at pivot 0.336 and a film curve that eases into black and white
  4 Saturation      an HSV node with channels 1 and 3 off: colors get denser as they saturate
  5 Hue rotation    color families present in the shot move toward the look's palette
  6 Split tone      Lift and Gain per channel: complementary color in shadows and highlights
  7 Relight         background darker around tracked faces and shoulders, the speaker keeps the light
  8 Output          DaVinci Wide Gamut / Intermediate to Rec.709 with DaVinci tone mapping

The per-shot steps (1 and 2) come from the balance. The look carries steps 2 to 6.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable

import numpy as np

from ..analysis import names
from ..color import ops
from ..color import working as W
from . import auto, lookfit, model, pipeline, skin as skin_mod, suggest

F32 = np.float32
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=F32)
SECTOR = 360.0 / lookfit.SECTORS   # 30 degrees: red 0, orange 30, yellow 60, ..., blue 240, magenta 300


def _ramp(v: float, lo: float, hi: float) -> float:
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


# ----------------------------------------------------------------- reading
def read_scene(b: auto.Builder, shot_id: str | None = None) -> dict[str, Any]:
    """Measure the scene of one shot: light, palette, faces, sky, contrast, exposure."""
    lead = next((it for it in b.items if it["shot"]["id"] == shot_id), None) if shot_id else None
    if lead is None:
        lead = next((it for it in b.items if it["shot"]["id"] == suggest._preview_shot(b)["id"]), b.items[0])
    members = [it for it in b.items if it.get("scene") == lead.get("scene")] or [lead]
    weights = np.array([it["duration"] * (2.0 if it is lead else 1.0) for it in members], dtype=np.float64)
    weights /= weights.sum()
    parts = [b._di(it)[:: max(1, len(b._di(it)) // 4000)].astype(F32) for it in members]
    pooled = np.concatenate(parts)
    bounds, start = [], 0
    for part in parts:
        bounds.append((start, start + len(part)))
        start += len(part)
    f0 = lookfit.features_multi(pooled, bounds)
    disp_y = [np.clip(b.odt(p), 0.0, 1.0) @ LUMA for p in parts]
    key = float(sum(w * float(np.median(y)) for w, y in zip(weights, disp_y)))
    top = float(max(np.percentile(y, 99.0) for y in disp_y))
    clipped = float(max((y >= 0.985).mean() for y in disp_y))
    prof = lookfit.hue_profile(pooled)
    mass = prof["mass"]
    # the light the footage was shot in: the cast the balance removed, in stops
    temp = float(sum(w * float(it["wb"].get("temperature", 0.0)) * float(it["wb"].get("confidence", 0.5)) for w, it in zip(weights, members)))
    tint = float(sum(w * float(it["wb"].get("tint", 0.0)) * float(it["wb"].get("confidence", 0.5)) for w, it in zip(weights, members)))
    sky = float(sum(w * float(((it["wb"].get("detail") or {}).get("sky") or {}).get("share") or 0.0) for w, it in zip(weights, members)))
    foliage = float(sum(w * float(((it["wb"].get("detail") or {}).get("foliage") or {}).get("share") or 0.0) for w, it in zip(weights, members)))
    faces = [it for it in members if it["skin"] is not None]
    skin_level = skin_hue_off = None
    if faces:
        sk = np.concatenate([b._di(it, "skin") for it in faces])
        skin_level = float(np.median(np.clip(b.odt(sk), 0.0, 1.0) @ LUMA))
        ang, _ = skin_mod.skin_on_display(sk, b.odt)
        skin_hue_off = float(skin_mod.offset_from_line(ang))
    spread = float(f0["spread"][1])
    chroma = float(f0["sats"][1])
    autumn = float(mass[0] + mass[11] * 0.5 + mass[2])          # reds and yellows: skin sits at orange, so orange counts apart
    greens = float(mass[3] + mass[4])
    blues = float(mass[6] + mass[7] + mass[8])
    traits = {
        "skin": len(faces) / len(members),
        "soft_light": _ramp(spread, 0.44, 0.30),
        "hard_light": _ramp(spread, 0.46, 0.58),
        "cool_light": _ramp(-temp, 0.05, 0.40),
        "warm_light": _ramp(temp, 0.15, 0.55),
        "autumn": _ramp(autumn, 0.12, 0.35) * _ramp(prof["colored_share"], 0.04, 0.2),
        "greenery": max(_ramp(greens, 0.10, 0.35), _ramp(foliage, 0.03, 0.25)),
        "blue": max(_ramp(blues, 0.10, 0.35), _ramp(sky, 0.03, 0.2)),
        "muted": _ramp(chroma, 0.075, 0.04),
        "colorful": _ramp(chroma, 0.09, 0.13),
        "low_key": _ramp(key, 0.32, 0.20),
        "high_key": _ramp(key, 0.48, 0.60),
        "bright_top": _ramp(top, 0.90, 0.98) + _ramp(clipped, 0.002, 0.02),
        "green_cast": _ramp(-tint, 0.05, 0.25),
    }
    traits["overcast"] = traits["soft_light"] * (1.0 - traits["warm_light"]) * (1.0 - _ramp(chroma, 0.10, 0.14))
    return {"lead": lead, "members": members, "parts": parts, "features": f0, "profile": prof, "traits": traits,
            "key": key, "top": top, "clipped": clipped, "temp": temp, "tint": tint, "spread": spread, "chroma": chroma,
            "skin_level": skin_level, "skin_offset": skin_hue_off, "sky": sky, "foliage": foliage}


def describe_scene(sc: dict[str, Any]) -> list[str]:
    """The scene read in plain words, for the panel above the suggestions."""
    t = sc["traits"]
    light = ("soft, overcast light" if t["overcast"] > 0.5 else "hard sunlight" if t["hard_light"] > 0.5
             else "warm, low sun" if t["warm_light"] > 0.5 else "open daylight")
    cast = []
    if abs(sc["temp"]) >= 0.08:
        cast.append(f"{abs(sc['temp']):.2f} stops {'warm' if sc['temp'] > 0 else 'cool'}")
    if abs(sc["tint"]) >= 0.08:
        cast.append(f"{abs(sc['tint']):.2f} stops {'green' if sc['tint'] < 0 else 'magenta'}")
    out = [f"Light: {light}" + (f", came in {' and '.join(cast)}." if cast else ", close to neutral.")]
    palette = []
    if t["autumn"] > 0.4:
        palette.append("autumn reds and yellows")
    if t["greenery"] > 0.4:
        palette.append("greenery")
    if t["blue"] > 0.4:
        palette.append("sky or blue tones")
    out.append("Palette: " + (", ".join(palette) if palette else "mostly neutral") + (", quiet color." if t["muted"] > 0.5 else "."))
    if sc["skin_level"] is not None:
        side = "on the skin tone line" if abs(sc["skin_offset"]) <= 4 else \
            f"{abs(sc['skin_offset']):.0f} degrees toward {'yellow-green' if sc['skin_offset'] > 0 else 'red-magenta'}"
        out.append(f"Faces in {t['skin'] * 100:.0f}% of the scene's shots. Skin at {sc['skin_level'] * 100:.0f}% on the waveform, {side}.")
    else:
        out.append("No faces found in this scene: balance anchors on neutral objects, sky and foliage.")
    contrast = "flat" if t["soft_light"] > 0.5 else "contrasty" if t["hard_light"] > 0.5 else "medium"
    out.append(f"Contrast: {contrast}. Brightest tones at {sc['top'] * 100:.0f}%" + (", sky or background close to clipping." if t["bright_top"] > 0.6 else "."))
    return out


# ------------------------------------------------------------- directions
def _present(sc: dict[str, Any], hue: float, floor: float = 0.025) -> float:
    """Share of the scene's color in the 30-degree sector around a hue."""
    return float(sc["profile"]["mass"][int(round((hue % 360.0) / SECTOR)) % lookfit.SECTORS])


SECTOR_NAMES = {0: "red", 30: "orange", 60: "yellow", 90: "yellow-green", 120: "green", 150: "green-cyan",
                180: "cyan", 210: "teal", 240: "blue", 270: "violet", 300: "magenta", 330: "red"}


def sector_name(h: float) -> str:
    return SECTOR_NAMES[int(round((h % 360.0) / SECTOR)) * 30 % 360]


def _moves(sc: dict[str, Any], wanted: dict[float, float]) -> dict[float, float]:
    """Keep only the hue moves for colors the scene holds. Scarce colors move less."""
    out = {}
    for h, deg in wanted.items():
        m = _present(sc, h)
        if m >= 0.015:
            out[h] = round(deg * min(1.0, 0.6 + m / 0.06), 1)
    return out


def directions(sc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every direction that could suit this scene, with its score and every number computed from the scene."""
    t = sc["traits"]
    chroma = max(sc["chroma"], 0.01)
    # contrast: a target tonal spread, turned into a ratio against all shots at build time (the look is shared)
    ratio = lambda target, lo=0.88, hi=1.35: (float(target), float(lo), float(hi))
    def hsv_for(target: float, lo: float, hi: float, gamma_bias: float = 0.0) -> dict[str, float]:
        """HSV saturation that reaches the color target without pushing strong colors past full saturation.

        Gain multiplies every saturation, so strong colors clip first: it stays at or under 1.04. Gamma lifts
        gentle colors most and never takes a color past 1, so it carries the rest of the boost. Less color
        comes from Gain alone, which never clips.
        """
        r = float(np.clip((target / chroma) ** 0.6, lo, hi))
        # the Gamma part needs the footage's own saturation under the look: build_look() sets it
        return {"ratio": round(r, 3), "bias": float(gamma_bias), "gain": round(min(r, 1.04), 3), "gamma": round(gamma_bias, 3)}

    room_up = _ramp(0.97 - sc["top"], 0.0, 0.12)   # headroom above the brightest tones
    out = []

    # Clean interview: the honest version of the shot. Always a candidate for an interview.
    out.append({
        "id": "clean", "family": "natural", "name": "Clean interview",
        "mood": "True colors, healthy skin, gentle contrast. What the day looked like, finished.",
        "score": 0.6 + 0.2 * t["skin"],
        "fits": "A safe base for any interview: skin and whites carry the shot.",
        "exposure": 0.0, "contrast": ratio(0.50, 0.95, 1.30), "floor": None, "relight": 0.4,
        "tints": {"shadows": [0, 0.0], "midtones": [0, 0.0], "highlights": [40, 0.003]},
        "hsv": hsv_for(0.115, 0.98, 1.3, 0.0),
        "hue_vs_hue": _moves(sc, {120: -6.0, 90: -4.0}), "hue_vs_sat": {},
        "rolloff": {"shadows": 0.9, "highlights": 0.88}, "finish": 1.0,
    })

    # Warmth: an answer to cool or overcast light, or a lean into autumn color and low sun.
    warmth = 0.010 + 0.010 * max(t["cool_light"], t["overcast"], t["autumn"])
    if t["autumn"] > 0.4:
        name, mood = "Warm autumn", "Leaves pulled to orange, warm highlights, skin with a glow. A crisp fall afternoon."
        moves = _moves(sc, {0: 8.0, 330: 6.0, 60: -8.0, 90: -10.0, 120: -12.0})
        hvs = {30: 1.08, 0: 1.04}
    elif t["warm_light"] > 0.5:
        name, mood = "Golden hour", "Low-sun warmth kept and shaped: amber highlights, soft shadows, rich skin."
        moves = _moves(sc, {60: -6.0, 120: -10.0, 90: -6.0})
        hvs = {30: 1.05}
    elif t["overcast"] > 0.5:
        name, mood = "Warm overcast", "Puts back the warmth gray skies take out. Skin comes alive, the day feels milder."
        moves = _moves(sc, {120: -10.0, 90: -6.0, 210: -6.0})
        hvs = {30: 1.04}
    else:
        name, mood = "Warm daylight", "A friendly, sunny lean: warm highlights and midtones, natural shadows."
        moves = _moves(sc, {120: -8.0, 90: -5.0})
        hvs = {}
    out.append({
        "id": "warm", "family": "warm", "name": name, "mood": mood,
        "score": 0.35 + 0.35 * max(t["cool_light"], t["overcast"], t["autumn"], t["warm_light"]) + 0.1 * t["skin"],
        "fits": ("Your light came in cool" if t["cool_light"] > 0.4 else "Gray sky flattens skin" if t["overcast"] > 0.5
                 else "The scene holds autumn color" if t["autumn"] > 0.4 else "The scene has warm light" if t["warm_light"] > 0.5
                 else "Daylight with faces") + ": warmth in the highlights flatters skin.",
        "exposure": 0.12 * room_up * t["overcast"], "contrast": ratio(0.51, 0.95, 1.32), "floor": None, "relight": 0.5,
        "tints": {"shadows": [25, 0.002], "midtones": [35, 0.004 + 0.003 * t["cool_light"]], "highlights": [42, warmth]},
        "hsv": hsv_for(0.12, 1.0, 1.35, 0.02),
        "hue_vs_hue": moves, "hue_vs_sat": hvs,
        "rolloff": {"shadows": 0.9, "highlights": 0.9}, "finish": 0.85,
    })

    # Cinematic teal and orange (hue compression and split tone).
    out.append({
        "id": "cinematic", "family": "cinematic", "name": "Cinematic teal and orange",
        "mood": "Colors compressed to two: teal shadows and backgrounds, warm skin. Deeper contrast, the modern film look.",
        "score": 0.45 + 0.15 * t["skin"] + 0.15 * t["blue"] + 0.1 * t["hard_light"] + 0.05 * t["greenery"],
        "fits": "Faces against " + ("sky and blue tones" if t["blue"] > 0.4 else "the background") + ": teal behind, warm skin in front separates the speaker.",
        "exposure": -0.2 * (1.0 - t["low_key"]), "contrast": ratio(0.54, 1.0, 1.38), "floor": None, "relight": 0.7,
        "tints": {"shadows": [195, 0.016 + 0.004 * t["blue"]], "midtones": [30, 0.003], "highlights": [40, 0.012]},
        "hsv": hsv_for(0.115, 1.0, 1.3, 0.0),
        "hue_vs_hue": _moves(sc, {240: -16.0, 210: -8.0, 120: -14.0, 90: -10.0, 60: -8.0, 300: 14.0, 330: 8.0}),
        "hue_vs_sat": {180: 1.06, 210: 1.06},
        "rolloff": {"shadows": 0.85, "highlights": 0.86}, "finish": 1.0,
    })

    # Rich and moody: darker reads richer. For flat light and bright, busy backgrounds.
    out.append({
        "id": "moody", "family": "moody", "name": "Rich and moody",
        "mood": "Darker, denser color, deep blacks, restrained greens. The speaker stands out from a quieter world.",
        "score": 0.3 + 0.3 * t["overcast"] + 0.2 * t["bright_top"] + 0.1 * t["greenery"] - 0.3 * t["low_key"],
        "fits": ("Flat light" if t["overcast"] > 0.5 else "A bright background" if t["bright_top"] > 0.5 else "This scene")
                + ": lower exposure and denser color give it weight and calm the background.",
        "exposure": -0.35 * (1.0 - t["low_key"]), "contrast": ratio(0.54, 1.0, 1.38), "floor": None, "relight": 1.0,
        "tints": {"shadows": [200, 0.008], "midtones": [0, 0.0], "highlights": [45, 0.005]},
        "hsv": hsv_for(0.125, 1.05, 1.4, -0.01),
        "hue_vs_hue": _moves(sc, {120: -8.0, 90: -6.0}), "hue_vs_sat": {120: 0.86, 90: 0.9, 240: 0.92},
        "rolloff": {"shadows": 0.8, "highlights": 0.85}, "finish": 1.0,
    })

    # Punchy interview: the studio interview grade. Levels filled, strong contrast and saturation,
    # cool shadows, mids nudged to magenta against green, warm skin, background relit down.
    out.append({
        "id": "punchy", "family": "punchy", "name": "Punchy interview",
        "mood": "Deep blacks, strong contrast, rich warm skin, cool shadows. The background steps back and the speaker pops.",
        "score": (0.45 + 0.25 * t["skin"] + 0.2 * t["muted"] + 0.15 * t["soft_light"] + 0.1 * t["green_cast"]) if t["skin"] > 0 else 0.2,
        "fits": ("Faces in flat light" if t["soft_light"] > 0.5 else "Faces in quiet color" if t["muted"] > 0.5 else "An interview")
                + ": contrast, color and a relight make the speaker pop.",
        "exposure": -0.1 * (1.0 - t["low_key"]), "contrast": ratio(0.57, 1.0, 1.42), "floor": None, "relight": 0.9,
        "tints": {"shadows": [215, 0.010], "midtones": [300, 0.003], "highlights": [30, 0.010]},
        "hsv": hsv_for(0.13, 1.05, 1.45, 0.02),
        "hue_vs_hue": _moves(sc, {120: -8.0, 90: -6.0}), "hue_vs_sat": {120: 0.9, 150: 0.92},
        "rolloff": {"shadows": 0.85, "highlights": 0.9}, "finish": 1.0,
    })

    # Bright and airy: soft light lifted, pastel color. Needs headroom above the brightest tones.
    out.append({
        "id": "airy", "family": "airy", "name": "Bright and airy",
        "mood": "Light, open and soft: lifted shadows, pastel color, clean whites. Friendly and approachable.",
        "score": (0.3 + 0.35 * t["soft_light"] * (1.0 - t["low_key"]) + 0.2 * t["high_key"]) * (0.4 + 0.6 * room_up) - 0.3 * t["hard_light"],
        "fits": "Soft light with room above the brightest tones: the shot takes a lift without clipping.",
        "exposure": 0.25 * room_up * (1.0 - t["high_key"]), "contrast": ratio(0.42, 0.88, 1.15), "floor": 0.045, "relight": 0.25,
        "tints": {"shadows": [210, 0.004], "midtones": [0, 0.0], "highlights": [45, 0.005]},
        "hsv": hsv_for(0.09, 0.9, 1.12, 0.02),
        "hue_vs_hue": _moves(sc, {120: -6.0}), "hue_vs_sat": {120: 0.92, 240: 0.95},
        "rolloff": {"shadows": 0.95, "highlights": 0.85}, "finish": 0.35,
    })
    return out


def relight_for(sc: dict[str, Any], d: dict[str, Any]) -> float:
    """Background darkening in stops for one look: only with faces, more when the background is bright."""
    t = sc["traits"]
    if t["skin"] <= 0.0:
        return 0.0
    return round(float(min(1.2, d.get("relight", 0.0) * (0.8 + 0.4 * t["bright_top"] + 0.2 * t["high_key"]))), 2)


def pick(cands: list[dict[str, Any]], count: int = 3) -> list[dict[str, Any]]:
    """The clean base first, then the best scoring other directions."""
    base = [c for c in cands if c["family"] == "natural"]
    rest = sorted([c for c in cands if c["family"] != "natural"], key=lambda c: -c["score"])
    return (base + rest)[:count]


# ----------------------------------------------------------------- building
def _expose(parts: list[np.ndarray], stops: float) -> list[np.ndarray]:
    g = pipeline.exposure_gain(stops)
    return parts if g == 1.0 else [W.di_encode(W.di_decode(p) * F32(g)) for p in parts]


def hsv_settings(look: dict[str, Any], parts: list[np.ndarray], hsv: dict[str, float]) -> dict[str, float]:
    """Gain and Gamma of the HSV saturation node for a wanted color ratio, measured on this footage.

    Gain multiplies every saturation, so strong colors clip first: it stays at or under 1.04. Gamma lifts
    gentle colors most and never takes a color past full saturation, so it carries the rest. In the log
    working space most colors sit near saturation 0.1, far lower than on a display, so the Gamma that gives
    the wanted lift is found at the footage's own median saturation.
    """
    r = float(hsv.get("ratio", hsv.get("gain", 1.0)))
    bias = float(hsv.get("bias", hsv.get("gamma", 0.0)))
    if r <= 1.0:
        return {"gain": round(r, 3), "gamma": round(bias, 3)}
    gain = min(r, 1.04)
    di = pipeline.look_stage(np.concatenate(parts), dict(look, hsv_sat={"gain": 1.0, "gamma": 0.0}))
    v = di.max(axis=1)
    s = (v - di.min(axis=1)) / np.maximum(v, 1e-6)
    s = s[(s > 0.02) & (v > 0.05)]
    s_med = float(np.clip(np.median(s) if len(s) else 0.12, 0.05, 0.5))
    e = 1.0 + float(np.log(r / gain)) / float(np.log(s_med))
    gamma = float(np.clip((1.0 / max(e, 0.5) - 1.0) / 4.0 + bias, -0.05, 0.06))
    return {"gain": round(gain, 3), "gamma": round(gamma, 3)}


def build_look(b: auto.Builder, d: dict[str, Any], polish_amount: float) -> dict[str, Any]:
    """Every number of one direction, fitted to the footage with the same solver as the reference match."""
    exp = round(float(d["exposure"]), 2)
    parts = _expose(b.pooled_footage(12000), exp)
    bounds, start = [], 0
    for part in parts:
        bounds.append((start, start + len(part)))
        start += len(part)
    spread = float(lookfit.features_multi(np.concatenate(parts), bounds)["spread"][1])
    target, lo, hi = d["contrast"]
    contrast = float(np.clip(target / max(spread, 0.05), lo, hi))
    skin = b.pooled_skin()
    skin_e = _expose([skin], exp)[0] if skin is not None else None
    template = {"name": d["name"], "mood": d["mood"], "contrast": contrast, "floor": d["floor"], "tints": d["tints"],
                "saturation": 1.0, "rolloff": d["rolloff"],
                "hue_vs_hue": [[h, v] for h, v in d["hue_vs_hue"].items()], "hue_vs_sat": [[h, v] for h, v in d["hue_vs_sat"].items()]}
    look, _ = suggest.fit_template(template, parts, skin_e, adapt=False)
    look["exposure"] = exp
    look["hsv_sat"] = hsv_settings(look, parts, d["hsv"])
    look = auto.finish_tone(b, look, polish_amount * float(d.get("finish", 1.0)))
    look["sat_vs_lum"] = []   # the HSV node already makes saturated colors denser
    look["name"], look["description"], look["source"] = d["name"], d["mood"], "suggestion"
    return look


def _skin_level(b: auto.Builder, members: list[dict[str, Any]], look: dict[str, Any]) -> float | None:
    faces = [it for it in members if it["skin"] is not None]
    if not faces:
        return None
    sk = np.concatenate([b._di(it, "skin") for it in faces])
    return float(np.median(np.clip(b.odt(pipeline.look_stage(sk, look)), 0.0, 1.0) @ LUMA))


def _tone_word(rgb_off: list[float], scale: float) -> str | None:
    """Name the color a Lift or Gain triple leans toward, or None when it is close to neutral."""
    v = np.asarray(rgb_off, dtype=np.float64) - float(np.mean(rgb_off))
    if float(np.abs(v).max()) < scale:
        return None
    _, c1, c2, _, _ = ops.hsl_split(v.astype(F32)[None, :] + F32(0.5))
    return names.hue_name(float(np.degrees(np.arctan2(c2[0], c1[0])) % 360.0))


def steps(b: auto.Builder, sc: dict[str, Any], look: dict[str, Any], strength: float, output_name: str,
          relight_stops: float = 0.0) -> list[dict[str, str]]:
    """The workflow, step by step, with the values this look uses and why."""
    applied = model.at_strength(look, strength)
    members = sc["members"]
    gains = [np.asarray(it["grade"]["balance"]["gain"], dtype=np.float64) for it in members]
    stops = [float(np.log2(max(float(np.mean(g)), 1e-6))) for g in gains]
    casts = []
    if abs(sc["temp"]) >= 0.05:
        casts.append(f"{abs(sc['temp']):.2f} stops {'warm' if sc['temp'] > 0 else 'cool'}")
    if abs(sc["tint"]) >= 0.05:
        casts.append(f"{abs(sc['tint']):.2f} stops {'green' if sc['tint'] < 0 else 'magenta'}")
    anchors = "skin and neutral objects" if sc["traits"]["skin"] > 0 else "neutral objects, sky and foliage"
    out = [{"step": "1 White balance", "value": "Gain on a linear node, per shot",
            "why": (f"The scene came in {' and '.join(casts)}. " if casts else "The scene came in close to neutral. ")
                   + f"Each shot is balanced on {anchors}, so blacks stay clean."}]
    lvl = _skin_level(b, members, applied)
    exp_txt = f"per shot {min(stops):+.2f} to {max(stops):+.2f} stops" if len(stops) > 1 else f"{stops[0]:+.2f} stops"
    look_exp = float(applied.get("exposure", 0.0))
    out.append({"step": "2 Exposure", "value": exp_txt + (f", look {look_exp:+.2f} stops" if abs(look_exp) >= 0.005 else ""),
                "why": (f"Skin lands at {lvl * 100:.0f}% on the waveform. " if lvl is not None else "")
                       + ("Darker reads richer." if look_exp < -0.005 else "A lift for a lighter feel, with the highlights checked." if look_exp > 0.005
                          else "The look keeps the balanced exposure.")})
    lows, highs = [], []
    for it in members:   # measured after each shot's own trim under the look
        y = np.clip(b.odt(pipeline.look_stage(b._di(it), applied)), 0.0, 1.0) @ LUMA
        lows.append(float(np.percentile(y, 1.0)))
        highs.append(float(np.percentile(y, 99.0)))
    lo, hi = float(np.median(lows)), float(max(highs))
    out.append({"step": "3 Contrast", "value": f"Contrast {applied['contrast']:.3f} at pivot {applied['pivot']:.3f}"
                + (", film curve" if applied.get("tone_curve") else ""),
                "why": f"Blacks at {max(lo, 0) * 100:.0f}%, brightest tones at {hi * 100:.0f}%. "
                       + ("The curve eases into black and white, so nothing new clips." if applied.get("tone_curve") else "")})
    hs = applied.get("hsv_sat") or {}
    g, gm = float(hs.get("gain", 1.0)), float(hs.get("gamma", 0.0))
    out.append({"step": "4 Saturation", "value": f"HSV node, Gain {g:.3f}" + (f", Gamma {gm:+.3f}" if abs(gm) >= 0.001 else ""),
                "why": ("Your footage holds quiet color: " if sc["traits"]["muted"] > 0.5 else "")
                       + ("Gain adds density to strong colors" if g > 1.005 else "Gain pulls strong colors back" if g < 0.995 else "Strong colors stay")
                       + ("; Gamma lifts gentle colors such as skin." if gm > 0.001 else "; Gamma keeps gentle colors quiet." if gm < -0.001 else ".")})
    moves = [(h, v) for h, v in applied.get("hue_vs_hue") or [] if abs(v) >= 1.0]
    if moves:
        words = []
        for h, v in moves:
            words.append(f"{sector_name(h)} {abs(v):.0f}° toward {sector_name(h + float(np.sign(v)) * 30.0)}")
        out.append({"step": "5 Hue rotation", "value": ", ".join(words),
                    "why": "Only colors this shot holds move. Skin stays out of every move."})
    else:
        out.append({"step": "5 Hue rotation", "value": "none", "why": "The palette stays true to the scene."})
    sh, hl = _tone_word(applied["lift"], 0.003), _tone_word([v - 1.0 for v in applied["gain"]], 0.006)
    lift_txt = "Lift " + " ".join(f"{c} {v:+.3f}" for c, v in zip("RGB", applied["lift"]))
    gain_txt = "Gain " + " ".join(f"{c} {v:.3f}" for c, v in zip("RGB", applied["gain"]))
    out.append({"step": "6 Split tone", "value": f"{lift_txt}, {gain_txt}",
                "why": (f"{sh.capitalize()} shadows" if sh else "Neutral shadows") + " against "
                       + (f"{hl} highlights." if hl else "neutral highlights.")
                       + (" Complementary colors shape the face." if sh and hl and sh != hl else "")})
    if relight_stops > 0:
        out.append({"step": "7 Relight", "value": f"Background {relight_stops:.2f} stops down, soft tracked shapes over faces and shoulders",
                    "why": "The eye goes to the speaker without a visible vignette. The subject keeps its exposure."})
    else:
        out.append({"step": "7 Relight", "value": "none",
                    "why": "No faces in this scene." if sc["traits"]["skin"] <= 0 else "This look keeps the background as it is."})
    out.append({"step": "8 Output", "value": f"DaVinci Wide Gamut / Intermediate to {output_name}",
                "why": "DaVinci tone mapping rolls highlights off. Every step above works in the wide space before it."})
    return out


def propose(proj, progress: Callable[[float, str], None] | None = None, count: int = 3,
            shot_id: str | None = None) -> list[dict[str, Any]]:
    """Read the scene of a shot and build `count` looks for it. Stores them in the project."""
    from .. import settings
    from ..color import working as Wk

    say = progress or (lambda f, m: None)
    if not proj.state.get("graded"):   # the looks sit on balanced, matched shots
        auto.build(proj, lambda f, m: say(0.3 * f, m), refit=False)
    say(0.32, "Reading the shot")
    b = auto.prepared_builder(proj)
    sc = read_scene(b, shot_id)
    cands = directions(sc)
    finish = copy.deepcopy(proj.state.get("finish") or model.neutral_finish())
    strength = 1.0   # each look is designed for this footage: it runs in full, the slider can take it back
    polish_amount = float(proj.state.get("polish", 85)) / 100.0
    output_name = Wk.OUTPUT_SPACES.get(settings.load()["output_space"], {}).get("name", "Rec.709")
    snap = b.snapshot()
    base_reports = b.check(model.neutral_look(), strength, finish)
    base_fail = {(sid, r["id"]) for sid, rules in base_reports.items() for r in rules if r["status"] == "fail"}
    lead = sc["lead"]["shot"]
    t_mid = lead["times"][len(lead["times"]) // 2] if lead.get("times") else 0.5 * (lead["start"] + lead["end"])
    chosen: list[dict[str, Any]] = []
    stamp = int(time.time())
    ranked = pick(cands, len(cands))
    for d in ranked:
        if len(chosen) >= count:
            break
        say(0.35 + 0.6 * len(chosen) / count, f"Building {d['name']}")
        look = build_look(b, d, polish_amount)
        look = b.apply_skin(look)
        b.shot_contrast(model.at_strength(look, strength))   # each shot trimmed under this look
        look, reports = auto.enforce_rules(b, look, finish, strength)
        failed = {(sid, r["id"]) for sid, rules in reports.items() for r in rules if r["status"] == "fail"} - base_fail
        fixes = list(b.fixes)
        rl = relight_for(sc, d)
        step_list = steps(b, sc, look, strength, output_name, rl)
        b.restore(snap)
        if failed:
            continue
        look = model.quantize_look(look) | {"reasons": look.get("reasons", {})}
        look["reasons"]["exposure"] = next(s["why"] for s in step_list if s["step"].startswith("2"))
        look["reasons"]["hsv_gain"] = next(s["why"] for s in step_list if s["step"].startswith("4"))
        try:
            fin = copy.deepcopy(proj.state.get("finish") or model.neutral_finish())
            fin["relight"] = {"enabled": rl > 0, "amount": rl or 0.6, "subject": 0.0}
            img = proj.render_frame(lead["clip_id"], t_mid, "graded", 480, lead["id"], look=model.at_strength(look, strength),
                                    finish_override=fin)
            suggest.thumb_path(d["id"]).write_bytes(pipeline.to_jpeg(img, 86))
        except Exception:
            suggest.thumb_path(d["id"]).unlink(missing_ok=True)
        chosen.append({"id": d["id"], "name": d["name"], "mood": d["mood"], "family": d["family"], "why": d["fits"],
                       "score": round(float(d["score"]), 3), "look": look, "fixes": fixes, "stamp": stamp,
                       "strength": int(round(strength * 100)), "steps": step_list, "shot_id": lead["id"],
                       "relight": {"enabled": rl > 0, "amount": rl or 0.6, "subject": 0.0}})
    with proj.lock:
        proj.state["suggestions"] = chosen
        proj.state["scene_read"] = {"shot_id": lead["id"], "shot_index": lead["index"], "clip_id": lead["clip_id"],
                                    "lines": describe_scene(sc), "traits": {k: round(float(v), 3) for k, v in sc["traits"].items()}}
        proj.state["scene_profile"] = proj.state["scene_read"]["traits"]
        proj.save()
    say(1.0, "Looks ready")
    return chosen
