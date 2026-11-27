"""Step 3 of the grade: group shots into scenes and match them.

Shots of one scene must look filmed at the same moment. Shots from different
cameras get a common base before the look goes on.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from ..color import ops

F32 = np.float32


def signature(di: np.ndarray) -> np.ndarray:
    """Compact picture of a balanced shot: tone histogram plus chroma-weighted hue histogram."""
    lum, c1, c2, hue, chroma = ops.hsl_split(di)
    h_l, _ = np.histogram(np.clip(lum, 0, 0.9), bins=16, range=(0.0, 0.9))
    h_h, _ = np.histogram(hue, bins=12, range=(0.0, 360.0), weights=chroma)
    h_l = h_l / max(h_l.sum(), 1)
    h_h = h_h / max(h_h.sum(), 1e-9)
    return np.concatenate([h_l, h_h]).astype(np.float64)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    tone = 1.0 - 0.5 * np.abs(a[:16] - b[:16]).sum()
    hue = 1.0 - 0.5 * np.abs(a[16:] - b[16:]).sum()
    return float(0.5 * tone + 0.5 * hue)


def _time(text: str) -> float | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def group_scenes(shots: list[dict[str, Any]], sigs: dict[str, np.ndarray], clip_times: dict[str, str]) -> dict[str, int]:
    """Assign a scene number to every shot.

    Consecutive shots of one clip share a scene when they look alike. Shots of
    different clips share a scene when recorded within 20 minutes and alike.
    """
    scenes: list[dict[str, Any]] = []
    out: dict[str, int] = {}
    prev: dict[str, Any] | None = None
    for s in shots:
        sig = sigs[s["id"]]
        t = _time(clip_times.get(s["clip_id"], ""))
        if t is not None:
            t += s["start"]
        best, best_sim = None, 0.0
        for sc in scenes:
            sim = similarity(sig, sc["sig"])
            same_clip_next = prev is not None and prev["clip_id"] == s["clip_id"] and out[prev["id"]] == sc["id"]
            if same_clip_next:
                need = 0.62
            elif sc["time"] is not None and t is not None:
                need = 0.55 if abs(t - sc["time"]) <= 1200 else 0.80
            else:
                need = 0.86  # no timestamps: join clips only when they look nearly identical
            if sim >= need and sim > best_sim:
                best, best_sim = sc, sim
        if best is None:
            best = {"id": len(scenes) + 1, "sig": sig.copy(), "time": t, "n": 0}
            scenes.append(best)
        else:
            best["sig"] = (best["sig"] * best["n"] + sig) / (best["n"] + 1)
            if t is not None and best["time"] is not None:
                best["time"] = 0.5 * (best["time"] + t)
        best["n"] += 1
        out[s["id"]] = best["id"]
        prev = s
    return out


def consensus_casts(group: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Shots of one clip in one scene share the camera's white balance, so they share the correction.

    group items: {"id", "dr", "db", "confidence"}. Confident shots keep most of their own estimate,
    uncertain shots follow the group.
    """
    if len(group) < 2:
        return {g["id"]: (g["dr"], g["db"]) for g in group}
    w = np.array([max(g["confidence"], 0.02) for g in group])
    dr = float((w * np.array([g["dr"] for g in group])).sum() / w.sum())
    db = float((w * np.array([g["db"] for g in group])).sum() / w.sum())
    out = {}
    for g in group:
        own = 0.7 * g["confidence"]
        out[g["id"]] = (own * g["dr"] + (1 - own) * dr, own * g["db"] + (1 - own) * db)
    return out


def pick_hero(candidates: list[dict[str, Any]]) -> str:
    """The reference shot of a scene: log footage first, then firm white balance, then length."""
    def score(c: dict[str, Any]) -> tuple:
        return (1 if c["scene_referred"] else 0, round(c["confidence"], 1), c["duration"])
    return max(candidates, key=score)["id"]


SKIN_FINE = 0.35       # stops: a skin difference this small is aligned in any case
SKIN_CORROBORATE = 0.5  # a larger one needs the whole picture to differ by at least this share of it, the same way


def exposure_trim(hero: dict[str, Any], shot: dict[str, Any]) -> tuple[float, str]:
    """Extra exposure for a shot so it sits with the hero. Items hold 'median' (stops) and 'skin' (stops or None).

    Skin brightness belongs to the person: people's skin differs by two stops and more. A skin difference counts as an
    exposure difference only when it is small, or when the whole picture differs the same way (the same person filmed
    darker or brighter). A larger skin difference with a picture that matches is a different person, kept as filmed."""
    d_pic = hero["median"] - shot["median"]
    if hero.get("skin") is not None and shot.get("skin") is not None:
        d = hero["skin"] - shot["skin"]
        if abs(d) <= SKIN_FINE or (d * d_pic > 0 and abs(d_pic) >= SKIN_CORROBORATE * abs(d)):
            trim = float(np.clip(0.8 * d, -1.0, 1.0))
            if abs(trim) >= 0.08:
                return trim, f"skin brightness matched to the scene's lead shot ({trim:+.2f} stops)"
            return 0.0, ""
        note = (f"skin sits {abs(d):.1f} stops {'darker' if d > 0 else 'brighter'} than in the lead shot while the picture "
                f"differs by {abs(d_pic):.1f}: a different person, so the skin keeps its own brightness")
        trim, why = _median_trim(d_pic)
        return trim, f"{note}; {why}" if why else note
    return _median_trim(d_pic)


def _median_trim(d: float) -> tuple[float, str]:
    if abs(d) > 0.5:
        trim = float(np.clip(0.5 * (d - np.sign(d) * 0.5), -0.6, 0.6))
        if abs(trim) >= 0.08:
            return trim, f"midtones moved toward the scene's lead shot ({trim:+.2f} stops)"
    return 0.0, ""


def tone_stats(di: np.ndarray) -> dict[str, float]:
    """Tone spread and color strength of a balanced sample, for camera matching."""
    lum, _, _, _, chroma = ops.hsl_split(di)
    p10, p90 = np.percentile(lum, [10, 90])
    mid = (lum > p10) & (lum < p90)
    return {"spread": float(p90 - p10), "chroma": float(chroma[mid].mean()) if mid.any() else 0.0}


def camera_match(primary: dict[str, float], other: dict[str, float], kinds_differ: bool) -> tuple[float, float, str]:
    """Contrast and saturation (multipliers) that bring another camera profile to the primary one."""
    # Pooled statistics of different pictures are a rough guide, so the match moves part of the way.
    c = float(np.clip((primary["spread"] / max(other["spread"], 1e-4)) ** 0.6, 0.75, 1.20))
    s = float(np.clip((primary["chroma"] / max(other["chroma"], 1e-5)) ** 0.5, 0.85, 1.15))
    if not kinds_differ:
        # same kind of source: correct clear mismatches only, and only half way
        c = 1.0 + 0.5 * (c - 1.0) if abs(c - 1.0) > 0.12 else 1.0
        s = 1.0 + 0.5 * (s - 1.0) if abs(s - 1.0) > 0.12 else 1.0
    why = ""
    if abs(c - 1.0) >= 0.02 or abs(s - 1.0) >= 0.02:
        why = "matched to the project's main camera before the look"
    return c, s, why
