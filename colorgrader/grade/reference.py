"""The reference: a still, a film screenshot or a short clip with the look to copy."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .. import library, media
from ..analysis import measure as measure_mod
from ..analysis import names
from ..analysis.neutral import vector_angle
from ..color.transforms import OutputTransform, input_transform_for
from ..color.ycbcr import rgb_to_ycbcr
from . import lookfit

F32 = np.float32
PIXELS = 230400


def load(path: str | Path, odt: OutputTransform) -> dict[str, Any]:
    """Read the reference. Returns display frame, working-space sample and file info.

    The reference is a finished picture. Its code values are what the viewer sees, so they
    go through the inverse of the app's own output conversion. A grade matched this way
    shows the same code values as the reference on any screen.
    """
    path = Path(path)
    info = media.probe(path)
    frames = []
    note = ""
    if info["is_image"]:
        rgb, meta = media.read_still(path)
        note = meta.get("note", "")
        frames.append(media.resize_rgb(rgb, pixels=PIXELS))
        kind = "still"
    else:
        kind = "clip"
        n = 5 if info["duration"] > 1.0 else 2
        hdr = info["color_transfer"] in ("arib-std-b67", "smpte2084")
        for i in range(n):
            t = info["duration"] * (i + 0.5) / n
            try:
                buf, w, h = media.read_frame(info, t, max_width=media.fit_size(info, pixels=PIXELS)[0])
            except media.MediaError:
                continue
            sig = media.raw_to_rgb(buf, w, h, info)
            if hdr:  # HDR reference clip: tone map it the way footage is tone mapped
                prof = {"input": {"type": "curve", "curve": "hlg" if info["color_transfer"] == "arib-std-b67" else "pq", "gamut": "rec2020"}}
                idt = input_transform_for(prof, info["full_range"], info["bit_depth"], odt.max_input_nits)
                sig = odt(idt(sig))
                note = "HDR clip, tone mapped to Rec.709"
            frames.append(np.clip(sig, 0.0, 1.0).astype(F32))
        if not frames:
            raise media.MediaError(f"Could not read frames from {path.name}")
    display = frames[len(frames) // 2]
    rng = np.random.default_rng(21)
    samples = []
    per = max(4000, 60000 // len(frames))
    for f in frames:
        flat = f.reshape(-1, 3)
        samples.append(flat[rng.choice(len(flat), min(per, len(flat)), replace=False)])
    disp_sample = np.concatenate(samples).astype(F32)
    di = odt.inverse(disp_sample)
    return {"info": info, "kind": kind, "note": note, "display": display.astype(F32), "display_sample": disp_sample, "di": di}


def thumbs_luma(frame_display: np.ndarray) -> np.ndarray:
    """Rank-normalized 24x14 luma thumbnail: compares picture content, ignores the grade."""
    import cv2

    y = frame_display @ np.array([0.2126, 0.7152, 0.0722], dtype=F32)
    small = cv2.resize(y, (24, 14), interpolation=cv2.INTER_AREA).ravel()
    rank = np.argsort(np.argsort(small)).astype(np.float64)
    return (rank - rank.mean()) / (rank.std() + 1e-9)


def content_similarity(ref_display: np.ndarray, frame_display: np.ndarray) -> float:
    """Correlation of tone layout between the reference and a footage frame. 1.0 = same picture."""
    a, b = thumbs_luma(ref_display), thumbs_luma(frame_display)
    return float((a * b).mean())


def zone_tints_display(disp: np.ndarray) -> list[tuple[float, float]]:
    """Color of near-neutral tones in shadows, midtones, highlights on the display (Cb, Cr)."""
    y, cb, cr = rgb_to_ycbcr(np.clip(disp, 0, 1))
    order = np.argsort(y)
    rank = np.empty(len(y), dtype=F32)
    rank[order] = np.linspace(0, 1, len(y), dtype=F32)
    w = np.exp(-(np.hypot(cb, cr) / 0.06) ** 2)
    out = []
    for lo, hi in lookfit.ZONES:
        sel = (rank >= lo) & (rank < hi)
        tot = max(float(w[sel].sum()), 1e-6)
        out.append((float((w[sel] * cb[sel]).sum() / tot), float((w[sel] * cr[sel]).sum() / tot)))
    return out


def describe(disp_sample: np.ndarray, m: dict[str, Any], di: np.ndarray) -> dict[str, Any]:
    """The reference look in plain words, for example 'lifted blacks, teal shadows, warm highlights, muted greens'."""
    words: list[str] = []
    black, white = m["black_point"]["level"], m["white_point"]["level"]
    crushed = m["clipping"]["shadows"]
    if black >= 4.0:
        words.append("lifted blacks")
    elif crushed >= 2.0:
        words.append("crushed blacks")
    elif black <= 1.2:
        words.append("deep blacks")
    spread = m["contrast"]["display_spread"]
    words.append("high contrast" if spread >= 58 else "soft contrast" if spread <= 34 else "medium contrast")
    tints = zone_tints_display(disp_sample)
    mags = [float(np.hypot(*t)) for t in tints]
    zone_words = []
    for name, t, mag in zip(lookfit.ZONE_NAMES, tints, mags):
        if mag >= 0.010:
            zone_words.append((name, names.tint_name(*t)))
    if len(zone_words) == 3 and len({z[1] for z in zone_words}) == 1:
        words.append(f"{zone_words[0][1]} cast overall")
    else:
        for name, tint in zone_words:
            words.append(f"{tint} {name}")
    if not zone_words:
        words.append("neutral tones")
    sat = m["saturation"]["overall"]
    mono = sat < 1.5
    if mono:
        words.append("monochrome")
    else:
        words.append("muted color" if sat < 9 else "rich color" if sat > 22 else "natural saturation")
        prof = lookfit.hue_profile(di)
        present = prof["mass"] >= 0.04
        if present.sum() >= 3:
            med = float(np.median(prof["chroma"][present]))
            for i in np.nonzero(present)[0]:
                if prof["chroma"][i] < 0.6 * med:
                    words.append(f"muted {names.hue_name(i * 360.0 / lookfit.SECTORS)}s")
        top = [c["name"] for c in m["palette"][:2]]
        if top:
            words.append("palette of " + " and ".join(top))
    if white <= 91.0:
        words.append("soft whites")
    y = disp_sample @ np.array([0.2126, 0.7152, 0.0722], dtype=F32)
    ceil = float(np.percentile(y, 99.6))
    capped = bool(ceil < 0.92 and float((y > ceil - 0.03).mean()) > 0.03)
    seen, uniq = set(), []
    for w_ in words:
        if w_ not in seen:
            seen.add(w_)
            uniq.append(w_)
    # colors the reference itself runs into the edge of the display gamut: bright ones and other saturated ones
    mx, mn = disp_sample.max(axis=1), disp_sample.min(axis=1)
    zero = mn <= 0.004
    return {"words": uniq, "text": ", ".join(uniq), "monochrome": mono, "capped": capped,
            "crushed": crushed / 100.0, "clipped": m["clipping"]["highlights"] / 100.0,
            "zero_bright": float((zero & (mx >= 0.85)).mean()), "zero_mid": float((zero & (mx >= 0.3) & (mx < 0.85)).mean())}


def analyze(path: str | Path, odt: OutputTransform) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Full reference analysis. Returns (state dict, arrays to cache)."""
    ref = load(path, odt)
    m = measure_mod.measure(ref["di"], odt, None, None, with_cast=False)
    desc = describe(ref["display_sample"], m, ref["di"])
    info = ref["info"]
    state = {"path": str(path), "name": Path(path).name, "kind": ref["kind"], "note": ref["note"],
             "width": info["width"], "height": info["height"], "measure": m, "summary": measure_mod.summarize(m),
             "description": desc["text"], "words": desc["words"],
             "flags": {k: desc[k] for k in ("monochrome", "capped", "crushed", "clipped", "zero_bright", "zero_mid")}}
    return state, {"display": ref["display"], "di": ref["di"], "display_sample": ref["display_sample"]}
