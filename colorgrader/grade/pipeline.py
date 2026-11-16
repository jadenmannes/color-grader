"""The grading chain, in the order a colorist builds it.

  1. input conversion   camera profile -> linear DaVinci Wide Gamut
  2. camera corrections  profile defaults (linear gain, saturation)
  3. balance             per shot, linear gain and offset
  4. match               per shot, contrast and saturation trims in log
  5. look                shared: contrast, lift/gamma/gain, saturation, curves
  6. skin                shared: hue curves inside the skin band
  7. finish              vignette and grain
  8. output conversion   -> Rec.709 display

Steps 3 to 6 use the operators in color/ops.py, which follow Resolve's math.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..color import ops
from ..color import working as W
from ..color.transforms import InputTransform, OutputTransform, correction_gains
from . import model

F32 = np.float32


def camera_stage(signal_rgb: np.ndarray, idt: InputTransform, corrections: dict[str, Any] | None) -> np.ndarray:
    """Steps 1 and 2 up to linear light. Returns linear DaVinci Wide Gamut."""
    lin = idt.to_linear_working(signal_rgb)
    gains = correction_gains(corrections)
    if not np.allclose(gains, 1.0):
        lin = lin * gains
    return lin


def camera_saturation(corrections: dict[str, Any] | None) -> float:
    return round(50.0 * float((corrections or {}).get("saturation", 1.0)), 1)


def balance_linear(lin: np.ndarray, shot: dict[str, Any]) -> np.ndarray:
    """Step 3: gain and offset on linear light (a Resolve node with Gamma set to Linear)."""
    b = shot["balance"]
    gain = np.asarray(b["gain"], dtype=F32)
    off = (np.asarray(b["offset"], dtype=np.float64) - ops.OFFSET_BASE) * ops.OFFSET_UNIT
    return lin * gain + off.astype(F32)


def shot_stage(lin: np.ndarray, shot: dict[str, Any], cam_sat: float = 50.0) -> np.ndarray:
    """Steps 3 and 4. Linear in, DaVinci Intermediate out."""
    di = W.di_encode(balance_linear(lin, shot))
    di = ops.saturation(di, cam_sat)
    m = shot["match"]
    di = ops.contrast(di, m["contrast"], m.get("pivot", model.PIVOT))
    di = ops.saturation(di, m["saturation"])
    return ops.hsl_curves(di, hue_vs_hue=m.get("hue_vs_hue"), hue_vs_sat=m.get("hue_vs_sat"))


def look_stage(di: np.ndarray, look: dict[str, Any]) -> np.ndarray:
    """Steps 5 and 6 on DaVinci Intermediate values."""
    di = ops.contrast(di, look["contrast"], look.get("pivot", model.PIVOT))
    di = ops.lggo(di, look["lift"], look["gamma"], look["gain"])
    di = ops.saturation(di, look["saturation"])
    di = ops.hsl_curves(di, lum_vs_sat=look.get("lum_vs_sat"))
    di = ops.hsl_curves(di, hue_vs_sat=look.get("hue_vs_sat"), hue_vs_hue=look.get("hue_vs_hue"))
    skin = look.get("skin") or {}
    di = ops.hsl_curves(di, hue_vs_sat=skin.get("hue_vs_sat"), hue_vs_hue=skin.get("hue_vs_hue"))
    return ops.hsl_curves(di, sat_vs_sat=look.get("sat_vs_sat"))


def vignette_mask(height: int, width: int, start: float) -> np.ndarray:
    """0 in the center, 1 in the corners, smooth in between."""
    y = (np.arange(height, dtype=F32) + 0.5) / height * 2.0 - 1.0
    x = (np.arange(width, dtype=F32) + 0.5) / width * 2.0 - 1.0
    rr = np.sqrt((x[None, :] ** 2 + y[:, None] ** 2) / 2.0)
    t = np.clip((rr - start) / max(1e-3, 1.0 - start), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(F32)


def grain_field(height: int, width: int, size: float, seed: int = 11) -> np.ndarray:
    """Zero-mean monochrome grain with unit standard deviation."""
    import cv2

    rng = np.random.default_rng(seed)
    g = rng.standard_normal((height, width)).astype(F32)
    sigma = 0.55 * max(0.3, size) * max(1.0, width / 1920.0)
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    return g / max(float(g.std()), 1e-6)


def finish_stage(di: np.ndarray, finish: dict[str, Any], seed: int = 11) -> np.ndarray:
    """Step 7 on a full frame (H, W, 3) in DaVinci Intermediate."""
    if di.ndim != 3:
        return di
    h, w = di.shape[:2]
    v = finish.get("vignette") or {}
    if v.get("enabled") and v.get("amount", 0) > 0:
        di = di - (vignette_mask(h, w, float(v.get("start", 0.45))) * F32(float(v["amount"]) * W.STOP))[..., None]
    g = finish.get("grain") or {}
    if g.get("enabled") and g.get("amount", 0) > 0:
        di = di + (grain_field(h, w, float(g.get("size", 1.0)), seed) * F32(float(g["amount"]) / 100.0))[..., None]
    return di


def render(signal_rgb: np.ndarray, idt: InputTransform, odt: OutputTransform, corrections: dict[str, Any] | None,
           shot: dict[str, Any] | None = None, look: dict[str, Any] | None = None, finish: dict[str, Any] | None = None,
           stage: str = "graded") -> np.ndarray:
    """Render display R'G'B' at a named stage: converted, balanced, graded."""
    lin = camera_stage(signal_rgb, idt, corrections)
    cam_sat = camera_saturation(corrections)
    if stage == "converted" or shot is None:
        di = ops.saturation(W.di_encode(lin), cam_sat)
        return odt(di)
    di = shot_stage(lin, shot, cam_sat)
    if stage == "balanced" or look is None:
        return odt(di)
    di = look_stage(di, look)
    if finish:
        di = finish_stage(di, finish)
    return odt(di)


def to_jpeg(display_rgb: np.ndarray, quality: int = 92, seed: int = 1) -> bytes:
    """Display R'G'B' float -> JPEG bytes, dithered to hide 8-bit steps."""
    import cv2

    rgb = np.clip(display_rgb, 0.0, 1.0) * 255.0
    noise = np.random.default_rng(seed).random(rgb.shape[:2], dtype=np.float32)[..., None] - 0.5
    img = np.clip(np.round(rgb + noise * 0.9), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]),
                           [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
