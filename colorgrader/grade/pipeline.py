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

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import numpy as np

from ..color import ops
from ..color import working as W
from . import controls, relight
from ..color.transforms import InputTransform, OutputTransform, correction_gains
from . import model

F32 = np.float32


def camera_stage(signal_rgb: np.ndarray, idt: InputTransform, corrections: dict[str, Any] | None) -> np.ndarray:
    """Steps 1 and 2 up to linear light. Returns linear DaVinci Wide Gamut."""
    lin = idt.to_linear_working(signal_rgb)
    gains = correction_gains(corrections)
    if not np.allclose(gains, 1.0):
        lin = lin * gains
    return limit_gamut(lin)


_GAMUT_WEIGHTS = np.array([0.25, 0.65, 0.10], dtype=F32)


def limit_gamut(lin: np.ndarray) -> np.ndarray:
    """Bring colors outside the working gamut back to its edge, keeping hue and brightness.

    Real scenes stay untouched: they hold no negative channel. Burned-in graphics and titles
    decoded through a log curve do. A negative channel turns into a far negative log value,
    and every later control reads the color as blacker than black.
    """
    mn = lin.min(axis=-1)
    if not (mn < 0.0).any():
        return lin
    # values under the black level of a log curve are negative in every channel: those stay as they are
    bad = (mn < 0.0) & ((lin @ _GAMUT_WEIGHTS) > 0.0)
    if not bad.any():
        return lin
    out = np.array(lin, dtype=F32, copy=True)
    px = out[bad]
    y = (px @ _GAMUT_WEIGHTS)[:, None]
    t = y / np.maximum(y - px.min(axis=-1, keepdims=True), 1e-9)
    out[bad] = y + t * (px - y)
    return out


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
    di = ops.hsl_curves(di, hue_vs_hue=m.get("hue_vs_hue"), hue_vs_sat=m.get("hue_vs_sat"))
    trim = shot.get("trim")
    if trim:   # this shot's own color controls, after the match and before the shared look
        di = controls.apply(di, trim)
    return di


def exposure_gain(stops: float) -> float:
    """Gain of the look's exposure node, rounded as the recipe prints it."""
    return controls.wb_gains(stops)[0]


def look_stage(di: np.ndarray, look: dict[str, Any]) -> np.ndarray:
    """Steps 5 and 6 on DaVinci Intermediate values: every color control of the look, in node order (controls.py)."""
    return controls.apply(di, look)


@lru_cache(maxsize=4)
def vignette_mask(height: int, width: int, start: float) -> np.ndarray:
    """0 in the center, 1 in the corners, smooth in between. Cached and read-only: strips of a large frame share it."""
    y = (np.arange(height, dtype=F32) + 0.5) / height * 2.0 - 1.0
    x = (np.arange(width, dtype=F32) + 0.5) / width * 2.0 - 1.0
    rr = np.sqrt((x[None, :] ** 2 + y[:, None] ** 2) / 2.0)
    t = np.clip((rr - start) / max(1e-3, 1.0 - start), 0.0, 1.0)
    out = (t * t * (3.0 - 2.0 * t)).astype(F32)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=2)
def grain_field(height: int, width: int, size: float, seed: int = 11) -> np.ndarray:
    """Zero-mean monochrome grain with unit standard deviation. Cached: every strip of a large frame shares one field."""
    import cv2

    rng = np.random.default_rng(seed)
    g = rng.standard_normal((height, width)).astype(F32)
    sigma = 0.55 * max(0.3, size) * max(1.0, width / 1920.0)
    g = cv2.GaussianBlur(g, (0, 0), sigma)
    out = (g / max(float(g.std()), 1e-6)).astype(F32)
    out.setflags(write=False)
    return out


def finish_stage(di: np.ndarray, finish: dict[str, Any], seed: int = 11, frame: tuple[int, int, int] | None = None) -> np.ndarray:
    """Step 7 in DaVinci Intermediate on a full frame (H, W, 3), or on rows y0.. of a frame when frame = (H, W, y0)."""
    if di.ndim != 3:
        return di
    h, w = di.shape[:2]
    fh, fw, y0 = frame or (h, w, 0)
    v = finish.get("vignette") or {}
    if v.get("enabled") and v.get("amount", 0) > 0:
        mask = vignette_mask(fh, fw, float(v.get("start", 0.45)))[y0:y0 + h]
        di = di - (mask * F32(float(v["amount"]) * W.STOP))[..., None]
    g = finish.get("grain") or {}
    if g.get("enabled") and g.get("amount", 0) > 0:
        field = grain_field(fh, fw, float(g.get("size", 1.0)), seed)[y0:y0 + h]
        di = di + (field * F32(float(g["amount"]) / 100.0))[..., None]
    return di


_LUMA709 = np.array([0.2126, 0.7152, 0.0722], dtype=F32)


def detail_values(finish: dict[str, Any] | None) -> tuple[float, float, float]:
    """(denoise, sharpen, clarity) as 0..1."""
    d = (finish or {}).get("detail") or {}
    return tuple(float(np.clip(d.get(k, 0), 0, 100)) / 100.0 for k in ("denoise", "sharpen", "clarity"))


def detail_sigmas(width: int) -> tuple[float, float]:
    """Blur widths in pixels for sharpening and clarity. Both scale with the frame, so every size looks alike."""
    return max(0.6, 1.0 * width / 1920.0), max(4.0, 0.022 * width)


def sharpen_gain(amount: float) -> float:
    return 0.9 * amount


def clarity_gain(amount: float) -> float:
    return 0.45 * amount


def denoise_signal(sig: np.ndarray, amount: float) -> np.ndarray:
    """Preview of the noise reduction on one source frame (H, W, 3). The export uses a temporal filter on the full clip."""
    if sig.ndim != 3 or amount <= 0:
        return sig
    import cv2

    scale = max(0.5, sig.shape[1] / 1920.0)
    y = sig @ _LUMA709
    chroma = sig - y[..., None]
    chroma = cv2.GaussianBlur(chroma, (0, 0), (1.0 + 3.0 * amount) * scale)        # color noise: the eye never misses fine chroma
    y2 = np.ascontiguousarray(y)
    for _ in range(2):   # two gentle passes clean flat areas further and still stop at edges
        y2 = cv2.bilateralFilter(y2, 7, 0.015 + 0.05 * amount, (1.2 + 2.0 * amount) * scale)
    return (y2[..., None] + chroma).astype(F32)


def wide_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur. A wide one runs on a smaller copy and scales back: the result is smooth either way, and a 4K
    clarity blur drops from seconds to milliseconds."""
    import cv2

    k = int(sigma // 6)
    h, w = img.shape[:2]
    if k < 2 or min(h, w) < 16 * k:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    small = cv2.resize(img, (max(1, round(w / k)), max(1, round(h / k))), interpolation=cv2.INTER_AREA)
    # the box average of the downscale already spreads by (k^2 - 1) / 12 pixels squared
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, (max(sigma * sigma - (k * k - 1) / 12.0, 1.0)) ** 0.5 / k))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def detail_display(disp: np.ndarray, sharpen: float, clarity: float) -> np.ndarray:
    """Clarity (wide local contrast) and sharpening on the display luma of a full frame. Colors keep their chroma."""
    if disp.ndim != 3 or (sharpen <= 0 and clarity <= 0):
        return disp
    import cv2

    s_sharp, s_clar = detail_sigmas(disp.shape[1])
    y = np.ascontiguousarray(np.clip(disp, 0.0, 1.0) @ _LUMA709)
    out = y
    if clarity > 0:
        out = out + F32(clarity_gain(clarity)) * (out - wide_blur(out, s_clar))
        out = np.clip(out, 0.0, 1.0)
    if sharpen > 0:
        out = out + F32(sharpen_gain(sharpen)) * (out - cv2.GaussianBlur(out, (0, 0), s_sharp))
    return (disp + (out - y)[..., None]).astype(F32)


STRIP_PIXELS = 1_500_000   # frames above this size render in strips on several cores
STRIP_ROWS_PIXELS = 400_000


def render_threads() -> int:
    return max(1, min(6, (os.cpu_count() or 2) - 1))


def _color(sig: np.ndarray, idt: InputTransform, odt: OutputTransform, corrections: dict[str, Any] | None,
           shot: dict[str, Any] | None, look: dict[str, Any] | None, finish: dict[str, Any] | None, stage: str,
           relight_mask: np.ndarray | None, frame: tuple[int, int, int] | None) -> np.ndarray:
    """Every per-pixel step, from the camera signal to display R'G'B'. Runs on a whole frame or on a strip of rows."""
    lin = camera_stage(sig, idt, corrections)
    cam_sat = camera_saturation(corrections)
    if stage == "converted" or shot is None:
        return odt(ops.saturation(W.di_encode(lin), cam_sat))
    di = shot_stage(lin, shot, cam_sat)
    if stage == "balanced" or look is None:
        return odt(di)
    di = look_stage(di, look)
    if finish:
        on, amount, subject = relight.settings(finish)
        if on and relight_mask is not None and di.ndim == 3:
            di = relight.apply(di, relight_mask, amount, subject)
        di = finish_stage(di, finish, frame=frame)
    return odt(di)


def render(signal_rgb: np.ndarray, idt: InputTransform, odt: OutputTransform, corrections: dict[str, Any] | None,
           shot: dict[str, Any] | None = None, look: dict[str, Any] | None = None, finish: dict[str, Any] | None = None,
           stage: str = "graded", relight_mask: np.ndarray | None = None) -> np.ndarray:
    """Render display R'G'B' at a named stage: converted, balanced, graded.

    A large frame renders in strips of rows on several cores. Every color step works per pixel, so the strips give
    exactly the whole-frame result. Noise reduction, sharpening and clarity look at neighbors: they run on the full frame.
    """
    dn, sharp, clar = detail_values(finish if (stage == "graded" and shot is not None and look is not None) else None)
    if dn > 0:
        signal_rgb = denoise_signal(signal_rgb, dn)
    if signal_rgb.ndim == 3 and relight_mask is not None and relight_mask.shape != signal_rgb.shape[:2]:
        import cv2

        relight_mask = cv2.resize(relight_mask, (signal_rgb.shape[1], signal_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    args = (idt, odt, corrections, shot, look, finish, stage)
    if signal_rgb.ndim != 3 or signal_rgb.shape[0] * signal_rgb.shape[1] <= STRIP_PIXELS or render_threads() < 2:
        disp = _color(signal_rgb, *args, relight_mask, None)
    else:
        h, w = signal_rgb.shape[:2]
        rows = max(16, STRIP_ROWS_PIXELS // w)
        disp = np.empty((h, w, 3), dtype=F32)
        if finish and stage == "graded":   # build the shared full-frame fields once, before the strips ask for them
            g = finish.get("grain") or {}
            if g.get("enabled") and g.get("amount", 0) > 0:
                grain_field(h, w, float(g.get("size", 1.0)), 11)

        def strip(y0: int) -> None:
            y1 = min(h, y0 + rows)
            m = relight_mask[y0:y1] if relight_mask is not None else None
            disp[y0:y1] = _color(signal_rgb[y0:y1], *args, m, (h, w, y0))

        with ThreadPoolExecutor(render_threads()) as pool:
            list(pool.map(strip, range(0, h, rows)))
    return detail_display(disp, sharp, clar)


def to_png(display_rgb: np.ndarray, bits: int = 8, seed: int = 1, compression: int = 1) -> bytes:
    """Display R'G'B' float -> lossless PNG. 8-bit is dithered like the JPEG view, 16-bit keeps the full precision.
    compression 0 encodes fastest (for the screen), 1 makes a smaller file (for saving)."""
    import cv2

    rgb = np.clip(display_rgb, 0.0, 1.0)
    if bits == 16:
        img = np.round(rgb * 65535.0).astype(np.uint16)
    else:
        noise = np.random.default_rng(seed).random(rgb.shape[:2], dtype=np.float32)[..., None] - 0.5
        img = np.clip(np.round(rgb * 255.0 + noise * 0.9), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(img[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, int(compression)])
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


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
