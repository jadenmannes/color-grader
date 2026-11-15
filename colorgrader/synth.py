"""Synthetic footage for tests and demos.

Builds scene-linear frames (procedural, or from a photo) and records them
through a camera profile from the library: curve, gamut, range, bit depth.
The result is a real video file with the camera's encoding, so the whole app
runs on it: metadata matching, decode, input conversion, grading.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import library, media
from .color import working as W
from .color.gamuts import apply_matrix, conversion_matrix
from .color.ycbcr import KR_KB, absolute_to_signal

F32 = np.float32


# ------------------------------------------------------------------ scenes
def procedural_scene(width: int = 640, height: int = 360, seed: int = 1, skin: bool = True,
                     sky: bool = True) -> np.ndarray:
    """A scene-linear Rec.709 test scene: sky gradient, gray wall, foliage, color patches, skin."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(F32)
    u, v = xx / (width - 1), yy / (height - 1)
    img = np.zeros((height, width, 3), dtype=F32)
    horizon = 0.45
    if sky:
        t = np.clip(v / horizon, 0, 1)[..., None]
        top = np.array([0.10, 0.22, 0.55], dtype=F32)
        low = np.array([0.55, 0.68, 0.85], dtype=F32)
        img = top * (1 - t) + low * t
    else:
        img[:] = np.array([0.30, 0.30, 0.30], dtype=F32) * (0.6 + 0.6 * (1 - v[..., None]))
    ground = v > horizon
    wall = ground & (u < 0.45)
    foliage = ground & (u >= 0.45)
    shade = (0.75 + 0.5 * (1 - (v - horizon) / (1 - horizon)))[..., None]
    img = np.where(wall[..., None], np.array([0.32, 0.32, 0.32], dtype=F32) * shade, img)
    leaf = np.array([0.07, 0.16, 0.035], dtype=F32) * (0.7 + 0.8 * rng.random((height, width, 1)).astype(F32))
    img = np.where(foliage[..., None], leaf, img)
    # neutral ramp strip along the bottom: black to bright white
    strip = v > 0.9
    ramp = (0.005 + 1.6 * u ** 2.2)[..., None] * np.ones(3, dtype=F32)
    img = np.where(strip[..., None], ramp, img)
    # color patches
    patches = [(0.75, 0.10, 0.08), (0.80, 0.55, 0.05), (0.10, 0.45, 0.12), (0.05, 0.35, 0.55), (0.08, 0.10, 0.50), (0.55, 0.08, 0.40),
               (0.02, 0.02, 0.02), (0.85, 0.85, 0.85)]
    pw, ph = width // 12, height // 9
    for i, col in enumerate(patches):
        x0 = int(width * 0.05) + i * (pw + 4)
        y0 = int(height * 0.74)
        img[y0:y0 + ph, x0:x0 + pw] = np.array(col, dtype=F32)
    if skin:
        for cx, cy, tone in ((0.30, 0.55, (0.42, 0.26, 0.18)), (0.62, 0.58, (0.22, 0.12, 0.075))):
            d = ((u - cx) / 0.075) ** 2 + ((v - cy) / 0.16) ** 2
            m = np.clip(1.2 - d, 0, 1)[..., None]
            lit = 0.75 + 0.5 * np.clip((cx + 0.06 - u) / 0.12, 0, 1)[..., None]
            img = img * (1 - np.minimum(m * 3, 1)) + np.array(tone, dtype=F32) * lit * np.minimum(m * 3, 1)
    return np.ascontiguousarray(img.astype(F32))


def photo_to_scene(path: str | Path, width: int, height: int) -> np.ndarray:
    """Approximate scene-linear Rec.709 light from a finished sRGB photo."""
    import cv2

    rgb, _ = media.read_still(path)
    h, w = rgb.shape[:2]
    scale = max(width / w, height / h)
    rgb = cv2.resize(rgb, (int(np.ceil(w * scale)), int(np.ceil(h * scale))), interpolation=cv2.INTER_CUBIC)
    y0, x0 = (rgb.shape[0] - height) // 2, (rgb.shape[1] - width) // 2
    rgb = np.clip(rgb[y0:y0 + height, x0:x0 + width], 0, 1)
    lin = library.get().curve("srgb").decode(rgb)
    scene = W.tm_inverse(np.minimum(lin, 0.985), 600.0)
    return np.ascontiguousarray((0.18 * np.power(np.maximum(scene, 0) / 0.18, 1.0 / 1.15)).astype(F32))


def ken_burns(scene: np.ndarray, frames: int, zoom: float = 0.06) -> Iterable[np.ndarray]:
    """Slow push-in over a still scene."""
    import cv2

    h, w = scene.shape[:2]
    for i in range(frames):
        k = 1.0 + zoom * (i / max(frames - 1, 1))
        cw, ch = w / k, h / k
        x0, y0 = (w - cw) / 2, (h - ch) / 2
        m = np.array([[k, 0, -x0 * k], [0, k, -y0 * k]], dtype=np.float32)
        yield cv2.warpAffine(scene, m, (w, h), flags=cv2.INTER_LINEAR)


# --------------------------------------------------------------- recording
def record(frames: Iterable[np.ndarray], out_path: str | Path, *, curve: str | dict, gamut: str | dict, fps: float = 25.0,
           bit_depth: int = 10, chroma: str = "422", full_range: bool = True, matrix: str = "bt709", codec: str = "libx265",
           exposure: float = 0.0, wb_gains=(1.0, 1.0, 1.0), flare: float = 0.0, noise: float = 0.0, display_look: dict | None = None,
           tags: dict[str, str] | None = None, color_tags: dict[str, str] | None = None, crf: int = 12, seed: int = 7,
           max_input_nits: float = 1000.0) -> Path:
    """Encode scene-linear Rec.709 frames as a camera would.

    display_look: for finished-picture profiles, {"contrast": c, "saturation": s} applied in the
    working space before the display encoding (a stand-in for a film simulation).
    """
    from .color import ops
    from .color.transforms import OutputTransform

    lib = library.get()
    cv, gm = lib.curve(curve), lib.gamut(gamut)
    rec709, dwg = lib.gamut("rec709"), lib.gamut(W.WORKING_GAMUT)
    m_cam = conversion_matrix(rec709, gm)
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    pix = {("420", 8): "yuv420p", ("422", 8): "yuv422p", ("444", 8): "yuv444p",
           ("420", 10): "yuv420p10le", ("422", 10): "yuv422p10le", ("444", 10): "yuv444p10le"}[(chroma, bit_depth)]
    proc = None
    kr, kb = KR_KB[matrix]
    kg = 1 - kr - kb
    peak = float((1 << bit_depth) - 1)
    s = float(1 << (bit_depth - 8))
    dtype = np.uint16 if bit_depth > 8 else np.uint8
    gains = np.asarray(wb_gains, dtype=F32) * F32(2.0 ** exposure)
    for scene in frames:
        h, w = scene.shape[:2]
        lin = scene * gains + F32(flare)
        if noise > 0:
            lin = lin + rng.normal(0, 1, lin.shape).astype(F32) * F32(noise) * np.sqrt(np.maximum(lin, 0) + 0.002)
        if cv.kind == "display":
            di = W.di_encode(apply_matrix(lin, conversion_matrix(rec709, dwg)))
            if display_look:
                di = ops.contrast(di, display_look.get("contrast", 1.0), W.MID_GRAY)
                di = ops.saturation(di, 50.0 * display_look.get("saturation", 1.0))
            ot = OutputTransform("rec709-gamma24", max_input_nits)
            disp = apply_matrix(ot.display_linear(di), conversion_matrix(rec709, gm))
            code = cv.encode(np.clip(disp, 0, 1))
        elif cv.kind == "pq":
            disp = W.pq_scene_to_display(apply_matrix(np.maximum(lin, 0), m_cam), cv.params, gm.luma_weights())
            code = cv.encode(disp)
        else:
            code = cv.encode(apply_matrix(lin, m_cam))
        if cv.code_values == "absolute":
            code = absolute_to_signal(code, full_range, bit_depth)
        r, g, b = code[..., 0], code[..., 1], code[..., 2]
        yv = kr * r + kg * g + kb * b
        cb = (b - yv) / (2 * (1 - kb))
        cr = (r - yv) / (2 * (1 - kr))
        if full_range:
            yc, cbc, crc = yv * peak, cb * peak + 128 * s, cr * peak + 128 * s
        else:
            yc, cbc, crc = yv * 219 * s + 16 * s, cb * 224 * s + 128 * s, cr * 224 * s + 128 * s
        if chroma in ("422", "420"):
            cbc = 0.5 * (cbc[:, 0::2] + cbc[:, 1::2])
            crc = 0.5 * (crc[:, 0::2] + crc[:, 1::2])
        if chroma == "420":
            cbc = 0.5 * (cbc[0::2] + cbc[1::2])
            crc = 0.5 * (crc[0::2] + crc[1::2])
        planes = []
        for p in (yc, cbc, crc):
            d = rng.random(p.shape).astype(F32) - 0.5  # dither before rounding
            planes.append(np.clip(np.round(p + d * 0.6), 0, peak).astype(dtype).tobytes())
        if proc is None:
            ct = {"color_primaries": "bt709", "color_trc": "bt709", "colorspace": matrix if matrix != "bt601" else "smpte170m"}
            ct.update(color_tags or {})
            cmd = [media.find_binary("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}",
                   "-r", f"{fps}", "-color_range", "pc" if full_range else "tv", "-i", "pipe:0", "-an", "-c:v", codec, "-pix_fmt", pix,
                   "-color_range", "pc" if full_range else "tv", "-color_primaries", ct["color_primaries"],
                   "-color_trc", ct["color_trc"], "-colorspace", ct["colorspace"]]
            if codec == "libx265":
                rng_name = "full" if full_range else "limited"
                cmd += ["-preset", "fast", "-crf", str(crf), "-tag:v", "hvc1", "-x265-params", f"log-level=error:range={rng_name}"]
            elif codec == "libx264":
                cmd += ["-preset", "fast", "-crf", str(crf)]
            elif codec == "prores_ks":
                cmd += ["-profile:v", "3"]
            for k, val in (tags or {}).items():
                cmd += ["-metadata", f"{k}={val}"]
            cmd += [str(out_path)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        proc.stdin.write(b"".join(planes))
    if proc is None:
        raise ValueError("no frames")
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg encode failed: {err[:500]}")
    return out_path


def concat_scenes(scenes: list[np.ndarray], frames_each: int) -> Iterable[np.ndarray]:
    for sc in scenes:
        yield from ken_burns(sc, frames_each)
