"""Rebuild compressed color from the full-resolution brightness.

Heavily compressed 4:2:0 and 4:2:2 files (an edit exported at a low bitrate, a phone clip, a download) store their
color as a mosaic of flat blocks, 16 by 16 pixels for H.264. In a flat log picture nobody sees them. A grade that adds
saturation turns them into visible squares on skin and staircase edges on colored clothes.

The fix is the guided filter (He, Sun, Tang 2010) in its fast form (He, Sun 2015): inside every small window the color
is modeled as a straight-line function of the brightness, which the file keeps at full resolution with far less
damage. Where brightness is smooth, the color settles to the window's average and the block steps vanish. At a
brightness edge, the color edge snaps to it. Measured on a real 8-bit S-Log3 H.264 file at 25 Mbps: color block steps
4 to 5 times the in-block changes before, 1.0 after (no blocks), knit and skin detail kept.

The viewer, the full-quality still and the export all run this same code, so the three show the same picture.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32
BLOCK_RADIUS = 8      # source pixels: half of the 16-pixel color blocks of H.264 (H.265 blocks are similar or larger)
EPS = 1e-4            # in 0..1 code values, squared. Smaller keeps color edges sharper where the brightness has an edge
AUTO_MAX_BPP = 0.2    # Auto rebuilds 8-bit 4:2:0 and 4:2:2 files below this many bits per pixel (H.264 terms).
                      # Camera files sit at 0.3 to 0.5 and keep their color as recorded: on clean footage the rebuild
                      # costs some color accuracy (measured: color error up to 13% higher, detail unchanged).


def rebuild(buf: np.ndarray, w: int, h: int, scale: float = 1.0) -> np.ndarray:
    """buf: one frame as yuv444p16le planes (3, h, w), flattened uint16, as media.read_frame returns it.
    scale: decoded width / source width, so the window follows the source's blocks at any preview size.
    Returns the same layout with Cb and Cr rebuilt. The brightness plane stays untouched."""
    import cv2

    planes = buf.reshape(3, h, w)
    r = max(1, int(round(BLOCK_RADIUS * scale)))
    s = max(1, min(4, r // 2))                       # work on an s-times smaller copy: same result, a fraction of the time
    sw, sh = max(1, w // s), max(1, h // s)
    rs = max(1, r // s)
    k = (2 * rs + 1, 2 * rs + 1)
    unit = F32(1.0 / 65535.0)
    y = planes[0].astype(F32)
    y *= unit
    ys = cv2.resize(y, (sw, sh), interpolation=cv2.INTER_AREA) if s > 1 else y
    box = lambda a: cv2.boxFilter(a, -1, k, borderType=cv2.BORDER_REFLECT)
    m_y = box(ys)
    var_y = box(ys * ys) - m_y * m_y
    out = np.empty_like(planes)
    out[0] = planes[0]
    for i in (1, 2):
        # shrink first, then convert: no full-size float copy of the color planes
        cs = (cv2.resize(planes[i], (sw, sh), interpolation=cv2.INTER_AREA) if s > 1 else planes[i]).astype(F32)
        cs *= unit
        m_c = box(cs)
        a = (box(ys * cs) - m_y * m_c) / (var_y + F32(EPS))
        b = m_c - a * m_y
        a, b = box(a), box(b)
        if s > 1:
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
        # in place: at 4K every full-size float temporary is 33 MB, and the export runs this on several frames at once
        a *= y
        a += b
        a *= F32(65535.0)
        np.rint(a, out=a)
        np.clip(a, 0, 65535, out=a)
        out[i] = a
    return out.reshape(-1)


def block_ratio(plane: np.ndarray, period: int = 8) -> float:
    """How much larger the steps at block borders are than the changes inside blocks (1.0: no block grid)."""
    p = plane.astype(F32)
    d = np.abs(np.diff(p, axis=1))
    inside = np.delete(d, np.s_[period - 1::period], axis=1).mean()
    return float(d[:, period - 1::period].mean() / max(float(inside), 1e-6))
