"""Scopes rendered from display R'G'B' frames: waveform, RGB parade, vectorscope, histogram.

All scopes return BGR uint8 images (OpenCV order) ready for PNG encoding.
"""
from __future__ import annotations

import numpy as np

from ..color.ycbcr import rgb_to_ycbcr
from .neutral import SKIN_LINE

F32 = np.float32
BG = (22, 20, 19)
GRID = (70, 66, 62)
TEXT = (150, 146, 140)
V_LO, V_HI = -0.05, 1.05


def _prep(rgb: np.ndarray, max_w: int = 512) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    if w > max_w:
        rgb = cv2.resize(rgb, (max_w, max(2, int(round(h * max_w / w)))), interpolation=cv2.INTER_AREA)
    return rgb.astype(F32, copy=False)


def _trace(values: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """values: (H, W) in signal units. Returns a (rows, cols) density map, 0..1."""
    h, w = values.shape
    target = cols
    cols = min(cols, w)  # narrow frames: bin at their own width, then stretch
    col = np.minimum((np.arange(w) * cols) // w, cols - 1)
    row = np.clip(((V_HI - values) / (V_HI - V_LO) * (rows - 1)).round().astype(np.int32), 0, rows - 1)
    idx = row * cols + col[None, :]
    dens = np.bincount(idx.ravel(), minlength=rows * cols).reshape(rows, cols).astype(F32)
    scale = h * w / float(cols * rows)
    out = 1.0 - np.exp(-dens / (scale * 6.0 + 1e-6))
    if cols != target:
        import cv2

        out = cv2.resize(out, (target, rows), interpolation=cv2.INTER_LINEAR)
    return out


def _wave_grid(img: np.ndarray, x0: int, width: int, labels: bool = True) -> None:
    import cv2

    rows = img.shape[0]
    for pct in (0, 25, 50, 75, 100):
        y = int(round((V_HI - pct / 100.0) / (V_HI - V_LO) * (rows - 1)))
        cv2.line(img, (x0, y), (x0 + width - 1, y), GRID, 1)
        if labels:
            cv2.putText(img, str(pct), (x0 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, TEXT, 1, cv2.LINE_AA)


def waveform(rgb: np.ndarray, mode: str = "rgb", cols: int = 512, rows: int = 256) -> np.ndarray:
    """Waveform. mode 'luma' shows Y', mode 'rgb' overlays the three channels."""
    rgb = _prep(rgb, cols)
    img = np.empty((rows, cols, 3), dtype=np.uint8)
    img[:] = BG
    _wave_grid(img, 0, cols)
    acc = img.astype(F32)
    if mode == "luma":
        y, _, _ = rgb_to_ycbcr(rgb)
        d = _trace(y, cols, rows)[..., None]
        acc += d * np.array([200, 235, 200], dtype=F32)
    else:
        for c, color in enumerate(((60, 60, 255), (60, 235, 60), (255, 110, 60))):  # BGR for R, G, B
            d = _trace(rgb[..., c], cols, rows)[..., None]
            acc += d * np.array(color, dtype=F32) * 0.85
    return np.clip(acc, 0, 255).astype(np.uint8)


def parade(rgb: np.ndarray, cols: int = 512, rows: int = 256) -> np.ndarray:
    rgb = _prep(rgb, cols)
    each = cols // 3
    img = np.empty((rows, each * 3, 3), dtype=np.uint8)
    img[:] = BG
    for c in range(3):
        _wave_grid(img, c * each, each, labels=(c == 0))
    acc = img.astype(F32)
    for c, color in enumerate(((70, 70, 255), (70, 235, 70), (255, 120, 70))):
        d = _trace(rgb[..., c], each, rows)[..., None]
        acc[:, c * each:(c + 1) * each] += d * np.array(color, dtype=F32)
    return np.clip(acc, 0, 255).astype(np.uint8)


def vectorscope(rgb: np.ndarray, size: int = 360) -> np.ndarray:
    """Vectorscope (Cb right, Cr up) with 75% targets and the skin tone line."""
    import cv2

    rgb = _prep(rgb, 512)
    _, cb, cr = rgb_to_ycbcr(np.clip(rgb, 0, 1))
    half = (size - 1) / 2.0
    radius = half * 0.94          # chroma 0.5 sits at the circle
    x = np.clip((half + cb / 0.5 * radius).round().astype(np.int32), 0, size - 1)
    y = np.clip((half - cr / 0.5 * radius).round().astype(np.int32), 0, size - 1)
    idx = (y * size + x).ravel()
    dens = np.bincount(idx, minlength=size * size).astype(F32)
    n = max(dens.sum(), 1.0)
    d = 1.0 - np.exp(-dens / (n / (size * size) * 25.0 + 1e-6))
    # color the trace with the average color of the pixels landing in each cell
    flat = np.clip(rgb.reshape(-1, 3), 0, 1)
    col = np.stack([np.bincount(idx, weights=flat[:, c], minlength=size * size) for c in range(3)], axis=-1)
    col = col / np.maximum(dens[:, None], 1.0)
    col = col / np.maximum(col.max(axis=1, keepdims=True), 1e-3)
    col = 0.35 + 0.65 * col
    img = np.empty((size, size, 3), dtype=np.uint8)
    img[:] = BG
    c0 = (int(round(half)), int(round(half)))
    cv2.circle(img, c0, int(radius), GRID, 1, cv2.LINE_AA)
    cv2.circle(img, c0, int(radius * 0.5), (48, 46, 44), 1, cv2.LINE_AA)
    cv2.line(img, (c0[0] - int(radius), c0[1]), (c0[0] + int(radius), c0[1]), (48, 46, 44), 1)
    cv2.line(img, (c0[0], c0[1] - int(radius)), (c0[0], c0[1] + int(radius)), (48, 46, 44), 1)
    a = np.radians(SKIN_LINE)
    cv2.line(img, c0, (int(round(half + np.cos(a) * radius)), int(round(half - np.sin(a) * radius))), (120, 170, 215), 1, cv2.LINE_AA)
    targets = {"R": (0.75, 0, 0), "Yl": (0.75, 0.75, 0), "G": (0, 0.75, 0), "Cy": (0, 0.75, 0.75), "B": (0, 0, 0.75), "Mg": (0.75, 0, 0.75)}
    for name, t in targets.items():
        _, tcb, tcr = rgb_to_ycbcr(np.array([[t]], dtype=F32))
        tx, ty = int(round(half + float(tcb[0, 0]) / 0.5 * radius)), int(round(half - float(tcr[0, 0]) / 0.5 * radius))
        bgr = tuple(int(v * 255 / 0.75 * 0.8) for v in t[::-1])
        cv2.rectangle(img, (tx - 5, ty - 5), (tx + 5, ty + 5), bgr, 1, cv2.LINE_AA)
        cv2.putText(img, name, (tx + 8, ty + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT, 1, cv2.LINE_AA)
    acc = img.astype(F32).reshape(-1, 3)
    acc += d[:, None] * col[:, ::-1] * 255.0
    return np.clip(acc, 0, 255).reshape(size, size, 3).astype(np.uint8)


def histogram(rgb: np.ndarray, cols: int = 512, rows: int = 180) -> np.ndarray:
    import cv2

    rgb = _prep(rgb, 512)
    img = np.empty((rows, cols, 3), dtype=np.uint8)
    img[:] = BG
    for pct in (0, 25, 50, 75, 100):
        x = int(round((pct / 100.0 - V_LO) / (V_HI - V_LO) * (cols - 1)))
        cv2.line(img, (x, 0), (x, rows - 1), GRID, 1)
    acc = img.astype(F32)
    bins = 256
    y, _, _ = rgb_to_ycbcr(rgb)
    chans = [(rgb[..., 0], (60, 60, 255)), (rgb[..., 1], (60, 225, 60)), (rgb[..., 2], (255, 110, 60)), (y, (215, 215, 215))]
    hs = []
    for v, _ in chans:
        h, _ = np.histogram(v, bins=bins, range=(V_LO, V_HI))
        hs.append(np.convolve(h.astype(F32), np.ones(3) / 3, mode="same"))
    top = max(float(np.percentile(np.concatenate(hs), 99.5)), 1.0)
    xs = np.linspace(0, cols - 1, bins)
    for (v, color), h in zip(chans, hs):
        hh = np.clip(h / top, 0, 1)
        pts = np.stack([xs, (rows - 1) - hh * (rows - 6)], axis=1).round().astype(np.int32)
        if color == (215, 215, 215):
            line = np.zeros_like(img)
            cv2.polylines(line, [pts], False, color, 1, cv2.LINE_AA)
            acc = np.maximum(acc, line.astype(F32))
        else:
            poly = np.concatenate([[[0, rows - 1]], pts, [[cols - 1, rows - 1]]]).astype(np.int32)
            layer = np.zeros_like(img)
            cv2.fillPoly(layer, [poly], color, cv2.LINE_AA)
            acc += layer.astype(F32) * 0.55
    return np.clip(acc, 0, 255).astype(np.uint8)


SCOPES = {"waveform": waveform, "parade": parade, "vectorscope": vectorscope, "histogram": histogram}


def render(kind: str, rgb: np.ndarray) -> bytes:
    """PNG bytes of one scope."""
    import cv2

    if kind not in SCOPES:
        raise KeyError(kind)
    ok, buf = cv2.imencode(".png", SCOPES[kind](rgb))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()
