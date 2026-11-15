"""Face detection and skin sampling.

Faces come from OpenCV's YuNet detector (model file shipped with the app, MIT
license). Skin pixels are the dominant color cluster in the middle of each
face box, so the sample works under any color cast.
"""
from __future__ import annotations

import threading

import numpy as np

from .. import paths

F32 = np.float32
_MODEL = paths.SHIPPED_DATA / "models" / "face_detection_yunet_2023mar.onnx"
_lock = threading.Lock()
_detector = None


def _get():
    global _detector
    if _detector is None:
        import cv2

        _detector = cv2.FaceDetectorYN.create(str(_MODEL), "", (320, 320), 0.7, 0.3, 50)
    return _detector


def detect_faces(display_rgb: np.ndarray, min_size: float = 0.035) -> list[list[float]]:
    """Faces in a display-referred R'G'B' frame. Returns [x, y, w, h, score] in pixels."""
    import cv2

    h, w = display_rgb.shape[:2]
    bgr = (np.clip(display_rgb[..., ::-1], 0, 1) * 255.0 + 0.5).astype(np.uint8)
    bgr = np.ascontiguousarray(bgr)
    with _lock:
        det = _get()
        det.setInputSize((w, h))
        try:
            _, faces = det.detect(bgr)
        except cv2.error:
            return []
    out = []
    for f in (faces if faces is not None else []):
        x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
        if fh < min_size * h:
            continue
        out.append([x, y, fw, fh, score])
    return out


def skin_pixels(frame: np.ndarray, faces: list[list[float]], limit: int = 4000) -> np.ndarray:
    """Skin samples (N, 3) from the central part of each face box, in the frame's own color space.

    Keeps the dominant cluster: drops eyes, brows, hair, teeth and highlights.
    """
    h, w = frame.shape[:2]
    out = []
    for x, y, fw, fh, _ in faces:
        x0, x1 = int(max(0, x + 0.22 * fw)), int(min(w, x + 0.78 * fw))
        y0, y1 = int(max(0, y + 0.20 * fh)), int(min(h, y + 0.85 * fh))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        px = frame[y0:y1, x0:x1].reshape(-1, 3)
        lum = px @ np.array([0.2126, 0.7152, 0.0722], dtype=F32)
        lo, hi = np.percentile(lum, [25, 92])
        px = px[(lum >= lo) & (lum <= hi)]
        if len(px) < 8:
            continue
        ch = px - (px @ np.array([0.2126, 0.7152, 0.0722], dtype=F32))[:, None]
        med = np.median(ch, axis=0)
        dist = np.linalg.norm(ch - med, axis=1)
        keep = dist <= np.percentile(dist, 70)
        out.append(px[keep])
    if not out:
        return np.zeros((0, 3), dtype=F32)
    px = np.concatenate(out)
    if len(px) > limit:
        px = px[np.random.default_rng(3).choice(len(px), limit, replace=False)]
    return px.astype(F32)
