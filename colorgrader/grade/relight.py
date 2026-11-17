"""Subject relight: the background falls into shadow, the face and shoulders stay lit.

The interview technique from the colorist workflow: darken the whole frame, then keep the brightness
inside soft shapes over the face and the body, tracked through the shot. The eye goes to the speaker
without a visible vignette.

Faces come from the analysis (YuNet detector on every analyzed frame, two per second by default). The
shapes follow them with smoothing, so they never jitter:

  head    an ellipse a little larger than the face box, reaching above the hairline and below the chin
  body    a wide ellipse under the head, over the shoulders and chest, running out of the frame bottom

Both have a wide feather: full strength inside, fading out over most of a face width, so no edge shows.

The darkening works in the working space (DaVinci Intermediate) as stops: background
`amount` stops down, subject `subject` stops up or down (0 keeps it exactly as it was, unlike the
2 - g shortcut that leaves the subject 13% darker).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..color import working as W

F32 = np.float32
HEAD_SCALE = (0.95, 1.30)      # head ellipse radii, in face widths and face heights
HEAD_SHIFT = -0.10             # head center above the face box center, in face heights
BODY_SCALE = (1.90, 2.20)      # body ellipse radii, in face widths and face heights
BODY_SHIFT = 2.40              # body center below the face box center, in face heights
INNER, OUTER = 0.85, 1.35      # feather: full strength up to INNER of the radius, none from OUTER (about a third of the radius, as drawn by hand)
SMOOTH = 1                     # samples on each side averaged for a steady track


def normalize_faces(faces: list[list[float]], w: int, h: int) -> list[list[float]]:
    """[x, y, w, h, score] in pixels -> [cx, cy, fw, fh, score] as fractions of the frame."""
    out = []
    for x, y, fw, fh, score in faces:
        out.append([round((x + fw / 2) / w, 4), round((y + fh / 2) / h, 4), round(fw / w, 4), round(fh / h, 4), round(float(score), 3)])
    return out


def _pair(prev: list[list[float]], nxt: list[list[float]]) -> list[tuple[list[float], list[float] | None]]:
    """Match each face of one sample to the nearest face of the next one."""
    out, used = [], set()
    for f in prev:
        best, bd = None, 1e9
        for j, g in enumerate(nxt):
            if j in used:
                continue
            d = (f[0] - g[0]) ** 2 + (f[1] - g[1]) ** 2
            if d < bd and d < (1.5 * max(f[2], g[2])) ** 2:
                best, bd = j, d
        if best is not None:
            used.add(best)
            out.append((f, nxt[best]))
        else:
            out.append((f, None))
    return out


def smooth_track(track: list[list[Any]]) -> list[list[Any]]:
    """Average each face with its matched neighbours over SMOOTH samples on each side."""
    if len(track) < 3:
        return track
    out = []
    for i, (t, faces) in enumerate(track):
        new = []
        for f in faces:
            acc, n = np.array(f[:4], dtype=np.float64), 1
            for k in range(1, SMOOTH + 1):
                for j in (i - k, i + k):
                    if 0 <= j < len(track):
                        m = _pair([f], track[j][1])[0][1]
                        if m is not None:
                            acc += np.array(m[:4], dtype=np.float64)
                            n += 1
            new.append([*(acc / n).round(4).tolist(), f[4]])
        out.append([t, new])
    return out


def faces_at(track: list[list[Any]], t: float) -> list[list[float]]:
    """Faces at time t: linear between the two nearest samples, faces matched by position."""
    if not track:
        return []
    if t <= track[0][0]:
        return track[0][1]
    if t >= track[-1][0]:
        return track[-1][1]
    for i in range(len(track) - 1):
        t0, f0 = track[i]
        t1, f1 = track[i + 1]
        if t0 <= t <= t1:
            a = (t - t0) / max(t1 - t0, 1e-6)
            if not f0 or not f1:
                return f0 if a < 0.5 else f1
            out, matched = [], []
            for f, g in _pair(f0, f1):
                if g is None:            # no partner: the face holds until halfway, then the next sample takes over
                    if a < 0.5:
                        out.append(f)
                    continue
                matched.append(g)
                out.append([f[k] + a * (g[k] - f[k]) for k in range(4)] + [max(f[4], g[4])])
            if a >= 0.5:
                out += [g for g in f1 if not any(g is m for m in matched)]
            return out
    return track[-1][1]


def _ellipse(xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    d = np.sqrt(((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2)
    t = np.clip((d - INNER) / (OUTER - INNER), 0.0, 1.0)
    return (1.0 - t * t * (3.0 - 2.0 * t)).astype(F32)


def mask(faces: list[list[float]], width: int, height: int) -> np.ndarray | None:
    """1 over the subjects, 0 in the background, soft in between. None when there is no face."""
    if not faces:
        return None
    ys = (np.arange(height, dtype=F32) + 0.5) / height
    xs = (np.arange(width, dtype=F32) + 0.5) / width
    xx, yy = np.meshgrid(xs, ys)
    m = np.zeros((height, width), dtype=F32)
    for cx, cy, fw, fh, _ in faces:
        m = np.maximum(m, _ellipse(xx, yy, cx, cy + HEAD_SHIFT * fh, HEAD_SCALE[0] * fw, HEAD_SCALE[1] * fh))
        m = np.maximum(m, _ellipse(xx, yy, cx, cy + BODY_SHIFT * fh, BODY_SCALE[0] * fw, BODY_SCALE[1] * fh))
    return m


def apply(di: np.ndarray, m: np.ndarray | None, amount: float, subject: float = 0.0) -> np.ndarray:
    """Background `amount` stops down, subject `subject` stops, on a DaVinci Intermediate frame (H, W, 3)."""
    if m is None or di.ndim != 3 or (amount == 0.0 and subject == 0.0):
        return di
    shift = F32(W.STOP) * (F32(subject) * m - F32(amount) * (F32(1.0) - m))
    return (di + shift[..., None]).astype(F32)


def settings(finish: dict[str, Any] | None) -> tuple[bool, float, float]:
    r = (finish or {}).get("relight") or {}
    return bool(r.get("enabled")), float(r.get("amount", 0.0)), float(r.get("subject", 0.0))
