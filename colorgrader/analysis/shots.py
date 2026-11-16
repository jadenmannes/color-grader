"""Shot detection on tiny frames decoded at the clip frame rate."""
from __future__ import annotations

from typing import Any

import numpy as np

F32 = np.float32


def _features(frames: np.ndarray) -> np.ndarray:
    """Per-frame color histograms: luma (32 bins) and two color differences (16 bins each)."""
    f = frames.astype(F32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    n = f.shape[0]
    px = y.shape[1] * y.shape[2]
    out = np.zeros((n, 64), dtype=F32)
    yi = np.clip((y / 8.0).astype(np.int32), 0, 31).reshape(n, -1)
    c1 = np.clip(((r - g) / 8.0 + 8.0).astype(np.int32), 0, 15).reshape(n, -1) + 32
    c2 = np.clip(((b - g) / 8.0 + 8.0).astype(np.int32), 0, 15).reshape(n, -1) + 48
    rows = np.repeat(np.arange(n), px)
    for idx in (yi, c1, c2):
        np.add.at(out, (rows, idx.ravel()), 1.0)
    return out / px


def detect_cuts(cuts_path: str, size: tuple[int, int], n_frames: int, fps: float, min_shot: float = 0.6,
                threshold: float = 0.22) -> tuple[list[float], dict[str, Any]]:
    """Return cut times in seconds (start of each new shot) plus diagnostics.

    A hard cut shows as a one-frame jump in the color histogram, well above the
    local level of motion. A dissolve shows as a steady drift over about a second.
    """
    w, h = size
    if n_frames < 3:
        return [], {"frames": n_frames}
    frames = np.memmap(cuts_path, dtype=np.uint8, mode="r", shape=(n_frames, h, w, 3))
    feats = np.concatenate([_features(np.asarray(frames[i:i + 512])) for i in range(0, n_frames, 512)])
    d = np.zeros(n_frames, dtype=F32)
    d[1:] = 0.5 * np.abs(feats[1:] - feats[:-1]).sum(axis=1) / 3.0  # 0..1
    # spatial change on a coarse grid: catches cuts between shots with the same colors
    # (a reverse angle in one room), where the histogram barely moves
    small = np.concatenate([np.asarray(frames[i:i + 512]).astype(F32).mean(axis=3) for i in range(0, n_frames, 512)])
    gy, gx = max(1, h // 4), max(1, w // 4)
    grid = small[:, :gy * 4, :gx * 4].reshape(n_frames, gy, 4, gx, 4).mean(axis=(2, 4)).reshape(n_frames, -1)
    s = np.zeros(n_frames, dtype=F32)
    s[1:] = np.abs(grid[1:] - grid[:-1]).mean(axis=1) / 255.0
    score = np.maximum(d, np.minimum(s * 4.5, 1.0))
    win = max(3, int(round(fps * 0.5)))
    min_gap = max(1, int(round(min_shot * fps)))
    cuts: list[int] = []
    for i in range(1, n_frames):
        if score[i] < threshold:
            continue
        lo, hi = max(1, i - win), min(n_frames, i + win + 1)
        local = np.concatenate([score[lo:i], score[i + 1:hi]])
        base = float(np.median(local)) if local.size else 0.0
        if score[i] >= 4.0 * base + 0.06:
            # flash guard: the frame after the jump looks like the frame before it
            if i + 1 < n_frames and score[i + 1] >= threshold and 0.5 * np.abs(feats[i + 1] - feats[i - 1]).sum() / 3.0 < 0.5 * float(d[i]):
                continue
            if cuts and i - cuts[-1] < min_gap:
                if score[i] > score[cuts[-1]]:
                    cuts[-1] = i
                continue
            if i < min_gap or n_frames - i < min_gap:
                continue
            cuts.append(i)
    # dissolves: large drift over about one second with no hard cut inside
    k = max(2, int(round(fps * 1.0)))
    if n_frames > 3 * k:
        drift = np.zeros(n_frames, dtype=F32)
        drift[k:] = 0.5 * np.abs(feats[k:] - feats[:-k]).sum(axis=1) / 3.0
        i = k
        while i < n_frames:
            if drift[i] > 0.45:
                j = i
                while j + 1 < n_frames and drift[j + 1] > 0.30:
                    j += 1
                mid = (i + j) // 2 - k // 2
                near_cut = any(abs(mid - c) < k + min_gap for c in cuts)
                if not near_cut and mid > min_gap and n_frames - mid > min_gap and float(score[max(1, i - k):j + 1].max()) < threshold:
                    cuts.append(int(mid))
                i = j + k
            else:
                i += 1
        cuts.sort()
    return [c / fps for c in cuts], {"frames": n_frames, "max_score": float(score.max()), "cut_frames": cuts}


def shots_from_cuts(cuts: list[float], duration: float) -> list[tuple[float, float]]:
    edges = [0.0] + [c for c in cuts if 0.0 < c < duration] + [duration]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i + 1] - edges[i] > 1e-3]
