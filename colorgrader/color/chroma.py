"""Rebuild compressed color from the full-resolution brightness.

Heavily compressed 4:2:0 and 4:2:2 files (an edit exported at a low bitrate, a phone clip, a download) store their
color as a mosaic of flat blocks, 16 by 16 pixels for H.264, each block a little off in color. In a flat log picture
nobody sees them. A grade that adds saturation, Color Boost most of all, turns them into squares on skin, staircase
edges on colored clothes, and purple and green blobs on gray walls, white shirts and overcast skies.

First, a frame the export stored with its color wrecked (whole blocks off, magenta on a pink sweater, while the frames
around it are fine) takes its color from its neighbors, moved along the motion: see repair() below.

Then three steps, all on the color planes. The brightness plane stays untouched.

1. Clean the block colors: an edge-aware average (a bilateral filter on brightness and color together) over about 48
   source pixels, three blocks. Pixels of a different brightness or a clearly different color do not mix, so edges and
   real color changes stay. Each block's color error averages out with its neighbors' instead of staying as a blob.
   The color range that counts as "the same" grows with the block error measured on the frame's flat areas.
2. Snap color edges to brightness edges: the guided filter (He, Sun, Tang 2010) in its fast form (He, Sun 2015). Inside
   every small window the color is modeled as a straight-line function of the brightness, which the file keeps at full
   resolution with far less damage.
3. Give back real detail: compare the result with the color the file stored, at the file's own color resolution. Where
   the two disagree by more than the frame's block error (measured on its flattest parts), the file's color returns:
   subtitles, thin colored lines, colored detail of the same brightness as its surroundings, which steps 1 and 2 can
   wash out. Sharper edges never count as a disagreement: at the file's color resolution they average to the same color.

Measured on a clean scene (overcast sky, gray wall, white shirt with a lav mic, skin, bricks, a checked shirt, foliage,
yellow subtitles), damaged like a 25 Mbps 4K edit export (8-bit 4:2:0, block color errors matching a real file),
graded with saturation and Color Boost, color error against the same grade of the clean scene:
  plain 4:2:0 color 5.07%, guided filter alone 2.76%, these three steps 1.56%.
  Blobs on gray areas 2.97%, 2.01%, 1.05%. Subtitles 6.6%, 13.6%, 5.7%. Skin 4.4%, 2.3%, 1.3%.
On a file without block errors the three steps still beat plain 4:2:0 color: 1.83% against 1.99%. On a real 4K S-Log3
edit exported at 25 Mbps, the flat areas that showed squares without the rebuild and soft blobs with the guided filter
alone come out clean. A 4K frame takes about 300 ms, and a 4K export about a sixth longer than with the guided filter alone.

The viewer, the full-quality still and the export all run this same code, so the three show the same picture.
"""
from __future__ import annotations

from typing import Any

import numpy as np

F32 = np.float32
VERSION = 3           # bump when the result changes: playback files rendered with an older rebuild render again
                      # (1 guided filter, 2 the three steps, 3 bad frames repaired from their neighbors)
BLOCK_RADIUS = 8      # source pixels: half of the 16-pixel color blocks of H.264 (H.265 blocks are similar or larger)
EPS = 1e-4            # in 0..1 code values, squared. Smaller keeps color edges sharper where the brightness has an edge
CLEAN_RADIUS = 48     # source pixels: the block-color cleanup averages over about three blocks
CLEAN_RANGE = 0.02    # 0..1 code values, brightness (doubled) plus Cb plus Cr: farther apart than this, pixels do not mix
CLEAN_LUMA = 2.0      # brightness counts double, so color never crosses a brightness edge
RANGE_PER_ERROR = 4.5  # the range also covers this many times the measured block error: typical steps between blocks mix
GIVE_BACK = 3.0       # the file's color returns where it differs by more than this many times the frame's block error
BLOCK_QUANTILE = 0.9  # the block error is read off this quantile of the disagreement on flat areas
GIVE_BACK_FLOOR = 1.0  # and by more than this many code values at the file's bit depth (one 8-bit step between blocks is error)
AUTO_MAX_BPP = 0.2    # Auto rebuilds 8-bit 4:2:0 and 4:2:2 files below this many bits per pixel (H.264 terms).
                      # Camera files sit at 0.3 to 0.5 and keep their color as recorded.


def subsampling(info: dict[str, Any]) -> tuple[int, int]:
    """How many source pixels share one stored color sample, across and down."""
    return {"420": (2, 2), "422": (2, 1), "411": (4, 1), "440": (1, 2)}.get(str(info.get("chroma", "420")), (1, 1))


def rebuild(buf: np.ndarray, w: int, h: int, scale: float = 1.0, sub: tuple[int, int] = (2, 2), bits: int = 8) -> np.ndarray:
    """buf: one frame as yuv444p16le planes (3, h, w), flattened uint16, as media.read_frame returns it.
    scale: decoded width / source width, so the windows follow the source's blocks at any preview size.
    sub: the file's color subsampling, (2, 2) for 4:2:0 and (2, 1) for 4:2:2. bits: the file's bit depth.
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
    # shrink first, then convert: no full-size float copy of the color planes
    cs = [(cv2.resize(planes[i], (sw, sh), interpolation=cv2.INTER_AREA) if s > 1 else planes[i]).astype(F32) * unit for i in (1, 2)]

    # 1. clean the block colors: edge-aware, over about three blocks, on the same small copy. A smaller copy is faster
    # but mixes colors across edges before the filter can tell them apart (measured: 2.6 times the edge error).
    rc = max(1, int(round(CLEAN_RADIUS * scale / s)))
    if rc >= 2:
        guide = np.empty((sh, sw, 3), F32)
        guide[..., 0] = ys * F32(CLEAN_LUMA)
        guide[..., 1], guide[..., 2] = cs
        # a harder export has bigger steps between blocks: the range grows with them, or the filter keeps every
        # block border as if it were a real edge
        sr = max(CLEAN_RANGE, RANGE_PER_ERROR * _wobble(ys, cs, rc))
        clean = cv2.bilateralFilter(guide, 2 * rc + 1, sr, rc / 2.0)
        cs = [np.ascontiguousarray(clean[..., 1]), np.ascontiguousarray(clean[..., 2])]

    # 2. snap color edges to brightness edges
    box = lambda a: cv2.boxFilter(a, -1, k, borderType=cv2.BORDER_REFLECT)
    m_y = box(ys)
    var_y = box(ys * ys) - m_y * m_y
    rebuilt = []
    for c in cs:
        m_c = box(c)
        a = (box(ys * c) - m_y * m_c) / (var_y + F32(EPS))
        b = m_c - a * m_y
        a, b = box(a), box(b)
        if s > 1:
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
        # in place: at 4K every full-size float temporary is 33 MB, and the export runs this on several frames at once
        a *= y
        a += b
        rebuilt.append(a)

    # 3. give back the detail the file holds beyond its block error
    _give_back(planes, rebuilt, w, h, scale, sub, bits)
    out = np.empty_like(planes)
    out[0] = planes[0]
    for i, a in zip((1, 2), rebuilt):
        a *= F32(65535.0)
        np.rint(a, out=a)
        np.clip(a, 0, 65535, out=a)
        out[i] = a
    return out.reshape(-1)


def _wobble(ys: np.ndarray, cs: list[np.ndarray], rc: int) -> float:
    """How far the color strays from its own neighborhood where the brightness is flat: the block color error, one
    channel, as a normal distribution's sigma. Flat brightness hides no real detail, so the stray is the damage."""
    import cv2

    k = (2 * rc + 1, 2 * rc + 1)
    mu = cv2.boxFilter(ys, -1, (5, 5))
    var = cv2.boxFilter(ys * ys, -1, (5, 5)) - mu * mu
    step = max(1, int(round(min(ys.shape) / 300)))
    flat = var[::step, ::step] <= np.quantile(var[::step, ::step], 0.3)
    if not flat.any():
        return 0.0
    dev = [np.abs(c - cv2.boxFilter(c, -1, k, borderType=cv2.BORDER_REFLECT))[::step, ::step][flat] for c in cs]
    return float(np.quantile(np.concatenate(dev), BLOCK_QUANTILE)) / 1.645     # 90% of a normal's size: 1.645 sigma


def _give_back(planes: np.ndarray, rebuilt: list[np.ndarray], w: int, h: int, scale: float, sub: tuple[int, int], bits: int) -> None:
    """Adds the file's own color back to rebuilt (0..1 floats, changed in place) where the two disagree by more than the
    frame's block error, compared at the file's color resolution."""
    import cv2

    fx, fy = max(1, int(round(sub[0] * scale))), max(1, int(round(sub[1] * scale)))
    nw, nh = max(1, w // fx), max(1, h // fy)
    unit = F32(1.0 / 65535.0)
    small = (lambda a: cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)) if fx > 1 or fy > 1 else (lambda a: a)
    diff = []
    for i, q in zip((1, 2), rebuilt):
        d = small(planes[i]).astype(F32)      # shrink the 16-bit plane first: a quarter of the values to convert
        d *= unit
        d -= small(q)
        diff.append(d)
    m = cv2.magnitude(diff[0], diff[1])
    # the block error: the typical disagreement on the flattest third of the frame, where no real detail can hide.
    # Never below GIVE_BACK_FLOOR code values: in an 8-bit file a one-code step between blocks is the error itself,
    # and most blocks are right, so the typical disagreement understates the blocks that are off.
    yl = small(planes[0]).astype(F32) * unit
    mu = cv2.boxFilter(yl, -1, (9, 9))
    var = cv2.boxFilter(yl * yl, -1, (9, 9)) - mu * mu
    step = max(1, int(round(min(nw, nh) / 400)))          # a sample of the frame is enough for two statistics
    vs, ms = var[::step, ::step], m[::step, ::step]
    flat = vs <= np.quantile(vs, 0.3)
    # a high quantile, not the median: in a hard export most blocks are right and a few are far off
    sigma = float(np.quantile(ms[flat], BLOCK_QUANTILE)) / _rayleigh(BLOCK_QUANTILE) if flat.any() else 0.0
    t = max(GIVE_BACK * sigma, GIVE_BACK_FLOOR / ((1 << int(bits)) - 1))
    # nothing below the threshold, everything above twice the threshold: real detail returns whole, not shaved
    gain = np.clip((m - F32(t)) / F32(max(t, 1e-9)), 0, 1)
    for q, d in zip(rebuilt, diff):
        d *= gain
        q += cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) if fx > 1 or fy > 1 else d


def _rayleigh(q: float) -> float:
    """The q quantile of the length of a 2-D normal with sigma 1."""
    return float(np.sqrt(-2.0 * np.log(1.0 - q)))


# ------------------------------------------------------------ single bad frames
# An editor's export at a low bitrate sometimes stores one frame with its color wrecked: whole blocks of a pink sweater
# magenta, others orange, while the frames before and after hold the right color and the brightness is fine. Measured
# on a real 25 Mbps 4K export: one frame in 122 with 8.7% of its still blocks over 4 code values off, every other frame
# at 0.1% or less. A color error that size reads as real color to the rebuild, so such a frame takes its color from its
# neighbors first. People move between frames, so the neighbors' color is moved onto the frame along the motion its
# brightness shows (optical flow), and only where both neighbors agree with it.
REPAIR_WIDTH = 960       # the motion and the checks run on a copy this wide
REPAIR_GATE = 0.02       # a frame is repaired when more than this share of its checkable area is off
REPAIR_OFF = (2.0, 4.0)  # 8-bit code values: color counts as off from 2, fully replaced from 4
REPAIR_MATCH = 4.0       # 8-bit code values: a moved neighbor this close in brightness (and color) shows the same thing
REPAIR_DUPLICATE = 0.4   # 8-bit code values, mean over the frame: a repeated frame (25 fps edits exported at 30)


def _small(buf: np.ndarray, w: int, h: int, sw: int, sh: int) -> np.ndarray:
    """Y, Cb, Cr at the check size, in 8-bit code values, (3, sh, sw)."""
    import cv2

    planes = buf.reshape(3, h, w)
    return np.stack([cv2.resize(planes[i], (sw, sh), interpolation=cv2.INTER_AREA).astype(F32) for i in range(3)]) * F32(255.0 / 65535.0)


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Where each pixel of brightness a sits in brightness b (8-bit code values), (h, w, 2)."""
    import cv2

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    return dis.calc(np.clip(a, 0, 255).astype(np.uint8), np.clip(b, 0, 255).astype(np.uint8), None)


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    import cv2

    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=F32), np.arange(h, dtype=F32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def repair(frames: list[np.ndarray], center: int, w: int, h: int, scale: float = 1.0, small: dict | None = None) -> np.ndarray:
    """frames: consecutive decoded frames (yuv444p16le, flattened) around frames[center], at most two on each side.
    Returns frames[center] itself when its color is fine (the usual case, bit for bit), otherwise a copy whose wrecked
    color comes from the frames before and after, moved along the motion. Brightness stays as decoded, and so does the
    color wherever the neighbors disagree with the frame or with each other (something new appears, an occlusion).
    A neighbor that repeats this frame (an edit at 25 fps exported at 30) is skipped: an encoder copies the damage
    into the repeat."""
    import cv2

    cur = frames[center]
    sw = min(w, REPAIR_WIDTH)
    sh = max(1, int(round(h * sw / w)))
    small = {} if small is None else small     # shrunk copies by position in frames; a stream passes its own

    def sm(i: int) -> np.ndarray:
        if i not in small:
            small[i] = _small(frames[i], w, h, sw, sh)
        return small[i]

    def neighbor(step: int) -> int | None:
        i = center + step
        if not 0 <= i < len(frames):
            return None
        if float(np.abs(sm(i)[0] - sm(center)[0]).mean()) < REPAIR_DUPLICATE:
            j = i + step
            return j if 0 <= j < len(frames) else None
        return i

    ip, inx = neighbor(-1), neighbor(1)
    if ip is None or inx is None:
        return cur
    c = sm(center)
    fa, fz = _flow(c[0], sm(ip)[0]), _flow(c[0], sm(inx)[0])
    a = np.stack([_warp(sm(ip)[i], fa) for i in range(3)])
    z = np.stack([_warp(sm(inx)[i], fz) for i in range(3)])
    mid = (a + z) * F32(0.5)
    box = lambda v: cv2.boxFilter(v, -1, (5, 5))
    valid = (box(np.abs(a[0] - c[0])) < REPAIR_MATCH) & (box(np.abs(z[0] - c[0])) < REPAIR_MATCH) \
        & (box(np.hypot(a[1] - z[1], a[2] - z[2])) < REPAIR_MATCH)
    off = box(np.hypot(c[1] - mid[1], c[2] - mid[2]))
    if valid.mean() < 0.05 or float((off[valid] > REPAIR_OFF[1]).mean()) <= REPAIR_GATE:
        return cur
    lo, hi = REPAIR_OFF
    t = np.clip((off - lo) / (hi - lo), 0, 1)
    wgt = cv2.resize((t * t * (3 - 2 * t) * valid).astype(F32), (w, h), interpolation=cv2.INTER_LINEAR)
    k = F32(w / float(sw))
    fa_full = cv2.resize(fa, (w, h), interpolation=cv2.INTER_LINEAR) * k
    fz_full = cv2.resize(fz, (w, h), interpolation=cv2.INTER_LINEAR) * k
    out = cur.copy().reshape(3, h, w)
    pa, pz = frames[ip].reshape(3, h, w), frames[inx].reshape(3, h, w)
    for i in (1, 2):
        v = out[i].astype(F32)
        target = (_warp(pa[i].astype(F32), fa_full) + _warp(pz[i].astype(F32), fz_full)) * F32(0.5)
        v += wgt * (target - v)
        out[i] = np.clip(np.rint(v), 0, 65535)
    return out.reshape(-1)


def repair_stream(frames, w: int, h: int, scale: float = 1.0):
    """Every frame of a sequence in order, each through repair() with up to two frames on each side: the export's
    view of the same neighbors the viewer reads around one frame (media.read_frames)."""
    from collections import deque

    window: deque = deque()          # (index, frame), unrepaired
    shrunk: dict[int, np.ndarray] = {}   # shrunk copies by frame index: each frame sits in up to five windows
    read = done = 0

    def one(i: int) -> np.ndarray:
        first = window[0][0]
        small = {k - first: shrunk[k] for k in shrunk if k >= first}
        out = repair([g for _, g in window], i - first, w, h, scale, small)
        for k, v in small.items():
            shrunk[k + first] = v
        return out

    def drop(upto: int) -> None:
        while window and window[0][0] < upto:
            shrunk.pop(window.popleft()[0], None)

    for f in frames:
        window.append((read, f))
        read += 1
        while done <= read - 3:      # frame `done` has its two frames after it
            yield one(done)
            done += 1
            drop(done - 2)
    while done < read:
        yield one(done)
        done += 1
        drop(done - 2)


def block_ratio(plane: np.ndarray, period: int = 8) -> float:
    """How much larger the steps at block borders are than the changes inside blocks (1.0: no block grid)."""
    p = plane.astype(F32)
    d = np.abs(np.diff(p, axis=1))
    inside = np.delete(d, np.s_[period - 1::period], axis=1).mean()
    return float(d[:, period - 1::period].mean() / max(float(inside), 1e-6))
