"""Before / after / reference sheet for the current project.

Usage:
  python tools/contact_sheet.py OUT.png [--stages source,converted,balanced,graded] [--width 420]

One row per shot. Columns: the chosen stages, then the reference when the
project holds one. Reads the project in the user folder (COLORGRADER_HOME).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colorgrader import project  # noqa: E402

NAMES = {"source": "Source", "converted": "Converted", "balanced": "Balanced", "graded": "Graded"}


def to_u8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 16, dtype=np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--stages", default="source,converted")
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--scope", default="", help="add a scope column for the last stage: waveform, parade, vectorscope")
    a = ap.parse_args()
    stages = [s for s in a.stages.split(",") if s]
    p = project.get()
    st = p.state
    ref = None
    if st.get("reference"):
        try:
            ref = to_u8(p.reference_display())
        except Exception:
            ref = None
    w = a.width
    h = int(w * 9 / 16)
    pad, head = 8, 26
    cols = len(stages) + (1 if ref is not None else 0) + (1 if a.scope else 0)
    rows = len(st["shots"])
    sheet = np.full((rows * (h + head + pad) + pad, cols * (w + pad) + pad, 3), 28, dtype=np.uint8)
    for r, shot in enumerate(st["shots"]):
        clip = p.clip(shot["clip_id"])
        t = shot["times"][len(shot["times"]) // 2] if shot.get("times") else 0.5 * (shot["start"] + shot["end"])
        y = pad + r * (h + head + pad)
        cells = []
        for stage in stages:
            img = to_u8(p.render_frame(clip["id"], t, stage, 960, shot["id"]))
            cells.append((f"{NAMES[stage]}  {clip['name']} shot {shot['index'] + 1}", img))
        if a.scope:
            from colorgrader.analysis import scopes

            sc = scopes.SCOPES[a.scope](p.render_frame(clip["id"], t, stages[-1], 640, shot["id"]))[..., ::-1]
            cells.append((f"{a.scope} of {NAMES[stages[-1]]}", sc))
        if ref is not None:
            cells.append(("Reference", ref))
        for c, (label, img) in enumerate(cells):
            x = pad + c * (w + pad)
            sheet[y + head:y + head + h, x:x + w] = fit(img, w, h)
            cv2.putText(sheet, label[:60], (x + 2, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 215, 215), 1, cv2.LINE_AA)
    cv2.imwrite(a.out, sheet[..., ::-1])
    print(a.out, sheet.shape)


if __name__ == "__main__":
    main()
