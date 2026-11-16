"""Make a test reference: grade any image with a known look and save it.

Usage:
  python tools/make_reference.py IN_IMAGE OUT.png --look teal-orange

The known look gives a ground truth for matching tests.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colorgrader import cameras, media  # noqa: E402
from colorgrader.color.transforms import OutputTransform  # noqa: E402
from colorgrader.grade import lookfit, model, pipeline  # noqa: E402

LOOKS = {
    "teal-orange": dict(contrast=1.30, lift=[-0.004, 0.006, 0.014], gamma=[0.010, 0.0, -0.012], gain=[1.035, 1.0, 0.95], saturation=46.0,
                        lum_vs_sat=lookfit.lum_vs_sat_points(0.7, 0.8)),
    "faded-warm": dict(contrast=0.92, lift=[0.036, 0.031, 0.024], gamma=[0.016, 0.008, -0.006], gain=[1.0, 0.995, 0.97], saturation=40.0),
    "cool-night": dict(contrast=1.25, lift=[-0.006, 0.0, 0.010], gamma=[-0.10, -0.085, -0.06], gain=[0.93, 0.97, 1.03], saturation=38.0),
    "punchy": dict(contrast=1.45, lift=[-0.006, -0.006, -0.006], gamma=[0.0, 0.0, 0.0], gain=[1.0, 1.0, 1.0], saturation=58.0),
    "mono": dict(contrast=1.35, saturation=0.0),
}


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--look", default="teal-orange", choices=sorted(LOOKS))
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    cameras.seed()
    rgb, _ = media.read_still(a.src)
    rgb = media.resize_rgb(rgb, max_width=a.width)
    odt = OutputTransform("rec709-gamma24", 1000.0)
    look = model.neutral_look()
    look.update(LOOKS[a.look])
    out = odt(pipeline.look_stage(odt.inverse(rgb), model.quantize_look(look)))
    img = (np.clip(out, 0, 1) * 65535 + 0.5).astype(np.uint16)
    cv2.imwrite(a.out, np.ascontiguousarray(img[..., ::-1]))
    print(a.out, a.look)


if __name__ == "__main__":
    main()
