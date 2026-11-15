"""Build synthetic demo clips from photos, recorded through camera profiles.

Usage:
  python tools/make_demo_clips.py PHOTO_DIR OUT_DIR [--size 1920x1080] [--seconds 2]

Makes stand-ins for the three test clips:
  demo_flog2.mov     Fujifilm X-M5 F-Log2, 10-bit 4:2:2 H.265, three shots
  demo_filmsim.mov   Fujifilm X-M5 film simulation, 8-bit H.264
  demo_applelog.mov  Phone Apple Log, 10-bit H.265

These clips test the pipeline. They are no substitute for camera originals.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colorgrader import cameras, synth  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("photos")
    ap.add_argument("out")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args()
    w, h = (int(v) for v in a.size.split("x"))
    photos = Path(a.photos)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cameras.seed()
    n = int(round(a.seconds * a.fps))

    def scenes(names):
        return [synth.photo_to_scene(photos / nm, w, h) for nm in names]

    fuji_tags = {"make": "FUJIFILM", "model": "X-M5", "comment": "FUJIFILM DIGITAL CAMERA X-M5"}
    # F-Log2: exposed one stop bright with a cool cast, as log footage often is
    synth.record(synth.concat_scenes(scenes(["zidane.jpg", "kodim22.png", "building.jpg"]), n), out / "demo_flog2.mov",
                 curve="fujifilm-f-log2", gamut="f-gamut", fps=a.fps, bit_depth=10, chroma="422", full_range=True, matrix="bt2020nc",
                 exposure=0.8, wb_gains=(0.93, 1.0, 1.10), flare=0.002, noise=0.004, tags=fuji_tags,
                 color_tags={"color_primaries": "bt2020", "color_trc": "bt709", "colorspace": "bt2020nc"})
    synth.record(synth.concat_scenes(scenes(["kodim15.png", "kodim21.png"]), n), out / "demo_filmsim.mov",
                 curve="rec709-gamma24", gamut="rec709", fps=a.fps, bit_depth=8, chroma="420", full_range=False, matrix="bt709",
                 codec="libx264", display_look={"contrast": 1.38, "saturation": 0.9}, wb_gains=(1.06, 1.0, 0.94), tags=fuji_tags)
    synth.record(synth.concat_scenes(scenes(["bus.jpg", "china.jpg"]), n), out / "demo_applelog.mov",
                 curve="apple-log", gamut="rec2020", fps=a.fps, bit_depth=10, chroma="420", full_range=False, matrix="bt2020nc",
                 exposure=-0.4, wb_gains=(1.04, 1.0, 0.97), noise=0.005,
                 tags={"make": "Apple", "model": "iPhone 15 Pro", "com.apple.quicktime.make": "Apple"},
                 color_tags={"color_primaries": "bt2020", "color_trc": "unknown", "colorspace": "bt2020nc"})
    print("wrote", [p.name for p in sorted(out.glob("demo_*.mov"))])


if __name__ == "__main__":
    main()
