"""Shared builders for synthetic clips."""
from pathlib import Path

import numpy as np

from colorgrader import synth

FUJI_TAGS = {"make": "FUJIFILM", "model": "X-M5", "comment": "FUJIFILM DIGITAL CAMERA X-M5"}


def flog2_clip(path: Path, scenes=None, frames_each: int = 12, size=(320, 180), **kw) -> Path:
    scenes = scenes or [synth.procedural_scene(*size)]
    args = dict(curve="fujifilm-f-log2", gamut="f-gamut", bit_depth=10, chroma="422", full_range=True, matrix="bt2020nc",
                tags=FUJI_TAGS, color_tags={"color_primaries": "bt2020", "color_trc": "bt709", "colorspace": "bt2020nc"}, crf=6)
    args.update(kw)
    return synth.record(synth.concat_scenes(scenes, frames_each), path, **args)


def filmsim_clip(path: Path, scenes=None, frames_each: int = 12, size=(320, 180), **kw) -> Path:
    scenes = scenes or [synth.procedural_scene(*size)]
    args = dict(curve="rec709-gamma24", gamut="rec709", bit_depth=8, chroma="420", full_range=False, matrix="bt709",
                codec="libx264", display_look={"contrast": 1.35, "saturation": 1.0}, tags=FUJI_TAGS, crf=8)
    args.update(kw)
    return synth.record(synth.concat_scenes(scenes, frames_each), path, **args)


def slog3_clip(path: Path, scenes=None, frames_each: int = 12, size=(320, 180), **kw) -> Path:
    scenes = scenes or [synth.procedural_scene(*size)]
    args = dict(curve="sony-s-log3", gamut="s-gamut3-cine", bit_depth=10, chroma="422", full_range=False, matrix="bt709",
                codec="libx264", tags={"make": "SONY", "model": "ILCE-7SM3"}, crf=6)
    args.update(kw)
    return synth.record(synth.concat_scenes(scenes, frames_each), path, **args)


def wall_patch(img: np.ndarray) -> np.ndarray:
    """Mean color of the neutral wall in synth.procedural_scene."""
    h, w = img.shape[:2]
    return img[int(h * 0.50):int(h * 0.56), int(w * 0.04):int(w * 0.20)].reshape(-1, 3).mean(axis=0)


def load_clip(p, path, profile=("fujifilm-x-m5", "f-log2")):
    """Add a clip to the project, set its profile and analyze it."""
    clip = p.add_clip(str(path))
    if profile:
        p.set_profile(clip["id"], *profile)
    p.analyze_clip(clip["id"])
    return p.clip(clip["id"])
