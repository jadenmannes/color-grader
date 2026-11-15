"""Input and output transforms.

Input: file R'G'B' signal -> DaVinci Wide Gamut / DaVinci Intermediate, driven
only by the camera profile data (curve, gamut, range, or a maker LUT).
Output: working space -> display R'G'B' (Rec.709).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import library, paths
from . import working as W
from .curves import Curve
from .gamuts import Gamut, apply_matrix, conversion_matrix
from .lut import Lut3D, read_cube
from .ycbcr import signal_to_absolute

F32 = np.float32

_lut_cache: dict[str, Lut3D] = {}


def load_user_lut(name: str) -> Lut3D:
    """Maker LUTs live in the user folder under luts/."""
    path = Path(name)
    if not path.is_absolute():
        path = paths.sub("luts") / name
    key = f"{path}:{path.stat().st_mtime_ns}"
    if key not in _lut_cache:
        _lut_cache[key] = read_cube(path)
    return _lut_cache[key]


@dataclass
class InputTransform:
    """Everything needed to bring one clip into the working space."""

    curve: Curve
    gamut: Gamut
    full_range: bool          # how the file stores code values
    bit_depth: int
    max_input_nits: float     # tone mapping range, shared with the output transform
    lut: Lut3D | None = None  # maker LUT applied before the curve
    lut_input: str = "absolute"

    def to_linear_working(self, signal_rgb: np.ndarray) -> np.ndarray:
        """R'G'B' signal (normalized by the file's range) -> linear DaVinci Wide Gamut."""
        lib = library.get()
        dwg = lib.gamut(W.WORKING_GAMUT)
        v = signal_rgb.astype(F32, copy=False)
        if self.lut is not None:
            if self.lut_input == "absolute":
                v = signal_to_absolute(v, self.full_range, self.bit_depth)
            v = self.lut.apply(v)
        elif self.curve.code_values == "absolute":
            v = signal_to_absolute(v, self.full_range, self.bit_depth)
        lin = self.curve.decode(v)
        if self.curve.kind == "display":
            lin = W.tm_inverse(np.clip(lin, 0.0, 1.0), self.max_input_nits)
        elif self.curve.kind == "pq":
            lin = W.pq_display_to_scene(lin, self.curve.params, self.gamut.luma_weights())
        return apply_matrix(lin, conversion_matrix(self.gamut, dwg))

    def __call__(self, signal_rgb: np.ndarray) -> np.ndarray:
        return W.di_encode(self.to_linear_working(signal_rgb))


def input_transform_for(profile: dict[str, Any], full_range: bool, bit_depth: int, max_input_nits: float) -> InputTransform:
    """Build the input transform from a camera profile dict. Reads only data."""
    lib = library.get()
    spec = profile.get("input") or {}
    rng = profile.get("data_range", "auto")
    if rng == "full":
        full_range = True
    elif rng == "video":
        full_range = False
    lut = None
    lut_input = "absolute"
    if spec.get("type") == "lut":
        lut = load_user_lut(spec["lut_file"])
        lut_input = spec.get("lut_input", "absolute")
        out = spec.get("lut_output") or {}
        curve = lib.curve(out.get("curve", "rec709-gamma24"))
        gamut = lib.gamut(out.get("gamut", "rec709"))
    else:
        curve = lib.curve(spec["curve"])
        gamut = lib.gamut(spec["gamut"])
    return InputTransform(curve=curve, gamut=gamut, full_range=full_range, bit_depth=bit_depth,
                          max_input_nits=max_input_nits, lut=lut, lut_input=lut_input)


@dataclass
class OutputTransform:
    """Working space -> display R'G'B'."""

    output_space: str = "rec709-gamma24"
    max_input_nits: float = 1000.0

    def _curve(self) -> Curve:
        return library.get().curve(W.OUTPUT_SPACES[self.output_space]["curve"])

    def _m(self, inverse: bool = False) -> np.ndarray:
        lib = library.get()
        a, b = lib.gamut(W.WORKING_GAMUT), lib.gamut(W.DISPLAY_GAMUT)
        return conversion_matrix(b, a) if inverse else conversion_matrix(a, b)

    def display_linear(self, di: np.ndarray) -> np.ndarray:
        lin = apply_matrix(W.di_decode(di), self._m())
        return W.tm_forward(lin, self.max_input_nits)

    def __call__(self, di: np.ndarray) -> np.ndarray:
        return self._curve().encode(self.display_linear(di))

    def inverse(self, display_rgb: np.ndarray, curve: Curve | None = None) -> np.ndarray:
        """Display R'G'B' -> working space. `curve` overrides the display curve (sRGB stills)."""
        curve = curve or self._curve()
        lin = curve.decode(np.clip(display_rgb.astype(F32, copy=False), 0.0, 1.0))
        scene = W.tm_inverse(lin, self.max_input_nits)
        return W.di_encode(apply_matrix(scene, self._m(inverse=True)))


def correction_gains(corr: dict[str, Any] | None) -> np.ndarray:
    """Linear RGB gains from a profile's default corrections.

    exposure in stops. temperature: +100 warms by half a stop of red over blue.
    tint: +100 moves half a stop toward magenta. Gains keep neutral brightness.
    """
    corr = corr or {}
    exposure = float(corr.get("exposure", 0.0))
    temp = float(corr.get("temperature", 0.0)) / 100.0
    tint = float(corr.get("tint", 0.0)) / 100.0
    stops = np.array([0.25 * temp + 0.125 * tint, -0.25 * tint, -0.25 * temp + 0.125 * tint], dtype=np.float64)
    gains = np.power(2.0, stops)
    gains = gains / float(gains @ np.array([0.2126, 0.7152, 0.0722]))
    return (gains * (2.0 ** exposure)).astype(F32)
