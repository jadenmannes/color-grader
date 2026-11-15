"""Working space and display transform.

Working space: DaVinci Wide Gamut with the DaVinci Intermediate log curve,
the native grading space in DaVinci Resolve.

Display transform: gamut matrix to Rec.709, then the DaVinci tone mapping
used by Resolve's Color Space Transform, f(x) = a*x/(x+b), per channel on
linear light, then the display gamma. Finished Rec.709 sources enter the
working space through the exact inverse, so an untouched clip round-trips.

Tone mapping constants follow Thatcher Freeman's measured recreation of
Resolve's CST (utility-dctls, MIT), Adaptation = 9.
"""
from __future__ import annotations

import numpy as np

F32 = np.float32

DI_A = 0.0075
DI_B = 7.0
DI_C = 0.07329248
DI_M = 10.44426855
DI_LIN_CUT = 0.00262409
DI_LOG_CUT = 0.02740668

WORKING_GAMUT = "davinci-wide-gamut"
WORKING_CURVE = "davinci-intermediate"
DISPLAY_GAMUT = "rec709"
STOP = DI_C  # one stop of exposure in DaVinci Intermediate code units

OUTPUT_SPACES = {
    "rec709-gamma24": {"name": "Rec.709 Gamma 2.4", "curve": "rec709-gamma24", "trc_tag": "bt709",
                       "resolve_gamma": "Gamma 2.4", "resolve_output": "Rec.709 Gamma 2.4"},
    "rec709-gamma22": {"name": "Rec.709 Gamma 2.2", "curve": "gamma22", "trc_tag": "bt709",
                       "resolve_gamma": "Gamma 2.2", "resolve_output": "Rec.709 Gamma 2.2"},
    "srgb": {"name": "sRGB", "curve": "srgb", "trc_tag": "iec61966-2-1",
             "resolve_gamma": "sRGB", "resolve_output": "sRGB"},
}


def di_encode(lin) -> np.ndarray:
    lin = np.asarray(lin, dtype=F32)
    log = (np.log2(np.maximum(lin, F32(DI_LIN_CUT)) + F32(DI_A)) + F32(DI_B)) * F32(DI_C)
    return np.where(lin > F32(DI_LIN_CUT), log, lin * F32(DI_M)).astype(F32, copy=False)


def di_decode(v) -> np.ndarray:
    v = np.asarray(v, dtype=F32)
    lin = np.exp2(np.maximum(v, F32(DI_LOG_CUT)) / F32(DI_C) - F32(DI_B)) - F32(DI_A)
    return np.where(v > F32(DI_LOG_CUT), lin, v / F32(DI_M)).astype(F32, copy=False)


MID_GRAY = float(di_encode(np.array([0.18], dtype=F32))[0])  # 0.336043


def tm_constants(max_input_nits: float, max_output_nits: float = 100.0, adaptation: float = 9.0):
    """Return (a, b, input_white, output_white) of f(x) = a*x/(x+b)."""
    iw = max_input_nits / 100.0
    ow = max_output_nits / 100.0
    if abs(iw - ow) < 1e-9:
        return 1.0, 0.0, iw, ow
    b = (iw - (adaptation / 100.0) * (iw / ow)) / ((iw / ow) - 1.0)
    a = ow / (iw / (iw + b))
    return a, b, iw, ow


def tm_forward(x, max_input_nits: float) -> np.ndarray:
    """Scene-linear to display-linear (1.0 = 100 nits)."""
    a, b, iw, ow = tm_constants(max_input_nits)
    x = np.clip(np.asarray(x, dtype=F32), F32(0.0), F32(iw))
    if b == 0.0:
        return np.minimum(x, F32(ow))
    return np.clip(F32(a) * x / (x + F32(b)), F32(0.0), F32(ow)).astype(F32, copy=False)


def tm_inverse(y, max_input_nits: float) -> np.ndarray:
    """Display-linear to scene-linear. Exact inverse of tm_forward."""
    a, b, iw, ow = tm_constants(max_input_nits)
    y = np.clip(np.asarray(y, dtype=F32), F32(0.0), F32(ow))
    if b == 0.0:
        return y
    den = np.maximum(F32(a) - y, F32(1e-9))
    return np.minimum(F32(b) * y / den, F32(iw)).astype(F32, copy=False)


def pq_display_to_scene(display_rel, params: dict, luma_weights) -> np.ndarray:
    """PQ display light (1.0 = reference white) to scene light.

    BT.2408 bridge: undo the HLG OOTF for a 1000 nit display, then scale so
    HDR reference white lands on 1.0. Keeps chromaticity.
    """
    ref = float(params.get("reference_white_nits", 203.0))
    peak = float(params.get("bridge_peak_nits", 1000.0))
    gamma = float(params.get("ootf_gamma", 1.2))
    hlg_white = float(params.get("hlg_reference_white", 0.26496256))
    fd = np.maximum(np.asarray(display_rel, dtype=F32) * F32(ref / peak), F32(0.0))  # relative to peak
    w = np.asarray(luma_weights, dtype=F32)
    yd = np.maximum(fd @ w, F32(1e-10))
    scale = np.power(yd, F32((1.0 - gamma) / gamma))
    es = fd * scale[..., None]
    return (es / F32(hlg_white)).astype(F32, copy=False)


def pq_scene_to_display(scene, params: dict, luma_weights) -> np.ndarray:
    ref = float(params.get("reference_white_nits", 203.0))
    peak = float(params.get("bridge_peak_nits", 1000.0))
    gamma = float(params.get("ootf_gamma", 1.2))
    hlg_white = float(params.get("hlg_reference_white", 0.26496256))
    es = np.maximum(np.asarray(scene, dtype=F32) * F32(hlg_white), F32(0.0))
    w = np.asarray(luma_weights, dtype=F32)
    ys = np.maximum(es @ w, F32(1e-10))
    fd = es * np.power(ys, F32(gamma - 1.0))[..., None]
    return (fd * F32(peak / ref)).astype(F32, copy=False)
