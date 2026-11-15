"""Plain-word names for colors and tints."""
from __future__ import annotations

import colorsys

import numpy as np

HUES = [(14, "red"), (40, "orange"), (66, "yellow"), (82, "yellow-green"), (150, "green"), (192, "teal"),
        (215, "cyan-blue"), (255, "blue"), (285, "purple"), (335, "magenta"), (361, "red")]


def hue_name(h_deg: float) -> str:
    h = h_deg % 360.0
    for limit, name in HUES:
        if h < limit:
            return name
    return "red"


def color_name(rgb) -> str:
    """Name a display color, for example 'dark teal' or 'pale orange'."""
    r, g, b = (float(np.clip(v, 0, 1)) for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.10 or v < 0.06:
        if v < 0.10:
            return "black"
        if v < 0.32:
            return "dark gray"
        if v < 0.62:
            return "gray"
        if v < 0.88:
            return "light gray"
        return "white"
    name = hue_name(h * 360.0)
    if name == "orange" and v < 0.45:
        name = "brown"
    if v < 0.28:
        prefix = "dark "
    elif s < 0.28:
        prefix = "pale " if v > 0.6 else "muted "
    elif v > 0.85 and s > 0.7:
        prefix = "vivid "
    else:
        prefix = ""
    return prefix + name


def tint_name(cb: float, cr: float) -> str:
    """Name a small color offset on the vectorscope (Cb right, Cr up)."""
    ang = float(np.degrees(np.arctan2(cr, cb))) % 360.0
    table = [(20, "blue"), (75, "magenta"), (112, "red"), (150, "warm orange"), (185, "yellow"), (225, "green-yellow"),
             (262, "green"), (305, "teal"), (345, "cyan-blue"), (361, "blue")]
    for limit, name in table:
        if ang < limit:
            return name
    return "blue"


def hex_color(rgb) -> str:
    r, g, b = (int(round(float(np.clip(v, 0, 1)) * 255)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"
