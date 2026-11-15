"""App settings, stored in the user folder."""
from __future__ import annotations

import json
from typing import Any

from . import paths
from .color import working as W

DEFAULTS: dict[str, Any] = {
    "target_editor": "resolve",
    "resolve_edition": "free",          # free | studio
    "lut_size": 33,                     # 17 | 33 | 65
    "lut_format": "cube",
    "export_format": "h265-10bit-mp4",  # see export.FORMATS
    "output_space": "rec709-gamma24",
    "look_strength": 80,                # percent, default for new looks
    "max_input_nits": 1000,             # highlight range of the tone mapping
    "analysis_fps": 2.0,                # analysis frames per second (minimum 1)
    "preview_width": 1280,
    "export_folder": "",
}

CHOICES = {
    "target_editor": ["resolve"],
    "resolve_edition": ["free", "studio"],
    "lut_size": [17, 33, 65],
    "lut_format": ["cube"],
    "output_space": list(W.OUTPUT_SPACES.keys()),
}


def _file():
    return paths.home() / "settings.json"


def load() -> dict[str, Any]:
    out = dict(DEFAULTS)
    try:
        data = json.loads(_file().read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out.update({k: v for k, v in data.items() if k in DEFAULTS})
    except (OSError, json.JSONDecodeError):
        pass
    return out


def save(values: dict[str, Any]) -> dict[str, Any]:
    cur = load()
    for k, v in values.items():
        if k not in DEFAULTS:
            continue
        if k in CHOICES and v not in CHOICES[k]:
            raise ValueError(f"{k}: pick one of {CHOICES[k]}")
        if k == "look_strength":
            v = int(min(100, max(0, int(v))))
        if k == "analysis_fps":
            v = float(min(8.0, max(1.0, float(v))))
        if k == "max_input_nits":
            v = int(min(10000, max(200, int(v))))
        if k == "preview_width":
            v = int(min(3840, max(480, int(v))))
        cur[k] = v
    _file().write_text(json.dumps(cur, indent=2) + "\n", encoding="utf-8")
    return cur
