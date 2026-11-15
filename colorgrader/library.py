"""Curve and gamut lookup.

Sources, later ones win on the same id:
  1. shipped data files (colorgrader/data/curves, .../gamuts)
  2. "definitions" blocks inside camera files in the user folder
  3. user files (~/ColorGrader/curves, ~/ColorGrader/gamuts)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import paths
from .color.curves import Curve
from .color.gamuts import Gamut


def _load_dir(folder: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not folder.exists():
        return out
    for f in sorted(folder.glob("*.json")):
        try:
            spec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(spec, dict) and spec.get("id"):
            out[spec["id"]] = spec
    return out


class ColorLibrary:
    def __init__(self):
        self.reload()

    def reload(self) -> None:
        curves = _load_dir(paths.SHIPPED_DATA / "curves")
        gamuts = _load_dir(paths.SHIPPED_DATA / "gamuts")
        for cam in _load_dir(paths.sub("cameras")).values():
            defs = cam.get("definitions") or {}
            curves.update({k: v for k, v in (defs.get("curves") or {}).items() if isinstance(v, dict)})
            gamuts.update({k: v for k, v in (defs.get("gamuts") or {}).items() if isinstance(v, dict)})
        curves.update(_load_dir(paths.sub("curves")))
        gamuts.update(_load_dir(paths.sub("gamuts")))
        self.curve_specs = curves
        self.gamut_specs = gamuts
        self._curves: dict[str, Curve] = {}
        self._gamuts: dict[str, Gamut] = {}

    def curve(self, ref: str | dict[str, Any]) -> Curve:
        if isinstance(ref, dict):
            return Curve(ref)
        if ref not in self._curves:
            if ref not in self.curve_specs:
                raise KeyError(f"unknown curve: {ref}")
            self._curves[ref] = Curve(self.curve_specs[ref])
        return self._curves[ref]

    def gamut(self, ref: str | dict[str, Any]) -> Gamut:
        if isinstance(ref, dict):
            return Gamut(ref)
        if ref not in self._gamuts:
            if ref not in self.gamut_specs:
                raise KeyError(f"unknown gamut: {ref}")
            self._gamuts[ref] = Gamut(self.gamut_specs[ref])
        return self._gamuts[ref]

    def list_curves(self) -> list[dict]:
        return [{"id": s["id"], "name": s.get("name", s["id"]), "kind": s.get("kind", "scene"),
                 "source": s.get("source", ""), "notes": s.get("notes", ""),
                 "recommended_range": s.get("recommended_range", "auto")}
                for s in sorted(self.curve_specs.values(), key=lambda s: s.get("name", s["id"]).lower())]

    def list_gamuts(self) -> list[dict]:
        return [{"id": s["id"], "name": s.get("name", s["id"]), "source": s.get("source", "")}
                for s in sorted(self.gamut_specs.values(), key=lambda s: s.get("name", s["id"]).lower())]


_lib: ColorLibrary | None = None


def get() -> ColorLibrary:
    global _lib
    if _lib is None:
        _lib = ColorLibrary()
    return _lib


def reset() -> None:
    global _lib
    _lib = None
