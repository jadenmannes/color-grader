"""Editor recipes: the grade as steps for one editing program.

Each editor lives in its own module in this folder. A module exposes

    ID = "resolve"                    # short id, stored in the settings
    NAME = "DaVinci Resolve"          # shown in the app
    EDITIONS = [("free", "Free"), ("studio", "Studio")]   # optional
    def build(ctx: dict) -> dict      # the recipe, see base.py for the layout

The registry finds every module here on first use. Adding Premiere Pro or
Final Cut Pro means adding one file. No other code changes.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from . import base

_registry: dict[str, Any] | None = None


def registry() -> dict[str, Any]:
    global _registry
    if _registry is None:
        found: dict[str, Any] = {}
        for mod in pkgutil.iter_modules(__path__):
            if mod.name.startswith("_") or mod.name == "base":
                continue
            module = importlib.import_module(f"{__name__}.{mod.name}")
            if hasattr(module, "ID") and hasattr(module, "build"):
                found[module.ID] = module
        _registry = found
    return _registry


def available() -> list[dict[str, Any]]:
    """Editors for the settings screen."""
    return [{"id": m.ID, "name": m.NAME, "editions": [{"id": a, "name": b} for a, b in getattr(m, "EDITIONS", [])]}
            for m in sorted(registry().values(), key=lambda m: getattr(m, "ORDER", 50))]


def build(proj, editor: str | None = None, edition: str | None = None) -> dict[str, Any]:
    """Build the recipe for the project's current grade."""
    from .. import settings

    st = settings.load()
    editor = editor or st["target_editor"]
    reg = registry()
    if editor not in reg:
        editor = "resolve" if "resolve" in reg else next(iter(reg))
    module = reg[editor]
    ctx = base.context(proj, st)
    ctx["edition"] = edition or st.get("resolve_edition", "free")
    recipe = module.build(ctx)
    recipe.setdefault("editor", module.ID)
    recipe.setdefault("editor_name", module.NAME)
    recipe["text"] = base.to_text(recipe)
    return recipe
