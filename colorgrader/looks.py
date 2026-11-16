"""Saved looks: one JSON file per look in the user folder, with a small preview picture.

A saved look holds the shared look at full strength, the strength and the
finish. It holds no per-shot balance, so it carries over to any footage.
"""
from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from . import paths
from .grade import model


class LookError(ValueError):
    pass


def _dir():
    return paths.sub("looks")


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "look"


def list_looks() -> list[dict[str, Any]]:
    out = []
    for f in sorted(_dir().glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict) or "look" not in d:
            continue
        d["id"] = f.stem
        d["has_thumb"] = f.with_suffix(".jpg").exists()
        out.append(d)
    out.sort(key=lambda d: d.get("created", 0), reverse=True)
    return out


def get_look(look_id: str) -> dict[str, Any]:
    f = _dir() / f"{paths.safe_id(look_id)}.json"
    if not f.exists():
        raise LookError("No such saved look.")
    d = json.loads(f.read_text(encoding="utf-8"))
    d["id"] = f.stem
    return d


def save_look(proj, name: str, thumb_jpeg: bytes | None = None) -> dict[str, Any]:
    name = (name or "").strip()
    if not name:
        raise LookError("Give the look a name.")
    look = copy.deepcopy(proj.state["look"])
    if model.is_neutral_look(look):
        raise LookError("The current look is neutral. Build a grade first.")
    look["name"] = name
    base = _slug(name)
    look_id, n = base, 2
    while (_dir() / f"{look_id}.json").exists():
        look_id, n = f"{base}-{n}", n + 1
    ref = proj.state.get("reference") or {}
    data = {"schema": 1, "name": name, "description": look.get("description", ""), "created": time.time(),
            "strength": proj.state.get("strength", 80), "look": look, "finish": copy.deepcopy(proj.state.get("finish") or model.neutral_finish()),
            "origin": {"source": look.get("source", ""), "reference": ref.get("name", "")}}
    (_dir() / f"{look_id}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if thumb_jpeg:
        (_dir() / f"{look_id}.jpg").write_bytes(thumb_jpeg)
    data["id"] = look_id
    data["has_thumb"] = bool(thumb_jpeg)
    return data


def delete_look(look_id: str) -> None:
    f = _dir() / f"{paths.safe_id(look_id)}.json"
    if not f.exists():
        raise LookError("No such saved look.")
    f.unlink()
    f.with_suffix(".jpg").unlink(missing_ok=True)


def thumb_path(look_id: str):
    return _dir() / f"{paths.safe_id(look_id)}.jpg"


def apply_look(proj, look_id: str) -> dict[str, Any]:
    """Put a saved look into the project. The caller rebuilds the grade so skin and the quality rules follow the footage."""
    d = get_look(look_id)
    look = model.neutral_look()
    look.update({k: v for k, v in d["look"].items() if k in look})
    look["source"] = "saved"
    look["name"] = d.get("name", "Saved look")
    look["skin"] = {"hue_vs_hue": [], "hue_vs_sat": []}   # skin protection depends on the footage: rebuilt on apply
    reasons = dict(look.get("reasons") or {})
    reasons.pop("skin", None)
    look["reasons"] = reasons
    proj.set_look(look)
    proj.set_strength(float(d.get("strength", proj.state.get("strength", 80))))
    if d.get("finish"):
        proj.set_finish(d["finish"])
    return d
