"""Where the app keeps its files.

Everything the user owns lives in one folder in the user's home directory:
cameras, curves, settings, saved looks, projects and caches. App updates
replace the code folder only, so they never touch this data.

Set COLORGRADER_HOME to move the folder (tests do this).
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SHIPPED_DATA = PACKAGE_DIR / "data"
WEB_DIR = PACKAGE_DIR / "web"

SUBDIRS = ("cameras", "curves", "gamuts", "luts", "looks", "projects", "cache", "exports")


def home() -> Path:
    env = os.environ.get("COLORGRADER_HOME")
    root = Path(env).expanduser() if env else Path.home() / "ColorGrader"
    root.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (root / sub).mkdir(exist_ok=True)
    return root


def sub(name: str) -> Path:
    p = home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_id(text: str) -> str:
    """Turn any name into a file-safe id."""
    out = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-") or "item"
