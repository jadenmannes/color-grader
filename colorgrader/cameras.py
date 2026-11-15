"""Camera library: JSON files in the user folder.

A camera holds recording profiles. A profile stores the input conversion
(curve + gamut, or a maker LUT), bit depth, data range, detection hints and
default corrections. Nothing camera-specific lives in code.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from . import library, media, paths

SCHEMA = 1
RANGES = ("auto", "video", "full")
DEFAULT_CORRECTIONS = {"exposure": 0.0, "temperature": 0.0, "tint": 0.0, "saturation": 1.0}


class CameraError(ValueError):
    pass


def _dir() -> Path:
    return paths.sub("cameras")


def _seed_file() -> Path:
    return _dir() / ".seeded.json"


def seed() -> None:
    """Copy shipped cameras into the user folder once. Deleted cameras stay deleted."""
    try:
        seeded = set(json.loads(_seed_file().read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        seeded = set()
    changed = False
    for src in sorted((paths.SHIPPED_DATA / "cameras").glob("*.json")):
        if src.stem in seeded:
            continue
        dst = _dir() / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
        seeded.add(src.stem)
        changed = True
    if changed:
        _seed_file().write_text(json.dumps(sorted(seeded)), encoding="utf-8")
        library.reset()


def restore_defaults() -> list[str]:
    """Write the shipped cameras again, replacing edited copies."""
    restored = []
    for src in sorted((paths.SHIPPED_DATA / "cameras").glob("*.json")):
        shutil.copyfile(src, _dir() / src.name)
        restored.append(src.stem)
    library.reset()
    return restored


def list_cameras() -> list[dict[str, Any]]:
    seed()
    out = []
    for f in sorted(_dir().glob("*.json")):
        if f.name.startswith("."):
            continue
        try:
            cam = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(cam, dict) and cam.get("id"):
            out.append(cam)
    out.sort(key=lambda c: c.get("name", "").lower())
    return out


def get_camera(cam_id: str) -> dict[str, Any] | None:
    for cam in list_cameras():
        if cam["id"] == cam_id:
            return cam
    return None


def get_profile(cam: dict[str, Any], profile_id: str) -> dict[str, Any] | None:
    for p in cam.get("profiles", []):
        if p.get("id") == profile_id:
            return p
    return None


def normalize(cam: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and check a camera dict. Raises CameraError with a readable message."""
    cam = copy.deepcopy(cam)
    name = str(cam.get("name", "")).strip()
    if not name:
        raise CameraError("Give the camera a name.")
    cam["name"] = name
    cam["schema"] = SCHEMA
    cam["id"] = paths.safe_id(cam.get("id") or name)
    cam["maker"] = str(cam.get("maker", "")).strip()
    names = cam.get("metadata_names") or []
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",")]
    cam["metadata_names"] = [n for n in (str(x).strip() for x in names) if n]
    cam["notes"] = str(cam.get("notes", ""))
    defs = cam.get("definitions") or {}
    cam["definitions"] = {"curves": dict(defs.get("curves") or {}), "gamuts": dict(defs.get("gamuts") or {})}
    profiles = cam.get("profiles") or []
    if not profiles:
        raise CameraError("Add at least one recording profile.")
    lib = library.get()
    seen = set()
    for p in profiles:
        pname = str(p.get("name", "")).strip()
        if not pname:
            raise CameraError("Every profile needs a name.")
        p["name"] = pname
        p["id"] = paths.safe_id(p.get("id") or pname)
        if p["id"] in seen:
            raise CameraError(f"Two profiles share the id '{p['id']}'. Rename one.")
        seen.add(p["id"])
        inp = p.get("input") or {}
        kind = inp.get("type", "curve")
        if kind == "curve":
            for key, known, local in (("curve", lib.curve_specs, cam["definitions"]["curves"]),
                                      ("gamut", lib.gamut_specs, cam["definitions"]["gamuts"])):
                ref = inp.get(key)
                if not ref:
                    raise CameraError(f"Profile '{pname}': pick a {key}.")
                if ref not in known and ref not in local:
                    raise CameraError(f"Profile '{pname}': unknown {key} '{ref}'.")
            p["input"] = {"type": "curve", "curve": inp["curve"], "gamut": inp["gamut"]}
        elif kind == "lut":
            lut_file = inp.get("lut_file")
            if not lut_file:
                raise CameraError(f"Profile '{pname}': load the maker's conversion LUT (.cube).")
            if not (paths.sub("luts") / lut_file).is_file() and not inp.get("lut_text"):
                raise CameraError(f"Profile '{pname}': LUT file '{lut_file}' is missing from the luts folder.")
            out = inp.get("lut_output") or {}
            p["input"] = {"type": "lut", "lut_file": lut_file, "lut_input": inp.get("lut_input", "absolute"),
                          "lut_output": {"curve": out.get("curve", "rec709-gamma24"), "gamut": out.get("gamut", "rec709")}}
            if inp.get("lut_text"):
                p["input"]["lut_text"] = inp["lut_text"]
            for key, known in (("curve", lib.curve_specs), ("gamut", lib.gamut_specs)):
                if p["input"]["lut_output"][key] not in known:
                    raise CameraError(f"Profile '{pname}': unknown LUT output {key}.")
        else:
            raise CameraError(f"Profile '{pname}': input type must be 'curve' or 'lut'.")
        try:
            p["bit_depth"] = int(p.get("bit_depth") or 8)
        except (TypeError, ValueError):
            raise CameraError(f"Profile '{pname}': bit depth must be a number.")
        if p["bit_depth"] not in (8, 10, 12, 14, 16):
            raise CameraError(f"Profile '{pname}': bit depth must be 8, 10, 12, 14 or 16.")
        p["data_range"] = p.get("data_range", "auto")
        if p["data_range"] not in RANGES:
            raise CameraError(f"Profile '{pname}': data range must be auto, video or full.")
        det = p.get("detect") or {}
        p["detect"] = {
            "bit_depth": [int(b) for b in det.get("bit_depth", []) or []],
            "transfer": [str(t) for t in det.get("transfer", []) or []],
            "primaries": [str(t) for t in det.get("primaries", []) or []],
            "image": det.get("image", "any"),
        }
        corr = dict(DEFAULT_CORRECTIONS)
        for k, v in (p.get("corrections") or {}).items():
            if k in corr:
                try:
                    corr[k] = float(v)
                except (TypeError, ValueError):
                    raise CameraError(f"Profile '{pname}': correction '{k}' must be a number.")
        p["corrections"] = corr
    cam["profiles"] = profiles
    return cam


def save_camera(cam: dict[str, Any], replace_id: str | None = None) -> dict[str, Any]:
    """Validate and write a camera file. `replace_id` removes the old file on rename."""
    seed()
    cam = normalize(cam)
    for p in cam["profiles"]:  # materialize LUT text carried by an imported camera
        inp = p["input"]
        if inp.get("type") == "lut" and inp.get("lut_text"):
            target = paths.sub("luts") / inp["lut_file"]
            if not target.exists():
                target.write_text(inp["lut_text"], encoding="utf-8")
            del inp["lut_text"]
    if replace_id and replace_id != cam["id"]:
        old = _dir() / f"{replace_id}.json"
        if old.exists():
            old.unlink()
    (_dir() / f"{cam['id']}.json").write_text(json.dumps(cam, indent=2) + "\n", encoding="utf-8")
    library.reset()
    return cam


def delete_camera(cam_id: str) -> bool:
    f = _dir() / f"{paths.safe_id(cam_id)}.json"
    if f.exists():
        f.unlink()
        library.reset()
        return True
    return False


def export_camera(cam_id: str) -> dict[str, Any]:
    """A self-contained camera file: curve and gamut definitions and LUT text travel with it."""
    cam = get_camera(cam_id)
    if cam is None:
        raise CameraError(f"No camera with id '{cam_id}'.")
    cam = copy.deepcopy(cam)
    lib = library.get()
    defs = cam.setdefault("definitions", {"curves": {}, "gamuts": {}})
    defs.setdefault("curves", {})
    defs.setdefault("gamuts", {})
    for p in cam["profiles"]:
        inp = p["input"]
        refs = inp if inp.get("type") == "curve" else inp.get("lut_output", {})
        if refs.get("curve") in lib.curve_specs:
            defs["curves"][refs["curve"]] = lib.curve_specs[refs["curve"]]
        if refs.get("gamut") in lib.gamut_specs:
            defs["gamuts"][refs["gamut"]] = lib.gamut_specs[refs["gamut"]]
        if inp.get("type") == "lut":
            f = paths.sub("luts") / inp["lut_file"]
            if f.is_file():
                inp["lut_text"] = f.read_text(encoding="utf-8", errors="replace")
    return cam


def import_camera(data: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    if not isinstance(data, dict) or not data.get("profiles"):
        raise CameraError("This file is not a camera file.")
    cam_id = paths.safe_id(data.get("id") or data.get("name", ""))
    if get_camera(cam_id) is not None and not overwrite:
        raise CameraError(f"A camera with id '{cam_id}' exists. Import with overwrite to replace it.")
    # definitions must be visible before validation
    data = copy.deepcopy(data)
    data["id"] = cam_id
    tmp = _dir() / f"{cam_id}.json"
    backup = tmp.read_text(encoding="utf-8") if tmp.exists() else None
    tmp.write_text(json.dumps(data), encoding="utf-8")
    library.reset()
    try:
        return save_camera(data)
    except CameraError:
        if backup is None:
            tmp.unlink(missing_ok=True)
        else:
            tmp.write_text(backup, encoding="utf-8")
        library.reset()
        raise


# ---------------------------------------------------------------- matching
def match_camera(info: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Find the camera whose metadata name appears in the file metadata. Longest name wins."""
    text = media.metadata_text(info).lower()
    best, best_len, best_name = None, 0, ""
    for cam in list_cameras():
        for name in cam.get("metadata_names", []):
            n = name.lower()
            if n and n in text and len(n) > best_len:
                best, best_len, best_name = cam, len(n), name
    return best, best_name


def image_stats(info: dict[str, Any]) -> dict[str, float]:
    """Quick luma statistics in absolute code values, from up to three frames."""
    from .color.ycbcr import signal_to_absolute

    times = [0.0] if info["is_image"] else [info["duration"] * f for f in (0.15, 0.5, 0.85)]
    lumas = []
    for t in times:
        try:
            buf, w, h = media.read_frame(info, t, max_width=480)
        except media.MediaError:
            continue
        rgb = media.raw_to_rgb(buf, w, h, info)
        rgb = signal_to_absolute(rgb, info["full_range"], info["bit_depth"])
        lumas.append((rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).ravel())
    if not lumas:
        return {}
    y = np.concatenate(lumas)
    q = np.percentile(y, [0.1, 1, 50, 99, 99.9])
    return {"p0_1": float(q[0]), "p1": float(q[1]), "p50": float(q[2]), "p99": float(q[3]), "p99_9": float(q[4]),
            "spread": float(q[3] - q[1])}


def rank_profiles(cam: dict[str, Any], info: dict[str, Any], stats: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Score each profile of a camera against file tags, then against the picture."""
    lib = library.get()
    out = []
    transfer = (info.get("color_transfer") or "unknown").lower()
    primaries = (info.get("color_primaries") or "unknown").lower()
    for p in cam.get("profiles", []):
        det = p.get("detect") or {}
        score, why = 0.0, []
        if det.get("bit_depth"):
            if info["bit_depth"] in det["bit_depth"]:
                score += 1.0
                why.append(f"{info['bit_depth']}-bit matches")
            else:
                score -= 2.0
                why.append(f"{info['bit_depth']}-bit does not match")
        if det.get("transfer"):
            if transfer in [t.lower() or "unknown" for t in det["transfer"]]:
                score += 3.0
                why.append(f"transfer tag '{transfer}' matches")
            else:
                score -= 3.0
                why.append(f"transfer tag '{transfer}' does not match")
        if det.get("primaries"):
            if primaries in [t.lower() or "unknown" for t in det["primaries"]]:
                score += 2.0
                why.append(f"primaries tag '{primaries}' matches")
            else:
                score -= 2.0
                why.append(f"primaries tag '{primaries}' does not match")
        if stats:
            inp = p.get("input") or {}
            ref = inp.get("curve") if inp.get("type", "curve") == "curve" else None
            hint = det.get("image", "any")
            if ref and ref in lib.curve_specs or (ref and ref in (cam.get("definitions", {}).get("curves") or {})):
                try:
                    curve = lib.curve(ref)
                except KeyError:
                    curve = None
                if curve is not None and curve.kind == "scene" and hint == "log":
                    black = curve.black_code()
                    if stats["p0_1"] < black - 0.035:
                        score -= 3.0
                        why.append("picture holds values below this curve's black level")
                    else:
                        score += 1.5
                        why.append("black level fits this log curve")
                    mid = float(curve.decode(np.array([stats["p50"]], dtype=np.float32))[0])
                    if 0.01 <= mid <= 1.5:
                        score += 0.5
                    else:
                        score -= 1.0
                        why.append("midtones land outside a plausible exposure")
                elif hint == "display":
                    if stats["p0_1"] < 0.06 or stats["spread"] > 0.75:
                        score += 1.5
                        why.append("picture reaches true black, as a finished picture does")
                    else:
                        score -= 1.0
                        why.append("picture looks flat for a finished picture")
        out.append({"profile_id": p["id"], "profile_name": p["name"], "score": round(score, 2), "why": why})
    out.sort(key=lambda r: -r["score"])
    return out


def detect(info: dict[str, Any]) -> dict[str, Any]:
    """Match camera and profile for a clip. Flags the result when the user must confirm."""
    cam, matched_name = match_camera(info)
    stats = image_stats(info)
    result: dict[str, Any] = {"camera_id": None, "profile_id": None, "matched_name": matched_name, "method": "none",
                              "needs_confirm": True, "candidates": [], "image_stats": stats, "message": ""}
    if cam is None:
        result["message"] = "No camera in your library matches this file's metadata."
        result["draft"] = draft_from_metadata(info, stats)
        return result
    ranked = rank_profiles(cam, info, stats)
    result["camera_id"] = cam["id"]
    result["candidates"] = ranked
    if not ranked:
        return result
    top = ranked[0]
    result["profile_id"] = top["profile_id"]
    gap = top["score"] - ranked[1]["score"] if len(ranked) > 1 else 99.0
    tag_based = any("tag" in w for w in top["why"])
    if len(ranked) == 1 and top["score"] >= 0.0:
        result["method"] = "metadata"
        result["needs_confirm"] = False
        result["message"] = f"{cam['name']}, {top['profile_name']}: camera name read from metadata."
    elif top["score"] >= 3.0 and gap >= 1.5:
        result["method"] = "metadata" if tag_based else "image"
        result["needs_confirm"] = False
        result["message"] = f"{cam['name']}, {top['profile_name']}: read from metadata."
    else:
        result["method"] = "image"
        close = [r["profile_name"] for r in ranked if top["score"] - r["score"] < 1.5]
        result["message"] = (f"{cam['name']} matched by name. The metadata does not settle the profile. "
                             f"Best estimate from the picture: {top['profile_name']}. Confirm or pick: {', '.join(close)}.")
    return result


_TRANSFER_TO_CURVE = {"bt709": "rec709-gamma24", "smpte170m": "rec709-gamma24", "bt470bg": "rec709-gamma24",
                      "iec61966-2-1": "srgb", "arib-std-b67": "hlg", "smpte2084": "pq", "linear": "linear"}
_PRIMARIES_TO_GAMUT = {"bt709": "rec709", "smpte170m": "rec709", "bt470bg": "rec709", "bt2020": "rec2020",
                       "smpte432": "p3-d65", "smpte431": "p3-d65"}


def draft_from_metadata(info: dict[str, Any], stats: dict[str, float] | None = None) -> dict[str, Any]:
    """Pre-fill the add-camera form with what the file shows."""
    make, model = info.get("make", "").strip(), info.get("model", "").strip()
    name = (f"{make} {model}".strip()) or Path(info["path"]).stem
    transfer = (info.get("color_transfer") or "").lower()
    primaries = (info.get("color_primaries") or "").lower()
    curve = _TRANSFER_TO_CURVE.get(transfer)
    guess_note = ""
    if curve is None:
        looks_log = bool(stats) and stats.get("p0_1", 0) > 0.06 and stats.get("spread", 1) < 0.8
        curve = "" if looks_log else "rec709-gamma24"
        guess_note = ("The file carries no transfer tag. The picture looks like log footage: pick the log curve."
                      if looks_log else "The file carries no transfer tag. The picture looks finished: Rec.709 assumed.")
    gamut = _PRIMARIES_TO_GAMUT.get(primaries, "rec709" if curve in ("rec709-gamma24", "srgb") else "")
    return {
        "name": name, "maker": make, "metadata_names": [n for n in [model or make] if n],
        "notes": guess_note,
        "profiles": [{
            "name": "Default",
            "input": {"type": "curve", "curve": curve, "gamut": gamut},
            "bit_depth": info["bit_depth"],
            "data_range": "auto",
            "detect": {"bit_depth": [info["bit_depth"]], "transfer": [transfer] if transfer else [],
                       "primaries": [primaries] if primaries else [], "image": "any"},
            "corrections": dict(DEFAULT_CORRECTIONS),
        }],
        "definitions": {"curves": {}, "gamuts": {}},
    }
