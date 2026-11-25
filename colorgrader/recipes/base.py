"""Shared parts for editor recipes.

context() collects everything a recipe module needs from the project, as plain
data. The helpers below build the recipe layout:

    recipe = {
      "editor", "editor_name", "edition", "title", "intro": [text, ...],
      "sections": [
        {"id", "title", "where", "notes": [text], "items": [setting],
         "groups": [{"title", "subtitle", "notes": [text], "nodes": [node]}],
         "nodes": [node]}
      ]
    }
    setting = {"control", "value", "reason"}
    node    = {"number", "label", "title", "tool", "how", "settings": [setting],
               "curves": [{"curve", "columns", "rows", "reason", "how"}], "notes": [text]}
"""
from __future__ import annotations

from typing import Any

from .. import media
from ..analysis import names
from ..color import working as W
from ..color.transforms import correction_gains
from ..grade import model, pipeline


def context(proj, st: dict[str, Any]) -> dict[str, Any]:
    """Plain data for recipe modules: look, finish, output, clips with profiles and shots."""
    from .. import export

    state = proj.state
    strength = float(state.get("strength", 100))
    look_full = state["look"]
    look = model.at_strength(look_full, strength / 100.0)
    look["reasons"] = dict(look_full.get("reasons") or {})
    out = W.OUTPUT_SPACES[st["output_space"]]
    lut_names = export.lut_names(proj)
    clips = []
    for clip in state["clips"]:
        info = clip["info"]
        entry: dict[str, Any] = {"id": clip["id"], "name": clip["name"], "fps": info["fps"] or 25.0, "is_image": info["is_image"],
                                 "duration": info["duration"], "bit_depth": info["bit_depth"], "shots": [],
                                 "color_rebuild": proj.color_rebuild_on(clip), "bits_per_pixel": media.bits_per_pixel(info)}
        try:
            cam, prof = proj.profile_of(clip)
            idt, _ = proj.idt(clip)
        except Exception as exc:  # no camera or a broken profile: the recipe says so
            entry["problem"] = str(exc)
            clips.append(entry)
            continue
        spec = prof.get("input") or {}
        corr = prof.get("corrections") or {}
        entry.update({
            "camera": cam["name"], "profile": prof["name"], "profile_key": f"{clip.get('camera_id')}/{clip.get('profile_id')}",
            "input_type": spec.get("type", "curve"), "lut_file": spec.get("lut_file"),
            "curve": {"id": idt.curve.id, "name": idt.curve.name, "kind": idt.curve.kind, "code_values": idt.curve.code_values,
                      "labels": dict(idt.curve.spec.get("labels") or {})},
            "gamut": {"id": idt.gamut.id, "name": idt.gamut.name, "labels": dict(idt.gamut.spec.get("labels") or {})},
            "data_range": prof.get("data_range", "auto"), "full_range": bool(idt.full_range),
            "corrections": corr, "camera_gains": [round(float(v), 3) for v in correction_gains(corr)],
            "camera_saturation": pipeline.camera_saturation(corr),
            "luts": lut_names["profiles"].get(f"{clip.get('camera_id')}/{clip.get('profile_id')}"),
        })
        for s in proj.shots_of(clip["id"]):
            g = model.quantize_shot(s["grade"]) if s.get("grade") else model.neutral_shot()
            entry["shots"].append({"id": s["id"], "index": s["index"], "start": s["start"], "end": s["end"], "grade": g,
                                   "reasons": dict((s.get("grade") or {}).get("reasons") or {}),
                                   "scene": s.get("scene"), "is_lead": bool(s.get("is_lead"))})
        clips.append(entry)
    return {
        "settings": dict(st), "strength": strength, "look": look, "look_name": look_full.get("name") or "Look",
        "look_description": look_full.get("description") or "", "neutral_look": model.is_neutral_look(look),
        "finish": state.get("finish") or model.neutral_finish(), "graded": bool(state.get("graded")),
        "output": {"id": st["output_space"], "name": out["name"], "resolve_gamma": out["resolve_gamma"],
                   "max_input_nits": int(st["max_input_nits"])},
        "clips": clips, "lut_names": lut_names, "edition": st.get("resolve_edition", "free"),
        "pivot": model.PIVOT, "stop_points": 10.0,
    }


# ------------------------------------------------------------------ builders
def setting(control: str, value: Any, reason: str = "") -> dict[str, str]:
    return {"control": control, "value": str(value), "reason": reason}


def node(number: str, label: str, title: str, tool: str, how: str = "", settings: list[dict[str, str]] | None = None,
         curves: list[dict[str, Any]] | None = None, notes: list[str] | None = None) -> dict[str, Any]:
    return {"number": number, "label": label, "title": title, "tool": tool, "how": how, "settings": settings or [],
            "curves": curves or [], "notes": notes or []}


def curve_table(curve: str, columns: list[str], rows: list[list[str]], reason: str = "", how: str = "") -> dict[str, Any]:
    return {"curve": curve, "columns": columns, "rows": rows, "reason": reason, "how": how}


# ------------------------------------------------------------------- formats
def num(v: float, decimals: int, signed: bool = False) -> str:
    text = f"{float(v):+.{decimals}f}" if signed else f"{float(v):.{decimals}f}"
    if float(text) == 0.0 and signed:
        return f"{0.0:.{decimals}f}"
    return text


def rgb(values, decimals: int, signed: bool = False) -> str:
    r, g, b = values
    return f"R {num(r, decimals, signed)}   G {num(g, decimals, signed)}   B {num(b, decimals, signed)}"


def timecode(seconds: float, fps: float) -> str:
    """Clip time as minutes:seconds:frames."""
    fps_i = max(1, int(round(fps)))
    total = int(round(seconds * fps))
    return f"{total // fps_i // 60:02d}:{total // fps_i % 60:02d}:{total % fps_i:02d}"


def hue_label(deg: float) -> str:
    """Color name plus the place between Resolve's six color buttons."""
    deg = deg % 360.0
    anchors = [(0, "red"), (60, "yellow"), (120, "green"), (180, "cyan"), (240, "blue"), (300, "magenta"), (360, "red")]
    for (a, an), (b, bn) in zip(anchors[:-1], anchors[1:]):
        if a <= deg <= b:
            t = (deg - a) / 60.0
            if t < 0.08:
                place = f"on {an}"
            elif t > 0.92:
                place = f"on {bn}"
            elif 0.42 <= t <= 0.58:
                place = f"halfway between {an} and {bn}"
            else:
                place = f"{round(t * 100):d}% of the way from {an} to {bn}"
            return f"{names.hue_name(deg)} ({place})"
    return names.hue_name(deg)


def hue_direction(deg: float, rotation: float) -> str:
    """Where a hue rotation sends a color, in words."""
    if abs(rotation) < 1e-9:
        return "no change"
    order = ["red", "yellow", "green", "cyan", "blue", "magenta"]
    i = int((deg % 360.0) // 60)
    return f"toward {order[(i + 1) % 6] if rotation > 0 else order[i if (deg % 60.0) > 1e-6 else (i - 1) % 6]}"


def is_neutral_curve(points, neutral: float) -> bool:
    return not points or all(abs(v - neutral) < 1e-9 for _, v in points)


# ---------------------------------------------------------------------- text
def to_text(recipe: dict[str, Any]) -> str:
    """The recipe as plain text, for printing or pasting into notes."""
    out: list[str] = [recipe.get("title", "Recipe"), "=" * len(recipe.get("title", "Recipe")), ""]
    out += list(recipe.get("intro", [])) + [""]

    def put_settings(items, indent="  "):
        width = max((len(i["control"]) for i in items), default=0)
        for i in items:
            out.append(f"{indent}{i['control'].ljust(width)}  {i['value']}")
            if i.get("reason"):
                out.append(f"{indent}{' ' * width}  Why: {i['reason']}")

    def put_node(n):
        out.append(f"  Node {n['number']}  [{n['label']}]  {n['title']}")
        out.append(f"    Tool: {n['tool']}")
        if n.get("how"):
            out.append(f"    How: {n['how']}")
        put_settings(n.get("settings", []), "    ")
        for c in n.get("curves", []):
            out.append(f"    Curve: {c['curve']}")
            if c.get("how"):
                out.append(f"      How: {c['how']}")
            widths = [max(len(str(col)), *(len(str(r[i])) for r in c["rows"])) for i, col in enumerate(c["columns"])]
            out.append("      " + "  ".join(str(col).ljust(w) for col, w in zip(c["columns"], widths)))
            for r in c["rows"]:
                out.append("      " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))
            if c.get("reason"):
                out.append(f"      Why: {c['reason']}")
        for note in n.get("notes", []):
            out.append(f"    Note: {note}")
        out.append("")

    for sec in recipe.get("sections", []):
        out += [sec["title"], "-" * len(sec["title"])]
        if sec.get("where"):
            out.append(f"Where: {sec['where']}")
        for note in sec.get("notes", []):
            out.append(note)
        if sec.get("items"):
            out.append("")
            put_settings(sec["items"])
        out.append("")
        for n in sec.get("nodes", []):
            put_node(n)
        for grp in sec.get("groups", []):
            out.append(f"* {grp['title']}")
            if grp.get("subtitle"):
                out.append(f"  {grp['subtitle']}")
            for note in grp.get("notes", []):
                out.append(f"  {note}")
            out.append("")
            for n in grp.get("nodes", []):
                put_node(n)
    return "\n".join(out).rstrip() + "\n"
