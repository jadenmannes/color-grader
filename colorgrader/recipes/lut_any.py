"""Recipe for any editor with LUT support (Premiere Pro, Final Cut Pro, CapCut, others).

One camera LUT per recording profile carries input conversion, look and output.
Per-shot balance stays a manual step with the editor's own basic controls.
This module doubles as the template for a new editor module.
"""
from __future__ import annotations

from typing import Any

from .base import setting

ID = "lut"
NAME = "Any editor (LUT)"
ORDER = 90
EDITIONS: list[tuple[str, str]] = []


def build(ctx: dict[str, Any]) -> dict[str, Any]:
    out = ctx["output"]
    names = ctx["lut_names"]
    title = f"LUT recipe for any editor: {ctx['look_name']} at {ctx['strength']:g}%"
    intro = [
        f"Output: {out['name']}. Each LUT has {ctx['settings']['lut_size']} points per side.",
        "One LUT per camera profile holds the input conversion, the look and the output conversion.",
        "Apply the LUT to clips as recorded. Put exposure and white balance corrections before the LUT.",
    ]
    sections: list[dict[str, Any]] = [{
        "id": "export", "title": "1. Export the LUT set", "where": "Export tab, Export LUT set",
        "notes": ["The folder holds the .cube files and a text file with the placement of each one."], "items": []}]
    items = []
    for p in names["profiles"].values():
        items.append(setting(p["camera"], f"{p['camera_name']} {p['profile_name']} clips, as recorded",
                             "Input conversion, look and output in one file."))
    sections.append({
        "id": "apply", "title": "2. Apply the camera LUT", "where": "Your editor's LUT slot",
        "notes": [
            "Premiere Pro: Lumetri Color, Basic Correction, Input LUT, Browse. Keep the clip's color space interpretation at its default.",
            "Final Cut Pro: Effects, Color, Custom LUT. Pick the file, set Input and Output to Rec. 709.",
            "DaVinci Resolve: right-click the clip node, LUT. Use no other conversion on the clip.",
            "Other editors: the first LUT or input LUT slot of the color panel.",
        ], "items": items})
    groups = []
    for clip in ctx["clips"]:
        if clip.get("problem") or not clip["shots"]:
            continue
        for shot in clip["shots"]:
            why = shot["reasons"]
            notes = [t for t in (why.get("gain"), why.get("offset"), why.get("match"), why.get("skin")) if t]
            label = clip["name"] + (f", shot {shot['index'] + 1}" if len(clip["shots"]) > 1 else "")
            groups.append({"title": label, "subtitle": f"{clip['camera']}, {clip['profile']}",
                           "notes": notes or ["No balance change needed."], "nodes": []})
    sections.append({
        "id": "balance", "title": "3. Balance each shot before the LUT",
        "where": "Exposure, temperature and tint controls of your editor, placed before the LUT",
        "notes": ["A LUT holds no per-shot values. The lines below say what the app changed in each shot. Set the same by eye with the scopes.",
                  "For exact per-shot values use the DaVinci Resolve recipe, or export the graded clip from this app."],
        "groups": groups})
    sections.append({
        "id": "limits", "title": "4. Limits of the LUT route", "where": "",
        "notes": ["Vignette, grain, noise reduction and sharpening stay outside the LUT: a LUT changes color only. Add them with your editor's own tools, or export the graded clip from this app.",
                  "Rebuild the LUT set after you change the look or its strength."], "items": []})
    return {"editor": ID, "editor_name": NAME, "edition": "", "title": title, "intro": intro, "sections": sections}
