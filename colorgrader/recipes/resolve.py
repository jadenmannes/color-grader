"""DaVinci Resolve recipe.

Project settings first, then clip nodes (input conversion, balance, match),
then timeline nodes (the shared look, finish, output conversion). Every value
is the number the app renders with. Each setting carries one short reason.

The node math follows measured recreations of Resolve's tools (see ops.py).
Places where Resolve offers no typed value (hue curve positions, window
shapes, film grain) are marked as set by eye.
"""
from __future__ import annotations

from typing import Any

from . import base
from .base import curve_table, node, num, rgb, setting

ID = "resolve"
NAME = "DaVinci Resolve"
ORDER = 10
EDITIONS = [("free", "Free"), ("studio", "Studio")]

CST = "Color Space Transform"
CST_HOW = "Effects panel, ResolveFX Color, Color Space Transform. Drag it onto the node."
BARS_HOW = "Primaries palette, mode Color Bars. Type the values into the R, G and B fields."
ADJ_HOW = "Primaries palette, adjustment row under the wheels."
LUM_MIX = ("Lum Mix", "0.00", "Each channel then moves alone, the way the app computes it.")


def _cst_tail(tone: str = "None") -> list[dict[str, str]]:
    return [
        setting("Tone Mapping Method", tone, "No tone mapping here. The output conversion at the end of the timeline does it once."
                if tone == "None" else ""),
        setting("Gamut Mapping Method", "None", "Colors pass unchanged. The look holds the saturation limits."),
        setting("Apply Forward OOTF", "Off", "The conversion stays a plain curve and gamut change."),
        setting("Apply Inverse OOTF", "Off", "Same reason."),
    ]


def _data_levels(clip: dict[str, Any], lut_route: bool = False) -> dict[str, str]:
    if not lut_route and clip["curve"]["code_values"] == "absolute":
        return setting("Clip Attributes, Data Levels", "Full",
                       f"{clip['curve']['name']} counts code values on the full scale. Right-click the clip, Clip Attributes, Video.")
    value = "Full" if clip["full_range"] else "Video"
    return setting("Clip Attributes, Data Levels", value,
                   "Matches how the file stores its levels. Right-click the clip, Clip Attributes, Video.")


def _input_nodes(clip: dict[str, Any], ctx: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Nodes for the input conversion of one clip, numbered from 1, plus notes."""
    curve, gamut = clip["curve"], clip["gamut"]
    luts = clip.get("luts") or {}
    peak = ctx["output"]["max_input_nits"]
    notes: list[str] = []
    lut_alt = (f"LUT route: load {luts['input']} on this node in place of the transform (right-click the node, LUT). "
               f"Set Data Levels to {'Full' if clip['full_range'] else 'Video'} for the LUT route.") if luts.get("input") else ""
    gamma_name, gamut_name = curve["labels"].get("resolve"), gamut["labels"].get("resolve")
    nodes: list[dict[str, Any]] = []

    def lut_only(why: str) -> list[dict[str, Any]]:
        n = node("1", "IN", "Input conversion", "LUT", "Right-click the node, LUT, pick the file.",
                 [setting("LUT", luts.get("input", "Export the LUT set first"), why),
                  _data_levels(clip, lut_route=True)])
        return [n]

    if clip["input_type"] == "lut":
        n0 = node("1", "CAM LUT", "Maker conversion LUT", "LUT", "Right-click the node, LUT, pick the file.",
                  [setting("LUT", clip.get("lut_file") or "the maker LUT of this camera profile",
                           "The camera profile uses the maker's own conversion."), _data_levels(clip, lut_route=True)])
        nodes.append(n0)
        notes.append(f"The maker LUT delivers {gamut['name']} with {curve['name']}. The next node brings it into the working space.")
    if curve["kind"] == "pq" or not gamma_name or not gamut_name:
        why = ("HDR PQ clips pass through a display-to-scene step with no matching Resolve control. The LUT holds the exact conversion."
               if curve["kind"] == "pq" else "This curve or gamut has no entry in Resolve's list. The LUT holds the exact conversion.")
        if clip["input_type"] == "lut":
            notes.append("Export the LUT set and chain the INPUT LUT after the maker LUT.")
        return nodes + lut_only(why), notes
    first = str(len(nodes) + 1)
    if curve["kind"] == "display":
        # finished picture: undo the display curve and the tone mapping in Rec.709, then change gamut
        a = node(first, "IN A", "Input conversion, step 1: display to scene light", CST, CST_HOW, [
            _data_levels(clip),
            setting("Input Color Space", gamut_name, f"The clip holds a finished {gamut['name']} picture."),
            setting("Input Gamma", gamma_name, "The display curve of the finished picture."),
            setting("Output Color Space", gamut_name, "The gamut stays here. Tone mapping then runs on the original primaries."),
            setting("Output Gamma", "Linear", "Scene light for the next node."),
            setting("Tone Mapping Method", "DaVinci", "Run backwards, it restores highlight range. The output conversion compresses it again."),
            setting("Custom Max Input", "On, 100 nits", "A finished SDR picture peaks at 100 nits."),
            setting("Custom Max Output", f"On, {peak} nits", "The mirror image of the output conversion. An untouched clip comes back unchanged."),
            setting("Adaptation", "9.00", "Resolve's default. The output conversion uses the same value."),
            setting("Gamut Mapping Method", "None", "Colors pass unchanged."),
            setting("Apply Forward OOTF", "Off", "The conversion stays a plain curve change."),
            setting("Apply Inverse OOTF", "Off", "Same reason."),
        ], notes=[lut_alt + " The LUT replaces both input nodes."] if lut_alt else [])
        b = node(str(int(first) + 1), "IN B", "Input conversion, step 2: into the working space", CST, CST_HOW, [
            setting("Input Color Space", gamut_name, "Output of the node before."),
            setting("Input Gamma", "Linear", "Output of the node before."),
            setting("Output Color Space", "DaVinci Wide Gamut", "The working space of the grade."),
            setting("Output Gamma", "DaVinci Intermediate", "The working curve of the grade."),
        ] + _cst_tail())
        return nodes + [a, b], notes
    n = node(first, "IN", "Input conversion", CST, CST_HOW, [
        _data_levels(clip),
        setting("Input Color Space", gamut_name, f"{clip['camera']} records {clip['profile']} in {gamut['name']}."),
        setting("Input Gamma", gamma_name, f"The log curve of the recording profile, {curve['name']}."),
        setting("Output Color Space", "DaVinci Wide Gamut", "The working space of the grade."),
        setting("Output Gamma", "DaVinci Intermediate", "The working curve of the grade. Every control after this node expects it."),
    ] + _cst_tail(), notes=[lut_alt] if lut_alt else [])
    if curve["id"] == "hlg":
        n["notes"].append("HLG: Resolve and the app scale scene light slightly differently. For an exact match use the LUT route.")
    return nodes + [n], notes


def _hue_rows(points, kind: str) -> list[list[str]]:
    """Rows for the points that change something, plus their neighbors as anchors."""
    neutral = 0.0 if kind == "rotate" else 1.0
    pts = sorted(points, key=lambda p: p[0] % 360.0)
    n = len(pts)
    active = [abs(v - neutral) > 1e-9 for _, v in pts]
    rows = []
    for i, (h, v) in enumerate(pts):
        if not (active[i] or active[(i - 1) % n] or active[(i + 1) % n]):
            continue
        role = "change" if active[i] else "anchor, holds the curve flat"
        if kind == "rotate":
            rows.append([base.hue_label(h), f"{h:.1f}", num(v, 1, signed=True), base.hue_direction(h, v) if active[i] else role])
        else:
            rows.append([base.hue_label(h), f"{h:.1f}", num(v, 3), role])
    return rows


HUE_HOW = ("Curves palette. Click the color dots under the curve to add points on red, yellow, green, cyan, blue and magenta. "
           "Click on the curve for points between them. Select a point and type its value in the field below the curve. "
           "Place points by the color under the curve: Resolve numbers Input Hue on its own scale.")


def _hue_curves(hue_vs_sat, hue_vs_hue, reason_sat: str, reason_hue: str) -> list[dict[str, Any]]:
    out = []
    if not base.is_neutral_curve(hue_vs_sat, 1.0):
        out.append(curve_table("Hue vs Sat", ["Color", "Hue angle", "Saturation", "Role"], _hue_rows(hue_vs_sat, "sat"), reason_sat, HUE_HOW))
    if not base.is_neutral_curve(hue_vs_hue, 0.0):
        out.append(curve_table("Hue vs Hue", ["Color", "Hue angle", "Hue Rotate", "Moves"], _hue_rows(hue_vs_hue, "rotate"), reason_hue, HUE_HOW))
    return out


def _clip_group(clip: dict[str, Any], shot: dict[str, Any], ctx: dict[str, Any], many: bool, repeat: bool = False) -> dict[str, Any]:
    nodes, notes = _input_nodes(clip, ctx)
    if repeat:  # later shots of one clip: the same input conversion, shown short
        nodes = [node(n["number"], n["label"], n["title"], n["tool"], "", [], notes=["Same as shot 1 of this clip. Copy the node."]) for n in nodes]
        notes = []
    k = len(nodes)
    g, why = shot["grade"], shot["reasons"]
    gains = clip["camera_gains"]
    if any(abs(v - 1.0) > 5e-4 for v in gains):
        k += 1
        nodes.append(node(str(k), "CAM", "Camera default correction", "Primaries, node Gamma set to Linear",
                          "Right-click the node, Gamma, Linear. " + BARS_HOW,
                          [setting("Gain", rgb(gains, 3), f"Default correction stored with the {clip['camera']} profile. Applies to every clip of this profile."),
                           setting(*LUM_MIX)]))
    bal = g["balance"]
    has_gain = any(abs(v - 1.0) > 5e-4 for v in bal["gain"])
    has_off = any(abs(v - 25.0) > 5e-3 for v in bal["offset"])
    if has_gain or has_off:
        k += 1
        items = []
        if has_gain:
            items.append(setting("Gain", rgb(bal["gain"], 3), why.get("gain", "Exposure and white balance of this shot.")))
        if has_off:
            items.append(setting("Offset", rgb(bal["offset"], 2), why.get("offset", "Black point of this shot.")))
        items.append(setting(*LUM_MIX))
        nodes.append(node(str(k), "BAL", "Balance: exposure, white balance, black point", "Primaries, node Gamma set to Linear",
                          "Right-click the node, Gamma, Linear. The node then works on linear light, like light itself. " + BARS_HOW,
                          items))
    else:
        notes.append("Balance: no change needed. " + why.get("gain", ""))
    m = g["match"]
    sat = m["saturation"]
    cam_sat = clip["camera_saturation"]
    items = []
    if abs(cam_sat - 50.0) > 0.05:
        k += 1
        nodes.append(node(str(k), "CAM SAT", "Camera default saturation", "Primaries", ADJ_HOW,
                          [setting("Sat", num(cam_sat, 1), f"Default saturation stored with the {clip['camera']} profile.")]))
    if abs(m["contrast"] - 1.0) > 5e-4:
        items.append(setting("Contrast", num(m["contrast"], 3), why.get("match", "Matches this shot to the lead shot before the look.")))
        items.append(setting("Pivot", num(m["pivot"], 3), "Middle gray in DaVinci Intermediate. Contrast turns around it, so exposure stays."))
    if abs(sat - 50.0) > 0.05:
        items.append(setting("Sat", num(sat, 1), why.get("match", "Matches this shot to the lead shot before the look.")))
    if items:
        k += 1
        nodes.append(node(str(k), "MATCH", "Shot match", "Primaries", ADJ_HOW, items))
    curves = _hue_curves(m.get("hue_vs_sat"), m.get("hue_vs_hue"),
                         why.get("skin", "Skin saturation trim for this shot."), why.get("skin", "Skin hue trim for this shot."))
    if curves:
        k += 1
        nodes.append(node(str(k), "SKIN TRIM", "Skin trim for this shot", "Curves", "", curves=curves))
    title = clip["name"]
    if many:
        title += f", shot {shot['index'] + 1}  ({base.timecode(shot['start'], clip['fps'])} to {base.timecode(shot['end'], clip['fps'])})"
    sub = f"{clip['camera']}, {clip['profile']}"
    if shot.get("scene"):
        sub += f". Scene {shot['scene']}" + (", lead shot" if shot.get("is_lead") else "")
    return {"title": title, "subtitle": sub, "notes": notes, "nodes": nodes}


def _timeline_nodes(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    look, why = ctx["look"], ctx["look"].get("reasons") or {}
    nodes: list[dict[str, Any]] = []

    def add(label: str, title: str, tool: str, how: str = "", settings=None, curves=None, notes=None):
        nodes.append(node(f"T{len(nodes) + 1}", label, title, tool, how, settings, curves, notes))

    detail = (ctx["finish"].get("detail") or {})
    if detail.get("denoise", 0) > 0:
        level = float(detail["denoise"])
        if ctx["edition"] == "studio":
            add("NR", "Noise reduction", "Motion Effects",
                "Motion Effects palette. Zoom the viewer to 100% on a flat, dark area and raise the values until the noise calms down.",
                [setting("Temporal NR Frames", "2", "The frames before and after share the same detail and different noise."),
                 setting("Temporal Threshold, Luma and Chroma", f"start at {5 + level * 0.25:.0f}", "Set by eye. Too high smears movement."),
                 setting("Spatial NR Mode", "Better", "Keeps edges and texture."),
                 setting("Spatial Threshold, Luma", f"start at {2 + level * 0.12:.0f}", "Set by eye. Skin must keep its pores."),
                 setting("Spatial Threshold, Chroma", f"start at {5 + level * 0.25:.0f}", "Color noise takes more reduction than luma noise.")],
                notes=[f"The app cleans the camera signal at {level:.0f}% before the grade. Set by eye against Play graded in the app."])
        else:
            add("NR", "Noise reduction", "Not in the free edition", "",
                [setting("Noise Reduction", "Needs Resolve Studio",
                         f"The app cleans camera noise at {level:.0f}%. Two ways around it: export the graded clip from this app, or leave the noise in.")])
    if abs(look["contrast"] - 1.0) > 5e-4:
        add("CONTRAST", "Look contrast", "Primaries", ADJ_HOW, [
            setting("Contrast", num(look["contrast"], 3), why.get("contrast", "Contrast of the look.")),
            setting("Pivot", num(look["pivot"], 3), "Middle gray in DaVinci Intermediate. Your exposure stays where the balance put it."),
        ])
    tone = []
    if any(abs(v) > 5e-4 for v in look["lift"]):
        tone.append(setting("Lift", rgb(look["lift"], 3, signed=True), why.get("lift", "Shadow level and shadow color of the look.")))
    if any(abs(v) > 5e-4 for v in look["gamma"]):
        tone.append(setting("Gamma", rgb(look["gamma"], 3, signed=True), why.get("gamma", "Midtone color of the look.")))
    if any(abs(v - 1.0) > 5e-4 for v in look["gain"]):
        tone.append(setting("Gain", rgb(look["gain"], 3), why.get("gain", "Highlight color of the look.")))
    if tone:
        add("TONE COLOR", "Shadow, midtone and highlight color", "Primaries", BARS_HOW, tone + [setting(*LUM_MIX)])
    sat_items, sat_curves = [], []
    if abs(look["saturation"] - 50.0) > 0.05:
        sat_items.append(setting("Sat", num(look["saturation"], 1), why.get("saturation", "Overall saturation of the look.")))
    if not base.is_neutral_curve(look.get("lum_vs_sat"), 1.0):
        rows = [[num(x, 3), num(v, 3)] for x, v in look["lum_vs_sat"]]
        sat_curves.append(curve_table("Lum vs Sat", ["Input Lum", "Saturation"], rows,
                                      why.get("lum_vs_sat", "Less color in deep shadows and bright highlights."),
                                      "Curves palette, Lum vs Sat. Click the curve to add a point, then type both values in the fields below."))
    if sat_items or sat_curves:
        add("SATURATION", "Saturation and its roll-off", "Primaries and Curves", ADJ_HOW if sat_items else "", sat_items, sat_curves)
    pal = _hue_curves(look.get("hue_vs_sat"), look.get("hue_vs_hue"),
                      why.get("hue_vs_sat", "Color families trimmed toward the reference palette."),
                      why.get("hue_vs_hue", "Color families nudged toward the hues of the reference."))
    if pal:
        add("PALETTE", "Palette: color families", "Curves", "", curves=pal)
    skin = look.get("skin") or {}
    sk = _hue_curves(skin.get("hue_vs_sat"), skin.get("hue_vs_hue"), why.get("skin", "Keeps skin healthy under the look."),
                     why.get("skin", "Returns skin to the skin tone line under the look."))
    if sk:
        add("SKIN", "Skin protection", "Curves", "", curves=sk,
            notes=["Check on the vectorscope with the skin tone line switched on. Skin sits on the line."])
    if not base.is_neutral_curve(look.get("sat_vs_sat"), 1.0):
        rows = [[num(x, 3), num(v, 3)] for x, v in look["sat_vs_sat"]]
        add("LIMIT", "Saturation limit", "Curves", "", curves=[curve_table(
            "Sat vs Sat", ["Input Sat", "Output Sat"], rows,
            why.get("sat_vs_sat", "Safety limit: bends extreme colors under the display gamut. Normal colors pass untouched."),
            "Curves palette, Sat vs Sat. Click the curve to add a point, then type both values in the fields below.")])
    fin = ctx["finish"]
    vig, grain = fin.get("vignette") or {}, fin.get("grain") or {}
    if vig.get("enabled") and vig.get("amount", 0) > 0:
        off = 25.0 - ctx["stop_points"] * float(vig["amount"])
        inner = round(1.4142 * float(vig.get("start", 0.45)) * 100)
        add("VIGNETTE", "Vignette", "Window and Primaries",
            "Window palette, circle. Center it. Stretch it to the frame shape. Press Invert so the node works outside the circle.",
            [setting("Window shape", f"Ellipse. Inner edge of the soft zone at {inner}% of the frame half width and half height. "
                                     "Outer edge through the frame corners.",
                     "The darkening starts well outside the subject and reaches full strength in the corners. Set by eye against the app preview."),
             setting("Softness", "High. The edge must stay invisible.", "A vignette nobody notices keeps the eye in the frame."),
             setting("Offset", rgb([off, off, off], 2), f"{float(vig['amount']):.2f} stops darker in the corners. One stop is 10 Offset points in DaVinci Intermediate.")])
    if grain.get("enabled") and grain.get("amount", 0) > 0:
        if ctx["edition"] == "studio":
            add("GRAIN", "Fine grain", "Film Grain (ResolveFX Texture)", "Effects panel, ResolveFX Texture, Film Grain. Drag it onto the node.",
                [setting("Preset", "35mm, the finest stock in the list", "Fine grain hides banding in skies and walls."),
                 setting("Grain Strength", "Low. Lower it until the grain is hard to see at 100% zoom.",
                         f"The app adds grain at {float(grain['amount']):.2f}% of the signal. Set by eye against the app preview."),
                 setting("Saturation", "0", "Monochrome grain, like film.")])
        else:
            add("GRAIN", "Fine grain", "Not in the free edition", "",
                [setting("Film Grain", "Needs Resolve Studio",
                         "Two ways around it: export the graded clip from this app with grain on, or leave grain out.")])
    out = ctx["output"]
    add("OUT A", "Output conversion, step 1: gamut", CST, CST_HOW, [
        setting("Input Color Space", "DaVinci Wide Gamut", "The working space of the grade."),
        setting("Input Gamma", "DaVinci Intermediate", "The working curve of the grade."),
        setting("Output Color Space", "Rec.709", "The display gamut."),
        setting("Output Gamma", "DaVinci Intermediate", "The curve stays here. Tone mapping then runs on Rec.709 primaries in the next node."),
    ] + _cst_tail())
    add("OUT B", "Output conversion, step 2: tone mapping and display curve", CST, CST_HOW, [
        setting("Input Color Space", "Rec.709", "Output of the node before."),
        setting("Input Gamma", "DaVinci Intermediate", "Output of the node before."),
        setting("Output Color Space", "Rec.709", "The display gamut."),
        setting("Output Gamma", out["resolve_gamma"], f"The display curve of {out['name']}."),
        setting("Tone Mapping Method", "DaVinci", "Rolls bright highlights off smoothly. A real conversion, never a contrast curve."),
        setting("Custom Max Input", f"On, {out['max_input_nits']} nits", "The highlight range of the working space. Highlights roll off with no clipping."),
        setting("Custom Max Output", "On, 100 nits", "Peak white of an SDR display."),
        setting("Adaptation", "9.00", "Resolve's default. Middle gray stays where the grade put it."),
        setting("Gamut Mapping Method", "None", "The look holds the saturation limits."),
        setting("Apply Forward OOTF", "Off", "The display curve alone finishes the picture."),
        setting("Apply Inverse OOTF", "Off", "Same reason."),
    ], notes=[f"LUT route: {ctx['lut_names']['output']} on one node replaces OUT A and OUT B."])
    sharp, clar = float(detail.get("sharpen", 0)), float(detail.get("clarity", 0))
    if sharp > 0 or clar > 0:
        items = []
        if clar > 0:
            items.append(setting("Midtone Detail (MD)", f"start at {clar * 0.6:.0f}",
                                 f"Wide local contrast: shape for faces and buildings. The app runs clarity at {clar:.0f}%. Set by eye against the app preview."))
        if sharp > 0:
            items.append(setting("Blur palette, Radius", f"start at {0.50 - sharp * 0.0006:.2f}",
                                 f"Below 0.50 the Blur palette sharpens. The app runs sharpening at {sharp:.0f}%. Set by eye at 100% zoom."))
        add("DETAIL", "Clarity and sharpening", "Primaries and Blur",
            "Midtone Detail sits on page 2 of the Primaries palette. The Blur palette holds Radius: link the three channels.",
            items, notes=["Last node: sharpening works on the finished picture."])
    return nodes


def build(ctx: dict[str, Any]) -> dict[str, Any]:
    out = ctx["output"]
    edition = "Studio" if ctx["edition"] == "studio" else "free edition"
    title = f"DaVinci Resolve recipe: {ctx['look_name']} at {ctx['strength']:g}%"
    intro = [
        f"Target: DaVinci Resolve, {edition}. Output: {out['name']}.",
        f"Look: {ctx['look_name']}" + (f" ({ctx['look_description']})" if ctx["look_description"] else "") + f". Values hold the look at {ctx['strength']:g}% strength.",
        "Build the nodes in the order shown, one serial node per entry. Fields show two decimals. Type every digit shown here.",
        "Change the strength in the app, then open the recipe again. The numbers follow.",
    ]
    if not ctx["graded"]:
        intro.append("No grade built yet. Press Build grade first. The values below hold the neutral state.")
    sections: list[dict[str, Any]] = []
    sections.append({
        "id": "project", "title": "1. Project settings", "where": "File, Project Settings",
        "notes": [], "items": [
            setting("Color Management, Color science", "DaVinci YRGB", "Every conversion happens in a node you see. Nothing runs hidden."),
            setting("Color Management, Timeline color space", "DaVinci WG/Intermediate", "The controls then respond the way the app expects."),
            setting("Color Management, Output color space", out["name"], "Tags the viewer and the render to match the output conversion."),
            setting("General Options, Color, Use S-curve for contrast", "Off", "Contrast then follows a straight line around the pivot, the math of the app."),
            setting("General Options, Color, Luminance mixer defaults to zero", "On", "New nodes start with Lum Mix at zero, so each channel moves alone."),
        ]})
    groups = []
    for clip in ctx["clips"]:
        if clip.get("problem"):
            groups.append({"title": clip["name"], "subtitle": "", "notes": [f"Skipped: {clip['problem']}"], "nodes": []})
            continue
        if not clip["shots"]:
            groups.append({"title": clip["name"], "subtitle": f"{clip['camera']}, {clip['profile']}", "notes": ["Analyze this clip first."], "nodes": []})
            continue
        many = len(clip["shots"]) > 1
        for i, shot in enumerate(clip["shots"]):
            grp = _clip_group(clip, shot, ctx, many, repeat=i > 0)
            if many and i == 0:
                cuts = ", ".join(base.timecode(s["start"], clip["fps"]) for s in clip["shots"][1:])
                grp["notes"].insert(0, f"This clip holds {len(clip['shots'])} shots. Cut it at {cuts} (clip time, minutes:seconds:frames) "
                                       "so each shot takes its own nodes. The input conversion node stays the same for all of them.")
            groups.append(grp)
    sections.append({
        "id": "clips", "title": "2. Clip nodes: input conversion, balance, match",
        "where": "Color page, node graph in Clip mode. One set of serial nodes per shot.",
        "notes": ["Shots with the same camera profile share the input conversion. Copy the first node, then set the rest per shot."],
        "groups": groups})
    sections.append({
        "id": "timeline", "title": "3. Timeline nodes: the shared look, finish, output conversion",
        "where": "Color page, node graph switched from Clip to Timeline (menu at the top right of the node graph). These nodes apply to every clip.",
        "notes": ([] if not ctx["neutral_look"] else ["The look is neutral. Only the output conversion follows."]),
        "nodes": _timeline_nodes(ctx)})
    names = ctx["lut_names"]
    lut_items = [
        setting(names["look"], "Timeline node, in place of the look nodes", "Holds contrast, color, saturation and skin of the look. Working space in, working space out."),
        setting(names["look_output"], "Last timeline node, alone", f"Holds the look and the output conversion to {out['name']}."),
        setting(names["output"], "Last timeline node", "Holds the output conversion alone. Replaces OUT A and OUT B."),
    ]
    for p in names["profiles"].values():
        lut_items.append(setting(p["input"], f"First clip node of {p['camera_name']} {p['profile_name']} clips", "Replaces the input conversion nodes."))
        lut_items.append(setting(p["camera"], f"Any editor, on {p['camera_name']} {p['profile_name']} clips as recorded",
                                 "Input conversion, look and output in one file. No per-shot balance inside."))
    sections.append({
        "id": "luts", "title": "4. LUT route",
        "where": "Export tab, Export LUT set. Copy the files into Resolve's LUT folder (Project Settings, Color Management, Open LUT Folder), then press Update Lists.",
        "notes": ["Each LUT replaces a group of nodes. The balance nodes stay: a LUT holds no per-shot values.",
                  f"LUT size: {ctx['settings']['lut_size']} points per side. Vignette, grain, noise reduction and sharpening stay outside the LUTs."],
        "items": lut_items})
    sections.append({
        "id": "check", "title": "5. Check",
        "where": "Color page, scopes",
        "notes": [
            "Compare the Resolve viewer with the app preview on the same frame. Waveform and vectorscope shapes match.",
            "Skin sits on the skin tone line of the vectorscope.",
            "Deliver page: Rec.709 with the gamma tag of your output color space. The file then looks the same in QuickTime, on a phone and on YouTube.",
        ], "items": []})
    return {"editor": ID, "editor_name": NAME, "edition": ctx["edition"], "title": title, "intro": intro, "sections": sections}
