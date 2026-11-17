"""MCP server: an AI assistant grades live in the running app.

Start the app first (start.bat or start.sh), then register this server with an MCP client such as
Claude Desktop or Claude Code:

    command:  <app folder>/.venv/Scripts/python.exe   (Windows)   or   <app folder>/.venv/bin/python
    args:     <app folder>/colorgrader/mcp.py

`python -m colorgrader.mcp --config` prints these lines with this computer's paths.

The server speaks the Model Context Protocol over stdin and stdout (JSON-RPC, one message per line) and
talks to the app over its local HTTP API, so every change shows up in the app's viewer while you watch.
The assistant sees the picture (JPEG frames) and the numbers (levels, clipping, skin position on the
vectorscope, saturation, quality rules), changes the grade, looks again, and can undo.

COLORGRADER_URL overrides the app address (default http://127.0.0.1:8765). No new packages: the
protocol is small enough to speak with the standard library.
"""
from __future__ import annotations

import base64
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

PROTOCOL = "2025-06-18"
SUPPORTED = ("2025-06-18", "2025-03-26", "2024-11-05")
BASE = os.environ.get("COLORGRADER_URL", "http://127.0.0.1:8765").rstrip("/")


class ToolError(Exception):
    """A failure the assistant should read and act on."""


# --------------------------------------------------------------------- http
def _request(method: str, path: str, body: Any = None, params: dict[str, Any] | None = None, raw: bool = False, timeout: float = 120.0):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            payload = res.read()
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode()).get("detail")
        except Exception:
            detail = exc.reason
        raise ToolError(f"The app refused: {detail}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise ToolError(f"Color Grader is not running at {BASE} ({exc}). Start it with start.bat or start.sh, then try again.")
    return payload if raw else json.loads(payload.decode() or "null")


def _get(path, **params):
    return _request("GET", path, params=params)


def _put(path, body):
    return _request("PUT", path, body)


def _post(path, body=None):
    return _request("POST", path, body if body is not None else {})


def _wait(job_id: str, timeout: float = 3600.0) -> dict[str, Any]:
    end = time.time() + timeout
    while time.time() < end:
        job = _get(f"/api/jobs/{job_id}")
        if job["state"] == "done":
            return job
        if job["state"] == "error":
            raise ToolError(job.get("error") or "The job failed.")
        time.sleep(0.5)
    raise ToolError("The job is still running. Check project_status in a moment.")


def _state() -> dict[str, Any]:
    return _get("/api/state")["project"]


def _shot(proj: dict[str, Any], shot_id: str) -> dict[str, Any]:
    for s in proj["shots"]:
        if s["id"] == shot_id:
            return s
    ids = ", ".join(s["id"] for s in proj["shots"][:12])
    raise ToolError(f"No shot {shot_id!r}. Shots: {ids}")


def _mid(shot: dict[str, Any]) -> float:
    return shot["times"][len(shot["times"]) // 2] if shot.get("times") else 0.5 * (shot["start"] + shot["end"])


def _image(clip_id: str, t: float, stage: str, shot_id: str, width: int) -> dict[str, Any]:
    jpg = _request("GET", "/api/frame", params={"clip": clip_id, "t": round(t, 3), "stage": stage, "w": width, "shot": shot_id}, raw=True)
    return {"type": "image", "data": base64.b64encode(jpg).decode(), "mimeType": "image/jpeg"}


def _text(obj: Any) -> dict[str, Any]:
    return {"type": "text", "text": obj if isinstance(obj, str) else json.dumps(obj, indent=1)}


# ------------------------------------------------------------------ history
_history: list[dict[str, Any]] = []


def _remember() -> None:
    """Keep the grade as it is now, so undo can put it back."""
    p = _state()
    _history.append({"look": p["look"], "finish": p["finish"], "strength": p["strength"],
                     "shots": {s["id"]: s["grade"] for s in p["shots"] if s.get("grade")}})
    del _history[:-30]


def _rules_summary(p: dict[str, Any]) -> dict[str, Any]:
    rep = p.get("report") or {}
    failing = [{"shot": s["id"], "rule": r["name"], "detail": r["detail"]}
               for s in p["shots"] for r in (s.get("rules") or []) if r["status"] == "fail"]
    return {"totals": rep.get("totals"), "failing": failing[:12]}


# -------------------------------------------------------------------- tools
def project_status(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    clips = [{"id": c["id"], "name": c["name"], "camera": c.get("camera_id"), "profile": c.get("profile_id"),
              "duration_s": round(c["info"]["duration"], 2), "size": f"{c['info']['width']}x{c['info']['height']}",
              "bit_depth": c["info"]["bit_depth"]} for c in p["clips"]]
    shots = [{"id": s["id"], "clip": s["clip_id"], "shot": s["index"] + 1, "from_s": s["start"], "to_s": s["end"],
              "faces": bool(s.get("has_faces")), "scene": s.get("scene"),
              "rules": (s.get("report") or {})} for s in p["shots"]]
    look = p["look"]
    out = {"clips": clips, "shots": shots, "graded": p.get("graded"),
           "look": {"name": look.get("name"), "source": look.get("source")}, "strength_percent": p.get("strength"),
           "finish": p.get("finish"), "reference": (p.get("reference") or {}).get("name"),
           "suggestions": [{"id": s["id"], "name": s["name"]} for s in p.get("suggestions") or []],
           "notes": (p.get("report") or {}).get("notes", []), "quality": _rules_summary(p)}
    return [_text(out)]


def view_shot(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    s = _shot(p, args["shot_id"])
    t = float(args["time"]) if args.get("time") is not None else _mid(s)
    t = min(max(t, s["start"]), max(s["start"], s["end"] - 0.02))
    width = int(args.get("width", 960))
    stage = args.get("stage", "both")
    _put("/api/focus", {"shot_id": s["id"], "t": t})        # the app's viewer jumps to this shot
    content = []
    if stage in ("both", "source"):
        content.append(_text(f"Source, shot {s['id']} at {t:.2f} s (camera signal as recorded):"))
        content.append(_image(s["clip_id"], t, "source", s["id"], width))
    if stage in ("both", "graded", "balanced", "converted"):
        st = "graded" if stage == "both" else stage
        content.append(_text(f"{st.capitalize()}, shot {s['id']} at {t:.2f} s:"))
        content.append(_image(s["clip_id"], t, st, s["id"], width))
    content.append(_text(_get("/api/readout", shot=s["id"], t=round(t, 3), stage="graded" if stage in ("both", "graded") else stage)))
    return content


def read_shot(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    s = _shot(p, args["shot_id"])
    return [_text(_get("/api/readout", shot=s["id"], t=args.get("time"), stage=args.get("stage", "graded")))]


def get_grade(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    look = {k: p["look"].get(k) for k in ("name", "exposure", "contrast", "pivot", "lift", "gamma", "gain", "saturation", "hsv_sat",
                                          "tone_curve", "lum_vs_sat", "hue_vs_hue", "hue_vs_sat", "sat_vs_sat")}
    out = {"look": look, "strength_percent": p["strength"], "finish": p["finish"]}
    if args.get("shot_id"):
        s = _shot(p, args["shot_id"])
        out["shot"] = {"id": s["id"], "balance": s["grade"]["balance"], "match": {k: s["grade"]["match"].get(k) for k in ("contrast", "pivot", "saturation")},
                       "reasons": s["grade"].get("reasons", {})}
    out["units"] = ("Look: exposure in stops on a linear node. Contrast around pivot 0.336 (middle gray). Lift and Gamma are offsets, "
                    "Gain a multiplier, per channel R G B (Resolve wheels). Saturation 50 = unchanged. hsv_sat gain 1 and gamma 0 = unchanged. "
                    "hue_vs_hue: [hue degrees, rotation degrees], red 0, yellow 60, green 120, cyan 180, blue 240, magenta 300. "
                    "Shot balance: gain per channel on linear light, offset 25 = unchanged.")
    return [_text(out)]


LOOK_KEYS = {"exposure", "contrast", "pivot", "lift", "gamma", "gain", "saturation", "tone_curve", "hue_vs_hue", "hue_vs_sat", "lum_vs_sat", "sat_vs_sat"}


def adjust_look(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    look = copy.deepcopy(p["look"])
    changed = []
    for k, v in args.items():
        if k in LOOK_KEYS and v is not None:
            if k in ("lift", "gamma", "gain") and (not isinstance(v, list) or len(v) != 3):
                raise ToolError(f"{k} takes three numbers: R, G, B.")
            look[k] = v
            changed.append(k)
    if args.get("hsv_gain") is not None or args.get("hsv_gamma") is not None:
        hs = dict(look.get("hsv_sat") or {"gain": 1.0, "gamma": 0.0})
        if args.get("hsv_gain") is not None:
            hs["gain"] = float(args["hsv_gain"])
        if args.get("hsv_gamma") is not None:
            hs["gamma"] = float(args["hsv_gamma"])
        look["hsv_sat"] = hs
        changed.append("hsv_sat")
    if not changed:
        raise ToolError("Nothing to change. Pass at least one look control.")
    _remember()
    look["source"] = look.get("source") or "edited"
    look.setdefault("reasons", {})
    for k in changed:
        look["reasons"][k] = args.get("reason") or "Set by the assistant."
    state = _put("/api/grade/look", {"look": look})
    return [_text({"changed": changed, "quality": _rules_summary(state),
                   "next": "view_shot shows the result. undo puts the previous grade back."})]


def adjust_shot(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    s = _shot(p, args["shot_id"])
    g = copy.deepcopy(s["grade"])
    gain = [float(v) for v in g["balance"]["gain"]]
    ex = float(args.get("exposure_stops") or 0.0)
    warm = float(args.get("warmth_stops") or 0.0)
    tint = float(args.get("tint_stops") or 0.0)
    f = [2.0 ** (ex + warm / 2), 2.0 ** (ex - tint), 2.0 ** (ex - warm / 2)]   # +tint is magenta: less green
    g["balance"]["gain"] = [round(a * b, 4) for a, b in zip(gain, f)]
    if args.get("black_offset") is not None:
        g["balance"]["offset"] = [round(v + float(args["black_offset"]), 2) for v in g["balance"]["offset"]]
    if args.get("contrast") is not None:
        g["match"]["contrast"] = float(args["contrast"])
    if args.get("saturation") is not None:
        g["match"]["saturation"] = float(args["saturation"])
    _remember()
    state = _put(f"/api/shots/{s['id']}/grade", {"grade": g})
    return [_text({"shot": s["id"], "balance": g["balance"], "match": {k: g["match"].get(k) for k in ("contrast", "pivot", "saturation")},
                   "quality": _rules_summary(state)})]


def set_finish(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    fin = copy.deepcopy(p["finish"])
    mapping = {"relight_on": ("relight", "enabled"), "relight_stops": ("relight", "amount"), "subject_stops": ("relight", "subject"),
               "vignette_on": ("vignette", "enabled"), "vignette_stops": ("vignette", "amount"),
               "grain_on": ("grain", "enabled"), "grain_amount": ("grain", "amount"),
               "denoise": ("detail", "denoise"), "sharpen": ("detail", "sharpen"), "clarity": ("detail", "clarity")}
    changed = []
    for key, (group, field) in mapping.items():
        if args.get(key) is not None:
            fin.setdefault(group, {})[field] = args[key]
            changed.append(key)
    if not changed:
        raise ToolError("Nothing to change.")
    _remember()
    state = _put("/api/grade/finish", {"finish": fin})
    return [_text({"changed": changed, "finish": state["finish"], "quality": _rules_summary(state)})]


def set_strength(args: dict[str, Any]) -> list[dict[str, Any]]:
    _remember()
    state = _put("/api/grade/strength", {"strength": float(args["percent"])})
    return [_text({"strength_percent": state["strength"], "quality": _rules_summary(state)})]


def design_looks(args: dict[str, Any]) -> list[dict[str, Any]]:
    _wait(_post("/api/suggestions", {"shot_id": args.get("shot_id")})["job"])
    p = _state()
    content = [_text({"scene_read": (p.get("scene_read") or {}).get("lines"),
                      "looks": [{"id": s["id"], "name": s["name"], "mood": s["mood"], "why": s["why"], "steps": s.get("steps")}
                                for s in p.get("suggestions") or []]})]
    for s in p.get("suggestions") or []:
        jpg = _request("GET", f"/api/suggestions/{s['id']}/thumb.jpg", raw=True)
        content += [_text(f"Preview: {s['name']}"), {"type": "image", "data": base64.b64encode(jpg).decode(), "mimeType": "image/jpeg"}]
    return content


def apply_look(args: dict[str, Any]) -> list[dict[str, Any]]:
    _remember()
    job = _wait(_post(f"/api/suggestions/{args['look_id']}/apply")["job"])
    p = _state()
    return [_text({"applied": p["look"].get("name"), "result": job.get("result"), "quality": _rules_summary(p)})]


def rebuild_grade(args: dict[str, Any]) -> list[dict[str, Any]]:
    _remember()
    _wait(_post("/api/grade/build", {})["job"])
    p = _state()
    return [_text({"look": p["look"].get("name"), "notes": (p.get("report") or {}).get("notes"),
                   "fixes": (p.get("report") or {}).get("fixes"), "quality": _rules_summary(p)})]


def undo(args: dict[str, Any]) -> list[dict[str, Any]]:
    if not _history:
        raise ToolError("Nothing to undo in this session.")
    h = _history.pop()
    _put("/api/grade/look", {"look": h["look"]})
    _put("/api/grade/finish", {"finish": h["finish"]})
    for sid, grade in h["shots"].items():
        _put(f"/api/shots/{sid}/grade", {"grade": grade})
    state = _put("/api/grade/strength", {"strength": h["strength"]})
    return [_text({"restored": state["look"].get("name"), "steps_left": len(_history), "quality": _rules_summary(state)})]


def save_look(args: dict[str, Any]) -> list[dict[str, Any]]:
    return [_text(_post("/api/looks", {"name": args["name"]}))]


def export_video(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    ids = args.get("clip_ids") or [c["id"] for c in p["clips"] if not c["info"]["is_image"]]
    job = _wait(_post("/api/export/video", {"clip_ids": ids, "format": args.get("format") or "auto"})["job"])
    return [_text(job.get("result"))]


RGB = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
POINTS = {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}

TOOLS: list[tuple[str, str, dict[str, Any], Callable[[dict[str, Any]], list[dict[str, Any]]]]] = [
    ("project_status", "Clips, shots (ids, time ranges, faces, scene), the current look, strength, finish and quality rules. Start here.",
     {"type": "object", "properties": {}}, project_status),
    ("view_shot", "See a shot: the source and the graded picture as images, plus its readout (levels, clipping, skin on the vectorscope, "
                  "saturation, failing rules). The app's viewer jumps to the same shot.",
     {"type": "object", "properties": {"shot_id": {"type": "string"},
                                       "stage": {"type": "string", "enum": ["both", "graded", "source", "balanced", "converted"], "default": "both"},
                                       "time": {"type": "number", "description": "Seconds in the clip. Default: the middle of the shot."},
                                       "width": {"type": "integer", "default": 960, "minimum": 320, "maximum": 1920}},
      "required": ["shot_id"]}, view_shot),
    ("read_shot", "Numbers only, no image: levels in display percent, clipping, saturation, skin position and failing rules for one shot.",
     {"type": "object", "properties": {"shot_id": {"type": "string"}, "time": {"type": "number"},
                                       "stage": {"type": "string", "enum": ["graded", "balanced", "converted", "source"], "default": "graded"}},
      "required": ["shot_id"]}, read_shot),
    ("get_grade", "The current look values (shared by all shots), strength, finish, and one shot's own balance if shot_id is given, with units.",
     {"type": "object", "properties": {"shot_id": {"type": "string"}}}, get_grade),
    ("adjust_look", "Change the shared look on every shot. Pass only the controls to change, as absolute values (read get_grade first). "
                    "Small moves: wheels 0.005 to 0.03, contrast 0.05, exposure 0.1 to 0.3 stops.",
     {"type": "object", "properties": {
         "exposure": {"type": "number", "description": "Stops on a linear node, 0 = unchanged."},
         "contrast": {"type": "number", "description": "1.0 = unchanged."}, "pivot": {"type": "number", "description": "0.336 = middle gray."},
         "lift": {**RGB, "description": "R G B offsets for shadows, 0 = unchanged."},
         "gamma": {**RGB, "description": "R G B offsets for midtones, 0 = unchanged."},
         "gain": {**RGB, "description": "R G B multipliers for highlights, 1 = unchanged."},
         "saturation": {"type": "number", "description": "Resolve Sat, 50 = unchanged."},
         "hsv_gain": {"type": "number", "description": "HSV node saturation Gain: lifts strong colors. 1 = unchanged."},
         "hsv_gamma": {"type": "number", "description": "HSV node saturation Gamma: lifts gentle colors such as skin. 0 = unchanged."},
         "tone_curve": {**POINTS, "description": "Custom curve points [input, output] in DaVinci Intermediate, all channels."},
         "hue_vs_hue": {**POINTS, "description": "[hue degrees, rotation degrees]."},
         "hue_vs_sat": {**POINTS, "description": "[hue degrees, saturation multiplier]."},
         "lum_vs_sat": {**POINTS, "description": "[luma 0..1, saturation multiplier]."},
         "reason": {"type": "string", "description": "One line on why: it shows in the app and the recipe."}}}, adjust_look),
    ("adjust_shot", "Correct one shot on its own: exposure and white balance on its linear node, black point, contrast and saturation. "
                    "Relative for exposure, warmth and tint; absolute for contrast and saturation.",
     {"type": "object", "properties": {
         "shot_id": {"type": "string"},
         "exposure_stops": {"type": "number", "description": "Brighter or darker, in stops, relative."},
         "warmth_stops": {"type": "number", "description": "Positive warmer (more red, less blue), relative."},
         "tint_stops": {"type": "number", "description": "Positive toward magenta (less green), relative."},
         "black_offset": {"type": "number", "description": "Offset points on linear light, relative. Negative deepens hazy blacks."},
         "contrast": {"type": "number", "description": "This shot's own contrast, absolute, 1 = unchanged."},
         "saturation": {"type": "number", "description": "This shot's own Sat, absolute, 50 = unchanged."}},
      "required": ["shot_id"]}, adjust_shot),
    ("set_finish", "Relight (background darker around tracked faces), vignette, grain, noise reduction, sharpening, clarity.",
     {"type": "object", "properties": {
         "relight_on": {"type": "boolean"}, "relight_stops": {"type": "number", "description": "Background down, 0.2 to 1.5."},
         "subject_stops": {"type": "number", "description": "Subject up or down, usually 0."},
         "vignette_on": {"type": "boolean"}, "vignette_stops": {"type": "number"},
         "grain_on": {"type": "boolean"}, "grain_amount": {"type": "number"},
         "denoise": {"type": "integer", "minimum": 0, "maximum": 100}, "sharpen": {"type": "integer", "minimum": 0, "maximum": 100},
         "clarity": {"type": "integer", "minimum": 0, "maximum": 100}}}, set_finish),
    ("set_strength", "How much of the look shows, 0 to 100 percent.",
     {"type": "object", "properties": {"percent": {"type": "number", "minimum": 0, "maximum": 100}}, "required": ["percent"]}, set_strength),
    ("design_looks", "The app reads a shot and designs three looks for it, with previews and every step's values.",
     {"type": "object", "properties": {"shot_id": {"type": "string"}}}, design_looks),
    ("apply_look", "Apply one of the designed looks (id from design_looks) to every shot and rebuild.",
     {"type": "object", "properties": {"look_id": {"type": "string"}}, "required": ["look_id"]}, apply_look),
    ("rebuild_grade", "Run the app's full build: balance, shot match, look, skin protection and quality rules with their fixes.",
     {"type": "object", "properties": {}}, rebuild_grade),
    ("undo", "Put back the grade from before the last change made in this session.", {"type": "object", "properties": {}}, undo),
    ("save_look", "Save the current look by name, to apply it to other footage later.",
     {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}, save_look),
    ("export_video", "Render graded clips at source resolution. Returns the file paths.",
     {"type": "object", "properties": {"clip_ids": {"type": "array", "items": {"type": "string"}},
                                       "format": {"type": "string", "description": "auto, h265-10bit-mp4, h264-8bit-mp4, prores-422hq-mov, dnxhr-hqx-mov"}}},
     export_video),
]
HANDLERS = {name: fn for name, _, _, fn in TOOLS}

INSTRUCTIONS = (
    "You grade video in Color Grader, live: every change shows in the app the user watches. Work like a colorist: "
    "project_status, then view_shot to see the source and the grade with numbers. Fix each shot's balance first "
    "(adjust_shot: exposure, white balance, black point), then the shared look (adjust_look), then finish (set_finish). "
    "After every change, view_shot again and judge picture and numbers together: skin near the skin tone line and around "
    "40 to 70 percent luma, blacks near 2 to 5 percent, nothing new clipped, no failing rules. Keep moves small. "
    "design_looks gives three starting points built from the shot. undo reverts the last change."
)


# ----------------------------------------------------------------- protocol
def handle(msg: dict[str, Any]) -> dict[str, Any] | None:
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:   # notifications need no answer
        return None
    if method == "initialize":
        asked = (msg.get("params") or {}).get("protocolVersion")
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": asked if asked in SUPPORTED else PROTOCOL,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "color-grader", "version": "1.0.0"},
            "instructions": INSTRUCTIONS}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [
            {"name": n, "description": d, "inputSchema": schema} for n, d, schema, _ in TOOLS]}}
    if method == "tools/call":
        params = msg.get("params") or {}
        fn = HANDLERS.get(params.get("name"))
        if fn is None:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"Unknown tool: {params.get('name')}"}}
        try:
            content = fn(params.get("arguments") or {})
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": content, "isError": False}}
        except ToolError as exc:
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [_text(str(exc))], "isError": True}}
        except Exception as exc:   # report, keep serving
            return {"jsonrpc": "2.0", "id": mid, "result": {"content": [_text(f"{type(exc).__name__}: {exc}")], "isError": True}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}


def print_config() -> None:
    """The lines to paste into an MCP client, with this computer's paths filled in."""
    # The server needs only the standard library: pointing at the file works from any folder, which is
    # how MCP clients start it.
    py = sys.executable
    script = str(Path(__file__).resolve())
    print("Claude Desktop: Settings, Developer, Edit Config. Add this inside the file (merge with any mcpServers already there):\n")
    print(json.dumps({"mcpServers": {"color-grader": {"command": py, "args": [script]}}}, indent=2))
    print("\nClaude Code: run this once in a terminal:\n")
    print(f'claude mcp add color-grader -- "{py}" "{script}"')
    print("\nStart Color Grader first (start.bat or start.sh). The assistant then works in the app you have open.")


def main() -> None:
    if "--config" in sys.argv[1:]:
        print_config()
        return
    out = sys.stdout.buffer
    for line in sys.stdin.buffer:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            reply = handle(msg) if isinstance(msg, dict) else None
        if reply is not None:
            out.write(json.dumps(reply).encode("utf-8") + b"\n")
            out.flush()


if __name__ == "__main__":
    main()
