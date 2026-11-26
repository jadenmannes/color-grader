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
    payload, _ = _request_full(method, path, body, params, timeout)
    return payload if raw else json.loads(payload.decode() or "null")


def _request_full(method: str, path: str, body: Any = None, params: dict[str, Any] | None = None, timeout: float = 120.0):
    """The response body as bytes and its headers."""
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.read(), dict(res.headers)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode()).get("detail")
        except Exception:
            detail = exc.reason
        raise ToolError(f"The app refused: {detail}")
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise ToolError(f"Color Grader is not running at {BASE} ({exc}). Start it with start.bat or start.sh, then try again.")


def _get(path, **params):
    return _request("GET", path, params=params)


def _put(path, body):
    return _request("PUT", path, body)


def _patch(path, body):
    return _request("PATCH", path, body)


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


def _remember() -> dict[str, Any]:
    """Keep the grade as it is now, so undo can put it back. Returns the project as it was."""
    p = _state()
    _history.append({"look": p["look"], "finish": p["finish"], "strength": p["strength"],
                     "shots": {s["id"]: s["grade"] for s in p["shots"] if s.get("grade")}})
    del _history[:-30]
    return p


# ----------------------------------------------------------------- activity
NAMES = {"exposure": "exposure", "temp": "temperature", "tint": "tint", "contrast": "contrast", "pivot": "pivot", "lift": "lift",
         "gamma": "gamma", "gain": "gain", "offset": "offset", "saturation": "saturation", "hue": "hue", "color_boost": "color boost",
         "shadows": "shadows", "highlights": "highlights", "hdr": "HDR zones", "rgb_mixer": "RGB mixer", "log": "log wheels",
         "hsv_sat": "HSV saturation", "tone_curve": "custom curve", "curves_rgb": "RGB curves", "lum_vs_sat": "Lum vs Sat",
         "hue_vs_hue": "Hue vs Hue", "hue_vs_sat": "Hue vs Sat", "hue_vs_lum": "Hue vs Lum", "sat_vs_sat": "Sat vs Sat",
         "sat_vs_lum": "Sat vs Lum", "secondaries": "secondaries"}


def _say(key: str, value: Any) -> str:
    name = NAMES.get(key, key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        signed = key in ("exposure", "temp", "tint", "shadows", "highlights", "hue", "color_boost")
        return f"{name} {value:+g}" if signed else f"{name} {value:g}"
    if isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
        return f"{name} " + " ".join(f"{v:g}" for v in value)
    return name


def _activity(**ev: Any) -> None:
    """Tell the app what the assistant does, so the person watching sees it: a cursor moves to the control, drags it,
    clicks where a color is sampled, and a feed lists each step. Never fails a tool."""
    try:
        _post("/api/activity", {k: v for k, v in ev.items() if v is not None})
    except ToolError:
        pass


def _shot_name(s: dict[str, Any]) -> str:
    return f"shot {s['index'] + 1}"


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
              "bit_depth": c["info"]["bit_depth"], "color_rebuild": bool(c.get("color_rebuild_active"))} for c in p["clips"]]
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
    _activity(tool="view_shot", title=f"Looking at {_shot_name(s)}", shot_id=s["id"], t=t, area="viewer")
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
    _activity(tool="read_shot", title=f"Reading the scopes of {_shot_name(s)}", shot_id=s["id"], area="viewer")
    return [_text(_get("/api/readout", shot=s["id"], t=args.get("time"), stage=args.get("stage", "graded")))]


UNITS = ("Look and trim controls: exposure, temp, tint, shadows, highlights and HDR zone exposure in stops. Contrast 1, saturation 50, "
         "offset 25, gain 1, lift and gamma 0, hue 0, color_boost 0 = unchanged. Curves: [input, output] points; DaVinci Intermediate "
         "inputs (black 0, middle gray 0.336), hue inputs in degrees (red 0, yellow 60, green 120, cyan 180, blue 240, magenta 300), "
         "chroma inputs (skin about 0.03 to 0.06). Shot balance: gain per channel on linear light, offset 25 = unchanged. "
         "The look is stored at full strength; strength_percent scales it toward neutral on screen.")


def get_grade(args: dict[str, Any]) -> list[dict[str, Any]]:
    out = _get("/api/grade/controls", shot=args.get("shot_id"))
    p = _state()
    out["finish"] = p["finish"]
    if args.get("shot_id"):
        s = _shot(p, args["shot_id"])
        out["shot"] = {"id": s["id"], "balance": s["grade"]["balance"],
                       "match": {k: s["grade"]["match"].get(k) for k in ("contrast", "pivot", "saturation")},
                       "reasons": s["grade"].get("reasons", {})}
    out["note"] = "look and trim list only the controls that differ from neutral. list_controls shows every control."
    out["units"] = UNITS
    return [_text(out)]


def list_controls(args: dict[str, Any]) -> list[dict[str, Any]]:
    spec = _get("/api/controls")
    keep = ("key", "label", "group", "type", "neutral", "min", "max", "unit", "help", "resolve", "exact")
    return [_text({"controls": [{k: c.get(k) for k in keep if c.get(k) is not None} for c in spec["controls"]],
                   "finish": [{k: c.get(k) for k in ("key", "label", "min", "max", "unit", "help") if c.get(k) is not None} for c in spec["finish"]],
                   "how": "adjust_look changes the shared look on every shot. adjust_shot with trim changes one shot only. set_secondary "
                          "adds a qualifier correction (sky, skin, foliage). exact false: the Resolve recipe sets that control by eye.",
                   "units": UNITS})]


def _changes(args: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    if strict:
        unknown = [k for k in args if k not in CONTROL_PROPS]
        if unknown:
            raise ToolError(f"Unknown control {', '.join(unknown)}. Controls: {', '.join(CONTROL_PROPS)}.")
    ch = {k: v for k, v in args.items() if k in CONTROL_PROPS and k not in ("hsv_gain", "hsv_gamma") and v is not None}
    hs = {f: float(args[f"hsv_{f}"]) for f in ("gain", "gamma") if args.get(f"hsv_{f}") is not None}
    if hs:
        ch["hsv_sat"] = hs
    return ch


def adjust_look(args: dict[str, Any]) -> list[dict[str, Any]]:
    ch = _changes(args)
    if not ch:
        raise ToolError("Nothing to change. Pass at least one control (list_controls shows them all).")
    before = _remember()
    state = _patch("/api/grade/look", {"changes": ch, "reasons": {"*": args.get("reason") or "Set by the assistant."}})
    _activity(tool="adjust_look", title="Look: " + ", ".join(_say(k, state["look"].get(k)) for k in ch), target="look",
              reason=args.get("reason"), controls=[{"key": k, "before": before["look"].get(k), "after": state["look"].get(k)} for k in ch])
    return [_text({"changed": list(ch), "now": {k: state["look"].get(k) for k in ch}, "quality": _rules_summary(state),
                   "next": "view_shot shows the result. undo puts the previous grade back."})]


def adjust_shot(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    s = _shot(p, args["shot_id"])
    balance = [k for k in ("exposure_stops", "warmth_stops", "tint_stops", "black_offset", "contrast", "saturation") if args.get(k) is not None]
    trim = _changes(args.get("trim") or {}, strict=True)
    if not balance and not trim and not args.get("reset_trim"):
        raise ToolError("Nothing to change. Pass balance values or trim controls.")
    _remember()
    out: dict[str, Any] = {"shot": s["id"]}
    state = None
    if balance:
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
        state = _put(f"/api/shots/{s['id']}/grade", {"grade": g})
        moves = [f"{lab} {float(args[k]):+g}" for k, lab in (("exposure_stops", "exposure"), ("warmth_stops", "warmth"),
                                                           ("tint_stops", "tint"), ("black_offset", "black point"),
                                                           ("contrast", "contrast"), ("saturation", "saturation")) if args.get(k) is not None]
        _activity(tool="adjust_shot", title=f"{_shot_name(s).capitalize()} balance: " + ", ".join(moves), shot_id=s["id"],
                  area="balance", reason=args.get("reason"))
        out["balance"] = g["balance"]
        out["match"] = {k: g["match"].get(k) for k in ("contrast", "pivot", "saturation")}
    if trim or args.get("reset_trim"):
        state = _patch(f"/api/shots/{s['id']}/trim", {"changes": trim, "replace": bool(args.get("reset_trim")),
                                                       "reasons": {"*": args.get("reason") or "Set by the assistant."}})
        out["trim"] = _shot(state, s["id"])["grade"].get("trim") or {}
        old_trim = s["grade"].get("trim") or {}
        keys = list(trim) or list(old_trim)
        _activity(tool="adjust_shot", title=f"{_shot_name(s).capitalize()} only: " + (", ".join(_say(k, out["trim"].get(k)) for k in trim) or "trim reset"),
                  target="shot", shot_id=s["id"], reason=args.get("reason"),
                  controls=[{"key": k, "before": old_trim.get(k), "after": out["trim"].get(k)} for k in keys])
    out["quality"] = _rules_summary(state)
    return [_text(out)]


def _owner_secs(p: dict[str, Any], shot_id: str | None) -> list[dict[str, Any]]:
    if shot_id:
        return copy.deepcopy((_shot(p, shot_id)["grade"].get("trim") or {}).get("secondaries") or [])
    return copy.deepcopy(p["look"].get("secondaries") or [])


def _view_shot_id(p: dict[str, Any], *candidates: str | None) -> str:
    for c in candidates:
        if c:
            return c
    focus = (p.get("focus") or {}).get("shot_id") if isinstance(p.get("focus"), dict) else None
    if focus:
        return focus
    if not p["shots"]:
        raise ToolError("No shots yet. Load footage in the app first.")
    return p["shots"][0]["id"]


def _key_image(p: dict[str, Any], shot_id: str, owner: str | None, index: int, mode: str = "highlight",
               t: float | None = None, width: int = 640) -> tuple[dict[str, Any], float]:
    s = _shot(p, shot_id)
    t = _mid(s) if t is None else float(t)
    jpg, headers = _request_full("GET", "/api/key", params={"clip": s["clip_id"], "t": round(t, 3), "w": width, "shot": s["id"],
                                                            "target": "shot" if owner else "look", "index": index - 1, "mode": mode})
    cover = float({k.lower(): v for k, v in headers.items()}.get("x-key-coverage", "0"))
    return {"type": "image", "data": base64.b64encode(jpg).decode(), "mimeType": "image/jpeg"}, cover


SEC_ADJUST = {"exposure": "exposure", "temp": "temp", "tint": "tint", "saturation": "saturation", "hue_rotate": "hue"}


def set_secondary(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    owner = args.get("shot_id")          # None: a secondary of the shared look
    secs = _owner_secs(p, owner)
    where = f"shot {owner}" if owner else "the look"
    idx = args.get("index")
    if idx is not None:
        idx = int(idx)
        if not 1 <= idx <= len(secs):
            raise ToolError(f"No secondary {idx} on {where}: it has {len(secs)}.")
    body_reason = {"*": args.get("reason") or "Set by the assistant."}
    save = (lambda ss: _patch(f"/api/shots/{owner}/trim", {"changes": {"secondaries": ss}, "reasons": body_reason})) if owner \
        else (lambda ss: _patch("/api/grade/look", {"changes": {"secondaries": ss}, "reasons": body_reason}))
    if args.get("remove"):
        if idx is None:
            raise ToolError("remove needs the index of the secondary.")
        gone = secs.pop(idx - 1)
        _remember()
        state = save(secs)
        _activity(tool="set_secondary", title=f"Removing secondary {idx} ({gone.get('name')})", target="shot" if owner else "look",
                  shot_id=owner, controls=[{"key": "secondaries"}])
        return [_text({"removed": gone.get("name"), "left": [x.get("name") for x in _owner_secs(state, owner)],
                       "quality": _rules_summary(state)})]
    if idx is None:
        secs.append({"name": args.get("name") or f"Secondary {len(secs) + 1}", "enabled": True, "invert": False, "adjust": {}})
        idx = len(secs)
    sec = secs[idx - 1]
    picked = None
    pick = args.get("pick")
    if pick:
        ps = _shot(p, _view_shot_id(p, pick.get("shot_id"), owner))
        t = float(pick["time"]) if pick.get("time") is not None else _mid(ps)
        picked = _get("/api/sample", clip=ps["clip_id"], t=round(t, 3), x=pick["x"], y=pick["y"], shot=ps["id"],
                      target="shot" if owner else "look")
        _activity(tool="set_secondary", title=f"Picking {picked.get('color', 'this color')} for {sec.get('name')}", shot_id=ps["id"],
                  t=t, point={"x": float(pick["x"]), "y": float(pick["y"])})
        for q, v in picked["suggested_qualifier"].items():
            if args.get(q) is None:
                sec[q] = v
        if args.get("hue") is None and "hue" not in picked["suggested_qualifier"]:
            sec.pop("hue", None)
    for q in ("hue", "sat", "lum"):
        if q in args:
            if args[q]:
                sec[q] = args[q]
            else:
                sec.pop(q, None)       # [] or null: that part of the key is off
    for k in ("name", "invert", "enabled"):
        if args.get(k) is not None:
            sec[k] = args[k]
    adj = dict(sec.get("adjust") or {})
    for a, k in SEC_ADJUST.items():
        if args.get(a) is not None:
            adj[k] = float(args[a])
    sec["adjust"] = adj
    _remember()
    state = save(secs)
    saved = _owner_secs(state, owner)[idx - 1]
    view = _view_shot_id(state, (pick or {}).get("shot_id"), owner)
    moves = [f"{k} {v:+g}" if k != "saturation" else f"saturation x{v:g}" for k, v in (saved.get("adjust") or {}).items()
             if abs(v - (1.0 if k == "saturation" else 0.0)) > 1e-9]
    _activity(tool="set_secondary", title=f"Secondary {idx} ({saved.get('name')})" + (": " + ", ".join(moves) if moves else ""),
              target="shot" if owner else "look", shot_id=view, reason=args.get("reason"), controls=[{"key": "secondaries"}],
              key={"target": "shot" if owner else "look", "index": idx - 1, "name": saved.get("name")})
    img, cover = _key_image(state, view, owner, idx)
    out = {"secondary": idx, "on": where, "saved": saved, "key_coverage_percent": cover,
           "next": "The image shows what the key selects (color) and what it leaves (gray). Narrow or widen hue, sat and lum "
                   "until only the wanted area shows, then view_shot to judge the correction.", "quality": _rules_summary(state)}
    if picked:
        out["picked"] = {k: picked[k] for k in ("color", "qualifier_sees", "graded_luma_percent") if k in picked}
    return [_text(out), img]


def view_key(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    owner = args["shot_id"] if args.get("of") == "shot" else None
    secs = _owner_secs(p, owner)
    idx = int(args.get("index", 1))
    if not 1 <= idx <= len(secs):
        raise ToolError(f"No secondary {idx} on {'this shot' if owner else 'the look'}: it has {len(secs)}.")
    s = _shot(p, args["shot_id"])
    t = float(args["time"]) if args.get("time") is not None else _mid(s)
    img, cover = _key_image(p, s["id"], owner, idx, args.get("mode", "highlight"), t, int(args.get("width", 640)))
    _put("/api/focus", {"shot_id": s["id"], "t": t})
    _activity(tool="view_key", title=f"Checking what {secs[idx - 1].get('name')} selects: {cover:.1f}% of the frame", shot_id=s["id"], t=t,
              key={"target": "shot" if owner else "look", "index": idx - 1, "name": secs[idx - 1].get("name")})
    mode = args.get("mode", "highlight")
    say = ("Highlight mode: the selected parts in color, the rest flat gray." if mode == "highlight"
           else "Matte: white is selected, black is left alone.")
    return [_text(f"Secondary {idx} ({secs[idx - 1].get('name')}) on shot {s['id']} at {t:.2f} s: {cover:.1f}% of the frame selected. {say}"), img]


def sample_color(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    s = _shot(p, args["shot_id"])
    t = float(args["time"]) if args.get("time") is not None else _mid(s)
    res = _get("/api/sample", clip=s["clip_id"], t=round(t, 3), x=args["x"], y=args["y"], shot=s["id"],
               target="shot" if args.get("of") == "shot" else "look")
    _activity(tool="sample_color", title=f"Sampling the color here: {res.get('color', '')}", shot_id=s["id"], t=t,
              point={"x": float(args["x"]), "y": float(args["y"])})
    return [_text(res)]


def reset_controls(args: dict[str, Any]) -> list[dict[str, Any]]:
    neutral = _get("/api/controls")["neutral"]
    keys = args.get("controls") or []
    unknown = [k for k in keys if k not in neutral]
    if unknown:
        raise ToolError(f"Unknown control {', '.join(unknown)}.")
    owner = args.get("shot_id")
    if owner:
        _shot(_state(), owner)
    before = _remember()
    if owner:
        body = {"changes": {k: neutral[k] for k in keys}} if keys else {"changes": {}, "replace": True}
        state = _patch(f"/api/shots/{owner}/trim", body)
    else:
        state = _patch("/api/grade/look", {"changes": {k: neutral[k] for k in (keys or neutral)}, "reasons": {"*": "Reset to neutral."}})
    if owner:
        old, new = _shot(before, owner)["grade"].get("trim") or {}, _shot(state, owner)["grade"].get("trim") or {}
        ks = keys or list(old)
    else:
        old, new = before["look"], state["look"]
        ks = keys or [k for k in neutral if json.dumps(old.get(k)) != json.dumps(new.get(k))]
    _activity(tool="reset_controls", title="Reset to neutral: " + (", ".join(NAMES.get(k, k) for k in ks) or "nothing changed"),
              target="shot" if owner else "look", shot_id=owner,
              controls=[{"key": k, "before": old.get(k), "after": new.get(k, neutral.get(k))} for k in ks])
    return [_text({"reset": keys or "all controls", "on": f"shot {owner}" if owner else "the look", "quality": _rules_summary(state)})]


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
    _activity(tool="set_finish", title="Finish: " + ", ".join(f"{k.replace('_', ' ')} {args[k]}" for k in changed), area="finish")
    return [_text({"changed": changed, "finish": state["finish"], "quality": _rules_summary(state)})]


def set_strength(args: dict[str, Any]) -> list[dict[str, Any]]:
    _remember()
    state = _put("/api/grade/strength", {"strength": float(args["percent"])})
    _activity(tool="set_strength", title=f"Look strength {float(args['percent']):g}%", area="strength")
    return [_text({"strength_percent": state["strength"], "quality": _rules_summary(state)})]


def design_looks(args: dict[str, Any]) -> list[dict[str, Any]]:
    _activity(tool="design_looks", title="Designing three looks from the shot", shot_id=args.get("shot_id"), area="looks")
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
    _activity(tool="apply_look", title="Applying a designed look", area="looks")
    job = _wait(_post(f"/api/suggestions/{args['look_id']}/apply")["job"])
    p = _state()
    return [_text({"applied": p["look"].get("name"), "result": job.get("result"), "quality": _rules_summary(p)})]


def rebuild_grade(args: dict[str, Any]) -> list[dict[str, Any]]:
    _remember()
    _activity(tool="rebuild_grade", title="Rebuilding the grade", area="build")
    _wait(_post("/api/grade/build", {})["job"])
    p = _state()
    return [_text({"look": p["look"].get("name"), "notes": (p.get("report") or {}).get("notes"),
                   "fixes": (p.get("report") or {}).get("fixes"), "quality": _rules_summary(p)})]


def undo(args: dict[str, Any]) -> list[dict[str, Any]]:
    if not _history:
        raise ToolError("Nothing to undo in this session.")
    h = _history.pop()
    _activity(tool="undo", title="Undo: the grade goes back one step")
    _put("/api/grade/look", {"look": h["look"]})
    _put("/api/grade/finish", {"finish": h["finish"]})
    for sid, grade in h["shots"].items():
        _put(f"/api/shots/{sid}/grade", {"grade": grade})
    state = _put("/api/grade/strength", {"strength": h["strength"]})
    return [_text({"restored": state["look"].get("name"), "steps_left": len(_history), "quality": _rules_summary(state)})]


def save_look(args: dict[str, Any]) -> list[dict[str, Any]]:
    _activity(tool="save_look", title=f"Saving the look as {args['name']}")
    return [_text(_post("/api/looks", {"name": args["name"]}))]


def export_video(args: dict[str, Any]) -> list[dict[str, Any]]:
    p = _state()
    ids = args.get("clip_ids") or [c["id"] for c in p["clips"] if not c["info"]["is_image"]]
    _activity(tool="export_video", title=f"Exporting {len(ids)} graded clip{'s' if len(ids) != 1 else ''}")
    job = _wait(_post("/api/export/video", {"clip_ids": ids, "format": args.get("format") or "auto"})["job"])
    return [_text(job.get("result"))]


RGB = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
POINTS = {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}}
TRIPLE = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}


def _n(text: str) -> dict[str, Any]:
    return {"type": "number", "description": text}


_ZONE = {"type": "object", "properties": {"exposure": _n("stops"), "saturation": _n("1 = unchanged"),
                                          "temp": _n("stops, + warmer"), "tint": _n("stops, + magenta")}}
CONTROL_PROPS: dict[str, Any] = {
    "exposure": _n("Stops on linear light, 0 = unchanged. Black stays black."),
    "temp": _n("White balance in stops: + warmer, - cooler, 0 = unchanged. Typical moves 0.05 to 0.3."),
    "tint": _n("Stops: + toward magenta, - toward green, 0 = unchanged."),
    "contrast": _n("1 = unchanged. Typical 1.05 to 1.25."),
    "pivot": _n("Contrast pivot in DaVinci Intermediate, 0.336 = middle gray."),
    "lift": {**RGB, "description": "Shadow level and color, R G B offsets, 0 = unchanged. Typical 0.005 to 0.03."},
    "gamma": {**RGB, "description": "Midtone level and color, R G B offsets, 0 = unchanged."},
    "gain": {**RGB, "description": "Highlight level and color, R G B multipliers, 1 = unchanged."},
    "offset": {**RGB, "description": "Every tone, R G B in Resolve points, 25 = unchanged. 10 points = one stop."},
    "saturation": _n("Resolve Sat, 50 = unchanged, 0 = black and white."),
    "hue": _n("Turns every color, degrees, 0 = unchanged."),
    "color_boost": _n("Vibrance: saturates gentle colors and leaves strong ones. 0 = unchanged, 1 = gentle colors doubled."),
    "shadows": _n("Stops at the center of the shadows: + opens, - deepens. Black and middle gray stay."),
    "highlights": _n("Stops at the center of the highlights: - recovers skies and windows, + lifts. Middle gray stays."),
    "hdr": {"type": "object", "properties": {z: _ZONE for z in ("dark", "shadow", "light", "highlight", "specular")},
            "description": "HDR zones by brightness: dark (under -3.5 stops), shadow (about -2), light (about +0.75: skin and upper mids), "
                           "highlight (about +2.5), specular (+4.5 and up). Each takes exposure (stops), saturation (1 = unchanged), "
                           "temp and tint (stops). Pass only the zones and fields to change."},
    "rgb_mixer": {"type": "array", "items": RGB, "minItems": 3, "maxItems": 3,
                  "description": "Rows: red, green, blue output, each the R G B input weights. [[1,0,0],[0,1,0],[0,0,1]] = unchanged."},
    "log": {"type": "object", "properties": {"shadow": RGB, "midtone": RGB, "highlight": RGB,
                                             "low_range": {"type": "number"}, "high_range": {"type": "number"}},
            "description": "Log wheels: shadow, midtone, highlight (R G B offsets in DI, typical 0.005 to 0.03), each only inside its range. "
                           "low_range (default 0.333) and high_range (0.55) set the borders."},
    "hsv_gain": _n("HSV node saturation Gain: lifts strong colors. 1 = unchanged."),
    "hsv_gamma": _n("HSV node saturation Gamma: lifts gentle colors such as skin. 0 = unchanged."),
    "tone_curve": {**POINTS, "description": "Custom curve, all channels: [input, output] in DaVinci Intermediate (0 to 1, middle gray 0.336). [] = unchanged."},
    "curves_rgb": {"type": "object", "properties": {"r": POINTS, "g": POINTS, "b": POINTS},
                   "description": "One curve per channel (gang off): {r, g, b}, each [input, output] points in DI. [] = unchanged."},
    "lum_vs_sat": {**POINTS, "description": "[luma in DI, saturation multiplier]."},
    "hue_vs_hue": {**POINTS, "description": "[hue degrees, rotation degrees]. Red 0, yellow 60, green 120, cyan 180, blue 240, magenta 300."},
    "hue_vs_sat": {**POINTS, "description": "[hue degrees, saturation multiplier]."},
    "hue_vs_lum": {**POINTS, "description": "[hue degrees, brightness multiplier]. Grays stay."},
    "sat_vs_sat": {**POINTS, "description": "[input chroma, saturation multiplier]. Chroma: skin about 0.03 to 0.06, strong colors 0.15 and up."},
    "sat_vs_lum": {**POINTS, "description": "[input chroma, brightness multiplier]."},
}

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
    ("get_grade", "The look's controls that differ from neutral (shared by all shots), strength, finish, and with shot_id that shot's "
                  "balance and its own trim, with units.",
     {"type": "object", "properties": {"shot_id": {"type": "string"}}}, get_grade),
    ("list_controls", "Every color control with its range, unit, what it does and the DaVinci Resolve tool it matches.",
     {"type": "object", "properties": {}}, list_controls),
    ("adjust_look", "Change the shared look on every shot. Every Resolve-style control: exposure, temp, tint, contrast, pivot, lift, "
                    "gamma, gain, offset, saturation, hue, color boost, shadows, highlights, HDR zones, log wheels, RGB mixer, HSV "
                    "saturation, custom and RGB curves, hue and sat curves. Pass only the controls to change, as absolute values "
                    "(read get_grade first). Keep moves small.",
     {"type": "object", "properties": {**CONTROL_PROPS,
                                       "reason": {"type": "string", "description": "One line on why: it shows in the app and the recipe."}}},
     adjust_look),
    ("adjust_shot", "Correct one shot on its own. Balance (relative): exposure_stops, warmth_stops, tint_stops, black_offset; match "
                    "(absolute): contrast, saturation. trim: any look control for this shot only, absolute, same names and units as "
                    "adjust_look. reset_trim clears the shot's trim first.",
     {"type": "object", "properties": {
         "shot_id": {"type": "string"},
         "exposure_stops": {"type": "number", "description": "Brighter or darker, in stops, relative."},
         "warmth_stops": {"type": "number", "description": "Positive warmer (more red, less blue), relative."},
         "tint_stops": {"type": "number", "description": "Positive toward magenta (less green), relative."},
         "black_offset": {"type": "number", "description": "Offset points on linear light, relative. Negative deepens hazy blacks."},
         "contrast": {"type": "number", "description": "This shot's match contrast, absolute, 1 = unchanged."},
         "saturation": {"type": "number", "description": "This shot's match Sat, absolute, 50 = unchanged."},
         "trim": {"type": "object", "properties": CONTROL_PROPS, "description": "This shot's own color controls."},
         "reset_trim": {"type": "boolean"},
         "reason": {"type": "string"}},
      "required": ["shot_id"]}, adjust_shot),
    ("sample_color", "The color under one point of a shot (x and y from 0 to 1, top left origin): its hue, chroma and lum as a "
                     "qualifier sees it, the graded display value, and a starting qualifier around it.",
     {"type": "object", "properties": {"shot_id": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"},
                                       "time": {"type": "number"},
                                       "of": {"type": "string", "enum": ["look", "shot"], "default": "look",
                                              "description": "Sample where a secondary of the look (or of this shot's trim) would key."}},
      "required": ["shot_id", "x", "y"]}, sample_color),
    ("set_secondary", "Add, change or remove a qualifier secondary: key one color family or area (hue, sat, lum ranges), then change "
                      "exposure, temp, tint, saturation or hue only there, like a Qualifier node in Resolve. With shot_id it belongs to "
                      "that shot's trim, without it to the look on every shot. pick seeds the key from a point of the frame. Omit index "
                      "to add a new one. Returns the key as an image.",
     {"type": "object", "properties": {
         "shot_id": {"type": "string", "description": "Owner: this shot's trim. Omit for the shared look."},
         "index": {"type": "integer", "minimum": 1, "description": "Which secondary, from 1. Omit to add a new one."},
         "remove": {"type": "boolean"},
         "name": {"type": "string", "description": "Sky, Skin, Foliage..."},
         "pick": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "time": {"type": "number"},
                                                   "shot_id": {"type": "string"}},
                  "required": ["x", "y"], "description": "Point to key on, x and y from 0 to 1 (top left origin). Fills hue, sat and lum you leave out."},
         "hue": {**TRIPLE, "description": "[center, width, soft] in degrees. [] turns the hue key off."},
         "sat": {**TRIPLE, "description": "[low, high, soft] in chroma (gray 0, skin about 0.03 to 0.06). [] turns it off."},
         "lum": {**TRIPLE, "description": "[low, high, soft] in DI (black 0, middle gray 0.336). [] turns it off."},
         "invert": {"type": "boolean", "description": "Correct everything outside the key."},
         "enabled": {"type": "boolean"},
         "exposure": _n("Stops inside the key."), "temp": _n("Stops inside the key, + warmer."), "tint": _n("Stops inside the key, + magenta."),
         "saturation": _n("Multiplier inside the key, 1 = unchanged."), "hue_rotate": _n("Degrees inside the key."),
         "reason": {"type": "string"}}}, set_secondary),
    ("view_key", "See what a secondary selects on a shot: Highlight mode (selection in color, the rest gray) or the matte, plus the "
                 "share of the frame selected.",
     {"type": "object", "properties": {"shot_id": {"type": "string"}, "index": {"type": "integer", "minimum": 1, "default": 1},
                                       "of": {"type": "string", "enum": ["look", "shot"], "default": "look"},
                                       "mode": {"type": "string", "enum": ["highlight", "matte"], "default": "highlight"},
                                       "time": {"type": "number"}, "width": {"type": "integer", "default": 640, "minimum": 320, "maximum": 1920}},
      "required": ["shot_id"]}, view_key),
    ("reset_controls", "Put controls back to neutral: on the look, or with shot_id on that shot's trim. Omit controls to reset all of them.",
     {"type": "object", "properties": {"shot_id": {"type": "string"}, "controls": {"type": "array", "items": {"type": "string"}}}},
     reset_controls),
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
    "You grade video in Color Grader, live: every change shows in the app the user watches. Work like a colorist. "
    "Start with project_status, then view_shot to see the source and the grade with numbers. Order of work: "
    "1. each shot's balance (adjust_shot: exposure, warmth, tint, black point; its trim takes any control for that shot alone); "
    "2. the shared look (adjust_look: exposure, temp, tint, lift, gamma, gain, offset, contrast, saturation, hue, color boost, "
    "shadows, highlights, HDR zones, log wheels, RGB mixer, curves; list_controls has ranges and units); "
    "3. secondaries for one color family or area such as sky, skin or foliage (sample_color on a point, set_secondary with pick, "
    "view_key to check the selection); 4. finish (set_finish: relight, vignette, grain, noise reduction, sharpening, clarity). "
    "Keep noise reduction at 0 unless the footage is visibly noisy in low light: it removes texture along with the noise. "
    "On clips with color_rebuild true (hard-compressed 8-bit files) add color with saturation, and keep color boost low: it "
    "lifts near-gray colors the most and turns leftover compression color into purple and green blotches on gray skies, "
    "walls and white clothes. "
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
