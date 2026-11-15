"""Local web server: JSON API plus the static web UI. Listens on localhost only."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, cameras, jobs, library, media, paths, project, settings
from .analysis import scopes
from .color import working as W
from .color.transforms import OutputTransform, input_transform_for
from .grade import pipeline

app = FastAPI(title="Color Grader", version=__version__)

_previews: dict[str, dict[str, Any]] = {}


def _err(exc: Exception, code: int = 400) -> HTTPException:
    return HTTPException(status_code=code, detail=str(exc))


def _draft_signature(cam: dict[str, Any]) -> str:
    core = [(p.get("name"), p.get("input"), p.get("data_range")) for p in cam.get("profiles", [])]
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ------------------------------------------------------------------- state
@app.get("/api/state")
def get_state() -> dict[str, Any]:
    p = project.get()
    return {"project": p.public_state(), "settings": settings.load(), "jobs": jobs.active(), "version": __version__,
            "home": str(paths.home())}


@app.get("/api/library")
def get_library() -> dict[str, Any]:
    cameras.seed()
    lib = library.get()
    return {"curves": lib.list_curves(), "gamuts": lib.list_gamuts(),
            "output_spaces": [{"id": k, "name": v["name"]} for k, v in W.OUTPUT_SPACES.items()],
            "ranges": list(cameras.RANGES), "settings_choices": settings.CHOICES}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return settings.load()


@app.put("/api/settings")
def put_settings(values: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return settings.save(values)
    except (ValueError, TypeError) as exc:
        raise _err(exc)


@app.post("/api/project/reset")
def reset_project() -> dict[str, Any]:
    project.get().reset()
    return get_state()


# ----------------------------------------------------------------- cameras
@app.get("/api/cameras")
def list_cameras() -> list[dict[str, Any]]:
    return cameras.list_cameras()


@app.post("/api/cameras/preview")
def preview_camera(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Convert a sample clip with a camera that is not saved yet. Shows original next to converted."""
    try:
        cam = cameras.normalize(body.get("camera") or {})
    except cameras.CameraError as exc:
        raise _err(exc)
    idx = int(body.get("profile_index", 0))
    if not (0 <= idx < len(cam["profiles"])):
        raise _err(ValueError("Pick a profile to preview."))
    sample = body.get("sample_path") or ""
    try:
        info = media.probe(sample)
    except media.MediaError as exc:
        raise _err(exc)
    prof = cam["profiles"][idx]
    st = settings.load()
    # curve and gamut definitions inside the draft must resolve before the camera is saved
    lib = library.get()
    defs = cam.get("definitions") or {}
    added_c = {k: v for k, v in (defs.get("curves") or {}).items() if k not in lib.curve_specs}
    added_g = {k: v for k, v in (defs.get("gamuts") or {}).items() if k not in lib.gamut_specs}
    lib.curve_specs.update(added_c)
    lib.gamut_specs.update(added_g)
    try:
        t = 0.0 if info["is_image"] else float(body.get("t", info["duration"] * 0.3))
        buf, w, h = media.read_frame(info, t, max_width=960)
        idt = input_transform_for(prof, info["full_range"], info["bit_depth"], float(st["max_input_nits"]))
        sig = media.raw_to_rgb(buf, w, h, info, full_range=idt.full_range)
        odt = OutputTransform(st["output_space"], float(st["max_input_nits"]))
        conv = pipeline.render(sig, idt, odt, prof.get("corrections"), stage="converted")
    except (media.MediaError, KeyError, OSError, ValueError) as exc:
        raise _err(exc)
    finally:
        for k in added_c:
            lib.curve_specs.pop(k, None)
        for k in added_g:
            lib.gamut_specs.pop(k, None)
    token = uuid.uuid4().hex[:12]
    yv = conv @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    _previews[token] = {"original": pipeline.to_jpeg(np.clip(sig, 0, 1)), "converted": pipeline.to_jpeg(conv),
                        "signature": _draft_signature(cam), "time": time.time()}
    while len(_previews) > 12:
        _previews.pop(next(iter(_previews)))
    return {"token": token, "signature": _draft_signature(cam), "sample": {k: info[k] for k in (
        "name", "codec", "pix_fmt", "bit_depth", "chroma", "color_range", "color_space", "color_transfer", "color_primaries",
        "width", "height", "make", "model")},
        "stats": {"black": round(float(np.percentile(yv, 0.5)) * 100, 1), "mid": round(float(np.median(yv)) * 100, 1),
                  "white": round(float(np.percentile(yv, 99.5)) * 100, 1)}}


@app.get("/api/preview/{token}/{which}.jpg")
def preview_image(token: str, which: str) -> Response:
    item = _previews.get(token)
    if item is None or which not in ("original", "converted"):
        raise HTTPException(404, "Preview expired. Convert the sample again.")
    return Response(item[which], media_type="image/jpeg")


@app.post("/api/cameras")
def save_camera(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    cam = body.get("camera") or {}
    replace_id = body.get("replace_id")
    is_new = not replace_id
    try:
        norm = cameras.normalize(cam)
        if is_new:
            if cameras.get_camera(norm["id"]) is not None:
                raise cameras.CameraError(f"A camera named '{norm['name']}' exists. Pick another name.")
            item = _previews.get(body.get("preview_token") or "")
            if item is None or item["signature"] != _draft_signature(norm):
                raise cameras.CameraError("Convert a sample clip first. Check the result next to the original, then save.")
        saved = cameras.save_camera(cam, replace_id=replace_id)
    except cameras.CameraError as exc:
        raise _err(exc)
    p = project.get()
    for c in list(p.state["clips"]):
        if c.get("camera_id") is None:
            p.redetect(c["id"])
    return saved


@app.delete("/api/cameras/{cam_id}")
def delete_camera(cam_id: str) -> dict[str, Any]:
    return {"deleted": cameras.delete_camera(cam_id)}


@app.get("/api/cameras/{cam_id}/export")
def export_camera(cam_id: str) -> Response:
    try:
        cam = cameras.export_camera(cam_id)
    except cameras.CameraError as exc:
        raise _err(exc, 404)
    return Response(json.dumps(cam, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{cam["id"]}.camera.json"'})


@app.post("/api/cameras/import")
def import_camera(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        data = body.get("camera")
        if data is None and body.get("path"):
            data = json.loads(Path(body["path"]).read_text(encoding="utf-8"))
        return cameras.import_camera(data or {}, overwrite=bool(body.get("overwrite")))
    except (cameras.CameraError, OSError, json.JSONDecodeError) as exc:
        raise _err(exc)


@app.post("/api/cameras/restore-defaults")
def restore_cameras() -> dict[str, Any]:
    return {"restored": cameras.restore_defaults()}


@app.post("/api/luts")
def add_lut(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Copy a maker LUT into the user folder. Returns the stored file name."""
    from .color.lut import read_cube

    src = Path(body.get("path") or "")
    try:
        lut = read_cube(src)
    except (OSError, ValueError) as exc:
        raise _err(exc)
    name = paths.safe_id(src.stem) + ".cube"
    shutil.copyfile(src, paths.sub("luts") / name)
    return {"lut_file": name, "size": lut.size, "title": lut.title}


# ------------------------------------------------------------------- files
@app.post("/api/pick")
def pick(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    kind = body.get("kind", "clips")
    try:
        r = subprocess.run([sys.executable, "-m", "colorgrader.pick", kind], capture_output=True, timeout=600)
        out = json.loads(r.stdout.decode("utf-8", errors="replace").strip().splitlines()[-1])
    except Exception as exc:
        return {"paths": [], "error": f"No file dialog here ({exc}). Paste the file path instead."}
    return {"paths": out.get("paths", []), "error": out.get("error")}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Fallback for drag and drop: copies the file into the app cache."""
    folder = paths.sub("cache") / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "upload").name
    target = folder / f"{uuid.uuid4().hex[:6]}_{name}"
    with target.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    return {"path": str(target)}


# ------------------------------------------------------------------- clips
@app.post("/api/clips")
def add_clips(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    p = project.get()
    added, errors = [], []
    for path in body.get("paths") or []:
        try:
            added.append(p.add_clip(path))
        except (media.MediaError, OSError) as exc:
            errors.append({"path": path, "error": str(exc)})
    return {"added": added, "errors": errors}


@app.delete("/api/clips/{clip_id}")
def remove_clip(clip_id: str) -> dict[str, Any]:
    project.get().remove_clip(clip_id)
    return {"ok": True}


@app.post("/api/clips/{clip_id}/profile")
def set_profile(clip_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return project.get().set_profile(clip_id, body.get("camera_id", ""), body.get("profile_id", ""))
    except project.ProjectError as exc:
        raise _err(exc)


@app.post("/api/clips/{clip_id}/redetect")
def redetect(clip_id: str) -> dict[str, Any]:
    try:
        return project.get().redetect(clip_id)
    except project.ProjectError as exc:
        raise _err(exc)


@app.post("/api/analyze")
def analyze(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    p = project.get()
    ids = body.get("clip_ids") or [c["id"] for c in p.state["clips"]]
    todo = []
    for cid in ids:
        c = p.clip(cid)
        if not c.get("camera_id") or not c.get("profile_id"):
            raise _err(ValueError(f"{c['name']}: assign a camera and profile first."))
        todo.append(cid)

    def work(progress):
        for i, cid in enumerate(todo):
            p.analyze_clip(cid, lambda f, m, i=i: progress((i + f) / len(todo), m))
        return {"clips": todo}

    return {"job": jobs.start("analyze", work, "Analyzing footage")}


@app.get("/api/jobs/{jid}")
def get_job(jid: str) -> dict[str, Any]:
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404, "No such job")
    return job


# ------------------------------------------------------------------ images
def _frame(clip: str, t: float, stage: str, w: int, shot: str | None) -> np.ndarray:
    p = project.get()
    try:
        if shot is None and stage in ("balanced", "graded"):
            s = p.shot_at(clip, t)
            shot = s["id"] if s else None
        return p.render_frame(clip, t, stage, w, shot)
    except (project.ProjectError, media.MediaError, KeyError) as exc:
        raise _err(exc)


@app.get("/api/frame")
def frame(clip: str, t: float = 0.0, stage: str = "converted", w: int = Query(1280, ge=64, le=3840), shot: str | None = None) -> Response:
    return Response(pipeline.to_jpeg(_frame(clip, t, stage, w, shot)), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/scope")
def scope(clip: str, kind: str = "waveform", t: float = 0.0, stage: str = "converted", shot: str | None = None) -> Response:
    if kind not in scopes.SCOPES:
        raise HTTPException(404, "Unknown scope")
    return Response(scopes.render(kind, _frame(clip, t, stage, 640, shot)), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/thumb")
def thumb(shot: str) -> Response:
    p = project.get()
    try:
        s = p.shot(shot)
        t = s["times"][len(s["times"]) // 2] if s.get("times") else 0.5 * (s["start"] + s["end"])
        img = p.render_frame(s["clip_id"], t, "converted", 320)
    except (project.ProjectError, media.MediaError, KeyError) as exc:
        raise _err(exc)
    return Response(pipeline.to_jpeg(img, 82), media_type="image/jpeg", headers={"Cache-Control": "max-age=30"})


# ------------------------------------------------------------------ static
@app.get("/")
def index() -> FileResponse:
    return FileResponse(paths.WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=str(paths.WEB_DIR)), name="static")


@app.exception_handler(Exception)
async def on_error(request, exc: Exception) -> JSONResponse:  # last resort: readable error for the UI
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
