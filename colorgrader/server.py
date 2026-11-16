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
FEATURE_SUGGESTIONS = True


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
            "home": str(paths.home()), "features": {"suggestions": FEATURE_SUGGESTIONS}}


@app.get("/api/library")
def get_library() -> dict[str, Any]:
    cameras.seed()
    lib = library.get()
    return {"curves": lib.list_curves(), "gamuts": lib.list_gamuts(),
            "output_spaces": [{"id": k, "name": v["name"]} for k, v in W.OUTPUT_SPACES.items()],
            "ranges": list(cameras.RANGES), "settings_choices": settings.choices()}


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


# --------------------------------------------------------------- reference
@app.post("/api/reference")
def set_reference(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        return project.get().set_reference(body.get("path") or "")
    except (media.MediaError, OSError, ValueError) as exc:
        raise _err(exc)


@app.delete("/api/reference")
def clear_reference() -> dict[str, Any]:
    project.get().clear_reference()
    return {"ok": True}


@app.get("/api/reference/frame")
def reference_frame() -> Response:
    try:
        return Response(pipeline.to_jpeg(project.get().reference_display()), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except project.ProjectError as exc:
        raise _err(exc, 404)


@app.get("/api/reference/scope")
def reference_scope(kind: str = "waveform") -> Response:
    if kind not in scopes.SCOPES:
        raise HTTPException(404, "Unknown scope")
    try:
        return Response(scopes.render(kind, project.get().reference_display()), media_type="image/png", headers={"Cache-Control": "no-store"})
    except project.ProjectError as exc:
        raise _err(exc, 404)


# ------------------------------------------------------------------- grade
@app.post("/api/grade/build")
def build_grade(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    from .grade import auto

    from .grade import model, suggest

    p = project.get()
    refit = bool(body.get("refit", True))

    def work(progress):
        report = auto.build(p, lambda f, m: progress(0.6 * f, m), refit=refit)
        # no reference and no look yet: propose three looks for this footage
        if not p.state.get("reference") and model.is_neutral_look(p.state["look"]) and not p.state.get("suggestions"):
            suggest.suggest(p, lambda f, m: progress(0.6 + 0.4 * f, m))
        progress(1.0, "Grade ready")
        return report

    return {"job": jobs.start("grade", work, "Building the grade")}


def _recheck() -> dict[str, Any]:
    from .grade import auto

    p = project.get()
    if p.state.get("graded"):
        auto.recheck(p)
    return p.public_state()


@app.put("/api/grade/strength")
def set_strength(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    project.get().set_strength(float(body.get("strength", 80)))
    return _recheck() if body.get("recheck", True) else {"ok": True}


@app.put("/api/grade/brightness")
def set_brightness(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    project.get().set_brightness(float(body.get("brightness", 100)))
    return {"ok": True}


@app.put("/api/grade/look")
def set_look(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    project.get().set_look(body.get("look") or {})
    return _recheck()


@app.put("/api/grade/finish")
def set_finish(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    project.get().set_finish(body.get("finish") or {})
    return _recheck()


@app.put("/api/shots/{shot_id}/grade")
def set_shot_grade(shot_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        project.get().set_shot_grade(shot_id, body.get("grade") or {})
    except project.ProjectError as exc:
        raise _err(exc, 404)
    return _recheck()


@app.put("/api/shots/{shot_id}/scene")
def set_scene(shot_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        project.get().set_scene(shot_id, int(body.get("scene", 1)))
    except (project.ProjectError, ValueError) as exc:
        raise _err(exc)
    return {"ok": True}


# ------------------------------------------------------------------ recipe
@app.get("/api/recipe/editors")
def recipe_editors() -> list[dict[str, Any]]:
    from . import recipes

    return recipes.available()


@app.get("/api/recipe")
def get_recipe(editor: str | None = None, edition: str | None = None) -> dict[str, Any]:
    from . import recipes

    return recipes.build(project.get(), editor, edition)


@app.get("/api/recipe.txt")
def get_recipe_text(editor: str | None = None, edition: str | None = None) -> Response:
    from . import recipes

    r = recipes.build(project.get(), editor, edition)
    return Response(r["text"], media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="recipe_{r["editor"]}.txt"'})


# ------------------------------------------------------------------ export
@app.get("/api/export/formats")
def export_formats() -> list[dict[str, Any]]:
    from . import export

    return [{"id": k, "name": v["name"]} for k, v in export.FORMATS.items()]


@app.post("/api/export/luts")
def export_luts() -> dict[str, Any]:
    from . import export

    p = project.get()

    def work(progress):
        progress(0.1, "Building the LUT set")
        return export.export_luts(p)

    return {"job": jobs.start("export-luts", work, "Exporting the LUT set")}


@app.post("/api/export/video")
def export_video(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    from . import export

    p = project.get()
    ids = body.get("clip_ids") or None
    fmt = body.get("format") or None
    if fmt and fmt not in export.FORMATS:
        raise _err(ValueError("Unknown export format."))
    return {"job": jobs.start("export-video", lambda progress: export.render_clips(p, ids, fmt, progress), "Exporting graded clips")}


@app.post("/api/open-folder")
def open_folder(body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Show an export folder in the system file manager. Only folders inside the app's own folders open."""
    from . import export

    target = Path(body.get("path") or export.export_dir()).expanduser().resolve()
    if target.is_file():
        target = target.parent
    roots = [export.export_dir().resolve(), paths.home().resolve()]
    if not target.is_dir() or not any(target == r or r in target.parents for r in roots):
        raise _err(ValueError("This folder does not belong to the app."))
    try:
        if sys.platform.startswith("win"):
            import os

            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except OSError as exc:
        return {"ok": False, "path": str(target), "error": str(exc)}
    return {"ok": True, "path": str(target)}


# -------------------------------------------------------- graded playback
@app.get("/api/playback/status")
def playback_status(clip: str, stage: str = "graded") -> dict[str, Any]:
    from . import export

    p = project.get()
    try:
        ready = export.preview_ready(p, clip, stage)
    except project.ProjectError as exc:
        raise _err(exc, 404)
    return {"ready": ready, "url": f"/api/playback/video?clip={clip}&stage={stage}&v={int(time.time())}" if ready else None}


@app.post("/api/playback/build")
def playback_build(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from . import export

    p = project.get()
    clip, stage = body.get("clip_id", ""), body.get("stage", "graded")
    if stage not in ("graded", "converted", "balanced"):
        raise _err(ValueError("Unknown stage."))
    try:
        p.clip(clip)
    except project.ProjectError as exc:
        raise _err(exc, 404)
    return {"job": jobs.start("playback", lambda progress: export.build_preview(p, clip, stage, progress), "Rendering playback")}


@app.get("/api/playback/video")
def playback_video(clip: str, stage: str = "graded") -> FileResponse:
    from . import export

    try:
        f = export.preview_file(project.get(), clip, stage)
    except project.ProjectError as exc:
        raise _err(exc, 404)
    if not f.exists():
        raise HTTPException(404, "Render the playback first.")
    return FileResponse(f, media_type="video/mp4", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------- saved looks
@app.get("/api/looks")
def list_looks() -> list[dict[str, Any]]:
    from . import looks

    return looks.list_looks()


@app.post("/api/looks")
def save_look(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    from . import looks

    p = project.get()
    thumb = None
    try:  # preview picture: the lead shot of the project under the look
        shots = p.state["shots"]
        if shots:
            s = next((x for x in shots if x.get("is_lead")), shots[0])
            t = s["times"][len(s["times"]) // 2] if s.get("times") else 0.5 * (s["start"] + s["end"])
            thumb = pipeline.to_jpeg(p.render_frame(s["clip_id"], t, "graded", 480, s["id"]), 85)
    except Exception:
        thumb = None
    try:
        return looks.save_look(p, body.get("name", ""), thumb)
    except looks.LookError as exc:
        raise _err(exc)


@app.delete("/api/looks/{look_id}")
def delete_look(look_id: str) -> dict[str, Any]:
    from . import looks

    try:
        looks.delete_look(look_id)
    except looks.LookError as exc:
        raise _err(exc, 404)
    return {"ok": True}


@app.get("/api/looks/{look_id}/thumb.jpg")
def look_thumb(look_id: str) -> FileResponse:
    from . import looks

    f = looks.thumb_path(look_id)
    if not f.exists():
        raise HTTPException(404, "No preview")
    return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/looks/{look_id}/apply")
def apply_look(look_id: str) -> dict[str, Any]:
    from . import looks
    from .grade import auto

    p = project.get()
    try:
        looks.apply_look(p, look_id)
    except looks.LookError as exc:
        raise _err(exc, 404)
    if not p.state["shots"]:
        return {"job": None, "project": p.public_state()}
    return {"job": jobs.start("grade", lambda progress: auto.build(p, progress, refit=False), "Applying the saved look")}


# ------------------------------------------------------- suggested looks
@app.post("/api/suggestions")
def make_suggestions() -> dict[str, Any]:
    from .grade import suggest

    p = project.get()
    if not p.state["shots"]:
        raise _err(ValueError("Analyze at least one clip first."))
    return {"job": jobs.start("suggest", lambda progress: {"count": len(suggest.suggest(p, progress))}, "Finding looks for your footage")}


@app.get("/api/suggestions/{sid}/thumb.jpg")
def suggestion_thumb(sid: str) -> FileResponse:
    from .grade import suggest

    f = suggest.thumb_path(sid)
    if not f.exists():
        raise HTTPException(404, "No preview")
    return FileResponse(f, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/suggestions/{sid}/apply")
def apply_suggestion(sid: str) -> dict[str, Any]:
    from .grade import auto, suggest

    p = project.get()
    try:
        suggest.apply(p, sid)
    except KeyError as exc:
        raise _err(ValueError(str(exc).strip("'\"")), 404)
    return {"job": jobs.start("grade", lambda progress: auto.build(p, progress, refit=False), "Applying the look")}


# ------------------------------------------------------------------ static
@app.get("/")
def index() -> FileResponse:
    return FileResponse(paths.WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=str(paths.WEB_DIR)), name="static")


@app.exception_handler(Exception)
async def on_error(request, exc: Exception) -> JSONResponse:  # last resort: readable error for the UI
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
