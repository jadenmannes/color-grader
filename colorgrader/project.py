"""Project state: clips, shots, analysis results, reference, grade.

State is plain JSON in the user folder. Pixel data sits in the cache folder.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import cameras, media, paths, settings
from .analysis import faces as faces_mod
from .analysis import measure as measure_mod
from .analysis import shots as shots_mod
from .color import working as W
from .color.transforms import InputTransform, OutputTransform, input_transform_for
from .color.ycbcr import signal_to_absolute
from .grade import model, pipeline
from .grade import controls as controls_mod
from .grade import relight as relight_mod
from .color import chroma as chroma_mod

F32 = np.float32
SAMPLE_PER_FRAME = 3000
SAMPLE_MAX = 60000
KEEP_FRAMES = 4


class ProjectError(RuntimeError):
    pass


def _hash(*parts: Any) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:12]


def fingerprint(path: Path) -> str:
    """Content key of a media file: size plus the first and last megabyte. Two copies of one file share it, whatever their names."""
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(1 << 20))
        if size > 2 << 20:
            fh.seek(size - (1 << 20))
            h.update(fh.read(1 << 20))
    return h.hexdigest()[:16]


def clip_id_for(path: Path) -> str:
    st = path.stat()
    return _hash(str(path.resolve()), st.st_size, int(st.st_mtime))


class Project:
    def __init__(self, name: str = "current"):
        self.name = name
        self.lock = threading.RLock()
        self._frame_cache: OrderedDict[tuple, tuple[np.ndarray, int, int]] = OrderedDict()
        self._shot_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.state = self._blank()
        self.load()

    # ------------------------------------------------------------ storage
    def _blank(self) -> dict[str, Any]:
        st = settings.load()
        return {"version": 1, "clips": [], "shots": [], "scenes": [], "reference": None,
                "look": model.neutral_look(), "strength": int(st["look_strength"]),
                "finish": model.neutral_finish(), "suggestions": [], "graded": False, "report": None,
                "brightness": 100, "polish": 85}

    def _file(self) -> Path:
        return paths.sub("projects") / f"{self.name}.json"

    def load(self) -> None:
        try:
            data = json.loads(self._file().read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("version") == 1:
                blank = self._blank()
                blank.update(data)
                self.state = blank
        except (OSError, json.JSONDecodeError):
            pass

    def save(self) -> None:
        with self.lock:
            tmp = self._file().with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
            tmp.replace(self._file())

    def reset(self) -> None:
        with self.lock:
            self.state = self._blank()
            self._frame_cache.clear()
            self._shot_cache.clear()
            self._ref_cache = None
            self.save()

    # ------------------------------------------------------------ lookups
    def clip(self, clip_id: str) -> dict[str, Any]:
        for c in self.state["clips"]:
            if c["id"] == clip_id:
                return c
        raise ProjectError(f"No clip {clip_id}")

    def shot(self, shot_id: str) -> dict[str, Any]:
        for s in self.state["shots"]:
            if s["id"] == shot_id:
                return s
        raise ProjectError(f"No shot {shot_id}")

    def shots_of(self, clip_id: str) -> list[dict[str, Any]]:
        return [s for s in self.state["shots"] if s["clip_id"] == clip_id]

    def profile_of(self, clip: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        cam = cameras.get_camera(clip.get("camera_id") or "")
        if cam is None:
            raise ProjectError(f"{clip['name']}: no camera assigned. Add or pick a camera first.")
        prof = cameras.get_profile(cam, clip.get("profile_id") or "")
        if prof is None:
            raise ProjectError(f"{clip['name']}: pick a recording profile.")
        return cam, prof

    def odt(self) -> OutputTransform:
        st = settings.load()
        return OutputTransform(st["output_space"], float(st["max_input_nits"]))

    def idt(self, clip: dict[str, Any]) -> tuple[InputTransform, dict[str, Any]]:
        _, prof = self.profile_of(clip)
        st = settings.load()
        info = clip["info"]
        return input_transform_for(prof, info["full_range"], info["bit_depth"], float(st["max_input_nits"])), prof

    def cache_dir(self, clip_id: str) -> Path:
        p = paths.sub("cache") / clip_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -------------------------------------------------------------- clips
    def find_copy(self, p: Path) -> dict[str, Any] | None:
        """The clip in the project that holds the same file content, under any name or folder."""
        try:
            fp = fingerprint(p)
        except OSError:
            return None
        with self.lock:
            for c in self.state["clips"]:
                if c.get("fingerprint") is None:
                    try:
                        c["fingerprint"] = fingerprint(Path(c["path"]))
                    except OSError:
                        continue
                if c["fingerprint"] == fp:
                    return c
        return None

    def add_clip(self, path: str) -> dict[str, Any]:
        p = Path(path).expanduser()
        info = media.probe(p)
        cid = clip_id_for(p)
        with self.lock:
            for c in self.state["clips"]:
                if c["id"] == cid:
                    return c
        same = self.find_copy(p)
        if same is not None:   # the same footage under another name: one clip, one analysis
            return dict(same, duplicate=True)
        det = cameras.detect(info)
        clip = {"id": cid, "path": str(p), "name": p.name, "info": info, "fingerprint": fingerprint(p),
                "camera_id": det["camera_id"], "profile_id": det["profile_id"], "detect": det,
                "confirmed": not det["needs_confirm"], "analysis": None, "error": None}
        with self.lock:
            self.state["clips"].append(clip)
            self.save()
        return clip

    def remove_clip(self, clip_id: str) -> None:
        with self.lock:
            self.state["clips"] = [c for c in self.state["clips"] if c["id"] != clip_id]
            self.state["shots"] = [s for s in self.state["shots"] if s["clip_id"] != clip_id]
            self.state["suggestions"] = []   # suggested looks belong to the footage they were made for
            self.save()
        shutil.rmtree(paths.sub("cache") / clip_id, ignore_errors=True)

    def set_profile(self, clip_id: str, camera_id: str, profile_id: str) -> dict[str, Any]:
        cam = cameras.get_camera(camera_id)
        if cam is None or cameras.get_profile(cam, profile_id) is None:
            raise ProjectError("Unknown camera or profile.")
        with self.lock:
            clip = self.clip(clip_id)
            changed = (clip.get("camera_id"), clip.get("profile_id")) != (camera_id, profile_id)
            clip["camera_id"], clip["profile_id"], clip["confirmed"] = camera_id, profile_id, True
            if changed:
                for s in self.shots_of(clip_id):
                    s["measure"] = None
            self.save()
        return clip

    def redetect(self, clip_id: str) -> dict[str, Any]:
        """Run camera matching again, after the camera library changed."""
        with self.lock:
            clip = self.clip(clip_id)
            det = cameras.detect(clip["info"])
            clip["detect"] = det
            if not clip.get("confirmed") or clip.get("camera_id") is None:
                clip["camera_id"], clip["profile_id"] = det["camera_id"], det["profile_id"]
                clip["confirmed"] = not det["needs_confirm"]
            self.save()
            return clip

    # ----------------------------------------------------------- analysis
    def profile_signature(self, clip: dict[str, Any]) -> str:
        _, prof = self.profile_of(clip)
        st = settings.load()
        return _hash(prof, clip["info"]["full_range"], clip["info"]["bit_depth"], st["max_input_nits"])

    def analyze_clip(self, clip_id: str, progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
        """Decode once, split into shots, measure every shot."""
        say = progress or (lambda f, m: None)
        clip = self.clip(clip_id)
        info = clip["info"]
        self.profile_of(clip)  # fail early when no profile is set
        st = settings.load()
        cdir = self.cache_dir(clip_id)
        t0 = time.time()
        if info["is_image"]:
            meta = {"still": True}
            ranges = [(0.0, 0.0)]
        else:
            fps_a = max(1.0, float(st["analysis_fps"]))
            dur = max(info["duration"], 0.04)
            if dur * fps_a > 900:
                fps_a = max(1.0, 900.0 / dur)
            if dur * fps_a < 6:
                fps_a = min(info["fps"] or 25.0, 6.0 / dur)
            meta = clip.get("analysis") or {}
            have = meta.get("analysis_path") and Path(meta["analysis_path"]).exists() and abs(meta.get("analysis_fps", 0) - fps_a) < 1e-6
            if not have:
                say(0.02, f"Decoding {clip['name']}")
                meta = media.analysis_pass(info, cdir, fps_a, progress=lambda f: say(0.02 + 0.6 * f, f"Decoding {clip['name']}"))
            cuts, cut_info = shots_mod.detect_cuts(meta["cuts_path"], tuple(meta["cuts_size"]), meta["cuts_frames"], meta["cuts_fps"])
            meta["cuts"] = cuts
            ranges = shots_mod.shots_from_cuts(cuts, dur)
        meta["decode_seconds"] = round(time.time() - t0, 2)
        new_shots = []
        for i, (a, b) in enumerate(ranges):
            say(0.65 + 0.33 * i / max(1, len(ranges)), f"Measuring shot {i + 1} of {len(ranges)}")
            new_shots.append(self._analyze_shot(clip, meta, i, a, b))
        meta["seconds"] = round(time.time() - t0, 2)
        with self.lock:
            old = {s["id"]: s for s in self.shots_of(clip_id)}
            for s in new_shots:  # keep grades when the shot list did not change
                if s["id"] in old and abs(old[s["id"]]["start"] - s["start"]) < 1e-3 and abs(old[s["id"]]["end"] - s["end"]) < 1e-3:
                    s["grade"] = old[s["id"]].get("grade", s["grade"])
                    s["scene"] = old[s["id"]].get("scene", s["scene"])
            if [(s["id"], s["start"], s["end"]) for s in new_shots] != [(s["id"], s["start"], s["end"]) for s in old.values()]:
                self.state["suggestions"] = []   # new footage: earlier suggestions no longer describe it
            self.state["shots"] = [s for s in self.state["shots"] if s["clip_id"] != clip_id] + new_shots
            clip["analysis"] = meta
            clip["error"] = None
            self.save()
        say(1.0, "Done")
        return clip

    def stale_clips(self) -> list[str]:
        """Clips whose shots were measured under another profile or tone mapping setting."""
        out = []
        for c in self.state["clips"]:
            if not c.get("camera_id") or not c.get("profile_id"):
                continue
            try:
                sig = self.profile_signature(c)
            except ProjectError:
                continue
            shots = self.shots_of(c["id"])
            if shots and any(s.get("signature") != sig or not s.get("measure") for s in shots):
                out.append(c["id"])
        return out

    def _frame_lin(self, clip: dict[str, Any], buf: np.ndarray, w: int, h: int, idt: InputTransform, prof: dict[str, Any]):
        info = clip["info"]
        sig = media.raw_to_rgb(buf, w, h, info, full_range=idt.full_range)
        return sig, pipeline.camera_stage(sig, idt, prof.get("corrections"))

    def _analyze_shot(self, clip: dict[str, Any], meta: dict[str, Any], index: int, start: float, end: float) -> dict[str, Any]:
        info = clip["info"]
        idt, prof = self.idt(clip)
        odt = self.odt()
        sid = f"{clip['id']}-{index:03d}"
        frames: list[tuple[float, np.ndarray, int, int]] = []
        if meta.get("still"):
            buf, w, h = media.read_frame(info, 0.0, max_width=media.fit_size(info, pixels=230400)[0])
            frames.append((0.0, buf, w, h))
        else:
            mm = media.open_analysis(meta)
            aw, ah = meta["analysis_size"]
            fps_a = meta["analysis_fps"]
            margin = min(0.15, 0.1 * (end - start))
            idxs = [i for i in range(meta["analysis_frames"]) if start + margin <= i / fps_a < end - margin]
            if not idxs:
                mid = 0.5 * (start + end)
                buf, w, h = media.read_frame(info, mid, max_width=aw)
                frames.append((mid, buf, w, h))
            else:
                for i in idxs:
                    frames.append((i / fps_a, np.asarray(mm[i]), aw, ah))
        rng = np.random.default_rng(1234 + index)
        keep_idx = set(np.unique(np.linspace(0, len(frames) - 1, min(KEEP_FRAMES, len(frames))).round().astype(int)).tolist())
        per_frame = max(500, min(SAMPLE_PER_FRAME, SAMPLE_MAX // max(1, len(frames))))
        samp, samp_y, skins, kept, kept_t, face_list, track = [], [], [], [], [], [], []
        code_hist = np.zeros(1024, dtype=np.float64)
        for n, (t, buf, w, h) in enumerate(frames):
            sig, lin = self._frame_lin(clip, buf, w, h, idt, prof)
            code = signal_to_absolute(sig, idt.full_range, info["bit_depth"])
            code_hist += np.bincount(np.clip((code.max(axis=-1) * 1023.0).round().astype(np.int32), 0, 1023).ravel(), minlength=1024)
            flat = lin.reshape(-1, 3)
            pick = rng.choice(flat.shape[0], min(per_frame, flat.shape[0]), replace=False)
            samp.append(flat[pick])
            samp_y.append((pick // w).astype(F32) / max(1, h - 1))
            disp = odt(W.di_encode(lin))
            fl = faces_mod.detect_faces(disp)          # every analyzed frame: the relight follows the faces
            track.append([round(float(t), 3), relight_mod.normalize_faces(fl, w, h)])
            if n in keep_idx:
                kept.append(lin.astype(np.float16))
                kept_t.append(t)
                face_list.append(fl)
                if fl:
                    skins.append(faces_mod.skin_pixels(lin, fl))
        sample = np.concatenate(samp).astype(F32)
        ypos = np.concatenate(samp_y).astype(F32)
        skin = np.concatenate(skins).astype(F32) if skins else np.zeros((0, 3), dtype=F32)
        src_clip = _pileup(code_hist)
        npz = self.cache_dir(clip["id"]) / f"shot_{index:03d}.npz"
        np.savez(npz, sample=sample, ypos=ypos, skin=skin, frames=np.stack(kept), times=np.asarray(kept_t, dtype=F32))
        self._shot_cache.pop(sid, None)
        m = measure_mod.measure(W.di_encode(sample), odt, ypos, W.di_encode(skin) if len(skin) else None, src_clip)
        return {"id": sid, "clip_id": clip["id"], "index": index, "start": round(start, 4), "end": round(end, 4),
                "frames_analyzed": len(frames), "times": [round(float(t), 3) for t in kept_t],
                "faces": [[[round(float(v), 2) for v in f] for f in fl] for fl in face_list],
                "has_faces": any(len(fl) > 0 for fl in face_list) or any(f for _, f in track),
                "track": relight_mod.smooth_track(track),
                "scene": 1, "signature": self.profile_signature(clip), "measure": m, "summary": measure_mod.summarize(m),
                "source_clip": src_clip,
                "grade": model.neutral_shot(), "data": str(npz)}

    def face_track(self, shot_id: str) -> list[list[Any]]:
        """Faces through a shot, two samples per second. Built on first use for shots analyzed before tracking."""
        shot = self.shot(shot_id)
        if shot.get("track") is not None:
            return shot["track"]
        clip = self.clip(shot["clip_id"])
        meta = clip.get("analysis") or {}
        track: list[list[Any]] = []
        if meta.get("analysis_path") and Path(meta["analysis_path"]).exists():
            idt, prof = self.idt(clip)
            odt = self.odt()
            mm = media.open_analysis(meta)
            aw, ah = meta["analysis_size"]
            fps_a = meta["analysis_fps"]
            for i in range(meta["analysis_frames"]):
                t = i / fps_a
                if not (shot["start"] <= t < shot["end"]):
                    continue
                _, lin = self._frame_lin(clip, np.asarray(mm[i]), aw, ah, idt, prof)
                fl = faces_mod.detect_faces(odt(W.di_encode(lin)))
                track.append([round(t, 3), relight_mod.normalize_faces(fl, aw, ah)])
        else:   # no decode cache: the few frames measured with the shot
            aw, ah = (meta.get("analysis_size") or [clip["info"]["display_width"], clip["info"]["display_height"]])
            for t, fl in zip(shot.get("times") or [], shot.get("faces") or []):
                track.append([t, relight_mod.normalize_faces(fl, aw, ah)])
        track = relight_mod.smooth_track(track)
        with self.lock:
            shot["track"] = track
            self.save()
        return track

    def relight_mask(self, shot_id: str | None, t: float, width: int, height: int) -> np.ndarray | None:
        if not shot_id or not relight_mod.settings(self.state.get("finish"))[0]:
            return None
        try:
            return relight_mod.mask(relight_mod.faces_at(self.face_track(shot_id), t), width, height)
        except ProjectError:
            return None

    def shot_data(self, shot_id: str) -> dict[str, np.ndarray]:
        """Pooled linear sample, skin sample and a few frames of a shot (linear DaVinci Wide Gamut)."""
        with self.lock:
            if shot_id in self._shot_cache:
                self._shot_cache.move_to_end(shot_id)
                return self._shot_cache[shot_id]
        shot = self.shot(shot_id)
        f = Path(shot.get("data") or "")
        if not f.exists():
            raise ProjectError("Shot data is missing. Analyze the clip again.")
        with np.load(f) as z:
            data = {k: z[k] for k in z.files}
        with self.lock:
            self._shot_cache[shot_id] = data
            while len(self._shot_cache) > 24:
                self._shot_cache.popitem(last=False)
        return data

    # ------------------------------------------------------------- frames
    FRAME_CACHE_BYTES = 320 * 1024 * 1024   # decoded frames kept for the viewer: about six 4K frames or many previews

    @staticmethod
    def color_rebuild_auto(info: dict[str, Any]) -> bool:
        """Auto: an 8-bit 4:2:0 or 4:2:2 file compressed hard (an export, a download) gets its color rebuilt."""
        if info["is_image"] or info["is_rgb"] or info.get("chroma") not in ("420", "422") or int(info.get("bit_depth", 8)) > 8:
            return False
        bpp = media.bits_per_pixel(info)
        return bpp is not None and bpp < chroma_mod.AUTO_MAX_BPP

    def color_rebuild_on(self, clip: dict[str, Any]) -> bool:
        """Rebuild the clip's compressed color from its brightness: on, off, or auto (see color_rebuild_auto)."""
        mode = clip.get("color_rebuild", "auto")
        if mode in ("on", "off"):
            return mode == "on"
        return self.color_rebuild_auto(clip["info"])

    def set_color_rebuild(self, clip_id: str, mode: str) -> None:
        if mode not in ("auto", "on", "off"):
            raise ProjectError("Color rebuild: auto, on or off.")
        with self.lock:
            self.clip(clip_id)["color_rebuild"] = mode
            self._frame_cache.clear()
            self._still = None
        self.touch()

    def frame_signal(self, clip: dict[str, Any], t: float, width: int, prefilter: str = "", hq: bool = False) -> tuple[np.ndarray, int, int]:
        rebuild = self.color_rebuild_on(clip)
        key = (clip["id"], round(float(t), 3), int(width), prefilter, hq, rebuild)
        with self.lock:
            if key in self._frame_cache:
                self._frame_cache.move_to_end(key)
                return self._frame_cache[key]
        val = media.read_frame(clip["info"], t, max_width=width, prefilter=prefilter, hq=hq)
        if rebuild:
            buf, w, h = val
            val = (chroma_mod.rebuild(buf, w, h, w / float(clip["info"]["display_width"])), w, h)
        with self.lock:
            self._frame_cache[key] = val
            total = sum(v[0].nbytes for v in self._frame_cache.values())
            while len(self._frame_cache) > 1 and (total > self.FRAME_CACHE_BYTES or len(self._frame_cache) > 48):
                _, old = self._frame_cache.popitem(last=False)
                total -= old[0].nbytes
        return val

    def render_still(self, clip_id: str, t: float, stage: str = "graded", shot_id: str | None = None) -> np.ndarray:
        """One frame at the source's full resolution with the full grade: the best picture the app makes of a moment.
        Same steps as the viewer and the export, with the best scaler and exact chroma. The last still stays in memory,
        so saving the frame you look at costs no second render."""
        info = self.clip(clip_id)["info"]
        key = (clip_id, round(float(t), 3), stage, shot_id, int(self.state.get("stamp", 0)), self.state.get("strength"))
        cached = getattr(self, "_still", None)
        if cached and cached[0] == key:
            return cached[1]
        img = self.render_frame(clip_id, t, stage, int(info["display_width"]), shot_id, hq=True)
        self._still = (key, img)
        return img

    def render_frame(self, clip_id: str, t: float, stage: str = "graded", width: int | None = None,
                     shot_id: str | None = None, look: dict[str, Any] | None = None,
                     finish_override: dict[str, Any] | None = None, hq: bool = False) -> np.ndarray:
        """Display R'G'B' of one frame at a stage: source, converted, balanced, graded."""
        clip = self.clip(clip_id)
        info = clip["info"]
        width = int(width or settings.load()["preview_width"])
        finish = copy.deepcopy(finish_override or self.state["finish"]) if stage == "graded" else None
        prefilter = ""
        if finish and not info["is_image"]:
            # the viewer cleans the frame with the same filter as the export, so both show the same picture
            from . import export

            dn = pipeline.detail_values(finish)[0]
            prefilter = export.denoise_chain(dn, info["display_width"], info) if dn > 0 else ""
            if prefilter:
                finish["detail"]["denoise"] = 0
        buf, w, h = self.frame_signal(clip, t, width, prefilter, hq)
        idt, prof = self.idt(clip) if stage != "source" or clip.get("profile_id") else (None, {})
        if stage == "source":
            fr = idt.full_range if idt is not None else None
            return np.clip(media.raw_to_rgb(buf, w, h, info, full_range=fr), 0.0, 1.0)
        sig = media.raw_to_rgb(buf, w, h, info, full_range=idt.full_range)
        shot = None
        if shot_id:
            shot = model.quantize_shot(self.shot(shot_id)["grade"])
        if look is None and stage == "graded":
            look = model.at_strength(self.state["look"], self.state["strength"] / 100.0)
        rmask = None
        if finish and relight_mod.settings(finish)[0] and shot_id and not info["is_image"]:
            try:
                rmask = relight_mod.mask(relight_mod.faces_at(self.face_track(shot_id), t), w, h)
            except ProjectError:
                rmask = None
        return pipeline.render(sig, idt, self.odt(), prof.get("corrections"), shot, look, finish, stage, relight_mask=rmask)

    # ---------------------------------------------------------- reference
    def _ref_file(self) -> Path:
        d = paths.sub("cache") / "reference"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.name}.npz"

    def set_reference(self, path: str) -> dict[str, Any]:
        from .grade import reference

        state, arrays = reference.analyze(Path(path).expanduser(), self.odt())
        np.savez(self._ref_file(), **arrays)
        with self.lock:
            self._ref_cache = None
            self.state["reference"] = state
            self.state["stamp"] = int(self.state.get("stamp", 0)) + 1
            self.save()
        return state

    def clear_reference(self) -> None:
        with self.lock:
            self._ref_cache = None
            self.state["reference"] = None
            self.state["stamp"] = int(self.state.get("stamp", 0)) + 1
            self.save()

    def reference_arrays(self) -> dict[str, np.ndarray]:
        if getattr(self, "_ref_cache", None) is None:
            f = self._ref_file()
            if not f.exists():
                raise ProjectError("Reference data is missing. Add the reference again.")
            with np.load(f) as z:
                self._ref_cache = {k: z[k] for k in z.files}
        return self._ref_cache

    def reference_display(self) -> np.ndarray:
        return self.reference_arrays()["display"]

    # -------------------------------------------------------------- grade
    def touch(self) -> None:
        with self.lock:
            self.state["stamp"] = int(self.state.get("stamp", 0)) + 1
            self.save()

    def set_strength(self, strength: float) -> None:
        with self.lock:
            self.state["strength"] = int(min(100, max(0, round(float(strength)))))
        self.touch()

    def set_brightness(self, brightness: float) -> None:
        """How far the midtone level follows the reference, 0 to 100. Takes effect on the next build."""
        with self.lock:
            self.state["brightness"] = int(min(100, max(0, round(float(brightness)))))
        self.touch()

    def set_polish(self, polish: float) -> None:
        """Strength of the finishing pass on tone, 0 to 100. Takes effect on the next build."""
        with self.lock:
            self.state["polish"] = int(min(100, max(0, round(float(polish)))))
        self.touch()

    def set_look(self, look: dict[str, Any]) -> None:
        base = model.neutral_look()
        base.update({k: v for k, v in look.items() if k in base or k in ("reasons",)})
        with self.lock:
            self.state["look"] = model.quantize_look(base) | {"reasons": base.get("reasons", {})}
        self.touch()

    def set_finish(self, finish: dict[str, Any]) -> None:
        base = model.neutral_finish()
        for key in ("vignette", "grain", "relight"):
            base[key].update({k: v for k, v in (finish.get(key) or {}).items() if k in base[key]})
        for k in base["detail"]:
            try:
                base["detail"][k] = int(min(100, max(0, round(float((finish.get("detail") or {}).get(k, 0))))))
            except (TypeError, ValueError):
                pass
        with self.lock:
            self.state["finish"] = base
            self.state["detail_set"] = True
        self.touch()

    def set_shot_grade(self, shot_id: str, grade: dict[str, Any]) -> None:
        with self.lock:
            s = self.shot(shot_id)
            g = model.neutral_shot()
            g["balance"].update({k: v for k, v in (grade.get("balance") or {}).items() if k in g["balance"]})
            g["match"].update({k: v for k, v in (grade.get("match") or {}).items() if k in g["match"]})
            trim = grade["trim"] if "trim" in grade else (s.get("grade") or {}).get("trim", {})
            g["trim"] = {k: v for k, v in (trim or {}).items() if k in controls_mod.full_neutral()}
            g["reasons"] = dict(s.get("grade", {}).get("reasons", {}))
            g["reasons"].update({k: str(v) for k, v in (grade.get("reasons") or {}).items() if k.startswith("trim_")})
            s["grade"] = model.quantize_shot(g)
        self.touch()

    def adjust_controls(self, changes: dict[str, Any], shot_id: str | None = None, reasons: dict[str, Any] | None = None,
                        replace: bool = False) -> dict[str, Any]:
        """Change some color controls of the shared look (shot_id None) or of one shot's own trim. The rest stay.

        reasons: {control: why} or {"*": why for all}. replace=True starts the shot's trim from neutral.
        Returns the controls of that target that now differ from neutral. Raises ValueError on bad input.
        """
        ch = controls_mod.clean(changes)
        why = {str(k): str(v) for k, v in (reasons or {}).items() if v}
        say = lambda k: why.get(k, why.get("*", "Set by hand."))
        with self.lock:
            if shot_id is None:
                look = controls_mod.merge(copy.deepcopy(self.state["look"]), ch)
                look["reasons"] = {**(look.get("reasons") or {}), **{k: say(k) for k in ch}}
                if look.get("source") in (None, "", "none"):
                    look["source"], look["name"] = "edited", "Custom"
                self.set_look(look)
                return controls_mod.changed(self.state["look"])
            s = self.shot(shot_id)
            g = copy.deepcopy(s["grade"])
            g["trim"] = controls_mod.changed(controls_mod.merge({} if replace else (g.get("trim") or {}), ch))
            g["reasons"] = {**(g.get("reasons") or {}), **{f"trim_{k}": say(k) for k in ch}}
            self.set_shot_grade(shot_id, g)
            return controls_mod.changed(self.shot(shot_id)["grade"]["trim"])

    def qualifier_view(self, clip_id: str, t: float, width: int, shot_id: str | None = None, target: str = "look",
                       index: int | None = None) -> tuple[np.ndarray, dict[str, Any], str | None]:
        """The picture a secondary keys on (DaVinci Intermediate), its control set, and the shot.

        target "look": a secondary of the shared look. target "shot": one of the shot's own trim.
        index None: a new secondary at the end of the list.
        """
        clip = self.clip(clip_id)
        if shot_id is None:
            s = self.shot_at(clip_id, t)
            shot_id = s["id"] if s else None
        if target == "shot" and not shot_id:
            raise ProjectError("A shot trim needs a shot.")
        buf, w, h = self.frame_signal(clip, t, int(width))
        idt, prof = self.idt(clip)
        sig = media.raw_to_rgb(buf, w, h, clip["info"], full_range=idt.full_range)
        lin = pipeline.camera_stage(sig, idt, prof.get("corrections"))
        cam_sat = pipeline.camera_saturation(prof.get("corrections"))
        shot = model.quantize_shot(self.shot(shot_id)["grade"]) if shot_id else model.neutral_shot()
        if target == "shot":
            c = shot.get("trim") or {}
            di = pipeline.shot_stage(lin, {**shot, "trim": {}}, cam_sat)
        else:
            c = model.at_strength(self.state["look"], self.state["strength"] / 100.0)
            di = pipeline.shot_stage(lin, shot, cam_sat)
        return controls_mod.qualifier_input(di, c, index), c, shot_id

    def key_frame(self, clip_id: str, t: float, width: int, shot_id: str | None = None, target: str = "look",
                  index: int = 0, mode: str = "highlight") -> tuple[np.ndarray, float]:
        """What secondary `index` selects, as Resolve's Highlight mode (keyed picture in color, the rest gray) or as
        a black and white matte. Returns display R'G'B' and the share of the frame inside the key in percent."""
        di, c, _ = self.qualifier_view(clip_id, t, width, shot_id, target, index)
        secs = c.get("secondaries") or []
        if not 0 <= index < len(secs):
            where = "this shot's trim" if target == "shot" else "the look"
            raise ProjectError(f"No secondary {index + 1} on {where}: it has {len(secs)}.")
        k = controls_mod.secondary_key(di, secs[index])
        if mode == "matte":
            out = np.repeat(k[..., None], 3, axis=-1)
        else:
            disp = np.clip(self.odt()(di), 0.0, 1.0)
            out = disp * k[..., None] + np.float32(0.5) * (1.0 - k[..., None])
        return out.astype(np.float32), round(float(k.mean()) * 100.0, 1)

    def sample(self, clip_id: str, t: float, x: float, y: float, shot_id: str | None = None, target: str = "look",
               radius: float = 0.01) -> dict[str, Any]:
        """The color under one point of the frame (x, y from 0 to 1, left top origin): what a qualifier would key on
        there, the graded display value, and a starting qualifier around it."""
        from .recipes.base import hue_label

        x, y = min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)
        di, _, shot_id = self.qualifier_view(clip_id, t, 640, shot_id, target, None)

        def patch(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            cx, cy, r = int(round(x * (w - 1))), int(round(y * (h - 1))), max(1, int(round(radius * w)))
            return img[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1].reshape(-1, 3).mean(axis=0)

        col = controls_mod.describe_color(patch(di))
        shown = patch(np.clip(self.render_frame(clip_id, t, "graded", 640, shot_id), 0.0, 1.0)) if shot_id else None
        out = {"x": round(x, 4), "y": round(y, 4), "time": round(float(t), 3), "shot_id": shot_id, "target": target,
               "qualifier_sees": col, "color": hue_label(col["hue"]) if col["chroma"] >= 0.012 else "near gray",
               "suggested_qualifier": controls_mod.suggest_qualifier(col["hue"], col["chroma"], col["lum"])}
        if shown is not None:
            out["graded_rgb_percent"] = [round(float(v) * 100.0, 1) for v in shown]
            out["graded_luma_percent"] = round(float(shown @ np.array([0.2126, 0.7152, 0.0722])) * 100.0, 1)
        return out

    def readout(self, shot_id: str, t: float | None = None, stage: str = "graded") -> dict[str, Any]:
        """The numbers a colorist reads off the scopes for one shot: levels, clipping, color, skin. Display percent."""
        from .color.ycbcr import rgb_to_ycbcr
        from .grade import skin as skin_mod

        s = self.shot(shot_id)
        clip = self.clip(s["clip_id"])
        if t is None:
            t = s["times"][len(s["times"]) // 2] if s.get("times") else 0.5 * (s["start"] + s["end"])
        img = np.clip(self.render_frame(clip["id"], float(t), stage, 640, shot_id), 0.0, 1.0).reshape(-1, 3)
        y, cb, cr = rgb_to_ycbcr(img)
        sat = np.hypot(cb, cr) * 200.0
        mids = (y > 0.25) & (y < 0.65)
        pct = lambda v, q: round(float(np.percentile(v, q)) * 100.0, 1)
        out: dict[str, Any] = {
            "shot_id": shot_id, "time": round(float(t), 2), "stage": stage,
            "luma_percent": {"black_p1": pct(y, 1), "shadows_p10": pct(y, 10), "median": pct(y, 50), "highlights_p90": pct(y, 90), "white_p99": pct(y, 99)},
            "clipped_percent_of_pixels": {"at_white": round(float((img.min(axis=1) >= 0.995).mean()) * 100, 2),
                                          "at_black": round(float((img.max(axis=1) <= 0.004).mean()) * 100, 2)},
            "saturation_percent": {"midtones_mean": round(float(sat[mids].mean()) if mids.any() else float(sat.mean()), 1),
                                   "peak_p99": round(float(np.percentile(sat, 99)), 1)},
            "average_color_rgb_percent": [round(float(v) * 100, 1) for v in img.mean(axis=0)],
            "faces_at_time": len(relight_mod.faces_at(self.face_track(shot_id), float(t))) if not clip["info"]["is_image"] else 0,
        }
        data = self.shot_data(shot_id)
        if len(data.get("skin", [])) >= 30 and stage in ("graded", "balanced"):
            idt, prof = self.idt(clip)
            g = model.quantize_shot(s["grade"])
            sk = pipeline.shot_stage(data["skin"].astype(np.float32), g, pipeline.camera_saturation(prof.get("corrections")))
            if stage == "graded":
                sk = pipeline.look_stage(sk, model.at_strength(self.state["look"], self.state["strength"] / 100.0))
            odt = self.odt()
            ang, skin_sat = skin_mod.skin_on_display(sk, odt)
            off = skin_mod.offset_from_line(ang)
            skin_y = float(np.median(np.clip(odt(sk), 0.0, 1.0) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
            out["skin"] = {"luma_percent": round(skin_y * 100, 1), "saturation_percent": round(float(skin_sat), 1),
                           "degrees_from_skin_line": round(off, 1),
                           "leans": "on the line" if abs(off) <= 4 else ("toward yellow-green" if off > 0 else "toward red-magenta")}
        out["rules"] = [{"rule": r["name"], "status": r["status"], "detail": r["detail"]} for r in s.get("rules") or [] if r["status"] != "pass"]
        return out

    def set_focus(self, shot_id: str | None, t: float | None = None) -> None:
        """Which shot an outside tool is working on: the app's viewer follows it. Kept in memory only."""
        with self.lock:
            n = int((getattr(self, "_focus", None) or {}).get("n", 0)) + 1
            self._focus = {"shot_id": shot_id, "t": t, "n": n}

    def focus(self) -> dict[str, Any]:
        return dict(getattr(self, "_focus", None) or {"shot_id": None, "t": None, "n": 0})

    # ------------------------------------------------------ assistant activity
    ACTIVITY_KEYS = ("tool", "title", "reason", "shot_id", "t", "target", "controls", "point", "key", "area")

    def log_activity(self, ev: dict[str, Any]) -> dict[str, Any]:
        """One step an outside assistant (the MCP server) takes. The viewer shows it: a cursor on the control, a line in
        the feed. Kept in memory only, the last 60 steps."""
        e: dict[str, Any] = {k: ev[k] for k in self.ACTIVITY_KEYS if ev.get(k) is not None}
        for k, n in (("tool", 40), ("title", 200), ("reason", 300), ("shot_id", 80), ("target", 10), ("area", 20)):
            if k in e:
                e[k] = str(e[k])[:n]
        if not isinstance(e.get("controls"), list):
            e.pop("controls", None)
        else:
            e["controls"] = [c for c in e["controls"][:30] if isinstance(c, dict) and isinstance(c.get("key"), str)]
        for k in ("point", "key"):
            if k in e and not isinstance(e[k], dict):
                e.pop(k)
        with self.lock:
            n = int(getattr(self, "_activity_n", 0)) + 1
            self._activity_n = n
            e.update(n=n, time=round(time.time(), 3))
            log = getattr(self, "_activity", [])
            log.append(e)
            self._activity = log[-60:]
        return e

    def activity(self, since: int = 0) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(e) for e in getattr(self, "_activity", []) if e["n"] > since]

    def activity_n(self) -> int:
        return int(getattr(self, "_activity_n", 0))

    def set_scene(self, shot_id: str, scene: int) -> None:
        with self.lock:
            s = self.shot(shot_id)
            s["scene"], s["scene_locked"] = int(scene), True
        self.touch()

    def shot_at(self, clip_id: str, t: float) -> dict[str, Any] | None:
        for s in self.shots_of(clip_id):
            if s["start"] <= t < s["end"] or (s["start"] == s["end"]):
                return s
        ss = self.shots_of(clip_id)
        return ss[-1] if ss else None

    def public_state(self) -> dict[str, Any]:
        """State for the UI."""
        with self.lock:
            out = json.loads(json.dumps(self.state, default=str))
        out["applied_look"] = model.at_strength(out["look"], out.get("strength", 80) / 100.0)
        for c, raw in zip(out["clips"], self.state["clips"]):   # compression and the color rebuild, for the Analysis tab
            c["bits_per_pixel"] = media.bits_per_pixel(raw["info"])
            c["color_rebuild_auto"] = self.color_rebuild_auto(raw["info"])
            c["color_rebuild_active"] = self.color_rebuild_on(raw)
        out["applied_look"]["reasons"] = out["look"].get("reasons", {})
        return out


def _pileup(hist: np.ndarray) -> dict[str, float]:
    """Share of pixels piled at the top and bottom code levels of a shot (percent).

    A sensor at its limit, or a crushed black, shows as a spike at the end of the
    histogram. Smooth tails do not count.
    """
    total = max(float(hist.sum()), 1.0)
    nz = np.nonzero(hist > total * 1e-6)[0]
    if nz.size == 0:
        return {"high": 0.0, "low": 0.0, "max_code": 0.0, "min_code": 0.0}
    top, bot = int(nz[-1]), int(nz[0])
    hi = float(hist[max(0, top - 2):top + 1].sum())
    hi_ref = float(hist[max(0, top - 12):max(0, top - 2)].sum()) / 10.0 * 3.0
    lo = float(hist[bot:bot + 3].sum())
    lo_ref = float(hist[bot + 3:bot + 13].sum()) / 10.0 * 3.0
    high = hi / total * 100.0 if hi > 2.5 * hi_ref and hi / total > 0.0005 else 0.0
    low = lo / total * 100.0 if lo > 2.5 * lo_ref and lo / total > 0.0005 else 0.0
    return {"high": round(high, 3), "low": round(low, 3), "max_code": round(top / 1023.0, 4), "min_code": round(bot / 1023.0, 4)}


_project: Project | None = None


def get() -> Project:
    global _project
    if _project is None:
        cameras.seed()
        _project = Project()
    return _project


def reset_singleton() -> None:
    global _project
    _project = None
