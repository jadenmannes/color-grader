"""Project state: clips, shots, analysis results, reference, grade.

State is plain JSON in the user folder. Pixel data sits in the cache folder.
"""
from __future__ import annotations

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
                "finish": model.neutral_finish(), "suggestions": [], "graded": False, "report": None}

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
    def add_clip(self, path: str) -> dict[str, Any]:
        p = Path(path).expanduser()
        info = media.probe(p)
        cid = clip_id_for(p)
        with self.lock:
            for c in self.state["clips"]:
                if c["id"] == cid:
                    return c
        det = cameras.detect(info)
        clip = {"id": cid, "path": str(p), "name": p.name, "info": info,
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
        samp, samp_y, skins, kept, kept_t, face_list = [], [], [], [], [], []
        code_hist = np.zeros(1024, dtype=np.float64)
        for n, (t, buf, w, h) in enumerate(frames):
            sig, lin = self._frame_lin(clip, buf, w, h, idt, prof)
            code = signal_to_absolute(sig, idt.full_range, info["bit_depth"])
            code_hist += np.bincount(np.clip((code.max(axis=-1) * 1023.0).round().astype(np.int32), 0, 1023).ravel(), minlength=1024)
            flat = lin.reshape(-1, 3)
            pick = rng.choice(flat.shape[0], min(per_frame, flat.shape[0]), replace=False)
            samp.append(flat[pick])
            samp_y.append((pick // w).astype(F32) / max(1, h - 1))
            if n in keep_idx:
                kept.append(lin.astype(np.float16))
                kept_t.append(t)
                disp = odt(W.di_encode(lin))
                fl = faces_mod.detect_faces(disp)
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
                "has_faces": any(len(fl) > 0 for fl in face_list),
                "scene": 1, "signature": self.profile_signature(clip), "measure": m, "summary": measure_mod.summarize(m),
                "source_clip": src_clip,
                "grade": model.neutral_shot(), "data": str(npz)}

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
    def frame_signal(self, clip: dict[str, Any], t: float, width: int) -> tuple[np.ndarray, int, int]:
        key = (clip["id"], round(float(t), 3), int(width))
        with self.lock:
            if key in self._frame_cache:
                self._frame_cache.move_to_end(key)
                return self._frame_cache[key]
        val = media.read_frame(clip["info"], t, max_width=width)
        with self.lock:
            self._frame_cache[key] = val
            while len(self._frame_cache) > 16:
                self._frame_cache.popitem(last=False)
        return val

    def render_frame(self, clip_id: str, t: float, stage: str = "graded", width: int | None = None,
                     shot_id: str | None = None, look: dict[str, Any] | None = None) -> np.ndarray:
        """Display R'G'B' of one frame at a stage: source, converted, balanced, graded."""
        clip = self.clip(clip_id)
        info = clip["info"]
        width = int(width or settings.load()["preview_width"])
        buf, w, h = self.frame_signal(clip, t, width)
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
        finish = self.state["finish"] if stage == "graded" else None
        return pipeline.render(sig, idt, self.odt(), prof.get("corrections"), shot, look, finish, stage)

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

    def set_look(self, look: dict[str, Any]) -> None:
        base = model.neutral_look()
        base.update({k: v for k, v in look.items() if k in base or k in ("reasons",)})
        with self.lock:
            self.state["look"] = model.quantize_look(base) | {"reasons": base.get("reasons", {})}
        self.touch()

    def set_finish(self, finish: dict[str, Any]) -> None:
        base = model.neutral_finish()
        for key in ("vignette", "grain"):
            base[key].update({k: v for k, v in (finish.get(key) or {}).items() if k in base[key]})
        with self.lock:
            self.state["finish"] = base
        self.touch()

    def set_shot_grade(self, shot_id: str, grade: dict[str, Any]) -> None:
        with self.lock:
            s = self.shot(shot_id)
            g = model.neutral_shot()
            g["balance"].update({k: v for k, v in (grade.get("balance") or {}).items() if k in g["balance"]})
            g["match"].update({k: v for k, v in (grade.get("match") or {}).items() if k in g["match"]})
            g["reasons"] = dict(s.get("grade", {}).get("reasons", {}))
            s["grade"] = model.quantize_shot(g)
        self.touch()

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
