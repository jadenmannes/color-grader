"""LUT export and graded clip export.

LUTs carry the grade into any editor. The graded clip renders with ffmpeg:
decode at full bit depth, convert to R'G'B' in float, apply the grade as
65-point LUTs with tetrahedral interpolation, add vignette and grain in the
working space, convert to Rec.709, dither, encode and tag.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import media, paths, settings
from .color import chroma
from .color import working as W
from .color.lut import Lut3D, write_cube
from .grade import relight as relight_mod
from .grade import model, pipeline

F32 = np.float32

FORMATS: dict[str, dict[str, Any]] = {
    "auto": {"name": "Auto: H.265 10-bit, best quality for every source"},
    "h265-10bit-mp4": {"name": "H.265 10-bit 4:2:0 (MP4)", "ext": ".mp4", "pix": "yuv420p10le",
                       "codec": ["-c:v", "libx265", "-preset", "medium", "-crf", "14", "-tag:v", "hvc1"], "x265": True},
    "h264-8bit-mp4": {"name": "H.264 8-bit 4:2:0 (MP4), for players without H.265", "ext": ".mp4", "pix": "yuv420p",
                      "codec": ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-profile:v", "high"]},
    "prores-422hq-mov": {"name": "ProRes 422 HQ 10-bit (MOV)", "ext": ".mov", "pix": "yuv422p10le",
                         "codec": ["-c:v", "prores_ks", "-profile:v", "3", "-vendor", "apl0"]},
    "dnxhr-hqx-mov": {"name": "DNxHR HQX 10-bit (MOV)", "ext": ".mov", "pix": "yuv422p10le",
                      "codec": ["-c:v", "dnxhd", "-profile:v", "dnxhr_hqx"]},
}


class ExportError(RuntimeError):
    pass


def export_dir() -> Path:
    st = settings.load()
    d = Path(st["export_folder"]).expanduser() if st.get("export_folder") else paths.sub("exports")
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"The export folder {d} cannot be made ({exc.strerror or exc}). Pick another folder in Settings.") from exc
    return d


def _safe(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9._-]+", "_", name)).strip("_") or "clip"


# -------------------------------------------------------------------- LUTs
def _look_now(proj) -> dict[str, Any]:
    return model.at_strength(proj.state["look"], proj.state["strength"] / 100.0)


def lut_look(proj, size: int) -> Lut3D:
    """Look only: working space in, working space out."""
    look = _look_now(proj)
    return Lut3D.from_function(lambda v: np.clip(pipeline.look_stage(v, look), 0.0, 1.0), size, title="Look, DaVinci WG/Intermediate")


Z_GAMMA = 2.2


def _encode(display_signed: np.ndarray, encoding: str) -> np.ndarray:
    """LUT output encoding. "display": display code values. "z22": signed display code raised to 2.2.

    Near black a display curve rises with infinite slope and then clips, which a LUT lattice
    cannot follow. The "z22" form stays close to linear light and runs straight through black,
    negative values included. The export undoes it with an exact transfer step after the LUT
    (zscale, linear to gamma 2.2), which also clips at black.
    """
    if encoding == "z22":
        d = np.clip(display_signed, -1.0, 1.0)
        return np.sign(d) * np.power(np.abs(d), F32(Z_GAMMA))
    return np.clip(display_signed, 0.0, 1.0)


def best_encoding(fn_signed: Callable[[np.ndarray], np.ndarray], size: int) -> str:
    """Pick the LUT output encoding with the smaller interpolation error on the gray axis.

    fn_signed returns signed display code values. The error counts in lightness (display
    code to the power 0.8, close to how the eye weighs it), averaged over the ramp.
    Log sources favor "z22", finished pictures "display".
    """
    grid = np.linspace(0.0, 1.0, size, dtype=F32)
    fine = np.linspace(0.0, 1.0, (size - 1) * 8 + 1, dtype=F32)
    true = np.clip(fn_signed(np.repeat(fine[:, None], 3, axis=1)), 0.0, 1.0)[:, 1].astype(np.float64)
    at_grid = fn_signed(np.repeat(grid[:, None], 3, axis=1))
    light = lambda d: np.power(np.maximum(d, 0.0), 0.8)
    direct = np.interp(fine, grid, _encode(at_grid, "display")[:, 1])
    z = np.interp(fine, grid, _encode(at_grid, "z22")[:, 1])
    z = np.power(np.clip(z, 0.0, 1.0), 1.0 / Z_GAMMA)
    err_direct = float(np.mean(np.abs(light(direct) - light(true))))
    err_z = float(np.mean(np.abs(light(z) - light(true))))
    return "z22" if err_z < err_direct else "display"


def lut_output(proj, size: int, encoding: str = "display") -> Lut3D:
    odt = proj.odt()
    return Lut3D.from_function(lambda v: _encode(odt.signed(v), encoding), size, title="Output: DaVinci WG/Intermediate to Rec.709")


def lut_look_output(proj, size: int) -> Lut3D:
    look, odt = _look_now(proj), proj.odt()
    return Lut3D.from_function(lambda v: np.clip(odt(pipeline.look_stage(v, look)), 0.0, 1.0), size, title="Look + output to Rec.709")


def lut_input(proj, clip: dict[str, Any], size: int) -> Lut3D:
    """Camera signal to the working space, with the profile's default corrections."""
    idt, prof = proj.idt(clip)
    corr, cam_sat = prof.get("corrections"), pipeline.camera_saturation(prof.get("corrections"))

    def fn(v):
        from .color import ops

        return np.clip(ops.saturation(W.di_encode(pipeline.camera_stage(v, idt, corr)), cam_sat), 0.0, 1.0)

    return Lut3D.from_function(fn, size, title="Input: camera to DaVinci WG/Intermediate")


def camera_function(proj, clip: dict[str, Any], shot: dict[str, Any] | None = None, to: str = "display",
                    with_look: bool = True) -> Callable[[np.ndarray], np.ndarray]:
    """Camera signal through the grade, as a function. shot=None leaves the per-shot balance out.

    to: "display" ends in Rec.709 as signed display code values (see OutputTransform.signed),
    "working" stops before finish and output (the export pipeline).
    """
    idt, prof = proj.idt(clip)
    corr, cam_sat = prof.get("corrections"), pipeline.camera_saturation(prof.get("corrections"))
    grade = model.quantize_shot(shot["grade"]) if shot else model.neutral_shot()
    look, odt = (_look_now(proj) if with_look else None), proj.odt()

    def fn(v):
        di = pipeline.shot_stage(pipeline.camera_stage(v, idt, corr), grade, cam_sat)
        if look is not None:
            di = pipeline.look_stage(di, look)
        return di if to == "working" else odt.signed(di)

    return fn


def lut_camera(proj, clip: dict[str, Any], size: int, shot: dict[str, Any] | None = None, to: str = "display",
               with_look: bool = True, encoding: str = "display") -> Lut3D:
    """Camera signal through the grade as a LUT. See camera_function()."""
    fn = camera_function(proj, clip, shot, to, with_look)
    if to == "working":  # values below black stay negative: the output table clips them
        return Lut3D.from_function(lambda v: np.clip(fn(v), -1.0, 1.0), size, title="Camera to DaVinci WG/Intermediate")
    return Lut3D.from_function(lambda v: _encode(fn(v), encoding), size, title="Camera to Rec.709 with look")


OUTPUT_TAGS = {"rec709-gamma24": "Rec709", "rec709-gamma22": "Rec709-Gamma22", "srgb": "sRGB"}


def lut_names(proj) -> dict[str, Any]:
    """File names of the LUT set. The recipe refers to the same names."""
    st = settings.load()
    look = proj.state["look"]
    name = _safe(look.get("name") or "look")
    out_tag = OUTPUT_TAGS.get(st["output_space"], "Rec709")
    names: dict[str, Any] = {"look": f"{name}_LOOK_DWG-Intermediate.cube",
                             "look_output": f"{name}_LOOK_AND_OUTPUT_DWG-Intermediate_to_{out_tag}.cube",
                             "output": f"OUTPUT_DWG-Intermediate_to_{out_tag}.cube", "profiles": {}}
    for clip in proj.state["clips"]:
        key = (clip.get("camera_id"), clip.get("profile_id"))
        if not all(key) or "/".join(key) in names["profiles"]:
            continue
        try:
            cam, prof = proj.profile_of(clip)
        except Exception:
            continue
        tag = _safe(f"{cam['name']}_{prof['name'].split(' (')[0]}")
        names["profiles"]["/".join(key)] = {"camera": f"{name}_{tag}_to_{out_tag}.cube", "input": f"INPUT_{tag}_to_DWG-Intermediate.cube",
                                            "camera_name": cam["name"], "profile_name": prof["name"], "clip_id": clip["id"]}
    return names


def export_luts(proj, folder: Path | None = None) -> dict[str, Any]:
    """Write the LUT set and a text file with placement instructions."""
    st = settings.load()
    size = int(st["lut_size"])
    look = proj.state["look"]
    name = _safe(look.get("name") or "look")
    folder = folder or (export_dir() / f"luts_{name}_{time.strftime('%Y%m%d_%H%M%S')}")
    folder.mkdir(parents=True, exist_ok=True)
    strength = proj.state["strength"]
    out_name = W.OUTPUT_SPACES[st["output_space"]]["name"]
    names = lut_names(proj)
    files = []

    def add(fname: str, lut: Lut3D, where: str, comments: list[str]):
        path = write_cube(folder / fname, lut, comments=[f"Color Grader: {look.get('name', 'look')} at {strength:g}%"] + comments)
        files.append({"file": str(path), "name": fname, "where": where})

    add(names["look"], lut_look(proj, size),
        "DaVinci Resolve, timeline level. Place it after the balance nodes and before the output conversion. It replaces the look nodes of the recipe.",
        ["Input: DaVinci Wide Gamut / DaVinci Intermediate", "Output: DaVinci Wide Gamut / DaVinci Intermediate",
         "Place after input conversion and balance, before the output conversion."])
    add(names["look_output"], lut_look_output(proj, size),
        f"DaVinci Resolve, last node. It replaces the look nodes and both output conversion nodes. Output is {out_name}.",
        ["Input: DaVinci Wide Gamut / DaVinci Intermediate", f"Output: {out_name}", "Place as the last node, after balance."])
    add(names["output"], lut_output(proj, size),
        "Optional. The app's output conversion as one LUT, for the last node. Same result as the two output nodes of the recipe.",
        ["Input: DaVinci Wide Gamut / DaVinci Intermediate", f"Output: {out_name}"])
    for key, n in names["profiles"].items():
        clip = proj.clip(n["clip_id"])
        add(n["camera"], lut_camera(proj, clip, size),
            f"Any editor. Apply to {n['camera_name']} {n['profile_name']} clips as they come from the camera. Holds input conversion, look and output. "
            "Balance each shot before this LUT where your editor allows it.",
            [f"Input: {n['camera_name']} {n['profile_name']}, as recorded", f"Output: {out_name}",
             "Premiere Pro: Lumetri, Basic Correction, Input LUT. Final Cut Pro: Custom LUT effect. Resolve: on the clip, with no other conversion."])
        add(n["input"], lut_input(proj, clip, max(size, 33)),
            f"DaVinci Resolve, first node of {n['camera_name']} {n['profile_name']} clips. Same job as the input conversion node of the recipe.",
            [f"Input: {n['camera_name']} {n['profile_name']}, as recorded", "Output: DaVinci Wide Gamut / DaVinci Intermediate"])
    lines = [f"LUTs for the look '{look.get('name', 'look')}' at {strength:g}% strength, {size} points per side.", ""]
    for f in files:
        lines += [f["name"], "  " + f["where"], ""]
    lines += ["The LUTs hold the shared look. They hold no per-shot balance, vignette or grain.",
              "Rebuild the LUTs after you change the look or its strength.",
              "",
              "Resolve: copy the files into the LUT folder (Project Settings, Color Management, Open LUT Folder), then press Update Lists.",
              "Full data levels: the camera LUTs expect the clip as recorded. For log clips set Clip Attributes, Data Levels to Full when the recipe says so."]
    (folder / "WHERE_THE_LUTS_GO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"folder": str(folder), "files": files, "size": size}


# ------------------------------------------------------------------- video
_filters: dict[str, bool] = {}


def denoise_filter(amount: float, width: int) -> str:
    """ffmpeg noise reduction for the camera signal. amount 0..1.

    A wavelet denoiser: it removes noise and keeps edges and texture with no block grid. The block-based FFT
    filter used before left a 16-pixel grid in fine foliage once a grade raised saturation (measured: edges at
    block borders 1.4 times stronger in brightness, 1.6 times in color; the wavelet filter 1.0). It also smooths
    the color blocks of 8-bit 4:2:0 files. The same chain runs in the viewer, the still and the export.
    """
    if amount <= 0:
        return ""
    if has_filter("vaguedenoiser"):
        return f"vaguedenoiser=threshold={2.0 + 14.0 * amount:.2f}:nsteps=5"
    if has_filter("fftdnoiz"):   # older ffmpeg: more overlap keeps the block grid near invisible
        return f"fftdnoiz=sigma={2.0 + 12.0 * amount:.2f}:amount=1:block=32:overlap=0.7"
    if has_filter("hqdn3d"):
        return f"hqdn3d={1.5 + 5.0 * amount:.2f}:{3.0 + 6.0 * amount:.2f}:{3.0 + 10.0 * amount:.2f}:{4.0 + 10.0 * amount:.2f}"
    return ""


def denoise_chain(amount: float, width: int, info: dict[str, Any]) -> str:
    """The noise reduction as an ffmpeg filter chain for this source, or "" when it does not apply."""
    if amount <= 0 or info.get("is_rgb") or not str(info.get("pix_fmt", "")).startswith("yuv"):
        return ""
    chain = denoise_filter(amount, width)
    if not chain:
        return ""
    # Averaging noise creates in-between values. A 16-bit buffer keeps them, so the cleaned picture holds
    # smoother gradients than the source file: 8-bit skies come out without steps.
    base = re.match(r"yuvj?(4\d\d)p", info["pix_fmt"])
    if base and info.get("bit_depth", 8) < 16:
        chain = f"format=yuv{base.group(1)}p16le,{chain}"
    return chain


def has_filter(name: str) -> bool:
    if name not in _filters:
        r = media.run([media.find_binary("ffmpeg"), "-hide_banner", "-filters"])
        text = r.stdout.decode(errors="replace")
        for f in ("zscale", "lut3d", "blend", "noise", "gblur", "mergeplanes", "hqdn3d", "fftdnoiz", "vaguedenoiser"):
            _filters[f] = bool(re.search(rf"\s{f}\s", text))
    return _filters.get(name, False)


_grain_stats: dict[float, tuple[int, float]] = {}


def _to_float_rgb() -> str:
    """Filter text that turns integer RGB planes into float RGB without touching the values."""
    return "zscale=rangein=full:range=full,format=gbrpf32le" if has_filter("zscale") else "format=gbrpf32le"


GRAIN_FRAMES = 17


def _grain_chain(w: int, h: int, fps: float, sigma: float, offset: int = 0, seed: int = 7, loop: bool = False) -> str:
    """Monochrome film grain source: temporal noise, slightly blurred, the same in all three channels.

    The gray plane goes into the RGB planes with mergeplanes. A direct gray to float RGB
    conversion gives wrong values in some ffmpeg builds.
    """
    lut = f",lut=y=val{offset:+d}" if offset else ""
    # The noise generator is slow at 4K. A short run of grain frames repeats instead: with fine grain
    # nobody sees a pattern that returns every GRAIN_FRAMES frames.
    cycle = f",trim=end_frame={GRAIN_FRAMES},loop=loop=-1:size={GRAIN_FRAMES}:start=0,setpts=N/({fps:.6f}*TB)" if loop else ""
    return (f"color=c=gray:s={w}x{h}:r={fps:.6f},format=gray,noise=alls=40:allf=t:all_seed={seed},gblur=sigma={sigma:.3f}{lut}{cycle},"
            f"mergeplanes=0x000000:gbrp,{_to_float_rgb()}")


def grain_source_stats(sigma: float) -> tuple[int, float]:
    """Measure ffmpeg's grain source once per blur size.

    Returns the 8-bit offset that centers it on 0.5 and its standard deviation (0..1 scale).
    Raises ExportError when this ffmpeg build produces an unusable source.
    """
    key = round(sigma, 3)
    if key not in _grain_stats:
        def measure(offset: int) -> tuple[float, float]:
            r = media.run([media.find_binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", _grain_chain(256, 256, 25.0, sigma, offset),
                           "-frames:v", "4", "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "pipe:1"])
            arr = np.frombuffer(r.stdout, dtype="<f4").astype(np.float64)
            if arr.size != 4 * 3 * 256 * 256 or not np.isfinite(arr).all():
                raise ExportError("This ffmpeg build cannot make the grain layer. Switch grain off for the export.")
            return float(arr.mean()), float(arr.std())

        mean, std = measure(0)
        offset = int(round((0.5 - mean) * 255.0))
        if offset:
            mean, std = measure(offset)
        if not (0.45 < mean < 0.55 and 0.004 < std < 0.3):
            raise ExportError("This ffmpeg build makes an unusable grain layer. Switch grain off for the export.")
        _grain_stats[key] = (offset, std)
    return _grain_stats[key]


def _resolve_format(fmt: str, info: dict[str, Any]) -> dict[str, Any]:
    if fmt not in FORMATS or fmt == "auto":
        # 10-bit for 8-bit sources too: the grade stretches the log signal and noise reduction fills in new levels.
        # 8-bit H.264 threw them away (measured on 8-bit S-Log3 fog: twice the noise of 10-bit and visible steps).
        fmt = "h265-10bit-mp4"
    return dict(FORMATS[fmt], id=fmt)


def render_clip(proj, clip_id: str, out_path: Path | None = None, fmt: str | None = None, max_width: int | None = None,
                progress: Callable[[float, str], None] | None = None, stage: str = "graded", preview: bool = False) -> dict[str, Any]:
    """Render one clip through the grade. preview=True makes a small H.264 file for playback in the app.
    When the color rebuild fails, the clip renders again without it, and the result's note says why."""
    clip = proj.clip(clip_id)
    rebuild = stage != "source" and not clip["info"]["is_rgb"] and proj.color_rebuild_on(clip)
    try:
        return _render_clip(proj, clip_id, out_path, fmt, max_width, progress, stage, preview, rebuild)
    except ExportError as exc:
        if not rebuild:
            raise
        (progress or (lambda f, m: None))(0.03, "The color rebuild failed: rendering again without it")
        res = _render_clip(proj, clip_id, out_path, fmt, max_width, progress, stage, preview, False)
        res["note"] = f"Exported without the color rebuild, which failed: {exc}"
        return res


def _render_clip(proj, clip_id: str, out_path: Path | None, fmt: str | None, max_width: int | None,
                 progress: Callable[[float, str], None] | None, stage: str, preview: bool, rebuild: bool) -> dict[str, Any]:
    say = progress or (lambda f, m: None)
    clip = proj.clip(clip_id)
    info = clip["info"]
    if info["is_image"]:
        raise ExportError("Stills have no clip export.")
    st = settings.load()
    shots = proj.shots_of(clip_id)
    if not shots:
        raise ExportError("Analyze the clip first.")
    f = _resolve_format("h264-8bit-mp4" if preview else (fmt or st["export_format"]), info)
    work = proj.cache_dir(clip_id) / ("preview" if preview else "export")
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("*.cube"):
        old.unlink()
    if out_path is None:
        out_path = (work / f"{stage}.mp4") if preview else export_dir() / f"{_safe(Path(clip['name']).stem)}_graded{f['ext']}"
    out_path = Path(out_path)
    finish = proj.state["finish"] if stage == "graded" else model.neutral_finish()
    vig = finish["vignette"] if finish["vignette"].get("enabled") else None
    grain = finish["grain"] if finish["grain"].get("enabled") else None
    rl_on, rl_amount, rl_subject = relight_mod.settings(finish)
    tracks = {}
    if rl_on and stage == "graded" and (rl_amount or rl_subject):
        for s in shots:
            tr = proj.face_track(s["id"])
            if any(f for _, f in tr):
                tracks[s["id"]] = tr
    relit = bool(tracks)
    two_stage = bool(vig or grain or relit)
    lut_size = 33 if preview else 65
    use_z = has_filter("zscale")
    say(0.02, "Building the grade tables")
    # Output encoding of the last LUT: the form that follows the deepest shadows best (see _encode).
    enc = "display"
    if use_z and stage != "source":
        enc = best_encoding(proj.odt().signed if two_stage else camera_function(proj, clip, None, "display", with_look=False), lut_size)
    lut_files = []
    for s in shots:
        target = "working" if two_stage else "display"
        if stage == "source":
            lut = None
        elif stage == "converted":
            lut = lut_camera(proj, clip, lut_size, None, target, with_look=False, encoding=enc)
        else:
            lut = lut_camera(proj, clip, lut_size, s, target, with_look=(stage == "graded"), encoding=enc)
        if lut is not None:
            p = write_cube(work / f"shot_{s['index']:03d}.cube", lut)
            lut_files.append((s, p))
    lut_b = write_cube(work / "output.cube", lut_output(proj, lut_size, enc)) if two_stage else None

    w, h = info["display_width"], info["display_height"]
    if max_width and w > max_width:
        h = int(round(h * max_width / w / 2)) * 2
        w = int(round(max_width / 2)) * 2
    fps = info["fps"] or 25.0
    matrix = {"bt709": "709", "bt2020nc": "2020_ncl", "bt2020c": "2020_cl", "bt601": "170m", "smpte170m": "170m", "bt470bg": "470bg"}[info["matrix"]]
    sws_matrix = {"bt709": "bt709", "bt2020nc": "bt2020", "bt2020c": "bt2020", "bt601": "smpte170m", "smpte170m": "smpte170m", "bt470bg": "bt470bg"}[info["matrix"]]
    idt_full = proj.idt(clip)[0].full_range if clip.get("profile_id") else info["full_range"]
    rng_z, rng_s = ("full", "pc") if idt_full else ("limited", "tv")
    dn, sharp, clar = pipeline.detail_values(finish)
    chain = []
    scale_first = bool(max_width and info["display_width"] > max_width)
    denoise = denoise_chain(dn, info["display_width"] if not preview else w, info)
    # Blocky compressed color gets rebuilt from the brightness by the app's own code (color/chroma.py), the same code
    # the viewer and the still run: ffmpeg decodes, the app rebuilds each frame, ffmpeg grades and encodes.
    pre = []
    if denoise and not preview:
        pre.append(denoise)   # noise reduction on the camera signal, before any conversion
    if scale_first or rebuild:
        flags = "lanczos+accurate_rnd+full_chroma_int" if rebuild else "bicubic"
        pre.append(f"scale={w}:{h}:flags={flags}:in_range={rng_s}:out_range={rng_s}")
    if denoise and preview:
        pre.append(denoise)   # the small playback file cleans after scaling: same look, a fraction of the time
    if not rebuild:
        chain += pre
    if info["is_rgb"]:
        chain.append("format=gbrpf32le")
    elif use_z:
        chain.append(f"zscale=matrixin={matrix}:rangein={rng_z}:range=full,format=gbrpf32le")
    else:
        chain.append(f"scale=in_color_matrix={sws_matrix}:in_range={rng_s}:out_range=pc:flags=accurate_rnd+full_chroma_int+bicubic,format=gbrpf32le")
    half = 0.5 / fps
    for i, (s, p) in enumerate(lut_files):
        cond = ""
        if len(lut_files) > 1:
            a = 0.0 if i == 0 else s["start"] - half
            b = 1e9 if i == len(lut_files) - 1 else s["end"] - half
            cond = f":enable='gte(t,{a:.5f})*lt(t,{b:.5f})'"
        chain.append(f"lut3d=file='{_ff_path(p)}':interp=tetrahedral{cond}")
    graph = f"[0:v]{','.join(chain)}[g0]"
    last = "g0"
    decode_cmd: list[str] = []
    if rebuild:   # a short queue: each raw 4K frame is 50 MB
        inputs = ["-thread_queue_size", "4", "-f", "rawvideo", "-pix_fmt", "yuv444p16le", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "pipe:0"]
        decode_cmd = _decoder_command(info["path"], pre)
    else:
        inputs = ["-i", info["path"]]
    n_in = 1
    rl_file = None
    if relit:
        # The relight follows the faces, so its mask moves: one small gray frame per video frame, scaled up
        # in ffmpeg. Stored as 0.5 + shift / 2K with opacity 2K: grainmerge gives A + shift, in stops of DI.
        k = W.STOP * max(abs(rl_amount), abs(rl_subject), 1e-3)
        mw = 256
        mh = max(2, int(round(mw * h / w / 2)) * 2)
        n_frames = int(round(info["duration"] * fps)) + 2
        rl_file = work / "relight_mask.raw"
        say(0.03, "Tracking faces for the relight")
        with open(rl_file, "wb") as fh:
            for fi in range(n_frames):
                t = fi / fps
                s = next((x for x in shots if x["start"] - 0.5 / fps <= t < x["end"] - 0.5 / fps), shots[-1])
                m = relight_mod.mask(relight_mod.faces_at(tracks[s["id"]], t), mw, mh) if s["id"] in tracks else None
                if m is None:
                    frame = np.full((mh, mw), 128, dtype=np.uint8)
                else:
                    shift = W.STOP * (rl_subject * m - rl_amount * (1.0 - m))
                    frame = np.clip(np.round((0.5 + 0.5 * shift / k) * 255.0), 0, 255).astype(np.uint8)
                fh.write(frame.tobytes())
        inputs += ["-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{mw}x{mh}", "-r", f"{fps:.6f}", "-i", str(rl_file)]
        graph += (f";[{n_in}:v]scale={w}:{h}:flags=bicubic,format=gray,mergeplanes=0x000000:gbrp,{_to_float_rgb()},"
                  f"setpts=N/({fps:.6f}*TB)[rm];[{last}][rm]blend=all_mode=grainmerge:all_opacity={min(2.0 * k, 1.0):.6f}:shortest=1[gr0]")
        last, n_in = "gr0", n_in + 1
    if vig:
        # ffmpeg's subtract blend clamps at zero. grainmerge computes A + (B - 0.5) * opacity with no clamp,
        # so the mask is stored as 0.5 - mask / 2 and the opacity doubles: A - mask * k.
        mask = pipeline.vignette_mask(h, w, float(vig.get("start", 0.45)))
        mask16 = np.round((0.5 - 0.5 * mask) * 65535.0).astype(np.uint16)
        import cv2

        mpath = work / "vignette.png"
        cv2.imwrite(str(mpath), np.repeat(mask16[..., None], 3, axis=2))  # 16-bit RGB: converts to float exactly
        inputs += ["-i", str(mpath)]
        k = float(vig["amount"]) * W.STOP
        # the mask decodes and converts once, then the loop filter repeats this one frame
        graph += (f";[{n_in}:v]format=gbrp16le,{_to_float_rgb()},loop=loop=-1:size=1:start=0,setpts=N/({fps:.6f}*TB)[vm];"
                  f"[{last}][vm]blend=all_mode=grainmerge:all_opacity={min(2.0 * k, 1.0):.6f}:shortest=1[g1]")
        last, n_in = "g1", n_in + 1
    if grain:
        sigma = 0.55 * max(0.3, float(grain.get("size", 1.0))) * max(1.0, w / 1920.0)
        offset, std = grain_source_stats(sigma)
        op = (float(grain["amount"]) / 100.0) / std
        graph += (f";{_grain_chain(w, h, fps, sigma, offset, loop=True)}[gr];"
                  f"[{last}][gr]blend=all_mode=grainmerge:all_opacity={min(op, 1.0):.6f}:shortest=1[g2]")
        last = "g2"
    tail = []
    if lut_b is not None:
        tail.append(f"lut3d=file='{_ff_path(lut_b)}':interp=tetrahedral")
    trc = W.OUTPUT_SPACES[st["output_space"]]["trc_tag"]
    def to_yuv(pix: str, dither: bool) -> str:
        if use_z:
            undo = "transferin=linear:transfer=bt470m:primariesin=709:primaries=709:" if enc == "z22" else ""  # exact power 1/2.2
            return f"zscale={undo}matrix=709:range=limited{':dither=error_diffusion' if dither else ''},format={pix}"
        return f"scale=out_color_matrix=bt709:out_range=tv{':sws_dither=ed' if dither else ''}:flags=accurate_rnd+full_chroma_int,format={pix}"

    def deliver(pix: str) -> str:
        """16-bit 4:4:4 to the delivery format, dithered once, at the very end."""
        if use_z:
            return f"zscale=matrixin=709:matrix=709:rangein=limited:range=limited:dither=error_diffusion,format={pix}"
        return f"scale=in_range=tv:out_range=tv:sws_dither=ed:flags=accurate_rnd+full_chroma_int,format={pix}"

    params = f"setparams=color_primaries=bt709:color_trc={trc}:colorspace=bt709:range=tv"
    steps = []
    s_sharp, s_clar = pipeline.detail_sigmas(w)
    if clar > 0:
        steps.append((s_clar, pipeline.clarity_gain(clar)))
    if sharp > 0:
        steps.append((s_sharp, pipeline.sharpen_gain(sharp)))
    if steps and has_filter("gblur"):
        # Unsharp masks on the finished luma: Y + k * (Y - blur). grainextract gives Y - blur + half,
        # grainmerge at opacity k adds k * (difference). Chroma passes through untouched.
        # They work on 16-bit 4:4:4: sharpening after the 8-bit conversion amplified its steps and its dither
        # noise (measured on 8-bit S-Log3 fog: nine times the noise of the full-quality still).
        tail.append(to_yuv("yuv444p16le", False))
        graph += f";[{last}]{','.join(tail)}[d0]"
        for i, (sigma, k) in enumerate(steps):
            a, b_, c, blur, diff, nxt = (f"d{i}{t}" for t in ("a", "b", "c", "blur", "diff", "n"))
            graph += (f";[d{i}]split=3[{a}][{b_}][{c}];[{b_}]gblur=sigma={sigma:.3f}:steps=2:planes=1[{blur}];"
                      f"[{a}][{blur}]blend=c0_mode=grainextract:c1_mode=normal:c2_mode=normal[{diff}];"
                      f"[{c}][{diff}]blend=c0_mode=grainmerge:c0_opacity={min(k, 1.0):.4f}:c1_mode=normal:c2_mode=normal[d{i + 1}]")
        graph += f";[d{len(steps)}]{deliver(f['pix'])},{params}[out]"
    else:
        tail += [to_yuv(f["pix"], True), params]
        graph += f";[{last}]{','.join(tail)}[out]"
    src_in = 0
    if rebuild:   # sound and metadata come from the original file
        src_in = n_in
        inputs += ["-i", info["path"]]
    cmd = [media.find_binary("ffmpeg"), "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", graph, "-map", "[out]"]
    codec = list(f["codec"])
    if f.get("x265") and w * h > 2560 * 1440:
        # Above 1440p the fast preset, with the texture settings below, keeps as much fine detail as medium and
        # takes 35% less time. Measured on the 4K 8-bit footage that needed the color rebuild: SSIM 0.960 against
        # 0.961, fine detail kept 81 to 85% against 77 to 82%.
        codec[codec.index("-preset") + 1] = "fast"
    if preview:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    cmd += codec + ["-pix_fmt", f["pix"], "-color_primaries", "bt709", "-color_trc", trc, "-colorspace", "bt709", "-color_range", "tv"]
    if f.get("x265") and not preview:
        # texture first: no SAO smoothing, stronger psycho-visual tuning, adaptive quantization that keeps detail in
        # textured areas, lighter deblocking. Measured on 4K knit and leaves: 95% of the still's fine texture, 92% before.
        cmd += ["-x265-params", f"log-level=error:colorprim=bt709:transfer={trc}:colormatrix=bt709:range=limited:"
                                "no-sao=1:psy-rd=2.0:psy-rdoq=2.0:aq-mode=3:deblock=-1,-1"]
    if info["has_audio"]:
        cmd += ["-map", f"{src_in}:a:0?"]
        if f["ext"] == ".mp4" or preview:
            cmd += (["-c:a", "copy"] if info["audio_codec"] == "aac" else ["-c:a", "aac", "-b:a", "256k"])
        else:
            cmd += (["-c:a", "copy"] if info["audio_codec"].startswith("pcm") else ["-c:a", "pcm_s24le"])
    cmd += ["-map_metadata", str(src_in), "-movflags", "+faststart+write_colr", "-progress", "pipe:1", "-nostats", str(out_path)]
    t0 = time.time()
    total = max(1, int(info.get("frames") or 0) or int(round(info["duration"] * fps)))
    log_path, dec_log = work / "ffmpeg.log", work / "decode.log"
    watch = _Watch(total)
    feed: _Feed | None = None
    with open(log_path, "wb") as log:
        log.write(_command_text(cmd, decode_cmd))
        log.flush()
        head = log.tell()
        say(0.03, "Rendering preview" if preview else "Rendering graded clip")
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if rebuild else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log,
                                **media._popen_kwargs())
        try:
            if rebuild:
                feed = _Feed(decode_cmd, proc, w, h, w / float(info["display_width"]), chroma.subsampling(info), int(info.get("bit_depth", 8)), dec_log, watch)
                feed.start()
            threading.Thread(target=watch.guard, args=(proc, feed), daemon=True, name="export-watch").start()
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.startswith(b"frame="):
                    try:
                        n = int(line[6:])
                    except ValueError:
                        continue
                    if n > watch.encoded:
                        watch.frame(n)
                        say(min(0.99, 0.03 + 0.96 * n / total), watch.text())
            rc = proc.wait()
        finally:
            if proc.poll() is None:   # an error in this thread: never leave ffmpeg running
                proc.kill()
                rc = proc.wait()
            if feed is not None:
                feed.finish(30)
    if rl_file is not None:   # after ffmpeg exits: Windows keeps an open file locked
        rl_file.unlink(missing_ok=True)
    if watch.stalled:
        raise ExportError(f"The export stopped making progress: no new frame for {STALL_SECONDS / 60:.0f} minutes, at frame "
                          f"{watch.encoded} of {total}. Details: {log_path}")
    if rc != 0:
        msg = _failure_text(_tail(log_path, head), out_path) or f"ffmpeg stopped with code {_exit_code(rc)}"
        if feed is not None and feed.error and "broken pipe" not in feed.error.lower():
            msg += f" The color rebuild reported: {feed.error}"
        raise ExportError(f"ffmpeg failed: {msg} Details: {log_path}")
    if feed is not None and feed.error:
        raise ExportError(f"Rebuilding the color failed: {feed.error} Details: {dec_log}")
    _check_output(out_path, info, log_path)
    say(1.0, "Done")
    return {"path": str(out_path), "format": f["id"], "format_name": f["name"], "seconds": round(time.time() - t0, 1),
            "width": w, "height": h, "pix_fmt": f["pix"], "tags": {"primaries": "bt709", "transfer": trc, "matrix": "bt709", "range": "tv"},
            "color_rebuild": rebuild, "log": str(log_path)}


STALL_SECONDS = 300.0   # five minutes without one more frame decoded, rebuilt or encoded: the export has stopped


def _decoder_command(path: str, pre: list[str]) -> list[str]:
    """ffmpeg decoding the source to 16-bit 4:4:4 for the color rebuild, through the steps that come before it.
    Two decode and two filter threads and a two-frame output queue: 4K still decodes at 27 frames a second, ten times
    what the export needs, with 670 MB instead of 1.26 GB (more on computers with many cores)."""
    return [media.find_binary("ffmpeg"), "-v", "error", "-nostdin", "-threads", "2", "-i", path, "-an", "-sn",
            "-filter_threads", "2", "-vf", ",".join(pre + ["format=yuv444p16le"]), "-f", "rawvideo", "-pix_fmt", "yuv444p16le",
            "-thread_queue_size", "2", "pipe:1"]


def _command_text(cmd: list[str], decode_cmd: list[str]) -> bytes:
    lines = [time.strftime("%Y-%m-%d %H:%M:%S"), "encode: " + subprocess.list2cmdline(cmd)]
    if decode_cmd:
        lines.append("decode: " + subprocess.list2cmdline(decode_cmd))
    return ("\n".join(lines) + "\n\n").encode("utf-8", "replace")


def _tail(path: Path, start: int = 0, chars: int = 700) -> str:
    """The last lines ffmpeg wrote, after the command header."""
    try:
        text = path.read_bytes()[start:].decode("utf-8", "replace").strip()
    except OSError:
        return ""
    return text[-chars:]


def _exit_code(rc: int) -> str:
    return hex(rc & 0xFFFFFFFF) if rc < 0 or rc > 255 else str(rc)   # Windows crash codes read as 0xC0000005


def _failure_text(text: str, out_path: Path) -> str:
    """Plain words for the failures people meet, ffmpeg's own text for the rest."""
    low = text.lower()
    if "permission denied" in low:
        return f"Windows did not let ffmpeg write {out_path.name}. Close any player or editor that has it open, then export again."
    if "no space left" in low or "not enough space" in low or "disk full" in low:
        return f"The disk that holds {out_path.parent} is full. Free some space, then export again."
    if "cannot allocate memory" in low or "out of memory" in low or "not enough memory" in low:
        return "Your computer ran out of memory during the export. Close other programs, then export again."
    return text


def _check_output(out_path: Path, info: dict[str, Any], log_path: Path) -> None:
    """The file exists, plays and holds the whole clip."""
    try:
        got = media.probe(out_path)
    except Exception as exc:   # noqa: BLE001 - any reason the file does not read is a failed export
        raise ExportError(f"The export wrote no readable file ({exc}). Details: {log_path}") from exc
    want = float(info["duration"])
    if want > 1.0 and got["duration"] < want - max(0.5, 0.02 * want):
        raise ExportError(f"The export stopped early: {got['duration']:.1f} of {want:.1f} seconds. Details: {log_path}")


def _left(seconds: float) -> str:
    if seconds < 60:
        return "under a minute left"
    if seconds < 5400:
        return f"about {int(round(seconds / 60))} min left"
    return f"about {seconds / 3600:.1f} h left"


class _Watch:
    """Progress of one render: frames done, time left, and a guard that stops an export that stopped moving."""

    def __init__(self, total: int, clock: Callable[[], float] = time.monotonic):
        self.total, self.encoded, self.stalled, self.clock = total, 0, False, clock
        self.last = clock()
        self.first: tuple[int, float] | None = None   # frame and time of the first progress report

    def tick(self) -> None:
        self.last = self.clock()

    def frame(self, n: int) -> None:
        self.encoded = n
        self.tick()
        if self.first is None:
            self.first = (n, self.last)

    def text(self) -> str:
        n = self.encoded
        msg = f"Frame {n} of {self.total}"
        if self.first is not None:
            n0, t0 = self.first
            dt = self.clock() - t0
            if n - n0 >= max(3, 0.02 * self.total) and dt >= 10:
                msg += " · " + _left((self.total - n) * dt / (n - n0))
        return msg

    def guard(self, proc: subprocess.Popen, feed: "_Feed | None") -> None:
        while proc.poll() is None:
            time.sleep(1.0)
            if self.encoded >= 0.98 * self.total:   # the last frames are in: finishing the file takes its time
                continue
            if self.clock() - self.last > STALL_SECONDS:
                self.stalled = True
                if feed is not None:
                    feed.kill()
                try:
                    proc.kill()
                except OSError:
                    pass
                return


def _read_frame(stream, n_values: int) -> np.ndarray | None:
    """One frame of 16-bit values, read straight into a new array. None at the end of the stream."""
    frame = np.empty(n_values, dtype="<u2")
    view = memoryview(frame).cast("B")
    got = 0
    while got < len(view):
        k = stream.readinto(view[got:])
        if not k:
            if got == 0:
                return None
            raise ExportError(f"the decoder stopped inside a frame ({got} of {len(view)} bytes)")
        got += k
    return frame


class _Feed:
    """Decodes with ffmpeg, rebuilds each frame's color, writes it to the encoder. Frames keep their order.
    At 4K each frame is 50 MB raw and the rebuild needs about 160 MB more while it works: two workers and one frame
    waiting keep the whole export under the memory a plain export takes plus about 1 GB."""

    def __init__(self, decode_cmd: list[str], enc: subprocess.Popen, w: int, h: int, scale: float, sub: tuple[int, int], bits: int, log_path: Path,
                 watch: _Watch):
        self.cmd, self.enc, self.w, self.h, self.scale, self.sub, self.bits = decode_cmd, enc, w, h, scale, sub, bits
        self.log_path, self.watch = log_path, watch
        self.dec: subprocess.Popen | None = None
        self.frames = 0
        self.error: str | None = None
        self.done = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="export-feed").start()

    def kill(self) -> None:
        dec = self.dec
        if dec is not None and dec.poll() is None:
            try:
                dec.kill()
            except OSError:
                pass

    def finish(self, timeout: float) -> None:
        if not self.done.wait(timeout):
            self.kill()
            self.done.wait(5)

    def _write(self, out: np.ndarray) -> None:
        self.enc.stdin.write(memoryview(out).cast("B"))
        self.frames += 1
        self.watch.tick()

    def _run(self) -> None:
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        try:
            with open(self.log_path, "wb") as log:
                self.dec = subprocess.Popen(self.cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=log, **media._popen_kwargs())
                workers = max(1, min(pipeline.render_threads(), 2 if self.w * self.h > 4_000_000 else 4))
                with ThreadPoolExecutor(workers, thread_name_prefix="rebuild") as pool:
                    pending: deque = deque()
                    while True:
                        frame = _read_frame(self.dec.stdout, self.w * self.h * 3)
                        if frame is None:
                            break
                        self.watch.tick()
                        pending.append(pool.submit(chroma.rebuild, frame, self.w, self.h, self.scale, self.sub, self.bits))
                        while len(pending) > workers:
                            self._write(pending.popleft().result())
                    while pending:
                        self._write(pending.popleft().result())
                rc = self.dec.wait()
            if rc != 0:
                self.error = f"decoding the source failed (code {_exit_code(rc)}): {_tail(self.log_path)}"
            elif self.frames == 0:
                self.error = f"decoding the source gave no frames. {_tail(self.log_path)}".strip()
        except BaseException as exc:   # a broken pipe when the encoder stopped, a decode failure, memory
            if isinstance(exc, MemoryError):
                self.error = "the computer ran out of memory"
            else:
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self.enc.stdin.close()
            except OSError:
                pass
            self.kill()
            if self.dec is not None:
                try:
                    self.dec.wait(10)
                except Exception:   # noqa: BLE001
                    pass
            self.done.set()


def render_clips(proj, clip_ids: list[str] | None = None, fmt: str | None = None,
                 progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
    """Export several clips, one file each."""
    say = progress or (lambda f, m: None)
    todo = [c for c in proj.state["clips"] if not c["info"]["is_image"] and (not clip_ids or c["id"] in clip_ids)]
    if not todo:
        raise ExportError("No video clips to export.")
    total = sum(max(c["info"]["duration"], 0.1) for c in todo)
    done, files, used = 0.0, [], set()
    for c in todo:
        share = max(c["info"]["duration"], 0.1) / total
        f = _resolve_format(fmt or settings.load()["export_format"], c["info"])
        stem = _safe(Path(c["name"]).stem) + "_graded"
        name, n = stem, 2
        while name in used:   # two clips with the same file name in different folders
            name, n = f"{stem}_{n}", n + 1
        used.add(name)
        out = export_dir() / f"{name}{f['ext']}"
        r = render_clip(proj, c["id"], out, fmt, progress=lambda fr, m, d=done, sh=share, nm=c["name"]: say(d + sh * fr, f"{nm} · {m}"))
        r["clip_id"], r["name"] = c["id"], Path(r["path"]).name
        files.append(r)
        done += share
    say(1.0, "Done")
    return {"folder": str(export_dir()), "files": files}


# ----------------------------------------------------------- graded playback
def _preview_signature(proj, clip_id: str, stage: str) -> str:
    import hashlib
    import json

    st = settings.load()
    clip = proj.clip(clip_id)
    core = [stage, proj.state["look"] if stage == "graded" else None, proj.state["strength"] if stage == "graded" else None,
            proj.state["finish"] if stage == "graded" else None,
            [(s["start"], s["end"], s.get("grade")) for s in proj.shots_of(clip_id)], clip.get("camera_id"), clip.get("profile_id"),
            proj.profile_signature(clip), st["output_space"], st["max_input_nits"], PREVIEW_WIDTH, proj.color_rebuild_on(clip)]
    return hashlib.sha1(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


PREVIEW_WIDTH = 960


def preview_file(proj, clip_id: str, stage: str = "graded") -> Path:
    return proj.cache_dir(clip_id) / "preview" / f"{stage}.mp4"


def preview_ready(proj, clip_id: str, stage: str = "graded") -> bool:
    f = preview_file(proj, clip_id, stage)
    sig = f.with_suffix(".sig")
    try:
        return f.exists() and f.stat().st_size > 0 and sig.read_text() == _preview_signature(proj, clip_id, stage)
    except OSError:
        return False


def build_preview(proj, clip_id: str, stage: str = "graded", progress: Callable[[float, str], None] | None = None) -> dict[str, Any]:
    """Small H.264 file of the clip at one stage, for playback in the app."""
    sig = _preview_signature(proj, clip_id, stage)
    f = preview_file(proj, clip_id, stage)
    if not preview_ready(proj, clip_id, stage):
        f.with_suffix(".sig").unlink(missing_ok=True)
        render_clip(proj, clip_id, f, max_width=PREVIEW_WIDTH, progress=progress, stage=stage, preview=True)
        f.with_suffix(".sig").write_text(sig)
    return {"clip_id": clip_id, "stage": stage, "signature": sig}


def _ff_path(p: Path) -> str:
    """Escape a path for use inside an ffmpeg filter option."""
    s = str(p).replace("\\", "/")
    return s.replace(":", "\\:").replace("'", "\\'")
