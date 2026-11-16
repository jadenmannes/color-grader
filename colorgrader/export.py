"""LUT export and graded clip export.

LUTs carry the grade into any editor. The graded clip renders with ffmpeg:
decode at full bit depth, convert to R'G'B' in float, apply the grade as
65-point LUTs with tetrahedral interpolation, add vignette and grain in the
working space, convert to Rec.709, dither, encode and tag.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import media, paths, settings
from .color import working as W
from .color.lut import Lut3D, write_cube
from .grade import model, pipeline

F32 = np.float32

FORMATS: dict[str, dict[str, Any]] = {
    "auto": {"name": "Auto: H.265 10-bit for 10-bit sources, H.264 for 8-bit"},
    "h265-10bit-mp4": {"name": "H.265 10-bit 4:2:0 (MP4)", "ext": ".mp4", "pix": "yuv420p10le",
                       "codec": ["-c:v", "libx265", "-preset", "medium", "-crf", "16", "-tag:v", "hvc1"], "x265": True},
    "h264-8bit-mp4": {"name": "H.264 8-bit 4:2:0 (MP4)", "ext": ".mp4", "pix": "yuv420p",
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
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "clip"


# -------------------------------------------------------------------- LUTs
def _look_now(proj) -> dict[str, Any]:
    return model.at_strength(proj.state["look"], proj.state["strength"] / 100.0)


def lut_look(proj, size: int) -> Lut3D:
    """Look only: working space in, working space out."""
    look = _look_now(proj)
    return Lut3D.from_function(lambda v: np.clip(pipeline.look_stage(v, look), 0.0, 1.0), size, title="Look, DaVinci WG/Intermediate")


def lut_output(proj, size: int) -> Lut3D:
    odt = proj.odt()
    return Lut3D.from_function(lambda v: np.clip(odt(v), 0.0, 1.0), size, title="Output: DaVinci WG/Intermediate to Rec.709")


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


def lut_camera(proj, clip: dict[str, Any], size: int, shot: dict[str, Any] | None = None, to: str = "display",
               with_look: bool = True) -> Lut3D:
    """Camera signal through the grade. shot=None leaves the per-shot balance out (a general camera LUT).

    to: "display" ends in Rec.709, "working" stops before finish and output (the export pipeline).
    """
    idt, prof = proj.idt(clip)
    corr, cam_sat = prof.get("corrections"), pipeline.camera_saturation(prof.get("corrections"))
    grade = model.quantize_shot(shot["grade"]) if shot else model.neutral_shot()
    look, odt = (_look_now(proj) if with_look else None), proj.odt()

    def fn(v):
        di = pipeline.shot_stage(pipeline.camera_stage(v, idt, corr), grade, cam_sat)
        if look is not None:
            di = pipeline.look_stage(di, look)
        return np.clip(di if to == "working" else odt(di), 0.0, 1.0)

    return Lut3D.from_function(fn, size, title="Camera to Rec.709 with look")


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
    files = []

    def add(fname: str, lut: Lut3D, where: str, comments: list[str]):
        path = write_cube(folder / fname, lut, comments=[f"Color Grader: {look.get('name', 'look')} at {strength}%"] + comments)
        files.append({"file": str(path), "name": fname, "where": where})

    add(f"{name}_LOOK_DWG-Intermediate.cube", lut_look(proj, size),
        "DaVinci Resolve, timeline level. Place it after the balance nodes and before the output conversion. It replaces the look nodes of the recipe.",
        ["Input: DaVinci Wide Gamut / DaVinci Intermediate", "Output: DaVinci Wide Gamut / DaVinci Intermediate",
         "Place after input conversion and balance, before the output conversion."])
    add(f"{name}_LOOK+OUTPUT_DWG-Intermediate_to_Rec709.cube", lut_look_output(proj, size),
        f"DaVinci Resolve, last node. It replaces the look nodes and both output conversion nodes. Output is {out_name}.",
        ["Input: DaVinci Wide Gamut / DaVinci Intermediate", f"Output: {out_name}", "Place as the last node, after balance."])
    add("OUTPUT_DWG-Intermediate_to_Rec709.cube", lut_output(proj, size),
        "Optional. The app's output conversion as one LUT, for the last node. Same result as the two output nodes of the recipe.",
        ["Input: DaVinci Wide Gamut / DaVinci Intermediate", f"Output: {out_name}"])
    seen = set()
    for clip in proj.state["clips"]:
        key = (clip.get("camera_id"), clip.get("profile_id"))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        try:
            cam, prof = proj.profile_of(clip)
        except Exception:
            continue
        tag = _safe(f"{cam['name']}_{prof['name'].split(' (')[0]}")
        add(f"{name}_{tag}_to_Rec709.cube", lut_camera(proj, clip, size),
            f"Any editor. Apply to {cam['name']} {prof['name']} clips as they come from the camera. Holds input conversion, look and output. "
            "Balance each shot before this LUT where your editor allows it.",
            [f"Input: {cam['name']} {prof['name']}, as recorded", f"Output: {out_name}",
             "Premiere Pro: Lumetri, Basic Correction, Input LUT. Final Cut Pro: Custom LUT effect. Resolve: on the clip, with no other conversion."])
        add(f"INPUT_{tag}_to_DWG-Intermediate.cube", lut_input(proj, clip, max(size, 33)),
            f"DaVinci Resolve, first node of {cam['name']} {prof['name']} clips. Same job as the input conversion node of the recipe.",
            [f"Input: {cam['name']} {prof['name']}, as recorded", "Output: DaVinci Wide Gamut / DaVinci Intermediate"])
    lines = [f"LUTs for the look '{look.get('name', 'look')}' at {strength}% strength, {size} points per side.", ""]
    for f in files:
        lines += [f["name"], "  " + f["where"], ""]
    lines += ["The LUTs hold the shared look. They hold no per-shot balance, vignette or grain.",
              "Rebuild the LUTs after you change the look or its strength."]
    (folder / "WHERE_THE_LUTS_GO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"folder": str(folder), "files": files, "size": size}


# ------------------------------------------------------------------- video
_filters: dict[str, bool] = {}


def has_filter(name: str) -> bool:
    if name not in _filters:
        r = media.run([media.find_binary("ffmpeg"), "-hide_banner", "-filters"])
        text = r.stdout.decode(errors="replace")
        for f in ("zscale", "lut3d", "blend", "noise", "gblur", "mergeplanes"):
            _filters[f] = bool(re.search(rf"\s{f}\s", text))
    return _filters.get(name, False)


_grain_stats: dict[float, tuple[int, float]] = {}


def _to_float_rgb() -> str:
    """Filter text that turns integer RGB planes into float RGB without touching the values."""
    return "zscale=rangein=full:range=full,format=gbrpf32le" if has_filter("zscale") else "format=gbrpf32le"


def _grain_chain(w: int, h: int, fps: float, sigma: float, offset: int = 0, seed: int = 7) -> str:
    """Monochrome film grain source: temporal noise, slightly blurred, the same in all three channels.

    The gray plane goes into the RGB planes with mergeplanes. A direct gray to float RGB
    conversion gives wrong values in some ffmpeg builds.
    """
    lut = f",lut=y=val{offset:+d}" if offset else ""
    return (f"color=c=gray:s={w}x{h}:r={fps:.6f},format=gray,noise=alls=40:allf=t:all_seed={seed},gblur=sigma={sigma:.3f}{lut},"
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
        fmt = "h265-10bit-mp4" if info["bit_depth"] > 8 else "h264-8bit-mp4"
    return dict(FORMATS[fmt], id=fmt)


def render_clip(proj, clip_id: str, out_path: Path | None = None, fmt: str | None = None, max_width: int | None = None,
                progress: Callable[[float, str], None] | None = None, stage: str = "graded", preview: bool = False) -> dict[str, Any]:
    """Render one clip through the grade. preview=True makes a small H.264 file for playback in the app."""
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
    two_stage = bool(vig or grain)
    lut_size = 33 if preview else 65
    say(0.02, "Building the grade tables")
    lut_files = []
    for s in shots:
        target = "working" if two_stage else "display"
        if stage == "source":
            lut = None
        elif stage == "converted":
            lut = lut_camera(proj, clip, lut_size, None, target, with_look=False)
        else:
            lut = lut_camera(proj, clip, lut_size, s, target, with_look=(stage == "graded"))
        if lut is not None:
            p = write_cube(work / f"shot_{s['index']:03d}.cube", lut)
            lut_files.append((s, p))
    lut_b = write_cube(work / "output.cube", lut_output(proj, lut_size)) if two_stage else None

    w, h = info["display_width"], info["display_height"]
    if max_width and w > max_width:
        h = int(round(h * max_width / w / 2)) * 2
        w = int(round(max_width / 2)) * 2
    fps = info["fps"] or 25.0
    use_z = has_filter("zscale")
    matrix = {"bt709": "709", "bt2020nc": "2020_ncl", "bt2020c": "2020_cl", "bt601": "170m", "smpte170m": "170m", "bt470bg": "470bg"}[info["matrix"]]
    sws_matrix = {"bt709": "bt709", "bt2020nc": "bt2020", "bt2020c": "bt2020", "bt601": "smpte170m", "smpte170m": "smpte170m", "bt470bg": "bt470bg"}[info["matrix"]]
    idt_full = proj.idt(clip)[0].full_range if clip.get("profile_id") else info["full_range"]
    rng_z, rng_s = ("full", "pc") if idt_full else ("limited", "tv")
    chain = []
    if max_width and info["display_width"] > max_width:
        chain.append(f"scale={w}:{h}:flags=bicubic:in_range={rng_s}:out_range={rng_s}")
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
    inputs = ["-i", info["path"]]
    n_in = 1
    if vig:
        # ffmpeg's subtract blend clamps at zero. grainmerge computes A + (B - 0.5) * opacity with no clamp,
        # so the mask is stored as 0.5 - mask / 2 and the opacity doubles: A - mask * k.
        mask = pipeline.vignette_mask(h, w, float(vig.get("start", 0.45)))
        mask16 = np.round((0.5 - 0.5 * mask) * 65535.0).astype(np.uint16)
        import cv2

        mpath = work / "vignette.png"
        cv2.imwrite(str(mpath), np.repeat(mask16[..., None], 3, axis=2))  # 16-bit RGB: converts to float exactly
        inputs += ["-loop", "1", "-framerate", f"{fps:.6f}", "-i", str(mpath)]
        k = float(vig["amount"]) * W.STOP
        graph += (f";[{n_in}:v]format=gbrp16le,{_to_float_rgb()}[vm];"
                  f"[{last}][vm]blend=all_mode=grainmerge:all_opacity={min(2.0 * k, 1.0):.6f}:shortest=1[g1]")
        last, n_in = "g1", n_in + 1
    if grain:
        sigma = 0.55 * max(0.3, float(grain.get("size", 1.0))) * max(1.0, w / 1920.0)
        offset, std = grain_source_stats(sigma)
        op = (float(grain["amount"]) / 100.0) / std
        graph += (f";{_grain_chain(w, h, fps, sigma, offset)}[gr];"
                  f"[{last}][gr]blend=all_mode=grainmerge:all_opacity={min(op, 1.0):.6f}:shortest=1[g2]")
        last = "g2"
    tail = []
    if lut_b is not None:
        tail.append(f"lut3d=file='{_ff_path(lut_b)}':interp=tetrahedral")
    if use_z:
        tail.append(f"zscale=matrix=709:range=limited:dither=error_diffusion,format={f['pix']}")
    else:
        tail.append(f"scale=out_color_matrix=bt709:out_range=tv:sws_dither=ed:flags=accurate_rnd+full_chroma_int,format={f['pix']}")
    graph += f";[{last}]{','.join(tail)}[out]"
    trc = W.OUTPUT_SPACES[st["output_space"]]["trc_tag"]
    cmd = [media.find_binary("ffmpeg"), "-v", "error", "-nostdin", "-y", *inputs, "-filter_complex", graph, "-map", "[out]"]
    codec = list(f["codec"])
    if preview:
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    cmd += codec + ["-pix_fmt", f["pix"], "-color_primaries", "bt709", "-color_trc", trc, "-colorspace", "bt709", "-color_range", "tv"]
    if f.get("x265") and not preview:
        cmd += ["-x265-params", f"log-level=error:colorprim=bt709:transfer={trc}:colormatrix=bt709:range=limited"]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0?"]
        if f["ext"] == ".mp4" or preview:
            cmd += (["-c:a", "copy"] if info["audio_codec"] == "aac" else ["-c:a", "aac", "-b:a", "256k"])
        else:
            cmd += (["-c:a", "copy"] if info["audio_codec"].startswith("pcm") else ["-c:a", "pcm_s24le"])
    cmd += ["-map_metadata", "0", "-movflags", "+faststart+write_colr", "-progress", "pipe:1", "-nostats", str(out_path)]
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **media._popen_kwargs())
    dur = max(info["duration"], 1e-3)
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith(b"out_time_ms="):
            try:
                say(min(0.99, 0.03 + 0.96 * int(line.split(b"=")[1]) / 1e6 / dur), "Rendering preview" if preview else "Rendering graded clip")
            except ValueError:
                pass
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise ExportError(f"ffmpeg failed: {err[-600:]}")
    say(1.0, "Done")
    return {"path": str(out_path), "format": f["id"], "format_name": f["name"], "seconds": round(time.time() - t0, 1),
            "width": w, "height": h, "pix_fmt": f["pix"], "tags": {"primaries": "bt709", "transfer": trc, "matrix": "bt709", "range": "tv"}}


def _ff_path(p: Path) -> str:
    """Escape a path for use inside an ffmpeg filter option."""
    s = str(p).replace("\\", "/")
    return s.replace(":", "\\:").replace("'", "\\'")
