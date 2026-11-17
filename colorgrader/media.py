"""ffmpeg and ffprobe access: probe metadata, decode frames at full bit depth.

Frames leave ffmpeg as planar Y'CbCr 4:4:4 16-bit with code values untouched
(no range or matrix conversion). The app converts to R'G'B' itself in float32.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from . import paths
from .color.ycbcr import pick_matrix, ycbcr_to_rgb

F32 = np.float32

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".mkv", ".mxf", ".avi", ".mts", ".m2ts", ".webm"}


class MediaError(RuntimeError):
    pass


def _popen_kwargs() -> dict:
    if sys.platform == "win32":
        return {"creationflags": 0x08000000}  # CREATE_NO_WINDOW
    return {}


_bin_cache: dict[str, str] = {}


def find_binary(name: str) -> str:
    """Locate ffmpeg or ffprobe: env override, the app's bin folder, then PATH."""
    if name in _bin_cache:
        return _bin_cache[name]
    exe = name + (".exe" if sys.platform == "win32" else "")
    candidates = []
    env = os.environ.get(f"COLORGRADER_{name.upper()}")
    if env:
        candidates.append(Path(env))
    candidates.append(paths.home() / "bin" / exe)
    candidates.append(paths.PACKAGE_DIR.parent / "bin" / exe)
    for c in candidates:
        if c.is_file():
            _bin_cache[name] = str(c)
            return str(c)
    found = shutil.which(name)
    if found:
        _bin_cache[name] = found
        return found
    raise MediaError(f"{name} not found. Run the install script again, or put {exe} on your PATH.")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, **_popen_kwargs(), **kw)


# ----------------------------------------------------------------- probing
_PIX_RE = re.compile(r"(\d+)(le|be)?$")


def _pix_info(pix_fmt: str) -> tuple[int, str, bool]:
    """Bit depth, chroma layout, and whether the format is RGB."""
    p = pix_fmt or ""
    is_rgb = p.startswith(("rgb", "bgr", "gbr", "argb", "abgr", "rgba", "bgra", "pal8", "gray"))
    depth = 8
    m = _PIX_RE.search(p)
    if m and not p.startswith(("rgb24", "bgr24", "nv12", "nv21", "nv16", "nv24")):
        d = int(m.group(1))
        if d in (9, 10, 12, 14, 16):
            depth = d
        elif d in (48, 64):
            depth = 16
    if p.startswith("p010"):
        depth = 10
    if p.startswith(("p012", "p212", "p412")):
        depth = 12
    if p.startswith("p016"):
        depth = 16
    chroma = "444"
    if "420" in p or p.startswith(("nv12", "nv21", "p010", "p012", "p016")):
        chroma = "420"
    elif "422" in p or p.startswith(("nv16", "uyvy", "yuyv", "p210", "p212", "p216")):
        chroma = "422"
    return depth, chroma, is_rgb


def _fraction(text: str | None) -> float:
    try:
        f = Fraction(text or "0")
        return float(f)
    except (ValueError, ZeroDivisionError):
        return 0.0


XML_SCAN_BYTES = 4 * 1024 * 1024
_XML_BLOCK = re.compile(rb"<NonRealTimeMeta\b.*?</NonRealTimeMeta>", re.S)
_XML_ELEMENT = re.compile(r"<([A-Za-z][\w.-]*)\s+([^<>]*?)/?>")
_XML_ATTR = re.compile(r"([A-Za-z_][\w.-]*)\s*=\s*\"([^\"]*)\"")


def _sidecar_xml(path: Path) -> Path | None:
    """Camera XML written next to the clip: C0001.MP4 -> C0001M01.XML or C0001.XML."""
    wanted = {f"{path.stem}m01.xml".lower(), f"{path.stem}.xml".lower()}
    try:
        for f in path.parent.iterdir():
            if f.name.lower() in wanted and f.is_file():
                return f
    except OSError:
        pass
    return None


def camera_xml_tags(path: str | Path) -> dict[str, str]:
    """Settings from camera XML metadata, as flat 'xml.*' tags.

    Reads the sidecar file first, then looks for the same XML block inside the
    clip (some cameras embed it in MP4 files). Element attributes become
    'xml.<element>.<attribute>'. <Item name=".." value=".."/> becomes
    'xml.<name>'. Camera files match on these tags with
    detect.metadata entries such as 'xml.capturegammaequation=<value>'.
    """
    path = Path(path)
    blob = b""
    side = _sidecar_xml(path)
    if side is not None:
        try:
            blob = side.read_bytes()[:XML_SCAN_BYTES]
        except OSError:
            blob = b""
    if b"NonRealTimeMeta" not in blob:
        try:
            size = path.stat().st_size
            with open(path, "rb") as fh:
                head = fh.read(XML_SCAN_BYTES)
                tail = b""
                if size > XML_SCAN_BYTES:
                    fh.seek(max(XML_SCAN_BYTES, size - XML_SCAN_BYTES))
                    tail = fh.read(XML_SCAN_BYTES)
            m = _XML_BLOCK.search(head) or _XML_BLOCK.search(tail)
            blob = m.group(0) if m else b""
        except OSError:
            blob = b""
    m = _XML_BLOCK.search(blob)
    if not m:
        return {}
    text = m.group(0).decode("utf-8", errors="replace")
    out: dict[str, str] = {}
    for el, attrs in _XML_ELEMENT.findall(text):
        pairs = dict((k.lower(), v) for k, v in _XML_ATTR.findall(attrs))
        name = el.split(":")[-1].lower()
        if name == "item" and "name" in pairs:
            out[f"xml.{pairs['name'].lower()}"] = pairs.get("value", "")
            continue
        for k, v in pairs.items():
            if k.startswith("xmlns") or k in ("lastupdate",):
                continue
            out[f"xml.{name}.{k}"] = v
        if len(out) > 400:
            break
    return out


def probe(path: str | Path) -> dict[str, Any]:
    """Read container and stream metadata into one flat dict."""
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"File not found: {path}")
    cmd = [find_binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    r = run(cmd)
    if r.returncode != 0:
        raise MediaError(f"ffprobe failed on {path.name}: {r.stderr.decode(errors='replace')[:400]}")
    data = json.loads(r.stdout.decode("utf-8", errors="replace") or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise MediaError(f"No video or image stream in {path.name}")
    fmt = data.get("format", {})
    tags: dict[str, str] = {}
    for src in (fmt.get("tags") or {}, v.get("tags") or {}):
        for k, val in src.items():
            tags[str(k).lower()] = str(val)
    pix = v.get("pix_fmt", "")
    depth, chroma, is_rgb = _pix_info(pix)
    try:
        raw_bits = int(v.get("bits_per_raw_sample") or 0)
    except ValueError:
        raw_bits = 0
    if raw_bits in (8, 10, 12, 16):
        depth = raw_bits
    rotation = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rotation = int(round(float(sd["rotation"])))
            except (TypeError, ValueError):
                pass
    if not rotation and "rotate" in tags:
        try:
            rotation = int(tags["rotate"])
        except ValueError:
            pass
    side = [sd.get("side_data_type", "") for sd in (v.get("side_data_list") or [])]
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    sar = _fraction((v.get("sample_aspect_ratio") or "1:1").replace(":", "/")) or 1.0
    dw, dh = int(round(w * sar)), h
    if abs(rotation) % 180 == 90:
        dw, dh = dh, dw
    fps = _fraction(v.get("avg_frame_rate")) or _fraction(v.get("r_frame_rate"))
    duration = float(v.get("duration") or fmt.get("duration") or 0.0)
    is_image = path.suffix.lower() in IMAGE_EXT or (fmt.get("format_name", "") in ("image2", "png_pipe", "jpeg_pipe", "tiff_pipe", "webp_pipe", "bmp_pipe"))
    color_range = v.get("color_range") or ("pc" if pix.startswith("yuvj") else "")
    is_image_ext = path.suffix.lower() in IMAGE_EXT
    if not is_image_ext:
        for k, val in camera_xml_tags(path).items():
            tags.setdefault(k, val)
    make = (tags.get("com.apple.quicktime.make") or tags.get("make") or tags.get("com.android.manufacturer")
            or tags.get("xml.device.manufacturer") or "")
    model = (tags.get("com.apple.quicktime.model") or tags.get("model") or tags.get("com.android.model")
             or tags.get("xml.device.modelname") or "")
    info = {
        "path": str(path),
        "name": path.name,
        "size": int(fmt.get("size") or 0),
        "container": fmt.get("format_name", ""),
        "is_image": bool(is_image),
        "duration": 0.0 if is_image else duration,
        "fps": 0.0 if is_image else fps,
        "frames": int(v.get("nb_frames") or 0) or (int(round(duration * fps)) if fps else 0),
        "width": w, "height": h, "display_width": dw, "display_height": dh, "rotation": rotation,
        "codec": v.get("codec_name", ""), "profile": v.get("profile", ""),
        "pix_fmt": pix, "bit_depth": depth, "chroma": chroma, "is_rgb": is_rgb,
        "color_range": color_range, "color_space": v.get("color_space", "") or "",
        "color_transfer": v.get("color_transfer", "") or "", "color_primaries": v.get("color_primaries", "") or "",
        "bit_rate": int(v.get("bit_rate") or fmt.get("bit_rate") or 0),
        "tags": tags, "side_data": side, "make": make, "model": model,
        "creation_time": tags.get("creation_time", ""),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "audio_codec": next((s.get("codec_name", "") for s in streams if s.get("codec_type") == "audio"), ""),
    }
    info["full_range"] = info["color_range"] in ("pc", "jpeg", "full") or info["is_rgb"]
    info["matrix"] = pick_matrix(info["color_space"], w, h)
    return info


def metadata_text(info: dict[str, Any]) -> str:
    """All metadata values in one searchable string (camera name matching)."""
    parts = [info.get("make", ""), info.get("model", "")]
    parts += [f"{k}={v}" for k, v in (info.get("tags") or {}).items()]
    return "\n".join(parts)


# ---------------------------------------------------------------- decoding
def fit_size(info: dict[str, Any], max_width: int | None = None, pixels: int | None = None) -> tuple[int, int]:
    """Even output size keeping the display aspect ratio."""
    dw, dh = info["display_width"], info["display_height"]
    if pixels:
        scale = min(1.0, (pixels / float(dw * dh)) ** 0.5)
    elif max_width:
        scale = min(1.0, max_width / float(dw))
    else:
        scale = 1.0
    w = max(2, int(round(dw * scale / 2.0)) * 2)
    h = max(2, int(round(dh * scale / 2.0)) * 2)
    return w, h


def _scale_filter(w: int, h: int, flags: str = "area") -> str:
    # Same range in and out: swscale then leaves code values alone.
    return f"scale={w}:{h}:flags={flags}:in_range=tv:out_range=tv"


def raw_to_rgb(buf: np.ndarray, w: int, h: int, info: dict[str, Any], full_range: bool | None = None) -> np.ndarray:
    """One yuv444p16le (or gbrp16le) frame -> R'G'B' float32 signal."""
    planes = buf.reshape(3, h, w)
    depth = info["bit_depth"]
    if info["is_rgb"]:
        g, b, r = planes[0], planes[1], planes[2]
        return np.stack([r, g, b], axis=-1).astype(F32) / F32(65535.0)
    shift = F32(1.0 / float(1 << (16 - depth)))
    fr = info["full_range"] if full_range is None else full_range
    return ycbcr_to_rgb(planes[0].astype(F32) * shift, planes[1].astype(F32) * shift, planes[2].astype(F32) * shift,
                        info["matrix"], fr, depth)


def out_pix_fmt(info: dict[str, Any]) -> str:
    return "gbrp16le" if info["is_rgb"] else "yuv444p16le"


def read_frame(info: dict[str, Any], t: float, max_width: int = 1280, prefilter: str = "") -> tuple[np.ndarray, int, int]:
    """Decode one frame near time t. Returns raw uint16 planes (3, H, W) flattened, width, height.

    prefilter: an ffmpeg filter chain that runs on the source frames before scaling (noise reduction).
    """
    w, h = fit_size(info, max_width=max_width)
    cmd = [find_binary("ffmpeg"), "-v", "error", "-nostdin"]
    if not info["is_image"] and t > 0:
        cmd += ["-ss", f"{max(0.0, t):.4f}"]
    cmd += ["-i", info["path"], "-frames:v", "1", "-an", "-sn",
            "-vf", (prefilter + "," if prefilter else "") + _scale_filter(w, h, "bicubic"),
            "-f", "rawvideo", "-pix_fmt", out_pix_fmt(info), "pipe:1"]
    r = run(cmd)
    need = w * h * 3 * 2
    if r.returncode != 0 or len(r.stdout) < need:
        if not info["is_image"] and t > 0.5:  # past the end: step back
            return read_frame(info, max(0.0, t - 0.5), max_width, prefilter)
        raise MediaError(f"Could not decode a frame from {info['name']}: {r.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(r.stdout[:need], dtype="<u2").copy(), w, h


def decode_threads() -> int:
    """Decoder threads for long decodes. Every thread holds its own full-size frames: 8 keeps 4K decoding fast and its memory in bounds."""
    return max(2, min(8, os.cpu_count() or 4))


def analysis_pass(info: dict[str, Any], out_dir: Path, analysis_fps: float, analysis_pixels: int = 230400,
                  cut_size: tuple[int, int] = (64, 36),
                  progress: Callable[[float], None] | None = None) -> dict[str, Any]:
    """One decode of the whole clip. Writes two raw files:

    analysis.raw  yuv444p16le frames at `analysis_fps`, downscaled, full bit depth
    cuts.raw      tiny rgb24 frames at the clip frame rate, for shot detection
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    aw, ah = fit_size(info, pixels=analysis_pixels)
    cw, ch = cut_size
    if info["display_height"] > info["display_width"]:
        cw, ch = ch, cw
    fps = info["fps"] or 25.0
    cut_fps = min(fps, 120.0) if fps > 0 else 25.0  # every frame: cut times stay frame accurate for the export
    a_path, c_path = out_dir / "analysis.raw", out_dir / "cuts.raw"
    graph = (f"[0:v]split=2[a][b];"
             f"[a]fps={analysis_fps:.6f},{_scale_filter(aw, ah)},format={out_pix_fmt(info)}[oa];"
             f"[b]fps={cut_fps:.6f},scale={cw}:{ch}:flags=fast_bilinear,format=rgb24[ob]")
    cmd = [find_binary("ffmpeg"), "-v", "error", "-nostdin", "-y", "-threads", str(decode_threads()), "-i", info["path"], "-an", "-sn",
           "-filter_complex", graph,
           "-map", "[oa]", "-f", "rawvideo", str(a_path),
           "-map", "[ob]", "-f", "rawvideo", str(c_path),
           "-progress", "pipe:1", "-nostats"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **_popen_kwargs())
    dur = max(info["duration"], 1e-3)
    assert proc.stdout is not None
    for line in proc.stdout:
        if progress and line.startswith(b"out_time_ms="):
            try:
                progress(min(1.0, int(line.split(b"=")[1]) / 1e6 / dur))
            except ValueError:
                pass
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise MediaError(f"ffmpeg failed on {info['name']}: {err[:400]}")
    a_frames = a_path.stat().st_size // (aw * ah * 6)
    c_frames = c_path.stat().st_size // (cw * ch * 3)
    return {"analysis_path": str(a_path), "analysis_size": [aw, ah], "analysis_fps": analysis_fps, "analysis_frames": int(a_frames),
            "cuts_path": str(c_path), "cuts_size": [cw, ch], "cuts_fps": cut_fps, "cuts_frames": int(c_frames)}


def open_analysis(meta: dict[str, Any]) -> np.memmap:
    aw, ah = meta["analysis_size"]
    return np.memmap(meta["analysis_path"], dtype="<u2", mode="r", shape=(meta["analysis_frames"], 3 * ah * aw))


# ------------------------------------------------------------------ stills
def read_still(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an image as display R'G'B' float32 in sRGB, honoring an embedded color profile."""
    from PIL import Image, ImageCms, ImageOps
    import io

    path = Path(path)
    note = ""
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)
        icc = im.info.get("icc_profile")
        if im.mode in ("I;16", "I;16B", "I;16L", "I"):
            raise ValueError("16-bit image: use OpenCV path")
        if im.mode not in ("RGB",):
            im = im.convert("RGB")
        if icc:
            try:
                src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                desc = ImageCms.getProfileDescription(src).strip()
                if "srgb" not in desc.lower():
                    dst = ImageCms.createProfile("sRGB")
                    im = ImageCms.profileToProfile(im, src, dst, renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB")
                    note = f"converted from {desc} to sRGB"
            except Exception:  # unreadable profile: treat as sRGB
                note = "embedded profile unreadable, treated as sRGB"
        arr = np.asarray(im, dtype=np.uint8).astype(F32) / F32(255.0)
    except Exception:
        import cv2

        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise MediaError(f"Could not read image: {path.name}")
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=-1)
        raw = raw[..., :3][..., ::-1]
        peak = 65535.0 if raw.dtype == np.uint16 else 255.0
        arr = raw.astype(F32) / F32(peak)
    return np.ascontiguousarray(arr), {"note": note, "width": arr.shape[1], "height": arr.shape[0]}


def resize_rgb(rgb: np.ndarray, max_width: int | None = None, pixels: int | None = None) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    if pixels:
        s = min(1.0, (pixels / float(w * h)) ** 0.5)
    elif max_width:
        s = min(1.0, max_width / float(w))
    else:
        s = 1.0
    if s >= 0.999:
        return rgb
    return cv2.resize(rgb, (max(2, int(round(w * s))), max(2, int(round(h * s)))), interpolation=cv2.INTER_AREA)
