"""Rebuilding compressed color from the brightness (color/chroma.py) in the viewer, the still and the export."""
import subprocess

import cv2
import numpy as np

from colorgrader import export, media, synth
from colorgrader.color import chroma
from colorgrader.grade import auto, model, pipeline

from helpers import flog2_clip, load_clip


def _planes(y, cb, cr):
    return np.round(np.clip(np.stack([y, cb, cr]), 0, 1) * 65535).astype(np.uint16).reshape(-1)


def test_rebuild_removes_color_blocks_and_keeps_brightness_and_edges():
    h, w = 256, 384
    r = np.random.default_rng(2)
    y = np.full((h, w), 0.45, np.float32) + 0.02 * cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (0, 0), 1.0)
    y[:, w // 2:] += 0.25                                                    # a brightness edge
    true_cb = np.where(np.arange(w)[None, :] < w // 2, 0.42, 0.60).astype(np.float32) * np.ones((h, w), np.float32)
    blocks = np.kron(r.uniform(-0.03, 0.03, (h // 16, w // 16)), np.ones((16, 16))).astype(np.float32)   # flat 16-pixel color blocks
    cb = true_cb + blocks
    buf = _planes(y, cb, np.full((h, w), 0.5, np.float32))
    out = chroma.rebuild(buf, w, h, 1.0).reshape(3, h, w).astype(np.float32) / 65535
    assert np.array_equal(out[0], buf.reshape(3, h, w)[0].astype(np.float32) / 65535)   # brightness untouched
    def excess_step(c):   # how much larger steps at the 16-pixel block borders are than steps inside blocks, left half
        d = np.abs(np.diff(c[:, :w // 2 - 16], axis=1))
        return float(d[:, 15::16].mean() - np.delete(d, np.s_[15::16], axis=1).mean())
    assert excess_step(cb) > 0.01 and excess_step(out[1]) < 0.15 * excess_step(cb)
    left, right = out[1][:, w // 2 - 24:w // 2 - 12].mean(), out[1][:, w // 2 + 12:w // 2 + 24].mean()
    assert abs((right - left) - 0.18) < 0.03                                  # the color edge stays where the brightness edge is
    err_before = np.abs(cb - true_cb)[:, 16:w // 2 - 16].mean()
    err_after = np.abs(out[1] - true_cb)[:, 16:w // 2 - 16].mean()
    assert err_after < 0.5 * err_before                                       # closer to the clean color


def _damage(planes, err_codes, seed=1):
    """An 8-bit 4:2:0 edit export of clean (3, h, w) Y, Cb, Cr planes in 0..1: every 8x8 color block (16 picture pixels)
    off by a random amount, rounded to 8 bits, then the decoder's upsampling. Returns media.read_frame's layout."""
    _, h, w = planes.shape
    r = np.random.default_rng(seed)
    out = [np.round(planes[0] * 255) / 255]
    for c in planes[1:]:
        half = cv2.resize(c, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        half += np.kron(r.normal(0, err_codes / 255, (h // 16 + 1, w // 16 + 1)), np.ones((8, 8)))[:h // 2, :w // 2].astype(np.float32)
        out.append(cv2.resize(np.round(np.clip(half, 0, 1) * 255) / 255, (w, h), interpolation=cv2.INTER_LANCZOS4))
    return _planes(*out)


def test_rebuild_clears_color_blobs_and_keeps_subtitles():
    """An overcast sky over a white shirt with yellow subtitles, exported like a 25 Mbps 4K edit: each color block a code
    or so off. A strong grade (Color Boost most of all) showed the leftover block colors as purple and green blobs. The
    rebuild averages them away, and gives back the subtitles, which have the brightness of the shirt behind them."""
    h, w = 360, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y = np.where(yy < 180, 0.62 + 0.08 * yy / 180, 0.74 + 0.03 * np.sin(xx / 23.0))          # sky, then a shirt with folds
    cb = np.where(yy < 180, 0.495 - 0.004 * xx / w, 0.500)                                       # faint green-blue sky, neutral shirt
    cr = np.where(yy < 180, 0.497, 0.501)
    text = np.zeros((h, w), np.uint8)
    cv2.putText(text, "I am 27 years old", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 3, cv2.LINE_AA)
    a = text.astype(np.float32) / 255
    y, cb, cr = y * (1 - a) + 0.72 * a, cb * (1 - a) + 0.40 * a, cr * (1 - a) + 0.53 * a       # yellow, as bright as the shirt
    clean = np.stack([y, cb, cr]).astype(np.float32)
    buf = _damage(clean, 1.0)
    out = chroma.rebuild(buf, w, h).reshape(3, h, w).astype(np.float32) / 65535
    off = buf.reshape(3, h, w).astype(np.float32) / 65535
    assert np.array_equal(out[0], off[0])                                                       # brightness untouched
    flat = (cv2.dilate(text, np.ones((15, 15), np.uint8)) == 0) & (np.abs(yy - 180) > 12)
    blob = lambda im: np.hypot(*[cv2.GaussianBlur(im[i] - clean[i], (0, 0), 4) for i in (1, 2)])[flat].mean()
    assert blob(out) < 0.45 * blob(off), (blob(out), blob(off))                                  # the blobs average away
    letters = text > 128
    err = lambda im: np.hypot(im[1] - clean[1], im[2] - clean[2])[letters].mean()
    assert err(out) < 1.25 * err(off), (err(out), err(off))                                     # the subtitles keep their yellow
    assert np.hypot(out[1] - 0.5, out[2] - 0.5)[letters].mean() > 0.07                          # still clearly yellow


def _hard_export(src, out):
    """An edit exported hard: 8-bit 4:2:0 H.264 at a low bitrate, color quantized coarser than brightness."""
    subprocess.run([media.find_binary("ffmpeg"), "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-crf", "32", "-pix_fmt", "yuv420p",
                    "-x264-params", "chroma-qp-offset=12", str(out)], check=True)
    return out


def test_hard_compressed_clip_gets_its_color_rebuilt_everywhere(proj, workdir, tmp_path):
    scenes = [synth.procedural_scene(640, 360), synth.procedural_scene(640, 360, seed=5)]
    clean = flog2_clip(workdir / "rb_clean.mov", scenes=scenes, frames_each=8, size=(640, 360))
    hard = _hard_export(clean, workdir / "rb_hard.mp4")
    info = media.probe(hard)
    assert info["bit_depth"] == 8 and media.bits_per_pixel(info) < 0.2
    clip = load_clip(proj, hard)
    tenbit = load_clip(proj, clean)
    auto.build(proj, refit=False)
    proj.adjust_controls({"saturation": 70, "color_boost": 0.3})
    c, c10 = proj.clip(clip["id"]), proj.clip(tenbit["id"])
    assert proj.color_rebuild_on(c) and not proj.color_rebuild_on(c10)         # Auto: on for the hard export, off for 10-bit
    state = proj.public_state()
    row = next(x for x in state["clips"] if x["id"] == clip["id"])
    assert row["color_rebuild_active"] and row["color_rebuild_auto"] and row["bits_per_pixel"] < 0.2

    shot = proj.shots_of(clip["id"])[0]
    t = 0.5 * (shot["start"] + shot["end"])
    on = np.clip(proj.render_still(clip["id"], t, "graded", shot["id"]), 0, 1)
    proj.set_color_rebuild(clip["id"], "off")
    assert not proj.color_rebuild_on(proj.clip(clip["id"]))
    off = np.clip(proj.render_still(clip["id"], t, "graded", shot["id"]), 0, 1)
    proj.set_color_rebuild(clip["id"], "auto")

    # the ideal: the clean 10-bit file through the same grade
    idt, prof = proj.idt(proj.clip(clip["id"]))
    cbuf, cw, ch = media.read_frame(c10["info"], t, 640, "", True)
    look = model.at_strength(proj.state["look"], proj.state["strength"] / 100.0)
    ideal = np.clip(pipeline.render(media.raw_to_rgb(cbuf, cw, ch, c10["info"], full_range=idt.full_range), idt, proj.odt(),
                                    prof.get("corrections"), model.quantize_shot(proj.shot(shot["id"])["grade"]), look,
                                    proj.state["finish"], "graded"), 0, 1)
    yl = lambda im: im @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    color_error = lambda im: float(np.linalg.norm((im - yl(im)[..., None]) - (ideal - yl(ideal)[..., None]), axis=2).mean())
    assert color_error(on) < 0.95 * color_error(off), (color_error(on), color_error(off))     # closer to the clean file
    assert abs(float(yl(on).mean()) - float(yl(off).mean())) < 0.005             # brightness stays

    res = export.render_clip(proj, clip["id"], tmp_path / "rb.mp4", fmt="h265-10bit-mp4")
    out = media.probe(res["path"])
    assert out["frames"] == info["frames"] or abs(out["duration"] - info["duration"]) < 0.1
    buf, w, h = media.read_frame(out, t, 640, "", True)
    got = np.clip(media.raw_to_rgb(buf, w, h, out), 0, 1)
    assert np.abs(got - on).mean() < np.abs(got - off).mean()                    # the export carries the rebuilt color
    assert np.abs(got - on).mean() < 0.012
