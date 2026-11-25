"""Rebuilding compressed color from the brightness (color/chroma.py) in the viewer, the still and the export."""
import subprocess

import cv2
import numpy as np

from colorgrader import export, media, synth
from colorgrader.color import chroma
from colorgrader.grade import auto

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

    def color_blocks(img):
        y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        return max(chroma.block_ratio(img[..., k] - y, 16) for k in (0, 2))
    assert color_blocks(on) < color_blocks(off)
    yl = lambda im: im @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    assert abs(float(yl(on).mean()) - float(yl(off).mean())) < 0.005             # brightness stays

    res = export.render_clip(proj, clip["id"], tmp_path / "rb.mp4", fmt="h265-10bit-mp4")
    out = media.probe(res["path"])
    assert out["frames"] == info["frames"] or abs(out["duration"] - info["duration"]) < 0.1
    buf, w, h = media.read_frame(out, t, 640, "", True)
    got = np.clip(media.raw_to_rgb(buf, w, h, out), 0, 1)
    assert np.abs(got - on).mean() < np.abs(got - off).mean()                    # the export carries the rebuilt color
    assert np.abs(got - on).mean() < 0.012
