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


def _ycbcr_clip(path, frames):
    """8-bit 4:2:0 H.264 from 8-bit Y, Cb, Cr planes (limited range, BT.709 tags)."""
    h, w = frames[0][0].shape
    cmd = [media.find_binary("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", f"{w}x{h}", "-r", "30",
           "-i", "pipe:0", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-colorspace", "bt709", "-color_range", "tv", str(path)]
    data = b"".join(np.stack(f).astype(np.uint8).tobytes() for f in frames)
    subprocess.run(cmd, input=data, check=True)
    return path


def _moving_frames(n=24, damage=(), flash=None, repeat=None, w=640, h=360):
    """A scene panning 3 pixels a frame. damage: frames whose color is wrecked in blocks (magenta), as a hard export
    stores it now and then. flash: a frame where a patch turns brighter and redder, for real. repeat: a frame that
    repeats the one before it, damage and all (an edit at 25 fps exported at 30)."""
    rgb = np.clip(synth.procedural_scene(w + 3 * n, h), 0, 1) ** (1 / 2.4)
    r = np.random.default_rng(4)
    hit = np.kron(r.random((h // 16 + 1, w // 16 + 1)) < 0.25, np.ones((16, 16)))[:h, :w]
    out = []
    for i in range(n):
        f = rgb[:, 3 * i:3 * i + w]
        y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
        cb, cr = (f[..., 2] - y) / 1.8556, (f[..., 0] - y) / 1.5748
        planes = [16 + 219 * y, 128 + 224 * cb, 128 + 224 * cr]
        if i in damage:
            planes[1] = planes[1] + 9 * hit
            planes[2] = planes[2] + 7 * hit
        if i == flash:
            planes[0][100:200, 100:300] += 30
            planes[2][100:200, 100:300] += 10
        out.append([np.clip(np.round(pl), 0, 255) for pl in planes])
    if repeat is not None:
        out[repeat] = [pl.copy() for pl in out[repeat - 1]]
    return out


def test_a_frame_stored_with_wrecked_color_takes_its_color_from_its_neighbors(workdir):
    """A 25 Mbps 4K edit export stored one frame in about a hundred with whole blocks of a pink sweater magenta. The
    frame gets its color from the frames before and after, moved along the motion. Here frame 12 is wrecked and frame 13
    repeats it (an edit at 25 fps exported at 30); frame 6 turns brighter and redder for real and keeps it."""
    clean = media.probe(_ycbcr_clip(workdir / "pan_clean.mp4", _moving_frames(repeat=13)))
    hard = media.probe(_ycbcr_clip(workdir / "pan_hard.mp4", _moving_frames(damage=(12,), flash=6, repeat=13)))
    w, h = 640, 360
    at = lambda n: max(0.0, (n - 0.5) / 30.0)      # half a frame before frame n: the reader gives the first frame from t on
    def chroma_of(buf):
        return buf.reshape(3, h, w)[1:].astype(np.float32) / 257.0
    for n in (12, 13):
        t = at(n)
        frames, c, _, _ = media.read_frames(hard, t, 2, 2, w, "", True)
        truth = chroma_of(media.read_frame(clean, t, w, "", True)[0])
        before = np.hypot(*(chroma_of(frames[c]) - truth))
        after = np.hypot(*(chroma_of(chroma.repair(frames, c, w, h)) - truth))
        hit = before > 4
        assert hit.mean() > 0.1 and after[hit].mean() < 0.25 * before[hit].mean(), (n, before[hit].mean(), after[hit].mean())
    for n in (6, 8, 0, 23):                          # the flash, a plain frame, both ends: untouched, bit for bit
        frames, c, _, _ = media.read_frames(hard, at(n), 2, 2, w, "", True)
        assert chroma.repair(frames, c, w, h) is frames[c], n
    # the export's stream sees the same neighbors as the viewer's window: same frames out, bit for bit
    cmd = export._decoder_command(hard["path"], [media._scale_filter(w, h, "lanczos+accurate_rnd+full_chroma_int")])
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    seq = [np.frombuffer(raw[i * w * h * 6:(i + 1) * w * h * 6], "<u2").copy() for i in range(len(raw) // (w * h * 6))]
    stream = list(chroma.repair_stream(seq, w, h))
    assert len(stream) == len(seq) == 24
    for n in (0, 6, 11, 12, 13, 14, 23):
        frames, c, _, _ = media.read_frames(hard, at(n), 2, 2, w, "", True)
        assert np.array_equal(frames[c], seq[n]), n                  # the same frame
        assert np.array_equal(stream[n], chroma.repair(frames, c, w, h)), n


def test_the_export_carries_the_frame_repair_like_the_still(proj, workdir, tmp_path):
    hard = _ycbcr_clip(workdir / "pan_hard2.mp4", _moving_frames(damage=(12,)))
    clip = load_clip(proj, hard)
    auto.build(proj, refit=False)
    proj.set_color_rebuild(clip["id"], "on")
    shot = proj.shots_of(clip["id"])[0]
    t = 11.5 / 30.0
    still = np.clip(proj.render_still(clip["id"], t, "graded", shot["id"]), 0, 1)
    real = chroma.repair
    try:
        chroma.repair = lambda frames, center, *a, **k: frames[center]
        proj.set_color_rebuild(clip["id"], "off")
        proj.set_color_rebuild(clip["id"], "on")
        broken = np.clip(proj.render_still(clip["id"], t, "graded", shot["id"]), 0, 1)
    finally:
        chroma.repair = real
    res = export.render_clip(proj, clip["id"], tmp_path / "pan.mp4", fmt="h265-10bit-mp4")
    out = media.probe(res["path"])
    buf, w, h = media.read_frame(out, t, 640, "", True)
    got = np.clip(media.raw_to_rgb(buf, w, h, out), 0, 1)
    yl = lambda im: (im @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    cdiff = lambda a, b: float(np.linalg.norm((a - yl(a)) - (b - yl(b)), axis=2).mean())
    assert cdiff(got, still) < 0.5 * cdiff(got, broken), (cdiff(got, still), cdiff(got, broken))
