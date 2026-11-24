"""Full-quality stills, strip rendering and the fast wide blur."""
import cv2
import numpy as np

from colorgrader import synth
from colorgrader.grade import auto, pipeline

from helpers import flog2_clip, load_clip


def _graded(proj, workdir, name, size=(960, 540)):
    clip = load_clip(proj, flog2_clip(workdir / name, scenes=[synth.procedural_scene(*size)], frames_each=6, size=size))
    auto.build(proj, refit=False)
    proj.adjust_controls({"temp": 0.2, "hdr": {"highlight": {"exposure": -0.3}}, "shadows": 0.2,
                          "secondaries": [{"name": "Sky", "hue": [214, 50, 25], "adjust": {"saturation": 1.2}}]})
    proj.set_finish({"vignette": {"enabled": True, "amount": 0.3}, "grain": {"enabled": True, "amount": 0.3},
                     "detail": {"denoise": 0, "sharpen": 40, "clarity": 30}})
    return clip, proj.shots_of(clip["id"])[0]


def test_strips_render_exactly_the_whole_frame(proj, workdir, monkeypatch):
    clip, shot = _graded(proj, workdir, "strips.mov")
    whole = proj.render_frame(clip["id"], 0.1, "graded", 960, shot["id"])
    monkeypatch.setattr(pipeline, "STRIP_PIXELS", 10_000)          # force strips on a small frame
    monkeypatch.setattr(pipeline, "STRIP_ROWS_PIXELS", 960 * 37)   # rows that do not divide the height
    monkeypatch.setattr(pipeline, "render_threads", lambda: 3)
    strips = proj.render_frame(clip["id"], 0.1, "graded", 960, shot["id"])
    assert strips.shape == whole.shape and float(np.abs(strips - whole).max()) == 0.0


def test_wide_blur_matches_a_true_gaussian():
    img = cv2.GaussianBlur(np.random.default_rng(3).random((540, 960)).astype(np.float32), (0, 0), 2.0)
    img[200:300, 300:600] += 0.5                                  # an edge, where blurs differ most
    for sigma in (21.0, 40.0):
        want = cv2.GaussianBlur(img, (0, 0), sigma)
        got = pipeline.wide_blur(img, sigma)
        assert float(np.abs(got - want).max()) < 0.01 and float(np.abs(got - want).mean()) < 0.001


def test_still_renders_at_full_resolution_and_saves_16_bit(proj, client, workdir):
    clip, shot = _graded(proj, workdir, "still.mov")
    t = 0.5 * (shot["start"] + shot["end"])
    r = client.get("/api/still", params={"clip": clip["id"], "t": t, "shot": shot["id"]})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.headers["x-still-size"] == "960x540"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.shape == (540, 960, 3) and img.dtype == np.uint8
    # the still is the viewer's picture at full size, with the best scaler: same grade, same brightness
    preview = proj.render_frame(clip["id"], t, "graded", 960, shot["id"])
    assert abs(float(img.mean()) / 255.0 - float(np.clip(preview, 0, 1).mean())) < 0.01
    d = client.get("/api/still", params={"clip": clip["id"], "t": t, "shot": shot["id"], "stage": "source", "download": 1})
    assert d.status_code == 200 and "attachment" in d.headers["content-disposition"]
    assert "still_shot1_" in d.headers["content-disposition"] and d.headers["content-disposition"].endswith('_source.png"')
    img16 = cv2.imdecode(np.frombuffer(d.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img16.dtype == np.uint16 and img16.shape == (540, 960, 3)
    assert client.get("/api/still", params={"clip": clip["id"], "stage": "nope"}).status_code == 400
    assert client.get("/api/still", params={"clip": "missing"}).status_code == 400


def test_interface_files_stay_fresh_after_an_update(client):
    """Browsers keep scripts without a Cache-Control header for hours: after a git pull the app looked old."""
    r = client.get("/static/js/main.js")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    assert client.get("/static/js/main.js", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    assert client.get("/").headers["cache-control"] == "no-store"
    v = client.get("/api/version").json()
    assert v["version"] and "commit" in v


def _block_edges(img, period):
    """How much stronger brightness jumps are at block borders than inside blocks. 1.0: no grid."""
    y = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    d = np.abs(np.diff(y, axis=1))
    return float(d[:, period - 1::period].mean() / np.delete(d, np.s_[period - 1::period], axis=1).mean())


def test_noise_reduction_leaves_no_block_grid_in_foliage(proj, workdir):
    """A block-based denoiser left a 16-pixel grid in graded leaves of 8-bit log footage. The noise reduction now
    keeps fine foliage free of any grid, and still removes noise."""
    from helpers import slog3_clip

    w, h = 960, 540
    r = np.random.default_rng(1)
    base = sum(a * cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (0, 0), s) for s, a in ((1.2, 1.0), (3, 0.8), (8, 0.6)))
    base = (base - base.min()) / (base.max() - base.min())
    leaves = (np.array([0.6, 0.22, 0.04], np.float32) * (0.15 + 0.85 * base[..., None])).astype(np.float32)
    leaves *= 1 + 0.05 * r.standard_normal((h, w, 1)).astype(np.float32)        # sensor noise
    clip = load_clip(proj, slog3_clip(workdir / "leaves8.mp4", scenes=[leaves.clip(0, 1)], frames_each=4, size=(w, h), bit_depth=8,
                                      chroma="420", crf=17, tags={"make": "SONY", "model": "ILCE-7CM2"}), ("sony-a7c-ii", "s-log3-s-gamut3-cine"))
    auto.build(proj, refit=False)
    shot = proj.shots_of(clip["id"])[0]
    proj.adjust_controls({"saturation": 70, "color_boost": 0.4, "contrast": 1.15})
    proj.set_finish({"detail": {"denoise": 0, "sharpen": 0, "clarity": 0}})
    raw = np.clip(proj.render_still(clip["id"], 0.04, "graded", shot["id"]), 0, 1)
    proj.set_finish({"detail": {"denoise": 35, "sharpen": 0, "clarity": 0}})
    clean = np.clip(proj.render_still(clip["id"], 0.04, "graded", shot["id"]), 0, 1)
    for period in (8, 16, 32):
        assert _block_edges(clean, period) < 1.05, (period, _block_edges(clean, period))
    fine = lambda im: float(np.abs(np.diff(im @ np.array([0.2126, 0.7152, 0.0722], np.float32), axis=1)).mean())
    assert fine(clean) < 0.97 * fine(raw)                                        # noise went down
