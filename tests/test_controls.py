"""Every color control: what it does to a picture, neutral when untouched, strength and rounding."""
import copy

import numpy as np
import pytest

from colorgrader.color import ops
from colorgrader.color import working as W
from colorgrader.grade import controls, model, pipeline

F32 = np.float32


def di(rgb_lin):
    return W.di_encode(np.asarray(rgb_lin, dtype=F32))


def run(changes, x):
    return controls.apply(np.asarray(x, dtype=F32), controls.merge(controls.full_neutral(), changes))


def hue_of(x):
    return ops.hsl_split(np.asarray(x, dtype=F32))[3]


def test_neutral_changes_nothing_and_every_control_counts():
    x = np.random.default_rng(3).random((500, 3)).astype(F32) * 0.8
    assert np.array_equal(controls.apply(x, controls.full_neutral()), x)
    assert model.is_neutral_look(model.neutral_look())
    moves = {"temp": 0.3, "tint": -0.2, "hue": 10.0, "color_boost": 0.4, "shadows": 0.3, "highlights": -0.3, "offset": [26.0, 25.0, 25.0],
             "hdr": {"shadow": {"exposure": 0.5}}, "log": {"shadow": [0.01, 0.0, 0.0]}, "curves_rgb": {"r": [[0.0, 0.0], [0.5, 0.55], [1.0, 1.0]]},
             "hue_vs_lum": [[240.0, 0.8]], "rgb_mixer": [[0.9, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
             "secondaries": [{"name": "sky", "hue": [220.0, 60.0, 20.0], "adjust": {"exposure": -0.5}}]}
    for key, val in moves.items():
        look = controls.merge(model.neutral_look(), {key: val})
        assert not model.is_neutral_look(look), key
        assert model.is_neutral_look(model.at_strength(look, 0.0)), key          # strength 0 removes every control
        assert not np.allclose(controls.apply(x, look), x), key                  # and every control moves pixels


def test_exposure_temp_tint_work_on_linear_light():
    gray = di([[0.18, 0.18, 0.18]])
    lin = lambda v: W.di_decode(v)[0]
    assert np.allclose(lin(run({"exposure": 1.0}, gray)), 0.36, rtol=2e-3)       # one stop doubles the light
    warm = lin(run({"temp": 1.0}, gray))
    assert warm[0] > 0.18 * 1.4 and warm[2] < 0.18 * 0.72 and abs(warm[1] - 0.18) < 1e-4
    mag = lin(run({"tint": 0.5}, gray))
    assert mag[1] < 0.18 * 0.72 and abs(mag[0] - 0.18) < 1e-4
    black = run({"exposure": 1.0, "temp": 1.0}, di([[0.0, 0.0, 0.0]]))
    assert np.allclose(W.di_decode(black), 0.0, atol=1e-6)                       # black stays black
    assert controls.wb_gains(0.0, 0.0, 0.0) == [1.0, 1.0, 1.0]


def test_offset_hue_and_color_boost():
    x = np.array([[0.30, 0.30, 0.30]], dtype=F32)
    assert np.allclose(run({"offset": [35.0, 35.0, 35.0]}, x), 0.30 + 10 * ops.OFFSET_UNIT)   # 10 points
    red = np.array([[0.45, 0.30, 0.30]], dtype=F32)
    turned = run({"hue": 120.0}, red)
    assert abs((hue_of(turned)[0] - 120.0 + 180) % 360 - 180) < 2.0              # red turned to green
    gentle, strong = np.array([[0.36, 0.33, 0.32]], dtype=F32), np.array([[0.50, 0.30, 0.25]], dtype=F32)
    chroma = lambda v: ops.hsl_split(v)[4][0]
    g_gain = chroma(run({"color_boost": 1.0}, gentle)) / chroma(gentle)
    s_gain = chroma(run({"color_boost": 1.0}, strong)) / chroma(strong)
    assert g_gain > 1.6 and s_gain < 1.1                                         # vibrance: gentle colors gain most


def test_shadows_and_highlights_keep_black_gray_and_white():
    pts = controls.shadows_highlights_points(1.0, -1.0)
    f = lambda v: float(ops.tone_curve(np.array([[v, v, v]], dtype=F32), pts)[0, 0])
    assert abs(f(0.0)) < 1e-6 and abs(f(controls.PIVOT) - controls.PIVOT) < 1e-6 and abs(f(1.0) - 1.0) < 1e-6
    assert f(0.18) > 0.18 + 0.8 * W.STOP and f(0.57) < 0.57 - 0.8 * W.STOP     # about a stop at each center
    ys = [y for _, y in controls.shadows_highlights_points(-1.5, 1.5)]
    assert all(b > a for a, b in zip(ys, ys[1:]))                                # never folds back


def test_hdr_zones_move_their_own_band():
    stops = np.array([-2.0, 0.75, 2.5], dtype=F32)
    lin = np.repeat((0.18 * 2.0 ** stops)[:, None], 3, axis=1).astype(F32)
    out = W.di_decode(run({"hdr": {"shadow": {"exposure": 1.0}}}, W.di_encode(lin)))
    moved = np.log2(out[:, 1] / lin[:, 1])
    assert moved[0] > 0.9 and moved[2] < 0.05                                     # shadows up a stop, highlights stay
    warm = W.di_decode(run({"hdr": {"highlight": {"temp": 1.0}}}, W.di_encode(lin)))
    assert warm[2, 0] > warm[2, 2] * 1.3 and abs(warm[0, 0] / warm[0, 2] - 1.0) < 0.05


def test_log_wheels_rgb_curves_mixer_and_hue_vs_lum():
    dark, bright = np.array([[0.15, 0.15, 0.15]], dtype=F32), np.array([[0.65, 0.65, 0.65]], dtype=F32)
    lg = {"log": {"shadow": [0.0, 0.0, 0.02]}}
    assert run(lg, dark)[0, 2] > 0.165 and abs(run(lg, bright)[0, 2] - 0.65) < 1e-3
    mid = np.array([[0.4, 0.4, 0.4]], dtype=F32)
    out = run({"curves_rgb": {"r": [[0.0, 0.0], [0.4, 0.5], [1.0, 1.0]]}}, mid)
    assert abs(out[0, 0] - 0.5) < 1e-3 and abs(out[0, 1] - 0.4) < 1e-6
    sw = run({"rgb_mixer": [[0, 0, 1], [0, 1, 0], [1, 0, 0]]}, np.array([[0.1, 0.2, 0.3]], dtype=F32))
    assert np.allclose(sw, [[0.3, 0.2, 0.1]])
    blue, gray = np.array([[0.25, 0.30, 0.50]], dtype=F32), np.array([[0.4, 0.4, 0.4]], dtype=F32)
    lum = lambda v: float(ops.hsl_split(v)[0][0])
    hvl = {"hue_vs_lum": [[h, 0.7 if h == 240 else 1.0] for h in range(0, 360, 30)]}
    assert lum(run(hvl, blue)) < lum(blue) * 0.85 and abs(lum(run(hvl, gray)) - lum(gray)) < 1e-6


def test_secondary_changes_only_the_keyed_color():
    sky, skin = np.array([[0.30, 0.38, 0.52]], dtype=F32), np.array([[0.48, 0.38, 0.33]], dtype=F32)
    sec = {"name": "sky", "hue": [float(hue_of(sky)[0]), 40.0, 20.0], "sat": [0.02, 1.0, 0.01], "adjust": {"exposure": -1.0, "saturation": 1.3}}
    assert controls.secondary_key(sky, sec)[0] > 0.99 and controls.secondary_key(skin, sec)[0] < 0.01
    out = run({"secondaries": [sec]}, np.vstack([sky, skin]))
    assert out[0].mean() < sky.mean() - 0.05 and np.allclose(out[1], skin[0])
    assert controls.secondary_key(sky, dict(sec, invert=True))[0] < 0.01
    off = run({"secondaries": [dict(sec, enabled=False)]}, sky)
    assert np.allclose(off, sky)


def test_strength_scales_and_rounding_is_stable():
    look = controls.merge(model.neutral_look(), {"temp": 0.4, "hue": 20.0, "offset": [27.0, 25.0, 23.0], "color_boost": 0.5,
                                                 "hdr": {"light": {"exposure": 0.4, "saturation": 1.2}},
                                                 "secondaries": [{"name": "s", "hue": [200.0, 40.0, 10.0], "adjust": {"exposure": -0.6}}]})
    half = model.at_strength(look, 0.5)
    assert half["temp"] == 0.2 and half["hue"] == 10.0 and half["offset"] == [26.0, 25.0, 24.0]
    assert half["hdr"]["light"] == {"exposure": 0.2, "saturation": 1.1, "temp": 0.0, "tint": 0.0}
    assert half["secondaries"][0]["adjust"]["exposure"] == -0.3
    q = model.quantize_look(look)
    assert model.quantize_look(q) == q


def test_a_shot_trim_runs_after_the_match_and_survives_rounding():
    shot = model.neutral_shot()
    shot["trim"] = {"exposure": 0.5, "lift": [0.0, 0.0, 0.02], "hdr": {"shadow": {"exposure": 0.3}}}
    q = model.quantize_shot(shot)
    assert q["trim"]["exposure"] == 0.5 and q["trim"]["lift"] == [0.0, 0.0, 0.02]
    lin = np.full((4, 3), 0.18, dtype=F32)
    plain = pipeline.shot_stage(lin, model.neutral_shot())
    trimmed = pipeline.shot_stage(lin, q)
    assert trimmed.mean() > plain.mean() + 0.03
