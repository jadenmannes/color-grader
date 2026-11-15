"""Working space, display transform, Resolve-style operators and LUT tools."""
import numpy as np

from colorgrader import library
from colorgrader.color import ops
from colorgrader.color import working as W
from colorgrader.color.lut import Lut3D, read_cube, write_cube
from colorgrader.color.transforms import OutputTransform, input_transform_for
from colorgrader.color.ycbcr import rgb_to_ycbcr, ycbcr_to_rgb

RNG = np.random.default_rng(0)


def test_davinci_intermediate_constants():
    assert abs(W.MID_GRAY - 0.336043) < 1e-5
    assert abs(float(W.di_encode([100.0])[0]) - 1.0) < 1e-4
    x = np.geomspace(1e-4, 90.0, 100).astype(np.float32)
    assert np.max(np.abs(W.di_decode(W.di_encode(x)) - x) / x) < 1e-4
    # one stop is a fixed step in the log part, away from the toe
    assert abs(float(W.di_encode([4.0])[0] - W.di_encode([2.0])[0]) - W.STOP) < 3e-4


def test_tone_mapping_matches_resolve_form_and_inverts():
    a, b, iw, ow = W.tm_constants(10000.0)
    assert abs(b - 91.0 / 99.0) < 1e-12 and abs(a - (100.0 + b) / 100.0) < 1e-12
    x = np.geomspace(1e-4, 9.5, 200).astype(np.float32)
    y = W.tm_forward(x, 1000.0)
    assert np.all(np.diff(y) > 0) and y.max() < 1.0
    assert abs(float(W.tm_forward([10.0], 1000.0)[0]) - 1.0) < 1e-6
    assert np.max(np.abs(W.tm_inverse(y, 1000.0) - x) / x) < 2e-3


def test_finished_picture_round_trips_through_the_working_space():
    """A Rec.709 film simulation clip with no grade comes out as it went in."""
    prof = {"input": {"type": "curve", "curve": "rec709-gamma24", "gamut": "rec709"}, "data_range": "auto"}
    idt = input_transform_for(prof, False, 8, 1000.0)
    odt = OutputTransform("rec709-gamma24", 1000.0)
    rgb = RNG.random((4000, 3)).astype(np.float32) * 0.96 + 0.02
    back = odt(idt(rgb))
    assert np.percentile(np.abs(back - rgb), 99) < 1.5e-3


def test_log_footage_gray_and_white_land_where_expected():
    lib = library.get()
    prof = {"input": {"type": "curve", "curve": "sony-s-log3", "gamut": "s-gamut3-cine"}, "data_range": "full"}
    idt = input_transform_for(prof, True, 10, 1000.0)
    gray = np.full((1, 3), 420.0 / 1023.0, dtype=np.float32)
    di = idt(gray)
    assert np.allclose(di, W.MID_GRAY, atol=2e-4)          # 18% gray stays 18% gray, neutral
    odt = OutputTransform("rec709-gamma24", 1000.0)
    level = float(odt(di)[0, 1])
    assert 0.44 < level < 0.50
    assert lib.curve("sony-s-log3").code_values == "absolute"


def test_lggo_matches_resolve_formulas():
    x = np.linspace(0.0, 1.0, 11, dtype=np.float32)[:, None] * np.ones(3, dtype=np.float32)
    assert np.allclose(ops.lggo(x), x, atol=1e-7)
    assert np.allclose(ops.lggo(x, gain=(1.2, 1.2, 1.2)), 1.2 * x, atol=1e-6)
    # lift: wheel value 0.05 lifts black to 0.10 and pins white
    lifted = ops.lggo(x, lift=(0.05, 0.05, 0.05))
    assert abs(float(lifted[0, 0]) - 0.10) < 1e-6 and abs(float(lifted[-1, 0]) - 1.0) < 1e-6
    # offset: one point is 7.5 ten-bit code values
    assert np.allclose(ops.lggo(x, offset=(26.0, 25.0, 24.0)) - x, np.array([7.5, 0.0, -7.5]) / 1023.0, atol=1e-6)
    # gamma: +0.25 is exponent 1/2, -0.25 is exponent 2
    assert np.allclose(ops.lggo(x, gamma=(0.25, 0.25, 0.25)), np.sqrt(x), atol=1e-6)
    assert np.allclose(ops.lggo(x, gamma=(-0.25, -0.25, -0.25)), x ** 2, atol=1e-6)
    for g in (-0.4, -0.1, 0.0, 0.1, 0.6):
        assert abs(float(ops.gamma_exponent_to_ui(ops.gamma_ui_to_exponent(g))) - g) < 1e-9


def test_offset_point_is_a_tenth_of_a_stop_in_davinci_intermediate():
    assert abs(ops.OFFSET_UNIT / W.STOP - 0.1) < 1e-3


def test_contrast_and_saturation():
    x = RNG.random((500, 3)).astype(np.float32)
    c = ops.contrast(x, 1.3, 0.336)
    assert np.allclose(c, (x - 0.336) * 1.3 + 0.336, atol=1e-6)
    gray = np.full((5, 3), 0.4, dtype=np.float32)
    assert np.allclose(ops.saturation(gray, 80.0), gray, atol=1e-6)
    mono = ops.saturation(x, 0.0)
    assert np.allclose(mono[:, 0], mono[:, 1], atol=1e-6) and np.allclose(mono[:, 0], x @ ops.SAT_WEIGHTS, atol=1e-6)
    assert np.allclose(ops.saturation(x, 50.0), x)


def test_hsl_curves():
    x = RNG.random((2000, 3)).astype(np.float32)
    lum, c1, c2, hue, chroma = ops.hsl_split(x)
    assert np.max(np.abs(ops.hsl_join(lum, c1, c2) - x)) < 1e-5
    red, green, blue = (np.array([c], dtype=np.float32) for c in ([1, 0, 0], [0, 1, 0], [0, 0, 1]))
    assert abs(float(ops.hsl_split(red)[3][0])) < 0.5
    assert abs(float(ops.hsl_split(green)[3][0]) - 120.0) < 0.5
    assert abs(float(ops.hsl_split(blue)[3][0]) - 240.0) < 0.5
    # pull green saturation down: greens lose chroma, reds keep theirs, luma stays
    pts = [[60.0, 1.0], [120.0, 0.5], [180.0, 1.0]]
    out_g = ops.hsl_curves(green * 0.6 + 0.2, hue_vs_sat=pts)
    assert abs(float(ops.hsl_split(out_g)[4][0]) / float(ops.hsl_split(green * 0.6 + 0.2)[4][0]) - 0.5) < 0.02
    assert np.allclose(ops.hsl_curves(red * 0.6 + 0.2, hue_vs_sat=pts), red * 0.6 + 0.2, atol=1e-4)
    assert abs(float(ops.hsl_split(out_g)[0][0]) - float(ops.hsl_split(green * 0.6 + 0.2)[0][0])) < 1e-5
    # hue rotation keeps chroma size
    rot = ops.hsl_curves(x, hue_vs_hue=[[0.0, 10.0], [180.0, 10.0]])
    assert np.max(np.abs(ops.hsl_split(rot)[4] - chroma)) < 1e-4


def test_lut_cube_round_trip_and_interpolation(tmp_path):
    fn = lambda v: v ** 1.3 * np.array([0.9, 1.0, 0.95], dtype=np.float32)
    lut = Lut3D.from_function(fn, size=33, title="test")
    path = write_cube(tmp_path / "t.cube", lut, comments=["unit test"])
    back = read_cube(path)
    assert back.size == 33 and back.title == "test"
    assert np.max(np.abs(back.table - lut.table)) < 1e-6
    x = RNG.random((5000, 3)).astype(np.float32)
    assert np.max(np.abs(back.apply(x) - fn(x))) < 2e-3
    ident = Lut3D.from_function(lambda v: v, size=17)
    assert np.max(np.abs(ident.apply(x) - x)) < 1e-6


def test_ycbcr_round_trip_video_and_full_range():
    rgb = RNG.random((64, 3)).astype(np.float32)
    for matrix in ("bt709", "bt2020nc", "bt601"):
        y, cb, cr = rgb_to_ycbcr(rgb, matrix)
        for full, depth in ((True, 10), (False, 10), (False, 8)):
            s = float(1 << (depth - 8))
            peak = float((1 << depth) - 1)
            if full:
                planes = (y * peak, cb * peak + 128 * s, cr * peak + 128 * s)
            else:
                planes = (y * 219 * s + 16 * s, cb * 224 * s + 128 * s, cr * 224 * s + 128 * s)
            back = ycbcr_to_rgb(*planes, matrix, full, depth)
            assert np.max(np.abs(back - rgb)) < 1e-5
