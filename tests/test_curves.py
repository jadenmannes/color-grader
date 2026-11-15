"""Log curve math against the makers' published numbers and against colour-science."""
import json
import warnings

import numpy as np
import pytest

from colorgrader import cameras, library, paths
from colorgrader.color.curves import Curve


def fuji(curve_id: str) -> Curve:
    """The Fujifilm curves ship inside the X-M5 camera file, as data."""
    cam = json.loads((paths.SHIPPED_DATA / "cameras" / "fujifilm-x-m5.json").read_text())
    return Curve(cam["definitions"]["curves"][curve_id])


# Fujifilm F-Log Data Sheet Ver.1.1 and F-Log2 Data Sheet Ver.1.0:
# reflection (%) -> 10-bit code value and IRE (%)
FLOG_SHEET = [(0.0, 95, 3.5), (0.18, 470, 46.3), (0.90, 705, 73.2)]
FLOG2_SHEET = [(0.0, 95, 3.5), (0.18, 400, 38.4), (0.90, 570, 57.8)]


@pytest.mark.parametrize("curve_id,sheet", [("fujifilm-f-log", FLOG_SHEET), ("fujifilm-f-log2", FLOG2_SHEET)])
def test_fuji_data_sheet_values(curve_id, sheet):
    c = fuji(curve_id)
    for reflection, code, ire in sheet:
        v = float(c.encode([reflection])[0])
        assert round(v * 1023) == code, (curve_id, reflection, v * 1023)
        # IRE on the data sheet uses the video scale: (code - 64) / 876
        assert abs((v * 1023 - 64) / 876 * 100 - ire) < 0.06
        # and decoding the published code value returns the reflection
        back = float(c.decode([code / 1023.0])[0])
        assert abs(back - reflection) < 0.004 * max(reflection, 0.05)


def test_fuji_constants_match_data_sheets():
    flog = fuji("fujifilm-f-log").spec["segments"]
    assert flog[0] == {"type": "lin", "e": 8.735631, "f": 0.092864, "below": 0.00089}
    assert (flog[1]["a"], flog[1]["b"], flog[1]["c"], flog[1]["d"]) == (0.555556, 0.009468, 0.344676, 0.790453)
    flog2 = fuji("fujifilm-f-log2").spec["segments"]
    assert flog2[0] == {"type": "lin", "e": 8.799461, "f": 0.092864, "below": 0.000889}
    assert (flog2[1]["a"], flog2[1]["b"], flog2[1]["c"], flog2[1]["d"]) == (5.555556, 0.064829, 0.245281, 0.384316)


def test_fuji_decode_cut_points():
    # cut2 on the data sheets: code value where the log segment begins
    assert abs(fuji("fujifilm-f-log")._y_bounds[0] - 0.100537775223865) < 2e-6
    assert abs(fuji("fujifilm-f-log2")._y_bounds[0] - 0.100686685370811) < 2e-6


def _colour_pairs():
    import colour.models as m

    lib = library.get()
    return [
        ("fujifilm-f-log", fuji("fujifilm-f-log"), m.log_encoding_FLog, 0.0),
        ("fujifilm-f-log2", fuji("fujifilm-f-log2"), m.log_encoding_FLog2, 0.0),
        ("sony-s-log3", lib.curve("sony-s-log3"), m.log_encoding_SLog3, -0.005),
        ("canon-log-3", lib.curve("canon-log-3"), m.log_encoding_CanonLog3, -0.04),
        ("panasonic-v-log", lib.curve("panasonic-v-log"), m.log_encoding_VLog, 0.0),
        ("nikon-n-log", lib.curve("nikon-n-log"), m.log_encoding_NLog, 0.0),
        ("apple-log", lib.curve("apple-log"), m.log_encoding_AppleLogProfile, -0.05),
        ("davinci-intermediate", lib.curve("davinci-intermediate"), m.oetf_DaVinciIntermediate, 0.0),
    ]


def test_curves_match_colour_science():
    x = np.concatenate([np.linspace(-0.05, 0.05, 300), np.geomspace(0.05, 50.0, 400)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, curve, ref_encode, lowest in _colour_pairs():
            xs = x[x >= lowest]
            ref = np.asarray(ref_encode(xs.astype(np.float64)))
            mine = curve.encode(xs.astype(np.float32))
            assert np.max(np.abs(mine - ref)) < 5e-6, name
            back = curve.decode(ref.astype(np.float32))
            assert np.max(np.abs(back - xs) / (np.abs(xs) + 0.02)) < 2e-4, name


def test_published_code_values():
    lib = library.get()
    assert round(float(lib.curve("sony-s-log3").encode([0.18])[0]) * 1023) == 420
    assert round(float(lib.curve("sony-s-log3").encode([0.0])[0]) * 1023) == 95
    assert round(float(lib.curve("panasonic-v-log").encode([0.18])[0]) * 1023) == 433
    assert round(float(lib.curve("panasonic-v-log").encode([0.0])[0]) * 1023) == 128
    assert round(float(lib.curve("nikon-n-log").encode([0.18])[0]) * 1023) == 372
    assert round(float(lib.curve("canon-log-3").encode([0.18])[0]) * 1023) == 351
    assert abs(float(lib.curve("apple-log").encode([0.18])[0]) - 0.4883) < 1e-3
    assert abs(float(lib.curve("davinci-intermediate").encode([0.18])[0]) - 0.336043) < 1e-5


def test_hlg_and_pq():
    import colour.models as m

    lib = library.get()
    hlg = lib.curve("hlg")
    assert abs(float(hlg.decode([0.75])[0]) - 1.0) < 1e-5            # HDR reference white
    e = np.linspace(0.0, 1.0, 64)
    white = hlg.spec["segments"][0]["a"] / 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.asarray(m.oetf_BT2100_HLG(e))
    assert np.max(np.abs(hlg.encode((e / white).astype(np.float32)) - ref)) < 1e-6
    pq = lib.curve("pq")
    assert abs(float(pq.encode([1.0])[0]) - float(m.eotf_inverse_ST2084(203.0))) < 1e-5
    assert abs(float(pq.decode(pq.encode([1.0]))[0]) - 1.0) < 1e-3


def test_every_shipped_curve_round_trips():
    lib = library.get()
    x = np.geomspace(0.002, 8.0, 200).astype(np.float32)
    for spec in lib.curve_specs.values():
        c = lib.curve(spec["id"])
        xs = np.clip(x, 0.0, 1.0) if c.kind == "display" else x
        assert np.max(np.abs(c.decode(c.encode(xs)) - xs) / (xs + 0.01)) < 2e-3, spec["id"]


def test_shipped_cameras_hold_their_own_definitions():
    cam = cameras.get_camera("fujifilm-x-m5")
    assert {"fujifilm-f-log", "fujifilm-f-log2"} <= set(cam["definitions"]["curves"])
    assert {p["id"] for p in cam["profiles"]} >= {"f-log", "f-log2", "film-simulation"}
    phone = cameras.get_camera("phone-generic")
    assert {p["id"] for p in phone["profiles"]} >= {"standard", "hdr-hlg", "hdr-pq", "apple-log"}
