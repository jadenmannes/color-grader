"""Generate the shipped curve and gamut data files from published constants.

Run:  python tools/build_data.py

The app never imports this file. Curves and gamuts live in JSON so a new
camera or log format needs no code change. Every curve lists its source.

Segment types (encode direction, scene/display linear x -> code value y):
  lin    y = e*x + f
  log    y = c*log_base(a*x + b) + d
  pow    y = c*(a*x + b)**p + d
  const  y = value
Each segment applies below its "below" bound; the last segment has none.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "colorgrader" / "data"


def lin(e, f, below=None):
    s = {"type": "lin", "e": e, "f": f}
    if below is not None:
        s["below"] = below
    return s


def log(a, b, c, d, base=10, below=None):
    s = {"type": "log", "base": base, "a": a, "b": b, "c": c, "d": d}
    if below is not None:
        s["below"] = below
    return s


def pw(a, b, c, d, p, below=None):
    s = {"type": "pow", "a": a, "b": b, "c": c, "d": d, "p": p}
    if below is not None:
        s["below"] = below
    return s


def const(value, below):
    return {"type": "const", "value": value, "below": below}


CURVES = []

# ---------------------------------------------------------------- scene logs
CURVES.append({
    "id": "sony-s-log3",
    "name": "Sony S-Log3",
    "kind": "scene",
    "code_values": "absolute",
    "recommended_range": "auto",
    "source": "Sony, Technical Summary for S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3",
    "notes": "18% gray = code 420, 0% = 95, 90% = 598 (10-bit).",
    "labels": {"resolve": "S-Log3"},
    "segments": [
        lin((171.2102946929 - 95.0) / 0.01125 / 1023.0, 95.0 / 1023.0, below=0.01125),
        log(1.0 / 0.19, 0.01 / 0.19, 261.5 / 1023.0, 420.0 / 1023.0),
    ],
})

CURVES.append({
    "id": "canon-log-3",
    "name": "Canon Log 3",
    "kind": "scene",
    "code_values": "absolute",
    "recommended_range": "auto",
    "source": "Canon, Canon Log Gamma Curves white paper (2020), v1.2 constants",
    "notes": "Canon defines x with 90% reflectance = 1.0. Constants here take 18% gray = 0.18.",
    "labels": {"resolve": "Canon Log 3"},
    "segments": [
        log(-14.98325 / 0.9, 1.0, -0.36726845, 0.12783901, below=-0.014 * 0.9),
        lin(1.9754798 / 0.9, 0.12512219, below=0.014 * 0.9),
        log(14.98325 / 0.9, 1.0, 0.36726845, 0.12240537),
    ],
})

CURVES.append({
    "id": "panasonic-v-log",
    "name": "Panasonic V-Log",
    "kind": "scene",
    "code_values": "absolute",
    "recommended_range": "auto",
    "source": "Panasonic, V-Log/V-Gamut Reference Manual (2014)",
    "notes": "18% gray = code 433, 0% = 128 (10-bit).",
    "labels": {"resolve": "Panasonic V-Log"},
    "segments": [
        lin(5.6, 0.125, below=0.01),
        log(1.0, 0.00873, 0.241514, 0.598206),
    ],
})

CURVES.append({
    "id": "nikon-n-log",
    "name": "Nikon N-Log",
    "kind": "scene",
    "code_values": "absolute",
    "recommended_range": "auto",
    "source": "Nikon, N-Log Specification Document v1.0.0",
    "notes": "18% gray = code 372 (10-bit).",
    "labels": {"resolve": "Nikon N-Log"},
    "segments": [
        pw(1.0, 0.0075, 650.0 / 1023.0, 0.0, 1.0 / 3.0, below=0.328),
        log(1.0, 0.0, 150.0 / 1023.0, 619.0 / 1023.0, base=math.e),
    ],
})

_apple = dict(R0=-0.05641088, Rt=0.01, c=47.28711236, beta=0.00964052, gamma=0.08550479, delta=0.69336945)
CURVES.append({
    "id": "apple-log",
    "name": "Apple Log",
    "kind": "scene",
    "code_values": "normalized",
    "recommended_range": "auto",
    "source": "Apple, Apple Log Profile White Paper (2023)",
    "notes": "18% gray = 0.488 signal.",
    "labels": {"resolve": "Apple Log"},
    "segments": [
        const(0.0, below=_apple["R0"]),
        pw(1.0, -_apple["R0"], _apple["c"], 0.0, 2.0, below=_apple["Rt"]),
        log(1.0, _apple["beta"], _apple["gamma"], _apple["delta"], base=2),
    ],
})

CURVES.append({
    "id": "davinci-intermediate",
    "name": "DaVinci Intermediate",
    "kind": "scene",
    "code_values": "normalized",
    "recommended_range": "auto",
    "source": "Blackmagic Design, DaVinci Wide Gamut Intermediate white paper (2021)",
    "notes": "The app's working curve. 18% gray = 0.336.",
    "labels": {"resolve": "DaVinci Intermediate"},
    "segments": [
        lin(10.44426855, 0.0, below=0.00262409),
        log(1.0, 0.0075, 0.07329248, 7.0 * 0.07329248, base=2),
    ],
})

CURVES.append({
    "id": "linear",
    "name": "Linear",
    "kind": "scene",
    "code_values": "normalized",
    "recommended_range": "auto",
    "source": "Identity",
    "labels": {"resolve": "Linear"},
    "segments": [lin(1.0, 0.0)],
})

# HLG, BT.2100. Scene light E is scaled so HDR reference white (75% signal,
# BT.2408) decodes to 1.0, in line with log footage where diffuse white ~ 0.9.
_ha, _hb, _hc = 0.17883277, 0.28466892, 0.55991073
_hlg_white = (math.exp((0.75 - _hc) / _ha) + _hb) / 12.0  # scene E at 75% signal
CURVES.append({
    "id": "hlg",
    "name": "HLG (BT.2100)",
    "kind": "scene",
    "code_values": "normalized",
    "recommended_range": "auto",
    "source": "ITU-R BT.2100-2 HLG OETF, reference white per ITU-R BT.2408",
    "notes": "Inverse OETF gives scene light. 75% signal (HDR reference white) decodes to 1.0.",
    "labels": {"resolve": "Rec.2100 HLG"},
    "segments": [
        pw(3.0 * _hlg_white, 0.0, 1.0, 0.0, 0.5, below=(1.0 / 12.0) / _hlg_white),
        log(12.0 * _hlg_white, -_hb, _ha, _hc, base=math.e),
    ],
})

# ------------------------------------------------------------ display curves
CURVES.append({
    "id": "rec709-gamma24",
    "name": "Rec.709 (Gamma 2.4 display)",
    "kind": "display",
    "code_values": "normalized",
    "recommended_range": "auto",
    "source": "ITU-R BT.1886 reference EOTF, pure power 2.4",
    "notes": "For finished Rec.709 pictures: film simulations, standard phone clips.",
    "labels": {"resolve": "Gamma 2.4"},
    "segments": [pw(1.0, 0.0, 1.0, 0.0, 1.0 / 2.4)],
})

CURVES.append({
    "id": "gamma22",
    "name": "Gamma 2.2 display",
    "kind": "display",
    "code_values": "normalized",
    "recommended_range": "auto",
    "source": "Pure power 2.2",
    "labels": {"resolve": "Gamma 2.2"},
    "segments": [pw(1.0, 0.0, 1.0, 0.0, 1.0 / 2.2)],
})

CURVES.append({
    "id": "srgb",
    "name": "sRGB display",
    "kind": "display",
    "code_values": "normalized",
    "recommended_range": "full",
    "source": "IEC 61966-2-1",
    "labels": {"resolve": "sRGB"},
    "segments": [
        lin(12.92, 0.0, below=0.0031308),
        pw(1.0, 0.0, 1.055, -0.055, 1.0 / 2.4),
    ],
})

CURVES.append({
    "id": "rec709-scene",
    "name": "Rec.709 (scene, BT.709 OETF)",
    "kind": "scene",
    "code_values": "normalized",
    "recommended_range": "auto",
    "source": "ITU-R BT.709-6 OETF",
    "notes": "Camera encoding with no tone curve. Rare in practice. Prefer Rec.709 (Gamma 2.4 display) for finished pictures.",
    "labels": {"resolve": "Rec.709"},
    "segments": [
        lin(4.5, 0.0, below=0.018),
        pw(1.0, 0.0, 1.099, -0.099, 0.45),
    ],
})

CURVES.append({
    "id": "pq",
    "name": "PQ (ST 2084, HDR10)",
    "kind": "pq",
    "code_values": "normalized",
    "recommended_range": "auto",
    "source": "SMPTE ST 2084, ITU-R BT.2100. Display light converts to scene light through the BT.2408 PQ-to-HLG bridge (1000 nit HLG OOTF, gamma 1.2).",
    "notes": "203 nits (HDR reference white) decodes to 1.0.",
    "labels": {"resolve": "Rec.2100 ST2084"},
    "params": {"reference_white_nits": 203.0, "bridge_peak_nits": 1000.0, "ootf_gamma": 1.2, "hlg_reference_white": _hlg_white},
    "segments": [],
})

# -------------------------------------------------------------------- gamuts
D65 = [0.3127, 0.3290]
GAMUTS = [
    {"id": "rec709", "name": "Rec.709 / sRGB", "primaries": [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], "white": D65,
     "source": "ITU-R BT.709", "labels": {"resolve": "Rec.709"}},
    {"id": "rec2020", "name": "Rec.2020", "primaries": [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], "white": D65,
     "source": "ITU-R BT.2020", "labels": {"resolve": "Rec.2020"}},
    {"id": "p3-d65", "name": "P3-D65 (Display P3)", "primaries": [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]], "white": D65,
     "source": "SMPTE RP 431-2 primaries, D65 white", "labels": {"resolve": "P3-D65"}},
    {"id": "s-gamut3", "name": "Sony S-Gamut3", "primaries": [[0.730, 0.280], [0.140, 0.855], [0.100, -0.050]], "white": D65,
     "source": "Sony S-Gamut3 technical summary", "labels": {"resolve": "S-Gamut3"}},
    {"id": "s-gamut3-cine", "name": "Sony S-Gamut3.Cine", "primaries": [[0.766, 0.275], [0.225, 0.800], [0.089, -0.087]], "white": D65,
     "source": "Sony S-Gamut3.Cine technical summary", "labels": {"resolve": "S-Gamut3.Cine"}},
    {"id": "canon-cinema-gamut", "name": "Canon Cinema Gamut", "primaries": [[0.740, 0.270], [0.170, 1.140], [0.080, -0.100]], "white": D65,
     "source": "Canon Cinema Gamut white paper", "labels": {"resolve": "Canon Cinema Gamut"}},
    {"id": "v-gamut", "name": "Panasonic V-Gamut", "primaries": [[0.730, 0.280], [0.165, 0.840], [0.100, -0.030]], "white": D65,
     "source": "Panasonic V-Log/V-Gamut Reference Manual", "labels": {"resolve": "Panasonic V-Gamut"}},
    {"id": "davinci-wide-gamut", "name": "DaVinci Wide Gamut", "primaries": [[0.8000, 0.3130], [0.1682, 0.9877], [0.0790, -0.1155]], "white": D65,
     "source": "Blackmagic Design, DaVinci Wide Gamut Intermediate white paper", "labels": {"resolve": "DaVinci Wide Gamut"}},
    {"id": "aces-ap1", "name": "ACEScg (AP1)", "primaries": [[0.713, 0.293], [0.165, 0.830], [0.128, 0.044]], "white": [0.32168, 0.33767],
     "source": "AMPAS S-2014-004", "labels": {"resolve": "ACES (AP1)"}},
    {"id": "aces-ap0", "name": "ACES (AP0)", "primaries": [[0.7347, 0.2653], [0.0, 1.0], [0.0001, -0.077]], "white": [0.32168, 0.33767],
     "source": "SMPTE ST 2065-1", "labels": {"resolve": "ACES (AP0)"}},
]

# ------------------------------------------------------------------- cameras
FUJI_DEFS = {
    "curves": {
        "fujifilm-f-log": {
            "id": "fujifilm-f-log",
            "name": "Fujifilm F-Log",
            "kind": "scene",
            "code_values": "absolute",
            "recommended_range": "auto",
            "source": "Fujifilm, F-Log Data Sheet Ver.1.1",
            "notes": "0% = code 95, 18% gray = 470, 90% = 705 (10-bit).",
            "labels": {"resolve": "FujiFilm F-Log"},
            "segments": [
                lin(8.735631, 0.092864, below=0.00089),
                log(0.555556, 0.009468, 0.344676, 0.790453),
            ],
        },
        "fujifilm-f-log2": {
            "id": "fujifilm-f-log2",
            "name": "Fujifilm F-Log2",
            "kind": "scene",
            "code_values": "absolute",
            "recommended_range": "auto",
            "source": "Fujifilm, F-Log2 Data Sheet Ver.1.0",
            "notes": "0% = code 95, 18% gray = 400, 90% = 570 (10-bit).",
            "labels": {"resolve": "FujiFilm F-Log2"},
            "segments": [
                lin(8.799461, 0.092864, below=0.000889),
                log(5.555556, 0.064829, 0.245281, 0.384316),
            ],
        },
    },
    "gamuts": {
        "f-gamut": {"id": "f-gamut", "name": "Fujifilm F-Gamut", "primaries": [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
                    "white": D65, "source": "Fujifilm F-Log Data Sheet. Same primaries as ITU-R BT.2020.",
                    "labels": {"resolve": "Rec.2020"}},
    },
}

NO_CORR = {"exposure": 0.0, "temperature": 0.0, "tint": 0.0, "saturation": 1.0}

XM5 = {
    "schema": 1,
    "id": "fujifilm-x-m5",
    "name": "Fujifilm X-M5",
    "maker": "FUJIFILM",
    "metadata_names": ["X-M5"],
    "notes": "F-Log and F-Log2 record F-Gamut (Rec.2020 primaries). Film simulations record finished Rec.709 pictures.",
    "profiles": [
        {
            "id": "f-log2",
            "name": "F-Log2",
            "input": {"type": "curve", "curve": "fujifilm-f-log2", "gamut": "f-gamut"},
            "bit_depth": 10,
            "data_range": "auto",
            "detect": {"bit_depth": [10], "transfer": [], "primaries": ["bt2020"], "image": "log"},
            "corrections": dict(NO_CORR),
        },
        {
            "id": "f-log",
            "name": "F-Log",
            "input": {"type": "curve", "curve": "fujifilm-f-log", "gamut": "f-gamut"},
            "bit_depth": 10,
            "data_range": "auto",
            "detect": {"bit_depth": [10], "transfer": [], "primaries": ["bt2020"], "image": "log"},
            "corrections": dict(NO_CORR),
        },
        {
            "id": "film-simulation",
            "name": "Film simulation (Eterna, Classic Chrome and others)",
            "input": {"type": "curve", "curve": "rec709-gamma24", "gamut": "rec709"},
            "bit_depth": 8,
            "data_range": "auto",
            "detect": {"bit_depth": [8, 10], "transfer": ["bt709"], "primaries": ["bt709"], "image": "display"},
            "corrections": dict(NO_CORR),
        },
        {
            "id": "hlg",
            "name": "HLG",
            "input": {"type": "curve", "curve": "hlg", "gamut": "rec2020"},
            "bit_depth": 10,
            "data_range": "auto",
            "detect": {"bit_depth": [10], "transfer": ["arib-std-b67"], "primaries": ["bt2020"], "image": "any"},
            "corrections": dict(NO_CORR),
        },
    ],
    "definitions": FUJI_DEFS,
}

PHONE = {
    "schema": 1,
    "id": "phone-generic",
    "name": "Phone (generic)",
    "maker": "",
    "metadata_names": ["iPhone", "Apple", "samsung", "SM-", "Pixel", "Google", "Xiaomi", "OnePlus", "HUAWEI", "OPPO", "vivo",
                       "com.android", "com.apple.quicktime"],
    "notes": "Covers standard clips, HDR clips (HLG and PQ) and Apple Log. The transfer tag in the file picks the profile.",
    "profiles": [
        {
            "id": "standard",
            "name": "Standard (Rec.709)",
            "input": {"type": "curve", "curve": "rec709-gamma24", "gamut": "rec709"},
            "bit_depth": 8,
            "data_range": "auto",
            "detect": {"bit_depth": [8, 10], "transfer": ["bt709", "smpte170m", "bt470bg", "iec61966-2-1"], "primaries": ["bt709", "smpte170m", "bt470bg"], "image": "display"},
            "corrections": dict(NO_CORR),
        },
        {
            "id": "hdr-hlg",
            "name": "HDR (HLG, Dolby Vision)",
            "input": {"type": "curve", "curve": "hlg", "gamut": "rec2020"},
            "bit_depth": 10,
            "data_range": "auto",
            "detect": {"bit_depth": [10], "transfer": ["arib-std-b67"], "primaries": ["bt2020"], "image": "any"},
            "corrections": dict(NO_CORR),
        },
        {
            "id": "hdr-pq",
            "name": "HDR (PQ, HDR10 and HDR10+)",
            "input": {"type": "curve", "curve": "pq", "gamut": "rec2020"},
            "bit_depth": 10,
            "data_range": "auto",
            "detect": {"bit_depth": [10], "transfer": ["smpte2084"], "primaries": ["bt2020"], "image": "any"},
            "corrections": dict(NO_CORR),
        },
        {
            "id": "apple-log",
            "name": "Apple Log",
            "input": {"type": "curve", "curve": "apple-log", "gamut": "rec2020"},
            "bit_depth": 10,
            "data_range": "auto",
            "detect": {"bit_depth": [10], "transfer": ["unknown", "unspecified", ""], "primaries": ["bt2020"], "image": "log"},
            "corrections": dict(NO_CORR),
        },
    ],
    "definitions": {"curves": {}, "gamuts": {}},
}


def main():
    (DATA / "curves").mkdir(parents=True, exist_ok=True)
    (DATA / "gamuts").mkdir(parents=True, exist_ok=True)
    (DATA / "cameras").mkdir(parents=True, exist_ok=True)
    for c in CURVES:
        (DATA / "curves" / f"{c['id']}.json").write_text(json.dumps(c, indent=2) + "\n", encoding="utf-8")
    for g in GAMUTS:
        (DATA / "gamuts" / f"{g['id']}.json").write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
    for cam in (XM5, PHONE):
        (DATA / "cameras" / f"{cam['id']}.json").write_text(json.dumps(cam, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(CURVES)} curves, {len(GAMUTS)} gamuts, 2 cameras to {DATA}")


if __name__ == "__main__":
    main()
