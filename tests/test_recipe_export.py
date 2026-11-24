"""Recipe, LUT export, clip export, graded playback and saved looks."""
import re
import subprocess

import numpy as np
import pytest

from colorgrader import export, jobs, looks, media, recipes, synth
from colorgrader.color import ops
from colorgrader.color import working as W
from colorgrader.color.lut import Lut3D, read_cube, write_cube
from colorgrader.grade import auto, model, pipeline

from helpers import filmsim_clip, flog2_clip, load_clip

LOOK = dict(contrast=1.12, lift=[0.004, 0.006, 0.012], gamma=[-0.010, -0.004, 0.006], gain=[1.02, 1.0, 0.985], saturation=47.0,
            lum_vs_sat=[[0.0, 0.8], [0.04, 0.9], [0.12, 1.0], [0.5, 1.0], [0.62, 0.95], [0.74, 0.9]],
            tone_curve=[[0.0, 0.0], [0.03, 0.03], [0.22, 0.2], [0.35, 0.35], [0.5, 0.52], [0.75, 0.75], [1.0, 1.0]],
            exposure=-0.25, hsv_sat={"gain": 1.08, "gamma": 0.03},
            sat_vs_lum=[[0.0, 1.0], [0.08, 1.0], [0.16, 0.97], [0.26, 0.95], [1.0, 0.95]],
            hue_vs_sat=[[h, 0.92 if h == 120 else 1.0] for h in range(0, 360, 30)],
            hue_vs_hue=[[h, 4.0 if h == 210 else 0.0] for h in range(0, 360, 30)])


def build_graded(proj, workdir, name="rx_src.mov", frames_each=20, size=(320, 180)):
    """A two-shot F-Log2 clip with balance and a known look at 80% strength."""
    scenes = [synth.procedural_scene(*size), synth.procedural_scene(*size, seed=5)[:, ::-1] * np.float32(0.8)]
    clip = load_clip(proj, flog2_clip(workdir / name, scenes=scenes, frames_each=frames_each, size=size))
    look = model.neutral_look()
    look.update(LOOK)
    look.update(name="Test look", source="saved")
    proj.set_look(look)
    proj.set_strength(80)
    auto.build(proj, refit=False)
    return proj.clip(clip["id"])


# ------------------------------------------------- recipe numbers = picture
def _rgb(text):
    return [float(v) for v in re.findall(r"[RGB] ([-+]?\d+\.\d+)", text)]


def _curves(node):
    out = {}
    for c in node["curves"]:
        col = 2 if c["curve"].startswith("Hue") else 1
        out[c["curve"]] = [[float(r[col - 1]), float(r[col])] for r in c["rows"]]
    return out


def _run_node(x, node):
    """Apply one recipe node with Resolve's math. x is linear light for linear nodes, DaVinci Intermediate otherwise."""
    s = {i["control"]: i["value"] for i in node["settings"]}
    if s.get("Node Color Space") == "HSV":
        return ops.hsv_saturation(x, _rgb(s["Gain"])[0] if "Gain" in s else 1.0, _rgb(s["Gamma"])[0] if "Gamma" in s else 0.0)
    if node["label"].endswith("EXPOSURE"):
        return W.di_encode(ops.lggo(W.di_decode(x), gain=_rgb(s["Gain"])))
    if "Red output" in s:
        return ops.rgb_mixer(x, [_rgb(s[f"{c} output"]) for c in ("Red", "Green", "Blue")])
    if "Contrast" in s:
        x = ops.contrast(x, float(s["Contrast"]), float(s["Pivot"]))
    if "Lift" in s or "Gamma" in s or "Gain" in s or "Offset" in s:
        x = ops.lggo(x, _rgb(s["Lift"]) if "Lift" in s else (0, 0, 0), _rgb(s["Gamma"]) if "Gamma" in s else (0, 0, 0),
                     _rgb(s["Gain"]) if "Gain" in s else (1, 1, 1), _rgb(s["Offset"]) if "Offset" in s else (25, 25, 25))
    if "Sat" in s:
        x = ops.saturation(x, float(s["Sat"]))
    c = _curves(node)
    if "Custom" in c:
        x = ops.tone_curve(x, c["Custom"])
    if any(k.startswith("Custom ") for k in c):
        x = ops.channel_curves(x, c.get("Custom Red"), c.get("Custom Green"), c.get("Custom Blue"))
    if "Sat vs Lum" in c:
        x = ops.hsl_curves(x, sat_vs_lum=c["Sat vs Lum"])
    if "Lum vs Sat" in c:
        x = ops.hsl_curves(x, lum_vs_sat=c["Lum vs Sat"])
    if "Hue vs Sat" in c or "Hue vs Hue" in c:
        x = ops.hsl_curves(x, hue_vs_sat=c.get("Hue vs Sat"), hue_vs_hue=c.get("Hue vs Hue"))
    if "Sat vs Sat" in c:
        x = ops.hsl_curves(x, sat_vs_sat=c["Sat vs Sat"])
    return x


def _rebuild_from_recipe(proj, clip):
    """Type the recipe into Resolve's math, node by node, for every shot of the clip. Returns the recipe sections."""
    recipe = recipes.build(proj, "resolve")
    sections = {s["id"]: s for s in recipe["sections"]}
    timeline = sections["timeline"]["nodes"]
    assert [n["label"] for n in timeline][-2:] == ["OUT A", "OUT B"]
    idt, prof = proj.idt(clip)
    odt = proj.odt()
    shots = proj.shots_of(clip["id"])
    groups = [g for g in sections["clips"]["groups"] if g["title"].startswith(clip["name"])]
    assert len(groups) == len(shots) == 2
    for shot, group in zip(shots, groups):
        t = 0.5 * (shot["start"] + shot["end"])
        buf, w, h = proj.frame_signal(clip, t, 320)
        sig = media.raw_to_rgb(buf, w, h, clip["info"], full_range=idt.full_range)
        x = pipeline.camera_stage(sig, idt, prof.get("corrections"))   # node IN: the input conversion
        linear = True
        for node in group["nodes"]:
            if node["label"].startswith("IN"):
                continue
            if linear and ("Linear" not in node["tool"] or node["label"].endswith("EXPOSURE")):
                x, linear = W.di_encode(x), False
            x = _run_node(x, node)
        if linear:
            x = W.di_encode(x)
        for node in timeline:
            if node["label"] in ("OUT A", "OUT B", "VIGNETTE", "GRAIN", "RELIGHT", "NR", "DETAIL"):
                continue
            x = _run_node(x, node)
        want = proj.render_frame(clip["id"], t, "graded", 320, shot["id"])
        assert np.max(np.abs(np.clip(odt(x), 0, 1) - np.clip(want, 0, 1))) < 2e-4
    return sections, groups


def test_recipe_numbers_alone_rebuild_the_picture(proj, workdir):
    """Type the recipe into Resolve's math, node by node: the result equals the app's own render."""
    _rebuild_from_recipe(proj, build_graded(proj, workdir))


# Every new control whose Resolve math is known: the recipe carries it exactly.
EXACT = dict(temp=0.3, tint=-0.1, offset=[24.6, 25.0, 25.5], shadows=0.4, highlights=-0.5, hue=3.0, color_boost=0.3,
             rgb_mixer=[[1.05, -0.03, -0.02], [0.0, 1.0, 0.0], [-0.02, 0.03, 0.99]],
             curves_rgb={"r": [[0.0, 0.0], [0.336, 0.346], [1.0, 1.0]], "g": [], "b": [[0.0, 0.01], [0.5, 0.49], [1.0, 1.0]]})
TRIM = dict(exposure=0.2, temp=-0.15, tint=0.05, contrast=1.06, offset=[25.3, 25.0, 24.8], shadows=0.3, highlights=-0.2,
            saturation=54.0, hue=2.0, rgb_mixer=[[1.0, 0.02, -0.02], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_recipe_rebuilds_every_exact_control_and_each_shot_trim(proj, workdir, monkeypatch):
    """Temp, tint, offset, shadows, highlights, hue, color boost, the RGB mixer and RGB curves on the look, plus one
    shot's own trim: typed into Resolve's math from the recipe alone, the picture equals the app's render."""
    monkeypatch.setitem(globals(), "LOOK", {**LOOK, **EXACT})
    clip = build_graded(proj, workdir, name="rx_exact.mov")
    shot = proj.shots_of(clip["id"])[0]
    proj.set_shot_grade(shot["id"], {**proj.shot(shot["id"])["grade"], "trim": TRIM, "reasons": {"trim_contrast": "Flat shot."}})
    sections, groups = _rebuild_from_recipe(proj, clip)
    labels = [n["label"] for n in sections["timeline"]["nodes"]]
    for want in ("EXPOSURE", "RGB MIXER", "TONE COLOR", "SHADOWS HIGHLIGHTS", "RGB CURVES", "COLOR BOOST", "PALETTE"):
        assert want in labels
    trim = [n for n in groups[0]["nodes"] if n["label"].startswith("TRIM ")]
    assert [n["label"] for n in trim] == ["TRIM EXPOSURE", "TRIM RGB MIXER", "TRIM CONTRAST", "TRIM TONE COLOR",
                                          "TRIM SHADOWS HIGHLIGHTS", "TRIM SATURATION", "TRIM PALETTE"]
    contrast = next(n for n in trim if n["label"] == "TRIM CONTRAST")
    assert contrast["settings"][0]["reason"] == "Flat shot." and contrast["title"].endswith("(this shot)")
    assert not any(n["label"].startswith("TRIM") for n in groups[1]["nodes"])        # the other shot has no trim
    assert proj.shot(shot["id"])["grade"]["trim"]["contrast"] == 1.06


def test_recipe_marks_controls_set_by_eye(proj, workdir):
    """HDR zones, Log wheels, Hue vs Lum and qualifier secondaries have no published Resolve math: the recipe lists them,
    says they are set by eye, and the LUT carries the exact result."""
    clip = build_graded(proj, workdir, name="rx_eye.mov")
    look = dict(proj.state["look"])
    look.update(hdr={**look["hdr"], "highlight": {"exposure": -0.4, "saturation": 0.9, "temp": 0.0, "tint": 0.0}},
                log={**look["log"], "shadow": [0.0, 0.0, 0.01]}, hue_vs_lum=[[30.0, 1.0], [210.0, 0.9]],
                secondaries=[{"name": "Sky", "hue": [215.0, 50.0, 20.0], "sat": [0.03, 1.0, 0.02], "adjust": {"saturation": 0.85}}])
    proj.set_look(look)
    r = recipes.build(proj, "resolve")
    nodes = {n["label"]: n for n in next(s for s in r["sections"] if s["id"] == "timeline")["nodes"]}
    for label in ("HDR ZONES", "LOG WHEELS", "HUE VS LUM", "SEC 1"):
        assert label in nodes and any("by eye" in n for n in nodes[label]["notes"]), label
    assert "Highlight zone" in {i["control"] for i in nodes["HDR ZONES"]["settings"]}
    sec = {i["control"]: i["value"] for i in nodes["SEC 1"]["settings"]}
    assert sec["Qualifier Hue"].startswith("Center 215 degrees") and sec["Sat"] == "44.0"   # 0.85 at 80% strength
    assert proj.clip(clip["id"])


def test_resolve_recipe_lists_settings_nodes_and_reasons(proj, workdir):
    clip = build_graded(proj, workdir)
    film = load_clip(proj, filmsim_clip(workdir / "rx_film.mov"), ("fujifilm-x-m5", "film-simulation"))
    auto.build(proj, refit=False)
    proj.set_finish({"vignette": {"enabled": True, "amount": 0.3}, "grain": {"enabled": True, "amount": 0.3}})
    r = recipes.build(proj, "resolve", "free")
    sec = {s["id"]: s for s in r["sections"]}
    project_items = {i["control"]: i["value"] for i in sec["project"]["items"]}
    assert project_items["Color Management, Color science"] == "DaVinci YRGB"
    assert project_items["Color Management, Timeline color space"] == "DaVinci WG/Intermediate"
    assert project_items["General Options, Color, Use S-curve for contrast"] == "Off"
    groups = sec["clips"]["groups"]
    first = next(g for g in groups if g["title"].startswith(clip["name"]))
    assert "Cut it at 00:00:20" in first["notes"][0]                       # 20 frames into the clip at 25 fps
    node_in = first["nodes"][0]
    values = {i["control"]: i["value"] for i in node_in["settings"]}
    assert values["Input Gamma"] == "FujiFilm F-Log2" and values["Input Color Space"] == "Rec.2020"
    assert values["Output Color Space"] == "DaVinci Wide Gamut" and values["Output Gamma"] == "DaVinci Intermediate"
    assert values["Clip Attributes, Data Levels"] == "Full"
    film_group = next(g for g in groups if g["title"].startswith(film["name"]))
    assert [n["label"] for n in film_group["nodes"]][:2] == ["IN A", "IN B"]  # finished picture: display to scene, then gamut
    inv = {i["control"]: i["value"] for i in film_group["nodes"][0]["settings"]}
    assert inv["Custom Max Input"] == "On, 100 nits" and inv["Custom Max Output"] == "On, 10000 nits"
    labels = [n["label"] for n in sec["timeline"]["nodes"]]
    assert labels == ["CONTRAST", "TONE COLOR", "SATURATION", "PALETTE", "VIGNETTE", "GRAIN", "OUT A", "OUT B"] or "LIMIT" in labels
    out_b = {i["control"]: i["value"] for i in sec["timeline"]["nodes"][-1]["settings"]}
    assert out_b["Tone Mapping Method"] == "DaVinci" and out_b["Custom Max Input"] == "On, 10000 nits"
    assert out_b["Output Gamma"] == "Gamma 2.4"
    look = model.at_strength(proj.state["look"], 0.8)
    contrast = next(n for n in sec["timeline"]["nodes"] if n["label"] == "CONTRAST")
    assert contrast["settings"][0]["value"] == f"{look['contrast']:.3f}"
    vig = next(n for n in sec["timeline"]["nodes"] if n["label"] == "VIGNETTE")
    assert "R 22.00" in vig["settings"][-1]["value"]                        # 0.3 stops = 3 Offset points under 25
    grain_free = next(n for n in sec["timeline"]["nodes"] if n["label"] == "GRAIN")
    assert "Studio" in grain_free["settings"][0]["value"]
    studio = recipes.build(proj, "resolve", "studio")
    grain_studio = next(n for n in studio["sections"][2]["nodes"] if n["label"] == "GRAIN")
    assert "Film Grain" in grain_studio["tool"]
    # one reason per setting, everywhere
    for s in r["sections"]:
        nodes = list(s.get("nodes", [])) + [n for g in s.get("groups", []) for n in g["nodes"]]
        for item in list(s.get("items", [])) + [i for n in nodes for i in n["settings"]]:
            assert item["reason"], f"no reason for {item['control']}"
        for n in nodes:
            for c in n["curves"]:
                assert c["reason"] and c["rows"]
    assert "Node T1" in r["text"] and "Why:" in r["text"] and r["text"].count("\n") > 60


def test_editors_are_plug_in_modules(proj, workdir):
    ids = [e["id"] for e in recipes.available()]
    assert ids[0] == "resolve" and "lut" in ids                 # found by scanning the recipes folder
    build_graded(proj, workdir)
    r = recipes.build(proj, "lut")
    assert r["editor"] == "lut" and "Premiere Pro" in r["text"] and "Final Cut Pro" in r["text"]
    names = export.lut_names(proj)
    assert names["profiles"]["fujifilm-x-m5/f-log2"]["camera"] in r["text"]


# --------------------------------------------------------------------- LUTs
def test_cube_file_round_trip_and_ffmpeg_agree(tmp_path):
    """The .cube file means the same to this app and to another program (ffmpeg's lut3d)."""
    lut = Lut3D.from_function(lambda v: np.stack([v[:, 0] ** 1.5, 0.8 * v[:, 1] + 0.1 * v[:, 2], 1.0 - v[:, 2]], axis=1), 17)
    path = write_cube(tmp_path / "t.cube", lut, comments=["test"])
    back = read_cube(path)
    assert np.allclose(back.table, lut.table, atol=1e-6)
    rng = np.random.default_rng(3)
    px = rng.random((64, 64, 3)).astype(np.float32)
    raw = (px * 65535 + 0.5).astype("<u2")
    px = raw.astype(np.float32) / 65535
    r = subprocess.run([media.find_binary("ffmpeg"), "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", "64x64", "-i", "pipe:0",
                        "-vf", f"lut3d=file='{export._ff_path(path)}':interp=tetrahedral", "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"],
                       input=raw.tobytes(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode()
    got = np.frombuffer(r.stdout, dtype="<u2").reshape(64, 64, 3).astype(np.float32) / 65535
    assert np.max(np.abs(got - np.clip(lut.apply(px), 0, 1))) < 2e-3


def test_lut_set_matches_the_grade(proj, workdir, tmp_path):
    clip = build_graded(proj, workdir)
    res = export.export_luts(proj, tmp_path / "luts")
    names = export.lut_names(proj)
    files = {f["name"]: f["file"] for f in res["files"]}
    assert set(files) == {names["look"], names["look_output"], names["output"], *[v for p in names["profiles"].values() for v in (p["camera"], p["input"])]}
    assert (tmp_path / "luts" / "WHERE_THE_LUTS_GO.txt").read_text().count(".cube") >= len(files)
    look = model.at_strength(proj.state["look"], proj.state["strength"] / 100.0)
    odt = proj.odt()
    idt, prof = proj.idt(clip)
    buf, w, h = proj.frame_signal(clip, 0.2, 320)
    sig = media.raw_to_rgb(buf, w, h, clip["info"], full_range=idt.full_range).reshape(-1, 3)
    di = pipeline.shot_stage(pipeline.camera_stage(sig, idt, prof.get("corrections")), model.neutral_shot())   # the picture's own colors
    lut = read_cube(files[names["look_output"]])
    assert lut.size == 33
    want = np.clip(odt(pipeline.look_stage(di, look)), 0, 1)
    err = np.abs(lut.apply(di) - want)
    assert err.mean() < 0.004 and np.percentile(err, 99) < 0.02   # a curved tone curve costs a 33-point table a little
    lut = read_cube(files[names["look"]])
    assert np.abs(lut.apply(di) - np.clip(pipeline.look_stage(di, look), 0, 1)).mean() < 0.001
    # camera LUT: recorded signal straight to the graded display picture (no per-shot balance)
    lut = read_cube(files[names["profiles"]["fujifilm-x-m5/f-log2"]["camera"]])
    err = np.abs(lut.apply(sig) - want)
    assert err.mean() < 0.004 and np.percentile(err, 99) < 0.03
    # input LUT then look+output LUT equals the camera LUT route
    lut_in = read_cube(files[names["profiles"]["fujifilm-x-m5/f-log2"]["input"]])
    chain = read_cube(files[names["look_output"]]).apply(lut_in.apply(sig))
    assert np.abs(chain - want).mean() < 0.005


def test_export_tables_follow_deep_shadows(proj, workdir):
    """The export keeps its table close to linear light for log sources, so the darkest tones stay exact."""
    clip = build_graded(proj, workdir)
    film = load_clip(proj, filmsim_clip(workdir / "rx_film_tbl.mov"), ("fujifilm-x-m5", "film-simulation"))
    ramp = np.repeat(np.linspace(0, 1, 8001, dtype=np.float32)[:, None], 3, axis=1)
    fn = export.camera_function(proj, clip, None, "display", with_look=False)
    assert export.best_encoding(fn, 65) == "z22"
    true = np.clip(fn(ramp), 0, 1)[:, 1]
    z = export.lut_camera(proj, clip, 65, None, "display", with_look=False, encoding="z22").apply(ramp)[:, 1]
    plain = export.lut_camera(proj, clip, 65, None, "display", with_look=False, encoding="display").apply(ramp)[:, 1]
    got = np.clip(z, 0, 1) ** (1 / export.Z_GAMMA)
    dark = (true > 0) & (true < 0.10)
    assert np.abs(got - true)[dark].mean() < 0.001 < np.abs(plain - true)[dark].mean()
    assert np.abs(got - true).max() < 0.015
    assert got[true == 0].max() < 0.012                         # black stays black
    assert export.best_encoding(export.camera_function(proj, film, None, "display", with_look=False), 65) == "display"
    assert export.best_encoding(proj.odt().signed, 65) == "z22"


# -------------------------------------------------------------- clip export
def _decode(path, t, width):
    info = media.probe(path)
    buf, w, h = media.read_frame(info, t, max_width=width)
    return media.raw_to_rgb(buf, w, h, info), info


def test_export_matches_the_preview_and_carries_rec709_tags(proj, workdir, tmp_path):
    clip = build_graded(proj, workdir, "rx_export.mov", frames_each=20, size=(640, 360))
    out = tmp_path / "graded.mp4"
    res = export.render_clip(proj, clip["id"], out, "auto")
    assert res["format"] == "h265-10bit-mp4"                    # 10-bit source: 10-bit export
    for shot in proj.shots_of(clip["id"]):                       # each shot carries its own balance
        t = 0.5 * (shot["start"] + shot["end"])
        got, info = _decode(out, t, 640)
        want = np.clip(proj.render_frame(clip["id"], t, "graded", 640, shot["id"]), 0, 1)
        err = np.abs(got - want)
        assert err.mean() < 0.008 and np.percentile(err, 99) < 0.05
    assert (info["width"], info["height"]) == (640, 360) and info["bit_depth"] == 10 and info["codec"] == "hevc"
    assert (info["color_primaries"], info["color_transfer"], info["color_space"]) == ("bt709", "bt709", "bt709")
    assert info["full_range"] is False
    assert abs(info["duration"] - clip["info"]["duration"]) < 0.05


def test_export_adds_vignette_and_grain_like_the_preview(proj, workdir, tmp_path):
    clip = build_graded(proj, workdir, "rx_finish.mov", frames_each=20, size=(640, 360))
    shot = proj.shots_of(clip["id"])[0]
    t = 0.5 * (shot["start"] + shot["end"])
    plain, _ = _decode(export.render_clip(proj, clip["id"], tmp_path / "plain.mp4")["path"], t, 640)
    proj.set_finish({"vignette": {"enabled": True, "amount": 0.6, "start": 0.45}})
    vig, _ = _decode(export.render_clip(proj, clip["id"], tmp_path / "vig.mp4")["path"], t, 640)
    want = np.clip(proj.render_frame(clip["id"], t, "graded", 640, shot["id"]), 0, 1)
    assert np.abs(vig - want).mean() < 0.008
    corner, center = (slice(0, 30), slice(0, 50)), (slice(165, 195), slice(295, 345))
    assert abs((plain[corner].mean() - vig[corner].mean()) - (plain[corner].mean() - want[corner].mean())) < 0.01
    assert plain[corner].mean() - vig[corner].mean() > 0.03 and abs(plain[center].mean() - vig[center].mean()) < 0.004
    proj.set_finish({"vignette": {"enabled": False}, "grain": {"enabled": True, "amount": 1.0, "size": 1.0}})
    grain, _ = _decode(export.render_clip(proj, clip["id"], tmp_path / "grain.mp4")["path"], t, 640)
    assert abs(grain.mean() - plain.mean()) < 0.003            # grain adds texture, never a brightness shift
    assert np.std(grain - plain) > 0.004


def test_eight_bit_source_exports_ten_bit_and_playback_renders(proj, workdir, tmp_path):
    film = load_clip(proj, filmsim_clip(workdir / "rx_film8.mov"), ("fujifilm-x-m5", "film-simulation"))
    auto.build(proj)
    res = export.render_clip(proj, film["id"], tmp_path / "film.mp4", "auto")
    assert res["format"] == "h265-10bit-mp4" and media.probe(res["path"])["bit_depth"] == 10   # the grade needs the levels
    res8 = export.render_clip(proj, film["id"], tmp_path / "film8.mp4", "h264-8bit-mp4")
    assert media.probe(res8["path"])["bit_depth"] == 8
    assert not export.preview_ready(proj, film["id"])
    export.build_preview(proj, film["id"])
    assert export.preview_ready(proj, film["id"])
    proj.set_strength(35)                                       # any grade change makes the playback file stale
    assert not export.preview_ready(proj, film["id"])


# -------------------------------------------------------------- saved looks
def test_saved_looks_store_apply_and_delete(proj, workdir):
    build_graded(proj, workdir)
    saved = looks.save_look(proj, "Evening warm", b"jpegbytes")
    assert saved["id"] == "evening-warm" and [l["name"] for l in looks.list_looks()][0] == "Evening warm"
    assert looks.save_look(proj, "Evening warm")["id"] == "evening-warm-2"      # same name: a second file
    proj.set_look(model.neutral_look())
    with pytest.raises(looks.LookError):
        looks.save_look(proj, "Nothing")                        # a neutral look holds nothing to save
    looks.apply_look(proj, "evening-warm")
    assert proj.state["look"]["contrast"] == LOOK["contrast"] and proj.state["look"]["source"] == "saved"
    assert proj.state["strength"] == 80
    looks.delete_look("evening-warm")
    looks.delete_look("evening-warm-2")
    assert not any(l["id"].startswith("evening-warm") for l in looks.list_looks())


def test_recipe_export_and_looks_api(client, workdir):
    from colorgrader import project

    p = project.get()
    clip = build_graded(p, workdir, "rx_api.mov")
    assert {e["id"] for e in client.get("/api/recipe/editors").json()} >= {"resolve", "lut"}
    r = client.get("/api/recipe").json()
    assert r["editor"] == "resolve" and len(r["sections"]) == 5
    txt = client.get("/api/recipe.txt?editor=lut")
    assert txt.status_code == 200 and "attachment" in txt.headers["content-disposition"]
    assert "auto" in [f["id"] for f in client.get("/api/export/formats").json()]
    job = jobs.wait(client.post("/api/export/luts").json()["job"])
    assert job["state"] == "done" and len(job["result"]["files"]) == 5
    job = jobs.wait(client.post("/api/playback/build", json={"clip_id": clip["id"]}).json()["job"])
    assert job["state"] == "done"
    status = client.get(f"/api/playback/status?clip={clip['id']}").json()
    assert status["ready"]
    video = client.get(status["url"], headers={"Range": "bytes=0-63"})
    assert video.status_code == 206 and video.headers["content-type"] == "video/mp4"
    job = jobs.wait(client.post("/api/export/video", json={"clip_ids": [clip["id"]], "format": "h264-8bit-mp4"}).json()["job"], 300)
    assert job["state"] == "done" and job["result"]["files"][0]["name"].endswith("_graded.mp4")
    assert client.post("/api/export/video", json={"format": "nope"}).status_code == 400
    assert client.post("/api/open-folder", json={"path": "/"}).status_code == 400
    saved = client.post("/api/looks", json={"name": "API look"}).json()
    assert saved["has_thumb"] and client.get(f"/api/looks/{saved['id']}/thumb.jpg").status_code == 200
    job = jobs.wait(client.post(f"/api/looks/{saved['id']}/apply").json()["job"])
    assert job["state"] == "done" and client.get("/api/state").json()["project"]["look"]["name"] == "API look"
    assert client.delete(f"/api/looks/{saved['id']}").json()["ok"]
    assert client.put("/api/settings", json={"target_editor": "lut"}).status_code == 200
    assert client.get("/api/recipe").json()["editor"] == "lut"
    assert client.put("/api/settings", json={"target_editor": "premiere"}).status_code == 400
    client.put("/api/settings", json={"target_editor": "resolve"})


def test_sharpening_and_clarity_match_between_preview_and_export(proj, workdir, tmp_path):
    clip = build_graded(proj, workdir, "rx_detail.mov", frames_each=20, size=(640, 360))
    shot = proj.shots_of(clip["id"])[0]
    t = 0.5 * (shot["start"] + shot["end"])
    soft = np.clip(proj.render_frame(clip["id"], t, "graded", 640, shot["id"]), 0, 1)
    proj.set_finish({"detail": {"sharpen": 60, "clarity": 50, "denoise": 0}})
    want = np.clip(proj.render_frame(clip["id"], t, "graded", 640, shot["id"]), 0, 1)
    got, _ = _decode(export.render_clip(proj, clip["id"], tmp_path / "detail.mp4")["path"], t, 640)
    edges = lambda im: float(np.abs(np.diff(im @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=1)).mean())
    assert edges(want) > 1.15 * edges(soft)                       # visibly crisper
    assert abs(edges(got) - edges(want)) < 0.12 * edges(want)     # the export sharpens as much as the preview
    assert np.abs(got - want).mean() < 0.01
    assert abs(got.mean() - soft.mean()) < 0.01                   # detail work never shifts brightness


def test_noise_reduction_cleans_the_export(proj, workdir, tmp_path):
    scene = synth.procedural_scene(640, 360)
    clip = load_clip(proj, flog2_clip(workdir / "rx_noisy.mov", scenes=[scene], frames_each=25, size=(640, 360), noise=0.12, crf=4))
    look = model.neutral_look()
    look.update(LOOK)
    look.update(name="Test look", source="saved")
    proj.set_look(look)
    auto.build(proj, refit=False)
    shot = proj.shots_of(clip["id"])[0]
    t = 0.5 * (shot["start"] + shot["end"])
    wall = (slice(180, 200), slice(30, 120))                      # the flat gray wall of the test scene
    noisy, _ = _decode(export.render_clip(proj, clip["id"], tmp_path / "noisy.mp4")["path"], t, 640)
    proj.set_finish({"detail": {"denoise": 60}})
    clean, _ = _decode(export.render_clip(proj, clip["id"], tmp_path / "clean.mp4")["path"], t, 640)
    import cv2

    def grit(im):   # pixel noise: what stays after removing the smooth part of the wall
        y = np.ascontiguousarray(im[wall] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
        return float((y - cv2.GaussianBlur(y, (0, 0), 3.0)).std())

    assert grit(clean) < 0.4 * grit(noisy)
    assert abs(clean[wall].mean() - noisy[wall].mean()) < 0.01
    proj.set_finish({"detail": {"denoise": 0}})
    raw = np.clip(proj.render_frame(clip["id"], t, "graded", 640, shot["id"]), 0, 1)
    proj.set_finish({"detail": {"denoise": 60}})
    prev = np.clip(proj.render_frame(clip["id"], t, "graded", 640, shot["id"]), 0, 1)
    assert grit(prev) < 0.6 * grit(raw)                           # the preview shows the cleaner picture too


def test_first_build_sets_a_finished_picture(proj, workdir):
    from colorgrader import settings

    clip = load_clip(proj, flog2_clip(workdir / "rx_first.mov", frames_each=20))
    settings.save({"auto_detail": True})
    try:
        auto.build(proj)
        d = proj.state["finish"]["detail"]
        assert d["sharpen"] > 0 and d["clarity"] > 0 and d["denoise"] > 0      # log footage: clean and crisp from the start
        assert not proj.state["finish"]["grain"]["enabled"]                    # the build never adds grain on its own
        proj.set_finish({"detail": {"sharpen": 0, "clarity": 0, "denoise": 0}})
        auto.build(proj)
        assert proj.state["finish"]["detail"] == {"denoise": 0, "sharpen": 0, "clarity": 0}   # your own values stay
    finally:
        settings.save({"auto_detail": False})


def test_relight_darkens_the_background_and_keeps_the_subject(proj, workdir, tmp_path):
    """Tracked soft shapes over the face and body: the background drops by the set stops in preview and export."""
    clip = build_graded(proj, workdir, "rx_relight.mov", frames_each=20, size=(640, 360))
    shots = proj.shots_of(clip["id"])
    for s in shots:   # a face moving left to right through each shot, as the tracker would report it
        s["track"] = [[s["start"], [[0.40, 0.40, 0.10, 0.18, 0.9]]], [s["end"], [[0.50, 0.40, 0.10, 0.18, 0.9]]]]
    shot = shots[0]
    t = 0.5 * (shot["start"] + shot["end"])
    plain = np.clip(proj.render_frame(clip["id"], t, "graded", 640, shot["id"]), 0, 1)
    proj.set_finish({"relight": {"enabled": True, "amount": 0.8, "subject": 0.0}})
    lit = np.clip(proj.render_frame(clip["id"], t, "graded", 640, shot["id"]), 0, 1)
    face = (slice(130, 160), slice(270, 330))          # face center at x 0.45, y 0.40 halfway through the shot
    corner = (slice(0, 40), slice(560, 640))
    y = lambda im: float((im @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).mean())
    assert abs(y(lit[face]) - y(plain[face])) < 0.01                 # subject stays where it was
    assert y(lit[corner]) < y(plain[corner]) - 0.04                   # background falls into shadow
    got, _ = _decode(export.render_clip(proj, clip["id"], tmp_path / "relight.mp4")["path"], t, 640)
    assert np.abs(got - lit).mean() < 0.01                           # the export follows the same shapes
    assert abs(y(got[corner]) - y(lit[corner])) < 0.015
