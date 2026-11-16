"""Balance, scene match, reference matching, skin protection and quality rules."""
import numpy as np
import pytest

from colorgrader import jobs, library, project, synth
from colorgrader.analysis.neutral import SKIN_LINE
from colorgrader.color import working as W
from colorgrader.color.gamuts import apply_matrix, conversion_matrix
from colorgrader.color.transforms import OutputTransform
from colorgrader.grade import auto, lookfit, model, pipeline, quality
from colorgrader.grade import skin as skin_mod

from helpers import filmsim_clip, flog2_clip, wall_patch

ODT = OutputTransform("rec709-gamma24", 10000.0)


def save_png16(path, rgb):
    import cv2

    img = (np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16)
    cv2.imwrite(str(path), np.ascontiguousarray(img[..., ::-1]))
    return path


def load_clip(p, path, profile=("fujifilm-x-m5", "f-log2")):
    clip = p.add_clip(str(path))
    if profile:
        p.set_profile(clip["id"], *profile)
    p.analyze_clip(clip["id"])
    return p.clip(clip["id"])


@pytest.fixture()
def proj():
    p = project.get()
    p.reset()
    return p


# ------------------------------------------------------------ matching test
KNOWN_LOOKS = {
    "warm contrast": dict(contrast=1.22, lift=[0.006, 0.004, 0.0], gamma=[0.006, 0.0, -0.008], gain=[1.02, 1.0, 0.975], saturation=46.0),
    "cool fade": dict(contrast=0.94, lift=[0.014, 0.018, 0.024], gamma=[-0.006, 0.0, 0.006], gain=[0.985, 1.0, 1.015], saturation=44.0),
}


@pytest.mark.parametrize("name", sorted(KNOWN_LOOKS))
def test_known_grade_comes_back_from_a_reference(proj, workdir, name):
    """Apply a known grade to a clip, use the result as the reference, recover the settings."""
    scene = synth.procedural_scene(640, 360, skin=False)
    clip = load_clip(proj, flog2_clip(workdir / "match_src.mov", scenes=[scene], frames_each=25, size=(640, 360), crf=4))
    shot = proj.shots_of(clip["id"])[0]
    auto.build(proj)                                   # balance only: no reference yet
    known = model.neutral_look()
    known.update(KNOWN_LOOKS[name])
    known = model.quantize_look(known)
    graded = proj.render_frame(clip["id"], 0.5, "graded", 640, shot["id"], look=known)
    ref_path = save_png16(workdir / f"match_ref_{name.replace(' ', '_')}.png", graded)

    proj.set_strength(100)
    proj.set_reference(str(ref_path))
    report = auto.build(proj)
    found = proj.state["look"]
    assert report["fit"]["same_content"], "the app must notice the reference shows the same picture"

    assert abs(found["contrast"] - known["contrast"]) <= 0.03
    # lift, gamma and gain trade a little against each other while the picture stays the same
    assert np.max(np.abs(np.array(found["lift"]) - known["lift"])) <= 0.010
    assert np.max(np.abs(np.array(found["gamma"]) - known["gamma"])) <= 0.012
    assert np.max(np.abs(np.array(found["gain"]) - known["gain"])) <= 0.015
    assert abs(found["saturation"] - known["saturation"]) <= 2.0

    # and the picture itself matches the reference: under half a percent on average
    again = proj.render_frame(clip["id"], 0.5, "graded", 640, shot["id"])
    assert np.mean(np.abs(again - graded)) < 0.004
    assert np.percentile(np.abs(again - graded), 99) < 0.02
    assert report["totals"]["failed"] == 0


def test_night_reference_keeps_daytime_exposure(proj, workdir):
    """A dark reference with other content passes on color and contrast, never its darkness."""
    scene = synth.procedural_scene(640, 360, skin=False)
    clip = load_clip(proj, flog2_clip(workdir / "day.mov", scenes=[scene], frames_each=25, size=(640, 360)))
    shot = proj.shots_of(clip["id"])[0]
    night = synth.procedural_scene(640, 360, seed=9, sky=False, skin=False)[:, ::-1] * np.array([0.10, 0.13, 0.20], dtype=np.float32)
    lib = library.get()
    di = W.di_encode(apply_matrix(night, conversion_matrix(lib.gamut("rec709"), lib.gamut(W.WORKING_GAMUT))))
    ref = save_png16(workdir / "night_ref.png", ODT(di))
    proj.set_strength(100)
    proj.set_reference(str(ref))
    report = auto.build(proj)
    assert not report["fit"]["same_content"]
    assert report["fit"]["exposure_gap_stops"] < -1.2 and report["fit"]["midtone_carry"] == 0.0
    assert any("exposure stays" in n.lower() for n in report["notes"])
    bal = proj.render_frame(clip["id"], 0.5, "balanced", 640, shot["id"])
    gra = proj.render_frame(clip["id"], 0.5, "graded", 640, shot["id"])
    y = lambda im: float(np.median(im @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
    assert abs(y(gra) - y(bal)) < 0.05                    # midtones stay within 5 points
    wb, wg = wall_patch(bal), wall_patch(gra)
    assert wg[2] / wg[0] > wb[2] / wb[0] + 0.01           # the cool cast of the reference comes across on the gray wall


# ------------------------------------------------------------------ balance
def test_balance_fixes_cast_exposure_and_black_point(proj, workdir):
    scene = synth.procedural_scene(640, 360, skin=False)
    clip = load_clip(proj, flog2_clip(workdir / "off.mov", scenes=[scene], frames_each=25, size=(640, 360),
                                      exposure=1.6, wb_gains=(0.88, 1.0, 1.15), flare=0.012))
    shot = proj.shots_of(clip["id"])[0]
    assert shot["measure"]["exposure"]["median_stops"] > 1.0
    auto.build(proj)
    shot = proj.shot(shot["id"])
    g = shot["grade"]["balance"]
    assert max(g["gain"]) < 0.75                           # exposure pulled down
    assert g["gain"][0] > g["gain"][1] > g["gain"][2]      # red up, blue down: the cool cast leaves
    assert all(o < 25.0 for o in g["offset"])              # flare removed
    data = proj.shot_data(shot["id"])
    lin = pipeline.balance_linear(data["frames"][0].astype(np.float32), shot["grade"])
    wall = wall_patch(lin)
    assert np.max(np.abs(np.log2(wall / wall[1]))) < 0.06  # the gray wall ends neutral
    stops = np.log2(np.maximum(lin @ np.array([0.2741, 0.8736, -0.1478], dtype=np.float32), 1e-4) / 0.18)
    assert -2.0 <= float(np.median(stops)) <= 1.1
    assert "stops" in shot["grade"]["reasons"]["gain"] and shot["grade"]["reasons"]["offset"]
    rules = {r["id"]: r["status"] for r in shot["rules"]}
    assert rules["neutral"] == "pass"


def test_balance_leaves_a_good_shot_alone(proj, workdir):
    scene = np.maximum(synth.procedural_scene(640, 360, skin=False) * 0.5 - 0.0024, 0.0)    # normal exposure, true blacks
    clip = load_clip(proj, flog2_clip(workdir / "good.mov", scenes=[scene], frames_each=25, size=(640, 360)))
    auto.build(proj)
    g = proj.shots_of(clip["id"])[0]["grade"]["balance"]
    assert all(abs(v - 1.0) < 0.05 for v in g["gain"]), g
    assert all(abs(v - 25.0) < 0.25 for v in g["offset"]), g


# -------------------------------------------------------------- scene match
def test_two_cameras_on_one_scene_match(proj, workdir):
    scene = synth.procedural_scene(640, 360, skin=False)
    a = load_clip(proj, flog2_clip(workdir / "cam_a.mov", scenes=[scene], frames_each=25, size=(640, 360)))
    b = load_clip(proj, filmsim_clip(workdir / "cam_b.mov", scenes=[scene], frames_each=25, size=(640, 360),
                                     display_look={"contrast": 1.4, "saturation": 1.15}), profile=None)
    auto.build(proj)
    sa, sb = proj.shots_of(a["id"])[0], proj.shots_of(b["id"])[0]
    assert sb["grade"]["match"]["contrast"] < 0.97          # the film simulation eases toward the log camera
    assert "main camera" in sb["grade"]["reasons"].get("match", "")
    fa = proj.render_frame(a["id"], 0.5, "balanced", 640, sa["id"])
    fb_matched = proj.render_frame(b["id"], 0.5, "balanced", 640, sb["id"])
    fb_plain = proj.render_frame(b["id"], 0.5, "converted", 640)
    assert np.mean(np.abs(fb_matched - fa)) < np.mean(np.abs(fb_plain - fa))   # closer after the match


def test_shots_of_one_clip_share_their_white_balance(proj, workdir):
    scenes = [synth.procedural_scene(640, 360, skin=False), synth.procedural_scene(640, 360, seed=5, skin=False)[:, ::-1]]
    clip = load_clip(proj, flog2_clip(workdir / "two.mov", scenes=scenes, frames_each=25, size=(640, 360), wb_gains=(1.12, 1.0, 0.9)))
    auto.build(proj)
    shots = proj.shots_of(clip["id"])
    assert len(shots) == 2
    ratios = [np.array(s["grade"]["balance"]["gain"]) / s["grade"]["balance"]["gain"][1] for s in shots]
    assert np.max(np.abs(ratios[0] - ratios[1])) < 0.08
    assert all(r[0] < 1.0 < r[2] for r in ratios)          # warm cast removed in both


# --------------------------------------------------------------------- skin
def checker_skin(n=400):
    lib = library.get()
    rgb = np.array([[194, 150, 130], [115, 82, 68]], dtype=np.float32) / 255.0      # ColorChecker light and dark skin
    lin = lib.curve("srgb").decode(rgb)
    lin = np.repeat(lin, n // 2, axis=0) * np.random.default_rng(1).uniform(0.8, 1.25, (n, 1)).astype(np.float32)
    return W.di_encode(apply_matrix(lin, conversion_matrix(lib.gamut("rec709"), lib.gamut(W.WORKING_GAMUT))))


def test_skin_correction_returns_skin_to_the_line():
    sk = checker_skin()
    ang, _ = skin_mod.skin_on_display(sk, ODT)
    assert abs(skin_mod.offset_from_line(ang)) < 5.0
    look = model.neutral_look()
    look.update(contrast=1.15, gamma=[-0.006, 0.003, 0.002], gain=[0.9925, 1.0025, 1.0], saturation=52.0)   # pushes the frame toward green
    look = model.quantize_look(look)
    pushed, _ = skin_mod.skin_on_display(pipeline.look_stage(sk, look), ODT)
    fix, info = skin_mod.skin_fix(sk, look, ODT)
    fixed, _ = skin_mod.skin_on_display(pipeline.look_stage(sk, dict(look, skin=fix)), ODT)
    assert abs(skin_mod.offset_from_line(pushed)) > 5.0
    assert abs(skin_mod.offset_from_line(fixed)) < abs(skin_mod.offset_from_line(pushed))
    assert abs(skin_mod.offset_from_line(fixed)) <= 6.0
    assert max(abs(v) for _, v in fix["hue_vs_hue"]) <= skin_mod.MAX_ROTATION + 1e-6
    # the correction touches the skin band only
    green = np.array([[0.30, 0.42, 0.28]], dtype=np.float32)
    assert np.allclose(pipeline.look_stage(green, dict(look, skin=fix)), pipeline.look_stage(green, look), atol=1e-5)


def test_neutral_look_needs_no_skin_correction():
    fix, info = skin_mod.skin_fix(checker_skin(), model.neutral_look(), ODT)
    assert fix == {"hue_vs_hue": [], "hue_vs_sat": []}


# ------------------------------------------------------------ quality rules
def sample_scene():
    lib = library.get()
    scene = synth.procedural_scene(320, 180, skin=False)
    lin = apply_matrix(scene, conversion_matrix(lib.gamut("rec709"), lib.gamut(W.WORKING_GAMUT))).reshape(-1, 3)
    ypos = (np.mgrid[0:180, 0:320][0] / 179.0).ravel().astype(np.float32)
    return lin, ypos


def test_quality_rules_catch_a_harsh_look():
    lin, ypos = sample_scene()
    shot = model.neutral_shot()
    gentle = model.neutral_look()
    gentle.update(contrast=1.15, saturation=48.0)
    harsh = model.neutral_look()
    harsh.update(contrast=1.8, lift=[-0.06, -0.06, -0.06], gain=[1.25, 1.25, 1.25], saturation=95.0)
    ok = {r["id"]: r["status"] for r in quality.check_shot(lin, ypos, None, shot, model.quantize_look(gentle), ODT, 50.0, None, None, 1.0)}
    bad = {r["id"]: r["status"] for r in quality.check_shot(lin, ypos, None, shot, model.quantize_look(harsh), ODT, 50.0, None, None, 1.0)}
    assert all(v == "pass" for v in ok.values()), ok
    assert bad["shadows"] == "fail" and bad["saturation"] == "fail"
    assert quality.subtlety(model.quantize_look(harsh))["status"] == "warn"
    assert quality.subtlety(model.quantize_look(gentle))["status"] == "pass"


def test_rules_blame_a_look_only_for_what_it_adds():
    """Footage with colors at the edge of the display gamut passes under no look. A look that pushes more colors over fails."""
    rng = np.random.default_rng(4)
    lib = library.get()
    base = rng.uniform(0.05, 0.6, (6000, 1)).astype(np.float32) * np.ones(3, dtype=np.float32)
    neon = base.copy()
    neon[:900] = rng.uniform(0.2, 0.5, (900, 1)).astype(np.float32) * np.array([0.02, 1.0, 0.03], dtype=np.float32)   # laser green, outside Rec.709
    lin = apply_matrix(neon, conversion_matrix(lib.gamut("rec2020"), lib.gamut(W.WORKING_GAMUT)))
    ypos = np.linspace(0, 1, len(lin), dtype=np.float32)
    shot = model.neutral_shot()
    none = {r["id"]: r for r in quality.check_shot(lin, ypos, None, shot, model.neutral_look(), ODT, 50.0, None, None, 0.0)}
    assert none["color_clip"]["status"] == "pass" and "footage itself" in none["color_clip"]["detail"]
    assert none["saturation"]["status"] in ("pass", "warn")
    # an ordinary scene, then a look that drives its warm highlights out of gamut
    warm = base.copy()
    warm[:900] = rng.uniform(2.0, 6.0, (900, 1)).astype(np.float32) * np.array([1.0, 0.8, 0.35], dtype=np.float32)
    lin = apply_matrix(warm, conversion_matrix(lib.gamut("rec709"), lib.gamut(W.WORKING_GAMUT)))
    push = model.neutral_look()
    push.update(contrast=1.35, gain=[1.06, 1.0, 0.9], saturation=60.0)
    calm = {r["id"]: r for r in quality.check_shot(lin, ypos, None, shot, model.neutral_look(), ODT, 50.0, None, None, 0.0)}
    hot = {r["id"]: r for r in quality.check_shot(lin, ypos, None, shot, model.quantize_look(push), ODT, 50.0, None, None, 0.0)}
    assert calm["color_clip"]["status"] == "pass"
    assert hot["color_clip"]["status"] == "fail" and "zero_bright" in hot["color_clip"]["tags"]
    # a reference that shows the same neon colors allows them
    allowed = {r["id"]: r for r in quality.check_shot(lin, ypos, None, shot, model.quantize_look(push), ODT, 50.0, None,
                                                       {"zero_bright": 0.5, "zero_mid": 0.5}, 0.0)}
    assert "zero_bright" not in allowed["color_clip"]["tags"]


def test_build_repairs_a_look_that_breaks_the_rules(proj, workdir):
    scene = synth.procedural_scene(640, 360, skin=False)
    clip = load_clip(proj, flog2_clip(workdir / "repair.mov", scenes=[scene], frames_each=25, size=(640, 360)))
    harsh = model.neutral_look()
    harsh.update(name="Harsh", source="saved", contrast=1.75, lift=[-0.05, -0.05, -0.05], saturation=80.0)
    proj.set_look(harsh)
    proj.set_strength(100)
    report = auto.build(proj)
    look = proj.state["look"]
    assert look["contrast"] < 1.75 and float(np.mean(look["lift"])) > -0.05      # the build eased it
    assert report["fixes"]
    assert report["totals"]["failed"] == 0


def test_saturation_limiter_and_strength():
    pts = lookfit.saturation_limiter()
    xs = np.array([p[0] for p in pts])
    out = xs * np.array([p[1] for p in pts])
    assert np.all(np.diff(out) >= -1e-9) and out.max() <= 0.26 + 1e-6      # monotone, bounded
    assert all(abs(p[1] - 1.0) < 1e-9 for p in pts if p[0] <= 0.16)       # normal colors untouched
    look = model.neutral_look()
    look.update(contrast=1.4, lift=[0.02, 0.0, -0.02], gain=[1.1, 1.0, 0.9], saturation=30.0, hue_vs_sat=[[120.0, 0.6]], hue_vs_hue=[[200.0, 8.0]])
    half = model.at_strength(look, 0.5)
    assert abs(half["contrast"] - 1.2) < 1e-9 and abs(half["saturation"] - 40.0) < 1e-9
    assert half["lift"] == [0.01, 0.0, -0.01] and half["gain"] == [1.05, 1.0, 0.95]
    assert half["hue_vs_sat"] == [[120.0, 0.8]] and half["hue_vs_hue"] == [[200.0, 4.0]]
    assert model.is_neutral_look(model.at_strength(look, 0.0))


# ---------------------------------------------------------------------- api
def test_reference_and_grade_api(client, workdir):
    scene = synth.procedural_scene(640, 360, skin=False)
    path = flog2_clip(workdir / "api.mov", scenes=[scene], frames_each=25, size=(640, 360))
    clip = client.post("/api/clips", json={"paths": [str(path)]}).json()["added"][0]
    client.post(f"/api/clips/{clip['id']}/profile", json={"camera_id": "fujifilm-x-m5", "profile_id": "f-log2"})
    assert jobs.wait(client.post("/api/analyze", json={"clip_ids": [clip["id"]]}).json()["job"])["state"] == "done"
    lib = library.get()
    other = synth.procedural_scene(640, 360, seed=4, sky=False, skin=False)[:, ::-1]
    di = W.di_encode(apply_matrix(other, conversion_matrix(lib.gamut("rec709"), lib.gamut(W.WORKING_GAMUT))))
    look = model.neutral_look()
    look.update(contrast=1.25, lift=[0.0, 0.006, 0.014], gain=[1.03, 1.0, 0.96], saturation=44.0)
    ref = save_png16(workdir / "api_ref.png", ODT(pipeline.look_stage(di, model.quantize_look(look))))
    r = client.post("/api/reference", json={"path": str(ref)})
    assert r.status_code == 200 and r.json()["description"]
    assert client.get("/api/reference/frame").content[:2] == b"\xff\xd8"
    assert client.get("/api/reference/scope?kind=vectorscope").content[:4] == b"\x89PNG"
    job = jobs.wait(client.post("/api/grade/build", json={}).json()["job"])
    assert job["state"] == "done", job
    st = client.get("/api/state").json()["project"]
    assert st["graded"] and st["look"]["source"] == "reference" and st["report"]["totals"]["failed"] == 0
    shot = st["shots"][0]
    assert shot["rules"] and shot["report"]["failed"] == 0
    for stage in ("balanced", "graded"):
        assert client.get(f"/api/frame?clip={clip['id']}&t=0.5&stage={stage}&w=320").content[:2] == b"\xff\xd8"
        assert client.get(f"/api/scope?clip={clip['id']}&t=0.5&stage={stage}&kind=parade").content[:4] == b"\x89PNG"
    full = st["look"]["contrast"]
    st2 = client.put("/api/grade/strength", json={"strength": 50}).json()
    assert st2["strength"] == 50 and abs(st2["applied_look"]["contrast"] - (1 + 0.5 * (full - 1))) < 2e-3
    st3 = client.put("/api/grade/finish", json={"finish": {"vignette": {"enabled": True, "amount": 0.3}}}).json()
    assert st3["finish"]["vignette"]["enabled"]
    assert client.put(f"/api/shots/{shot['id']}/scene", json={"scene": 3}).status_code == 200
    assert client.delete("/api/reference").status_code == 200
