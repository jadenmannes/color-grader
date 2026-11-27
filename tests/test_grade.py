"""Balance, scene match, reference matching, skin protection and quality rules."""
import numpy as np
import pytest

from colorgrader import jobs, library, project, synth
from colorgrader.analysis.neutral import SKIN_LINE
from colorgrader.color import working as W
from colorgrader.color.gamuts import apply_matrix, conversion_matrix
from colorgrader.color.transforms import OutputTransform
from colorgrader.grade import auto, balance, lookfit, match, model, pipeline, quality
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


def test_reference_brightness_follows_or_stays(proj, workdir):
    """Reference brightness at 0 keeps the footage's exposure. At 100 the midtones move toward a dark reference, 1.25 stops at most."""
    scene = synth.procedural_scene(640, 360, skin=False)
    clip = load_clip(proj, flog2_clip(workdir / "day.mov", scenes=[scene], frames_each=25, size=(640, 360)))
    shot = proj.shots_of(clip["id"])[0]
    night = synth.procedural_scene(640, 360, seed=9, sky=False, skin=False)[:, ::-1] * np.array([0.10, 0.13, 0.20], dtype=np.float32)
    lib = library.get()
    di = W.di_encode(apply_matrix(night, conversion_matrix(lib.gamut("rec709"), lib.gamut(W.WORKING_GAMUT))))
    ref = save_png16(workdir / "night_ref.png", ODT(di))
    proj.set_strength(100)
    proj.set_reference(str(ref))
    y = lambda im: float(np.median(im @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
    proj.set_brightness(0)
    report = auto.build(proj)
    assert not report["fit"]["same_content"]
    assert report["fit"]["exposure_gap_stops"] < -1.2 and report["fit"]["midtone_carry"] == 0.0
    assert any("exposure stays" in n.lower() for n in report["notes"])
    bal = proj.render_frame(clip["id"], 0.5, "balanced", 640, shot["id"])
    gra = proj.render_frame(clip["id"], 0.5, "graded", 640, shot["id"])
    assert abs(y(gra) - y(bal)) < 0.05                    # midtones stay within 5 points
    wb, wg = wall_patch(bal), wall_patch(gra)
    assert wg[2] / wg[0] > wb[2] / wb[0] + 0.01           # the cool cast of the reference comes across on the gray wall
    proj.set_brightness(100)
    report = auto.build(proj)
    gap, carry = report["fit"]["exposure_gap_stops"], report["fit"]["midtone_carry"]
    assert 1.0 <= abs(gap) * carry <= 1.3                 # a firm move, capped: day never turns into night
    dark = proj.render_frame(clip["id"], 0.5, "graded", 640, shot["id"])
    assert y(dark) < y(gra) - 0.08


def test_bright_reference_brightens_the_footage(proj, workdir):
    """A high-key reference with other content lifts the midtones of darker footage close to its own level."""
    scene = synth.procedural_scene(640, 360, skin=False) * np.float32(0.6)
    clip = load_clip(proj, flog2_clip(workdir / "dim.mov", scenes=[scene], frames_each=25, size=(640, 360)))
    shot = proj.shots_of(clip["id"])[0]
    bright = synth.procedural_scene(640, 360, seed=9, sky=False, skin=False)[:, ::-1] * np.float32(1.7)
    lib = library.get()
    di = W.di_encode(apply_matrix(bright, conversion_matrix(lib.gamut("rec709"), lib.gamut(W.WORKING_GAMUT))))
    ref_disp = ODT(di)
    ref = save_png16(workdir / "bright_ref.png", ref_disp)
    proj.set_strength(100)
    proj.set_reference(str(ref))
    report = auto.build(proj)
    y = lambda im: float(np.median(np.clip(im, 0, 1) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
    bal = proj.render_frame(clip["id"], 0.5, "balanced", 640, shot["id"])
    gra = proj.render_frame(clip["id"], 0.5, "graded", 640, shot["id"])
    gap = report["fit"]["exposure_gap_stops"]
    assert gap > 0.3
    assert y(gra) > y(bal) + 0.03
    assert abs(y(gra) - y(ref_disp)) < 0.6 * abs(y(bal) - y(ref_disp))   # most of the distance covered
    assert not any(r["status"] == "fail" and r["id"] in ("highlights", "color_clip") for r in proj.shots_of(clip["id"])[0]["rules"])


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


# ------------------------------------------------------- people differ
def _skin_lin(level_stops=0.0, n=400):
    """Linear working-space skin samples (ColorChecker light skin) shifted by level_stops: a person with lighter or
    darker skin under the same light."""
    return (W.di_decode(checker_skin(n)) * np.float32(2.0 ** level_stops)).astype(np.float32)


def test_scene_match_keeps_each_persons_skin_brightness(proj, workdir):
    """Two people in one scene under the same light, one with skin 1.7 stops darker. The scene match used to lift the
    darker person's whole shot by a stop to give both the same skin brightness, washing their skin out."""
    scene = synth.procedural_scene(640, 360, skin=False)
    clip = load_clip(proj, flog2_clip(workdir / "people.mov", scenes=[scene, scene[:, ::-1]], frames_each=25, size=(640, 360)))
    shots = proj.shots_of(clip["id"])
    assert len(shots) == 2
    for sh, stops in zip(shots, (0.0, -1.7)):
        proj._shot_cache[sh["id"]] = dict(proj.shot_data(sh["id"]), skin=_skin_lin(stops))
    auto.build(proj, refit=False)
    shots = proj.shots_of(clip["id"])
    assert shots[0]["scene"] == shots[1]["scene"]                            # one scene, as in the user's edit
    ex = [s["grade"]["info"]["exposure"] for s in shots]
    assert abs(ex[0] - ex[1]) < 0.1, ex                                       # the same light keeps the same exposure
    gain_text = " ".join(s["grade"]["reasons"]["gain"] for s in shots)
    assert "different person" in gain_text and "skin brightness matched" not in gain_text


def test_scene_match_still_matches_one_person_filmed_darker():
    hero = {"median": -0.4, "high": 2.0, "skin": -0.9}
    shot = {"median": -1.1, "high": 1.3, "skin": -1.6}                       # everything 0.7 stops under, skin too
    trim, why = match.exposure_trim(hero, shot)
    assert abs(trim - 0.56) < 0.01 and "skin brightness matched" in why
    trim, why = match.exposure_trim(hero, {"median": -0.45, "high": 2.0, "skin": -2.6})   # same picture, skin 1.7 darker
    assert trim == 0.0 and "different person" in why
    # a darker person in a darker frame under the same light: the middle tones drop with the content, the whites stay
    trim, why = match.exposure_trim(hero, {"median": -1.7, "high": 1.95, "skin": -2.6})
    assert "different person" in why and "skin brightness matched" not in why and trim < 0.5
    trim, _ = match.exposure_trim(hero, {"median": -0.4, "high": 2.0, "skin": -1.1})      # small differences align either way
    assert abs(trim - 0.16) < 0.01


def test_scene_match_keeps_a_darker_person_in_a_darker_frame(proj, workdir):
    """Both shots exposed right under the same light: the second holds more dark content (a dark hoodie, shade) and a
    person with skin 1.7 stops darker. The middle tones differ about as much as the skin, the whites do not."""
    scene = synth.procedural_scene(640, 360, skin=False)
    dark = scene.copy()
    dark[150:, :] *= np.float32(2.0 ** -1.8)                                  # the lower part: dark clothes and shade
    clip = load_clip(proj, flog2_clip(workdir / "people_dark.mov", scenes=[scene, dark[:, ::-1]], frames_each=25, size=(640, 360)))
    shots = proj.shots_of(clip["id"])
    assert len(shots) == 2
    for sh, stops in zip(shots, (0.0, -1.7)):
        proj._shot_cache[sh["id"]] = dict(proj.shot_data(sh["id"]), skin=_skin_lin(stops))
        proj.set_scene(sh["id"], 1)                                            # one scene, as in the user's edit
    auto.build(proj, refit=False)
    shots = proj.shots_of(clip["id"])
    assert shots[0]["scene"] == shots[1]["scene"]
    gain_text = " ".join(s["grade"]["reasons"]["gain"] for s in shots)
    assert "skin brightness matched" not in gain_text and "different person" in gain_text, gain_text


def test_darker_skin_exposed_right_is_not_lifted():
    """With a gray card at 0.18, the darkest skin (about 5% reflectance) sits near 0.05, and half that in partial shadow.
    The old floor of 0.07 lifted it."""
    gray = np.full((4000, 3), 0.18, np.float32) * np.random.default_rng(2).uniform(0.5, 2.0, (4000, 1)).astype(np.float32)
    for level in (0.05, 0.025):
        sk = _skin_lin(float(np.log2(level / 0.18)) - 0.0)
        sk *= np.float32(level / float(np.median(sk @ balance.DWG_Y)))
        e, why = balance.exposure_decision(gray, sk, False)
        assert not any("skin" in w for w in why), (level, e, why)
    e, why = balance.exposure_decision(gray * np.float32(0.15), _skin_lin() * np.float32(0.03), False)   # truly underexposed
    assert e > 0.5


def test_an_edited_file_keeps_each_shots_own_white_balance(proj, workdir):
    """Shots of one camera recording share its white balance. An edited master (no camera make or model) cuts street
    and boat recordings together: sharing pulled the street shots toward the boat's correction."""
    scene = synth.procedural_scene(640, 360, skin=False)
    warm = (scene * np.array([1.15, 1.0, 0.85], np.float32)).astype(np.float32)
    casts = {}
    for name, tags in (("camera", None), ("edit", {})):
        proj.reset()
        kw = {} if tags is None else {"tags": tags}
        clip = load_clip(proj, flog2_clip(workdir / f"wb_{name}.mov", scenes=[scene, warm[:, ::-1]], frames_each=25, size=(640, 360), **kw))
        shots = proj.shots_of(clip["id"])
        for sh in shots:
            proj.set_scene(sh["id"], 1)
        auto.build(proj, refit=False)
        casts[name] = [np.array(s["grade"]["info"]["cast_applied"]) for s in proj.shots_of(clip["id"])]
    spread = lambda c: float(np.abs(c[0] - c[1]).max())
    assert spread(casts["edit"]) > spread(casts["camera"]) + 0.02, casts        # the camera file shares, the edit does not


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


def test_scope_readout_reads_skin_on_the_balanced_and_graded_stages(client, workdir):
    """view_shot and read_shot crashed on every shot with a face (Project.odt() got the skin pixels as an argument).
    Synthetic faces never reach the face detector, so the shot gets the skin samples an interview gives it."""
    proj = project.get()
    clip = load_clip(proj, flog2_clip(workdir / "faces.mov", scenes=[synth.procedural_scene(320, 180)], frames_each=8))
    auto.build(proj, refit=False)
    shot = proj.shots_of(clip["id"])[0]
    skin = W.di_decode(checker_skin(60)).astype(np.float32)                  # linear, as the analysis stores it
    proj._shot_cache[shot["id"]] = dict(proj.shot_data(shot["id"]), skin=skin)
    for stage in ("graded", "balanced"):
        r = client.get("/api/readout", params={"shot": shot["id"], "stage": stage})
        assert r.status_code == 200, r.text
        sk = r.json()["skin"]
        assert 20 < sk["luma_percent"] < 90 and sk["saturation_percent"] > 0 and sk["leans"]
    assert "skin" not in client.get("/api/readout", params={"shot": shot["id"], "stage": "source"}).json()


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


def test_burned_in_graphics_keep_their_color():
    """A saturated title color decoded through a log curve stays a color: no black patches, no negative log values."""
    from colorgrader.grade import pipeline
    from colorgrader.color import working as W2

    lin = np.array([[16.2, -0.27, 1.46], [7.5, -0.08, 1.07], [0.4, 0.4, 0.4], [-0.01, -0.01, -0.01]], dtype=np.float32)
    out = pipeline.limit_gamut(lin)
    assert out[:2].min() >= -1e-6 and W2.di_encode(out[:2]).min() >= -1e-3
    assert out[0, 0] > out[0, 2] > out[0, 1]                       # still pink: red over blue over green
    assert np.allclose(out[2:], lin[2:])                           # in-gamut colors and sub-black values untouched
