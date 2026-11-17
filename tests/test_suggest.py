"""Look suggestions for footage with no reference."""
import json

import numpy as np
import pytest

from colorgrader import jobs, paths, synth
from colorgrader.grade import auto, model, suggest

from helpers import flog2_clip, load_clip

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def day_scene():
    return synth.procedural_scene(480, 270, skin=False)


def night_scene():
    return synth.procedural_scene(480, 270, seed=9, sky=False, skin=False) * np.array([0.05, 0.07, 0.11], dtype=np.float32)


def load(proj, workdir, name, scene):
    return load_clip(proj, flog2_clip(workdir / name, scenes=[scene], frames_each=25, size=(480, 270)))


def test_three_looks_with_name_mood_steps_and_preview(proj, workdir):
    clip = load(proj, workdir, "sg_day.mov", day_scene())
    found = suggest.suggest(proj)
    assert len(found) == 3
    assert len({s["family"] for s in found}) == 3                 # three different directions, never three variations of one
    assert found[0]["family"] == "natural"                        # the clean interview grade always comes first
    for s in found:
        assert s["name"] and s["mood"].count(".") >= 1 and s["why"]
        assert suggest.thumb_path(s["id"]).stat().st_size > 2000  # the preview picture
        assert s["look"]["source"] == "suggestion" and s["look"]["reasons"]["contrast"].startswith(s["name"])
        assert [st["step"][0] for st in s["steps"]] == list("1234567")   # the colorist workflow, step by step
        assert all(st["value"] and st["why"] for st in s["steps"])
    assert proj.state["suggestions"] == found and proj.state["scene_read"]["lines"]
    # every suggestion holds the quality rules; exposure moves only as far as the look says
    shot = proj.shots_of(clip["id"])[0]
    base = np.median(np.clip(proj.render_frame(clip["id"], 0.5, "balanced", 480, shot["id"]), 0, 1) @ LUMA)
    for s in found:
        suggest.apply(proj, s["id"])
        report = auto.build(proj, refit=False)
        assert report["totals"]["failed"] == 0, s["name"]
        assert proj.state["look"]["name"] == s["name"]
        graded = np.median(np.clip(proj.render_frame(clip["id"], 0.5, "graded", 480, shot["id"]), 0, 1) @ LUMA)
        allowed = 0.06 + 0.15 * abs(s["look"].get("exposure", 0.0))
        assert abs(graded - base) < allowed, f"{s['name']} moved exposure from {base:.3f} to {graded:.3f}"


def _scene(**kw):
    """A scene read with chosen traits, for direction tests."""
    traits = dict.fromkeys(["skin", "soft_light", "hard_light", "cool_light", "warm_light", "autumn", "greenery", "blue",
                            "muted", "colorful", "low_key", "high_key", "bright_top", "green_cast", "overcast"], 0.0)
    traits.update(kw)
    mass = np.full(12, 1.0 / 12)
    return {"traits": traits, "spread": 0.4, "chroma": 0.08, "top": 0.85, "profile": {"mass": mass}}


def test_the_shot_decides_which_looks_come_up():
    from colorgrader.grade import design

    pick = lambda sc: [d["name"] for d in design.pick(design.directions(sc))]
    autumn = pick(_scene(skin=1.0, autumn=1.0, overcast=0.8, soft_light=0.8, cool_light=0.5))
    sunny = pick(_scene(skin=1.0, blue=1.0, hard_light=1.0))
    gray = pick(_scene(skin=1.0, overcast=1.0, soft_light=1.0, cool_light=0.6))
    assert autumn[0] == sunny[0] == gray[0] == "Clean interview"
    assert "Warm autumn" in autumn and "Cinematic teal and orange" in sunny and "Warm overcast" in gray
    assert "Bright and airy" not in sunny                          # hard sun has no room for a lifted, airy look
    assert autumn != sunny


def test_look_numbers_come_from_the_measurements():
    from colorgrader.grade import design

    cool = {d["id"]: d for d in design.directions(_scene(skin=1.0, overcast=1.0, cool_light=1.0))}
    neutral = {d["id"]: d for d in design.directions(_scene(skin=1.0))}
    assert cool["warm"]["tints"]["highlights"][1] > neutral["warm"]["tints"]["highlights"][1]   # cooler light, more warmth back
    dull = {d["id"]: d for d in design.directions(dict(_scene(skin=1.0), chroma=0.05))}
    vivid = {d["id"]: d for d in design.directions(dict(_scene(skin=1.0), chroma=0.14))}
    assert dull["clean"]["hsv"]["gain"] > 1.0 > vivid["clean"]["hsv"]["gain"]                 # quiet color gets more, loud color less
    sparse = _scene(skin=1.0)
    sparse["profile"] = {"mass": np.array([0.5, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)}
    moves = {d["id"]: d for d in design.directions(sparse)}["cinematic"]["hue_vs_hue"]
    assert 60 in moves and 240 not in moves and 120 not in moves  # only colors the shot holds get moved


def test_suggestions_follow_the_chosen_shot(proj, workdir):
    day = load(proj, workdir, "sg_pick_day.mov", day_scene())
    night = load(proj, workdir, "sg_pick_night.mov", night_scene())
    d_shot, n_shot = proj.shots_of(day["id"])[0]["id"], proj.shots_of(night["id"])[0]["id"]
    suggest.suggest(proj, shot_id=d_shot)
    day_read = proj.state["scene_read"]
    suggest.suggest(proj, shot_id=n_shot)
    night_read = proj.state["scene_read"]
    assert day_read["shot_id"] == d_shot and night_read["shot_id"] == n_shot
    assert day_read["traits"]["low_key"] < night_read["traits"]["low_key"]
    assert day_read["lines"] != night_read["lines"]
    with pytest.raises(KeyError):
        suggest.apply(proj, "no-such-look")


def test_cinematic_look_follows_the_six_step_workflow(proj, workdir):
    """Split tone and hue compression of the cinematic direction, built on real footage."""
    from colorgrader.grade import design

    load(proj, workdir, "sg_cine.mov", day_scene())
    auto.build(proj)
    b = auto.prepared_builder(proj)
    sc = design.read_scene(b)
    d = next(x for x in design.directions(sc) if x["id"] == "cinematic")
    look = design.build_look(b, d, 0.7)
    assert look["tone_curve"]                                      # contrast comes from a soft S-curve
    assert look["lift"][0] < look["lift"][1] and look["lift"][0] < look["lift"][2]    # red pulled down in the shadows: teal
    assert look["gain"][2] < look["gain"][0]                       # blue pulled down in the highlights: warm
    assert look["exposure"] <= 0.0                                 # darker reads richer
    assert look["hsv_sat"]["gain"] > 0.9 and not look["sat_vs_lum"]   # saturation from the HSV node
    turn = {round(h): v for h, v in look["hue_vs_hue"]}
    assert all(turn.get(h, 0) <= 0 for h in (240, 120)) and turn.get(300, 0) >= 0


def test_build_without_reference_proposes_looks(client, workdir):
    from colorgrader import project

    p = project.get()
    load(p, workdir, "sg_api.mov", day_scene())
    assert client.get("/api/state").json()["features"]["suggestions"]
    job = jobs.wait(client.post("/api/grade/build", json={}).json()["job"])
    assert job["state"] == "done"
    state = client.get("/api/state").json()["project"]
    assert len(state["suggestions"]) == 3 and model.is_neutral_look(state["look"])   # proposed, never applied unasked
    first = state["suggestions"][0]
    assert client.get(f"/api/suggestions/{first['id']}/thumb.jpg").headers["content-type"] == "image/jpeg"
    job = jobs.wait(client.post(f"/api/suggestions/{first['id']}/apply").json()["job"])
    assert job["state"] == "done" and job["result"]["totals"]["failed"] == 0
    state = client.get("/api/state").json()["project"]
    assert state["look"]["name"] == first["name"] and state["look"]["source"] == "suggestion"
    assert first["name"] in client.get("/api/recipe").json()["title"]
    assert client.post("/api/suggestions/nope/apply").status_code == 404
    job = jobs.wait(client.post("/api/suggestions", json={}).json()["job"])       # ask again on demand
    assert job["state"] == "done" and job["result"]["count"] == 3
    client.delete(f"/api/clips/{p.state['clips'][0]['id']}")
    assert client.get("/api/state").json()["project"]["suggestions"] == []        # suggestions belong to their footage
