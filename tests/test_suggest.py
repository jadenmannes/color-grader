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


def test_three_looks_with_name_mood_and_preview(proj, workdir):
    clip = load(proj, workdir, "sg_day.mov", day_scene())
    found = suggest.suggest(proj)
    assert len(found) == 3
    assert len({s["family"] for s in found}) == 3                 # three different directions, never three variations of one
    for s in found:
        assert s["name"] and s["mood"].count(".") >= 1 and s["why"]
        assert suggest.thumb_path(s["id"]).stat().st_size > 2000  # the preview picture
        assert s["look"]["source"] == "suggestion" and s["look"]["reasons"]["contrast"].startswith(s["name"])
    assert proj.state["suggestions"] == found
    # every suggestion holds the quality rules on this footage and keeps the footage's exposure
    shot = proj.shots_of(clip["id"])[0]
    base = np.median(np.clip(proj.render_frame(clip["id"], 0.5, "balanced", 480, shot["id"]), 0, 1) @ LUMA)
    for s in found:
        suggest.apply(proj, s["id"])
        report = auto.build(proj, refit=False)
        assert report["totals"]["failed"] == 0, s["name"]
        assert proj.state["look"]["name"] == s["name"]
        graded = np.median(np.clip(proj.render_frame(clip["id"], 0.5, "graded", 480, shot["id"]), 0, 1) @ LUMA)
        assert abs(graded - base) < 0.05, f"{s['name']} moved exposure from {base:.3f} to {graded:.3f}"


def test_the_footage_decides_which_looks_come_up(proj, workdir):
    load(proj, workdir, "sg_day2.mov", day_scene())
    day = [s["id"] for s in suggest.suggest(proj)]
    day_traits = proj.state["scene_profile"]
    proj.reset()
    load(proj, workdir, "sg_night.mov", night_scene())
    night = [s["id"] for s in suggest.suggest(proj)]
    night_traits = proj.state["scene_profile"]
    assert day_traits["daylight"] > 0.8 and day_traits["sky"] > 0.8 and day_traits["low_key"] == 0.0
    assert night_traits["low_key"] > 0.3 and night_traits["daylight"] < 0.1
    assert "vivid-travel" in day and "moonlight" not in day       # sky and greenery: postcard color, no night look
    assert "moonlight" in night and "golden-hour" not in night and "vivid-travel" not in night
    assert day != night


def test_scores_follow_the_traits():
    tpl = {t["id"]: t for t in suggest.load_templates()}
    none = dict.fromkeys(suggest.TRAIT_WORDS, 0.0)
    faces = dict(none, skin=1.0)
    assert suggest.score(tpl["teal-orange"], faces)[0] > suggest.score(tpl["teal-orange"], none)[0]
    assert suggest.score(tpl["vivid-travel"], faces)[0] < suggest.score(tpl["vivid-travel"], none)[0]   # close-ups dislike heavy color
    points, pros = suggest.score(tpl["moonlight"], dict(none, low_key=1.0, cool=1.0))
    assert points > 0.9 and pros[0] == "low_key"
    assert len(tpl) >= 8 and all(t["name"] and t["mood"] and t["family"] for t in tpl.values())


def test_templates_are_data_in_the_user_folder(proj, workdir):
    """A new look template is one entry in a JSON file. No code change."""
    mine = {"templates": [{"id": "my-green", "name": "My green", "family": "custom", "mood": "Green shadows. A test.", "contrast": 1.05,
                           "floor": None, "tints": {"shadows": [120, 0.012], "midtones": [0, 0.0], "highlights": [0, 0.0]},
                           "saturation": 0.95, "rolloff": {"shadows": 0.9, "highlights": 0.9}, "base": 5.0, "suits": {}}]}
    f = paths.sub("looks") / "suggestions.json"
    f.write_text(json.dumps(mine))
    try:
        assert "my-green" in [t["id"] for t in suggest.load_templates()]
        load(proj, workdir, "sg_custom.mov", day_scene())
        found = suggest.suggest(proj)
        assert found[0]["id"] == "my-green"                       # the highest score comes first
        lift = found[0]["look"]["lift"]
        assert lift[1] > lift[0] and lift[1] > lift[2]            # green leads in the shadows
    finally:
        f.unlink()
    f.write_text("not json")
    try:
        assert len(suggest.load_templates()) >= 8                 # a broken user file never breaks the shipped looks
    finally:
        f.unlink()


def test_black_and_white_runs_at_full_strength(proj, workdir):
    load(proj, workdir, "sg_mono.mov", day_scene())
    mono = next(t for t in suggest.load_templates() if t.get("mono"))
    boosted = dict(mono, base=9.0)
    f = paths.sub("looks") / "suggestions.json"
    f.write_text(json.dumps({"templates": [boosted]}))
    try:
        proj.set_strength(60)
        found = suggest.suggest(proj)
    finally:
        f.unlink()
    assert found[0]["id"] == mono["id"] and found[0]["strength"] == 100 and found[0]["look"]["saturation"] == 0.0
    suggest.apply(proj, mono["id"])
    assert proj.state["strength"] == 100                          # part strength would leave color in a monochrome look
    with pytest.raises(KeyError):
        suggest.apply(proj, "no-such-look")


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
