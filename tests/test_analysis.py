"""Shot detection, measurements, white balance estimate and scopes."""
import numpy as np
import pytest

from colorgrader import jobs, library, project, synth
from colorgrader.analysis import measure, neutral, scopes
from colorgrader.color import working as W
from colorgrader.color.gamuts import apply_matrix, conversion_matrix
from colorgrader.color.transforms import OutputTransform

from helpers import flog2_clip, wall_patch

ODT = OutputTransform("rec709-gamma24", 1000.0)


def dwg(scene):
    lib = library.get()
    return apply_matrix(scene, conversion_matrix(lib.gamut("rec709"), lib.gamut(W.WORKING_GAMUT)))


@pytest.fixture(scope="module")
def three_shots(workdir):
    scenes = [synth.procedural_scene(320, 180, seed=1), synth.procedural_scene(320, 180, seed=2, sky=False),
              synth.procedural_scene(320, 180, seed=3, skin=False) * np.array([1.6, 0.7, 0.5], dtype=np.float32)]
    return flog2_clip(workdir / "three_shots.mov", scenes=scenes, frames_each=25)


def test_shot_detection_and_per_shot_analysis(client, three_shots):
    clip = client.post("/api/clips", json={"paths": [str(three_shots)]}).json()["added"][0]
    client.post(f"/api/clips/{clip['id']}/profile", json={"camera_id": "fujifilm-x-m5", "profile_id": "f-log2"})
    job = client.post("/api/analyze", json={"clip_ids": [clip["id"]]}).json()["job"]
    done = jobs.wait(job)
    assert done["state"] == "done", done
    state = client.get("/api/state").json()["project"]
    shots = [s for s in state["shots"] if s["clip_id"] == clip["id"]]
    assert len(shots) == 3
    assert abs(shots[1]["start"] - 1.0) <= 0.05 and abs(shots[2]["start"] - 2.0) <= 0.05
    for s in shots:
        m = s["measure"]
        assert s["frames_analyzed"] >= 1                       # at least one frame per second
        for key in ("exposure", "black_point", "white_point", "clipping", "contrast", "saturation", "palette", "skin", "white_balance"):
            assert key in m
        assert len(m["palette"]) >= 3 and len(s["summary"]) >= 6
    # frames and scopes for every stage the UI shows
    for stage in ("source", "converted"):
        assert client.get(f"/api/frame?clip={clip['id']}&t=0.5&stage={stage}&w=320").content[:2] == b"\xff\xd8"
        for kind in scopes.SCOPES:
            r = client.get(f"/api/scope?clip={clip['id']}&t=0.5&stage={stage}&kind={kind}")
            assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get(f"/api/thumb?shot={shots[0]['id']}").content[:2] == b"\xff\xd8"


def test_analysis_speed_budget(client, three_shots):
    """A one-minute clip must finish in under two minutes. This 3 s clip scales far below that."""
    p = project.get()
    clip = p.add_clip(str(three_shots))
    p.set_profile(clip["id"], "fujifilm-x-m5", "f-log2")
    import time

    t0 = time.time()
    p.analyze_clip(clip["id"])
    per_second_of_footage = (time.time() - t0) / 3.0
    assert per_second_of_footage < 2.0


@pytest.mark.parametrize("gains", [(1.0, 1.0, 1.0), (0.88, 1.0, 1.15), (1.18, 1.0, 0.84), (1.06, 0.92, 1.06)])
def test_white_balance_estimate_recovers_a_known_cast(gains):
    scene = synth.procedural_scene(480, 270)
    lin = dwg(scene * np.array(gains, dtype=np.float32))
    ypos = (np.mgrid[0:270, 0:480][0] / 269.0).ravel().astype(np.float32)
    wb = neutral.estimate_cast(lin.reshape(-1, 3), ypos, ODT, None)
    fixed = lin * neutral.gains_from_cast(wb["dr"], wb["db"])
    wall = wall_patch(fixed)
    assert np.max(np.abs(np.log2(wall / wall[1]))) < 0.035       # the gray wall comes back neutral
    assert wb["confidence"] > 0.5 and "neutral objects" in wb["anchors"]


def test_white_balance_stays_put_without_anchors():
    """A frame with no neutral object, face, sky or foliage keeps the camera's balance."""
    rng = np.random.default_rng(4)
    reds = np.array([0.5, 0.08, 0.06], dtype=np.float32) * (0.4 + rng.random((20000, 1)).astype(np.float32))
    wb = neutral.estimate_cast(dwg(reds), np.full(20000, 0.8, dtype=np.float32), ODT, None)
    assert abs(wb["dr"]) < 0.06 and abs(wb["db"]) < 0.06 and wb["confidence"] < 0.2


def test_skin_anchor_direction():
    """ColorChecker skin patches sit on the skin tone line: no correction asked."""
    lib = library.get()
    light = lib.curve("srgb").decode(np.array([[194, 150, 130]], dtype=np.float32) / 255.0)
    skin = np.repeat(dwg(light), 200, axis=0)
    blues = dwg(np.array([[0.05, 0.08, 0.30]], dtype=np.float32)) * np.linspace(0.5, 1.5, 5000, dtype=np.float32)[:, None]
    wb = neutral.estimate_cast(blues, np.full(5000, 0.8, dtype=np.float32), ODT, skin)
    assert abs(wb["dr"]) < 0.05 and abs(wb["db"]) < 0.05
    ang, _ = neutral.vector_angle(np.array([[194, 150, 130]], dtype=np.float32) / 255.0)
    assert abs(float(ang[0]) - neutral.SKIN_LINE) < 1.0


def test_measurements_read_sensible_numbers():
    scene = synth.procedural_scene(480, 270)
    di = W.di_encode(dwg(scene)).reshape(-1, 3)
    m = measure.measure(di, ODT, None, None)
    assert -2.5 < m["exposure"]["median_stops"] < 1.5
    assert m["black_point"]["level"] < 14 and m["white_point"]["level"] > 75
    assert m["saturation"]["midtones"] > 2
    gray = W.di_encode(np.full((5000, 3), 0.18, dtype=np.float32))
    mg = measure.measure(gray, ODT, None, None, with_cast=False)
    assert abs(mg["exposure"]["median_stops"]) < 0.01 and mg["saturation"]["overall"] < 0.01
    assert 44 < mg["exposure"]["midtone_level"] < 50


def test_scopes_draw_the_signal():
    gray = np.full((90, 160, 3), 0.5, dtype=np.float32)
    wf = scopes.waveform(gray, "luma")
    rows = wf.shape[0]
    line = int(round((scopes.V_HI - 0.5) / (scopes.V_HI - scopes.V_LO) * (rows - 1)))
    assert wf[line - 1:line + 2, :, 1].max(axis=0).mean() > 200    # bright trace on the 50% line
    assert wf[line + 30, :, 1].mean() < 60
    vs = scopes.vectorscope(gray)
    c = vs.shape[0] // 2
    assert vs[c - 1:c + 2, c - 1:c + 2].max() > 200                # neutral gray sits in the center
    red = np.zeros((90, 160, 3), dtype=np.float32)
    red[..., 0] = 0.75
    vr = scopes.vectorscope(red)
    ys, xs = np.nonzero(vr[..., 2] > 200)
    assert ys.mean() < c and xs.mean() < c                         # red: up and left of center
    assert scopes.parade(gray).shape[1] % 3 == 0 and scopes.histogram(gray).shape[0] > 50
