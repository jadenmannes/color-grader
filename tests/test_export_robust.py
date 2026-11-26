"""The export never hangs and never fails in silence: ffmpeg logs to a file, a guard stops an export that stopped
moving, the decoder and the finished file get checked, and a failed color rebuild renders again without it."""
import sys
import time
from pathlib import Path

import pytest

from colorgrader import export, jobs, media, settings, synth
from colorgrader.grade import auto

from helpers import flog2_clip, load_clip
from test_chroma import _hard_export


def _hard_clip(proj, workdir, name):
    """A hard-compressed 8-bit clip: the color rebuild turns on by itself."""
    clean = flog2_clip(workdir / f"{name}_clean.mov", scenes=[synth.procedural_scene(640, 360, seed=3)], frames_each=16, size=(640, 360))
    clip = load_clip(proj, _hard_export(clean, workdir / f"{name}.mp4"))
    auto.build(proj, refit=False)
    assert proj.color_rebuild_on(proj.clip(clip["id"]))
    return proj.clip(clip["id"])


def test_a_stalled_rebuild_stops_and_the_clip_exports_without_it(proj, workdir, tmp_path, monkeypatch):
    clip = _hard_clip(proj, workdir, "stall")
    monkeypatch.setattr(export, "STALL_SECONDS", 3.0)
    monkeypatch.setattr(export, "_decoder_command", lambda path, pre: [sys.executable, "-c", "import time; time.sleep(120)"])
    msgs = []
    t0 = time.time()
    res = export.render_clip(proj, clip["id"], tmp_path / "stall.mp4", fmt="h265-10bit-mp4", progress=lambda f, m: msgs.append(m))
    assert time.time() - t0 < 60                                         # stopped, never hung
    assert res["note"].startswith("Exported without the color rebuild") and "stopped making progress" in res["note"]
    assert res["color_rebuild"] is False
    assert "rendering again without it" in " ".join(msgs) and any(m.startswith("Frame ") for m in msgs)
    out = media.probe(res["path"])
    assert abs(out["duration"] - clip["info"]["duration"]) < 0.1


def test_a_decoder_failure_falls_back_with_the_reason(proj, workdir, tmp_path, monkeypatch):
    clip = _hard_clip(proj, workdir, "nodec")
    missing = str(workdir / "missing.mp4")
    monkeypatch.setattr(export, "_decoder_command",
                        lambda path, pre: [media.find_binary("ffmpeg"), "-v", "error", "-nostdin", "-i", missing, "-f", "rawvideo", "pipe:1"])
    res = export.render_clip(proj, clip["id"], tmp_path / "nodec.mp4", fmt="h265-10bit-mp4")
    assert "decoding the source" in res["note"]
    assert media.probe(res["path"])["frames"] > 0


def test_the_rebuild_export_logs_both_commands(proj, workdir, tmp_path):
    clip = _hard_clip(proj, workdir, "logged")
    res = export.render_clip(proj, clip["id"], tmp_path / "logged.mp4", fmt="h265-10bit-mp4")
    assert res["color_rebuild"] is True and "note" not in res
    log = Path(res["log"]).read_text()
    assert "encode: " in log and "decode: " in log and "pipe:0" in log


def test_ffmpeg_failures_read_plainly_and_point_to_the_log(proj, workdir, tmp_path):
    clip = load_clip(proj, flog2_clip(workdir / "fail10.mov", scenes=[synth.procedural_scene(320, 180)], frames_each=6))
    auto.build(proj, refit=False)
    target = tmp_path / "folder.mp4"
    target.mkdir()                                                       # ffmpeg cannot write a file where a folder is
    with pytest.raises(export.ExportError) as err:
        export.render_clip(proj, clip["id"], target, fmt="h265-10bit-mp4")
    assert "ffmpeg failed" in str(err.value) and "ffmpeg.log" in str(err.value)
    assert export._failure_text("[out#0] Error opening output a.mp4: Permission denied", Path("C:/Videos/a.mp4")) == (
        "Windows did not let ffmpeg write a.mp4. Close any player or editor that has it open, then export again.")
    assert "is full" in export._failure_text("No space left on device", Path("/x/a.mp4"))
    assert "out of memory" in export._failure_text("Cannot allocate memory", Path("/x/a.mp4"))
    assert export._exit_code(-1073741819) == "0xc0000005" and export._exit_code(1) == "1"


def test_a_short_file_counts_as_a_failed_export(proj, workdir, tmp_path):
    clip = load_clip(proj, flog2_clip(workdir / "short10.mov", scenes=[synth.procedural_scene(320, 180)], frames_each=6))
    auto.build(proj, refit=False)
    res = export.render_clip(proj, clip["id"], tmp_path / "short.mp4", fmt="h264-8bit-mp4")
    info = dict(clip["info"], duration=30.0)
    with pytest.raises(export.ExportError, match="stopped early"):
        export._check_output(Path(res["path"]), info, Path("ffmpeg.log"))
    with pytest.raises(export.ExportError, match="no readable file"):
        export._check_output(tmp_path / "nothing.mp4", info, Path("ffmpeg.log"))


def test_progress_counts_frames_and_the_time_left():
    now = [100.0]
    w = export._Watch(1000, clock=lambda: now[0])
    w.frame(10)
    assert w.text() == "Frame 10 of 1000"                                  # too early to tell
    now[0] += 60
    w.frame(110)                                                          # 100 frames a minute: 890 left
    assert w.text() == "Frame 110 of 1000 · about 9 min left"
    now[0] += 480
    w.frame(910)
    assert w.text() == "Frame 910 of 1000 · under a minute left"
    assert export._left(2 * 3600) == "about 2.0 h left"


def test_the_last_export_stays_available(client, workdir, tmp_path):
    from colorgrader import project

    proj = project.get()
    clip = load_clip(proj, flog2_clip(workdir / "last10.mov", scenes=[synth.procedural_scene(320, 180)], frames_each=6))
    auto.build(proj, refit=False)
    settings.save({"export_folder": str(tmp_path / "out")})
    try:
        jid = client.post("/api/export/video", json={"clip_ids": [clip["id"]], "format": "h264-8bit-mp4"}).json()["job"]
        jobs.wait(jid, 120)
        last = client.get("/api/export/last").json()
        assert last["id"] == jid and last["state"] == "done" and last["result"]["files"][0]["name"].endswith("_graded.mp4")
        jid = client.post("/api/export/video", json={"clip_ids": ["no-such-clip"]}).json()["job"]
        jobs.wait(jid, 30)
        last = client.get("/api/export/last").json()
        assert last["id"] == jid and last["state"] == "error" and "No video clips" in last["error"]
    finally:
        settings.save({"export_folder": ""})
