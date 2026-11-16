"""Camera library: a new camera arrives through the form, as data, with no code change."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from colorgrader import cameras, jobs, paths, project, synth
from colorgrader.color import working as W
from colorgrader.color.gamuts import apply_matrix, conversion_matrix
from colorgrader import library

from helpers import filmsim_clip, flog2_clip, slog3_clip, wall_patch


@pytest.fixture(scope="module")
def slog3(workdir) -> Path:
    return slog3_clip(workdir / "sony_slog3.mov", frames_each=20)


def _code_strings_and_names(path: Path):
    """String literals and identifiers of a Python file, without docstrings and comments."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                doc.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in doc:
            yield node.value
        elif isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            yield node.name


def test_no_camera_specific_code():
    """Camera and log format names appear in data files only, never in code."""
    pattern = re.compile(r"x-m5|xm5|f-log|flog|s-log|slog|v-log|vlog|n-log|nlog|canon|sony|panasonic|nikon|fujifilm|iphone", re.I)
    for f in paths.PACKAGE_DIR.rglob("*.py"):
        hits = [t for t in _code_strings_and_names(f) if pattern.search(t)]
        assert not hits, (f.name, hits[:3])


def test_shipped_cameras_and_user_folder(home):
    ids = {c["id"] for c in cameras.list_cameras()}
    assert {"fujifilm-x-m5", "phone-generic"} <= ids
    assert (home / "cameras" / "fujifilm-x-m5.json").is_file()


def test_unknown_camera_opens_prefilled_form(client, slog3):
    res = client.post("/api/clips", json={"paths": [str(slog3)]}).json()
    clip = res["added"][0]
    assert clip["camera_id"] is None and not clip["confirmed"]
    draft = clip["detect"]["draft"]
    assert draft["name"] == "SONY ILCE-7SM3" and draft["maker"] == "SONY"
    assert draft["metadata_names"] == ["ILCE-7SM3"]
    prof = draft["profiles"][0]
    assert prof["bit_depth"] == 10 and prof["detect"]["transfer"] == ["bt709"]


def test_add_slog3_camera_through_the_form_and_convert(client, slog3, home):
    """The form flow: prefilled draft, pick curve and gamut from the list, convert a sample, save, analyze."""
    lib = client.get("/api/library").json()
    assert "sony-s-log3" in {c["id"] for c in lib["curves"]} and "s-gamut3-cine" in {g["id"] for g in lib["gamuts"]}
    clip = client.post("/api/clips", json={"paths": [str(slog3)]}).json()["added"][0]
    draft = clip["detect"]["draft"]
    draft["profiles"][0]["name"] = "S-Log3 / S-Gamut3.Cine"
    draft["profiles"][0]["input"] = {"type": "curve", "curve": "sony-s-log3", "gamut": "s-gamut3-cine"}
    draft["profiles"][0]["detect"]["transfer"] = []       # the file carries a plain bt709 tag

    # saving a new camera needs a checked sample conversion first
    r = client.post("/api/cameras", json={"camera": draft})
    assert r.status_code == 400 and "sample" in r.json()["detail"].lower()

    prev = client.post("/api/cameras/preview", json={"camera": draft, "profile_index": 0, "sample_path": str(slog3)})
    assert prev.status_code == 200, prev.text
    token = prev.json()["token"]
    for which in ("original", "converted"):
        img = client.get(f"/api/preview/{token}/{which}.jpg")
        assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    saved = client.post("/api/cameras", json={"camera": draft, "preview_token": token})
    assert saved.status_code == 200, saved.text
    cam = saved.json()
    assert cam["id"] == "sony-ilce-7sm3"
    assert (home / "cameras" / "sony-ilce-7sm3.json").is_file()

    # the clip now matches by metadata, with no confirmation needed
    state = client.get("/api/state").json()["project"]
    clip = next(c for c in state["clips"] if c["path"] == str(slog3))
    assert clip["camera_id"] == "sony-ilce-7sm3" and clip["profile_id"] == cam["profiles"][0]["id"] and clip["confirmed"]

    job = client.post("/api/analyze", json={"clip_ids": [clip["id"]]}).json()["job"]
    assert jobs.wait(job)["state"] == "done"
    state = client.get("/api/state").json()["project"]
    shots = [s for s in state["shots"] if s["clip_id"] == clip["id"]]
    assert len(shots) == 1 and shots[0]["measure"]["exposure"]["midtone_level"] > 5

    # conversion accuracy against the scene the clip was recorded from
    p = project.get()
    data = p.shot_data(shots[0]["id"])
    frame = data["frames"][0].astype(np.float32)
    lib_ = library.get()
    truth = apply_matrix(synth.procedural_scene(frame.shape[1], frame.shape[0]),
                         conversion_matrix(lib_.gamut("rec709"), lib_.gamut(W.WORKING_GAMUT)))
    got, want = wall_patch(frame), wall_patch(truth)
    assert np.max(np.abs(np.log2(got / want))) < 0.06          # within 0.06 stops on the gray wall
    assert np.max(np.abs(np.log2(got / got[1]))) < 0.03        # and neutral

    img = client.get(f"/api/frame?clip={clip['id']}&t=0.2&stage=converted&w=320")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"


def test_export_import_edit_delete(client, home):
    assert cameras.get_camera("sony-ilce-7sm3") is not None
    exported = client.get("/api/cameras/sony-ilce-7sm3/export")
    assert exported.status_code == 200
    data = exported.json()
    # the file carries its own curve and gamut definitions
    assert "sony-s-log3" in data["definitions"]["curves"] and "s-gamut3-cine" in data["definitions"]["gamuts"]
    assert client.delete("/api/cameras/sony-ilce-7sm3").json()["deleted"]
    assert cameras.get_camera("sony-ilce-7sm3") is None
    back = client.post("/api/cameras/import", json={"camera": data})
    assert back.status_code == 200 and cameras.get_camera("sony-ilce-7sm3") is not None
    assert client.post("/api/cameras/import", json={"camera": data}).status_code == 400   # exists: needs overwrite
    # edit: set a default tint fix once
    cam = cameras.get_camera("sony-ilce-7sm3")
    cam["profiles"][0]["corrections"]["tint"] = 12
    r = client.post("/api/cameras", json={"camera": cam, "replace_id": cam["id"]})
    assert r.status_code == 200 and cameras.get_camera("sony-ilce-7sm3")["profiles"][0]["corrections"]["tint"] == 12.0


def test_profile_default_correction_applies_to_every_clip(workdir):
    from colorgrader.color.transforms import correction_gains, input_transform_for
    from colorgrader.grade import pipeline

    gains = correction_gains({"tint": 20.0})
    assert gains[1] < gains[0] and abs(gains[0] - gains[2]) < 1e-6            # magenta: green down
    assert abs(float(gains @ np.array([0.2126, 0.7152, 0.0722])) - 1.0) < 1e-6
    prof = {"input": {"type": "curve", "curve": "sony-s-log3", "gamut": "s-gamut3-cine"}, "data_range": "full",
            "corrections": {"exposure": 1.0, "temperature": 0, "tint": 0, "saturation": 1}}
    idt = input_transform_for(prof, True, 10, 1000.0)
    gray = np.full((1, 3), 420.0 / 1023.0, dtype=np.float32)
    lin = pipeline.camera_stage(gray, idt, prof["corrections"])
    assert np.allclose(lin, 0.36, rtol=2e-3)


def test_x_m5_detection(workdir):
    from colorgrader import media

    flog2 = flog2_clip(workdir / "xm5_flog2.mov")
    det = cameras.detect(media.probe(flog2))
    assert det["camera_id"] == "fujifilm-x-m5"
    # F-Log and F-Log2 carry the same tags here: the app estimates and asks
    assert det["needs_confirm"] and {c["profile_id"] for c in det["candidates"][:2]} == {"f-log", "f-log2"}
    sim = filmsim_clip(workdir / "xm5_filmsim.mov")
    det = cameras.detect(media.probe(sim))
    assert det["camera_id"] == "fujifilm-x-m5" and det["profile_id"] == "film-simulation" and not det["needs_confirm"]


def test_maker_lut_profile(client, workdir, home):
    """A camera with no published math: load the maker LUT, say what it outputs."""
    from colorgrader.color.lut import Lut3D, write_cube

    lib = library.get()
    slog3, cine, rec709 = lib.curve("sony-s-log3"), lib.gamut("s-gamut3-cine"), lib.gamut("rec709")

    def maker(v):  # stand-in for an official "log to Rec.709" LUT, without tone mapping
        lin = apply_matrix(slog3.decode(v), conversion_matrix(cine, rec709))
        return lib.curve("rec709-gamma24").encode(np.clip(lin, 0, 1))

    lut_path = write_cube(workdir / "Maker_Log_to_709.cube", Lut3D.from_function(maker, 33))
    stored = client.post("/api/luts", json={"path": str(lut_path)}).json()
    cam = {"name": "Mystery Cam", "metadata_names": ["MYSTERY-1"], "profiles": [{
        "name": "Log via LUT", "bit_depth": 10, "data_range": "full",
        "input": {"type": "lut", "lut_file": stored["lut_file"], "lut_output": {"curve": "rec709-gamma24", "gamut": "rec709"}}}]}
    norm = cameras.normalize(cam)
    from colorgrader.color.transforms import OutputTransform, input_transform_for

    idt = input_transform_for(norm["profiles"][0], True, 10, 1000.0)
    gray = np.full((1, 3), 420.0 / 1023.0, dtype=np.float32)
    out = OutputTransform("rec709-gamma24", 1000.0)(idt(gray))
    assert abs(float(out[0, 1]) - 0.18 ** (1 / 2.4)) < 4e-3     # finished-picture path round-trips the LUT result
    exported_ok = cameras.save_camera(cam)
    data = cameras.export_camera(exported_ok["id"])
    assert "LUT_3D_SIZE 33" in data["profiles"][0]["input"]["lut_text"]
    cameras.delete_camera(exported_ok["id"])


SIDECAR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<NonRealTimeMeta xmlns="urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.20" lastUpdate="2026-10-31T12:30:57+01:00">
  <Duration value="8184"/>
  <VideoFormat>
    <VideoRecPort port="DIRECT"/>
    <VideoFrame videoCodec="AVC_3840_2160_HP@L51" captureFps="25p" formatFps="25p"/>
    <VideoLayout pixel="3840" numOfVerticalLine="2160" aspectRatio="16:9"/>
  </VideoFormat>
  <Device manufacturer="Sony" modelName="ILCE-7CM2" serialNo="0000000"/>
  <Lens modelName="FE 28-60mm F4-5.6"/>
  <AcquisitionRecord>
    <Group name="CameraUnitMetadataSet">
      <Item name="CaptureGammaEquation" value="s-log3-cine"/>
      <Item name="CaptureColorPrimaries" value="S-Gamut3Cine"/>
      <Item name="CodingEquations" value="rec709"/>
    </Group>
  </AcquisitionRecord>
</NonRealTimeMeta>
"""


def test_camera_xml_sidecar_settles_camera_and_profile(workdir):
    """A clip with a camera XML sidecar matches camera and profile from metadata, no confirm needed."""
    from colorgrader import media

    folder = workdir / "xml_sidecar"
    folder.mkdir(exist_ok=True)
    clip = slog3_clip(folder / "C0001.MP4", frames_each=8)
    det = cameras.detect(media.probe(clip))
    assert det["camera_id"] != "sony-a7c-ii"   # no sidecar yet: model unknown
    (folder / "C0001M01.XML").write_text(SIDECAR_XML, encoding="utf-8")
    info = media.probe(clip)
    assert info["model"] == "ILCE-7CM2" and info["tags"]["xml.capturegammaequation"] == "s-log3-cine"
    det = cameras.detect(info)
    assert det["camera_id"] == "sony-a7c-ii"
    assert det["profile_id"] == "s-log3-s-gamut3-cine" and not det["needs_confirm"] and det["method"] == "metadata"


def test_camera_xml_inside_the_clip(workdir):
    """The same XML block embedded in the media file works without a sidecar."""
    from colorgrader import media

    folder = workdir / "xml_embedded"
    folder.mkdir(exist_ok=True)
    clip = slog3_clip(folder / "C0002.mov", frames_each=8)
    data = clip.read_bytes()
    clip.write_bytes(data + SIDECAR_XML.encode("utf-8"))   # trailing bytes after the last atom: decoders ignore them
    tags = media.camera_xml_tags(clip)
    assert tags["xml.device.modelname"] == "ILCE-7CM2" and tags["xml.capturecolorprimaries"] == "S-Gamut3Cine"
