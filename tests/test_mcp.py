"""The MCP server: an assistant sees shots, changes the grade live, and undoes, over stdin and stdout."""
import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from colorgrader.grade import auto

from helpers import flog2_clip, load_clip


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def app_url(proj):
    import uvicorn

    from colorgrader import server

    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=5)


class Client:
    def __init__(self, url: str):
        env = dict(os.environ, COLORGRADER_URL=url)
        self.p = subprocess.Popen([sys.executable, "-m", "colorgrader.mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        self.n = 0

    def call(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            self.n += 1
            msg["id"] = self.n
        self.p.stdin.write(json.dumps(msg).encode() + b"\n")
        self.p.stdin.flush()
        if notify:
            return None
        return json.loads(self.p.stdout.readline())

    def tool(self, tool_name, /, **args):
        r = self.call("tools/call", {"name": tool_name, "arguments": args})
        return r["result"]

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)


def test_mcp_server_sees_edits_and_undoes(proj, workdir, app_url):
    clip = load_clip(proj, flog2_clip(workdir / "mcp.mov", frames_each=20, size=(320, 180)))
    auto.build(proj, refit=False)
    shot = proj.shots_of(clip["id"])[0]
    c = Client(app_url)
    try:
        init = c.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}})
        assert init["result"]["serverInfo"]["name"] == "color-grader" and init["result"]["capabilities"]["tools"]
        c.call("notifications/initialized", notify=True)
        names = {t["name"] for t in c.call("tools/list")["result"]["tools"]}
        assert {"project_status", "view_shot", "adjust_look", "adjust_shot", "set_finish", "design_looks", "undo"} <= names

        status = json.loads(c.tool("project_status")["content"][0]["text"])
        assert status["shots"][0]["id"] == shot["id"]

        seen = c.tool("view_shot", shot_id=shot["id"], width=320)
        images = [x for x in seen["content"] if x["type"] == "image"]
        assert len(images) == 2 and base64.b64decode(images[0]["data"])[:2] == b"\xff\xd8"   # source and graded JPEGs
        readout = json.loads(seen["content"][-1]["text"])
        assert "luma_percent" in readout and readout["shot_id"] == shot["id"]
        assert proj.focus()["shot_id"] == shot["id"]                                          # the app's viewer follows

        before = proj.state["look"]["contrast"]
        res = c.tool("adjust_look", contrast=round(before + 0.2, 3), hsv_gamma=0.02, reason="More punch")
        assert not res["isError"]
        assert abs(proj.state["look"]["contrast"] - (before + 0.2)) < 1e-6                   # live in the app
        assert proj.state["look"]["hsv_sat"]["gamma"] == 0.02
        gain_before = list(proj.shot(shot["id"])["grade"]["balance"]["gain"])
        assert not c.tool("adjust_shot", shot_id=shot["id"], exposure_stops=0.5)["isError"]
        assert proj.shot(shot["id"])["grade"]["balance"]["gain"][1] > gain_before[1] * 1.3
        assert not c.tool("set_finish", relight_on=True, relight_stops=0.7)["isError"]
        assert proj.state["finish"]["relight"] == {"enabled": True, "amount": 0.7, "subject": 0.0}

        for _ in range(3):
            assert not c.tool("undo")["isError"]
        assert abs(proj.state["look"]["contrast"] - before) < 1e-6
        assert proj.shot(shot["id"])["grade"]["balance"]["gain"] == gain_before
        assert c.tool("undo")["isError"]                                                     # nothing left to undo
        bad = c.tool("view_shot", shot_id="no-such-shot")
        assert bad["isError"] and "No shot" in bad["content"][0]["text"]
    finally:
        c.close()


def test_mcp_server_says_when_the_app_is_not_running():
    c = Client("http://127.0.0.1:9")
    try:
        c.call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}})
        res = c.tool("project_status")
        assert res["isError"] and "not running" in res["content"][0]["text"]
        assert c.call("nope/method")["error"]["code"] == -32601
    finally:
        c.close()


def test_mcp_server_drives_every_control_secondaries_and_undo(proj, workdir, app_url):
    clip = load_clip(proj, flog2_clip(workdir / "mcp_all.mov", frames_each=12, size=(320, 180)))
    auto.build(proj, refit=False)
    shot = proj.shots_of(clip["id"])[0]
    temp0, trim0 = proj.state["look"]["temp"], proj.shot(shot["id"])["grade"]["trim"]
    c = Client(app_url)
    try:
        c.call("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}})
        names = {t["name"] for t in c.call("tools/list")["result"]["tools"]}
        assert {"list_controls", "set_secondary", "view_key", "sample_color", "reset_controls"} <= names
        spec = json.loads(c.tool("list_controls")["content"][0]["text"])
        assert {"hdr", "log", "rgb_mixer", "secondaries"} <= {x["key"] for x in spec["controls"]}

        res = c.tool("adjust_look", temp=0.25, tint=-0.05, offset=[24.8, 25, 25.2], hdr={"highlight": {"exposure": -0.4}},
                     log={"shadow": [0, 0, 0.01]}, curves_rgb={"b": [[0, 0.01], [1, 1]]}, color_boost=0.2, shadows=0.3, reason="Autumn")
        assert not res["isError"], res
        look = proj.state["look"]
        assert look["temp"] == 0.25 and look["hdr"]["highlight"]["exposure"] == -0.4 and look["log"]["shadow"] == [0, 0, 0.01]
        assert look["curves_rgb"]["b"] == [[0.0, 0.01], [1.0, 1.0]] and look["reasons"]["temp"] == "Autumn"
        step = [e for e in proj.activity() if e["tool"] == "adjust_look"][-1]          # the app's viewer shows this step
        temp = next(c for c in step["controls"] if c["key"] == "temp")
        assert temp["before"] == temp0 and temp["after"] == 0.25 and step["target"] == "look" and step["reason"] == "Autumn"
        assert "temperature +0.25" in step["title"]
        assert not c.tool("adjust_shot", shot_id=shot["id"], trim={"exposure": 0.3, "highlights": -0.4})["isError"]
        assert proj.shot(shot["id"])["grade"]["trim"] == {"exposure": 0.3, "highlights": -0.4}
        bad = c.tool("adjust_shot", shot_id=shot["id"], trim={"brightness": 1})
        assert bad["isError"] and "Unknown control" in bad["content"][0]["text"]

        sample = json.loads(c.tool("sample_color", shot_id=shot["id"], x=0.3, y=0.4)["content"][0]["text"])
        assert "suggested_qualifier" in sample and "graded_luma_percent" in sample
        res = c.tool("set_secondary", name="Picked", pick={"x": 0.3, "y": 0.4, "shot_id": shot["id"]}, saturation=0.7)
        assert not res["isError"], res
        info = json.loads(res["content"][0]["text"])
        assert info["key_coverage_percent"] > 0.5 and res["content"][1]["type"] == "image"
        sec = proj.state["look"]["secondaries"][0]
        assert sec["name"] == "Picked" and sec["adjust"]["saturation"] == 0.7 and sec["lum"]
        picked = [e for e in proj.activity() if e["tool"] == "set_secondary"]
        assert picked[0]["point"] == {"x": 0.3, "y": 0.4} and picked[-1]["key"]["index"] == 0
        key = c.tool("view_key", shot_id=shot["id"], index=1, mode="matte")
        assert not key["isError"] and key["content"][1]["type"] == "image" and "selected" in key["content"][0]["text"]
        assert c.tool("view_key", shot_id=shot["id"], index=2)["isError"]
        assert not c.tool("set_secondary", index=1, remove=True)["isError"]
        assert proj.state["look"]["secondaries"] == []

        assert not c.tool("reset_controls", controls=["temp", "hdr"])["isError"]
        assert proj.state["look"]["temp"] == 0.0 and proj.state["look"]["hdr"]["highlight"]["exposure"] == 0.0
        assert proj.state["look"]["shadows"] == 0.3                                   # the other controls stay
        assert not c.tool("reset_controls", shot_id=shot["id"])["isError"]
        assert proj.shot(shot["id"])["grade"]["trim"] == {}

        for _ in range(2):                                                            # undo both resets
            assert not c.tool("undo")["isError"]
        assert proj.state["look"]["temp"] == 0.25 and proj.shot(shot["id"])["grade"]["trim"] == {"exposure": 0.3, "highlights": -0.4}
        assert not c.tool("undo")["isError"]                                          # undo the removal
        assert proj.state["look"]["secondaries"][0]["name"] == "Picked"
        for _ in range(3):
            assert not c.tool("undo")["isError"]
        assert proj.state["look"]["temp"] == temp0 and proj.shot(shot["id"])["grade"]["trim"] == trim0
        assert c.tool("undo")["isError"]
    finally:
        c.close()


def test_instructions_hold_no_skin_brightness_target():
    """The assistant once read 'skin around 40 to 70 percent luma' as a target and brightened darker skin toward it."""
    from colorgrader import mcp

    text = mcp.INSTRUCTIONS.lower()
    assert "40 to 70" not in text and "skin brightness belongs to the person" in text
    assert "white and gray objects" in text and "never set one person's skin to another person's brightness" in text
