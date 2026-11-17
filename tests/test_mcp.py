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

    def tool(self, name, **args):
        r = self.call("tools/call", {"name": name, "arguments": args})
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
