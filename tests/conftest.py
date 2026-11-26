"""Test setup: every test session gets its own empty user folder."""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

_HOME = tempfile.mkdtemp(prefix="colorgrader-test-")
os.environ["COLORGRADER_HOME"] = _HOME
atexit.register(shutil.rmtree, _HOME, ignore_errors=True)   # each run renders clips and exports: hundreds of MB


@pytest.fixture(scope="session")
def home() -> Path:
    return Path(_HOME)


@pytest.fixture(scope="session")
def workdir(home) -> Path:
    d = home / "work"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture(scope="session", autouse=True)
def _seed():
    from colorgrader import cameras, settings

    cameras.seed()
    settings.save({"auto_detail": False})   # tests set noise reduction and sharpening themselves


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from colorgrader import project, server

    project.get().reset()
    return TestClient(server.app)


@pytest.fixture()
def proj():
    from colorgrader import project

    p = project.get()
    p.reset()
    return p
