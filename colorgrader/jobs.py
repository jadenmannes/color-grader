"""Background jobs with progress, for analysis and exports."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Callable

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()
# One job at a time. Decoding 4K footage, building a grade and rendering an export each use
# several gigabytes at their peak. Side by side they run a laptop out of memory.
_turn = threading.Lock()


def _friendly(exc: BaseException, label: str) -> str:
    """A readable message for out-of-memory failures. Windows reports them as WinError 8 or 1450."""
    win = getattr(exc, "winerror", None)
    if isinstance(exc, MemoryError) or win in (8, 14, 1450) or getattr(exc, "errno", None) == 12:
        return (f"Your computer ran out of memory during {label.lower() or 'this job'}. Close other programs and run it again. "
                "The app runs one job at a time, so a second try usually works.")
    return str(exc)


def start(kind: str, fn: Callable[[Callable[[float, str], None]], Any], label: str = "") -> str:
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "kind": kind, "label": label, "state": "running", "progress": 0.0, "message": "", "result": None,
           "error": None, "started": time.time(), "finished": None}
    with _lock:
        _jobs[jid] = job

    def progress(frac: float, message: str = "") -> None:
        job["progress"] = float(min(1.0, max(0.0, frac)))
        if message:
            job["message"] = message

    def run() -> None:
        if not _turn.acquire(blocking=False):
            job["message"] = "Waiting for the job before it to finish"
            _turn.acquire()
            job["message"] = ""
        try:
            job["result"] = fn(progress)
            job["state"] = "done"
            job["progress"] = 1.0
        except Exception as exc:  # report to the UI, keep the server alive
            job["state"] = "error"
            job["error"] = _friendly(exc, label)
            job["trace"] = traceback.format_exc()
        finally:
            job["finished"] = time.time()
            _turn.release()

    threading.Thread(target=run, daemon=True, name=f"job-{kind}-{jid}").start()
    return jid


def get(jid: str) -> dict[str, Any] | None:
    with _lock:
        job = _jobs.get(jid)
    if job is None:
        return None
    return {k: v for k, v in job.items() if k != "trace"}


def latest(kind: str) -> dict[str, Any] | None:
    """The newest job of one kind, running or finished. The Export panel shows it again after a page reload."""
    with _lock:
        found = [j for j in _jobs.values() if j["kind"] == kind]
    if not found:
        return None
    return {k: v for k, v in max(found, key=lambda j: j["started"]).items() if k != "trace"}


def active() -> list[dict[str, Any]]:
    with _lock:
        return [{k: v for k, v in j.items() if k not in ("trace", "result")} for j in _jobs.values() if j["state"] == "running"]


def wait(jid: str, timeout: float = 600.0) -> dict[str, Any]:
    end = time.time() + timeout
    while time.time() < end:
        job = get(jid)
        if job and job["state"] != "running":
            return job
        time.sleep(0.05)
    raise TimeoutError(jid)
