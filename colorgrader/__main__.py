"""Start the app:  python -m colorgrader   (or the start script)."""
from __future__ import annotations

import argparse
import threading
import webbrowser


def main() -> None:
    ap = argparse.ArgumentParser(prog="colorgrader", description="Local color grading assistant")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="Keep 127.0.0.1: footage stays on this computer")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    from . import cameras, media, paths

    try:
        ff = media.find_binary("ffmpeg")
        media.find_binary("ffprobe")
    except media.MediaError as exc:
        raise SystemExit(f"{exc}\nRun the install script first.")
    cameras.seed()
    url = f"http://{args.host}:{args.port}"
    print(f"Color Grader running at {url}")
    print(f"Data folder: {paths.home()}")
    print(f"ffmpeg: {ff}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    import uvicorn

    uvicorn.run("colorgrader.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
