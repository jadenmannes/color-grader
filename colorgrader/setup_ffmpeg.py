"""Make sure ffmpeg and ffprobe exist. Run by the install script.

Looks on PATH and in the app's own bin folder. When missing, downloads a
static build into ~/ColorGrader/bin, so nothing else on the system changes.
"""
from __future__ import annotations

import io
import os
import platform
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from . import media, paths

WINDOWS_ZIPS = [
    "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip",
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
]
MAC_ZIPS = {"ffmpeg": "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip", "ffprobe": "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"}


def bin_dir() -> Path:
    d = paths.home() / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def check() -> tuple[bool, str]:
    """True when both tools run and the build holds what the app needs."""
    media._bin_cache.clear()
    try:
        ff = media.find_binary("ffmpeg")
        media.find_binary("ffprobe")
    except media.MediaError as exc:
        return False, str(exc)
    try:
        enc = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, timeout=30).stdout.decode(errors="replace")
        flt = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, timeout=30).stdout.decode(errors="replace")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"ffmpeg does not run: {exc}"
    missing = [name for name, text in (("libx265", enc), ("libx264", enc), ("lut3d", flt)) if name not in text]
    if missing:
        return False, f"ffmpeg at {ff} lacks: {', '.join(missing)}"
    return True, ff


def _download(url: str) -> bytes:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "colorgrader-setup"})
    with urllib.request.urlopen(req, timeout=120) as res:
        return res.read()


def _extract(data: bytes, wanted: set[str], dest: Path) -> list[str]:
    done = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            base = Path(name).name.lower()
            if base in wanted:
                target = dest / Path(name).name
                with z.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                done.append(base)
    return done


def install_windows() -> bool:
    for url in WINDOWS_ZIPS:
        try:
            got = _extract(_download(url), {"ffmpeg.exe", "ffprobe.exe"}, bin_dir())
            if {"ffmpeg.exe", "ffprobe.exe"} <= set(got):
                return True
        except Exception as exc:  # try the next mirror
            print(f"  failed: {exc}")
    if shutil.which("winget"):
        print("  trying winget")
        r = subprocess.run(["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--accept-source-agreements", "--accept-package-agreements"])
        if r.returncode == 0:
            links = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"
            for exe in ("ffmpeg.exe", "ffprobe.exe"):
                if (links / exe).is_file():
                    shutil.copyfile(links / exe, bin_dir() / exe)
            return True
    return False


def install_mac() -> bool:
    if shutil.which("brew"):
        print("  installing with Homebrew")
        if subprocess.run(["brew", "install", "ffmpeg"]).returncode == 0:
            return True
    try:
        for name, url in MAC_ZIPS.items():
            _extract(_download(url), {name}, bin_dir())
        return True
    except Exception as exc:
        print(f"  failed: {exc}")
        return False


def main() -> int:
    ok, detail = check()
    if ok:
        print(f"ffmpeg ready: {detail}")
        return 0
    print(f"ffmpeg setup needed ({detail})")
    system = platform.system()
    if system == "Windows":
        done = install_windows()
    elif system == "Darwin":
        done = install_mac()
    else:
        done = False
        print("  Install ffmpeg with your package manager, for example: sudo apt install ffmpeg")
    ok, detail = check()
    if ok:
        print(f"ffmpeg ready: {detail}")
        return 0
    print("\nffmpeg is still missing or incomplete.")
    print(f"Fix: put ffmpeg and ffprobe (a GPL build with libx264 and libx265) into {bin_dir()} and run the install again.")
    if system == "Windows":
        print("Or run:  winget install Gyan.FFmpeg")
    elif system == "Darwin":
        print("Or run:  brew install ffmpeg")
    return 0 if done and ok else 1


if __name__ == "__main__":
    sys.exit(main())
