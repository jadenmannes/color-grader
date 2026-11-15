"""Native file dialog, run as a separate process so the dialog owns its own main thread.

Prints a JSON list of chosen paths. Used by the web UI: a browser cannot hand
the server a real file path, and footage stays where it is.
"""
from __future__ import annotations

import json
import sys

TYPES = {
    "clips": ("Video", "*.mov *.mp4 *.m4v *.mkv *.mxf *.avi *.mts *.MOV *.MP4 *.MXF"),
    "reference": ("Image or video", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp *.mov *.mp4 *.m4v *.mkv *.PNG *.JPG *.MOV *.MP4"),
    "lut": ("Cube LUT", "*.cube *.CUBE"),
    "camera": ("Camera file", "*.json"),
    "look": ("Look file", "*.json"),
}


def main() -> None:
    kind = sys.argv[1] if len(sys.argv) > 1 else "clips"
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # no Tk on this machine
        print(json.dumps({"error": f"No file dialog on this system: {exc}"}))
        return
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if kind == "folder":
            p = filedialog.askdirectory(title="Choose a folder")
            paths = [p] if p else []
        elif kind == "clips":
            paths = list(filedialog.askopenfilenames(title="Add clips", filetypes=[TYPES["clips"], ("All files", "*.*")]))
        else:
            p = filedialog.askopenfilename(title="Choose a file", filetypes=[TYPES.get(kind, ("All files", "*.*")), ("All files", "*.*")])
            paths = [p] if p else []
        root.destroy()
        print(json.dumps({"paths": paths}))
    except Exception as exc:
        print(json.dumps({"error": f"File dialog failed: {exc}"}))


if __name__ == "__main__":
    main()
