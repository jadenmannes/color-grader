# Color Grader

Color Grader analyzes your footage and a reference, builds a grade the way a colorist works, and hands you the result three ways: a DaVinci Resolve recipe, a `.cube` LUT, and a graded export. It runs on your laptop. Your footage never leaves your computer.

## Install and start

Windows:

```bat
install.bat
```

```bat
start.bat
```

macOS and Linux:

```bash
./install.sh
```

```bash
./start.sh
```

The install script sets up Python through `uv`, installs the packages, and fetches ffmpeg when your system has none. The start script opens the app in your browser at `http://127.0.0.1:8765`.

## The stack, in two sentences

A Python backend does all color math in 32-bit float with NumPy and colour-science, and ffmpeg decodes and encodes at full bit depth, so no browser video player ever touches your pixels. The interface is a local web page served by the same process, with no build step and no cloud service.

## How it works

1. Add clips. The app reads the file metadata, matches the camera from your camera library, and splits long clips and finished edits into shots.
2. Add a reference: a still, a film screenshot or a short clip.
3. The app measures every shot and the reference: exposure, black and white point, clipping, white balance, contrast, saturation by tonal range, main colors and skin tones.
4. The app builds the grade and shows before and after next to the reference, with scopes.
5. You get the Resolve recipe, the LUT and the graded export.

### The grading chain

| Step | Where | What |
| --- | --- | --- |
| 1 Input conversion | per clip | Camera curve and gamut to DaVinci Wide Gamut / Intermediate. Finished Rec.709 pictures enter through the exact inverse of the output conversion. |
| 2 Balance | per shot | Exposure, white balance, black point. Gain and offset on linear light. |
| 3 Shot match | per shot | Shots of one scene match each other, across cameras. |
| 4 Look | shared | Contrast, lift/gamma/gain, saturation, hue curves. One look for all shots. |
| 5 Skin | shared | Keeps skin on the skin tone line. |
| 6 Finish | shared | Optional vignette and grain. |
| 7 Output conversion | shared | To Rec.709 with Resolve's DaVinci tone mapping. |

Every step uses standard grading controls with Resolve's own math, so the recipe values rebuild the same picture in Resolve.

## Camera library

Cameras are data. Each camera is one JSON file in `~/ColorGrader/cameras`. Adding a camera needs no code change and no rebuild.

- A camera holds recording profiles. A profile stores the log curve, color gamut, bit depth, data range, how to recognize it in file metadata, and default corrections applied to every clip.
- Add a camera in the Cameras tab. Pick the curve and gamut from the list, or load the maker's conversion LUT when no published math exists. The form converts a sample clip and shows it next to the original before it saves.
- A clip from an unknown camera opens the form, filled in with what the metadata shows.
- Edit, delete, export and import cameras as files. An exported file carries its own curve and gamut definitions.

Shipped cameras: Fujifilm X-M5 (F-Log2, F-Log, film simulations, HLG) and a generic phone (standard, HDR HLG, HDR PQ, Apple Log).

Shipped curves with published math: Sony S-Log3, Canon Log 3, Panasonic V-Log, Nikon N-Log, Apple Log, HLG, PQ, Rec.709, sRGB, DaVinci Intermediate. The Fujifilm F-Log and F-Log2 curves live inside the X-M5 camera file.

## Your data

Everything you own sits in `~/ColorGrader`: cameras, settings, saved looks, projects, caches and exports. App updates replace the code folder only.

## Tests

```bash
uv sync --extra dev
```

```bash
uv run pytest
```

- `tests/test_curves.py` checks F-Log and F-Log2 against the values in Fujifilm's data sheets (0% = code 95, 18% gray = 470 and 400, 90% = 705 and 570) and every shipped curve against colour-science.
- `tests/test_camera_library.py` records a synthetic S-Log3 clip, adds a matching camera through the form's API with no code edits, converts and analyzes the clip.
- `tests/e2e/run.sh` drives the real add-camera form in a headless browser.

## Layout

```
colorgrader/
  color/        curves, gamuts, working space, Resolve-style operators, LUT tools
  analysis/     shot detection, measurements, white balance estimate, scopes, faces
  grade/        the grade as data and the grading chain
  data/         shipped curves, gamuts, cameras (JSON) and the face model
  web/          the interface (plain JavaScript modules, no build step)
  cameras.py    camera library
  media.py      ffmpeg and ffprobe access
  project.py    clips, shots, analysis state
  server.py     local API
tools/          data generator and demo clip builder
tests/
```

See `THIRD_PARTY.md` for licenses and math references.
