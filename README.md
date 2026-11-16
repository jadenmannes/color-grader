# Color Grader

Color Grader analyzes your footage and a reference, builds a grade the way a colorist works, and hands you the result three ways: a DaVinci Resolve recipe, a `.cube` LUT set, and a graded export. It runs on your laptop. Your footage never leaves your computer.

## Install and start

Windows (PowerShell or the Command Prompt, from the project folder):

```bat
.\install.bat
```

```bat
.\start.bat
```

The `.\` matters in PowerShell: it runs scripts from the current folder only when you point at it. Double-clicking the files in Explorer works too.

macOS and Linux:

```bash
./install.sh
```

```bash
./start.sh
```

The install script sets up Python through `uv`, installs the packages, and fetches ffmpeg when your system has none. The start script opens the app in your browser at `http://127.0.0.1:8765`.

## The stack, in two sentences

A Python backend does all color math in 32-bit float with NumPy and colour-science, and ffmpeg decodes and encodes at full bit depth, so no browser video player ever touches your pixels during analysis. The interface is a local web page served by the same process, with no build step and no cloud service.

## How it works

1. Add clips. The app reads the file metadata, matches the camera from your camera library, and splits long clips and finished edits into shots.
2. Add a reference: a still, a film screenshot or a short clip. With no reference the app proposes three looks for your footage.
3. The app measures every shot and the reference: exposure, black and white point, clipping, white balance, contrast, saturation by tonal range, main colors and skin tones.
4. Press Build grade. The app balances every shot, matches the shots of a scene, fits one shared look, protects skin and checks the quality rules.
5. You get a before and after view next to the reference, scopes, the editor recipe, the LUT set and the graded export.

### The grading chain

| Step | Where | What |
| --- | --- | --- |
| 1 Input conversion | per clip | Camera curve and gamut to DaVinci Wide Gamut / Intermediate. Finished Rec.709 pictures enter through the exact inverse of the output conversion. |
| 2 Balance | per shot | Exposure, white balance, black point. Gain and offset on linear light. Shots with no faces anchor on neutral objects, sky and foliage. |
| 3 Shot match | per shot | Shots of one scene match each other. Different cameras match each other before the look. |
| 4 Look | shared | Contrast, lift/gamma/gain, saturation, Lum vs Sat, hue curves. One look for all shots. |
| 5 Skin | shared | Keeps skin on the skin tone line. |
| 6 Finish | shared | Optional vignette and fine grain. |
| 7 Output conversion | shared | To Rec.709 with Resolve's DaVinci tone mapping. |

Every step uses standard grading controls with Resolve's own math. The values in the recipe are the values the app renders with.

### Matching a reference

The app describes the reference in plain words and fits the shared look to its statistics: tonal spread, black level, color of shadows, midtones and highlights, saturation by tonal range, and palette. Brightness and color level follow the reference too. The Reference brightness slider on the Grade tab sets how far: 100 percent takes on the reference's midtone level (1.25 stops at most), 0 percent keeps your own exposure. When a reference asks for more than your footage carries, the app says why and builds the closest believable version. No pixel-by-pixel color transfer happens anywhere.

### Looks with no reference

A build with no reference proposes three looks. The app reads the footage (brightness, contrast, color level, faces, sky, foliage, warm or cool palette), scores every look template, keeps the best of three different families, fits each to your footage and checks the quality rules. Each proposal shows a name, a one-line mood, a preview and the reason for the pick. The templates are data: `colorgrader/data/looks/suggestions.json`, plus an optional `suggestions.json` in `~/ColorGrader/looks`.

### Quality rules

A grade fails when it breaks one of these. The build repairs failures on its own and lists every fix.

- No new clipped highlights or crushed shadows, unless the reference does it.
- No clipped colors: no channel driven to its limit or to zero inside a bright or saturated color.
- Skin stays healthy and on the skin tone line.
- Whites and grays stay neutral after balance.
- Saturation stays controlled, with less in deep shadows and bright highlights.
- Gradients stay smooth.
- Shots in one scene match.
- The grade stays subtle.

## What you get

- **Viewer**: before and after with a drag slider, next to the reference. Play graded renders a small preview file and plays the clip with the full grade.
- **Scopes**: waveform, RGB parade, vectorscope with skin tone line, histogram, for before, after and reference side by side.
- **Strength slider**: 0 to 100 percent. Recipe, LUTs and export follow it.
- **Recipe tab**: project color settings, then node by node, every control with its exact value and one short reason. Copy it as text or download it. The large view fits a second screen.
- **LUT set** (Export tab): the look for Resolve, the look with output conversion, the output conversion alone, and per camera profile an input LUT and an any-editor LUT. A text file in the folder says where each file goes. Size 17, 33 or 65 points (Settings).
- **Graded clip** (Export tab): source resolution, 10-bit H.265 for 10-bit sources, tagged Rec.709 so QuickTime, phones and YouTube show the same picture. ProRes 422 HQ and DNxHR HQX are in the format list.
- **Saved looks**: save the current look with a name, apply it to other footage later.

## Camera library

Cameras are data. Each camera is one JSON file in `~/ColorGrader/cameras`. Adding a camera needs no code change and no rebuild.

- A camera holds recording profiles. A profile stores the log curve, color gamut, bit depth, data range, how to recognize it in file metadata, and default corrections applied to every clip.
- Add a camera in the Cameras tab. Pick the curve and gamut from the list, or load the maker's conversion LUT when no published math exists. The form converts a sample clip and shows it next to the original before it saves.
- A clip from an unknown camera opens the form, filled in with what the metadata shows.
- Edit, delete, export and import cameras as files. An exported file carries its own curve and gamut definitions.

Shipped cameras: Fujifilm X-M5 (F-Log2, F-Log, film simulations, HLG), Sony A7C II (S-Log3 with S-Gamut3.Cine or S-Gamut3, HLG, standard and S-Cinetone) and a generic phone (standard, HDR HLG, HDR PQ, Apple Log).

Sony and other cameras write their settings into an XML file next to each clip (`C0001M01.XML`) and into the clip itself. The app reads it and shows the values as `xml.*` tags. A profile names the tags that settle it, for example `xml.capturegammaequation=s-log3-cine`. Keep the XML next to the clip when you copy footage off the card.

Shipped curves with published math: Sony S-Log3, Canon Log 3, Panasonic V-Log, Nikon N-Log, Apple Log, HLG, PQ, Rec.709, sRGB, DaVinci Intermediate. The Fujifilm F-Log and F-Log2 curves live inside the X-M5 camera file.

## Editors

Each editor is one module in `colorgrader/recipes`. The app finds the modules on its own. Shipped: `resolve.py` (DaVinci Resolve, free and Studio) and `lut_any.py` (any editor with a LUT slot, the template for new modules). Adding Premiere Pro or Final Cut Pro means adding one file there.

## Settings

Target editor and Resolve edition, output color space (Rec.709 Gamma 2.4, Gamma 2.2 or sRGB), highlight range of the tone mapping, default look strength, analysis frames per second, LUT size, export format and export folder.

## Your data

Everything you own sits in `~/ColorGrader`: cameras, settings, saved looks, projects, caches and exports. App updates replace the code folder only.

## Known limits

- The Resolve control math follows measured recreations of Resolve's tools (see `THIRD_PARTY.md`). Nobody has checked the recipe inside Resolve yet. Compare the Resolve viewer with the app on your first project. The LUT route carries the exact app result either way.
- Resolve numbers the Input Hue of its hue curves on its own scale. The recipe places hue points by color name and by position between the six color buttons.
- Vignette window shape and film grain have no typed values in Resolve. The recipe marks them as set by eye. Film Grain needs Resolve Studio.
- The X-M5 profiles are built from Fujifilm's published F-Log and F-Log2 math. Detection of F-Log against F-Log2 from metadata is untested on real X-M5 files, so the app asks you to confirm the profile when the metadata leaves it open.
- A 33-point LUT follows the grade within 0.2 percent on average on normal picture content (65 points: 0.05 percent). Its largest error sits in the deepest shadows of log footage and at the edge of the display gamut. The node recipe and the app's own export stay exact there.
- HDR PQ clips convert through a display-to-scene step with no matching Resolve control. Use the input LUT for them.
- The Windows and macOS start scripts are written but ran on Linux only so far.

## Tests

```bash
uv sync --extra dev
```

```bash
uv run pytest
```

- `tests/test_curves.py` checks F-Log and F-Log2 against the values in Fujifilm's data sheets (0% = code 95, 18% gray = 470 and 400, 90% = 705 and 570) and every shipped curve against colour-science.
- `tests/test_camera_library.py` records a synthetic S-Log3 clip, adds a matching camera through the form's API with no code edits, converts and analyzes the clip.
- `tests/test_grade.py` applies a known grade to a clip, uses the result as the reference and checks the app recovers the settings. It also covers balance, scene match, skin protection and the quality rules.
- `tests/test_recipe_export.py` types the recipe numbers into Resolve's math and compares with the app's render, checks the `.cube` files against ffmpeg, and compares exported frames and file tags with the preview.
- `tests/test_suggest.py` covers the look suggestions: three different looks, each inside the quality rules, chosen by the footage.
- `tests/e2e/run.sh` drives the real add-camera form in a headless browser.

## Layout

```
colorgrader/
  color/        curves, gamuts, working space, Resolve-style operators, LUT tools
  analysis/     shot detection, measurements, white balance estimate, scopes, faces
  grade/        balance, match, look fit, skin, quality rules, suggestions
  recipes/      one module per editor
  data/         shipped curves, gamuts, cameras, look templates (JSON) and the face model
  web/          the interface (plain JavaScript modules, no build step)
  cameras.py    camera library
  export.py     LUT set, graded clip export, playback preview
  looks.py      saved looks
  media.py      ffmpeg and ffprobe access
  project.py    clips, shots, analysis state
  server.py     local API
tools/          data generator and demo clip builder
tests/
```

See `THIRD_PARTY.md` for licenses and math references.
