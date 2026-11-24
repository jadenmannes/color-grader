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

### Updating

Pull the update in the color-grader folder, then restart the app (`.\start.bat` or `./start.sh`) and reload the page. The top bar shows the version and the update (git commit) the app runs. Versions before 4541dc0 let the browser keep old interface files: after updating from one of those, reload once with Ctrl+F5 (Cmd+Shift+R on a Mac).

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
| 3b Shot trim | per shot | Any color control on one shot only. Empty unless you or an assistant set it. |
| 4 Look | shared | Every color control below. One look for all shots. |
| 5 Skin | shared | Keeps skin on the skin tone line. |
| 6 Finish | shared | Noise reduction, clarity and sharpening. Optional vignette and fine grain. |
| 7 Output conversion | shared | To Rec.709 with Resolve's DaVinci tone mapping. |

Every step uses standard grading controls with Resolve's own math. The values in the recipe are the values the app renders with.

### Matching a reference

The app describes the reference in plain words and fits the shared look to its statistics: tonal spread, black level, color of shadows, midtones and highlights, saturation by tonal range, and palette. Brightness and color level follow the reference too. The Reference brightness slider on the Grade tab sets how far: 100 percent takes on the reference's midtone level (1.25 stops at most), 0 percent keeps your own exposure. When a reference asks for more than your footage carries, the app says why and builds the closest believable version. No pixel-by-pixel color transfer happens anywhere.

### A finished picture

A corrected picture is flat: blacks sit in gray, the top end stops short of white, noise shows. Three steps close the gap to a finished one.

- **Punch** (Grade tab, default 70 percent): the finishing pass on tone. It sets the black point and opens the top end with the look's own contrast, lift, gamma and gain. A film-style curve then steepens the midtones and eases into black and white, so the picture gains depth with no new clipping. Color density (Sat vs Lum) makes strong colors a little darker and richer, the way film dye takes light away. The brightest skin stays under 76 percent, so faces keep shape and color.
- **Noise reduction**: cleans the camera signal before the grade, at 16-bit precision, with the frames before and after. 8-bit log footage gains smooth gradients from it. The first build switches it on for log footage.
- **Clarity and sharpening**: wide and fine edge contrast on the finished luma. The first build sets both to a moderate level.

The build never adds grain on its own. Grain stays a choice on the Grade tab. LUTs carry color only: noise reduction and sharpening live in the app's export and, as starting values, in the Resolve recipe.

### Three looks designed from your shot

Select a shot and press Design three looks. There is no preset list: the app reads that shot and the other shots of its scene, then builds every number of every look from what it measured. It is tuned for interviews shot outdoors.

The scene read: the light the footage came in (soft and overcast, hard sun, low warm sun, the cast the balance removed in stops), the palette (autumn reds and yellows, greenery, sky), the faces (skin level on the waveform, distance from the skin tone line), the contrast, and how close the brightest tones sit to clipping.

The directions, chosen and tuned by that read:

| Look | Comes up when | Built from the shot |
| --- | --- | --- |
| Clean interview | Always, first | Contrast to a finished range, saturation lifted for quiet color or pulled back for loud color, greens a touch toward yellow when the shot holds them |
| Warm autumn, Golden hour, Warm overcast, Warm daylight | Cool or gray light, autumn color, low sun | Warmth in the highlights scaled to how cool the light came in; leaves compressed to orange when the shot holds them |
| Cinematic teal and orange | Faces against sky, blue tones or a busy background | Hue compression only for colors the shot holds, teal shadows, warm highlights, a slightly darker exposure |
| Rich and moody | Flat light, bright backgrounds | Lower exposure, denser HSV saturation, quieter greens, deep blacks |
| Bright and airy | Soft light with room above the brightest tones | A lift sized to that room, lifted shadows, pastel color |
| Punchy interview | Faces, above all in flat light or quiet color | The studio interview grade: levels filled, strong contrast and saturation, cool shadows, mids nudged against green, warm skin, a strong relight |

Each look lists the colorist workflow with its values and reasons: 1 white balance, 2 exposure, 3 contrast, 4 saturation, 5 hue rotation, 6 split tone, 7 relight, 8 output. A designed look runs at full strength; the Look strength slider takes it back. Steps 1 and 2 stay per shot; the look carries the rest, so every shot of the interview matches.

Every shot also gets its own black point and contrast (the clip-level EXP/CON step): haze comes off in linear light, crushed blacks open, and a flat shot gets contrast around its own middle so its exposure stays. Once a look is on, each shot is trimmed again under it, so one dark or hazy shot never shifts the rest.

### Subject relight

The interview technique of darkening the frame and keeping the speaker lit. The analysis finds faces on every analyzed frame (two per second) and smooths their path. A soft ellipse sits over each head, from above the hairline to below the chin, and a wide one over the shoulders and chest. Outside them the picture drops by the set number of stops in the working space; inside, the subject keeps its exposure exactly (the hand-made `2 - g` shortcut leaves the subject 13% darker). The feather is about a third of each radius, so no edge shows.

- The first build switches it on at 0.5 stops when most shots hold a face. Each designed look sets its own amount: more for bright backgrounds, none without faces.
- Preview, Play graded and the export all follow the faces. The export feeds ffmpeg one small mask frame per video frame.
- The recipe lists a RELIGHT node: two windows, the tracker, and an Outside node with the Offset. Window shapes and tracking are set by eye in Resolve. LUTs cannot carry it.
- Grade tab, Finish: Relight on or off and its strength in stops.

### Tools from the colorist workflow

- **White balance and exposure** on a node in linear gamma with Gain: blacks stay black, the way HDR Global exposure behaves.
- **Contrast** at pivot 0.336 (middle gray of DaVinci Intermediate) plus a film curve.
- **Saturation** in an HSV node with channels 1 and 3 off: Gain lifts strong colors, Gamma lifts gentle colors such as skin, and colors get denser as they saturate, like film.
- **Hue rotation** with Hue vs Hue curves, **split tone** with Lift and Gain per channel.
- The recipe lists each of these as its own node with exact values. The app renders with the same numbers.

The old template file `colorgrader/data/looks/suggestions.json` stays for template authors (`suggest.suggest_from_templates`). The app no longer proposes from it.

### Color controls

The Color tab, the MCP server and the recipe share one control set. Each control works on the shared look (every shot) or on one shot's own trim. The app applies them in this order, one Resolve node each:

| Control | Resolve tool | Recipe |
| --- | --- | --- |
| Exposure, temperature, tint (stops) | Gain on a node with Gamma set to Linear | exact |
| HDR zones (dark, shadow, light, highlight, specular), each with exposure, saturation, temp and tint | HDR palette | by eye |
| RGB mixer | RGB Mixer palette | exact |
| Contrast and pivot | Primaries | exact |
| Lift, gamma, gain, offset (color wheels) | Primaries | exact |
| Log wheels: shadow, midtone, highlight, low and high range | Primaries, Log mode | by eye |
| Shadows, highlights (stops) | Custom curve the app generates | exact points |
| Custom curve, RGB curves with gang off | Curves, Custom | exact points |
| Saturation, Lum vs Sat | Primaries and Curves | exact |
| Color boost (vibrance) | Sat vs Sat curve the app generates | exact |
| HSV saturation gain and gamma | Primaries in an HSV node | exact |
| Hue (global), Hue vs Hue, Hue vs Sat | Curves | exact |
| Hue vs Lum | Curves | by eye |
| Sat vs Lum, Sat vs Sat | Curves | exact points |
| Secondaries: hue, saturation and luminance key, then exposure, temp, tint, saturation and hue inside it | Qualifier, HSL | by eye |

Exact: typed into Resolve's math, the recipe values rebuild the app's picture (a test checks this). Exact points: the same points, with Resolve's own spline between them. By eye: Resolve's math for that tool is unpublished, so the recipe gives a starting point and the LUT and export carry the exact result.

Units stay the same everywhere. Exposure, temp, tint, shadows and highlights are in stops. Hue is in degrees: red 0, yellow 60, green 120, cyan 180, blue 240, magenta 300. The primaries use Resolve's own numbers.

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

- **Viewer**: before and after with a drag slider, next to the reference. The preview renders at the size your screen shows it, sharp on large and high-DPI screens. Play graded renders a small preview file and plays the clip with the full grade.
- **Full screen** (button under the viewer): the viewer with its before and after slider on the whole screen. Esc or the button in the corner leaves.
- **Full quality** (button under the viewer, F, or double-click the picture): the frame on screen at the source's full resolution with the full grade, lossless. Its own Full screen button fills the screen. Graded, source, or a split compare. Fit or 100% (one frame pixel per screen pixel) to judge noise, grain and sharpness. Save PNG writes a 16-bit file. A 4K frame takes a few seconds: large frames render on several cores at once.
- **Scopes**: waveform, RGB parade, vectorscope with skin tone line, histogram, for before, after and reference side by side.
- **Color tab**: every control above, live. Color wheels with a level slider and R, G, B fields, curve editors (click to add a point, drag, double-click to remove), HDR zones, the RGB mixer, and secondaries. A secondary starts from a preset (sky, skin, foliage, autumn leaves) or from a color you click in the viewer, and Show key turns the viewer into Resolve's Highlight view. Switch between the look and This shot only. Double-click a control's name to reset it.
- **Strength slider**: 0 to 100 percent. Recipe, LUTs and export follow it.
- **Recipe tab**: project color settings, then node by node, every control with its exact value and one short reason. Copy it as text or download it. The large view fits a second screen.
- **LUT set** (Export tab): the look for Resolve, the look with output conversion, the output conversion alone, and per camera profile an input LUT and an any-editor LUT. A text file in the folder says where each file goes. Size 17, 33 or 65 points (Settings).
- **Graded clip** (Export tab): source resolution, 10-bit H.265 for 10-bit sources, tagged Rec.709 so QuickTime, phones and YouTube show the same picture. ProRes 422 HQ and DNxHR HQX are in the format list.
- **Saved looks**: save the current look with a name, apply it to other footage later.

## Grade with an AI assistant (MCP)

Color Grader includes an MCP server. An assistant such as Claude, in the Claude desktop app or Claude Code on your computer, then grades in the app you have open: it looks at a shot (source and graded picture plus the numbers a colorist reads off the scopes), changes the grade, looks again, and can undo. Every change shows in your viewer within a second or two, and the viewer jumps to the shot the assistant works on.

1. Start the app (`.\start.bat` or `./start.sh`).
2. Connect the MCP server. On Windows, run:

```bat
.\mcp.bat
```

It registers the server with Claude Code when the `claude` command exists, then prints the lines for Claude Desktop. On macOS and Linux, print the setup lines with:

```bash
uv run python -m colorgrader.mcp --config
```

3. For Claude Desktop, paste the printed lines into its config file (Settings, Developer, Edit Config) and restart it. For Claude Code on macOS and Linux, run the printed `claude mcp add` command.
4. Ask, for example: "Grade shot 1 of my interview: warmer skin, darker background, keep it natural."

Tools the assistant gets:

- Look and measure: `project_status`, `view_shot`, `read_shot`, `get_grade`, `list_controls`.
- Change the grade: `adjust_look` (every control on every shot), `adjust_shot` (balance, plus any control as that shot's own trim), `reset_controls`, `set_finish`, `set_strength`.
- Secondaries: `sample_color` reads the color under a point, `set_secondary` keys a color (from a point or from ranges) and corrects inside it, `view_key` shows what the key selects.
- Looks and output: `design_looks`, `apply_look`, `rebuild_grade`, `undo`, `save_look`, `export_video`.

You see the assistant at work in the app. An orange cursor moves to each control it changes, drags sliders and wheels from the old value to the new one, clicks where it samples a color, and switches to the shot it works on. The viewer shows the key of a secondary while the assistant checks it. A short feed on the viewer lists each step with its reason. The cursor only shows the steps: the MCP server makes the changes. Untick Show assistant under the viewer to keep the panels still.

The app and the MCP server talk only on 127.0.0.1 and send nothing anywhere by themselves. With an assistant connected, the frames it views (single JPEG stills, 960 pixels wide by default) and the numbers go to the assistant's AI service, like an image you share in a chat. Your video files stay on your computer. The MCP server has to run on the same computer as the app, so a cloud session (Claude Code on the web) cannot reach it.

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
- The film curve in the recipe is a point table for Resolve's Custom curve. Resolve draws its own spline between the points, and its Sat vs Lum math is unpublished, so both nodes need a check by eye. The LUT route is exact.
- Vignette window shape and film grain have no typed values in Resolve. The recipe marks them as set by eye. Film Grain needs Resolve Studio.
- HDR zones, Log wheels, Hue vs Lum and qualifier softness use the app's own math. Resolve's is unpublished. The recipe lists them as starting points to set by eye. The LUT and the export carry the app's exact result.
- The X-M5 profiles are built from Fujifilm's published F-Log and F-Log2 math. Detection of F-Log against F-Log2 from metadata is untested on real X-M5 files, so the app asks you to confirm the profile when the metadata leaves it open.
- A 33-point LUT follows the grade within 0.3 percent on average on normal picture content (65 points: about 0.05 percent). The film curve costs a small table a little accuracy. Its largest error sits in the deepest shadows of log footage and at the edge of the display gamut. The node recipe and the app's own export stay exact there.
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
- `tests/test_controls.py` checks every color control: neutral changes nothing, each control moves only what it should, strength and rounding hold, and the API takes partial edits and rejects bad input.
- `tests/test_mcp.py` drives the MCP server over stdin and stdout against a running app: every control, shot trims, secondaries with a picked color, the key view and undo.
- `tests/test_still.py` checks the full-quality still (full resolution, 16-bit save), shows strip rendering equals whole-frame rendering bit for bit, and checks the fast wide blur against a true Gaussian.
- `tests/e2e/run.sh` drives the real add-camera form, the Color tab, the assistant cursor and the full-quality still in a headless browser.

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
