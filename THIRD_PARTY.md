# Third-party components

| Component | Where | License |
| --- | --- | --- |
| Preact and htm (standalone bundle) | `colorgrader/web/vendor/htm-preact-standalone.js` | MIT (Preact), Apache-2.0 (htm) |
| YuNet face detection model, 2023mar | `colorgrader/data/models/face_detection_yunet_2023mar.onnx` | MIT, see `face_detection_yunet_LICENSE.txt` |

Math references:

- DaVinci Resolve control math (Lift/Gamma/Gain/Offset, saturation, HSL curves, CST tone mapping): measured recreations by Thatcher Freeman, https://github.com/thatcherfreeman/utility-dctls (MIT). The app implements the same formulas in its own code.
- Log curves: maker data sheets and white papers, listed in each file under `colorgrader/data/curves` and in the camera files.
- Color matrices and reference values for tests: colour-science (BSD-3-Clause).
