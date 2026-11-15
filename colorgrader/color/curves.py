"""Transfer curves defined as data.

A curve is a JSON object with a list of segments. This module holds the only
curve math in the app: four generic segment types plus the PQ standard. No
camera or log format is named here. Adding a format means adding a JSON file
or a "definitions" block inside a camera file.

All math runs in float32.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

F32 = np.float32


class CurveError(ValueError):
    pass


def _f(x) -> np.float32:
    return np.float32(x)


class Curve:
    """Encode: linear light -> code value. Decode: code value -> linear light."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.id: str = spec.get("id", "")
        self.name: str = spec.get("name", self.id)
        self.kind: str = spec.get("kind", "scene")  # scene | display | pq
        self.code_values: str = spec.get("code_values", "normalized")  # absolute | normalized
        self.recommended_range: str = spec.get("recommended_range", "auto")
        self.params: dict = spec.get("params", {})
        self.segments: list[dict] = list(spec.get("segments", []))
        if self.kind not in ("scene", "display", "pq"):
            raise CurveError(f"curve {self.id}: unknown kind {self.kind!r}")
        if self.kind != "pq":
            if not self.segments:
                raise CurveError(f"curve {self.id}: no segments")
            for i, s in enumerate(self.segments):
                if s.get("type") not in ("lin", "log", "pow", "const"):
                    raise CurveError(f"curve {self.id}: unknown segment type {s.get('type')!r}")
                if i < len(self.segments) - 1 and "below" not in s:
                    raise CurveError(f"curve {self.id}: segment {i} needs a 'below' bound")
            # code value at each boundary, for decoding
            self._y_bounds = [float(self._enc_seg(s, np.array([s["below"]], dtype=np.float64))[0])
                              for s in self.segments[:-1]]

    # ------------------------------------------------------------ segments
    @staticmethod
    def _enc_seg(s: dict, x: np.ndarray) -> np.ndarray:
        t = s["type"]
        dt = x.dtype
        if t == "lin":
            return x * dt.type(s["e"]) + dt.type(s["f"])
        if t == "log":
            arg = np.maximum(x * dt.type(s["a"]) + dt.type(s["b"]), dt.type(1e-12))
            return np.log(arg) * dt.type(s["c"] / math.log(s["base"])) + dt.type(s["d"])
        if t == "pow":
            arg = np.maximum(x * dt.type(s["a"]) + dt.type(s["b"]), dt.type(0.0))
            return np.power(arg, dt.type(s["p"])) * dt.type(s["c"]) + dt.type(s["d"])
        if t == "const":
            return np.full_like(x, s["value"])
        raise CurveError(t)

    @staticmethod
    def _dec_seg(s: dict, y: np.ndarray) -> np.ndarray:
        t = s["type"]
        dt = y.dtype
        if t == "lin":
            return (y - dt.type(s["f"])) / dt.type(s["e"])
        if t == "log":
            ex = np.exp((y - dt.type(s["d"])) * dt.type(math.log(s["base"]) / s["c"]))
            return (ex - dt.type(s["b"])) / dt.type(s["a"])
        if t == "pow":
            base = np.maximum((y - dt.type(s["d"])) / dt.type(s["c"]), dt.type(0.0))
            return (np.power(base, dt.type(1.0 / s["p"])) - dt.type(s["b"])) / dt.type(s["a"])
        if t == "const":
            return np.full_like(y, s["below"])
        raise CurveError(t)

    # -------------------------------------------------------------- public
    def encode(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=F32)
        if self.kind == "pq":
            return pq_inverse_eotf(x * F32(self.params.get("reference_white_nits", 203.0)))
        out = self._enc_seg(self.segments[-1], x)
        for s in reversed(self.segments[:-1]):
            out = np.where(x < F32(s["below"]), self._enc_seg(s, x), out)
        return out.astype(F32, copy=False)

    def decode(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=F32)
        if self.kind == "pq":
            return pq_eotf(y) / F32(self.params.get("reference_white_nits", 203.0))
        out = self._dec_seg(self.segments[-1], y)
        for s, yb in zip(reversed(self.segments[:-1]), reversed(self._y_bounds)):
            out = np.where(y < F32(yb), self._dec_seg(s, y), out)
        return out.astype(F32, copy=False)

    def black_code(self) -> float:
        """Code value of zero light."""
        return float(self.encode(np.zeros(1, dtype=F32))[0])

    def gray_code(self) -> float:
        return float(self.encode(np.array([0.18], dtype=F32))[0])


# ------------------------------------------------------------------ PQ (ST 2084)
_M1 = 2610.0 / 16384.0
_M2 = 2523.0 / 4096.0 * 128.0
_C1 = 3424.0 / 4096.0
_C2 = 2413.0 / 4096.0 * 32.0
_C3 = 2392.0 / 4096.0 * 32.0


def pq_eotf(v) -> np.ndarray:
    """ST 2084 EOTF. Signal 0..1 -> display light in nits."""
    v = np.clip(np.asarray(v, dtype=F32), 0.0, 1.0)
    vp = np.power(v, F32(1.0 / _M2))
    num = np.maximum(vp - F32(_C1), F32(0.0))
    den = F32(_C2) - F32(_C3) * vp
    return (F32(10000.0) * np.power(num / den, F32(1.0 / _M1))).astype(F32)


def pq_inverse_eotf(nits) -> np.ndarray:
    y = np.clip(np.asarray(nits, dtype=F32) / F32(10000.0), 0.0, 1.0)
    yp = np.power(y, F32(_M1))
    return np.power((F32(_C1) + F32(_C2) * yp) / (F32(1.0) + F32(_C3) * yp), F32(_M2)).astype(F32)
