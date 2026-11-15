"""Color gamuts defined as data (xy primaries and white point).

Matrices come from colour-science: normalised primary matrices and CAT02
chromatic adaptation when white points differ (the default in Resolve's
Color Space Transform).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

F32 = np.float32


class Gamut:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.id: str = spec.get("id", "")
        self.name: str = spec.get("name", self.id)
        self.primaries = np.asarray(spec["primaries"], dtype=np.float64).reshape(3, 2)
        self.white = np.asarray(spec["white"], dtype=np.float64).reshape(2)

    def key(self) -> tuple:
        return tuple(np.round(self.primaries, 6).ravel()) + tuple(np.round(self.white, 6))

    def to_xyz(self) -> np.ndarray:
        return _npm(self.key())

    def luma_weights(self) -> np.ndarray:
        """Y row of RGB->XYZ: true luminance weights for linear RGB."""
        return self.to_xyz()[1].astype(F32)


@lru_cache(maxsize=64)
def _npm(key: tuple) -> np.ndarray:
    import colour

    prim = np.asarray(key[:6]).reshape(3, 2)
    white = np.asarray(key[6:])
    return np.asarray(colour.models.normalised_primary_matrix(prim, white), dtype=np.float64)


@lru_cache(maxsize=256)
def _matrix(src_key: tuple, dst_key: tuple) -> np.ndarray:
    import colour

    m_src = _npm(src_key)
    m_dst = _npm(dst_key)
    w_src = np.asarray(src_key[6:])
    w_dst = np.asarray(dst_key[6:])
    m = m_src
    if not np.allclose(w_src, w_dst, atol=1e-5):
        cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(
            colour.xy_to_XYZ(w_src), colour.xy_to_XYZ(w_dst), transform="CAT02")
        m = cat @ m
    return np.linalg.inv(m_dst) @ m


def conversion_matrix(src: Gamut, dst: Gamut) -> np.ndarray:
    """3x3 float32 matrix taking linear RGB in src to linear RGB in dst."""
    return _matrix(src.key(), dst.key()).astype(F32)


def apply_matrix(rgb: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply a 3x3 matrix to an (..., 3) float32 array."""
    return (rgb.reshape(-1, 3) @ m.T.astype(F32)).reshape(rgb.shape)
