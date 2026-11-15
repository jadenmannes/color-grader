"""3D LUTs: .cube files and tetrahedral interpolation in float32."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

F32 = np.float32


class Lut3D:
    """table[r, g, b] -> RGB. Domain is the input range covered by the table."""

    def __init__(self, table: np.ndarray, domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0), title: str = ""):
        table = np.asarray(table, dtype=F32)
        if table.ndim != 4 or table.shape[3] != 3 or not (table.shape[0] == table.shape[1] == table.shape[2]):
            raise ValueError("LUT table must have shape (N, N, N, 3)")
        self.table = table
        self.size = table.shape[0]
        self.domain_min = np.asarray(domain_min, dtype=F32)
        self.domain_max = np.asarray(domain_max, dtype=F32)
        self.title = title

    @classmethod
    def from_function(cls, fn: Callable[[np.ndarray], np.ndarray], size: int = 33, domain_min=0.0, domain_max=1.0,
                      title: str = "") -> "Lut3D":
        axis = np.linspace(domain_min, domain_max, size, dtype=F32)
        r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
        grid = np.stack([r, g, b], axis=-1).reshape(-1, 3)
        out = np.asarray(fn(grid), dtype=F32).reshape(size, size, size, 3)
        return cls(out, (domain_min,) * 3, (domain_max,) * 3, title)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Tetrahedral interpolation. Input outside the domain clamps to the edge."""
        shape = rgb.shape
        x = rgb.reshape(-1, 3).astype(F32, copy=False)
        n = self.size
        span = np.maximum(self.domain_max - self.domain_min, F32(1e-12))
        x = np.clip((x - self.domain_min) / span, 0.0, 1.0) * F32(n - 1)
        i0 = np.minimum(np.floor(x).astype(np.int32), n - 2)
        f = x - i0
        order = np.argsort(-f, axis=1, kind="stable")  # axis with the largest fraction first
        fs = np.take_along_axis(f, order, axis=1)
        rows = np.arange(x.shape[0])
        step1 = np.zeros_like(i0)
        step1[rows, order[:, 0]] = 1
        step2 = step1.copy()
        step2[rows, order[:, 1]] = 1
        flat = self.table.reshape(-1, 3)

        def at(idx):
            return flat[(idx[:, 0] * n + idx[:, 1]) * n + idx[:, 2]]

        c0 = at(i0)
        c1 = at(i0 + step1)
        c2 = at(i0 + step2)
        c3 = at(i0 + 1)
        out = c0 + fs[:, 0:1] * (c1 - c0) + fs[:, 1:2] * (c2 - c1) + fs[:, 2:3] * (c3 - c2)
        return out.reshape(shape)


def write_cube(path: str | Path, lut: Lut3D, comments: list[str] | None = None) -> Path:
    """Write a .cube file (red index changes fastest)."""
    path = Path(path)
    lines = []
    for c in comments or []:
        lines.append(f"# {c}")
    if lut.title:
        lines.append(f'TITLE "{lut.title}"')
    lines.append(f"LUT_3D_SIZE {lut.size}")
    if not (np.allclose(lut.domain_min, 0.0) and np.allclose(lut.domain_max, 1.0)):
        lines.append("DOMAIN_MIN " + " ".join(f"{v:.6f}" for v in lut.domain_min))
        lines.append("DOMAIN_MAX " + " ".join(f"{v:.6f}" for v in lut.domain_max))
    data = np.transpose(lut.table, (2, 1, 0, 3)).reshape(-1, 3)  # b outer, g, r inner
    body = "\n".join(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in data.tolist())
    path.write_text("\n".join(lines) + "\n" + body + "\n", encoding="utf-8")
    return path


def read_cube(path: str | Path) -> Lut3D:
    """Read a 3D .cube file. A 1D shaper is not supported."""
    size = None
    dmin = [0.0, 0.0, 0.0]
    dmax = [1.0, 1.0, 1.0]
    title = ""
    values: list[float] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head = line.split()[0].upper()
            if head == "TITLE":
                title = line[5:].strip().strip('"')
            elif head == "LUT_3D_SIZE":
                size = int(line.split()[1])
            elif head == "LUT_1D_SIZE":
                raise ValueError("1D LUT files are not supported. Load a 3D .cube file.")
            elif head == "DOMAIN_MIN":
                dmin = [float(v) for v in line.split()[1:4]]
            elif head == "DOMAIN_MAX":
                dmax = [float(v) for v in line.split()[1:4]]
            elif head in ("LUT_3D_INPUT_RANGE",):
                lo, hi = (float(v) for v in line.split()[1:3])
                dmin, dmax = [lo] * 3, [hi] * 3
            elif head[0].isdigit() or head[0] in "-+.":
                parts = line.split()
                if len(parts) >= 3:
                    values.extend(float(v) for v in parts[:3])
    if size is None:
        raise ValueError("No LUT_3D_SIZE found. The file is not a 3D .cube LUT.")
    arr = np.asarray(values, dtype=F32)
    if arr.size != size ** 3 * 3:
        raise ValueError(f"LUT holds {arr.size // 3} entries, expected {size ** 3}.")
    table = np.transpose(arr.reshape(size, size, size, 3), (2, 1, 0, 3))
    return Lut3D(np.ascontiguousarray(table), dmin, dmax, title)
