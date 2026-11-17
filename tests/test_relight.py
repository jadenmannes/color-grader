"""Subject relight: face tracks, soft shapes, background down in stops."""
import numpy as np

from colorgrader.grade import relight


def test_track_interpolates_and_smooths():
    track = [[0.0, [[0.40, 0.40, 0.10, 0.18, 0.9]]], [0.5, [[0.42, 0.40, 0.10, 0.18, 0.9]]], [1.0, [[0.44, 0.40, 0.10, 0.18, 0.9]]]]
    mid = relight.faces_at(track, 0.25)
    assert len(mid) == 1 and abs(mid[0][0] - 0.41) < 1e-6           # halfway between two samples
    jitter = [[t, [[0.40 + (0.02 if i % 2 else 0.0), 0.40, 0.10, 0.18, 0.9]]] for i, t in enumerate(np.arange(0, 3, 0.5))]
    smooth = relight.smooth_track(jitter)
    spread = lambda tr: np.ptp([f[0][0] for _, f in tr[1:-1]])
    assert spread(smooth) < spread(jitter)                          # no jitter in the shapes
    assert relight.faces_at([], 0.3) == []


def test_mask_covers_head_and_shoulders_and_leaves_the_corners():
    m = relight.mask([[0.5, 0.35, 0.12, 0.22, 0.9]], 320, 180)
    assert m[int(0.35 * 180), 160] == 1.0                            # face
    assert m[int(0.80 * 180), 160] > 0.9                             # chest under the face
    assert m[5, 5] == 0.0 and m[5, 314] == 0.0                       # top corners in the background
    assert 0.05 < (m > 0.95).mean() < 0.5
    assert relight.mask([], 320, 180) is None


def test_apply_moves_only_the_background():
    di = np.full((18, 32, 3), 0.4, dtype=np.float32)
    m = np.zeros((18, 32), dtype=np.float32)
    m[6:12, 12:20] = 1.0
    out = relight.apply(di, m, amount=1.0)
    assert np.allclose(out[9, 16], 0.4)                              # subject exactly as it was
    assert np.allclose(out[0, 0], 0.4 - 0.07329248, atol=1e-5)       # one stop down in DaVinci Intermediate
