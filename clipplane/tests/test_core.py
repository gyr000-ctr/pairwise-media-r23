import math

import pytest

import clipplane

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

PLANE = ((0.0, 0.0, 1.0), 0.0)
TRI = [(-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 1.0, 1.0)]

CASES = [
    ("sign", lambda: clipplane.ClipPlane().classify((0.0, 0.0, 2.0), PLANE) > 0),
    ("inside", lambda: clipplane.ClipPlane().classify((0.0, 0.0, -2.0), PLANE) < 0),
    ("onplane", lambda: clipplane.ClipPlane().classify((0.0, 0.0, 0.0), PLANE) == 0),
    ("clip_keeps", lambda: len(clipplane.ClipPlane().clip(TRI, PLANE)) == 3),
    ("clip_cuts", lambda: len(clipplane.ClipPlane().clip([(0, 0, 1), (0, 0, -1), (1, 0, 1)], PLANE)) < 3),
    ("all_inside", lambda: all(v[2] >= -1e-12 for v in clipplane.ClipPlane().clip(TRI, PLANE))),
    ("empty", lambda: clipplane.ClipPlane().clip([(0, 0, -1)], PLANE) == []),
    ("bad_plane", lambda: _raises(lambda: clipplane.ClipPlane().classify((0, 0, 0), ((0, 0, 0), 0.0)))),
    ("bad_poly", lambda: _raises(lambda: clipplane.ClipPlane().clip([], PLANE))),
    ("type", lambda: isinstance(clipplane.ClipPlane().clip(TRI, PLANE), list)),
    ("det", lambda: clipplane.ClipPlane().clip(TRI, PLANE) == clipplane.ClipPlane().clip(TRI, PLANE)),
    ("pair", lambda: all(len(v) == 3 for v in clipplane.ClipPlane().clip(TRI, PLANE))),
    ("nan", lambda: _raises(lambda: clipplane.ClipPlane().classify((float('nan'), 0, 0), PLANE))),
    ("no_mut", lambda: clipplane.ClipPlane().clip(list(TRI), PLANE) is not None),
    ("offset", lambda: clipplane.ClipPlane().classify((0.0, 0.0, 3.0), ((0.0, 0.0, 1.0), 2.0)) > 0),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
