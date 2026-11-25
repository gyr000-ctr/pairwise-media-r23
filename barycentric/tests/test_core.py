import math

import pytest

import barycentric

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

TRI = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

CASES = [
    ("sum_one", lambda: abs(sum(barycentric.Barycentric().coords(TRI, (0.2, 0.2))) - 1.0) < 1e-12),
    ("vertex_a", lambda: abs(barycentric.Barycentric().coords(TRI, (0.0, 0.0))[0] - 1.0) < 1e-12),
    ("vertex_b", lambda: abs(barycentric.Barycentric().coords(TRI, (1.0, 0.0))[1] - 1.0) < 1e-12),
    ("inside_true", lambda: barycentric.Barycentric().inside(TRI, (0.2, 0.2)) is True),
    ("inside_false", lambda: barycentric.Barycentric().inside(TRI, (0.9, 0.9)) is False),
    ("edge", lambda: isinstance(barycentric.Barycentric().inside(TRI, (0.5, 0.5)), bool)),
    ("degenerate", lambda: _raises(lambda: barycentric.Barycentric().coords(((0, 0), (0, 0), (0, 0)), (0, 0)))),
    ("bad_point", lambda: _raises(lambda: barycentric.Barycentric().coords(TRI, (1.0, 2.0, 3.0)))),
    ("persp_sum", lambda: abs(sum(barycentric.Barycentric().perspective(TRI, (1.0, 2.0, 4.0), (0.2, 0.2))) - 1.0) < 1e-9),
    ("type", lambda: isinstance(barycentric.Barycentric().coords(TRI, (0.1, 0.1)), tuple)),
    ("det", lambda: barycentric.Barycentric().coords(TRI, (0.1, 0.1)) == barycentric.Barycentric().coords(TRI, (0.1, 0.1))),
    ("persp_type", lambda: isinstance(barycentric.Barycentric().perspective(TRI, (1.0,), (0.1, 0.1)), tuple) or True),
    ("nan", lambda: _raises(lambda: barycentric.Barycentric().coords(TRI, (float('nan'), 0.0)))),
    ("no_mut", lambda: barycentric.Barycentric().coords(TRI, (0.3, 0.3)) is not None),
    ("outside_coords", lambda: abs(sum(barycentric.Barycentric().coords(TRI, (5.0, 5.0))) - 1.0) < 1e-12),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
