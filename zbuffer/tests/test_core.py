import math

import pytest

import zbuffer

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _z():
    return zbuffer.ZBuffer(4, 4, 0.1, 100.0)

CASES = [
    ("empty_depth", lambda: _z().depth(0, 0) == 1.0),
    ("write", lambda: (_z().test(0, 0, 0.5, 1.0), _z().depth(0, 0))[1] == 0.5),
    ("occlude", lambda: (_z().test(0, 0, 0.5, 1.0), _z().test(0, 0, 0.9, 2.0), _z().attr(0, 0))[2] == 1.0),
    ("nearer_wins", lambda: (_z().test(0, 0, 0.9, 1.0), _z().test(0, 0, 0.2, 2.0), _z().attr(0, 0))[2] == 2.0),
    ("ndc", lambda: abs(_z().ndc(0.1) - 0.0) < 1e-9),
    ("ndc_far", lambda: abs(_z().ndc(100.0) - 1.0) < 1e-9),
    ("bad_near", lambda: _raises(lambda: zbuffer.ZBuffer(4, 4, 0.0, 100.0))),
    ("bad_range", lambda: _raises(lambda: zbuffer.ZBuffer(4, 4, 10.0, 1.0))),
    ("out_of_bounds", lambda: _raises(lambda: _z().test(9, 9, 0.5, 1.0))),
    ("type", lambda: isinstance(_z().depth(0, 0), float)),
    ("det", lambda: (_z().test(0, 0, 0.5, 1.0), _z().depth(0, 0))[1] == (_z().test(0, 0, 0.5, 1.0), _z().depth(0, 0))[1]),
    ("no_mut", lambda: _z().test(0, 0, 0.5, 1.0) is not None),
    ("persp", lambda: isinstance(_z().perspective(0.5, (1.0, 2.0), (1.0, 3.0)), tuple)),
    ("nan", lambda: _raises(lambda: _z().test(0, 0, float('nan'), 1.0))),
    ("bool_test", lambda: isinstance(_z().test(1, 1, 0.5, 1.0), bool)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
