import math

import pytest

import msaa

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

TRI = ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))

CASES = [
    ("full_inside", lambda: abs(msaa.Msaa(4).coverage(TRI, 0.5, 0.5) - 1.0) < 1e-6),
    ("full_outside", lambda: msaa.Msaa(4).coverage(TRI, 5.0, 5.0) == 0.0),
    ("edge_partial", lambda: 0.0 < msaa.Msaa(8).coverage(TRI, 1.5, 0.5) < 1.0),
    ("weights_sum", lambda: abs(sum(msaa.Msaa(4).weights()) - 1.0) < 1e-12),
    ("bad_samples", lambda: _raises(lambda: msaa.Msaa(3))),
    ("resolve_len", lambda: len(msaa.Msaa(4).resolve([(1.0, 0.0, 0.0)] * 4)) == 4),
    ("resolve_avg", lambda: abs(msaa.Msaa(4).resolve([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)])[0] - 0.5) < 1e-9),
    ("bad_resolve", lambda: _raises(lambda: msaa.Msaa(4).resolve([]))),
    ("depth_resolve", lambda: isinstance(msaa.Msaa(4).resolve_depth([0.5, 0.2, 0.9, 0.7]), float)),
    ("type", lambda: isinstance(msaa.Msaa(4).coverage(TRI, 0.5, 0.5), float)),
    ("det", lambda: msaa.Msaa(4).coverage(TRI, 0.5, 0.5) == msaa.Msaa(4).coverage(TRI, 0.5, 0.5)),
    ("nan", lambda: _raises(lambda: msaa.Msaa(4).coverage(TRI, float('nan'), 0.5))),
    ("no_mut", lambda: msaa.Msaa(4).coverage(TRI, 0.5, 0.5) is not None),
    ("eight_samples", lambda: abs(msaa.Msaa(8).coverage(TRI, 0.5, 0.5) - 1.0) < 1e-6),
    ("weights_type", lambda: isinstance(msaa.Msaa(4).weights(), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
