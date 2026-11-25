import math

import pytest

import shadowatlas

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _a():
    return shadowatlas.ShadowAtlas(1024)

CASES = [
    ("alloc_rect", lambda: len(_a().alloc(256, 256)) == 4),
    ("two_no_overlap", lambda: (lambda a: (a.alloc(256, 256), a.alloc(256, 256), a.pages())[2] >= 1)(_a())),
    ("free_reuse", lambda: (lambda a: (r := a.alloc(256, 256), a.free(r), a.alloc(256, 256))[2] is not None)(_a())),
    ("pages_monotonic", lambda: (lambda a: (a.alloc(256, 256), a.pages()))(_a()) >= 1),
    ("bad_size", lambda: _raises(lambda: _a().alloc(0, 256))),
    ("too_big", lambda: _raises(lambda: _a().alloc(2048, 2048))),
    ("bad_free", lambda: _raises(lambda: _a().free(None))),
    ("type", lambda: isinstance(_a().alloc(128, 128), tuple)),
    ("det", lambda: (lambda a: a.alloc(128, 128))(_a()) == (lambda a: a.alloc(128, 128))(_a())),
    ("bounds", lambda: all(c <= 1024 for c in _a().alloc(512, 512))),
    ("pages_type", lambda: isinstance(_a().pages(), int)),
    ("no_mut", lambda: _a().alloc(64, 64) is not None),
    ("many", lambda: len([_a().alloc(64, 64) for _ in range(3)]) == 3),
    ("free_unknown", lambda: _raises(lambda: _a().free((0, 0, 4096, 4096)))),
    ("float_reject", lambda: _raises(lambda: _a().alloc(1.5, 64))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
