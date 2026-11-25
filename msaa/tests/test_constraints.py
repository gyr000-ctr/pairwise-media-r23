"""msaa 约束测试：覆盖率精度、权重、性能、并发、错误分类。

与 tests/test_core.py（验收用例）互补，覆盖 README 中“评审侧隐藏用例”的
边界要求。
"""

import math
import threading

import pytest

import msaa

TRI = ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))
ROT = ((0.1, 0.2), (2.7, 1.3), (0.4, 2.9))


def _exact_area(tri, px, py):
    """用 Fraction 独立实现的多边形裁剪，作为几何真值。"""
    from fractions import Fraction as F

    def fr(v):
        return F(str(v))

    (ax, ay), (bx, by), (cx, cy) = [(fr(p[0]), fr(p[1])) for p in tri]
    h = F(1, 2)
    poly = [
        (fr(px) - h, fr(py) - h),
        (fr(px) + h, fr(py) - h),
        (fr(px) + h, fr(py) + h),
        (fr(px) - h, fr(py) + h),
    ]
    d = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if d < 0:
        (bx, by), (cx, cy) = (cx, cy), (bx, by)
    for (ex, ey), (fx, fy) in (
        ((ax, ay), (bx, by)),
        ((bx, by), (cx, cy)),
        ((cx, cy), (ax, ay)),
    ):
        A = -(fy - ey)
        B = fx - ex
        C = (fy - ey) * ex - (fx - ex) * ey  # A*x + B*y + C >= 0
        if not poly:
            break
        out = []
        cur = poly[-1]
        dp = A * cur[0] + B * cur[1] + C
        for q in poly:
            dq = A * q[0] + B * q[1] + C
            if dq >= 0:
                if dp < 0:
                    t = dp / (dp - dq)
                    out.append(
                        (cur[0] + t * (q[0] - cur[0]), cur[1] + t * (q[1] - cur[1]))
                    )
                out.append(q)
            elif dp >= 0:
                t = dp / (dp - dq)
                out.append(
                    (cur[0] + t * (q[0] - cur[0]), cur[1] + t * (q[1] - cur[1]))
                )
            cur, dp = q, dq
        poly = out
    if len(poly) < 3:
        return 0.0
    s = F(0)
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        s += x0 * y1 - x1 * y0
        x0, y0 = x1, y1
    return float(abs(s) / 2)


@pytest.mark.parametrize("n", msaa.MSAA_SAMPLES)
def test_weights_sum_and_layout(n):
    m = msaa.Msaa(n)
    w = m.weights()
    assert isinstance(w, list) and len(w) == n
    assert abs(sum(w) - 1.0) <= 1e-12
    assert all(wi > 0 for wi in w)
    # 采样点全部严格位于像素方格内
    assert all(abs(x) < 0.5 and abs(y) < 0.5 for x, y in m.template)


def test_four_x_is_subset_of_eight_x():
    assert set(msaa.Msaa(4).template) <= set(msaa.Msaa(8).template)


@pytest.mark.parametrize("tri", [TRI, ROT, tuple(reversed(TRI))])
@pytest.mark.parametrize("n", [4, 8])
def test_edge_coverage_matches_exact_area(tri, n):
    m = msaa.Msaa(n)
    for px in range(-1, 5):
        for py in range(-1, 5):
            cov = m.coverage(tri, float(px), float(py))
            truth = _exact_area(tri, float(px), float(py))
            assert abs(cov - truth) <= 1e-6, (tri, n, px, py, cov, truth)


def test_full_inside_outside_bit_exact():
    m = msaa.Msaa(4)
    assert m.coverage(TRI, 0.5, 0.5) == 1.0
    assert m.coverage(TRI, 5.0, 5.0) == 0.0
    # 相切（只接触边界）也必须是精确 0
    assert m.coverage(TRI, 2.5, 0.5) == 0.0


def test_winding_order_irrelevant():
    a = msaa.Msaa(8)
    assert a.coverage(TRI, 1.5, 0.5) == a.coverage(tuple(reversed(TRI)), 1.5, 0.5)
    assert a.coverage(TRI, 0.5, 0.5) == a.coverage(tuple(reversed(TRI)), 0.5, 0.5)


def test_depth_keeps_nearest():
    m = msaa.Msaa(4)
    assert m.resolve_depth([0.5, 0.2, 0.9, 0.7]) == 0.2
    assert m.resolve_depth([3.0, 1.5, 2.0, 4.0]) == 1.5
    assert isinstance(m.resolve_depth([0.5, 0.2, 0.9, 0.7]), float)


def test_color_weighted_average():
    m = msaa.Msaa(4)
    out = m.resolve([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] * 2)
    assert abs(out[0] - 0.5) < 1e-9
    # RGBA 全 1 → 精确 1
    assert m.resolve([(1.0, 1.0, 1.0, 1.0)] * 4) == (1.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "bad",
    [3, 0, -1, 5, 6, 32],
)
def test_invalid_sample_counts_value_errors(bad):
    with pytest.raises(ValueError):
        msaa.Msaa(bad)


@pytest.mark.parametrize("bad", [None, "4", True, 4.5])
def test_invalid_sample_counts_type_and_value_errors(bad):
    with pytest.raises(ValueError):
        msaa.Msaa(bad)
    with pytest.raises(msaa.MsaaError):
        msaa.Msaa(bad)


def test_type_error_distinguishable_from_value_error():
    with pytest.raises(TypeError):
        msaa.Msaa(None)
    with pytest.raises(TypeError):
        msaa.Msaa("4")
    # 取值错误的实例也同时是 ValueError 与 TypeError（可被任一 except 捕获）
    with pytest.raises(ValueError):
        msaa.Msaa(3)


@pytest.mark.parametrize(
    "fn",
    [
        lambda m: m.coverage(TRI, float("nan"), 0.5),
        lambda m: m.coverage(TRI, 0.5, float("inf")),
        lambda m: m.coverage(((0, 0), (1, float("nan")), (0, 1)), 0.5, 0.5),
        lambda m: m.coverage(((0, 0), (1, 1), (2, 2)), 0.5, 0.5),
        lambda m: m.resolve([(float("nan"), 0, 0)] * 4),
        lambda m: m.resolve_depth([float("nan")] * 4),
    ],
)
def test_nonfinite_and_degenerate_classified(fn):
    m = msaa.Msaa(4)
    with pytest.raises(ValueError):
        fn(m)


def test_resolve_and_depth_length_validation():
    m = msaa.Msaa(4)
    with pytest.raises(ValueError):
        m.resolve([])
    with pytest.raises(ValueError):
        m.resolve([(0, 0, 0)] * 3)
    with pytest.raises(ValueError):
        m.resolve_depth([0.1] * 5)


def test_no_shared_mutation_and_determinism():
    m = msaa.Msaa(4)
    first = m.coverage(ROT, 1.3, 1.7)
    assert m.coverage(ROT, 1.3, 1.7) == first
    w = m.weights()
    w[0] = 99.0
    assert m.weights()[0] != 99.0
    try:
        m.template[0] = (9.0, 9.0)  # tuple 不可变
    except TypeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("template 应不可变")


def test_concurrent_use_is_safe():
    m = msaa.Msaa(4)
    ref = m.coverage(ROT, 1.3, 1.7)
    errors = []

    def work():
        try:
            for i in range(500):
                if not 0.0 <= m.coverage(ROT, i % 7 * 0.5, i % 5 * 0.5) <= 1.0:
                    raise AssertionError("coverage 越界")
                if m.coverage(ROT, 1.3, 1.7) != ref:
                    raise AssertionError("并发下结果不一致")
                if m.resolve([(0.2, 0.3, 0.4, 0.5)] * 4)[0] <= 0:
                    raise AssertionError("resolve 异常")
                if m.resolve_depth([0.5, 0.1, 0.9, 0.3]) != 0.1:
                    raise AssertionError("resolve_depth 异常")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_one_million_pixels_under_500ms():
    np = pytest.importorskip("numpy")
    tri = ((0.0, 0.0), (3000.0, 10.0), (10.0, 3000.0))
    xs, ys = np.meshgrid(
        np.arange(1000, dtype=float) + 0.5, np.arange(1000, dtype=float) + 0.5
    )
    xs = xs.ravel()
    ys = ys.ravel()
    assert xs.size == 1_000_000
    m = msaa.Msaa(4)
    import time

    t0 = time.perf_counter()
    cov = m.coverage(tri, xs, ys)
    dt = time.perf_counter() - t0
    assert cov.shape == xs.shape
    assert 0.0 <= cov.min() and cov.max() <= 1.0
    assert dt < 0.5, f"coverage 1M 像素耗时 {dt*1000:.0f} ms"

    colors = np.random.default_rng(0).random((1_000_000, 4, 4))
    t0 = time.perf_counter()
    out = m.resolve(colors)
    dt = time.perf_counter() - t0
    assert out.shape == (1_000_000, 4)
    assert dt < 0.5, f"resolve 1M 像素耗时 {dt*1000:.0f} ms"

    depths = np.random.default_rng(1).random((1_000_000, 4))
    t0 = time.perf_counter()
    d = m.resolve_depth(depths)
    dt = time.perf_counter() - t0
    assert d.shape == (1_000_000,)
    assert dt < 0.5, f"resolve_depth 1M 像素耗时 {dt*1000:.0f} ms"


def test_vector_path_matches_scalar():
    np = pytest.importorskip("numpy")
    m = msaa.Msaa(8)
    lo = np.arange(-1, 5, dtype=float)
    x, y = np.meshgrid(lo, lo)
    vec = m.coverage(ROT, x, y)
    scalar = np.array(
        [[m.coverage(ROT, float(a), float(b)) for a in lo] for b in lo]
    )
    assert np.array_equal(vec, scalar)
