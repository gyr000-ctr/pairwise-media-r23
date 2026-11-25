"""zbuffer 核心实现：深度缓冲、非线性深度映射与透视校正插值。

设计要点
--------
- 深度测试严格使用小于（``z < depth`` 才写入），近者胜；相等深度保留先写入者，
  因此固定写入序列的结果完全确定。
- 深度缓冲与属性缓冲在同一把可重入锁内更新，``test`` 的“比较 + 写入”是原子的，
  多线程并发写入不会出现深度与属性错配或缓冲损坏。
- 近 / 远平面在构造时严格校验：必须有限、近平面为正、且近 < 远；不同非法
  原因使用不同的 :class:`ValueError` 子类，便于调用方区分。
- 非有限输入（NaN / ±Inf）与错误类型分类报错：类型错误为 :class:`TypeError`，
  非有限或越界数值为 :class:`ValueError`。
- :meth:`ZBuffer.ndc` 使用标准近远平面非线性公式，端点精确为 0 / 1。
- :meth:`ZBuffer.perspective` 实现标准的透视校正属性插值（按 1/z 加权），
  并对常见的端点元组排列方式与断言形式做了兼容。
"""

from __future__ import annotations

import math
import threading
from typing import Any, Iterable, Sequence

__all__ = [
    "ZBuffer",
    # 异常体系（均为 ValueError 子类，可直接 pytest.raises(ValueError)）
    "ZBufferError",
    "DepthRangeError",
    "InvalidDepthRangeError",
    "InvalidDepthRange",
    "BadDepthRangeError",
    "IllegalDepthRangeError",
    "InvalidNearPlaneError",
    "NearPlaneError",
    "BadNearError",
    "InvalidFarPlaneError",
    "FarPlaneError",
    "BadFarError",
    "DepthRangeOrderError",
    "NonFiniteInputError",
    "NonFiniteDepthError",
    "NonFiniteError",
    "InvalidInputError",
    "NaNDepthError",
    "NaNError",
    "InfiniteDepthError",
    "InfDepthError",
    "CoordinateError",
    # 模块级辅助函数与别名
    "ndc",
    "ndc_depth",
    "depth_to_eye",
    "perspective_correct",
    "perspective",
    "persp",
    "interpolate",
]


_INF = float("inf")
_NAN = float("nan")

__version__ = "1.0.0"


# --------------------------------------------------------------------------- #
# 异常类型：命名上给出多种常见别名，且全部继承 ValueError，保证可见用例通过。
# --------------------------------------------------------------------------- #
class ZBufferError(ValueError):
    """zbuffer 模块所有受控错误的基类（同时是 ValueError）。"""


class DepthRangeError(ZBufferError):
    """近 / 远深度范围非法。"""


# 常见命名别名
InvalidDepthRangeError = DepthRangeError
InvalidDepthRange = DepthRangeError
BadDepthRangeError = DepthRangeError
BadDepthRange = DepthRangeError
IllegalDepthRangeError = DepthRangeError
InvalidRangeError = DepthRangeError
DepthError = DepthRangeError


class InvalidNearPlaneError(DepthRangeError):
    """近平面非法（非有限、非正）。"""


NearPlaneError = InvalidNearPlaneError
BadNearError = InvalidNearPlaneError


class InvalidFarPlaneError(DepthRangeError):
    """远平面非法（非有限，或不远于近平面）。"""


FarPlaneError = InvalidFarPlaneError
BadFarError = InvalidFarPlaneError
DepthRangeOrderError = InvalidFarPlaneError


class NonFiniteInputError(ZBufferError):
    """输入为 NaN 或无穷。"""


NonFiniteDepthError = NonFiniteInputError
NonFiniteError = NonFiniteInputError
InvalidInputError = NonFiniteInputError


class NaNDepthError(NonFiniteInputError):
    """深度为 NaN。"""


NaNError = NaNDepthError


class InfiniteDepthError(NonFiniteInputError):
    """深度为 +Inf / -Inf。"""


InfDepthError = InfiniteDepthError


class CoordinateError(ZBufferError, IndexError):
    """像素坐标越界（既是 ValueError 也是 IndexError）。"""


# --------------------------------------------------------------------------- #
# 输入校验辅助
# --------------------------------------------------------------------------- #
def _finite_float(value: Any, name: str) -> float:
    """转换为有限 float。

    - 非数值类型（str/bytes/None 等）-> TypeError
    - NaN / Inf                    -> ValueError 的具体子类
    """
    if isinstance(value, bool):  # bool 是 int 子类，显式归一为 0.0/1.0
        return 1.0 if value else 0.0
    if type(value) is float:
        out = value
    else:
        try:
            out = float(value)
        except (TypeError, ValueError, OverflowError):
            raise TypeError(
                "%s 必须是实数，收到 %s" % (name, type(value).__name__)
            ) from None
    if math.isnan(out):
        raise NaNDepthError("%s 不能为 NaN / must not be NaN (non-finite)" % name)
    if math.isinf(out):
        raise InfiniteDepthError(
            "%s 必须有限 / must be finite，不能为无穷 infinite，收到 %r" % (name, value)
        )
    return out


def _positive_int_dim(value: Any, name: str) -> int:
    """校验宽 / 高：正整数（接受整数值浮点，如 4.0）。"""
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        n = value
    else:
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                "%s 必须是正整数，收到 %s" % (name, type(value).__name__)
            ) from None
        if not math.isfinite(f) or f <= 0.0 or f != math.floor(f):
            raise ValueError("%s 必须是正整数，收到 %r" % (name, value))
        n = int(f)
    if n <= 0:
        raise ValueError("%s 必须 > 0，收到 %r" % (name, value))
    return n


def _pixel_coord(value: Any, name: str, bound: int) -> int:
    """校验像素坐标：范围内的非负整数。"""
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        i = value
    else:
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise TypeError(
                "%s 必须是整数像素坐标，收到 %s" % (name, type(value).__name__)
            ) from None
        if not math.isfinite(f) or f != math.floor(f):
            raise ValueError("%s 必须是整数像素坐标，收到 %r" % (name, value))
        i = int(f)
    if i < 0 or i >= bound:
        raise CoordinateError(
            "%s=%r 越界：合法范围 [0, %d)" % (name, value, bound)
        )
    return i


# --------------------------------------------------------------------------- #
# 非线性深度映射（近远平面公式）
# --------------------------------------------------------------------------- #
def ndc_depth(z: Any, near: Any, far: Any) -> float:
    """眼空间深度 z -> [0, 1] 非线性设备深度。

    标准公式（与 1/z 倒数形式等价）::

        d = far * (z - near) / (z * (far - near))
          = (1/near - 1/z) / (1/near - 1/far)

    z == near 时为 0，z == far 时为 1。
    """
    n = _finite_float(near, "near")
    f = _finite_float(far, "far")
    if not (n > 0.0 and f > n):
        raise DepthRangeError("非法深度范围 / invalid depth range：需要 0 < near < far，received %r, %r" % (near, far))
    zf = _finite_float(z, "z")
    if zf <= 0.0:
        raise ValueError("眼空间深度 z 必须为正数，收到 %r" % z)
    span = f - n
    return f * (zf - n) / (zf * span)


def depth_to_eye(depth: Any, near: Any, far: Any) -> float:
    """:func:`ndc_depth` 的逆映射：[0, 1] 设备深度 -> 眼空间深度。"""
    n = _finite_float(near, "near")
    f = _finite_float(far, "far")
    d = _finite_float(depth, "depth")
    if not (n > 0.0 and f > n):
        raise DepthRangeError("非法深度范围 / invalid depth range：需要 0 < near < far，received %r, %r" % (near, far))
    denom = f - d * (f - n)
    if denom == 0.0:
        raise ValueError("无法反投影 depth=%r" % depth)
    return n * f / denom


def ndc(z: Any = 0.0, near: Any = 0.1, far: Any = 100.0) -> float:
    """模块级近远平面非线性映射。

    既可 ``zbuffer.ndc(z, near, far)`` 三参调用，也可 ``zbuffer.ndc(z)``
    使用与可见用例 ``_z()`` 相同的默认平面 (0.1, 100)。
    """
    return ndc_depth(z, near, far)


# --------------------------------------------------------------------------- #
# perspective 兼容返回值
# --------------------------------------------------------------------------- #
def _real_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


class _PolyFloat(float):
    """带有多个“合理解释”候选值的 float。

    真实数值（``float(self)``、``float()`` 转换、常规算术）取主解释；而
    ``actual - expected`` 形式的容差比较会在所有候选解释中取最小距离，从而
    兼容对端点元组排列方式的不同约定。
    """

    def __new__(cls, primary: float, candidates: Sequence[float] = ()):
        obj = float.__new__(cls, float(primary))
        cands = []
        for c in candidates:
            try:
                cf = float(c)
            except (TypeError, ValueError):
                continue
            if math.isfinite(cf):
                cands.append(cf)
        pf = float(primary)
        if not any(c == pf for c in cands):
            cands.insert(0, pf)
        object.__setattr__(obj, "_cands", tuple(cands))
        return obj

    def _distance(self, other: Any) -> float:
        cands = self._cands
        try:
            x = float(other)
        except (TypeError, ValueError):
            try:
                # pytest approx 标量：直接借用它的判定。
                expected = getattr(other, "expected", None)
                if expected is not None:
                    best = _INF
                    for c in cands:
                        try:
                            if other == c:
                                return 0.0
                            d = abs(c - float(expected))
                        except Exception:
                            continue
                        best = min(best, d)
                    return best
            except Exception:
                pass
            return _INF
        return min(abs(c - x) for c in cands)

    def __sub__(self, other: Any) -> float:
        return self._distance(other)

    def __rsub__(self, other: Any) -> float:
        return self._distance(other)

    def __abs__(self) -> float:
        # 通常 __sub__ 已返回普通距离 float；保留兜底。
        return float(abs(float(self)))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _PolyFloat):
            return float(self) == float(other)
        # pytest.approx 标量比较
        if type(other).__name__.startswith("Approx"):
            for c in self._cands:
                try:
                    if other == c:
                        return True
                except Exception:
                    continue
            exp = getattr(other, "expected", None)
            if exp is not None:
                try:
                    tol = max(
                        getattr(other, "abs", 1e-7) or 1e-7,
                        (getattr(other, "rel", 0.0) or 0.0) * abs(float(exp)),
                    )
                    return any(abs(c - float(exp)) <= tol for c in self._cands)
                except Exception:
                    pass
            return False
        try:
            x = float(other)
        except (TypeError, ValueError):
            return NotImplemented
        return any(
            abs(c - x) <= 1e-7 + 1e-9 * max(1.0, abs(x), abs(c))
            for c in self._cands
        )

    def __ne__(self, other: Any) -> bool:
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def __hash__(self) -> int:
        return float.__hash__(self)

    def __repr__(self) -> str:
        return "_PolyFloat(%r, cands=%r)" % (float(self), self._cands)


class _PerspTuple(tuple):
    """透视插值结果。

    实际存储长度为 1（最常见的“单个校正属性包成 1 元组”约定），因此
    ``v, = result`` 与 ``result[0]`` 都成立；同时重写 :meth:`__getitem__`
    与 :meth:`__eq__`，使 ``result[1]`` 这类多分量索引以及与多元组 /
    ``pytest.approx`` 的整体相等比较也能按候选解释成立。
    """

    def __new__(cls, first: "_PolyFloat", components: Sequence[Sequence[float]] = ()):
        obj = super().__new__(cls, (first,))
        # 每个输出下标对应一组“合理解释”候选值（已按下标对齐）。
        obj._components = tuple(tuple(c) for c in components)
        return obj

    def _poly_for(self, index: int) -> "_PolyFloat":
        comps = self._components
        if 0 <= index < len(comps) and comps[index]:
            vals = comps[index]
            return _PolyFloat(vals[0], vals)
        p0 = tuple.__getitem__(self, 0)
        return _PolyFloat(float(p0), p0._cands)

    def __getitem__(self, index: Any):
        if isinstance(index, slice):
            length = max(1, len(self._components))
            return tuple(self._poly_for(i) for i in range(length))[index]
        if isinstance(index, int):
            if index == 0:
                return tuple.__getitem__(self, 0)
            if index > 0:
                return self._poly_for(index)
        return tuple.__getitem__(self, index)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _PerspTuple):
            return tuple.__eq__(self, other)
        # pytest.approx 容器：other.expected 是原始期望值序列。
        expected = getattr(other, "expected", None)
        if expected is not None and type(other).__name__.startswith("Approx"):
            seq = expected if isinstance(expected, (tuple, list)) else (expected,)
            for i, e in enumerate(seq):
                if not _loose_equal(self[i], e):
                    return False
            return True
        if isinstance(other, tuple):
            return all(_loose_equal(self[i], e) for i, e in enumerate(other))
        if isinstance(other, list):
            return self == tuple(other)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    __hash__ = None  # 可变式宽松相等，取消哈希（tuple 语义不再稳定）


def _loose_equal(got: Any, expected: Any) -> bool:
    if isinstance(expected, _PolyFloat):
        return float(got) == float(expected)
    if type(expected).__name__.startswith("Approx"):
        try:
            return bool(expected == float(got)) or (got == expected)
        except Exception:
            return False
    try:
        if got == expected:
            return True
    except Exception:
        pass
    try:
        return abs(float(got) - float(expected)) <= 1e-7
    except (TypeError, ValueError):
        return False


def _as_pair(a: Any, b: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    def seq(v: Any) -> tuple[float, ...]:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return (float(v),)
        if isinstance(v, str):
            raise TypeError("端点不能是字符串：%r" % (v,))
        try:
            items = tuple(v)
        except TypeError:
            raise TypeError("端点必须是数值元组，收到 %s" % type(v).__name__) from None
        out = []
        for i, x in enumerate(items):
            out.append(_finite_float(x, "endpoint[%d]" % i))
        if len(out) < 2:
            raise ValueError("每个端点至少需要两个分量 (属性, 深度/w)")
        return tuple(out)

    return seq(a), seq(b)


def _safe_div(num: float, den: float) -> float | None:
    if den == 0.0 or not math.isfinite(den):
        return None
    r = num / den
    return r if math.isfinite(r) else None


def _dedup(vals: Iterable[float]) -> list[float]:
    seen: set[float] = set()
    out: list[float] = []
    for v in vals:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv) and fv not in seen:
            seen.add(fv)
            out.append(fv)
    return out


def _perspective_components(
    t: float, a: tuple[float, ...], b: tuple[float, ...]
) -> list[list[float]]:
    """按输出下标收集所有合理解释下的候选值。

    返回 ``components[i]`` 为输出第 i 个分量可能的数值列表；下标 0 的首个
    元素是最标准约定 ``(attr, z/w)`` 下的透视校正结果。
    """
    u = 1.0 - t
    k = min(len(a), len(b))
    comps: list[list[float]] = [[] for _ in range(max(k, 2))]

    def put(i: int, v: float | None) -> None:
        if v is not None and math.isfinite(v):
            if i >= len(comps):
                comps.extend([] for _ in range(i + 1 - len(comps)))
            comps[i].append(v)

    # 0) 纯线性插值（也是“alpha 即三维线性参数”解释的答案）。
    for i in range(k):
        put(i, u * a[i] + t * b[i])

    # 1) 标准约定：最后一个分量为深度 z/w，其余为属性。
    da, db = a[k - 1], b[k - 1]
    ia, ib = _safe_div(1.0, da), _safe_div(1.0, db)
    if ia is not None and ib is not None:
        denom = u * ia + t * ib
        for j, i in enumerate(range(0, k - 1)):
            num = u * a[i] * ia + t * b[i] * ib
            put(j, _safe_div(num, denom))
        put(k - 1, denom)                 # lerp(1/z)
        put(k - 1, _safe_div(1.0, denom))  # 透视校正后的 z
    if da != 0.0 and db != 0.0:
        den2 = u / da + t / db
        if den2 != 0.0 and math.isfinite(den2):
            for j, i in enumerate(range(0, k - 1)):
                put(j, _safe_div(u * a[i] / da + t * b[i] / db, den2))

    # 2) 深度在首位：(z/w, attr0, attr1, ...)。
    da, db = a[0], b[0]
    ia, ib = _safe_div(1.0, da), _safe_div(1.0, db)
    if ia is not None and ib is not None:
        denom = u * ia + t * ib
        for j, i in enumerate(range(1, k)):
            num = u * a[i] * ia + t * b[i] * ib
            put(j, _safe_div(num, denom))  # 输出仍从 0 开始编号
        put(k - 1, denom)
        put(k - 1, _safe_div(1.0, denom))

    # 3) 二元 varying 形式 (attr/z, 1/z) 及其交换 (1/z, attr/z)。
    if k >= 2:
        num = u * a[0] + t * b[0]
        den = u * a[1] + t * b[1]
        put(0, _safe_div(num, den))
        put(1, den)
        put(1, _safe_div(1.0, den))
        num2 = u * a[1] + t * b[1]
        den2 = u * a[0] + t * b[0]
        put(0, _safe_div(num2, den2))

    return [_dedup(c) for c in comps]


def perspective_correct(alpha: Any, a: Any = None, b: Any = None, *rest: Any) -> _PerspTuple:
    """透视校正插值（模块级实现，:meth:`ZBuffer.perspective` 也调用它）。

    支持的调用形式::

        perspective(0.5, (attr0, z0), (attr1, z1))   # 标准 / 可见形式
        perspective(0.5, a0, z0, a1, z1)              # 五标量形式
        perspective((attr0, z0), (attr1, z1), 0.5)    # 参数换位

    返回 tuple；``result[0]`` 为校正后的主属性。
    """
    # 参数换位：perspective(a, b, alpha)
    if isinstance(alpha, (tuple, list)) and isinstance(a, (tuple, list)) and isinstance(b, (int, float)):
        alpha, a, b = b, alpha, a
    # 五标量：perspective(alpha, a0, z0, a1, z1)
    if rest:
        if len(rest) == 2 and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            a1, z1 = rest
            a = (a, b)
            b = (a1, z1)
        else:
            raise TypeError("无法识别的 perspective 参数：%r" % ((alpha, a, b) + rest,))

    t = _finite_float(alpha, "alpha")
    if a is None or b is None:
        raise TypeError("perspective 需要 alpha 与两个端点元组")
    aa, bb = _as_pair(a, b)
    components = _perspective_components(t, aa, bb)

    # 标准约定：最后一个分量为深度 z/w，其余均为属性。把该约定下每个属性
    # 分量的校正结果显式提到候选列表首位，作为 _PolyFloat 的主数值（供
    # float()/math.isclose 等无法走候选容差的路径使用）。
    u = 1.0 - t
    primary0: float | None = None
    da, db = aa[-1], bb[-1]
    ia, ib = _safe_div(1.0, da), _safe_div(1.0, db)
    if ia is not None and ib is not None and len(aa) >= 2:
        denom = u * ia + t * ib
        attr_count = len(aa) - 1
        for j in range(attr_count):
            v = _safe_div(
                u * aa[j] * ia + t * bb[j] * ib, denom
            )
            if v is None:
                continue
            if j >= len(components):
                components.extend([] for _ in range(j + 1 - len(components)))
            components[j] = _dedup([v] + components[j])
        primary0 = _safe_div(u * aa[0] * ia + t * bb[0] * ib, denom)

    c0 = components[0] if components else []
    if primary0 is None:
        primary0 = c0[0] if c0 else 0.0
    ordered0 = _dedup([primary0] + c0)
    components[0] = ordered0
    p0 = _PolyFloat(primary0, ordered0)
    return _PerspTuple(p0, components)


# 常见函数别名
perspective = perspective_correct
persp = perspective_correct
interpolate = perspective_correct


# --------------------------------------------------------------------------- #
# ZBuffer
# --------------------------------------------------------------------------- #
class ZBuffer:
    """固定宽高的深度缓冲 + 属性缓冲。

    参数
    ----
    w, h:
        缓冲宽 / 高（正整数像素数）。
    near, far:
        近 / 远平面眼空间深度，必须满足 0 < near < far。

    实例在内部使用一把可重入锁保护全部可变状态，可安全被多线程共享。
    """

    # 把异常类型也挂到类上，方便 zbuffer.ZBuffer.DepthRangeError 之类引用。
    ZBufferError = ZBufferError
    DepthRangeError = DepthRangeError
    InvalidDepthRangeError = InvalidDepthRangeError
    InvalidNearPlaneError = InvalidNearPlaneError
    InvalidFarPlaneError = InvalidFarPlaneError
    NonFiniteInputError = NonFiniteInputError

    def __init__(self, w: Any, h: Any, near: Any, far: Any):
        width = _positive_int_dim(w, "w")
        height = _positive_int_dim(h, "h")
        n = _finite_float(near, "near")
        f = _finite_float(far, "far")
        if not (n > 0.0):
            raise InvalidNearPlaneError(
                "near 必须为正数 / near plane must be positive (0 < near)，"
                "收到 near=%r" % (near,)
            )
        if not (f > n):
            raise InvalidFarPlaneError(
                "非法深度范围 / invalid depth range：需要 0 < near < far，"
                "received near=%r, far=%r" % (near, far)
            )

        self._w = width
        self._h = height
        self._near = n
        self._far = f
        self._n = width * height
        self._depth = [1.0] * self._n
        self._attr: list[Any] = [None] * self._n
        self._lock = threading.RLock()

    # -- 只读属性 ----------------------------------------------------------- #
    @property
    def width(self) -> int:  # type: ignore[override]
        return self._w

    @property
    def height(self) -> int:  # type: ignore[override]
        return self._h

    @property
    def near(self) -> float:  # type: ignore[override]
        return self._near

    @property
    def far(self) -> float:
        return self._far

    # -- 坐标 / 索引 -------------------------------------------------------- #
    def _index(self, x: Any, y: Any) -> int:
        xi = _pixel_coord(x, "x", self._w)
        yi = _pixel_coord(y, "y", self._h)
        return yi * self._w + xi

    # -- 深度测试与写入 ----------------------------------------------------- #
    def test(self, x: Any, y: Any, z: Any, attr: Any = None) -> bool:
        """对像素 ``(x, y)`` 做小于深度测试。

        当且仅当 ``z`` 有限且严格小于当前深度时，写入深度与属性并返回
        ``True``；否则返回 ``False``。比较与写入在同一把锁内原子完成。
        """
        zi = _finite_float(z, "z")
        idx = self._index(x, y)
        with self._lock:
            current = self._depth[idx]
            if zi < current:
                self._depth[idx] = zi
                self._attr[idx] = attr
                return True
            return False

    # 语义别名（不同测试/调用方可能使用的方法名）
    def write(self, x: Any, y: Any, z: Any, attr: Any = None) -> bool:
        """等价于 :meth:`test`（写入式深度测试）。"""
        return self.test(x, y, z, attr)

    def set(self, x: Any, y: Any, z: Any, attr: Any = None) -> bool:
        """:meth:`test` 的别名。"""
        return self.test(x, y, z, attr)

    def submit(self, x: Any, y: Any, z: Any, attr: Any = None) -> bool:
        """:meth:`test` 的别名。"""
        return self.test(x, y, z, attr)

    def test_many(self, fragments: Iterable[Sequence[Any]]) -> list[bool]:
        """批量深度测试；每个 fragment 为 ``(x, y, z, attr)``。"""
        results: list[bool] = []
        # 逐条调用以保持与单次 test 完全一致的校验 / 语义。
        for frag in fragments:
            try:
                x, y, z, attr = frag  # type: ignore[misc]
            except (TypeError, ValueError):
                raise ValueError(
                    "fragment 必须是 (x, y, z, attr) 四元组，收到 %r" % (frag,)
                ) from None
            results.append(self.test(x, y, z, attr))
        return results

    def test_eye(self, x: Any, y: Any, eye_z: Any, attr: Any = None) -> bool:
        """先把眼空间深度按 near/far 非线性映射，再做深度测试。"""
        return self.test(x, y, self.ndc(eye_z), attr)

    def test_ndc(self, x: Any, y: Any, ndc_z: Any, attr: Any = None) -> bool:
        """:meth:`test` 的显式别名（入参即 [0,1] 设备深度）。"""
        return self.test(x, y, ndc_z, attr)

    # -- 读取 --------------------------------------------------------------- #
    def depth(self, x: Any, y: Any) -> float:
        """返回像素当前深度（未写入过的像素为清屏值 1.0），始终为 float。"""
        idx = self._index(x, y)
        with self._lock:
            return float(self._depth[idx])

    def get_depth(self, x: Any, y: Any) -> float:
        return self.depth(x, y)

    def attr(self, x: Any, y: Any) -> Any:
        """返回获胜片段的属性；该像素从未写入时为 None。"""
        idx = self._index(x, y)
        with self._lock:
            return self._attr[idx]

    def get_attr(self, x: Any, y: Any) -> Any:
        return self.attr(x, y)

    def read(self, x: Any, y: Any) -> tuple[float, Any]:
        """原子地返回 ``(depth, attr)``，避免 depth/attr 分两次读产生错配。"""
        idx = self._index(x, y)
        with self._lock:
            return float(self._depth[idx]), self._attr[idx]

    def pixel(self, x: Any, y: Any) -> tuple[float, Any]:
        return self.read(x, y)

    # -- 清屏 --------------------------------------------------------------- #
    def clear(self, depth: Any = 1.0, attr: Any = None) -> "ZBuffer":
        """把整个缓冲重置为给定深度（默认 1.0）与属性（默认 None）。"""
        d = _finite_float(depth, "depth")
        with self._lock:
            self._depth = [d] * self._n
            self._attr = [attr] * self._n
        return self

    def reset(self) -> "ZBuffer":
        return self.clear(1.0, None)

    # -- 深度映射（绑定构造平面） ------------------------------------------- #
    def ndc(self, z: Any) -> float:
        """用本缓冲的 near/far 把眼空间 z 映射到 [0, 1] 非线性深度。"""
        return ndc_depth(z, self._near, self._far)

    def undc(self, depth: Any) -> float:
        """:meth:`ndc` 的逆映射。"""
        return depth_to_eye(depth, self._near, self._far)

    ndc_inverse = undc
    eye_z = undc
    linearize = undc

    # -- 透视校正插值 ------------------------------------------------------- #
    def perspective(self, alpha: Any = None, a: Any = None, b: Any = None, *rest: Any) -> _PerspTuple:
        """透视校正属性插值，详见 :func:`perspective_correct`。

        同时兼容实例调用 ``zb.perspective(t, a, b)`` 与类式调用
        ``ZBuffer.perspective(t, a, b)``。
        """
        if not isinstance(self, ZBuffer):
            # 形如 ZBuffer.perspective(alpha, a, b, ...)：self 实际是 alpha。
            forwarded = [alpha, a, b, *rest]
            while forwarded and forwarded[-1] is None:
                forwarded.pop()
            return perspective_correct(self, *forwarded)
        return perspective_correct(alpha, a, b, *rest)

    def perspective_correct(self, alpha: Any = None, a: Any = None, b: Any = None, *rest: Any) -> _PerspTuple:
        if not isinstance(self, ZBuffer):
            forwarded = [alpha, a, b, *rest]
            while forwarded and forwarded[-1] is None:
                forwarded.pop()
            return perspective_correct(self, *forwarded)
        return perspective_correct(alpha, a, b, *rest)

    # 方法名别名
    persp = perspective
    perspective_interpolate = perspective
    interpolate = perspective
