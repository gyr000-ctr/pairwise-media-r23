"""msaa 核心 API：多重采样解析。

设计要点
--------
* 采样模板：只接受 1/2/4/8/16 路。4x 为旋转网格；8x 由 4x 每点向像素中心收缩
  得到，故 4x 点集是 8x 点集的精确子集。权重等值，权重和严格为 ``1.0``。
* 像素约定：``(px, py)`` 是像素中心，像素方格为 ``[px-.5, px+.5] x
  [py-.5, py+.5]``。
* 覆盖率：用三条边所定义半平面的有符号距离把像素方格精确分类——
  全内 → ``1.0``；全外（含相切）→ ``0.0``；只有真正被边界穿过的边缘像素才用
  三角形与像素方格的多边形裁剪求精确相交面积。全内/全外逐位精确，边缘像素
  覆盖率即几何面积。
* 颜色解析：按覆盖率含义做等权加权平均，每通道用 ``math.fsum`` 求和（向量路径
  用 numpy 的成对求和，与标量路径在 1e-12 内一致）。
* 深度：按“最近采样保留”规则取有效采样最小值，不做平均。
* 错误：``MsaaError`` 同时派生 ``ValueError`` 与 ``TypeError``，两种 ``except``
  都能命中；``MsaaValueError`` / ``MsaaTypeError`` 进一步区分取值与类型错误。
* 性能：``coverage`` / ``resolve`` / ``resolve_depth`` 接受 numpy 数组批量输入，
  一次处理上百万像素（向量化分类 + 仅对边缘像素做标量精确面积）。
* 并发安全：构造完成后全部字段只读，所有公开方法不修改实例、不使用共享可变
  状态，可在多线程下并发调用。
"""

import math

try:  # numpy 为可选加速依赖
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

__all__ = [
    "MSAA_SAMPLES",
    "Msaa",
    "MsaaError",
    "MsaaValueError",
    "MsaaTypeError",
]

#: 支持的采样数（MSAA 多重采样标准档位）
MSAA_SAMPLES = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class MsaaError(ValueError, TypeError):
    """msaa 输入错误基类。

    同时派生 ``ValueError`` 与 ``TypeError``（二者互不继承，多继承合法），
    使调用方按任一种惯例捕获都能命中。
    """


class MsaaValueError(MsaaError):
    """取值非法（采样数不支持、坐标非有限、三角形退化等）。"""


class MsaaTypeError(MsaaError):
    """类型非法（参数缺失、None、类型错误、容器形状不合法等）。"""


def _val_err(msg):
    return MsaaValueError(msg)


def _type_err(msg):
    return MsaaTypeError(msg)


# --------------------------------------------------------------------------- #
# 采样模板
# --------------------------------------------------------------------------- #
def _build_template(n):
    """构造 n 个采样点，坐标为相对像素中心的偏移（等权）。

    - 1x：像素中心；
    - 2x：对角两点；
    - 4x：2x2 旋转网格（D3D 风格）；
    - 8x：4x 每点向像素中心收缩一半再取一点，故 4x 点集是 8x 点集的子集；
    - 16x：4x4 规则网格。

    所有点严格位于像素方格内（|偏移| < 0.5）。
    """
    raw = _build_template_unit(n)
    return tuple((x - 0.5, y - 0.5) for x, y in raw)


def _build_template_unit(n):
    """单位方格 [0, 1)^2 内的采样点。"""
    if n == 1:
        return ((0.5, 0.5),)
    if n == 2:
        return ((0.25, 0.25), (0.75, 0.75))
    if n == 4:
        return ((0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875))
    if n == 8:
        out = []
        for x, y in _build_template_unit(4):
            out.append((x, y))
            out.append((x + (0.5 - x) * 0.5, y + (0.5 - y) * 0.5))
        return tuple(out)
    if n == 16:
        return tuple(
            ((i + 0.5) / 4.0, (j + 0.5) / 4.0) for j in range(4) for i in range(4)
        )
    raise _val_err(f"不支持的采样数: {n!r}；合法取值: {MSAA_SAMPLES}")


# --------------------------------------------------------------------------- #
# 输入校验辅助
# --------------------------------------------------------------------------- #
def _check_number(v, name):
    if isinstance(v, bool):
        raise _type_err(f"{name} 不能是布尔值")
    if isinstance(v, (int, float)):
        return float(v)
    try:  # 兼容 numpy 数值标量等
        return float(v)
    except (TypeError, ValueError):
        raise _type_err(f"{name} 必须是实数，收到 {type(v).__name__}") from None


def _check_finite(v, name):
    v = _check_number(v, name)
    if not math.isfinite(v):
        raise _val_err(f"{name} 必须有限，收到 {v!r}")
    return v


def _check_xy(v, name):
    if isinstance(v, (str, bytes)):
        raise _type_err(f"{name} 不是坐标")
    try:
        x, y = v
    except (TypeError, ValueError):
        raise _type_err(f"{name} 必须是 (x, y) 二元组") from None
    return _check_finite(x, f"{name}.x"), _check_finite(y, f"{name}.y")


def _check_tri(tri):
    if tri is None:
        raise _type_err("三角形不能为 None")
    if isinstance(tri, (str, bytes)):
        raise _type_err("三角形不能是字符串")
    try:
        verts = tuple(tri)
    except TypeError:
        raise _type_err("三角形必须是 3 个顶点的序列") from None
    if len(verts) != 3:
        raise _val_err(f"三角形必须恰好有 3 个顶点，收到 {len(verts)} 个")
    return (
        _check_xy(verts[0], "tri[0]"),
        _check_xy(verts[1], "tri[1]"),
        _check_xy(verts[2], "tri[2]"),
    )


def _check_deg_colors(c, idx):
    if isinstance(c, (str, bytes)):
        raise _type_err(f"采样[{idx}] 不是颜色")
    try:
        chans = tuple(c)
    except TypeError:
        raise _type_err(f"采样[{idx}] 必须是颜色序列") from None
    if not chans:
        raise _val_err(f"采样[{idx}] 颜色为空")
    if len(chans) > 4:
        raise _val_err(f"采样[{idx}] 颜色最多 4 个通道（RGBA），收到 {len(chans)}")
    vals = tuple(
        _check_finite(ch, f"采样[{idx}][{k}]") for k, ch in enumerate(chans)
    )
    # 归一为 RGBA：单通道视为灰度，双通道视为灰度+alpha，RGB 视为不透明
    if len(vals) == 1:
        return (vals[0], vals[0], vals[0], 1.0)
    if len(vals) == 2:
        return (vals[0], vals[0], vals[0], vals[1])
    if len(vals) == 3:
        return (vals[0], vals[1], vals[2], 1.0)
    return vals


def _check_seq_arg(seq, name):
    if seq is None:
        raise _type_err(f"{name}不能为 None")
    if isinstance(seq, (str, bytes)):
        raise _type_err(f"{name}不能是字符串")
    try:
        return tuple(seq)
    except TypeError:
        raise _type_err(f"{name}必须是可迭代对象") from None


def _as_float_array(v, name):
    """把批量输入转成 float64 数组，非数值/非有限即报错。"""
    try:
        arr = _np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        raise _type_err(f"{name} 必须是数值数组") from None
    if arr.dtype != _np.float64:
        try:
            arr = arr.astype(_np.float64)
        except (TypeError, ValueError):
            raise _type_err(f"{name} 必须是数值数组") from None
    if not _np.isfinite(arr).all():
        raise _val_err(f"{name} 包含非有限值")
    return arr


def _is_vec(v):
    return _np is not None and _np.ndim(v) > 0


# --------------------------------------------------------------------------- #
# 几何（与绕序、像素中心约定绑定）
# --------------------------------------------------------------------------- #
def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def _orient(tri):
    """返回 (逆时针三角形, 有向面积 2 倍)；退化时抛错。"""
    (ax, ay), (bx, by), (cx, cy) = tri
    area2 = _cross(bx - ax, by - ay, cx - ax, cy - ay)
    if not math.isfinite(area2):
        raise _val_err("三角形顶点量级过大导致面积溢出")
    scale = max(
        abs(bx - ax) + abs(by - ay),
        abs(cx - bx) + abs(cy - by),
        abs(ax - cx) + abs(ay - cy),
        1.0,
    )
    if area2 == 0.0 or abs(area2) <= (scale * scale) * 1e-15:
        raise _val_err("退化三角形（三点共线或重合），覆盖率无定义")
    if area2 < 0.0:
        return (tri[0], tri[2], tri[1]), -area2
    return tri, area2


def _edges(tri):
    (ax, ay), (bx, by), (cx, cy) = tri
    return ((ax, ay, bx, by), (bx, by, cx, cy), (cx, cy, ax, ay))


def _edge_coeffs(e):
    """边 (ex,ey)->(fx,fy) 的内部半平面 ``A*x + B*y + C >= 0``。"""
    ex, ey, fx, fy = e
    dx = fx - ex
    dy = fy - ey
    return -dy, dx, dy * ex - dx * ey


def _classify_square(edges, px, py):
    """精确分类像素方格与三角形的覆盖关系。

    通过边半平面在方格上的最小值 / 最大值判定：最大值 ≤ 0 → 整格在外（面积 0）；
    所有最小值 ≥ 0 → 整格在内（面积 1）；否则返回 None（边缘像素）。
    """
    cut = False
    for e in edges:
        A, B, C = _edge_coeffs(e)
        dc = A * px + B * py + C
        r = 0.5 * (abs(A) + abs(B))
        if dc + r <= 0.0:  # 整格都在该半平面之外
            return 0.0
        if dc - r < 0.0:  # 有角落在外部
            cut = True
    return None if cut else 1.0


def _clip_nonneg(poly, A, B, C):
    """Sutherland–Hodgman：保留 ``A*x + B*y + C >= 0`` 的部分。"""
    out = []
    if not poly:
        return out
    px, py = poly[-1]
    dp = A * px + B * py + C
    for qx, qy in poly:
        dq = A * qx + B * qy + C
        if dq >= 0.0:
            if dp < 0.0:
                t = dp / (dp - dq)
                out.append((px + t * (qx - px), py + t * (qy - py)))
            out.append((qx, qy))
        elif dp >= 0.0:
            t = dp / (dp - dq)
            out.append((px + t * (qx - px), py + t * (qy - py)))
        px, py, dp = qx, qy, dq
    return out


def _area(poly):
    """凸多边形面积（鞋带公式，取绝对值）。"""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        s += x0 * y1 - x1 * y0
        x0, y0 = x1, y1
    return abs(s) * 0.5


def _exact_area(edges, px, py):
    """边缘像素：三角形与像素方格的精确相交面积。"""
    x0 = px - 0.5
    y0 = py - 0.5
    x1 = px + 0.5
    y1 = py + 0.5
    poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    for e in edges:
        A, B, C = _edge_coeffs(e)
        poly = _clip_nonneg(poly, A, B, C)
        if len(poly) < 3:
            return 0.0
    cov = _area(poly)
    if cov <= 0.0:
        return 0.0
    if cov >= 1.0:
        return 1.0
    return cov


# --------------------------------------------------------------------------- #
# 主类
# --------------------------------------------------------------------------- #
class Msaa:
    """多重采样解析器。

    参数
    ----
    samples : int
        采样数，必须是 ``MSAA_SAMPLES`` 之一（1/2/4/8/16）。
    """

    __slots__ = ("_samples", "_weights", "_template")

    def __init__(self, samples=4):
        if samples is None:
            raise _type_err("采样数不能为 None")
        if isinstance(samples, bool):
            raise _type_err("采样数不能是布尔值")
        if isinstance(samples, float):
            if not samples.is_integer():
                raise _val_err(f"采样数必须是整数，收到 {samples!r}")
            samples = int(samples)
        elif not isinstance(samples, int):
            raise _type_err(f"采样数必须是整数，收到 {type(samples).__name__}")
        if samples not in MSAA_SAMPLES:
            raise _val_err(f"不支持的采样数: {samples}；合法取值: {MSAA_SAMPLES}")

        # 等权权重；1/2/4/8/16 均为 2 的幂，1.0/n 精确，残差补齐后和严格为 1.0
        w = 1.0 / samples
        weights = [w] * samples
        if samples > 1:
            weights[-1] = 1.0 - math.fsum(weights[:-1])

        self._samples = samples
        self._weights = tuple(weights)
        self._template = _build_template(samples)

    # -- 元数据 ---------------------------------------------------------- #
    @property
    def samples(self):
        """采样数。"""
        return self._samples

    @property
    def template(self):
        """采样点相对像素中心的偏移（只读元组）。"""
        return self._template

    def weights(self):
        """返回采样权重列表，和严格为 1。"""
        return list(self._weights)

    # -- 覆盖率 ---------------------------------------------------------- #
    def coverage(self, tri, px, py):
        """三角形 ``tri`` 对像素 ``(px, py)`` 的覆盖率，取值 [0, 1]。

        ``(px, py)`` 为像素中心。全内 → ``1.0``、全外 → ``0.0``（逐位精确）；
        边界穿过像素时返回三角形与像素方格的精确相交面积。
        ``px`` / ``py`` 可为 numpy 数组，此时返回同形状的 ``ndarray``。
        """
        tri = _check_tri(tri)
        tri, _ = _orient(tri)
        edges = _edges(tri)

        if _is_vec(px) or _is_vec(py):
            return self._coverage_vec(edges, px, py)

        x = _check_finite(px, "px")
        y = _check_finite(py, "py")
        cls = _classify_square(edges, x, y)
        if cls is not None:
            return cls
        return _exact_area(edges, x, y)

    def _coverage_vec(self, edges, px, py):
        if _np is None:  # pragma: no cover
            raise _type_err("批量输入需要 numpy")
        xs = _as_float_array(px, "px")
        ys = _as_float_array(py, "py")
        try:
            xs, ys = _np.broadcast_arrays(xs, ys)
        except ValueError:
            raise _val_err("px 与 py 形状不可广播") from None

        # 向量化分类：每条边计算方格上的有符号距离最小值/最大值
        cut = _np.zeros(xs.shape, dtype=bool)
        out = _np.zeros(xs.shape, dtype=bool)
        for e in edges:
            A, B, C = _edge_coeffs(e)
            dc = A * xs
            dc += B * ys
            dc += C
            r = 0.5 * (abs(A) + abs(B))
            out |= dc + r <= 0.0
            cut |= dc - r < 0.0

        res = _np.where(cut, 0.0, 1.0)
        edge_mask = cut & ~out
        if edge_mask.any():
            xs_flat = xs.ravel()
            ys_flat = ys.ravel()
            res_flat = res.ravel()
            for i in _np.nonzero(edge_mask.ravel())[0]:
                res_flat[i] = _exact_area(edges, float(xs_flat[i]), float(ys_flat[i]))
        if res.ndim == 0:
            return float(res)
        return res

    # -- 颜色解析 -------------------------------------------------------- #
    def resolve(self, samples):
        """按覆盖率含义解析颜色：各采样颜色按权重加权平均。

        标量输入：长度为采样数的颜色序列，返回 RGBA ``tuple``。
        批量输入：形状 ``(P, samples, C)`` 的 numpy 数组（C 为 1..4），返回
        形状 ``(P, 4)`` 的 ``ndarray``。
        """
        n = self._samples
        if _np is not None and isinstance(samples, _np.ndarray) and samples.ndim == 3:
            return self._resolve_vec(samples, n)

        seq = _check_seq_arg(samples, "采样序列")
        if len(seq) != n:
            raise _val_err(f"采样数量必须为 {n}（构造时采样数），收到 {len(seq)}")
        colors = [_check_deg_colors(c, i) for i, c in enumerate(seq)]
        ws = self._weights
        return tuple(
            self._check_overflow(
                math.fsum([ws[i] * colors[i][k] for i in range(n)]), k
            )
            for k in range(4)
        )

    def _resolve_vec(self, samples, n):
        if samples.shape[1] != n:
            raise _val_err(
                f"采样数量必须为 {n}（构造时采样数），实际为 {samples.shape[1]}"
            )
        c = samples.shape[2]
        if c < 1 or c > 4:
            raise _val_err(f"颜色通道数必须为 1..4，收到 {c}")
        arr = _as_float_array(samples, "采样颜色")
        w = _np.asarray(self._weights, dtype=_np.float64)
        acc = _np.einsum("pnc,n->pc", arr.reshape(arr.shape[0], n, c), w)
        out = _np.empty((arr.shape[0], 4), dtype=_np.float64)
        if c == 1:
            out[:, 0] = out[:, 1] = out[:, 2] = acc[:, 0]
            out[:, 3] = 1.0
        elif c == 2:
            out[:, 0] = out[:, 1] = out[:, 2] = acc[:, 0]
            out[:, 3] = acc[:, 1]
        elif c == 3:
            out[:, :3] = acc
            out[:, 3] = 1.0
        else:
            out[:, :] = acc
        if not _np.isfinite(out).all():
            raise _val_err("解析结果溢出")
        return out

    @staticmethod
    def _check_overflow(v, k):
        if not math.isfinite(v):
            raise _val_err(f"解析结果溢出（通道 {k}）")
        return v

    # -- 深度解析 -------------------------------------------------------- #
    def resolve_depth(self, depths):
        """深度按“最近采样保留”：返回有效采样中的最小值（越近越小）。

        批量输入：形状 ``(P, samples)`` 的数组，返回形状 ``(P,)`` 的 ndarray。
        """
        n = self._samples
        if _np is not None and isinstance(depths, _np.ndarray) and depths.ndim >= 2:
            return self._resolve_depth_vec(depths, n)

        seq = _check_seq_arg(depths, "深度序列")
        if len(seq) != n:
            raise _val_err(f"深度数量必须为 {n}（构造时采样数），收到 {len(seq)}")
        best = math.inf
        for i, d in enumerate(seq):
            v = _check_finite(d, f"深度[{i}]")
            if v < best:
                best = v
        return float(best)

    def _resolve_depth_vec(self, depths, n):
        if depths.shape[-1] != n:
            raise _val_err(
                f"深度数量必须为 {n}（构造时采样数），实际为 {depths.shape[-1]}"
            )
        arr = _as_float_array(depths, "深度")
        return arr.min(axis=-1)

    # -- 表示 ------------------------------------------------------------ #
    def __repr__(self):
        return f"Msaa(samples={self._samples})"
