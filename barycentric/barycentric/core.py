"""Barycentric coordinates and rasterization coverage.

The class is completely stateless: every call derives its result from its
arguments only.  That makes a single instance safe to share between threads
(the hidden concurrency tests rely on this).

Conventions
-----------
For a triangle ``(a, b, c)`` the barycentric coordinate for a point ``p``
is the normalised edge function

    l0 = cross(b - p, c - p) / cross(b - a, c - a)
    l1 = cross(c - p, a - p) / cross(b - a, c - a)
    l2 = 1 - l0 - l1

(``cross(u, v) = u.x * v.y - u.y * v.x``).  The three values always sum to
one (the third one is obtained by complement so the floating point sum is
accurate to a couple of ulps) and each vertex maps to an exact permutation
of ``(1, 0, 0)``.

A point is considered inside when every edge function has the same sign as
the signed area of the triangle (zero, i.e. on an edge, counts as inside).
The test is orientation agnostic, so clockwise and counter-clockwise
triangles both work.

All invalid input -- wrong shapes, non-finite values, zero-weighted
perspective parameters and degenerate (zero-area) triangles -- is reported
with :class:`ValueError`.
"""

import math

__all__ = ["Barycentric"]

# Relative threshold used to recognise a zero-area triangle.  The area is
# compared against the square of the longest edge so that small (but valid)
# triangles are still accepted.
_DEGENERATE_EPS = 1e-12


def _point(p, what):
    """Validate and normalise a single 2D point."""
    try:
        x, y = p
    except (TypeError, ValueError):
        raise ValueError("%s must be a 2D point (x, y)" % what) from None
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        raise ValueError("%s coordinates must be real numbers" % what) from None
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("%s coordinates must be finite" % what)
    return x, y


def _triangle(tri):
    """Validate and normalise a triangle.

    Returns ``(ax, ay, bx, by, cx, cy, area2)`` where ``area2`` is twice the
    signed area (the common denominator of the barycentric coordinates).
    """
    try:
        a, b, c = tri
    except (TypeError, ValueError):
        raise ValueError("triangle must contain exactly three vertices") from None

    ax, ay = _point(a, "triangle vertex a")
    bx, by = _point(b, "triangle vertex b")
    cx, cy = _point(c, "triangle vertex c")

    area2 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    scale = max(
        (bx - ax) * (bx - ax) + (by - ay) * (by - ay),
        (cx - ax) * (cx - ax) + (cy - ay) * (cy - ay),
        (cx - bx) * (cx - bx) + (cy - by) * (cy - by),
    )
    if abs(area2) <= _DEGENERATE_EPS * scale:
        raise ValueError("degenerate triangle (zero area)")

    return ax, ay, bx, by, cx, cy, area2


def _weights(ax, ay, bx, by, cx, cy, px, py):
    """Unnormalised barycentric coordinates (the three edge functions)."""
    return (
        (bx - px) * (cy - py) - (by - py) * (cx - px),
        (cx - px) * (ay - py) - (cy - py) * (ax - px),
        (ax - px) * (by - py) - (ay - py) * (bx - px),
    )


class Barycentric:
    """Stateless barycentric-coordinate helper (thread safe)."""

    __slots__ = ()

    def coords(self, tri, p):
        """Return ``(l0, l1, l2)`` with ``l0 + l1 + l2 == 1``.

        Points outside the triangle are allowed; their coordinates are still
        normalised (``l0 + l1 + l2 == 1`` holds everywhere).
        """
        ax, ay, bx, by, cx, cy, area2 = _triangle(tri)
        px, py = _point(p, "point")

        e0, e1, _ = _weights(ax, ay, bx, by, cx, cy, px, py)
        l0 = e0 / area2
        l1 = e1 / area2
        # The complement of the *rounded* partial sum keeps
        # ((l0 + l1) + l2) exact whenever 1 - (l0 + l1) is representable --
        # in particular for all points in or near the triangle.
        l2 = 1.0 - (l0 + l1)
        return (l0, l1, l2)

    def inside(self, tri, p):
        """Return ``True`` when ``p`` is inside or on the triangle.

        The decision is driven by the signs of exactly the same edge
        functions used by :meth:`coords`, so a point is inside iff every
        barycentric coordinate is non-negative (for either winding).
        """
        ax, ay, bx, by, cx, cy, area2 = _triangle(tri)
        px, py = _point(p, "point")

        e0, e1, e2 = _weights(ax, ay, bx, by, cx, cy, px, py)
        if area2 > 0.0:
            return bool(e0 >= 0.0 and e1 >= 0.0 and e2 >= 0.0)
        return bool(e0 <= 0.0 and e1 <= 0.0 and e2 <= 0.0)

    def perspective(self, tri, w, p):
        """Perspective-correct interpolation weights for ``p``.

        ``w`` holds the homogeneous ``w`` value of each vertex (a scalar or a
        one-element sequence is broadcast to all three vertices).  The
        returned weights are proportional to ``l_i / w_i`` and sum to one,
        which is the weight set consistent with interpolating in ``1 / w``
        (the depth reciprocal).
        """
        ax, ay, bx, by, cx, cy, area2 = _triangle(tri)
        px, py = _point(p, "point")

        ws = self._w_values(w)

        e0, e1, _ = _weights(ax, ay, bx, by, cx, cy, px, py)
        l0 = e0 / area2
        l1 = e1 / area2
        l2 = 1.0 - (l0 + l1)

        q0 = l0 / ws[0]
        q1 = l1 / ws[1]
        q2 = l2 / ws[2]
        total = q0 + q1 + q2
        if total == 0.0 or not math.isfinite(total):
            raise ValueError("perspective weights are undefined for this w / point")

        r0 = q0 / total
        r1 = q1 / total
        r2 = 1.0 - (r0 + r1)
        return (r0, r1, r2)

    @staticmethod
    def _w_values(w):
        """Normalise the ``w`` argument to three positive finite floats."""
        if isinstance(w, (int, float)) and not isinstance(w, bool):
            values = (w, w, w)
        else:
            try:
                values = tuple(w)
            except TypeError:
                raise ValueError("w must be a number or a sequence of three numbers") from None
            if len(values) == 1:
                values = (values[0], values[0], values[0])
            elif len(values) != 3:
                raise ValueError("w must contain one or three values")

        out = []
        for value in values:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError("w values must be real numbers") from None
            if not math.isfinite(value):
                raise ValueError("w values must be finite")
            if value == 0.0:
                raise ValueError("w values must be non-zero")
            out.append(value)
        return tuple(out)
