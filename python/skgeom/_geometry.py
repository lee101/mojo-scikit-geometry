"""A small, exact-by-contract 2D subset of the :mod:`skgeom` API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
import math
from typing import Iterable, Sequence

import numpy as np

from ._lib import addr, f64, lib


class Sign(IntEnum):
    NEGATIVE = -1
    ZERO = 0
    POSITIVE = 1


Orientation = Sign


class BoundedSide(IntEnum):
    ON_UNBOUNDED_SIDE = -1
    ON_BOUNDARY = 0
    ON_BOUNDED_SIDE = 1


ON_UNBOUNDED_SIDE = BoundedSide.ON_UNBOUNDED_SIDE
ON_BOUNDARY = BoundedSide.ON_BOUNDARY
ON_BOUNDED_SIDE = BoundedSide.ON_BOUNDED_SIDE
NEGATIVE, ZERO, POSITIVE = Sign.NEGATIVE, Sign.ZERO, Sign.POSITIVE


@dataclass(frozen=True, slots=True)
class Vector2:
    x: float
    y: float

    def __post_init__(self):
        x, y = float(self.x), float(self.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("coordinates must be finite float64 values")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def squared_length(self) -> float:
        return self.x * self.x + self.y * self.y

    def __mul__(self, scalar: float) -> "Vector2":
        return Vector2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__


@dataclass(frozen=True, slots=True)
class Point2:
    x: float
    y: float

    def __post_init__(self):
        x, y = float(self.x), float(self.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("coordinates must be finite float64 values")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __add__(self, vector: Vector2) -> "Point2":
        return Point2(self.x + vector.x, self.y + vector.y)

    def __sub__(self, other: "Point2") -> Vector2:
        return Vector2(self.x - other.x, self.y - other.y)


def _point(value) -> Point2:
    if isinstance(value, Point2):
        return value
    if isinstance(value, Vector2):
        return Point2(value.x, value.y)
    if len(value) != 2:
        raise ValueError("a point must contain exactly two coordinates")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("coordinates must be finite float64 values")
    return Point2(x, y)


def _fraction(x: float) -> Fraction:
    return Fraction.from_float(float(x))


def _sign(value: Fraction | float) -> Sign:
    return Sign.POSITIVE if value > 0 else Sign.NEGATIVE if value < 0 else Sign.ZERO


def _orientation_exact(a: Point2, b: Point2, c: Point2) -> Sign:
    ax, ay, bx, by, cx, cy = map(_fraction, (a.x, a.y, b.x, b.y, c.x, c.y))
    return _sign((ax - cx) * (by - cy) - (ay - cy) * (bx - cx))


def orientation(a, b, c) -> Sign:
    """Exact sign of the oriented area of triangle ``(a, b, c)``.

    A Mojo floating-point filter handles ordinary inputs; a dyadic-rational
    fallback makes the returned sign exact for every finite float input.
    """
    a, b, c = _point(a), _point(b), _point(c)
    return Sign(int(orientations([[[a.x, a.y], [b.x, b.y], [c.x, c.y]]])[0]))


def orientations(triples) -> np.ndarray:
    """Return exact orientation signs for an ``(n, 3, 2)`` point array."""
    raw = f64(triples)
    if raw.ndim != 3 or raw.shape[1:] != (3, 2):
        raise ValueError("expected points with shape (n, 3, 2)")
    packed = raw.reshape(-1, 6)
    result = np.empty(len(packed), dtype=np.int64)
    lib().msg_orient2d_batch(addr(packed), len(packed), addr(result))
    for i in np.flatnonzero(result == 0):
        a, b, c = (Point2(*raw[i, j]) for j in range(3))
        result[i] = int(_orientation_exact(a, b, c))
    return result


def _incircle_exact(a: Point2, b: Point2, c: Point2, d: Point2) -> Sign:
    ax, ay, bx, by, cx, cy, dx, dy = map(_fraction, (a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y))
    adx, ady, bdx, bdy, cdx, cdy = ax - dx, ay - dy, bx - dx, by - dy, cx - dx, cy - dy
    abdet = adx * bdy - bdx * ady
    bcdet = bdx * cdy - cdx * bdy
    cadet = cdx * ady - adx * cdy
    return _sign((adx * adx + ady * ady) * bcdet + (bdx * bdx + bdy * bdy) * cadet + (cdx * cdx + cdy * cdy) * abdet)


def side_of_oriented_circle(a, b, c, d) -> Sign:
    a, b, c, d = _point(a), _point(b), _point(c), _point(d)
    return Sign(int(incircles([[[a.x, a.y], [b.x, b.y], [c.x, c.y], [d.x, d.y]]])[0]))


def incircles(quads) -> np.ndarray:
    """Return exact incircle signs for an ``(n, 4, 2)`` point array."""
    raw = f64(quads)
    if raw.ndim != 3 or raw.shape[1:] != (4, 2):
        raise ValueError("expected points with shape (n, 4, 2)")
    packed = raw.reshape(-1, 8)
    result = np.empty(len(packed), dtype=np.int64)
    lib().msg_incircle_batch(addr(packed), len(packed), addr(result))
    for i in np.flatnonzero(result == 0):
        a, b, c, d = (Point2(*raw[i, j]) for j in range(4))
        result[i] = int(_incircle_exact(a, b, c, d))
    return result


@dataclass(frozen=True, slots=True)
class Segment2:
    source: Point2
    target: Point2

    def __init__(self, source, target):
        object.__setattr__(self, "source", _point(source))
        object.__setattr__(self, "target", _point(target))

    def squared_length(self) -> float:
        d = self.target - self.source
        return d.squared_length()

    def has_on(self, point) -> bool:
        return _on_segment(self.source, self.target, _point(point))


def _on_segment(a: Point2, b: Point2, p: Point2) -> bool:
    if _orientation_exact(a, b, p) != ZERO:
        return False
    ax, ay, bx, by, px, py = map(_fraction, (a.x, a.y, b.x, b.y, p.x, p.y))
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


def _line_intersection(a: Point2, b: Point2, c: Point2, d: Point2) -> Point2:
    ax, ay, bx, by, cx, cy, dx, dy = map(_fraction, (a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y))
    rx, ry, sx, sy = bx - ax, by - ay, dx - cx, dy - cy
    den = rx * sy - ry * sx
    t = ((cx - ax) * sy - (cy - ay) * sx) / den
    return Point2(float(ax + t * rx), float(ay + t * ry))


def intersection(first: Segment2, second: Segment2):
    """Return ``None``, a ``Point2``, or the overlapping ``Segment2``."""
    if not isinstance(first, Segment2) or not isinstance(second, Segment2):
        raise TypeError("only Segment2/Segment2 intersection is covered")
    a, b, c, d = first.source, first.target, second.source, second.target
    ab_c, ab_d = _orientation_exact(a, b, c), _orientation_exact(a, b, d)
    cd_a, cd_b = _orientation_exact(c, d, a), _orientation_exact(c, d, b)
    if ab_c * ab_d < 0 and cd_a * cd_b < 0:
        return _line_intersection(a, b, c, d)
    candidates = []
    for point in (a, b):
        if _on_segment(c, d, point):
            candidates.append(point)
    for point in (c, d):
        if _on_segment(a, b, point):
            candidates.append(point)
    unique = list(dict.fromkeys(candidates))
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    # The only way there are two distinct candidates is a collinear overlap.
    unique.sort(key=lambda p: ((p.x - a.x) ** 2 + (p.y - a.y) ** 2))
    return Segment2(unique[0], unique[-1])


def segment_pair_kinds(pairs) -> np.ndarray:
    """Classify segment pairs: 0 disjoint, 1 proper crossing, 2 touching/overlap."""
    raw = f64(pairs)
    if raw.ndim != 3 or raw.shape[1:] != (4, 2):
        raise ValueError("expected pairs with shape (n, 4, 2)")
    packed = raw.reshape(-1, 8)
    result = np.empty(len(packed), dtype=np.int64)
    lib().msg_segment_pairs(addr(packed), len(packed), addr(result))
    # A 2 also means an inconclusive filtered orientation. Resolve it exactly.
    for i in np.flatnonzero(result == 2):
        a, b, c, d = (Point2(*raw[i, j]) for j in range(4))
        hit = intersection(Segment2(a, b), Segment2(c, d))
        result[i] = 0 if hit is None else 2
    return result


def convex_hull(points: Iterable) -> list[Point2]:
    """Counter-clockwise monotone-chain hull, without a repeated first point."""
    unique = sorted(set(_point(p) for p in points), key=lambda p: (p.x, p.y))
    if len(unique) <= 1:
        return unique

    def half(seq):
        result: list[Point2] = []
        for point in seq:
            while len(result) >= 2 and orientation(result[-2], result[-1], point) <= ZERO:
                result.pop()
            result.append(point)
        return result

    return half(unique)[:-1] + half(reversed(unique))[:-1]


class Polygon:
    def __init__(self, vertices: Iterable = ()):
        values = [_point(p) for p in vertices]
        if len(values) > 1 and values[0] == values[-1]:
            values.pop()
        self.vertices = values

    def __iter__(self):
        return iter(self.vertices)

    def __len__(self):
        return len(self.vertices)

    def area(self) -> float:
        if not self.vertices:
            return 0.0
        raw = f64([(p.x, p.y) for p in self.vertices])
        return float(lib().msg_polygon_area(addr(raw), len(raw)))

    def orientation(self) -> Sign:
        total = Fraction(0)
        for a, b in zip(self.vertices, self.vertices[1:] + self.vertices[:1]):
            total += _fraction(a.x) * _fraction(b.y) - _fraction(a.y) * _fraction(b.x)
        return _sign(total)

    def bounded_side(self, point) -> BoundedSide:
        p = _point(point)
        n = len(self.vertices)
        if n < 3:
            return ON_UNBOUNDED_SIDE
        winding = 0
        for a, b in zip(self.vertices, self.vertices[1:] + self.vertices[:1]):
            side = _orientation_exact(a, b, p)
            if side == ZERO and _on_segment(a, b, p):
                return ON_BOUNDARY
            if a.y <= p.y < b.y and side > ZERO:
                winding += 1
            elif b.y <= p.y < a.y and side < ZERO:
                winding -= 1
        return ON_BOUNDED_SIDE if winding else ON_UNBOUNDED_SIDE

    def has_on_boundary(self, point) -> bool:
        return self.bounded_side(point) == ON_BOUNDARY

    def has_on_bounded_side(self, point) -> bool:
        return self.bounded_side(point) == ON_BOUNDED_SIDE

    def is_simple(self) -> bool:
        n = len(self.vertices)
        if n < 3:
            return False
        edges = [Segment2(a, b) for a, b in zip(self.vertices, self.vertices[1:] + self.vertices[:1])]
        for i, first in enumerate(edges):
            for j in range(i + 1, n):
                if j == i + 1 or (i == 0 and j == n - 1):
                    continue
                if intersection(first, edges[j]) is not None:
                    return False
        return True

    def triangulate(self) -> list["Polygon"]:
        """Ear-clipping triangulation of a simple polygon."""
        if not self.is_simple():
            raise ValueError("triangulation requires a simple polygon")
        points = self.vertices[:]
        if self.orientation() < ZERO:
            points.reverse()
        triangles: list[Polygon] = []
        while len(points) > 3:
            clipped = False
            for i, b in enumerate(points):
                a, c = points[(i - 1) % len(points)], points[(i + 1) % len(points)]
                if orientation(a, b, c) <= ZERO:
                    continue
                if any(_in_triangle(p, a, b, c) for k, p in enumerate(points) if k not in ((i - 1) % len(points), i, (i + 1) % len(points))):
                    continue
                triangles.append(Polygon((a, b, c)))
                points.pop(i)
                clipped = True
                break
            if not clipped:
                raise ValueError("polygon has degenerate ears")
        triangles.append(Polygon(points))
        return triangles


def _in_triangle(p: Point2, a: Point2, b: Point2, c: Point2) -> bool:
    return orientation(a, b, p) >= ZERO and orientation(b, c, p) >= ZERO and orientation(c, a, p) >= ZERO


def points_in_polygon(points, polygon: Polygon | Sequence) -> np.ndarray:
    """Classify many points, returning CGAL-style ``BoundedSide`` integer values."""
    poly = polygon if isinstance(polygon, Polygon) else Polygon(polygon)
    raw = f64(points)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("expected points with shape (n, 2)")
    vertices = f64([[p.x for p in poly.vertices], [p.y for p in poly.vertices]])
    result = np.empty(len(raw), dtype=np.int64)
    vertex_count = vertices.shape[1]
    if vertex_count < 3:
        result.fill(int(ON_UNBOUNDED_SIDE))
        return result
    lib().msg_points_in_polygon(addr(raw), len(raw), addr(vertices), vertex_count, addr(result))
    # The native kernel uses 2 for any edge whose floating-point filter was
    # inconclusive (including a possible boundary edge).  Reclassify those
    # points exactly; this keeps a fast path for ordinary points without
    # treating an uncertified orientation as a geometric zero.
    outside = result == 0
    for i in np.flatnonzero(result == 2):
        result[i] = int(poly.bounded_side(Point2(*raw[i])))
    result[outside] = int(ON_UNBOUNDED_SIDE)
    return result


class Arrangement:
    """Finite segment arrangement with exact predicate decisions and split edges."""
    def __init__(self, segments: Iterable[Segment2] = ()):
        self._segments = [s if isinstance(s, Segment2) else Segment2(*s) for s in segments]

    def insert(self, segment: Segment2) -> None:
        self._segments.append(segment if isinstance(segment, Segment2) else Segment2(*segment))

    insert_non_intersecting_curve = insert

    @property
    def edges(self) -> list[Segment2]:
        pieces: list[Segment2] = []
        for i, segment in enumerate(self._segments):
            cuts = [segment.source, segment.target]
            for other in self._segments:
                hit = intersection(segment, other)
                if isinstance(hit, Point2):
                    cuts.append(hit)
                elif isinstance(hit, Segment2):
                    cuts.extend((hit.source, hit.target))
            dx, dy = segment.target.x - segment.source.x, segment.target.y - segment.source.y
            cuts = list(dict.fromkeys(cuts))
            cuts.sort(key=lambda p: (p.x - segment.source.x) * dx + (p.y - segment.source.y) * dy)
            pieces.extend(Segment2(a, b) for a, b in zip(cuts, cuts[1:]) if a != b)
        unique: dict[tuple[Point2, Point2], Segment2] = {}
        for edge in pieces:
            key = tuple(sorted((edge.source, edge.target), key=lambda p: (p.x, p.y)))
            unique[key] = edge
        return list(unique.values())

    @property
    def vertices(self) -> list[Point2]:
        result = set()
        for edge in self.edges:
            result.add(edge.source)
            result.add(edge.target)
        return sorted(result, key=lambda p: (p.x, p.y))
