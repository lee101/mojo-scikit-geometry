from fractions import Fraction

import numpy as np
import pytest

import skgeom as sg


def ref_orientation(a, b, c):
    ax, ay, bx, by, cx, cy = (Fraction.from_float(float(x)) for x in (*a, *b, *c))
    value = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return (value > 0) - (value < 0)


def ref_incircle(a, b, c, d):
    values = [Fraction.from_float(float(x)) for x in (*a, *b, *c, *d)]
    ax, ay, bx, by, cx, cy, dx, dy = values
    adx, ady, bdx, bdy, cdx, cdy = ax - dx, ay - dy, bx - dx, by - dy, cx - dx, cy - dy
    value = (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
    value += (bdx * bdx + bdy * bdy) * (cdx * ady - adx * cdy)
    value += (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
    return (value > 0) - (value < 0)


def test_orientation_published_and_near_collinear_vectors():
    assert sg.orientation((0, 0), (1, 0), (0, 1)) == sg.POSITIVE
    assert sg.orientation((0, 0), (0, 1), (1, 0)) == sg.NEGATIVE
    assert sg.orientation((0, 0), (1, 1), (2, 2)) == sg.ZERO
    c = (2.0, np.nextafter(2.0, np.inf))
    assert sg.orientation((0, 0), (1, 1), c) == ref_orientation((0, 0), (1, 1), c)


def test_orientation_batch_matches_independent_exact_reference():
    rng = np.random.default_rng(9)
    triples = rng.normal(size=(2000, 3, 2))
    triples[:3] = [[[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, np.nextafter(2.0, np.inf)]], [[1e100, 1e100], [2e100, 2e100], [3e100, 3e100 + 1e84]]]
    expected = np.array([ref_orientation(*row) for row in triples])
    assert np.array_equal(sg.orientations(triples), expected)


@pytest.mark.parametrize("size", [3, 4, 5, 9])
def test_orientation_batch_sizes(size):
    rng = np.random.default_rng(size)
    triples = rng.normal(size=(size, 3, 2))
    expected = np.array([ref_orientation(*row) for row in triples])
    assert np.array_equal(sg.orientations(triples), expected)


def test_orientation_parallel_threshold():
    rng = np.random.default_rng(81)
    triples = rng.normal(size=(100_003, 3, 2))
    a, b, c = triples[:, 0], triples[:, 1], triples[:, 2]
    expected = np.sign((a[:, 0] - c[:, 0]) * (b[:, 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (b[:, 0] - c[:, 0]))
    assert np.array_equal(sg.orientations(triples), expected)


def test_incircle_batch_matches_independent_exact_reference():
    rng = np.random.default_rng(11)
    quads = rng.normal(size=(1000, 4, 2))
    quads[:3] = [[[1, 0], [0, 1], [-1, 0], [0, 0]], [[1, 0], [0, 1], [-1, 0], [0, -1]], [[1, 0], [0, 1], [-1, 0], [0, np.nextafter(-1.0, 0.0)]]]
    expected = np.array([ref_incircle(*row) for row in quads])
    assert np.array_equal(sg.incircles(quads), expected)


def test_scalar_predicates_and_coordinate_validation():
    assert sg.side_of_oriented_circle((1, 0), (0, 1), (-1, 0), (0, 0)) == sg.POSITIVE
    assert sg.Vector2(3, 4).squared_length() == 25
    with pytest.raises(ValueError, match="exactly representable"):
        sg.orientations(np.full((1, 3, 2), 2**53 + 1, dtype=np.int64))
    with pytest.raises(ValueError, match="finite"):
        sg.incircles(np.array([[[np.nan, 0]] * 4], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        sg.Point2(float("inf"), 0)


def test_segment_intersection_handles_crossing_touch_overlap_and_disjoint():
    assert sg.intersection(sg.Segment2((0, 0), (2, 2)), sg.Segment2((0, 2), (2, 0))) == sg.Point2(1, 1)
    assert sg.intersection(sg.Segment2((0, 0), (1, 0)), sg.Segment2((1, 0), (2, 0))) == sg.Point2(1, 0)
    assert sg.intersection(sg.Segment2((0, 0), (3, 0)), sg.Segment2((1, 0), (2, 0))) == sg.Segment2((1, 0), (2, 0))
    assert sg.intersection(sg.Segment2((0, 0), (1, 0)), sg.Segment2((0, 1), (1, 1))) is None
    pairs = np.array([[[0, 0], [2, 2], [0, 2], [2, 0]], [[0, 0], [1, 0], [1, 0], [2, 0]], [[0, 0], [1, 0], [0, 1], [1, 1]]])
    assert np.array_equal(sg.segment_pair_kinds(pairs), [1, 2, 0])


def test_polygon_area_side_and_bulk_location():
    polygon = sg.Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])
    assert polygon.area() == pytest.approx(6.0)
    assert polygon.orientation() == sg.POSITIVE
    assert polygon.bounded_side((1, 1)) == sg.ON_BOUNDED_SIDE
    assert polygon.bounded_side((0, 1)) == sg.ON_BOUNDARY
    assert polygon.bounded_side((4, 1)) == sg.ON_UNBOUNDED_SIDE
    points = np.array([[1, 1], [0, 1], [4, 1], [3, 2]])
    assert np.array_equal(sg.points_in_polygon(points, polygon), [1, 0, -1, 0])


def test_points_in_polygon_handles_small_and_large_batches():
    polygon = sg.Polygon([(0, 0), (3, 0), (3, 2), (0, 2)])
    small = np.array([[1, 1], [0, 1], [4, 1], [3, 2]])
    assert np.array_equal(sg.points_in_polygon(small, polygon), [1, 0, -1, 0])
    points = np.tile([[1, 1], [4, 1]], (125_001, 1))
    expected = np.tile([1, -1], 125_001)
    assert np.array_equal(sg.points_in_polygon(points, polygon), expected)


def test_points_in_polygon_simd_tail():
    angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    polygon = sg.Polygon(np.c_[np.cos(angles), np.sin(angles)])
    points = np.array([[0, 0], [2, 0], [0.25, -0.25]])
    assert np.array_equal(sg.points_in_polygon(points, polygon), [1, -1, 1])


def test_convex_hull_and_triangulation_preserve_area():
    hull = sg.convex_hull([(0, 0), (2, 0), (2, 2), (0, 2), (1, 1), (0, 0)])
    assert hull == [sg.Point2(0, 0), sg.Point2(2, 0), sg.Point2(2, 2), sg.Point2(0, 2)]
    polygon = sg.Polygon([(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)])
    triangles = polygon.triangulate()
    assert len(triangles) == len(polygon) - 2
    assert sum(t.area() for t in triangles) == pytest.approx(polygon.area())


def test_arrangement_splits_crossing_segments_into_atomic_edges():
    arrangement = sg.Arrangement([sg.Segment2((0, 0), (2, 2)), sg.Segment2((0, 2), (2, 0))])
    assert sg.Point2(1, 1) in arrangement.vertices
    assert len(arrangement.vertices) == 5
    assert len(arrangement.edges) == 4


def test_simple_polygon_detection():
    assert sg.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]).is_simple()
    assert not sg.Polygon([(0, 0), (2, 2), (0, 2), (2, 0)]).is_simple()
