# mojo-scikit-geometry

`mojo-scikit-geometry` is a standalone Mojo port of the compute-heavy 2D core of
[scikit-geometry](https://github.com/scikit-geometry/scikit-geometry). It provides
the familiar `skgeom` import path while keeping bulk predicate work in a compact,
compiled shared library.

The covered subset is deliberately useful rather than broad: exact-by-contract
`orientation` and `side_of_oriented_circle` predicates; bulk `orientations` and
`incircles`; `Point2`, `Vector2`, and `Segment2`; segment intersection; convex
hulls; simple polygons (area, orientation, point location, simplicity, and ear
clipping triangulation); bulk point-in-polygon; and finite segment `Arrangement`s
whose crossing segments are split into atomic edges.

It is not a CGAL replacement. Circular arcs, 3D geometry, constrained Delaunay,
polygon set booleans, Voronoi diagrams, kinetic data structures, and the rest of
the upstream bindings are not covered yet. `intersection` currently accepts two
`Segment2` values only.

## Install and use

```bash
pixi install
pixi run build
pixi run python - <<'PY'
import skgeom as sg

square = sg.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
print(square.area())                         # 4.0
print(square.bounded_side((1, 1)))           # 1 (ON_BOUNDED_SIDE)
print(sg.orientation((0, 0), (1, 0), (0, 1)))  # 1 (POSITIVE)
print(sg.intersection(
    sg.Segment2((0, 0), (2, 2)), sg.Segment2((0, 2), (2, 0))
))                                           # Point2(x=1.0, y=1.0)
PY
```

Run the complete verification suite with `pixi run test`; use `pixi run bench`
for a fresh, machine-locked benchmark run.

## Correctness and compatibility

For ordinary floating-point inputs Mojo evaluates predicates using conservative
error bounds. If a sign is not certified, the Python binding recomputes it with
exact `Fraction.from_float` dyadic arithmetic. Thus predicate decisions are exact
for finite IEEE-754 float inputs, including collinear, cocircular, and nearly
degenerate cases. Constructed intersection coordinates are returned as `float`,
matching the practical Python-facing convention rather than exposing CGAL exact
number types.

The included suite compares predicate results with an independent pure-Python
dyadic reference implementation and published predicate vectors. Public names and
return conventions follow the covered `skgeom` subset; this project does not claim
full API or ABI compatibility with the upstream CGAL bindings.

## Benchmarks

Measured with `pixi run bench` on x86_64, Python 3.13.14, NumPy 2.5.1.
Timings are the best of seven runs; reference is vectorized NumPy for the two
algebraic predicates.

| kernel | Mojo | reference | ratio |
| --- | ---: | ---: | ---: |
| orient2d, 500k | 7.24 ms | 11.04 ms | 1.53x |
| incircle, 500k | 13.34 ms | 112.47 ms | 8.43x |
| point-in-polygon, 500k x 128 | 435.76 ms | n/a | n/a |

The point-in-polygon row has no NumPy baseline because NumPy has no equivalent
single-call polygon classifier. It remains included as a reproducible throughput
measurement rather than a fabricated comparison.

## How it works

All Mojo exports are in one compilation unit, `src/capi.mojo`, built as
`dist/libmojo-scikit-geometry.so`. NumPy supplies C-contiguous `float64` coordinate
arrays and `int64` output arrays; ctypes passes their addresses as `Int`, which is
the non-parametric C ABI supported by this Mojo nightly. The kernel makes no
allocations and writes caller-owned output buffers directly. Scalar Python objects
are a thin compatibility layer over the same batch functions, with exact dyadic
fallback only when a floating-point filter is inconclusive.

MIT licensed.
