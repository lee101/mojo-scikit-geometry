"""Reproducible timings for the bulk predicates and polygon classifier."""

from __future__ import annotations

import platform
import time

import numpy as np

import skgeom as sg


def best(call, repeats=7):
    call()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        values.append(time.perf_counter() - start)
    return min(values)


def numpy_orientation(data):
    a, b, c = data[:, 0], data[:, 1], data[:, 2]
    return np.sign((a[:, 0] - c[:, 0]) * (b[:, 1] - c[:, 1]) - (a[:, 1] - c[:, 1]) * (b[:, 0] - c[:, 0]))


def numpy_incircle(data):
    a, b, c, d = (data[:, i] - data[:, 3] for i in range(4))
    ab = a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]
    bc = b[:, 0] * c[:, 1] - c[:, 0] * b[:, 1]
    ca = c[:, 0] * a[:, 1] - a[:, 0] * c[:, 1]
    return np.sign((a * a).sum(1) * bc + (b * b).sum(1) * ca + (c * c).sum(1) * ab)


def ms(value):
    return f"{value * 1e3:.2f} ms"


def main():
    rng = np.random.default_rng(42)
    n = 500_000
    triples = rng.normal(size=(n, 3, 2))
    quads = rng.normal(size=(n, 4, 2))
    polygon = sg.Polygon(np.c_[np.cos(np.linspace(0, 2 * np.pi, 128, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 128, endpoint=False))])
    points = rng.uniform(-1.5, 1.5, size=(n, 2))
    rows = [
        ("orient2d, 500k", best(lambda: sg.orientations(triples)), best(lambda: numpy_orientation(triples))),
        ("incircle, 500k", best(lambda: sg.incircles(quads)), best(lambda: numpy_incircle(quads))),
        ("point-in-polygon, 500k x 128", best(lambda: sg.points_in_polygon(points, polygon)), None),
    ]
    print(f"machine: {platform.processor() or platform.machine()}, Python {platform.python_version()}, NumPy {np.__version__}")
    print("| kernel | Mojo | reference | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, reference in rows:
        if reference is None:
            print(f"| {name} | {ms(ours)} | n/a | n/a |")
        else:
            print(f"| {name} | {ms(ours)} | {ms(reference)} | {reference / ours:.2f}x |")


if __name__ == "__main__":
    main()
