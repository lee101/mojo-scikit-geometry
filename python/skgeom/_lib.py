"""Build and load the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_SKG_LIB") or os.path.join(ROOT, "dist", "libmojo-scikit-geometry.so")
I, F = ctypes.c_int64, ctypes.c_double
SIGNATURES = {
    "msg_orient2d_batch": ([I, I, I], None),
    "msg_incircle_batch": ([I, I, I], None),
    "msg_polygon_area": ([I, I], F),
    "msg_points_in_polygon": ([I, I, I, I, I], None),
    "msg_segment_pairs": ([I, I, I], None),
}
_library: ctypes.CDLL | None = None


def build() -> str:
    if os.path.exists(LIB) and not os.environ.get("MOJO_SKG_LIB"):
        source = os.path.join(ROOT, "src", "capi.mojo")
        if os.path.getmtime(LIB) >= os.path.getmtime(source):
            return LIB
    if os.environ.get("MOJO_SKG_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise RuntimeError(f"MOJO_SKG_LIB does not exist: {LIB}")
    mojo = shutil.which("mojo")
    if not mojo:
        raise RuntimeError("mojo is unavailable; run through `pixi run`")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run([mojo, "build", "--emit", "shared-lib", os.path.join(ROOT, "src", "capi.mojo"), "-o", LIB], capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (args, result) in SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes, fn.restype = args, result
    return _library


def f64(value) -> np.ndarray:
    """Return a finite, C-contiguous ``float64`` array for the native ABI.

    Narrower floating-point arrays are exactly widened.  Integers are accepted
    only while every value is exactly representable as float64; wider floating
    types and lossy integers are rejected before crossing the native boundary.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f" and value.dtype.itemsize > np.dtype(np.float64).itemsize:
            raise TypeError("NumPy coordinate arrays wider than float64 are unsupported")
        if value.dtype.kind in "iu" and value.size and np.any(np.abs(value.astype(object)) > 2**53):
            raise ValueError("integer coordinates must be exactly representable as float64")
        if value.dtype.kind not in "fiub":
            raise TypeError("NumPy coordinate arrays must be numeric")
    array = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("coordinates must be finite float64 values")
    return array


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise TypeError("native buffers must be C-contiguous NumPy arrays")
    # Empty arrays are permitted only with a zero element count; NumPy still
    # provides an address, and the Mojo loops return before dereferencing it.
    pointer = int(value.ctypes.data)
    if pointer == 0:
        raise ValueError("native buffer has a null data pointer")
    return pointer
