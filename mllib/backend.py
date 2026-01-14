#numpy/cupy

import numpy as np

try:
    import cupy as cp
    cupy_available = True
except Exception:
    cp = None
    cupy_available = False

def get_xp(backend: str):
    if backend == "gpu":
        if not cupy_available:
            raise RuntimeError("CuPy not available")
        return cp
    return np

def to_backend(x, xp):
    return xp.asarray(x, dtype=xp.float32)

def asnumpy(x):
    if cupy_available and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x
