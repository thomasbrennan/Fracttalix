# fracttalix/_compat.py
# Shared optional-dependency imports for the Fracttalix Sentinel package.
# All submodules import third-party libraries from here so there is exactly
# one try/except per dependency.

import math
import os
import warnings

# ---------------------------------------------------------------------------
# numpy
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NP = True
except ImportError:
    np = None  # type: ignore
    _NP = False
    if os.environ.get("FRACTTALIX_QUIET", "0") != "1":
        warnings.warn(
            "numpy not available; FFT falls back to pure-Python stub "
            "(degraded accuracy). Install numpy for full functionality.",
            ImportWarning,
            stacklevel=2,
        )

# ---------------------------------------------------------------------------
# numba
# ---------------------------------------------------------------------------
try:
    from numba import njit as _numba_njit
    _NUMBA = True
except ImportError:
    def _numba_njit(*a, **kw):  # type: ignore
        def _d(f): return f
        return _d
    _NUMBA = False

# ---------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL = True
except ImportError:
    plt = None  # type: ignore
    mpatches = None  # type: ignore
    _MPL = False

# ---------------------------------------------------------------------------
# tqdm
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    def _tqdm(it, *a, **kw): return it  # type: ignore
    _TQDM = False

# ---------------------------------------------------------------------------
# scipy
# ---------------------------------------------------------------------------
try:
    import scipy.signal as _scipy_signal
    _SCIPY = True
except ImportError:
    _scipy_signal = None  # type: ignore
    _SCIPY = False

# ---------------------------------------------------------------------------
# multiprocessing
# ---------------------------------------------------------------------------
import multiprocessing as _mp

# ---------------------------------------------------------------------------
# Module-level helpers (must be top-level for pickle / multiprocessing safety)
# ---------------------------------------------------------------------------

def _mean(seq):
    """Plain-Python mean — pickle-safe (T0-03)."""
    s = list(seq)
    return sum(s) / len(s) if s else 0.0


def _np_rng(seed=None):
    if _NP:
        return np.random.default_rng(seed)
    import random
    class _R:
        def __init__(self, s): random.seed(s)
        def uniform(self, lo, hi, n): return [random.uniform(lo, hi) for _ in range(n)]
    return _R(seed)


def _to_np(data):
    if _NP:
        return np.asarray(data, dtype=float)
    return list(data)


def _np_fft(arr):
    if _NP:
        return np.fft.rfft(arr)
    n = len(arr)
    return [complex(arr[k]) for k in range(n)]  # stub


def _np_ifft(arr):
    if _NP:
        return np.fft.irfft(arr)
    return arr  # stub


def _phase_randomize_worker(args):
    """Pool worker for phase-randomization surrogates (T0-03)."""
    data, seed = args
    rng = _np_rng(seed)
    arr = _to_np(data)
    fft = _np_fft(arr)
    phases = rng.uniform(0, 2 * math.pi, len(fft) // 2 + 1)
    fft_r = fft.copy()
    n = len(fft_r)
    for i in range(1, n // 2 + 1):
        fft_r[i] *= math.cos(phases[i - 1]) + 1j * math.sin(phases[i - 1])
        if n - i != i:
            fft_r[n - i] = fft_r[i].conjugate()
    return _np_ifft(fft_r).real.tolist()
