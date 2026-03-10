# fracttalix/_utils.py
# Pure-Python fallback utilities and helper functions.
# Separated from _compat.py to keep dependency management distinct from
# algorithmic helpers.

import math

from fracttalix._compat import _NP, np


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
    result = _np_ifft(fft_r)
    if isinstance(result, list):
        return [x.real if isinstance(x, complex) else x for x in result]
    return result.real.tolist()
