# fracttalix/_compat.py
# Shared optional-dependency imports for the Fracttalix Sentinel package.
# All submodules import third-party libraries from here so there is exactly
# one try/except per dependency.

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
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
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

# ---------------------------------------------------------------------------
# Pure-Python helpers — re-exported from _utils for backward compatibility.
# Implementations live in _utils.py to separate dependency management from
# algorithmic helpers.
# ---------------------------------------------------------------------------
from fracttalix._utils import (  # noqa: E402, F401
    _mean,
    _np_fft,
    _np_ifft,
    _np_rng,
    _phase_randomize_worker,
    _to_np,
)
