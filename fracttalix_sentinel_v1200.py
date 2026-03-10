# fracttalix_sentinel_v1200.py
# Fracttalix Sentinel v12.0 — Three-Channel Extension
#
# V12.0 extends the v8.0 detection architecture to implement the
# three-channel model of dissipative network information transmission
# derived in the Meta-Kaizen Paper 6 theoretical framework.
#
# Three-channel model:
#   Channel 1 (Structural): Network topology as active transmitter.
#     Monitors structural properties of the input data stream
#     as an independent information channel.
#
#   Channel 2 (Rhythmic): Broadband multiplexed oscillatory transmission.
#     Decomposes the composite rhythmicity signal into independent
#     frequency band carrier waves and monitors cross-frequency coupling
#     between bands as a higher-order information channel.
#
#   Channel 3 (Temporal): One-way irreversible carrier wave.
#     Logs the temporal sequence of channel degradation events
#     as diagnostic information about regime change type and severity.
#
# Alert classes:
#   BAND_ANOMALY: Per-carrier-wave anomaly invisible to composite detection
#   COUPLING_DEGRADATION: Cross-frequency coupling breakdown (earlier warning)
#   STRUCTURAL_RHYTHMIC_DECOUPLING: Channel 1-2 coherence loss
#   CASCADE_PRECURSOR: Tipping cascade precursor — CRITICAL severity
#
# V8.0 root-cause fixes (α-ε) preserved:
#   α: SentinelConfig frozen dataclass — immutable, picklable, inspectable
#   β: WindowBank — named independent deques; each consumer owns its window slot
#   γ: Pipeline decomposition — 26 DetectorStep subclasses (19 v8 + 7 v12)
#   δ: Soft regime boost (T0-02) — replaces hard alpha reset with multiplicative boost
#   ε: SSI replaces RSI naming (T0-05); rsi alias preserved
#
# Governing principle: The v8.0 19-step pipeline is the foundation.
# V12.0 extends it. No existing step is removed. No existing behavior changed.
# All extensions are additive. Full backward compatibility with v8.0 and v7.x.
#
# Theoretical foundation: Fractal Rhythm Model Papers 1-6
# DOI: 10.5281/zenodo.18859299
# GitHub: https://github.com/thomasbrennan/Fracttalix

__version__ = "12.0.0"
__author__ = "Thomas Brennan & Claude (Anthropic) & Grok (xAI)"
__license__ = "MIT"

__all__ = [
    "SentinelConfig", "WindowBank", "StepContext", "DetectorStep",
    "SentinelDetector", "SentinelResult", "MultiStreamSentinel",
    "SentinelBenchmark", "SentinelServer",
    "Detector_7_10", "register_step",
    # V12.0 data structures
    "FrequencyBands", "StructuralSnapshot", "CouplingMatrix",
    "ChannelCoherence", "DegradationSequence",
    "AlertSeverity", "AlertType", "Alert",
]

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import abc
import argparse
import asyncio
import csv
import dataclasses
import io
import json
import math
import os
import socket
import sys
import threading
import time
import warnings
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Optional third-party imports
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NP = True
except ImportError:
    np = None  # type: ignore
    _NP = False

try:
    from numba import njit as _numba_njit
    _NUMBA = True
except ImportError:
    def _numba_njit(*a, **kw):  # type: ignore
        def _d(f): return f
        return _d
    _NUMBA = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL = True
except ImportError:
    plt = None  # type: ignore
    _MPL = False

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    def _tqdm(it, *a, **kw): return it  # type: ignore
    _TQDM = False

import multiprocessing as _mp

# ---------------------------------------------------------------------------
# Module-level helpers (must be top-level for pickle / multiprocessing safety)
# T0-03: lambda replaced with named functions
# ---------------------------------------------------------------------------

def _mean(seq):
    """Plain-Python mean — pickle-safe (T0-03)."""
    s = list(seq)
    return sum(s) / len(s) if s else 0.0


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


# ===========================================================================
# V12.0 DATA STRUCTURES — Three-Channel Model
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class FrequencyBands:
    """Channel 2 decomposition into five independent carrier waves.

    Computed via FFT decomposition of windowed signal.
    Five bands: ultra-low (trend), low (slow oscillation),
    mid (primary rhythmicity), high (fast fluctuation),
    ultra-high (noise floor).
    """
    ultra_low_power: float
    low_power: float
    mid_power: float
    high_power: float
    ultra_high_power: float
    ultra_low_phase: float
    low_phase: float
    mid_phase: float
    high_phase: float
    ultra_high_phase: float
    timestamp: int


@dataclasses.dataclass(frozen=True)
class StructuralSnapshot:
    """Channel 1 structural properties at current timestep.

    Treats network topology as active transmitter — not passive substrate.
    Encodes history, identity, and capacity of the input data stream
    at the current moment independently of rhythmic properties.
    """
    mean: float
    variance: float
    skewness: float
    kurtosis: float
    autocorrelation_lag1: float
    autocorrelation_lag2: float
    stationarity_score: float
    timestamp: int


@dataclasses.dataclass(frozen=True)
class CouplingMatrix:
    """Cross-frequency coupling coefficients between adjacent band pairs.

    Phase-amplitude coupling: lower band phase to higher band amplitude.
    Coupling degradation is an earlier warning signal than single-band
    anomaly — the heterodyned information channel degrades first.
    Declining composite_coupling_score precedes regime change.
    """
    ultra_low_to_low: float
    low_to_mid: float
    mid_to_high: float
    high_to_ultra_high: float
    composite_coupling_score: float
    coupling_trend: float       # Positive = strengthening, negative = degrading
    timestamp: int


@dataclasses.dataclass(frozen=True)
class ChannelCoherence:
    """Structural-rhythmic coherence measurement.

    Measures degree to which Channel 1 and Channel 2 remain coupled.
    In a healthy network structural changes are reflected in rhythmic
    changes and vice versa. Decoupling indicates channels have lost
    coherence — itself a regime change signal independent of anomalies
    in either individual channel.
    coherence_score: 0.0 = fully decoupled, 1.0 = fully coherent.
    """
    coherence_score: float
    structural_change_rate: float
    rhythmic_change_rate: float
    decoupling_trend: float
    timestamp: int


@dataclasses.dataclass(frozen=True)
class DegradationSequence:
    """Temporal ordering of channel degradation events (Channel 3 information).

    The sequence and ordering of degradation is diagnostic about regime
    change type and severity.
    """
    first_channel_anomaly: str
    first_anomaly_timestamp: int
    second_channel_anomaly: Optional[str]
    second_anomaly_timestamp: Optional[int]
    coupling_degradation_timestamp: Optional[int]
    decoupling_timestamp: Optional[int]
    cascade_precursor_timestamp: Optional[int]
    sequence_pattern: str       # Human-readable degradation narrative


class AlertSeverity(Enum):
    """Severity levels for v12.0 structured alerts."""
    INFO = 1
    WARNING = 2
    ALERT = 3
    CRITICAL = 4    # Reserved for CASCADE_PRECURSOR only


class AlertType(Enum):
    """Alert type classification for v12.0 structured alerts."""
    # v8.0 alert type identifiers (as string labels)
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    DRIFT_ANOMALY = "drift_anomaly"
    VARIANCE_ANOMALY = "variance_anomaly"
    REGIME_CHANGE = "regime_change"
    EWS_WARNING = "ews_warning"
    # V12.0 alert types
    BAND_ANOMALY = "band_anomaly"
    COUPLING_DEGRADATION = "coupling_degradation"
    STRUCTURAL_RHYTHMIC_DECOUPLING = "sr_decoupling"
    CASCADE_PRECURSOR = "cascade_precursor"


@dataclasses.dataclass(frozen=True)
class Alert:
    """Structured alert object for v12.0 three-channel detection."""
    alert_type: AlertType
    severity: AlertSeverity
    score: float
    message: str


def _build_sequence_narrative(alert_types_set: set) -> str:
    """Build human-readable degradation narrative from active alert type set."""
    parts = []
    if AlertType.BAND_ANOMALY in alert_types_set:
        parts.append("carrier wave anomaly detected")
    if AlertType.COUPLING_DEGRADATION in alert_types_set:
        parts.append("cross-frequency coupling degrading")
    if AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING in alert_types_set:
        parts.append("structural-rhythmic channels decoupling")
    if AlertType.CASCADE_PRECURSOR in alert_types_set:
        parts.append("CASCADE PRECURSOR — scale-level reversion risk")
    if not parts:
        parts.append("no active degradation")
    return " \u2192 ".join(parts)


# ===========================================================================
# SECTION 1 — SentinelConfig (frozen dataclass, α)
# ===========================================================================

@dataclasses.dataclass(frozen=True, slots=True)
class SentinelConfig:
    """Immutable configuration for SentinelDetector.

    All parameters validated in ``__post_init__``.  Use the factory class
    methods (``fast``, ``production``, ``sensitive``, ``realtime``) for
    common presets, or override individual fields via ``dataclasses.replace``.
    """

    # ------------------------------------------------------------------
    # A: Core EWMA
    # ------------------------------------------------------------------
    alpha: float = 0.1
    """EWMA smoothing factor (0 < α ≤ 1).  Smaller = slower, more stable."""

    dev_alpha: float = 0.1
    """EWMA factor for deviation (volatility) estimation."""

    multiplier: float = 3.0
    """Alert threshold = EWMA ± multiplier × dev_ewma."""

    warmup_periods: int = 30
    """Observations collected before alerts are issued."""

    # ------------------------------------------------------------------
    # B: Regime detection
    # ------------------------------------------------------------------
    regime_threshold: float = 3.5
    """Z-score magnitude that triggers a regime change."""

    regime_alpha_boost: float = 2.0
    """Multiplicative boost applied to alpha during regime transitions (δ)."""

    regime_boost_decay: float = 0.9
    """Decay rate of the regime boost per observation."""

    # ------------------------------------------------------------------
    # C: Multivariate
    # ------------------------------------------------------------------
    multivariate: bool = False
    """Enable multivariate (Mahalanobis) mode."""

    n_channels: int = 1
    """Number of input channels when multivariate=True."""

    cov_alpha: float = 0.05
    """EWMA factor for covariance matrix update."""

    # ------------------------------------------------------------------
    # D: FRM metrics
    # ------------------------------------------------------------------
    rpi_window: int = 64
    """Window length for RPI FFT computation."""

    rfi_window: int = 64
    """Window length for RFI inter-beat analysis."""

    rpi_threshold: float = 0.6
    """Minimum RPI for 'rhythm healthy' classification."""

    rfi_threshold: float = 0.4
    """RFI alert threshold (higher = more irregular)."""

    # ------------------------------------------------------------------
    # E: Complexity & EWS
    # ------------------------------------------------------------------
    pe_order: int = 3
    """Permutation Entropy embedding dimension."""

    pe_window: int = 50
    """Sliding window for PE computation."""

    pe_threshold: float = 0.05
    """PE deviation alert threshold (fraction of log(pe_order!))."""

    ews_window: int = 40
    """EWS rolling window (T0-01: independent from scalar_window)."""

    ews_threshold: float = 0.6
    """EWS score threshold for 'approaching critical' classification."""

    # ------------------------------------------------------------------
    # F: Fluid dynamics
    # ------------------------------------------------------------------
    sti_window: int = 20
    """Shear-Turbulence Index window."""

    tps_window: int = 30
    """Temporal Phase Space reconstruction window."""

    osc_damp_window: int = 20
    """Oscillation damping window."""

    osc_threshold: float = 1.5
    """Oscillation damping alert multiplier."""

    cpd_window: int = 30
    """Change-Point Detection comparison window."""

    cpd_threshold: float = 2.0
    """CPD alert z-score threshold."""

    # ------------------------------------------------------------------
    # G: Drift / Volatility / Seasonal
    # ------------------------------------------------------------------
    ph_delta: float = 0.01
    """Page-Hinkley incremental delta (sensitivity)."""

    ph_lambda: float = 50.0
    """Page-Hinkley cumulative threshold."""

    var_cusum_k: float = 0.5
    """VarCUSUM allowance (half the expected shift in std-devs)."""

    var_cusum_h: float = 5.0
    """VarCUSUM decision threshold."""

    seasonal_period: int = 0
    """Seasonal period (0 = auto-detect via FFT)."""

    # ------------------------------------------------------------------
    # H: AQB / Scoring / IO
    # ------------------------------------------------------------------
    quantile_threshold_mode: bool = False
    """Use Adaptive Quantile Baseline instead of EWMA ± mult threshold."""

    aqb_window: int = 200
    """Rolling window for AQB quantile estimation."""

    aqb_q_low: float = 0.01
    """Lower quantile for AQB."""

    aqb_q_high: float = 0.99
    """Upper quantile for AQB."""

    history_maxlen: int = 5000
    """Maximum result records kept in memory."""

    csv_path: str = ""
    """If non-empty, stream results to this CSV file."""

    log_level: str = "WARNING"
    """Python logging level name."""

    # ------------------------------------------------------------------
    # V12.0 — Channel 2 frequency decomposition
    # ------------------------------------------------------------------
    enable_frequency_decomposition: bool = True
    """Enable FFT decomposition of signal into five frequency band carrier waves."""

    min_window_for_fft: int = 32
    """Minimum window length required before FFT decomposition runs."""

    # ------------------------------------------------------------------
    # V12.0 — Cross-frequency coupling detection
    # ------------------------------------------------------------------
    enable_coupling_detection: bool = True
    """Enable cross-frequency phase-amplitude coupling measurement."""

    coupling_degradation_threshold: float = 0.3
    """composite_coupling_score below this triggers COUPLING_DEGRADATION alert."""

    coupling_trend_window: int = 10
    """Number of FrequencyBands snapshots used for coupling measurement."""

    # ------------------------------------------------------------------
    # V12.0 — Structural-rhythmic coherence
    # ------------------------------------------------------------------
    enable_channel_coherence: bool = True
    """Enable structural-rhythmic channel coherence measurement."""

    coherence_threshold: float = 0.4
    """coherence_score below this triggers STRUCTURAL_RHYTHMIC_DECOUPLING alert."""

    coherence_window: int = 20
    """Rolling window length for coherence computation."""

    # ------------------------------------------------------------------
    # V12.0 — Cascade precursor
    # ------------------------------------------------------------------
    enable_cascade_detection: bool = True
    """Enable CASCADE_PRECURSOR detection (requires all three conditions)."""

    cascade_ews_threshold: int = 2
    """Minimum number of EWS indicators elevated for cascade precursor."""

    # ------------------------------------------------------------------
    # V12.0 — Degradation sequence logging
    # ------------------------------------------------------------------
    enable_sequence_logging: bool = True
    """Enable temporal logging of channel degradation sequences."""

    sequence_retention: int = 1000
    """Maximum number of completed degradation sequences to retain."""

    def __post_init__(self):
        errs = []
        if not (0.0 < self.alpha <= 1.0):
            errs.append(f"alpha={self.alpha} must be in (0, 1]")
        if not (0.0 < self.dev_alpha <= 1.0):
            errs.append(f"dev_alpha={self.dev_alpha} must be in (0, 1]")
        if self.multiplier <= 0:
            errs.append(f"multiplier={self.multiplier} must be > 0")
        if self.warmup_periods < 1:
            errs.append(f"warmup_periods={self.warmup_periods} must be >= 1")
        if self.n_channels < 1:
            errs.append(f"n_channels={self.n_channels} must be >= 1")
        if not (0 < self.aqb_q_low < self.aqb_q_high < 1):
            errs.append("aqb_q_low/high must satisfy 0 < low < high < 1")
        if errs:
            raise ValueError("SentinelConfig validation errors:\n  " + "\n  ".join(errs))

    # ------------------------------------------------------------------
    # Factory presets
    # ------------------------------------------------------------------

    @classmethod
    def fast(cls) -> "SentinelConfig":
        """High α, low warmup — react instantly, higher false-positive rate."""
        return cls(alpha=0.3, dev_alpha=0.3, warmup_periods=10)

    @classmethod
    def production(cls) -> "SentinelConfig":
        """Balanced defaults — suitable for most production deployments."""
        return cls()

    @classmethod
    def sensitive(cls) -> "SentinelConfig":
        """Low α, tight multiplier — catches subtle anomalies."""
        return cls(alpha=0.05, dev_alpha=0.05, multiplier=2.5, warmup_periods=50)

    @classmethod
    def realtime(cls) -> "SentinelConfig":
        """Fast response with quantile-adaptive thresholds."""
        return cls(alpha=0.2, warmup_periods=15, quantile_threshold_mode=True)


# ===========================================================================
# SECTION 2 — WindowBank (β)
# ===========================================================================

class WindowBank:
    """Named collection of independent deques.

    Each consumer registers its own slot with its own maxlen; the bank's
    ``append`` method fans out each new value to *all* registered deques.
    Slots are independent — one consumer's window size does not affect others.

    T0-01 fix: EWSStep registers ``"ews_w"`` independently of ``"scalar"``
    so it never receives stale scalar_window data.
    """

    def __init__(self):
        self._windows: Dict[str, deque] = {}

    def register(self, name: str, maxlen: int) -> None:
        """Create a named window.  No-op if already registered."""
        if name not in self._windows:
            self._windows[name] = deque(maxlen=maxlen)

    def append(self, value: float) -> None:
        """Fan out *value* to every registered window."""
        for d in self._windows.values():
            d.append(value)

    def get(self, name: str) -> deque:
        """Return the named deque (raises KeyError if not registered)."""
        return self._windows[name]

    def reset(self) -> None:
        """Clear all windows, preserving registrations."""
        for d in self._windows.values():
            d.clear()

    def state_dict(self) -> Dict[str, list]:
        return {k: list(v) for k, v in self._windows.items()}

    def load_state(self, sd: Dict[str, list]) -> None:
        for k, vals in sd.items():
            if k in self._windows:
                self._windows[k].clear()
                self._windows[k].extend(vals)


# ===========================================================================
# SECTION 3 — StepContext
# ===========================================================================

@dataclasses.dataclass
class StepContext:
    """Mutable scratchpad passed through the pipeline on each observation.

    Steps read previous results from ``scratch`` and write their own outputs
    into it.  ``value`` is the raw input; ``step`` is the observation counter.
    """

    value: Any
    """Raw input value (scalar float or list/array for multivariate)."""

    step: int
    """Monotonically increasing observation counter (0-based)."""

    config: SentinelConfig
    """Immutable detector configuration."""

    bank: WindowBank
    """Shared window bank."""

    scratch: Dict[str, Any] = dataclasses.field(default_factory=dict)
    """Shared mutable scratchpad — steps read/write intermediate results."""

    # ------------------------------------------------------------------
    # Convenience accessors (populated by CoreEWMAStep)
    # ------------------------------------------------------------------

    @property
    def current(self) -> float:
        """Scalar current value (last element if multivariate)."""
        v = self.value
        if isinstance(v, (list, tuple)):
            return float(v[-1])
        try:
            if hasattr(v, '__len__'):
                return float(v[-1])
        except Exception:
            pass
        return float(v)

    @property
    def ewma(self) -> float:
        return self.scratch.get("ewma", 0.0)

    @property
    def dev_ewma(self) -> float:
        return self.scratch.get("dev_ewma", 1.0)

    @property
    def baseline_mean(self) -> float:
        return self.scratch.get("baseline_mean", 0.0)

    @property
    def baseline_std(self) -> float:
        return self.scratch.get("baseline_std", 1.0)

    @property
    def is_warmup(self) -> bool:
        return self.step < self.config.warmup_periods


# ===========================================================================
# SECTION 4 — DetectorStep ABC + registry
# ===========================================================================

_STEP_REGISTRY: Dict[str, type] = {}


def register_step(cls):
    """Class decorator — register a DetectorStep subclass by name."""
    _STEP_REGISTRY[cls.__name__] = cls
    return cls


class DetectorStep(abc.ABC):
    """Abstract base for pipeline steps.

    Each subclass implements a single well-scoped responsibility.  The pipeline
    calls ``update(ctx)`` in order; steps modify ``ctx.scratch`` in-place.
    """

    @abc.abstractmethod
    def update(self, ctx: StepContext) -> None:
        """Process one observation.  Modifies ctx.scratch in-place."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state to factory defaults."""

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable snapshot of internal state."""

    @abc.abstractmethod
    def load_state(self, sd: Dict[str, Any]) -> None:
        """Restore internal state from a snapshot."""


# ===========================================================================
# SECTION 5 — Pipeline Steps
# ===========================================================================

# ---------------------------------------------------------------------------
# 5.1 CoreEWMAStep
# ---------------------------------------------------------------------------

@register_step
class CoreEWMAStep(DetectorStep):
    """Compute EWMA baseline + anomaly score.  Must be first in pipeline."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self._bank_registered = False
        self.reset()

    def reset(self):
        self._ewma = 0.0
        self._dev_ewma = 1.0
        self._warmup_buf: List[float] = []
        self._initialized = False
        self._n = 0
        # per-channel state (multivariate)
        self._ch_ewma: List[float] = []
        self._ch_dev: List[float] = []
        self._ch_init: List[bool] = []
        # AQB rolling buffer
        self._aqb_buf: deque = deque(maxlen=self.cfg.aqb_window)
        # boost state (regime soft-boost injected here by RegimeStep)
        self._alpha_boost: float = 1.0

    def _ensure_bank(self, bank: WindowBank):
        if not self._bank_registered:
            bank.register("scalar", max(self.cfg.pe_window, self.cfg.rpi_window,
                                        self.cfg.rfi_window, self.cfg.sti_window,
                                        self.cfg.tps_window, self.cfg.osc_damp_window,
                                        self.cfg.cpd_window, 64))
            self._bank_registered = True

    def _eff_alpha(self) -> float:
        a = self.cfg.alpha * self._alpha_boost
        return min(a, 1.0)

    def _eff_dev_alpha(self) -> float:
        a = self.cfg.dev_alpha * self._alpha_boost
        return min(a, 1.0)

    def _scalar_update(self, v: float, ctx: StepContext) -> Dict[str, Any]:
        cfg = self.cfg
        self._n += 1
        self._aqb_buf.append(abs(v - self._ewma) if self._initialized else 0.0)

        if not self._initialized:
            self._warmup_buf.append(v)
            if len(self._warmup_buf) >= cfg.warmup_periods:
                self._ewma = _mean(self._warmup_buf)
                sq = [(x - self._ewma) ** 2 for x in self._warmup_buf]
                self._dev_ewma = math.sqrt(_mean(sq)) or 1.0
                self._initialized = True
            return {"ewma": v, "dev_ewma": 1.0, "baseline_mean": v,
                    "baseline_std": 1.0, "z_score": 0.0,
                    "anomaly_score": 0.0, "anomaly": False,
                    "alert": False, "warmup": True}

        a = self._eff_alpha()
        da = self._eff_dev_alpha()
        prev_ewma = self._ewma
        self._ewma = a * v + (1 - a) * self._ewma
        err = abs(v - prev_ewma)
        self._dev_ewma = da * err + (1 - da) * self._dev_ewma
        self._dev_ewma = max(self._dev_ewma, 1e-10)

        z = (v - self._ewma) / self._dev_ewma
        sigma = self.cfg.multiplier

        if cfg.quantile_threshold_mode and len(self._aqb_buf) >= 10:
            sorted_buf = sorted(self._aqb_buf)
            n = len(sorted_buf)
            hi_idx = min(int(cfg.aqb_q_high * n), n - 1)
            lo_idx = max(int(cfg.aqb_q_low * n), 0)
            hi_thresh = self._ewma + sorted_buf[hi_idx]
            lo_thresh = self._ewma - sorted_buf[hi_idx]
            alert = v > hi_thresh or v < lo_thresh
        else:
            alert = abs(z) > sigma

        anomaly_score = min(1.0, abs(z) / (sigma + 1e-10))

        return {
            "ewma": self._ewma,
            "dev_ewma": self._dev_ewma,
            "baseline_mean": self._ewma,
            "baseline_std": self._dev_ewma,
            "z_score": z,
            "anomaly_score": anomaly_score,
            "anomaly": alert,
            "alert": alert,
            "warmup": False,
        }

    def _mv_update(self, vs: List[float], ctx: StepContext) -> Dict[str, Any]:
        cfg = self.cfg
        nc = cfg.n_channels
        if len(self._ch_ewma) < nc:
            self._ch_ewma = [0.0] * nc
            self._ch_dev = [1.0] * nc
            self._ch_init = [False] * nc

        scores = []
        for i, v in enumerate(vs[:nc]):
            if not self._ch_init[i]:
                self._ch_ewma[i] = v
                self._ch_init[i] = True
                scores.append(0.0)
            else:
                a = self._eff_alpha()
                da = self._eff_dev_alpha()
                prev = self._ch_ewma[i]
                self._ch_ewma[i] = a * v + (1 - a) * self._ch_ewma[i]
                err = abs(v - prev)
                self._ch_dev[i] = da * err + (1 - da) * self._ch_dev[i]
                self._ch_dev[i] = max(self._ch_dev[i], 1e-10)
                scores.append(abs(v - self._ch_ewma[i]) / self._ch_dev[i])

        agg = _mean(scores) if scores else 0.0
        z = agg
        sigma = self.cfg.multiplier
        alert = z > sigma
        anomaly_score = min(1.0, z / (sigma + 1e-10))

        # Use last channel as "scalar" representative
        sv = float(vs[-1]) if vs else 0.0
        self._ewma = self._ch_ewma[-1] if self._ch_ewma else sv
        self._dev_ewma = self._ch_dev[-1] if self._ch_dev else 1.0

        return {
            "ewma": self._ewma,
            "dev_ewma": self._dev_ewma,
            "baseline_mean": self._ewma,
            "baseline_std": self._dev_ewma,
            "z_score": z,
            "anomaly_score": anomaly_score,
            "anomaly": alert,
            "alert": alert,
            "warmup": self._n < self.cfg.warmup_periods,
            "ch_ewma": list(self._ch_ewma),
            "ch_dev": list(self._ch_dev),
        }

    def update(self, ctx: StepContext) -> None:
        self._ensure_bank(ctx.bank)
        v = ctx.value
        cfg = self.cfg

        if cfg.multivariate and isinstance(v, (list, tuple, type(None))):
            if v is None:
                v = [0.0] * cfg.n_channels
            vals = [float(x) for x in v]
            result = self._mv_update(vals, ctx)
            sv = float(vals[-1]) if vals else 0.0
        else:
            sv = float(v)
            result = self._scalar_update(sv, ctx)

        ctx.bank.append(sv)
        ctx.scratch.update(result)

        # Decay boost
        self._alpha_boost = max(1.0, self._alpha_boost * self.cfg.regime_boost_decay)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ewma": self._ewma, "dev_ewma": self._dev_ewma,
            "initialized": self._initialized, "n": self._n,
            "warmup_buf": list(self._warmup_buf),
            "ch_ewma": list(self._ch_ewma), "ch_dev": list(self._ch_dev),
            "ch_init": list(self._ch_init), "alpha_boost": self._alpha_boost,
            "aqb_buf": list(self._aqb_buf),
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._ewma = sd.get("ewma", 0.0)
        self._dev_ewma = sd.get("dev_ewma", 1.0)
        self._initialized = sd.get("initialized", False)
        self._n = sd.get("n", 0)
        self._warmup_buf = sd.get("warmup_buf", [])
        self._ch_ewma = sd.get("ch_ewma", [])
        self._ch_dev = sd.get("ch_dev", [])
        self._ch_init = sd.get("ch_init", [])
        self._alpha_boost = sd.get("alpha_boost", 1.0)
        self._aqb_buf = deque(sd.get("aqb_buf", []), maxlen=self.cfg.aqb_window)


# ---------------------------------------------------------------------------
# 5.2 CUSUMStep
# ---------------------------------------------------------------------------

@register_step
class CUSUMStep(DetectorStep):
    """Bidirectional CUSUM for persistent mean shift detection."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._s_hi = 0.0
        self._s_lo = 0.0
        self._k = 0.5  # allowance in sigmas

    def update(self, ctx: StepContext) -> None:
        if ctx.is_warmup:
            return
        z = ctx.scratch.get("z_score", 0.0)
        k = self._k
        self._s_hi = max(0.0, self._s_hi + z - k)
        self._s_lo = max(0.0, self._s_lo - z - k)
        h = 5.0
        cusum_alert = (self._s_hi > h) or (self._s_lo > h)
        ctx.scratch["cusum_hi"] = self._s_hi
        ctx.scratch["cusum_lo"] = self._s_lo
        ctx.scratch["cusum_alert"] = cusum_alert
        if cusum_alert:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"s_hi": self._s_hi, "s_lo": self._s_lo}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._s_hi = sd.get("s_hi", 0.0)
        self._s_lo = sd.get("s_lo", 0.0)


# ---------------------------------------------------------------------------
# 5.3 RegimeStep (δ — soft boost, T0-02)
# ---------------------------------------------------------------------------

@register_step
class RegimeStep(DetectorStep):
    """Detect regime changes; apply soft alpha boost instead of hard reset (δ)."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._in_regime = False
        self._regime_count = 0

    def update(self, ctx: StepContext) -> None:
        if ctx.is_warmup:
            ctx.scratch["regime_change"] = False
            return
        z = abs(ctx.scratch.get("z_score", 0.0))
        thresh = self.cfg.regime_threshold
        if z > thresh:
            self._in_regime = True
            self._regime_count += 1
            # Soft boost: tell CoreEWMAStep to temporarily raise alpha (T0-02)
            core = ctx.scratch.get("_core_step_ref")
            if core is not None:
                core._alpha_boost = self.cfg.regime_alpha_boost
            ctx.scratch["regime_change"] = True
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        else:
            if self._in_regime:
                self._in_regime = False
            ctx.scratch["regime_change"] = False
        ctx.scratch["in_regime"] = self._in_regime

    def state_dict(self) -> Dict[str, Any]:
        return {"in_regime": self._in_regime, "regime_count": self._regime_count}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._in_regime = sd.get("in_regime", False)
        self._regime_count = sd.get("regime_count", 0)


# ---------------------------------------------------------------------------
# 5.4 VarCUSUMStep
# ---------------------------------------------------------------------------

@register_step
class VarCUSUMStep(DetectorStep):
    """CUSUM on squared deviation — catches volatility explosions."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._s_hi = 0.0
        self._s_lo = 0.0
        self._var_ewma = 1.0
        self._warmed = False

    def update(self, ctx: StepContext) -> None:
        if ctx.is_warmup:
            return
        if not self._warmed:
            self._s_hi = 0.0
            self._s_lo = 0.0
            self._var_ewma = 0.0  # seed from real data, not default 1.0
            self._warmed = True
        dev = ctx.dev_ewma
        z = ctx.scratch.get("z_score", 0.0)
        v2 = z * z  # variance proxy
        self._var_ewma = 0.9 * self._var_ewma + 0.1 * v2
        k = self.cfg.var_cusum_k
        self._s_hi = max(0.0, self._s_hi + v2 - k)
        # Only accumulate _s_lo when variance baseline is established; prevents
        # false "variance drop" alerts on data that has always been constant.
        if self._var_ewma > 1e-4:
            self._s_lo = max(0.0, self._s_lo + k - v2)
        else:
            self._s_lo = 0.0
        alert = self._s_hi > self.cfg.var_cusum_h or self._s_lo > self.cfg.var_cusum_h
        ctx.scratch["var_cusum_hi"] = self._s_hi
        ctx.scratch["var_cusum_lo"] = self._s_lo
        ctx.scratch["var_cusum_alert"] = alert
        if alert:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"s_hi": self._s_hi, "s_lo": self._s_lo, "var_ewma": self._var_ewma, "warmed": self._warmed}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._s_hi = sd.get("s_hi", 0.0)
        self._s_lo = sd.get("s_lo", 0.0)
        self._var_ewma = sd.get("var_ewma", 1.0)
        self._warmed = sd.get("warmed", False)


# ---------------------------------------------------------------------------
# 5.5 PageHinkleyStep
# ---------------------------------------------------------------------------

@register_step
class PageHinkleyStep(DetectorStep):
    """Page-Hinkley test for slow gradual mean drift."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._cum_hi = 0.0
        self._cum_lo = 0.0
        self._m_hi = 0.0
        self._m_lo = 0.0
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        if ctx.is_warmup:
            ctx.scratch["ph_alert"] = False
            return
        v = ctx.current
        mu = ctx.ewma
        delta = self.cfg.ph_delta
        lam = self.cfg.ph_lambda

        self._cum_hi += (v - mu - delta)
        self._cum_lo += (mu - v - delta)
        self._m_hi = max(self._m_hi, self._cum_hi)
        self._m_lo = max(self._m_lo, self._cum_lo)
        self._n += 1

        alert = (self._m_hi - self._cum_hi > lam) or (self._m_lo - self._cum_lo > lam)
        ctx.scratch["ph_alert"] = alert
        ctx.scratch["ph_cum_hi"] = self._cum_hi
        ctx.scratch["ph_cum_lo"] = self._cum_lo
        if alert:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
            # reset to avoid continuous triggering
            self._cum_hi = 0.0; self._cum_lo = 0.0
            self._m_hi = 0.0; self._m_lo = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {"cum_hi": self._cum_hi, "cum_lo": self._cum_lo,
                "m_hi": self._m_hi, "m_lo": self._m_lo, "n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._cum_hi = sd.get("cum_hi", 0.0)
        self._cum_lo = sd.get("cum_lo", 0.0)
        self._m_hi = sd.get("m_hi", 0.0)
        self._m_lo = sd.get("m_lo", 0.0)
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# 5.6 STIStep
# ---------------------------------------------------------------------------

@register_step
class STIStep(DetectorStep):
    """Shear-Turbulence Index — fluid-dynamics anomaly proxy."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._prev = None
        self._diffs: deque = deque(maxlen=self.cfg.sti_window)

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        if self._prev is not None:
            d = abs(v - self._prev)
            self._diffs.append(d)
        self._prev = v

        if len(self._diffs) >= 4:
            mu = _mean(self._diffs)
            sq = [(x - mu) ** 2 for x in self._diffs]
            std = math.sqrt(_mean(sq)) or 1e-10
            sti = mu / std if std > 0 else 0.0
        else:
            sti = 0.0

        ctx.scratch["sti"] = sti
        if not ctx.is_warmup and sti > 2.0:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"prev": self._prev, "diffs": list(self._diffs)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._prev = sd.get("prev")
        self._diffs = deque(sd.get("diffs", []), maxlen=self.cfg.sti_window)


# ---------------------------------------------------------------------------
# 5.7 TPSStep — Temporal Phase Space
# ---------------------------------------------------------------------------

@register_step
class TPSStep(DetectorStep):
    """Temporal Phase Space reconstruction — detects attractor deformation."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._win: deque = deque(maxlen=self.cfg.tps_window)
        self._ref_radius: Optional[float] = None

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        self._win.append(v)
        if len(self._win) < self.cfg.tps_window:
            ctx.scratch["tps_score"] = 0.0
            return

        w = list(self._win)
        mu = _mean(w)
        # Use pairs (x[i], x[i+1]) as 2D phase space
        radii = [math.sqrt((w[i] - mu) ** 2 + (w[i+1] - mu) ** 2)
                 for i in range(len(w) - 1)]
        r_mean = _mean(radii) if radii else 0.0

        if self._ref_radius is None:
            self._ref_radius = r_mean or 1.0
        score = abs(r_mean - self._ref_radius) / (self._ref_radius + 1e-10)
        ctx.scratch["tps_score"] = score
        if not ctx.is_warmup and score > 0.5:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        # Slowly update reference
        self._ref_radius = 0.99 * self._ref_radius + 0.01 * r_mean

    def state_dict(self) -> Dict[str, Any]:
        return {"win": list(self._win), "ref_radius": self._ref_radius}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._win = deque(sd.get("win", []), maxlen=self.cfg.tps_window)
        self._ref_radius = sd.get("ref_radius")


# ---------------------------------------------------------------------------
# 5.8 OscDampStep
# ---------------------------------------------------------------------------

@register_step
class OscDampStep(DetectorStep):
    """Oscillation damping — detects sudden amplitude shifts."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._amp_buf: deque = deque(maxlen=self.cfg.osc_damp_window)
        self._amp_ewma = 0.0

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        w = ctx.bank.get("scalar")
        if len(w) >= 2:
            amp = max(w) - min(w)
        else:
            amp = 0.0
        self._amp_buf.append(amp)
        alpha = 0.1
        self._amp_ewma = alpha * amp + (1 - alpha) * self._amp_ewma

        if not ctx.is_warmup and self._amp_ewma > 1e-10:
            ratio = amp / self._amp_ewma
            osc_alert = ratio > self.cfg.osc_threshold or ratio < (1.0 / self.cfg.osc_threshold)
        else:
            osc_alert = False

        ctx.scratch["osc_amp"] = amp
        ctx.scratch["osc_alert"] = osc_alert
        if osc_alert:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"amp_buf": list(self._amp_buf), "amp_ewma": self._amp_ewma}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._amp_buf = deque(sd.get("amp_buf", []), maxlen=self.cfg.osc_damp_window)
        self._amp_ewma = sd.get("amp_ewma", 0.0)


# ---------------------------------------------------------------------------
# 5.9 CPDStep — Change Point Detection
# ---------------------------------------------------------------------------

@register_step
class CPDStep(DetectorStep):
    """Two-window CPD: compare means of recent vs historical windows."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._hist: deque = deque(maxlen=self.cfg.cpd_window * 2)

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        self._hist.append(v)
        w = self.cfg.cpd_window
        if len(self._hist) < w * 2 or ctx.is_warmup:
            ctx.scratch["cpd_score"] = 0.0
            return

        h = list(self._hist)
        recent = h[-w:]
        historical = h[:w]
        mu_r = _mean(recent)
        mu_h = _mean(historical)
        sq_h = [(x - mu_h) ** 2 for x in historical]
        std_h = math.sqrt(_mean(sq_h)) if sq_h else 1.0
        std_h = max(std_h, 1e-10)
        score = abs(mu_r - mu_h) / std_h
        ctx.scratch["cpd_score"] = score
        if score > self.cfg.cpd_threshold:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"hist": list(self._hist)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._hist = deque(sd.get("hist", []), maxlen=self.cfg.cpd_window * 2)


# ---------------------------------------------------------------------------
# 5.10 RPIStep — Rhythm Periodicity Index (FRM Axiom 6)
# ---------------------------------------------------------------------------

@register_step
class RPIStep(DetectorStep):
    """RPI via FFT dominant peak ratio."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        if len(w) < self.cfg.rpi_window or not _NP:
            ctx.scratch["rpi"] = 0.0
            return
        arr = np.array(list(w)[-self.cfg.rpi_window:])
        arr = arr - arr.mean()
        spec = np.abs(np.fft.rfft(arr)) ** 2
        total = spec.sum()
        if total < 1e-12:
            ctx.scratch["rpi"] = 0.0
            return
        rpi = float(spec.max() / total)
        ctx.scratch["rpi"] = rpi
        if not ctx.is_warmup and rpi < self.cfg.rpi_threshold:
            pass  # low RPI is informational, not an alert by itself

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# 5.11 RFIStep — Rhythm Fractal Index (FRM Axiom 8)
# ---------------------------------------------------------------------------

@register_step
class RFIStep(DetectorStep):
    """RFI: Hurst-like fractal dimension of rhythm irregularity."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._n = 0

    def _hurst(self, ts: List[float]) -> float:
        n = len(ts)
        if n < 8:
            return 0.5
        mu = _mean(ts)
        dev = [x - mu for x in ts]
        # R/S analysis
        cumdev = []
        s = 0.0
        for d in dev:
            s += d
            cumdev.append(s)
        R = max(cumdev) - min(cumdev)
        sq = [(x - mu) ** 2 for x in ts]
        S = math.sqrt(_mean(sq)) or 1e-10
        rs = R / S
        if rs <= 0:
            return 0.5
        return math.log(rs) / math.log(n / 2) if n > 2 else 0.5

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        if len(w) < self.cfg.rfi_window:
            ctx.scratch["rfi"] = 0.0
            ctx.scratch["hurst"] = 0.5
            return
        ts = list(w)[-self.cfg.rfi_window:]
        h = self._hurst(ts)
        rfi = abs(h - 0.5) * 2  # 0=random, 1=perfectly fractal
        ctx.scratch["rfi"] = rfi
        ctx.scratch["hurst"] = h
        if not ctx.is_warmup and rfi > self.cfg.rfi_threshold:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# 5.12 SSIStep — Synchrony Stability Index (FRM Axiom 10) — was RSI (T0-05)
# ---------------------------------------------------------------------------

@register_step
class SSIStep(DetectorStep):
    """SSI: Kuramoto synchronization proxy via FFT phase coherence.

    Formerly mislabeled 'RSI' in v7.10 (T0-05 fix).  ``rsi`` attribute alias
    preserved for backward compatibility.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        n = self.cfg.rpi_window
        if len(w) < n or not _NP:
            ctx.scratch["ssi"] = 0.0
            ctx.scratch["rsi"] = 0.0  # alias (T0-05)
            return
        arr = np.array(list(w)[-n:])
        arr = arr - arr.mean()
        fft = np.fft.rfft(arr)
        phases = np.angle(fft[1:])  # skip DC
        # Kuramoto order param: |mean(exp(i*phases))|
        order = float(np.abs(np.mean(np.exp(1j * phases))))
        ctx.scratch["ssi"] = order
        ctx.scratch["rsi"] = order  # backward compat alias

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# 5.13 PEStep — Permutation Entropy (FRM Axiom 3)
# ---------------------------------------------------------------------------

@register_step
class PEStep(DetectorStep):
    """Streaming Permutation Entropy."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def _ordinal_pattern(self, window: List[float]) -> tuple:
        return tuple(sorted(range(len(window)), key=lambda i: window[i]))

    def _pe(self, data: List[float], m: int) -> float:
        from math import log, factorial
        n = len(data)
        counts: Dict[tuple, int] = {}
        total = 0
        for i in range(n - m + 1):
            pat = self._ordinal_pattern(data[i:i+m])
            counts[pat] = counts.get(pat, 0) + 1
            total += 1
        if total == 0:
            return 0.0
        h = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                h -= p * math.log(p)
        max_h = math.log(math.factorial(m)) if m > 1 else 1.0
        return h / max_h if max_h > 0 else 0.0

    def reset(self):
        self._pe_ewma = 0.5
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        m = self.cfg.pe_order
        if len(w) < max(self.cfg.pe_window, m + 1):
            ctx.scratch["pe"] = 0.5
            ctx.scratch["pe_baseline"] = self._pe_ewma
            return

        data = list(w)[-self.cfg.pe_window:]
        pe = self._pe(data, m)
        ctx.scratch["pe"] = pe

        # Rolling PE baseline
        alpha = 0.05
        self._pe_ewma = alpha * pe + (1 - alpha) * self._pe_ewma
        ctx.scratch["pe_baseline"] = self._pe_ewma

        # Alert on contextual deviation (not absolute low PE — fixes v7.11 false alarm)
        dev = abs(pe - self._pe_ewma)
        if not ctx.is_warmup and dev > self.cfg.pe_threshold:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"pe_ewma": self._pe_ewma, "n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._pe_ewma = sd.get("pe_ewma", 0.5)
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# 5.14 EWSStep — Early Warning Signals (FRM Axiom 9) — T0-01 fix
# ---------------------------------------------------------------------------

@register_step
class EWSStep(DetectorStep):
    """EWS: Critical slowing down via rising variance + AC(1).

    T0-01: Uses bank.get("ews_w") — independent window, not scalar_window.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self._bank_registered = False
        self.reset()

    def reset(self):
        self._var_ewma = 0.0
        self._ac1_ewma = 0.0
        self._ews_score = 0.0
        self._n = 0

    def _ensure_bank(self, bank: WindowBank):
        if not self._bank_registered:
            bank.register("ews_w", self.cfg.ews_window)  # T0-01
            self._bank_registered = True

    def _ac1(self, data: List[float]) -> float:
        n = len(data)
        if n < 3:
            return 0.0
        mu = _mean(data)
        num = sum((data[i] - mu) * (data[i+1] - mu) for i in range(n-1))
        den = sum((x - mu) ** 2 for x in data)
        return (num / den) if den > 1e-12 else 0.0

    def update(self, ctx: StepContext) -> None:
        self._ensure_bank(ctx.bank)
        self._n += 1

        w = list(ctx.bank.get("ews_w"))  # T0-01: independent window
        if len(w) < max(10, self.cfg.ews_window // 2):
            ctx.scratch["ews_score"] = 0.0
            ctx.scratch["ews_regime"] = "stable"
            return

        mu = _mean(w)
        var = _mean([(x - mu) ** 2 for x in w])
        ac1 = self._ac1(w)

        alpha = 0.1
        self._var_ewma = alpha * var + (1 - alpha) * self._var_ewma
        self._ac1_ewma = alpha * ac1 + (1 - alpha) * self._ac1_ewma

        # EWS score: combine rising var + rising AC(1)
        # Both should be trending upward near a critical transition
        var_trend = min(1.0, var / (self._var_ewma + 1e-10) if var > 0 else 0.0)
        ac1_clamped = max(0.0, min(1.0, self._ac1_ewma))
        self._ews_score = 0.5 * var_trend + 0.5 * ac1_clamped

        thresh = self.cfg.ews_threshold
        if self._ews_score > thresh * 1.5:
            regime = "critical"
        elif self._ews_score > thresh:
            regime = "approaching"
        else:
            regime = "stable"

        ctx.scratch["ews_score"] = self._ews_score
        ctx.scratch["ews_regime"] = regime
        ctx.scratch["ews_var"] = var
        ctx.scratch["ews_ac1"] = ac1
        if not ctx.is_warmup and regime == "critical":
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"var_ewma": self._var_ewma, "ac1_ewma": self._ac1_ewma,
                "ews_score": self._ews_score, "n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._var_ewma = sd.get("var_ewma", 0.0)
        self._ac1_ewma = sd.get("ac1_ewma", 0.0)
        self._ews_score = sd.get("ews_score", 0.0)
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# 5.15 AQBStep — Adaptive Quantile Baseline (FRM Axiom 1)
# ---------------------------------------------------------------------------

@register_step
class AQBStep(DetectorStep):
    """AQB: distribution-free quantile thresholds."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._buf: deque = deque(maxlen=self.cfg.aqb_window)

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        self._buf.append(v)
        if len(self._buf) < 10 or not self.cfg.quantile_threshold_mode:
            ctx.scratch["aqb_lo"] = None
            ctx.scratch["aqb_hi"] = None
            return

        s = sorted(self._buf)
        n = len(s)
        lo_idx = max(0, int(self.cfg.aqb_q_low * n))
        hi_idx = min(n - 1, int(self.cfg.aqb_q_high * n))
        lo = s[lo_idx]
        hi = s[hi_idx]
        ctx.scratch["aqb_lo"] = lo
        ctx.scratch["aqb_hi"] = hi
        if not ctx.is_warmup and (v > hi or v < lo):
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"buf": list(self._buf)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._buf = deque(sd.get("buf", []), maxlen=self.cfg.aqb_window)


# ---------------------------------------------------------------------------
# 5.16 SeasonalStep — Seasonal Periodic Baseline
# ---------------------------------------------------------------------------

@register_step
class SeasonalStep(DetectorStep):
    """Per-phase EWMA baseline; period auto-detected via FFT."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._period: Optional[int] = self.cfg.seasonal_period if self.cfg.seasonal_period > 0 else None
        self._phase_ewma: Dict[int, float] = {}
        self._phase_dev: Dict[int, float] = {}
        self._detect_buf: List[float] = []
        self._n = 0

    def _detect_period(self) -> Optional[int]:
        if not _NP or len(self._detect_buf) < 32:
            return None
        arr = np.array(self._detect_buf)
        arr = arr - arr.mean()
        spec = np.abs(np.fft.rfft(arr)) ** 2
        freqs = np.fft.rfftfreq(len(arr))
        if len(spec) < 2:
            return None
        spec[0] = 0  # skip DC
        peak_idx = int(np.argmax(spec))
        if peak_idx == 0 or freqs[peak_idx] < 1e-6:
            return None
        period = int(round(1.0 / freqs[peak_idx]))
        return period if 2 <= period <= len(self._detect_buf) // 2 else None

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        self._n += 1

        if self._period is None:
            self._detect_buf.append(v)
            if len(self._detect_buf) >= 64:
                self._period = self._detect_period()
            ctx.scratch["seasonal_err"] = 0.0
            return

        phase = self._n % self._period
        alpha = self.cfg.alpha

        if phase not in self._phase_ewma:
            self._phase_ewma[phase] = v
            self._phase_dev[phase] = 1.0
        else:
            prev = self._phase_ewma[phase]
            self._phase_ewma[phase] = alpha * v + (1 - alpha) * prev
            err = abs(v - prev)
            self._phase_dev[phase] = 0.1 * err + 0.9 * self._phase_dev[phase]

        base = self._phase_ewma[phase]
        dev = self._phase_dev[phase] or 1.0
        err = abs(v - base) / dev
        ctx.scratch["seasonal_err"] = err
        ctx.scratch["seasonal_phase"] = phase
        ctx.scratch["seasonal_period"] = self._period

        if not ctx.is_warmup and err > self.cfg.multiplier:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"period": self._period, "phase_ewma": self._phase_ewma,
                "phase_dev": self._phase_dev, "n": self._n,
                "detect_buf": self._detect_buf}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._period = sd.get("period")
        self._phase_ewma = {int(k): v for k, v in sd.get("phase_ewma", {}).items()}
        self._phase_dev = {int(k): v for k, v in sd.get("phase_dev", {}).items()}
        self._n = sd.get("n", 0)
        self._detect_buf = sd.get("detect_buf", [])


# ---------------------------------------------------------------------------
# 5.17 MahalStep — Mahalanobis distance (multivariate)
# ---------------------------------------------------------------------------

@register_step
class MahalStep(DetectorStep):
    """Rolling EWMA covariance + Woodbury rank-1 update for Mahalanobis distance."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._cov_inv = None
        self._mean_vec = None
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.multivariate or not _NP:
            ctx.scratch["mahal_dist"] = 0.0
            return

        v = ctx.value
        if not isinstance(v, (list, tuple)):
            ctx.scratch["mahal_dist"] = 0.0
            return

        x = np.array([float(xi) for xi in v[:self.cfg.n_channels]])
        self._n += 1
        a = self.cfg.cov_alpha

        if self._mean_vec is None:
            self._mean_vec = x.copy()
            d = self.cfg.n_channels
            self._cov_inv = np.eye(d)
            ctx.scratch["mahal_dist"] = 0.0
            return

        # EWMA mean update
        self._mean_vec = a * x + (1 - a) * self._mean_vec
        diff = x - self._mean_vec

        # Woodbury rank-1 covariance inverse update
        # C_new = (1-a)*C + a * diff @ diff.T
        # Use Sherman-Morrison: (A + uv^T)^{-1} = A^{-1} - (A^{-1}u v^T A^{-1})/(1 + v^T A^{-1} u)
        C_inv = self._cov_inv
        u = diff.reshape(-1, 1)
        C_inv_u = C_inv @ u
        denom = (1.0 - a) + a * float(u.T @ C_inv_u)
        if abs(denom) > 1e-12:
            self._cov_inv = (C_inv - a * (C_inv_u @ C_inv_u.T) / denom) / (1.0 - a)
        else:
            d = self.cfg.n_channels
            self._cov_inv = np.eye(d)

        # Mahalanobis distance
        md = float(math.sqrt(max(0.0, float(diff @ self._cov_inv @ diff))))
        ctx.scratch["mahal_dist"] = md

        thresh = self.cfg.multiplier * math.sqrt(self.cfg.n_channels)
        if not ctx.is_warmup and md > thresh:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean_vec": self._mean_vec.tolist() if self._mean_vec is not None and _NP else None,
            "cov_inv": self._cov_inv.tolist() if self._cov_inv is not None and _NP else None,
            "n": self._n,
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)
        mv = sd.get("mean_vec")
        ci = sd.get("cov_inv")
        if mv is not None and _NP:
            self._mean_vec = np.array(mv)
        if ci is not None and _NP:
            self._cov_inv = np.array(ci)


# ---------------------------------------------------------------------------
# 5.18 RRSStep — Rhythm Resonance Score (FRM Axiom 11)
# ---------------------------------------------------------------------------

@register_step
class RRSStep(DetectorStep):
    """RRS: harmonic resonance score — ratio of harmonic power to total power."""

    def __init__(self, config: SentinelConfig, regime_step: Optional["RegimeStep"] = None):
        self.cfg = config
        self._regime = regime_step
        self.reset()

    def reset(self):
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        n = self.cfg.rpi_window
        if len(w) < n or not _NP:
            ctx.scratch["rrs"] = 0.0
            return

        arr = np.array(list(w)[-n:])
        arr = arr - arr.mean()
        spec = np.abs(np.fft.rfft(arr)) ** 2
        total = spec.sum()
        if total < 1e-12:
            ctx.scratch["rrs"] = 0.0
            return

        # Fundamental: peak frequency; harmonics: 2x, 3x
        peak_idx = int(np.argmax(spec[1:])) + 1
        harm_power = spec[peak_idx]
        for mult in [2, 3]:
            hi = peak_idx * mult
            if hi < len(spec):
                harm_power += spec[hi]
        rrs = float(harm_power / total)
        ctx.scratch["rrs"] = rrs

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# 5.19 AlertReasonsStep
# ---------------------------------------------------------------------------

@register_step
class AlertReasonsStep(DetectorStep):
    """Aggregate alert reasons list — must be last step."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        pass

    def update(self, ctx: StepContext) -> None:
        reasons: List[str] = []
        s = ctx.scratch
        if s.get("cusum_alert"):
            reasons.append("cusum_mean_shift")
        if s.get("var_cusum_alert"):
            reasons.append("cusum_variance_spike")
        if s.get("regime_change"):
            reasons.append("regime_change")
        if s.get("ph_alert"):
            reasons.append("gradual_drift")
        if s.get("osc_alert"):
            reasons.append("oscillation_damping")
        if s.get("cpd_score", 0) > self.cfg.cpd_threshold:
            reasons.append("change_point")
        if abs(s.get("z_score", 0)) > self.cfg.multiplier:
            reasons.append("ewma_threshold")
        if s.get("rfi", 0) > self.cfg.rfi_threshold:
            reasons.append("high_fractal_irregularity")
        ews = s.get("ews_regime", "stable")
        if ews in ("approaching", "critical"):
            reasons.append(f"ews_{ews}")
        pe = s.get("pe", 0.5)
        pe_base = s.get("pe_baseline", 0.5)
        if abs(pe - pe_base) > self.cfg.pe_threshold:
            if pe < pe_base:
                reasons.append("low_entropy_ordered")
            else:
                reasons.append("high_entropy_chaotic")
        if s.get("mahal_dist", 0) > self.cfg.multiplier * math.sqrt(self.cfg.n_channels):
            reasons.append("mahalanobis_multivariate")
        # V12.0 — include structured alert types in reasons list
        for alert in s.get("v9_active_alerts", []):
            if alert.alert_type.value not in reasons:
                reasons.append(alert.alert_type.value)
        # V12.0 — compute channel summary string
        channel_parts: List[str] = []
        if s.get("structural_snapshot") is not None:
            channel_parts.append("structural:active")
        if s.get("frequency_bands") is not None:
            channel_parts.append("rhythmic:active")
        cm = s.get("coupling_matrix")
        if cm is not None:
            cst = ("degraded" if cm.composite_coupling_score < self.cfg.coupling_degradation_threshold
                   else "healthy")
            channel_parts.append(f"coupling:{cst}")
        cc = s.get("channel_coherence")
        if cc is not None:
            coh = "decoupled" if cc.coherence_score < self.cfg.coherence_threshold else "coherent"
            channel_parts.append(f"coherence:{coh}")
        ctx.scratch["channel_summary"] = (
            " | ".join(channel_parts) if channel_parts else "channels:initializing"
        )
        ctx.scratch["alert_reasons"] = reasons

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass


# ===========================================================================
# V12.0 PIPELINE STEPS — Three-Channel Extension
# ===========================================================================

# ---------------------------------------------------------------------------
# V12 Step 4a: StructuralSnapshotStep — Channel 1
# ---------------------------------------------------------------------------

@register_step
class StructuralSnapshotStep(DetectorStep):
    """Compute Channel 1 structural snapshot from current windowed data.

    Step 4a — inserted immediately after CoreEWMAStep.
    Treats network topology as active transmitter: computes statistical
    properties of the input data stream as an independent information channel.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._snapshot_history: deque = deque(
            maxlen=max(self.cfg.coherence_window * 2, 40))

    def update(self, ctx: StepContext) -> None:
        if not (self.cfg.enable_channel_coherence or self.cfg.enable_cascade_detection):
            ctx.scratch["structural_snapshot"] = None
            ctx.scratch["_structural_snapshot_history"] = self._snapshot_history
            return

        w = list(ctx.bank.get("scalar"))
        n = len(w)
        if n < 4:
            ctx.scratch["structural_snapshot"] = None
            ctx.scratch["_structural_snapshot_history"] = self._snapshot_history
            return

        mean = _mean(w)
        variance = _mean([(x - mean) ** 2 for x in w])
        std = math.sqrt(variance) if variance > 1e-20 else 1e-10

        skewness = (sum((x - mean) ** 3 for x in w) / (n * std ** 3)
                    if std > 1e-10 else 0.0)
        kurtosis = (sum((x - mean) ** 4 for x in w) / (n * std ** 4) - 3
                    if std > 1e-10 else 0.0)

        denom = sum((x - mean) ** 2 for x in w)
        ac1 = (sum((w[i] - mean) * (w[i - 1] - mean) for i in range(1, n)) / denom
               if denom > 1e-20 else 0.0)
        ac2 = (sum((w[i] - mean) * (w[i - 2] - mean) for i in range(2, n)) / denom
               if denom > 1e-20 else 0.0)

        half = n // 2
        if half > 1:
            mu1 = _mean(w[:half])
            mu2 = _mean(w[half:])
            v1 = _mean([(x - mu1) ** 2 for x in w[:half]])
            v2 = _mean([(x - mu2) ** 2 for x in w[half:]])
            stationarity = 1.0 - abs(v2 - v1) / (v1 + v2 + 1e-10)
        else:
            stationarity = 1.0

        snapshot = StructuralSnapshot(
            mean=mean,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            autocorrelation_lag1=ac1,
            autocorrelation_lag2=ac2,
            stationarity_score=max(0.0, min(1.0, stationarity)),
            timestamp=ctx.step,
        )
        self._snapshot_history.append(snapshot)
        ctx.scratch["structural_snapshot"] = snapshot
        ctx.scratch["_structural_snapshot_history"] = self._snapshot_history

    def state_dict(self) -> Dict[str, Any]:
        return {"n_snapshots": len(self._snapshot_history)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass  # in-memory history only


# ---------------------------------------------------------------------------
# V12 Step 4b: FrequencyDecompositionStep — Channel 2
# ---------------------------------------------------------------------------

@register_step
class FrequencyDecompositionStep(DetectorStep):
    """Decompose windowed signal into five frequency band carrier waves.

    Step 4b — inserted immediately after StructuralSnapshotStep.
    Implements Channel 2: broadband multiplexed oscillatory transmission.
    Uses numpy FFT if available; falls back to pure-Python DFT.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._bands_history: deque = deque(
            maxlen=max(self.cfg.coupling_trend_window * 4, 60))

    def _pure_python_bands(self, data: List[float]) -> tuple:
        N = len(data)
        spectrum = []
        for k in range(N // 2 + 1):
            re = sum(data[t] * math.cos(2 * math.pi * k * t / N) for t in range(N))
            im = sum(data[t] * math.sin(2 * math.pi * k * t / N) for t in range(N))
            mag = math.sqrt(re ** 2 + im ** 2)
            phase = math.atan2(-im, re)
            spectrum.append((mag, phase))
        total = len(spectrum)

        def pure_band(lo_frac, hi_frac):
            lo_i = int(lo_frac * total)
            hi_i = int(hi_frac * total)
            if hi_i <= lo_i:
                return 0.0, 0.0
            s = spectrum[lo_i:hi_i]
            avg_mag = sum(x[0] for x in s) / len(s)
            avg_phase = sum(x[1] for x in s) / len(s)
            return avg_mag, avg_phase

        ul_p, ul_ph = pure_band(0.00, 0.05)
        l_p,  l_ph  = pure_band(0.05, 0.15)
        m_p,  m_ph  = pure_band(0.15, 0.40)
        h_p,  h_ph  = pure_band(0.40, 0.70)
        uh_p, uh_ph = pure_band(0.70, 1.00)
        return ul_p, ul_ph, l_p, l_ph, m_p, m_ph, h_p, h_ph, uh_p, uh_ph

    def _compute_bands(self, data: List[float], timestamp: int) -> Optional[FrequencyBands]:
        n = len(data)
        try:
            import numpy as _np_local
            arr = _np_local.array(data, dtype=float)
            fft = _np_local.fft.rfft(arr)
            freqs = _np_local.fft.rfftfreq(n)
            magnitudes = _np_local.abs(fft)
            phases = _np_local.angle(fft)

            def band_stats(lo, hi):
                mask = (freqs >= lo) & (freqs < hi)
                if not _np_local.any(mask):
                    return 0.0, 0.0
                return float(_np_local.mean(magnitudes[mask])), float(_np_local.mean(phases[mask]))

            ul_p, ul_ph = band_stats(0.00, 0.05)
            l_p,  l_ph  = band_stats(0.05, 0.15)
            m_p,  m_ph  = band_stats(0.15, 0.40)
            h_p,  h_ph  = band_stats(0.40, 0.70)
            uh_p, uh_ph = band_stats(0.70, 1.00)
        except ImportError:
            ul_p, ul_ph, l_p, l_ph, m_p, m_ph, h_p, h_ph, uh_p, uh_ph = \
                self._pure_python_bands(data)

        return FrequencyBands(
            ultra_low_power=ul_p, low_power=l_p, mid_power=m_p,
            high_power=h_p, ultra_high_power=uh_p,
            ultra_low_phase=ul_ph, low_phase=l_ph, mid_phase=m_ph,
            high_phase=h_ph, ultra_high_phase=uh_ph,
            timestamp=timestamp,
        )

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_frequency_decomposition:
            ctx.scratch["frequency_bands"] = None
            ctx.scratch["_bands_history"] = self._bands_history
            return

        w = list(ctx.bank.get("scalar"))
        n = len(w)
        if n < self.cfg.min_window_for_fft:
            ctx.scratch["frequency_bands"] = None
            ctx.scratch["_bands_history"] = self._bands_history
            return

        bands = self._compute_bands(w, ctx.step)
        if bands is not None:
            self._bands_history.append(bands)
        ctx.scratch["frequency_bands"] = bands
        ctx.scratch["_bands_history"] = self._bands_history

    def state_dict(self) -> Dict[str, Any]:
        return {"n_bands": len(self._bands_history)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass  # in-memory history only


# ---------------------------------------------------------------------------
# V12 Step 15a: BandAnomalyStep
# ---------------------------------------------------------------------------

@register_step
class BandAnomalyStep(DetectorStep):
    """Per-carrier-wave anomaly detection invisible to composite signal.

    Step 15a — run after existing detection steps.
    Applies independent EWMA anomaly detection to each frequency band's
    power time series. A band-specific anomaly not present in the composite
    signal is invisible to v8.0 but detectable here.
    """

    BAND_NAMES: List[str] = ["ultra_low", "low", "mid", "high", "ultra_high"]

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._band_ewma: Dict[str, float] = {}
        self._band_dev: Dict[str, float] = {}

    def _ewma_anomaly_score(self, power: float, band_name: str) -> float:
        """Return normalized anomaly score (>1.0 = anomalous) for a band."""
        alpha = self.cfg.alpha
        if band_name not in self._band_ewma:
            self._band_ewma[band_name] = power
            self._band_dev[band_name] = 1.0
            return 0.0
        prev = self._band_ewma[band_name]
        self._band_ewma[band_name] = alpha * power + (1 - alpha) * prev
        err = abs(power - prev)
        self._band_dev[band_name] = (
            alpha * err + (1 - alpha) * self._band_dev.get(band_name, 1.0))
        self._band_dev[band_name] = max(self._band_dev[band_name], 1e-10)
        z = abs(power - self._band_ewma[band_name]) / self._band_dev[band_name]
        return z / (self.cfg.multiplier + 1e-10)

    def update(self, ctx: StepContext) -> None:
        fb = ctx.scratch.get("frequency_bands")
        if fb is None:
            ctx.scratch["band_anomalies"] = {}
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        current_powers = [
            fb.ultra_low_power, fb.low_power, fb.mid_power,
            fb.high_power, fb.ultra_high_power,
        ]
        anomalies: Dict[str, float] = {}
        for name, power in zip(self.BAND_NAMES, current_powers):
            score = self._ewma_anomaly_score(power, name)
            if score > 1.0:
                anomalies[name] = min(1.0, score)

        ctx.scratch["band_anomalies"] = anomalies
        alerts: List[Alert] = list(ctx.scratch.get("v9_active_alerts", []))
        if anomalies and not ctx.is_warmup and not ctx.scratch.get("alert", False):
            # Per-band anomaly not present in composite — new information
            alert = Alert(
                alert_type=AlertType.BAND_ANOMALY,
                severity=AlertSeverity.WARNING,
                score=max(anomalies.values()),
                message=(
                    f"Per-band anomaly in bands: {sorted(anomalies.keys())}. "
                    f"Invisible to composite detection."
                ),
            )
            alerts.append(alert)
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        ctx.scratch["v9_active_alerts"] = alerts

    def state_dict(self) -> Dict[str, Any]:
        return {"band_ewma": dict(self._band_ewma), "band_dev": dict(self._band_dev)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._band_ewma = sd.get("band_ewma", {})
        self._band_dev = sd.get("band_dev", {})


# ---------------------------------------------------------------------------
# V12 Steps 15b+15c: CrossFrequencyCouplingStep
# ---------------------------------------------------------------------------

@register_step
class CrossFrequencyCouplingStep(DetectorStep):
    """Measure cross-frequency phase-amplitude coupling and detect degradation.

    Step 15b+15c — computes CouplingMatrix and generates COUPLING_DEGRADATION
    alert when composite_coupling_score falls below threshold.
    Earlier warning than single-band anomaly detection.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._coupling_degradation_active: bool = False

    @staticmethod
    def _pac(low_phases: List[float], high_powers: List[float]) -> float:
        """Phase-amplitude coupling: modulation index approximation."""
        n = len(low_phases)
        if n < 2:
            return 0.0
        bins: List[List[float]] = [[] for _ in range(8)]
        for ph, pw in zip(low_phases, high_powers):
            bin_idx = int((ph + math.pi) / (2 * math.pi) * 8) % 8
            bins[bin_idx].append(pw)
        bin_means = [sum(b) / len(b) if b else 0.0 for b in bins]
        overall_mean = sum(bin_means) / 8.0
        variance = sum((m - overall_mean) ** 2 for m in bin_means) / 8.0
        best_dev = max(bin_means) - overall_mean
        max_variance = best_dev ** 2 if best_dev > 0 else 1e-10
        return variance / (max_variance + 1e-10)

    def _composite_from_slice(self, bands_slice: list) -> float:
        if len(bands_slice) < 2:
            return 0.0
        return sum([
            self._pac([b.ultra_low_phase for b in bands_slice],
                      [b.low_power for b in bands_slice]),
            self._pac([b.low_phase for b in bands_slice],
                      [b.mid_power for b in bands_slice]),
            self._pac([b.mid_phase for b in bands_slice],
                      [b.high_power for b in bands_slice]),
            self._pac([b.high_phase for b in bands_slice],
                      [b.ultra_high_power for b in bands_slice]),
        ]) / 4.0

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_coupling_detection:
            ctx.scratch["coupling_matrix"] = None
            ctx.scratch["coupling_degradation_active"] = False
            self._coupling_degradation_active = False
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        bands_history = ctx.scratch.get("_bands_history")
        if bands_history is None or len(bands_history) < self.cfg.coupling_trend_window:
            ctx.scratch["coupling_matrix"] = None
            ctx.scratch["coupling_degradation_active"] = self._coupling_degradation_active
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        recent = list(bands_history)[-self.cfg.coupling_trend_window:]
        ul_phases = [b.ultra_low_phase for b in recent]
        l_phases  = [b.low_phase for b in recent]
        m_phases  = [b.mid_phase for b in recent]
        h_phases  = [b.high_phase for b in recent]
        l_powers  = [b.low_power for b in recent]
        m_powers  = [b.mid_power for b in recent]
        h_powers  = [b.high_power for b in recent]
        uh_powers = [b.ultra_high_power for b in recent]

        ul_to_l = self._pac(ul_phases, l_powers)
        l_to_m  = self._pac(l_phases,  m_powers)
        m_to_h  = self._pac(m_phases,  h_powers)
        h_to_uh = self._pac(h_phases,  uh_powers)
        composite = (ul_to_l + l_to_m + m_to_h + h_to_uh) / 4.0

        half = len(recent) // 2
        if half >= 2:
            early = self._composite_from_slice(recent[:half])
            late  = self._composite_from_slice(recent[half:])
            trend = late - early
        else:
            trend = 0.0

        coupling = CouplingMatrix(
            ultra_low_to_low=ul_to_l,
            low_to_mid=l_to_m,
            mid_to_high=m_to_h,
            high_to_ultra_high=h_to_uh,
            composite_coupling_score=composite,
            coupling_trend=trend,
            timestamp=ctx.step,
        )
        ctx.scratch["coupling_matrix"] = coupling

        alerts: List[Alert] = list(ctx.scratch.get("v9_active_alerts", []))
        if composite < self.cfg.coupling_degradation_threshold and not ctx.is_warmup:
            self._coupling_degradation_active = True
            alert = Alert(
                alert_type=AlertType.COUPLING_DEGRADATION,
                severity=AlertSeverity.WARNING,
                score=composite,
                message=(
                    f"Cross-frequency coupling degraded to {composite:.3f} "
                    f"(threshold: {self.cfg.coupling_degradation_threshold}). "
                    f"Trend: {trend:+.3f}. Heterodyned information channel at risk."
                ),
            )
            alerts.append(alert)
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        else:
            self._coupling_degradation_active = False

        ctx.scratch["coupling_degradation_active"] = self._coupling_degradation_active
        ctx.scratch["v9_active_alerts"] = alerts

    def state_dict(self) -> Dict[str, Any]:
        return {"coupling_degradation_active": self._coupling_degradation_active}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._coupling_degradation_active = sd.get("coupling_degradation_active", False)


# ---------------------------------------------------------------------------
# V12 Steps 15d+15e: ChannelCoherenceStep
# ---------------------------------------------------------------------------

@register_step
class ChannelCoherenceStep(DetectorStep):
    """Measure structural-rhythmic channel coherence and detect decoupling.

    Step 15d+15e — computes ChannelCoherence and generates
    STRUCTURAL_RHYTHMIC_DECOUPLING alert when coherence_score falls below
    coherence_threshold. Independent of anomalies in either individual channel.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._rhythmic_history: deque = deque(
            maxlen=self.cfg.coherence_window * 2)
        self._sr_decoupling_active: bool = False

    def _coherence_score_from_slices(
        self, struct_slice: list, rhythmic_slice: list
    ) -> float:
        if len(struct_slice) < 2 or len(rhythmic_slice) < 2:
            return 0.5
        n = min(len(struct_slice), len(rhythmic_slice))
        ss = struct_slice[-n:]
        rs = rhythmic_slice[-n:]
        sc = [abs(ss[i].variance - ss[i - 1].variance) for i in range(1, len(ss))]
        rc = [abs(rs[i] - rs[i - 1]) for i in range(1, len(rs))]
        s_rate = _mean(sc) if sc else 0.0
        r_rate = _mean(rc) if rc else 0.0
        max_rate = max(s_rate, r_rate, 1e-10)
        return max(0.0, min(1.0, 1.0 - abs(s_rate - r_rate) / max_rate))

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_channel_coherence:
            ctx.scratch["channel_coherence"] = None
            ctx.scratch["sr_decoupling_active"] = False
            self._sr_decoupling_active = False
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        # Accumulate rhythmic proxy (EWMA value)
        self._rhythmic_history.append(ctx.scratch.get("ewma", ctx.current))

        structural_history = ctx.scratch.get("_structural_snapshot_history")
        window = self.cfg.coherence_window

        if (structural_history is None
                or len(structural_history) < window
                or len(self._rhythmic_history) < window):
            ctx.scratch["channel_coherence"] = None
            ctx.scratch["sr_decoupling_active"] = self._sr_decoupling_active
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        recent_struct = list(structural_history)[-window:]
        recent_rhythmic = list(self._rhythmic_history)[-window:]

        sc = [abs(recent_struct[i].variance - recent_struct[i - 1].variance)
              for i in range(1, len(recent_struct))]
        rc = [abs(recent_rhythmic[i] - recent_rhythmic[i - 1])
              for i in range(1, len(recent_rhythmic))]
        structural_change_rate = _mean(sc) if sc else 0.0
        rhythmic_change_rate = _mean(rc) if rc else 0.0
        max_rate = max(structural_change_rate, rhythmic_change_rate, 1e-10)
        coherence_score = max(
            0.0, min(1.0, 1.0 - abs(structural_change_rate - rhythmic_change_rate) / max_rate)
        )

        half = window // 2
        early_coh = self._coherence_score_from_slices(
            recent_struct[:half], recent_rhythmic[:half])
        late_coh = self._coherence_score_from_slices(
            recent_struct[half:], recent_rhythmic[half:])
        decoupling_trend = late_coh - early_coh

        coherence = ChannelCoherence(
            coherence_score=coherence_score,
            structural_change_rate=structural_change_rate,
            rhythmic_change_rate=rhythmic_change_rate,
            decoupling_trend=decoupling_trend,
            timestamp=ctx.step,
        )
        ctx.scratch["channel_coherence"] = coherence

        alerts: List[Alert] = list(ctx.scratch.get("v9_active_alerts", []))
        if coherence_score < self.cfg.coherence_threshold and not ctx.is_warmup:
            self._sr_decoupling_active = True
            alert = Alert(
                alert_type=AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING,
                severity=AlertSeverity.ALERT,
                score=coherence_score,
                message=(
                    f"Channel 1-2 coherence degraded to {coherence_score:.3f} "
                    f"(threshold: {self.cfg.coherence_threshold}). "
                    f"Structural rate: {structural_change_rate:.4f}. "
                    f"Rhythmic rate: {rhythmic_change_rate:.4f}. "
                    f"Channels operating independently."
                ),
            )
            alerts.append(alert)
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        else:
            self._sr_decoupling_active = False

        ctx.scratch["sr_decoupling_active"] = self._sr_decoupling_active
        ctx.scratch["v9_active_alerts"] = alerts

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rhythmic_history": list(self._rhythmic_history),
            "sr_decoupling_active": self._sr_decoupling_active,
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._rhythmic_history = deque(
            sd.get("rhythmic_history", []),
            maxlen=self.cfg.coherence_window * 2,
        )
        self._sr_decoupling_active = sd.get("sr_decoupling_active", False)


# ---------------------------------------------------------------------------
# V12 Step 19a: CascadePrecursorStep
# ---------------------------------------------------------------------------

@register_step
class CascadePrecursorStep(DetectorStep):
    """Detect incipient scale-level reversion cascade precursor.

    Step 19a — requires ALL conditions simultaneously:
      1. COUPLING_DEGRADATION alert active
      2. STRUCTURAL_RHYTHMIC_DECOUPLING alert active
      3. At least cascade_ews_threshold EWS indicators elevated
    This combined signature indicates not a local anomaly but an incipient
    scale-level reversion event.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._cascade_active: bool = False

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_cascade_detection:
            ctx.scratch["cascade_precursor_active"] = False
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        coupling_active = ctx.scratch.get("coupling_degradation_active", False)
        decoupling_active = ctx.scratch.get("sr_decoupling_active", False)

        ews_indicators = {
            "ews_score": ctx.scratch.get("ews_score", 0.0),
            "anomaly_score": ctx.scratch.get("anomaly_score", 0.0),
            "rfi": ctx.scratch.get("rfi", 0.0),
        }
        elevated_ews = sum(
            1 for v in ews_indicators.values() if v > self.cfg.ews_threshold
        )

        alerts: List[Alert] = list(ctx.scratch.get("v9_active_alerts", []))
        if (coupling_active and decoupling_active
                and elevated_ews >= self.cfg.cascade_ews_threshold
                and not ctx.is_warmup):
            self._cascade_active = True
            alert = Alert(
                alert_type=AlertType.CASCADE_PRECURSOR,
                severity=AlertSeverity.CRITICAL,
                score=1.0,
                message=(
                    f"CASCADE PRECURSOR DETECTED. "
                    f"Cross-frequency coupling degraded. "
                    f"Structural-rhythmic decoupling active. "
                    f"{elevated_ews} EWS indicators elevated. "
                    f"Combined signature indicates incipient scale-level "
                    f"reversion event. Immediate review warranted."
                ),
            )
            alerts.append(alert)
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        else:
            self._cascade_active = False

        ctx.scratch["cascade_precursor_active"] = self._cascade_active
        ctx.scratch["v9_active_alerts"] = alerts

    def state_dict(self) -> Dict[str, Any]:
        return {"cascade_active": self._cascade_active}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._cascade_active = sd.get("cascade_active", False)


# ---------------------------------------------------------------------------
# V12 Step 19b: DegradationSequenceStep
# ---------------------------------------------------------------------------

@register_step
class DegradationSequenceStep(DetectorStep):
    """Log temporal ordering of channel degradation events (Channel 3).

    Step 19b — the sequence and ordering of degradation is diagnostic
    about regime change type and severity.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._active_sequence: Optional[DegradationSequence] = None
        self._sequence_history: deque = deque(maxlen=self.cfg.sequence_retention)

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_sequence_logging:
            ctx.scratch["degradation_sequence"] = None
            return

        active_alerts: List[Alert] = ctx.scratch.get("v9_active_alerts", [])
        alert_types = {a.alert_type for a in active_alerts}
        now = ctx.step

        if not alert_types:
            if self._active_sequence is not None:
                self._sequence_history.append(self._active_sequence)
                self._active_sequence = None
        elif self._active_sequence is None:
            first = list(alert_types)[0]
            self._active_sequence = DegradationSequence(
                first_channel_anomaly=first.value,
                first_anomaly_timestamp=now,
                second_channel_anomaly=None,
                second_anomaly_timestamp=None,
                coupling_degradation_timestamp=(
                    now if AlertType.COUPLING_DEGRADATION in alert_types else None),
                decoupling_timestamp=(
                    now if AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING in alert_types else None),
                cascade_precursor_timestamp=(
                    now if AlertType.CASCADE_PRECURSOR in alert_types else None),
                sequence_pattern=_build_sequence_narrative(alert_types),
            )
        else:
            existing = self._active_sequence
            alert_values = {a.value for a in alert_types}
            new_vals = alert_values - {existing.first_channel_anomaly}
            second = existing.second_channel_anomaly or (
                next(iter(new_vals), None) if new_vals else None
            )
            self._active_sequence = DegradationSequence(
                first_channel_anomaly=existing.first_channel_anomaly,
                first_anomaly_timestamp=existing.first_anomaly_timestamp,
                second_channel_anomaly=second,
                second_anomaly_timestamp=existing.second_anomaly_timestamp or (
                    now if second is not None
                    and existing.second_channel_anomaly is None else None
                ),
                coupling_degradation_timestamp=(
                    existing.coupling_degradation_timestamp or
                    (now if AlertType.COUPLING_DEGRADATION in alert_types else None)
                ),
                decoupling_timestamp=(
                    existing.decoupling_timestamp or
                    (now if AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING in alert_types else None)
                ),
                cascade_precursor_timestamp=(
                    existing.cascade_precursor_timestamp or
                    (now if AlertType.CASCADE_PRECURSOR in alert_types else None)
                ),
                sequence_pattern=_build_sequence_narrative(alert_types),
            )

        ctx.scratch["degradation_sequence"] = self._active_sequence

    def state_dict(self) -> Dict[str, Any]:
        return {"n_sequences": len(self._sequence_history)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass  # in-memory history only


# ===========================================================================
# Pipeline builder + legacy mapper
# ===========================================================================

def _build_default_pipeline(config: SentinelConfig) -> List[DetectorStep]:
    """Return ordered list of DetectorStep instances for a SentinelDetector.

    V12.0: 26 steps (19 v8 foundation + 7 three-channel extension).
    V8.0 steps are UNCHANGED — v12.0 steps are inserted at defined positions.
    """
    regime = RegimeStep(config)
    rrs = RRSStep(config, regime)
    return [
        CoreEWMAStep(config),               # Step 1  — MUST be first (populates bank)
        StructuralSnapshotStep(config),     # Step 4a — V12.0 Channel 1
        FrequencyDecompositionStep(config), # Step 4b — V12.0 Channel 2
        CUSUMStep(config),                  # Steps 2-19: existing v8.0 (UNCHANGED)
        regime,
        VarCUSUMStep(config),
        PageHinkleyStep(config),
        STIStep(config),
        TPSStep(config),
        OscDampStep(config),
        CPDStep(config),
        RPIStep(config),
        RFIStep(config),
        SSIStep(config),
        PEStep(config),
        EWSStep(config),
        AQBStep(config),
        SeasonalStep(config),
        MahalStep(config),
        rrs,
        BandAnomalyStep(config),            # Step 15a — V12.0
        CrossFrequencyCouplingStep(config), # Step 15b+15c — V12.0
        ChannelCoherenceStep(config),       # Step 15d+15e — V12.0
        CascadePrecursorStep(config),       # Step 19a — V12.0
        DegradationSequenceStep(config),    # Step 19b — V12.0
        AlertReasonsStep(config),           # MUST be last
    ]


def _legacy_kwargs_to_config(kw: dict) -> SentinelConfig:
    """Map v7.x flat kwargs to SentinelConfig (backward compat)."""
    mapping = {
        "alpha": "alpha",
        "dev_alpha": "dev_alpha",
        "multiplier": "multiplier",
        "warmup_periods": "warmup_periods",
        "regime_threshold": "regime_threshold",
        "regime_alpha_boost": "regime_alpha_boost",
        "multivariate": "multivariate",
        "n_channels": "n_channels",
        "cov_alpha": "cov_alpha",
        "rpi_window": "rpi_window",
        "rfi_window": "rfi_window",
        "rpi_threshold": "rpi_threshold",
        "rfi_threshold": "rfi_threshold",
        "pe_order": "pe_order",
        "pe_window": "pe_window",
        "pe_threshold": "pe_threshold",
        "ews_window": "ews_window",
        "ews_threshold": "ews_threshold",
        "sti_window": "sti_window",
        "tps_window": "tps_window",
        "osc_damp_window": "osc_damp_window",
        "osc_threshold": "osc_threshold",
        "cpd_window": "cpd_window",
        "cpd_threshold": "cpd_threshold",
        "ph_delta": "ph_delta",
        "ph_lambda": "ph_lambda",
        "var_cusum_k": "var_cusum_k",
        "var_cusum_h": "var_cusum_h",
        "seasonal_period": "seasonal_period",
        "quantile_threshold_mode": "quantile_threshold_mode",
        "aqb_window": "aqb_window",
        "aqb_q_low": "aqb_q_low",
        "aqb_q_high": "aqb_q_high",
        "history_maxlen": "history_maxlen",
        "csv_path": "csv_path",
    }
    mapped = {}
    for old, new in mapping.items():
        if old in kw:
            mapped[new] = kw[old]
    # v7.x compat: 'rsi_window' -> rpi_window (they shared same window)
    if "rsi_window" in kw and "rpi_window" not in mapped:
        mapped["rpi_window"] = kw["rsi_window"]
    return SentinelConfig(**mapped)


# ===========================================================================
# V12.0 SentinelResult — dict subclass with Three-Channel convenience methods
# ===========================================================================


class SentinelResult(dict):
    """V12.0 result object — dict subclass with Three-Channel convenience methods.

    All existing dict access patterns preserved for backward compatibility::

        result = det.update_and_check(value)
        if result["alert"]:          # v8.0 pattern — still works
            ...
        if result.is_cascade_precursor():  # v12.0 pattern
            ...

    """

    def is_cascade_precursor(self) -> bool:
        """Returns True if CASCADE_PRECURSOR alert is active."""
        return bool(self.get("cascade_precursor_active", False))

    def get_channel_status(self) -> Dict[str, str]:
        """Returns status of each channel: healthy, degrading, anomalous, or unknown."""
        status: Dict[str, str] = {}

        ss = self.get("structural_snapshot")
        if ss is not None:
            if ss.stationarity_score < 0.3:
                status["structural"] = "anomalous"
            elif ss.stationarity_score < 0.6:
                status["structural"] = "degrading"
            else:
                status["structural"] = "healthy"
        else:
            status["structural"] = "unknown"

        score = self.get("anomaly_score", 0.0)
        if score >= 1.0:
            status["rhythmic_composite"] = "anomalous"
        elif score > 0.5:
            status["rhythmic_composite"] = "degrading"
        else:
            status["rhythmic_composite"] = "healthy"

        cm = self.get("coupling_matrix")
        if cm is not None:
            if cm.coupling_trend < -0.1:
                status["coupling"] = "degrading"
            elif cm.composite_coupling_score < 0.3:
                status["coupling"] = "anomalous"
            else:
                status["coupling"] = "healthy"
        else:
            status["coupling"] = "unknown"

        cc = self.get("channel_coherence")
        if cc is not None:
            if cc.coherence_score < 0.2:
                status["coherence"] = "anomalous"
            elif cc.coherence_score < 0.4:
                status["coherence"] = "degrading"
            else:
                status["coherence"] = "healthy"
        else:
            status["coherence"] = "unknown"

        return status

    def get_degradation_narrative(self) -> str:
        """Returns human-readable description of current degradation sequence."""
        ds = self.get("degradation_sequence")
        if ds is not None:
            return ds.sequence_pattern
        return "No active degradation sequence."

    def get_primary_carrier_wave(self) -> str:
        """Returns frequency band currently carrying the most information."""
        fb = self.get("frequency_bands")
        if fb is None:
            return "decomposition_disabled"
        bands = {
            "ultra_low": fb.ultra_low_power,
            "low": fb.low_power,
            "mid": fb.mid_power,
            "high": fb.high_power,
            "ultra_high": fb.ultra_high_power,
        }
        return max(bands, key=bands.get)


# ===========================================================================
# SECTION 6 — SentinelDetector (main orchestrator)
# ===========================================================================

class SentinelDetector:
    """Streaming anomaly detector — v8.0 pipeline architecture.

    Usage::

        det = SentinelDetector()
        for value in stream:
            result = det.update_and_check(value)
            if result["alert"]:
                print(result["alert_reasons"])

    Backward-compatible with all v7.x kwargs.
    ``Detector_7_10`` is an alias for this class.
    """

    def __init__(self, config: Optional[SentinelConfig] = None,
                 *, steps: Optional[List[DetectorStep]] = None,
                 **legacy_kwargs):
        if config is None:
            if legacy_kwargs:
                config = _legacy_kwargs_to_config(legacy_kwargs)
            else:
                config = SentinelConfig()
        self.config = config
        self._bank = WindowBank()
        if steps is not None:
            self._pipeline = list(steps)
        else:
            self._pipeline = _build_default_pipeline(config)
        # Wire CoreEWMAStep reference into scratch for RegimeStep soft-boost
        self._core_step: Optional[CoreEWMAStep] = None
        for s in self._pipeline:
            if isinstance(s, CoreEWMAStep):
                self._core_step = s
                break

        self._n = 0
        self._history: deque = deque(maxlen=config.history_maxlen)
        self._csv_file = None
        self._csv_writer = None
        if config.csv_path:
            self._open_csv(config.csv_path)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update_and_check(self, value) -> "SentinelResult":
        """Process one observation; return SentinelResult (dict subclass)."""
        ctx = StepContext(
            value=value,
            step=self._n,
            config=self.config,
            bank=self._bank,
            scratch={},
        )
        # Inject core step ref for regime soft-boost
        if self._core_step is not None:
            ctx.scratch["_core_step_ref"] = self._core_step

        for step in self._pipeline:
            step.update(ctx)

        result = SentinelResult(
            (k, v) for k, v in ctx.scratch.items()
            if not k.startswith("_")
        )
        result.setdefault("alert", False)
        result.setdefault("anomaly", False)
        result.setdefault("warmup", ctx.is_warmup)
        result.setdefault("z_score", 0.0)
        result.setdefault("anomaly_score", 0.0)
        result.setdefault("alert_reasons", [])
        # V12.0 defaults
        result.setdefault("frequency_bands", None)
        result.setdefault("structural_snapshot", None)
        result.setdefault("coupling_matrix", None)
        result.setdefault("channel_coherence", None)
        result.setdefault("degradation_sequence", None)
        result.setdefault("cascade_precursor_active", False)
        result.setdefault("band_anomalies", {})
        result.setdefault("channel_summary", "")
        result.setdefault("v9_active_alerts", [])
        result["step"] = self._n
        result["value"] = value if not isinstance(value, (list, tuple)) else list(value)

        self._n += 1
        self._history.append(result)
        if self._csv_writer is not None:
            self._write_csv_row(result)
        return result

    async def aupdate(self, value) -> Dict[str, Any]:
        """Async wrapper for update_and_check."""
        return self.update_and_check(value)

    # ------------------------------------------------------------------
    # Fitting / tuning
    # ------------------------------------------------------------------

    def fit(self, data: Sequence) -> "SentinelDetector":
        """Warm up detector on unlabeled data (returns self)."""
        for v in data:
            self.update_and_check(v)
        return self

    @classmethod
    def auto_tune(cls, data: Sequence,
                  labeled_data: Optional[Sequence[Tuple[Any, bool]]] = None,
                  alphas: Optional[List[float]] = None,
                  multipliers: Optional[List[float]] = None) -> "SentinelDetector":
        """Grid-search alpha/multiplier to maximise F1 on labeled_data
        (or minimise false-positive rate if labels not provided).
        """
        if alphas is None:
            alphas = [0.05, 0.1, 0.2, 0.3]
        if multipliers is None:
            multipliers = [2.0, 2.5, 3.0, 3.5, 4.0]

        best_score = -1.0
        best_cfg: Optional[SentinelConfig] = None

        for a in alphas:
            for m in multipliers:
                cfg = SentinelConfig(alpha=a, dev_alpha=a, multiplier=m)
                det = cls(config=cfg)
                if labeled_data is not None:
                    results = []
                    labels = []
                    for v, lbl in labeled_data:
                        r = det.update_and_check(v)
                        results.append(r)
                        labels.append(lbl)
                    # Compute F1
                    tp = fp = fn = 0
                    for r, lbl in zip(results, labels):
                        pred = r["alert"]
                        if pred and lbl: tp += 1
                        elif pred and not lbl: fp += 1
                        elif not pred and lbl: fn += 1
                    prec = tp / (tp + fp + 1e-10)
                    rec = tp / (tp + fn + 1e-10)
                    f1 = 2 * prec * rec / (prec + rec + 1e-10)
                    score = f1
                else:
                    results = [det.update_and_check(v) for v in data]
                    fp_rate = sum(1 for r in results if r["alert"] and not r["warmup"]) / (len(results) + 1)
                    score = 1.0 - fp_rate

                if score > best_score:
                    best_score = score
                    best_cfg = cfg

        return cls(config=best_cfg or SentinelConfig())

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self) -> str:
        """Serialize full detector state to JSON string."""
        sd: Dict[str, Any] = {
            "version": __version__,
            "n": self._n,
            "config": dataclasses.asdict(self.config),
            "bank": self._bank.state_dict(),
            "steps": [],
        }
        for i, step in enumerate(self._pipeline):
            sd["steps"].append({
                "cls": type(step).__name__,
                "idx": i,
                "state": step.state_dict(),
            })
        return json.dumps(sd)

    def load_state(self, json_str: str) -> None:
        """Restore detector state from JSON string."""
        sd = json.loads(json_str)
        self._n = sd.get("n", 0)
        self._bank.load_state(sd.get("bank", {}))
        step_states = sd.get("steps", [])
        for ss in step_states:
            idx = ss.get("idx", -1)
            if 0 <= idx < len(self._pipeline):
                self._pipeline[idx].load_state(ss.get("state", {}))

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, soft: bool = False) -> None:
        """Reset detector state.

        Parameters
        ----------
        soft:
            If True, only reset accumulators but keep warmup data.
            If False (default), full reset to factory state.
        """
        self._n = 0
        self._history.clear()
        self._bank.reset()
        for step in self._pipeline:
            step.reset()

    # ------------------------------------------------------------------
    # CSV / history
    # ------------------------------------------------------------------

    def _open_csv(self, path: str) -> None:
        self._csv_file = open(path, "w", newline="")
        self._csv_writer = None  # created lazily on first row

    def _write_csv_row(self, result: Dict[str, Any]) -> None:
        if self._csv_writer is None:
            fieldnames = [k for k in result.keys() if k != "alert_reasons"]
            fieldnames.append("alert_reasons")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames,
                                               extrasaction="ignore")
            self._csv_writer.writeheader()
        row = dict(result)
        row["alert_reasons"] = "|".join(result.get("alert_reasons", []))
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def flush_csv(self) -> None:
        """Flush pending CSV writes."""
        if self._csv_file is not None:
            self._csv_file.flush()

    def close(self) -> None:
        """Close CSV file if open."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_history(self, title: str = "Sentinel v8 Dashboard",
                     show: bool = True) -> Optional[Any]:
        """4-panel dashboard: value+EWMA, anomaly_score, PE, z_score."""
        if not _MPL:
            warnings.warn("matplotlib not available")
            return None
        history = list(self._history)
        if not history:
            return None

        steps = [r["step"] for r in history]
        values = [r.get("value", 0) if not isinstance(r.get("value"), list) else r["value"][-1]
                  for r in history]
        ewmas = [r.get("ewma", 0) for r in history]
        scores = [r.get("anomaly_score", 0) for r in history]
        pes = [r.get("pe", 0.5) for r in history]
        zs = [r.get("z_score", 0) for r in history]
        alerts = [i for i, r in enumerate(history) if r.get("alert")]

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        axes[0].plot(steps, values, color="steelblue", linewidth=0.8, label="Value")
        axes[0].plot(steps, ewmas, color="orange", linewidth=1.5, label="EWMA")
        for ai in alerts:
            axes[0].axvline(steps[ai], color="red", alpha=0.3, linewidth=0.5)
        axes[0].set_ylabel("Value / EWMA")
        axes[0].legend(fontsize=8)

        axes[1].plot(steps, scores, color="purple", linewidth=0.8)
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("Anomaly Score")

        axes[2].plot(steps, pes, color="teal", linewidth=0.8)
        axes[2].set_ylabel("Permutation Entropy")

        axes[3].plot(steps, zs, color="gray", linewidth=0.8)
        axes[3].axhline(self.config.multiplier, color="red", linestyle="--", linewidth=0.8)
        axes[3].axhline(-self.config.multiplier, color="red", linestyle="--", linewidth=0.8)
        axes[3].set_ylabel("Z-score")
        axes[3].set_xlabel("Step")

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"SentinelDetector(v{__version__}, n={self._n}, "
                f"alpha={self.config.alpha}, warmup={self.config.warmup_periods}, "
                f"three_channel=True)")

    def __len__(self) -> int:
        return self._n


# Backward-compat alias (ε)
Detector_7_10 = SentinelDetector


# ===========================================================================
# SECTION 7 — MultiStreamSentinel
# ===========================================================================

class MultiStreamSentinel:
    """Manage multiple independent named streams, each with its own detector.

    Each stream is lazily initialized on first observation.  Streams share
    a common ``config`` but have fully independent state.
    """

    def __init__(self, config: Optional[SentinelConfig] = None,
                 detector_factory: Optional[Callable] = None,
                 **legacy_kwargs):
        if config is None:
            if legacy_kwargs:
                config = _legacy_kwargs_to_config(legacy_kwargs)
            else:
                config = SentinelConfig()
        self.config = config
        self._factory = detector_factory or (lambda cfg: SentinelDetector(config=cfg))
        self._streams: Dict[str, SentinelDetector] = {}
        self._lock = threading.Lock()

    def update(self, stream_id: str, value) -> Dict[str, Any]:
        """Update a named stream; auto-create detector on first call."""
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = self._factory(self.config)
        return self._streams[stream_id].update_and_check(value)

    async def aupdate(self, stream_id: str, value) -> Dict[str, Any]:
        """Async variant of update."""
        return self.update(stream_id, value)

    def get_detector(self, stream_id: str) -> Optional[SentinelDetector]:
        return self._streams.get(stream_id)

    def list_streams(self) -> List[str]:
        return list(self._streams.keys())

    def reset_stream(self, stream_id: str) -> bool:
        """Reset a named stream; return False if not found."""
        if stream_id in self._streams:
            self._streams[stream_id].reset()
            return True
        return False

    def delete_stream(self, stream_id: str) -> bool:
        """Delete a named stream; return False if not found."""
        with self._lock:
            if stream_id in self._streams:
                self._streams.pop(stream_id)
                return True
        return False

    def status(self, stream_id: str) -> Dict[str, Any]:
        det = self._streams.get(stream_id)
        if det is None:
            return {"error": "stream not found"}
        history = list(det._history)
        alerts = sum(1 for r in history if r.get("alert"))
        return {
            "stream_id": stream_id,
            "n": det._n,
            "alert_count": alerts,
            "last_result": history[-1] if history else None,
        }

    def save_all(self) -> str:
        """Serialize all stream states to JSON string."""
        return json.dumps({
            sid: json.loads(det.save_state())
            for sid, det in self._streams.items()
        })

    def load_all(self, json_str: str) -> None:
        """Restore all stream states from JSON string."""
        data = json.loads(json_str)
        for sid, sd in data.items():
            if sid not in self._streams:
                self._streams[sid] = self._factory(self.config)
            self._streams[sid].load_state(json.dumps(sd))

    def __repr__(self) -> str:
        return f"MultiStreamSentinel(streams={list(self._streams.keys())})"


# ===========================================================================
# SECTION 8 — SentinelBenchmark
# ===========================================================================

class SentinelBenchmark:
    """Built-in evaluation harness.

    Generates five labeled anomaly archetypes and reports F1, AUPRC (PR curve
    via trapezoidal integration), VUS-PR (avg AUPRC over 5 buffer tolerances),
    mean detection lag, and naive 3-sigma baseline comparison.
    """

    ARCHETYPES: List[str] = ["point", "contextual", "collective", "drift", "variance"]

    def __init__(self, n: int = 500, seed: int = 42,
                 config: Optional[SentinelConfig] = None):
        self.n = n
        self.seed = seed
        self.config = config or SentinelConfig()
        self._rng: Any = None

    def _get_rng(self) -> Any:
        if self._rng is None:
            if _NP:
                self._rng = np.random.default_rng(self.seed)
            else:
                import random
                random.seed(self.seed)
                self._rng = random
        return self._rng

    def _randn(self, n: int) -> List[float]:
        rng = self._get_rng()
        if _NP:
            return rng.standard_normal(n).tolist()
        import random
        return [random.gauss(0, 1) for _ in range(n)]

    def generate(self, archetype: str) -> Tuple[List[float], List[bool]]:
        """Generate (data, labels) for a given archetype."""
        n = self.n
        data = self._randn(n)
        labels = [False] * n

        if archetype == "point":
            # Sparse large spikes
            idxs = list(range(50, n, 80))
            for i in idxs:
                if i < n:
                    data[i] += 8.0
                    labels[i] = True

        elif archetype == "contextual":
            # Normal values that are anomalous given seasonal context
            period = 20
            for i in range(n):
                # Add sinusoidal mean
                data[i] += 3.0 * math.sin(2 * math.pi * i / period)
            idxs = list(range(60, n, 100))
            for i in idxs:
                if i < n:
                    data[i] = data[i] - 6.0  # contextually anomalous
                    labels[i] = True

        elif archetype == "collective":
            # Runs of anomalous observations
            idxs = list(range(100, 120)) + list(range(300, 315))
            for i in idxs:
                if i < n:
                    data[i] += 4.0
                    labels[i] = True

        elif archetype == "drift":
            # Slow mean drift
            for i in range(n):
                if i > n // 2:
                    data[i] += (i - n // 2) * 0.02
            for i in range(n // 2 + 20, n):
                labels[i] = True

        elif archetype == "variance":
            # Sudden variance explosion
            for i in range(n // 2, n):
                data[i] *= 4.0
                labels[i] = True

        else:
            raise ValueError(f"Unknown archetype: {archetype}")

        return data, labels

    def _pr_auc(self, scores: List[float], labels: List[bool],
                tolerance: int = 0) -> float:
        """Trapezoidal AUPRC with buffer tolerance."""
        n = len(scores)
        # Expand labels with tolerance
        tol_labels = list(labels)
        if tolerance > 0:
            for i, lbl in enumerate(labels):
                if lbl:
                    for j in range(max(0, i - tolerance), min(n, i + tolerance + 1)):
                        tol_labels[j] = True

        paired = sorted(zip(scores, tol_labels), key=lambda x: -x[0])
        tp = fp = 0
        total_pos = sum(1 for lbl in tol_labels if lbl)
        if total_pos == 0:
            return 0.0
        precs = []
        recs = []
        for score, lbl in paired:
            if lbl:
                tp += 1
            else:
                fp += 1
            precs.append(tp / (tp + fp))
            recs.append(tp / total_pos)

        # Trapezoidal integration
        auc = 0.0
        for i in range(1, len(precs)):
            auc += (recs[i] - recs[i-1]) * (precs[i] + precs[i-1]) / 2
        return max(0.0, auc)

    def _vus_pr(self, scores: List[float], labels: List[bool]) -> float:
        """Volume Under PR Surface over tolerances {0,5,10,15,20}."""
        tols = [0, 5, 10, 15, 20]
        return sum(self._pr_auc(scores, labels, t) for t in tols) / len(tols)

    def _naive_baseline(self, data: List[float], labels: List[bool]) -> Dict[str, float]:
        """3-sigma naive baseline."""
        mu = _mean(data)
        sq = [(x - mu) ** 2 for x in data]
        std = math.sqrt(_mean(sq)) or 1.0
        preds = [abs(x - mu) > 3 * std for x in data]
        tp = sum(1 for p, l in zip(preds, labels) if p and l)
        fp = sum(1 for p, l in zip(preds, labels) if p and not l)
        fn = sum(1 for p, l in zip(preds, labels) if not p and l)
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        return {"f1": f1, "precision": prec, "recall": rec}

    def evaluate(self, archetype: str) -> Dict[str, Any]:
        """Run detector on one archetype; return metrics dict."""
        data, labels = self.generate(archetype)
        det = SentinelDetector(config=self.config)
        results = [det.update_and_check(v) for v in data]

        scores = [r.get("anomaly_score", 0.0) for r in results]
        preds = [r.get("alert", False) for r in results]

        # F1
        tp = fp = fn = 0
        detection_lags = []
        for i, (pred, lbl) in enumerate(zip(preds, labels)):
            if pred and lbl: tp += 1
            elif pred and not lbl: fp += 1
            elif not pred and lbl: fn += 1
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)

        # Detection lag for true anomalies
        anom_starts = [i for i, l in enumerate(labels) if l and (i == 0 or not labels[i-1])]
        for start in anom_starts:
            for j in range(start, min(start + 50, len(preds))):
                if preds[j]:
                    detection_lags.append(j - start)
                    break

        auprc = self._pr_auc(scores, labels)
        vus = self._vus_pr(scores, labels)
        naive = self._naive_baseline(data, labels)
        mean_lag = _mean(detection_lags) if detection_lags else float("inf")

        return {
            "archetype": archetype,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "auprc": auprc,
            "vus_pr": vus,
            "mean_lag": mean_lag,
            "naive_f1": naive["f1"],
            "n_alerts": sum(preds),
            "n_true_anomalies": sum(labels),
        }

    def run_suite(self, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """Run all 5 archetypes; print summary table."""
        results = {}
        for arch in self.ARCHETYPES:
            results[arch] = self.evaluate(arch)
        if verbose:
            print(f"\n{'─'*72}")
            print(f"{'Sentinel v8 Benchmark':^72}")
            print(f"{'─'*72}")
            header = f"{'Archetype':<14} {'F1':>6} {'AUPRC':>7} {'VUS-PR':>7} "
            header += f"{'Lag':>6} {'Naive F1':>9} {'Alerts':>7}"
            print(header)
            print(f"{'─'*72}")
            for arch, r in results.items():
                lag = f"{r['mean_lag']:.1f}" if r['mean_lag'] != float('inf') else "∞"
                print(f"{arch:<14} {r['f1']:>6.3f} {r['auprc']:>7.3f} {r['vus_pr']:>7.3f} "
                      f"{lag:>6} {r['naive_f1']:>9.3f} {r['n_alerts']:>7}")
            print(f"{'─'*72}\n")
        return results


# ===========================================================================
# SECTION 9 — SentinelServer (asyncio)
# ===========================================================================

class SentinelServer:
    """Async HTTP server wrapping MultiStreamSentinel.

    Endpoints:
      POST /update/<stream_id>       body: {"value": ...}
      GET  /streams                  returns list of stream IDs
      GET  /status/<stream_id>       returns stream status
      DELETE /stream/<stream_id>     delete a stream
      POST /reset/<stream_id>        reset a stream
      GET  /health                   version + uptime
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765,
                 config: Optional[SentinelConfig] = None):
        self.host = host
        self.port = port
        self.mss = MultiStreamSentinel(config=config)
        self._start_time = time.time()
        self._server: Optional[Any] = None

    async def _handle(self, reader: asyncio.StreamReader,
                      writer: asyncio.StreamWriter) -> None:
        try:
            raw = await asyncio.wait_for(reader.read(65536), timeout=5.0)
            lines = raw.decode("utf-8", errors="replace").split("\r\n")
            if not lines:
                writer.close()
                return
            req_line = lines[0].split()
            if len(req_line) < 2:
                writer.close()
                return
            method = req_line[0].upper()
            path = req_line[1]
            # Find body (after blank line)
            body_str = ""
            try:
                sep = raw.index(b"\r\n\r\n")
                body_str = raw[sep + 4:].decode("utf-8", errors="replace").strip()
            except ValueError:
                pass

            status, resp = await self._route(method, path, body_str)
            resp_bytes = json.dumps(resp).encode()
            http = (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(resp_bytes)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode() + resp_bytes
            writer.write(http)
            await writer.drain()
        except Exception as exc:
            err = json.dumps({"error": str(exc)}).encode()
            http = (f"HTTP/1.1 500 Internal Server Error\r\nContent-Length: {len(err)}\r\n\r\n").encode() + err
            writer.write(http)
            await writer.drain()
        finally:
            writer.close()

    async def _route(self, method: str, path: str,
                     body: str) -> Tuple[str, Any]:
        parts = [p for p in path.strip("/").split("/") if p]

        if method == "GET" and parts == ["health"]:
            return "200 OK", {
                "version": __version__,
                "uptime": time.time() - self._start_time,
                "streams": len(self.mss.list_streams()),
            }

        if method == "GET" and parts == ["streams"]:
            return "200 OK", {"streams": self.mss.list_streams()}

        if method == "GET" and len(parts) == 2 and parts[0] == "status":
            return "200 OK", self.mss.status(parts[1])

        if method == "POST" and len(parts) == 2 and parts[0] == "update":
            try:
                payload = json.loads(body) if body else {}
            except json.JSONDecodeError:
                return "400 Bad Request", {"error": "invalid JSON"}
            value = payload.get("value")
            if value is None:
                return "400 Bad Request", {"error": "missing 'value'"}
            result = await self.mss.aupdate(parts[1], value)
            return "200 OK", result

        if method == "DELETE" and len(parts) == 2 and parts[0] == "stream":
            ok = self.mss.delete_stream(parts[1])
            return ("200 OK", {"deleted": parts[1]}) if ok else ("404 Not Found", {"error": "not found"})

        if method == "POST" and len(parts) == 2 and parts[0] == "reset":
            ok = self.mss.reset_stream(parts[1])
            return ("200 OK", {"reset": parts[1]}) if ok else ("404 Not Found", {"error": "not found"})

        return "404 Not Found", {"error": f"no route for {method} {path}"}

    async def serve_forever(self) -> None:
        """Start async server (runs until cancelled)."""
        self._server = await asyncio.start_server(
            self._handle, self.host, self.port
        )
        async with self._server:
            await self._server.serve_forever()

    def run(self) -> None:
        """Blocking entry point (wraps asyncio.run)."""
        try:
            asyncio.run(self.serve_forever())
        except KeyboardInterrupt:
            pass


# ===========================================================================
# CLI entry point
# ===========================================================================

def _cli_main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fracttalix_sentinel_v1200",
        description="Fracttalix Sentinel v12.0 streaming anomaly detector",
    )
    parser.add_argument("--file", "-f", help="CSV file path (reads first column)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--serve", action="store_true", help="Start HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    cfg = SentinelConfig(alpha=args.alpha, multiplier=args.multiplier,
                         warmup_periods=args.warmup)

    if args.benchmark:
        bench = SentinelBenchmark(config=cfg)
        bench.run_suite()
        return

    if args.serve:
        server = SentinelServer(host=args.host, port=args.port, config=cfg)
        print(f"Sentinel v{__version__} server on {args.host}:{args.port}")
        server.run()
        return

    if args.file:
        det = SentinelDetector(config=cfg)
        with open(args.file, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    v = float(row[0])
                except (ValueError, IndexError):
                    continue
                r = det.update_and_check(v)
                if r.get("alert"):
                    print(f"[ALERT] step={r['step']} value={v:.4f} "
                          f"z={r.get('z_score', 0):.2f} reasons={r.get('alert_reasons', [])}")
        return

    parser.print_help()


# ===========================================================================
# 75-test smoke suite
# ===========================================================================

def _run_tests():
    import traceback
    passed = 0
    failed = 0
    errors = []

    def ok(name):
        nonlocal passed
        passed += 1
        print(f"  [PASS] {name}")

    def fail(name, reason):
        nonlocal failed
        failed += 1
        errors.append((name, reason))
        print(f"  [FAIL] {name}: {reason}")

    def run(name, fn):
        try:
            fn()
            ok(name)
        except Exception as e:
            fail(name, f"{type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"  Fracttalix Sentinel v12.0 — 75-test Smoke Suite")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # T01 — T05: SentinelConfig
    # ------------------------------------------------------------------
    def t01():
        cfg = SentinelConfig()
        assert cfg.alpha == 0.1
        assert cfg.warmup_periods == 30
    run("T01 SentinelConfig defaults", t01)

    def t02():
        cfg = SentinelConfig(alpha=0.2, multiplier=2.5)
        assert cfg.alpha == 0.2
        assert cfg.multiplier == 2.5
    run("T02 SentinelConfig custom values", t02)

    def t03():
        try:
            SentinelConfig(alpha=0.0)
            assert False, "should raise"
        except ValueError:
            pass
    run("T03 SentinelConfig validation: alpha=0 raises", t03)

    def t04():
        cfg1 = SentinelConfig(alpha=0.1)
        cfg2 = dataclasses.replace(cfg1, alpha=0.2)
        assert cfg1.alpha == 0.1
        assert cfg2.alpha == 0.2
    run("T04 SentinelConfig replace returns new instance", t04)

    def t05():
        f = SentinelConfig.fast()
        p = SentinelConfig.production()
        s = SentinelConfig.sensitive()
        r = SentinelConfig.realtime()
        assert f.alpha > p.alpha
        assert s.alpha < p.alpha
        assert r.quantile_threshold_mode
    run("T05 SentinelConfig factory presets", t05)

    # ------------------------------------------------------------------
    # T06 — T10: WindowBank
    # ------------------------------------------------------------------
    def t06():
        bank = WindowBank()
        bank.register("a", 5)
        bank.register("b", 10)
        for i in range(7):
            bank.append(float(i))
        assert len(bank.get("a")) == 5
        assert len(bank.get("b")) == 7
    run("T06 WindowBank independent maxlens", t06)

    def t07():
        bank = WindowBank()
        bank.register("x", 4)
        for i in range(3):
            bank.append(float(i))
        bank.reset()
        assert len(bank.get("x")) == 0
    run("T07 WindowBank reset clears windows", t07)

    def t08():
        bank = WindowBank()
        bank.register("s", 10)
        for i in range(5):
            bank.append(float(i))
        sd = bank.state_dict()
        bank2 = WindowBank()
        bank2.register("s", 10)
        bank2.load_state(sd)
        assert list(bank2.get("s")) == list(bank.get("s"))
    run("T08 WindowBank state dict round-trip", t08)

    def t09():
        bank = WindowBank()
        bank.register("a", 3)
        bank.register("b", 3)
        bank.append(1.0)
        assert list(bank.get("a")) == [1.0]
        assert list(bank.get("b")) == [1.0]
    run("T09 WindowBank append fans to all", t09)

    def t10():
        bank = WindowBank()
        bank.register("q", 5)
        bank.register("q", 5)  # no-op
        assert len(bank._windows) == 1
    run("T10 WindowBank double-register is no-op", t10)

    # ------------------------------------------------------------------
    # T11 — T15: SentinelDetector basic
    # ------------------------------------------------------------------
    def t11():
        det = SentinelDetector()
        r = det.update_and_check(1.0)
        assert "alert" in r
        assert "warmup" in r
        assert r["warmup"] is True
    run("T11 SentinelDetector: first result has warmup=True", t11)

    def t12():
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for i in range(5):
            det.update_and_check(1.0)
        r = det.update_and_check(1.0)
        assert r["warmup"] is False
    run("T12 SentinelDetector: warmup exits after warmup_periods", t12)

    def t13():
        det = SentinelDetector(SentinelConfig(warmup_periods=5, multiplier=2.0))
        for _ in range(5):
            det.update_and_check(0.0)
        r = det.update_and_check(100.0)
        assert r["alert"] is True
    run("T13 SentinelDetector: large spike triggers alert", t13)

    def t14():
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for _ in range(5):
            det.update_and_check(0.0)
        for _ in range(20):
            r = det.update_and_check(0.0)
        assert r["alert"] is False
    run("T14 SentinelDetector: stable data no alert", t14)

    def t15():
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for i in range(10):
            det.update_and_check(float(i))
        assert det._n == 10
        assert len(det._history) == 10
    run("T15 SentinelDetector: step counter and history", t15)

    # ------------------------------------------------------------------
    # T16 — T20: Pipeline steps
    # ------------------------------------------------------------------
    def t16():
        cfg = SentinelConfig(warmup_periods=5)
        det = SentinelDetector(cfg)
        for _ in range(5):
            det.update_and_check(0.0)
        r = det.update_and_check(100.0)
        assert "cusum_hi" in r
        assert r["cusum_hi"] > 0
    run("T16 CUSUMStep: cusum_hi positive after spike", t16)

    def t17():
        cfg = SentinelConfig(warmup_periods=5)
        det = SentinelDetector(cfg)
        for _ in range(40):
            det.update_and_check(0.1)
        r = det.update_and_check(0.1)
        assert "ews_score" in r
        assert "ews_regime" in r
    run("T17 EWSStep: ews keys present", t17)

    def t18():
        cfg = SentinelConfig(warmup_periods=5)
        det = SentinelDetector(cfg)
        for _ in range(70):
            det.update_and_check(1.0)
        r = det.update_and_check(1.0)
        assert "pe" in r
        assert 0.0 <= r["pe"] <= 1.0
    run("T18 PEStep: pe in [0,1]", t18)

    def t19():
        cfg = SentinelConfig(warmup_periods=5)
        det = SentinelDetector(cfg)
        for _ in range(70):
            det.update_and_check(1.0)
        r = det.update_and_check(1.0)
        assert "rpi" in r
        assert "rfi" in r
        assert "ssi" in r
        assert "rsi" in r  # backward compat alias (T0-05)
    run("T19 RPIStep/RFIStep/SSIStep: keys present + rsi alias (T0-05)", t19)

    def t20():
        cfg = SentinelConfig(warmup_periods=5)
        det = SentinelDetector(cfg)
        for _ in range(5):
            det.update_and_check(0.0)
        r = det.update_and_check(100.0)
        assert isinstance(r.get("alert_reasons"), list)
        assert len(r["alert_reasons"]) > 0
    run("T20 AlertReasonsStep: reasons populated on alert", t20)

    # ------------------------------------------------------------------
    # T21 — T25: State persistence
    # ------------------------------------------------------------------
    def t21():
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for i in range(10):
            det.update_and_check(float(i))
        state = det.save_state()
        assert isinstance(state, str)
        sd = json.loads(state)
        assert sd["version"] == __version__
        assert sd["n"] == 10
    run("T21 save_state: JSON with version and n", t21)

    def t22():
        cfg = SentinelConfig(warmup_periods=5)
        det1 = SentinelDetector(cfg)
        for i in range(10):
            det1.update_and_check(float(i))
        state = det1.save_state()
        det2 = SentinelDetector(cfg)
        det2.load_state(state)
        assert det2._n == det1._n
    run("T22 load_state: restores step counter", t22)

    def t23():
        cfg = SentinelConfig(warmup_periods=5)
        det = SentinelDetector(cfg)
        for i in range(10):
            det.update_and_check(float(i))
        det.reset()
        assert det._n == 0
        assert len(det._history) == 0
    run("T23 reset: clears counter and history", t23)

    def t24():
        cfg = SentinelConfig(warmup_periods=5)
        det = SentinelDetector(cfg)
        for i in range(15):
            det.update_and_check(float(i))
        r1 = det.update_and_check(0.0)
        state = det.save_state()
        det2 = SentinelDetector(cfg)
        det2.load_state(state)
        r2 = det2.update_and_check(0.0)
        assert abs(r2.get("ewma", 0) - r1.get("ewma", 0)) < 1.0
    run("T24 save/load: EWMA approximately preserved", t24)

    def t25():
        cfg = SentinelConfig(warmup_periods=5)
        det = SentinelDetector(cfg)
        for i in range(5):
            det.update_and_check(float(i))
        state = det.save_state()
        sd = json.loads(state)
        assert "bank" in sd
        assert "steps" in sd
    run("T25 save_state: bank and steps keys present", t25)

    # ------------------------------------------------------------------
    # T26 — T30: Backward compat / alias
    # ------------------------------------------------------------------
    def t26():
        det = Detector_7_10()
        r = det.update_and_check(1.0)
        assert "alert" in r
    run("T26 Detector_7_10 alias works", t26)

    def t27():
        det = SentinelDetector(alpha=0.2, multiplier=2.5, warmup_periods=10)
        assert det.config.alpha == 0.2
        assert det.config.multiplier == 2.5
    run("T27 v7.x flat kwargs via legacy mapper", t27)

    def t28():
        det = SentinelDetector(rsi_window=32)  # v7.10 arg → rpi_window
        assert det.config.rpi_window == 32
    run("T28 rsi_window legacy kwarg mapped to rpi_window", t28)

    def t29():
        cfg = SentinelConfig.fast()
        det = SentinelDetector(config=cfg)
        assert det.config.alpha == SentinelConfig.fast().alpha
    run("T29 SentinelDetector accepts config object", t29)

    def t30():
        # fit() warms up and returns self
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        ret = det.fit([float(i) for i in range(10)])
        assert ret is det
        assert det._n == 10
    run("T30 fit() returns self and advances counter", t30)

    # ------------------------------------------------------------------
    # T31 — T35: MultiStreamSentinel
    # ------------------------------------------------------------------
    def t31():
        mss = MultiStreamSentinel()
        r = mss.update("s1", 1.0)
        assert "alert" in r
    run("T31 MultiStreamSentinel basic update", t31)

    def t32():
        mss = MultiStreamSentinel()
        mss.update("s1", 1.0)
        mss.update("s2", 2.0)
        assert set(mss.list_streams()) == {"s1", "s2"}
    run("T32 MultiStreamSentinel: two streams independent", t32)

    def t33():
        mss = MultiStreamSentinel()
        mss.update("s1", 1.0)
        ok = mss.reset_stream("s1")
        assert ok
        det = mss.get_detector("s1")
        assert det._n == 0
    run("T33 MultiStreamSentinel reset_stream", t33)

    def t34():
        mss = MultiStreamSentinel()
        mss.update("s1", 1.0)
        ok = mss.delete_stream("s1")
        assert ok
        assert "s1" not in mss.list_streams()
    run("T34 MultiStreamSentinel delete_stream", t34)

    def t35():
        mss = MultiStreamSentinel()
        for i in range(10):
            mss.update("x", float(i))
        st = mss.status("x")
        assert st["n"] == 10
    run("T35 MultiStreamSentinel status", t35)

    # ------------------------------------------------------------------
    # T36 — T38: Benchmark
    # ------------------------------------------------------------------
    def t36():
        bench = SentinelBenchmark(n=200, seed=0)
        data, labels = bench.generate("point")
        assert len(data) == 200
        assert any(labels)
    run("T36 Benchmark generate point archetype", t36)

    def t37():
        bench = SentinelBenchmark(n=200, seed=0)
        r = bench.evaluate("point")
        assert "f1" in r
        assert "auprc" in r
        assert "vus_pr" in r
        assert r["f1"] >= 0.0
    run("T37 Benchmark evaluate point", t37)

    def t38():
        bench = SentinelBenchmark(n=150, seed=1)
        results = bench.run_suite(verbose=False)
        assert set(results.keys()) == set(SentinelBenchmark.ARCHETYPES)
    run("T38 Benchmark run_suite all archetypes", t38)

    # ------------------------------------------------------------------
    # T39 — T40: Async + registry
    # ------------------------------------------------------------------
    def t39():
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        r = asyncio.get_event_loop().run_until_complete(det.aupdate(1.0))
        assert "alert" in r
    run("T39 aupdate async wrapper", t39)

    def t40():
        assert "CoreEWMAStep" in _STEP_REGISTRY
        assert "EWSStep" in _STEP_REGISTRY
        assert "SSIStep" in _STEP_REGISTRY
        assert "AlertReasonsStep" in _STEP_REGISTRY
        assert len(_STEP_REGISTRY) >= 19
    run("T40 _STEP_REGISTRY contains all steps", t40)

    # ------------------------------------------------------------------
    # T41 — T45: Channel 2 Frequency Decomposition
    # ------------------------------------------------------------------
    def t41():
        cfg = SentinelConfig(warmup_periods=5, min_window_for_fft=8)
        det = SentinelDetector(cfg)
        for i in range(50):
            r = det.update_and_check(math.sin(2 * math.pi * i / 10))
        assert r["frequency_bands"] is not None
        fb = r["frequency_bands"]
        assert isinstance(fb, FrequencyBands)
        assert hasattr(fb, "ultra_low_power")
        assert hasattr(fb, "high_power")
        assert hasattr(fb, "timestamp")
    run("T41 FrequencyDecomposition produces FrequencyBands instance", t41)

    def t42():
        cfg = SentinelConfig(warmup_periods=5, min_window_for_fft=8)
        det = SentinelDetector(cfg)
        for i in range(50):
            r = det.update_and_check(float(i % 10))
        fb = r.get("frequency_bands")
        assert fb is not None
        assert fb.ultra_low_power >= 0.0
        assert fb.low_power >= 0.0
        assert fb.mid_power >= 0.0
        assert fb.high_power >= 0.0
        assert fb.ultra_high_power >= 0.0
    run("T42 Band powers are non-negative", t42)

    def t43():
        cfg = SentinelConfig(warmup_periods=5, min_window_for_fft=8)
        det = SentinelDetector(cfg)
        for i in range(50):
            r = det.update_and_check(math.sin(2 * math.pi * i / 7))
        fb = r.get("frequency_bands")
        assert fb is not None
        assert -math.pi - 0.01 <= fb.ultra_low_phase <= math.pi + 0.01
        assert -math.pi - 0.01 <= fb.mid_phase <= math.pi + 0.01
        assert -math.pi - 0.01 <= fb.ultra_high_phase <= math.pi + 0.01
    run("T43 Band phases are bounded in [-pi, pi]", t43)

    def t44():
        # min_window_for_fft larger than bank maxlen -> always None
        cfg = SentinelConfig(warmup_periods=2, min_window_for_fft=200)
        det = SentinelDetector(cfg)
        for i in range(50):
            r = det.update_and_check(float(i))
        assert r["frequency_bands"] is None
    run("T44 Decomposition returns None below min_window_for_fft", t44)

    def t45():
        cfg = SentinelConfig(warmup_periods=5, enable_frequency_decomposition=False)
        det = SentinelDetector(cfg)
        for i in range(50):
            r = det.update_and_check(float(i % 10))
        assert r["frequency_bands"] is None
    run("T45 Decomposition disabled by config returns None", t45)

    # ------------------------------------------------------------------
    # T46 — T52: Cross-Frequency Coupling
    # ------------------------------------------------------------------
    def t46():
        cfg = SentinelConfig(warmup_periods=5, min_window_for_fft=8,
                             coupling_trend_window=5)
        det = SentinelDetector(cfg)
        for i in range(80):
            r = det.update_and_check(math.sin(2 * math.pi * i / 8))
        assert r["coupling_matrix"] is not None
        cm = r["coupling_matrix"]
        assert isinstance(cm, CouplingMatrix)
        assert hasattr(cm, "composite_coupling_score")
        assert hasattr(cm, "coupling_trend")
    run("T46 CouplingMatrix computed for long-running signal", t46)

    def t47():
        # Direct unit test of CrossFrequencyCouplingStep
        cfg = SentinelConfig(warmup_periods=1, coupling_degradation_threshold=0.99,
                             coupling_trend_window=3, min_window_for_fft=8)
        step = CrossFrequencyCouplingStep(cfg)
        # Build a fake bands_history deque
        fake_history = deque(maxlen=20)
        for i in range(5):
            fake_history.append(FrequencyBands(
                ultra_low_power=0.1, low_power=0.1, mid_power=0.1,
                high_power=0.1, ultra_high_power=0.1,
                ultra_low_phase=float(i), low_phase=float(i),
                mid_phase=float(i), high_phase=float(i), ultra_high_phase=float(i),
                timestamp=i,
            ))
        bank = WindowBank()
        bank.register("scalar", 64)
        for i in range(10):
            bank.append(float(i))
        ctx = StepContext(value=1.0, step=5, config=cfg, bank=bank,
                         scratch={"_bands_history": fake_history,
                                  "v9_active_alerts": []})
        step.update(ctx)
        # composite < 0.99 threshold -> alert
        assert ctx.scratch.get("coupling_degradation_active") is True
        alerts = ctx.scratch.get("v9_active_alerts", [])
        assert any(a.alert_type == AlertType.COUPLING_DEGRADATION for a in alerts)
    run("T47 Coupling degradation alert generated at threshold", t47)

    def t48():
        cfg = SentinelConfig(warmup_periods=1, coupling_degradation_threshold=0.99,
                             coupling_trend_window=3, min_window_for_fft=8)
        step = CrossFrequencyCouplingStep(cfg)
        fake_history = deque(maxlen=20)
        for i in range(5):
            fake_history.append(FrequencyBands(
                ultra_low_power=0.1, low_power=0.1, mid_power=0.1,
                high_power=0.1, ultra_high_power=0.1,
                ultra_low_phase=0.0, low_phase=0.0, mid_phase=0.0,
                high_phase=0.0, ultra_high_phase=0.0, timestamp=i,
            ))
        bank = WindowBank()
        bank.register("scalar", 64)
        for i in range(5):
            bank.append(float(i))
        ctx = StepContext(value=1.0, step=5, config=cfg, bank=bank,
                         scratch={"_bands_history": fake_history,
                                  "v9_active_alerts": []})
        step.update(ctx)
        alerts = ctx.scratch.get("v9_active_alerts", [])
        coupling_alerts = [a for a in alerts
                           if a.alert_type == AlertType.COUPLING_DEGRADATION]
        assert coupling_alerts
        assert coupling_alerts[0].severity == AlertSeverity.WARNING
    run("T48 Coupling degradation alert has WARNING severity", t48)

    def t49():
        cfg = SentinelConfig(warmup_periods=5, min_window_for_fft=8,
                             coupling_trend_window=5)
        det = SentinelDetector(cfg)
        for i in range(80):
            r = det.update_and_check(math.sin(2 * math.pi * i / 8))
        cm = r.get("coupling_matrix")
        assert cm is not None
        assert isinstance(cm.coupling_trend, float)
    run("T49 Coupling trend is a float in CouplingMatrix", t49)

    def t50():
        cfg = SentinelConfig(warmup_periods=5, enable_coupling_detection=False)
        det = SentinelDetector(cfg)
        for i in range(80):
            r = det.update_and_check(float(i % 10))
        assert r["coupling_matrix"] is None
        assert r.get("coupling_degradation_active") is False
    run("T50 Coupling detection disabled by config", t50)

    def t51():
        # BandAnomalyStep unit test
        cfg = SentinelConfig(warmup_periods=1, multiplier=0.5)
        step = BandAnomalyStep(cfg)
        # Warm up with normal band values
        bank = WindowBank()
        bank.register("scalar", 64)
        for i in range(5):
            bank.append(1.0)
        for i in range(5):
            fb_normal = FrequencyBands(
                ultra_low_power=1.0, low_power=1.0, mid_power=1.0,
                high_power=1.0, ultra_high_power=1.0,
                ultra_low_phase=0.0, low_phase=0.0, mid_phase=0.0,
                high_phase=0.0, ultra_high_phase=0.0, timestamp=i,
            )
            ctx = StepContext(value=1.0, step=i + 2, config=cfg, bank=bank,
                              scratch={"frequency_bands": fb_normal,
                                       "v9_active_alerts": []})
            step.update(ctx)
        # Now inject a spike in mid band only
        fb_spike = FrequencyBands(
            ultra_low_power=1.0, low_power=1.0, mid_power=1000.0,
            high_power=1.0, ultra_high_power=1.0,
            ultra_low_phase=0.0, low_phase=0.0, mid_phase=0.0,
            high_phase=0.0, ultra_high_phase=0.0, timestamp=10,
        )
        ctx2 = StepContext(value=1000.0, step=10, config=cfg, bank=bank,
                           scratch={"frequency_bands": fb_spike,
                                    "v9_active_alerts": [], "alert": False})
        step.update(ctx2)
        anomalies = ctx2.scratch.get("band_anomalies", {})
        assert "mid" in anomalies
    run("T51 BandAnomalyStep detects per-band spike", t51)

    def t52():
        # Coupling alert clears when coupling recovers
        cfg = SentinelConfig(warmup_periods=1, coupling_degradation_threshold=0.99,
                             coupling_trend_window=3)
        step = CrossFrequencyCouplingStep(cfg)
        step._coupling_degradation_active = True

        fake_history = deque(maxlen=20)
        for i in range(5):
            fake_history.append(FrequencyBands(
                ultra_low_power=0.1, low_power=0.1, mid_power=0.1,
                high_power=0.1, ultra_high_power=0.1,
                ultra_low_phase=0.0, low_phase=0.0, mid_phase=0.0,
                high_phase=0.0, ultra_high_phase=0.0, timestamp=i,
            ))
        bank = WindowBank()
        bank.register("scalar", 64)
        for i in range(5):
            bank.append(float(i))
        # Use very low threshold so coupling is now "healthy"
        cfg2 = SentinelConfig(warmup_periods=1, coupling_degradation_threshold=0.0,
                              coupling_trend_window=3)
        step2 = CrossFrequencyCouplingStep(cfg2)
        ctx = StepContext(value=1.0, step=5, config=cfg2, bank=bank,
                          scratch={"_bands_history": fake_history,
                                   "v9_active_alerts": []})
        step2.update(ctx)
        assert ctx.scratch.get("coupling_degradation_active") is False
    run("T52 Coupling alert cleared when coupling recovers", t52)

    # ------------------------------------------------------------------
    # T53 — T57: Structural-Rhythmic Coherence
    # ------------------------------------------------------------------
    def t53():
        cfg = SentinelConfig(warmup_periods=5, min_window_for_fft=8,
                             coupling_trend_window=5, coherence_window=10)
        det = SentinelDetector(cfg)
        for i in range(100):
            r = det.update_and_check(math.sin(2 * math.pi * i / 8))
        # Coherence may or may not be computed depending on window fill
        cc = r.get("channel_coherence")
        if cc is not None:
            assert isinstance(cc, ChannelCoherence)
            assert hasattr(cc, "coherence_score")
    run("T53 ChannelCoherence produces ChannelCoherence when data sufficient", t53)

    def t54():
        cfg = SentinelConfig(warmup_periods=5, coherence_window=5)
        det = SentinelDetector(cfg)
        for i in range(80):
            r = det.update_and_check(1.0)  # stable — should be coherent
        cc = r.get("channel_coherence")
        if cc is not None:
            assert 0.0 <= cc.coherence_score <= 1.0
    run("T54 Coherence score is in [0.0, 1.0]", t54)

    def t55():
        # Direct unit test of ChannelCoherenceStep
        cfg = SentinelConfig(warmup_periods=1, coherence_threshold=0.99,
                             coherence_window=4)
        step = ChannelCoherenceStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 64)
        for i in range(10):
            bank.append(float(i))
        # Build fake structural history — all same value (stable)
        fake_struct = deque(maxlen=20)
        for i in range(10):
            fake_struct.append(StructuralSnapshot(
                mean=0.0, variance=0.0, skewness=0.0, kurtosis=0.0,
                autocorrelation_lag1=0.0, autocorrelation_lag2=0.0,
                stationarity_score=1.0, timestamp=i,
            ))
        # Build rhythmic history with large jumps (diverging from structural)
        for i in range(10):
            step._rhythmic_history.append(float(i * 100))  # big rhythmic changes
        ctx = StepContext(value=1.0, step=10, config=cfg, bank=bank,
                          scratch={"ewma": 1.0,
                                   "_structural_snapshot_history": fake_struct,
                                   "v9_active_alerts": []})
        step.update(ctx)
        alerts = ctx.scratch.get("v9_active_alerts", [])
        assert any(a.alert_type == AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING
                   for a in alerts)
    run("T55 SR decoupling alert generated when structural and rhythmic diverge", t55)

    def t56():
        # SR decoupling alert has ALERT severity
        cfg = SentinelConfig(warmup_periods=1, coherence_threshold=0.99,
                             coherence_window=4)
        step = ChannelCoherenceStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 64)
        for i in range(10):
            bank.append(float(i))
        fake_struct = deque(maxlen=20)
        for i in range(10):
            fake_struct.append(StructuralSnapshot(
                mean=0.0, variance=0.0, skewness=0.0, kurtosis=0.0,
                autocorrelation_lag1=0.0, autocorrelation_lag2=0.0,
                stationarity_score=1.0, timestamp=i,
            ))
        for i in range(10):
            step._rhythmic_history.append(float(i * 100))
        ctx = StepContext(value=1.0, step=10, config=cfg, bank=bank,
                          scratch={"ewma": 1.0,
                                   "_structural_snapshot_history": fake_struct,
                                   "v9_active_alerts": []})
        step.update(ctx)
        alerts = ctx.scratch.get("v9_active_alerts", [])
        sr_alerts = [a for a in alerts
                     if a.alert_type == AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING]
        assert sr_alerts
        assert sr_alerts[0].severity == AlertSeverity.ALERT
    run("T56 SR decoupling alert has ALERT severity", t56)

    def t57():
        cfg = SentinelConfig(warmup_periods=5, enable_channel_coherence=False)
        det = SentinelDetector(cfg)
        for i in range(80):
            r = det.update_and_check(float(i % 10))
        assert r["channel_coherence"] is None
        assert r.get("sr_decoupling_active") is False
    run("T57 Coherence detection disabled by config", t57)

    # ------------------------------------------------------------------
    # T58 — T62: Cascade Precursor
    # ------------------------------------------------------------------
    def t58():
        # All conditions met -> cascade precursor fires
        cfg = SentinelConfig(warmup_periods=1, cascade_ews_threshold=2,
                             ews_threshold=0.5)
        step = CascadePrecursorStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 64)
        ctx = StepContext(value=1.0, step=5, config=cfg, bank=bank,
                          scratch={
                              "coupling_degradation_active": True,
                              "sr_decoupling_active": True,
                              "ews_score": 0.8,
                              "anomaly_score": 0.9,
                              "rfi": 0.7,
                              "v9_active_alerts": [],
                          })
        step.update(ctx)
        assert ctx.scratch.get("cascade_precursor_active") is True
        alerts = ctx.scratch.get("v9_active_alerts", [])
        assert any(a.alert_type == AlertType.CASCADE_PRECURSOR for a in alerts)
    run("T58 CascadePrecursor fires when all conditions met", t58)

    def t59():
        # Coupling active alone — no cascade
        cfg = SentinelConfig(warmup_periods=1, cascade_ews_threshold=2,
                             ews_threshold=0.5)
        step = CascadePrecursorStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 64)
        ctx = StepContext(value=1.0, step=5, config=cfg, bank=bank,
                          scratch={
                              "coupling_degradation_active": True,
                              "sr_decoupling_active": False,
                              "ews_score": 0.8,
                              "anomaly_score": 0.9,
                              "rfi": 0.7,
                              "v9_active_alerts": [],
                          })
        step.update(ctx)
        assert ctx.scratch.get("cascade_precursor_active") is False
    run("T59 CascadePrecursor not triggered by coupling alone", t59)

    def t60():
        # Decoupling active alone — no cascade
        cfg = SentinelConfig(warmup_periods=1, cascade_ews_threshold=2,
                             ews_threshold=0.5)
        step = CascadePrecursorStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 64)
        ctx = StepContext(value=1.0, step=5, config=cfg, bank=bank,
                          scratch={
                              "coupling_degradation_active": False,
                              "sr_decoupling_active": True,
                              "ews_score": 0.8,
                              "anomaly_score": 0.9,
                              "rfi": 0.7,
                              "v9_active_alerts": [],
                          })
        step.update(ctx)
        assert ctx.scratch.get("cascade_precursor_active") is False
    run("T60 CascadePrecursor not triggered by decoupling alone", t60)

    def t61():
        # Cascade alert has CRITICAL severity
        cfg = SentinelConfig(warmup_periods=1, cascade_ews_threshold=2,
                             ews_threshold=0.5)
        step = CascadePrecursorStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 64)
        ctx = StepContext(value=1.0, step=5, config=cfg, bank=bank,
                          scratch={
                              "coupling_degradation_active": True,
                              "sr_decoupling_active": True,
                              "ews_score": 0.8,
                              "anomaly_score": 0.9,
                              "rfi": 0.7,
                              "v9_active_alerts": [],
                          })
        step.update(ctx)
        alerts = ctx.scratch.get("v9_active_alerts", [])
        cascade_alerts = [a for a in alerts
                          if a.alert_type == AlertType.CASCADE_PRECURSOR]
        assert cascade_alerts
        assert cascade_alerts[0].severity == AlertSeverity.CRITICAL
    run("T61 CascadePrecursor has CRITICAL severity", t61)

    def t62():
        # cascade_precursor_active in SentinelResult
        cfg = SentinelConfig(warmup_periods=1, enable_cascade_detection=False)
        det = SentinelDetector(cfg)
        r = det.update_and_check(1.0)
        assert "cascade_precursor_active" in r
        assert isinstance(r["cascade_precursor_active"], bool)
        assert isinstance(r, SentinelResult)
    run("T62 cascade_precursor_active key in SentinelResult", t62)

    # ------------------------------------------------------------------
    # T63 — T65: Degradation Sequence + SentinelResult methods
    # ------------------------------------------------------------------
    def t63():
        cfg = SentinelConfig(warmup_periods=1)
        step = DegradationSequenceStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 64)
        # Inject a fake alert
        fake_alert = Alert(
            alert_type=AlertType.COUPLING_DEGRADATION,
            severity=AlertSeverity.WARNING,
            score=0.2,
            message="test",
        )
        ctx = StepContext(value=1.0, step=10, config=cfg, bank=bank,
                          scratch={"v9_active_alerts": [fake_alert]})
        step.update(ctx)
        ds = ctx.scratch.get("degradation_sequence")
        assert ds is not None
        assert isinstance(ds, DegradationSequence)
        assert ds.first_channel_anomaly == AlertType.COUPLING_DEGRADATION.value
        assert ds.first_anomaly_timestamp == 10
    run("T63 DegradationSequence logs first channel anomaly", t63)

    def t64():
        # Sequence narrative is human-readable string
        cfg = SentinelConfig(warmup_periods=1)
        step = DegradationSequenceStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 64)
        fake_alert = Alert(
            alert_type=AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING,
            severity=AlertSeverity.ALERT,
            score=0.1,
            message="test",
        )
        ctx = StepContext(value=1.0, step=5, config=cfg, bank=bank,
                          scratch={"v9_active_alerts": [fake_alert]})
        step.update(ctx)
        ds = ctx.scratch.get("degradation_sequence")
        assert ds is not None
        assert isinstance(ds.sequence_pattern, str)
        assert len(ds.sequence_pattern) > 0
        assert "decoupling" in ds.sequence_pattern.lower()
    run("T64 DegradationSequence narrative is human-readable string", t64)

    def t65():
        # SentinelResult convenience methods
        cfg = SentinelConfig(warmup_periods=5, min_window_for_fft=8)
        det = SentinelDetector(cfg)
        for i in range(50):
            r = det.update_and_check(math.sin(2 * math.pi * i / 8))
        assert isinstance(r, SentinelResult)
        assert isinstance(r.is_cascade_precursor(), bool)
        channel_status = r.get_channel_status()
        assert isinstance(channel_status, dict)
        assert "structural" in channel_status
        assert "rhythmic_composite" in channel_status
        narrative = r.get_degradation_narrative()
        assert isinstance(narrative, str)
        carrier = r.get_primary_carrier_wave()
        assert isinstance(carrier, str)
    run("T65 SentinelResult convenience methods all return correct types", t65)

    # ------------------------------------------------------------------
    # T66 — T75: Stress Tests
    # ------------------------------------------------------------------
    def t66():
        # Long stream: 10,000 observations without error
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for i in range(10_000):
            r = det.update_and_check(math.sin(i * 0.01) + (i % 100) * 0.001)
        assert det._n == 10_000
        assert "alert" in r
    run("T66 Stress: 10k observations no crash", t66)

    def t67():
        # Large spike (within float range but extreme for detector)
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for _ in range(10):
            det.update_and_check(1.0)
        r = det.update_and_check(1e15)
        assert "alert" in r
        assert r["alert"] is True
    run("T67 Stress: large positive spike (1e15)", t67)

    def t68():
        # Large negative spike
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for _ in range(10):
            det.update_and_check(0.0)
        r = det.update_and_check(-1e15)
        assert "alert" in r
        assert r["alert"] is True
    run("T68 Stress: large negative spike (-1e15)", t68)

    def t69():
        # Zero-variance stream (all identical values) — no crash
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for _ in range(200):
            r = det.update_and_check(42.0)
        assert det._n == 200
        assert "alert" in r
    run("T69 Stress: zero-variance stream completes (200 identical values)", t69)

    def t70():
        # Alternating extreme values
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for i in range(100):
            v = 1e6 if i % 2 == 0 else -1e6
            r = det.update_and_check(v)
        assert det._n == 100
        assert "alert" in r
    run("T70 Stress: alternating extreme values", t70)

    def t71():
        # Very small values near zero
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for i in range(100):
            r = det.update_and_check(1e-300 * (i % 3))
        assert det._n == 100
        assert "alert" in r
    run("T71 Stress: very small values near zero (1e-300)", t71)

    def t72():
        # Rapid regime changes
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for block in range(10):
            baseline = block * 100.0
            for _ in range(20):
                r = det.update_and_check(baseline)
        assert det._n == 200
        assert "alert" in r
    run("T72 Stress: rapid regime changes (10 blocks)", t72)

    def t73():
        # Save/load round-trip after long stream
        det = SentinelDetector(SentinelConfig(warmup_periods=5))
        for i in range(500):
            det.update_and_check(math.sin(i * 0.1))
        state = det.save_state()
        det2 = SentinelDetector(SentinelConfig(warmup_periods=5))
        det2.load_state(state)
        r1 = det.update_and_check(0.5)
        r2 = det2.update_and_check(0.5)
        assert r1["alert"] == r2["alert"]
        assert abs(r1.get("ewma", 0) - r2.get("ewma", 0)) < 0.01
    run("T73 Stress: save/load fidelity after 500 observations", t73)

    def t74():
        # MultiStreamSentinel with many streams
        mss = MultiStreamSentinel()
        for s in range(50):
            for i in range(10):
                mss.update(f"stream_{s}", float(i))
        assert len(mss.list_streams()) == 50
        for s in range(50):
            st = mss.status(f"stream_{s}")
            assert st["n"] == 10
    run("T74 Stress: MultiStreamSentinel 50 streams", t74)

    def t75():
        # All config presets run full pipeline without error
        for factory in [SentinelConfig.fast, SentinelConfig.production,
                        SentinelConfig.sensitive, SentinelConfig.realtime]:
            det = SentinelDetector(factory())
            for i in range(100):
                r = det.update_and_check(math.sin(i * 0.05) * 10)
            assert det._n == 100
            assert "alert" in r
    run("T75 Stress: all config presets complete pipeline", t75)

    # ------------------------------------------------------------------
    total = passed + failed
    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed / {total} total")
    if errors:
        print(f"\n  Failed tests:")
        for name, reason in errors:
            print(f"    {name}: {reason}")
    print(f"{'='*60}\n")
    return failed == 0


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = _run_tests()
        sys.exit(0 if success else 1)
    elif len(sys.argv) == 1:
        # Default: run tests
        success = _run_tests()
        sys.exit(0 if success else 1)
    else:
        _cli_main()
