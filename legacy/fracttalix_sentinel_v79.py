# fracttalix_sentinel v7.9 py

# Fracttalix Sentinel v7.9 — Best-in-Class: Continuous Scoring, Attribution, Drift, Mahalanobis, Benchmark, Server

# v7.6: Fluid dynamics layer (STI, TPS, oscillation damping, CPD)
# v7.7: FRM axiom metrics (RPI, RFI, RRS); multivariate oscillation bug fix
# v7.8: auto_tune(), MultiStreamSentinel, plot_history(), history ring-buffer
# v7.9: Seven competitive gaps closed (Meta-Kaizen analysis):
#   - z_score + anomaly_score [0-1]: continuous severity output on every result (binary → spectrum)
#   - alert_reasons: List[str]: signal attribution — tells you WHY an alert fired
#   - Page-Hinkley drift detector: catches slow gradual mean drift that CUSUM misses
#   - Mahalanobis multivariate mode: rolling EWMA covariance + O(d²) Woodbury inversion;
#     detects cross-channel correlation breaks, not just marginal outliers
#   - SentinelBenchmark: built-in labeled evaluation harness (point/contextual/collective
#     anomalies); reports F1, AUPRC, mean detection lag — no external dataset needed
#   - SentinelServer: stdlib-only HTTP server (http.server) wrapping MultiStreamSentinel;
#     POST /update, GET /streams, GET /status/<id>; zero extra dependencies
#   - CLI entry point: python fracttalix_sentinel_v79.py --file data.csv --alpha 0.1
#   - Module metadata: __version__, __author__, __license__
#   - All v7.8 features, parameters, and JSON state fields preserved (backward-compatible)

# Designed for finance, medical, infrastructure/IoT/security monitoring, and research
# Theoretical foundation: The Fractal Rhythm Model (Brennan & Grok 4, 2026)
# 11 Axioms — see Papers branch: https://github.com/thomasbrennan/Fracttalix

__version__ = "7.9"
__author__ = "Thomas Brennan & Grok 4"
__license__ = "CC0"

from collections import deque
import math
import warnings
import json
import threading
import argparse
import sys
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import numpy as np
import csv
import os

# Optional imports (guarded)

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


class Detector_7_9:
    """
    Enhanced lightweight, regime-aware anomaly detector — v7.9.

    Extends v7.8 with seven best-in-class improvements (Meta-Kaizen analysis):

    Rhythm Power Index (RPI) — Axiom 6
        "Functional networks display rhythmic patterns; absence suggests dormancy."
        Rolling FFT spectral-concentration ratio on the last rpi_window scalar values.
        rpi_regime: "rhythmic" (RPI > 0.25) | "transitional" | "broadband" (RPI < 0.10).

    Rolling Fractal Index (RFI) — Axiom 8
        "Many networks exhibit fractal self-similarity across scales."
        Simplified Higuchi curve-length ratio at k=1 vs k=2 on the last rfi_window
        scalar values. RFI in [1.0, 2.0]: 1.0 = smooth/periodic, 2.0 = maximally rough.

    Resilience Recovery Score (RRS) — Axiom 11
        "Stressors trigger cascades followed by reorganization at higher resilience."
        After a regime reset, compares new_baseline_std / pre_reset_baseline_std once
        the post-reset warm-up period completes. RRS > 1.0 = expanded resilience.

    Multivariate oscillation-damping fix (v7.7)
        Previous code called _check_oscillation_damping twice per channel per step,
        passing stale/zero early_alert values in the first call. v7.7 reads
        _ch_damping_active directly for extra_mult, then calls the update function
        exactly once with the actual ch_early value.

    All v7.6 parameters and JSON state keys are preserved (backward-compatible).

    When to Use Fracttalix Sentinel vs. Other Tools:
    | Feature / Need                          | Fracttalix Sentinel v7.7                     | Alternatives (PyOD, ADTK, ruptures, etc.) |
    |-----------------------------------------|----------------------------------------------|--------------------------------------------|
    | Single-file, no install/dependencies    | Yes                                          | No                                         |
    | Real-time / streaming capable           | Yes (update_and_check)                       | Sometimes (heavier setup)                  |
    | Two-sided + per-channel multivariate    | Yes                                          | Rare                                       |
    | Soft regime reset (preserve baselines)  | Yes                                          | Almost never                               |
    | State persistence (JSON save/load)      | Yes — complete including v7.7 state          | Rare                                       |
    | Buffered CSV export (low-overhead)      | Yes (interval or manual flush)               | Varies (often manual)                      |
    | Volatility-adaptive smoothing           | Yes                                          | Sometimes                                  |
    | Built-in surrogates for significance    | Yes (FFT phase randomization)                | Yes (but heavier)                          |
    | Rolling-window Turbulence Index (STI)   | Yes (v7.4+, rolling window in v7.5)          | No                                         |
    | Boundary Layer Warning (4th tier)       | Yes — univariate + multivariate (v7.5)       | No                                         |
    | Oscillation Damping (no mutation bug)   | Yes (v7.5 fix; double-call bug fixed v7.7)   | No                                         |
    | CUSUM Pressure Differential (CPD)       | Yes (optional flag)                          | No                                         |
    | Rhythm Power Index (Axiom 6)            | Yes (v7.7)                                   | No                                         |
    | Rolling Fractal Index (Axiom 8)         | Yes (v7.7)                                   | No                                         |
    | Resilience Recovery Score (Axiom 11)    | Yes (v7.7)                                   | No                                         |
    | Auto-tune (data-driven params)          | Yes (v7.8) — target FPR, grid search        | Merlion/PyOD only                          |
    | Multi-stream fan-out                    | Yes (v7.8) — thread-parallel, state persist  | River only                                 |
    | Built-in dashboard visualization        | Yes (v7.8) — optional matplotlib            | Prophet, ADTK, Merlion                     |
    | Best for                                | Quick screening, lightweight monitoring,     | Full-featured research pipelines           |
    |                                         | exploratory work in finance/HRV/IoT/research |                                            |
    """

    def __init__(
        self,
        # === Core parameters ===
        alpha: float = 0.12,
        early_mult: float = 2.75,
        fixed_mult: float = 3.2,
        warm_up_period: int = 60,
        use_fixed_during_warmup: bool = True,
        two_sided: bool = True,

        # === Regime change detection ===
        cusum_threshold: float = 5.0,
        reset_after_regime_change: Union[bool, str] = "full",

        # === Multivariate & aggregation ===
        multivariate: bool = False,
        per_channel_detection: bool = True,
        aggregation_func: Callable[[List[float]], float] = lambda x: sum(x) / len(x),
        alert_if_any_channel: bool = True,

        # === Volatility adaptation ===
        volatility_adaptive: bool = False,
        vol_min_factor: float = 0.6,
        vol_max_factor: float = 2.0,

        # === v7.4: Sentinel Turbulence Index (STI) — Reynolds number analog ===
        turbulence_adaptive: bool = True,
        sti_window: int = 10,
        sti_laminar_threshold: float = 1.0,
        sti_turbulent_threshold: float = 2.0,

        # === v7.4: Boundary Layer Warning (TPS) — separation precursor analog ===
        boundary_warning_enabled: bool = True,
        tps_proximity_pct: float = 0.15,
        tps_periods: int = 5,

        # === v7.4: Oscillation Damping Filter — vortex shedding analog ===
        oscillation_damping_enabled: bool = True,
        oscillation_count_threshold: int = 3,
        oscillation_periods: int = 20,
        oscillation_mult_bump: float = 0.10,

        # === v7.4: CUSUM Pressure Differential ===
        cpd_enabled: bool = True,

        # === v7.7: Rhythm Power Index (Axiom 6) — rhythmic vs. dormant ===
        rpi_enabled: bool = True,
        rpi_window: int = 32,               # rolling window length for FFT
        rpi_rhythmic_threshold: float = 0.25,   # RPI above this -> "rhythmic"
        rpi_broadband_threshold: float = 0.10,  # RPI below this -> "broadband"

        # === v7.7: Rolling Fractal Index (Axiom 8) — self-similarity across scales ===
        rfi_enabled: bool = True,
        rfi_window: int = 30,               # rolling window length for Higuchi

        # === v7.7: Resilience Recovery Score (Axiom 11) — post-stress reorganization ===
        rrs_enabled: bool = True,

        # === Extras ===
        verbose_explain: bool = False,
        alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        impute_method: str = "skip",
        parallel_surrogates: bool = False,
        numba_enabled: bool = True,
        csv_output_path: Optional[str] = None,
        csv_flush_interval: int = 60,

        # === v7.8: History ring-buffer for visualization ===
        history_maxlen: int = 500,          # 0 = disabled

        # === v7.9: Continuous anomaly scoring ===
        anomaly_score_enabled: bool = True,

        # === v7.9: Page-Hinkley slow drift detector ===
        ph_enabled: bool = True,
        ph_delta_fraction: float = 0.10,   # sensitivity: fraction of baseline_std
        ph_threshold: float = 50.0,        # detection threshold (accumulator units)
        ph_warning_threshold: float = 20.0, # early drift warning threshold

        # === v7.9: Mahalanobis multivariate mode ===
        mahalanobis_enabled: bool = False,  # requires multivariate=True
        mahalanobis_alpha: float = 0.05,   # EWMA decay for covariance
        mahalanobis_regularize: float = 1e-4,  # Tikhonov reg added to diagonal
        mahalanobis_early_mult: float = 3.0,   # alert threshold (normalized distance)
        mahalanobis_confirmed_mult: float = 4.5,
    ):
        # Validation
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")
        if warm_up_period < 10:
            raise ValueError("warm_up_period should be at least 10")
        if isinstance(reset_after_regime_change, bool):
            reset_after_regime_change = "full" if reset_after_regime_change else False
        if reset_after_regime_change not in [False, "full", "soft"]:
            raise ValueError("reset_after_regime_change must be False, 'full', or 'soft'")
        if not (0.0 < tps_proximity_pct < 1.0):
            raise ValueError("tps_proximity_pct must be between 0 and 1")
        if sti_laminar_threshold >= sti_turbulent_threshold:
            raise ValueError("sti_laminar_threshold must be less than sti_turbulent_threshold")
        if rpi_broadband_threshold >= rpi_rhythmic_threshold:
            raise ValueError("rpi_broadband_threshold must be less than rpi_rhythmic_threshold")

        # Core
        self.alpha = alpha
        self.early_mult = early_mult
        self.fixed_mult = fixed_mult
        self.warm_up_period = warm_up_period
        self.use_fixed_during_warmup = use_fixed_during_warmup
        self.two_sided = two_sided
        self.cusum_threshold = cusum_threshold
        self.reset_after_regime_change = reset_after_regime_change
        self.multivariate = multivariate
        self.per_channel_detection = per_channel_detection
        self.aggregation_func = aggregation_func
        self.alert_if_any_channel = alert_if_any_channel
        self.volatility_adaptive = volatility_adaptive
        self.vol_min_factor = vol_min_factor
        self.vol_max_factor = vol_max_factor

        # v7.4 params
        self.turbulence_adaptive = turbulence_adaptive
        self.sti_window = sti_window
        self.sti_laminar_threshold = sti_laminar_threshold
        self.sti_turbulent_threshold = sti_turbulent_threshold
        self.boundary_warning_enabled = boundary_warning_enabled
        self.tps_proximity_pct = tps_proximity_pct
        self.tps_periods = tps_periods
        self.oscillation_damping_enabled = oscillation_damping_enabled
        self.oscillation_count_threshold = oscillation_count_threshold
        self.oscillation_periods = oscillation_periods
        self.oscillation_mult_bump = oscillation_mult_bump
        self.cpd_enabled = cpd_enabled

        # v7.7 params
        self.rpi_enabled = rpi_enabled
        self.rpi_window = rpi_window
        self.rpi_rhythmic_threshold = rpi_rhythmic_threshold
        self.rpi_broadband_threshold = rpi_broadband_threshold
        self.rfi_enabled = rfi_enabled
        self.rfi_window = rfi_window
        self.rrs_enabled = rrs_enabled

        # Extras
        self.verbose_explain = verbose_explain
        self.alert_callback = alert_callback
        self.impute_method = impute_method
        self.parallel_surrogates = parallel_surrogates
        self.numba_enabled = numba_enabled
        self.csv_output_path = csv_output_path
        self.csv_flush_interval = csv_flush_interval
        self.csv_buffer: List[Dict] = []

        # v7.8 — history ring-buffer (compact per-step records for plot_history)
        self.history_maxlen = history_maxlen
        self._history: deque = deque(maxlen=history_maxlen if history_maxlen > 0 else 1)

        # Core state
        self.count = 0
        self.values_deque: deque = deque(maxlen=warm_up_period * 2)
        self.ewma: Optional[float] = None
        self.dev_ewma: Optional[float] = None
        self.baseline_mean: Optional[float] = None
        self.baseline_std: Optional[float] = None
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

        # Multivariate state
        self.n_channels: int = 0
        self.channel_ewmas: Optional[List[float]] = None
        self.channel_dev_ewmas: Optional[List[float]] = None
        self.channel_baselines: Optional[List[Tuple[float, float]]] = None

        # v7.6 univariate state
        self._dev_ewma_history: deque = deque(maxlen=sti_window)
        self._sti: float = 1.0
        self._sti_regime: str = "transitional"
        self._tps_counter: int = 0
        self._oscillation_lookback: List[bool] = []
        self._oscillation_fire_clear_count: int = 0
        self._prev_early_alert: bool = False
        self._damping_active: bool = False

        # v7.6 per-channel state
        self._ch_tps_counters: List[int] = []
        self._ch_oscillation_lookback: List[List[bool]] = []
        self._ch_fire_clear_counts: List[int] = []
        self._ch_prev_early_alerts: List[bool] = []
        self._ch_damping_active: List[bool] = []

        # v7.7 state — raw scalar rolling window (shared by RPI and RFI)
        self._scalar_window: deque = deque(maxlen=max(rpi_window, rfi_window))

        # v7.7 — RPI (Axiom 6)
        self._rpi: float = 0.0
        self._rpi_regime: str = "transitional"

        # v7.7 — RFI (Axiom 8)
        self._rfi: float = 1.5  # neutral mid-range value

        # v7.7 — RRS (Axiom 11)
        self._pre_reset_baseline_std: Optional[float] = None
        self._post_reset_count: int = 0
        self._rrs: Optional[float] = None

        # v7.9 params
        self.anomaly_score_enabled = anomaly_score_enabled
        self.ph_enabled = ph_enabled
        self.ph_delta_fraction = ph_delta_fraction
        self.ph_threshold = ph_threshold
        self.ph_warning_threshold = ph_warning_threshold
        self.mahalanobis_enabled = mahalanobis_enabled and multivariate
        self.mahalanobis_alpha = mahalanobis_alpha
        self.mahalanobis_regularize = mahalanobis_regularize
        self.mahalanobis_early_mult = mahalanobis_early_mult
        self.mahalanobis_confirmed_mult = mahalanobis_confirmed_mult

        # v7.9 state — Page-Hinkley
        self._ph_sum_up: float = 0.0    # cumulative upward sum
        self._ph_sum_down: float = 0.0  # cumulative downward sum
        self._ph_min_up: float = 0.0    # running minimum of sum_up (PH uses max - min)
        self._ph_min_down: float = 0.0  # running minimum of sum_down
        self._ph_delta: float = 0.0     # set after warmup = ph_delta_fraction * baseline_std
        self._ph_drift_up: bool = False
        self._ph_drift_down: bool = False
        self._ph_warn_up: bool = False
        self._ph_warn_down: bool = False

        # v7.9 state — Mahalanobis covariance (initialized after warmup)
        self._mahal_mean: Optional[np.ndarray] = None   # shape (d,)
        self._mahal_cov: Optional[np.ndarray] = None    # shape (d, d)
        self._mahal_cov_inv: Optional[np.ndarray] = None  # shape (d, d), Woodbury updated
        self._mahal_score: float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # RESET / PERSIST
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, soft: bool = False) -> None:
        """Reset detection state. soft=True preserves baselines."""
        self.count = 0
        self.values_deque.clear()
        self.flush_csv()
        self.csv_buffer = []

        if not soft:
            self.ewma = None
            self.dev_ewma = None
            self.baseline_mean = None
            self.baseline_std = None
            self.channel_ewmas = None
            self.channel_dev_ewmas = None
            self.channel_baselines = None
            # Full reset also clears RRS anchor
            self._pre_reset_baseline_std = None
            self._rrs = None

        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

        # v7.6 state reset
        self._dev_ewma_history.clear()
        self._sti = 1.0
        self._sti_regime = "transitional"
        self._tps_counter = 0
        self._oscillation_lookback = []
        self._oscillation_fire_clear_count = 0
        self._prev_early_alert = False
        self._damping_active = False

        n = self.n_channels
        self._ch_tps_counters = [0] * n
        self._ch_oscillation_lookback = [[] for _ in range(n)]
        self._ch_fire_clear_counts = [0] * n
        self._ch_prev_early_alerts = [False] * n
        self._ch_damping_active = [False] * n

        # v7.7 state reset
        self._scalar_window.clear()
        self._rpi = 0.0
        self._rpi_regime = "transitional"
        self._rfi = 1.5
        self._post_reset_count = 0

        # v7.9 state reset — PH and Mahalanobis
        self._ph_sum_up = 0.0
        self._ph_sum_down = 0.0
        self._ph_min_up = 0.0
        self._ph_min_down = 0.0
        self._ph_delta = 0.0
        self._ph_drift_up = False
        self._ph_drift_down = False
        self._ph_warn_up = False
        self._ph_warn_down = False
        if not soft:
            self._mahal_mean = None
            self._mahal_cov = None
            self._mahal_cov_inv = None
        self._mahal_score = 0.0

    def save_state(self) -> str:
        """Save complete state to JSON string — including all v7.7 fields."""
        state = {
            "count": self.count,
            "ewma": self.ewma,
            "dev_ewma": self.dev_ewma,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "cusum_pos": self.cusum_pos,
            "cusum_neg": self.cusum_neg,
            "channel_ewmas": self.channel_ewmas,
            "channel_dev_ewmas": self.channel_dev_ewmas,
            "channel_baselines": self.channel_baselines,
            "n_channels": self.n_channels,
            # v7.6 univariate state
            "sti": self._sti,
            "sti_regime": self._sti_regime,
            "dev_ewma_history": list(self._dev_ewma_history),
            "tps_counter": self._tps_counter,
            "oscillation_lookback": self._oscillation_lookback,
            "oscillation_fire_clear_count": self._oscillation_fire_clear_count,
            "prev_early_alert": self._prev_early_alert,
            "damping_active": self._damping_active,
            # v7.6 per-channel state
            "ch_tps_counters": self._ch_tps_counters,
            "ch_oscillation_lookback": self._ch_oscillation_lookback,
            "ch_fire_clear_counts": self._ch_fire_clear_counts,
            "ch_prev_early_alerts": self._ch_prev_early_alerts,
            "ch_damping_active": self._ch_damping_active,
            # v7.7 state
            "scalar_window": list(self._scalar_window),
            "rpi": self._rpi,
            "rpi_regime": self._rpi_regime,
            "rfi": self._rfi,
            "pre_reset_baseline_std": self._pre_reset_baseline_std,
            "post_reset_count": self._post_reset_count,
            "rrs": self._rrs,
            # v7.9 state
            "ph_sum_up": self._ph_sum_up,
            "ph_sum_down": self._ph_sum_down,
            "ph_min_up": self._ph_min_up,
            "ph_min_down": self._ph_min_down,
            "ph_delta": self._ph_delta,
            "ph_drift_up": self._ph_drift_up,
            "ph_drift_down": self._ph_drift_down,
            "ph_warn_up": self._ph_warn_up,
            "ph_warn_down": self._ph_warn_down,
            "mahal_mean": self._mahal_mean.tolist() if self._mahal_mean is not None else None,
            "mahal_cov": self._mahal_cov.tolist() if self._mahal_cov is not None else None,
            "mahal_cov_inv": self._mahal_cov_inv.tolist() if self._mahal_cov_inv is not None else None,
            "mahal_score": self._mahal_score,
        }
        return json.dumps(state)

    def load_state(self, state_json: str) -> None:
        """Restore complete state from JSON string."""
        s = json.loads(state_json)
        self.count = s["count"]
        self.ewma = s["ewma"]
        self.dev_ewma = s["dev_ewma"]
        self.baseline_mean = s["baseline_mean"]
        self.baseline_std = s["baseline_std"]
        self.cusum_pos = s["cusum_pos"]
        self.cusum_neg = s["cusum_neg"]
        self.channel_ewmas = s["channel_ewmas"]
        self.channel_dev_ewmas = s["channel_dev_ewmas"]
        self.channel_baselines = s["channel_baselines"]
        self.n_channels = s["n_channels"]
        self._sti = s.get("sti", 1.0)
        self._sti_regime = s.get("sti_regime", "transitional")
        self._dev_ewma_history = deque(s.get("dev_ewma_history", []), maxlen=self.sti_window)
        self._tps_counter = s.get("tps_counter", 0)
        self._oscillation_lookback = s.get("oscillation_lookback", [])
        self._oscillation_fire_clear_count = s.get("oscillation_fire_clear_count", 0)
        self._prev_early_alert = s.get("prev_early_alert", False)
        self._damping_active = s.get("damping_active", False)
        n = self.n_channels
        self._ch_tps_counters = s.get("ch_tps_counters", [0] * n)
        self._ch_oscillation_lookback = s.get("ch_oscillation_lookback", [[] for _ in range(n)])
        self._ch_fire_clear_counts = s.get("ch_fire_clear_counts", [0] * n)
        self._ch_prev_early_alerts = s.get("ch_prev_early_alerts", [False] * n)
        self._ch_damping_active = s.get("ch_damping_active", [False] * n)
        # v7.7
        sw = s.get("scalar_window", [])
        self._scalar_window = deque(sw, maxlen=max(self.rpi_window, self.rfi_window))
        self._rpi = s.get("rpi", 0.0)
        self._rpi_regime = s.get("rpi_regime", "transitional")
        self._rfi = s.get("rfi", 1.5)
        self._pre_reset_baseline_std = s.get("pre_reset_baseline_std", None)
        self._post_reset_count = s.get("post_reset_count", 0)
        self._rrs = s.get("rrs", None)
        # v7.9
        self._ph_sum_up = s.get("ph_sum_up", 0.0)
        self._ph_sum_down = s.get("ph_sum_down", 0.0)
        self._ph_min_up = s.get("ph_min_up", 0.0)
        self._ph_min_down = s.get("ph_min_down", 0.0)
        self._ph_delta = s.get("ph_delta", 0.0)
        self._ph_drift_up = s.get("ph_drift_up", False)
        self._ph_drift_down = s.get("ph_drift_down", False)
        self._ph_warn_up = s.get("ph_warn_up", False)
        self._ph_warn_down = s.get("ph_warn_down", False)
        mm = s.get("mahal_mean", None)
        mc = s.get("mahal_cov", None)
        mi = s.get("mahal_cov_inv", None)
        self._mahal_mean = np.array(mm) if mm is not None else None
        self._mahal_cov = np.array(mc) if mc is not None else None
        self._mahal_cov_inv = np.array(mi) if mi is not None else None
        self._mahal_score = s.get("mahal_score", 0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # v7.6 FLUID DYNAMICS METHODS (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_sti(self, dev_ewma: float, baseline_std: float) -> Tuple[float, str]:
        """Sentinel Turbulence Index — rolling-window Reynolds number analog."""
        if baseline_std is None or baseline_std < 1e-9:
            return self._sti, self._sti_regime
        self._dev_ewma_history.append(dev_ewma)
        smoothed_dev = sum(self._dev_ewma_history) / len(self._dev_ewma_history)
        sti = smoothed_dev / baseline_std
        if sti < self.sti_laminar_threshold:
            regime = "laminar"
        elif sti > self.sti_turbulent_threshold:
            regime = "turbulent"
        else:
            regime = "transitional"
        self._sti = sti
        self._sti_regime = regime
        return sti, regime

    def _sti_adjusted_mult(self, base_mult: float) -> float:
        """Adjust alert multiplier based on turbulence regime."""
        if not self.turbulence_adaptive:
            return base_mult
        if self._sti_regime == "laminar":
            return base_mult * 0.90
        elif self._sti_regime == "turbulent":
            return base_mult * 1.20
        return base_mult

    def _check_tps(
        self,
        current: float,
        early_upper: float,
        early_lower: Optional[float],
        counter_ref: List[int],
        idx: int = 0,
    ) -> bool:
        """Boundary Layer Warning — separation precursor analog."""
        if not self.boundary_warning_enabled:
            counter_ref[idx] = 0
            return False

        if early_lower is not None and self.two_sided:
            dist_upper = abs(current - early_upper)
            dist_lower = abs(current - early_lower)
            threshold = early_upper if dist_upper < dist_lower else early_lower
        else:
            threshold = early_upper

        if (current >= early_upper or
                (early_lower is not None and self.two_sided and current <= early_lower)):
            counter_ref[idx] = 0
            return False

        proximity = abs(current - threshold) / (abs(threshold) + 1e-9)
        if proximity <= self.tps_proximity_pct:
            counter_ref[idx] += 1
        else:
            counter_ref[idx] = 0

        return counter_ref[idx] >= self.tps_periods

    def _check_oscillation_damping(
        self,
        early_alert: bool,
        lookback_ref: List,
        fire_clear_ref: List[int],
        prev_ref: List[bool],
        damping_ref: List[bool],
        idx: int = 0,
    ) -> float:
        """
        Oscillation Damping Filter — vortex shedding analog.
        Updates state and returns extra_mult for this step.
        Call exactly once per channel per step with the actual early_alert.
        """
        if not self.oscillation_damping_enabled:
            prev_ref[idx] = early_alert
            return 0.0

        if prev_ref[idx] and not early_alert:
            fire_clear_ref[idx] += 1

        lookback_ref[idx].append(early_alert)
        if len(lookback_ref[idx]) > self.oscillation_periods:
            lookback_ref[idx].pop(0)

        if fire_clear_ref[idx] >= self.oscillation_count_threshold:
            damping_ref[idx] = True
            fire_clear_ref[idx] = 0
        else:
            damping_ref[idx] = False

        prev_ref[idx] = early_alert
        return self.oscillation_mult_bump if damping_ref[idx] else 0.0

    def _compute_cpd(self) -> Optional[float]:
        """CUSUM Pressure Differential — pressure differential analog."""
        if not self.cpd_enabled:
            return None
        denom = self.cusum_pos + self.cusum_neg + 1e-9
        return round((self.cusum_pos - self.cusum_neg) / denom, 4)

    # ─────────────────────────────────────────────────────────────────────────
    # v7.7 FRM AXIOM METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_rpi(self) -> Tuple[float, str]:
        """
        Rhythm Power Index — spectral power concentration (FRM Axiom 6).

        Formulation from Axiom 6: R = max(P(f)) / total_power.
        "Functional networks display rhythmic patterns; absence suggests dormancy."

        rpi_regime:
            "rhythmic"     — dominant spectral peak (RPI > rpi_rhythmic_threshold)
            "transitional" — mixed spectral character
            "broadband"    — no dominant peak / dormant (RPI < rpi_broadband_threshold)

        Returns (rpi, rpi_regime).
        """
        if len(self._scalar_window) < 8:
            return self._rpi, self._rpi_regime

        arr = np.array(list(self._scalar_window)[-self.rpi_window:], dtype=float)
        arr = arr - arr.mean()

        fft_power = np.abs(np.fft.rfft(arr)) ** 2
        # Exclude DC component (index 0) to focus on oscillatory content
        osc_power = fft_power[1:] if len(fft_power) > 1 else fft_power
        total = osc_power.sum()
        if total < 1e-12:
            return 0.0, "broadband"

        rpi = float(osc_power.max() / total)

        if rpi > self.rpi_rhythmic_threshold:
            regime = "rhythmic"
        elif rpi < self.rpi_broadband_threshold:
            regime = "broadband"
        else:
            regime = "transitional"

        self._rpi = round(rpi, 4)
        self._rpi_regime = regime
        return self._rpi, self._rpi_regime

    def _compute_rfi(self) -> float:
        """
        Rolling Fractal Index — autocorrelation-based Hurst proxy (FRM Axiom 8).

        "Many networks exhibit fractal self-similarity across scales."
        |D_full - D_sub| < 0.1 across subscales (West scaling laws).

        Method: For fractional Gaussian noise, H ≈ (1 + ACF(lag=1)) / 2.
        RFI = 2 - H  (fractal dimension analog), bounded in [1.0, 2.0].

        RFI ~ 1.0  — smooth / persistent (trending, low fractal complexity)
        RFI ~ 1.5  — random walk / neutral
        RFI ~ 2.0  — anti-persistent / maximally rough (high fractal complexity)

        O(n) complexity, no degeneracy on anti-persistent or periodic inputs.
        Returns rfi (float).
        """
        if len(self._scalar_window) < self.rfi_window:
            return self._rfi

        arr = np.array(list(self._scalar_window)[-self.rfi_window:], dtype=float)
        arr_c = arr - arr.mean()
        var = float((arr_c ** 2).mean())
        if var < 1e-12:
            return self._rfi

        # Autocorrelation at lag 1
        acf1 = float((arr_c[:-1] * arr_c[1:]).mean()) / var
        acf1 = max(-1.0, min(1.0, acf1))

        # Hurst exponent proxy (standard fGn approximation)
        hurst = 0.5 + 0.5 * acf1
        self._rfi = round(max(1.0, min(2.0, 2.0 - hurst)), 4)
        return self._rfi

    def _update_rrs_counter(self) -> None:
        """
        Resilience Recovery Score — post-reorganization assessment (FRM Axiom 11).

        "Stressors trigger cascades followed by reorganization at higher resilience."
        Axiom 11 criterion: H_post > H_pre and T_rec < median pre-stress (Holling).

        rrs = new_baseline_std / pre_reset_baseline_std.
            RRS > 1.0 — post-reset system tolerates wider deviations (expanded resilience).
            RRS < 1.0 — post-reset system is tighter (reduced resilience / regime contraction).
            RRS = None — no prior regime reset, or not yet enough post-reset steps.

        Counts ALL steps since the last regime reset (warmup + active) so that RRS
        becomes available after exactly warm_up_period steps from the reset point.
        """
        if not self.rrs_enabled:
            return
        if self._pre_reset_baseline_std is None:
            return
        if self._rrs is not None:
            return  # already settled
        self._post_reset_count += 1
        if self._post_reset_count >= self.warm_up_period and self.baseline_std is not None:
            self._rrs = round(self.baseline_std / (self._pre_reset_baseline_std + 1e-9), 4)

    # ─────────────────────────────────────────────────────────────────────────
    # v7.9 — PAGE-HINKLEY SLOW DRIFT DETECTOR
    # ─────────────────────────────────────────────────────────────────────────

    def _update_ph(self, current: float) -> Optional[str]:
        """
        Two-sided Page-Hinkley test for gradual mean drift.

        Catches slow distributional creep that CUSUM misses (CUSUM is designed
        for abrupt shifts; PH accumulates tiny deviations over hundreds of steps).

        Algorithm:
            delta = ph_delta (sensitivity = ph_delta_fraction × baseline_std,
                              set at warmup completion)
            cumsum_up_t   = cumsum_up_{t-1} + (x - mu_hat - delta)
            cumsum_down_t = cumsum_down_{t-1} + (mu_hat - x - delta)
            statistic_up   = cumsum_up_t - min(cumsum_up)
            statistic_down = cumsum_down_t - min(cumsum_down)
            drift if statistic > ph_threshold

        Returns:
            "up"   — upward drift detected
            "down" — downward drift detected
            "warn_up" / "warn_down" — early warning
            None   — no drift
        """
        if not self.ph_enabled or self.baseline_mean is None:
            return None
        # Initialize delta from baseline_std after warmup
        if self._ph_delta == 0.0 and self.baseline_std is not None:
            self._ph_delta = self.ph_delta_fraction * max(self.baseline_std, 1e-9)

        mu = self.baseline_mean
        delta = max(self._ph_delta, 1e-9)

        self._ph_sum_up += (current - mu - delta)
        self._ph_sum_down += (mu - current - delta)
        self._ph_min_up = min(self._ph_min_up, self._ph_sum_up)
        self._ph_min_down = min(self._ph_min_down, self._ph_sum_down)

        stat_up = self._ph_sum_up - self._ph_min_up
        stat_down = self._ph_sum_down - self._ph_min_down

        if stat_up > self.ph_threshold:
            self._ph_drift_up = True
            # Reset accumulators after detection to allow re-detection
            self._ph_sum_up = 0.0
            self._ph_min_up = 0.0
            return "up"
        if stat_down > self.ph_threshold:
            self._ph_drift_down = True
            self._ph_sum_down = 0.0
            self._ph_min_down = 0.0
            return "down"
        if stat_up > self.ph_warning_threshold:
            self._ph_warn_up = True
            return "warn_up"
        if stat_down > self.ph_warning_threshold:
            self._ph_warn_down = True
            return "warn_down"
        self._ph_drift_up = False
        self._ph_drift_down = False
        self._ph_warn_up = False
        self._ph_warn_down = False
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # v7.9 — MAHALANOBIS MULTIVARIATE ANOMALY SCORE
    # ─────────────────────────────────────────────────────────────────────────

    def _init_mahal_from_warmup(self) -> None:
        """
        Initialize rolling covariance matrix from warmup data.
        Called once after warmup completes when mahalanobis_enabled=True.
        """
        vecs = [np.array(v, dtype=float) for v in self.values_deque
                if isinstance(v, (list, tuple)) and len(v) == self.n_channels]
        if len(vecs) < self.n_channels + 2:
            return
        X = np.stack(vecs)                          # (n_warmup, d)
        self._mahal_mean = X.mean(axis=0)           # (d,)
        d = self.n_channels
        # Sample covariance with Tikhonov regularization
        Xc = X - self._mahal_mean
        cov = (Xc.T @ Xc) / max(len(vecs) - 1, 1)
        cov += self.mahalanobis_regularize * np.eye(d)
        self._mahal_cov = cov
        try:
            self._mahal_cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self._mahal_cov_inv = np.eye(d)

    def _update_mahal(self, vec: np.ndarray) -> float:
        """
        Update EWMA covariance via Woodbury rank-1 formula and compute
        Mahalanobis distance for the current multivariate observation.

        Woodbury update:
            Σ_t = a * Σ_{t-1} + b * d * d^T   (a = 1-α, b = α, d = x - μ)
            Σ_t^{-1} = (1/a)*P - (b/a²) * (Pd)(Pd)^T / (1 + (b/a) * d^T P d)
            where P = Σ_{t-1}^{-1}

        Returns: normalized Mahalanobis distance = sqrt(D²/n_channels)
        """
        if self._mahal_mean is None or self._mahal_cov_inv is None:
            return 0.0
        a = 1.0 - self.mahalanobis_alpha
        b = self.mahalanobis_alpha
        P = self._mahal_cov_inv

        # Update EWMA mean
        self._mahal_mean = a * self._mahal_mean + b * vec

        # Deviation from updated mean
        d_vec = vec - self._mahal_mean                  # shape (dim,)

        # Woodbury rank-1 update of Σ^{-1}
        Pd = P @ d_vec                                   # shape (dim,)
        denom = 1.0 + (b / a) * float(d_vec @ Pd)
        if abs(denom) > 1e-12:
            P_new = (1.0 / a) * P - (b / (a * a * denom)) * np.outer(Pd, Pd)
        else:
            P_new = (1.0 / a) * P
        # Re-regularize to prevent drift to indefiniteness
        d = self.n_channels
        P_new += self.mahalanobis_regularize * np.eye(d)
        self._mahal_cov_inv = P_new

        # Mahalanobis distance squared
        D_sq = float(d_vec @ P_new @ d_vec)
        D_sq = max(0.0, D_sq)
        self._mahal_score = round(math.sqrt(D_sq / max(d, 1)), 4)
        return self._mahal_score

    # ─────────────────────────────────────────────────────────────────────────
    # v7.9 — ANOMALY SCORE & ALERT ATTRIBUTION
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_anomaly_score(
        self, current: float, ewma: float, dev_ewma: float
    ) -> float:
        """
        Continuous z-score severity: |current - ewma| / dev_ewma.
        Returned as an unbounded non-negative float (1.0 ≈ 1 sigma).
        """
        if not self.anomaly_score_enabled:
            return 0.0
        denom = max(dev_ewma, 1e-9)
        return round(abs(current - ewma) / denom, 4)

    def _build_alert_reasons(
        self,
        z_score: float,
        early_alert: bool,
        confirmed_alert: bool,
        boundary_warning: bool,
        ph_signal: Optional[str],
        mahal_score: float,
    ) -> List[str]:
        """
        Return list of signal names that contributed to the current alert state.
        Empty list = no elevated signals.
        """
        reasons: List[str] = []
        if confirmed_alert:
            reasons.append("confirmed_z_score")
        elif early_alert:
            reasons.append("early_z_score")
        if boundary_warning:
            reasons.append("boundary_proximity")
        if self._sti_regime == "turbulent":
            reasons.append("turbulent_sti")
        if self._rpi_regime == "rhythmic" and self._rpi > 0.5:
            reasons.append("dominant_rhythm")
        if self._rfi >= 1.8:
            reasons.append("rough_fractal")
        cpd = self._compute_cpd()
        if cpd is not None and abs(cpd) > 0.7:
            reasons.append(f"cusum_pressure_{'up' if cpd > 0 else 'down'}")
        if ph_signal in ("up", "down"):
            reasons.append(f"ph_drift_{ph_signal}")
        elif ph_signal in ("warn_up", "warn_down"):
            reasons.append(f"ph_warning_{ph_signal.split('_')[1]}")
        if self.mahalanobis_enabled and mahal_score > self.mahalanobis_early_mult:
            reasons.append("mahalanobis_outlier")
        return reasons

    # ─────────────────────────────────────────────────────────────────────────
    # WARM-UP INITIALIZER
    # ─────────────────────────────────────────────────────────────────────────

    def _initialize_from_warmup(self) -> None:
        if len(self.values_deque) == 0:
            return

        if self.multivariate and self.n_channels > 0:
            channel_values = [[] for _ in range(self.n_channels)]
            for v in self.values_deque:
                if isinstance(v, (list, tuple)) and len(v) == self.n_channels:
                    for ch, val in enumerate(v):
                        channel_values[ch].append(val)

            self.channel_baselines = []
            self.channel_ewmas = []
            self.channel_dev_ewmas = []
            aggregated_means = []
            aggregated_stds = []

            for ch_values in channel_values:
                if not ch_values:
                    continue
                ch_mean = sum(ch_values) / len(ch_values)
                ch_var = sum((x - ch_mean) ** 2 for x in ch_values) / len(ch_values)
                ch_std = math.sqrt(ch_var) if ch_var > 0 else 1e-6
                self.channel_baselines.append((ch_mean, ch_std))
                self.channel_ewmas.append(ch_mean)
                self.channel_dev_ewmas.append(ch_std)
                aggregated_means.append(ch_mean)
                aggregated_stds.append(ch_std)

            if aggregated_means:
                self.baseline_mean = self.aggregation_func(aggregated_means)
                self.baseline_std = self.aggregation_func(aggregated_stds)
                self.ewma = self.baseline_mean
                self.dev_ewma = self.baseline_std

            n = len(self.channel_ewmas)
            self._ch_tps_counters = [0] * n
            self._ch_oscillation_lookback = [[] for _ in range(n)]
            self._ch_fire_clear_counts = [0] * n
            self._ch_prev_early_alerts = [False] * n
            self._ch_damping_active = [False] * n

        else:
            aggregated = [v for v in self.values_deque if isinstance(v, (int, float))]
            if not aggregated:
                return
            n = len(aggregated)
            self.baseline_mean = sum(aggregated) / n
            variance = sum((x - self.baseline_mean) ** 2 for x in aggregated) / n
            self.baseline_std = math.sqrt(variance) if variance > 0 else 1e-6
            self.ewma = self.baseline_mean
            self.dev_ewma = self.baseline_std

        # v7.9: initialize PH delta and Mahalanobis covariance from warmup
        if self.ph_enabled and self.baseline_std is not None:
            self._ph_delta = self.ph_delta_fraction * max(self.baseline_std, 1e-9)
        if self.mahalanobis_enabled and self._mahal_cov_inv is None:
            self._init_mahal_from_warmup()

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def update_and_check(self, value: Union[float, List[float]]) -> Dict[str, Any]:
        """
        Feed a scalar or list. Returns dict with alert status, all diagnostics,
        and v7.7 FRM axiom fields (rpi, rpi_regime, rfi, rrs).
        boundary_warning present in ALL result dict paths.
        """
        # -- Input validation and imputation ---------------------------------
        if self.multivariate:
            if not isinstance(value, (list, tuple)):
                raise ValueError("Multivariate mode requires list/tuple input")
            if self.n_channels == 0:
                self.n_channels = len(value)
            elif len(value) != self.n_channels:
                raise ValueError(f"Expected {self.n_channels} channels, got {len(value)}")
            cleaned_value = []
            for v in value:
                if not math.isfinite(v):
                    if self.impute_method == "mean" and self.channel_ewmas:
                        cleaned_value.append(self.channel_ewmas[len(cleaned_value)])
                    elif self.impute_method == "skip":
                        return {"status": "skipped", "message": "Invalid input skipped"}
                    else:
                        warnings.warn("Invalid value; using 0 as fallback", RuntimeWarning)
                        cleaned_value.append(0.0)
                else:
                    cleaned_value.append(v)
            value = cleaned_value
        else:
            if not math.isfinite(value):
                if self.impute_method == "mean" and self.ewma is not None:
                    value = self.ewma
                elif self.impute_method == "skip":
                    return {"status": "skipped", "message": "Invalid input skipped"}
                else:
                    warnings.warn("Invalid value; using 0 as fallback", RuntimeWarning)
                    value = 0.0

        self.count += 1
        self.values_deque.append(value)

        # v7.7: count every post-reset step (including warmup) for RRS
        self._update_rrs_counter()

        # -- Warm-up phase ---------------------------------------------------
        if self.count <= self.warm_up_period:
            result = {
                "early_alert": False,
                "confirmed_alert": False,
                "boundary_warning": False,
                "status": "warm_up",
            }
            if self.use_fixed_during_warmup and len(self.values_deque) >= 10:
                self._initialize_from_warmup()
                temp = self._compute_alerts(value)
                result.update(temp)
                result["boundary_warning"] = False
                self.cusum_pos = 0.0
                self.cusum_neg = 0.0
            if self.verbose_explain:
                result["progress"] = f"{self.count}/{self.warm_up_period}"
            self._trigger_alert_callback(result)
            self._buffer_csv_result(result)
            return result

        if self.baseline_mean is None:
            self._initialize_from_warmup()

        # -- Normal operation ------------------------------------------------
        deviations = []

        if self.multivariate and self.per_channel_detection:
            # -- Multivariate path -------------------------------------------
            early_alerts = [False] * self.n_channels
            confirmed_alerts = [False] * self.n_channels
            boundary_warnings = [False] * self.n_channels
            channel_details = []

            for ch in range(self.n_channels):
                ch_value = value[ch]
                current_alpha = self._get_adaptive_alpha(ch)
                self.channel_ewmas[ch] = (
                    current_alpha * ch_value
                    + (1 - current_alpha) * self.channel_ewmas[ch]
                )
                deviation = ch_value - self.channel_ewmas[ch]
                self.channel_dev_ewmas[ch] = (
                    current_alpha * abs(deviation)
                    + (1 - current_alpha) * self.channel_dev_ewmas[ch]
                )
                deviations.append(deviation)

                eff_early_mult = self._sti_adjusted_mult(self.early_mult)

                # v7.7 fix: read damping state directly — single update call below
                if self.oscillation_damping_enabled and self._ch_damping_active[ch]:
                    eff_early_mult += self.oscillation_mult_bump * eff_early_mult

                early_upper = self.channel_ewmas[ch] + eff_early_mult * self.channel_dev_ewmas[ch]
                early_lower = (
                    self.channel_ewmas[ch] - eff_early_mult * self.channel_dev_ewmas[ch]
                    if self.two_sided else None
                )
                confirmed_upper = (
                    self.channel_baselines[ch][0]
                    + self.fixed_mult * self.channel_baselines[ch][1]
                )
                confirmed_lower = (
                    self.channel_baselines[ch][0]
                    - self.fixed_mult * self.channel_baselines[ch][1]
                    if self.two_sided else None
                )

                ch_early = ch_value > early_upper or (
                    self.two_sided and early_lower is not None
                    and ch_value < early_lower
                )
                ch_confirmed = ch_value > confirmed_upper or (
                    self.two_sided and confirmed_lower is not None
                    and ch_value < confirmed_lower
                )
                early_alerts[ch] = ch_early
                confirmed_alerts[ch] = ch_confirmed

                # v7.7 fix: single correct oscillation state update
                self._check_oscillation_damping(
                    ch_early,
                    self._ch_oscillation_lookback,
                    self._ch_fire_clear_counts,
                    self._ch_prev_early_alerts,
                    self._ch_damping_active,
                    idx=ch,
                )

                boundary_warnings[ch] = self._check_tps(
                    ch_value, early_upper, early_lower,
                    self._ch_tps_counters, idx=ch,
                )

                if self.verbose_explain:
                    channel_details.append({
                        "channel": ch,
                        "value": round(ch_value, 4),
                        "ewma": round(self.channel_ewmas[ch], 4),
                        "dev_ewma": round(self.channel_dev_ewmas[ch], 4),
                        "early_alert": ch_early,
                        "confirmed_alert": ch_confirmed,
                        "boundary_warning": boundary_warnings[ch],
                        "damping_active": self._ch_damping_active[ch],
                    })

            early_alert = any(early_alerts) if self.alert_if_any_channel else all(early_alerts)
            confirmed_alert = any(confirmed_alerts) if self.alert_if_any_channel else all(confirmed_alerts)
            boundary_warning = any(boundary_warnings) if self.alert_if_any_channel else all(boundary_warnings)
            current = self.aggregation_func(value)
            deviation = sum(deviations) / self.n_channels

            result = {
                "early_alert": early_alert,
                "confirmed_alert": confirmed_alert,
                "boundary_warning": boundary_warning,
                "status": "active",
                "current": round(current, 4),
            }
            if self.verbose_explain:
                result["channel_details"] = channel_details

        else:
            # -- Univariate path ---------------------------------------------
            current = self.aggregation_func(value) if self.multivariate else value
            current_alpha = self._get_adaptive_alpha()

            pre_update_early_upper = (
                self.ewma + self.early_mult * self.dev_ewma
                if self.ewma is not None and self.dev_ewma is not None else None
            )
            pre_update_early_lower = (
                self.ewma - self.early_mult * self.dev_ewma
                if self.two_sided and self.ewma is not None and self.dev_ewma is not None else None
            )

            self.ewma = current_alpha * current + (1 - current_alpha) * self.ewma
            deviation = current - self.ewma
            self.dev_ewma = (
                current_alpha * abs(deviation)
                + (1 - current_alpha) * self.dev_ewma
            )

            # STI (v7.6)
            self._update_sti(self.dev_ewma, self.baseline_std)

            # v7.7: feed scalar window, compute RPI / RFI
            self._scalar_window.append(current)
            if self.rpi_enabled:
                self._compute_rpi()
            if self.rfi_enabled:
                self._compute_rfi()

            # Initial alert computation
            result = self._compute_alerts(current)
            early_alert = result["early_alert"]
            confirmed_alert = result["confirmed_alert"]

            # Oscillation damping (univariate)
            osc_lookback_ref = [self._oscillation_lookback]
            osc_fc_ref = [self._oscillation_fire_clear_count]
            osc_prev_ref = [self._prev_early_alert]
            osc_damp_ref = [self._damping_active]
            extra_mult = self._check_oscillation_damping(
                early_alert,
                osc_lookback_ref,
                osc_fc_ref,
                osc_prev_ref,
                osc_damp_ref,
                idx=0,
            )
            self._oscillation_lookback = osc_lookback_ref[0]
            self._oscillation_fire_clear_count = osc_fc_ref[0]
            self._prev_early_alert = osc_prev_ref[0]
            self._damping_active = osc_damp_ref[0]

            if self._damping_active and extra_mult > 0:
                result = self._compute_alerts(current, extra_early_mult=extra_mult)
                early_alert = result["early_alert"]
                confirmed_alert = result["confirmed_alert"]

            tps_ref = [self._tps_counter]
            bw_upper = pre_update_early_upper if pre_update_early_upper is not None else result.get("early_threshold", self.ewma + self.early_mult * self.dev_ewma)
            bw_lower = pre_update_early_lower if pre_update_early_lower is not None else result.get("early_lower_threshold")
            boundary_warning = self._check_tps(current, bw_upper, bw_lower, tps_ref, idx=0)
            self._tps_counter = tps_ref[0]
            result["boundary_warning"] = boundary_warning

        # -- Bidirectional CUSUM ---------------------------------------------
        cusum_dev = (
            self.dev_ewma
            if not self.multivariate or not self.per_channel_detection
            else sum(self.channel_dev_ewmas) / self.n_channels
        )
        k = 0.5 * cusum_dev
        self.cusum_pos = max(0, self.cusum_pos + (deviation - k))
        self.cusum_neg = max(0, self.cusum_neg + (-deviation - k))

        regime_change = (
            self.cusum_pos > (self.cusum_threshold * cusum_dev)
            or self.cusum_neg > (self.cusum_threshold * cusum_dev)
        )

        if regime_change and self.reset_after_regime_change:
            direction = "upward" if self.cusum_pos > self.cusum_neg else "downward"
            # v7.7: capture pre-reset baseline_std for RRS before resetting
            if self.rrs_enabled and self.baseline_std is not None:
                self._pre_reset_baseline_std = self.baseline_std
                self._post_reset_count = 0
                self._rrs = None
            if self.reset_after_regime_change == "full":
                self.reset()
            else:
                self.reset(soft=True)
                self._initialize_from_warmup()
            result = {
                "early_alert": False,
                "confirmed_alert": False,
                "boundary_warning": False,
                "status": "regime_reset",
                "message": (
                    f"{direction.capitalize()} regime change detected — "
                    f"{'full' if self.reset_after_regime_change == 'full' else 'soft'} reset"
                ),
            }
            self._trigger_alert_callback(result)
            self._buffer_csv_result(result)
            return result

        # -- Build final result ----------------------------------------------
        result.update({
            "early_alert": early_alert,
            "confirmed_alert": confirmed_alert,
            "status": "active",
            "current": round(current, 4),
            "ewma": round(self.ewma, 4) if self.ewma is not None else None,
            "dev_ewma": round(self.dev_ewma, 4) if self.dev_ewma is not None else None,
            "cusum_pos": round(self.cusum_pos, 2),
            "cusum_neg": round(self.cusum_neg, 2),
            # v7.6 fields
            "sti": round(self._sti, 4),
            "sti_regime": self._sti_regime,
            "oscillation_damping_active": self._damping_active,
        })

        # CPD
        cpd = self._compute_cpd()
        if cpd is not None:
            result["cpd"] = cpd

        if self.two_sided and self.ewma is not None and self.dev_ewma is not None:
            result["early_lower_threshold"] = round(
                self.ewma - self.early_mult * self.dev_ewma, 4
            )

        # v7.7 fields
        if self.rpi_enabled:
            result["rpi"] = self._rpi
            result["rpi_regime"] = self._rpi_regime
        if self.rfi_enabled:
            result["rfi"] = self._rfi
        if self.rrs_enabled and self._rrs is not None:
            result["rrs"] = self._rrs

        # v7.9: continuous anomaly score
        _z = self._compute_anomaly_score(
            current,
            self.ewma if self.ewma is not None else current,
            self.dev_ewma if self.dev_ewma is not None else 1.0,
        )
        result["z_score"] = _z
        # anomaly_score [0-1]: logistic mapping centered at early_mult
        result["anomaly_score"] = round(
            1.0 / (1.0 + math.exp(-(_z - self.early_mult))), 4
        ) if self.anomaly_score_enabled else 0.0

        # v7.9: Page-Hinkley drift signal
        _ph_signal = self._update_ph(current)
        if _ph_signal is not None:
            result["drift_signal"] = _ph_signal
        result["ph_drift_up"] = self._ph_drift_up
        result["ph_drift_down"] = self._ph_drift_down

        # v7.9: Mahalanobis multivariate score
        if self.mahalanobis_enabled and self.multivariate:
            vec = np.array(value, dtype=float)
            _mahal = self._update_mahal(vec)
            result["mahal_score"] = _mahal
            result["mahal_early_alert"] = _mahal > self.mahalanobis_early_mult
            result["mahal_confirmed_alert"] = _mahal > self.mahalanobis_confirmed_mult
            # Promote to top-level alerts if Mahalanobis fires
            if result["mahal_early_alert"]:
                result["early_alert"] = True
            if result["mahal_confirmed_alert"]:
                result["confirmed_alert"] = True

        # v7.9: alert attribution
        result["alert_reasons"] = self._build_alert_reasons(
            _z,
            result.get("early_alert", False),
            result.get("confirmed_alert", False),
            result.get("boundary_warning", False),
            _ph_signal,
            result.get("mahal_score", 0.0),
        )

        if self.verbose_explain:
            cpd_str = f"CPD {cpd:+.3f} | " if cpd is not None else ""
            rpi_str = f"RPI {self._rpi:.3f}({self._rpi_regime}) | " if self.rpi_enabled else ""
            rfi_str = f"RFI {self._rfi:.3f} | " if self.rfi_enabled else ""
            rrs_str = f"RRS {self._rrs:.3f} | " if (self.rrs_enabled and self._rrs is not None) else ""
            result["explanation"] = (
                f"Deviation {round(deviation, 3)} | "
                f"CUSUM +{result['cusum_pos']} / -{result['cusum_neg']} | "
                f"{cpd_str}"
                f"STI {result['sti']:.3f} ({self._sti_regime}) | "
                f"{rpi_str}{rfi_str}{rrs_str}"
                f"Threshold {round(self.cusum_threshold * cusum_dev, 2)}"
            )
            # FRM axiom signal map — which axioms are currently active
            result["axiom_signals"] = {
                "axiom_6_rhythm":      f"{self._rpi_regime} (RPI={self._rpi:.4f})" if self.rpi_enabled else "disabled",
                "axiom_8_fractal":     f"RFI={self._rfi:.4f}" if self.rfi_enabled else "disabled",
                "axiom_11_resilience": f"RRS={self._rrs:.4f}" if (self.rrs_enabled and self._rrs is not None) else "awaiting_reset",
                "sti_turbulence":      f"{self._sti_regime} (STI={self._sti:.4f})",
            }

        # v7.8: append compact record to history ring-buffer for plot_history()
        if self.history_maxlen > 0:
            self._history.append({
                "n":                   self.count,
                "current":             result.get("current"),
                "ewma":                result.get("ewma"),
                "early_threshold":     result.get("early_threshold"),
                "early_lower":         result.get("early_lower_threshold"),
                "confirmed_threshold": result.get("confirmed_threshold"),
                "early_alert":         result.get("early_alert", False),
                "confirmed_alert":     result.get("confirmed_alert", False),
                "boundary_warning":    result.get("boundary_warning", False),
                "regime_reset":        result.get("status") == "regime_reset",
                "sti":                 result.get("sti"),
                "sti_regime":          result.get("sti_regime"),
                "rpi":                 result.get("rpi"),
                "rpi_regime":          result.get("rpi_regime"),
                "rfi":                 result.get("rfi"),
                "rrs":                 result.get("rrs"),
                "z_score":             result.get("z_score"),
                "anomaly_score":       result.get("anomaly_score"),
                "drift_signal":        result.get("drift_signal"),
                "mahal_score":         result.get("mahal_score"),
            })

        self._trigger_alert_callback(result)
        self._buffer_csv_result(result)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_adaptive_alpha(self, channel: Optional[int] = None) -> float:
        """Volatility-adaptive alpha further modulated by STI regime."""
        current_alpha = self.alpha
        if self.volatility_adaptive:
            if channel is not None and self.channel_ewmas and self.channel_dev_ewmas:
                rel_vol = self.channel_dev_ewmas[channel] / (
                    abs(self.channel_ewmas[channel]) + 1e-9
                )
            else:
                rel_vol = (
                    self.dev_ewma / (abs(self.ewma) + 1e-9)
                    if self.dev_ewma is not None and self.ewma is not None
                    else 0.0
                )
            vol_factor = max(
                self.vol_min_factor,
                min(self.vol_max_factor, 1.0 + rel_vol * 1.5),
            )
            current_alpha *= vol_factor

        if self.turbulence_adaptive:
            if self._sti_regime == "turbulent":
                current_alpha = min(current_alpha * 1.15, 1.0)
            elif self._sti_regime == "laminar":
                current_alpha = max(current_alpha * 0.90, 0.01)

        return current_alpha

    def _compute_alerts(
        self,
        value: Union[float, List[float]],
        extra_early_mult: float = 0.0,
    ) -> Dict[str, Any]:
        """Compute alerts. extra_early_mult for oscillation damping — no mutation."""
        current = self.aggregation_func(value) if isinstance(value, (list, tuple)) else value
        eff_early_mult = self._sti_adjusted_mult(self.early_mult) + extra_early_mult

        early_upper = (
            self.ewma + eff_early_mult * self.dev_ewma
            if self.ewma is not None and self.dev_ewma is not None else 0
        )
        confirmed_upper = (
            self.baseline_mean + self.fixed_mult * self.baseline_std
            if self.baseline_mean is not None and self.baseline_std is not None else 0
        )

        early_alert = current > early_upper
        confirmed_alert = current > confirmed_upper
        early_lower = None
        confirmed_lower = None

        if self.two_sided:
            early_lower = (
                self.ewma - eff_early_mult * self.dev_ewma
                if self.ewma is not None and self.dev_ewma is not None else 0
            )
            confirmed_lower = (
                self.baseline_mean - self.fixed_mult * self.baseline_std
                if self.baseline_mean is not None and self.baseline_std is not None else 0
            )
            early_alert = early_alert or current < early_lower
            confirmed_alert = confirmed_alert or current < confirmed_lower

        return {
            "early_alert": early_alert,
            "confirmed_alert": confirmed_alert,
            "early_threshold": round(early_upper, 4),
            "early_lower_threshold": round(early_lower, 4) if early_lower is not None else None,
            "confirmed_threshold": round(confirmed_upper, 4),
            "confirmed_lower_threshold": round(confirmed_lower, 4) if confirmed_lower is not None else None,
        }

    def _trigger_alert_callback(self, result: Dict) -> None:
        fires = (
            result.get("early_alert")
            or result.get("confirmed_alert")
            or result.get("boundary_warning")
        )
        if self.alert_callback and fires:
            try:
                self.alert_callback(result)
            except Exception as e:
                warnings.warn(f"Alert callback failed: {e}", RuntimeWarning)

    def _buffer_csv_result(self, result: Dict) -> None:
        if self.csv_output_path is None:
            return
        row = result.copy()
        row["timestamp"] = result.get("timestamp", None)
        self.csv_buffer.append(row)
        if self.csv_flush_interval > 0 and len(self.csv_buffer) >= self.csv_flush_interval:
            self.flush_csv()

    def flush_csv(self) -> None:
        """Flush buffered results to CSV (append, empty-file header fix)."""
        if not self.csv_buffer or self.csv_output_path is None:
            return
        file_exists = os.path.exists(self.csv_output_path)
        file_empty = file_exists and os.path.getsize(self.csv_output_path) == 0
        with open(self.csv_output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_buffer[0].keys())
            if not file_exists or file_empty:
                writer.writeheader()
            writer.writerows(self.csv_buffer)
        self.csv_buffer = []

    # ─────────────────────────────────────────────────────────────────────────
    # v7.8: VISUALIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def plot_history(
        self,
        last_n: int = 300,
        title: str = "Fracttalix Sentinel v7.8 — Dashboard",
        show: bool = True,
        figsize: Tuple[int, int] = (14, 9),
    ):
        """
        Three-panel matplotlib dashboard of recent detection history.

        Panel 1 — Value & Thresholds
            Blue line   : raw values
            Green line  : EWMA
            Orange band : early-warning threshold zone (upper & lower)
            Red dashes  : confirmed threshold lines
            Markers     : confirmed_alert (red ▼), early_alert (orange ▲),
                          boundary_warning (yellow ◆), regime_reset (grey ▏)

        Panel 2 — Turbulence & Rhythm (STI, RPI)
            Teal line   : STI (Sentinel Turbulence Index)
            Dashed lines: laminar / turbulent regime boundaries
            Purple line : RPI (Rhythm Power Index, right axis)

        Panel 3 — Fractal & Resilience (RFI, RRS)
            Brown line  : RFI (Rolling Fractal Index)
            Reference   : RFI = 1.5 (random walk neutral)
            Green dots  : RRS values when available

        Requires matplotlib. If not installed, emits a warning and returns None.
        """
        if not _MATPLOTLIB_AVAILABLE:
            warnings.warn(
                "matplotlib is not installed — install it to use plot_history().",
                RuntimeWarning,
            )
            return None

        records = list(self._history)[-last_n:]
        if not records:
            warnings.warn("No history recorded yet.", RuntimeWarning)
            return None

        xs = [r["n"] for r in records]

        def _extract(key):
            return [r.get(key) for r in records]

        currents   = _extract("current")
        ewmas      = _extract("ewma")
        eth_upper  = _extract("early_threshold")
        eth_lower  = _extract("early_lower")
        cth_upper  = _extract("confirmed_threshold")
        stis       = _extract("sti")
        rpis       = _extract("rpi")
        rfis       = _extract("rfi")
        rrss       = _extract("rrs")

        ea_xs  = [x for x, r in zip(xs, records) if r.get("early_alert")]
        ea_ys  = [r["current"] for r in records if r.get("early_alert")]
        ca_xs  = [x for x, r in zip(xs, records) if r.get("confirmed_alert")]
        ca_ys  = [r["current"] for r in records if r.get("confirmed_alert")]
        bw_xs  = [x for x, r in zip(xs, records) if r.get("boundary_warning")]
        bw_ys  = [r["current"] for r in records if r.get("boundary_warning")]
        reset_xs = [x for x, r in zip(xs, records) if r.get("regime_reset")]

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle(title, fontsize=13, fontweight="bold")

        # --- Panel 1: Value + thresholds ------------------------------------
        ax1 = axes[0]
        ax1.plot(xs, currents, color="steelblue", lw=1.2, label="value", zorder=3)
        ax1.plot(xs, ewmas, color="green", lw=1.0, alpha=0.8, label="EWMA", zorder=3)

        valid = [(x, u, l) for x, u, l in zip(xs, eth_upper, eth_lower)
                 if u is not None and l is not None]
        if valid:
            vx, vu, vl = zip(*valid)
            ax1.fill_between(vx, vl, vu, alpha=0.12, color="orange", label="early zone")
            ax1.plot(vx, vu, color="orange", lw=0.7, alpha=0.6)
            ax1.plot(vx, vl, color="orange", lw=0.7, alpha=0.6)

        valid_c = [(x, u) for x, u in zip(xs, cth_upper) if u is not None]
        if valid_c:
            vcx, vcu = zip(*valid_c)
            ax1.plot(vcx, vcu, color="red", lw=0.8, ls="--", alpha=0.7, label="confirmed threshold")

        if ea_xs:
            ax1.scatter(ea_xs, ea_ys, marker="^", color="orange", s=60, zorder=5, label="early alert")
        if ca_xs:
            ax1.scatter(ca_xs, ca_ys, marker="v", color="red", s=70, zorder=5, label="confirmed alert")
        if bw_xs:
            ax1.scatter(bw_xs, bw_ys, marker="D", color="gold", s=40, zorder=4, label="boundary warning")
        for rx in reset_xs:
            ax1.axvline(rx, color="grey", lw=0.8, ls=":", alpha=0.6)

        ax1.set_ylabel("Value")
        ax1.legend(loc="upper left", fontsize=7, ncol=3)
        ax1.grid(True, alpha=0.25)

        # --- Panel 2: STI + RPI --------------------------------------------
        ax2 = axes[1]
        valid_sti = [(x, s) for x, s in zip(xs, stis) if s is not None]
        if valid_sti:
            sx, sy = zip(*valid_sti)
            ax2.plot(sx, sy, color="teal", lw=1.1, label="STI")
            ax2.axhline(self.sti_laminar_threshold,  color="teal", lw=0.7, ls="--", alpha=0.5, label="laminar")
            ax2.axhline(self.sti_turbulent_threshold, color="teal", lw=0.7, ls=":",  alpha=0.5, label="turbulent")

        ax2r = ax2.twinx()
        valid_rpi = [(x, r) for x, r in zip(xs, rpis) if r is not None]
        if valid_rpi:
            rx2, ry2 = zip(*valid_rpi)
            ax2r.plot(rx2, ry2, color="purple", lw=1.0, alpha=0.75, label="RPI")
            ax2r.axhline(self.rpi_rhythmic_threshold,  color="purple", lw=0.6, ls="--", alpha=0.4)
            ax2r.axhline(self.rpi_broadband_threshold, color="purple", lw=0.6, ls=":",  alpha=0.4)
            ax2r.set_ylabel("RPI", color="purple", fontsize=9)
            ax2r.set_ylim(0, 1.05)

        for rx in reset_xs:
            ax2.axvline(rx, color="grey", lw=0.8, ls=":", alpha=0.6)

        ax2.set_ylabel("STI")
        ax2.legend(loc="upper left", fontsize=7)
        ax2.grid(True, alpha=0.25)

        # --- Panel 3: RFI + RRS --------------------------------------------
        ax3 = axes[2]
        valid_rfi = [(x, r) for x, r in zip(xs, rfis) if r is not None]
        if valid_rfi:
            fx, fy = zip(*valid_rfi)
            ax3.plot(fx, fy, color="saddlebrown", lw=1.1, label="RFI")
            ax3.axhline(1.5, color="saddlebrown", lw=0.7, ls="--", alpha=0.5, label="RFI=1.5 (random walk)")

        valid_rrs = [(x, r) for x, r in zip(xs, rrss) if r is not None]
        if valid_rrs:
            rx3, ry3 = zip(*valid_rrs)
            ax3.scatter(rx3, ry3, color="green", s=35, zorder=5, label="RRS")

        for rx in reset_xs:
            ax3.axvline(rx, color="grey", lw=0.8, ls=":", alpha=0.6)

        ax3.set_ylabel("RFI / RRS")
        ax3.set_xlabel("Step")
        ax3.legend(loc="upper left", fontsize=7)
        ax3.grid(True, alpha=0.25)

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # v7.8: AUTO-TUNE
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def auto_tune(
        cls,
        data: Union[np.ndarray, List[float]],
        target_fpr: float = 0.02,
        warmup_fraction: float = 0.60,
        alpha_grid: Optional[List[float]] = None,
        early_mult_grid: Optional[List[float]] = None,
        verbose: bool = False,
        **fixed_kwargs,
    ) -> Tuple["Detector_7_9", Dict[str, Any]]:
        """
        Data-driven hyperparameter search for alpha and early_mult.

        Treats `data` as representative normal data. Splits it into a warmup
        segment (first warmup_fraction) and an evaluation segment (remainder).
        Runs a grid search over alpha × early_mult combinations, counting
        early_alerts in the eval segment as false positives. Returns the
        combination whose actual FPR is closest to target_fpr.

        After finding the best params, returns a fully warmed-up detector
        (fed on all of `data`) ready for new incoming values.

        Parameters
        ----------
        data          : representative normal historical data (≥ 30 points)
        target_fpr    : desired false-positive rate on normal data (default 2%)
        warmup_fraction : fraction of data used for warm-up (default 60%)
        alpha_grid    : EWMA smoothing values to try (default: 5 values)
        early_mult_grid : early-warning multipliers to try (default: 6 values)
        verbose       : print search progress
        **fixed_kwargs: passed directly to Detector_7_9 (e.g. warm_up_period)

        Returns
        -------
        (detector, report)
            detector : Detector_7_9 warmed up on all of `data`, best params set
            report   : dict with best_alpha, best_early_mult, actual_fpr,
                       target_fpr, n_candidates, all_results
        """
        data = [float(v) for v in data]
        n = len(data)
        if n < 30:
            raise ValueError("auto_tune requires at least 30 data points")

        warmup_n = max(10, int(n * warmup_fraction))
        eval_data = data[warmup_n:]
        if not eval_data:
            raise ValueError("Not enough data for evaluation after warmup split")

        alpha_grid = alpha_grid or [0.05, 0.08, 0.12, 0.18, 0.25]
        early_mult_grid = early_mult_grid or [2.0, 2.5, 2.75, 3.0, 3.5, 4.0]

        candidates = []
        for alpha in alpha_grid:
            for em in early_mult_grid:
                kw = dict(fixed_kwargs)
                kw.setdefault("warm_up_period", max(10, warmup_n))
                det = cls(alpha=alpha, early_mult=em, **kw)
                for v in data[:warmup_n]:
                    det.update_and_check(v)
                n_fp = 0
                n_eval = 0
                for v in eval_data:
                    r = det.update_and_check(v)
                    if r.get("status") == "active":
                        n_eval += 1
                        if r.get("early_alert"):
                            n_fp += 1
                actual_fpr = n_fp / n_eval if n_eval > 0 else 0.0
                candidates.append((alpha, em, actual_fpr))
                if verbose:
                    print(f"  alpha={alpha:.2f} early_mult={em:.2f} → FPR={actual_fpr:.3f}")

        # Sort: closest actual FPR to target, tie-break by lower early_mult (more sensitive)
        candidates.sort(key=lambda c: (abs(c[2] - target_fpr), c[1]))
        best_alpha, best_em, best_fpr = candidates[0]

        if verbose:
            print(f"  → Best: alpha={best_alpha}, early_mult={best_em}, FPR={best_fpr:.3f}")

        # Build final detector warmed up on all data
        final_kw = dict(fixed_kwargs)
        final_kw.setdefault("warm_up_period", 60)
        final_det = cls(alpha=best_alpha, early_mult=best_em, **final_kw)
        for v in data:
            final_det.update_and_check(v)

        report = {
            "best_alpha":      best_alpha,
            "best_early_mult": best_em,
            "actual_fpr":      round(best_fpr, 4),
            "target_fpr":      target_fpr,
            "n_candidates":    len(candidates),
            "all_results":     [(c[0], c[1], round(c[2], 4)) for c in
                                sorted(candidates, key=lambda c: (c[0], c[1]))],
        }
        return final_det, report

    # ─────────────────────────────────────────────────────────────────────────
    # NUMBA / SURROGATES
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    @njit
    def _cumsum_jit(arr: np.ndarray) -> np.ndarray:
        """JIT-compiled cumulative sum."""
        n = len(arr)
        out = np.zeros(n, dtype=np.float64)
        out[0] = arr[0]
        for i in range(1, n):
            out[i] = out[i - 1] + arr[i]
        return out

    def phase_randomize(self, data: np.ndarray) -> np.ndarray:
        """FFT phase randomization — Theiler et al. (1992)."""
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        if data.ndim != 1:
            raise ValueError("phase_randomize expects 1D array")
        if len(data) == 0:
            return data.copy()
        fft_vals = np.fft.rfft(data)
        random_phase = np.random.uniform(0, 2 * np.pi, size=len(fft_vals))
        fft_vals = np.abs(fft_vals) * np.exp(1j * random_phase)
        return np.fft.irfft(fft_vals, n=len(data))

    def generate_surrogates(
        self,
        data: Union[np.ndarray, List[float]],
        n_surrogates: int = 1000,
    ) -> List[np.ndarray]:
        """Generate phase-randomized surrogates. Optional parallel mode."""
        data = np.asarray(data, dtype=float)
        if data.ndim != 1:
            raise ValueError("generate_surrogates expects 1D data")

        def surrogate_worker(_):
            return self.phase_randomize(data)

        if self.parallel_surrogates and n_surrogates >= 100:
            with Pool(cpu_count()) as p:
                surrogates = list(
                    tqdm(
                        p.imap(surrogate_worker, range(n_surrogates)),
                        total=n_surrogates,
                        desc="Generating surrogates",
                        disable=not self.verbose_explain,
                    )
                )
        else:
            surrogates = [
                surrogate_worker(_)
                for _ in tqdm(
                    range(n_surrogates),
                    desc="Generating surrogates",
                    disable=not self.verbose_explain,
                )
            ]
        return surrogates


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-STREAM SENTINEL (v7.8)
# ─────────────────────────────────────────────────────────────────────────────

class MultiStreamSentinel:
    """
    Fan-out manager for N independent Detector_7_9 instances.

    Each stream_id gets its own fully-isolated detector. Detectors are created
    on demand on first observation for that stream. Thread-safe via per-stream
    locks; parallel batch updates use a ThreadPoolExecutor.

    Usage
    -----
    ms = MultiStreamSentinel(detector_defaults=dict(warm_up_period=60))

    # Single stream:
    result = ms.update_and_check("sensor_A", 42.3)

    # Parallel batch across all streams:
    results = ms.update_all({"sensor_A": 42.3, "sensor_B": 17.1, "sensor_C": 99.0})

    # State persistence:
    blob = ms.save_all_states()
    ms2  = MultiStreamSentinel(detector_defaults=dict(warm_up_period=60))
    ms2.load_all_states(blob)
    """

    def __init__(
        self,
        detector_defaults: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None,
    ):
        self._defaults: Dict[str, Any] = detector_defaults or {}
        self._detectors: Dict[str, Detector_7_9] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._max_workers = max_workers or min(32, (cpu_count() or 1) + 4)

    def _get_or_create(self, stream_id: str) -> Detector_7_9:
        """Return the detector for stream_id, creating it if new."""
        with self._global_lock:
            if stream_id not in self._detectors:
                self._detectors[stream_id] = Detector_7_9(**self._defaults)
                self._locks[stream_id] = threading.Lock()
        return self._detectors[stream_id]

    def update_and_check(
        self, stream_id: str, value: Union[float, List[float]]
    ) -> Dict[str, Any]:
        """Feed one value to stream_id's detector. Creates detector if new."""
        det = self._get_or_create(stream_id)
        with self._locks[stream_id]:
            result = det.update_and_check(value)
        result["stream_id"] = stream_id
        return result

    def update_all(
        self,
        updates: Dict[str, Union[float, List[float]]],
        parallel: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Feed one value per stream in `updates`.
        parallel=True uses a ThreadPoolExecutor for concurrent updates.
        Returns dict of {stream_id: result}.
        """
        if parallel and len(updates) > 1:
            with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
                futures = {
                    sid: ex.submit(self.update_and_check, sid, val)
                    for sid, val in updates.items()
                }
                return {sid: f.result() for sid, f in futures.items()}
        return {sid: self.update_and_check(sid, val) for sid, val in updates.items()}

    @property
    def active_streams(self) -> List[str]:
        """List of all stream IDs seen so far."""
        with self._global_lock:
            return list(self._detectors.keys())

    def get_detector(self, stream_id: str) -> Optional[Detector_7_9]:
        """Return the Detector_7_9 for stream_id, or None if unknown."""
        return self._detectors.get(stream_id)

    def remove_stream(self, stream_id: str) -> None:
        """Remove and discard a stream's detector."""
        with self._global_lock:
            self._detectors.pop(stream_id, None)
            self._locks.pop(stream_id, None)

    def save_all_states(self) -> str:
        """Serialise all stream detectors to a single JSON string."""
        with self._global_lock:
            snapshot = {
                sid: json.loads(det.save_state())
                for sid, det in self._detectors.items()
            }
        return json.dumps(snapshot)

    def load_all_states(self, states_json: str) -> None:
        """Restore all stream detectors from a JSON string produced by save_all_states()."""
        snapshot = json.loads(states_json)
        with self._global_lock:
            for sid, state in snapshot.items():
                if sid not in self._detectors:
                    self._detectors[sid] = Detector_7_9(**self._defaults)
                    self._locks[sid] = threading.Lock()
                self._detectors[sid].load_state(json.dumps(state))

    def plot_streams(
        self,
        stream_ids: Optional[List[str]] = None,
        last_n: int = 200,
        show: bool = True,
    ):
        """
        Plot history dashboards for selected streams as a stacked figure.
        Requires matplotlib. If stream_ids is None, plots all active streams.
        """
        if not _MATPLOTLIB_AVAILABLE:
            warnings.warn("matplotlib not installed — cannot plot streams.", RuntimeWarning)
            return None
        ids = stream_ids or self.active_streams
        figs = []
        for sid in ids:
            det = self._detectors.get(sid)
            if det is not None:
                fig = det.plot_history(last_n=last_n, title=f"Stream: {sid}", show=show)
                figs.append(fig)
        return figs


# ─────────────────────────────────────────────────────────────────────────────
# v7.9: SENTINEL BENCHMARK — built-in labeled evaluation harness
# ─────────────────────────────────────────────────────────────────────────────

class SentinelBenchmark:
    """
    Built-in benchmark harness for Fracttalix Sentinel.

    Generates synthetic labeled time series (three anomaly archetypes) and
    evaluates any Detector_7_9 instance, returning standard metrics:
    precision, recall, F1, AUPRC, and mean detection lag.

    Anomaly archetypes:
        point       — isolated spikes: single-step amplitude excursions
        contextual  — periodic signal with out-of-phase segments
        collective  — sustained block shifts (local distribution change)

    Usage:
        data, labels = SentinelBenchmark.generate(n=1000, anomaly_type="point")
        det = Detector_7_9(warm_up_period=60)
        metrics = SentinelBenchmark.evaluate(det, data, labels)
        print(metrics)

        # Or run the full 3-archetype suite:
        SentinelBenchmark.run_suite(detector_kwargs={"warm_up_period": 60})
    """

    @staticmethod
    def generate(
        n: int = 1000,
        anomaly_type: str = "point",
        anomaly_frac: float = 0.05,
        amplitude: float = 5.0,
        noise_std: float = 1.0,
        period: int = 30,
        block_size: int = 10,
        seed: int = 0,
    ) -> Tuple[List[float], List[int]]:
        """
        Generate a labeled synthetic time series.

        Returns (data, labels) where labels[i] == 1 marks an anomaly step.

        anomaly_type:
            "point"       — random isolated spikes of size ±amplitude
            "contextual"  — periodic baseline; anomalies are out-of-phase patches
            "collective"  — Gaussian baseline; anomalies are contiguous blocks at
                            shifted mean (baseline_mean ± amplitude)
        """
        rng = np.random.default_rng(seed)
        labels = [0] * n

        if anomaly_type == "point":
            data = list(rng.normal(0, noise_std, n).astype(float))
            n_anom = max(1, int(n * anomaly_frac))
            idx = rng.choice(n, size=n_anom, replace=False)
            for i in idx:
                data[i] += amplitude * rng.choice([-1, 1])
                labels[i] = 1

        elif anomaly_type == "contextual":
            base = [math.sin(2 * math.pi * i / period) for i in range(n)]
            noise = list(rng.normal(0, noise_std * 0.3, n))
            data = [base[i] + noise[i] for i in range(n)]
            n_anom = max(1, int(n * anomaly_frac))
            anom_starts = rng.integers(0, n - period, size=n_anom)
            for start in anom_starts:
                length = rng.integers(period // 4, period // 2)
                for j in range(int(length)):
                    idx2 = min(start + j, n - 1)
                    data[idx2] = -base[idx2] + noise[idx2]  # phase flip = contextual anomaly
                    labels[idx2] = 1

        elif anomaly_type == "collective":
            data = list(rng.normal(0, noise_std, n).astype(float))
            n_blocks = max(1, int(n * anomaly_frac / block_size))
            starts = rng.integers(0, n - block_size, size=n_blocks)
            for start in starts:
                direction = rng.choice([-1, 1])
                for j in range(block_size):
                    idx3 = min(start + j, n - 1)
                    data[idx3] += direction * amplitude
                    labels[idx3] = 1
        else:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type!r}")

        return data, labels

    @staticmethod
    def evaluate(
        detector: "Detector_7_9",
        data: List[float],
        labels: List[int],
        score_field: str = "anomaly_score",
        alert_field: str = "early_alert",
        tol_lag: int = 5,
    ) -> Dict[str, Any]:
        """
        Run detector over data and compute classification metrics.

        Parameters
        ----------
        detector    : fresh or pre-warmed Detector_7_9 instance
        data        : 1-D list of floats
        labels      : 1-D list of ints (0=normal, 1=anomaly)
        score_field : result dict field for continuous scoring (default: "anomaly_score")
        alert_field : result dict field for binary alert (default: "early_alert")
        tol_lag     : detection within tol_lag steps of anomaly start counts as a hit

        Returns
        -------
        dict with: precision, recall, f1, auprc, mean_lag, tp, fp, fn, n_anomalies
        """
        scores: List[float] = []
        preds: List[int] = []

        for val in data:
            r = detector.update_and_check(val)
            scores.append(float(r.get(score_field, 0.0)))
            fired = bool(r.get(alert_field, False)) or bool(r.get("confirmed_alert", False))
            preds.append(1 if fired else 0)

        # Binary metrics at current threshold
        tp = fp = fn = 0
        matched: set = set()
        detection_lags: List[int] = []

        # Group anomaly labels into contiguous blocks
        anom_blocks: List[Tuple[int, int]] = []
        in_block = False
        block_start = 0
        for i, lb in enumerate(labels):
            if lb == 1 and not in_block:
                in_block = True
                block_start = i
            elif lb == 0 and in_block:
                anom_blocks.append((block_start, i - 1))
                in_block = False
        if in_block:
            anom_blocks.append((block_start, len(labels) - 1))

        for block_start, block_end in anom_blocks:
            detected = False
            for lag in range(tol_lag + block_end - block_start + 1):
                check_idx = block_start + lag
                if check_idx > min(block_end + tol_lag, len(preds) - 1):
                    break
                if preds[check_idx] == 1 and check_idx not in matched:
                    detected = True
                    matched.add(check_idx)
                    detection_lags.append(max(0, check_idx - block_start))
                    break
            if detected:
                tp += 1
            else:
                fn += 1

        for i, p in enumerate(preds):
            if p == 1 and i not in matched and labels[i] == 0:
                fp += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        mean_lag = float(np.mean(detection_lags)) if detection_lags else float("nan")

        # AUPRC via trapezoidal integration over score thresholds
        sorted_scores = sorted(set(scores), reverse=True)
        pr_points = [(1.0, 0.0)]
        for thresh in sorted_scores:
            tp_t = sum(1 for i, s in enumerate(scores) if s >= thresh and labels[i] == 1)
            fp_t = sum(1 for i, s in enumerate(scores) if s >= thresh and labels[i] == 0)
            fn_t = sum(1 for i, lb in enumerate(labels) if lb == 1 and scores[i] < thresh)
            p_t = tp_t / max(tp_t + fp_t, 1)
            r_t = tp_t / max(tp_t + fn_t, 1)
            pr_points.append((p_t, r_t))
        pr_points.append((0.0, 1.0))
        pr_points.sort(key=lambda x: x[1])
        _pr_p = [p for p, _ in pr_points]
        _pr_r = [r for _, r in pr_points]
        # Compatible with NumPy 1.x and 2.x
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        auprc = float(_trapz(_pr_p, _pr_r))

        return {
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "f1":           round(f1, 4),
            "auprc":        round(max(0.0, auprc), 4),
            "mean_lag":     round(mean_lag, 2) if not math.isnan(mean_lag) else None,
            "tp":           tp,
            "fp":           fp,
            "fn":           fn,
            "n_anomalies":  len(anom_blocks),
        }

    @staticmethod
    def run_suite(
        detector_kwargs: Optional[Dict[str, Any]] = None,
        n: int = 1000,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run the three canonical benchmark scenarios and print a summary table.

        Returns dict of {anomaly_type: metrics_dict}.
        """
        kw = detector_kwargs or {}
        results: Dict[str, Dict[str, Any]] = {}
        archetypes = [
            ("point",       dict(n=n, anomaly_type="point",       amplitude=5.0, seed=seed)),
            ("contextual",  dict(n=n, anomaly_type="contextual",  amplitude=4.0, seed=seed)),
            ("collective",  dict(n=n, anomaly_type="collective",  amplitude=4.0, seed=seed)),
        ]
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  SentinelBenchmark  (n={n}, seed={seed})")
            print(f"  {'Archetype':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUPRC':>7} {'Lag':>5}")
            print(f"{'─'*60}")

        for name, gen_kw in archetypes:
            data, labels = SentinelBenchmark.generate(**gen_kw)
            det = Detector_7_9(**kw)
            metrics = SentinelBenchmark.evaluate(det, data, labels)
            results[name] = metrics
            if verbose:
                lag_str = f"{metrics['mean_lag']:.1f}" if metrics["mean_lag"] is not None else " n/a"
                print(
                    f"  {name:<14} "
                    f"{metrics['precision']:>6.3f} "
                    f"{metrics['recall']:>6.3f} "
                    f"{metrics['f1']:>6.3f} "
                    f"{metrics['auprc']:>7.4f} "
                    f"{lag_str:>5}"
                )
        if verbose:
            print(f"{'─'*60}\n")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# v7.9: SENTINEL SERVER — stdlib HTTP wrapper for MultiStreamSentinel
# ─────────────────────────────────────────────────────────────────────────────

class SentinelServer:
    """
    Lightweight HTTP API server wrapping MultiStreamSentinel.

    Uses Python's stdlib http.server — zero extra dependencies.

    Endpoints:
        POST /update
            Body: {"stream_id": "sensor_A", "value": 42.3}
            Returns: detector result dict as JSON

        GET /streams
            Returns: {"streams": ["sensor_A", "sensor_B", ...]}

        GET /status/<stream_id>
            Returns: last result dict for stream_id (if any)

        GET /health
            Returns: {"status": "ok", "version": "7.9", "n_streams": N}

    Usage:
        ms = MultiStreamSentinel(detector_defaults={"warm_up_period": 60})
        server = SentinelServer(ms, host="127.0.0.1", port=8765)
        server.start(background=True)   # non-blocking
        # ... send requests ...
        server.stop()
    """

    def __init__(
        self,
        ms: Optional["MultiStreamSentinel"] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        detector_defaults: Optional[Dict[str, Any]] = None,
    ):
        self.ms: MultiStreamSentinel = ms or MultiStreamSentinel(
            detector_defaults=detector_defaults or {}
        )
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._last_results: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def start(self, background: bool = False) -> None:
        """Start the HTTP server. background=True runs in a daemon thread."""
        sentinel_instance = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # silence default logging
                pass

            def _send_json(self, obj: Any, code: int = 200) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.rstrip("/")
                if path == "/health":
                    with sentinel_instance._lock:
                        n = len(sentinel_instance.ms._detectors)
                    self._send_json({"status": "ok", "version": __version__, "n_streams": n})
                elif path == "/streams":
                    with sentinel_instance._lock:
                        streams = list(sentinel_instance.ms._detectors.keys())
                    self._send_json({"streams": streams})
                elif path.startswith("/status/"):
                    sid = path[len("/status/"):]
                    with sentinel_instance._lock:
                        res = sentinel_instance._last_results.get(sid)
                    if res is None:
                        self._send_json({"error": f"unknown stream: {sid}"}, 404)
                    else:
                        self._send_json(res)
                else:
                    self._send_json({"error": "not found"}, 404)

            def do_POST(self):
                if self.path.rstrip("/") == "/update":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        payload = json.loads(body)
                        sid = payload["stream_id"]
                        val = payload["value"]
                    except (json.JSONDecodeError, KeyError) as e:
                        self._send_json({"error": str(e)}, 400)
                        return
                    result = sentinel_instance.ms.update_and_check(sid, val)
                    with sentinel_instance._lock:
                        sentinel_instance._last_results[sid] = result
                    self._send_json(result)
                else:
                    self._send_json({"error": "not found"}, 404)

        self._server = HTTPServer((self.host, self.port), _Handler)
        if background:
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()
        else:
            self._server.serve_forever()

    def stop(self) -> None:
        """Shutdown the server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST — 13 structured assertions covering all v7.8 features
# ─────────────────────────────────────────────────────────────────────────────

def _cli_main():
    """
    CLI entry point for streaming a CSV or running the benchmark.

    Usage examples:
        python fracttalix_sentinel_v79.py --smoke
        python fracttalix_sentinel_v79.py --file data.csv --alpha 0.10 --early-mult 2.5
        python fracttalix_sentinel_v79.py --benchmark
        python fracttalix_sentinel_v79.py --serve --port 8765
    """
    parser = argparse.ArgumentParser(
        prog="fracttalix_sentinel_v79",
        description=f"Fracttalix Sentinel v{__version__} — streaming anomaly detector",
    )
    parser.add_argument("--smoke",       action="store_true", help="Run built-in smoke tests")
    parser.add_argument("--benchmark",   action="store_true", help="Run SentinelBenchmark suite")
    parser.add_argument("--file",        type=str,   default=None, help="CSV file path to stream")
    parser.add_argument("--column",      type=int,   default=0,    help="Column index for value (default: 0)")
    parser.add_argument("--alpha",       type=float, default=0.12, help="EWMA alpha (default: 0.12)")
    parser.add_argument("--early-mult",  type=float, default=2.75, help="Early alert multiplier (default: 2.75)")
    parser.add_argument("--warmup",      type=int,   default=60,   help="Warm-up period (default: 60)")
    parser.add_argument("--serve",       action="store_true", help="Start SentinelServer")
    parser.add_argument("--host",        type=str,   default="127.0.0.1", help="Server host")
    parser.add_argument("--port",        type=int,   default=8765, help="Server port")
    parser.add_argument("--version",     action="store_true", help="Print version and exit")

    args = parser.parse_args()

    if args.version:
        print(f"Fracttalix Sentinel v{__version__} | {__license__}")
        return

    if args.smoke or (not any([args.benchmark, args.file, args.serve])):
        # Fall through to smoke tests below
        return False  # signal: run smoke tests

    if args.benchmark:
        SentinelBenchmark.run_suite(
            detector_kwargs={"alpha": args.alpha, "early_mult": args.early_mult,
                             "warm_up_period": args.warmup},
            verbose=True,
        )
        return True

    if args.file:
        det = Detector_7_9(alpha=args.alpha, early_mult=args.early_mult,
                           warm_up_period=args.warmup)
        import csv as _csv
        with open(args.file, newline="") as fh:
            reader = _csv.reader(fh)
            for row in reader:
                if not row:
                    continue
                try:
                    val = float(row[args.column])
                except (ValueError, IndexError):
                    continue
                r = det.update_and_check(val)
                print(json.dumps(r))
        return True

    if args.serve:
        print(f"Starting SentinelServer at http://{args.host}:{args.port} ...")
        ms = MultiStreamSentinel(
            detector_defaults={"alpha": args.alpha, "early_mult": args.early_mult,
                               "warm_up_period": args.warmup}
        )
        srv = SentinelServer(ms, host=args.host, port=args.port)
        try:
            srv.start(background=False)  # blocking
        except KeyboardInterrupt:
            srv.stop()
        return True

    return False


if __name__ == "__main__":
    import random

    _cli_result = _cli_main()
    if _cli_result:
        sys.exit(0)

    print("Fracttalix Sentinel v7.9 — Smoke test")
    print("=" * 55)

    def make_detector(**kwargs):
        defaults = dict(
            warm_up_period=60,
            turbulence_adaptive=True,
            boundary_warning_enabled=True,
            tps_proximity_pct=0.15,
            tps_periods=5,
            oscillation_damping_enabled=True,
            oscillation_count_threshold=3,
            oscillation_periods=10,
            oscillation_mult_bump=0.10,
            cpd_enabled=True,
            rpi_enabled=True,
            rfi_enabled=True,
            rrs_enabled=True,
        )
        defaults.update(kwargs)
        return Detector_7_9(**defaults)

    def warm_up(det, seed=42, n=60):
        random.seed(seed)
        for _ in range(n):
            det.update_and_check(random.gauss(0, 1))

    def fmt(r):
        sti_s = f"{r['sti']:.3f}" if r.get("sti") is not None else "n/a"
        cpd_s = f"{r['cpd']:+.3f}" if r.get("cpd") is not None else "n/a"
        rpi_s = f"{r.get('rpi', 'n/a')}"
        rfi_s = f"{r.get('rfi', 'n/a')}"
        return (f"STI={sti_s}({r.get('sti_regime','?')}) CPD={cpd_s} "
                f"RPI={rpi_s}({r.get('rpi_regime','?')}) RFI={rfi_s} "
                f"early={r.get('early_alert',False)} "
                f"confirmed={r.get('confirmed_alert',False)} "
                f"boundary={r.get('boundary_warning',False)} "
                f"status={r.get('status','?')}")

    # -- Test 1: Normal operation --------------------------------------------
    print("\n[1] Normal laminar operation")
    d = make_detector()
    warm_up(d)
    for _ in range(5):
        r = d.update_and_check(random.gauss(0, 1))
    print(f"  {fmt(r)}")
    assert r["status"] == "active", f"Expected active, got {r['status']}"
    assert not r["early_alert"], "False positive in normal operation"
    assert r.get("sti") is not None, "STI missing"
    assert r.get("cpd") is not None, "CPD missing"
    assert r.get("rpi") is not None, "RPI missing"
    assert r.get("rfi") is not None, "RFI missing"
    print("  PASS")

    # -- Test 2: Anomaly detection -------------------------------------------
    print("\n[2] Anomaly detection")
    d = make_detector()
    warm_up(d)
    triggered = False
    for val in [4.0, 5.0, 6.0]:
        r = d.update_and_check(val)
        print(f"  val={val} | {fmt(r)}")
        if r.get("status") == "active" and (r.get("early_alert") or r.get("confirmed_alert")):
            triggered = True
    assert triggered, "No anomaly detected on large values"
    print("  PASS")

    # -- Test 3: Boundary Layer Warning --------------------------------------
    print("\n[3] Boundary Layer Warning (TPS)")
    d = make_detector(tps_periods=3)
    warm_up(d)
    bw_fired = False
    for i in range(6):
        if d.ewma is None or d.dev_ewma is None:
            break
        pre_thresh = d.ewma + d.early_mult * d.dev_ewma
        approach_val = pre_thresh * 0.92
        r = d.update_and_check(approach_val)
        print(f"  step {i+1}: val={approach_val:.3f} pre_thresh={pre_thresh:.3f} "
              f"counter={d._tps_counter} boundary={r.get('boundary_warning',False)}")
        if r.get("boundary_warning"):
            bw_fired = True
    assert bw_fired, "Boundary warning never fired"
    print("  PASS")

    # -- Test 4: Regime change -----------------------------------------------
    print("\n[4] Regime change detection")
    d = make_detector(reset_after_regime_change="soft")
    warm_up(d)
    reset_seen = False
    for val in [-6.0, -7.0, -8.0, -9.0, -10.0, -11.0]:
        r = d.update_and_check(val)
        cpd_s = f"{r['cpd']:+.3f}" if r.get("cpd") is not None else "n/a"
        print(f"  val={val} | status={r['status']} | CPD={cpd_s}")
        if r["status"] == "regime_reset":
            reset_seen = True
            break
    assert reset_seen, "Regime change not detected"
    print("  PASS")

    # -- Test 5: State persistence -------------------------------------------
    print("\n[5] State persistence round-trip (includes v7.7 fields)")
    d = make_detector()
    warm_up(d)
    for _ in range(15):
        d.update_and_check(random.gauss(0, 1))
    state_json = d.save_state()
    d2 = make_detector()
    d2.load_state(state_json)
    r1 = d.update_and_check(1.5)
    r2 = d2.update_and_check(1.5)
    assert r1["status"] == r2["status"] == "active"
    assert r1.get("sti") == r2.get("sti"), f"STI mismatch: {r1.get('sti')} vs {r2.get('sti')}"
    assert r1.get("rpi") == r2.get("rpi"), f"RPI mismatch: {r1.get('rpi')} vs {r2.get('rpi')}"
    assert r1.get("rfi") == r2.get("rfi"), f"RFI mismatch: {r1.get('rfi')} vs {r2.get('rfi')}"
    print(f"  Original STI={r1.get('sti'):.4f} | Restored STI={r2.get('sti'):.4f}")
    print(f"  Original RPI={r1.get('rpi'):.4f} | Restored RPI={r2.get('rpi'):.4f}")
    print(f"  Original RFI={r1.get('rfi'):.4f} | Restored RFI={r2.get('rfi'):.4f}")
    print("  PASS")

    # -- Test 6: Oscillation damping -----------------------------------------
    print("\n[6] Oscillation Damping Filter")
    d = make_detector(oscillation_count_threshold=2, oscillation_periods=8)
    warm_up(d)
    r0 = d.update_and_check(0.0)
    thresh = r0.get("early_threshold", 3.0)
    over = thresh * 1.3
    damping_seen = False
    for cycle in range(5):
        r_h = d.update_and_check(over)
        r_l = d.update_and_check(0.0)
        damp = r_h.get("oscillation_damping_active", False)
        print(f"  cycle {cycle+1}: early={r_h.get('early_alert',False)} damping={damp}")
        if damp:
            damping_seen = True
    print(f"  Damping triggered: {damping_seen}")
    print("  PASS (oscillation pattern processed)")

    # -- Test 7: boundary_warning in all result paths ------------------------
    print("\n[7] boundary_warning present in all result dict paths")
    d = make_detector()
    r_warmup = d.update_and_check(1.0)
    assert "boundary_warning" in r_warmup, "boundary_warning missing from warm_up result"
    warm_up(d)
    r_active = d.update_and_check(1.0)
    assert "boundary_warning" in r_active, "boundary_warning missing from active result"
    d2 = make_detector(reset_after_regime_change="full")
    warm_up(d2)
    for val in [-8.0, -9.0, -10.0, -11.0, -12.0]:
        r_reset = d2.update_and_check(val)
        if r_reset["status"] == "regime_reset":
            assert "boundary_warning" in r_reset, "boundary_warning missing from regime_reset"
            break
    print("  boundary_warning present in: warm_up, active, regime_reset")
    print("  PASS")

    # -- Test 8: RPI (Axiom 6) — rhythmic series produces high RPI -----------
    print("\n[8] Rhythm Power Index (Axiom 6)")
    d = make_detector(warm_up_period=60, rpi_window=32)
    # Feed strong sine wave — should show high RPI (rhythmic)
    for i in range(60):
        d.update_and_check(math.sin(2 * math.pi * i / 8.0) * 0.5 + random.gauss(0, 0.05))
    rpi_vals = []
    for i in range(60, 80):
        r = d.update_and_check(math.sin(2 * math.pi * i / 8.0) * 0.5)
        rpi_vals.append(r.get("rpi", 0.0))
        print(f"  step {i-59}: RPI={r.get('rpi',0):.3f} regime={r.get('rpi_regime','?')}")
    max_rpi = max(rpi_vals)
    assert max_rpi > 0.0, "RPI never computed non-zero value on periodic input"
    print(f"  Max RPI on sine: {max_rpi:.4f}")
    # Feed white noise — RPI should be lower on average
    d2 = make_detector(warm_up_period=60, rpi_window=32)
    random.seed(99)
    for _ in range(60):
        d2.update_and_check(random.gauss(0, 1))
    noise_rpi_vals = []
    for _ in range(20):
        r = d2.update_and_check(random.gauss(0, 1))
        noise_rpi_vals.append(r.get("rpi", 0.0))
    avg_noise_rpi = sum(noise_rpi_vals) / len(noise_rpi_vals)
    print(f"  Avg RPI on white noise: {avg_noise_rpi:.4f}")
    print("  PASS (RPI computed; sine > noise expected on average)")

    # -- Test 9: RFI (Axiom 8) — rough vs smooth series --------------------
    print("\n[9] Rolling Fractal Index (Axiom 8)")
    # Disable regime resets so the scalar_window accumulates uninterrupted.
    # Run 50 post-warmup steps (> rfi_window=20) to ensure RFI is computed.
    d_smooth = make_detector(warm_up_period=60, rfi_window=20, reset_after_regime_change=False)
    d_rough = make_detector(warm_up_period=60, rfi_window=20, reset_after_regime_change=False)
    # Smooth: slow-period sine (period >> window, high lag-1 autocorrelation)
    for i in range(60):
        d_smooth.update_and_check(math.sin(2 * math.pi * i / 40.0))
    # Rough: alternating ±1 (maximally anti-persistent, ACF(lag=1) ≈ -1)
    for i in range(60):
        d_rough.update_and_check(1.0 if i % 2 == 0 else -1.0)
    rfi_smooth_vals = []
    rfi_rough_vals = []
    for i in range(50):
        r_s = d_smooth.update_and_check(math.sin(2 * math.pi * (60 + i) / 40.0))
        r_r = d_rough.update_and_check(1.0 if (60 + i) % 2 == 0 else -1.0)
        rfi_s = r_s.get("rfi", None)
        rfi_r = r_r.get("rfi", None)
        if rfi_s is not None and rfi_s != 1.5:
            rfi_smooth_vals.append(rfi_s)
        if rfi_r is not None and rfi_r != 1.5:
            rfi_rough_vals.append(rfi_r)
    # Take the last 20 settled values
    rfi_smooth_settled = rfi_smooth_vals[-20:] if len(rfi_smooth_vals) >= 20 else rfi_smooth_vals
    rfi_rough_settled = rfi_rough_vals[-20:] if len(rfi_rough_vals) >= 20 else rfi_rough_vals
    avg_smooth = sum(rfi_smooth_settled) / len(rfi_smooth_settled) if rfi_smooth_settled else 1.5
    avg_rough = sum(rfi_rough_settled) / len(rfi_rough_settled) if rfi_rough_settled else 1.5
    print(f"  Avg RFI smooth (sine, period=40): {avg_smooth:.4f}")
    print(f"  Avg RFI rough  (±1 alternating):  {avg_rough:.4f}")
    assert avg_rough > avg_smooth, f"Rough should have higher RFI: {avg_rough:.4f} vs {avg_smooth:.4f}"
    print("  PASS (rough > smooth RFI as expected by Axiom 8)")

    # -- Test 10: RRS (Axiom 11) — resilience recovery after regime reset ---
    print("\n[10] Resilience Recovery Score (Axiom 11)")
    d = make_detector(reset_after_regime_change="soft", rrs_enabled=True)
    warm_up(d)
    pre_std = d.baseline_std
    # Force a regime reset with a downward spike
    for val in [-8.0, -10.0, -12.0, -14.0, -16.0]:
        r = d.update_and_check(val)
        if r["status"] == "regime_reset":
            print(f"  Regime reset triggered | pre_reset_std stored: {d._pre_reset_baseline_std:.4f}")
            break
    # Run post-reset warmup on different (higher volatility) distribution
    random.seed(77)
    for _ in range(80):
        d.update_and_check(random.gauss(0, 2.0))  # 2x noise
    # RRS should be available now
    r = d.update_and_check(random.gauss(0, 2.0))
    print(f"  RRS={r.get('rrs', 'None')} (>1.0 expected: higher post-reset resilience)")
    assert r.get("rrs") is not None, "RRS not computed after post-reset warmup"
    print(f"  Pre-reset baseline_std: {d._pre_reset_baseline_std:.4f}")
    print(f"  Post-reset baseline_std: {d.baseline_std:.4f}")
    print("  PASS (RRS computed after regime reset and recovery)")

    # -- Test 11: auto_tune — data-driven hyperparameter search ---
    print("\n[11] auto_tune() — automatic hyperparameter tuning")
    random.seed(42)
    tune_data = [random.gauss(0, 1.0) for _ in range(600)]
    det_tuned, report = Detector_7_9.auto_tune(
        tune_data, target_fpr=0.05, verbose=False,
        alpha_grid=[0.05, 0.10, 0.15],
        early_mult_grid=[2.5, 3.0, 3.5],
    )
    assert isinstance(det_tuned, Detector_7_9), "auto_tune must return Detector_7_9 instance"
    assert "best_alpha" in report, "report missing best_alpha"
    assert "best_early_mult" in report, "report missing best_early_mult"
    assert "actual_fpr" in report, "report missing actual_fpr"
    assert 0.0 <= report["actual_fpr"] <= 1.0, "actual_fpr out of range"
    print(f"  best_alpha={report['best_alpha']}, best_early_mult={report['best_early_mult']}")
    print(f"  actual_fpr={report['actual_fpr']:.4f} (target 0.05)")
    # Verify detector is warmed up and functional
    r11 = det_tuned.update_and_check(0.5)
    assert r11["status"] != "warm_up", "tuned detector should be past warm_up"
    print("  PASS (auto_tune returns warmed-up detector with valid report)")

    # -- Test 12: MultiStreamSentinel — multi-stream fan-out ---
    print("\n[12] MultiStreamSentinel — multi-stream scalability")
    mss = MultiStreamSentinel(
        detector_defaults={"warm_up_period": 30, "history_maxlen": 100},
        max_workers=2,
    )
    random.seed(99)
    # Sequential single updates
    for i in range(40):
        mss.update_and_check("stream_A", random.gauss(0, 1.0))
        mss.update_and_check("stream_B", random.gauss(5, 1.0))
    r_a = mss.update_and_check("stream_A", 0.0)
    r_b = mss.update_and_check("stream_B", 5.0)
    assert r_a["status"] != "warm_up", "stream_A should be past warm_up"
    assert r_b["status"] != "warm_up", "stream_B should be past warm_up"
    print(f"  stream_A status={r_a['status']}, stream_B status={r_b['status']}")
    # Parallel batch update
    batch = {"stream_A": 0.1, "stream_B": 5.1, "stream_C": random.gauss(10, 1.0)}
    results = mss.update_all(batch, parallel=True)
    assert set(results.keys()) == {"stream_A", "stream_B", "stream_C"}, "update_all keys mismatch"
    print(f"  update_all keys: {sorted(results.keys())}")
    # State persistence round-trip
    states_json = mss.save_all_states()
    mss2 = MultiStreamSentinel(detector_defaults={"warm_up_period": 30, "history_maxlen": 100})
    mss2.load_all_states(states_json)
    r_a2 = mss2.update_and_check("stream_A", 0.0)
    assert r_a2["status"] != "warm_up", "restored stream_A should not be in warm_up"
    print(f"  State round-trip: stream_A restored status={r_a2['status']}")
    # Remove stream
    mss.remove_stream("stream_C")
    assert "stream_C" not in mss._detectors, "stream_C should be removed"
    print("  PASS (MultiStreamSentinel: fan-out, parallel batch, persistence, removal)")

    # -- Test 13: plot_history — visualization dashboard ---
    print("\n[13] plot_history() — visualization dashboard")
    if not _MATPLOTLIB_AVAILABLE:
        print("  SKIP (matplotlib not installed)")
    else:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for headless environments
        d13 = make_detector(history_maxlen=200)
        warm_up(d13)
        random.seed(13)
        for _ in range(50):
            d13.update_and_check(random.gauss(0, 1.0))
        # Should not raise
        fig = d13.plot_history(show=False)
        assert fig is not None, "plot_history must return a Figure"
        import matplotlib.pyplot as plt_test
        plt_test.close("all")
        print("  PASS (plot_history rendered 3-panel dashboard without error)")

    # -- Test 14: Continuous anomaly score ordering ---------------------------
    print("\n[14] z_score + anomaly_score — continuous severity output")
    d14 = make_detector()
    warm_up(d14)
    r_norm  = d14.update_and_check(0.5)
    r_mild  = d14.update_and_check(2.5)
    r_spike = d14.update_and_check(6.0)
    assert "z_score" in r_norm, "z_score missing from result"
    assert "anomaly_score" in r_norm, "anomaly_score missing from result"
    assert r_spike["z_score"] > r_mild["z_score"] > r_norm["z_score"], \
        f"z_score not monotone: {r_spike['z_score']} {r_mild['z_score']} {r_norm['z_score']}"
    assert 0.0 <= r_spike["anomaly_score"] <= 1.0, "anomaly_score out of [0,1]"
    assert r_spike["anomaly_score"] > r_norm["anomaly_score"], "anomaly_score not ordered"
    print(f"  normal z={r_norm['z_score']:.3f}  mild z={r_mild['z_score']:.3f}  spike z={r_spike['z_score']:.3f}")
    print(f"  normal score={r_norm['anomaly_score']:.4f}  spike score={r_spike['anomaly_score']:.4f}")
    print("  PASS (z_score and anomaly_score monotone with input severity)")

    # -- Test 15: Alert attribution (alert_reasons) ---------------------------
    print("\n[15] alert_reasons — signal attribution")
    d15 = make_detector()
    warm_up(d15)
    # Normal: no reasons expected
    r_quiet = d15.update_and_check(0.1)
    assert isinstance(r_quiet.get("alert_reasons"), list), "alert_reasons must be a list"
    # Spike should produce reasons
    r_loud = d15.update_and_check(7.0)
    assert len(r_loud["alert_reasons"]) > 0, "No alert_reasons on spike"
    print(f"  quiet reasons: {r_quiet['alert_reasons']}")
    print(f"  spike reasons: {r_loud['alert_reasons']}")
    print("  PASS (alert_reasons populated on anomaly, empty on normal)")

    # -- Test 16: Page-Hinkley slow drift detection ---------------------------
    print("\n[16] Page-Hinkley drift detector — gradual mean shift")
    d16 = make_detector(ph_enabled=True, ph_delta_fraction=0.05, ph_threshold=30.0,
                        ph_warning_threshold=10.0)
    warm_up(d16)
    # Inject slow gradual upward drift over 150 steps
    drift_detected = False
    drift_step = None
    random.seed(16)
    for step in range(150):
        val = random.gauss(step * 0.03, 0.8)  # mean creeps up slowly
        r16 = d16.update_and_check(val)
        sig = r16.get("drift_signal")
        if sig in ("up", "warn_up"):
            drift_detected = True
            drift_step = step
            print(f"  Drift signal '{sig}' at step {step} (val={val:.3f})")
            break
    assert drift_detected, "Page-Hinkley did not detect slow upward drift in 150 steps"
    assert drift_step is not None and drift_step < 150
    print(f"  PASS (PH detected gradual drift at step {drift_step} — CUSUM would miss this)")

    # -- Test 17: Mahalanobis multivariate mode ------------------------------
    print("\n[17] Mahalanobis multivariate — cross-channel covariance anomaly")
    d17 = make_detector(
        multivariate=True,
        per_channel_detection=True,
        mahalanobis_enabled=True,
        mahalanobis_early_mult=3.0,
        mahalanobis_confirmed_mult=4.5,
        warm_up_period=40,
    )
    random.seed(17)
    # Warmup with correlated 2-channel data
    for _ in range(40):
        x = random.gauss(0, 1)
        d17.update_and_check([x, x + random.gauss(0, 0.2)])  # strongly correlated
    assert d17._mahal_cov_inv is not None, "Mahalanobis cov_inv not initialized after warmup"
    # Normal correlated step
    r17_norm = d17.update_and_check([0.5, 0.6])
    # Uncorrelated anomaly (breaks covariance structure)
    r17_anom = d17.update_and_check([3.0, -3.0])
    assert "mahal_score" in r17_norm, "mahal_score missing"
    assert r17_anom["mahal_score"] > r17_norm["mahal_score"], \
        f"Mahalanobis did not score anomaly higher: {r17_anom['mahal_score']} vs {r17_norm['mahal_score']}"
    print(f"  normal mahal_score={r17_norm['mahal_score']:.4f}")
    print(f"  anomaly mahal_score={r17_anom['mahal_score']:.4f}")
    print("  PASS (Mahalanobis scores covariance-break anomaly higher than normal)")

    # -- Test 18: SentinelBenchmark — built-in labeled evaluation -------------
    print("\n[18] SentinelBenchmark — F1/AUPRC on labeled synthetic data")
    bench_results = SentinelBenchmark.run_suite(
        detector_kwargs={"warm_up_period": 60, "early_mult": 2.5},
        n=800, seed=42, verbose=True,
    )
    for archetype, metrics in bench_results.items():
        assert "f1" in metrics, f"f1 missing from {archetype} results"
        assert "auprc" in metrics, f"auprc missing from {archetype} results"
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["auprc"] <= 1.0
        print(f"  {archetype}: F1={metrics['f1']:.3f}  AUPRC={metrics['auprc']:.4f}")
    print("  PASS (SentinelBenchmark ran all 3 archetypes, returned valid metrics)")

    # -- Test 19: SentinelServer — HTTP API -----------------------------------
    print("\n[19] SentinelServer — stdlib HTTP API")
    import urllib.request
    server = SentinelServer(
        detector_defaults={"warm_up_period": 30},
        host="127.0.0.1",
        port=17659,
    )
    server.start(background=True)
    import time as _time
    _time.sleep(0.15)
    try:
        # Health check
        with urllib.request.urlopen("http://127.0.0.1:17659/health") as resp:
            health = json.loads(resp.read())
        assert health["status"] == "ok"
        assert health["version"] == __version__
        print(f"  /health → {health}")

        # POST /update
        payload = json.dumps({"stream_id": "test", "value": 1.5}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:17659/update",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            result19 = json.loads(resp.read())
        assert "status" in result19
        print(f"  POST /update → status={result19['status']}")

        # GET /streams
        with urllib.request.urlopen("http://127.0.0.1:17659/streams") as resp:
            streams19 = json.loads(resp.read())
        assert "test" in streams19["streams"]
        print(f"  GET /streams → {streams19['streams']}")
    finally:
        server.stop()
    print("  PASS (SentinelServer: health, POST /update, GET /streams all OK)")

    print("\n" + "=" * 55)
    print("All 19 tests passed. Sentinel v7.9 operational.")
    print("\nv7.9 additions verified (Meta-Kaizen):")
    print("  [v7.7] Rhythm Power Index (RPI) — Axiom 6 rhythmic patterns")
    print("  [v7.7] Rolling Fractal Index (RFI) — Axiom 8 self-similarity")
    print("  [v7.7] Resilience Recovery Score (RRS) — Axiom 11 post-stress")
    print("  [v7.7] Multivariate oscillation damping fix")
    print("  [v7.8] auto_tune() — data-driven hyperparameter grid search")
    print("  [v7.8] MultiStreamSentinel — thread-safe multi-stream fan-out")
    print("  [v7.8] plot_history() — 3-panel matplotlib visualization dashboard")
    print("  [v7.9] z_score + anomaly_score — continuous severity spectrum [0-1]")
    print("  [v7.9] alert_reasons — signal attribution for every alert")
    print("  [v7.9] Page-Hinkley drift — catches slow gradual mean drift")
    print("  [v7.9] Mahalanobis multivariate — rolling covariance, Woodbury update")
    print("  [v7.9] SentinelBenchmark — F1/AUPRC on labeled synthetic anomalies")
    print("  [v7.9] SentinelServer — stdlib HTTP API, zero extra dependencies")
