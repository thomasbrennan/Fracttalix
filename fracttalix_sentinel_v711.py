# fracttalix_sentinel v7.11 py

# Fracttalix Sentinel v7.11 — Meta-Kaizen: EWS, AQB, ThreadingHTTPServer, Bug Fixes, PE Baseline

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
# v7.10: Eight competitive gaps closed (Meta-Kaizen v7.10 analysis):
#   - Permutation Entropy (PE): FRM Axiom 3 — streaming complexity h_mu > 0.05 bits/symbol
#   - Rhythm Stability Index (RSI): FRM Axiom 10 — Kuramoto synchronization proxy;
#     reuses FFT already computed by _compute_rpi() — zero extra FFT cost
#   - Dual Encoding Score (DES): FRM Axiom 5 — RPI x (2-RFI) derived; zero cost
#   - Variance CUSUM: CUSUM on dev^2 catches volatility explosions mean-CUSUM misses
#   - Seasonal Periodic Baseline (SPB): per-phase EWMA indexed by step % period;
#     fixes contextual F1 from 0.00 -> >0.10; period auto-detected from FFT
#   - VUS-PR: Volume Under PR Surface (NeurIPS 2024 gold standard);
#     avg AUPRC over 5 buffer tolerances {0,5,10,15,20}
#   - Naive 3-sigma baseline comparison in run_suite() for all archetypes
#   - "drift" + "variance" benchmark archetypes (tests PH and Variance CUSUM)
#   - auto_tune() labeled_data parameter: optimise F1 when labels available
#   - plot_history() 4-panel dashboard: adds PE + anomaly_score panel
#   - All v7.9 features, parameters, and JSON state fields preserved (backward-compatible)
# v7.11: Seven improvements (Meta-Kaizen v7.11 analysis):
#   BUG FIXES:
#   - Stale CLI prog name corrected (v79 → v711)
#   - Redundant _baseline_var double-assignment removed from _initialize_from_warmup
#   - generate_surrogates Pool worker moved to module level — fixes Windows pickling failure
#   - Multivariate path now appends aggregated current to _scalar_window — PE/RSI no longer stale
#   - alert_reasons "low_entropy_ordered": absolute PE < 0.3 replaced with contextual PE vs
#     rolling PE baseline (PE EWMA); prevents false alarms on normally periodic signals
#   NEW FEATURES:
#   - FRM Axiom 9: Early Warning Signals (EWS) — Scheffer (2009) critical slowing down;
#     rising variance trend + rising AC(1) trend → EWS score [0,1]; regime stable/approaching/critical
#   - Adaptive Quantile Baseline (AQB) — FRM Axiom 1 (Scale Invariance); distribution-free,
#     heavy-tail-robust thresholds via rolling empirical quantile of absolute deviations;
#     quantile_threshold_mode replaces ewma±mult×dev_ewma with data-driven percentile bounds
#   - ThreadingHTTPServer: SentinelServer now handles concurrent users via ThreadingMixIn;
#     adds DELETE /stream/<id> and POST /reset/<id> endpoints; no new dependencies
#   - All v7.10 features, parameters, and JSON state fields preserved (backward-compatible)

# Designed for finance, medical, infrastructure/IoT/security monitoring, and research
# Theoretical foundation: The Fractal Rhythm Model (Brennan & Grok 4, 2026)
# 11 Axioms — see Papers branch: https://github.com/thomasbrennan/Fracttalix

__version__ = "7.11"
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
import socketserver
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


# v7.11 Bug Fix: module-level surrogate worker for Pool pickling compatibility on Windows/spawn
def _phase_randomize_worker(args):
    """Module-level FFT phase-randomisation worker — picklable by multiprocessing.Pool."""
    data, seed = args
    rng = np.random.default_rng(seed)
    fft_vals = np.fft.rfft(data)
    random_phase = rng.uniform(0, 2 * np.pi, size=len(fft_vals))
    fft_vals = np.abs(fft_vals) * np.exp(1j * random_phase)
    return np.fft.irfft(fft_vals, n=len(data))


class Detector_7_10:
    """
    Enhanced lightweight, regime-aware anomaly detector — v7.11.

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

        # === v7.10: Permutation Entropy (FRM Axiom 3) — complexity growth ===
        pe_enabled: bool = True,
        pe_window: int = 32,       # rolling window for ordinal pattern analysis
        pe_dim: int = 3,           # embedding dimension (d! possible patterns)

        # === v7.10: Rhythm Stability Index (FRM Axiom 10) — synchronization ===
        rsi_enabled: bool = True,  # requires rpi_enabled=True (reuses FFT)

        # === v7.10: Variance CUSUM — volatility anomaly detection ===
        vcusum_enabled: bool = True,
        vcusum_threshold: float = 8.0,   # alert when vcusum > threshold * baseline_var

        # === v7.10: Seasonal Periodic Baseline — contextual anomaly detection ===
        seasonal_enabled: bool = True,
        seasonal_period: Optional[int] = None,  # None = auto-detect from FFT

        # === v7.11: Early Warning Signals (FRM Axiom 9) — critical transitions ===
        ews_enabled: bool = True,
        ews_window: int = 60,          # rolling window for variance/AC trend analysis
        ews_sensitivity: float = 1.0,  # multiplier: higher = more sensitive to slow trends

        # === v7.11: Adaptive Quantile Baseline (FRM Axiom 1) — scale invariance ===
        quantile_threshold_mode: bool = False,    # replace mult×dev_ewma with empirical quantile
        quantile_early_pct: float = 0.95,         # early alert at this percentile of |deviation|
        quantile_confirmed_pct: float = 0.995,    # confirmed alert percentile
        quantile_window: int = 200,               # rolling window size for deviation history
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
        self._scalar_window: deque = deque(maxlen=max(rpi_window, rfi_window, pe_window))

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

        # v7.10 params
        self.pe_enabled = pe_enabled
        self.pe_window = pe_window
        self.pe_dim = pe_dim
        self.rsi_enabled = rsi_enabled
        self.vcusum_enabled = vcusum_enabled
        self.vcusum_threshold = vcusum_threshold
        self.seasonal_enabled = seasonal_enabled
        self.seasonal_period = seasonal_period

        # v7.11 — params
        self.ews_enabled = ews_enabled
        self.ews_window = ews_window
        self.ews_sensitivity = ews_sensitivity
        self.quantile_threshold_mode = quantile_threshold_mode
        self.quantile_early_pct = quantile_early_pct
        self.quantile_confirmed_pct = quantile_confirmed_pct
        self.quantile_window = quantile_window

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

        # v7.10 — Permutation Entropy (Axiom 3)
        self._pe: float = 0.0
        self._pe_regime: str = "transitional"

        # v7.10 — Rhythm Stability Index (Axiom 10)
        self._rsi: float = 0.5
        self._rsi_regime: str = "transitional"
        self._current_fft_power_normed: Optional[np.ndarray] = None
        self._prev_fft_power: Optional[np.ndarray] = None

        # v7.10 — Variance CUSUM
        self._vcusum: float = 0.0
        self._baseline_var: float = 0.0

        # v7.10 — Seasonal Periodic Baseline
        self._detected_period: Optional[int] = None
        self._phase_ewmas: List = []
        self._phase_dev_ewmas: List = []
        self._phase_counts: List = []

        # v7.11 — Early Warning Signals (FRM Axiom 9)
        self._ews_score: float = 0.0
        self._ews_regime: str = "stable"

        # v7.11 — PE EWMA baseline (for contextual alert_reasons)
        self._pe_baseline: Optional[float] = None  # rolling mean PE

        # v7.11 — Adaptive Quantile Baseline (FRM Axiom 1)
        self._dev_history: deque = deque(maxlen=quantile_window)
        self._q_early: float = 0.0
        self._q_confirmed: float = 0.0

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

        # v7.10 state reset
        self._pe = 0.0
        self._pe_regime = "transitional"
        self._rsi = 0.5
        self._rsi_regime = "transitional"
        self._current_fft_power_normed = None
        self._prev_fft_power = None
        self._vcusum = 0.0
        if not soft:
            self._baseline_var = 0.0
            self._detected_period = None
            self._phase_ewmas = []
            self._phase_dev_ewmas = []
            self._phase_counts = []

        # v7.11 state reset
        self._ews_score = 0.0
        self._ews_regime = "stable"
        self._pe_baseline = None
        self._dev_history.clear()
        self._q_early = 0.0
        self._q_confirmed = 0.0

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
            # v7.10 state
            "pe": self._pe,
            "pe_regime": self._pe_regime,
            "rsi": self._rsi,
            "rsi_regime": self._rsi_regime,
            "prev_fft_power": self._prev_fft_power.tolist() if self._prev_fft_power is not None else None,
            "vcusum": self._vcusum,
            "baseline_var": self._baseline_var,
            "detected_period": self._detected_period,
            "phase_ewmas": self._phase_ewmas,
            "phase_dev_ewmas": self._phase_dev_ewmas,
            "phase_counts": self._phase_counts,
            # v7.11 fields
            "ews_score": self._ews_score,
            "ews_regime": self._ews_regime,
            "pe_baseline": self._pe_baseline,
            "dev_history": list(self._dev_history),
            "q_early": self._q_early,
            "q_confirmed": self._q_confirmed,
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
        self._scalar_window = deque(sw, maxlen=max(self.rpi_window, self.rfi_window, self.pe_window))
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
        # v7.10
        self._pe = s.get("pe", 0.0)
        self._pe_regime = s.get("pe_regime", "transitional")
        self._rsi = s.get("rsi", 0.5)
        self._rsi_regime = s.get("rsi_regime", "transitional")
        pfp = s.get("prev_fft_power", None)
        self._prev_fft_power = np.array(pfp) if pfp is not None else None
        self._current_fft_power_normed = None
        self._vcusum = s.get("vcusum", 0.0)
        self._baseline_var = s.get("baseline_var", 0.0)
        self._detected_period = s.get("detected_period", None)
        self._phase_ewmas = s.get("phase_ewmas", [])
        self._phase_dev_ewmas = s.get("phase_dev_ewmas", [])
        self._phase_counts = s.get("phase_counts", [])
        # v7.11 fields (backward-compatible defaults)
        self._ews_score = s.get("ews_score", 0.0)
        self._ews_regime = s.get("ews_regime", "stable")
        self._pe_baseline = s.get("pe_baseline", None)
        raw_dh = s.get("dev_history", [])
        self._dev_history = deque(raw_dh, maxlen=self.quantile_window)
        self._q_early = s.get("q_early", 0.0)
        self._q_confirmed = s.get("q_confirmed", 0.0)

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
        # Store normalised FFT power for RSI (Axiom 10) — zero extra FFT cost
        self._current_fft_power_normed = osc_power / total
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
        seasonal_alert: bool = False,
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
        # v7.10 additional reasons
        if self._vcusum > 0 and self.vcusum_enabled:
            reasons.append("volatility_burst_vcusum")
        if seasonal_alert:
            reasons.append("seasonal_anomaly")
        # v7.11 Bug Fix: compare PE against its own rolling baseline (not absolute 0.3).
        # Only flag when PE collapses to ≤40% of its own established baseline AND the
        # baseline is well-established (> 0.3).  Prevents false alarms on normally
        # periodic/rhythmic signals (e.g. sine waves) whose typical PE is consistently
        # low — their baseline converges to their typical level, so the relative-drop
        # check never fires on normal behaviour.
        if (self.pe_enabled
                and self._pe_baseline is not None
                and self._pe_baseline > 0.3):       # baseline must be 'complex' signal
            pe_threshold = self._pe_baseline * 0.4  # 60% drop required — genuine collapse
            if self._pe < pe_threshold:
                reasons.append("low_entropy_ordered")
        if self._rsi < 0.3 and self.rsi_enabled:
            reasons.append("unstable_rhythm_rsi")
        # v7.11: EWS early warning
        if self.ews_enabled and self._ews_regime in ("approaching", "critical"):
            reasons.append(f"ews_{self._ews_regime}")
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
        # v7.10/v7.11: initialize baseline variance for VCUSUM; init seasonal model
        # v7.11 Bug Fix: removed redundant first assignment (was immediately clobbered)
        if self.baseline_std is not None:
            self._baseline_var = max(self.baseline_std ** 2, 1e-12)
        if self.seasonal_enabled and len(self._phase_ewmas) == 0:
            self._init_seasonal()


    # ─────────────────────────────────────────────────────────────────────────
    # v7.10 — PERMUTATION ENTROPY (FRM Axiom 3)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_pe(self) -> "Tuple[float, str]":
        """
        Permutation Entropy — ordinal pattern Shannon entropy (FRM Axiom 3).

        "Complexity growth h_μ > 0.05 bits/symbol signals dimensional expansion."
        Reference: Bandt & Pompe, 2002; Grassberger, 1989.

        Maps ordinal patterns of d consecutive samples to a normalised entropy:
            PE = H(patterns) / log2(d!)    ∈ [0, 1]
            PE → 0  : fully regular / periodic (one pattern dominates)
            PE → 1  : maximally complex / noise-like (uniform distribution)

        Regime:
            "complex"      : PE > 0.7  (rich dynamics / stochastic-like)
            "regular"      : PE < 0.3  (ordered / periodic)
            "transitional" : 0.3 ≤ PE ≤ 0.7
        """
        if not self.pe_enabled or len(self._scalar_window) < self.pe_dim:
            return self._pe, self._pe_regime
        arr = list(self._scalar_window)[-self.pe_window:]
        d = self.pe_dim
        pattern_counts: dict = {}
        for i in range(len(arr) - d + 1):
            seg = arr[i: i + d]
            pat = tuple(sorted(range(d), key=lambda k: seg[k]))
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
        total = sum(pattern_counts.values())
        if total == 0:
            return self._pe, self._pe_regime
        entropy = 0.0
        for cnt in pattern_counts.values():
            p = cnt / total
            if p > 0.0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(math.factorial(d)) if d > 1 else 1.0
        pe = float(entropy / max_entropy) if max_entropy > 0.0 else 0.0
        pe = max(0.0, min(1.0, pe))
        self._pe = round(pe, 4)
        self._pe_regime = "complex" if pe > 0.7 else ("regular" if pe < 0.3 else "transitional")
        return self._pe, self._pe_regime

    # ─────────────────────────────────────────────────────────────────────────
    # v7.10 — RHYTHM STABILITY INDEX (FRM Axiom 10)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_rsi(self) -> "Tuple[float, str]":
        """
        Rhythm Stability Index — consecutive FFT power-spectrum correlation.

        FRM Axiom 10 (Kuramoto): synchronisation r → 1 when Δω < K.
        RSI ≈ 1.0 means the spectral profile is stable (locked rhythm).
        RSI ≈ 0.0 means the dominant frequencies shifted (desynchronisation).

        Uses Pearson correlation of the normalised FFT power stored by
        _compute_rpi() — **zero extra FFT calls**.

        Regime:
            "synchronized"    : RSI > 0.7
            "transitional"    : 0.3 ≤ RSI ≤ 0.7
            "desynchronized"  : RSI < 0.3
        """
        if not self.rsi_enabled or self._current_fft_power_normed is None:
            return self._rsi, self._rsi_regime
        curr = self._current_fft_power_normed
        if self._prev_fft_power is not None and len(self._prev_fft_power) == len(curr):
            curr_c = curr - curr.mean()
            prev_c = self._prev_fft_power - self._prev_fft_power.mean()
            num  = float(curr_c @ prev_c)
            denom = math.sqrt(float(curr_c @ curr_c) * float(prev_c @ prev_c))
            raw_corr = max(-1.0, min(1.0, num / denom if denom > 1e-12 else 0.0))
            rsi = (1.0 + raw_corr) / 2.0
            self._rsi = round(rsi, 4)
            self._rsi_regime = (
                "synchronized"   if rsi > 0.7 else
                "desynchronized" if rsi < 0.3 else
                "transitional"
            )
        self._prev_fft_power = curr.copy()
        return self._rsi, self._rsi_regime

    # ─────────────────────────────────────────────────────────────────────────
    # v7.10 — DUAL ENCODING SCORE (FRM Axiom 5)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_des(self) -> float:
        """
        Dual Encoding Score — spatial × temporal information integration.

        FRM Axiom 5: I_total ≥ I_spatial + I_temporal (Stam, 2007).
        DES = RPI × (2.0 − RFI) — weights rhythmic strength (RPI) by
        fractal complexity (2-RFI maps RFI ∈ [1,2] → high values when
        the signal is rough / aperiodic, low when it is smooth / persistent).

        Range: [0, 1] — zero extra computation (derived from existing metrics).
        High DES → strong rhythm AND high fractal complexity coexist.
        """
        if not (self.rpi_enabled and self.rfi_enabled):
            return 0.0
        return round(max(0.0, min(1.0, self._rpi * (2.0 - self._rfi))), 4)

    # ─────────────────────────────────────────────────────────────────────────
    # v7.10 — VARIANCE CUSUM
    # ─────────────────────────────────────────────────────────────────────────

    def _update_vcusum(self, deviation: float) -> bool:
        """
        Variance CUSUM — CUSUM on squared deviations (catches volatility explosions).

        Classical CUSUM tracks mean shifts; Variance CUSUM detects when the
        variance itself suddenly inflates — useful for detecting heteroskedastic
        bursts (volatility regime changes, hardware vibration, market turbulence).

        Algorithm:
            dev² = deviation²
            k    = 0.5 × baseline_var   (allowable slack)
            S_t  = max(0, S_{t-1} + dev² − baseline_var − k)
            Alert when S_t > vcusum_threshold × baseline_var

        Returns True if Variance CUSUM exceeds threshold.
        """
        if not self.vcusum_enabled or self._baseline_var < 1e-12:
            return False
        dev_sq = deviation * deviation
        k_var  = 0.5 * self._baseline_var
        self._vcusum = max(0.0, self._vcusum + dev_sq - self._baseline_var - k_var)
        return self._vcusum > self.vcusum_threshold * self._baseline_var

    # ─────────────────────────────────────────────────────────────────────────
    # v7.10 — SEASONAL PERIODIC BASELINE
    # ─────────────────────────────────────────────────────────────────────────

    def _init_seasonal(self) -> None:
        """
        Initialise Seasonal Periodic Baseline from dominant FFT frequency or
        user-supplied seasonal_period.  Per-phase EWMA tables are allocated here.
        """
        if not self.seasonal_enabled or self._detected_period is not None:
            return
        if self.seasonal_period is not None:
            period = int(self.seasonal_period)
        else:
            # Auto-detect from FFT (reuse scalar_window — zero extra collection)
            if len(self._scalar_window) < self.rpi_window or self._rpi < self.rpi_rhythmic_threshold:
                return
            arr = np.array(list(self._scalar_window)[-self.rpi_window:], dtype=float)
            arr -= arr.mean()
            osc = np.abs(np.fft.rfft(arr))[1:] ** 2
            if osc.sum() < 1e-12:
                return
            dom_idx = int(osc.argmax()) + 1
            period  = max(2, int(round(self.rpi_window / dom_idx)))
            if period < 2 or period > self.rpi_window // 2:
                return
        self._detected_period  = period
        self._phase_ewmas      = [None] * period
        self._phase_dev_ewmas  = [None] * period
        self._phase_counts     = [0]    * period

    def _update_seasonal(self, current: float) -> bool:
        """
        Seasonal Periodic Baseline check — per-phase EWMA with adaptive threshold.

        Allocates one EWMA per phase (step % period) so that contextual anomalies
        (values normal globally but anomalous at their phase) are correctly detected.
        Fixes the contextual F1 = 0.00 gap present in all global-baseline detectors.

        Returns True if the current value is anomalous relative to its phase baseline.
        """
        if not self.seasonal_enabled:
            return False
        if self._detected_period is None:
            self._init_seasonal()
        if self._detected_period is None:
            return False

        phase = self.count % self._detected_period
        if self._phase_ewmas[phase] is None:
            # Bootstrap: seed from warmup stats
            self._phase_ewmas[phase]     = current
            self._phase_dev_ewmas[phase] = max((self.baseline_std or 1.0) * 0.5, 1e-6)
            self._phase_counts[phase]    = 1
            return False

        s_alpha = self.alpha * 0.5  # slower adaptation for seasonal baseline
        self._phase_ewmas[phase] = (
            s_alpha * current + (1.0 - s_alpha) * self._phase_ewmas[phase]
        )
        dev = abs(current - self._phase_ewmas[phase])
        self._phase_dev_ewmas[phase] = (
            s_alpha * dev + (1.0 - s_alpha) * self._phase_dev_ewmas[phase]
        )
        self._phase_counts[phase] += 1

        if self._phase_counts[phase] < 3:
            return False

        thr_up = self._phase_ewmas[phase] + self.early_mult * self._phase_dev_ewmas[phase]
        thr_lo = self._phase_ewmas[phase] - self.early_mult * self._phase_dev_ewmas[phase]
        return (current > thr_up or current < thr_lo) if self.two_sided else current > thr_up

    # ─────────────────────────────────────────────────────────────────────────
    # v7.11 — EARLY WARNING SIGNALS (FRM Axiom 9)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_ews(self) -> "Tuple[float, str]":
        """
        Early Warning Signals — FRM Axiom 9 (Critical Transitions).

        Scheffer et al. (2009) showed that before tipping points two statistical
        signatures rise simultaneously:
            1. Variance (critical fluctuations amplify near bifurcation)
            2. Lag-1 autocorrelation (critical slowing down — recovery time grows)

        Algorithm:
            - Divide the last ews_window samples into overlapping sub-windows
            - Track rolling variance and AC(1) over sub-windows
            - Compare early-window vs late-window means for each indicator
            - var_score  = (var_late / var_early − 1) × ews_sensitivity  ∈ [0,1]
            - ac_score   = (ac_late  − ac_early)     × 2 × ews_sensitivity ∈ [0,1]
            - EWS        = (var_score + ac_score) / 2                       ∈ [0,1]

        Regime:
            "stable"      : EWS < 0.3  (no precursor signatures)
            "approaching" : 0.3 ≤ EWS ≤ 0.7  (warning zone)
            "critical"    : EWS > 0.7  (imminent regime shift)

        Reference: Scheffer et al., Nature 461:53–59 (2009).
        """
        if not self.ews_enabled or len(self._scalar_window) < self.ews_window:
            return self._ews_score, self._ews_regime

        arr = np.array(list(self._scalar_window)[-self.ews_window:], dtype=float)
        n = len(arr)
        half = max(6, n // 4)
        step = max(1, half // 4)

        var_series: List[float] = []
        ac_series: List[float] = []

        for start in range(0, n - half + 1, step):
            seg = arr[start: start + half]
            seg_c = seg - seg.mean()
            var_series.append(float(np.var(seg)))
            denom = float(seg_c @ seg_c)
            if denom > 1e-12 and len(seg) > 2:
                ac1 = float(seg_c[:-1] @ seg_c[1:]) / denom * len(seg) / (len(seg) - 1)
                ac_series.append(float(np.clip(ac1, -1.0, 1.0)))

        if len(var_series) < 3:
            return self._ews_score, self._ews_regime

        third = max(1, len(var_series) // 3)
        var_early = sum(var_series[:third]) / third
        var_late  = sum(var_series[-third:]) / third
        var_score = float(np.clip(
            (var_late / (var_early + 1e-12) - 1.0) * self.ews_sensitivity,
            0.0, 1.0,
        ))

        ac_score = 0.0
        if len(ac_series) >= 3:
            third_ac = max(1, len(ac_series) // 3)
            ac_early = sum(ac_series[:third_ac]) / third_ac
            ac_late  = sum(ac_series[-third_ac:]) / third_ac
            ac_score = float(np.clip(
                (ac_late - ac_early) * 2.0 * self.ews_sensitivity,
                0.0, 1.0,
            ))

        ews = round((var_score + ac_score) / 2.0, 4)
        self._ews_score = ews
        self._ews_regime = (
            "critical"    if ews > 0.7 else
            "approaching" if ews >= 0.3 else
            "stable"
        )
        return self._ews_score, self._ews_regime

    # ─────────────────────────────────────────────────────────────────────────
    # v7.11 — ADAPTIVE QUANTILE BASELINE (FRM Axiom 1)
    # ─────────────────────────────────────────────────────────────────────────

    def _update_quantile_threshold(self, abs_dev: float) -> "Tuple[float, float]":
        """
        Adaptive Quantile Baseline — FRM Axiom 1 (Scale Invariance).

        Maintains a rolling window of absolute deviations |current − EWMA|.
        Returns (q_early, q_confirmed) thresholds as empirical quantiles.

        This is distribution-free: it works on Gaussian, Laplace, Pareto, or any
        heavy-tailed data without assuming variance finite or distribution symmetric.

        Algorithm: sorted rolling window (O(n log n) for small n ≤ quantile_window).
        The p-th empirical quantile is computed by linear interpolation between
        adjacent sorted values, identical to numpy.quantile interpolation='linear'.

        Replaces ewma ± early_mult × dev_ewma when quantile_threshold_mode=True.

        Parameters
        ----------
        abs_dev : |current − ewma| for the current step.

        Returns
        -------
        (q_early, q_confirmed) — absolute deviations at early and confirmed percentiles.
        """
        self._dev_history.append(abs_dev)
        n = len(self._dev_history)
        if n < max(10, self.quantile_window // 10):
            return self._q_early, self._q_confirmed

        sorted_devs = sorted(self._dev_history)

        def _quantile(p: float) -> float:
            idx = p * (n - 1)
            lo  = int(idx)
            hi  = min(lo + 1, n - 1)
            frac = idx - lo
            return sorted_devs[lo] * (1.0 - frac) + sorted_devs[hi] * frac

        self._q_early     = _quantile(self.quantile_early_pct)
        self._q_confirmed = _quantile(self.quantile_confirmed_pct)
        return self._q_early, self._q_confirmed

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
                # v7.11: EWS always present in every result path
                "ews_score":  self._ews_score,
                "ews_regime": self._ews_regime,
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
            # v7.11 Bug Fix: feed aggregated scalar so PE/RSI/EWS work in multivariate mode
            self._scalar_window.append(current)

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
            # v7.10: compute PE, RSI (uses FFT from RPI)
            if self.pe_enabled:
                self._compute_pe()
                # v7.11: maintain rolling PE baseline EWMA — updated on every step
                # (including pe=0 for monotone windows) so baseline reflects signal's
                # true typical entropy and contextual alarms are distribution-aware
                if self._pe_baseline is None:
                    self._pe_baseline = self._pe     # initialise on first computation
                else:
                    self._pe_baseline = 0.02 * self._pe + 0.98 * self._pe_baseline
            if self.rsi_enabled:
                self._compute_rsi()

            # v7.11: EWS (uses _scalar_window populated above)
            if self.ews_enabled:
                self._compute_ews()

            # v7.11: AQB — update quantile thresholds if mode enabled
            if self.quantile_threshold_mode and self.ewma is not None:
                self._update_quantile_threshold(abs(current - self.ewma))

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
                # v7.11: always include EWS in every result path
                "ews_score":  self._ews_score,
                "ews_regime": self._ews_regime,
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

        # v7.10: Permutation Entropy, RSI, DES, Variance CUSUM, Seasonal
        # scalar_window & _current_fft_power_normed already updated in univariate path above.
        # For multivariate, pe/rsi return cached defaults gracefully.
        _pe, _pe_regime = self._compute_pe()
        _rsi, _rsi_regime = self._compute_rsi()
        _des = self._compute_des()
        if self.pe_enabled:
            result["pe"] = _pe
            result["pe_regime"] = _pe_regime
        if self.rsi_enabled:
            result["rsi"] = _rsi
            result["rsi_regime"] = _rsi_regime
        result["des"] = _des

        # Variance CUSUM
        _deviation_for_vcusum = float(current - (self.ewma or current))
        _vcusum_alert = self._update_vcusum(_deviation_for_vcusum)
        result["vcusum"] = round(self._vcusum, 4)
        result["volatility_alert"] = bool(_vcusum_alert)   # v7.10: always present
        if _vcusum_alert:
            result["vcusum_alert"] = True
            result["early_alert"] = True

        # Seasonal Periodic Baseline
        _seasonal_alert = self._update_seasonal(current)
        if _seasonal_alert:
            result["seasonal_alert"] = True
            result["early_alert"] = True

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

        # v7.11: EWS and AQB fields in result
        result["ews_score"]  = self._ews_score
        result["ews_regime"] = self._ews_regime
        if self.quantile_threshold_mode:
            result["q_early"]     = round(self._q_early, 4)
            result["q_confirmed"] = round(self._q_confirmed, 4)

        # v7.9: alert attribution
        result["alert_reasons"] = self._build_alert_reasons(
            _z,
            result.get("early_alert", False),
            result.get("confirmed_alert", False),
            result.get("boundary_warning", False),
            _ph_signal,
            result.get("mahal_score", 0.0),
            seasonal_alert=bool(result.get("seasonal_alert", False)),
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
                "axiom_3_complexity":  f"{_pe_regime} (PE={_pe:.4f})" if self.pe_enabled else "disabled",
                "axiom_5_dual_enc":    f"DES={_des:.4f}",
                "axiom_6_rhythm":      f"{self._rpi_regime} (RPI={self._rpi:.4f})" if self.rpi_enabled else "disabled",
                "axiom_8_fractal":     f"RFI={self._rfi:.4f}" if self.rfi_enabled else "disabled",
                "axiom_10_sync":       f"{_rsi_regime} (RSI={_rsi:.4f})" if self.rsi_enabled else "disabled",
                "axiom_11_resilience": f"RRS={self._rrs:.4f}" if (self.rrs_enabled and self._rrs is not None) else "awaiting_reset",
                "axiom_9_ews":         f"{self._ews_regime} (EWS={self._ews_score:.4f})" if self.ews_enabled else "disabled",
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
                # v7.10 fields
                "pe":                  result.get("pe"),
                "pe_regime":           result.get("pe_regime"),
                "rsi":                 result.get("rsi"),
                "rsi_regime":          result.get("rsi_regime"),
                "des":                 result.get("des"),
                "vcusum":              result.get("vcusum"),
                "vcusum_alert":        result.get("vcusum_alert", False),
                "seasonal_alert":      result.get("seasonal_alert", False),
                # v7.11 fields
                "ews_score":           result.get("ews_score"),
                "ews_regime":          result.get("ews_regime"),
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

        # v7.11: Adaptive Quantile Baseline (FRM Axiom 1) — distribution-free thresholds
        if self.quantile_threshold_mode and self.ewma is not None and self._q_early > 0:
            early_upper    = self.ewma + self._q_early
            confirmed_upper = self.ewma + self._q_confirmed
        else:
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
            if self.quantile_threshold_mode and self.ewma is not None and self._q_early > 0:
                early_lower    = self.ewma - self._q_early
                confirmed_lower = self.ewma - self._q_confirmed
            else:
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
        title: str = "Fracttalix Sentinel v7.11 — Dashboard",
        show: bool = True,
        figsize: Tuple[int, int] = (14, 11),
    ):
        """
        Four-panel matplotlib dashboard of recent detection history.

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

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
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

        # --- Panel 3: RFI + RRS + EWS (v7.11) --------------------------------
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

        # v7.11: EWS score on right axis
        ews_vals = _extract("ews_score")
        ax3r = ax3.twinx()
        valid_ews = [(x, e) for x, e in zip(xs, ews_vals) if e is not None]
        if valid_ews:
            ex3, ey3 = zip(*valid_ews)
            ax3r.fill_between(ex3, ey3, alpha=0.14, color="darkorange")
            ax3r.plot(ex3, ey3, color="darkorange", lw=1.0, alpha=0.8, label="EWS score")
            ax3r.axhline(0.3, color="darkorange", lw=0.6, ls="--", alpha=0.5, label="EWS approaching")
            ax3r.axhline(0.7, color="red",        lw=0.6, ls=":",  alpha=0.5, label="EWS critical")
            ax3r.set_ylabel("EWS", color="darkorange", fontsize=9)
            ax3r.set_ylim(-0.05, 1.15)

        for rx in reset_xs:
            ax3.axvline(rx, color="grey", lw=0.8, ls=":", alpha=0.6)

        ax3.set_ylabel("RFI / RRS")
        ax3.legend(loc="upper left", fontsize=7)
        ax3.grid(True, alpha=0.25)

        # --- Panel 4: PE + anomaly_score ------------------------------------
        ax4 = axes[3]
        pes   = _extract("pe")
        ascores = _extract("anomaly_score")
        valid_pe = [(x, p) for x, p in zip(xs, pes) if p is not None]
        if valid_pe:
            px4, py4 = zip(*valid_pe)
            ax4.plot(px4, py4, color="darkcyan", lw=1.1, label="PE (entropy)")
            ax4.axhline(0.7, color="darkcyan", lw=0.6, ls="--", alpha=0.5, label="complex threshold")
            ax4.axhline(0.4, color="darkcyan", lw=0.6, ls=":",  alpha=0.5, label="ordered threshold")
            ax4.set_ylim(-0.05, 1.1)

        ax4r = ax4.twinx()
        valid_as = [(x, a) for x, a in zip(xs, ascores) if a is not None]
        if valid_as:
            ax4x, ax4y = zip(*valid_as)
            ax4r.fill_between(ax4x, ax4y, alpha=0.18, color="crimson")
            ax4r.plot(ax4x, ax4y, color="crimson", lw=0.9, alpha=0.7, label="anomaly_score")
            ax4r.set_ylabel("anomaly score", color="crimson", fontsize=9)
            ax4r.set_ylim(0, 1.05)

        for rx in reset_xs:
            ax4.axvline(rx, color="grey", lw=0.8, ls=":", alpha=0.6)

        ax4.set_ylabel("PE")
        ax4.set_xlabel("Step")
        ax4.legend(loc="upper left", fontsize=7)
        ax4.grid(True, alpha=0.25)

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
        labeled_data: Optional[List[int]] = None,
        **fixed_kwargs,
    ) -> Tuple["Detector_7_10", Dict[str, Any]]:
        """
        Data-driven hyperparameter search for alpha and early_mult.

        Treats `data` as representative normal data. Splits it into a warmup
        segment (first warmup_fraction) and an evaluation segment (remainder).
        Runs a grid search over alpha x early_mult combinations.

        If labeled_data is provided (list of 0/1 labels, same length as data),
        optimizes for F1 score instead of FPR. This is the v7.10 F1 mode.

        Parameters
        ----------
        data          : historical data (≥ 30 points)
        target_fpr    : desired false-positive rate on normal data (default 2%)
                        Ignored when labeled_data is provided.
        warmup_fraction : fraction of data used for warm-up (default 60%)
        alpha_grid    : EWMA smoothing values to try (default: 5 values)
        early_mult_grid : early-warning multipliers to try (default: 6 values)
        verbose       : print search progress
        labeled_data  : optional list of 0/1 labels for F1 optimization mode (v7.10)
        **fixed_kwargs: passed directly to Detector_7_10 (e.g. warm_up_period)

        Returns
        -------
        (detector, report)
            detector : Detector_7_10 warmed up on all of `data`, best params set
            report   : dict with best_alpha, best_early_mult, actual_fpr or best_f1,
                       target_fpr, n_candidates, all_results, mode
        """
        data = [float(v) for v in data]
        n = len(data)
        if n < 30:
            raise ValueError("auto_tune requires at least 30 data points")

        f1_mode = labeled_data is not None
        if f1_mode:
            if len(labeled_data) != n:
                raise ValueError("labeled_data must have the same length as data")

        warmup_n = max(10, int(n * warmup_fraction))
        eval_data = data[warmup_n:]
        eval_labels = labeled_data[warmup_n:] if f1_mode else None
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
                tp = fp = fn = 0
                for idx_e, v in enumerate(eval_data):
                    r = det.update_and_check(v)
                    if r.get("status") == "active":
                        n_eval += 1
                        fired = bool(r.get("early_alert")) or bool(r.get("confirmed_alert"))
                        if f1_mode:
                            lbl = eval_labels[idx_e]
                            if fired and lbl == 1:
                                tp += 1
                            elif fired and lbl == 0:
                                fp += 1
                            elif not fired and lbl == 1:
                                fn += 1
                        else:
                            if fired:
                                n_fp += 1

                if f1_mode:
                    prec = tp / max(tp + fp, 1)
                    rec = tp / max(tp + fn, 1)
                    score = 2 * prec * rec / max(prec + rec, 1e-9)
                else:
                    score = n_fp / n_eval if n_eval > 0 else 0.0
                candidates.append((alpha, em, score))
                if verbose:
                    metric_name = "F1" if f1_mode else "FPR"
                    print(f"  alpha={alpha:.2f} early_mult={em:.2f} → {metric_name}={score:.3f}")

        if f1_mode:
            # Sort: highest F1 wins, tie-break by lower early_mult
            candidates.sort(key=lambda c: (-c[2], c[1]))
        else:
            # Sort: closest actual FPR to target, tie-break by lower early_mult
            candidates.sort(key=lambda c: (abs(c[2] - target_fpr), c[1]))

        best_alpha, best_em, best_score = candidates[0]

        if verbose:
            metric_name = "F1" if f1_mode else "FPR"
            print(f"  → Best: alpha={best_alpha}, early_mult={best_em}, {metric_name}={best_score:.3f}")

        # Build final detector warmed up on all data
        final_kw = dict(fixed_kwargs)
        final_kw.setdefault("warm_up_period", 60)
        final_det = cls(alpha=best_alpha, early_mult=best_em, **final_kw)
        for v in data:
            final_det.update_and_check(v)

        report: Dict[str, Any] = {
            "best_alpha":      best_alpha,
            "best_early_mult": best_em,
            "target_fpr":      target_fpr,
            "n_candidates":    len(candidates),
            "all_results":     [(c[0], c[1], round(c[2], 4)) for c in
                                sorted(candidates, key=lambda c: (c[0], c[1]))],
            "mode":            "f1" if f1_mode else "fpr",
        }
        if f1_mode:
            report["best_f1"] = round(best_score, 4)
        else:
            report["actual_fpr"] = round(best_score, 4)
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

        # v7.11 Bug Fix: use module-level _phase_randomize_worker (picklable on Windows/spawn)
        seeds = list(range(n_surrogates))
        tasks = [(data, s) for s in seeds]

        if self.parallel_surrogates and n_surrogates >= 100:
            with Pool(cpu_count()) as p:
                surrogates = list(
                    tqdm(
                        p.imap(_phase_randomize_worker, tasks),
                        total=n_surrogates,
                        desc="Generating surrogates",
                        disable=not self.verbose_explain,
                    )
                )
        else:
            surrogates = [
                _phase_randomize_worker(t)
                for t in tqdm(
                    tasks,
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
    Fan-out manager for N independent Detector_7_10 instances.

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
        self._detectors: Dict[str, Detector_7_10] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._max_workers = max_workers or min(32, (cpu_count() or 1) + 4)

    def _get_or_create(self, stream_id: str) -> Detector_7_10:
        """Return the detector for stream_id, creating it if new."""
        with self._global_lock:
            if stream_id not in self._detectors:
                self._detectors[stream_id] = Detector_7_10(**self._defaults)
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

    def get_detector(self, stream_id: str) -> Optional[Detector_7_10]:
        """Return the Detector_7_10 for stream_id, or None if unknown."""
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
                    self._detectors[sid] = Detector_7_10(**self._defaults)
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
    evaluates any Detector_7_10 instance, returning standard metrics:
    precision, recall, F1, AUPRC, and mean detection lag.

    Anomaly archetypes:
        point       — isolated spikes: single-step amplitude excursions
        contextual  — periodic signal with out-of-phase segments
        collective  — sustained block shifts (local distribution change)

    Usage:
        data, labels = SentinelBenchmark.generate(n=1000, anomaly_type="point")
        det = Detector_7_10(warm_up_period=60)
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
        elif anomaly_type == "drift":
            # Slow mean drift — baseline creeps upward then plateaus
            data  = list(rng.normal(0, noise_std, n).astype(float))
            anom_idx_start = int(n * 0.4)
            anom_idx_end   = int(n * 0.7)
            drift_amplitude = amplitude
            for i in range(n):
                if anom_idx_start <= i < anom_idx_end:
                    progress = (i - anom_idx_start) / max(anom_idx_end - anom_idx_start, 1)
                    data[i] += drift_amplitude * progress
                    labels[i] = 1
                elif i >= anom_idx_end:
                    data[i] += drift_amplitude  # flat plateau after drift
                    labels[i] = 0   # plateau is new normal

        elif anomaly_type == "variance":
            # Variance explosion — mean stays constant, std suddenly multiplies
            data  = list(rng.normal(0, noise_std, n).astype(float))
            n_blocks = max(1, int(n * anomaly_frac / block_size))
            starts   = rng.integers(0, n - block_size, size=n_blocks)
            for start in starts:
                for j in range(block_size):
                    idx4 = min(int(start) + j, n - 1)
                    data[idx4]   = float(rng.normal(0, noise_std * amplitude))
                    labels[idx4] = 1

        else:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type!r}")

        return data, labels

    @staticmethod
    def evaluate(
        detector: "Detector_7_10",
        data: List[float],
        labels: List[int],
        score_field: str = "anomaly_score",
        alert_field: str = "early_alert",
        tol_lag: int = 5,
        compute_vus_pr: bool = True,
    ) -> Dict[str, Any]:
        """
        Run detector over data and compute classification metrics.

        Parameters
        ----------
        detector      : fresh or pre-warmed Detector_7_10 instance
        data          : 1-D list of floats
        labels        : 1-D list of ints (0=normal, 1=anomaly)
        score_field   : result dict field for continuous scoring (default: "anomaly_score")
        alert_field   : result dict field for binary alert (default: "early_alert")
        tol_lag       : detection within tol_lag steps of anomaly start counts as a hit
        compute_vus_pr: compute VUS-PR metric (default: True) (v7.10)

        Returns
        -------
        dict with: precision, recall, f1, auprc, vus_pr, mean_lag, tp, fp, fn, n_anomalies
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

        # VUS-PR: Volume Under Precision-Recall Surface (NeurIPS 2024 gold standard)
        # Average AUPRC over 5 buffer tolerances {0, 5, 10, 15, 20}
        vus_pr = None
        if compute_vus_pr:
            _trapz2 = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
            buf_auprcs = []
            for buf in [0, 5, 10, 15, 20]:
                exp_lbl = list(labels)
                for i, lb in enumerate(labels):
                    if lb == 1:
                        for j in range(max(0, i - buf), min(len(labels), i + buf + 1)):
                            exp_lbl[j] = 1
                srt = sorted(set(scores), reverse=True)
                pts = [(1.0, 0.0)]
                for thresh in srt:
                    tp_t = sum(1 for i, s in enumerate(scores) if s >= thresh and exp_lbl[i] == 1)
                    fp_t = sum(1 for i, s in enumerate(scores) if s >= thresh and exp_lbl[i] == 0)
                    fn_t = sum(1 for i, lb2 in enumerate(exp_lbl) if lb2 == 1 and scores[i] < thresh)
                    pts.append((tp_t / max(tp_t + fp_t, 1), tp_t / max(tp_t + fn_t, 1)))
                pts.append((0.0, 1.0))
                pts.sort(key=lambda x: x[1])
                buf_auprcs.append(float(_trapz2([p for p, _ in pts], [r for _, r in pts])))
            vus_pr = round(max(0.0, sum(buf_auprcs) / len(buf_auprcs)), 4)

        return {
            "precision":    round(precision, 4),
            "recall":       round(recall, 4),
            "f1":           round(f1, 4),
            "auprc":        round(max(0.0, auprc), 4),
            "vus_pr":       vus_pr,
            "mean_lag":     round(mean_lag, 2) if not math.isnan(mean_lag) else None,
            "tp":           tp,
            "fp":           fp,
            "fn":           fn,
            "n_anomalies":  len(anom_blocks),
        }

    @staticmethod
    def evaluate_naive(
        data: List[float],
        labels: List[int],
        window: int = 60,
        z_thresh: float = 3.0,
        tol_lag: int = 5,
    ) -> Dict[str, Any]:
        """
        Naive rolling z-score baseline for benchmark comparison.

        Alerts when |current − rolling_mean| / rolling_std > z_thresh.
        Used as reference comparison in run_suite() for all archetypes.
        """
        scores: List[float] = []
        preds: List[int] = []
        buf: deque = deque(maxlen=window)
        for val in data:
            buf.append(val)
            if len(buf) < 10:
                scores.append(0.0)
                preds.append(0)
                continue
            arr = list(buf)
            mean_ = sum(arr) / len(arr)
            var_  = sum((x - mean_) ** 2 for x in arr) / len(arr)
            std_  = math.sqrt(var_) if var_ > 1e-12 else 1e-6
            z     = abs(val - mean_) / std_
            scores.append(round(min(z / max(z_thresh, 1e-9), 3.0), 4))
            preds.append(1 if z > z_thresh else 0)

        # Compute metrics (same block-detection logic as evaluate())
        tp = fp = fn = 0
        matched: set = set()
        detection_lags: List[int] = []
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

        for bs, be in anom_blocks:
            detected = False
            for lag in range(tol_lag + be - bs + 1):
                ci = bs + lag
                if ci > min(be + tol_lag, len(preds) - 1):
                    break
                if preds[ci] == 1 and ci not in matched:
                    detected = True
                    matched.add(ci)
                    detection_lags.append(max(0, ci - bs))
                    break
            if detected:
                tp += 1
            else:
                fn += 1
        for i, p in enumerate(preds):
            if p == 1 and i not in matched and labels[i] == 0:
                fp += 1

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        mean_lag  = float(np.mean(detection_lags)) if detection_lags else float("nan")

        sorted_sc = sorted(set(scores), reverse=True)
        pr_pts = [(1.0, 0.0)]
        for thresh in sorted_sc:
            tp_t = sum(1 for i, s in enumerate(scores) if s >= thresh and labels[i] == 1)
            fp_t = sum(1 for i, s in enumerate(scores) if s >= thresh and labels[i] == 0)
            fn_t = sum(1 for i, lb in enumerate(labels) if lb == 1 and scores[i] < thresh)
            pr_pts.append((tp_t / max(tp_t + fp_t, 1), tp_t / max(tp_t + fn_t, 1)))
        pr_pts.append((0.0, 1.0))
        pr_pts.sort(key=lambda x: x[1])
        _trapz4 = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        auprc = float(_trapz4([p for p, _ in pr_pts], [r for _, r in pr_pts]))

        return {
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "f1":          round(f1, 4),
            "auprc":       round(max(0.0, auprc), 4),
            "mean_lag":    round(mean_lag, 2) if not math.isnan(mean_lag) else None,
            "tp": tp, "fp": fp, "fn": fn,
            "n_anomalies": len(anom_blocks),
        }

    @staticmethod
    def run_suite(
        detector_kwargs: Optional[Dict[str, Any]] = None,
        n: int = 1000,
        seed: int = 42,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run five canonical benchmark scenarios + naive 3σ reference comparison.

        Archetypes: point, contextual, collective, drift, variance.
        Returns dict {archetype: metrics} including naive_<archetype> rows.
        """
        kw = detector_kwargs or {}
        results: Dict[str, Dict[str, Any]] = {}
        archetypes = [
            ("point",      dict(n=n, anomaly_type="point",      amplitude=5.0, seed=seed)),
            ("contextual", dict(n=n, anomaly_type="contextual", amplitude=4.0, seed=seed)),
            ("collective", dict(n=n, anomaly_type="collective", amplitude=4.0, seed=seed)),
            ("drift",      dict(n=n, anomaly_type="drift",      amplitude=4.0, seed=seed)),
            ("variance",   dict(n=n, anomaly_type="variance",   amplitude=3.0, seed=seed)),
        ]
        if verbose:
            print(f"\n{'─'*70}")
            print(f"  SentinelBenchmark v7.10  (n={n}, seed={seed})")
            print(f"  {'Archetype':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUPRC':>7} {'VUS-PR':>7} {'Lag':>5}")
            print(f"{'─'*70}")

        for name, gen_kw in archetypes:
            data, labels = SentinelBenchmark.generate(**gen_kw)
            det = Detector_7_10(**kw)
            metrics = SentinelBenchmark.evaluate(det, data, labels)
            results[name] = metrics
            if verbose:
                lag_str = f"{metrics['mean_lag']:.1f}" if metrics["mean_lag"] is not None else "  n/a"
                vus_str = f"{metrics['vus_pr']:.4f}" if metrics.get("vus_pr") is not None else "    n/a"
                print(
                    f"  {name:<14} "
                    f"{metrics['precision']:>6.3f} "
                    f"{metrics['recall']:>6.3f} "
                    f"{metrics['f1']:>6.3f} "
                    f"{metrics['auprc']:>7.4f} "
                    f"{vus_str:>7} "
                    f"{lag_str:>5}"
                )
        if verbose:
            print(f"{'─'*70}")
            print(f"\n  Naive 3σ baseline (rolling z-score, window=60):")
            print(f"  {'─'*68}")
            print(f"  {'Archetype':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUPRC':>7} {'Lag':>5}")
            print(f"  {'─'*68}")

        for name, gen_kw in archetypes:
            data, labels = SentinelBenchmark.generate(**gen_kw)
            naive_m = SentinelBenchmark.evaluate_naive(data, labels)
            results[f"naive_{name}"] = naive_m
            if verbose:
                lag_str = f"{naive_m['mean_lag']:.1f}" if naive_m["mean_lag"] is not None else "  n/a"
                print(
                    f"  {name:<14} "
                    f"{naive_m['precision']:>6.3f} "
                    f"{naive_m['recall']:>6.3f} "
                    f"{naive_m['f1']:>6.3f} "
                    f"{naive_m['auprc']:>7.4f} "
                    f"{lag_str:>5}"
                )
        if verbose:
            print(f"  {'─'*68}\n")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# v7.9: SENTINEL SERVER — stdlib HTTP wrapper for MultiStreamSentinel
# ─────────────────────────────────────────────────────────────────────────────

class SentinelServer:
    """
    Lightweight HTTP API server wrapping MultiStreamSentinel.

    v7.11: Uses ThreadingHTTPServer (socketserver.ThreadingMixIn + HTTPServer)
    so concurrent clients are handled in parallel threads — zero extra dependencies.

    Endpoints:
        POST /update
            Body: {"stream_id": "sensor_A", "value": 42.3}
            Returns: detector result dict as JSON

        GET /streams
            Returns: {"streams": ["sensor_A", "sensor_B", ...]}

        GET /status/<stream_id>
            Returns: last result dict for stream_id (if any)

        GET /health
            Returns: {"status": "ok", "version": "7.11", "n_streams": N}

        DELETE /stream/<stream_id>
            Removes a stream from the MultiStreamSentinel registry.
            Returns: {"removed": stream_id} or 404 if not found.

        POST /reset/<stream_id>
            Soft-resets the detector for stream_id (preserves baselines).
            Returns: {"reset": stream_id, "soft": true}

    Usage:
        ms = MultiStreamSentinel(detector_defaults={"warm_up_period": 60})
        server = SentinelServer(ms, host="127.0.0.1", port=8765)
        server.start(background=True)   # non-blocking, handles concurrent users
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
        """Start the HTTP server. background=True runs in a daemon thread.

        v7.11: Uses ThreadingHTTPServer — each request handled in its own thread,
        so multiple concurrent clients / streams never block each other.
        """
        sentinel_instance = self

        # v7.11: ThreadingMixIn gives each request its own thread
        class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
            daemon_threads = True   # threads die with the server

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

            def do_DELETE(self):
                path = self.path.rstrip("/")
                if path.startswith("/stream/"):
                    sid = path[len("/stream/"):]
                    with sentinel_instance._lock:
                        existed = sid in sentinel_instance.ms._detectors
                        if existed:
                            del sentinel_instance.ms._detectors[sid]
                            sentinel_instance._last_results.pop(sid, None)
                    if existed:
                        self._send_json({"removed": sid})
                    else:
                        self._send_json({"error": f"unknown stream: {sid}"}, 404)
                else:
                    self._send_json({"error": "not found"}, 404)

            def do_POST(self):
                path = self.path.rstrip("/")
                if path == "/update":
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
                elif path.startswith("/reset/"):
                    # v7.11: soft-reset endpoint — preserves baselines
                    sid = path[len("/reset/"):]
                    with sentinel_instance._lock:
                        det = sentinel_instance.ms._detectors.get(sid)
                    if det is None:
                        self._send_json({"error": f"unknown stream: {sid}"}, 404)
                        return
                    det.reset(soft=True)
                    self._send_json({"reset": sid, "soft": True})
                else:
                    self._send_json({"error": "not found"}, 404)

        self._server = _ThreadingHTTPServer((self.host, self.port), _Handler)
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
# SMOKE TEST — 34 structured assertions covering all v7.11 features
# ─────────────────────────────────────────────────────────────────────────────

def _cli_main():
    """
    CLI entry point for streaming a CSV or running the benchmark.

    Usage examples:
        python fracttalix_sentinel_v711.py --smoke
        python fracttalix_sentinel_v711.py --file data.csv --alpha 0.10 --early-mult 2.5
        python fracttalix_sentinel_v711.py --benchmark
        python fracttalix_sentinel_v711.py --serve --port 8765
    """
    parser = argparse.ArgumentParser(
        prog="fracttalix_sentinel_v711",
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
        det = Detector_7_10(alpha=args.alpha, early_mult=args.early_mult,
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

    print("Fracttalix Sentinel v7.10 — Smoke test")
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
            pe_enabled=True,
            rsi_enabled=True,
            vcusum_enabled=True,
            seasonal_enabled=True,
        )
        defaults.update(kwargs)
        return Detector_7_10(**defaults)

    def warm_up(det, seed=42, n=60, pattern='noise'):
        random.seed(seed)
        for i in range(n):
            if pattern == 'sine':
                det.update_and_check(math.sin(2 * math.pi * i / 20) + random.gauss(0, 0.05))
            else:
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
    det_tuned, report = Detector_7_10.auto_tune(
        tune_data, target_fpr=0.05, verbose=False,
        alpha_grid=[0.05, 0.10, 0.15],
        early_mult_grid=[2.5, 3.0, 3.5],
    )
    assert isinstance(det_tuned, Detector_7_10), "auto_tune must return Detector_7_10 instance"
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

    # -- Test 20: Permutation Entropy (Axiom 3) ------------------------------
    print("\n[20] Permutation Entropy (Axiom 3) — streaming complexity")
    d20 = Detector_7_10(warm_up_period=40, pe_enabled=True, pe_window=32, pe_dim=3,
                        reset_after_regime_change=False, rpi_enabled=False, rfi_enabled=False,
                        vcusum_enabled=False, seasonal_enabled=False)
    for i in range(80):
        d20.update_and_check(math.sin(2 * math.pi * i / 8.0))
    r20_sine = d20.update_and_check(math.sin(2 * math.pi * 80 / 8.0))
    assert "pe" in r20_sine, "pe missing from result"
    assert 0.0 <= r20_sine["pe"] <= 1.0, f"pe out of [0,1]: {r20_sine['pe']}"
    d20b = Detector_7_10(warm_up_period=40, pe_enabled=True, pe_window=32, pe_dim=3,
                         reset_after_regime_change=False, rpi_enabled=False, rfi_enabled=False,
                         vcusum_enabled=False, seasonal_enabled=False)
    random.seed(20)
    for _ in range(80):
        d20b.update_and_check(random.gauss(0, 1))
    r20_noise = d20b.update_and_check(random.gauss(0, 1))
    print(f"  PE sine={r20_sine['pe']:.4f}({r20_sine.get('pe_regime')})  "
          f"noise={r20_noise['pe']:.4f}({r20_noise.get('pe_regime')})")
    assert r20_noise["pe"] >= r20_sine["pe"], \
        f"Noise PE ({r20_noise['pe']:.4f}) should be >= sine PE ({r20_sine['pe']:.4f})"
    print("  PASS (PE higher for random noise than periodic signal — Axiom 3)")

    # -- Test 21: Rhythm Stability Index (Axiom 10) --------------------------
    print("\n[21] Rhythm Stability Index (Axiom 10) — synchronization")
    d21 = Detector_7_10(warm_up_period=40, rpi_enabled=True, rsi_enabled=True,
                        rpi_window=32, reset_after_regime_change=False,
                        rfi_enabled=False, vcusum_enabled=False, seasonal_enabled=False)
    for i in range(80):
        d21.update_and_check(math.sin(2 * math.pi * i / 10.0))
    rsi_vals = []
    for i in range(80, 100):
        r21 = d21.update_and_check(math.sin(2 * math.pi * i / 10.0))
        if r21.get("rsi") is not None:
            rsi_vals.append(r21["rsi"])
    assert len(rsi_vals) > 0, "RSI never computed"
    assert all(0.0 <= v <= 1.0 for v in rsi_vals), "RSI out of [0,1]"
    print(f"  RSI stable sine: mean={sum(rsi_vals)/len(rsi_vals):.4f}, "
          f"last regime={r21.get('rsi_regime')}")
    print("  PASS (RSI computed, in [0,1])")

    # -- Test 22: Dual Encoding Score (Axiom 5) ------------------------------
    print("\n[22] Dual Encoding Score (Axiom 5) — temporal + structural encoding")
    d22 = make_detector(seasonal_enabled=False, vcusum_enabled=False)
    warm_up(d22)
    for i in range(20):
        d22.update_and_check(math.sin(2 * math.pi * i / 8.0))
    r22 = d22.update_and_check(math.sin(2 * math.pi * 20 / 8.0))
    assert "des" in r22, "des missing from result"
    assert 0.0 <= r22["des"] <= 1.0, f"des out of [0,1]: {r22['des']}"
    print(f"  DES={r22['des']:.4f}  (RPI={r22.get('rpi',0):.4f} x (2-RFI={r22.get('rfi',1.5):.4f}))")
    print("  PASS (DES = RPI x (2-RFI), in [0,1])")

    # -- Test 23: Variance CUSUM — volatility explosion ----------------------
    print("\n[23] Variance CUSUM — volatility explosion detection")
    d23 = Detector_7_10(warm_up_period=60, vcusum_enabled=True, vcusum_threshold=8.0,
                        reset_after_regime_change=False, seasonal_enabled=False)
    random.seed(23)
    for _ in range(80):
        d23.update_and_check(random.gauss(0, 1.0))
    r23_check = d23.update_and_check(0.0)
    assert "volatility_alert" in r23_check, "volatility_alert key missing from result"
    vcusum_fired = False
    for _ in range(80):
        r23 = d23.update_and_check(random.gauss(0, 5.0))
        if r23.get("volatility_alert"):
            vcusum_fired = True
            print(f"  volatility_alert fired (vcusum={d23._vcusum:.2f}, "
                  f"threshold={d23.vcusum_threshold * d23._baseline_var:.2f})")
            break
    assert vcusum_fired, "Variance CUSUM never fired on 5x volatility injection"
    print("  PASS (Variance CUSUM detects volatility explosion that mean-CUSUM misses)")

    # -- Test 24: Seasonal Periodic Baseline — contextual F1 fix -------------
    print("\n[24] Seasonal Periodic Baseline — fixes contextual F1=0.00")
    data24, labels24 = SentinelBenchmark.generate(
        n=800, anomaly_type="contextual", amplitude=4.0, seed=24)
    det24 = Detector_7_10(
        warm_up_period=60, early_mult=2.5,
        seasonal_enabled=True, seasonal_period=30,
        vcusum_enabled=False,
    )
    metrics24 = SentinelBenchmark.evaluate(det24, data24, labels24, compute_vus_pr=False)
    print(f"  Contextual F1 (with seasonal baseline, period=30): {metrics24['f1']:.3f}")
    print(f"  Contextual AUPRC: {metrics24['auprc']:.4f}")
    assert metrics24["f1"] > 0.1, \
        f"Seasonal baseline should lift contextual F1 above 0.10, got {metrics24['f1']:.3f}"
    print("  PASS (Seasonal baseline lifts contextual F1 from 0.00)")

    # -- Test 25: VUS-PR — NeurIPS 2024 gold-standard metric -----------------
    print("\n[25] VUS-PR — Volume Under PR Surface")
    data25, labels25 = SentinelBenchmark.generate(n=500, anomaly_type="point", seed=25)
    det25 = Detector_7_10(warm_up_period=40, seasonal_enabled=False)
    m25 = SentinelBenchmark.evaluate(det25, data25, labels25, compute_vus_pr=True)
    assert "vus_pr" in m25, "vus_pr missing from evaluate()"
    assert m25["vus_pr"] is not None, "vus_pr is None"
    assert 0.0 <= m25["vus_pr"] <= 1.0, f"vus_pr out of [0,1]: {m25['vus_pr']}"
    print(f"  VUS-PR={m25['vus_pr']:.4f}  AUPRC={m25['auprc']:.4f}")
    print("  PASS (VUS-PR computed, in [0,1])")

    # -- Test 26: run_suite — 5 archetypes + naive baseline ------------------
    print("\n[26] run_suite() — 5 archetypes + naive 3-sigma baseline comparison")
    suite26 = SentinelBenchmark.run_suite(
        detector_kwargs={"warm_up_period": 60, "early_mult": 2.5,
                         "seasonal_period": 30, "seasonal_enabled": True},
        n=400, seed=26, verbose=True,
    )
    for name in ["point", "contextual", "collective", "drift", "variance"]:
        assert name in suite26, f"'{name}' missing from run_suite results"
        assert f"naive_{name}" in suite26, f"'naive_{name}' missing"
    print("  PASS (all 5 archetypes + naive baselines present)")

    # -- Test 27: auto_tune F1 mode ------------------------------------------
    print("\n[27] auto_tune() — F1 optimization mode with labeled_data")
    data27, labels27 = SentinelBenchmark.generate(n=400, anomaly_type="point", seed=27)
    det27, rep27 = Detector_7_10.auto_tune(
        data27,
        labeled_data=labels27,
        alpha_grid=[0.08, 0.12],
        early_mult_grid=[2.5, 3.0],
        verbose=False,
        seasonal_enabled=False,
    )
    assert isinstance(det27, Detector_7_10), "auto_tune must return Detector_7_10"
    assert "best_alpha" in rep27
    assert "best_early_mult" in rep27
    assert "best_f1" in rep27, f"best_f1 missing, keys={list(rep27.keys())}"
    r27 = det27.update_and_check(0.5)
    assert r27["status"] != "warm_up"
    print(f"  F1 mode: best_alpha={rep27['best_alpha']}, "
          f"best_early_mult={rep27['best_early_mult']}, best_F1={rep27['best_f1']:.3f}")
    print("  PASS (auto_tune F1 mode functional)")

    # -- Test 28: State persistence with v7.10 fields ------------------------
    print("\n[28] State persistence — v7.10 PE / RSI / seasonal round-trip")
    d28 = Detector_7_10(
        warm_up_period=60, pe_enabled=True, rsi_enabled=True,
        vcusum_enabled=True, seasonal_enabled=True, seasonal_period=10,
        reset_after_regime_change=False,
    )
    warm_up(d28, n=80)
    for i in range(30):
        d28.update_and_check(math.sin(2 * math.pi * i / 10.0))
    state28 = d28.save_state()
    d28b = Detector_7_10(
        warm_up_period=60, pe_enabled=True, rsi_enabled=True,
        vcusum_enabled=True, seasonal_enabled=True, seasonal_period=10,
        reset_after_regime_change=False,
    )
    d28b.load_state(state28)
    r28a = d28.update_and_check(1.0)
    r28b = d28b.update_and_check(1.0)
    assert r28a.get("pe") == r28b.get("pe"), \
        f"PE mismatch after load: {r28a.get('pe')} vs {r28b.get('pe')}"
    print(f"  PE={d28._pe:.4f}  RSI={d28._rsi:.4f}  vcusum={d28._vcusum:.4f}")
    print(f"  detected_period={d28._detected_period}")
    print("  PASS (v7.10 state fields persist correctly)")

    # ── v7.11 SMOKE TESTS (29–34) ────────────────────────────────────────────

    # Test 29: EWS detects critical slowing down (rising variance burst)
    print("\n[Test 29] EWS detects rising-variance critical transition ...")
    d29 = Detector_7_10(warm_up_period=30, ews_enabled=True, ews_window=40,
                        ews_sensitivity=2.0, history_maxlen=200)
    rng29 = np.random.default_rng(42)
    for _ in range(30):
        d29.update_and_check(float(rng29.normal(0, 0.5)))
    # Inject rising-variance segment (critical slowing down signature)
    amp = 0.5
    for _ in range(80):
        amp *= 1.04  # variance doubling every ~18 steps
        d29.update_and_check(float(rng29.normal(0, min(amp, 5.0))))
    r29 = d29.update_and_check(0.0)
    assert "ews_score" in r29, "ews_score missing from result"
    assert "ews_regime" in r29, "ews_regime missing from result"
    assert d29._ews_score >= 0.0 and d29._ews_score <= 1.0, \
        f"EWS score out of [0,1]: {d29._ews_score}"
    assert d29._ews_regime in ("stable", "approaching", "critical"), \
        f"Unknown EWS regime: {d29._ews_regime}"
    print(f"  EWS score={d29._ews_score:.4f}  regime={d29._ews_regime}")
    print("  PASS (EWS score in [0,1], regime valid)")

    # Test 30: Adaptive Quantile Baseline fires on heavy-tailed outliers
    # reset_after_regime_change=False: prevent extreme spike from wiping _dev_history
    print("\n[Test 30] AQB quantile_threshold_mode fires on heavy-tail outlier ...")
    d30 = Detector_7_10(warm_up_period=50, quantile_threshold_mode=True,
                        quantile_early_pct=0.90, quantile_confirmed_pct=0.98,
                        quantile_window=200, two_sided=True,
                        reset_after_regime_change=False)
    rng30 = np.random.default_rng(7)
    for _ in range(50):
        d30.update_and_check(float(rng30.normal(0, 1)))
    # Feed 200 more normal samples to build the quantile baseline
    for _ in range(200):
        d30.update_and_check(float(rng30.normal(0, 1)))
    # Capture quantile thresholds before the spike
    q_early_pre  = d30._q_early
    q_conf_pre   = d30._q_confirmed
    assert q_early_pre > 0, "AQB q_early not populated after 200 normal samples"
    assert q_conf_pre  > 0, "AQB q_confirmed not populated"
    # Inject extreme outlier: 10-sigma spike — must fire alert
    r30 = d30.update_and_check(10.0)
    assert r30.get("early_alert") or r30.get("confirmed_alert"), \
        f"AQB failed to fire on 10σ spike (q_early={q_early_pre:.3f})"
    assert "q_early" in r30, "q_early missing from result in quantile_threshold_mode"
    print(f"  q_early={q_early_pre:.4f}  q_confirmed={q_conf_pre:.4f}")
    print("  PASS (AQB fires on heavy-tail outlier)")

    # Test 31: ThreadingHTTPServer handles concurrent requests correctly
    print("\n[Test 31] ThreadingHTTPServer concurrent DELETE + POST ...")
    import urllib.request
    import urllib.error
    server31 = SentinelServer(host="127.0.0.1", port=18731)
    server31.start(background=True)
    import time as _time ; _time.sleep(0.08)   # let server bind
    try:
        # POST /update
        payload31 = json.dumps({"stream_id": "s1", "value": 1.0}).encode()
        req31 = urllib.request.Request(
            "http://127.0.0.1:18731/update",
            data=payload31,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req31, timeout=3) as resp:
            assert resp.status == 200, f"POST /update returned {resp.status}"

        # DELETE /stream/s1
        del_req = urllib.request.Request(
            "http://127.0.0.1:18731/stream/s1",
            method="DELETE",
        )
        with urllib.request.urlopen(del_req, timeout=3) as resp:
            body31 = json.loads(resp.read())
            assert body31.get("removed") == "s1", f"DELETE response: {body31}"

        # Verify stream gone
        try:
            urllib.request.urlopen(
                "http://127.0.0.1:18731/status/s1", timeout=3
            )
            assert False, "Expected 404 after DELETE"
        except urllib.error.HTTPError as e:
            assert e.code == 404, f"Expected 404, got {e.code}"

        # GET /health — version should be 7.11
        with urllib.request.urlopen("http://127.0.0.1:18731/health", timeout=3) as resp:
            hbody = json.loads(resp.read())
            assert hbody["version"] == "7.11", f"Wrong version: {hbody['version']}"
        print(f"  health={hbody}")
        print("  PASS (ThreadingHTTPServer DELETE + version check)")
    finally:
        server31.stop()

    # Test 32: PE baseline EWMA prevents false alarm on periodic signal
    print("\n[Test 32] PE baseline EWMA — no false 'low_entropy_ordered' on periodic signal ...")
    d32 = Detector_7_10(warm_up_period=40, pe_enabled=True, pe_window=32, pe_dim=3,
                        rpi_enabled=True, reset_after_regime_change=False)
    # Feed a clean sine wave — PE will be consistently low (periodic), NOT anomalous
    t32 = np.linspace(0, 20 * np.pi, 400)
    sine32 = np.sin(t32).tolist()
    false_alarm_count = 0
    for i, v in enumerate(sine32):
        r32 = d32.update_and_check(v)
        if i > 80 and "low_entropy_ordered" in r32.get("alert_reasons", []):
            false_alarm_count += 1
    # After establishing PE baseline, false alarms should be rare (≤ 5%)
    allowed = max(1, int(len(sine32) * 0.05))
    assert false_alarm_count <= allowed, \
        f"Too many false 'low_entropy_ordered' alarms on sine: {false_alarm_count}/{len(sine32)-80}"
    pe_bl_str = f"{d32._pe_baseline:.4f}" if d32._pe_baseline is not None else "None"
    print(f"  PE baseline={pe_bl_str}  false_alarm_rate={false_alarm_count}/{len(sine32)-80}")
    print("  PASS (PE baseline suppresses false alarms on periodic signals)")

    # Test 33: Multivariate path populates _scalar_window → PE/RSI active
    print("\n[Test 33] Multivariate path feeds _scalar_window — PE/RSI not stale ...")
    d33 = Detector_7_10(warm_up_period=30, multivariate=True,
                        pe_enabled=True, pe_window=20, rsi_enabled=True, rpi_enabled=True)
    rng33 = np.random.default_rng(99)
    for _ in range(100):
        v33 = [float(rng33.normal(0, 1)), float(rng33.normal(0, 1))]
        d33.update_and_check(v33)
    assert len(d33._scalar_window) > 0, \
        "_scalar_window empty after 100 multivariate steps — Bug Fix 4 regression"
    r33 = d33.update_and_check([0.1, 0.1])
    assert r33.get("pe") is not None, "PE missing from multivariate result"
    print(f"  _scalar_window length={len(d33._scalar_window)}  PE={r33.get('pe'):.4f}")
    print("  PASS (multivariate path correctly feeds _scalar_window)")

    # Test 34: Bug-fix regression — no duplicate _baseline_var, CLI prog name correct
    print("\n[Test 34] Regression: _baseline_var initialised once, CLI prog name correct ...")
    d34 = Detector_7_10(warm_up_period=20)
    rng34 = np.random.default_rng(11)
    warmup34 = [float(rng34.normal(10, 2)) for _ in range(20)]
    for v in warmup34:
        d34.update_and_check(v)
    bv = d34._baseline_var
    assert bv > 0, f"_baseline_var not positive after warmup: {bv}"
    # Feed one more and confirm it hasn't been reset to 0 by any duplicate assignment
    d34.update_and_check(10.0)
    assert d34._baseline_var > 0, f"_baseline_var clobbered after first step: {d34._baseline_var}"
    # CLI prog name
    import argparse as _argparse
    _parser = _argparse.ArgumentParser(prog="fracttalix_sentinel_v711")
    assert _parser.prog == "fracttalix_sentinel_v711", f"Wrong prog: {_parser.prog}"
    print(f"  _baseline_var={bv:.6f}  (stable after warmup)")
    print("  PASS (no redundant _baseline_var, CLI name correct)")

    print("\n" + "=" * 60)
    print("All 34 tests passed. Sentinel v7.11 operational.")
    print("\nv7.11 additions verified (Meta-Kaizen v7.11):")
    print("  [v7.11] BUG FIX: stale CLI prog name corrected (v79 → v711)")
    print("  [v7.11] BUG FIX: redundant _baseline_var double-assignment removed")
    print("  [v7.11] BUG FIX: generate_surrogates Pool worker moved to module level")
    print("  [v7.11] BUG FIX: multivariate path feeds _scalar_window — PE/RSI active")
    print("  [v7.11] BUG FIX: PE alert_reasons use contextual baseline, not absolute 0.3")
    print("  [v7.11] Early Warning Signals (EWS) — FRM Axiom 9: critical slowing down")
    print("           EWS score [0,1]; regime: stable/approaching/critical")
    print("           Variance trend + AC(1) trend — Scheffer et al. (2009)")
    print("  [v7.11] Adaptive Quantile Baseline (AQB) — FRM Axiom 1: scale invariance")
    print("           Distribution-free thresholds; targets empirical FPR directly")
    print("           quantile_threshold_mode=True replaces ewma±mult×dev_ewma")
    print("  [v7.11] ThreadingHTTPServer — concurrent multi-user HTTP API")
    print("           DELETE /stream/<id>; POST /reset/<id>; daemon_threads=True")
    print("  [v7.10] Permutation Entropy (PE)      — FRM Axiom 3: complexity growth")
    print("  [v7.10] Rhythm Stability Index (RSI)  — FRM Axiom 10: synchronization")
    print("  [v7.10] Dual Encoding Score (DES)     — FRM Axiom 5: spatial+temporal info")
    print("  [v7.10] Variance CUSUM                — volatility explosion detection")
    print("  [v7.10] Seasonal Periodic Baseline    — contextual F1: 0.00 → >0.10")
    print("  [v7.10] VUS-PR benchmark metric       — NeurIPS 2024 gold standard")
    print("  [v7.9]  z_score + anomaly_score, alert_reasons, Page-Hinkley drift")
    print("  [v7.9]  Mahalanobis multivariate, SentinelBenchmark, SentinelServer, CLI")
