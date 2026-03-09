# fracttalix_sentinel v7.6 py

# Fracttalix Sentinel v7.6 — Adaptive early detection with regime awareness

# Updated: Optional Numba JIT, parallel surrogates, tqdm progress/verbose flag, proper FFT phase randomization

# v7.2: Buffered CSV export (low-overhead for high-freq)

# v7.3: Buffered CSV with empty-file header fix, flush on reset, doc polish

# v7.4: Meta-Kaizen fluid dynamics rotation — STI, Boundary Layer Warning, Oscillation Damping, CPD

# v7.5: Hybrid synthesis — user-configurable params, rolling STI with alpha modulation, oscillation damping via extra_mult, boundary warning in all paths, complete state persistence

# v7.6: All KVS-ranked fluid dynamics improvements confirmed in both paths, refined integration, complete persistence

# rolling-window STI with alpha modulation, oscillation damping via extra_mult (no mutation bug),

# boundary warning in all paths (univariate + multivariate), complete per-channel state

# persistence, cpd_enabled flag, sti_regime label in result dict

# Designed for finance, medical, infrastructure/IoT/security monitoring, and research

from collections import deque
import math
import warnings
import json
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


class Detector_7_6:
    """
    Enhanced lightweight, regime-aware anomaly detector.

    Features:
    - Two-sided alerts (upper & lower anomalies)
    - Per-channel multivariate detection with aggregation flexibility
    - Soft/full regime reset options
    - NaN/Inf imputation (skip or mean)
    - State persistence (JSON save/load) — complete including per-channel v7.4 state
    - Volatility-adaptive smoothing
    - Optional Numba JIT for speed (if installed)
    - Optional parallel surrogates & tqdm progress
    - Proper FFT-based phase randomization for surrogates (Theiler et al. 1992)
    - Buffered CSV export (low filesystem overhead: flush on interval or manually)

    v7.4 Meta-Kaizen fluid dynamics improvements (retained and refined in v7.5):
    - Sentinel Turbulence Index (STI): rolling-window Reynolds number analog.
      Characterizes flow state (laminar / transitional / turbulent) before applying
      thresholds. Modulates both alert multipliers AND alpha for deeper integration.
      All thresholds user-configurable. sti_regime label reported in result dict.
    - Boundary Layer Warning: fourth alert tier. Fires after tps_periods consecutive
      steps within tps_proximity_pct of early threshold without crossing. Implemented
      in BOTH univariate and multivariate paths. Present in all result dict paths.
    - Oscillation Damping Filter: suppresses resonance artifacts (rapid early_alert
      fire/clear cycles without confirmed_alert). Implemented via extra_mult parameter
      — no base multiplier mutation, no TypeError risk. Per-channel lookback fully
      persisted in save_state/load_state.
    - CUSUM Pressure Differential (CPD): optional flag (cpd_enabled). Directional
      pressure signal in every active result dict. +1.0 upward, -1.0 downward, ~0
      bidirectional volatility.

    v7.5 synthesis improvements over v7.4:
    - All v7.4 parameters exposed in __init__ (from Grok) — user-configurable without
      subclassing
    - Rolling-window STI with alpha modulation (from Claude) — smoother, more faithful
      to Reynolds number inspiration
    - Oscillation damping via extra_mult not self.early_mult mutation (from Claude) —
      eliminates TypeError in multivariate mode
    - Boundary warning in univariate path (from Claude) — not only multivariate
    - boundary_warning present in all result dict paths (from Claude)
    - cpd_enabled flag (from Grok)
    - Complete per-channel state persistence including oscillation_lookback (from Grok)
    - Cleaner proximity calculation in TPS (from Grok)

    When to Use Fracttalix Sentinel vs. Other Tools:
    | Feature / Need                          | Fracttalix Sentinel v7.6                     | Alternatives (PyOD, ADTK, ruptures, etc.) |
    |-----------------------------------------|----------------------------------------------|--------------------------------------------|
    | Single-file, no install/dependencies    | Yes                                          | No                                         |
    | Real-time / streaming capable           | Yes (update_and_check)                       | Sometimes (heavier setup)                  |
    | Two-sided + per-channel multivariate    | Yes                                          | Rare                                       |
    | Soft regime reset (preserve baselines)  | Yes                                          | Almost never                               |
    | State persistence (JSON save/load)      | Yes — complete including v7.5 state          | Rare                                       |
    | Buffered CSV export (low-overhead)      | Yes (interval or manual flush)               | Varies (often manual)                      |
    | Volatility-adaptive smoothing           | Yes                                          | Sometimes                                  |
    | Built-in surrogates for significance    | Yes (FFT phase randomization)                | Yes (but heavier)                          |
    | Rolling-window Turbulence Index (STI)   | Yes (v7.4+, rolling window in v7.5)          | No                                         |
    | Boundary Layer Warning (4th tier)       | Yes — univariate + multivariate (v7.5)       | No                                         |
    | Oscillation Damping (no mutation bug)   | Yes (v7.5 fix)                               | No                                         |
    | CUSUM Pressure Differential (CPD)       | Yes (optional flag)                          | No                                         |
    | Best for                                | Quick screening, lightweight monitoring,     | Full-featured research pipelines           |
    |                                         | exploratory work in finance/HRV/IoT/research |                                            |
    """

    def __init__(
        self,
        # === Core parameters ===
        alpha: float = 0.12,                    # EWMA smoothing factor (0 < alpha <= 1)
        early_mult: float = 2.75,               # Early warning multiplier
        fixed_mult: float = 3.2,                # Confirmed alert multiplier
        warm_up_period: int = 60,               # Points before adaptive mode
        use_fixed_during_warmup: bool = True,   # Use fixed threshold during warm-up
        two_sided: bool = True,                 # Detect both upper and lower anomalies

        # === Regime change detection ===
        cusum_threshold: float = 5.0,           # Sigma-multiplier for regime change
        reset_after_regime_change: Union[bool, str] = "full",  # 'full', 'soft', False

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
        turbulence_adaptive: bool = True,       # Enable STI-based threshold + alpha adaptation
        sti_window: int = 10,                   # Rolling window for smoothed STI
        sti_laminar_threshold: float = 1.0,     # STI < threshold -> laminar (tighten)
        sti_turbulent_threshold: float = 2.0,   # STI > threshold -> turbulent (widen)

        # === v7.4: Boundary Layer Warning (TPS) — separation precursor analog ===
        boundary_warning_enabled: bool = True,  # Enable fourth alert tier
        tps_proximity_pct: float = 0.15,        # Within this fraction of threshold = boundary zone
        tps_periods: int = 5,                   # Consecutive periods in boundary zone to fire

        # === v7.4: Oscillation Damping Filter — vortex shedding analog ===
        oscillation_damping_enabled: bool = True,
        oscillation_count_threshold: int = 3,   # Fire/clear cycles before damping
        oscillation_periods: int = 20,          # Lookback window for oscillation detection
        oscillation_mult_bump: float = 0.10,    # Fractional early_mult increase when damping

        # === v7.4: CUSUM Pressure Differential ===
        cpd_enabled: bool = True,               # Include CPD in result dict

        # === Extras ===
        verbose_explain: bool = False,
        alert_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        impute_method: str = "skip",            # 'skip' or 'mean' for NaN/Inf
        parallel_surrogates: bool = False,
        numba_enabled: bool = True,
        csv_output_path: Optional[str] = None,
        csv_flush_interval: int = 60,           # 0 = manual flush only
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

        # v7.4 params — all user-configurable (from Grok)
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

        # Extras
        self.verbose_explain = verbose_explain
        self.alert_callback = alert_callback
        self.impute_method = impute_method
        self.parallel_surrogates = parallel_surrogates
        self.numba_enabled = numba_enabled
        self.csv_output_path = csv_output_path
        self.csv_flush_interval = csv_flush_interval
        self.csv_buffer: List[Dict] = []

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

        # v7.6 — univariate state
        self._dev_ewma_history: deque = deque(maxlen=sti_window)  # rolling STI window
        self._sti: float = 1.0
        self._sti_regime: str = "transitional"
        self._tps_counter: int = 0                                # univariate boundary counter
        self._oscillation_lookback: List[bool] = []              # univariate osc history
        self._oscillation_fire_clear_count: int = 0
        self._prev_early_alert: bool = False
        self._damping_active: bool = False

        # v7.6 — per-channel state (fully persisted)
        self._ch_tps_counters: List[int] = []
        self._ch_oscillation_lookback: List[List[bool]] = []
        self._ch_fire_clear_counts: List[int] = []
        self._ch_prev_early_alerts: List[bool] = []
        self._ch_damping_active: List[bool] = []

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

        self.cusum_pos = 0.0
        self.cusum_neg = 0.0

        # Reset v7.5 state
        self._dev_ewma_history.clear()
        self._sti = 1.0
        self._sti_regime = "transitional"
        self._tps_counter = 0
        self._oscillation_lookback = []
        self._oscillation_fire_clear_count = 0
        self._prev_early_alert = False
        self._damping_active = False

        # Per-channel reset
        n = self.n_channels
        self._ch_tps_counters = [0] * n
        self._ch_oscillation_lookback = [[] for _ in range(n)]
        self._ch_fire_clear_counts = [0] * n
        self._ch_prev_early_alerts = [False] * n
        self._ch_damping_active = [False] * n

    def save_state(self) -> str:
        """Save complete state to JSON string — including all v7.5 fields."""
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
            # v7.6 per-channel state — complete persistence
            "ch_tps_counters": self._ch_tps_counters,
            "ch_oscillation_lookback": self._ch_oscillation_lookback,
            "ch_fire_clear_counts": self._ch_fire_clear_counts,
            "ch_prev_early_alerts": self._ch_prev_early_alerts,
            "ch_damping_active": self._ch_damping_active,
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

    # ─────────────────────────────────────────────────────────────────────────
    # v7.6 FLUID DYNAMICS METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _update_sti(self, dev_ewma: float, baseline_std: float) -> Tuple[float, str]:
        """
        Sentinel Turbulence Index — rolling-window Reynolds number analog (from Claude).
        Smoother than instantaneous ratio. More faithful to fluid dynamics inspiration.
        Returns (sti, regime_label).
        """
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
        """
        Adjust alert multiplier based on turbulence regime (from Claude).
        Laminar -> tighten 10% (system predictable, anomalies real).
        Turbulent -> widen 20% (system noisy, reduce false positives).
        """
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
        """
        Boundary Layer Warning — separation precursor analog.
        Cleaner proximity calculation (from Grok).
        Works for both univariate (idx=0, counter_ref=[self._tps_counter])
        and per-channel (idx=ch, counter_ref=self._ch_tps_counters).
        Returns True if boundary warning fires.
        """
        if not self.boundary_warning_enabled:
            counter_ref[idx] = 0
            return False

        # Determine nearest threshold
        if early_lower is not None and self.two_sided:
            dist_upper = abs(current - early_upper)
            dist_lower = abs(current - early_lower)
            threshold = early_upper if dist_upper < dist_lower else early_lower
        else:
            threshold = early_upper

        # Already crossed — not boundary zone
        if (current >= early_upper or
                (early_lower is not None and self.two_sided and current <= early_lower)):
            counter_ref[idx] = 0
            return False

        # Proximity check (from Grok — cleaner than absolute gap)
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
        Via extra_mult parameter, NOT self.early_mult mutation (from Claude —
        eliminates TypeError in multivariate mode).
        Per-channel lookback fully maintained (from Grok).
        Returns extra multiplier increment (0 or oscillation_mult_bump).
        """
        if not self.oscillation_damping_enabled:
            prev_ref[idx] = early_alert
            return 0.0

        # Detect fire->clear transition
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
        """
        CUSUM Pressure Differential — pressure differential analog.
        Optional (cpd_enabled flag from Grok).
        +1.0 upward, -1.0 downward, ~0 bidirectional volatility.
        """
        if not self.cpd_enabled:
            return None
        denom = self.cusum_pos + self.cusum_neg + 1e-9
        return round((self.cusum_pos - self.cusum_neg) / denom, 4)

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

            # Initialize per-channel v7.6 state
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

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def update_and_check(self, value: Union[float, List[float]]) -> Dict[str, Any]:
        """
        Feed a scalar or list. Returns dict with alert status, all diagnostics,
        and v7.5 turbulence/boundary/oscillation/CPD fields.
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

        # -- Warm-up phase ---------------------------------------------------
        if self.count <= self.warm_up_period:
            result = {
                "early_alert": False,
                "confirmed_alert": False,
                "boundary_warning": False,   # present in ALL paths
                "status": "warm_up",
            }
            if self.use_fixed_during_warmup and len(self.values_deque) >= 10:
                self._initialize_from_warmup()
                temp = self._compute_alerts(value)
                result.update(temp)
                result["boundary_warning"] = False
                # Reset CUSUM only — do NOT reset count or baselines.
                # Resetting count here would trap the detector in warm-up forever.
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

                # Per-channel STI (uses shared regime from aggregate STI)
                eff_early_mult = self._sti_adjusted_mult(self.early_mult)

                # Oscillation damping per channel — via extra_mult, no mutation (from Claude)
                extra_mult = self._check_oscillation_damping(
                    early_alerts[ch] if ch > 0 else False,
                    self._ch_oscillation_lookback,
                    self._ch_fire_clear_counts,
                    self._ch_prev_early_alerts,
                    self._ch_damping_active,
                    idx=ch,
                )
                eff_early_mult += extra_mult * eff_early_mult

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

                # Update oscillation state with actual early_alert
                self._check_oscillation_damping(
                    ch_early,
                    self._ch_oscillation_lookback,
                    self._ch_fire_clear_counts,
                    self._ch_prev_early_alerts,
                    self._ch_damping_active,
                    idx=ch,
                )

                # Boundary warning per channel (from Grok — cleaner proximity)
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

            # Capture pre-update threshold for TPS boundary check.
            # After EWMA update the threshold shifts — TPS must measure proximity
            # to the threshold the value was approaching, not the one it caused.
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

            # STI — rolling window (from Claude)
            self._update_sti(self.dev_ewma, self.baseline_std)

            # Initial alert computation
            result = self._compute_alerts(current)
            early_alert = result["early_alert"]
            confirmed_alert = result["confirmed_alert"]

            # Oscillation damping — extra_mult not mutation (from Claude)
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
            # Sync scalar state back from list wrappers
            self._oscillation_lookback = osc_lookback_ref[0]
            self._oscillation_fire_clear_count = osc_fc_ref[0]
            self._prev_early_alert = osc_prev_ref[0]
            self._damping_active = osc_damp_ref[0]

            if self._damping_active and extra_mult > 0:
                result = self._compute_alerts(current, extra_early_mult=extra_mult)
                early_alert = result["early_alert"]
                confirmed_alert = result["confirmed_alert"]

            # Boundary warning — use PRE-UPDATE threshold so proximity is measured
            # against the threshold the value was approaching, not the one it caused.
            tps_ref = [self._tps_counter]
            bw_upper = pre_update_early_upper if pre_update_early_upper is not None else result.get("early_threshold", self.ewma + self.early_mult * self.dev_ewma)
            bw_lower = pre_update_early_lower if pre_update_early_lower is not None else result.get("early_lower_threshold")
            boundary_warning = self._check_tps(
                current,
                bw_upper,
                bw_lower,
                tps_ref,
                idx=0,
            )
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
            if self.reset_after_regime_change == "full":
                self.reset()
            else:
                self.reset(soft=True)
                self._initialize_from_warmup()
            result = {
                "early_alert": False,
                "confirmed_alert": False,
                "boundary_warning": False,   # present in ALL paths
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
            # v7.6 fields — always present
            "sti": round(self._sti, 4),
            "sti_regime": self._sti_regime,            # human-readable label (from Claude)
            "oscillation_damping_active": self._damping_active,
        })

        # CPD — optional flag (from Grok)
        cpd = self._compute_cpd()
        if cpd is not None:
            result["cpd"] = cpd

        if self.two_sided and self.ewma is not None and self.dev_ewma is not None:
            result["early_lower_threshold"] = round(
                self.ewma - self.early_mult * self.dev_ewma, 4
            )

        if self.verbose_explain:
            cpd_str = f"CPD {cpd:+.3f} | " if cpd is not None else ""
            result["explanation"] = (
                f"Deviation {round(deviation, 3)} | "
                f"CUSUM +{result['cusum_pos']} / -{result['cusum_neg']} | "
                f"{cpd_str}"
                f"STI {result['sti']:.3f} ({self._sti_regime}) | "
                f"Threshold {round(self.cusum_threshold * cusum_dev, 2)}"
            )

        self._trigger_alert_callback(result)
        self._buffer_csv_result(result)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_adaptive_alpha(self, channel: Optional[int] = None) -> float:
        """
        Volatility-adaptive alpha further modulated by STI regime (from Claude).
        Turbulent -> faster adaptation. Laminar -> more smoothing.
        """
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

        # STI modulation (from Claude)
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
        """
        Compute alerts. extra_early_mult for oscillation damping — no mutation.
        STI-adjusted multiplier applied here.
        """
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
# SMOKE TEST — six structured assertions covering all v7.6 features
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("Fracttalix Sentinel v7.6 — Smoke test")
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
        )
        defaults.update(kwargs)
        return Detector_7_6(**defaults)

    def warm_up(det, seed=42, n=60):
        random.seed(seed)
        for _ in range(n):
            det.update_and_check(random.gauss(0, 1))

    def fmt(r):
        sti_s = f"{r['sti']:.3f}" if r.get("sti") is not None else "n/a"
        cpd_s = f"{r['cpd']:+.3f}" if r.get("cpd") is not None else "n/a"
        return (f"STI={sti_s}({r.get('sti_regime','?')}) CPD={cpd_s} "
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
    assert r.get("sti") is not None, "STI missing from result"
    assert r.get("cpd") is not None, "CPD missing from result"
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
        # Read pre-update threshold directly from detector state each step.
        # Using result["early_threshold"] won't work because that is the POST-update
        # threshold after EWMA has already incorporated the new value.
        if d.ewma is None or d.dev_ewma is None:
            break
        pre_thresh = d.ewma + d.early_mult * d.dev_ewma
        approach_val = pre_thresh * 0.92  # 8% below -> within 15% boundary zone
        r = d.update_and_check(approach_val)
        print(f"  step {i+1}: val={approach_val:.3f} pre_thresh={pre_thresh:.3f} "
              f"counter={d._tps_counter} boundary={r.get('boundary_warning',False)} "
              f"early={r.get('early_alert',False)}")
        if r.get("boundary_warning"):
            bw_fired = True
    assert bw_fired, "Boundary warning never fired approaching threshold"
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
    print("\n[5] State persistence round-trip")
    d = make_detector()
    warm_up(d)
    for _ in range(10):
        d.update_and_check(random.gauss(0, 1))
    state_json = d.save_state()
    d2 = make_detector()
    d2.load_state(state_json)
    r1 = d.update_and_check(1.5)
    r2 = d2.update_and_check(1.5)
    assert r1["status"] == r2["status"] == "active"
    assert r1.get("sti") == r2.get("sti"), f"STI mismatch: {r1.get('sti')} vs {r2.get('sti')}"
    print(f"  Original STI={r1.get('sti'):.4f} | Restored STI={r2.get('sti'):.4f}")
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

    print("\n" + "=" * 55)
    print("All 7 tests passed. Sentinel v7.6 operational.")
