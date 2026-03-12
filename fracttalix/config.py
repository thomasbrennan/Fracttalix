# fracttalix/config.py
# SentinelConfig frozen dataclass — all ~50 parameters with factory presets

import dataclasses


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
    # F: Temporal / oscillatory dynamics (signal-processing parameters)
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

    cusum_k: float = 0.5
    """CUSUM allowance k (half the expected mean shift in sigma units).
    Phase 2: was hardcoded 0.5 in CUSUMStep; now configurable."""

    cusum_h: float = 5.0
    """CUSUM decision threshold h.
    Phase 2: was hardcoded 5.0 in CUSUMStep; now configurable."""

    var_cusum_k: float = 0.5
    """VarCUSUM allowance (half the expected shift in std-devs)."""

    var_cusum_h: float = 5.0
    """VarCUSUM decision threshold."""

    alert_cooldown_steps: int = 0
    """Per-step quiet period after an alert fires (0 = no cooldown).
    Phase 4: prevents repeated alerts from sustained conditions."""

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
    # V9.0 — Channel 2 frequency decomposition
    # ------------------------------------------------------------------
    enable_frequency_decomposition: bool = True
    """Enable FFT decomposition of signal into five frequency band carrier waves."""

    min_window_for_fft: int = 32
    """Minimum window length required before FFT decomposition runs."""

    # ------------------------------------------------------------------
    # V9.0 — Cross-frequency coupling detection
    # ------------------------------------------------------------------
    enable_coupling_detection: bool = True
    """Enable cross-frequency phase-amplitude coupling measurement."""

    coupling_degradation_threshold: float = 0.3
    """composite_coupling_score below this triggers COUPLING_DEGRADATION alert."""

    coupling_trend_window: int = 10
    """Number of FrequencyBands snapshots used for coupling measurement."""

    # ------------------------------------------------------------------
    # V9.0 — Structural-rhythmic coherence
    # ------------------------------------------------------------------
    enable_channel_coherence: bool = True
    """Enable structural-rhythmic channel coherence measurement."""

    coherence_threshold: float = 0.4
    """coherence_score below this triggers STRUCTURAL_RHYTHMIC_DECOUPLING alert."""

    coherence_window: int = 20
    """Rolling window length for coherence computation."""

    # ------------------------------------------------------------------
    # V9.0 — Cascade precursor
    # ------------------------------------------------------------------
    enable_cascade_detection: bool = True
    """Enable CASCADE_PRECURSOR detection (requires all three conditions)."""

    cascade_ews_threshold: int = 2
    """Minimum number of EWS indicators elevated for cascade precursor."""

    # ------------------------------------------------------------------
    # V9.0 — Degradation sequence logging
    # ------------------------------------------------------------------
    enable_sequence_logging: bool = True
    """Enable temporal logging of channel degradation sequences."""

    sequence_retention: int = 1000
    """Maximum number of completed degradation sequences to retain."""

    # ------------------------------------------------------------------
    # V12.0 — numpy fallback warning control
    # ------------------------------------------------------------------
    warn_on_numpy_fallback: bool = True
    """Emit ImportWarning if numpy is not available (checked at import time)."""

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
        """High α, low warmup — react instantly, very high false-positive rate.

        Warning: multiplier=3.0 with alpha=0.3 produces approximately 60–80%
        normal alert rate on white noise (see benchmark/investigate_fpr_s47.py).
        Suitable only for contexts where false positives are tolerated or
        downstream filtering is applied.  Use production() for general use.
        """
        return cls(alpha=0.3, dev_alpha=0.3, warmup_periods=10)

    @classmethod
    def production(cls) -> "SentinelConfig":
        """Balanced defaults — suitable for most production deployments.

        Uses multiplier=4.5, which gives approximately 5–8% normal alert rate
        on white noise N(0,1) (see benchmark/investigate_fpr_s47.py for the
        full multiplier–FPR trade-off curve).

        Changed in v12.2: multiplier raised from 3.0 → 4.5.  The previous
        default produced a 35.6% normal alert rate.  Users who need the old
        behaviour can set SentinelConfig(multiplier=3.0) explicitly.
        """
        return cls(multiplier=4.5)

    @classmethod
    def sensitive(cls) -> "SentinelConfig":
        """Low α, tight multiplier — catches subtle anomalies."""
        return cls(alpha=0.05, dev_alpha=0.05, multiplier=2.5, warmup_periods=50)

    @classmethod
    def realtime(cls) -> "SentinelConfig":
        """Fast response with quantile-adaptive thresholds."""
        return cls(alpha=0.2, warmup_periods=15, quantile_threshold_mode=True)
