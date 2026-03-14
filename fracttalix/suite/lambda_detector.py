"""LambdaDetector v2 -- bifurcation proximity via variance and spectral width.

v1 failed because it fit the FRM parametric form to raw signal data.
The FRM form describes a linear ring-down, but real systems near Hopf
bifurcation are noise-driven and nonlinear.  The cubic saturation term
creates an effective damping floor that masks the true λ.

v2 estimates λ from two observables that actually track bifurcation
proximity in nonlinear noise-driven systems:

1. **Variance scaling**: Var(x) ∝ σ²_noise / (2λ).  Inverting this
   gives λ_hat = C / Var(x) where C is calibrated during warmup.
   As λ → 0, variance diverges — this holds even with cubic saturation.

2. **Spectral peak width**: The power spectrum near ω₀ is Lorentzian
   with half-width λ/(2π).  Fitting gives a second λ estimate.

The combined estimate is more robust than either alone.

Unique FRM contributions preserved:
- ω = π/(2·τ_gen) anchors the spectral analysis
- Time-to-transition estimate: Δt = λ_hat / |dλ_hat/dt|
- Scope awareness via spectral SNR and variance trend consistency

Requires: numpy only (scipy optional for Lorentzian refinement).
"""

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from fracttalix.suite.base import BaseDetector, _mean, _variance, _std


class LambdaDetector(BaseDetector):
    """Track bifurcation proximity via variance-inversion and spectral width.

    Parameters
    ----------
    tau_gen : float
        Generation timescale.  ω = π/(2·τ_gen).  If 0, estimated from FFT.
    fit_window : int
        Sliding window size for variance and spectral computation.
    fit_interval : int
        Compute every N steps (performance).
    lambda_window : int
        Rolling window for λ_hat history (trend estimation).
    lambda_warning : float
        Threshold below which λ_hat triggers CRITICAL_SLOWING.
    r_squared_min : float
        Unused (kept for API compatibility).  Scope now uses spectral SNR.
    warmup : int
        Explicit warmup period.  The detector also needs fit_window
        observations before the first estimate.
    """

    def __init__(
        self,
        tau_gen: float = 0.0,
        fit_window: int = 128,
        fit_interval: int = 4,
        lambda_window: int = 20,
        lambda_warning: float = 0.05,
        r_squared_min: float = 0.5,
        warmup: int = 0,
    ):
        super().__init__(
            name="Lambda",
            warmup=warmup,
            window_size=fit_window,
        )
        self._tau_gen = tau_gen
        self._fit_interval = fit_interval
        self._lambda_window = lambda_window
        self._lambda_warning = lambda_warning
        self._alert_threshold = 0.5

        # Internal state
        self._lambda_history: deque = deque(maxlen=lambda_window)
        self._var_history: deque = deque(maxlen=lambda_window)
        self._fit_counter = 0
        self._baseline_var: Optional[float] = None
        self._baseline_lambda: Optional[float] = None
        self._baseline_is_estimated: bool = False  # True only when from real AC1
        self._last_lambda: Optional[float] = None
        self._last_lam_rate = 0.0
        self._last_time_to_bif: Optional[float] = None
        self._last_scope = "INSUFFICIENT_DATA"
        self._last_spectral_snr = 0.0
        self._last_spectral_width: Optional[float] = None
        self._last_var_ratio = 1.0
        self._last_var_trend = 0.5
        self._last_baseline_ratio = 1.0
        # Confirmation counter: require alert conditions to persist across
        # multiple fit intervals before firing.  Transient dips in estimated
        # lambda (from noisy variance) will not persist; genuine CSD will.
        self._confirm_count = 0
        self._confirm_required = 5  # consecutive fit intervals

    def _check_scope(self, window: List[float]) -> bool:
        min_window = max(32, self._window_size // 2)
        return len(window) >= min_window

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        import numpy as np

        self._fit_counter += 1
        if self._fit_counter % self._fit_interval != 0:
            return self._score_from_state()

        data = np.array(window, dtype=float)
        n = len(data)

        # ── Step 1: Variance-based λ estimation ──
        current_var = float(np.var(data))
        self._var_history.append(current_var)

        # Calibrate baseline from first few windows
        if self._baseline_var is None and len(self._var_history) >= 3:
            self._baseline_var = float(np.median(list(self._var_history)))
            # λ_baseline estimated from AC1 during warmup
            centered = data - np.mean(data)
            c0 = np.sum(centered ** 2)
            c1 = np.sum(centered[:-1] * centered[1:])
            ac1 = c1 / c0 if abs(c0) > 1e-12 else 0.0
            if 0 < ac1 < 1:
                self._baseline_lambda = -math.log(ac1)
                self._baseline_is_estimated = True
            else:
                # AC1 failed — try spectral width as fallback baseline
                lam_spec = self._estimate_lambda_spectral(data, n)
                if lam_spec is not None and self._last_spectral_snr >= 3.0:
                    self._baseline_lambda = lam_spec
                    self._baseline_is_estimated = True
                else:
                    self._baseline_lambda = 0.1
                    self._baseline_is_estimated = False

        if self._baseline_var is None:
            self._last_scope = "INSUFFICIENT_DATA"
            return 0.0, "collecting baseline"

        # λ_hat from variance inversion: λ ∝ 1/Var
        # Using ratio: λ_hat = λ_baseline × (baseline_var / current_var)
        if current_var > 1e-12:
            lam_var = self._baseline_lambda * (self._baseline_var / current_var)
        else:
            lam_var = self._baseline_lambda

        # ── Step 2: Spectral width λ estimation ──
        lam_spectral = self._estimate_lambda_spectral(data, n)

        # ── Step 3: Combine estimates ──
        if lam_spectral is not None and self._last_spectral_snr >= 3.0:
            # Weighted combination: spectral has higher precision when SNR good
            lam_hat = 0.4 * lam_var + 0.6 * lam_spectral
        else:
            lam_hat = lam_var

        self._last_lambda = max(0.0, lam_hat)
        self._lambda_history.append(self._last_lambda)

        # ── Step 4: Scope classification ──
        self._last_scope = self._compute_scope(current_var)
        # Track variance ratio and trend for scoring
        if self._baseline_var and self._baseline_var > 1e-12:
            self._last_var_ratio = current_var / self._baseline_var
        else:
            self._last_var_ratio = 1.0
        # Variance trend: fraction of recent pairs where var increased
        self._last_var_trend = self._compute_var_trend()

        # ── Step 5: Trend and time-to-bifurcation ──
        self._last_lam_rate, self._last_time_to_bif = self._compute_trend()

        return self._score_from_state()

    def _estimate_lambda_spectral(self, data, n) -> Optional[float]:
        """Estimate λ from spectral peak width (Lorentzian half-width)."""
        import numpy as np

        # Determine expected peak frequency
        if self._tau_gen and self._tau_gen > 0:
            omega_expected = math.pi / (2.0 * self._tau_gen)
        else:
            omega_expected = None

        # FFT with Hann window
        centered = data - np.mean(data)
        hann = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(centered * hann)) ** 2  # power spectrum
        freqs = np.fft.rfftfreq(n)  # in cycles per sample

        if len(spectrum) <= 2:
            return None

        # Find spectral peak (excluding DC)
        peak_idx = np.argmax(spectrum[1:]) + 1
        peak_val = spectrum[peak_idx]
        mean_val = np.mean(spectrum[1:])
        self._last_spectral_snr = peak_val / mean_val if mean_val > 1e-12 else 0.0

        if self._last_spectral_snr < 2.0:
            return None

        # Measure half-width at half-maximum (HWHM)
        half_max = peak_val / 2.0
        # Search left from peak
        left_idx = peak_idx
        for i in range(peak_idx - 1, 0, -1):
            if spectrum[i] <= half_max:
                # Linear interpolation for sub-bin accuracy
                if spectrum[i + 1] - spectrum[i] > 1e-12:
                    frac = (half_max - spectrum[i]) / (spectrum[i + 1] - spectrum[i])
                    left_idx = i + frac
                else:
                    left_idx = i
                break

        # Search right from peak
        right_idx = peak_idx
        for i in range(peak_idx + 1, len(spectrum)):
            if spectrum[i] <= half_max:
                if spectrum[i - 1] - spectrum[i] > 1e-12:
                    frac = (half_max - spectrum[i]) / (spectrum[i - 1] - spectrum[i])
                    right_idx = i - frac
                else:
                    right_idx = i
                break

        hwhm_bins = (right_idx - left_idx) / 2.0
        if hwhm_bins <= 0:
            return None

        # Convert HWHM from bins to angular frequency
        # freq_resolution = 1/n cycles/sample per bin
        # HWHM in angular freq = hwhm_bins × (2π/n)
        hwhm_omega = hwhm_bins * (2.0 * math.pi / n)

        # For Lorentzian: HWHM = λ/(2π) in frequency, or HWHM = λ in angular freq
        # So λ ≈ hwhm_omega
        self._last_spectral_width = hwhm_omega
        return hwhm_omega

    def _compute_scope(self, current_var: float) -> str:
        """Classify scope based on spectral and variance evidence."""
        if self._last_spectral_snr < 2.0:
            return "OUT_OF_SCOPE"
        if current_var < 1e-12:
            return "OUT_OF_SCOPE"
        # Check baseline ratio — if λ has dropped significantly, not stable
        if self._baseline_var and self._baseline_var > 1e-12:
            var_ratio = current_var / self._baseline_var
            # Variance growth > 1.5× means system is changing
            if var_ratio > 1.5:
                return "IN_SCOPE"
            if 0.7 < var_ratio < 1.5 and self._last_lam_rate >= -1e-4:
                return "STABLE"
        return "IN_SCOPE"

    def _compute_trend(self) -> Tuple[float, Optional[float]]:
        """Estimate dλ/dt and time-to-bifurcation from λ history."""
        import numpy as np

        history = list(self._lambda_history)
        if len(history) < 3:
            return 0.0, None

        lams = np.array(history, dtype=float)
        t = np.arange(len(lams), dtype=float)

        # MAD outlier rejection
        median_lam = np.median(lams)
        mad = np.median(np.abs(lams - median_lam))
        if mad > 1e-12:
            threshold = 3.0 * mad / 0.6745
            mask = np.abs(lams - median_lam) < threshold
            if mask.sum() >= 3:
                lams = lams[mask]
                t = t[mask]

        t_mean = np.mean(t)
        lam_mean = np.mean(lams)
        numer = np.sum((t - t_mean) * (lams - lam_mean))
        denom = np.sum((t - t_mean) ** 2)
        if abs(denom) < 1e-12:
            return 0.0, None

        rate = float(numer / denom)

        # Time-to-bifurcation
        time_to_bif = None
        current = self._lambda_history[-1]
        if rate < -1e-8 and current > 0:
            time_to_bif = current / abs(rate) * self._fit_interval

        return rate, time_to_bif

    def _compute_var_trend(self) -> float:
        """Fraction of consecutive pairs in var_history where variance increased.

        Returns value in [0, 1]. Values > 0.6 suggest consistently rising
        variance (CSD signature). Values near 0.5 are random fluctuations.
        """
        vhist = list(self._var_history)
        if len(vhist) < 5:
            return 0.5
        increases = sum(1 for i in range(1, len(vhist)) if vhist[i] > vhist[i - 1])
        return increases / (len(vhist) - 1)

    def _score_from_state(self) -> Tuple[float, str]:
        """Score based on λ trajectory and baseline ratio.

        Uses two complementary signals:
        1. Rolling rate (dλ/dt) — sensitive to rapid changes
        2. Baseline ratio (λ_current / λ_baseline) — sensitive to gradual decline

        Confirmation window: alert-level conditions must persist for
        ``_confirm_required`` consecutive fit intervals before the score
        exceeds the alert threshold.  This filters transient dips in the
        noisy λ estimate (which recover quickly on stable systems) while
        passing genuine sustained decline (which does not recover).
        """
        if self._last_scope in ("OUT_OF_SCOPE", "INSUFFICIENT_DATA"):
            msg = f"scope={self._last_scope}"
            if self._last_lambda is not None:
                msg += f" λ={self._last_lambda:.4f} SNR={self._last_spectral_snr:.1f}"
            self._confirm_count = 0
            return 0.0, msg

        lam = self._last_lambda
        rate = self._last_lam_rate
        ttb = self._last_time_to_bif

        if lam is None:
            return 0.0, "no estimate yet"

        # Baseline ratio: how much has λ declined from calibration?
        # Uses median of recent λ history for noise robustness
        # Only meaningful when baseline_lambda was genuinely estimated (not fallback)
        # AND the baseline was above warning (system started healthy)
        baseline_ratio = 1.0
        has_meaningful_baseline = (
            self._baseline_lambda is not None
            and self._baseline_is_estimated
            and self._baseline_lambda > self._lambda_warning
        )
        if has_meaningful_baseline and len(self._lambda_history) >= 3:
            sorted_hist = sorted(self._lambda_history)
            mid = len(sorted_hist) // 2
            smoothed_lam = sorted_hist[mid]
            baseline_ratio = smoothed_lam / self._baseline_lambda

        self._last_baseline_ratio = baseline_ratio
        var_trend = self._last_var_trend

        # ── Check if alert-level conditions are met (pre-confirmation) ──
        alert_candidate = False
        candidate_score = 0.0
        candidate_msg = ""

        # ── Path 1: Rate-based detection (rapid decline) ──
        # Requires corroborating variance trend (> 0.6 = consistently rising)
        if rate < -1e-3 and lam < self._lambda_warning and var_trend > 0.6:
            alert_candidate = True
            if ttb is not None and ttb < 40.0:
                candidate_score = min(1.0, 0.7 + 0.3 * (40.0 - ttb) / 40.0)
                candidate_msg = f"TRANSITION λ={lam:.4f} rate={rate:.5f} Δt={ttb:.1f} vt={var_trend:.2f}"
            else:
                candidate_score = 0.6
                candidate_msg = f"CRITICAL_SLOWING λ={lam:.4f} rate={rate:.5f} vt={var_trend:.2f}"

        # ── Path 2: Baseline-ratio detection (gradual decline) ──
        if not alert_candidate and has_meaningful_baseline and baseline_ratio < 1.0:
            lam_ceiling = 10.0 * self._lambda_warning
            if baseline_ratio < 0.30 and var_trend > 0.55 and lam < lam_ceiling:
                alert_candidate = True
                candidate_score = min(0.8, 0.5 + 0.3 * (0.25 - baseline_ratio) / 0.25)
                candidate_msg = f"CSD_RATIO λ={lam:.4f} ratio={baseline_ratio:.2f} vt={var_trend:.2f}"
            elif baseline_ratio < 0.45 and lam < self._lambda_warning and var_trend > 0.55:
                alert_candidate = True
                candidate_score = min(0.7, 0.5 + 0.2 * (0.45 - baseline_ratio) / 0.45)
                candidate_msg = f"CSD_DECLINING λ={lam:.4f} ratio={baseline_ratio:.2f} vt={var_trend:.2f}"

        # ── Confirmation gate ──
        # Only count at fit intervals (when _compute actually ran)
        is_fit_step = (self._fit_counter % self._fit_interval == 0)
        if alert_candidate and is_fit_step:
            self._confirm_count += 1
        elif not alert_candidate and is_fit_step:
            # Decay rather than reset: allows brief interruptions in noisy signals
            self._confirm_count = max(0, self._confirm_count - 1)

        if alert_candidate:
            if self._confirm_count >= self._confirm_required:
                # Confirmed: genuine sustained decline
                return candidate_score, candidate_msg
            else:
                # Not yet confirmed: report sub-alert score showing buildup
                pending_score = min(0.49, candidate_score * self._confirm_count / self._confirm_required)
                return pending_score, f"confirming ({self._confirm_count}/{self._confirm_required}) {candidate_msg}"

        # ── Sub-alert: mild decline ──
        if has_meaningful_baseline and baseline_ratio < 0.5 and var_trend > 0.55:
            score = max(0.0, min(0.49, 0.3 * (0.5 - baseline_ratio) / 0.5))
            return score, f"λ={lam:.4f} ratio={baseline_ratio:.2f} weakening vt={var_trend:.2f}"

        # Stable
        if lam < self._lambda_warning:
            score = max(0.0, min(0.49, (self._lambda_warning - lam) / self._lambda_warning * 0.3))
            return score, f"λ={lam:.4f} rate={rate:.5f} stable SNR={self._last_spectral_snr:.1f}"
        return 0.0, f"λ={lam:.4f} rate={rate:.5f} stable SNR={self._last_spectral_snr:.1f}"

    @property
    def current_lambda(self) -> Optional[float]:
        """Most recently estimated λ (variance + spectral combined)."""
        return self._last_lambda

    @property
    def lambda_rate(self) -> float:
        """Rate of change of λ.  Negative = declining toward bifurcation."""
        return self._last_lam_rate

    @property
    def time_to_transition(self) -> Optional[float]:
        """Estimated steps until λ reaches zero."""
        return self._last_time_to_bif

    @property
    def r_squared(self) -> float:
        """Spectral SNR (replaces R² from v1 for API compatibility)."""
        return self._last_spectral_snr

    @property
    def scope_status(self) -> str:
        """Current scope: INSUFFICIENT_DATA, OUT_OF_SCOPE, STABLE, IN_SCOPE."""
        return self._last_scope

    @property
    def baseline_ratio(self) -> float:
        """Ratio of current λ to baseline λ (median-smoothed).

        Values < 1.0 indicate λ has declined from its baseline.
        Values < 0.5 suggest significant critical slowing down.
        Virtu uses this directly as an activation signal (more reliable
        than lambda_rate, which is smoothed over 20 rolling windows).
        Returns 1.0 when baseline has not yet been established.
        """
        return self._last_baseline_ratio

    def reset(self) -> None:
        super().reset()
        self._lambda_history.clear()
        self._var_history.clear()
        self._fit_counter = 0
        self._baseline_var = None
        self._baseline_lambda = None
        self._baseline_is_estimated = False
        self._last_lambda = None
        self._last_lam_rate = 0.0
        self._last_time_to_bif = None
        self._last_scope = "INSUFFICIENT_DATA"
        self._last_spectral_snr = 0.0
        self._last_spectral_width = None
        self._last_var_ratio = 1.0
        self._last_var_trend = 0.5
        self._last_baseline_ratio = 1.0
        self._confirm_count = 0

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update({
            "lambda_history": list(self._lambda_history),
            "var_history": list(self._var_history),
            "fit_counter": self._fit_counter,
            "baseline_var": self._baseline_var,
            "baseline_lambda": self._baseline_lambda,
            "last_lambda": self._last_lambda,
            "last_lam_rate": self._last_lam_rate,
            "last_time_to_bif": self._last_time_to_bif,
            "last_scope": self._last_scope,
            "last_spectral_snr": self._last_spectral_snr,
            "last_spectral_width": self._last_spectral_width,
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        super().load_state(sd)
        self._lambda_history = deque(
            sd.get("lambda_history", []), maxlen=self._lambda_window
        )
        self._var_history = deque(
            sd.get("var_history", []), maxlen=self._lambda_window
        )
        self._fit_counter = sd.get("fit_counter", 0)
        self._baseline_var = sd.get("baseline_var")
        self._baseline_lambda = sd.get("baseline_lambda")
        self._last_lambda = sd.get("last_lambda")
        self._last_lam_rate = sd.get("last_lam_rate", 0.0)
        self._last_time_to_bif = sd.get("last_time_to_bif")
        self._last_scope = sd.get("last_scope", "INSUFFICIENT_DATA")
        self._last_spectral_snr = sd.get("last_spectral_snr", 0.0)
        self._last_spectral_width = sd.get("last_spectral_width")
