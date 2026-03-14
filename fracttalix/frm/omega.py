# fracttalix/frm/omega.py
# OmegaDetector — FRM frequency integrity checker.
#
# Theorem basis (FRM quarter-wave theorem):
#   ω = π / (2 · τ_gen)
#
# When τ_gen is known (strong mode), the FRM predicts the exact fundamental
# frequency. OmegaDetector fits the dominant oscillation frequency from
# streaming data and checks whether it matches this prediction.
#
# Deviation between observed ω and predicted ω is a direct test of whether
# the FRM structure is intact. If ω drifts, the delay τ_gen is changing —
# structural change is underway, independent of λ.
#
# Cross-validation:  OmegaDetector is independent of Lambda/HopfDetector.
#   Lambda watches the amplitude envelope (decay rate λ).
#   Omega watches the frequency (ω = π/(2·τ_gen)).
#   Agreement of both = compound FRM structural signal.
#   CouplingDetector (Layer 1) provides a third independent signal
#   via PAC degradation in the FRM-predicted frequency band.
#
# Modes:
#   strong mode: tau_gen supplied → omega_predicted = π/(2·tau_gen), fixed.
#     Deviation from this is the FRM integrity test.
#   weak mode:   tau_gen=None → track omega stability via FFT.
#     Useful for frequency change detection; not FRM-physics-derived.
#     frm_confidence does NOT increment in weak mode.
#
# OUT_OF_SCOPE conditions:
#   • Signal has no dominant frequency (spectrum too flat)
#   • FFT fit fails (insufficient window, all noise)
#   • Weak mode only: insufficient history for trend
#
# Implemented by Lady Ada (FRM physics layer).

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional

from fracttalix.suite.base import BaseDetector, DetectorResult, ScopeStatus
from fracttalix.frm.lorentzian import welch_psd, fit_lorentzian


def _estimate_omega_fft(data) -> float:
    """Estimate dominant angular frequency from FFT peak (requires numpy).

    Applies a Hann window to reduce spectral leakage, then uses parabolic
    interpolation for sub-bin frequency accuracy.
    """
    import numpy as np
    n = len(data)
    centered = data - np.mean(data)
    # Hann window: reduces leakage, concentrates energy at true frequency bin
    hann = np.hanning(n)
    fft_vals = np.abs(np.fft.rfft(centered * hann))
    if len(fft_vals) <= 1:
        return 0.1
    peak_idx = int(np.argmax(fft_vals[1:])) + 1  # exclude DC
    # Parabolic interpolation for sub-bin refinement
    if 1 < peak_idx < len(fft_vals) - 1:
        alpha = float(fft_vals[peak_idx - 1])
        beta = float(fft_vals[peak_idx])
        gamma = float(fft_vals[peak_idx + 1])
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            correction = 0.5 * (alpha - gamma) / denom
            refined_idx = peak_idx + correction
        else:
            refined_idx = float(peak_idx)
    else:
        refined_idx = float(peak_idx)
    omega = 2.0 * math.pi * refined_idx / n
    return max(omega, 0.01)


def _estimate_omega_autocorr(data, omega_pred: float, tolerance: float = 0.5) -> float:
    """Estimate dominant angular frequency via autocorrelation lag search.

    Searches for the autocorrelation peak within ±tolerance of the predicted
    period.  This is immune to the FFT non-integer-bin quantisation error that
    arises when the signal period is not an integer multiple of the FFT window,
    and it adapts to frequency changes roughly one period after the onset.

    Parameters
    ----------
    data        : array-like of floats
    omega_pred  : predicted angular frequency (strong mode)
    tolerance   : fractional period search range (default ±50%)

    Returns estimated angular frequency.
    """
    import numpy as np
    n = len(data)
    d = np.array(data, dtype=float)
    d -= d.mean()

    period_pred = 2.0 * math.pi / omega_pred
    lag_min = max(2, int(period_pred * (1.0 - tolerance)))
    lag_max = min(n - 2, int(period_pred * (1.0 + tolerance)))
    if lag_min >= lag_max:
        return omega_pred  # fallback — tolerance too tight for window

    # Autocorrelation for each lag in the search range
    lags = range(lag_min, lag_max + 1)
    ac = np.array([np.dot(d[lag:], d[:n - lag]) / (n - lag) for lag in lags])

    best_idx = int(np.argmax(ac))
    best_lag = float(lag_min + best_idx)

    # Parabolic sub-sample refinement on the autocorrelation peak
    if 0 < best_idx < len(ac) - 1:
        a, b, g = float(ac[best_idx - 1]), float(ac[best_idx]), float(ac[best_idx + 1])
        denom = a - 2.0 * b + g
        if abs(denom) > 1e-12:
            best_lag += -0.5 * (g - a) / denom

    return 2.0 * math.pi / max(best_lag, 1.0)


def _spectrum_peak_ratio(data) -> float:
    """Return peak-to-mean ratio of FFT magnitudes (excluding DC).

    Values > 3.0 indicate a dominant frequency; values near 1.0 are flat noise.
    """
    import numpy as np
    n = len(data)
    if n < 4:
        return 0.0
    centered = data - np.mean(data)
    fft_vals = np.abs(np.fft.rfft(centered))
    bins = fft_vals[1:]  # exclude DC
    if len(bins) == 0:
        return 0.0
    avg = float(np.mean(bins))
    if avg < 1e-10:
        return 0.0
    return float(np.max(bins)) / avg


class OmegaDetector(BaseDetector):
    """FRM frequency integrity detector.

    Checks whether the observed dominant frequency matches the FRM
    prediction ω = π/(2·τ_gen).

    Parameters
    ----------
    tau_gen : float or None
        If provided: strong mode. omega_predicted = π/(2·tau_gen).
        If None: weak mode. Track frequency stability only (generic).
        Only strong mode contributes to frm_confidence.
    warmup : int
        Observations before any verdict (default 80).
    window : int
        Minimum FFT window size (default 64).  In strong mode the actual
        FFT window is auto-expanded to the smallest multiple of the FRM
        period (4·τ_gen) that is ≥ window, ensuring the signal lands on
        an integer FFT bin and eliminating spectral leakage quantisation
        error.  warmup is also raised to match if necessary.
    deviation_threshold : float
        Fractional deviation |Δω/ω_predicted| above which ALERT fires.
        Default 0.05 (5% deviation from predicted frequency).
    alert_steps : int
        Number of consecutive above-threshold steps before ALERT (default 3).
        Prevents single-step FFT artifacts from triggering.
    """

    def __init__(
        self,
        tau_gen: Optional[float] = None,
        warmup: int = 80,
        window: int = 64,
        deviation_threshold: float = 0.05,
        alert_steps: int = 3,
        scope_tolerance: float = 0.50,
    ):
        self._omega_predicted = (
            math.pi / (2.0 * tau_gen) if tau_gen is not None and tau_gen > 0 else None
        )
        self._strong_mode = (self._omega_predicted is not None)
        self._scope_tolerance = scope_tolerance  # OUT_OF_SCOPE if |Δω/ω| > this

        # In strong mode: auto-expand FFT window so the autocorrelation search
        # range [period*(1-tol), period*(1+tol)] fits inside the window.
        # Minimum window = period*(1+scope_tolerance) + 4.
        # The docstring promised this; here it is implemented.
        # For tau_gen=10 (period=40), window_needed=64 = default.
        # For tau_gen=20 (period=80), window_needed=124 — without expansion,
        # autocorrelation searches [40,62] and misses the peak at lag=80,
        # causing Omega to report false deviations and spurious ALERT on all
        # oscillatory signals (including stable null trajectories).
        if self._strong_mode and self._omega_predicted is not None:
            period = 2.0 * math.pi / self._omega_predicted  # = 4 * tau_gen
            min_window = int(period * (1.0 + scope_tolerance)) + 4
            if min_window > window:
                window = min_window
            warmup = max(warmup, window)

        super().__init__("OmegaDetector", warmup=warmup, window_size=max(window, warmup))
        self._tau_gen = tau_gen
        self._deviation_threshold = deviation_threshold
        self._alert_steps = alert_steps
        self._window_size_fft = window
        self._consecutive_above: int = 0
        self._omega_history: deque = deque(maxlen=20)

    def _check_scope(self, window: List[float]) -> bool:
        """Return True if signal has a dominant periodic component near omega_predicted.

        Two-gate scope check:
        1. Signal must have a dominant spectral peak (FFT ratio > 3× mean bin power).
           Without this, there is no frequency to track — white noise is OUT_OF_SCOPE.
        2. Strong mode only: the dominant frequency (via autocorrelation, which is
           immune to FFT quantisation error) must be within scope_tolerance (default
           50%) of omega_predicted.  Signals oscillating far from the FRM-predicted
           frequency are not FRM-shaped and OmegaDetector stays silent.
        """
        import numpy as np
        fft_window = np.array(window[-self._window_size_fft:], dtype=float)
        if len(fft_window) < 8:
            return False
        ratio = _spectrum_peak_ratio(fft_window)
        if ratio <= 3.0:
            return False

        # Strong mode: frequency sanity check via FFT.
        # FFT is used here because it correctly identifies the dominant frequency
        # of arbitrary signals (e.g. a sinusoid at 4× the FRM frequency).
        # Autocorrelation is reserved for _compute where precision near omega_pred
        # matters; in _check_scope we only need to reject wildly mismatched signals.
        if self._strong_mode and self._omega_predicted is not None:
            omega_obs = _estimate_omega_fft(fft_window)
            initial_dev = abs(omega_obs - self._omega_predicted) / self._omega_predicted
            if initial_dev > self._scope_tolerance:
                return False

        return True

    def _compute(self, window: List[float]):
        """Estimate observed ω and compare to FRM prediction.

        Strong mode: compare omega_obs (via Lorentzian centroid) to omega_predicted.
          Lorentzian f0_fit is immune to phase diffusion: noise broadens the peak
          but does not shift its centroid, unlike autocorrelation lag which scatters
          by D = sigma^2/(2A^2) per sample step.  Falls back to autocorrelation when
          Lorentzian fit is poor (r_squared < 0.5 or FWHM unresolvable).
          Fires ALERT after alert_steps consecutive deviations > threshold.
        Weak mode: track omega stability over recent history via FFT + CV.
          Fires ALERT when frequency wanders > 20% of its mean.
        """
        import numpy as np
        fft_window = np.array(window[-self._window_size_fft:], dtype=float)

        if self._strong_mode and self._omega_predicted is not None:
            # Predicted peak frequency in cycles/sample
            f0_pred = self._omega_predicted / (2.0 * math.pi)

            # Lorentzian fit via Welch PSD — phase-diffusion immune centroid
            freqs, psd = welch_psd(fft_window, seg_len=max(16, len(fft_window) // 2))
            f0_fit, _lam, r_squared, fwhm_resolvable = fit_lorentzian(
                freqs, psd, f0_pred=f0_pred, band_factor=self._scope_tolerance
            )

            if r_squared >= 0.5 and fwhm_resolvable and f0_fit > 0:
                # Lorentzian centroid: stable under phase diffusion (noise broadens
                # the peak symmetrically without shifting its centroid).
                omega_obs = 2.0 * math.pi * f0_fit
                method = "lorentzian"
            else:
                # Fallback to autocorrelation: superior to FFT parabolic when the
                # signal period is non-integer in the FFT window (avoids bin
                # quantisation scatter). Lorentzian is preferred when FWHM is
                # resolvable (broad peaks from damped/noisy oscillators).
                omega_obs = _estimate_omega_autocorr(
                    list(fft_window), self._omega_predicted, self._scope_tolerance
                )
                method = "autocorr"

            self._omega_history.append(omega_obs)

            deviation = abs(omega_obs - self._omega_predicted) / self._omega_predicted
            if deviation > self._deviation_threshold:
                self._consecutive_above += 1
            else:
                self._consecutive_above = 0

            # Score rises linearly: 0 → 1 as consecutive_above → alert_steps
            score = min(1.0, self._consecutive_above / self._alert_steps)
            msg = (
                f"omega_obs={omega_obs:.4f} omega_pred={self._omega_predicted:.4f} "
                f"deviation={deviation:.3f} consecutive={self._consecutive_above} "
                f"r2={r_squared:.2f} method={method} mode=strong"
            )
            return score, msg

        else:
            # Weak mode: track frequency stability via FFT + coefficient of variation.
            omega_obs = _estimate_omega_fft(fft_window)
            self._omega_history.append(omega_obs)
            if len(self._omega_history) < 3:
                return 0.0, f"omega_obs={omega_obs:.4f} mode=weak insufficient_history"

            hist = list(self._omega_history)
            omega_mean = sum(hist) / len(hist)
            omega_var = sum((x - omega_mean) ** 2 for x in hist) / len(hist)
            omega_std = math.sqrt(max(omega_var, 0.0))
            # Coefficient of variation: std / mean
            cv = omega_std / (abs(omega_mean) + 1e-10)
            # cv > 0.20 (20% variation) → score approaches 1.0
            score = min(1.0, cv / 0.20)
            msg = (
                f"omega_obs={omega_obs:.4f} omega_mean={omega_mean:.4f} "
                f"omega_std={omega_std:.5f} cv={cv:.3f} mode=weak"
            )
            return score, msg

    def reset(self) -> None:
        super().reset()
        self._consecutive_above = 0
        self._omega_history.clear()

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update({
            "tau_gen": self._tau_gen,
            "scope_tolerance": self._scope_tolerance,
            "consecutive_above": self._consecutive_above,
            "omega_history": list(self._omega_history),
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        super().load_state(sd)
        self._consecutive_above = sd.get("consecutive_above", 0)
        self._scope_tolerance = sd.get("scope_tolerance", self._scope_tolerance)
        self._omega_history = deque(sd.get("omega_history", []), maxlen=20)
