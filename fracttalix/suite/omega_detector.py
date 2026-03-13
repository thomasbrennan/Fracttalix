# fracttalix/suite/omega_detector.py
# OmegaDetector — FRM-derived timescale integrity monitoring.
#
# What makes this unique:
#   - The FRM derives ω = π/(2·τ_gen) at Hopf criticality
#   - This is an ABSOLUTE frequency reference, not a relative one
#   - Every other frequency-change detector (BOCPD, spectral CUSUM,
#     wavelet decomposition) detects change relative to the data's own
#     history — they can tell you "frequency changed" but not "frequency
#     is wrong"
#   - Omega can tell you "the observed frequency has deviated from the
#     physics-predicted value" — a structural integrity violation
#
# When τ_gen is unknown (weak mode), Omega estimates it from data and
# tracks frequency stability — which is less unique but still useful.
#
# Requires: numpy (no scipy needed)

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from fracttalix.suite.base import BaseDetector, _mean, _std, _variance


class OmegaDetector(BaseDetector):
    """Monitor oscillation frequency against FRM-predicted ω = π/(2·τ_gen).

    Two modes:
    - Strong mode (τ_gen provided): Alerts when observed ω deviates from
      the physics-predicted value. This is an absolute structural check.
    - Weak mode (τ_gen=0): Estimates baseline ω during warmup, alerts when
      ω shifts. Equivalent to spectral CUSUM but with FRM-aware scoring.

    Parameters
    ----------
    tau_gen : float
        Generation timescale. ω_predicted = π/(2·τ_gen). If 0, weak mode.
    fit_window : int
        FFT window size for frequency estimation.
    omega_history : int
        Rolling window of ω estimates for trend detection.
    deviation_threshold : float
        Fractional ω deviation that triggers alert (0.15 = 15%).
    min_spectral_snr : float
        Minimum spectral peak / mean ratio to be in scope.
    """

    def __init__(
        self,
        tau_gen: float = 0.0,
        fit_window: int = 128,
        omega_history: int = 20,
        deviation_threshold: float = 0.15,
        min_spectral_snr: float = 3.0,
        warmup: int = 0,
    ):
        super().__init__(
            name="Omega",
            warmup=warmup,
            window_size=fit_window,
        )
        self._tau_gen = tau_gen
        self._omega_predicted = (
            math.pi / (2.0 * tau_gen) if tau_gen > 0 else None
        )
        self._omega_history_size = omega_history
        self._deviation_threshold = deviation_threshold
        self._min_snr = min_spectral_snr
        self._alert_threshold = 0.5

        # Internal state
        self._omega_history: deque = deque(maxlen=omega_history)
        self._baseline_omega: Optional[float] = None
        self._baseline_set = False
        self._fit_counter = 0

    def _check_scope(self, window: List[float]) -> bool:
        if len(window) < 32:
            return False
        # Check for spectral content — white noise / constant → OUT_OF_SCOPE
        s = _std(window)
        if s < 1e-10:
            return False
        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        import numpy as np

        data = np.array(window, dtype=float)
        n = len(data)

        # Estimate ω via FFT with parabolic interpolation
        centered = data - np.mean(data)
        hann = np.hanning(n)
        fft_vals = np.abs(np.fft.rfft(centered * hann))

        if len(fft_vals) <= 2:
            return 0.0, "insufficient FFT bins"

        # Check spectral SNR (peak / mean)
        peak_idx = np.argmax(fft_vals[1:]) + 1
        peak_val = fft_vals[peak_idx]
        mean_val = np.mean(fft_vals[1:])
        snr = peak_val / mean_val if mean_val > 1e-12 else 0.0

        if snr < self._min_snr:
            return 0.0, f"low spectral SNR ({snr:.1f} < {self._min_snr})"

        # Parabolic interpolation for sub-bin accuracy
        if 1 < peak_idx < len(fft_vals) - 1:
            a = fft_vals[peak_idx - 1]
            b = fft_vals[peak_idx]
            g = fft_vals[peak_idx + 1]
            d = a - 2.0 * b + g
            if abs(d) > 1e-12:
                delta = 0.5 * (a - g) / d
                refined = peak_idx + delta
            else:
                refined = float(peak_idx)
        else:
            refined = float(peak_idx)

        omega_obs = 2.0 * math.pi * refined / n
        self._omega_history.append(omega_obs)

        # Set baseline in weak mode
        if not self._baseline_set and len(self._omega_history) >= 5:
            if self._omega_predicted is None:
                # Weak mode: use median of first observations as baseline
                self._baseline_omega = float(
                    np.median(list(self._omega_history))
                )
            else:
                # Strong mode: baseline IS the predicted value
                self._baseline_omega = self._omega_predicted
            self._baseline_set = True

        if not self._baseline_set:
            return 0.0, f"baseline collecting ω={omega_obs:.4f}"

        # Compute deviation from reference
        omega_ref = self._baseline_omega
        deviation = abs(omega_obs - omega_ref) / omega_ref if omega_ref > 1e-12 else 0.0

        # Compute trend in ω (is it drifting?)
        omega_trend = 0.0
        if len(self._omega_history) >= 5:
            omegas = np.array(list(self._omega_history), dtype=float)
            t = np.arange(len(omegas), dtype=float)
            t_mean = np.mean(t)
            o_mean = np.mean(omegas)
            numer = np.sum((t - t_mean) * (omegas - o_mean))
            denom = np.sum((t - t_mean) ** 2)
            if abs(denom) > 1e-12:
                omega_trend = float(numer / denom) / omega_ref

        # Score: how far from reference?
        if deviation > self._deviation_threshold:
            score = min(1.0, 0.5 + 0.5 * (deviation - self._deviation_threshold) / self._deviation_threshold)
            mode = "STRONG" if self._omega_predicted else "WEAK"
            return score, (
                f"FREQUENCY_DEVIATION [{mode}] "
                f"ω={omega_obs:.4f} ref={omega_ref:.4f} "
                f"dev={deviation:.1%} trend={omega_trend:.4f}"
            )

        # Check if ω is trending away from reference
        if abs(omega_trend) > 0.005:
            trend_score = min(0.49, abs(omega_trend) / 0.02 * 0.49)
            return trend_score, (
                f"ω={omega_obs:.4f} ref={omega_ref:.4f} "
                f"dev={deviation:.1%} trend={omega_trend:.4f}"
            )

        return 0.0, (
            f"ω={omega_obs:.4f} ref={omega_ref:.4f} "
            f"dev={deviation:.1%} stable"
        )

    @property
    def current_omega(self) -> Optional[float]:
        return self._omega_history[-1] if self._omega_history else None

    @property
    def omega_predicted(self) -> Optional[float]:
        return self._omega_predicted

    @property
    def omega_deviation(self) -> float:
        if not self._baseline_set or not self._omega_history:
            return 0.0
        ref = self._baseline_omega
        obs = self._omega_history[-1]
        return abs(obs - ref) / ref if ref > 1e-12 else 0.0

    def reset(self) -> None:
        super().reset()
        self._omega_history.clear()
        self._baseline_omega = None
        self._baseline_set = False
        self._fit_counter = 0

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update({
            "omega_history": list(self._omega_history),
            "baseline_omega": self._baseline_omega,
            "baseline_set": self._baseline_set,
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        super().load_state(sd)
        self._omega_history = deque(
            sd.get("omega_history", []), maxlen=self._omega_history_size
        )
        self._baseline_omega = sd.get("baseline_omega")
        self._baseline_set = sd.get("baseline_set", False)
