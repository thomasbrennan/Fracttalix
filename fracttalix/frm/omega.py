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
        Rolling window for FFT frequency estimation (default 64).
    deviation_threshold : float
        Fractional deviation |Δω/ω_predicted| above which ALERT fires.
        Default 0.05 (5% deviation from predicted frequency).
    alert_steps : int
        Number of consecutive above-threshold steps before ALERT (default 5).
        Prevents single-step FFT artifacts from triggering.
    """

    def __init__(
        self,
        tau_gen: Optional[float] = None,
        warmup: int = 80,
        window: int = 64,
        deviation_threshold: float = 0.05,
        alert_steps: int = 5,
    ):
        super().__init__("OmegaDetector", warmup=warmup, window_size=max(window, warmup))
        self._tau_gen = tau_gen
        self._deviation_threshold = deviation_threshold
        self._alert_steps = alert_steps
        self._window_size_fft = window
        self._omega_predicted = (
            math.pi / (2.0 * tau_gen) if tau_gen is not None and tau_gen > 0 else None
        )
        self._strong_mode = (self._omega_predicted is not None)
        self._consecutive_above: int = 0
        self._omega_history: deque = deque(maxlen=20)

    def _check_scope(self, window: List[float]) -> bool:
        """Return True if FFT has a dominant frequency peak (not flat noise)."""
        import numpy as np
        fft_window = np.array(window[-self._window_size_fft:], dtype=float)
        if len(fft_window) < 8:
            return False
        ratio = _spectrum_peak_ratio(fft_window)
        # Require peak at least 3× the mean bin magnitude
        return ratio > 3.0

    def _compute(self, window: List[float]):
        """Estimate observed ω and compare to FRM prediction.

        Strong mode: compare omega_obs to omega_predicted = π/(2·tau_gen).
          Fires ALERT after alert_steps consecutive deviations > threshold.
        Weak mode: track omega stability over recent history via CV.
          Fires ALERT when frequency wanders > 20% of its mean.
        """
        import numpy as np
        fft_window = np.array(window[-self._window_size_fft:], dtype=float)
        omega_obs = _estimate_omega_fft(fft_window)
        self._omega_history.append(omega_obs)

        if self._strong_mode:
            # Smooth omega estimate over recent history to suppress FFT quantization noise.
            # Use the mean of the last min(5, len) samples from _omega_history (already
            # includes omega_obs just appended above).
            hist = list(self._omega_history)
            n_smooth = min(5, len(hist))
            omega_smoothed = sum(hist[-n_smooth:]) / n_smooth

            deviation = abs(omega_smoothed - self._omega_predicted) / self._omega_predicted
            if deviation > self._deviation_threshold:
                self._consecutive_above += 1
            else:
                self._consecutive_above = 0

            # Score rises linearly: 0 → 1 as consecutive_above → alert_steps
            score = min(1.0, self._consecutive_above / self._alert_steps)
            msg = (
                f"omega_obs={omega_obs:.4f} omega_pred={self._omega_predicted:.4f} "
                f"deviation={deviation:.3f} consecutive={self._consecutive_above} mode=strong"
            )
            return score, msg

        else:
            # Weak mode: alert on frequency instability (high CV over history)
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
            "consecutive_above": self._consecutive_above,
            "omega_history": list(self._omega_history),
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        super().load_state(sd)
        self._consecutive_above = sd.get("consecutive_above", 0)
        self._omega_history = deque(sd.get("omega_history", []), maxlen=20)
