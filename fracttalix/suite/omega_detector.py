"""OmegaDetector -- FRM-derived timescale integrity monitoring.

What makes this unique:

- The FRM derives omega = pi / (2 * tau_gen) at Hopf criticality.
- This is an **absolute** frequency reference, not a relative one.
- Every other frequency-change detector (BOCPD, spectral CUSUM, wavelet
  decomposition) detects change relative to the data's own history -- they
  can tell you "frequency changed" but not "frequency is wrong".
- OmegaDetector can tell you "the observed frequency has deviated from
  the physics-predicted value" -- a structural integrity violation.

When tau_gen is unknown (weak mode), OmegaDetector estimates it from
data and tracks frequency stability -- less unique but still useful.

Requires: numpy (no scipy needed).
"""

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
        """Determine whether the current window contains analysable oscillatory content.

        The detector is out of scope when the window is too short (< 32 samples)
        or has negligible variance (constant / near-silent signal), since
        frequency estimation would be meaningless.

        Parameters
        ----------
        window : list of float
            Recent sample values from the sliding window.

        Returns
        -------
        bool
            ``True`` if spectral analysis should proceed, ``False`` otherwise.
        """
        if len(window) < 32:
            return False
        # Check for spectral content — white noise / constant → OUT_OF_SCOPE
        s = _std(window)
        if s < 1e-10:
            return False
        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        """Estimate the dominant oscillation frequency and score its deviation.

        The pipeline is:

        1. Mean-centre the window and apply a Hann taper.
        2. Compute the magnitude spectrum via real FFT.
        3. Verify that the spectral peak exceeds ``min_spectral_snr`` times
           the mean magnitude; reject low-SNR windows early.
        4. Refine the peak location to sub-bin accuracy using three-point
           parabolic interpolation on the magnitude spectrum.
        5. Convert the refined bin index to angular frequency
           (omega = 2 * pi * bin / N).
        6. In **strong mode** (tau_gen provided), compare the observed omega
           against the physics-predicted absolute reference
           omega_predicted = pi / (2 * tau_gen). In **weak mode** (tau_gen=0),
           compare against the median of the first few observations (baseline).
        7. Return a score in [0, 1] proportional to the fractional deviation,
           plus a human-readable diagnostic string.

        Parameters
        ----------
        window : list of float
            Recent sample values from the sliding window.

        Returns
        -------
        score : float
            Anomaly score between 0.0 (nominal) and 1.0 (severe deviation).
        detail : str
            Diagnostic message including observed omega, reference omega,
            deviation percentage, and trend information.
        """
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
        """Most recent observed angular frequency estimate.

        Returns
        -------
        float or None
            The latest omega value from the rolling history, or ``None``
            if no frequency estimate has been computed yet.
        """
        return self._omega_history[-1] if self._omega_history else None

    @property
    def omega_predicted(self) -> Optional[float]:
        """Physics-predicted angular frequency from the FRM.

        In strong mode this equals pi / (2 * tau_gen) -- the absolute
        Hopf-criticality reference. In weak mode (tau_gen=0) this is
        ``None`` because no physics prediction is available.

        Returns
        -------
        float or None
            The predicted omega, or ``None`` in weak mode.
        """
        return self._omega_predicted

    @property
    def omega_deviation(self) -> float:
        """Fractional deviation of the latest omega from the reference.

        Computed as ``|omega_obs - omega_ref| / omega_ref``. Returns 0.0
        when the baseline has not yet been established or no observations
        exist. In strong mode the reference is the physics-predicted value;
        in weak mode it is the data-derived baseline.

        Returns
        -------
        float
            Non-negative fractional deviation (e.g. 0.15 means 15%).
            Zero when the detector is still in warmup or has no data.
        """
        if not self._baseline_set or not self._omega_history:
            return 0.0
        ref = self._baseline_omega
        obs = self._omega_history[-1]
        return abs(obs - ref) / ref if ref > 1e-12 else 0.0

    def reset(self) -> None:
        """Reset the detector to its initial state.

        Clears the omega history, baseline, and fit counter while also
        resetting the base-class sliding window and warmup counters.
        After a reset the detector re-enters warmup / baseline-collection
        before scoring resumes.
        """
        super().reset()
        self._omega_history.clear()
        self._baseline_omega = None
        self._baseline_set = False
        self._fit_counter = 0

    def state_dict(self) -> Dict[str, Any]:
        """Serialize internal state for checkpointing.

        Extends the base-class snapshot with omega-specific fields so the
        detector can be restored later via :meth:`load_state`.

        Returns
        -------
        dict
            Dictionary containing base-class state plus ``omega_history``,
            ``baseline_omega``, and ``baseline_set``.

        See Also
        --------
        load_state : Restore from a previously saved state dict.
        """
        sd = super().state_dict()
        sd.update({
            "omega_history": list(self._omega_history),
            "baseline_omega": self._baseline_omega,
            "baseline_set": self._baseline_set,
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        """Restore detector state from a previously saved snapshot.

        Re-populates the omega history deque, baseline, and base-class
        internals so that scoring can resume exactly where it left off.

        Parameters
        ----------
        sd : dict
            State dictionary produced by :meth:`state_dict`.

        See Also
        --------
        state_dict : Produce the snapshot consumed by this method.
        """
        super().load_state(sd)
        self._omega_history = deque(
            sd.get("omega_history", []), maxlen=self._omega_history_size
        )
        self._baseline_omega = sd.get("baseline_omega")
        self._baseline_set = sd.get("baseline_set", False)
