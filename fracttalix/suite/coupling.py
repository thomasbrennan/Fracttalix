"""CouplingDetector -- Cross-frequency decoupling via phase-amplitude coupling.

Theorem basis (MK-P5 / PAC)
----------------------------
In a healthy oscillatory system, lower-frequency bands modulate higher-
frequency amplitudes (phase-amplitude coupling, PAC).  This cross-scale
coordination degrades before the system transitions -- the heterodyned
information channel degrades first.

Measurement algorithm:

1. FFT-decompose the rolling window into 5 bands (ultra-low, low, mid,
   high, ultra-high).
2. For each adjacent pair, compute PAC strength as the correlation between
   low-band phase and high-band magnitude across the window.
3. Track the composite coupling score and its trend.
4. Alert if coupling is declining (coupling_trend < threshold) and the
   composite score falls below a critical level.

OUT_OF_SCOPE conditions
-----------------------
- Signal has insufficient oscillatory content (noise floor dominates):
  ultra-high power > 70 % of total spectrum -- pure noise, no bands to
  couple.
- Signal is too short for reliable FFT (< min_window).
- Signal has no meaningful frequency structure (all bands equally flat).
- PAC bands (low, mid, high) carry < 30 % of total power.

Strengths and limitations
-------------------------
Best at
    Oscillatory systems losing cross-scale coordination before collapse
    (neural signals, power grids, physiological rhythms, market
    microstructure).
Mediocre at
    Non-oscillatory signals (random walks, pure step functions).
Useless at
    White noise (always OUT_OF_SCOPE by design -- no bands to couple).
"""

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from fracttalix.suite.base import (
    BaseDetector,
    _mean,
    _std,
)


def _fft_bands(data: List[float]) -> Optional[Dict[str, Tuple[float, float]]]:
    """Return band (power, phase) for 5 bands.  Returns None if FFT unavailable."""
    n = len(data)
    try:
        import numpy as np
        arr = np.array(data, dtype=float)
        arr = arr - arr.mean()
        fft = np.fft.rfft(arr)
        freqs = np.fft.rfftfreq(n)
        mags = np.abs(fft)
        phases = np.angle(fft)

        def band(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                return 0.0, 0.0
            return float(np.mean(mags[mask])), float(np.mean(phases[mask]))

        return {
            "ultra_low": band(0.00, 0.05),
            "low":       band(0.05, 0.15),
            "mid":       band(0.15, 0.40),
            "high":      band(0.40, 0.70),
            "ultra_high": band(0.70, 1.00),
        }
    except ImportError:
        # Pure-Python fallback DFT (slow but correct)
        # Mean-centre to match the NumPy path (line 57)
        mean_val = sum(data) / n
        centred = [x - mean_val for x in data]
        spectrum = []
        for k in range(n // 2 + 1):
            re = sum(centred[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
            im = sum(centred[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
            spectrum.append((math.hypot(re, im), math.atan2(-im, re)))

        total = len(spectrum)

        def pure_band(lo_frac, hi_frac):
            lo_i = int(lo_frac * total)
            hi_i = max(lo_i + 1, int(hi_frac * total))
            hi_i = min(hi_i, total)
            s = spectrum[lo_i:hi_i]
            if not s:
                return 0.0, 0.0
            avg_mag = sum(x[0] for x in s) / len(s)
            avg_phase = sum(x[1] for x in s) / len(s)
            return avg_mag, avg_phase

        return {
            "ultra_low": pure_band(0.00, 0.05),
            "low":       pure_band(0.05, 0.15),
            "mid":       pure_band(0.15, 0.40),
            "high":      pure_band(0.40, 0.70),
            "ultra_high": pure_band(0.70, 1.00),
        }


def _pac_coefficient(lo_phase: float, hi_mag: float,
                     lo_phases: List[float], hi_mags: List[float]) -> float:
    """Phase-amplitude coupling: correlation between lo-band phase and hi-band magnitude.

    We use the modulation index (MI) approximation: the amplitude envelope
    of the hi-band binned by the lo-band phase.  For streaming we use the
    simpler instantaneous version: |cos(lo_phase)| * hi_mag normalised.
    """
    if not lo_phases or not hi_mags:
        return 0.0
    n = min(len(lo_phases), len(hi_mags))
    if n < 3:
        return 0.0
    # Correlation between phase (via cosine) and magnitude
    cos_phases = [math.cos(p) for p in lo_phases[-n:]]
    mags = hi_mags[-n:]
    mu_c = _mean(cos_phases)
    mu_m = _mean(mags)
    cov = sum((c - mu_c) * (m - mu_m) for c, m in zip(cos_phases, mags)) / n
    std_c = _std(cos_phases)
    std_m = _std(mags)
    if std_c < 1e-10 or std_m < 1e-10:
        return 0.0
    return abs(cov / (std_c * std_m))


class CouplingDetector(BaseDetector):
    """Cross-frequency decoupling detector via phase-amplitude coupling trend.

    Parameters
    ----------
    warmup : int
        Observations before any verdict (default 120).
    window : int
        Rolling window for FFT (default 128).
    min_fft_window : int
        Minimum window for reliable FFT (default 64).
    noise_floor_threshold : float
        If ultra_high power fraction > this → OUT_OF_SCOPE (default 0.70).
    coupling_drop_threshold : float
        If composite_coupling < warmup_coupling * this → ALERT (default 0.55).
    trend_threshold : float
        If coupling_trend < this (negative = declining) → ALERT (default -0.05).
    pac_history : int
        Steps of band data to keep for PAC computation (default 20).
    coupling_threshold : float
        Score threshold for ALERT status (default 0.50).
    """

    def __init__(
        self,
        warmup: int = 120,
        window: int = 128,
        min_fft_window: int = 64,
        noise_floor_threshold: float = 0.70,
        coupling_drop_threshold: float = 0.55,
        trend_threshold: float = -0.05,
        pac_history: int = 20,
        coupling_threshold: float = 0.50,
    ):
        super().__init__("CouplingDetector", warmup=warmup, window_size=window)
        self._min_fft_window = min_fft_window
        self._noise_floor_threshold = noise_floor_threshold
        self._coupling_drop_threshold = coupling_drop_threshold
        self._trend_threshold = trend_threshold
        self._pac_history = pac_history
        self._alert_threshold = coupling_threshold

        # PAC history buffers
        self._lo_phases: deque = deque(maxlen=pac_history)
        self._hi_mags: deque = deque(maxlen=pac_history)
        self._mid_phases: deque = deque(maxlen=pac_history)
        self._high_mags: deque = deque(maxlen=pac_history)

        # Coupling score history
        self._coupling_history: deque = deque(maxlen=40)
        self._coupling_ewma: float = 0.0
        self._coupling_n: int = 0

        # Warmup baseline
        self._warmup_coupling: float = 0.0
        self._baseline_set: bool = False

    def _compute_coupling(self, bands: Dict[str, Tuple[float, float]]) -> float:
        """Compute composite PAC score from adjacent band pairs (low-mid, mid-high).

        Appends the current step's phase and magnitude values to the PAC
        history buffers, then computes the PAC coefficient for each adjacent
        band pair and returns their average.

        Parameters
        ----------
        bands : dict
            Band name to (power, phase) tuple, as returned by ``_fft_bands``.

        Returns
        -------
        float
            Mean PAC coefficient across the low-mid and mid-high pairs.
        """
        _, lo_ph = bands["low"]
        hi_mid_mag, _ = bands["mid"]
        _, mid_ph = bands["mid"]
        hi_mag, _ = bands["high"]

        self._lo_phases.append(lo_ph)
        self._hi_mags.append(hi_mid_mag)
        self._mid_phases.append(mid_ph)
        self._high_mags.append(hi_mag)

        pac_lo_mid = _pac_coefficient(lo_ph, hi_mid_mag,
                                      list(self._lo_phases), list(self._hi_mags))
        pac_mid_hi = _pac_coefficient(mid_ph, hi_mag,
                                      list(self._mid_phases), list(self._high_mags))
        return (pac_lo_mid + pac_mid_hi) / 2.0

    def _noise_fraction(self, bands: Dict[str, Tuple[float, float]]) -> float:
        """Compute the fraction of total spectral power in the ultra-high band.

        A high noise fraction (> ``noise_floor_threshold``) indicates the
        signal is dominated by noise with no meaningful frequency structure.

        Parameters
        ----------
        bands : dict
            Band name to (power, phase) tuple, as returned by ``_fft_bands``.

        Returns
        -------
        float
            Ultra-high power / total power.
        """
        total = sum(p for p, _ in bands.values()) + 1e-10
        uh_power = bands["ultra_high"][0]
        return uh_power / total

    def _check_scope(self, window: List[float]) -> bool:
        """Determine whether the signal is suitable for coupling detection.

        Scope gates
        -----------
        1. Window too short for reliable FFT (< ``min_fft_window``).
        2. FFT unavailable (numpy fallback also failed).
        3. Noise floor dominates: ultra-high power > ``noise_floor_threshold``
           of total -- pure noise, no bands to couple.
        4. Total spectral power near zero -- flat / constant signal.
        5. No dominant frequency band: ``max(band_power) / total < 0.40`` --
           energy is uniformly distributed (white noise).
        6. PAC bands (low + mid + high) carry < 30 % of total power -- the
           bands used for coupling computation are essentially empty.

        Parameters
        ----------
        window : list of float
            Current rolling window.

        Returns
        -------
        bool
            True if the signal is in scope for coupling analysis.
        """
        if len(window) < self._min_fft_window:
            return False

        bands = _fft_bands(window)
        if bands is None:
            return False

        # Scope gate: noise floor dominates → no meaningful frequency structure
        if self._noise_fraction(bands) > self._noise_floor_threshold:
            return False

        # Scope gate: all bands are near-zero → flat signal (constant or nearly so)
        powers = [p for p, _ in bands.values()]
        total_power = sum(powers) + 1e-10
        if total_power < 1e-6:
            return False

        # Scope gate: no dominant frequency band → white noise or flat spectrum.
        # White noise distributes energy roughly uniformly across all frequency
        # bands; the maximum band power / total stays ≤ 0.33 empirically.
        # Oscillatory and PAC signals have a dominant band (max/total ≥ 0.57).
        # A threshold of 0.40 cleanly separates the two classes.
        if max(powers) / total_power < 0.40:
            return False

        # Scope gate: PAC bands (low, mid, high) must carry meaningful energy.
        # If the signal's energy is entirely in ultra_low (f < 0.05), the bands
        # used for coupling computation (low=0.05-0.15, mid=0.15-0.40, high=0.40-0.70)
        # are essentially empty → PAC coefficient is noise, not signal.
        # Require lo+mid+high ≥ 30% of total power.
        pac_power = (
            bands["low"][0] + bands["mid"][0] + bands["high"][0]
        )
        if pac_power / total_power < 0.30:
            return False

        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        """Compute the coupling score via PAC drop and trend analysis.

        Algorithm:
        1. FFT-decompose the window into 5 frequency bands.
        2. Compute composite PAC coupling via ``_compute_coupling``.
        3. Update coupling EWMA; freeze baseline after 10 post-warmup steps.
        4. ``drop_score``: how much coupling has fallen from the warmup
           baseline, mapped to [0, 1] via ``coupling_drop_threshold``.
        5. ``trend_score``: linear slope of coupling history, mapped to
           [0, 1] via ``trend_threshold``.
        6. Final score = max(drop_score, trend_score).

        Parameters
        ----------
        window : list of float
            Current rolling window.

        Returns
        -------
        score : float
            Coupling score in [0, 1].
        msg : str
            Diagnostic string with coupling value, baseline, trend, and
            noise fraction.
        """
        bands = _fft_bands(window)
        if bands is None:
            return 0.0, "fft unavailable"

        coupling = self._compute_coupling(bands)

        # Update EWMA
        alpha = 0.15
        if self._coupling_n == 0:
            self._coupling_ewma = coupling
        else:
            self._coupling_ewma = alpha * coupling + (1 - alpha) * self._coupling_ewma
        self._coupling_history.append(coupling)
        self._coupling_n += 1

        # Set baseline after warmup
        if not self._baseline_set and self._coupling_n >= 10:
            self._warmup_coupling = self._coupling_ewma
            self._baseline_set = True

        # Coupling trend: slope over recent history
        h = list(self._coupling_history)
        if len(h) >= 5:
            n = len(h)
            xbar = (n - 1) / 2.0
            ybar = _mean(h)
            num = sum((i - xbar) * (h[i] - ybar) for i in range(n))
            den = sum((i - xbar) ** 2 for i in range(n))
            trend = num / (den + 1e-10)
        else:
            trend = 0.0

        # Score components
        drop_score = 0.0
        trend_score = 0.0

        if self._baseline_set and self._warmup_coupling > 1e-6:
            # How much has coupling dropped from baseline?
            ratio = coupling / (self._warmup_coupling + 1e-10)
            # ratio = 1.0 → no drop (score 0); ratio = coupling_drop_threshold → score 1
            drop_score = max(0.0, min(1.0,
                (1.0 - ratio) / (1.0 - self._coupling_drop_threshold + 1e-10)
            ))

        if trend < 0:
            # Negative trend: declining coupling
            # trend = trend_threshold → score 1
            trend_score = min(1.0, abs(trend) / (abs(self._trend_threshold) + 1e-10))

        score = max(drop_score, trend_score)

        noise_frac = self._noise_fraction(bands)
        msg = (
            f"coupling={coupling:.3f}(base={self._warmup_coupling:.3f}) "
            f"trend={trend:.4f} drop_score={drop_score:.3f} "
            f"noise_frac={noise_frac:.2f}"
        )
        return score, msg

    def reset(self) -> None:
        """Clear all state, including PAC history buffers, coupling EWMA, and baseline."""
        super().reset()
        self._lo_phases.clear()
        self._hi_mags.clear()
        self._mid_phases.clear()
        self._high_mags.clear()
        self._coupling_history.clear()
        self._coupling_ewma = 0.0
        self._coupling_n = 0
        self._warmup_coupling = 0.0
        self._baseline_set = False

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all detector state.

        Returns
        -------
        dict
            Contains coupling EWMA, baseline, coupling history, and
            base-class state.
        """
        sd = super().state_dict()
        sd.update({
            "coupling_ewma": self._coupling_ewma,
            "coupling_n": self._coupling_n,
            "warmup_coupling": self._warmup_coupling,
            "baseline_set": self._baseline_set,
            "coupling_history": list(self._coupling_history),
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        """Restore detector state from a snapshot produced by ``state_dict``.

        Parameters
        ----------
        sd : dict
            Snapshot dictionary.
        """
        super().load_state(sd)
        self._coupling_ewma = sd.get("coupling_ewma", 0.0)
        self._coupling_n = sd.get("coupling_n", 0)
        self._warmup_coupling = sd.get("warmup_coupling", 0.0)
        self._baseline_set = sd.get("baseline_set", False)
        self._coupling_history.clear()
        self._coupling_history.extend(sd.get("coupling_history", []))
