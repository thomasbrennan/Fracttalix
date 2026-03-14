"""DriftDetector -- Slow distribution shift via non-adaptive CUSUM (frozen baseline).

Theorem basis (P3 / CUSUM)
--------------------------
The key insight: an EWMA baseline adapts to slow drift, masking it.  A
non-adaptive (warmup-frozen) baseline does not adapt -- slow drift
accumulates in the CUSUM statistic and eventually crosses the threshold.

The test operates on ``z_raw = (x - warmup_mean) / warmup_std``, so:

- Point anomalies cause one large z_raw spike that resets the accumulator.
- Slow drift causes many small z_raw values that accumulate persistently.

This natural separation is why frozen-baseline CUSUM dominates drift
benchmarks.

Design note -- Page-Hinkley excluded by design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PH with a frozen baseline has ``E[ph_cum_lo increment] = -delta < 0`` on
stationary data, so the accumulator drifts monotonically downward and the
gap grows at ``delta/step``, reaching the threshold every ~50 steps on
stationary N(0,1) data (65 % FPR).  PH is correct only with an adaptive
baseline (which would mask drift) or with a one-shot post-hoc test.
Neither applies here.  CUSUM-only.

OUT_OF_SCOPE conditions
-----------------------
- Sudden large spike (point anomaly): ``|z_raw| >> 5`` in a single step.
  The CUSUM accumulator resets; we declare OUT_OF_SCOPE so the user knows
  this is DiscordDetector's job.
- High baseline variance: if variance has grown >4x since warmup,
  VarianceDetector owns it.
- Strongly autocorrelated / oscillatory signal: CUSUM accumulates on
  half-periods of a sinusoid; HopfDetector + CouplingDetector handle these.
- Insufficient warmup to freeze a reliable baseline.

Strengths and limitations
-------------------------
Best at
    Slow monotonic mean shifts (mu changes by 0.5-2 sigma over 50-500 steps).
Mediocre at
    Oscillating mean (CUSUM cancels out).
Useless at
    Point anomalies (resets the accumulator; DiscordDetector handles).
"""

from typing import Any, Dict, List, Tuple

from fracttalix.suite.base import (
    BaseDetector,
    _ac1,
    _mean,
    _std,
    _variance,
)


class DriftDetector(BaseDetector):
    """Slow distribution-shift detector via frozen-baseline CUSUM.

    Parameters
    ----------
    warmup : int
        Observations used to freeze the mean/std baseline (default 100).
    window : int
        Rolling window kept (default 300).
    cusum_k : float
        CUSUM allowance — half the minimum detectable shift in σ units
        (default 0.5).
    cusum_h : float
        CUSUM threshold.  Lower → more sensitive but more false positives
        (default 8.0).
    spike_z : float
        z_raw above this is classified as a point spike → OUT_OF_SCOPE
        (default 5.0).
    var_growth_threshold : float
        Variance ratio (current / warmup) above this → OUT_OF_SCOPE (default 4.0).
    drift_threshold : float
        CUSUM score threshold for ALERT (default 0.50).
    """

    def __init__(
        self,
        warmup: int = 100,
        window: int = 300,
        cusum_k: float = 0.5,
        cusum_h: float = 8.0,
        spike_z: float = 5.0,
        var_growth_threshold: float = 4.0,
        drift_threshold: float = 0.50,
    ):
        super().__init__("DriftDetector", warmup=warmup, window_size=window)
        self._cusum_k = cusum_k
        self._cusum_h = cusum_h
        self._spike_z = spike_z
        self._var_growth_threshold = var_growth_threshold
        self._alert_threshold = drift_threshold

        # Warmup-frozen baseline
        self._warmup_mean: float = 0.0
        self._warmup_std: float = 1.0
        self._warmup_var: float = 1.0
        self._baseline_set: bool = False

        # CUSUM state
        self._s_hi: float = 0.0
        self._s_lo: float = 0.0

        # Spike gate cooldown (prevent OUT_OF_SCOPE on single-step transient)
        self._spike_steps_ago: int = 999

        # Track consecutive drift alerts for stronger scoring
        self._drift_count: int = 0

    def _set_baseline(self, window: List[float]) -> None:
        """Freeze the warmup mean, std, and variance as the CUSUM reference.

        All subsequent z_raw values are computed relative to this frozen
        baseline so that slow drift accumulates rather than being absorbed.

        Parameters
        ----------
        window : list of float
            The warmup observations to compute the baseline from.
        """
        self._warmup_mean = _mean(window)
        self._warmup_std = max(_std(window), 1e-10)
        self._warmup_var = max(_variance(window), 1e-10)
        self._baseline_set = True

    def _check_scope(self, window: List[float]) -> bool:
        """Determine whether the signal is suitable for drift detection.

        Scope gates
        -----------
        1. Single-step spike: ``|z_raw| > spike_z`` -- point anomaly;
           DiscordDetector owns that.  Stays out of scope for 3 steps after.
        2. Strongly autocorrelated signal: ``|AC(1)| > 0.35`` -- oscillatory
           data causes false CUSUM accumulation on half-periods;
           HopfDetector + CouplingDetector own that.
        3. Variance has grown > ``var_growth_threshold`` * warmup variance --
           VarianceDetector owns that.

        Parameters
        ----------
        window : list of float
            Current rolling window.

        Returns
        -------
        bool
            True if the signal is in scope for drift analysis.
        """
        if not self._baseline_set:
            self._set_baseline(window[:self._warmup])

        self._spike_steps_ago += 1

        x = window[-1]
        z_raw = (x - self._warmup_mean) / self._warmup_std

        # Scope gate: single-step spike (point anomaly)
        if abs(z_raw) > self._spike_z:
            self._spike_steps_ago = 0
            return False

        # Stay out of scope for a few steps after a spike
        if self._spike_steps_ago < 3:
            return False

        # Scope gate: oscillatory signal.
        # CUSUM accumulates on half-periods of a sinusoid, causing false positives.
        # Oscillatory signals belong to HopfDetector + CouplingDetector.
        # We only fire if the signal is not strongly autocorrelated around its mean.
        recent = window[-min(50, len(window)):]
        recent_ac1 = _ac1(recent)
        if abs(recent_ac1) > 0.35:
            return False

        # Scope gate: variance has grown substantially → VarianceDetector
        recent_var = _variance(window[-min(40, len(window)):])
        if recent_var > self._var_growth_threshold * self._warmup_var:
            return False

        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        """Compute the drift score via bidirectional CUSUM on frozen-baseline z_raw.

        Algorithm:
        1. Compute ``z_raw = (x - warmup_mean) / warmup_std``.
        2. Update bidirectional CUSUM accumulators (s_hi, s_lo) with
           allowance ``cusum_k``.
        3. If either accumulator exceeds ``cusum_h``, fire a CUSUM alert
           and reset both accumulators.
        4. Score = 1.0 on threshold crossing; pre-crossing score is capped
           at 0.49 (monitoring signal only).

        Parameters
        ----------
        window : list of float
            Current rolling window.

        Returns
        -------
        score : float
            Drift score in [0, 1].
        msg : str
            Diagnostic string with z_raw, CUSUM accumulators, and signal type.
        """
        x = window[-1]
        z_raw = (x - self._warmup_mean) / self._warmup_std
        k = self._cusum_k

        # Bidirectional CUSUM on frozen-baseline z_raw.
        # PH excluded: with a frozen baseline E[ph_cum increment] = −δ < 0 on
        # stationary data, causing monotonic growth of the gap and ~65% FPR.
        self._s_hi = max(0.0, self._s_hi + z_raw - k)
        self._s_lo = max(0.0, self._s_lo - z_raw - k)
        cusum_alert = (self._s_hi > self._cusum_h) or (self._s_lo > self._cusum_h)
        if cusum_alert:
            self._s_hi = 0.0
            self._s_lo = 0.0

        # Score design: only reach alert threshold on threshold crossing.
        # Pre-crossing accumulation is monitoring signal, capped below 0.50.
        if cusum_alert:
            score = 1.0
        else:
            pre_score = max(self._s_hi, self._s_lo) / (self._cusum_h + 1e-10)
            score = min(0.49, pre_score * 0.49)

        if cusum_alert:
            self._drift_count += 1
        else:
            self._drift_count = max(0, self._drift_count - 1)

        sig_str = "cusum" if cusum_alert else "none"
        msg = (
            f"z_raw={z_raw:.3f} cusum=({self._s_hi:.2f},{self._s_lo:.2f}) "
            f"signals=[{sig_str}]"
        )
        return score, msg

    def reset(self) -> None:
        """Clear all state, including frozen baseline, CUSUM accumulators, and spike cooldown."""
        super().reset()
        self._baseline_set = False
        self._warmup_mean = 0.0
        self._warmup_std = 1.0
        self._warmup_var = 1.0
        self._s_hi = 0.0
        self._s_lo = 0.0
        self._spike_steps_ago = 999
        self._drift_count = 0

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all detector state.

        Returns
        -------
        dict
            Contains baseline, CUSUM accumulators, spike cooldown, and
            base-class state.
        """
        sd = super().state_dict()
        sd.update({
            "baseline_set": self._baseline_set,
            "warmup_mean": self._warmup_mean,
            "warmup_std": self._warmup_std,
            "warmup_var": self._warmup_var,
            "s_hi": self._s_hi,
            "s_lo": self._s_lo,
            "spike_steps_ago": self._spike_steps_ago,
            "drift_count": self._drift_count,
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
        self._baseline_set = sd.get("baseline_set", False)
        self._warmup_mean = sd.get("warmup_mean", 0.0)
        self._warmup_std = sd.get("warmup_std", 1.0)
        self._warmup_var = sd.get("warmup_var", 1.0)
        self._s_hi = sd.get("s_hi", 0.0)
        self._s_lo = sd.get("s_lo", 0.0)
        self._spike_steps_ago = sd.get("spike_steps_ago", 999)
        self._drift_count = sd.get("drift_count", 0)
