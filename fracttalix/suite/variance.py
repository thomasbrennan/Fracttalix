"""VarianceDetector -- Sudden volatility change via CUSUM on squared residuals.

Theorem basis (P4 / VarCUSUM)
------------------------------
Variance changes are invisible to mean-tracking statistics.  A CUSUM
running on z-squared (squared standardised residuals) accumulates when
variance has shifted.  A single large spike raises z-squared once; a
variance explosion raises z-squared persistently -- the CUSUM accumulates
through the latter and resets through the former.

Design note -- z_raw not z_adaptive
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``z_raw = (x - warmup_mean) / warmup_std`` uses the frozen baseline, so
``E[z_raw**2] = 1.0`` on stationary data.  An adaptive EWMA-MAD baseline
instead produces ``E[dev_ewma] ~ 0.8 * sigma`` (tracking MAE not std),
inflating ``E[z_adaptive**2] ~ 1.57`` and causing systematic CUSUM
accumulation on stationary data (false positives).

Two complementary tests:

1. **VarCUSUM**: CUSUM on z-squared (responsive, resets after each crossing).
2. **Sustained variance**: windowed variance vs warmup baseline (catches
   prolonged volatility regimes that individual crossings miss).

OUT_OF_SCOPE conditions
-----------------------
- Mean is shifting (DriftDetector's domain): if the warmup-frozen z_raw is
  drifting monotonically, CUSUM on z-squared will also fire.  We check for
  this and declare OUT_OF_SCOPE.
- Insufficient warmup to build a variance baseline.

Strengths and limitations
-------------------------
Best at
    Sudden volatility explosions, regime switches in noise level.
Mediocre at
    Slow variance growth (accumulates slowly; CouplingDetector may see it
    first via PAC degradation).
Useless at
    Mean shifts without variance change (always OUT_OF_SCOPE).
"""

import math
from collections import deque
from typing import Any, Dict, List, Tuple

from fracttalix.suite.base import (
    BaseDetector, ScopeStatus,
    _mean, _variance, _std, _linear_trend,
)


class VarianceDetector(BaseDetector):
    """Sudden volatility-change detector via CUSUM on squared residuals.

    Parameters
    ----------
    warmup : int
        Observations to build variance baseline (default 80).
    window : int
        Rolling window (default 200).
    var_cusum_k : float
        CUSUM reference value.  Derived from the log-likelihood ratio for
        detecting a 4× variance increase: k = log(2)×8/3 ≈ 1.848.  Under
        null N(0,1), E[z²−k] = 1−1.848 = −0.848 < 0 → correct null drift
        (default 1.848).
    var_cusum_h : float
        CUSUM threshold.  Calibrated for ~0.43% FPR on N(0,1) null using
        the LLR-derived k (default 13.5).
    sustained_ratio : float
        Ratio current_variance / warmup_variance above which sustained-variance
        alert fires (default 4.0).
    sustained_window : int
        Window for sustained variance check (default 40).
    drift_trend_threshold : float
        If |linear_trend(z_raw)| > this the mean is drifting → OUT_OF_SCOPE
        (default 0.06).
    variance_threshold : float
        Score threshold for ALERT (default 0.50).
    """

    def __init__(
        self,
        warmup: int = 80,
        window: int = 200,
        var_cusum_k: float = 1.848,
        var_cusum_h: float = 13.5,
        sustained_ratio: float = 4.0,
        sustained_window: int = 40,
        drift_trend_threshold: float = 0.06,
        variance_threshold: float = 0.50,
    ):
        super().__init__("VarianceDetector", warmup=warmup, window_size=window)
        self._var_cusum_k = var_cusum_k
        self._var_cusum_h = var_cusum_h
        self._sustained_ratio = sustained_ratio
        self._sustained_window = sustained_window
        self._drift_trend_threshold = drift_trend_threshold
        self._alert_threshold = variance_threshold

        # Warmup-frozen baseline (for sustained-variance ratio check only)
        self._warmup_mean: float = 0.0
        self._warmup_std: float = 1.0
        self._warmup_var: float = 1.0
        self._baseline_set: bool = False

        # VarCUSUM state
        self._s_hi: float = 0.0
        self._s_lo: float = 0.0
        self._var_ewma: float = 0.0
        self._warmed_cusum: bool = False

    def _set_baseline(self, window: List[float]) -> None:
        """Freeze the warmup mean, std, and variance as the VarCUSUM reference.

        The frozen baseline ensures ``E[z_raw**2] = 1.0`` on stationary data,
        giving the CUSUM correct null-distribution behaviour.

        Parameters
        ----------
        window : list of float
            The warmup observations to compute the baseline from.
        """
        self._warmup_mean = _mean(window)
        self._warmup_std = max(_std(window), 1e-10)
        self._warmup_var = max(_variance(window), 1e-6)
        self._baseline_set = True

    def _z_raws(self, window: List[float]) -> List[float]:
        """Standardise observations using the frozen warmup baseline.

        Parameters
        ----------
        window : list of float
            Raw observations to standardise.

        Returns
        -------
        list of float
            ``(x - warmup_mean) / warmup_std`` for each x in *window*.
        """
        mu = self._warmup_mean
        s = self._warmup_std
        return [(x - mu) / s for x in window]

    def _check_scope(self, window: List[float]) -> bool:
        """Determine whether the signal is suitable for variance detection.

        Scope gates
        -----------
        1. Mean is drifting: ``|linear_trend(z_raw)| > drift_trend_threshold``
           -- a monotonic mean shift inflates z-squared; DriftDetector owns
           that.

        Parameters
        ----------
        window : list of float
            Current rolling window.

        Returns
        -------
        bool
            True if the signal is in scope for variance analysis.
        """
        if not self._baseline_set:
            self._set_baseline(window[:self._warmup])

        # Scope gate: mean is drifting → DriftDetector's domain
        z_raws = self._z_raws(window[-min(60, len(window)):])
        trend = abs(_linear_trend(z_raws))
        if trend > self._drift_trend_threshold:
            return False

        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        """Compute the variance score via one-sided VarCUSUM + sustained-variance ratio.

        Algorithm:
        1. Compute ``z_raw = (x - warmup_mean) / warmup_std`` and ``v2 = z_raw**2``.
        2. Update one-sided CUSUM accumulator: ``s_hi = max(0, s_hi + v2 - k)``.
           Detects variance *increase* only (decrease is not in scope).
        3. If ``s_hi > var_cusum_h``, fire a VarCUSUM alert and reset.
        4. Sustained-variance check: ``recent_var / warmup_var`` mapped to [0, 1].
        5. Score = max of CUSUM alert (1.0) and sustained score.  Pre-crossing
           score is capped at 0.49.

        Parameters
        ----------
        window : list of float
            Current rolling window.

        Returns
        -------
        score : float
            Variance score in [0, 1].
        msg : str
            Diagnostic string with z_raw-squared, CUSUM state, variance ratio,
            and signal type.
        """
        if not self._warmed_cusum:
            self._s_hi = 0.0
            self._s_lo = 0.0
            self._var_ewma = 1.0
            self._warmed_cusum = True

        x = window[-1]

        # z_raw uses frozen warmup baseline: E[z_raw²] = 1.0 on null data.
        # CUSUM k=1.0 → increments (v2−k) have mean 0 → correct null behaviour.
        z_raw = (x - self._warmup_mean) / self._warmup_std
        v2 = z_raw ** 2   # χ²(1) proxy; E=1.0 under null

        k = self._var_cusum_k
        self._var_ewma = 0.9 * self._var_ewma + 0.1 * v2

        # One-sided CUSUM: detect variance INCREASE only.
        # The downside CUSUM on chi²(1) has positive drift because P(v2<k)≈68.3%
        # (i.e. positive increments occur 68% of the time), causing systematic
        # accumulation on stationary data.  Variance decrease is not in scope.
        self._s_hi = max(0.0, self._s_hi + v2 - k)
        self._s_lo = 0.0

        cusum_alert = self._s_hi > self._var_cusum_h
        if cusum_alert:
            self._s_hi = 0.0
            self._s_lo = 0.0

        # Sustained variance check: compare recent windowed variance to frozen baseline
        sw = min(self._sustained_window, len(window))
        recent_var = _variance(window[-sw:])
        var_ratio = recent_var / (self._warmup_var + 1e-10)
        sustained_alert = var_ratio > self._sustained_ratio
        # Sustained score: 0.0 at 1x, 1.0 at sustained_ratio×
        sustained_score = min(1.0, max(0.0,
            (var_ratio - 1.0) / (self._sustained_ratio - 1.0 + 1e-10)
        ))

        # Score design: only reach alert threshold on actual crossings.
        # Pre-crossing accumulation is a monitoring signal (capped below threshold).
        if cusum_alert or sustained_alert:
            score = max(0.5 + 0.5 * sustained_score, 1.0 if cusum_alert else 0.0)
            score = min(1.0, score)
        else:
            # Continuous pre-crossing monitoring: never reaches alert threshold
            pre_score = max(self._s_hi, self._s_lo) / (self._var_cusum_h + 1e-10)
            score = min(0.49, max(pre_score * 0.49, sustained_score * 0.49))

        signals = []
        if cusum_alert:
            signals.append("var_cusum")
        if sustained_alert:
            signals.append(f"sustained({var_ratio:.1f}x)")
        sig_str = "+".join(signals) if signals else "none"

        msg = (
            f"z_raw²={v2:.3f} cusum=({self._s_hi:.2f},{self._s_lo:.2f}) "
            f"var_ratio={var_ratio:.2f} signals=[{sig_str}]"
        )
        return score, msg

    def reset(self) -> None:
        """Clear all state, including frozen baseline, CUSUM accumulators, and EWMA tracker."""
        super().reset()
        self._baseline_set = False
        self._warmup_mean = 0.0
        self._warmup_std = 1.0
        self._warmup_var = 1.0
        self._s_hi = 0.0
        self._s_lo = 0.0
        self._var_ewma = 0.0
        self._warmed_cusum = False

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all detector state.

        Returns
        -------
        dict
            Contains baseline, CUSUM accumulators, EWMA tracker, and
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
            "var_ewma": self._var_ewma,
            "warmed_cusum": self._warmed_cusum,
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
        self._var_ewma = sd.get("var_ewma", 0.0)
        self._warmed_cusum = sd.get("warmed_cusum", False)
