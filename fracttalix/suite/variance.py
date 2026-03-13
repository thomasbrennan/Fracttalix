# fracttalix/suite/variance.py
# VarianceDetector — Sudden volatility change via CUSUM on squared residuals.
#
# Theorem basis (P4 / VarCUSUM):
#   Variance changes are invisible to mean-tracking statistics.  A CUSUM
#   running on z² (squared standardized residuals) accumulates when variance
#   has shifted.  A single large spike raises z² once; a variance explosion
#   raises z² persistently — the CUSUM accumulates through the latter and
#   resets through the former.
#
#   Two complementary tests:
#     1. VarCUSUM: CUSUM on z² (responsive, resets after each crossing).
#     2. Sustained variance: windowed variance vs warmup baseline (catches
#        prolonged volatility regimes that individual crossings miss).
#
# OUT_OF_SCOPE conditions:
#   • Mean is shifting (DriftDetector's domain): if the warmup-frozen z_raw
#     is drifting monotonically, CUSUM on z² will also fire (z² rises with
#     drift).  We check for this and declare OUT_OF_SCOPE.
#   • Insufficient warmup to build a variance baseline.
#
# Best at: sudden volatility explosions, regime switches in noise level.
# Mediocre at: slow variance growth (accumulates slowly — CouplingDetector
#              may see it first via PAC degradation).
# Useless at: mean shifts without variance change (always OUT_OF_SCOPE).

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
        CUSUM allowance for variance statistic z² (default 1.0 per v12.2 fix).
    var_cusum_h : float
        CUSUM threshold (default 12.0).
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
        var_cusum_k: float = 1.0,
        var_cusum_h: float = 12.0,
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

        # Adaptive EWMA for CUSUM z-score (tracks signal level so oscillations
        # don't accumulate as false variance changes)
        self._ewma: float = 0.0
        self._dev_ewma: float = 1.0
        self._ewma_init: bool = False

    def _set_baseline(self, window: List[float]) -> None:
        self._warmup_mean = _mean(window)
        self._warmup_std = max(_std(window), 1e-10)
        self._warmup_var = max(_variance(window), 1e-6)
        self._baseline_set = True

    def _z_raws(self, window: List[float]) -> List[float]:
        mu = self._warmup_mean
        s = self._warmup_std
        return [(x - mu) / s for x in window]

    def _check_scope(self, window: List[float]) -> bool:
        if not self._baseline_set:
            self._set_baseline(window[:self._warmup])

        # Scope gate: mean is drifting → DriftDetector's domain
        z_raws = self._z_raws(window[-min(60, len(window)):])
        trend = abs(_linear_trend(z_raws))
        if trend > self._drift_trend_threshold:
            return False

        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        if not self._warmed_cusum:
            self._s_hi = 0.0
            self._s_lo = 0.0
            self._var_ewma = 1.0
            self._ewma = _mean(window[-20:])
            self._dev_ewma = max(_std(window[-20:]), 1e-10)
            self._ewma_init = True
            self._warmed_cusum = True

        x = window[-1]

        # Update adaptive EWMA (tracks oscillation/trend, so z stays ~N(0,1) on stable data)
        alpha = 0.1
        prev_ewma = self._ewma
        self._ewma = alpha * x + (1 - alpha) * self._ewma
        err = abs(x - prev_ewma)
        self._dev_ewma = alpha * err + (1 - alpha) * self._dev_ewma
        self._dev_ewma = max(self._dev_ewma, 1e-10)

        # CUSUM z-score uses adaptive baseline (oscillation-immune)
        z_adaptive = (x - self._ewma) / self._dev_ewma
        v2 = z_adaptive ** 2   # variance proxy

        k = self._var_cusum_k
        self._var_ewma = 0.9 * self._var_ewma + 0.1 * v2

        self._s_hi = max(0.0, self._s_hi + v2 - k)
        # Only accumulate downside if variance baseline established
        if self._var_ewma > 1e-4:
            self._s_lo = max(0.0, self._s_lo + k - v2)
        else:
            self._s_lo = 0.0

        cusum_alert = (self._s_hi > self._var_cusum_h) or (self._s_lo > self._var_cusum_h)
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
            f"z_adap²={v2:.3f} cusum=({self._s_hi:.2f},{self._s_lo:.2f}) "
            f"var_ratio={var_ratio:.2f} signals=[{sig_str}]"
        )
        return score, msg

    def reset(self) -> None:
        super().reset()
        self._baseline_set = False
        self._warmup_mean = 0.0
        self._warmup_std = 1.0
        self._warmup_var = 1.0
        self._s_hi = 0.0
        self._s_lo = 0.0
        self._var_ewma = 0.0
        self._warmed_cusum = False
        self._ewma = 0.0
        self._dev_ewma = 1.0
        self._ewma_init = False

    def state_dict(self) -> Dict[str, Any]:
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
            "ewma": self._ewma,
            "dev_ewma": self._dev_ewma,
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        super().load_state(sd)
        self._baseline_set = sd.get("baseline_set", False)
        self._warmup_mean = sd.get("warmup_mean", 0.0)
        self._warmup_std = sd.get("warmup_std", 1.0)
        self._warmup_var = sd.get("warmup_var", 1.0)
        self._s_hi = sd.get("s_hi", 0.0)
        self._s_lo = sd.get("s_lo", 0.0)
        self._var_ewma = sd.get("var_ewma", 0.0)
        self._warmed_cusum = sd.get("warmed_cusum", False)
        self._ewma = sd.get("ewma", 0.0)
        self._dev_ewma = sd.get("dev_ewma", 1.0)
