# fracttalix/suite/drift.py
# DriftDetector — Slow distribution shift via non-adaptive CUSUM + Page-Hinkley.
#
# Theorem basis (P3 / CUSUM + Page-Hinkley):
#   The key insight: an EWMA baseline adapts to slow drift, masking it.
#   A non-adaptive (warmup-frozen) baseline does not adapt — slow drift
#   accumulates in the CUSUM statistic and eventually crosses the threshold.
#   Page-Hinkley is a second independent vote, sensitive to gradual monotonic
#   changes in the cumulative mean.
#
#   Both tests operate on z_raw = (x − warmup_mean) / warmup_std, so:
#     - Point anomalies cause one large z_raw spike that resets the accumulator.
#     - Slow drift causes many small z_raw values that accumulate persistently.
#   This natural separation is why CUSUM and PH dominate drift benchmarks.
#
# OUT_OF_SCOPE conditions:
#   • Sudden large spike (point anomaly): z_raw >> 5 in a single step.
#     The CUSUM accumulator resets, so individual spikes don't fool us, but
#     we explicitly declare OUT_OF_SCOPE so the user knows this is Discord's job.
#   • High baseline variance (signal was already noisy at warmup): the warmup
#     std absorbs the noise and z_raw is well-behaved; this is fine.  We only
#     go OUT_OF_SCOPE if variance has grown >4× since warmup (VarianceDetector).
#   • Insufficient warmup to freeze a reliable baseline.
#
# Best at: slow monotonic mean shifts (μ changes by 0.5–2σ over 50–500 steps).
# Mediocre at: oscillating mean (CUSUM and PH cancel out).
# Useless at: point anomalies (resets the accumulator; DiscordDetector handles).

import math
from collections import deque
from typing import Any, Dict, List, Tuple

from fracttalix.suite.base import (
    BaseDetector, ScopeStatus,
    _mean, _variance, _std, _ac1,
)


class DriftDetector(BaseDetector):
    """Slow distribution-shift detector via frozen-baseline CUSUM + Page-Hinkley.

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
    ph_delta : float
        Page-Hinkley minimum detectable change (default 0.3).
    ph_lambda : float
        Page-Hinkley threshold (default 15.0).
    spike_z : float
        z_raw above this is classified as a point spike → OUT_OF_SCOPE
        (default 5.0).
    var_growth_threshold : float
        Variance ratio (current / warmup) above this → OUT_OF_SCOPE (default 4.0).
    drift_threshold : float
        Combined CUSUM+PH score threshold for ALERT (default 0.50).
    """

    def __init__(
        self,
        warmup: int = 100,
        window: int = 300,
        cusum_k: float = 0.5,
        cusum_h: float = 8.0,
        ph_delta: float = 0.3,
        ph_lambda: float = 15.0,
        spike_z: float = 5.0,
        var_growth_threshold: float = 4.0,
        drift_threshold: float = 0.50,
    ):
        super().__init__("DriftDetector", warmup=warmup, window_size=window)
        self._cusum_k = cusum_k
        self._cusum_h = cusum_h
        self._ph_delta = ph_delta
        self._ph_lambda = ph_lambda
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

        # Page-Hinkley state
        self._ph_cum_hi: float = 0.0
        self._ph_cum_lo: float = 0.0
        self._ph_m_hi: float = 0.0
        self._ph_m_lo: float = 0.0

        # Spike gate cooldown (prevent OUT_OF_SCOPE on single-step transient)
        self._spike_steps_ago: int = 999

        # Track consecutive drift alerts for stronger scoring
        self._drift_count: int = 0

    def _set_baseline(self, window: List[float]) -> None:
        self._warmup_mean = _mean(window)
        self._warmup_std = max(_std(window), 1e-10)
        self._warmup_var = max(_variance(window), 1e-10)
        self._baseline_set = True

    def _check_scope(self, window: List[float]) -> bool:
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
        if abs(recent_ac1) > 0.30:
            return False

        # Scope gate: variance has grown substantially → VarianceDetector
        recent_var = _variance(window[-min(40, len(window)):])
        if recent_var > self._var_growth_threshold * self._warmup_var:
            return False

        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        x = window[-1]
        z_raw = (x - self._warmup_mean) / self._warmup_std
        k = self._cusum_k

        # Bidirectional CUSUM
        self._s_hi = max(0.0, self._s_hi + z_raw - k)
        self._s_lo = max(0.0, self._s_lo - z_raw - k)
        cusum_alert = (self._s_hi > self._cusum_h) or (self._s_lo > self._cusum_h)
        cusum_score = min(1.0, max(self._s_hi, self._s_lo) / (self._cusum_h + 1e-10))
        if cusum_alert:
            self._s_hi = 0.0
            self._s_lo = 0.0

        # Page-Hinkley
        mu = self._warmup_mean
        delta = self._ph_delta
        self._ph_cum_hi += (x - mu - delta)
        self._ph_cum_lo += (mu - x - delta)
        self._ph_m_hi = max(self._ph_m_hi, self._ph_cum_hi)
        self._ph_m_lo = max(self._ph_m_lo, self._ph_cum_lo)
        ph_alert = (
            (self._ph_m_hi - self._ph_cum_hi > self._ph_lambda) or
            (self._ph_m_lo - self._ph_cum_lo > self._ph_lambda)
        )
        ph_score = min(1.0, max(
            self._ph_m_hi - self._ph_cum_hi,
            self._ph_m_lo - self._ph_cum_lo,
        ) / (self._ph_lambda + 1e-10))
        if ph_alert:
            self._ph_cum_hi = 0.0
            self._ph_cum_lo = 0.0
            self._ph_m_hi = 0.0
            self._ph_m_lo = 0.0

        # Combined score: max of the two independent tests
        score = max(cusum_score, ph_score)

        if cusum_alert or ph_alert:
            self._drift_count += 1
        else:
            self._drift_count = max(0, self._drift_count - 1)

        signals = []
        if cusum_alert:
            signals.append("cusum")
        if ph_alert:
            signals.append("page-hinkley")
        sig_str = "+".join(signals) if signals else "none"

        msg = (
            f"z_raw={z_raw:.3f} cusum=({self._s_hi:.2f},{self._s_lo:.2f}) "
            f"ph_gap=({self._ph_m_hi - self._ph_cum_hi:.2f},{self._ph_m_lo - self._ph_cum_lo:.2f}) "
            f"signals=[{sig_str}]"
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
        self._ph_cum_hi = 0.0
        self._ph_cum_lo = 0.0
        self._ph_m_hi = 0.0
        self._ph_m_lo = 0.0
        self._spike_steps_ago = 999
        self._drift_count = 0

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update({
            "baseline_set": self._baseline_set,
            "warmup_mean": self._warmup_mean,
            "warmup_std": self._warmup_std,
            "warmup_var": self._warmup_var,
            "s_hi": self._s_hi,
            "s_lo": self._s_lo,
            "ph_cum_hi": self._ph_cum_hi,
            "ph_cum_lo": self._ph_cum_lo,
            "ph_m_hi": self._ph_m_hi,
            "ph_m_lo": self._ph_m_lo,
            "spike_steps_ago": self._spike_steps_ago,
            "drift_count": self._drift_count,
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
        self._ph_cum_hi = sd.get("ph_cum_hi", 0.0)
        self._ph_cum_lo = sd.get("ph_cum_lo", 0.0)
        self._ph_m_hi = sd.get("ph_m_hi", 0.0)
        self._ph_m_lo = sd.get("ph_m_lo", 0.0)
        self._spike_steps_ago = sd.get("spike_steps_ago", 999)
        self._drift_count = sd.get("drift_count", 0)
