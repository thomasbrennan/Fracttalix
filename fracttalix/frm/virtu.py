# fracttalix/frm/virtu.py
# VirtuDetector — FRM time-to-bifurcation decision support.
#
# Theorem basis (FRM perturbation expansion):
#   Time-to-bifurcation Δt ≈ λ / |dλ/dt|
#   where λ = |α|/(Γ·τ_gen) and λ → 0 signals the Hopf bifurcation.
#
# VirtuDetector is NOT a binary alert detector. It reads Lambda and Omega
# outputs and produces a time estimate — the actionable time window before
# the system crosses into its bifurcation.
#
# Operational use:
#   If Lambda says λ is declining (CRITICAL_SLOWING) and Omega confirms
#   ω is still structured (FRM intact), Virtu's estimate is trustworthy.
#   If Omega reports ω drift (FRM structure degrading), Virtu's estimate
#   should be treated as a lower bound (could happen sooner).
#
# The Kramers correction (asymmetric cost):
#   In many applications, acting too late is worse than acting too early.
#   The conservative estimate is: Δt_conservative = Δt / safety_factor.
#   Default safety_factor = 1.0 (no correction); user may override.
#
# Implemented by Lady Ada (FRM physics layer).

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fracttalix.suite.base import BaseDetector, DetectorResult, ScopeStatus

# Planning horizon: ttb beyond this many steps scores near 0 (no urgency)
_HORIZON = 200.0


class VirtuDetector(BaseDetector):
    """FRM time-to-bifurcation estimator.

    Reads from the shared scratch space populated by Lambda (HopfDetector frm)
    and Omega, and synthesizes a time-to-bifurcation estimate.

    In the FRMSuite context, VirtuDetector is called AFTER Lambda and Omega
    have run on the same step, and it reads their outputs from a shared dict.

    Standalone use (without FRMSuite): pass lambda_val and lam_rate explicitly
    via update_frm() instead of update().

    Parameters
    ----------
    safety_factor : float
        Conservative multiplier: Δt_reported = Δt_raw / safety_factor.
        Default 1.0 (no correction). Values > 1 give earlier warnings.
    omega_trust : bool
        If True (default), only report when Omega is also in scope (FRM intact).
        If False, report based on Lambda alone.
    warmup : int
        Minimum Lambda history steps before reporting (default 20).
    """

    def __init__(
        self,
        safety_factor: float = 1.0,
        omega_trust: bool = True,
        warmup: int = 20,
    ):
        super().__init__("VirtuDetector", warmup=warmup, window_size=warmup)
        self._safety_factor = safety_factor
        self._omega_trust = omega_trust
        self._last_ttb: Optional[float] = None
        self._last_confidence: str = "LOW"

    def _check_scope(self, window: List[float]) -> bool:
        return True  # Scope determined by Lambda/Omega, not raw data

    def _compute(self, window: List[float]):
        # Called only via the generic update() path (no Lambda/Omega context).
        # VirtuDetector is designed for use via update_frm(); without FRM
        # inputs there is nothing to report.
        return 0.0, "use update_frm() for FRM-aware time-to-bifurcation estimates"

    def update_frm(
        self,
        lambda_val: Optional[float],
        lam_rate: float,
        time_to_bif: Optional[float],
        omega_in_scope: bool,
        step: int,
        baseline_ratio: float = 1.0,
    ) -> DetectorResult:
        """FRMSuite-native update: receives pre-computed Lambda/Omega outputs.

        Parameters
        ----------
        lambda_val : float or None
            Current fitted λ from Lambda detector.
        lam_rate : float
            Rate of change dλ/dt from Lambda detector (negative = declining).
        time_to_bif : float or None
            Raw time-to-bifurcation estimate from Lambda: λ / |dλ/dt|.
        omega_in_scope : bool
            Whether OmegaDetector is currently in scope (FRM structure intact).
        step : int
            Current observation step (from FRMSuite).
        baseline_ratio : float
            Ratio of current λ to baseline λ (from LambdaDetector.baseline_ratio).
            Values < 0.5 indicate significant critical slowing down and activate
            Virtu even when lam_rate is too smooth to cross the rate threshold.
        """
        self._step += 1

        if step < self._warmup:
            return DetectorResult(
                detector=self._name,
                status=ScopeStatus.WARMUP,
                score=0.0,
                message=f"warmup ({step}/{self._warmup})",
                step=step,
            )

        # Activate if either:
        # 1. Lambda rate is rapidly declining (rate-based, responsive to fast changes)
        # 2. Lambda baseline_ratio < 0.5 (ratio-based, catches gradual sustained decline)
        #    The 20-window rolling rate estimate is often too smooth to cross -1e-3;
        #    baseline_ratio gives a direct measure of cumulative λ decline.
        rate_declining = lam_rate < -1e-3
        ratio_declining = baseline_ratio < 0.5

        if not rate_declining and not ratio_declining:
            return DetectorResult(
                detector=self._name,
                status=ScopeStatus.NORMAL,
                score=0.0,
                message=f"lambda stable (rate={lam_rate:.5f} ratio={baseline_ratio:.2f})",
                step=step,
            )

        # Need at least a time-to-bif estimate for urgency scoring.
        # When ratio-based activation fires without a rate-based ttb, estimate
        # ttb from baseline_ratio: at ratio=0.5, halfway to bifurcation.
        if time_to_bif is None:
            if ratio_declining and lambda_val is not None and lambda_val > 0:
                # Rough estimate: remaining distance ∝ lambda_val × lambda_window
                time_to_bif = lambda_val * 20.0 * 4.0  # 20-window × fit_interval
            else:
                return DetectorResult(
                    detector=self._name,
                    status=ScopeStatus.NORMAL,
                    score=0.0,
                    message="no ttb estimate available",
                    step=step,
                )

        # omega_trust gate: if FRM structure (ω) is uncertain, report OUT_OF_SCOPE
        if self._omega_trust and not omega_in_scope:
            return DetectorResult(
                detector=self._name,
                status=ScopeStatus.OUT_OF_SCOPE,
                score=0.0,
                message="omega out of scope (FRM structure uncertain)",
                step=step,
            )

        # Kramers correction: conservative estimate (act earlier)
        # safety_factor > 1.0 → shorter reported window → earlier warning
        ttb_conservative = time_to_bif / max(self._safety_factor, 1e-10)
        self._last_ttb = ttb_conservative

        # Confidence grading: rate takes precedence; ratio gives LOW confidence
        if lam_rate < -0.01:
            confidence = "HIGH"
        elif lam_rate < -0.003:
            confidence = "MEDIUM"
        elif rate_declining:
            confidence = "LOW"
        else:
            # ratio-only activation — less precise than rate-based
            confidence = "LOW"
        self._last_confidence = confidence

        # Urgency score: 0 = distant (ttb >> horizon), 1 = imminent (ttb → 0)
        score = 1.0 - min(1.0, ttb_conservative / _HORIZON)

        activation = "rate" if rate_declining else "ratio"
        omega_confirmed = omega_in_scope
        msg = (
            f"ttb={ttb_conservative:.1f} confidence={confidence} "
            f"activation={activation} ratio={baseline_ratio:.2f} "
            f"safety_factor={self._safety_factor:.2f} omega_confirmed={omega_confirmed}"
        )

        status = ScopeStatus.ALERT if score >= self._alert_threshold else ScopeStatus.NORMAL
        return DetectorResult(
            detector=self._name,
            status=status,
            score=score,
            message=msg,
            step=step,
        )

    def reset(self) -> None:
        super().reset()
        self._last_ttb = None
        self._last_confidence = "LOW"

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update({
            "last_ttb": self._last_ttb,
            "last_confidence": self._last_confidence,
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        super().load_state(sd)
        self._last_ttb = sd.get("last_ttb", None)
        self._last_confidence = sd.get("last_confidence", "LOW")
