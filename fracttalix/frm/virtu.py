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
# STATUS: PLACEHOLDER — awaiting Lady Ada's implementation.
# API contract defined below; Lady Ada implements the body.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fracttalix.suite.base import BaseDetector, DetectorResult, ScopeStatus


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
        # PLACEHOLDER — Lady Ada implements this.
        # Contract:
        #   VirtuDetector is called via update_frm(lambda_val, lam_rate,
        #     time_to_bif, omega_in_scope) in FRMSuite, bypassing the raw
        #     window-based update().
        #   - If time_to_bif is None or lambda is not declining: NORMAL
        #   - If omega_trust and omega not in scope: OUT_OF_SCOPE
        #   - Report: ttb_conservative = time_to_bif / safety_factor
        #   - Score: 1.0 - min(1.0, ttb_conservative / horizon)
        #     where horizon = some expected planning window (e.g. 200 steps)
        #   - Message includes: ttb=X steps, confidence=HIGH/MEDIUM/LOW,
        #     safety_factor=Y, omega_confirmed=True/False
        return 0.0, "placeholder — awaiting implementation"

    def update_frm(
        self,
        lambda_val: Optional[float],
        lam_rate: float,
        time_to_bif: Optional[float],
        omega_in_scope: bool,
        step: int,
    ) -> DetectorResult:
        """FRMSuite-native update: receives pre-computed Lambda/Omega outputs."""
        # PLACEHOLDER — Lady Ada fills in.
        self._step += 1
        if step < self._warmup:
            return DetectorResult(
                detector=self._name,
                status=ScopeStatus.WARMUP,
                score=0.0,
                message=f"warmup ({step}/{self._warmup})",
                step=step,
            )
        return DetectorResult(
            detector=self._name,
            status=ScopeStatus.NORMAL,
            score=0.0,
            message="placeholder — awaiting implementation",
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
