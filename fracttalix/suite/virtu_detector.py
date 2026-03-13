# fracttalix/suite/virtu_detector.py
# VirtuDetector — FRM-derived decision rationality via Kramers scaling.
#
# What makes this unique:
#   - Every other detector outputs "alert" or "no alert"
#   - Virtu outputs "your decision window is closing"
#   - Based on Kramers scaling: σ_τ ~ (μ_c - μ)^(-1/2)
#     As the system approaches bifurcation, the uncertainty in
#     timing diverges. Early action has lower uncertainty but may
#     be premature. Late action has higher urgency but worse odds.
#   - Virtu tracks the RATIO of actionable information to timing
#     uncertainty — the "decision quality" metric
#
# No other detection system incorporates decision theory.
# Requires: Lambda detector output (λ, dλ/dt, time-to-transition)
#
# Design: Virtu wraps a LambdaDetector. It doesn't fit data itself —
# it interprets Lambda's output through the lens of decision theory.

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from fracttalix.suite.base import BaseDetector


class VirtuDetector(BaseDetector):
    """Decision rationality detector: when should you act?

    Virtu consumes Lambda detector output and applies Kramers scaling
    to estimate the decision quality at each moment:

    - Far from bifurcation: low urgency, high uncertainty → WAIT
    - Approaching: rising urgency, narrowing window → ACT SOON
    - Near bifurcation: extreme urgency, diverging uncertainty → ACT NOW
    - Past optimal window: uncertainty has overwhelmed → TOO LATE

    The "Virtù Window" is the interval where:
    - λ is declining with HIGH or MEDIUM confidence
    - Time-to-transition is within the decision horizon
    - Timing uncertainty hasn't yet diverged past the action threshold

    Parameters
    ----------
    lambda_detector : LambdaDetector
        The Lambda detector whose output Virtu interprets.
    decision_horizon : float
        Maximum time-to-transition (in steps) at which action is relevant.
    urgency_floor : float
        Minimum λ_rate magnitude for urgency to register.
    """

    def __init__(
        self,
        lambda_detector=None,
        decision_horizon: float = 100.0,
        urgency_floor: float = 0.001,
        warmup: int = 0,
        window_size: int = 128,
    ):
        super().__init__(
            name="Virtu",
            warmup=warmup,
            window_size=window_size,
        )
        self._lambda_det = lambda_detector
        self._decision_horizon = decision_horizon
        self._urgency_floor = urgency_floor
        self._alert_threshold = 0.5

        # Decision state
        self._decision_quality_history: deque = deque(maxlen=20)
        self._virtu_window_open = False
        self._window_opened_at: Optional[int] = None
        self._peak_quality = 0.0

    def _check_scope(self, window: List[float]) -> bool:
        # Virtu is in scope only when Lambda is in scope and producing values
        if self._lambda_det is None:
            return False
        scope = self._lambda_det.scope_status
        if scope in ("OUT_OF_SCOPE", "INSUFFICIENT_DATA"):
            return False
        if self._lambda_det.current_lambda is None:
            return False
        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        lam = self._lambda_det.current_lambda
        rate = self._lambda_det.lambda_rate
        ttb = self._lambda_det.time_to_transition
        scope = self._lambda_det.scope_status

        if lam is None:
            return 0.0, "no λ available"

        # LIMIT_CYCLE: system is stable, no decision needed
        if scope == "LIMIT_CYCLE":
            self._virtu_window_open = False
            return 0.0, f"limit cycle — no action needed (λ={lam:.4f})"

        # Kramers scaling: timing uncertainty σ_τ ~ 1/√λ
        # As λ → 0, σ_τ → ∞ (uncertainty diverges)
        if lam > 1e-6:
            timing_uncertainty = 1.0 / math.sqrt(lam)
        else:
            timing_uncertainty = 1e6  # effectively infinite

        # Urgency: how fast is λ declining?
        urgency = abs(rate) if rate < -self._urgency_floor else 0.0

        # Information quality: how much do we know about the trajectory?
        # High when λ is declining consistently and ttb is finite
        if ttb is not None and ttb > 0 and ttb < self._decision_horizon:
            # Time pressure: closer = more urgent
            time_pressure = 1.0 - (ttb / self._decision_horizon)
            # Uncertainty ratio: timing_uncertainty / ttb
            # When this > 1, uncertainty exceeds the time remaining
            uncertainty_ratio = timing_uncertainty / ttb if ttb > 1e-6 else 1e6
        else:
            time_pressure = 0.0
            uncertainty_ratio = 0.0

        # Decision quality = urgency × time_pressure / uncertainty
        # High when: λ declining fast, transition near, uncertainty manageable
        if uncertainty_ratio < 1.0 and urgency > 0:
            # GOOD WINDOW: uncertainty is less than time remaining
            decision_quality = min(1.0, urgency * time_pressure * 10.0 / (1.0 + timing_uncertainty * 0.1))
        elif uncertainty_ratio >= 1.0 and urgency > 0:
            # CLOSING WINDOW: uncertainty exceeds time remaining
            decision_quality = min(1.0, 0.8 * urgency * 5.0)
        else:
            decision_quality = 0.0

        self._decision_quality_history.append(decision_quality)

        # Track Virtù Window state
        if decision_quality > 0.3 and not self._virtu_window_open:
            self._virtu_window_open = True
            self._window_opened_at = self._step
            self._peak_quality = decision_quality
        elif self._virtu_window_open:
            self._peak_quality = max(self._peak_quality, decision_quality)
            if decision_quality < 0.1:
                self._virtu_window_open = False

        # Build message
        if decision_quality >= 0.7:
            phase = "ACT NOW"
        elif decision_quality >= 0.4:
            phase = "ACT SOON"
        elif decision_quality >= 0.2:
            phase = "MONITOR"
        else:
            phase = "WAIT"

        parts = [phase]
        parts.append(f"λ={lam:.4f}")
        parts.append(f"rate={rate:.4f}")
        if ttb is not None:
            parts.append(f"Δt={ttb:.1f}")
        parts.append(f"σ_τ={timing_uncertainty:.1f}")
        parts.append(f"quality={decision_quality:.2f}")
        if self._virtu_window_open:
            parts.append("WINDOW_OPEN")

        return decision_quality, " ".join(parts)

    @property
    def decision_quality(self) -> float:
        return self._decision_quality_history[-1] if self._decision_quality_history else 0.0

    @property
    def virtu_window_open(self) -> bool:
        return self._virtu_window_open

    @property
    def peak_quality(self) -> float:
        return self._peak_quality

    def reset(self) -> None:
        super().reset()
        self._decision_quality_history.clear()
        self._virtu_window_open = False
        self._window_opened_at = None
        self._peak_quality = 0.0

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd.update({
            "decision_quality_history": list(self._decision_quality_history),
            "virtu_window_open": self._virtu_window_open,
            "window_opened_at": self._window_opened_at,
            "peak_quality": self._peak_quality,
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        super().load_state(sd)
        self._decision_quality_history = deque(
            sd.get("decision_quality_history", []), maxlen=20
        )
        self._virtu_window_open = sd.get("virtu_window_open", False)
        self._window_opened_at = sd.get("window_opened_at")
        self._peak_quality = sd.get("peak_quality", 0.0)
