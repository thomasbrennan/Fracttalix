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
# STATUS: PLACEHOLDER — awaiting Lady Ada's implementation.
# Bill Joy has defined the API contract; Lady Ada implements the body.

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional

from fracttalix.suite.base import BaseDetector, DetectorResult, ScopeStatus


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
        # Placeholder: always in scope if we have data
        # Lady Ada will implement: check FFT has a dominant peak
        return True

    def _compute(self, window: List[float]):
        # PLACEHOLDER — Lady Ada implements this.
        # Contract:
        #   - Estimate omega_observed from FFT of window[-self._window_size_fft:]
        #   - In strong mode: compare to self._omega_predicted
        #     deviation = |omega_obs - omega_pred| / omega_pred
        #     If deviation > self._deviation_threshold for alert_steps consecutive
        #     steps: score → 1.0 (ALERT)
        #   - In weak mode: track omega stability (std of recent omega history)
        #   - Return (score, message) where message includes:
        #     omega_obs=X omega_pred=Y deviation=Z% mode=strong/weak
        #
        # Until Lady Ada implements: return NORMAL with explanation.
        mode = "strong" if self._strong_mode else "weak"
        pred_str = f" omega_pred={self._omega_predicted:.4f}" if self._omega_predicted else ""
        return 0.0, f"placeholder mode={mode}{pred_str} — awaiting implementation"

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
