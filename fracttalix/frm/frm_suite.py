# fracttalix/frm/frm_suite.py
# FRMSuite — unified two-layer detection suite for FRM-shaped signals.
#
# Architecture:
#   Layer 1 (generic, no scipy): DetectorSuite
#     HopfDetector(ews), DiscordDetector, DriftDetector,
#     VarianceDetector, CouplingDetector
#
#   Layer 2 (FRM physics, numpy only):
#     Lambda  = LambdaDetector v2  — track λ → 0 via variance-inversion + spectral width
#     Omega   = OmegaDetector      — track ω vs π/(2·τ_gen)   [Lady Ada]
#     Virtu   = VirtuDetector      — time-to-bifurcation        [Lady Ada]
#
# frm_confidence (0–3):
#   Counts Layer 2 detectors in ALERT, strong mode only.
#   frm_confidence = 3 means Lambda + Omega + Virtu all alerting simultaneously.
#   Lambda alone = 1. Lambda + Omega = 2. Lambda + Omega + Virtu = 3.
#   Note: CouplingDetector is Layer 1; it does NOT add to frm_confidence.
#   frm_confidence_plus: frm_confidence + (1 if CouplingDetector alerts).
#   CouplingDetector provides independent structural cross-validation of Omega.
#
# Graceful degradation:
#   Without scipy: Layer 2 raises ImportError on first update().
#   FRMSuite catches this and sets layer2_available=False.
#   Layer 1 continues operating normally.
#
# Usage::
#
#   from fracttalix.frm import FRMSuite
#   suite = FRMSuite(tau_gen=12.5)
#   for value in stream:
#       result = suite.update(value)
#       print(result.summary())
#       if result.frm_confidence >= 2:
#           # Lambda + Omega both alerting: strong FRM bifurcation signal
#           print(f"FRM alert: confidence={result.frm_confidence}")

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Iterator, List, Optional

from fracttalix.suite import DetectorSuite, SuiteResult, ScopeStatus, DetectorResult
from fracttalix.suite.lambda_detector import LambdaDetector
from fracttalix.frm.omega import OmegaDetector
from fracttalix.frm.virtu import VirtuDetector


@dataclasses.dataclass(frozen=True)
class FRMSuiteResult:
    """One observation's output from the full FRM suite (Layer 1 + Layer 2).

    Attributes
    ----------
    layer1 : SuiteResult
        All 5 Layer 1 detector results.
    lambda_ : DetectorResult
        HopfDetector(frm) result — FRM damping decay.
    omega : DetectorResult
        OmegaDetector result — FRM frequency integrity.
    virtu : DetectorResult
        VirtuDetector result — time-to-bifurcation estimate.
    frm_confidence : int
        Number of FRM Layer 2 detectors in ALERT, strong mode (0–3).
        Lambda + Omega contribute 1 each when alerting.
        Lambda alone = 1. Lambda + Omega = 2.
        All three Layer 2 (Lambda + Omega + Virtu triggering) = 3.
    frm_confidence_plus : int
        frm_confidence + 1 if CouplingDetector (Layer 1) also alerts.
        CouplingDetector is independent of FRM physics — its confirmation
        is the strongest cross-validation signal available.
    layer2_available : bool
        False if scipy is absent. Layer 2 results will all be WARMUP.
    """

    layer1: SuiteResult
    lambda_: DetectorResult
    omega: DetectorResult
    virtu: DetectorResult
    frm_confidence: int
    frm_confidence_plus: int
    layer2_available: bool

    def __iter__(self) -> Iterator[DetectorResult]:
        return iter([
            self.layer1.hopf, self.layer1.discord, self.layer1.drift,
            self.layer1.variance, self.layer1.coupling,
            self.lambda_, self.omega, self.virtu,
        ])

    @property
    def any_alert(self) -> bool:
        return self.layer1.any_alert or any(
            r.is_alert for r in [self.lambda_, self.omega, self.virtu]
        )

    @property
    def alerts(self) -> List[DetectorResult]:
        return [r for r in self if r.is_alert]

    def summary(self) -> str:
        """Multi-line dashboard string."""
        l1 = self.layer1.summary()
        frm_parts = []
        for label, r in [("Lam", self.lambda_), ("Omg", self.omega), ("Vrt", self.virtu)]:
            if r.status == ScopeStatus.WARMUP:
                frm_parts.append(f"{label}:WARM")
            elif r.status == ScopeStatus.OUT_OF_SCOPE:
                frm_parts.append(f"{label}:OOS")
            elif r.status == ScopeStatus.ALERT:
                frm_parts.append(f"{label}:ALERT({r.score:.2f})")
            else:
                frm_parts.append(f"{label}:ok({r.score:.2f})")

        frm_line = " | ".join(frm_parts)
        avail = "scipy:ok" if self.layer2_available else "scipy:missing"
        conf = f"frm_conf={self.frm_confidence}+{self.frm_confidence_plus - self.frm_confidence}"
        return f"L1: {l1}\nL2: {frm_line} [{conf}] [{avail}]"


class FRMSuite:
    """Unified FRM detection suite: 5 generic + 3 FRM-physics detectors.

    Parameters
    ----------
    tau_gen : float or None
        FRM generation delay. Supplies both Lambda (fixes ω = π/(2·τ_gen))
        and Omega (strong mode: checks observed ω = predicted ω).
        If None: Lambda estimates ω from FFT; Omega runs in weak mode.
        Best results require tau_gen from domain knowledge.
    hopf_ews_kwargs : dict, optional
        Extra kwargs for HopfDetector(method='ews') in Layer 1.
    lambda_kwargs : dict, optional
        Extra kwargs for HopfDetector(method='frm') — Lambda.
    omega_kwargs : dict, optional
        Extra kwargs for OmegaDetector.
    virtu_kwargs : dict, optional
        Extra kwargs for VirtuDetector.
    layer1_kwargs : dict, optional
        Dict of dicts: keys 'discord', 'drift', 'variance', 'coupling'.
        Forwarded to respective Layer 1 detectors.
    """

    def __init__(
        self,
        tau_gen: Optional[float] = None,
        hopf_ews_kwargs: Optional[Dict] = None,
        lambda_kwargs: Optional[Dict] = None,
        omega_kwargs: Optional[Dict] = None,
        virtu_kwargs: Optional[Dict] = None,
        layer1_kwargs: Optional[Dict] = None,
    ):
        self.tau_gen = tau_gen
        l1kw = layer1_kwargs or {}

        # Layer 1: all 5 generic detectors
        self._layer1 = DetectorSuite(
            hopf_kwargs={**(hopf_ews_kwargs or {})},
            discord_kwargs=l1kw.get("discord", {}),
            drift_kwargs=l1kw.get("drift", {}),
            variance_kwargs=l1kw.get("variance", {}),
            coupling_kwargs=l1kw.get("coupling", {}),
        )

        # Layer 2: FRM physics
        # LambdaDetector v2: variance-inversion + spectral peak width.
        # Replaces HopfDetector(method='frm') which used parametric curve_fit —
        # validated to fail on nonlinear Hopf normal form data.
        lam_kw = {**(lambda_kwargs or {})}
        if tau_gen is not None:
            lam_kw.setdefault("tau_gen", tau_gen)
        self._lambda = LambdaDetector(**lam_kw)

        omg_kw = {**(omega_kwargs or {})}
        if tau_gen is not None:
            omg_kw.setdefault("tau_gen", tau_gen)
        self._omega = OmegaDetector(**omg_kw)

        self._virtu = VirtuDetector(**(virtu_kwargs or {}))

        self._lambda_available: Optional[bool] = None   # None = not yet tested
        self._omega_available: Optional[bool] = None    # tracked separately
        self._step = 0

    @property
    def _layer2_available(self) -> bool:
        """True if at least one Layer 2 detector is available."""
        return (self._lambda_available is not False) or (self._omega_available is not False)

    def update(self, value: float) -> FRMSuiteResult:
        """Feed one observation. Returns FRMSuiteResult."""
        step = self._step
        self._step += 1

        # Layer 1 always runs
        l1_result = self._layer1.update(value)

        # Layer 2: attempt each detector independently; scipy unavailability in
        # Lambda (curve_fit) must NOT block Omega (numpy-only).
        lambda_result = self._run_layer2_detector(
            self._lambda, value, "_lambda_available"
        )
        omega_result = self._run_layer2_detector(
            self._omega, value, "_omega_available"
        )

        # Virtu reads Lambda/Omega outputs via update_frm
        virtu_result = self._run_virtu(lambda_result, omega_result, step)

        # frm_confidence: count Layer 2 detectors alerting in strong mode
        frm_conf = 0
        lambda_strong = (self.tau_gen is not None)  # Lambda is strong when tau_gen fixed
        omega_strong = (self.tau_gen is not None)   # Omega is strong when tau_gen known

        if lambda_result.is_alert and lambda_strong:
            frm_conf += 1
        if omega_result.is_alert and omega_strong:
            frm_conf += 1
        if virtu_result.is_alert:
            frm_conf += 1

        # frm_confidence_plus: add CouplingDetector cross-validation
        frm_conf_plus = frm_conf + (1 if l1_result.coupling.is_alert else 0)

        return FRMSuiteResult(
            layer1=l1_result,
            lambda_=lambda_result,
            omega=omega_result,
            virtu=virtu_result,
            frm_confidence=frm_conf,
            frm_confidence_plus=frm_conf_plus,
            layer2_available=(
                self._lambda_available is not False
                or self._omega_available is not False
            ),
        )

    def _run_layer2_detector(
        self, detector, value: float, avail_attr: str
    ) -> DetectorResult:
        """Run one Layer 2 detector, tracking its own ImportError state.

        Each detector tracks availability independently: Lambda requires scipy,
        Omega requires only numpy.  A failed Lambda must not block Omega.
        """
        if getattr(self, avail_attr) is False:
            return DetectorResult(
                detector=detector._name,
                status=ScopeStatus.OUT_OF_SCOPE,
                score=0.0,
                message="optional dependency not available",
                step=self._step - 1,
            )
        try:
            result = detector.update(value)
            if getattr(self, avail_attr) is None:
                setattr(self, avail_attr, True)
            return result
        except ImportError as exc:
            setattr(self, avail_attr, False)
            return DetectorResult(
                detector=detector._name,
                status=ScopeStatus.OUT_OF_SCOPE,
                score=0.0,
                message=f"optional dependency not available: {exc}",
                step=self._step - 1,
            )

    def _run_virtu(
        self,
        lambda_result: DetectorResult,
        omega_result: DetectorResult,
        step: int,
    ) -> DetectorResult:
        """Run VirtuDetector with Lambda/Omega outputs.

        Reads LambdaDetector properties directly (not via message parsing)
        for reliability. baseline_ratio provides Virtu activation even when
        lam_rate is too smooth (20-window rolling) to cross the rate threshold.
        """
        if self._lambda_available is False:
            return DetectorResult(
                detector="VirtuDetector",
                status=ScopeStatus.OUT_OF_SCOPE,
                score=0.0,
                message="Lambda unavailable",
                step=step,
            )

        # Read Lambda state directly from the detector object
        lam_val = self._lambda.current_lambda
        lam_rate = self._lambda.lambda_rate
        ttb = self._lambda.time_to_transition
        baseline_ratio = self._lambda.baseline_ratio

        omega_in_scope = omega_result.status in (ScopeStatus.NORMAL, ScopeStatus.ALERT)

        return self._virtu.update_frm(
            lambda_val=lam_val,
            lam_rate=lam_rate,
            time_to_bif=ttb,
            omega_in_scope=omega_in_scope,
            step=step,
            baseline_ratio=baseline_ratio,
        )

    def reset(self) -> None:
        """Reset all detectors."""
        self._layer1.reset()
        self._lambda.reset()
        self._omega.reset()
        self._virtu.reset()
        self._step = 0
        self._lambda_available = None
        self._omega_available = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "layer1": self._layer1.state_dict(),
            "lambda": self._lambda.state_dict(),
            "omega": self._omega.state_dict(),
            "virtu": self._virtu.state_dict(),
            "step": self._step,
            "lambda_available": self._lambda_available,
            "omega_available": self._omega_available,
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        if "layer1" in sd:
            self._layer1.load_state(sd["layer1"])
        if "lambda" in sd:
            self._lambda.load_state(sd["lambda"])
        if "omega" in sd:
            self._omega.load_state(sd["omega"])
        if "virtu" in sd:
            self._virtu.load_state(sd["virtu"])
        self._step = sd.get("step", 0)
        # Back-compat: old state_dicts use "layer2_available" (single flag)
        _old = sd.get("layer2_available", None)
        self._lambda_available = sd.get("lambda_available", _old)
        self._omega_available = sd.get("omega_available", _old)
