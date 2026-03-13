# fracttalix/frm/__init__.py
# FRM Physics Layer — Layer 2 of the FRMSuite.
#
# This package contains detectors derived directly from FRM physics:
#   Lambda  (HopfDetector method='frm') — track λ → 0 (Hopf bifurcation)
#   Omega   (OmegaDetector)            — track ω vs π/(2·τ_gen)
#   Virtu   (VirtuDetector)            — time-to-bifurcation estimate
#
# Dependency notes:
#   Lambda (HopfDetector frm) requires scipy + numpy.
#   Omega (OmegaDetector)    requires numpy only.
#   Virtu (VirtuDetector)    requires neither (reads Lambda/Omega outputs).
# Import gracefully degrades: Lambda failure does NOT block Omega.
# FRMSuite tracks per-detector availability independently.
#
# Entry point:
#   from fracttalix.frm import FRMSuite
#   suite = FRMSuite(tau_gen=12.5)
#   result = suite.update(value)
#   if result.frm_confidence >= 2:
#       handle_frm_alert(result)

from fracttalix.frm.frm_suite import FRMSuite, FRMSuiteResult
from fracttalix.frm.omega import OmegaDetector
from fracttalix.frm.virtu import VirtuDetector
from fracttalix.suite.base import ScopeStatus, DetectorResult

__all__ = [
    "FRMSuite",
    "FRMSuiteResult",
    "OmegaDetector",
    "VirtuDetector",
    "ScopeStatus",
    "DetectorResult",
]
