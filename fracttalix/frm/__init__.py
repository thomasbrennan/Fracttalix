# fracttalix/frm/__init__.py
# FRM Physics Layer — Layer 2 of the FRMSuite.
#
# This package contains detectors derived directly from FRM physics:
#   Lambda  (HopfDetector method='frm') — track λ → 0 (Hopf bifurcation)
#   Omega   (OmegaDetector)            — track ω vs π/(2·τ_gen)
#   Virtu   (VirtuDetector)            — time-to-bifurcation estimate
#
# All require scipy + numpy. Import gracefully degrades: if scipy is absent
# the Layer 2 detectors raise ImportError on first update(), but FRMSuite
# continues to operate with Layer 1 only (frm_confidence stays at 0).
#
# Entry point:
#   from fracttalix.frm import FRMSuite
#   suite = FRMSuite(tau_gen=12.5)
#   result = suite.update(value)
#   if result.frm_confidence >= 2:
#       handle_frm_alert(result)

from fracttalix.frm.frm_suite import FRMSuite, FRMSuiteResult
from fracttalix.suite.base import ScopeStatus, DetectorResult

__all__ = [
    "FRMSuite",
    "FRMSuiteResult",
    "ScopeStatus",
    "DetectorResult",
]
