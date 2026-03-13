# fracttalix/suite/__init__.py
# Sentinel Detector Suite — five specialized detectors, clean slate design.
#
# Philosophy:
#   The v12.2 SentinelDetector is a 37-step monolith that always produces
#   an alert/no-alert verdict regardless of whether any of those steps apply.
#   The suite is the alternative: five small, focused, self-aware detectors
#   that each report OUT_OF_SCOPE when they don't apply, and ALERT only when
#   they are confident in their domain.
#
#   Run them in parallel.  Get a dashboard of independent opinions.
#   No false consensus.  Honest uncertainty.
#
# Detectors:
#   HopfDetector    — Pre-transition early warning (critical slowing down)
#   DiscordDetector — Point and contextual anomalies (subsequence discord)
#   DriftDetector   — Slow distribution shifts (non-adaptive CUSUM + PH)
#   VarianceDetector — Sudden volatility changes (CUSUM on z²)
#   CouplingDetector — Cross-frequency decoupling (PAC degradation)
#
# Quick start::
#
#   from fracttalix.suite import DetectorSuite
#   suite = DetectorSuite()
#   for value in stream:
#       result = suite.update(value)
#       print(result.summary())
#
# Or use individual detectors::
#
#   from fracttalix.suite import HopfDetector, DriftDetector
#   hopf = HopfDetector()
#   drift = DriftDetector()

from fracttalix.suite.base import ScopeStatus, DetectorResult, BaseDetector
from fracttalix.suite.hopf import HopfDetector
from fracttalix.suite.discord import DiscordDetector
from fracttalix.suite.drift import DriftDetector
from fracttalix.suite.variance import VarianceDetector
from fracttalix.suite.coupling import CouplingDetector
from fracttalix.suite.suite import DetectorSuite, SuiteResult

__all__ = [
    "ScopeStatus",
    "DetectorResult",
    "BaseDetector",
    "HopfDetector",
    "DiscordDetector",
    "DriftDetector",
    "VarianceDetector",
    "CouplingDetector",
    "DetectorSuite",
    "SuiteResult",
]
