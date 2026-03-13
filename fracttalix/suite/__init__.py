"""Fracttalix Sentinel Detector Suite — eight modular, scope-aware detectors.

Philosophy
----------
The v12.2 SentinelDetector is a 37-step monolith that always produces
an alert/no-alert verdict regardless of whether any of those steps apply.
The suite is the alternative: eight small, focused, self-aware detectors
that each report ``OUT_OF_SCOPE`` when they don't apply, and ``ALERT``
only when they are confident in their domain.

Run them in parallel.  Get a dashboard of independent opinions.
No false consensus.  Honest uncertainty.

Core Detectors (pure Python + NumPy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
====================  =============================================================
Detector              What it detects
====================  =============================================================
``HopfDetector``      Pre-transition early warning (rising variance + AC1)
``DiscordDetector``   Point and contextual anomalies (subsequence discord)
``DriftDetector``     Slow distribution shifts (frozen-baseline CUSUM)
``VarianceDetector``  Sudden volatility changes (CUSUM on z-squared residuals)
``CouplingDetector``  Cross-frequency decoupling (phase-amplitude coupling trend)
====================  =============================================================

FRM-Derived Detectors (require SciPy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These detectors leverage the Fractal Rhythm Model's physics — absolute
frequency references and parametric decay tracking that no other detection
system provides.

====================  =============================================================
Detector              What it detects
====================  =============================================================
``LambdaDetector``    Bifurcation proximity via parametric lambda tracking
``OmegaDetector``     Timescale integrity via omega = pi / (2 * tau_gen)
``VirtuDetector``     Decision rationality via Kramers scaling
====================  =============================================================

Quick Start
-----------
Run all five core detectors as a dashboard::

    from fracttalix.suite import DetectorSuite

    suite = DetectorSuite()
    for value in stream:
        result = suite.update(value)
        print(result.summary())

Use individual detectors for targeted analysis::

    from fracttalix.suite import HopfDetector, LambdaDetector

    hopf = HopfDetector(warmup=60, window=40)
    lam = LambdaDetector(tau_gen=20.0, fit_window=128)

    for value in stream:
        hopf_result = hopf.update(value)
        lam_result = lam.update(value)

Combine FRM detectors for full physics-aware monitoring::

    from fracttalix.suite import LambdaDetector, OmegaDetector, VirtuDetector

    lam = LambdaDetector(tau_gen=20.0)
    omega = OmegaDetector(tau_gen=20.0)
    virtu = VirtuDetector(lambda_detector=lam)

    for value in stream:
        lam_result = lam.update(value)
        omega_result = omega.update(value)
        virtu_result = virtu.update(value)

        if virtu_result.is_alert:
            print(f"Decision window closing: {virtu_result.message}")

See Also
--------
fracttalix.suite.base : Base types (ScopeStatus, DetectorResult, BaseDetector).
fracttalix.suite.suite : DetectorSuite for running core detectors as a group.
"""

from fracttalix.suite.base import ScopeStatus, DetectorResult, BaseDetector
from fracttalix.suite.hopf import HopfDetector
from fracttalix.suite.discord import DiscordDetector
from fracttalix.suite.drift import DriftDetector
from fracttalix.suite.variance import VarianceDetector
from fracttalix.suite.coupling import CouplingDetector
from fracttalix.suite.suite import DetectorSuite, SuiteResult
from fracttalix.suite.lambda_detector import LambdaDetector
from fracttalix.suite.omega_detector import OmegaDetector
from fracttalix.suite.virtu_detector import VirtuDetector

__all__ = [
    "ScopeStatus",
    "DetectorResult",
    "BaseDetector",
    "HopfDetector",
    "DiscordDetector",
    "DriftDetector",
    "VarianceDetector",
    "CouplingDetector",
    "LambdaDetector",
    "OmegaDetector",
    "VirtuDetector",
    "DetectorSuite",
    "SuiteResult",
]
