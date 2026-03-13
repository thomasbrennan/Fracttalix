"""DetectorSuite -- runs all 5 detectors in parallel.

The suite is not a consensus machine.  It is a dashboard.  Each detector
gives an independent opinion.  The user decides which opinions matter for
their domain.

Architecture
------------
- Detectors run in parallel (all see the same value).
- No blending, no averaging, no consensus gate.
- Each detector can be OUT_OF_SCOPE when data does not fit its model.
- ``SuiteResult`` contains all 5 individual results plus a summary.

Recommended combos
------------------
Power grid monitoring
    HopfDetector + VarianceDetector
API latency monitoring
    DiscordDetector + DriftDetector
Neural / physiological
    HopfDetector + CouplingDetector
"I don't know what I'm looking for"
    Use SentinelDetector (v12.2 pipeline).
"""

import dataclasses
from typing import Any, Dict, Iterator, List, Optional

from fracttalix.suite.base import DetectorResult, ScopeStatus
from fracttalix.suite.coupling import CouplingDetector
from fracttalix.suite.discord import DiscordDetector
from fracttalix.suite.drift import DriftDetector
from fracttalix.suite.hopf import HopfDetector
from fracttalix.suite.variance import VarianceDetector


@dataclasses.dataclass(frozen=True)
class SuiteResult:
    """One observation's worth of output from all 5 detectors.

    Access individual results via attribute or iteration::

        r = suite.update(value)
        if r.hopf.is_alert:
            ...
        for det_result in r:
            print(det_result.detector, det_result.status)

    """
    hopf: DetectorResult
    discord: DetectorResult
    drift: DetectorResult
    variance: DetectorResult
    coupling: DetectorResult

    def __iter__(self) -> Iterator[DetectorResult]:
        return iter([self.hopf, self.discord, self.drift, self.variance, self.coupling])

    @property
    def alerts(self) -> List[DetectorResult]:
        """Return the list of detectors currently firing (status == ALERT).

        Returns
        -------
        list of DetectorResult
            Only results whose status is ``ScopeStatus.ALERT``.
        """
        return [r for r in self if r.is_alert]

    @property
    def in_scope(self) -> List[DetectorResult]:
        """Return detectors whose model applies to the current data.

        Includes results with status NORMAL or ALERT (i.e. not WARMUP and not
        OUT_OF_SCOPE).

        Returns
        -------
        list of DetectorResult
            Results where ``in_scope`` is True.
        """
        return [r for r in self if r.in_scope]

    @property
    def out_of_scope(self) -> List[DetectorResult]:
        """Return detectors that have declared this data outside their domain.

        Returns
        -------
        list of DetectorResult
            Results whose status is ``ScopeStatus.OUT_OF_SCOPE``.
        """
        return [r for r in self if r.status == ScopeStatus.OUT_OF_SCOPE]

    @property
    def any_alert(self) -> bool:
        """Check whether at least one detector is in ALERT status.

        Returns
        -------
        bool
            True if any detector is currently alerting.
        """
        return len(self.alerts) > 0

    def summary(self) -> str:
        """Build a one-line dashboard string showing all 5 detector statuses.

        Format: ``Hopf:ok(0.12) | Disc:OOS | Drif:ALERT(0.87) | ...``

        Returns
        -------
        str
            Pipe-delimited status string with 4-char detector abbreviations.
        """
        parts = []
        for r in self:
            if r.status == ScopeStatus.WARMUP:
                parts.append(f"{r.detector[:4]}:WARM")
            elif r.status == ScopeStatus.OUT_OF_SCOPE:
                parts.append(f"{r.detector[:4]}:OOS")
            elif r.status == ScopeStatus.ALERT:
                parts.append(f"{r.detector[:4]}:ALERT({r.score:.2f})")
            else:
                parts.append(f"{r.detector[:4]}:ok({r.score:.2f})")
        return " | ".join(parts)


class DetectorSuite:
    """Run all 5 specialized detectors in parallel on a single stream.

    Each detector is independent.  Use the subset relevant to your domain.

    Usage::

        suite = DetectorSuite()
        for value in stream:
            result = suite.update(value)
            print(result.summary())
            if result.drift.is_alert:
                handle_drift(result.drift)

    Parameters
    ----------
    hopf_kwargs : dict, optional
        Kwargs forwarded to HopfDetector.
    discord_kwargs : dict, optional
        Kwargs forwarded to DiscordDetector.
    drift_kwargs : dict, optional
        Kwargs forwarded to DriftDetector.
    variance_kwargs : dict, optional
        Kwargs forwarded to VarianceDetector.
    coupling_kwargs : dict, optional
        Kwargs forwarded to CouplingDetector.
    """

    def __init__(
        self,
        hopf_kwargs: Optional[Dict[str, Any]] = None,
        discord_kwargs: Optional[Dict[str, Any]] = None,
        drift_kwargs: Optional[Dict[str, Any]] = None,
        variance_kwargs: Optional[Dict[str, Any]] = None,
        coupling_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.hopf = HopfDetector(**(hopf_kwargs or {}))
        self.discord = DiscordDetector(**(discord_kwargs or {}))
        self.drift = DriftDetector(**(drift_kwargs or {}))
        self.variance = VarianceDetector(**(variance_kwargs or {}))
        self.coupling = CouplingDetector(**(coupling_kwargs or {}))
        self._detectors = [
            self.hopf, self.discord, self.drift, self.variance, self.coupling
        ]

    def update(self, value: float) -> SuiteResult:
        """Feed one observation to all detectors. Returns SuiteResult."""
        results = [det.update(value) for det in self._detectors]
        return SuiteResult(
            hopf=results[0],
            discord=results[1],
            drift=results[2],
            variance=results[3],
            coupling=results[4],
        )

    def reset(self) -> None:
        """Reset all detectors to factory state."""
        for det in self._detectors:
            det.reset()

    def state_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot of all detector states."""
        return {
            "hopf": self.hopf.state_dict(),
            "discord": self.discord.state_dict(),
            "drift": self.drift.state_dict(),
            "variance": self.variance.state_dict(),
            "coupling": self.coupling.state_dict(),
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        """Restore all detector states from a snapshot."""
        if "hopf" in sd:
            self.hopf.load_state(sd["hopf"])
        if "discord" in sd:
            self.discord.load_state(sd["discord"])
        if "drift" in sd:
            self.drift.load_state(sd["drift"])
        if "variance" in sd:
            self.variance.load_state(sd["variance"])
        if "coupling" in sd:
            self.coupling.load_state(sd["coupling"])
