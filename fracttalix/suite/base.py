"""Base types for the Fracttalix Sentinel Detector Suite.

This module provides the foundational abstractions for all suite detectors:

- :class:`ScopeStatus` — Enum of detector verdicts (NORMAL, ALERT,
  OUT_OF_SCOPE, WARMUP).
- :class:`DetectorResult` — Immutable output from a single detector at a
  single timestep.
- :class:`BaseDetector` — Abstract base class that all detectors implement.

Design principles
-----------------
1. Each detector is small, focused, and self-aware.
2. Each reports ``OUT_OF_SCOPE`` when data does not fit its model.
3. Detectors run in parallel; no false consensus by blending.
4. A detector that stays silent is more useful than one that guesses.

Shared math helpers (``_mean``, ``_variance``, ``_std``, ``_ac1``,
``_linear_trend``) are pure-Python, zero-dependency functions used
internally by detectors that don't require NumPy.

Example
-------
Subclass ``BaseDetector`` to create a custom detector::

    from fracttalix.suite.base import BaseDetector

    class MyDetector(BaseDetector):
        def __init__(self):
            super().__init__(name="My", warmup=50, window_size=100)

        def _check_scope(self, window):
            return len(window) >= 30 and _std(window) > 0.01

        def _compute(self, window):
            score = ...  # your scoring logic (0.0 to 1.0)
            return score, "explanation of the score"

See Also
--------
fracttalix.suite.suite : DetectorSuite runs all detectors in parallel.
"""

import dataclasses
import math
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Any, Dict, List


class ScopeStatus(Enum):
    """Verdict from a detector at a single timestep.

    Every detector result carries exactly one of these statuses:

    - ``NORMAL`` — The detector's model applies and no anomaly was found.
      The ``score`` field is meaningful (0.0 = fully normal).
    - ``ALERT`` — The detector's model applies and an anomaly was detected.
      The ``score`` field indicates severity (higher = more severe).
    - ``OUT_OF_SCOPE`` — The data does not fit this detector's model.
      The ``score`` is fixed at 0.0 and should be ignored.
    - ``WARMUP`` — The detector is still collecting baseline data.
      The ``score`` is fixed at 0.0 and should be ignored.
    """
    NORMAL = "normal"
    ALERT = "alert"
    OUT_OF_SCOPE = "out_of_scope"
    WARMUP = "warmup"


@dataclasses.dataclass(frozen=True)
class DetectorResult:
    """Immutable output from a single detector at a single timestep.

    Attributes
    ----------
    detector : str
        Human-readable detector name (e.g. ``"HopfDetector"``).
    status : ScopeStatus
        The detector's verdict for this observation.
    score : float
        Anomaly score in the range [0.0, 1.0].  Meaningful only when
        ``status`` is ``NORMAL`` or ``ALERT``.  When ``status`` is
        ``OUT_OF_SCOPE`` or ``WARMUP``, the score is always 0.0.
    message : str
        Human-readable explanation of the score (e.g. internal metric
        values, regime classification, or reason for out-of-scope).
    step : int
        Zero-based observation index at which this result was produced.

    Examples
    --------
    >>> result = detector.update(42.0)
    >>> result.is_alert
    False
    >>> result.score
    0.12
    """
    detector: str
    status: ScopeStatus
    score: float
    message: str
    step: int

    @property
    def is_alert(self) -> bool:
        """``True`` if this result is an alert (``status == ALERT``)."""
        return self.status == ScopeStatus.ALERT

    @property
    def in_scope(self) -> bool:
        """``True`` if the detector's model applies (``NORMAL`` or ``ALERT``)."""
        return self.status in (ScopeStatus.NORMAL, ScopeStatus.ALERT)


class BaseDetector(ABC):
    """Abstract base class for all suite detectors.

    Subclasses must implement two methods:

    1. :meth:`_check_scope` — Decide whether the detector's model applies
       to the current data window.  Return ``True`` if in scope.
    2. :meth:`_compute` — Given the in-scope data window, return a tuple
       ``(score, message)`` where score is in [0.0, 1.0].

    The public entry point is :meth:`update`, which appends the new value
    to the internal sliding window, runs scope and compute logic, and
    returns a :class:`DetectorResult`.

    Parameters
    ----------
    name : str
        Human-readable detector name (appears in ``DetectorResult.detector``).
    warmup : int
        Number of initial observations during which the detector reports
        ``WARMUP`` status and does not evaluate.
    window_size : int
        Maximum length of the internal sliding window.

    Attributes
    ----------
    _alert_threshold : float
        Score at or above which ``status`` flips from ``NORMAL`` to ``ALERT``.
        Subclasses may override this in ``__init__`` (default 0.5).

    Examples
    --------
    >>> from fracttalix.suite import HopfDetector
    >>> det = HopfDetector(warmup=60, window=40)
    >>> result = det.update(3.14)
    >>> result.status
    <ScopeStatus.WARMUP: 'warmup'>

    See Also
    --------
    DetectorResult : The output type returned by :meth:`update`.
    ScopeStatus : The set of possible detector verdicts.
    """

    def __init__(self, name: str, warmup: int, window_size: int):
        self._name = name
        self._warmup = warmup
        self._window_size = window_size
        self._window: deque = deque(maxlen=window_size)
        self._step = 0
        self._alert_threshold: float = 0.5

    @abstractmethod
    def _check_scope(self, window: List[float]) -> bool:
        """Decide whether the detector's model applies to the current data.

        Parameters
        ----------
        window : list of float
            Current contents of the sliding window.

        Returns
        -------
        bool
            ``True`` if the data fits this detector's model and
            :meth:`_compute` should be called; ``False`` to return
            ``OUT_OF_SCOPE``.
        """

    @abstractmethod
    def _compute(self, window: List[float]) -> tuple:
        """Compute the anomaly score for the current in-scope window.

        Called only after :meth:`_check_scope` returns ``True`` and the
        warmup period has elapsed.

        Parameters
        ----------
        window : list of float
            Current contents of the sliding window.

        Returns
        -------
        score : float
            Anomaly score in [0.0, 1.0].  Values at or above
            ``_alert_threshold`` trigger ``ALERT`` status.
        message : str
            Human-readable explanation of the score.
        """

    def update(self, value: float) -> DetectorResult:
        """Feed one observation and receive a detector verdict.

        Parameters
        ----------
        value : float
            The new scalar observation.

        Returns
        -------
        DetectorResult
            Immutable result with status, score, message, and step index.
        """
        self._window.append(float(value))
        step = self._step
        self._step += 1
        window = list(self._window)

        if step < self._warmup:
            return DetectorResult(
                detector=self._name,
                status=ScopeStatus.WARMUP,
                score=0.0,
                message=f"warmup ({step}/{self._warmup})",
                step=step,
            )

        if not self._check_scope(window):
            return DetectorResult(
                detector=self._name,
                status=ScopeStatus.OUT_OF_SCOPE,
                score=0.0,
                message="data does not fit this detector's model",
                step=step,
            )

        score, message = self._compute(window)
        score = max(0.0, min(1.0, score))
        status = ScopeStatus.ALERT if score >= self._alert_threshold else ScopeStatus.NORMAL
        return DetectorResult(
            detector=self._name,
            status=status,
            score=score,
            message=message,
            step=step,
        )

    def reset(self) -> None:
        """Reset the detector to its initial (pre-warmup) state.

        Clears the sliding window and resets the step counter to zero.
        Subclasses that maintain additional state (e.g. EWMA trackers,
        CUSUM accumulators) should call ``super().reset()`` and then
        clear their own fields.
        """
        self._window.clear()
        self._step = 0

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of the detector's state.

        Use :meth:`load_state` to restore from the returned dictionary.
        Subclasses should call ``super().state_dict()`` and then update
        the dictionary with their own fields.

        Returns
        -------
        dict
            Keys ``"window"`` (list of float) and ``"step"`` (int), plus
            any subclass-specific keys.
        """
        return {
            "window": list(self._window),
            "step": self._step,
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        """Restore detector state from a previously saved snapshot.

        Parameters
        ----------
        sd : dict
            Dictionary produced by :meth:`state_dict`.
        """
        self._step = sd.get("step", 0)
        self._window.clear()
        self._window.extend(sd.get("window", []))


# ---------------------------------------------------------------------------
# Shared math helpers (pure Python, no numpy required)
#
# These are internal utilities used by detectors that don't need NumPy.
# They operate on plain ``list[float]`` and are intentionally simple.
# ---------------------------------------------------------------------------


def _mean(xs: List[float]) -> float:
    """Arithmetic mean.  Returns 0.0 for empty input."""
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs: List[float]) -> float:
    """Population variance (biased).  Returns 0.0 if fewer than 2 values."""
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


def _std(xs: List[float]) -> float:
    """Population standard deviation (biased).  See :func:`_variance`."""
    return math.sqrt(max(_variance(xs), 0.0))


def _ac1(xs: List[float]) -> float:
    """Lag-1 autocorrelation coefficient.

    Measures how strongly each observation correlates with its immediate
    predecessor.  Values near 1.0 indicate strong positive memory
    (critical slowing down); values near 0.0 indicate white noise.

    Returns 0.0 if the denominator (total variance) is negligible or
    if fewer than 3 observations are provided.
    """
    n = len(xs)
    if n < 3:
        return 0.0
    mu = _mean(xs)
    num = sum((xs[i] - mu) * (xs[i - 1] - mu) for i in range(1, n))
    den = sum((x - mu) ** 2 for x in xs)
    return (num / den) if den > 1e-12 else 0.0


def _linear_trend(xs: List[float]) -> float:
    """Slope of the least-squares regression line, normalised by standard deviation.

    A unit-free measure of monotonic trend strength.  Positive values
    indicate upward drift; negative values indicate downward drift.
    The normalisation by standard deviation makes the result comparable
    across signals of different scales.

    Returns 0.0 if fewer than 3 observations or zero variance.
    """
    n = len(xs)
    if n < 3:
        return 0.0
    xbar = (n - 1) / 2.0
    ybar = _mean(xs)
    num = sum((i - xbar) * (xs[i] - ybar) for i in range(n))
    den = sum((i - xbar) ** 2 for i in range(n))
    if den < 1e-12:
        return 0.0
    slope = num / den
    s = _std(xs)
    return slope / (s + 1e-10)
