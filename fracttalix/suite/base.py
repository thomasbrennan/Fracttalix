# fracttalix/suite/base.py
# Base types for the Sentinel Detector Suite.
#
# Design principles:
#   - Each detector is small, focused, and self-aware.
#   - Each reports OUT_OF_SCOPE when data does not fit its model.
#   - Detectors run in parallel; no false consensus by blending.
#   - A detector that stays silent is more useful than one that guesses.

import dataclasses
import math
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional


class ScopeStatus(Enum):
    """What the detector is telling you."""
    NORMAL = "normal"           # In scope, no anomaly detected
    ALERT = "alert"             # In scope, anomaly detected
    OUT_OF_SCOPE = "out_of_scope"   # Data does not fit this detector's model
    WARMUP = "warmup"           # Still collecting baseline data


@dataclasses.dataclass(frozen=True)
class DetectorResult:
    """Single detector output for one observation."""
    detector: str
    status: ScopeStatus
    score: float        # 0.0–1.0; meaningful only when status is NORMAL or ALERT
    message: str
    step: int

    @property
    def is_alert(self) -> bool:
        return self.status == ScopeStatus.ALERT

    @property
    def in_scope(self) -> bool:
        return self.status in (ScopeStatus.NORMAL, ScopeStatus.ALERT)


class BaseDetector(ABC):
    """Abstract base for suite detectors.

    Each subclass must:
      1. Implement _check_scope(window) -> bool to decide if data fits.
      2. Implement _compute(window) -> (score, message) when in scope.
      3. Call super().__init__(name, warmup, window_size) in __init__.

    update(value) returns a DetectorResult every call.
    """

    def __init__(self, name: str, warmup: int, window_size: int):
        self._name = name
        self._warmup = warmup
        self._window_size = window_size
        self._window: deque = deque(maxlen=window_size)
        self._step = 0
        self._alert_threshold: float = 0.5   # subclasses may override

    @abstractmethod
    def _check_scope(self, window: List[float]) -> bool:
        """Return True if this detector applies to this data."""

    @abstractmethod
    def _compute(self, window: List[float]) -> tuple:
        """Return (score: float, message: str). Called only when in scope and post-warmup."""

    def update(self, value: float) -> DetectorResult:
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
        self._window.clear()
        self._step = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "window": list(self._window),
            "step": self._step,
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._step = sd.get("step", 0)
        self._window.clear()
        self._window.extend(sd.get("window", []))


# ---------------------------------------------------------------------------
# Shared math helpers (pure Python, no numpy required)
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


def _std(xs: List[float]) -> float:
    return math.sqrt(max(_variance(xs), 0.0))


def _ac1(xs: List[float]) -> float:
    """Lag-1 autocorrelation."""
    n = len(xs)
    if n < 3:
        return 0.0
    mu = _mean(xs)
    num = sum((xs[i] - mu) * (xs[i - 1] - mu) for i in range(1, n))
    den = sum((x - mu) ** 2 for x in xs)
    return (num / den) if den > 1e-12 else 0.0


def _linear_trend(xs: List[float]) -> float:
    """Slope of least-squares line, normalized by std."""
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
