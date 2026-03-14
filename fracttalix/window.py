# fracttalix/window.py
# WindowBank and StepContext classes

import dataclasses
from collections import deque
from typing import Any, Dict

from fracttalix.config import SentinelConfig


class WindowBank:
    """Named collection of independent deques.

    Each consumer registers its own slot with its own maxlen; the bank's
    ``append`` method fans out each new value to *all* registered deques.
    Slots are independent — one consumer's window size does not affect others.

    T0-01 fix: EWSStep registers ``"ews_w"`` independently of ``"scalar"``
    so it never receives stale scalar_window data.
    """

    def __init__(self):
        self._windows: Dict[str, deque] = {}

    def register(self, name: str, maxlen: int) -> None:
        """Create a named window.  No-op if already registered."""
        if name not in self._windows:
            self._windows[name] = deque(maxlen=maxlen)

    def append(self, value: float) -> None:
        """Fan out *value* to every registered window."""
        for d in self._windows.values():
            d.append(value)

    def get(self, name: str) -> deque:
        """Return the named deque (raises KeyError if not registered)."""
        return self._windows[name]

    def reset(self) -> None:
        """Clear all windows, preserving registrations."""
        for d in self._windows.values():
            d.clear()

    def state_dict(self) -> Dict[str, list]:
        return {k: list(v) for k, v in self._windows.items()}

    def load_state(self, sd: Dict[str, list]) -> None:
        for k, vals in sd.items():
            if k in self._windows:
                self._windows[k].clear()
                self._windows[k].extend(vals)
            elif vals:
                # Window was saved but not yet registered (lazy registration).
                # Re-create with the saved maxlen so data survives round-trip.
                self._windows[k] = deque(vals, maxlen=len(vals))

    def state_dict_with_maxlen(self) -> Dict[str, dict]:
        """Return state including maxlen for each window (used by save_state)."""
        return {
            k: {"data": list(v), "maxlen": v.maxlen}
            for k, v in self._windows.items()
        }

    def load_state_with_maxlen(self, sd: Dict[str, dict]) -> None:
        """Restore state including maxlen (used by load_state)."""
        for k, info in sd.items():
            maxlen = info.get("maxlen", len(info.get("data", [])))
            data = info.get("data", [])
            if k in self._windows:
                self._windows[k].clear()
                self._windows[k].extend(data)
            else:
                self._windows[k] = deque(data, maxlen=maxlen or None)


@dataclasses.dataclass
class StepContext:
    """Mutable scratchpad passed through the pipeline on each observation.

    Steps read previous results from ``scratch`` and write their own outputs
    into it.  ``value`` is the raw input; ``step`` is the observation counter.
    """

    value: Any
    """Raw input value (scalar float or list/array for multivariate)."""

    step: int
    """Monotonically increasing observation counter (0-based)."""

    config: SentinelConfig
    """Immutable detector configuration."""

    bank: WindowBank
    """Shared window bank."""

    scratch: Dict[str, Any] = dataclasses.field(default_factory=dict)
    """Shared mutable scratchpad — steps read/write intermediate results."""

    # ------------------------------------------------------------------
    # Convenience accessors (populated by CoreEWMAStep)
    # ------------------------------------------------------------------

    @property
    def current(self) -> float:
        """Scalar current value (last element if multivariate)."""
        v = self.value
        if isinstance(v, (list, tuple)):
            return float(v[-1])
        try:
            if hasattr(v, '__len__'):
                return float(v[-1])
        except Exception:
            pass
        return float(v)

    @property
    def ewma(self) -> float:
        return self.scratch.get("ewma", 0.0)

    @property
    def dev_ewma(self) -> float:
        return self.scratch.get("dev_ewma", 1.0)

    @property
    def baseline_mean(self) -> float:
        return self.scratch.get("baseline_mean", 0.0)

    @property
    def baseline_std(self) -> float:
        return self.scratch.get("baseline_std", 1.0)

    @property
    def is_warmup(self) -> bool:
        return self.step < self.config.warmup_periods
