# fracttalix/steps/base.py
# DetectorStep abstract base class and RegimeBoostState shared object

import abc
from typing import Any, Dict

from fracttalix.window import StepContext


class RegimeBoostState:
    """Shared mutable object passed at construction time to CoreEWMAStep and RegimeStep.

    Eliminates the fragile _core_step_ref scratch-key side channel used in v10.0.
    RegimeStep writes boost; CoreEWMAStep reads it.  Persisted across save/load
    via each step's own state_dict.
    """

    __slots__ = ("boost",)

    def __init__(self):
        self.boost: float = 1.0


class DetectorStep(abc.ABC):
    """Abstract base for pipeline steps.

    Each subclass implements a single well-scoped responsibility.  The pipeline
    calls ``update(ctx)`` in order; steps modify ``ctx.scratch`` in-place.
    """

    @abc.abstractmethod
    def update(self, ctx: StepContext) -> None:
        """Process one observation.  Modifies ctx.scratch in-place."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state to factory defaults."""

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable snapshot of internal state."""

    @abc.abstractmethod
    def load_state(self, sd: Dict[str, Any]) -> None:
        """Restore internal state from a snapshot."""
