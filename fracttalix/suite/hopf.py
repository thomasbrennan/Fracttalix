"""HopfDetector -- Pre-transition early warning via critical slowing down.

Theorem basis (P1 / EWS)
------------------------
Near a Hopf bifurcation (or any fold/transcritical transition), the system
loses its ability to recover from perturbations.  This manifests as:

1. Rising variance (the noise is amplified more).
2. Rising lag-1 autocorrelation (recovery takes longer, memory builds).

Both signals rise *before* the transition, not at it -- that is the early
warning.

OUT_OF_SCOPE conditions
-----------------------
- White noise: AC(1) is persistently near zero; the slowing-down signal is
  indistinguishable from baseline -- report nothing.
- Sustained mean shift: the signal has already jumped; that is a regime
  change, not an approach to one.  DiscordDetector / DriftDetector see it.
- Variance already high at warmup end: no pre-transition baseline to
  compare against -- insufficient scope.

Strengths and limitations
-------------------------
Best at
    Oscillatory or autocorrelated signals approaching a qualitative state
    change (oscillation onset, equilibrium loss, phase transition).
Mediocre at
    Step functions (reports OUT_OF_SCOPE after the jump).
Useless at
    Pure white noise (always OUT_OF_SCOPE by design).
"""

from typing import Any, Dict, List, Tuple

from fracttalix.suite.base import (
    BaseDetector,
    _ac1,
    _mean,
    _std,
    _variance,
)


class HopfDetector(BaseDetector):
    """Detect pre-transition early warning via critical slowing down.

    Parameters
    ----------
    warmup : int
        Observations before any verdict (default 60).  Must be long enough
        to establish a stable variance+AC(1) baseline.
    window : int
        Rolling window for current variance/AC(1) computation (default 40).
    ews_threshold : float
        Score threshold above which status = ALERT (default 0.55).
    ac1_min : float
        Minimum baseline AC(1) for the signal to be in scope.  Signals with
        AC(1) < ac1_min are treated as white noise → OUT_OF_SCOPE (default 0.1).
    mean_shift_z : float
        If |current_mean − warmup_mean| > mean_shift_z * warmup_std the signal
        has already jumped; report OUT_OF_SCOPE (default 3.5).
    """

    def __init__(
        self,
        warmup: int = 60,
        window: int = 40,
        ews_threshold: float = 0.55,
        ac1_min: float = 0.10,
        mean_shift_z: float = 3.5,
    ):
        super().__init__("HopfDetector", warmup=warmup, window_size=max(window, warmup))
        self._ews_window = window
        self._ews_threshold = ews_threshold
        self._ac1_min = ac1_min
        self._mean_shift_z = mean_shift_z
        self._alert_threshold = ews_threshold

        # Warmup baseline (frozen after warmup)
        self._warmup_var: float = 0.0
        self._warmup_ac1: float = 0.0
        self._warmup_mean: float = 0.0
        self._warmup_std: float = 1.0
        self._baseline_set: bool = False

        # EWMA trackers for trend detection
        self._var_ewma: float = 0.0
        self._ac1_ewma: float = 0.0

    def _set_baseline(self, window: List[float]) -> None:
        """Freeze the warmup baseline for variance, AC(1), mean, and std.

        Called once at end of warmup.  All subsequent scores are measured
        relative to these frozen values so that slow changes are detectable.

        Parameters
        ----------
        window : list of float
            The warmup observations to compute the baseline from.
        """
        self._warmup_mean = _mean(window)
        self._warmup_std = max(_std(window), 1e-10)
        self._warmup_var = _variance(window)
        self._warmup_ac1 = _ac1(window)
        self._var_ewma = self._warmup_var
        self._ac1_ewma = self._warmup_ac1
        self._baseline_set = True

    def _check_scope(self, window: List[float]) -> bool:
        """Determine whether the signal is suitable for Hopf detection.

        Scope gates
        -----------
        1. Baseline AC(1) < ``ac1_min`` -- signal is white noise; slowing-down
           is indistinguishable from baseline.
        2. Current mean has shifted > ``mean_shift_z`` sigma from warmup mean --
           the transition already happened; this is DriftDetector's domain.

        Parameters
        ----------
        window : list of float
            Current rolling window of observations.

        Returns
        -------
        bool
            True if the signal is in scope for Hopf analysis.
        """
        if not self._baseline_set:
            self._set_baseline(window[-self._ews_window:])

        recent = window[-self._ews_window:]

        # Scope gate 1: baseline AC(1) too low → white noise
        if self._warmup_ac1 < self._ac1_min:
            return False

        # Scope gate 2: current mean has shifted beyond mean_shift_z
        cur_mean = _mean(recent)
        z_shift = abs(cur_mean - self._warmup_mean) / self._warmup_std
        if z_shift > self._mean_shift_z:
            return False

        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        """Compute the early-warning score via EWMA blending of variance ratio and AC(1) delta.

        Algorithm:
        1. Compute current variance and AC(1) over the EWS window.
        2. Update EWMA trackers for both statistics.
        3. ``var_score``: maps variance ratio (current / warmup) from [1, 4] to [0, 1].
        4. ``ac1_score``: maps AC(1) delta (current - warmup) from [0, 0.3] to [0, 1].
        5. Final EWS score = 0.5 * var_score + 0.5 * ac1_score.

        Parameters
        ----------
        window : list of float
            Current rolling window of observations.

        Returns
        -------
        score : float
            EWS score in [0, 1].
        msg : str
            Diagnostic string with component values and regime label.
        """
        recent = window[-self._ews_window:]
        alpha = 0.15

        cur_var = _variance(recent)
        cur_ac1 = _ac1(recent)

        self._var_ewma = alpha * cur_var + (1 - alpha) * self._var_ewma
        self._ac1_ewma = alpha * cur_ac1 + (1 - alpha) * self._ac1_ewma

        # Rising variance: how much has variance grown relative to warmup?
        var_ratio = cur_var / (self._warmup_var + 1e-10)
        # Map: 1.0 → 0.0 (no change), 4.0 → 1.0 (4× the baseline variance)
        var_score = min(1.0, max(0.0, (var_ratio - 1.0) / 3.0))

        # Rising AC(1): how much has it grown relative to warmup?
        ac1_delta = cur_ac1 - self._warmup_ac1
        # +0.3 delta → score = 1.0; negative → score = 0.0
        ac1_score = min(1.0, max(0.0, ac1_delta / 0.30))

        # EWS score: both must be rising for a true Hopf precursor
        ews_score = 0.5 * var_score + 0.5 * ac1_score

        if ews_score >= self._ews_threshold * 1.4:
            regime = "critical"
        elif ews_score >= self._ews_threshold:
            regime = "approaching"
        else:
            regime = "stable"

        msg = (
            f"ews={ews_score:.3f} var_ratio={var_ratio:.2f} "
            f"ac1={cur_ac1:.3f}(base={self._warmup_ac1:.3f}) regime={regime}"
        )
        return ews_score, msg

    def reset(self) -> None:
        """Clear all state, including the frozen warmup baseline and EWMA trackers."""
        super().reset()
        self._baseline_set = False
        self._warmup_var = 0.0
        self._warmup_ac1 = 0.0
        self._warmup_mean = 0.0
        self._warmup_std = 1.0
        self._var_ewma = 0.0
        self._ac1_ewma = 0.0

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all detector state.

        Returns
        -------
        dict
            Contains baseline values, EWMA trackers, and base-class state.
        """
        sd = super().state_dict()
        sd.update({
            "baseline_set": self._baseline_set,
            "warmup_var": self._warmup_var,
            "warmup_ac1": self._warmup_ac1,
            "warmup_mean": self._warmup_mean,
            "warmup_std": self._warmup_std,
            "var_ewma": self._var_ewma,
            "ac1_ewma": self._ac1_ewma,
        })
        return sd

    def load_state(self, sd: Dict[str, Any]) -> None:
        """Restore detector state from a snapshot produced by ``state_dict``.

        Parameters
        ----------
        sd : dict
            Snapshot dictionary.
        """
        super().load_state(sd)
        self._baseline_set = sd.get("baseline_set", False)
        self._warmup_var = sd.get("warmup_var", 0.0)
        self._warmup_ac1 = sd.get("warmup_ac1", 0.0)
        self._warmup_mean = sd.get("warmup_mean", 0.0)
        self._warmup_std = sd.get("warmup_std", 1.0)
        self._var_ewma = sd.get("var_ewma", 0.0)
        self._ac1_ewma = sd.get("ac1_ewma", 0.0)
