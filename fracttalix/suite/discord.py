"""DiscordDetector -- Point anomaly detection via subsequence discord.

Theorem basis (P2 / DAMP-inspired)
-----------------------------------
A "discord" is a subsequence of a time series that is maximally dissimilar
to all other subsequences.  The DAMP algorithm (2022 VLDB) dominates
point-anomaly benchmarks with a dead-simple approach: for each new
subsequence, find its nearest neighbour in the history and report the
distance.  If that distance >> the distribution of past NN-distances, it
is a discord (point anomaly or sudden contextual anomaly).

We use a lightweight online variant:

1. Maintain a rolling buffer of recent subsequences (stride 1).
2. For each new subsequence, compute its z-normalised distance to a random
   sample of historical subsequences.
3. Track the EWMA distribution of those NN distances.
4. Score = distance / (ewma_mean + k * ewma_std).

OUT_OF_SCOPE conditions
-----------------------
- Drifting data: if the signal has a sustained mean trend, subsequence
  distances systematically grow, causing false positives.  DriftDetector
  owns that.
- Rising variance baseline: VarianceDetector's domain.
- Insufficient history (< 2 * subseq_len subsequences stored).

Strengths and limitations
-------------------------
Best at
    Sharp point anomalies, sudden contextual anomalies.
Mediocre at
    Gradual drift (distances accumulate slowly, hard to threshold).
Useless at
    Slow variance changes (always OUT_OF_SCOPE by design).
"""

import math
import random
from collections import deque
from typing import Any, Dict, List, Tuple

from fracttalix.suite.base import (
    BaseDetector, ScopeStatus,
    _mean, _variance, _std, _linear_trend,
)


def _z_normalise(xs: List[float]) -> List[float]:
    """Z-normalise a sequence to zero mean and unit variance.

    Returns a zero vector if the standard deviation is near-zero (constant
    subsequence), preventing division-by-zero in distance calculations.

    Parameters
    ----------
    xs : list of float
        Raw subsequence values.

    Returns
    -------
    list of float
        Z-normalised values, or all zeros if std < 1e-10.
    """
    mu = _mean(xs)
    s = _std(xs)
    if s < 1e-10:
        return [0.0] * len(xs)
    return [(x - mu) / s for x in xs]


def _euclidean(a: List[float], b: List[float]) -> float:
    """Compute the Euclidean distance between two equal-length sequences.

    Parameters
    ----------
    a, b : list of float
        Input sequences (must have the same length).

    Returns
    -------
    float
        L2 distance between *a* and *b*.
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class DiscordDetector(BaseDetector):
    """Point anomaly detector via subsequence discord (DAMP-inspired).

    Parameters
    ----------
    warmup : int
        Observations before any verdict (default 80).
    window : int
        Total rolling window to keep (default 200).
    subseq_len : int
        Length of each subsequence (default 20).
    n_candidates : int
        Number of random historical subsequences to compare against (default 30).
        Trade-off: more = more accurate, slower.
    discord_threshold : float
        Alert threshold for the discord score (default 0.60).
    trend_threshold : float
        If |linear_trend| > this, data is drifting → OUT_OF_SCOPE (default 0.08).
    var_growth_threshold : float
        If current variance > var_growth_threshold × warmup_variance the data
        has experienced a variance shift → OUT_OF_SCOPE (default 4.0).
    """

    def __init__(
        self,
        warmup: int = 80,
        window: int = 200,
        subseq_len: int = 20,
        n_candidates: int = 30,
        discord_threshold: float = 0.60,
        trend_threshold: float = 0.08,
        var_growth_threshold: float = 4.0,
    ):
        super().__init__("DiscordDetector", warmup=warmup, window_size=window)
        self._subseq_len = subseq_len
        self._n_candidates = n_candidates
        self._discord_threshold = discord_threshold
        self._trend_threshold = trend_threshold
        self._var_growth_threshold = var_growth_threshold
        self._alert_threshold = discord_threshold

        # EWMA of NN distances (distribution tracker)
        self._dist_ewma: float = 0.0
        self._dist_var_ewma: float = 0.0
        self._dist_n: int = 0

        # Warmup baseline
        self._warmup_var: float = 1.0
        self._baseline_set: bool = False

    def _set_baseline(self, window: List[float]) -> None:
        """Freeze the warmup variance baseline.

        Used later to detect whether variance has grown substantially
        (indicating a regime shift that belongs to VarianceDetector).

        Parameters
        ----------
        window : list of float
            The warmup observations.
        """
        self._warmup_var = max(_variance(window), 1e-6)
        self._baseline_set = True

    def _check_scope(self, window: List[float]) -> bool:
        """Determine whether the signal is suitable for discord detection.

        Scope gates
        -----------
        1. Insufficient non-trivial history (< 2 * subseq_len + 10 points).
        2. Persistent linear trend exceeds ``trend_threshold`` -- drifting
           data inflates subsequence distances; DriftDetector owns that.
        3. Current variance > ``var_growth_threshold`` * warmup variance --
           variance regime shift; VarianceDetector owns that.

        Parameters
        ----------
        window : list of float
            Current rolling window.

        Returns
        -------
        bool
            True if the signal is in scope for discord analysis.
        """
        if not self._baseline_set:
            self._set_baseline(window)

        if len(window) < 2 * self._subseq_len + 10:
            return False

        # Scope gate: persistent linear trend → DriftDetector's domain
        trend = abs(_linear_trend(window))
        if trend > self._trend_threshold:
            return False

        # Scope gate: variance has grown substantially → VarianceDetector's domain
        cur_var = _variance(window[-self._subseq_len:])
        if cur_var > self._var_growth_threshold * self._warmup_var:
            return False

        return True

    def _compute(self, window: List[float]) -> Tuple[float, str]:
        """Compute the discord score for the current subsequence.

        Algorithm:
        1. Extract the most recent ``subseq_len`` values and z-normalise.
        2. Sample up to ``n_candidates`` historical subsequences (excluding
           the last 2L to avoid trivial matches) and z-normalise each.
        3. Compute the nearest-neighbour (NN) Euclidean distance.
        4. Update the EWMA distribution of NN distances.
        5. Score = z-score of NN distance / 3 (clamped to [0, 1]).  A 3-sigma
           discord maps to score 1.0.

        Parameters
        ----------
        window : list of float
            Current rolling window.

        Returns
        -------
        score : float
            Discord score in [0, 1].
        msg : str
            Diagnostic string with NN distance, EWMA, and z-score.
        """
        L = self._subseq_len
        # Current subsequence = the most recent L values, z-normalised
        cur_subseq = _z_normalise(window[-L:])

        # Random sample from history (excluding the last 2L to avoid trivial matches)
        safe_end = max(0, len(window) - 2 * L)
        if safe_end < L:
            return 0.0, "insufficient non-trivial history"

        if safe_end > self._n_candidates:
            starts = random.sample(range(safe_end), self._n_candidates)
        else:
            starts = list(range(safe_end))

        if not starts:
            return 0.0, "no candidates"

        dists = []
        for s in starts:
            subseq = _z_normalise(window[s:s + L])
            dists.append(_euclidean(cur_subseq, subseq))

        nn_dist = min(dists)

        # Update distribution EWMA
        alpha = 0.1
        if self._dist_n == 0:
            self._dist_ewma = nn_dist
            self._dist_var_ewma = 0.0
        else:
            old_mean = self._dist_ewma
            self._dist_ewma = alpha * nn_dist + (1 - alpha) * self._dist_ewma
            self._dist_var_ewma = alpha * (nn_dist - old_mean) ** 2 + (1 - alpha) * self._dist_var_ewma
        self._dist_n += 1

        # Score: how many standard deviations above the historical NN mean?
        dist_std = math.sqrt(max(self._dist_var_ewma, 1e-10))
        z_dist = (nn_dist - self._dist_ewma) / (dist_std + 1e-10)
        # Map: z=3 → score≈1.0 (3-sigma discord)
        score = min(1.0, max(0.0, z_dist / 3.0))

        msg = (
            f"nn_dist={nn_dist:.3f} ewma={self._dist_ewma:.3f} "
            f"z={z_dist:.2f} score={score:.3f}"
        )
        return score, msg

    def reset(self) -> None:
        """Clear all state, including NN-distance EWMA and warmup baseline."""
        super().reset()
        self._dist_ewma = 0.0
        self._dist_var_ewma = 0.0
        self._dist_n = 0
        self._warmup_var = 1.0
        self._baseline_set = False

    def state_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot of all detector state.

        Returns
        -------
        dict
            Contains EWMA trackers, warmup baseline, and base-class state.
        """
        sd = super().state_dict()
        sd.update({
            "dist_ewma": self._dist_ewma,
            "dist_var_ewma": self._dist_var_ewma,
            "dist_n": self._dist_n,
            "warmup_var": self._warmup_var,
            "baseline_set": self._baseline_set,
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
        self._dist_ewma = sd.get("dist_ewma", 0.0)
        self._dist_var_ewma = sd.get("dist_var_ewma", 0.0)
        self._dist_n = sd.get("dist_n", 0)
        self._warmup_var = sd.get("warmup_var", 1.0)
        self._baseline_set = sd.get("baseline_set", False)
