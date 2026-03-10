# benchmark/archetypes.py
# Archetype generation logic ported from SentinelBenchmark in v11.
# Uses only stdlib (math, random) — no numpy required.

import math
import random
from typing import List, Literal, Tuple, Union

# Supported archetype names
BenchmarkArchetype = Literal["point", "contextual", "collective", "drift", "variance"]

ARCHETYPES: List[str] = ["point", "contextual", "collective", "drift", "variance"]


def _randn_list(n: int, seed: int) -> List[float]:
    """Generate n standard-normal samples using Box-Muller, seeded reproducibly."""
    rng = random.Random(seed)
    out: List[float] = []
    while len(out) < n:
        u1 = rng.random()
        u2 = rng.random()
        if u1 == 0.0:
            u1 = 1e-10
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
        out.append(z0)
        if len(out) < n:
            out.append(z1)
    return out[:n]


def generate(
    archetype: Union[str, BenchmarkArchetype],
    n: int = 1000,
    seed: int = 42,
) -> Tuple[List[float], List[int]]:
    """Generate (data, labels) for the given anomaly archetype.

    Parameters
    ----------
    archetype:
        One of "point", "contextual", "collective", "drift", "variance".
    n:
        Number of observations to generate. Default 1000.
    seed:
        Random seed for reproducibility. Default 42.

    Returns
    -------
    data:
        List of float observations.
    labels:
        List of int (0 or 1) indicating anomaly ground truth.
    """
    if archetype not in ARCHETYPES:
        raise ValueError(
            f"Unknown archetype {archetype!r}. Choose from: {ARCHETYPES}"
        )

    data: List[float] = _randn_list(n, seed)
    labels: List[int] = [0] * n

    if archetype == "point":
        # Sparse large spikes every ~50 steps, 20 total
        idxs = list(range(50, n, 50))[:20]
        for i in idxs:
            if i < n:
                data[i] += 8.0
                labels[i] = 1

    elif archetype == "contextual":
        # Sinusoidal seasonal signal; anomalies are contextually wrong phase dips
        period = 20
        for i in range(n):
            data[i] += 3.0 * math.sin(2.0 * math.pi * i / period)
        idxs = list(range(60, n, 100))
        for i in idxs:
            if i < n:
                data[i] = data[i] - 6.0  # contextually anomalous
                labels[i] = 1

    elif archetype == "collective":
        # Extended runs of shifted mean (two blocks)
        idxs = list(range(100, 120)) + list(range(300, 315))
        for i in idxs:
            if i < n:
                data[i] += 4.0
                labels[i] = 1

    elif archetype == "drift":
        # Slow linear mean drift in second half
        for i in range(n):
            if i > n // 2:
                data[i] += (i - n // 2) * 0.02
        for i in range(n // 2 + 20, n):
            labels[i] = 1

    elif archetype == "variance":
        # 4x variance explosion in second half
        for i in range(n // 2, n):
            data[i] *= 4.0
            labels[i] = 1

    return data, labels
