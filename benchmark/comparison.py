# benchmark/comparison.py
# Baseline comparison for hostile-review compliance.
#
# Compares Fracttalix Sentinel against:
#   1. Naive 3-sigma (EWMA ± 3σ, no pipeline)
#   2. PyOD ECOD (if pyod installed)
#   3. River HalfSpaceTrees (if river installed)
#
# If PyOD/River are not installed, those baselines are gracefully skipped
# with a note.

import math
from typing import Any, Dict, List, Optional, Tuple

from benchmark.archetypes import generate
from benchmark.metrics import _pr_auc, _vus_pr, _compute_f1_with_tolerance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _ewma(data: List[float], alpha: float = 0.1) -> List[float]:
    """Exponentially weighted moving average."""
    if not data:
        return []
    out = [data[0]]
    for x in data[1:]:
        out.append(alpha * x + (1.0 - alpha) * out[-1])
    return out


def _naive_3sigma(
    data: List[float],
    labels: List[int],
) -> Dict[str, float]:
    """Simple EWMA baseline: alert when |z| > 3, no multi-step pipeline.

    Uses global mean and std — intentionally naïve to contrast with Sentinel.
    """
    mu = _mean(data)
    sq = [(x - mu) ** 2 for x in data]
    std = math.sqrt(_mean(sq)) or 1.0

    scores = [abs(x - mu) / std for x in data]
    preds = [s > 3.0 for s in scores]

    _prec, _rec, f1 = _compute_f1_with_tolerance(preds, labels, tolerance=5)
    auprc = _pr_auc(scores, labels, tolerance=5)
    vus = _vus_pr(scores, labels)
    return {"f1": f1, "auprc": auprc, "vus_pr": vus}


def _pyod_ecod(
    data: List[float],
    labels: List[int],
    batch_size: int = 100,
) -> Optional[Dict[str, float]]:
    """Run PyOD ECOD in sliding window batches.

    Returns None if pyod is not installed.
    """
    try:
        from pyod.models.ecod import ECOD  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None

    n = len(data)
    scores = [0.0] * n

    # Sliding window — need at least batch_size points
    for end in range(batch_size, n + 1, batch_size // 2):
        start = max(0, end - batch_size)
        window = np.array(data[start:end]).reshape(-1, 1)
        clf = ECOD()
        clf.fit(window)
        window_scores = clf.decision_scores_.tolist()
        # Assign score to last point in window
        idx = end - 1
        if idx < n:
            scores[idx] = float(window_scores[-1])

    # Fill remaining with 0
    threshold = sorted(scores, reverse=True)[max(1, n // 20)]  # top-5th percentile
    preds = [s >= threshold for s in scores]

    _prec, _rec, f1 = _compute_f1_with_tolerance(preds, labels, tolerance=5)
    auprc = _pr_auc(scores, labels, tolerance=5)
    vus = _vus_pr(scores, labels)
    return {"f1": f1, "auprc": auprc, "vus_pr": vus}


def _river_hst(
    data: List[float],
    labels: List[int],
) -> Optional[Dict[str, float]]:
    """Run River HalfSpaceTrees streaming detector.

    Returns None if river is not installed.
    """
    try:
        from river.anomaly import HalfSpaceTrees  # type: ignore
    except ImportError:
        return None

    hst = HalfSpaceTrees(seed=42)
    scores: List[float] = []
    for x in data:
        score = hst.score_one({"x": x})
        hst.learn_one({"x": x})
        scores.append(float(score))

    if not scores or max(scores) == 0.0:
        threshold = 0.5
    else:
        threshold = sorted(scores, reverse=True)[max(1, len(scores) // 20)]
    preds = [s >= threshold for s in scores]

    _prec, _rec, f1 = _compute_f1_with_tolerance(preds, labels, tolerance=5)
    auprc = _pr_auc(scores, labels, tolerance=5)
    vus = _vus_pr(scores, labels)
    return {"f1": f1, "auprc": auprc, "vus_pr": vus}


def _run_fracttalix(
    data: List[float],
    labels: List[int],
    config: Optional[Any] = None,
) -> Dict[str, float]:
    """Run Fracttalix Sentinel full pipeline."""
    from fracttalix.config import SentinelConfig
    from fracttalix.detector import SentinelDetector
    from benchmark.metrics import _pr_auc, _vus_pr, _compute_f1_with_tolerance

    if config is None:
        config = SentinelConfig()
    det = SentinelDetector(config=config)
    results = [det.update_and_check(v) for v in data]

    scores = [float(r.get("anomaly_score", 0.0)) for r in results]
    preds = [bool(r.get("alert", False)) for r in results]

    _prec, _rec, f1 = _compute_f1_with_tolerance(preds, labels, tolerance=5)
    auprc = _pr_auc(scores, labels, tolerance=5)
    vus = _vus_pr(scores, labels)
    return {"f1": f1, "auprc": auprc, "vus_pr": vus}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_baselines(
    archetype: str = "point",
    n: int = 1000,
    seed: int = 42,
    config: Optional[Any] = None,
) -> Dict[str, Optional[Dict[str, float]]]:
    """Compare Fracttalix against naive and optional third-party baselines.

    Parameters
    ----------
    archetype:
        Benchmark archetype name (default "point").
    n:
        Number of observations (default 1000).
    seed:
        RNG seed (default 42).
    config:
        Optional SentinelConfig for the Fracttalix detector.

    Returns
    -------
    dict with keys:
        "fracttalix"  — always present; dict with f1, auprc, vus_pr
        "naive_3sigma" — always present; dict with f1, auprc, vus_pr
        "pyod_ecod"   — dict if pyod installed, else None
        "river_hst"   — dict if river installed, else None
    """
    data, labels = generate(archetype, n=n, seed=seed)

    fracttalix_result = _run_fracttalix(data, labels, config=config)
    naive_result = _naive_3sigma(data, labels)
    pyod_result = _pyod_ecod(data, labels)
    river_result = _river_hst(data, labels)

    results: Dict[str, Optional[Dict[str, float]]] = {
        "fracttalix": fracttalix_result,
        "naive_3sigma": naive_result,
        "pyod_ecod": pyod_result,
        "river_hst": river_result,
    }

    print_comparison_table(results, archetype=archetype)
    return results


def print_comparison_table(
    results: Dict[str, Optional[Dict[str, float]]],
    archetype: str = "",
) -> None:
    """Pretty-print the baseline comparison table."""
    title = f"Baseline Comparison — archetype: {archetype}" if archetype else "Baseline Comparison"
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"{title:^62}")
    print(sep)
    header = f"{'Method':<22} {'F1':>8} {'AUPRC':>8} {'VUS-PR':>8}"
    print(header)
    print(sep)

    labels_map = {
        "fracttalix": "Fracttalix Sentinel",
        "naive_3sigma": "Naive 3-sigma",
        "pyod_ecod": "PyOD ECOD",
        "river_hst": "River HalfSpaceTrees",
    }

    for key, label in labels_map.items():
        val = results.get(key)
        if val is None:
            print(f"{label:<22} {'(not installed)':>26}")
        else:
            print(
                f"{label:<22} {val['f1']:>8.3f} {val['auprc']:>8.3f} "
                f"{val['vus_pr']:>8.3f}"
            )
    print(f"{sep}\n")
