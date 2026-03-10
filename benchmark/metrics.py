# benchmark/metrics.py
# Metric calculation ported from SentinelBenchmark in v11.
# Computes F1 (with buffer tolerance), AUPRC, VUS-PR, and mean detection lag.

from typing import Any, Dict, List, Optional, Tuple

from benchmark.archetypes import ARCHETYPES, generate

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _pr_auc(
    scores: List[float],
    labels: List[int],
    tolerance: int = 0,
) -> float:
    """Trapezoidal AUPRC with buffer tolerance.

    Parameters
    ----------
    scores:
        Continuous anomaly scores (higher = more anomalous).
    labels:
        Ground-truth binary labels (0/1).
    tolerance:
        Half-width of the label expansion window (steps).

    Returns
    -------
    float: Area under the precision-recall curve.
    """
    n = len(scores)
    tol_labels = list(labels)
    if tolerance > 0:
        for i, lbl in enumerate(labels):
            if lbl:
                for j in range(max(0, i - tolerance), min(n, i + tolerance + 1)):
                    tol_labels[j] = 1

    paired = sorted(zip(scores, tol_labels), key=lambda x: -x[0])
    tp = fp = 0
    total_pos = sum(1 for lbl in tol_labels if lbl)
    if total_pos == 0:
        return 0.0

    precs: List[float] = []
    recs: List[float] = []
    for _score, lbl in paired:
        if lbl:
            tp += 1
        else:
            fp += 1
        precs.append(tp / (tp + fp))
        recs.append(tp / total_pos)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(precs)):
        auc += (recs[i] - recs[i - 1]) * (precs[i] + precs[i - 1]) / 2.0
    return max(0.0, auc)


def _vus_pr(scores: List[float], labels: List[int]) -> float:
    """Volume Under PR Surface averaged over buffer tolerances [1, 3, 5, 7, 9]."""
    tols = [1, 3, 5, 7, 9]
    return sum(_pr_auc(scores, labels, t) for t in tols) / len(tols)


def _compute_f1_with_tolerance(
    preds: List[bool],
    labels: List[int],
    tolerance: int = 5,
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 with a tolerance buffer around anomalies.

    A prediction at step i is a true positive if any label in
    [i-tolerance, i+tolerance] is 1.
    """
    n = len(preds)
    # Build expanded label window
    tol_labels = [0] * n
    for i, lbl in enumerate(labels):
        if lbl:
            for j in range(max(0, i - tolerance), min(n, i + tolerance + 1)):
                tol_labels[j] = 1

    tp = fp = fn = 0
    for i, (pred, lbl) in enumerate(zip(preds, labels)):
        if pred:
            if tol_labels[i]:
                tp += 1
            else:
                fp += 1
        else:
            if lbl:
                fn += 1

    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-10)
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    archetype: str,
    config: Optional[Any] = None,
    n: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run SentinelDetector on the named archetype and return metrics.

    Parameters
    ----------
    archetype:
        One of the five benchmark archetypes.
    config:
        Optional SentinelConfig. Defaults to SentinelConfig().
    n:
        Dataset size (default 1000).
    seed:
        RNG seed (default 42).

    Returns
    -------
    dict with keys: archetype, f1, auprc, vus_pr, mean_lag,
                    n_anomalies, n_detections
    """
    from fracttalix.config import SentinelConfig
    from fracttalix.detector import SentinelDetector

    if config is None:
        config = SentinelConfig()

    data, labels = generate(archetype, n=n, seed=seed)
    det = SentinelDetector(config=config)
    results = [det.update_and_check(v) for v in data]

    scores: List[float] = [float(r.get("anomaly_score", 0.0)) for r in results]
    preds: List[bool] = [bool(r.get("alert", False)) for r in results]

    _prec, _rec, f1 = _compute_f1_with_tolerance(preds, labels, tolerance=5)

    # AUPRC at tolerance=5
    auprc = _pr_auc(scores, labels, tolerance=5)

    # VUS-PR
    vus = _vus_pr(scores, labels)

    # Mean detection lag: steps from anomaly-run start to first detection
    anom_starts = [
        i for i, lbl in enumerate(labels)
        if lbl and (i == 0 or not labels[i - 1])
    ]
    detection_lags: List[float] = []
    for start in anom_starts:
        for j in range(start, min(start + 50, len(preds))):
            if preds[j]:
                detection_lags.append(float(j - start))
                break

    mean_lag = _mean(detection_lags) if detection_lags else float("inf")

    return {
        "archetype": archetype,
        "f1": f1,
        "auprc": auprc,
        "vus_pr": vus,
        "mean_lag": mean_lag,
        "n_anomalies": int(sum(labels)),
        "n_detections": int(sum(1 for p in preds if p)),
    }


def run_suite(
    config: Optional[Any] = None,
    n: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run evaluate() on all 5 archetypes and print a formatted table.

    Parameters
    ----------
    config:
        Optional SentinelConfig passed to each evaluate() call.
    n:
        Dataset size per archetype (default 1000).
    seed:
        RNG seed (default 42).

    Returns
    -------
    List of result dicts, one per archetype.
    """
    all_results: List[Dict[str, Any]] = []
    for arch in ARCHETYPES:
        result = evaluate(arch, config=config, n=n, seed=seed)
        all_results.append(result)

    # Print formatted table
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"{'Fracttalix V12 Benchmark Suite':^70}")
    print(sep)
    header = (
        f"{'Archetype':<14} {'F1':>6} {'AUPRC':>7} {'VUS-PR':>7} "
        f"{'Lag':>6} {'#Anom':>6} {'#Det':>6}"
    )
    print(header)
    print(sep)
    for r in all_results:
        lag = f"{r['mean_lag']:.1f}" if r["mean_lag"] != float("inf") else "inf"
        print(
            f"{r['archetype']:<14} {r['f1']:>6.3f} {r['auprc']:>7.3f} "
            f"{r['vus_pr']:>7.3f} {lag:>6} {r['n_anomalies']:>6} "
            f"{r['n_detections']:>6}"
        )
    print(f"{sep}\n")
    return all_results
