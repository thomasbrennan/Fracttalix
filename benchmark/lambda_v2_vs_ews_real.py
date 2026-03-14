#!/usr/bin/env python3
"""
Lambda v2 vs Generic EWS — Head-to-Head on Real-World Data
============================================================

The test that determines whether Fracttalix is superior to competitors.

Runs both detectors on the same real-world datasets and compares:
  - True positive rate (TPR): % of transition trajectories detected
  - False positive rate (FPR): % of null/control trajectories that false-alarm
  - Lead time: how early before transition the first alert fires
  - F1 score: harmonic mean of precision and recall

Datasets:
  1. Thermoacoustic Hopf bifurcation (Bury et al. 2021 PNAS)
  2. Chick heart cell period-doubling (Bury et al.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from fracttalix.suite import LambdaDetector

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ──────────────────────────────────────────────────────────
#  GENERIC EWS BASELINE (Scheffer et al. 2009)
# ──────────────────────────────────────────────────────────

class GenericEWS:
    """Baseline Early Warning Signal detector.

    Standard approach: track rolling variance + lag-1 autocorrelation.
    Alert when both exceed thresholds relative to baseline.
    """

    def __init__(self, window=128, var_threshold=1.5, ac1_threshold=0.7):
        self.window = window
        self.var_threshold = var_threshold
        self.ac1_threshold = ac1_threshold
        self._buffer = []
        self._baseline_var = None
        self._warmup = window * 2

    def update(self, value):
        """Returns (alert, var_ratio, ac1)."""
        self._buffer.append(value)

        if len(self._buffer) < self.window:
            return False, None, None

        w = np.array(self._buffer[-self.window:])
        current_var = np.var(w)

        if len(w) > 1:
            centered = w - np.mean(w)
            c0 = np.sum(centered ** 2)
            c1 = np.sum(centered[:-1] * centered[1:])
            ac1 = c1 / c0 if abs(c0) > 1e-12 else 0.0
        else:
            ac1 = 0.0

        if len(self._buffer) == self._warmup:
            self._baseline_var = current_var

        if self._baseline_var is None:
            return False, None, None

        var_ratio = (
            current_var / self._baseline_var
            if self._baseline_var > 1e-12
            else 1.0
        )

        alert = var_ratio > self.var_threshold and ac1 > self.ac1_threshold
        return alert, var_ratio, ac1

    def reset(self):
        self._buffer = []
        self._baseline_var = None


# ──────────────────────────────────────────────────────────
#  HEAD-TO-HEAD RUNNER
# ──────────────────────────────────────────────────────────

def run_both_on_series(values, ews_var_thresh=1.5, ews_ac1_thresh=0.7):
    """Run Lambda v2 and Generic EWS on the same series."""
    # Lambda v2
    lam_det = LambdaDetector(
        tau_gen=0.0, fit_window=128, fit_interval=8,
        lambda_window=20, lambda_warning=0.05,
    )
    lam_alerts = []
    for t, val in enumerate(values):
        r = lam_det.update(float(val))
        if r.is_alert:
            lam_alerts.append(t)

    # Generic EWS
    ews = GenericEWS(window=128, var_threshold=ews_var_thresh,
                     ac1_threshold=ews_ac1_thresh)
    ews_alerts = []
    for t, val in enumerate(values):
        alert, _, _ = ews.update(float(val))
        if alert:
            ews_alerts.append(t)

    return {
        "lambda_alerts": lam_alerts,
        "lambda_any": len(lam_alerts) > 0,
        "lambda_first": lam_alerts[0] if lam_alerts else None,
        "ews_alerts": ews_alerts,
        "ews_any": len(ews_alerts) > 0,
        "ews_first": ews_alerts[0] if ews_alerts else None,
        "final_scope": lam_det.scope_status,
        "final_lambda": lam_det.current_lambda,
    }


def compute_metrics(tp, fp, fn, tn):
    """Compute TPR, FPR, precision, F1."""
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tpr": tpr, "fpr": fpr, "precision": precision, "recall": recall, "f1": f1}


def test_thermoacoustic():
    """Head-to-head on thermoacoustic Hopf data."""
    print("\n" + "=" * 70)
    print("  TEST 1: Thermoacoustic Hopf Bifurcation")
    print("  19 forced (transition) + 10 null (control)")
    print("=" * 70)

    forced_path = os.path.join(DATA_DIR, "thermoacoustic_ews_forced.csv")
    null_path = os.path.join(DATA_DIR, "thermoacoustic_ews_null.csv")

    if not os.path.exists(forced_path):
        print("  SKIP: data not found")
        return None

    df_forced = pd.read_csv(forced_path)
    df_null = pd.read_csv(null_path) if os.path.exists(null_path) else None

    # Forced trajectories
    lam_tp, ews_tp = 0, 0
    print("\n  Forced trajectories:")
    print(f"  {'tsid':>4s}  {'Lambda':>8s}  {'EWS':>8s}  {'scope':>10s}  {'winner':>12s}")
    for tsid in sorted(df_forced.tsid.unique()):
        sub = df_forced[df_forced.tsid == tsid]
        values = sub["state"].values
        r = run_both_on_series(values)

        if r["lambda_any"]:
            lam_tp += 1
        if r["ews_any"]:
            ews_tp += 1

        winner = "—"
        if r["lambda_any"] and r["ews_any"]:
            if r["lambda_first"] < r["ews_first"]:
                winner = "LAMBDA"
            elif r["ews_first"] < r["lambda_first"]:
                winner = "EWS"
            else:
                winner = "TIE"
        elif r["lambda_any"]:
            winner = "LAMBDA only"
        elif r["ews_any"]:
            winner = "EWS only"
        else:
            winner = "NEITHER"

        print(f"  {tsid:4d}  {len(r['lambda_alerts']):8d}  {len(r['ews_alerts']):8d}  "
              f"{r['final_scope']:>10s}  {winner:>12s}")

    # Null trajectories
    lam_fp, ews_fp = 0, 0
    n_null = 0
    if df_null is not None:
        print("\n  Null trajectories:")
        for tsid in sorted(df_null.tsid.unique()):
            sub = df_null[df_null.tsid == tsid]
            values = sub["state"].values
            r = run_both_on_series(values)
            n_null += 1
            if r["lambda_any"]:
                lam_fp += 1
            if r["ews_any"]:
                ews_fp += 1

    n_forced = len(df_forced.tsid.unique())
    lam_fn = n_forced - lam_tp
    ews_fn = n_forced - ews_tp
    lam_tn = n_null - lam_fp
    ews_tn = n_null - ews_fp

    lam_m = compute_metrics(lam_tp, lam_fp, lam_fn, lam_tn)
    ews_m = compute_metrics(ews_tp, ews_fp, ews_fn, ews_tn)

    print(f"\n  {'Metric':<15s}  {'Lambda v2':>10s}  {'Generic EWS':>12s}  {'Winner':>8s}")
    print(f"  {'─'*50}")
    for key in ["tpr", "fpr", "precision", "f1"]:
        lv = lam_m[key]
        ev = ews_m[key]
        if key == "fpr":
            w = "Lambda" if lv < ev else ("EWS" if ev < lv else "TIE")
        else:
            w = "Lambda" if lv > ev else ("EWS" if ev > lv else "TIE")
        print(f"  {key.upper():<15s}  {lv:10.1%}  {ev:12.1%}  {w:>8s}")

    return {"lambda": lam_m, "ews": ews_m, "name": "Thermoacoustic"}


def test_chick_heart():
    """Head-to-head on chick heart period-doubling data."""
    print("\n" + "=" * 70)
    print("  TEST 2: Chick Heart Cell Period-Doubling")
    print("  23 period-doubling + 23 neutral")
    print("=" * 70)

    chick_path = os.path.join(DATA_DIR, "df_chick.csv")
    if not os.path.exists(chick_path):
        print("  SKIP: data not found")
        return None

    df = pd.read_csv(chick_path)
    ibi_col = [c for c in df.columns if "IBI" in c or "ibi" in c][0]

    results = {"pd": [], "neutral": []}

    for ttype in ["pd", "neutral"]:
        sub_type = df[df["type"] == ttype]
        for tsid in sorted(sub_type.tsid.unique()):
            sub = sub_type[sub_type.tsid == tsid]
            values = sub[ibi_col].values
            r = run_both_on_series(values)
            results[ttype].append(r)

    # Period-doubling (positive class)
    lam_tp = sum(1 for r in results["pd"] if r["lambda_any"])
    ews_tp = sum(1 for r in results["pd"] if r["ews_any"])
    n_pd = len(results["pd"])

    # Neutral (negative class)
    lam_fp = sum(1 for r in results["neutral"] if r["lambda_any"])
    ews_fp = sum(1 for r in results["neutral"] if r["ews_any"])
    n_neutral = len(results["neutral"])

    lam_m = compute_metrics(lam_tp, lam_fp, n_pd - lam_tp, n_neutral - lam_fp)
    ews_m = compute_metrics(ews_tp, ews_fp, n_pd - ews_tp, n_neutral - ews_fp)

    print(f"\n  Period-doubling: Lambda detects {lam_tp}/{n_pd}, EWS detects {ews_tp}/{n_pd}")
    print(f"  Neutral FP:      Lambda {lam_fp}/{n_neutral}, EWS {ews_fp}/{n_neutral}")

    print(f"\n  {'Metric':<15s}  {'Lambda v2':>10s}  {'Generic EWS':>12s}  {'Winner':>8s}")
    print(f"  {'─'*50}")
    for key in ["tpr", "fpr", "precision", "f1"]:
        lv = lam_m[key]
        ev = ews_m[key]
        if key == "fpr":
            w = "Lambda" if lv < ev else ("EWS" if ev < lv else "TIE")
        else:
            w = "Lambda" if lv > ev else ("EWS" if ev > lv else "TIE")
        print(f"  {key.upper():<15s}  {lv:10.1%}  {ev:12.1%}  {w:>8s}")

    return {"lambda": lam_m, "ews": ews_m, "name": "Chick Heart"}


def main():
    print("=" * 70)
    print("  Lambda v2 vs Generic EWS — Head-to-Head on Real-World Data")
    print("  'Superior to competitors, or don't release'")
    print("=" * 70)

    all_results = []

    r1 = test_thermoacoustic()
    if r1:
        all_results.append(r1)

    r2 = test_chick_heart()
    if r2:
        all_results.append(r2)

    # ══════════════════════════════════════════════════════
    #  OVERALL VERDICT
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  OVERALL VERDICT")
    print("=" * 70)

    lambda_superior = 0
    ews_superior = 0
    ties = 0

    for r in all_results:
        lf1 = r["lambda"]["f1"]
        ef1 = r["ews"]["f1"]
        if lf1 > ef1 + 0.05:
            lambda_superior += 1
            verdict = "Lambda SUPERIOR"
        elif ef1 > lf1 + 0.05:
            ews_superior += 1
            verdict = "EWS SUPERIOR"
        else:
            ties += 1
            verdict = "COMPARABLE"
        print(f"\n  {r['name']:20s}: Lambda F1={lf1:.3f}  EWS F1={ef1:.3f}  → {verdict}")

    print(f"\n  Lambda superior:  {lambda_superior}/{len(all_results)}")
    print(f"  EWS superior:     {ews_superior}/{len(all_results)}")
    print(f"  Comparable:       {ties}/{len(all_results)}")

    print(f"\n  {'─'*60}")
    if lambda_superior > ews_superior and ews_superior == 0:
        print("  RELEASE GATE: PASS — Lambda v2 is superior to generic EWS")
    elif lambda_superior > ews_superior:
        print("  RELEASE GATE: PARTIAL — Lambda v2 wins some, loses some")
    elif lambda_superior == ews_superior:
        print("  RELEASE GATE: FAIL — No clear advantage over generic EWS")
    else:
        print("  RELEASE GATE: FAIL — Generic EWS is better")
    print(f"  {'─'*60}")

    return lambda_superior > ews_superior and ews_superior == 0


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
