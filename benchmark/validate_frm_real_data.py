#!/usr/bin/env python3
"""
FRM Suite — Real-World Data Validation
=======================================

Tests the FRM detectors (Lambda, Omega, Virtu) on genuine empirical data
with known transitions.  This is the test that determines whether the
software has any worth at all.

Datasets:
  1. Thermoacoustic Hopf bifurcation (Bury et al. 2021 PNAS)
     - 19 forced trajectories: Rijke tube voltage ramp → Hopf bifurcation
     - 10 null trajectories: steady-state, no transition
     - 1500 points each, 2 kHz sampling

  2. Chick heart cell aggregates (period-doubling bifurcation)
     - 'pd' type: approaching period-doubling bifurcation
     - 'neutral' type: no bifurcation (control)

Question:
  Does frm_confidence >= 2 on real-world oscillatory data reliably
  precede transitions?  Can the FRM form even fit this data?
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from fracttalix.suite import (
    LambdaDetector, OmegaDetector, VirtuDetector, ScopeStatus,
)


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def run_frm_on_series(values, tau_gen=0.0, fit_window=128, fit_interval=8):
    """Run all 3 FRM detectors on a time series.

    tau_gen=0 means Lambda estimates omega from FFT (weak mode).

    Returns dict with diagnostics.
    """
    lam_det = LambdaDetector(
        tau_gen=tau_gen, fit_window=fit_window, fit_interval=fit_interval,
        lambda_window=20, lambda_warning=0.05,
    )
    omega_det = OmegaDetector(
        tau_gen=tau_gen, fit_window=fit_window, deviation_threshold=0.15,
    )
    virtu_det = VirtuDetector(
        lambda_detector=lam_det, decision_horizon=100.0,
    )

    n = len(values)
    lambda_alerts = []
    omega_alerts = []
    virtu_phases = []
    scopes = []
    r2_values = []
    lambda_values = []

    for t, val in enumerate(values):
        v = float(val)
        lam_r = lam_det.update(v)
        omega_r = omega_det.update(v)
        virtu_r = virtu_det.update(v)

        if lam_r.is_alert:
            lambda_alerts.append(t)
        if omega_r.is_alert:
            omega_alerts.append(t)

        scopes.append(lam_det.scope_status)
        if lam_det.r_squared > 0:
            r2_values.append(lam_det.r_squared)
        if lam_det.current_lambda is not None:
            lambda_values.append(lam_det.current_lambda)

        dq = virtu_det.decision_quality
        if dq >= 0.7:
            virtu_phases.append(("ACT NOW", t, dq))
        elif dq >= 0.4:
            virtu_phases.append(("ACT SOON", t, dq))
        elif dq >= 0.2:
            virtu_phases.append(("MONITOR", t, dq))

    scope_counts = {}
    for s in scopes:
        scope_counts[s] = scope_counts.get(s, 0) + 1

    return {
        "n": n,
        "lambda_alerts": lambda_alerts,
        "omega_alerts": omega_alerts,
        "virtu_phases": virtu_phases,
        "scope_counts": scope_counts,
        "r2_values": r2_values,
        "lambda_values": lambda_values,
        "final_scope": lam_det.scope_status,
        "final_r2": lam_det.r_squared,
        "final_lambda": lam_det.current_lambda,
        "peak_quality": virtu_det.peak_quality,
        "time_to_transition": lam_det.time_to_transition,
    }


def test_thermoacoustic():
    """Test on Rijke tube Hopf bifurcation data."""
    print("\n" + "=" * 70)
    print("  TEST 1: Thermoacoustic Hopf Bifurcation (Bury et al. 2021)")
    print("  19 forced trajectories → Hopf bifurcation in Rijke tube")
    print("=" * 70)

    forced_path = os.path.join(DATA_DIR, "thermoacoustic_ews_forced.csv")
    null_path = os.path.join(DATA_DIR, "thermoacoustic_ews_null.csv")

    if not os.path.exists(forced_path):
        print("  SKIP: thermoacoustic data not found")
        return None

    df_forced = pd.read_csv(forced_path)
    df_null = pd.read_csv(null_path) if os.path.exists(null_path) else None

    # ── Forced trajectories (approaching Hopf bifurcation) ──
    print("\n  Forced trajectories (approaching Hopf bifurcation):")
    forced_results = []

    for tsid in sorted(df_forced.tsid.unique()):
        sub = df_forced[df_forced.tsid == tsid]
        values = sub["state"].values

        # Sampling: 2 kHz, 1500 points = 0.75 seconds
        # The oscillation frequency in a Rijke tube is typically 100-400 Hz
        # → tau_gen ≈ T/4 = 1/(4*f) ≈ 0.625 to 2.5 ms = 1.25 to 5 samples at 2 kHz
        # Use FFT estimation (tau_gen=0)
        result = run_frm_on_series(values, tau_gen=0.0, fit_window=128, fit_interval=8)
        forced_results.append(result)

        n_alerts = len(result["lambda_alerts"])
        n_virtu = len(result["virtu_phases"])
        r2_med = np.median(result["r2_values"]) if result["r2_values"] else 0
        scope = result["final_scope"]

        print(f"    tsid={tsid:2d}: scope={scope:15s} R²={r2_med:.3f} "
              f"λ_alerts={n_alerts:3d} virtu={n_virtu:3d} "
              f"peak_q={result['peak_quality']:.3f}")

    # Summary
    in_scope = sum(1 for r in forced_results
                   if r["final_scope"] in ("IN_SCOPE", "BOUNDARY"))
    any_alert = sum(1 for r in forced_results if len(r["lambda_alerts"]) > 0)
    any_virtu = sum(1 for r in forced_results if len(r["virtu_phases"]) > 0)
    r2_all = [v for r in forced_results for v in r["r2_values"]]

    print(f"\n  Summary ({len(forced_results)} forced trajectories):")
    print(f"    In-scope (IN_SCOPE or BOUNDARY): {in_scope}/{len(forced_results)}")
    print(f"    Any Lambda alert:                {any_alert}/{len(forced_results)}")
    print(f"    Any Virtu phase (MONITOR+):      {any_virtu}/{len(forced_results)}")
    if r2_all:
        r2a = np.array(r2_all)
        print(f"    R² median: {np.median(r2a):.3f}, mean: {np.mean(r2a):.3f}")
        print(f"    R² > 0.5: {(r2a > 0.5).sum()}/{len(r2a)} ({(r2a > 0.5).mean():.1%})")

    # ── Null trajectories (no transition — FPR check) ──
    if df_null is not None:
        print(f"\n  Null trajectories (steady-state, no transition):")
        null_results = []

        for tsid in sorted(df_null.tsid.unique()):
            sub = df_null[df_null.tsid == tsid]
            values = sub["state"].values
            result = run_frm_on_series(values, tau_gen=0.0, fit_window=128, fit_interval=8)
            null_results.append(result)

        null_alerts = sum(1 for r in null_results if len(r["lambda_alerts"]) > 0)
        null_virtu = sum(1 for r in null_results if len(r["virtu_phases"]) > 0)
        print(f"    Lambda false alarms: {null_alerts}/{len(null_results)}")
        print(f"    Virtu false alarms:  {null_virtu}/{len(null_results)}")
    else:
        null_results = []

    return {
        "forced": forced_results,
        "null": null_results,
        "in_scope_rate": in_scope / max(1, len(forced_results)),
        "alert_rate": any_alert / max(1, len(forced_results)),
        "virtu_rate": any_virtu / max(1, len(forced_results)),
    }


def test_chick_heart():
    """Test on chick heart cell period-doubling data."""
    print("\n" + "=" * 70)
    print("  TEST 2: Chick Heart Cell Period-Doubling (Bury et al.)")
    print("  IBI time series: 'pd' = period-doubling, 'neutral' = control")
    print("=" * 70)

    path = os.path.join(DATA_DIR, "df_chick.csv")
    if not os.path.exists(path):
        print("  SKIP: chick heart data not found")
        return None

    df = pd.read_csv(path)

    for dtype in ["pd", "neutral"]:
        sub_df = df[df.type == dtype]
        tsids = sorted(sub_df.tsid.unique())
        label = "Period-doubling" if dtype == "pd" else "Neutral (control)"
        print(f"\n  {label} ({len(tsids)} trajectories):")

        results = []
        for tsid in tsids:
            sub = sub_df[sub_df.tsid == tsid]
            values = sub["IBI (s)"].values

            # IBI data: beat-to-beat interval. Oscillation frequency = ~1 Hz
            # tau_gen estimated from FFT
            result = run_frm_on_series(values, tau_gen=0.0, fit_window=64, fit_interval=4)
            results.append(result)

            n_alerts = len(result["lambda_alerts"])
            scope = result["final_scope"]
            r2_med = np.median(result["r2_values"]) if result["r2_values"] else 0

            print(f"    tsid={tsid:2d}: scope={scope:15s} R²={r2_med:.3f} "
                  f"λ_alerts={n_alerts:3d} peak_q={result['peak_quality']:.3f}")

        in_scope = sum(1 for r in results
                       if r["final_scope"] in ("IN_SCOPE", "BOUNDARY"))
        any_alert = sum(1 for r in results if len(r["lambda_alerts"]) > 0)
        print(f"    In-scope: {in_scope}/{len(results)}, Alerts: {any_alert}/{len(results)}")

    return True


def test_sunspots():
    """Test on sunspot data (quasi-periodic, known NOT to be FRM-shaped)."""
    print("\n" + "=" * 70)
    print("  TEST 3: Sunspot Numbers (quasi-periodic, expected OUT_OF_SCOPE)")
    print("=" * 70)

    try:
        import statsmodels.api as sm
        data = sm.datasets.sunspots.load_pandas().data
        values = data.SUNACTIVITY.values
    except ImportError:
        print("  SKIP: statsmodels not available")
        return None

    # tau_gen for ~11-year solar cycle: T/4 ≈ 2.75 years (annual data)
    result = run_frm_on_series(values, tau_gen=2.75, fit_window=64, fit_interval=4)

    r2_med = np.median(result["r2_values"]) if result["r2_values"] else 0
    print(f"\n  n={result['n']}, scope={result['final_scope']}")
    print(f"  R² median: {r2_med:.3f}")
    print(f"  Scope distribution: {result['scope_counts']}")
    print(f"  Lambda alerts: {len(result['lambda_alerts'])}")

    oos = result["scope_counts"].get("OUT_OF_SCOPE", 0) + \
          result["scope_counts"].get("LIMIT_CYCLE", 0)
    total = sum(result["scope_counts"].values())
    print(f"  OOS + LIMIT_CYCLE: {oos}/{total} ({oos/max(1,total):.1%})")
    print(f"  (Expected: mostly OUT_OF_SCOPE — quasi-periodic != damped)")

    return result


def main():
    print("=" * 70)
    print("  FRM Suite — Real-World Data Validation")
    print("  'The test that determines whether the software has any worth'")
    print("=" * 70)

    thermo = test_thermoacoustic()
    chick = test_chick_heart()
    sunspot = test_sunspots()

    # ══════════════════════════════════════════════════════
    #  VERDICT
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  VERDICT: Does the FRM suite work on real-world data?")
    print("=" * 70)

    criteria = []

    if thermo:
        # C1: FRM fits at least some forced trajectories
        if thermo["in_scope_rate"] > 0.3:
            criteria.append(("FRM fits Hopf data (>30% in-scope)", "PASS",
                            f"{thermo['in_scope_rate']:.0%}"))
        else:
            criteria.append(("FRM fits Hopf data (>30% in-scope)", "FAIL",
                            f"{thermo['in_scope_rate']:.0%}"))

        # C2: Lambda fires on at least some forced trajectories
        if thermo["alert_rate"] > 0.2:
            criteria.append(("Lambda alerts on real Hopf data (>20%)", "PASS",
                            f"{thermo['alert_rate']:.0%}"))
        else:
            criteria.append(("Lambda alerts on real Hopf data (>20%)", "FAIL",
                            f"{thermo['alert_rate']:.0%}"))

    print()
    gate_pass = True
    for name, status, detail in criteria:
        marker = "+" if status == "PASS" else "-"
        print(f"  [{marker}] {status}: {name}")
        print(f"       {detail}")
        if status == "FAIL":
            gate_pass = False

    if not criteria:
        print("  No criteria could be evaluated (missing data)")
        gate_pass = False

    print(f"\n  {'─'*60}")
    if gate_pass:
        print("  GATE: PASS — FRM suite shows promise on real-world data")
    else:
        print("  GATE: FAIL — FRM suite does not yet demonstrate value")
        print("  on real-world data with known transitions.")
    print(f"  {'─'*60}")

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
