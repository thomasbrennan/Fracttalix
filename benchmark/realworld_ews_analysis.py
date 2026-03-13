"""
realworld_ews_analysis.py — Real-world EWS validation on paleoclimate records

Datasets: earlywarningtoolbox/datasets (Dakos et al. 2008, PNAS)
  - YD2PB grayscale: 2110 points, Younger Dryas → Pre-Boreal transition ~11,700 BP
  - Vostok1 deuterium: 512 points, Antarctic ice core 12,000–58,759 BP
  - GBA temp: 131 points, Greenland Basin Area

WHAT THIS CAN TEST:
  DetectorSuite (generic EWS): do variance, autocorrelation, drift signals rise
  before the known transition? This tests whether our Layer 1 detectors replicate
  the Dakos et al. findings (increased variance/AC before transition).

WHAT THIS CANNOT TEST:
  FRMSuite physics validation. These are paleoclimate fold/saddle-node bifurcations,
  NOT Hopf bifurcations. There is no characteristic oscillation frequency and no
  FRM-derived tau_gen. FRMSuite Layer 2 (Lambda, Omega, Virtu) is OUT_OF_SCOPE.
  frm_confidence=3 validation requires oscillatory data with known tau_gen.

TRANSITION POINTS:
  YD2PB: Younger Dryas/Pre-Boreal boundary at 11,700 BP.
    Data ordered oldest→newest (12,549→11,212 BP).
    Transition at ~68.9% through = index 1454 of 2110.
    Pre-transition window: indices 0..1453 (Younger Dryas, cold)
    Post-transition: indices 1454..2109 (Pre-Boreal, warming)

METHODOLOGY:
  1. Run DetectorSuite streaming on each full time series
  2. Track per-step alert state and per-detector scores
  3. For YD2PB: report alert rate in final 200 steps before transition
     vs alert rate in stable pre-transition period (steps 100–400)
  4. For Virtu/FRM scope: explicitly confirm OUT_OF_SCOPE on non-oscillatory data
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from fracttalix.suite import DetectorSuite
from fracttalix.suite.base import ScopeStatus

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load(fname):
    path = os.path.join(DATA_DIR, fname)
    return [float(x) for x in open(path).read().strip().split()]


def run_suite(signal, label, transition_idx=None):
    """Run DetectorSuite on signal, return per-step results."""
    suite = DetectorSuite()
    results = []
    for i, v in enumerate(signal):
        r = suite.update(v)
        results.append({
            "step": i,
            "any_alert": r.any_alert,
            "hopf": r.hopf.is_alert,
            "drift": r.drift.is_alert,
            "variance": r.variance.is_alert,
            "discord": r.discord.is_alert,
            "coupling": r.coupling.is_alert,
            "hopf_score": r.hopf.score,
            "drift_score": r.drift.score,
            "variance_score": r.variance.score,
            "coupling_scope": r.coupling.status != ScopeStatus.OUT_OF_SCOPE,
        })

    n = len(signal)
    warmup = 80  # DetectorSuite warmup

    print(f"\n{'='*60}")
    print(f"Dataset: {label}  (n={n})")
    if transition_idx:
        print(f"Transition index: {transition_idx} ({100*transition_idx/n:.1f}% through)")
    print(f"{'='*60}")

    # Alert rates by window
    def alert_rate(lo, hi, key="any_alert"):
        window = [r[key] for r in results[lo:hi]]
        return sum(window) / len(window) if window else 0.0

    # Stable pre-transition baseline (well before transition, after warmup)
    stable_lo = warmup
    stable_hi = min(warmup + 400, (transition_idx or n) - 200)

    print(f"\nAlert rates (DetectorSuite, post-warmup):")
    print(f"  {'Period':<35} {'Any':>5}  {'Hopf':>5}  {'Drift':>5}  {'Var':>5}")

    if transition_idx and transition_idx > stable_hi + 100:
        # Pre-transition stable window
        print(f"  {'Stable baseline (steps %d–%d)' % (stable_lo, stable_hi):<35} "
              f"{alert_rate(stable_lo, stable_hi):.1%}  "
              f"{alert_rate(stable_lo, stable_hi, 'hopf'):.1%}  "
              f"{alert_rate(stable_lo, stable_hi, 'drift'):.1%}  "
              f"{alert_rate(stable_lo, stable_hi, 'variance'):.1%}")

        # Approach window: final 200 steps before transition
        pre_lo = max(warmup, transition_idx - 200)
        pre_hi = transition_idx
        print(f"  {'Approach window (steps %d–%d)' % (pre_lo, pre_hi):<35} "
              f"{alert_rate(pre_lo, pre_hi):.1%}  "
              f"{alert_rate(pre_lo, pre_hi, 'hopf'):.1%}  "
              f"{alert_rate(pre_lo, pre_hi, 'drift'):.1%}  "
              f"{alert_rate(pre_lo, pre_hi, 'variance'):.1%}")

        # Immediate pre-transition: final 50 steps
        imm_lo = max(warmup, transition_idx - 50)
        imm_hi = transition_idx
        print(f"  {'Immediate pre-transition (steps %d–%d)' % (imm_lo, imm_hi):<35} "
              f"{alert_rate(imm_lo, imm_hi):.1%}  "
              f"{alert_rate(imm_lo, imm_hi, 'hopf'):.1%}  "
              f"{alert_rate(imm_lo, imm_hi, 'drift'):.1%}  "
              f"{alert_rate(imm_lo, imm_hi, 'variance'):.1%}")

        # Post-transition
        post_lo = transition_idx
        post_hi = min(n, transition_idx + 200)
        if post_lo < post_hi:
            print(f"  {'Post-transition (steps %d–%d)' % (post_lo, post_hi):<35} "
                  f"{alert_rate(post_lo, post_hi):.1%}  "
                  f"{alert_rate(post_lo, post_hi, 'hopf'):.1%}  "
                  f"{alert_rate(post_lo, post_hi, 'drift'):.1%}  "
                  f"{alert_rate(post_lo, post_hi, 'variance'):.1%}")

        # First alert before transition
        first_alert = next((r["step"] for r in results if r["step"] >= warmup and r["any_alert"]
                            and r["step"] < transition_idx), None)
        if first_alert:
            lead = transition_idx - first_alert
            print(f"\n  First alert: step {first_alert} → {lead} steps before transition")
        else:
            print(f"\n  No alert fired before transition")

        # Score trends: do EWS scores rise as transition approaches?
        q1_lo, q1_hi = warmup, warmup + (transition_idx - warmup) // 3
        q3_lo, q3_hi = transition_idx - (transition_idx - warmup) // 3, transition_idx

        def mean_score(lo, hi, key):
            vals = [r[key] for r in results[lo:hi]]
            return sum(vals) / len(vals) if vals else 0.0

        print(f"\n  EWS score trend (do scores rise approaching transition?):")
        print(f"  {'Detector':<20} {'Early third':>12}  {'Final third':>12}  {'Rising?':>8}")
        for det, key in [("HopfDetector(ews)", "hopf_score"),
                         ("DriftDetector", "drift_score"),
                         ("VarianceDetector", "variance_score")]:
            early = mean_score(q1_lo, q1_hi, key)
            late  = mean_score(q3_lo, q3_hi, key)
            rising = "YES ↑" if late > early * 1.2 else ("FLAT" if abs(late - early) < 0.05 else "NO ↓")
            print(f"  {det:<20} {early:>12.3f}  {late:>12.3f}  {rising:>8}")

    else:
        # No transition index — full series report
        print(f"  Full series (post-warmup):  {alert_rate(warmup, n):.1%} alert rate")

    # FRM scope check on a sample window
    print(f"\n  FRM scope check (is signal oscillatory?):")
    mid = n // 2
    window = signal[mid:mid+200]
    vals = window
    mean_v = sum(vals) / len(vals)
    variance = sum((v - mean_v)**2 for v in vals) / len(vals)
    # Simple zero-crossing frequency estimate
    crossings = sum(1 for i in range(1, len(vals)) if (vals[i]-mean_v)*(vals[i-1]-mean_v) < 0)
    freq_est = crossings / (2 * len(vals))  # cycles per sample
    print(f"  Mean={mean_v:.2f}  Std={variance**0.5:.2f}  Zero-crossing freq≈{freq_est:.4f} cycles/sample")
    if freq_est > 0.05:
        tau_gen_est = 1.0 / (4 * freq_est)  # quarter-wave: tau = 1/(4f)
        print(f"  → tau_gen estimate (quarter-wave): {tau_gen_est:.1f} samples")
        print(f"  → Oscillatory enough for FRMSuite weak mode")
    else:
        print(f"  → Signal is slow-varying, not oscillatory → FRMSuite OUT_OF_SCOPE (correct)")

    return results


def interpret(yd2pb_results, transition_idx):
    """Print overall interpretation."""
    n = len(yd2pb_results)
    warmup = 80

    print(f"\n{'='*60}")
    print(f"INTERPRETATION")
    print(f"{'='*60}")

    # Check if alert rate genuinely rises before transition
    stable_rate = sum(1 for r in yd2pb_results[warmup:warmup+400] if r["any_alert"]) / 400
    approach_rate = sum(1 for r in yd2pb_results[max(warmup, transition_idx-200):transition_idx] if r["any_alert"]) / 200

    ratio = approach_rate / stable_rate if stable_rate > 0 else float('inf')

    print(f"\nQ: Does DetectorSuite give more alerts approaching the transition?")
    if ratio >= 2.0:
        print(f"   YES — approach alert rate ({approach_rate:.1%}) is {ratio:.1f}× the stable baseline ({stable_rate:.1%}).")
        print(f"   This replicates the Dakos et al. finding of rising EWS before abrupt transitions.")
    elif ratio >= 1.3:
        print(f"   WEAKLY — approach rate ({approach_rate:.1%}) is {ratio:.1f}× baseline ({stable_rate:.1%}).")
        print(f"   Modest increase consistent with, but not strongly confirming, EWS theory.")
    else:
        print(f"   NO — approach rate ({approach_rate:.1%}) ≈ baseline ({stable_rate:.1%}) (ratio {ratio:.1f}×).")
        print(f"   DetectorSuite does not show rising EWS before this transition.")

    print(f"\nQ: Does this validate FRM physics?")
    print(f"   NO — for two independent reasons:")
    print(f"   1. These are fold/saddle-node bifurcations (abrupt climate shifts), not Hopf")
    print(f"      bifurcations. The FRM predicts Hopf dynamics (oscillation approach).")
    print(f"      These transitions lack a characteristic frequency and have no tau_gen.")
    print(f"   2. Even if EWS fired perfectly here, it would validate critical slowing down")
    print(f"      (a general phenomenon), not FRM-specific physics.")

    print(f"\nQ: What WOULD validate FRM physics?")
    print(f"   Real-world oscillatory data with known Hopf-type transitions:")
    print(f"   - EEG alpha rhythm → seizure onset (tau_gen ≈ 4–9 samples at 173 Hz)")
    print(f"   - Thermoacoustic pressure oscillations → instability onset")
    print(f"   - HRV/cardiac oscillations → arrhythmia transition")
    print(f"   Test: does frm_confidence=3 fire before the transition, and is Virtu's")
    print(f"   TTB estimate within 2× of the true remaining time?")
    print(f"   Status: appropriate datasets not freely downloadable; logged in collab thread.")


def main():
    print("Real-World EWS Analysis — Fracttalix DetectorSuite on Paleoclimate Records")
    print("Datasets: earlywarningtoolbox/datasets (Dakos et al. 2008, PNAS)")

    # --- YD2PB: Younger Dryas to Pre-Boreal (2110 pts, transition at ~68.9%) ---
    yd2pb = load("yd2pb_raw.txt")
    transition_idx = int(0.689 * len(yd2pb))  # ~11,700 BP = 68.9% through
    yd2pb_results = run_suite(yd2pb, "YD2PB Grayscale (Younger Dryas → Pre-Boreal)", transition_idx)

    # --- Vostok1: Antarctic ice core deuterium (512 pts) ---
    # Dakos 2008 used Vostok for the YD transition; time range 11,973–58,759 BP
    # The transition of interest (Termination I, ~18 ka BP) is at ~74% through
    vostok = load("vostok1_raw.txt")
    vostok_transition = int(0.74 * len(vostok))
    run_suite(vostok, "Vostok1 Deuterium (Antarctic ice core, Termination I ~18 ka)", vostok_transition)

    # --- GBA temp: 131 pts ---
    gba = load("gba_raw.txt")
    run_suite(gba, "GBA Temperature (Greenland Basin, 14.5–21 ka)", transition_idx=int(0.6 * len(gba)))

    # --- Interpretation ---
    interpret(yd2pb_results, transition_idx)


if __name__ == "__main__":
    main()
