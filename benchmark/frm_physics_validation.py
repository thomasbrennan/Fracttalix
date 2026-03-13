"""
frm_physics_validation.py — Prospective physics validation for FRMSuite

Question: When frm_confidence=3 first fires on a synthetic Hopf approach signal,
  (a) how much lead time before bifurcation does it give?
  (b) how accurate is Virtu's TTB estimate at that moment?

This is distinct from the F-S4 gate (which found the *best* TTB estimate across
all anomaly steps). This test asks the harder prospective question:
at the moment the suite first reaches maximum confidence, is the estimate useful?

Signal model (sig_hopf_approach):
  - Steps 0..n-201: stable oscillation, λ=0.5
  - Steps n-200..n-1: λ declines 0.5→0 linearly (Hopf approach)
  - "Bifurcation" is defined as step n (λ=0 reached)
  - True TTB at step i (during anomaly phase): n - i steps

Metrics reported per trial:
  - first_conf3_step: step at which frm_confidence=3 first fires (None if never)
  - lead_time: true TTB at that step (how much warning time)
  - virtu_est: Virtu's TTB estimate at that step
  - ratio: virtu_est / lead_time (1.0 = perfect; <1 = underestimate; >1 = overestimate)
  - within_2x: abs(ratio - 1.0) < 1.0 (i.e., ratio in [0.5, 2.0])

Also reports frm_confidence timeline: at each step of the anomaly phase,
what fraction of trials have reached confidence >= 1, 2, 3?
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fracttalix.frm import FRMSuite
    FRM_AVAILABLE = True
except ImportError:
    print("ERROR: FRMSuite not available. Run: pip install fracttalix[fast]")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Signal generator (mirrors frm_suite_sandbox.sig_hopf_approach exactly)
# ---------------------------------------------------------------------------

def sig_hopf_approach(n: int, seed: int, tau_gen: float = 10.0):
    """Synthetic FRM Hopf approach — λ declining 0.5→0 over last 200 steps."""
    rng = random.Random(seed)
    omega = math.pi / (2.0 * tau_gen)
    signal = []
    flags = []
    anomaly_start = n - 200
    for i in range(n):
        if i < anomaly_start:
            lam = 0.5
        else:
            progress = (i - anomaly_start) / 200.0
            lam = 0.5 * (1.0 - progress)
        t = float(i)
        val = 3.0 * math.exp(-lam * (t % 50)) * math.cos(omega * t) + rng.gauss(0, 0.2)
        signal.append(val)
        flags.append(i >= anomaly_start)
    return signal, flags


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(seed: int, n: int, tau_gen: float) -> dict:
    """Run one trial. Returns prospective validation metrics."""
    sig, flags = sig_hopf_approach(n, seed, tau_gen)
    bifurcation_step = n  # λ=0 is reached at the end of the signal

    suite = FRMSuite(tau_gen=tau_gen)

    first_conf3_step = None
    first_conf3_virtu_est = None
    first_conf3_lead_time = None

    # Track frm_confidence at each anomaly-phase step (relative offset from anomaly_start)
    anomaly_start = n - 200
    conf_by_offset = {}  # offset -> frm_confidence

    for i, v in enumerate(sig):
        r = suite.update(v)

        if flags[i]:
            offset = i - anomaly_start  # 0..199
            conf_by_offset[offset] = r.frm_confidence

            if first_conf3_step is None and r.frm_confidence == 3:
                first_conf3_step = i
                first_conf3_lead_time = bifurcation_step - i  # true TTB

                # Extract Virtu TTB estimate
                msg = r.virtu.message or ""
                if "ttb=" in msg:
                    try:
                        ttb_str = msg.split("ttb=")[1].split()[0]
                        first_conf3_virtu_est = float(ttb_str)
                    except (IndexError, ValueError):
                        first_conf3_virtu_est = None

    ratio = None
    within_2x = False
    if first_conf3_lead_time is not None and first_conf3_virtu_est is not None and first_conf3_lead_time > 0:
        ratio = first_conf3_virtu_est / first_conf3_lead_time
        within_2x = 0.5 <= ratio <= 2.0

    return {
        "seed": seed,
        "first_conf3_step": first_conf3_step,
        "lead_time": first_conf3_lead_time,
        "virtu_est": first_conf3_virtu_est,
        "ratio": ratio,
        "within_2x": within_2x,
        "conf_by_offset": conf_by_offset,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N = 500
    TAU_GEN = 10.0
    SEEDS = list(range(20))  # 20 trials for robust statistics

    print(f"FRM Physics Validation — Prospective TTB accuracy at frm_confidence=3")
    print(f"Signal: Hopf approach (n={N}, tau_gen={TAU_GEN}), {len(SEEDS)} seeds")
    print(f"Bifurcation: step {N} (λ=0). Anomaly phase: steps {N-200}..{N-1}.")
    print()

    results = []
    for seed in SEEDS:
        r = run_trial(seed, N, TAU_GEN)
        results.append(r)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------

    print(f"{'Seed':>4}  {'conf3@step':>10}  {'lead_time':>9}  {'virtu_est':>9}  {'ratio':>6}  {'within 2×':>9}")
    print("-" * 60)

    fired = [r for r in results if r["first_conf3_step"] is not None]
    no_fire = [r for r in results if r["first_conf3_step"] is None]

    for r in results:
        seed = r["seed"]
        if r["first_conf3_step"] is None:
            print(f"{seed:>4}  {'never':>10}  {'—':>9}  {'—':>9}  {'—':>6}  {'—':>9}")
        else:
            ratio_str = f"{r['ratio']:.2f}" if r["ratio"] is not None else "—"
            w2x = "YES" if r["within_2x"] else "NO"
            vest = f"{r['virtu_est']:.1f}" if r["virtu_est"] is not None else "—"
            print(f"{seed:>4}  {r['first_conf3_step']:>10}  {r['lead_time']:>9}  {vest:>9}  {ratio_str:>6}  {w2x:>9}")

    print()

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------

    n_trials = len(SEEDS)
    n_fired = len(fired)
    print(f"── Aggregate ({n_trials} trials) ──────────────────────────────")
    print(f"  frm_confidence=3 fired:  {n_fired}/{n_trials} ({100*n_fired/n_trials:.0f}%)")

    if fired:
        lead_times = [r["lead_time"] for r in fired]
        print(f"  Lead time when fired:    min={min(lead_times)}  mean={sum(lead_times)/len(lead_times):.1f}  max={max(lead_times)} steps")

        ratios_with_virtu = [r for r in fired if r["ratio"] is not None]
        if ratios_with_virtu:
            ratios = [r["ratio"] for r in ratios_with_virtu]
            within_2x = [r for r in ratios_with_virtu if r["within_2x"]]
            print(f"  Virtu TTB reported:      {len(ratios_with_virtu)}/{n_fired} trials")
            print(f"  Ratio (est/true) range:  min={min(ratios):.2f}  mean={sum(ratios)/len(ratios):.2f}  max={max(ratios):.2f}")
            print(f"  Within 2× of true TTB:  {len(within_2x)}/{len(ratios_with_virtu)} ({100*len(within_2x)/len(ratios_with_virtu):.0f}%)")
        else:
            print("  Virtu TTB: no estimates reported at conf3 moment")

    # ---------------------------------------------------------------------------
    # Confidence ramp: at each anomaly offset, what fraction have conf >= k?
    # ---------------------------------------------------------------------------

    print()
    print("── frm_confidence ramp (anomaly phase, all trials) ────────────")
    print(f"  {'offset':>6}  {'%conf≥1':>8}  {'%conf≥2':>8}  {'%conf=3':>8}  (steps before bifurcation: 200-offset)")

    checkpoints = [0, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 199]
    for offset in checkpoints:
        c1 = sum(1 for r in results if r["conf_by_offset"].get(offset, 0) >= 1)
        c2 = sum(1 for r in results if r["conf_by_offset"].get(offset, 0) >= 2)
        c3 = sum(1 for r in results if r["conf_by_offset"].get(offset, 0) == 3)
        remaining = 200 - offset
        print(f"  {offset:>6}  {100*c1/n_trials:>7.0f}%  {100*c2/n_trials:>7.0f}%  {100*c3/n_trials:>7.0f}%  ({remaining} steps to bifurcation)")

    # ---------------------------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------------------------

    print()
    print("── Interpretation ─────────────────────────────────────────────")

    if not fired:
        print("  frm_confidence=3 never fired. Test is inconclusive.")
        return

    fire_rate = n_fired / n_trials
    mean_lead = sum(r["lead_time"] for r in fired) / len(fired)

    ratios_with_virtu = [r for r in fired if r["ratio"] is not None]
    if ratios_with_virtu:
        within_2x_rate = len([r for r in ratios_with_virtu if r["within_2x"]]) / len(ratios_with_virtu)
        mean_ratio = sum(r["ratio"] for r in ratios_with_virtu) / len(ratios_with_virtu)
    else:
        within_2x_rate = 0.0
        mean_ratio = None

    print()
    print(f"  Q1: Does frm_confidence=3 fire before bifurcation?")
    if fire_rate >= 0.8:
        print(f"      YES — fired in {fire_rate:.0%} of trials, mean {mean_lead:.0f} steps before bifurcation.")
        print(f"      This is consistent with FRM physics: λ→0 produces a detectable signature")
        print(f"      in Lambda, Omega, and Virtu simultaneously, well in advance of transition.")
    elif fire_rate >= 0.5:
        print(f"      PARTIAL — fired in {fire_rate:.0%} of trials, mean {mean_lead:.0f} steps before bifurcation.")
        print(f"      Detects majority of cases but not robustly reliable.")
    else:
        print(f"      WEAK — only {fire_rate:.0%} of trials reached frm_confidence=3.")

    print()
    print(f"  Q2: Is Virtu's TTB estimate accurate at that moment?")
    if ratios_with_virtu:
        if within_2x_rate >= 0.6 and mean_ratio is not None and 0.5 <= mean_ratio <= 2.0:
            print(f"      YES — {within_2x_rate:.0%} of estimates within 2× of true TTB.")
            print(f"      Mean ratio {mean_ratio:.2f} (1.0=perfect). Virtu's estimate is useful.")
            print(f"      This is evidence that Δt ≈ λ/|dλ/dt| tracks the true FRM trajectory.")
        elif within_2x_rate >= 0.4:
            print(f"      PARTIAL — {within_2x_rate:.0%} within 2×, mean ratio {mean_ratio:.2f}.")
            print(f"      Estimate is directionally correct but noisy.")
        else:
            print(f"      WEAK — only {within_2x_rate:.0%} within 2×, mean ratio {mean_ratio:.2f}.")
            print(f"      Virtu's estimate does not reliably track true TTB at conf3 moment.")
    else:
        print(f"      UNKNOWN — Virtu did not report TTB at the conf3 moment in any trial.")

    print()
    print(f"  Q3: Does performance indicate veracity of FRM physics?")
    if fire_rate >= 0.8 and ratios_with_virtu and within_2x_rate >= 0.6:
        print(f"      PARTIAL SUPPORT — but with an important caveat:")
        print(f"      The signals are *generated by* the FRM model (λ = 0.5*(1-t/200),")
        print(f"      ω = π/(2·τ_gen)). The detectors are *fitted to* those signals.")
        print(f"      Good performance on synthetic FRM signals confirms internal consistency,")
        print(f"      not that real-world systems follow FRM dynamics.")
        print(f"      Real validation requires: frm_confidence=3 on real data →")
        print(f"        observed transition within Virtu's predicted window.")
    else:
        print(f"      INCONCLUSIVE on synthetic data alone.")


if __name__ == "__main__":
    main()
