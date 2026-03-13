#!/usr/bin/env python3
"""
FRM Confidence=3 Validation Benchmark
======================================

The question this answers:

  Does frm_confidence=3 on real-world data reliably precede the
  transitions the FRM predicts, at the timescales Virtu estimates?

frm_confidence is defined as the count of FRM detectors that
*confirm* an approaching transition:

  - Lambda: is_alert (score >= 0.5, λ declining toward 0)
  - Omega:  in_scope AND NOT alert (ω matches physics prediction —
            structural integrity holds during the approach)
  - Virtu:  score > 0 AND decision_quality > 0.2 (MONITOR or higher)

frm_confidence=3 means: Lambda sees λ declining, Omega confirms the
FRM frequency prediction still holds, and Virtu's decision theory
says the window is relevant.  This is the strongest possible signal
from the FRM framework.

Metrics:
  1. Reliability:  How often does confidence=3 precede the actual transition?
  2. Lead time:    How many steps before the transition does confidence=3 first appear?
  3. Virtu accuracy: How close is Virtu's Δt estimate to the actual time remaining?
  4. False positive rate: How often does confidence=3 fire on stable data?

Data: Stochastic Hopf normal form with known transition point (λ → 0).
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fracttalix.suite import LambdaDetector, OmegaDetector, VirtuDetector, ScopeStatus


# ──────────────────────────────────────────────────────────
#  DATA GENERATORS
# ──────────────────────────────────────────────────────────

def generate_approaching_bifurcation(
    n_steps=2000,
    tau_gen=20.0,
    lam_start=0.15,
    lam_end=0.0,
    noise_std=0.08,
    amplitude=3.0,
    baseline=10.0,
    seed=42,
):
    """Stochastic Hopf normal form approaching bifurcation.

    Uses Euler-Maruyama with dt=0.1 sub-stepping for numerical stability.

    Returns (values, lam_true, transition_step) where transition_step
    is the first step where true λ < 0.05 (the Lambda warning threshold).
    """
    np.random.seed(seed)
    omega0 = math.pi / (2.0 * tau_gen)
    dt = 0.1
    sub_steps = 10
    x, y = 0.01, 0.01
    values = np.zeros(n_steps)
    lam_true = np.zeros(n_steps)

    for t in range(n_steps):
        frac = t / (n_steps - 1) if n_steps > 1 else 0
        lam_t = lam_start + (lam_end - lam_start) * frac
        mu = -lam_t
        lam_true[t] = lam_t

        for _ in range(sub_steps):
            r_sq = x * x + y * y
            dx = (mu * x - omega0 * y - r_sq * x) * dt + noise_std * np.random.normal() * math.sqrt(dt)
            dy = (omega0 * x + mu * y - r_sq * y) * dt + noise_std * np.random.normal() * math.sqrt(dt)
            x += dx
            y += dy
        values[t] = baseline + amplitude * x

    # Transition = first step where λ < warning threshold
    warning = 0.05
    transition_step = n_steps
    for t in range(n_steps):
        if lam_true[t] <= warning:
            transition_step = t
            break

    return values, lam_true, transition_step


def generate_deterministic_bifurcation(n_steps=600, tau_gen=20.0,
                                        lam_start=0.15, lam_end=0.0,
                                        noise_std=0.2, amplitude=3.0, seed=42):
    """Deterministic FRM-form data with declining λ.

    This IS the data type the Lambda detector was designed for:
    f(t) = B + A·exp(-λ(t)·t_local)·cos(ω·t) + noise

    where λ(t) declines linearly over time. Each "episode" of the
    oscillation has a true FRM envelope with a slightly smaller λ.
    """
    np.random.seed(seed)
    omega = math.pi / (2.0 * tau_gen)
    values = np.zeros(n_steps)
    lam_true = np.zeros(n_steps)
    episode_len = int(4 * tau_gen)

    for t in range(n_steps):
        frac = t / (n_steps - 1) if n_steps > 1 else 0
        lam_t = lam_start + (lam_end - lam_start) * frac
        lam_true[t] = lam_t
        local_t = t % episode_len
        values[t] = (10.0 + amplitude * math.exp(-lam_t * local_t)
                     * math.cos(omega * t)
                     + np.random.normal(0, noise_std))

    warning = 0.05
    transition_step = n_steps
    for t in range(n_steps):
        if lam_true[t] <= warning:
            transition_step = t
            break

    return values, lam_true, transition_step


def generate_stable_oscillation(n=1000, tau_gen=20.0, noise_std=0.3,
                                 amplitude=3.0, seed=42):
    """Sustained oscillation — no transition. For FPR testing."""
    np.random.seed(seed)
    omega = math.pi / (2.0 * tau_gen)
    t = np.arange(n, dtype=float)
    values = 10.0 + amplitude * np.cos(omega * t) + np.random.normal(0, noise_std, n)
    return values


def generate_white_noise(n=500, seed=42):
    """White noise — no oscillation, no transition."""
    np.random.seed(seed)
    return np.random.normal(0, 1, n)


# ──────────────────────────────────────────────────────────
#  FRM CONFIDENCE COMPUTATION
# ──────────────────────────────────────────────────────────

def compute_frm_confidence(lam_result, omega_result, virtu_result, virtu_det):
    """Compute frm_confidence level (0-3).

    Level 3 = all three detectors confirm an approaching transition:
      - Lambda alerting (λ declining toward 0)
      - Omega in-scope and not alerting (FRM physics holds)
      - Virtu active (decision quality > 0.2, MONITOR or higher)
    """
    confidence = 0

    # Lambda: alerting means λ is declining and score >= threshold
    if lam_result.is_alert:
        confidence += 1

    # Omega: in_scope AND NOT alerting means ω matches prediction.
    # If Omega is alerting, frequency is deviating — physics breaking down.
    # If Omega is out_of_scope, can't confirm physics.
    if omega_result.in_scope and not omega_result.is_alert:
        confidence += 1

    # Virtu: decision quality above MONITOR threshold
    if virtu_det.decision_quality > 0.2:
        confidence += 1

    return confidence


# ──────────────────────────────────────────────────────────
#  BENCHMARK RUNNER
# ──────────────────────────────────────────────────────────

def run_bifurcation_scenario(name, n_steps, tau_gen, lam_start, lam_end,
                              noise_std, amplitude, seed):
    """Run all 3 FRM detectors on one bifurcation scenario."""

    values, lam_true, true_transition = generate_approaching_bifurcation(
        n_steps=n_steps, tau_gen=tau_gen, lam_start=lam_start,
        lam_end=lam_end, noise_std=noise_std, amplitude=amplitude, seed=seed,
    )

    # Create detectors
    lam_det = LambdaDetector(
        tau_gen=tau_gen, fit_window=128, fit_interval=8,
        lambda_window=20, lambda_warning=0.05,
    )
    omega_det = OmegaDetector(
        tau_gen=tau_gen, fit_window=128, deviation_threshold=0.15,
    )
    virtu_det = VirtuDetector(
        lambda_detector=lam_det, decision_horizon=100.0,
    )

    # Run
    confidence_history = []
    virtu_dt_history = []
    first_conf3_step = None
    first_conf2_step = None

    for t, val in enumerate(values):
        v = float(val)
        lam_r = lam_det.update(v)
        omega_r = omega_det.update(v)
        virtu_r = virtu_det.update(v)

        conf = compute_frm_confidence(lam_r, omega_r, virtu_r, virtu_det)
        confidence_history.append(conf)

        if conf >= 3 and first_conf3_step is None:
            first_conf3_step = t
        if conf >= 2 and first_conf2_step is None:
            first_conf2_step = t

        # Track Virtu's time estimate when confidence is high
        if conf >= 2 and lam_det.time_to_transition is not None:
            virtu_dt_history.append({
                "step": t,
                "virtu_dt": lam_det.time_to_transition,
                "actual_remaining": true_transition - t,
                "true_lambda": lam_true[t],
                "fitted_lambda": lam_det.current_lambda,
                "decision_quality": virtu_det.decision_quality,
                "confidence": conf,
            })

    # Analyze
    result = {
        "name": name,
        "n_steps": n_steps,
        "true_transition": true_transition,
        "first_conf3": first_conf3_step,
        "first_conf2": first_conf2_step,
        "conf3_lead": (true_transition - first_conf3_step) if first_conf3_step is not None else None,
        "conf2_lead": (true_transition - first_conf2_step) if first_conf2_step is not None else None,
        "conf3_precedes": first_conf3_step is not None and first_conf3_step < true_transition,
        "virtu_estimates": virtu_dt_history,
        "peak_quality": virtu_det.peak_quality,
        "final_lambda": lam_det.current_lambda,
        "final_r2": lam_det.r_squared,
        "final_scope": lam_det.scope_status,
    }

    return result


def run_fpr_scenario(name, data, tau_gen):
    """Run on stable/noise data to check false positive rate."""
    lam_det = LambdaDetector(
        tau_gen=tau_gen, fit_window=128, fit_interval=8,
        lambda_window=20, lambda_warning=0.05,
    )
    omega_det = OmegaDetector(
        tau_gen=tau_gen, fit_window=128, deviation_threshold=0.15,
    )
    virtu_det = VirtuDetector(
        lambda_detector=lam_det, decision_horizon=100.0,
    )

    conf3_count = 0
    conf2_count = 0
    total = 0

    for val in data:
        v = float(val)
        lam_r = lam_det.update(v)
        omega_r = omega_det.update(v)
        virtu_r = virtu_det.update(v)
        conf = compute_frm_confidence(lam_r, omega_r, virtu_r, virtu_det)
        total += 1
        if conf >= 3:
            conf3_count += 1
        if conf >= 2:
            conf2_count += 1

    return {
        "name": name,
        "total": total,
        "conf3_count": conf3_count,
        "conf2_count": conf2_count,
        "conf3_rate": conf3_count / max(1, total),
        "conf2_rate": conf2_count / max(1, total),
    }


# ──────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  FRM Confidence=3 Validation Benchmark")
    print("  Does frm_confidence=3 reliably precede transitions")
    print("  at the timescales Virtu estimates?")
    print("=" * 70)

    # ════════════════════════════════════════════
    #  PART 1: BIFURCATION SCENARIOS
    # ════════════════════════════════════════════

    seeds = [42, 123, 789]
    all_results = []
    conf3_precedes_count = 0
    conf3_total = 0
    virtu_errors = []

    # ── Part 1a: Deterministic FRM-form data (what the detector was designed for) ──
    print(f"\n  Part 1a: Deterministic FRM-form data")
    print(f"  (Does the detector work on its own data type?)")
    det_scenarios = [
        ("Det. slow (τ=20, σ=0.2)", 600, 20.0, 0.15, 0.0, 0.2, 3.0),
        ("Det. fast (τ=10, σ=0.2)", 400, 10.0, 0.20, 0.0, 0.2, 3.0),
    ]

    for det_sc in det_scenarios:
        dname, dn, dtau, dls, dle, dns, da = det_sc
        print(f"\n  Scenario: {dname}")
        for seed in seeds:
            print(f"    Running seed={seed}...", end=" ", flush=True)
            vals, lt, ts = generate_deterministic_bifurcation(
                n_steps=dn, tau_gen=dtau, lam_start=dls, lam_end=dle,
                noise_std=dns, amplitude=da, seed=seed,
            )
            result = run_bifurcation_scenario(
                f"{dname} (seed={seed})",
                dn, dtau, dls, dle, dns, da, seed,
            )
            # Override with deterministic data
            lam_det = LambdaDetector(
                tau_gen=dtau, fit_window=128, fit_interval=8,
                lambda_window=20, lambda_warning=0.05,
            )
            omega_det = OmegaDetector(
                tau_gen=dtau, fit_window=128, deviation_threshold=0.15,
            )
            virtu_det = VirtuDetector(lambda_detector=lam_det, decision_horizon=100.0)
            first_conf3 = None
            for t, val in enumerate(vals):
                v = float(val)
                lam_r = lam_det.update(v)
                omega_r = omega_det.update(v)
                virtu_r = virtu_det.update(v)
                conf = compute_frm_confidence(lam_r, omega_r, virtu_r, virtu_det)
                if conf >= 3 and first_conf3 is None:
                    first_conf3 = t

            precedes = first_conf3 is not None and first_conf3 < ts
            lead = ts - first_conf3 if precedes else None
            print(f"scope={lam_det.scope_status} R²={lam_det.r_squared:.3f} "
                  f"λ={lam_det.current_lambda} peak_q={virtu_det.peak_quality:.3f} "
                  f"conf3@{first_conf3} trans@{ts} "
                  f"{'PRECEDES' if precedes else 'MISS'} "
                  f"{'lead='+str(lead) if lead else ''}", flush=True)

            if precedes:
                conf3_precedes_count += 1
                all_results.append({"conf3_precedes": True, "conf3_lead": lead,
                                   "first_conf3": first_conf3, "first_conf2": None,
                                   "virtu_estimates": [], "peak_quality": virtu_det.peak_quality,
                                   "final_lambda": lam_det.current_lambda,
                                   "final_r2": lam_det.r_squared, "final_scope": lam_det.scope_status,
                                   "name": f"{dname} (seed={seed})"})
            else:
                all_results.append({"conf3_precedes": False, "conf3_lead": None,
                                   "first_conf3": first_conf3, "first_conf2": None,
                                   "virtu_estimates": [], "peak_quality": virtu_det.peak_quality,
                                   "final_lambda": lam_det.current_lambda,
                                   "final_r2": lam_det.r_squared, "final_scope": lam_det.scope_status,
                                   "name": f"{dname} (seed={seed})"})
            conf3_total += 1

    # ── Part 1b: Stochastic Hopf normal form (generic bifurcation data) ──
    print(f"\n  Part 1b: Stochastic Hopf normal form")
    print(f"  (Does the detector work on generic bifurcation data?)")

    scenarios = [
        # (name, n_steps, tau_gen, lam_start, lam_end, noise_std, amplitude)
        ("Stoch slow (τ=20, σ=0.08)", 800, 20.0, 0.15, 0.0, 0.08, 3.0),
        ("Stoch fast (τ=10, σ=0.08)", 600, 10.0, 0.20, 0.0, 0.08, 3.0),
    ]

    print(f"\n{'─'*70}")
    print(f"  PART 1b continued: Stochastic Hopf normal form")
    print(f"  {len(scenarios)} scenarios × {len(seeds)} seeds = {len(scenarios)*len(seeds)} trials")
    print(f"{'─'*70}")

    for scenario in scenarios:
        name, n_steps, tau_gen, lam_start, lam_end, noise_std, amplitude = scenario

        print(f"\n  Scenario: {name}")
        print(f"  λ: {lam_start} → {lam_end} over {n_steps} steps")

        scenario_precedes = 0
        scenario_leads = []
        scenario_virtu_errors = []

        for seed in seeds:
            import sys as _sys
            print(f"    Running seed={seed}...", end=" ", flush=True)
            result = run_bifurcation_scenario(
                f"{name} (seed={seed})",
                n_steps, tau_gen, lam_start, lam_end, noise_std, amplitude, seed,
            )
            all_results.append(result)
            conf3_total += 1

            # Diagnostic: what DID fire?
            max_conf = max(result.get("confidence_max", 0),
                          3 if result["first_conf3"] is not None else
                          2 if result["first_conf2"] is not None else 0)
            print(f"max_conf={max_conf} scope={result['final_scope']} "
                  f"R²={result['final_r2']:.3f} λ={result.get('final_lambda', 'N/A')} "
                  f"peak_q={result['peak_quality']:.3f}", flush=True)

            if result["conf3_precedes"]:
                conf3_precedes_count += 1
                scenario_precedes += 1
                scenario_leads.append(result["conf3_lead"])

            # Virtu timing accuracy: compare estimated Δt to actual remaining
            for est in result["virtu_estimates"]:
                if est["confidence"] >= 3 and est["actual_remaining"] > 0:
                    error = est["virtu_dt"] - est["actual_remaining"]
                    rel_error = error / est["actual_remaining"] if est["actual_remaining"] > 0 else 0
                    virtu_errors.append(rel_error)
                    scenario_virtu_errors.append(rel_error)

        precede_rate = scenario_precedes / len(seeds)
        mean_lead = np.mean(scenario_leads) if scenario_leads else 0

        print(f"    Precedes transition: {scenario_precedes}/{len(seeds)} ({precede_rate:.0%})")
        if scenario_leads:
            print(f"    Mean lead time:      {mean_lead:.0f} steps")
            print(f"    Lead range:          [{min(scenario_leads)}, {max(scenario_leads)}]")
        if scenario_virtu_errors:
            arr = np.array(scenario_virtu_errors)
            print(f"    Virtu Δt rel error:   median={np.median(arr):.2f}, "
                  f"MAD={np.median(np.abs(arr - np.median(arr))):.2f}")

    # ════════════════════════════════════════════
    #  PART 2: FALSE POSITIVE RATE
    # ════════════════════════════════════════════

    print(f"\n{'─'*70}")
    print(f"  PART 2: False positive rate on stable / noise data")
    print(f"{'─'*70}")

    fpr_results = []
    for seed in seeds:
        # Stable oscillation (no transition should fire)
        stable = generate_stable_oscillation(n=1000, tau_gen=20.0, seed=seed)
        fpr_results.append(run_fpr_scenario(
            f"Stable oscillation (seed={seed})", stable, tau_gen=20.0,
        ))

        # White noise
        noise = generate_white_noise(n=500, seed=seed)
        fpr_results.append(run_fpr_scenario(
            f"White noise (seed={seed})", noise, tau_gen=20.0,
        ))

    stable_fpr = [r for r in fpr_results if "Stable" in r["name"]]
    noise_fpr = [r for r in fpr_results if "noise" in r["name"]]

    stable_conf3 = sum(r["conf3_count"] for r in stable_fpr)
    stable_total = sum(r["total"] for r in stable_fpr)
    noise_conf3 = sum(r["conf3_count"] for r in noise_fpr)
    noise_total = sum(r["total"] for r in noise_fpr)

    print(f"\n  Stable oscillation:")
    print(f"    conf=3 steps: {stable_conf3}/{stable_total} "
          f"({stable_conf3/max(1,stable_total):.2%})")
    print(f"  White noise:")
    print(f"    conf=3 steps: {noise_conf3}/{noise_total} "
          f"({noise_conf3/max(1,noise_total):.2%})")

    # ════════════════════════════════════════════
    #  PART 3: VIRTU TIMESCALE ACCURACY
    # ════════════════════════════════════════════

    print(f"\n{'─'*70}")
    print(f"  PART 3: Virtu timescale accuracy (at frm_confidence >= 3)")
    print(f"{'─'*70}")

    if virtu_errors:
        ve = np.array(virtu_errors)
        print(f"\n  Total conf=3 estimates with known Δt: {len(ve)}")
        print(f"  Relative error (Virtu Δt vs actual):")
        print(f"    Median:  {np.median(ve):+.2f}")
        print(f"    Mean:    {np.mean(ve):+.2f}")
        print(f"    Std:     {np.std(ve):.2f}")
        print(f"    MAD:     {np.median(np.abs(ve - np.median(ve))):.2f}")
        print(f"    |error| < 0.5 (within 50%): {(np.abs(ve) < 0.5).sum()}/{len(ve)} "
              f"({(np.abs(ve) < 0.5).mean():.0%})")
        print(f"    |error| < 1.0 (within 100%): {(np.abs(ve) < 1.0).sum()}/{len(ve)} "
              f"({(np.abs(ve) < 1.0).mean():.0%})")
        # Direction: positive = overestimates time remaining (conservative)
        overestimates = (ve > 0).sum()
        print(f"    Overestimates (conservative): {overestimates}/{len(ve)} "
              f"({overestimates/len(ve):.0%})")
    else:
        print(f"\n  No conf=3 Virtu estimates were produced.")

    # ════════════════════════════════════════════
    #  VERDICT
    # ════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    reliability = conf3_precedes_count / max(1, conf3_total)
    stable_fpr_rate = stable_conf3 / max(1, stable_total)
    noise_fpr_rate = noise_conf3 / max(1, noise_total)

    criteria = []

    # C1: frm_confidence=3 precedes transitions > 60% of time
    if reliability >= 0.6:
        criteria.append(("conf=3 precedes transition (>60%)", "PASS",
                        f"{reliability:.0%} ({conf3_precedes_count}/{conf3_total})"))
    else:
        criteria.append(("conf=3 precedes transition (>60%)", "FAIL",
                        f"{reliability:.0%} ({conf3_precedes_count}/{conf3_total})"))

    # C2: conf=3 FPR < 5% on stable data
    if stable_fpr_rate < 0.05:
        criteria.append(("conf=3 FPR < 5% on stable", "PASS",
                        f"{stable_fpr_rate:.2%}"))
    else:
        criteria.append(("conf=3 FPR < 5% on stable", "FAIL",
                        f"{stable_fpr_rate:.2%}"))

    # C3: conf=3 FPR = 0% on white noise
    if noise_fpr_rate == 0:
        criteria.append(("conf=3 FPR = 0% on noise", "PASS", "0 false positives"))
    else:
        criteria.append(("conf=3 FPR = 0% on noise", "WARN",
                        f"{noise_fpr_rate:.2%}"))

    # C4: Virtu Δt within factor of 2 for >50% of conf=3 estimates
    if virtu_errors:
        ve = np.array(virtu_errors)
        within_2x = (np.abs(ve) < 1.0).mean()
        if within_2x >= 0.5:
            criteria.append(("Virtu Δt within 2× (>50%)", "PASS",
                            f"{within_2x:.0%}"))
        else:
            criteria.append(("Virtu Δt within 2× (>50%)", "FAIL",
                            f"{within_2x:.0%}"))
    else:
        criteria.append(("Virtu Δt accuracy", "SKIP", "No conf=3 estimates"))

    # C5: Mean lead time > 50 steps
    leads = [r["conf3_lead"] for r in all_results if r["conf3_lead"] is not None and r["conf3_lead"] > 0]
    if leads:
        mean_lead = np.mean(leads)
        if mean_lead > 50:
            criteria.append(("Mean lead > 50 steps", "PASS", f"{mean_lead:.0f} steps"))
        else:
            criteria.append(("Mean lead > 50 steps", "FAIL", f"{mean_lead:.0f} steps"))
    else:
        criteria.append(("Mean lead > 50 steps", "SKIP", "No conf=3 detections"))

    print()
    gate_pass = True
    for name, status, detail in criteria:
        marker = "+" if status == "PASS" else ("-" if status == "FAIL" else "~")
        print(f"  [{marker}] {status}: {name}")
        print(f"       {detail}")
        if status == "FAIL":
            gate_pass = False

    print(f"\n  {'─'*60}")
    if gate_pass:
        print("  GATE: PASS")
        print("  frm_confidence=3 reliably precedes transitions and")
        print("  Virtu timescale estimates are in the right ballpark.")
    else:
        print("  GATE: FAIL")
        print("  frm_confidence=3 does NOT yet reliably predict transitions.")
        print("  See individual criteria above for what needs improvement.")
    print(f"  {'─'*60}")

    return gate_pass


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
