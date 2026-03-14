#!/usr/bin/env python3
"""
Monte Carlo Lambda Detector Benchmark
======================================

Law of large numbers: run hundreds of synthetic trajectories so metrics
converge to stable values with tight confidence intervals.

Generates two trajectory classes using the linearized damped oscillator
(Ornstein-Uhlenbeck with oscillation), which is the standard model in
the CSD literature (Scheffer et al. 2009, Dakos et al. 2012):

  dx/dt = μ·x - ω₀·y + σ·η_x(t)
  dy/dt = ω₀·x + μ·y + σ·η_y(t)

No cubic saturation term — this ensures CSD indicators (rising variance,
rising AC1, spectral narrowing) are present and proportional to the true
distance from bifurcation.  The nonlinear Hopf normal form with cubic
saturation (-r²·x terms) suppresses these indicators and makes forced/null
trajectories observationally indistinguishable.

  - FORCED: λ declining toward 0 (approaching bifurcation)
  - NULL: λ held constant (stable oscillation, no transition)

Outputs: TPR, FPR, precision, F1 with 95% Wilson score intervals.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fracttalix.suite import LambdaDetector


# ──────────────────────────────────────────────────────────
#  DATA GENERATION
# ──────────────────────────────────────────────────────────

def generate_ou_oscillator(
    n_steps: int,
    tau_gen: float,
    lam_start: float,
    lam_end: float,
    noise_std: float,
    amplitude: float,
    baseline: float = 10.0,
    seed: int = 42,
):
    """Linearized damped oscillator (OU with oscillation).

    Standard CSD benchmark model:
        dx/dt = μ·x - ω₀·y + σ·η_x(t)
        dy/dt = ω₀·x + μ·y + σ·η_y(t)

    where μ = -λ.  Variance scales as σ²/(2λ) — the textbook CSD signature.
    AC1 = exp(-λ·dt).  Both increase monotonically as λ → 0.

    Returns (values, lam_true) arrays.
    """
    rng = np.random.RandomState(seed)
    omega0 = math.pi / (2.0 * tau_gen)
    n_sub = 4
    sub_dt = 1.0 / n_sub
    sqrt_sub_dt = math.sqrt(sub_dt)

    x, y = 0.0, 0.0
    values = np.zeros(n_steps)
    lam_true = np.zeros(n_steps)

    for t in range(n_steps):
        frac = t / max(n_steps - 1, 1)
        lam_t = lam_start + (lam_end - lam_start) * frac
        mu = -lam_t
        lam_true[t] = lam_t

        for _ in range(n_sub):
            # Linear only — no cubic saturation
            dx = (mu * x - omega0 * y) * sub_dt + noise_std * rng.normal() * sqrt_sub_dt
            dy = (omega0 * x + mu * y) * sub_dt + noise_std * rng.normal() * sqrt_sub_dt
            x += dx
            y += dy

        values[t] = baseline + amplitude * x

    return values, lam_true


def generate_forced_ensemble(n_traj: int, base_seed: int = 1000):
    """Generate diverse forced (transitioning) trajectories.

    Parameters sampled to ensure CSD indicators are present:
      - tau_gen: 5-40 (covers fast and slow oscillations)
      - lam_start: 0.08-0.25 (healthy baseline damping)
      - lam_end: 0.005 (near bifurcation but not singular)
      - noise_std: 0.1-0.5 (moderate noise, keeps variance finite near bifurcation)
      - n_steps: 500-2000
    """
    rng = np.random.RandomState(base_seed)
    trajectories = []

    for i in range(n_traj):
        seed = base_seed + i
        tau_gen = rng.uniform(5.0, 40.0)
        lam_start = rng.uniform(0.08, 0.25)
        lam_end = 0.005  # near bifurcation, not at it (avoids divergence)
        noise_std = rng.uniform(0.1, 0.5)
        amplitude = rng.uniform(1.0, 5.0)
        n_steps = rng.randint(500, 2001)

        values, lam_true = generate_ou_oscillator(
            n_steps=n_steps, tau_gen=tau_gen,
            lam_start=lam_start, lam_end=lam_end,
            noise_std=noise_std, amplitude=amplitude,
            seed=seed,
        )
        trajectories.append({
            "values": values, "lam_true": lam_true,
            "params": {"tau_gen": tau_gen, "lam_start": lam_start,
                       "noise_std": noise_std, "n_steps": n_steps,
                       "amplitude": amplitude},
        })

    return trajectories


def generate_null_ensemble(n_traj: int, base_seed: int = 5000):
    """Generate diverse null (stable, no transition) trajectories.

    Same parameter distributions as forced, but λ stays constant.
    """
    rng = np.random.RandomState(base_seed)
    trajectories = []

    for i in range(n_traj):
        seed = base_seed + i
        tau_gen = rng.uniform(5.0, 40.0)
        lam_constant = rng.uniform(0.08, 0.25)  # constant λ
        noise_std = rng.uniform(0.1, 0.5)
        amplitude = rng.uniform(1.0, 5.0)
        n_steps = rng.randint(500, 2001)

        values, lam_true = generate_ou_oscillator(
            n_steps=n_steps, tau_gen=tau_gen,
            lam_start=lam_constant, lam_end=lam_constant,  # no decline
            noise_std=noise_std, amplitude=amplitude,
            seed=seed,
        )
        trajectories.append({
            "values": values, "lam_true": lam_true,
            "params": {"tau_gen": tau_gen, "lam_constant": lam_constant,
                       "noise_std": noise_std, "n_steps": n_steps,
                       "amplitude": amplitude},
        })

    return trajectories


# ──────────────────────────────────────────────────────────
#  DETECTOR RUNNER
# ──────────────────────────────────────────────────────────

def run_lambda_detector(values):
    """Run LambdaDetector on a trajectory. Returns (alerted: bool, n_alerts: int, first_alert: int|None)."""
    det = LambdaDetector(
        tau_gen=0.0, fit_window=128, fit_interval=8,
        lambda_window=20, lambda_warning=0.05,
    )
    alerts = []
    for t, val in enumerate(values):
        r = det.update(float(val))
        if r.is_alert:
            alerts.append(t)

    return len(alerts) > 0, len(alerts), alerts[0] if alerts else None


# ──────────────────────────────────────────────────────────
#  STATISTICS
# ──────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Wilson score interval for binomial proportion (better than normal approx)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return p_hat, lo, hi


def compute_metrics_with_ci(tp: int, fp: int, fn: int, tn: int):
    """Compute TPR, FPR, precision, F1 with 95% Wilson CIs."""
    tpr_val, tpr_lo, tpr_hi = wilson_ci(tp, tp + fn)
    fpr_val, fpr_lo, fpr_hi = wilson_ci(fp, fp + tn)
    prec_val, prec_lo, prec_hi = wilson_ci(tp, tp + fp)
    recall = tpr_val
    precision = prec_val
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "tpr": (tpr_val, tpr_lo, tpr_hi),
        "fpr": (fpr_val, fpr_lo, fpr_hi),
        "precision": (prec_val, prec_lo, prec_hi),
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ──────────────────────────────────────────────────────────
#  MAIN BENCHMARK
# ──────────────────────────────────────────────────────────

def run_monte_carlo(n_forced: int = 500, n_null: int = 500, verbose: bool = True):
    """Run Monte Carlo benchmark. Returns metrics dict."""
    if verbose:
        print(f"Generating {n_forced} forced + {n_null} null trajectories...")

    forced = generate_forced_ensemble(n_forced, base_seed=1000)
    null = generate_null_ensemble(n_null, base_seed=5000)

    if verbose:
        print(f"Running Lambda detector on {n_forced + n_null} trajectories...")

    # Forced (positive class)
    tp = 0
    forced_results = []
    for i, traj in enumerate(forced):
        alerted, n_alerts, first_alert = run_lambda_detector(traj["values"])
        if alerted:
            tp += 1
        forced_results.append({
            "alerted": alerted, "n_alerts": n_alerts,
            "first_alert": first_alert, "params": traj["params"],
        })
        if verbose and (i + 1) % 100 == 0:
            print(f"  Forced: {i + 1}/{n_forced} (TPR so far: {tp}/{i + 1})")

    fn = n_forced - tp

    # Null (negative class)
    fp = 0
    null_results = []
    for i, traj in enumerate(null):
        alerted, n_alerts, first_alert = run_lambda_detector(traj["values"])
        if alerted:
            fp += 1
        null_results.append({
            "alerted": alerted, "n_alerts": n_alerts,
            "first_alert": first_alert, "params": traj["params"],
        })
        if verbose and (i + 1) % 100 == 0:
            print(f"  Null: {i + 1}/{n_null} (FPR so far: {fp}/{i + 1})")

    tn = n_null - fp

    metrics = compute_metrics_with_ci(tp, fp, fn, tn)

    if verbose:
        print("\n" + "=" * 70)
        print("  MONTE CARLO RESULTS (Lambda Detector v2)")
        print(f"  N_forced={n_forced}  N_null={n_null}")
        print("=" * 70)
        print(f"\n  {'Metric':<12s}  {'Value':>8s}  {'95% CI':>16s}  {'Count':>8s}")
        print(f"  {'─' * 50}")
        for key in ["tpr", "fpr", "precision"]:
            val, lo, hi = metrics[key]
            if key == "tpr":
                count = f"{tp}/{tp + fn}"
            elif key == "fpr":
                count = f"{fp}/{fp + tn}"
            else:
                count = f"{tp}/{tp + fp}"
            print(f"  {key.upper():<12s}  {val:8.1%}  [{lo:6.1%}, {hi:6.1%}]  {count:>8s}")
        print(f"  {'F1':<12s}  {metrics['f1']:8.3f}")

        # ── Convergence check: show running metrics at 100, 200, 300, 400, 500 ──
        print(f"\n  Convergence (TPR at N):")
        for n_check in [50, 100, 200, 300, 400, n_forced]:
            if n_check > n_forced:
                continue
            sub_tp = sum(1 for r in forced_results[:n_check] if r["alerted"])
            val, lo, hi = wilson_ci(sub_tp, n_check)
            width = hi - lo
            print(f"    N={n_check:4d}: TPR={val:5.1%} CI=[{lo:5.1%}, {hi:5.1%}] width={width:5.1%}")

        print(f"\n  Convergence (FPR at N):")
        for n_check in [50, 100, 200, 300, 400, n_null]:
            if n_check > n_null:
                continue
            sub_fp = sum(1 for r in null_results[:n_check] if r["alerted"])
            val, lo, hi = wilson_ci(sub_fp, n_check)
            width = hi - lo
            print(f"    N={n_check:4d}: FPR={val:5.1%} CI=[{lo:5.1%}, {hi:5.1%}] width={width:5.1%}")

        # ── Failure analysis: what parameters cause misses/FPs? ──
        missed = [r for r in forced_results if not r["alerted"]]
        false_alarms = [r for r in null_results if r["alerted"]]

        if missed:
            print(f"\n  Missed detections ({len(missed)}/{n_forced}):")
            noise_vals = [r["params"]["noise_std"] for r in missed]
            lam_vals = [r["params"]["lam_start"] for r in missed]
            print(f"    noise_std: mean={np.mean(noise_vals):.2f} "
                  f"range=[{np.min(noise_vals):.2f}, {np.max(noise_vals):.2f}]")
            print(f"    lam_start: mean={np.mean(lam_vals):.3f} "
                  f"range=[{np.min(lam_vals):.3f}, {np.max(lam_vals):.3f}]")

        if false_alarms:
            print(f"\n  False alarms ({len(false_alarms)}/{n_null}):")
            noise_vals = [r["params"]["noise_std"] for r in false_alarms]
            lam_vals = [r["params"]["lam_constant"] for r in false_alarms]
            print(f"    noise_std: mean={np.mean(noise_vals):.2f} "
                  f"range=[{np.min(noise_vals):.2f}, {np.max(noise_vals):.2f}]")
            print(f"    lam_constant: mean={np.mean(lam_vals):.3f} "
                  f"range=[{np.min(lam_vals):.3f}, {np.max(lam_vals):.3f}]")

        print(f"\n  {'─' * 50}")

    return {
        "metrics": metrics,
        "forced_results": forced_results,
        "null_results": null_results,
    }


if __name__ == "__main__":
    n = 500
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    result = run_monte_carlo(n_forced=n, n_null=n)
    metrics = result["metrics"]

    # Exit code: pass if TPR > 50% and FPR < 20%
    tpr = metrics["tpr"][0]
    fpr = metrics["fpr"][0]
    if tpr > 0.50 and fpr < 0.20:
        print(f"\n  GATE: PASS (TPR={tpr:.1%} > 50%, FPR={fpr:.1%} < 20%)")
        sys.exit(0)
    else:
        print(f"\n  GATE: FAIL (TPR={tpr:.1%}, FPR={fpr:.1%})")
        sys.exit(1)
