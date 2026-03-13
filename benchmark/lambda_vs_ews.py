#!/usr/bin/env python3
"""
Lambda Detector vs Generic EWS — Head-to-Head Benchmark
========================================================

The test that justifies the project.

Generates synthetic data where a damped oscillation approaches Hopf
bifurcation (λ declining linearly toward 0), then compares detection
timing between:

  1. FRM Lambda detector — parametric, fits f(t)=B+A·exp(-λt)·cos(ωt+φ)
  2. Generic EWS (Scheffer et al.) — variance + lag-1 autocorrelation

The Lambda detector should fire EARLIER because it directly estimates
the parameter that goes to zero, rather than watching statistical
shadows (rising variance, rising AC1) that are second-order effects.

Multiple scenarios tested at different noise levels and decline rates.
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fracttalix import SentinelConfig, SentinelDetector


# ──────────────────────────────────────────────────────────
#  GENERIC EWS BASELINE (Scheffer et al. 2009)
# ──────────────────────────────────────────────────────────

class GenericEWS:
    """Baseline Early Warning Signal detector.

    Implements the standard Scheffer et al. approach:
    - Track rolling variance (σ²)
    - Track rolling lag-1 autocorrelation (AC1)
    - Fire alert when both exceed thresholds

    This is what every other EWS system does. No parametric model,
    no physics, just statistics.
    """

    def __init__(self, window=128, var_threshold=1.5, ac1_threshold=0.7):
        self.window = window
        self.var_threshold = var_threshold
        self.ac1_threshold = ac1_threshold
        self._buffer = []
        self._baseline_var = None
        self._baseline_ac1 = None
        self._warmup = window * 2  # use first 2 windows as baseline

    def update(self, value):
        """Process one observation. Returns (alert, var_ratio, ac1)."""
        self._buffer.append(value)

        if len(self._buffer) < self.window:
            return False, None, None

        # Current window
        w = np.array(self._buffer[-self.window:])

        # Variance
        current_var = np.var(w)

        # Lag-1 autocorrelation
        if len(w) > 1:
            centered = w - np.mean(w)
            c0 = np.sum(centered ** 2)
            c1 = np.sum(centered[:-1] * centered[1:])
            ac1 = c1 / c0 if abs(c0) > 1e-12 else 0.0
        else:
            ac1 = 0.0

        # Set baseline from warmup period
        if len(self._buffer) == self._warmup:
            self._baseline_var = current_var
            self._baseline_ac1 = ac1

        if self._baseline_var is None:
            return False, None, None

        # Ratio of current variance to baseline
        var_ratio = (
            current_var / self._baseline_var
            if self._baseline_var > 1e-12
            else 1.0
        )

        # Alert: variance increased significantly AND AC1 is high
        alert = var_ratio > self.var_threshold and ac1 > self.ac1_threshold

        return alert, var_ratio, ac1


# ──────────────────────────────────────────────────────────
#  SYNTHETIC BIFURCATION APPROACH DATA
# ──────────────────────────────────────────────────────────

def generate_approaching_bifurcation(
    n_steps=2000,
    tau_gen=20.0,
    lam_start=0.15,
    lam_end=0.0,
    noise_std=0.5,
    amplitude=3.0,
    baseline=10.0,
    seed=42,
):
    """Generate noise-driven oscillation approaching Hopf bifurcation.

    Uses the stochastic Hopf normal form in Cartesian coordinates:
        dx/dt = μ·x - ω₀·y - (x²+y²)·x + σ·η_x(t)
        dy/dt = ω₀·x + μ·y - (x²+y²)·y + σ·η_y(t)

    where μ < 0 gives a damped oscillation (stable focus) and μ → 0
    approaches the bifurcation. The relationship to FRM: λ ≈ |μ|.

    As μ → 0:
    - Oscillation amplitude grows (less damping of noise excitations)
    - Variance increases proportional to 1/|μ|
    - Lag-1 autocorrelation increases toward 1
    - These are the genuine critical slowing down indicators

    The observed signal is baseline + amplitude * x(t) + noise.
    """
    np.random.seed(seed)
    omega0 = math.pi / (2.0 * tau_gen)
    dt = 1.0  # unit time step

    x = 0.01  # initial conditions near origin
    y = 0.01

    values = np.zeros(n_steps)
    lam_true = np.zeros(n_steps)

    for t in range(n_steps):
        # μ declines linearly: μ = -lam_start → -lam_end
        frac = t / (n_steps - 1) if n_steps > 1 else 0
        lam_t = lam_start + (lam_end - lam_start) * frac
        mu = -lam_t  # μ = -λ (negative = stable)
        lam_true[t] = lam_t

        # Stochastic Hopf normal form (Euler-Maruyama)
        r_sq = x * x + y * y
        dx = (mu * x - omega0 * y - r_sq * x) * dt + noise_std * np.random.normal() * math.sqrt(dt)
        dy = (omega0 * x + mu * y - r_sq * y) * dt + noise_std * np.random.normal() * math.sqrt(dt)
        x += dx
        y += dy
        # Clamp to prevent numerical blowup near/past bifurcation
        clamp = 10.0
        x = max(-clamp, min(clamp, x))
        y = max(-clamp, min(clamp, y))

        # Observed signal
        values[t] = baseline + amplitude * x

    # Find step where λ crosses warning threshold
    warning_threshold = 0.05
    transition_step = n_steps
    for t in range(n_steps):
        if lam_true[t] <= warning_threshold:
            transition_step = t
            break

    return values, lam_true, transition_step


# ──────────────────────────────────────────────────────────
#  BENCHMARK RUNNER
# ──────────────────────────────────────────────────────────

def run_scenario(name, n_steps, tau_gen, lam_start, lam_end,
                 noise_std, amplitude, seed=42):
    """Run both detectors on the same data and compare timing."""
    print(f"\n{'─'*60}")
    print(f"  Scenario: {name}")
    print(f"  λ: {lam_start} → {lam_end} over {n_steps} steps")
    print(f"  τ_gen={tau_gen}, noise_std={noise_std}, A={amplitude}")
    print(f"{'─'*60}")

    values, lam_true, true_transition = generate_approaching_bifurcation(
        n_steps=n_steps,
        tau_gen=tau_gen,
        lam_start=lam_start,
        lam_end=lam_end,
        noise_std=noise_std,
        amplitude=amplitude,
        seed=seed,
    )

    # ── Lambda Detector ──
    cfg = SentinelConfig(
        enable_hopf_detector=True,
        hopf_tau_gen=tau_gen,
        hopf_fit_window=128,
        hopf_fit_interval=4,
        hopf_lambda_window=20,
        hopf_lambda_warning=0.05,
        hopf_t_decision=10.0,
        hopf_r_squared_min=0.5,
        warmup_periods=5,
    )
    det = SentinelDetector(config=cfg)

    lambda_first_alert = None
    lambda_alert_type = None
    lambda_alerts = []

    for t, val in enumerate(values):
        result = det.update_and_check(float(val))
        hopf = result.get_hopf_status()
        if hopf["alert"] and lambda_first_alert is None:
            lambda_first_alert = t
            lambda_alert_type = hopf["alert_type"]
        if hopf["alert"]:
            lambda_alerts.append(t)

    # ── Generic EWS ──
    ews = GenericEWS(window=128, var_threshold=1.5, ac1_threshold=0.7)

    ews_first_alert = None
    ews_alerts = []

    for t, val in enumerate(values):
        alert, var_ratio, ac1 = ews.update(float(val))
        if alert and ews_first_alert is None:
            ews_first_alert = t
        if alert:
            ews_alerts.append(t)

    # ── Results ──
    print(f"\n  True λ < 0.05 at step:  {true_transition}")
    print(f"\n  Lambda Detector:")
    print(f"    First alert at step:  {lambda_first_alert or 'NEVER'}")
    if lambda_first_alert is not None:
        lead = true_transition - lambda_first_alert
        print(f"    Lead time:            {lead} steps "
              f"({'before' if lead > 0 else 'after'} transition)")
        print(f"    Alert type:           {lambda_alert_type}")
        print(f"    True λ at alert:      {lam_true[lambda_first_alert]:.4f}")
    print(f"    Total alerts:         {len(lambda_alerts)}")

    print(f"\n  Generic EWS (Scheffer et al.):")
    print(f"    First alert at step:  {ews_first_alert or 'NEVER'}")
    if ews_first_alert is not None:
        lead = true_transition - ews_first_alert
        print(f"    Lead time:            {lead} steps "
              f"({'before' if lead > 0 else 'after'} transition)")
        print(f"    True λ at alert:      {lam_true[ews_first_alert]:.4f}")
    print(f"    Total alerts:         {len(ews_alerts)}")

    # ── Verdict ──
    if lambda_first_alert is not None and ews_first_alert is not None:
        if lambda_first_alert < ews_first_alert:
            advantage = ews_first_alert - lambda_first_alert
            print(f"\n  VERDICT: Lambda wins by {advantage} steps")
            return "LAMBDA_WINS", advantage
        elif ews_first_alert < lambda_first_alert:
            advantage = lambda_first_alert - ews_first_alert
            print(f"\n  VERDICT: EWS wins by {advantage} steps")
            return "EWS_WINS", advantage
        else:
            print(f"\n  VERDICT: Tie")
            return "TIE", 0
    elif lambda_first_alert is not None:
        print(f"\n  VERDICT: Lambda detects, EWS misses entirely")
        return "LAMBDA_ONLY", 0
    elif ews_first_alert is not None:
        print(f"\n  VERDICT: EWS detects, Lambda misses entirely")
        return "EWS_ONLY", 0
    else:
        print(f"\n  VERDICT: Neither detects")
        return "NEITHER", 0


def main():
    print("=" * 60)
    print("  Lambda Detector vs Generic EWS — Head-to-Head Benchmark")
    print("  'The test that justifies the project'")
    print("=" * 60)

    results = {}

    # Scenario 1: Clean, slow decline — ideal conditions
    results["clean_slow"] = run_scenario(
        "Clean, slow decline",
        n_steps=800, tau_gen=20.0,
        lam_start=0.15, lam_end=0.0,
        noise_std=0.3, amplitude=3.0,
    )

    # Scenario 2: Noisy, slow decline — realistic
    results["noisy_slow"] = run_scenario(
        "Noisy, slow decline",
        n_steps=800, tau_gen=20.0,
        lam_start=0.15, lam_end=0.0,
        noise_std=1.0, amplitude=3.0,
    )

    # Scenario 3: Clean, fast decline — urgent transition
    results["clean_fast"] = run_scenario(
        "Clean, fast decline",
        n_steps=500, tau_gen=10.0,
        lam_start=0.20, lam_end=0.0,
        noise_std=0.3, amplitude=3.0,
    )

    # Scenario 4: Noisy, fast decline — hardest case
    results["noisy_fast"] = run_scenario(
        "Noisy, fast decline",
        n_steps=500, tau_gen=10.0,
        lam_start=0.20, lam_end=0.0,
        noise_std=1.5, amplitude=3.0,
    )

    # Scenario 5: Very slow decline, large oscillation
    results["large_osc"] = run_scenario(
        "Large oscillation, slow decline",
        n_steps=1000, tau_gen=30.0,
        lam_start=0.10, lam_end=0.0,
        noise_std=0.5, amplitude=5.0,
    )

    # Scenario 6: Different random seed
    results["seed_check"] = run_scenario(
        "Noisy, slow decline (seed=123)",
        n_steps=800, tau_gen=20.0,
        lam_start=0.15, lam_end=0.0,
        noise_std=1.0, amplitude=3.0,
        seed=123,
    )

    # ══════════════════════════════════════════════════════
    #  OVERALL RESULTS
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  OVERALL RESULTS")
    print("=" * 60)

    lambda_wins = sum(1 for v, _ in results.values() if v == "LAMBDA_WINS")
    ews_wins = sum(1 for v, _ in results.values() if v == "EWS_WINS")
    ties = sum(1 for v, _ in results.values() if v == "TIE")
    lambda_only = sum(1 for v, _ in results.values() if v == "LAMBDA_ONLY")
    ews_only = sum(1 for v, _ in results.values() if v == "EWS_ONLY")
    neither = sum(1 for v, _ in results.values() if v == "NEITHER")

    print(f"\n  Lambda wins:      {lambda_wins}/{len(results)}")
    print(f"  EWS wins:         {ews_wins}/{len(results)}")
    print(f"  Ties:             {ties}/{len(results)}")
    print(f"  Lambda only:      {lambda_only}/{len(results)}")
    print(f"  EWS only:         {ews_only}/{len(results)}")
    print(f"  Neither detects:  {neither}/{len(results)}")

    advantages = [a for v, a in results.values()
                  if v == "LAMBDA_WINS" and a > 0]
    if advantages:
        print(f"\n  Lambda average lead: {np.mean(advantages):.0f} steps")
        print(f"  Lambda max lead:     {max(advantages)} steps")

    # Verdict
    print(f"\n  {'─'*50}")
    if lambda_wins + lambda_only > ews_wins + ews_only:
        print("  BENCHMARK: PASS — Lambda detects earlier than generic EWS")
        print("  The parametric FRM form provides genuine advantage.")
    elif lambda_wins + lambda_only == ews_wins + ews_only:
        print("  BENCHMARK: INCONCLUSIVE — No clear winner")
    else:
        print("  BENCHMARK: FAIL — Generic EWS detects earlier")
        print("  Lambda detector does not justify its complexity.")
    print(f"  {'─'*50}")

    return lambda_wins + lambda_only > ews_wins + ews_only


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
