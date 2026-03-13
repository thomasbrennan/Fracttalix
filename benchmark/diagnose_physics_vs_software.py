#!/usr/bin/env python3
"""
Diagnosis: Is the physics wrong, or is the software looking in the wrong place?

FRM form: f(t) = B + A·exp(-λt)·cos(ωt + φ)

Physics claim: near Hopf bifurcation, λ → 0.

In a noise-driven damped oscillator, the autocorrelation function is:
  ACF(τ) = (σ²/2λ) · exp(-λ|τ|) · cos(ωτ)

This IS the FRM form. So if you fit the FRM form to the ACF instead
of the raw signal, you should recover λ correctly.

Test:
  1. Generate stochastic Hopf normal form with known λ
  2. Fit FRM to raw signal (what Lambda detector does) → check R² and λ
  3. Fit FRM to autocorrelation function → check R² and λ
  4. Compare fitted λ to true λ at each stage of the approach

If the ACF approach recovers true λ but the raw signal doesn't,
then: PHYSICS IS CORRECT, SOFTWARE IS LOOKING IN THE WRONG PLACE.
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import curve_fit


def generate_stochastic_hopf(n_steps, tau_gen, lam, noise_std, seed=42):
    """Generate noise-driven damped oscillation at fixed λ.

    Uses Euler-Maruyama with dt=0.1 for stability, sub-stepping 10x.
    noise_std should be small relative to λ (e.g., 0.05-0.1).
    """
    np.random.seed(seed)
    omega0 = math.pi / (2.0 * tau_gen)
    mu = -lam
    dt = 0.1  # sub-step size
    sub_steps = 10  # 10 sub-steps per observation
    x, y = 0.01, 0.01
    values = np.zeros(n_steps)

    for t in range(n_steps):
        for _ in range(sub_steps):
            r_sq = x * x + y * y
            dx = (mu * x - omega0 * y - r_sq * x) * dt + noise_std * np.random.normal() * math.sqrt(dt)
            dy = (omega0 * x + mu * y - r_sq * y) * dt + noise_std * np.random.normal() * math.sqrt(dt)
            x += dx
            y += dy
        values[t] = x

    return values


def compute_acf(signal, max_lag):
    """Compute autocorrelation function up to max_lag."""
    n = len(signal)
    centered = signal - np.mean(signal)
    c0 = np.sum(centered ** 2)
    if c0 < 1e-12:
        return np.zeros(max_lag + 1)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.sum(centered[:n - lag] * centered[lag:]) / c0
    return acf


def frm_model(t, B, A, lam, phi, omega):
    """FRM form with fixed omega."""
    return B + A * np.exp(-lam * t) * np.cos(omega * t + phi)


def fit_frm(data, omega, label=""):
    """Fit FRM form to data. Returns (lambda, r2, params) or None."""
    n = len(data)
    t = np.arange(n, dtype=float)

    def model(t_arr, B, A, lam, phi):
        return B + A * np.exp(-lam * t_arr) * np.cos(omega * t_arr + phi)

    B_init = float(np.mean(data))
    A_init = float(np.max(np.abs(data - B_init)))
    if A_init < 1e-10:
        A_init = 1.0

    bounds = ([-np.inf, -np.inf, 0.0, -2 * math.pi],
              [np.inf, np.inf, 50.0, 2 * math.pi])

    starts = [
        (B_init, A_init, 0.1, 0.0),
        (B_init, -A_init, 0.05, math.pi / 2),
        (B_init, A_init, 0.01, -math.pi / 4),
    ]

    ss_tot = np.sum((data - np.mean(data)) ** 2)
    if ss_tot < 1e-12:
        return None

    best_r2 = -np.inf
    best_popt = None

    for p0 in starts:
        try:
            popt, _ = curve_fit(model, t, data, p0=list(p0),
                                bounds=bounds, maxfev=5000)
            y_pred = model(t, *popt)
            ss_res = np.sum((data - y_pred) ** 2)
            r2 = 1.0 - ss_res / ss_tot
            if r2 > best_r2:
                best_r2 = r2
                best_popt = popt
        except (RuntimeError, ValueError):
            continue

    if best_popt is None:
        return None

    B_fit, A_fit, lam_fit, phi_fit = best_popt
    return {"lambda": lam_fit, "r2": best_r2, "B": B_fit, "A": A_fit, "phi": phi_fit}


def main():
    print("=" * 70)
    print("  DIAGNOSIS: Physics vs Software")
    print("  Is the FRM form wrong, or is the software looking in the wrong place?")
    print("=" * 70)

    tau_gen = 20.0
    omega = math.pi / (2.0 * tau_gen)
    noise_std = 0.08  # small noise relative to damping
    window = 256
    max_lag = 128

    # Test at different true λ values (far from → near bifurcation)
    true_lambdas = [0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.01, 0.005]

    print(f"\n  τ_gen={tau_gen}, ω={omega:.4f}, noise={noise_std}, window={window}")
    print(f"\n  {'True λ':>8s} | {'Raw: λ_fit':>10s} {'R²':>6s} | "
          f"{'ACF: λ_fit':>10s} {'R²':>6s} | {'Var':>8s} | {'AC1':>6s}")
    print(f"  {'─' * 75}")

    raw_results = []
    acf_results = []

    for true_lam in true_lambdas:
        # Generate enough data, use last `window` points (after transient)
        signal = generate_stochastic_hopf(
            n_steps=window + 500, tau_gen=tau_gen,
            lam=true_lam, noise_std=noise_std, seed=42,
        )
        seg = signal[-window:]

        # Variance and AC1 (what generic EWS uses)
        variance = np.var(seg)
        centered = seg - np.mean(seg)
        c0 = np.sum(centered ** 2)
        c1 = np.sum(centered[:-1] * centered[1:])
        ac1 = c1 / c0 if abs(c0) > 1e-12 else 0.0

        # Method 1: Fit FRM to raw signal (what Lambda detector does)
        raw_fit = fit_frm(seg, omega, label="raw")

        # Method 2: Fit FRM to ACF
        acf = compute_acf(seg, max_lag)
        # ACF starts at 1.0 by definition. Fit the decaying part.
        # The FRM form for ACF: ACF(τ) = exp(-λτ)·cos(ωτ)
        # (B=0, A=1 at lag 0)
        acf_fit = fit_frm(acf, omega, label="acf")

        raw_lam = raw_fit["lambda"] if raw_fit else None
        raw_r2 = raw_fit["r2"] if raw_fit else None
        acf_lam = acf_fit["lambda"] if acf_fit else None
        acf_r2 = acf_fit["r2"] if acf_fit else None

        raw_results.append((true_lam, raw_lam, raw_r2))
        acf_results.append((true_lam, acf_lam, acf_r2))

        raw_str = f"{raw_lam:10.4f} {raw_r2:6.3f}" if raw_lam is not None else "     FAIL   FAIL"
        acf_str = f"{acf_lam:10.4f} {acf_r2:6.3f}" if acf_lam is not None else "     FAIL   FAIL"

        print(f"  {true_lam:8.3f} | {raw_str} | {acf_str} | {variance:8.4f} | {ac1:6.3f}")

    # ── Multi-seed robustness ──
    print(f"\n{'─' * 70}")
    print(f"  Multi-seed robustness (3 seeds per λ value)")
    print(f"{'─' * 70}")

    seeds = [42, 123, 789]

    print(f"\n  {'True λ':>8s} | {'ACF λ_fit (mean±std)':>22s} {'R² mean':>8s} | "
          f"{'Raw λ_fit (mean±std)':>22s} {'R² mean':>8s}")
    print(f"  {'─' * 80}")

    for true_lam in true_lambdas:
        raw_lams = []
        acf_lams = []
        raw_r2s = []
        acf_r2s = []

        for seed in seeds:
            signal = generate_stochastic_hopf(
                n_steps=window + 500, tau_gen=tau_gen,
                lam=true_lam, noise_std=noise_std, seed=seed,
            )
            seg = signal[-window:]

            raw_fit = fit_frm(seg, omega)
            if raw_fit:
                raw_lams.append(raw_fit["lambda"])
                raw_r2s.append(raw_fit["r2"])

            acf = compute_acf(seg, max_lag)
            acf_fit = fit_frm(acf, omega)
            if acf_fit:
                acf_lams.append(acf_fit["lambda"])
                acf_r2s.append(acf_fit["r2"])

        if acf_lams:
            acf_str = f"{np.mean(acf_lams):7.4f}±{np.std(acf_lams):6.4f} {np.mean(acf_r2s):8.3f}"
        else:
            acf_str = "FAIL"
        if raw_lams:
            raw_str = f"{np.mean(raw_lams):7.4f}±{np.std(raw_lams):6.4f} {np.mean(raw_r2s):8.3f}"
        else:
            raw_str = "FAIL"

        print(f"  {true_lam:8.3f} | {acf_str:>31s} | {raw_str:>31s}")

    # ── Sliding window: does ACF-λ track declining true-λ? ──
    print(f"\n{'─' * 70}")
    print(f"  Sliding window: ACF-λ tracking during bifurcation approach")
    print(f"{'─' * 70}")

    # Generate approaching bifurcation
    n_total = 2000
    lam_start = 0.15
    lam_end = 0.0
    np.random.seed(42)
    omega0 = math.pi / (2.0 * tau_gen)
    x, y = 0.01, 0.01
    values = np.zeros(n_total)
    lam_true_arr = np.zeros(n_total)
    dt = 0.1
    sub_steps = 10

    for t in range(n_total):
        frac = t / (n_total - 1)
        lam_t = lam_start + (lam_end - lam_start) * frac
        lam_true_arr[t] = lam_t
        mu = -lam_t
        for _ in range(sub_steps):
            r_sq = x * x + y * y
            dx = (mu * x - omega0 * y - r_sq * x) * dt + noise_std * np.random.normal() * math.sqrt(dt)
            dy = (omega0 * x + mu * y - r_sq * y) * dt + noise_std * np.random.normal() * math.sqrt(dt)
            x += dx
            y += dy
        values[t] = x

    print(f"\n  λ declining: {lam_start} → {lam_end} over {n_total} steps")
    print(f"\n  {'Step':>6s} {'True λ':>8s} | {'ACF λ_fit':>10s} {'R²':>6s} | "
          f"{'Raw λ_fit':>10s} {'R²':>6s} | {'Variance':>10s}")
    print(f"  {'─' * 70}")

    step_size = 100
    for start in range(window, n_total - step_size, step_size):
        seg = values[start:start + window]
        true_lam_mid = lam_true_arr[start + window // 2]
        var = np.var(seg)

        raw_fit = fit_frm(seg, omega)
        acf = compute_acf(seg, max_lag)
        acf_fit = fit_frm(acf, omega)

        raw_str = f"{raw_fit['lambda']:10.4f} {raw_fit['r2']:6.3f}" if raw_fit else "      FAIL   FAIL"
        acf_str = f"{acf_fit['lambda']:10.4f} {acf_fit['r2']:6.3f}" if acf_fit else "      FAIL   FAIL"

        print(f"  {start + window//2:6d} {true_lam_mid:8.4f} | {acf_str} | {raw_str} | {var:10.4f}")

    # ── VERDICT ──
    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}")

    # Check: does ACF-λ correlate with true λ?
    acf_lam_track = []
    true_lam_track = []
    for start in range(window, n_total - step_size, step_size):
        seg = values[start:start + window]
        true_lam_mid = lam_true_arr[start + window // 2]
        acf = compute_acf(seg, max_lag)
        acf_fit = fit_frm(acf, omega)
        if acf_fit and acf_fit["r2"] > 0.3:
            acf_lam_track.append(acf_fit["lambda"])
            true_lam_track.append(true_lam_mid)

    if len(acf_lam_track) >= 5:
        corr = np.corrcoef(acf_lam_track, true_lam_track)[0, 1]
        print(f"\n  Correlation(ACF-λ, true-λ): {corr:.3f}")
        print(f"  (n={len(acf_lam_track)} windows with R² > 0.3)")

        if corr > 0.5:
            print(f"\n  DIAGNOSIS: PHYSICS IS CORRECT, SOFTWARE IS WRONG")
            print(f"  The FRM form fits the ACF and recovers λ that tracks the")
            print(f"  true bifurcation parameter. The Lambda detector should fit")
            print(f"  FRM to the autocorrelation function, not the raw signal.")
        elif corr > 0.0:
            print(f"\n  DIAGNOSIS: PHYSICS IS PARTIALLY CORRECT")
            print(f"  ACF-λ correlates weakly with true λ. The FRM form captures")
            print(f"  some of the dynamics but not reliably enough.")
        else:
            print(f"\n  DIAGNOSIS: PHYSICS MAY BE WRONG (for this data type)")
            print(f"  Neither raw nor ACF fitting recovers true λ.")
    else:
        print(f"\n  DIAGNOSIS: INCONCLUSIVE")
        print(f"  Too few valid ACF fits ({len(acf_lam_track)}) to determine correlation.")

    # Also check raw signal correlation
    raw_lam_track = []
    true_lam_track_raw = []
    for start in range(window, n_total - step_size, step_size):
        seg = values[start:start + window]
        true_lam_mid = lam_true_arr[start + window // 2]
        raw_fit = fit_frm(seg, omega)
        if raw_fit and raw_fit["r2"] > 0.3:
            raw_lam_track.append(raw_fit["lambda"])
            true_lam_track_raw.append(true_lam_mid)

    if len(raw_lam_track) >= 5:
        corr_raw = np.corrcoef(raw_lam_track, true_lam_track_raw)[0, 1]
        print(f"  Correlation(Raw-λ, true-λ):  {corr_raw:.3f} "
              f"(n={len(raw_lam_track)} windows)")
    else:
        print(f"  Raw signal: too few valid fits ({len(raw_lam_track)})")

    return True


if __name__ == "__main__":
    main()
