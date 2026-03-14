#!/usr/bin/env python3
"""validate_frm_real_data.py — Real-data validation of FRMSuite Layer 2.

GATE CRITERION
--------------
frm_confidence must rise across the 19 forced trajectories (mean confidence
in the final 25% of steps rises with increasing forcing) and remain flat on
the 10 null trajectories.

    PASS: mean_forced_confidence > mean_null_confidence AND
          any forced trajectory reaches frm_confidence >= 2

DATA
----
Primary: thermoacoustic_ews_forced.csv (Bury et al. 2021, thermoacoustic
  Hopf bifurcation with subcritical cubic saturation). 19 forced + 10 null
  trajectories. tau_gen from dominant FFT peak per file.

Fallback: Below-bifurcation CSD synthetic data (Option A).
  The system starts at λ=-0.3 (stable damped oscillator, fixed point at
  origin) and λ is forced toward 0 (Hopf bifurcation threshold). Variance
  ∝ σ²/(2|λ|) INCREASES as λ→0 — this is the standard CSD scenario from
  the EWS literature and matches Lambda's OU physics (designed for
  below-bifurcation approach where variance, not amplitude, is the signal).

  This replaces the previous supercritical Hopf from-above scenario (λ: +0.3→0)
  which was the wrong physics for Lambda: there variance DECREASES (amplitude
  shrinks toward bifurcation), exactly inverting the detector's expected signal.

Usage:
  python benchmark/validate_frm_real_data.py
  python benchmark/validate_frm_real_data.py --tau-gen 20 --n 800
  python benchmark/validate_frm_real_data.py --verbose
"""

import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fracttalix.frm import FRMSuite
except ImportError as e:
    print(f"ERROR: FRMSuite import failed: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Below-bifurcation CSD signal generator (Option A)
#
# Physics: damped noisy oscillator, λ < 0 (stable fixed point at origin).
# As λ is forced from -lambda_0 toward 0, critical slowing down occurs:
#   Var(x) ≈ σ²/(2|λ|)  →  INCREASES as λ → 0   (OU approximation near origin)
# This is the CSD scenario Lambda was designed for.
#
# Contrast with previous supercritical Hopf from-above (λ: +0.3 → 0):
#   there Var(x) ≈ λ/2 → DECREASES near bifurcation — the opposite direction,
#   which caused Lambda to mis-fire and inverted the frm_confidence signal.
# ---------------------------------------------------------------------------

def _sl_step(x: float, y: float, lam: float, omega: float,
             sigma: float, rng: random.Random) -> tuple:
    """One Euler step of the 2-D Stuart-Landau oscillator.

    Works for both λ < 0 (below bifurcation, stable fixed point) and
    λ > 0 (above bifurcation, stable limit cycle).  The cubic term
    (x²+y²)x provides saturation when λ > 0 and contributes additional
    damping toward the origin when λ < 0.
    """
    r2 = x * x + y * y
    x += (lam * x - omega * y - r2 * x) + sigma * rng.gauss(0, 1)
    y += (lam * y + omega * x - r2 * y) + sigma * rng.gauss(0, 1)
    return x, y


def _csd_burnin(tau_gen: float, lambda_0: float, sigma: float,
                rng: random.Random) -> tuple:
    """Run below-bifurcation oscillator at λ=-lambda_0 until steady state.

    The fixed point is the origin; burn-in reaches the stationary
    variance Var(x) ≈ σ²/(2·lambda_0).  Burn-in length = 6 full
    oscillation periods (enough for the OU process to reach equilibrium).
    FRMSuite never sees these steps — they are discarded.
    """
    omega = math.pi / (2.0 * tau_gen)
    period = 4 * tau_gen  # samples per oscillation cycle
    burn_steps = max(200, int(6 * period))
    lam = -lambda_0  # stable (negative)
    x = sigma / math.sqrt(2.0 * lambda_0) * 0.5  # start near equilibrium amplitude
    y = 0.0
    for _ in range(burn_steps):
        x, y = _sl_step(x, y, lam, omega, sigma, rng)
    return x, y


def sig_csd_forced(n: int, seed: int, tau_gen: float = 20.0,
                   lambda_0: float = 0.3, sigma: float = 0.10) -> list:
    """Forced trajectory: λ ramps linearly from -lambda_0 → 0 over n steps.

    This is the below-bifurcation CSD scenario (Option A):
        dx = (λx - ω·y - (x²+y²)·x) + sigma·randn   with λ: -0.3 → 0
        dy = (λy + ω·x - (x²+y²)·y) + sigma·randn

    As λ → 0 from below, Var(x) ≈ σ²/(2|λ|) INCREASES — the CSD signal
    that Lambda and FRMSuite are designed to detect.

    A burn-in at λ=-lambda_0 is run first and discarded so that the
    detector's baseline calibrates at the initial steady-state variance,
    not the initial transient.

    Returns list of x values, length n.
    """
    rng = random.Random(seed)
    omega = math.pi / (2.0 * tau_gen)
    x, y = _csd_burnin(tau_gen, lambda_0, sigma, rng)
    signal = []
    for i in range(n):
        lam = -lambda_0 * (1.0 - i / max(n - 1, 1))  # ramp: -lambda_0 → 0
        x, y = _sl_step(x, y, lam, omega, sigma, rng)
        signal.append(x)
    return signal


def sig_csd_null(n: int, seed: int, tau_gen: float = 20.0,
                 lambda_val: float = 0.3, sigma: float = 0.10) -> list:
    """Null trajectory: λ = -lambda_val (constant, below bifurcation).

    No bifurcation approach; variance is stationary at σ²/(2·lambda_val).
    frm_confidence should remain low throughout.
    Burn-in ensures detector calibrates at the correct steady-state variance.
    """
    rng = random.Random(seed + 10000)
    omega = math.pi / (2.0 * tau_gen)
    x, y = _csd_burnin(tau_gen, lambda_val, sigma, rng)
    lam = -lambda_val  # constant, stable
    signal = []
    for _ in range(n):
        x, y = _sl_step(x, y, lam, omega, sigma, rng)
        signal.append(x)
    return signal


# ---------------------------------------------------------------------------
# CSV loader (primary data path)
# ---------------------------------------------------------------------------

def try_load_csv(path: str):
    """Attempt to load thermoacoustic_ews_forced.csv.

    Expected format: first column = trajectory index (1-29), second column = value.
    Trajectories 1-19 are forced; 20-29 are null.

    Returns (forced_list, null_list) where each is a list of float arrays,
    or None if the file cannot be found/parsed.
    """
    if not os.path.exists(path):
        return None
    try:
        forced, null = [], []
        current_traj, current_idx, current_vals = None, None, []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                traj_idx = int(float(parts[0]))
                val = float(parts[1])
                if traj_idx != current_idx:
                    if current_vals:
                        (forced if current_idx <= 19 else null).append(current_vals)
                    current_idx = traj_idx
                    current_vals = []
                current_vals.append(val)
        if current_vals:
            (forced if current_idx <= 19 else null).append(current_vals)
        return forced, null
    except Exception as e:
        print(f"  [warn] CSV parse failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------

def evaluate_trajectory(signal: list, tau_gen: float,
                         verbose: bool = False) -> dict:
    """Run FRMSuite on one trajectory; return summary statistics."""
    suite = FRMSuite(tau_gen=tau_gen)
    conf_history = []
    lambda_oos_count = 0
    omega_alert_count = 0
    for val in signal:
        result = suite.update(val)
        conf_history.append(result.frm_confidence)
        if result.lambda_.status.name == "OUT_OF_SCOPE":
            lambda_oos_count += 1
        if result.omega.status.name == "ALERT":
            omega_alert_count += 1

    n = len(conf_history)
    head_end = n // 4          # first 25%
    tail_start = n - n // 4   # last 25%

    peak_conf = max(conf_history) if conf_history else 0
    head_conf = sum(conf_history[:head_end]) / max(1, head_end)
    tail_conf = sum(conf_history[tail_start:]) / max(1, n - tail_start)
    any_conf2 = any(c >= 2 for c in conf_history)
    any_conf3 = any(c >= 3 for c in conf_history)
    # Rising trend: tail higher than head (signal of approach to bifurcation)
    rising = (tail_conf > head_conf * 1.10) if head_conf > 0.05 else (tail_conf > 0.05)

    if verbose:
        # Print confidence timeline at 10% intervals
        step = max(1, n // 10)
        tl = [conf_history[i] for i in range(0, n, step)]
        print(f"    conf timeline: {tl} | head={head_conf:.2f} tail={tail_conf:.2f}")

    return {
        "peak_conf": peak_conf,
        "head_conf": head_conf,
        "tail_conf": tail_conf,
        "any_conf2": any_conf2,
        "any_conf3": any_conf3,
        "rising": rising,
        "lambda_oos_frac": lambda_oos_count / max(1, n),
        "omega_alert_frac": omega_alert_count / max(1, n),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="validate_frm_real_data.py")
    parser.add_argument("--tau-gen", type=float, default=20.0,
                        help="FRM generation timescale (default 20)")
    parser.add_argument("--n", type=int, default=800,
                        help="Steps per synthetic trajectory (default 800)")
    parser.add_argument("--sigma", type=float, default=0.10,
                        help="Noise amplitude for synthetic data (default 0.10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-trajectory confidence timelines")
    parser.add_argument("--data-dir",
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "data"),
                        help="Directory to search for thermoacoustic CSV")
    args = parser.parse_args()

    tau_gen = args.tau_gen
    n = args.n
    sigma = args.sigma

    print("=" * 65)
    print("validate_frm_real_data.py — FRMSuite Layer 2 real-data gate")
    print("=" * 65)

    # ── Try real thermoacoustic data first ──────────────────────────────────
    csv_path = os.path.join(args.data_dir, "thermoacoustic_ews_forced.csv")
    csv_data = try_load_csv(csv_path)

    if csv_data is not None:
        forced_signals, null_signals = csv_data
        data_source = f"thermoacoustic CSV ({len(forced_signals)} forced, {len(null_signals)} null)"
        # Estimate tau_gen from first forced trajectory dominant FFT peak
        try:
            import numpy as np
            first = np.array(forced_signals[0], dtype=float)
            first -= first.mean()
            fft_mag = np.abs(np.fft.rfft(first))
            freqs = np.fft.rfftfreq(len(first))
            peak_idx = int(np.argmax(fft_mag[1:])) + 1
            f_dom = freqs[peak_idx]
            if f_dom > 1e-6:
                tau_gen = 1.0 / (4.0 * f_dom)
                print(f"  tau_gen estimated from CSV: {tau_gen:.2f} (f_dom={f_dom:.4f})")
        except Exception:
            pass
    else:
        print(f"  thermoacoustic CSV not found at: {csv_path}")
        print("  Falling back to below-bifurcation CSD synthetic data (Option A).")
        print(f"  tau_gen={tau_gen}, n={n}, sigma={sigma}")
        print()
        print("  NOTE: λ starts at -0.3 (stable fixed point) and ramps toward 0.")
        print("  Var(x) ≈ σ²/(2|λ|) INCREASES as λ→0 — the CSD signal Lambda detects.")
        print("  This matches Lambda's OU physics (below-bifurcation approach).")
        data_source = f"below-bifurcation CSD synthetic (Option A), tau_gen={tau_gen}"

        forced_signals = [
            sig_csd_forced(n, seed=i, tau_gen=tau_gen, sigma=sigma)
            for i in range(19)
        ]
        null_signals = [
            sig_csd_null(n, seed=i, tau_gen=tau_gen, sigma=sigma)
            for i in range(10)
        ]

    print()
    print(f"Data source: {data_source}")
    print(f"tau_gen={tau_gen:.2f}  (omega_expected={math.pi/(2*tau_gen):.4f} rad/step)")
    print()

    # ── Evaluate forced trajectories ─────────────────────────────────────────
    print("FORCED TRAJECTORIES (λ rising from -lambda_0 → 0, CSD)")
    print(f"  {'idx':>3}  {'peak':>4}  {'head':>6}  {'tail':>6}  {'rise':>5}  {'L-OOS':>6}  {'Ω-ALT':>6}")
    print("-" * 60)
    forced_results = []
    for i, sig in enumerate(forced_signals):
        if args.verbose:
            print(f"  forced[{i:02d}] (n={len(sig)}):")
        r = evaluate_trajectory(sig, tau_gen=tau_gen, verbose=args.verbose)
        forced_results.append(r)
        flag = "↑" if r["rising"] else "─"
        print(f"  [{i:02d}]  {r['peak_conf']:>4}  {r['head_conf']:>6.3f}  "
              f"{r['tail_conf']:>6.3f}  {flag:>5}  "
              f"{r['lambda_oos_frac']:>6.1%}  {r['omega_alert_frac']:>6.1%}")

    # ── Evaluate null trajectories ───────────────────────────────────────────
    print()
    print("NULL TRAJECTORIES (λ = -lambda_0 constant, stable fixed point)")
    print(f"  {'idx':>3}  {'peak':>4}  {'head':>6}  {'tail':>6}  {'rise':>5}  {'L-OOS':>6}  {'Ω-ALT':>6}")
    print("-" * 60)
    null_results = []
    for i, sig in enumerate(null_signals):
        if args.verbose:
            print(f"  null[{i:02d}] (n={len(sig)}):")
        r = evaluate_trajectory(sig, tau_gen=tau_gen, verbose=args.verbose)
        null_results.append(r)
        flag = "↑" if r["rising"] else "─"
        print(f"  [{i:02d}]  {r['peak_conf']:>4}  {r['head_conf']:>6.3f}  "
              f"{r['tail_conf']:>6.3f}  {flag:>5}  "
              f"{r['lambda_oos_frac']:>6.1%}  {r['omega_alert_frac']:>6.1%}")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    forced_tail_mean = sum(r["tail_conf"] for r in forced_results) / max(1, len(forced_results))
    null_tail_mean = sum(r["tail_conf"] for r in null_results) / max(1, len(null_results))
    forced_head_mean = sum(r["head_conf"] for r in forced_results) / max(1, len(forced_results))
    null_head_mean = sum(r["head_conf"] for r in null_results) / max(1, len(null_results))
    forced_peak_mean = sum(r["peak_conf"] for r in forced_results) / max(1, len(forced_results))
    null_peak_mean = sum(r["peak_conf"] for r in null_results) / max(1, len(null_results))
    any_forced_conf2 = any(r["any_conf2"] for r in forced_results)
    any_null_conf3 = any(r["any_conf3"] for r in null_results)
    forced_conf2_count = sum(1 for r in forced_results if r["any_conf2"])
    forced_conf3_count = sum(1 for r in forced_results if r["any_conf3"])
    null_conf2_count = sum(1 for r in null_results if r["any_conf2"])
    forced_rising_count = sum(1 for r in forced_results if r["rising"])
    null_rising_count = sum(1 for r in null_results if r["rising"])
    forced_lam_oos_mean = sum(r["lambda_oos_frac"] for r in forced_results) / max(1, len(forced_results))
    null_lam_oos_mean = sum(r["lambda_oos_frac"] for r in null_results) / max(1, len(null_results))
    forced_omega_alert_mean = sum(r["omega_alert_frac"] for r in forced_results) / max(1, len(forced_results))
    null_omega_alert_mean = sum(r["omega_alert_frac"] for r in null_results) / max(1, len(null_results))

    # ── Gate evaluation ───────────────────────────────────────────────────────
    # Gate 1: forced tail confidence strictly exceeds null tail confidence
    gate1_pass = forced_tail_mean > null_tail_mean
    # Gate 2: majority of forced trajectories reach conf >= 2 (Lambda + one more)
    gate2_pass = forced_conf2_count >= len(forced_results) // 2
    # Gate 3: null trajectories show flat trend (rising count < 30%)
    gate3_pass = null_rising_count <= int(0.30 * len(null_results))

    print()
    print("=" * 65)
    print("GATE SUMMARY")
    print("=" * 65)
    print(f"  Forced mean peak_conf  : {forced_peak_mean:.3f}   Null: {null_peak_mean:.3f}")
    print(f"  Forced mean head_conf  : {forced_head_mean:.3f}   Null: {null_head_mean:.3f}")
    print(f"  Forced mean tail_conf  : {forced_tail_mean:.3f}   Null: {null_tail_mean:.3f}")
    print(f"  Forced rising trend    : {forced_rising_count}/{len(forced_results)}"
          f"         Null: {null_rising_count}/{len(null_results)}")
    print(f"  Forced conf≥2          : {forced_conf2_count}/{len(forced_results)}"
          f"         Null: {null_conf2_count}/{len(null_results)}")
    print(f"  Forced conf=3          : {forced_conf3_count}/{len(forced_results)}")
    print(f"  Lambda OOS (mean)      : forced={forced_lam_oos_mean:.1%}  null={null_lam_oos_mean:.1%}")
    print(f"  Omega ALERT (mean)     : forced={forced_omega_alert_mean:.1%}  null={null_omega_alert_mean:.1%}")
    print()

    print(f"  Gate 1 — forced tail_conf > null tail_conf:   {'PASS' if gate1_pass else 'FAIL'}")
    print(f"           ({forced_tail_mean:.3f} vs {null_tail_mean:.3f})")
    print(f"  Gate 2 — majority of forced reach conf≥2:     {'PASS' if gate2_pass else 'FAIL'}")
    print(f"           ({forced_conf2_count}/{len(forced_results)} trajectories)")
    print(f"  Gate 3 — null trajectories show flat trend:   {'PASS' if gate3_pass else 'FAIL'}")
    print(f"           ({null_rising_count}/{len(null_results)} null rising, threshold ≤30%)")
    print()

    overall = gate1_pass and gate2_pass and gate3_pass
    print("=" * 65)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 65)

    print()
    print("PHYSICS DIAGNOSIS:")
    print(f"  Lambda OOS rate on forced ({forced_lam_oos_mean:.0%}) vs null ({null_lam_oos_mean:.0%}):")
    if forced_lam_oos_mean > null_lam_oos_mean + 0.05:
        print("  → Lambda exits scope more often on forced trajectories.")
        print("    Near bifurcation (λ→0), spectral SNR may drop as oscillation")
        print("    amplitude is small. Consider relaxing SNR gate near threshold.")
    print(f"  Omega false-alert rate on null ({null_omega_alert_mean:.0%}):")
    if null_omega_alert_mean > 0.10:
        print("  → Omega alerting on stable fixed-point oscillations. Phase")
        print("    diffusion causes frequency estimate scatter > threshold.")
        print("    Fix: widen Omega deviation_threshold or use Lorentzian f0.")
    if not gate1_pass and forced_tail_mean < null_tail_mean:
        print("  → Forced tail_conf < null tail_conf: detector still fires more")
        print("    on stable signals. Check Lambda baseline calibration and")
        print("    variance trend direction for below-bifurcation scenario.")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
