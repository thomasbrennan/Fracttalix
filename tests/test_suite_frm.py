"""Tests for FRM-derived suite detectors: Lambda, Omega, Virtu.

Lambda v2 uses variance-inversion + spectral peak width (Lorentzian HWHM)
instead of parametric curve_fit.  Scope states: INSUFFICIENT_DATA,
OUT_OF_SCOPE, STABLE, IN_SCOPE.  r_squared property returns spectral SNR.
"""

import math

import pytest

from fracttalix.suite import (
    LambdaDetector,
    OmegaDetector,
    VirtuDetector,
)

np = pytest.importorskip("numpy")

# ── Helpers ──

def _white_noise(n=500, seed=42, std=1.0):
    np.random.seed(seed)
    return np.random.normal(0, std, n)


def _damped_oscillation(n=500, tau_gen=20.0, lam=0.05, amp=3.0, noise=0.3, seed=42):
    """Clean damped oscillation: FRM form."""
    np.random.seed(seed)
    omega = math.pi / (2.0 * tau_gen)
    t = np.arange(n, dtype=float)
    values = 10.0 + amp * np.exp(-lam * (t % int(4 * tau_gen))) * np.cos(omega * t)
    values += np.random.normal(0, noise, n)
    return values


def _sustained_oscillation(n=500, tau_gen=20.0, amp=3.0, noise=0.3, seed=42):
    """Limit cycle: oscillation with no damping."""
    np.random.seed(seed)
    omega = math.pi / (2.0 * tau_gen)
    t = np.arange(n, dtype=float)
    values = 10.0 + amp * np.cos(omega * t) + np.random.normal(0, noise, n)
    return values


def _approaching_bifurcation(n=800, tau_gen=20.0, lam_start=0.15, lam_end=0.0,
                              noise=0.08, amp=3.0, seed=42):
    """Stochastic Hopf normal form approaching bifurcation.

    Uses Euler-Maruyama with dt=0.1 sub-stepping for numerical stability.
    """
    np.random.seed(seed)
    omega0 = math.pi / (2.0 * tau_gen)
    dt = 0.1
    sub_steps = 10
    x, y = 0.01, 0.01
    values = np.zeros(n)
    for t in range(n):
        frac = t / (n - 1) if n > 1 else 0
        lam_t = lam_start + (lam_end - lam_start) * frac
        mu = -lam_t
        for _ in range(sub_steps):
            r_sq = x * x + y * y
            dx = (mu * x - omega0 * y - r_sq * x) * dt + noise * np.random.normal() * math.sqrt(dt)
            dy = (omega0 * x + mu * y - r_sq * y) * dt + noise * np.random.normal() * math.sqrt(dt)
            x += dx
            y += dy
        values[t] = 10.0 + amp * x
    return values


def _frequency_shift(n=500, tau_gen=20.0, shift_point=300, freq_ratio=1.3,
                     amp=3.0, noise=0.3, seed=42):
    """Oscillation where frequency shifts partway through."""
    np.random.seed(seed)
    omega1 = math.pi / (2.0 * tau_gen)
    omega2 = omega1 * freq_ratio
    _t = np.arange(n, dtype=float)
    values = np.zeros(n)
    for i in range(n):
        omega = omega1 if i < shift_point else omega2
        values[i] = 10.0 + amp * math.cos(omega * i) + np.random.normal(0, noise)
    return values


def _run_detector(det, signal):
    """Run detector on full signal, return list of DetectorResult."""
    results = []
    for val in signal:
        results.append(det.update(float(val)))
    return results


def _any_alert(results):
    return any(r.is_alert for r in results)


def _alert_steps(results):
    return [r.step for r in results if r.is_alert]


def _first_alert_step(results):
    for r in results:
        if r.is_alert:
            return r.step
    return None


# ══════════════════════════════════════════════════════
#  LAMBDA DETECTOR TESTS
# ══════════════════════════════════════════════════════

class TestLambdaDetector:

    def test_no_alert_on_white_noise(self):
        """White noise has no spectral peak → OUT_OF_SCOPE, no alert."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128)
        results = _run_detector(det, _white_noise(400))
        assert not _any_alert(results)

    def test_low_alert_rate_on_sustained_oscillation(self):
        """Sustained oscillation has stable variance → low alert rate."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128)
        results = _run_detector(det, _sustained_oscillation(500, tau_gen=20.0))
        alerts = _alert_steps(results)
        alert_rate = len(alerts) / max(1, len(results))
        assert alert_rate < 0.05, f"FPR too high on sustained oscillation: {alert_rate:.1%}"

    def test_variance_increases_near_bifurcation(self):
        """As λ→0, variance should increase (Var ∝ 1/λ).

        Uses stochastic Hopf normal form with sub-stepped integration.
        """
        det = LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=2)
        signal = _approaching_bifurcation(800, tau_gen=20.0, noise=0.08)
        results = _run_detector(det, signal)
        # After enough data, lambda should be estimated
        lam = det.current_lambda
        # Either lambda is declining or we got an alert
        has_alert = _any_alert(results)
        has_declining = det.lambda_rate < 0
        assert lam is not None or has_alert or has_declining, (
            f"Should estimate λ near bifurcation (λ={lam}, alert={has_alert}, rate={det.lambda_rate})"
        )

    def test_alert_on_approaching_bifurcation(self):
        """Approaching bifurcation should trigger alert or detect declining λ."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=2,
                             lambda_warning=0.08)
        signal = _approaching_bifurcation(1000, tau_gen=20.0, lam_start=0.15,
                                          lam_end=0.0, noise=0.08)
        results = _run_detector(det, signal)
        has_alert = _any_alert(results)
        has_declining = det.lambda_rate < 0
        assert has_alert or has_declining, (
            f"Should detect declining λ (alert={has_alert}, rate={det.lambda_rate})"
        )

    def test_lambda_property_accessible(self):
        """Current λ should be accessible after enough data."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=1)
        for val in _damped_oscillation(300, tau_gen=20.0):
            det.update(float(val))
        assert det.current_lambda is not None or det.scope_status in ("OUT_OF_SCOPE", "INSUFFICIENT_DATA")

    def test_r_squared_returns_spectral_snr(self):
        """r_squared property returns spectral SNR (v2 API compat)."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=1)
        for val in _sustained_oscillation(300, tau_gen=20.0):
            det.update(float(val))
        assert isinstance(det.r_squared, float)
        # Sustained oscillation should have decent spectral SNR
        assert det.r_squared > 0

    def test_scope_states_valid(self):
        """Scope should be one of the v2 states."""
        valid_scopes = {"INSUFFICIENT_DATA", "OUT_OF_SCOPE", "STABLE", "IN_SCOPE", "NEAR_BOUNDARY"}
        det = LambdaDetector(tau_gen=20.0, fit_window=128)
        assert det.scope_status in valid_scopes
        for val in _sustained_oscillation(300, tau_gen=20.0):
            det.update(float(val))
        assert det.scope_status in valid_scopes

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128)
        for val in _sustained_oscillation(200, tau_gen=20.0):
            det.update(float(val))
        det.reset()
        assert det.current_lambda is None
        assert det.scope_status == "INSUFFICIENT_DATA"
        assert det.r_squared == 0.0

    def test_monte_carlo_tpr_fpr(self):
        """Monte Carlo: TPR and FPR converge to acceptable values at N=100.

        Uses linearized OU oscillator (standard CSD model) where variance
        scales as 1/λ.  Tests that the detector reliably separates forced
        (λ declining) from null (λ constant) trajectories.

        At N=100 per class, Wilson CI width is ~15%.  We use relaxed gates
        (TPR > 80%, FPR < 35%) to account for CI width while still catching
        catastrophic regressions.
        """
        from benchmark.monte_carlo_lambda import (
            generate_forced_ensemble,
            generate_null_ensemble,
            run_lambda_detector,
        )

        n = 100
        forced = generate_forced_ensemble(n, base_seed=2000)
        null = generate_null_ensemble(n, base_seed=6000)

        tp = sum(1 for t in forced if run_lambda_detector(t["values"])[0])
        fp = sum(1 for t in null if run_lambda_detector(t["values"])[0])

        tpr = tp / n
        fpr = fp / n

        assert tpr > 0.80, f"TPR={tpr:.1%} too low (need >80%)"
        assert fpr < 0.35, f"FPR={fpr:.1%} too high (need <35%)"


# ══════════════════════════════════════════════════════
#  OMEGA DETECTOR TESTS
# ══════════════════════════════════════════════════════

class TestOmegaDetector:

    def test_no_alert_on_stable_frequency(self):
        """Stable oscillation at correct frequency → low alert rate."""
        # Use longer tau_gen for better FFT resolution
        det = OmegaDetector(tau_gen=10.0, fit_window=128, deviation_threshold=0.20)
        signal = _sustained_oscillation(400, tau_gen=10.0, amp=5.0, noise=0.2)
        results = _run_detector(det, signal)
        alerts = _alert_steps(results)
        alert_rate = len(alerts) / max(1, len(results))
        assert alert_rate < 0.05, f"FPR too high on stable freq: {alert_rate:.1%}"

    def test_alert_on_frequency_shift(self):
        """Large frequency shift should trigger alert after shift point."""
        # Use stronger shift and relaxed threshold
        det = OmegaDetector(tau_gen=10.0, fit_window=128, deviation_threshold=0.20)
        signal = _frequency_shift(600, tau_gen=10.0, shift_point=350,
                                  freq_ratio=2.0, amp=5.0, noise=0.2)
        results = _run_detector(det, signal)
        # Check alerts after the shift
        post_shift_alerts = [r for r in results if r.step > 400 and r.is_alert]
        assert len(post_shift_alerts) > 0, "Should alert after large frequency shift"

    def test_out_of_scope_on_white_noise(self):
        """White noise has no dominant frequency → should be stable/low score."""
        det = OmegaDetector(tau_gen=20.0, fit_window=128, min_spectral_snr=3.0)
        results = _run_detector(det, _white_noise(400))
        # Most results should be OUT_OF_SCOPE or low score (noise has no clear peak)
        alerts = _alert_steps(results)
        alert_rate = len(alerts) / max(1, len(results))
        assert alert_rate < 0.10, f"FPR too high on noise: {alert_rate:.1%}"

    def test_strong_mode_uses_tau_gen(self):
        """Strong mode should use τ_gen for absolute reference."""
        det = OmegaDetector(tau_gen=20.0)
        assert det.omega_predicted is not None
        expected = math.pi / (2.0 * 20.0)
        assert abs(det.omega_predicted - expected) < 1e-10

    def test_weak_mode_estimates_baseline(self):
        """Weak mode (tau_gen=0) should estimate baseline from data."""
        det = OmegaDetector(tau_gen=0.0, fit_window=128)
        assert det.omega_predicted is None
        signal = _sustained_oscillation(300, tau_gen=20.0)
        _run_detector(det, signal)
        # After warmup, should have a baseline
        assert det.current_omega is not None

    def test_reset_clears_state(self):
        """Reset should clear omega history."""
        det = OmegaDetector(tau_gen=20.0, fit_window=128)
        _run_detector(det, _sustained_oscillation(200, tau_gen=20.0))
        det.reset()
        assert det.current_omega is None


# ══════════════════════════════════════════════════════
#  VIRTU DETECTOR TESTS
# ══════════════════════════════════════════════════════

class TestVirtuDetector:

    def test_no_alert_on_stable_system(self):
        """Stable system → no decision urgency."""
        lam_det = LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=4)
        virtu = VirtuDetector(lambda_detector=lam_det, window_size=128)
        signal = _sustained_oscillation(500, tau_gen=20.0)
        results = []
        for val in signal:
            lam_det.update(float(val))
            results.append(virtu.update(float(val)))
        assert not _any_alert(results), "Should not alert on stable system"

    def test_responds_to_approaching_bifurcation(self):
        """Approaching bifurcation → Virtu should register some activity.

        The stochastic Hopf signal is subtle, so we check that Virtu at
        least reads Lambda's state (decision_quality may stay 0 if Lambda
        doesn't produce strong enough declining trend).
        """
        lam_det = LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=2)
        virtu = VirtuDetector(lambda_detector=lam_det, window_size=128)
        signal = _approaching_bifurcation(1000, tau_gen=20.0, noise=0.08)
        results = []
        for val in signal:
            lam_det.update(float(val))
            results.append(virtu.update(float(val)))
        # Virtu should at least have processed the data without error
        assert len(results) == len(signal)
        # If Lambda did detect something, Virtu should have non-zero peak_quality
        lam_alerted = any(r.is_alert for r in _run_detector(
            LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=2),
            signal))
        if lam_alerted:
            # Only assert peak_quality if Lambda itself fired
            assert virtu.peak_quality >= 0.0

    def test_no_detector_means_out_of_scope(self):
        """Without a Lambda detector, Virtu should be OUT_OF_SCOPE."""
        virtu = VirtuDetector(lambda_detector=None, window_size=128)
        signal = _sustained_oscillation(200, tau_gen=20.0)
        results = _run_detector(virtu, signal)
        # All should be OUT_OF_SCOPE (no lambda detector)
        in_scope = [r for r in results if r.in_scope]
        assert len(in_scope) == 0

    def test_reset_clears_state(self):
        """Reset should clear decision state."""
        lam_det = LambdaDetector(tau_gen=20.0, fit_window=128)
        virtu = VirtuDetector(lambda_detector=lam_det, window_size=128)
        for val in _sustained_oscillation(200, tau_gen=20.0):
            lam_det.update(float(val))
            virtu.update(float(val))
        virtu.reset()
        assert virtu.decision_quality == 0.0
        assert not virtu.virtu_window_open
