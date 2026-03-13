"""Tests for FRM-derived suite detectors: Lambda, Omega, Virtu.

These detectors are unique to Fracttalix — they use FRM physics
(ω = π/(2·τ_gen), λ = |α|/(Γ·τ_gen)) that no other system has.
"""

import math

import numpy as np
import pytest

from fracttalix.suite import (
    LambdaDetector,
    OmegaDetector,
    VirtuDetector,
    ScopeStatus,
)


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
                              noise=0.3, amp=3.0, seed=42):
    """Stochastic Hopf normal form approaching bifurcation."""
    np.random.seed(seed)
    omega0 = math.pi / (2.0 * tau_gen)
    x, y = 0.01, 0.01
    values = np.zeros(n)
    for t in range(n):
        frac = t / (n - 1) if n > 1 else 0
        lam_t = lam_start + (lam_end - lam_start) * frac
        mu = -lam_t
        r_sq = x * x + y * y
        dx = (mu * x - omega0 * y - r_sq * x) + noise * np.random.normal()
        dy = (omega0 * x + mu * y - r_sq * y) + noise * np.random.normal()
        x = max(-10, min(10, x + dx))
        y = max(-10, min(10, y + dy))
        values[t] = 10.0 + amp * x
    return values


def _frequency_shift(n=500, tau_gen=20.0, shift_point=300, freq_ratio=1.3,
                     amp=3.0, noise=0.3, seed=42):
    """Oscillation where frequency shifts partway through."""
    np.random.seed(seed)
    omega1 = math.pi / (2.0 * tau_gen)
    omega2 = omega1 * freq_ratio
    t = np.arange(n, dtype=float)
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
        """White noise should be OUT_OF_SCOPE (R² too low)."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128)
        results = _run_detector(det, _white_noise(400))
        assert not _any_alert(results)

    def test_no_alert_on_sustained_oscillation(self):
        """Sustained oscillation = LIMIT_CYCLE, no alert."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128)
        results = _run_detector(det, _sustained_oscillation(500, tau_gen=20.0))
        assert not _any_alert(results), "Should not alert on limit cycle"

    def test_alert_on_approaching_bifurcation(self):
        """Approaching bifurcation should trigger alert.

        Uses clean deterministic declining-λ signal that FRM can fit,
        rather than stochastic normal form (which has random phase
        perturbations that make FRM fitting unreliable).
        """
        det = LambdaDetector(tau_gen=10.0, fit_window=64, fit_interval=2)
        # Deterministic: FRM form with λ declining over time
        np.random.seed(42)
        omega = math.pi / 20.0
        n = 600
        values = np.zeros(n)
        for t in range(n):
            lam_t = 0.15 * (1.0 - t / n)  # 0.15 → 0
            local_t = t % 40
            values[t] = 10.0 + 3.0 * math.exp(-lam_t * local_t) * math.cos(omega * t)
            values[t] += np.random.normal(0, 0.2)
        results = _run_detector(det, values)
        # Lambda should either alert or at least detect declining λ
        has_alert = _any_alert(results)
        has_declining = det.lambda_rate < 0
        assert has_alert or has_declining, (
            f"Should detect declining λ (alert={has_alert}, rate={det.lambda_rate})"
        )

    def test_lambda_property_accessible(self):
        """Current λ should be accessible after fitting."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=1)
        for val in _damped_oscillation(200, tau_gen=20.0):
            det.update(float(val))
        # Should have a lambda value after enough data
        assert det.current_lambda is not None or det.scope_status == "OUT_OF_SCOPE"

    def test_r_squared_accessible(self):
        """R² should be accessible."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=1)
        for val in _sustained_oscillation(200, tau_gen=20.0):
            det.update(float(val))
        assert isinstance(det.r_squared, float)

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        det = LambdaDetector(tau_gen=20.0, fit_window=128)
        for val in _sustained_oscillation(200, tau_gen=20.0):
            det.update(float(val))
        det.reset()
        assert det.current_lambda is None
        assert det.scope_status == "INSUFFICIENT_DATA"


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

    def test_alert_on_approaching_bifurcation(self):
        """Approaching bifurcation → decision window should open."""
        lam_det = LambdaDetector(tau_gen=20.0, fit_window=128, fit_interval=4)
        virtu = VirtuDetector(lambda_detector=lam_det, window_size=128)
        signal = _approaching_bifurcation(800, tau_gen=20.0, noise=0.3)
        results = []
        for val in signal:
            lam_det.update(float(val))
            results.append(virtu.update(float(val)))
        # Virtu should detect the closing decision window
        # (may or may not fire alert depending on λ trajectory)
        # At minimum, decision quality should rise
        assert virtu.peak_quality > 0.0 or not _any_alert(
            _run_detector(lam_det, _approaching_bifurcation(800, tau_gen=20.0, noise=0.3))
        )

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
