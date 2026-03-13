# tests/test_frm.py
# Tests for the fracttalix.frm FRM physics layer.
#
# Covers:
#   OmegaDetector — FRM frequency integrity (numpy only)
#   VirtuDetector — time-to-bifurcation estimator
#   FRMSuite      — two-layer integration (Layer 1 + FRM physics)
#
# Note: HopfDetector(method='frm') / Lambda tests live in test_suite.py
# (TestHopfDetectorFRM).  This file covers the Lady Ada deliverables.

import math
import random

import pytest

from fracttalix.frm import FRMSuite, FRMSuiteResult, OmegaDetector, VirtuDetector
from fracttalix.suite.base import ScopeStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sinusoid(n, freq=0.10, amp=3.0, noise=0.2, seed=42):
    rng = random.Random(seed)
    return [amp * math.sin(2 * math.pi * freq * i) + rng.gauss(0, noise)
            for i in range(n)]


def _white_noise(n, seed=42):
    rng = random.Random(seed)
    return [rng.gauss(0, 1) for _ in range(n)]


def _omega_drift_signal(n, tau_gen=10.0, drift_frac=0.10, seed=42):
    """Sinusoid at FRM-predicted frequency, drifting +drift_frac at midpoint."""
    rng = random.Random(seed)
    omega_base = math.pi / (2.0 * tau_gen)
    drift_at = n // 2
    signal = []
    flags = []
    for i in range(n):
        omega = omega_base * (1.0 + drift_frac if i >= drift_at else 1.0)
        signal.append(3.0 * math.sin(omega * i) + rng.gauss(0, 0.2))
        flags.append(i >= drift_at)
    return signal, flags


# ---------------------------------------------------------------------------
# OmegaDetector
# ---------------------------------------------------------------------------

class TestOmegaDetector:

    def test_warmup_returns_warmup_status(self):
        """Returns WARMUP during warmup period."""
        tau_gen = 10.0
        det = OmegaDetector(tau_gen=tau_gen, warmup=80)
        for i in range(79):
            r = det.update(math.sin(math.pi / (2 * tau_gen) * i))
            assert r.status == ScopeStatus.WARMUP, f"Expected WARMUP at step {i}"

    def test_strong_mode_set_when_tau_gen_given(self):
        """Strong mode is active when tau_gen is supplied."""
        det = OmegaDetector(tau_gen=10.0)
        assert det._strong_mode is True

    def test_weak_mode_when_no_tau_gen(self):
        """Weak mode is active when tau_gen is None."""
        det = OmegaDetector(tau_gen=None)
        assert det._strong_mode is False

    def test_white_noise_out_of_scope(self):
        """White noise has no dominant frequency → OUT_OF_SCOPE, no ALERT."""
        det = OmegaDetector(tau_gen=10.0, warmup=80)
        signal = _white_noise(400)
        alerts = sum(
            1 for x in signal
            if det.update(x).status == ScopeStatus.ALERT
        )
        assert alerts == 0, f"OmegaDetector fired {alerts} ALERT(s) on white noise"

    def test_stable_oscillation_no_alert_strong_mode(self):
        """Stable sinusoid at FRM frequency → NORMAL (no drift), no ALERT.

        Uses tau_gen=8 so omega=π/16 lands exactly on FFT bin 2 of a 64-sample
        window (period=32, 64/32=2 exact cycles).  Parabolic interpolation
        returns the exact bin → deviation=0 → no ALERT.
        """
        # tau_gen=8 → omega=π/16 → period=32 samples → 64 samples = 2 exact cycles
        tau_gen = 8.0
        omega = math.pi / (2.0 * tau_gen)   # = π/16
        det = OmegaDetector(tau_gen=tau_gen, warmup=80, window=64, deviation_threshold=0.05)
        # Pure sinusoid with no noise — FFT estimate is exact at bin 2
        signal = [3.0 * math.sin(omega * i) for i in range(300)]
        post_warmup_alerts = sum(
            1 for r in [det.update(x) for x in signal]
            if r.status == ScopeStatus.ALERT
        )
        assert post_warmup_alerts == 0, (
            f"OmegaDetector fired {post_warmup_alerts} ALERT(s) on stable FRM frequency "
            f"(tau_gen=8, omega=π/16, 2 exact cycles per 64-sample window)."
        )

    def test_detects_omega_drift_strong_mode(self):
        """10% ω drift in strong mode → ALERT within 100 steps of onset (F-S3).

        Uses tau_gen=8 (FFT bin-aligned, period=32) so that the baseline
        omega estimate is exact and drift detection is reliable.
        """
        tau_gen = 8.0
        signal, flags = _omega_drift_signal(400, tau_gen=tau_gen, drift_frac=0.10)
        det = OmegaDetector(
            tau_gen=tau_gen, warmup=80, window=64,
            deviation_threshold=0.05, alert_steps=5
        )

        statuses = [det.update(x).status for x in signal]
        # F-S3: detect within 100 steps of onset (n//2 = 200)
        drift_start = 200
        window_end = min(drift_start + 100, len(statuses))
        alerts_in_window = sum(
            1 for s in statuses[drift_start:window_end]
            if s == ScopeStatus.ALERT
        )
        assert alerts_in_window > 0, (
            f"OmegaDetector (strong mode) failed F-S3: no ALERT in 100 steps after "
            f"10% ω drift onset (tau_gen=8, 10% drift)."
        )

    def test_reset_clears_state(self):
        """reset() returns detector to WARMUP state."""
        tau_gen = 10.0
        omega = math.pi / (2.0 * tau_gen)
        det = OmegaDetector(tau_gen=tau_gen, warmup=80)
        for i in range(150):
            det.update(3.0 * math.sin(omega * i))
        det.reset()
        r = det.update(0.0)
        assert r.status == ScopeStatus.WARMUP

    def test_state_dict_roundtrip(self):
        """state_dict / load_state preserves detector output."""
        tau_gen = 10.0
        omega = math.pi / (2.0 * tau_gen)
        det = OmegaDetector(tau_gen=tau_gen, warmup=80)
        signal = [3.0 * math.sin(omega * i) for i in range(150)]
        for x in signal[:120]:
            det.update(x)
        sd = det.state_dict()

        det2 = OmegaDetector(tau_gen=tau_gen, warmup=80)
        det2.load_state(sd)

        r1 = det.update(signal[120])
        r2 = det2.update(signal[120])
        assert r1.status == r2.status
        assert r1.score == pytest.approx(r2.score, abs=1e-9)

    def test_weak_mode_stable_no_alert(self):
        """Weak mode: stable frequency signal → no ALERT (CV near 0)."""
        det = OmegaDetector(tau_gen=None, warmup=80)
        signal = _sinusoid(400, freq=0.10, noise=0.1)
        alerts = sum(
            1 for x in signal if det.update(x).status == ScopeStatus.ALERT
        )
        # Weak mode fires on frequency instability; stable signal → at most a few
        assert alerts < 10, f"Too many weak-mode alerts ({alerts}) on stable frequency"

    def test_stable_oscillation_no_alert_tau10(self):
        """Stable sinusoid at FRM frequency (tau_gen=10) → no ALERT (regression test).

        tau_gen=10 gives period=40 samples, which is NOT an integer FFT bin in a
        64-sample window (bin 1.6).  Pure FFT-based estimation returns bin 1 (0.098)
        giving 37.5% deviation → false ALERT.  The autocorrelation-based estimator
        used in strong mode returns the correct period (lag=40) → deviation=0% → NORMAL.
        """
        tau_gen = 10.0
        omega = math.pi / (2.0 * tau_gen)
        rng = random.Random(99)
        det = OmegaDetector(tau_gen=tau_gen, warmup=80)
        signal = [3.0 * math.sin(omega * i) + rng.gauss(0, 0.2) for i in range(300)]
        post_warmup_alerts = sum(
            1 for r in [det.update(x) for x in signal]
            if r.status == ScopeStatus.ALERT
        )
        assert post_warmup_alerts == 0, (
            f"Autocorr regression: {post_warmup_alerts} false ALERTs on stable "
            f"tau_gen=10 signal (non-integer FFT bin 1.6 should not cause FP)."
        )

    def test_detects_drift_tau10(self):
        """OmegaDetector detects 10% drift at tau_gen=10 (benchmark standard).

        This is the default benchmark configuration (N=500, tau_gen=10).
        Detects ≥ 90% of anomaly steps (Signal 8 F-S7 gate requirement).
        """
        tau_gen = 10.0
        signal, flags = _omega_drift_signal(500, tau_gen=tau_gen, drift_frac=0.10)
        det = OmegaDetector(tau_gen=tau_gen, warmup=80)
        alerts = [det.update(x).status == ScopeStatus.ALERT for x in signal]
        tp = sum(a and f for a, f in zip(alerts, flags))
        pos = sum(flags)
        tpr = tp / pos
        assert tpr >= 0.90, (
            f"OmegaDetector (tau_gen=10) TPR={tpr:.1%} < 90% gate on Signal 8 drift."
        )

    def test_autocorr_estimator_accuracy(self):
        """_estimate_omega_autocorr returns accurate omega for non-integer-bin signals."""
        from fracttalix.frm.omega import _estimate_omega_autocorr
        tau_gen = 10.0
        omega_true = math.pi / (2.0 * tau_gen)
        # Steady-state window: all samples at true frequency
        rng = random.Random(7)
        window = [3.0 * math.sin(omega_true * i) + rng.gauss(0, 0.2) for i in range(64)]
        omega_obs = _estimate_omega_autocorr(window, omega_true, tolerance=0.5)
        dev = abs(omega_obs - omega_true) / omega_true
        assert dev < 0.05, (
            f"_estimate_omega_autocorr returned {omega_obs:.5f} (true {omega_true:.5f}), "
            f"deviation {dev:.3f} exceeds 5% threshold."
        )


# ---------------------------------------------------------------------------
# VirtuDetector
# ---------------------------------------------------------------------------

class TestVirtuDetector:

    def test_warmup_returns_warmup(self):
        """Returns WARMUP during warmup period."""
        det = VirtuDetector(warmup=20)
        for i in range(20):
            r = det.update_frm(
                lambda_val=0.3, lam_rate=-0.01, time_to_bif=30.0,
                omega_in_scope=True, step=i,
            )
            assert r.status == ScopeStatus.WARMUP

    def test_no_alert_stable_lambda(self):
        """Stable or rising λ (rate ≥ 0) → NORMAL, no ALERT."""
        det = VirtuDetector(warmup=5)
        result = det.update_frm(
            lambda_val=0.5, lam_rate=0.001, time_to_bif=None,
            omega_in_scope=True, step=10,
        )
        assert result.status == ScopeStatus.NORMAL

    def test_oos_when_omega_trust_and_omega_out(self):
        """omega_trust=True + omega_in_scope=False → OUT_OF_SCOPE."""
        det = VirtuDetector(warmup=5, omega_trust=True)
        result = det.update_frm(
            lambda_val=0.1, lam_rate=-0.02, time_to_bif=5.0,
            omega_in_scope=False, step=10,
        )
        assert result.status == ScopeStatus.OUT_OF_SCOPE

    def test_alert_on_declining_lambda_omega_trusted(self):
        """Declining λ with omega_in_scope=True → ALERT when ttb is short."""
        det = VirtuDetector(warmup=5, omega_trust=True)
        # ttb=5 → score = 1 - 5/200 = 0.975 → ALERT (threshold 0.5)
        result = det.update_frm(
            lambda_val=0.1, lam_rate=-0.02, time_to_bif=5.0,
            omega_in_scope=True, step=10,
        )
        assert result.status == ScopeStatus.ALERT
        assert result.score > 0.5

    def test_normal_when_ttb_distant(self):
        """Declining λ but ttb >> horizon → NORMAL (not urgent)."""
        det = VirtuDetector(warmup=5)
        # ttb=500 >> horizon=200 → score ≈ 0 (capped at 1 - 500/200 = negative → 0)
        result = det.update_frm(
            lambda_val=0.4, lam_rate=-0.002, time_to_bif=500.0,
            omega_in_scope=True, step=10,
        )
        assert result.status == ScopeStatus.NORMAL

    def test_safety_factor_increases_urgency(self):
        """Higher safety_factor → shorter reported ttb → higher score."""
        ttb_raw = 100.0
        det1 = VirtuDetector(warmup=5, safety_factor=1.0)
        det2 = VirtuDetector(warmup=5, safety_factor=2.0)

        r1 = det1.update_frm(None, -0.01, ttb_raw, True, step=10)
        r2 = det2.update_frm(None, -0.01, ttb_raw, True, step=10)
        assert r2.score >= r1.score, "Higher safety_factor should increase urgency score"

    def test_no_alert_via_generic_update(self):
        """VirtuDetector.update() (generic path) returns NORMAL with message."""
        det = VirtuDetector(warmup=5)
        for _ in range(10):
            r = det.update(0.5)
        assert "update_frm" in r.message.lower() or r.score == 0.0

    def test_reset_clears_state(self):
        """reset() returns detector to WARMUP."""
        det = VirtuDetector(warmup=5)
        for i in range(20):
            det.update_frm(0.3, -0.01, 30.0, True, step=i)
        det.reset()
        r = det.update_frm(0.3, -0.01, 30.0, True, step=0)
        assert r.status == ScopeStatus.WARMUP

    def test_state_dict_roundtrip(self):
        """state_dict / load_state preserves last_ttb and last_confidence."""
        det = VirtuDetector(warmup=5)
        for i in range(15):
            det.update_frm(0.3, -0.02, 15.0, True, step=i)
        sd = det.state_dict()

        det2 = VirtuDetector(warmup=5)
        det2.load_state(sd)
        assert det2._last_ttb == det._last_ttb
        assert det2._last_confidence == det._last_confidence


# ---------------------------------------------------------------------------
# FRMSuite integration
# ---------------------------------------------------------------------------

class TestFRMSuite:

    def test_smoke_white_noise(self):
        """FRMSuite runs on white noise without error."""
        suite = FRMSuite(tau_gen=10.0)
        signal = _white_noise(200)
        result = None
        for x in signal:
            result = suite.update(x)
        assert result is not None
        assert isinstance(result, FRMSuiteResult)

    def test_smoke_sinusoid(self):
        """FRMSuite runs on FRM-frequency sinusoid without error."""
        tau_gen = 10.0
        omega = math.pi / (2.0 * tau_gen)
        suite = FRMSuite(tau_gen=tau_gen)
        for i in range(300):
            suite.update(3.0 * math.sin(omega * i))

    def test_frm_confidence_type_and_range(self):
        """frm_confidence is int in [0, 3]; frm_confidence_plus in [0, 4]."""
        suite = FRMSuite(tau_gen=10.0)
        for x in _white_noise(100):
            r = suite.update(x)
        assert isinstance(r.frm_confidence, int)
        assert 0 <= r.frm_confidence <= 3
        assert 0 <= r.frm_confidence_plus <= 4

    def test_any_alert_is_bool(self):
        """any_alert and alerts work correctly."""
        suite = FRMSuite(tau_gen=10.0)
        for x in _white_noise(150):
            r = suite.update(x)
        assert isinstance(r.any_alert, bool)
        assert isinstance(r.alerts, list)

    def test_layer1_always_returns_results(self):
        """Layer 1 always runs; layer1 result has all 5 detectors."""
        suite = FRMSuite()  # no tau_gen
        for x in _white_noise(100):
            r = suite.update(x)
        assert r.layer1.hopf is not None
        assert r.layer1.discord is not None
        assert r.layer1.drift is not None
        assert r.layer1.variance is not None
        assert r.layer1.coupling is not None

    def test_omega_runs_without_scipy(self):
        """OmegaDetector (numpy only) is not blocked by Lambda's scipy dependency.

        This verifies the per-detector availability fix: even when scipy is absent
        (Lambda will fail), Omega should still run independently.
        """
        suite = FRMSuite(tau_gen=10.0)
        tau_gen = 10.0
        omega = math.pi / (2.0 * tau_gen)
        signal = [3.0 * math.sin(omega * i) for i in range(300)]
        for x in signal:
            r = suite.update(x)

        # If scipy is absent, Lambda should be OUT_OF_SCOPE but Omega should NOT be
        # blocked (it should be WARMUP, NORMAL, or ALERT — not the lambda-blocked OOS)
        try:
            import scipy  # noqa
            # scipy present: both should be runnable
        except ImportError:
            # scipy absent: Lambda is OOS, Omega should NOT be "optional dependency not available"
            assert r.lambda_.status == ScopeStatus.OUT_OF_SCOPE
            assert "optional dependency" not in r.omega.message or r.omega.status != ScopeStatus.OUT_OF_SCOPE, (
                "Omega was blocked by Lambda's scipy failure. Per-detector availability fix required."
            )

    def test_reset_restores_warmup(self):
        """reset() restores FRMSuite to fresh state."""
        suite = FRMSuite(tau_gen=10.0)
        for x in _white_noise(200):
            suite.update(x)
        suite.reset()
        r = suite.update(0.0)
        assert r.layer1.hopf.status == ScopeStatus.WARMUP

    def test_state_dict_roundtrip(self):
        """state_dict / load_state preserves suite state."""
        suite = FRMSuite(tau_gen=10.0)
        signal = _white_noise(150)
        for x in signal[:100]:
            suite.update(x)
        sd = suite.state_dict()

        suite2 = FRMSuite(tau_gen=10.0)
        suite2.load_state(sd)

        r1 = suite.update(signal[100])
        r2 = suite2.update(signal[100])
        assert r1.layer1.hopf.status == r2.layer1.hopf.status
        assert r1.layer1.drift.status == r2.layer1.drift.status

    def test_summary_format(self):
        """summary() contains L1 and L2 lines."""
        suite = FRMSuite(tau_gen=10.0)
        for x in _white_noise(50):
            r = suite.update(x)
        s = r.summary()
        assert "L1:" in s
        assert "L2:" in s

    def test_frm_confidence_plus_geq_frm_confidence(self):
        """frm_confidence_plus >= frm_confidence always."""
        suite = FRMSuite(tau_gen=10.0)
        for x in _white_noise(200):
            r = suite.update(x)
            assert r.frm_confidence_plus >= r.frm_confidence

    def test_no_tau_gen_weak_mode(self):
        """FRMSuite without tau_gen runs in weak mode (no frm_confidence from Omega)."""
        suite = FRMSuite()  # no tau_gen
        for x in _white_noise(200):
            r = suite.update(x)
        # In weak mode, Omega does not contribute to frm_confidence
        # (frm_confidence counts strong-mode only)
        # We can't assert frm_confidence == 0 always (Lambda still runs in weak mode)
        # but we can verify the suite completes and produces valid output
        assert 0 <= r.frm_confidence <= 3

    def test_iter_yields_eight_results(self):
        """Iterating FRMSuiteResult yields all 8 detector results."""
        suite = FRMSuite(tau_gen=10.0)
        for x in _white_noise(100):
            r = suite.update(x)
        results = list(r)
        assert len(results) == 8  # 5 Layer 1 + lambda + omega + virtu

    def test_frm_confidence_does_not_exceed_3(self):
        """frm_confidence never exceeds 3 regardless of signal."""
        suite = FRMSuite(tau_gen=10.0)
        tau_gen = 10.0
        omega = math.pi / (2.0 * tau_gen)
        # Use a signal that might trigger multiple Layer 2 detectors
        signal = [3.0 * math.sin(omega * i) * math.exp(-0.003 * i) for i in range(500)]
        for x in signal:
            r = suite.update(x)
            assert r.frm_confidence <= 3, (
                f"frm_confidence={r.frm_confidence} exceeds max of 3"
            )
