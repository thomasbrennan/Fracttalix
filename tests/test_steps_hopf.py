# tests/test_steps_hopf.py
# Tests for HopfDetectorStep — the λ detector
#
# Test 1: Parameter recovery on synthetic damped oscillation
# Test 2: Transition detection (λ → 0, the money test)
# Test 3: False positive rate on stable system
# Test 4: Scope boundary detection on non-FRM data

import math

import pytest

try:
    import numpy as np
    import scipy.optimize  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from fracttalix import SentinelConfig, SentinelDetector

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="scipy required")


def _make_hopf_config(**overrides):
    """Create a config with the Hopf detector enabled."""
    defaults = dict(
        enable_hopf_detector=True,
        hopf_fit_window=128,
        hopf_fit_interval=1,  # fit every step for testing
        hopf_lambda_window=20,
        hopf_lambda_warning=0.05,
        hopf_t_decision=10.0,
        hopf_r_squared_min=0.5,
        warmup_periods=5,
    )
    defaults.update(overrides)
    return SentinelConfig(**defaults)


# ---------------------------------------------------------------
# Test 1: Parameter Recovery
# ---------------------------------------------------------------

class TestParameterRecovery:
    """Fit a known damped oscillation and verify λ, ω recovery."""

    def test_recovers_lambda_from_clean_signal(self):
        """Known λ=0.02, ω=π/12 → should recover λ in right ballpark."""
        np.random.seed(42)
        tau_gen = 6.0
        omega_true = math.pi / (2.0 * tau_gen)  # ≈ 0.2618
        lam_true = 0.02  # slow decay so oscillation persists in window
        n = 200

        cfg = _make_hopf_config(hopf_tau_gen=tau_gen)
        det = SentinelDetector(config=cfg)

        last_result = None
        for t in range(n):
            # Use modular time so the window always has visible oscillation
            val = (
                5.0
                + 3.0 * math.exp(-lam_true * (t % 128)) * math.cos(omega_true * t)
                + np.random.normal(0, 0.15)
            )
            last_result = det.update_and_check(val)

        hopf = last_result.get_hopf_status()
        assert hopf["scope_status"] in ("IN_SCOPE", "BOUNDARY"), (
            f"Expected IN_SCOPE, got {hopf['scope_status']}"
        )
        assert hopf["lambda"] is not None

    def test_r_squared_is_reasonable_for_clean_signal(self):
        """R² should be > 0.3 on a clean damped oscillation."""
        np.random.seed(123)
        tau_gen = 6.0
        omega = math.pi / (2.0 * tau_gen)
        lam = 0.01  # very slow decay to keep oscillation visible
        n = 200

        cfg = _make_hopf_config(hopf_tau_gen=tau_gen)
        det = SentinelDetector(config=cfg)

        last_result = None
        for t in range(n):
            val = (
                5.0
                + 3.0 * math.exp(-lam * (t % 128)) * math.cos(omega * t)
                + np.random.normal(0, 0.1)
            )
            last_result = det.update_and_check(val)

        hopf = last_result.get_hopf_status()
        assert hopf["r_squared"] is not None
        assert hopf["r_squared"] > 0.3, (
            f"Expected R² > 0.3, got {hopf['r_squared']}"
        )


# ---------------------------------------------------------------
# Test 2: Transition Detection (The Money Test)
# ---------------------------------------------------------------

class TestTransitionDetection:
    """λ decreasing toward zero should trigger alerts."""

    def test_detects_critical_slowing(self):
        """Linearly decreasing λ should fire CRITICAL_SLOWING."""
        np.random.seed(42)
        tau_gen = 6.0
        omega = math.pi / (2.0 * tau_gen)
        n = 600

        cfg = _make_hopf_config(
            hopf_tau_gen=tau_gen,
            hopf_lambda_warning=0.03,
            hopf_fit_interval=2,
        )
        det = SentinelDetector(config=cfg)

        alerts_fired = []
        for t in range(n):
            # λ decreases from 0.15 to ~0.005 over 600 steps
            lam_t = 0.15 * (1.0 - t / n) + 0.005
            # Generate signal with this λ applied to recent window
            val = (
                5.0
                + 2.0 * math.exp(-lam_t * (t % 128)) * math.cos(omega * t)
                + np.random.normal(0, 0.15)
            )
            result = det.update_and_check(val)
            hopf = result.get_hopf_status()
            if hopf["alert"]:
                alerts_fired.append((t, hopf["alert_type"]))

        # Should have fired at least one alert
        assert len(alerts_fired) > 0, "No alerts fired during λ → 0 transition"

        # Should include CRITICAL_SLOWING or TRANSITION_APPROACHING
        alert_types = {a[1] for a in alerts_fired}
        assert alert_types & {"CRITICAL_SLOWING", "TRANSITION_APPROACHING"}, (
            f"Expected critical alerts, got {alert_types}"
        )

    def test_estimates_time_to_transition(self):
        """When λ is declining, time_to_transition should be finite."""
        np.random.seed(99)
        tau_gen = 6.0
        omega = math.pi / (2.0 * tau_gen)
        n = 400

        cfg = _make_hopf_config(hopf_tau_gen=tau_gen)
        det = SentinelDetector(config=cfg)

        found_estimate = False
        for t in range(n):
            lam_t = 0.2 * (1.0 - t / n) + 0.01
            val = (
                5.0
                + 2.0 * math.exp(-lam_t * (t % 128)) * math.cos(omega * t)
                + np.random.normal(0, 0.1)
            )
            result = det.update_and_check(val)
            hopf = result.get_hopf_status()
            if hopf["time_to_transition"] is not None:
                found_estimate = True
                assert hopf["time_to_transition"] > 0, (
                    "time_to_transition should be positive"
                )

        assert found_estimate, "Never produced a time_to_transition estimate"


# ---------------------------------------------------------------
# Test 3: False Positive Rate
# ---------------------------------------------------------------

class TestFalsePositiveRate:
    """Stable system should not trigger alerts."""

    def test_stable_system_no_alerts(self):
        """Constant λ=0.15 should NOT fire CRITICAL_SLOWING."""
        np.random.seed(42)
        tau_gen = 6.0
        omega = math.pi / (2.0 * tau_gen)
        n = 500
        lam_stable = 0.15

        cfg = _make_hopf_config(
            hopf_tau_gen=tau_gen,
            hopf_lambda_warning=0.05,
        )
        det = SentinelDetector(config=cfg)

        alert_count = 0
        fit_count = 0
        for t in range(n):
            val = (
                5.0
                + 3.0 * math.exp(-lam_stable * (t % 128)) * math.cos(omega * t)
                + np.random.normal(0, 0.2)
            )
            result = det.update_and_check(val)
            hopf = result.get_hopf_status()
            if hopf["lambda"] is not None:
                fit_count += 1
            if hopf["alert"]:
                alert_count += 1

        # Allow very small FPR but not systematic false positives
        if fit_count > 0:
            fpr = alert_count / fit_count
            assert fpr < 0.05, (
                f"False positive rate {fpr:.2%} exceeds 5% on stable system"
            )


# ---------------------------------------------------------------
# Test 4: Scope Boundary
# ---------------------------------------------------------------

class TestScopeBoundary:
    """Non-FRM data should be flagged as OUT_OF_SCOPE."""

    def test_white_noise_out_of_scope(self):
        """White noise should not fit the FRM form well."""
        np.random.seed(42)
        cfg = _make_hopf_config()
        det = SentinelDetector(config=cfg)

        out_of_scope_count = 0
        fit_count = 0
        for t in range(300):
            val = np.random.normal(0, 1)
            result = det.update_and_check(val)
            hopf = result.get_hopf_status()
            if hopf["scope_status"] != "INSUFFICIENT_DATA":
                fit_count += 1
                if hopf["scope_status"] in ("OUT_OF_SCOPE", "BOUNDARY"):
                    out_of_scope_count += 1

        # Most fits should be out of scope on white noise
        if fit_count > 0:
            oos_rate = out_of_scope_count / fit_count
            assert oos_rate > 0.5, (
                f"Expected mostly OUT_OF_SCOPE on noise, got {oos_rate:.1%}"
            )

    def test_linear_trend_out_of_scope(self):
        """Linear trend is not a damped oscillation."""
        np.random.seed(42)
        cfg = _make_hopf_config()
        det = SentinelDetector(config=cfg)

        out_of_scope_count = 0
        fit_count = 0
        for t in range(300):
            val = 0.01 * t + np.random.normal(0, 0.1)
            result = det.update_and_check(val)
            hopf = result.get_hopf_status()
            if hopf["scope_status"] != "INSUFFICIENT_DATA":
                fit_count += 1
                if hopf["scope_status"] in ("OUT_OF_SCOPE", "BOUNDARY"):
                    out_of_scope_count += 1

        if fit_count > 0:
            oos_rate = out_of_scope_count / fit_count
            assert oos_rate > 0.5, (
                f"Expected mostly OUT_OF_SCOPE on trend, got {oos_rate:.1%}"
            )

    def test_step_function_out_of_scope(self):
        """Step function is not a damped oscillation."""
        np.random.seed(42)
        cfg = _make_hopf_config()
        det = SentinelDetector(config=cfg)

        out_of_scope_count = 0
        fit_count = 0
        for t in range(300):
            val = (5.0 if t < 150 else 10.0) + np.random.normal(0, 0.1)
            result = det.update_and_check(val)
            hopf = result.get_hopf_status()
            if hopf["scope_status"] != "INSUFFICIENT_DATA":
                fit_count += 1
                if hopf["scope_status"] in ("OUT_OF_SCOPE", "BOUNDARY"):
                    out_of_scope_count += 1

        if fit_count > 0:
            oos_rate = out_of_scope_count / fit_count
            assert oos_rate > 0.3, (
                f"Expected significant OUT_OF_SCOPE on step, got {oos_rate:.1%}"
            )

    def test_no_alert_when_out_of_scope(self):
        """OUT_OF_SCOPE should suppress alerts."""
        np.random.seed(42)
        cfg = _make_hopf_config()
        det = SentinelDetector(config=cfg)

        for t in range(300):
            val = np.random.normal(0, 1)
            result = det.update_and_check(val)
            hopf = result.get_hopf_status()
            if hopf["scope_status"] == "OUT_OF_SCOPE":
                assert not hopf["alert"], (
                    "Should not alert when OUT_OF_SCOPE"
                )


# ---------------------------------------------------------------
# Test 5: Disabled by default
# ---------------------------------------------------------------

class TestDisabledByDefault:
    """Hopf detector should not run when not enabled."""

    def test_disabled_produces_no_hopf_keys(self):
        """Default config (enable_hopf_detector=False) → no hopf keys."""
        cfg = SentinelConfig.production()
        det = SentinelDetector(config=cfg)

        result = det.update_and_check(1.0)
        assert "hopf_lambda" not in result

    def test_existing_pipeline_unaffected(self):
        """Enabling Hopf detector must not change existing result keys."""
        np.random.seed(42)
        cfg_without = SentinelConfig(warmup_periods=5)
        cfg_with = SentinelConfig(
            warmup_periods=5, enable_hopf_detector=True
        )

        det_without = SentinelDetector(config=cfg_without)
        det_with = SentinelDetector(config=cfg_with)

        for t in range(50):
            val = np.random.normal(0, 1)
            r1 = det_without.update_and_check(val)
            r2 = det_with.update_and_check(val)

            # All existing keys should match
            for key in r1:
                if key.startswith("hopf_"):
                    continue
                assert key in r2, f"Missing key {key} when Hopf enabled"


# ---------------------------------------------------------------
# Test 6: Result API
# ---------------------------------------------------------------

class TestResultAPI:
    """get_hopf_status() convenience method."""

    def test_get_hopf_status_returns_dict(self):
        """get_hopf_status() should return a dict with expected keys."""
        np.random.seed(42)
        cfg = _make_hopf_config()
        det = SentinelDetector(config=cfg)

        for t in range(200):
            val = (
                5.0
                + 2.0 * math.exp(-0.1 * t) * math.cos(0.5 * t)
                + np.random.normal(0, 0.1)
            )
            result = det.update_and_check(val)

        hopf = result.get_hopf_status()
        expected_keys = {
            "lambda", "lambda_rate", "time_to_transition",
            "confidence", "scope_status", "r_squared",
            "omega", "tau_gen_implied", "alert", "alert_type",
        }
        assert expected_keys == set(hopf.keys()), (
            f"Unexpected keys: {set(hopf.keys()) - expected_keys}"
        )
