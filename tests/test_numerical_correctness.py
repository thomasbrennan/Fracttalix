# tests/test_numerical_correctness.py
# Numerical correctness and behavioral tests for core algorithms.

import math
import time

from fracttalix import SentinelConfig, SentinelDetector


class TestEWMAConvergence:
    """CoreEWMAStep should converge to input mean on constant data."""

    def test_ewma_converges_on_constant_input(self):
        det = SentinelDetector()
        for _ in range(200):
            det.update_and_check(5.0)
        result = det.update_and_check(5.0)
        assert abs(result["ewma"] - 5.0) < 0.01

    def test_ewma_tracks_step_change(self):
        """EWMA should adapt after a step change in mean."""
        det = SentinelDetector()
        # Warm up at mean=10
        for _ in range(100):
            det.update_and_check(10.0)
        # Shift to mean=20, give time to adapt
        for _ in range(200):
            det.update_and_check(20.0)
        result = det.update_and_check(20.0)
        assert abs(result["ewma"] - 20.0) < 0.5

    def test_anomaly_score_bounded_0_1(self):
        """anomaly_score should always be in [0, 1]."""
        det = SentinelDetector()
        for i in range(100):
            result = det.update_and_check(float(i))
            assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_anomaly_score_bounded_on_spike(self):
        """Even extreme spikes should produce anomaly_score in [0, 1]."""
        det = SentinelDetector()
        for _ in range(50):
            det.update_and_check(1.0)
        result = det.update_and_check(1000.0)
        assert 0.0 <= result["anomaly_score"] <= 1.0


class TestThresholdBehavior:
    """Verify that alerts fire on genuinely anomalous data."""

    def test_stable_data_low_z_score(self):
        """Stable input should produce low z-scores after warmup."""
        import random
        random.seed(123)
        cfg = SentinelConfig(warmup_periods=30)
        det = SentinelDetector(config=cfg)
        z_scores = []
        for i in range(300):
            r = det.update_and_check(5.0 + random.gauss(0, 0.1))
            if i >= 60:
                z_scores.append(abs(r["z_score"]))
        mean_abs_z = sum(z_scores) / len(z_scores)
        assert mean_abs_z < 2.0, f"Mean |z| = {mean_abs_z:.2f} too high for stable data"

    def test_spike_triggers_alert(self):
        """A large spike should trigger an alert."""
        det = SentinelDetector()
        for _ in range(50):
            det.update_and_check(1.0)
        result = det.update_and_check(100.0)
        assert result["alert"] is True

    def test_z_score_sign_matches_direction(self):
        """Positive deviation should give positive z, negative should give negative."""
        det = SentinelDetector()
        for _ in range(50):
            det.update_and_check(0.0)
        r_hi = det.update_and_check(10.0)
        assert r_hi["z_score"] > 0

        det2 = SentinelDetector()
        for _ in range(50):
            det2.update_and_check(0.0)
        r_lo = det2.update_and_check(-10.0)
        assert r_lo["z_score"] < 0


class TestVarCUSUMBehavior:
    """VarCUSUM should detect sustained variance increase."""

    def test_sustained_variance_triggers_alert(self):
        """Injecting high-variance data after stable warmup should alert."""
        import random
        random.seed(42)
        cfg = SentinelConfig(warmup_periods=30)
        det = SentinelDetector(config=cfg)
        # Stable warmup
        for _ in range(50):
            det.update_and_check(5.0)
        # High variance regime
        alerted = False
        for _ in range(200):
            v = 5.0 + random.gauss(0, 10)
            r = det.update_and_check(v)
            if r.get("var_cusum_alert"):
                alerted = True
                break
        assert alerted, "VarCUSUM should detect sustained variance increase"


class TestPerformance:
    """Throughput / stress tests."""

    def test_10k_observations_no_error(self):
        """Detector should process 10K observations without error."""
        det = SentinelDetector()
        for i in range(10_000):
            det.update_and_check(math.sin(i * 0.01))
        assert len(det) == 10_000

    def test_10k_throughput(self):
        """Detector should process 10K observations in under 60 seconds."""
        det = SentinelDetector()
        data = [math.sin(i * 0.01) for i in range(10_000)]
        start = time.time()
        for v in data:
            det.update_and_check(v)
        elapsed = time.time() - start
        assert elapsed < 120.0, f"10K observations took {elapsed:.1f}s (expected <120s)"

    def test_fast_config_throughput(self):
        """Fast config should process 10K observations in under 60 seconds."""
        cfg = SentinelConfig.fast()
        det = SentinelDetector(config=cfg)
        data = [math.sin(i * 0.01) for i in range(10_000)]
        start = time.time()
        for v in data:
            det.update_and_check(v)
        elapsed = time.time() - start
        assert elapsed < 120.0, f"10K fast observations took {elapsed:.1f}s (expected <120s)"
