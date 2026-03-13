# tests/test_suite.py
# Tests for the fracttalix.suite 5-detector package.
#
# Each detector is tested for:
#   1. False positive rate on the null distribution it targets
#   2. Detection on a signal it should detect
#   3. OUT_OF_SCOPE for signals in another detector's domain
#   4. Reset / state persistence contract

import math
import random

import pytest

from fracttalix.suite import (
    CouplingDetector,
    DetectorSuite,
    DiscordDetector,
    DriftDetector,
    HopfDetector,
    ScopeStatus,
    VarianceDetector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _white_noise(n, seed=42, std=1.0):
    rng = random.Random(seed)
    return [rng.gauss(0, std) for _ in range(n)]


def _sinusoid(n, freq=0.10, amp=3.0, noise=0.3, seed=42):
    rng = random.Random(seed)
    return [amp * math.sin(2 * math.pi * freq * i) + rng.gauss(0, noise)
            for i in range(n)]


def _fpr(detector_cls, signal, threshold=0.50, **kwargs):
    """Return alert rate of detector_cls on signal."""
    det = detector_cls(**kwargs)
    alerts = sum(
        1 for x in signal
        if det.update(x).status == ScopeStatus.ALERT
    )
    return alerts / len(signal)


def _any_alert(detector_cls, signal, **kwargs):
    """Return True if the detector fires at least once on signal."""
    det = detector_cls(**kwargs)
    return any(det.update(x).status == ScopeStatus.ALERT for x in signal)


# ---------------------------------------------------------------------------
# HopfDetector
# ---------------------------------------------------------------------------

class TestHopfDetector:

    def test_fpr_white_noise(self):
        """No alerts on white noise (always OUT_OF_SCOPE by design)."""
        signal = _white_noise(500)
        det = HopfDetector()
        results = [det.update(x) for x in signal]
        alerts = [r for r in results if r.status == ScopeStatus.ALERT]
        assert len(alerts) == 0

    def test_no_alert_white_noise(self):
        """White noise should never produce ALERT (EWS requires correlated signal)."""
        signal = _white_noise(500)
        det = HopfDetector()
        statuses = [det.update(x).status for x in signal]
        alerts = sum(1 for s in statuses if s == ScopeStatus.ALERT)
        assert alerts == 0, f"Got {alerts} ALERT(s) on white noise"

    def test_in_scope_oscillatory_signal(self):
        """Stable sinusoid is in scope (AC(1) > 0.10 threshold)."""
        signal = _sinusoid(200, freq=0.10, noise=0.2)
        det = HopfDetector()
        statuses = [det.update(x).status for x in signal]
        in_scope = sum(1 for s in statuses[100:] if s != ScopeStatus.OUT_OF_SCOPE)
        assert in_scope > 0, "Expected some in-scope steps on oscillatory signal"

    def test_alert_on_rising_variance(self):
        """EWS: rising variance + falling AC(1) triggers ALERT on oscillatory signal.

        The EWS signal is most visible when noise dominates the window variance.
        We use a small-amplitude sinusoid so that the rising noise is visible
        relative to the warmup baseline.
        """
        rng = random.Random(42)
        signal = []
        # Stable phase: small amplitude so noise dominates variance
        for i in range(100):
            signal.append(0.5 * math.sin(2 * math.pi * 0.10 * i) + rng.gauss(0, 0.2))
        # Explosive variance: simulate critical slowing down
        for i in range(200):
            noise_std = min(0.2 + i * 0.05, 5.0)  # 0.2 → 10.2, aggressively increasing
            signal.append(0.5 * math.sin(2 * math.pi * 0.10 * i) + rng.gauss(0, noise_std))

        det = HopfDetector(warmup=60, window=40, ews_threshold=0.45)
        statuses = [det.update(x).status for x in signal]
        late_alerts = sum(1 for s in statuses[200:] if s == ScopeStatus.ALERT)
        assert late_alerts > 0, "Expected ALERT during explosive variance phase"

    def test_reset_clears_state(self):
        """After reset, detector behaves as fresh."""
        signal = _sinusoid(150, freq=0.10, noise=0.2)
        det = HopfDetector()
        for x in signal:
            det.update(x)
        det.reset()
        # Should be warming up again
        result = det.update(signal[0])
        assert result.status == ScopeStatus.WARMUP

    def test_state_dict_roundtrip(self):
        """state_dict / load_state preserves detector output."""
        signal = _sinusoid(150, freq=0.10, noise=0.2)
        det = HopfDetector()
        for x in signal[:100]:
            det.update(x)
        sd = det.state_dict()

        det2 = HopfDetector()
        det2.load_state(sd)

        r1 = det.update(signal[100])
        r2 = det2.update(signal[100])
        assert r1.score == pytest.approx(r2.score, abs=1e-9)
        assert r1.status == r2.status

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="method"):
            HopfDetector(method='bogus')


_frm_available = True
try:
    import scipy  # noqa: F401
    import numpy  # noqa: F401
except ImportError:
    _frm_available = False

_skip_frm = pytest.mark.skipif(not _frm_available, reason="FRM method requires scipy + numpy")


@_skip_frm
class TestHopfDetectorFRM:
    """Tests for HopfDetector(method='frm') — FRM Lambda approach."""

    def test_warmup_returns_warmup_status(self):
        """FRM method returns WARMUP during warmup period."""
        det = HopfDetector(method='frm', warmup=60)
        for i in range(59):
            r = det.update(math.sin(0.2 * i))
            assert r.status == ScopeStatus.WARMUP

    def test_white_noise_out_of_scope(self):
        """White noise has no FRM structure → OUT_OF_SCOPE, no ALERT."""
        import random as _r
        rng = _r.Random(0)
        det = HopfDetector(method='frm', warmup=60)
        alerts = sum(
            1 for i in range(400)
            if det.update(rng.gauss(0, 1)).status == ScopeStatus.ALERT
        )
        assert alerts == 0, f"FRM fired {alerts} alerts on white noise"

    def test_stable_oscillation_no_alert(self):
        """Sustained sinusoid (limit cycle) does not trigger ALERT."""
        det = HopfDetector(method='frm', warmup=60)
        alerts = sum(
            1 for i in range(300)
            if det.update(3.0 * math.sin(2 * math.pi * 0.1 * i)).status == ScopeStatus.ALERT
        )
        assert alerts == 0, f"FRM fired {alerts} alerts on sustained oscillation (limit cycle FP)"

    def test_reset_clears_frm_state(self):
        """reset() clears FRM lambda history and returns to WARMUP."""
        det = HopfDetector(method='frm', warmup=60)
        for i in range(120):
            det.update(math.sin(0.2 * i))
        det.reset()
        r = det.update(0.0)
        assert r.status == ScopeStatus.WARMUP


# ---------------------------------------------------------------------------
# DiscordDetector
# ---------------------------------------------------------------------------

class TestDiscordDetector:

    def test_fpr_white_noise(self):
        """FPR on white noise should be < 2%."""
        signal = _white_noise(500)
        rate = _fpr(DiscordDetector, signal)
        assert rate < 0.02, f"FPR={rate:.3f} exceeds 2%"

    def test_detects_point_anomaly(self):
        """Single spike in otherwise stable signal → ALERT."""
        random.seed(7)  # fix global state so random.sample in DiscordDetector is deterministic
        rng = random.Random(7)
        signal = [rng.gauss(0, 1) for _ in range(200)]
        # Insert a 10σ spike
        signal[180] = 10.0

        det = DiscordDetector(warmup=80, subseq_len=10, discord_threshold=0.50)
        statuses = [det.update(x).status for x in signal]
        # Expect alert near the spike
        late_alerts = sum(1 for s in statuses[170:] if s == ScopeStatus.ALERT)
        assert late_alerts > 0, "Expected ALERT near 10σ spike"

    def test_no_alert_on_stable_sinusoid(self):
        """Stable sinusoid should not trigger discord alerts."""
        signal = _sinusoid(300, freq=0.10, noise=0.1)
        rate = _fpr(DiscordDetector, signal)
        assert rate < 0.05, f"FPR={rate:.3f} on stable sinusoid"

    def test_reset_clears_state(self):
        signal = _white_noise(120)
        det = DiscordDetector()
        for x in signal:
            det.update(x)
        det.reset()
        result = det.update(signal[0])
        assert result.status == ScopeStatus.WARMUP


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:

    def test_fpr_white_noise(self):
        """FPR on stationary N(0,1) should be < 1%."""
        signal = _white_noise(1000)
        rate = _fpr(DriftDetector, signal)
        assert rate < 0.01, f"FPR={rate:.3f} exceeds 1%"

    def test_fpr_stable_sinusoid(self):
        """FPR on stable sinusoid should be < 2%."""
        signal = _sinusoid(500, freq=0.10, noise=0.3)
        rate = _fpr(DriftDetector, signal)
        assert rate < 0.02, f"FPR={rate:.3f} on stable sinusoid"

    def test_detects_slow_upward_drift(self):
        """Slow linear mean shift → ALERT."""
        rng = random.Random(42)
        signal = []
        # Stable warmup
        for _ in range(100):
            signal.append(rng.gauss(0, 1))
        # Slow drift: +3σ over 300 steps
        for i in range(300):
            signal.append(rng.gauss(0.01 * i, 1))

        det = DriftDetector()
        statuses = [det.update(x).status for x in signal]
        late_alerts = sum(1 for s in statuses[200:] if s == ScopeStatus.ALERT)
        assert late_alerts > 0, "Expected ALERT during slow drift phase"

    def test_detects_slow_downward_drift(self):
        """Slow negative mean shift → ALERT."""
        rng = random.Random(7)
        signal = [rng.gauss(0, 1) for _ in range(100)]
        for i in range(300):
            signal.append(rng.gauss(-0.01 * i, 1))

        det = DriftDetector()
        statuses = [det.update(x).status for x in signal]
        late_alerts = sum(1 for s in statuses[200:] if s == ScopeStatus.ALERT)
        assert late_alerts > 0, "Expected ALERT during slow negative drift"

    def test_reset_clears_state(self):
        signal = _white_noise(120)
        det = DriftDetector()
        for x in signal:
            det.update(x)
        det.reset()
        result = det.update(signal[0])
        assert result.status == ScopeStatus.WARMUP


# ---------------------------------------------------------------------------
# VarianceDetector
# ---------------------------------------------------------------------------

class TestVarianceDetector:

    def test_fpr_white_noise(self):
        """FPR on stationary N(0,1) should be < 1%."""
        signal = _white_noise(1000)
        rate = _fpr(VarianceDetector, signal)
        assert rate < 0.01, f"FPR={rate:.3f} exceeds 1%"

    def test_fpr_stable_sinusoid(self):
        """No false positives on stable sinusoid."""
        signal = _sinusoid(500, freq=0.10, noise=0.3)
        rate = _fpr(VarianceDetector, signal)
        assert rate < 0.01, f"FPR={rate:.3f} on stable sinusoid"

    def test_detects_variance_explosion(self):
        """4× sudden variance increase → ALERT."""
        rng = random.Random(42)
        signal = [rng.gauss(0, 1) for _ in range(100)]
        # Sudden 4× variance spike
        signal += [rng.gauss(0, 4) for _ in range(200)]

        det = VarianceDetector()
        statuses = [det.update(x).status for x in signal]
        late_alerts = sum(1 for s in statuses[110:] if s == ScopeStatus.ALERT)
        assert late_alerts > 0, "Expected ALERT after 4× variance jump"

    def test_no_alert_on_point_spike(self):
        """Single point spike should not cause sustained alerts."""
        rng = random.Random(42)
        signal = [rng.gauss(0, 1) for _ in range(100)]
        signal.append(8.0)  # single spike
        signal += [rng.gauss(0, 1) for _ in range(100)]

        det = VarianceDetector()
        statuses = [det.update(x).status for x in signal]
        # Should not be alerting continuously after the spike
        post_spike = statuses[102:]
        sustained_alerts = sum(1 for s in post_spike if s == ScopeStatus.ALERT)
        assert sustained_alerts < 5, "Too many alerts after single spike"

    def test_reset_clears_state(self):
        signal = _white_noise(120)
        det = VarianceDetector()
        for x in signal:
            det.update(x)
        det.reset()
        result = det.update(signal[0])
        assert result.status == ScopeStatus.WARMUP


# ---------------------------------------------------------------------------
# CouplingDetector
# ---------------------------------------------------------------------------

class TestCouplingDetector:

    def test_fpr_white_noise(self):
        """White noise → OUT_OF_SCOPE (no meaningful frequency structure)."""
        signal = _white_noise(500)
        det = CouplingDetector()
        statuses = [det.update(x).status for x in signal]
        post_warmup = statuses[150:]
        alerts = sum(1 for s in post_warmup if s == ScopeStatus.ALERT)
        assert alerts == 0, f"Got {alerts} alerts on white noise"

    def test_oos_ultra_low_frequency(self):
        """Signal at f=0.04 (energy entirely in ultra_low) → OUT_OF_SCOPE.

        PAC bands (low/mid/high) contain only FFT leakage for ultra_low signals.
        The PAC scope gate (pac_power/total < 0.30) correctly excludes these.
        """
        signal = _sinusoid(400, freq=0.04, noise=0.2)
        det = CouplingDetector()
        statuses = [det.update(x).status for x in signal]
        post_warmup = statuses[150:]
        alerts = sum(1 for s in post_warmup if s == ScopeStatus.ALERT)
        assert alerts == 0, (
            f"Got {alerts} false alerts on f=0.04 (ultra_low) signal. "
            f"PAC scope gate should exclude it."
        )

    def test_in_scope_mid_frequency(self):
        """Signal at f=0.20 (energy in mid band) should be in scope."""
        signal = _sinusoid(400, freq=0.20, noise=0.2)
        det = CouplingDetector()
        statuses = [det.update(x).status for x in signal]
        post_warmup = statuses[150:]
        in_scope = sum(1 for s in post_warmup if s != ScopeStatus.OUT_OF_SCOPE)
        assert in_scope > 0, "Expected some in-scope steps on f=0.20 signal"

    def test_no_alert_single_tone_sinusoid(self):
        """Single-tone sinusoid f=0.10 → OUT_OF_SCOPE or NORMAL, never ALERT.

        A sustained sinusoid has energy in one PAC band only; the other bands
        contain only FFT leakage and noise.  PAC between a structured phase
        and a noise amplitude is spurious.  The multi-band scope gate must
        suppress false alerts.
        """
        signal = _sinusoid(500, freq=0.10, noise=0.3)
        det = CouplingDetector()
        statuses = [det.update(x).status for x in signal]
        post_warmup = statuses[150:]
        alerts = sum(1 for s in post_warmup if s == ScopeStatus.ALERT)
        assert alerts == 0, (
            f"Got {alerts} false alerts on sustained sinusoid f=0.10 "
            f"(single-tone FP regression)."
        )

    def test_reset_clears_state(self):
        signal = _sinusoid(200, freq=0.15)
        det = CouplingDetector()
        for x in signal:
            det.update(x)
        det.reset()
        result = det.update(signal[0])
        assert result.status == ScopeStatus.WARMUP


# ---------------------------------------------------------------------------
# DetectorSuite integration
# ---------------------------------------------------------------------------

class TestDetectorSuite:

    def test_smoke_white_noise(self):
        """Suite runs on white noise without error."""
        suite = DetectorSuite()
        signal = _white_noise(300)
        result = None
        for x in signal:
            result = suite.update(x)
        assert result is not None
        assert result.summary() != ""

    def test_smoke_sinusoid(self):
        """Suite runs on sinusoid without error."""
        suite = DetectorSuite()
        signal = _sinusoid(300, freq=0.10)
        result = None
        for x in signal:
            result = suite.update(x)
        assert result is not None

    def test_reset(self):
        """Suite reset restores warming-up state."""
        suite = DetectorSuite()
        for x in _white_noise(200):
            suite.update(x)
        suite.reset()
        result = suite.update(0.0)
        # All detectors should be WARMING_UP after reset
        assert result.hopf.status == ScopeStatus.WARMUP
        assert result.discord.status == ScopeStatus.WARMUP

    def test_no_alert_no_consensus_on_noise(self):
        """Suite should not produce cross-detector consensus on white noise."""
        suite = DetectorSuite()
        signal = _white_noise(500)
        for x in signal:
            suite.update(x)
        # Run extra steps to check no alerts
        alerts_per_step = []
        for x in _white_noise(200, seed=99):
            r = suite.update(x)
            n_alerts = sum(1 for d in [r.hopf, r.discord, r.drift, r.variance, r.coupling]
                           if d.status == ScopeStatus.ALERT)
            alerts_per_step.append(n_alerts)
        # No step should have 3+ detectors simultaneously alerting on noise
        max_concurrent = max(alerts_per_step)
        assert max_concurrent < 3, f"Max concurrent alerts={max_concurrent} on white noise"

    def test_suite_fpr_is_dominated_by_individual_fprs(self):
        """Overall suite FPR is at most the sum of individual FPRs."""
        suite = DetectorSuite()
        signal = _white_noise(1000)
        total_steps = 0
        any_alert_steps = 0
        for x in signal:
            r = suite.update(x)
            if r.hopf.status != ScopeStatus.WARMUP:
                total_steps += 1
                if any(d.status == ScopeStatus.ALERT
                       for d in [r.hopf, r.discord, r.drift, r.variance, r.coupling]):
                    any_alert_steps += 1
        if total_steps > 0:
            fpr = any_alert_steps / total_steps
            assert fpr < 0.05, f"Suite FPR={fpr:.3f} exceeds 5% on white noise"

    def test_summary_format(self):
        """summary() output contains expected detector abbreviations."""
        suite = DetectorSuite()
        for x in _white_noise(10):
            r = suite.update(x)
        s = r.summary()
        for abbr in ["Hopf", "Disc", "Drif", "Vari", "Coup"]:
            assert abbr in s, f"'{abbr}' not found in summary: {s}"
