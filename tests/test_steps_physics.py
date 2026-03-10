# tests/test_steps_physics.py
# Tests for Steps 26-37: physics/dynamics steps

from fracttalix import SentinelConfig
from fracttalix.steps import (
    AlertReasonsStep,
    CouplingRateStep,
    CriticalCouplingEstimationStep,
    DiagnosticWindowStep,
    KuramotoOrderStep,
    MaintenanceBurdenStep,
    PACCoefficientStep,
    PACDegradationStep,
    PhaseExtractionStep,
    ReversedSequenceStep,
    SequenceOrderingStep,
    ThroughputEstimationStep,
)
from fracttalix.window import StepContext, WindowBank


def make_ctx(value=1.0, step=50, config=None):
    """Create a StepContext past warmup with populated bank and common scratch keys."""
    if config is None:
        config = SentinelConfig()
    bank = WindowBank()
    bank.register("scalar", 128)
    bank.register("ews_w", 40)
    for i in range(80):
        bank.append(float(i % 10) + 0.5)
    ctx = StepContext(value=value, step=step, config=config, bank=bank, scratch={})
    ctx.scratch.update({
        "ewma": 5.0,
        "dev_ewma": 1.0,
        "z_score": 0.5,
        "alert": False,
        "anomaly": False,
        "anomaly_score": 0.1,
        "baseline_mean": 5.0,
        "baseline_std": 1.0,
        "band_anomalies": {},
        "v9_active_alerts": [],
        "coupling_matrix": None,
        "channel_coherence": None,
        "structural_snapshot": None,
        "frequency_bands": None,
        "_bands_history": None,
        "_structural_snapshot_history": None,
        "ews_score": 0.3,
        "ews_regime": "stable",
        "pe": 0.5,
        "pe_baseline": 0.5,
        "cusum_alert": False,
        "var_cusum_alert": False,
        "regime_change": False,
        "ph_alert": False,
        "osc_alert": False,
        "cpd_score": 0.0,
        "rfi": 0.0,
        "mahal_dist": 0.0,
        "band_filtered_signals": {},
        "band_phases": {},
        "mean_coupling_strength": 0.5,
        "node_count": 64,
        "throughput": 10.0,
        "critical_coupling": 0.5,
        "coupling_rate": -0.01,
        "coupling_history": [0.5, 0.49, 0.48, 0.47, 0.46],
        "kuramoto_order": 0.8,
        "phi_rate": -0.01,
        "coupling_degrading": False,
        "coherence_degrading": False,
        "sequence_history": [],
        "phi_history": [0.8, 0.79, 0.78],
        "band_amplitudes": {},
        "band_powers": {},
    })
    return ctx


class TestThroughputEstimationStep:
    def test_instantiate(self):
        step = ThroughputEstimationStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = ThroughputEstimationStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "throughput" in ctx.scratch

    def test_throughput_is_float(self):
        step = ThroughputEstimationStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch["throughput"], float)

    def test_state_dict_returns_dict(self):
        step = ThroughputEstimationStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_load_state(self):
        step = ThroughputEstimationStep(SentinelConfig())
        step.load_state(step.state_dict())

    def test_reset(self):
        step = ThroughputEstimationStep(SentinelConfig())
        step.reset()  # should not raise


class TestMaintenanceBurdenStep:
    def test_instantiate(self):
        step = MaintenanceBurdenStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = MaintenanceBurdenStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "maintenance_burden" in ctx.scratch

    def test_maintenance_burden_in_range(self):
        step = MaintenanceBurdenStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        mu = ctx.scratch["maintenance_burden"]
        assert 0.0 <= mu <= 1.0

    def test_tainter_regime_is_string(self):
        step = MaintenanceBurdenStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch["tainter_regime"], str)

    def test_state_dict_returns_dict(self):
        step = MaintenanceBurdenStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_load_state(self):
        step = MaintenanceBurdenStep(SentinelConfig())
        step.load_state(step.state_dict())

    def test_reset(self):
        step = MaintenanceBurdenStep(SentinelConfig())
        step.reset()


class TestPhaseExtractionStep:
    def test_instantiate(self):
        step = PhaseExtractionStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = PhaseExtractionStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "band_phases" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = PhaseExtractionStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_load_state(self):
        step = PhaseExtractionStep(SentinelConfig())
        step.load_state(step.state_dict())

    def test_reset(self):
        step = PhaseExtractionStep(SentinelConfig())
        step.reset()


class TestPACCoefficientStep:
    def test_instantiate(self):
        step = PACCoefficientStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = PACCoefficientStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "mean_pac" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = PACCoefficientStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_load_state(self):
        step = PACCoefficientStep(SentinelConfig())
        step.load_state(step.state_dict())

    def test_reset(self):
        step = PACCoefficientStep(SentinelConfig())
        step.reset()


class TestPACDegradationStep:
    def test_instantiate(self):
        step = PACDegradationStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = PACDegradationStep(SentinelConfig())
        ctx = make_ctx()
        ctx.scratch["mean_pac"] = 0.5
        step.update(ctx)
        assert "pac_degradation_rate" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = PACDegradationStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_state_round_trip(self):
        step = PACDegradationStep(SentinelConfig())
        ctx = make_ctx()
        ctx.scratch["mean_pac"] = 0.5
        step.update(ctx)
        sd = step.state_dict()
        step2 = PACDegradationStep(SentinelConfig())
        step2.load_state(sd)
        assert step2.state_dict() == sd

    def test_reset(self):
        step = PACDegradationStep(SentinelConfig())
        step.reset()


class TestCriticalCouplingEstimationStep:
    def test_instantiate(self):
        step = CriticalCouplingEstimationStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = CriticalCouplingEstimationStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "critical_coupling" in ctx.scratch

    def test_phi_kappa_separation_is_float(self):
        """phi_kappa_separation should be a float."""
        from fracttalix import SentinelDetector
        det = SentinelDetector()
        for i in range(40):
            r = det.update_and_check(float(i % 5))
        assert isinstance(r["phi_kappa_separation"], float)

    def test_state_dict_returns_dict(self):
        step = CriticalCouplingEstimationStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_load_state(self):
        step = CriticalCouplingEstimationStep(SentinelConfig())
        step.load_state(step.state_dict())

    def test_reset(self):
        step = CriticalCouplingEstimationStep(SentinelConfig())
        step.reset()


class TestCouplingRateStep:
    def test_instantiate(self):
        step = CouplingRateStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = CouplingRateStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "coupling_rate" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = CouplingRateStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_state_round_trip(self):
        step = CouplingRateStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()
        step2 = CouplingRateStep(SentinelConfig())
        step2.load_state(sd)
        assert step2.state_dict() == sd

    def test_reset(self):
        step = CouplingRateStep(SentinelConfig())
        step.reset()


class TestDiagnosticWindowStep:
    def test_instantiate(self):
        step = DiagnosticWindowStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = DiagnosticWindowStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        # diagnostic_window_steps key should exist in scratch
        assert "diagnostic_window_steps" in ctx.scratch or "diagnostic_window_confidence" in ctx.scratch

    def test_returns_triple(self):
        """DiagnosticWindowStep should populate pessimistic/expected/optimistic triple."""
        from fracttalix import SentinelDetector
        det = SentinelDetector()
        for i in range(40):
            det.update_and_check(float(i % 5))
        result = det.update_and_check(1.0)
        dw = result.get_diagnostic_window()
        assert "steps" in dw
        assert "steps_pessimistic" in dw
        assert "steps_optimistic" in dw
        assert "confidence" in dw

    def test_state_dict_returns_dict(self):
        step = DiagnosticWindowStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_load_state(self):
        step = DiagnosticWindowStep(SentinelConfig())
        step.load_state(step.state_dict())

    def test_reset(self):
        step = DiagnosticWindowStep(SentinelConfig())
        step.reset()


class TestKuramotoOrderStep:
    def test_instantiate(self):
        step = KuramotoOrderStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = KuramotoOrderStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "kuramoto_order" in ctx.scratch

    def test_kuramoto_order_in_range(self):
        """Kuramoto order parameter should be in [0, 1]."""
        step = KuramotoOrderStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        phi = ctx.scratch["kuramoto_order"]
        assert 0.0 <= phi <= 1.0

    def test_state_dict_returns_dict(self):
        step = KuramotoOrderStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_load_state(self):
        step = KuramotoOrderStep(SentinelConfig())
        step.load_state(step.state_dict())

    def test_reset(self):
        step = KuramotoOrderStep(SentinelConfig())
        step.reset()


class TestSequenceOrderingStep:
    def test_instantiate(self):
        step = SequenceOrderingStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = SequenceOrderingStep(SentinelConfig())
        ctx = make_ctx()
        ctx.scratch["coupling_rate"] = -0.01
        step.update(ctx)
        assert "phi_rate" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = SequenceOrderingStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_state_round_trip(self):
        step = SequenceOrderingStep(SentinelConfig())
        ctx = make_ctx()
        for _ in range(5):
            step.update(ctx)
        sd = step.state_dict()
        step2 = SequenceOrderingStep(SentinelConfig())
        step2.load_state(sd)
        assert step2.state_dict() == sd

    def test_reset(self):
        step = SequenceOrderingStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert len(step._phi_history) == 0


class TestReversedSequenceStep:
    def test_instantiate(self):
        step = ReversedSequenceStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = ReversedSequenceStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "reversed_sequence" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = ReversedSequenceStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_load_state(self):
        step = ReversedSequenceStep(SentinelConfig())
        step.load_state(step.state_dict())

    def test_reset(self):
        step = ReversedSequenceStep(SentinelConfig())
        step.reset()


class TestAlertReasonsStep:
    def test_instantiate(self):
        step = AlertReasonsStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = AlertReasonsStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "alert_reasons" in ctx.scratch

    def test_alert_reasons_is_list(self):
        step = AlertReasonsStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch["alert_reasons"], list)

    def test_state_dict_returns_dict(self):
        step = AlertReasonsStep(SentinelConfig())
        assert isinstance(step.state_dict(), dict)

    def test_state_round_trip(self):
        step = AlertReasonsStep(SentinelConfig())
        sd = step.state_dict()
        step2 = AlertReasonsStep(SentinelConfig())
        step2.load_state(sd)
        assert step2.state_dict() == sd

    def test_reset(self):
        step = AlertReasonsStep(SentinelConfig())
        step.reset()
        assert step._cooldown_remaining == 0
