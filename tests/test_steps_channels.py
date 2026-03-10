# tests/test_steps_channels.py
# Tests for Steps 21-25: BandAnomalyStep, CrossFrequencyCouplingStep,
#                        ChannelCoherenceStep, CascadePrecursorStep,
#                        DegradationSequenceStep

import pytest
from fracttalix import SentinelConfig
from fracttalix.steps import (
    BandAnomalyStep,
    CrossFrequencyCouplingStep,
    ChannelCoherenceStep,
    CascadePrecursorStep,
    DegradationSequenceStep,
)
from fracttalix.types import FrequencyBands
from fracttalix.window import StepContext, WindowBank


def make_ctx(value=1.0, step=50, config=None):
    """Create a StepContext past warmup with populated bank and frequency bands."""
    if config is None:
        config = SentinelConfig()
    bank = WindowBank()
    bank.register("scalar", 128)
    for i in range(80):
        bank.append(float(i % 10) + 0.5)
    ctx = StepContext(value=value, step=step, config=config, bank=bank, scratch={})
    ctx.scratch["ewma"] = 5.0
    ctx.scratch["dev_ewma"] = 1.0
    ctx.scratch["z_score"] = 0.5
    ctx.scratch["alert"] = False
    ctx.scratch["anomaly_score"] = 0.1
    ctx.scratch["baseline_mean"] = 5.0
    ctx.scratch["baseline_std"] = 1.0
    ctx.scratch["band_anomalies"] = {}
    ctx.scratch["v9_active_alerts"] = []
    ctx.scratch["ews_score"] = 0.3
    ctx.scratch["ews_regime"] = "stable"
    ctx.scratch["coupling_matrix"] = None
    ctx.scratch["channel_coherence"] = None
    ctx.scratch["structural_snapshot"] = None
    ctx.scratch["frequency_bands"] = None
    ctx.scratch["_bands_history"] = None
    ctx.scratch["_structural_snapshot_history"] = None
    return ctx


class TestBandAnomalyStep:
    def test_instantiate(self):
        step = BandAnomalyStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = BandAnomalyStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "band_anomalies" in ctx.scratch

    def test_band_anomalies_is_dict(self):
        step = BandAnomalyStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch["band_anomalies"], dict)

    def test_state_dict_returns_dict(self):
        step = BandAnomalyStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = BandAnomalyStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = BandAnomalyStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert step._band_ewma == {}
        assert step._band_dev == {}


class TestCrossFrequencyCouplingStep:
    def test_instantiate(self):
        step = CrossFrequencyCouplingStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = CrossFrequencyCouplingStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "coupling_matrix" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = CrossFrequencyCouplingStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = CrossFrequencyCouplingStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = CrossFrequencyCouplingStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()  # should not raise


class TestChannelCoherenceStep:
    def test_instantiate(self):
        step = ChannelCoherenceStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = ChannelCoherenceStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "channel_coherence" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = ChannelCoherenceStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = ChannelCoherenceStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = ChannelCoherenceStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()  # should not raise


class TestCascadePrecursorStep:
    def test_instantiate(self):
        step = CascadePrecursorStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = CascadePrecursorStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "cascade_precursor_active" in ctx.scratch

    def test_cascade_precursor_active_is_bool(self):
        step = CascadePrecursorStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch["cascade_precursor_active"], bool)

    def test_state_dict_returns_dict(self):
        step = CascadePrecursorStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = CascadePrecursorStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = CascadePrecursorStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()  # should not raise


class TestDegradationSequenceStep:
    def test_instantiate(self):
        step = DegradationSequenceStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = DegradationSequenceStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "degradation_sequence" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = DegradationSequenceStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = DegradationSequenceStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = DegradationSequenceStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()  # should not raise
