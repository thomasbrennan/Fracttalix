# tests/test_steps_frequency.py
# Tests for Steps 12-15: RPIStep, RFIStep, SSIStep, PEStep

import pytest
from fracttalix import SentinelConfig
from fracttalix.steps import RPIStep, RFIStep, SSIStep, PEStep
from fracttalix.window import StepContext, WindowBank


def make_ctx(value=1.0, step=50, config=None):
    """Create a StepContext past warmup with populated bank."""
    if config is None:
        config = SentinelConfig()
    bank = WindowBank()
    bank.register("scalar", 128)
    bank.register("ews_w", 40)
    for i in range(80):
        bank.append(float(i % 10) + 0.5)
    ctx = StepContext(value=value, step=step, config=config, bank=bank, scratch={})
    ctx.scratch["ewma"] = 5.0
    ctx.scratch["dev_ewma"] = 1.0
    ctx.scratch["z_score"] = 0.5
    ctx.scratch["alert"] = False
    ctx.scratch["baseline_mean"] = 5.0
    ctx.scratch["baseline_std"] = 1.0
    return ctx


class TestRPIStep:
    def test_instantiate(self):
        step = RPIStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = RPIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "rpi" in ctx.scratch

    def test_rpi_is_float(self):
        step = RPIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch["rpi"], float)

    def test_state_dict_returns_dict(self):
        step = RPIStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        step = RPIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()

        step2 = RPIStep(SentinelConfig())
        step2.load_state(sd)
        assert step2._n == step._n

    def test_reset(self):
        step = RPIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert step._n == 0


class TestRFIStep:
    def test_instantiate(self):
        step = RFIStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = RFIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "rfi" in ctx.scratch

    def test_rfi_is_float(self):
        step = RFIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch["rfi"], float)

    def test_hurst_also_populated(self):
        step = RFIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "hurst" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = RFIStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        step = RFIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()

        step2 = RFIStep(SentinelConfig())
        step2.load_state(sd)
        assert step2._n == step._n

    def test_reset(self):
        step = RFIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert step._n == 0


class TestSSIStep:
    def test_instantiate(self):
        step = SSIStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = SSIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "ssi" in ctx.scratch

    def test_rsi_alias_populated(self):
        """rsi is backward-compat alias for ssi."""
        step = SSIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "rsi" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = SSIStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        step = SSIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()

        step2 = SSIStep(SentinelConfig())
        step2.load_state(sd)
        assert step2._n == step._n

    def test_reset(self):
        step = SSIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert step._n == 0


class TestPEStep:
    def test_instantiate(self):
        step = PEStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = PEStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "pe" in ctx.scratch

    def test_pe_is_float(self):
        step = PEStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch["pe"], float)

    def test_pe_in_range_after_warmup(self):
        """After sufficient data, PE should be in [0, 1]."""
        step = PEStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        pe = ctx.scratch["pe"]
        assert 0.0 <= pe <= 1.0

    def test_state_dict_returns_dict(self):
        step = PEStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        step = PEStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()
        n_before = step._n

        step2 = PEStep(SentinelConfig())
        step2.load_state(sd)
        assert step2._n == n_before

    def test_reset(self):
        step = PEStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert step._n == 0
        assert step._pe_ewma == 0.5
