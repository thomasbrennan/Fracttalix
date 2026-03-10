# tests/test_steps_temporal.py
# Tests for Steps 8-11: STIStep, TPSStep, OscDampStep, CPDStep

from fracttalix import SentinelConfig
from fracttalix.steps import CPDStep, OscDampStep, STIStep, TPSStep
from fracttalix.window import StepContext, WindowBank


def make_ctx(value=1.0, step=50, config=None):
    """Create a StepContext past warmup with populated bank."""
    if config is None:
        config = SentinelConfig()
    bank = WindowBank()
    bank.register("scalar", 128)
    for i in range(64):
        bank.append(float(i % 10) + 0.1)
    ctx = StepContext(value=value, step=step, config=config, bank=bank, scratch={})
    ctx.scratch["ewma"] = 5.0
    ctx.scratch["dev_ewma"] = 1.0
    ctx.scratch["z_score"] = 0.5
    ctx.scratch["anomaly_score"] = 0.1
    ctx.scratch["alert"] = False
    ctx.scratch["baseline_mean"] = 5.0
    ctx.scratch["baseline_std"] = 1.0
    return ctx


class TestSTIStep:
    def test_instantiate(self):
        step = STIStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = STIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        # STI step should populate sti key
        assert "sti" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = STIStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = STIStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_state_round_trip(self):
        step = STIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()

        step2 = STIStep(SentinelConfig())
        step2.load_state(sd)
        # After round-trip, state should be equivalent
        sd2 = step2.state_dict()
        assert sd == sd2

    def test_reset(self):
        step = STIStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        # Post-reset state should match a fresh instance
        fresh = STIStep(SentinelConfig())
        assert step.state_dict() == fresh.state_dict()


class TestTPSStep:
    def test_instantiate(self):
        step = TPSStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = TPSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        # TPSStep populates tps_score key
        assert "tps_score" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = TPSStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = TPSStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)

    def test_state_round_trip(self):
        step = TPSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()

        step2 = TPSStep(SentinelConfig())
        step2.load_state(sd)
        assert step2.state_dict() == sd

    def test_reset(self):
        step = TPSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        fresh = TPSStep(SentinelConfig())
        assert step.state_dict() == fresh.state_dict()


class TestOscDampStep:
    def test_instantiate(self):
        step = OscDampStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = OscDampStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        # OscDampStep populates osc_amp and osc_alert keys
        assert "osc_amp" in ctx.scratch or "osc_alert" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = OscDampStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = OscDampStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)

    def test_state_round_trip(self):
        step = OscDampStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()

        step2 = OscDampStep(SentinelConfig())
        step2.load_state(sd)
        assert step2.state_dict() == sd

    def test_reset(self):
        step = OscDampStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        fresh = OscDampStep(SentinelConfig())
        assert step.state_dict() == fresh.state_dict()


class TestCPDStep:
    def test_instantiate(self):
        step = CPDStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = CPDStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        # CPDStep populates cpd_score key
        assert "cpd_score" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = CPDStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = CPDStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)

    def test_state_round_trip(self):
        step = CPDStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()

        step2 = CPDStep(SentinelConfig())
        step2.load_state(sd)
        assert step2.state_dict() == sd

    def test_reset(self):
        step = CPDStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        fresh = CPDStep(SentinelConfig())
        assert step.state_dict() == fresh.state_dict()
