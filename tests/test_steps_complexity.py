# tests/test_steps_complexity.py
# Tests for Steps 16-20: EWSStep, AQBStep, SeasonalStep, MahalStep, RRSStep

from fracttalix import SentinelConfig
from fracttalix.steps import AQBStep, EWSStep, MahalStep, RegimeStep, RRSStep, SeasonalStep
from fracttalix.window import StepContext, WindowBank


def make_ctx(value=1.0, step=50, config=None):
    """Create a StepContext past warmup with populated bank."""
    if config is None:
        config = SentinelConfig()
    bank = WindowBank()
    bank.register("scalar", 128)
    bank.register("ews_w", 40)
    for i in range(80):
        v = float(i % 10) + 0.5
        bank.append(v)
    ctx = StepContext(value=value, step=step, config=config, bank=bank, scratch={})
    ctx.scratch["ewma"] = 5.0
    ctx.scratch["dev_ewma"] = 1.0
    ctx.scratch["z_score"] = 0.5
    ctx.scratch["alert"] = False
    ctx.scratch["baseline_mean"] = 5.0
    ctx.scratch["baseline_std"] = 1.0
    ctx.scratch["ews_score"] = 0.3
    return ctx


class TestEWSStep:
    def test_instantiate(self):
        step = EWSStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = EWSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "ews_score" in ctx.scratch

    def test_ews_score_is_float(self):
        step = EWSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch["ews_score"], float)

    def test_ews_regime_is_string(self):
        step = EWSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert isinstance(ctx.scratch.get("ews_regime", ""), str)

    def test_state_dict_returns_dict(self):
        step = EWSStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        step = EWSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()
        var_ewma = step._var_ewma

        step2 = EWSStep(SentinelConfig())
        step2.load_state(sd)
        assert step2._var_ewma == var_ewma

    def test_reset(self):
        step = EWSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert step._n == 0
        assert step._ews_score == 0.0


class TestAQBStep:
    def test_instantiate(self):
        step = AQBStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = AQBStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        # AQB step populates aqb-related keys
        assert "aqb_hi" in ctx.scratch or "anomaly_score" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = AQBStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = AQBStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = AQBStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()  # should not raise
        # Post-reset state should be equivalent to fresh instance
        fresh = AQBStep(SentinelConfig())
        assert step.state_dict() == fresh.state_dict()


class TestSeasonalStep:
    def test_instantiate(self):
        step = SeasonalStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = SeasonalStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        # Should populate seasonal key
        assert "seasonal_err" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = SeasonalStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = SeasonalStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = SeasonalStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()  # should not raise


class TestMahalStep:
    def test_instantiate(self):
        step = MahalStep(SentinelConfig())
        assert step is not None

    def test_instantiate_multivariate(self):
        cfg = SentinelConfig(multivariate=True, n_channels=2)
        step = MahalStep(cfg)
        assert step is not None

    def test_update_scalar_mode(self):
        """In scalar mode (multivariate=False), mahal_dist should be 0.0."""
        step = MahalStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert ctx.scratch.get("mahal_dist", 0.0) == 0.0

    def test_update_multivariate_mode(self):
        """With multivariate=True config, mahal_dist should be computed."""
        cfg = SentinelConfig(multivariate=True, n_channels=2)
        step = MahalStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 128)
        for i in range(80):
            bank.append(float(i % 10))
        ctx = StepContext(
            value=[1.0, 2.0], step=50, config=cfg, bank=bank, scratch={}
        )
        ctx.scratch["ewma"] = 1.5
        ctx.scratch["dev_ewma"] = 0.5
        ctx.scratch["z_score"] = 0.5
        ctx.scratch["alert"] = False
        step.update(ctx)
        assert "mahal_dist" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = MahalStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = MahalStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = MahalStep(SentinelConfig(multivariate=True, n_channels=2))
        step.reset()
        assert step._cov_inv is None
        assert step._mean_vec is None
        assert step._n == 0


class TestRRSStep:
    def test_instantiate(self):
        step = RRSStep(SentinelConfig())
        assert step is not None

    def test_instantiate_with_regime_step(self):
        cfg = SentinelConfig()
        regime = RegimeStep(cfg)
        step = RRSStep(cfg, regime_step=regime)
        assert step is not None

    def test_update_runs(self):
        step = RRSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "rrs" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = RRSStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = RRSStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = RRSStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert step._n == 0
