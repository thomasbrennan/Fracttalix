# tests/test_steps_foundation.py
# Tests for Steps 1-7: foundation steps

import pytest
from fracttalix import SentinelConfig
from fracttalix.steps.base import RegimeBoostState
from fracttalix.steps import (
    CoreEWMAStep,
    StructuralSnapshotStep,
    FrequencyDecompositionStep,
    CUSUMStep,
    RegimeStep,
    VarCUSUMStep,
    PageHinkleyStep,
)
from fracttalix.window import StepContext, WindowBank


def make_ctx(value=1.0, step=50, config=None):
    """Create a minimal StepContext for testing individual steps."""
    if config is None:
        config = SentinelConfig()
    bank = WindowBank()
    bank.register("scalar", 128)
    # Pre-populate bank so steps have data to work with
    for i in range(64):
        bank.append(float(i % 10))
    ctx = StepContext(value=value, step=step, config=config, bank=bank, scratch={})
    # Pre-populate scratch with common fields that steps depend on
    ctx.scratch["ewma"] = 5.0
    ctx.scratch["dev_ewma"] = 1.0
    ctx.scratch["z_score"] = 0.5
    ctx.scratch["anomaly_score"] = 0.1
    ctx.scratch["alert"] = False
    ctx.scratch["baseline_mean"] = 5.0
    ctx.scratch["baseline_std"] = 1.0
    return ctx


class TestCoreEWMAStep:
    def test_instantiate(self):
        cfg = SentinelConfig()
        step = CoreEWMAStep(cfg)
        assert step is not None

    def test_instantiate_with_boost_state(self):
        cfg = SentinelConfig()
        boost = RegimeBoostState()
        step = CoreEWMAStep(cfg, boost_state=boost)
        assert step is not None

    def test_update_runs(self):
        cfg = SentinelConfig()
        step = CoreEWMAStep(cfg)
        ctx = make_ctx(value=1.0, step=0, config=cfg)
        ctx.scratch = {}  # CoreEWMAStep needs to start fresh
        step.update(ctx)
        assert "ewma" in ctx.scratch

    def test_state_dict_returns_dict(self):
        cfg = SentinelConfig()
        step = CoreEWMAStep(cfg)
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        cfg = SentinelConfig()
        step = CoreEWMAStep(cfg)
        # Run some updates first
        bank = WindowBank()
        bank.register("scalar", 128)
        for i in range(40):
            ctx = StepContext(value=float(i), step=i, config=cfg, bank=bank, scratch={})
            step.update(ctx)
        sd = step.state_dict()
        ewma_before = step._ewma

        # Load state into a fresh step
        step2 = CoreEWMAStep(cfg)
        step2.load_state(sd)
        assert step2._ewma == ewma_before

    def test_reset_clears_state(self):
        cfg = SentinelConfig()
        step = CoreEWMAStep(cfg)
        bank = WindowBank()
        bank.register("scalar", 128)
        for i in range(40):
            ctx = StepContext(value=float(i), step=i, config=cfg, bank=bank, scratch={})
            step.update(ctx)
        step.reset()
        assert step._n == 0
        assert step._ewma == 0.0


class TestStructuralSnapshotStep:
    def test_instantiate(self):
        step = StructuralSnapshotStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = StructuralSnapshotStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "structural_snapshot" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = StructuralSnapshotStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = StructuralSnapshotStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = StructuralSnapshotStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert len(step._snapshot_history) == 0


class TestFrequencyDecompositionStep:
    def test_instantiate(self):
        step = FrequencyDecompositionStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = FrequencyDecompositionStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "frequency_bands" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = FrequencyDecompositionStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_load_state(self):
        step = FrequencyDecompositionStep(SentinelConfig())
        sd = step.state_dict()
        step.load_state(sd)  # should not raise

    def test_reset(self):
        step = FrequencyDecompositionStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert len(step._bands_history) == 0


class TestCUSUMStep:
    def test_instantiate(self):
        step = CUSUMStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = CUSUMStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "cusum_hi" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = CUSUMStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        step = CUSUMStep(SentinelConfig())
        ctx = make_ctx()
        ctx.scratch["z_score"] = 2.0
        step.update(ctx)
        sd = step.state_dict()
        s_hi_before = step._s_hi

        step2 = CUSUMStep(SentinelConfig())
        step2.load_state(sd)
        assert step2._s_hi == s_hi_before

    def test_reset(self):
        step = CUSUMStep(SentinelConfig())
        ctx = make_ctx()
        ctx.scratch["z_score"] = 2.0
        step.update(ctx)
        step.reset()
        assert step._s_hi == 0.0
        assert step._s_lo == 0.0


class TestRegimeStep:
    def test_instantiate(self):
        step = RegimeStep(SentinelConfig())
        assert step is not None

    def test_instantiate_with_boost_state(self):
        boost = RegimeBoostState()
        step = RegimeStep(SentinelConfig(), boost_state=boost)
        assert step is not None

    def test_update_runs(self):
        step = RegimeStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "regime_change" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = RegimeStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        step = RegimeStep(SentinelConfig())
        ctx = make_ctx()
        ctx.scratch["z_score"] = 5.0
        step.update(ctx)
        sd = step.state_dict()

        step2 = RegimeStep(SentinelConfig())
        step2.load_state(sd)
        assert step2._regime_count == step._regime_count

    def test_reset(self):
        step = RegimeStep(SentinelConfig())
        ctx = make_ctx()
        ctx.scratch["z_score"] = 5.0
        step.update(ctx)
        step.reset()
        assert step._in_regime is False
        assert step._regime_count == 0

    def test_shared_boost_state_with_core_ewma(self):
        """RegimeBoostState is shared between CoreEWMAStep and RegimeStep."""
        boost = RegimeBoostState()
        cfg = SentinelConfig()
        core = CoreEWMAStep(cfg, boost_state=boost)
        regime = RegimeStep(cfg, boost_state=boost)
        # Both steps reference the same boost object
        assert core._boost_state is regime._boost_state


class TestVarCUSUMStep:
    def test_instantiate(self):
        step = VarCUSUMStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = VarCUSUMStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "var_cusum_hi" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = VarCUSUMStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        step = VarCUSUMStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()
        s_hi = step._s_hi

        step2 = VarCUSUMStep(SentinelConfig())
        step2.load_state(sd)
        assert step2._s_hi == s_hi

    def test_reset(self):
        step = VarCUSUMStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert step._s_hi == 0.0


class TestPageHinkleyStep:
    def test_instantiate(self):
        step = PageHinkleyStep(SentinelConfig())
        assert step is not None

    def test_update_runs(self):
        step = PageHinkleyStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        assert "ph_alert" in ctx.scratch

    def test_state_dict_returns_dict(self):
        step = PageHinkleyStep(SentinelConfig())
        sd = step.state_dict()
        assert isinstance(sd, dict)

    def test_state_round_trip(self):
        step = PageHinkleyStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        sd = step.state_dict()
        cum_hi = step._cum_hi

        step2 = PageHinkleyStep(SentinelConfig())
        step2.load_state(sd)
        assert step2._cum_hi == cum_hi

    def test_reset(self):
        step = PageHinkleyStep(SentinelConfig())
        ctx = make_ctx()
        step.update(ctx)
        step.reset()
        assert step._cum_hi == 0.0
        assert step._cum_lo == 0.0
        assert step._n == 0
