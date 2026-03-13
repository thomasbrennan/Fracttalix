# tests/test_config.py
# Tests for SentinelConfig frozen dataclass

import dataclasses

import pytest

from fracttalix import SentinelConfig


class TestSentinelConfigDefaults:
    def test_default_instantiation(self):
        cfg = SentinelConfig()
        assert isinstance(cfg, SentinelConfig)

    def test_default_alpha(self):
        cfg = SentinelConfig()
        assert cfg.alpha == 0.1

    def test_default_dev_alpha(self):
        cfg = SentinelConfig()
        assert cfg.dev_alpha == 0.1

    def test_default_multiplier(self):
        cfg = SentinelConfig()
        assert cfg.multiplier == 3.0

    def test_default_warmup_periods(self):
        cfg = SentinelConfig()
        assert cfg.warmup_periods == 30

    def test_warn_on_numpy_fallback_default_is_true(self):
        cfg = SentinelConfig()
        assert cfg.warn_on_numpy_fallback is True


class TestSentinelConfigFrozen:
    def test_frozen_alpha(self):
        cfg = SentinelConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.alpha = 0.5  # type: ignore

    def test_frozen_multiplier(self):
        cfg = SentinelConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.multiplier = 2.0  # type: ignore

    def test_frozen_warmup_periods(self):
        cfg = SentinelConfig()
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            cfg.warmup_periods = 5  # type: ignore


class TestSentinelConfigValidation:
    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError):
            SentinelConfig(alpha=0.0)

    def test_invalid_alpha_negative(self):
        with pytest.raises(ValueError):
            SentinelConfig(alpha=-0.1)

    def test_invalid_alpha_greater_than_one(self):
        with pytest.raises(ValueError):
            SentinelConfig(alpha=1.1)

    def test_valid_alpha_boundary_one(self):
        cfg = SentinelConfig(alpha=1.0)
        assert cfg.alpha == 1.0

    def test_invalid_dev_alpha(self):
        with pytest.raises(ValueError):
            SentinelConfig(dev_alpha=0.0)

    def test_invalid_multiplier_zero(self):
        with pytest.raises(ValueError):
            SentinelConfig(multiplier=0.0)

    def test_invalid_multiplier_negative(self):
        with pytest.raises(ValueError):
            SentinelConfig(multiplier=-1.0)

    def test_invalid_warmup_periods_zero(self):
        with pytest.raises(ValueError):
            SentinelConfig(warmup_periods=0)

    def test_invalid_n_channels_zero(self):
        with pytest.raises(ValueError):
            SentinelConfig(n_channels=0)

    def test_invalid_aqb_quantiles(self):
        with pytest.raises(ValueError):
            SentinelConfig(aqb_q_low=0.5, aqb_q_high=0.3)


class TestSentinelConfigPresets:
    def test_fast_returns_sentinelconfig(self):
        cfg = SentinelConfig.fast()
        assert isinstance(cfg, SentinelConfig)

    def test_fast_alpha(self):
        cfg = SentinelConfig.fast()
        assert cfg.alpha == 0.3

    def test_fast_warmup(self):
        cfg = SentinelConfig.fast()
        assert cfg.warmup_periods == 10

    def test_production_returns_sentinelconfig(self):
        cfg = SentinelConfig.production()
        assert isinstance(cfg, SentinelConfig)

    def test_production_has_raised_multiplier(self):
        """v12.2: production() uses multiplier=4.5, bare default stays 3.0."""
        prod = SentinelConfig.production()
        default = SentinelConfig()
        assert prod.multiplier == 4.5
        assert default.multiplier == 3.0
        assert prod != default

    def test_sensitive_returns_sentinelconfig(self):
        cfg = SentinelConfig.sensitive()
        assert isinstance(cfg, SentinelConfig)

    def test_sensitive_alpha(self):
        cfg = SentinelConfig.sensitive()
        assert cfg.alpha == 0.05

    def test_sensitive_multiplier(self):
        cfg = SentinelConfig.sensitive()
        assert cfg.multiplier == 2.5

    def test_sensitive_warmup(self):
        cfg = SentinelConfig.sensitive()
        assert cfg.warmup_periods == 50

    def test_realtime_returns_sentinelconfig(self):
        cfg = SentinelConfig.realtime()
        assert isinstance(cfg, SentinelConfig)

    def test_realtime_quantile_mode(self):
        cfg = SentinelConfig.realtime()
        assert cfg.quantile_threshold_mode is True

    def test_realtime_alpha(self):
        cfg = SentinelConfig.realtime()
        assert cfg.alpha == 0.2

    def test_realtime_warmup(self):
        cfg = SentinelConfig.realtime()
        assert cfg.warmup_periods == 15


class TestSentinelConfigReplace:
    def test_replace_creates_new_instance(self):
        cfg = SentinelConfig()
        cfg2 = dataclasses.replace(cfg, alpha=0.2)
        assert cfg2 is not cfg

    def test_replace_modifies_field(self):
        cfg = SentinelConfig()
        cfg2 = dataclasses.replace(cfg, alpha=0.2)
        assert cfg2.alpha == 0.2

    def test_replace_preserves_other_fields(self):
        cfg = SentinelConfig()
        cfg2 = dataclasses.replace(cfg, alpha=0.2)
        assert cfg2.multiplier == cfg.multiplier
        assert cfg2.warmup_periods == cfg.warmup_periods

    def test_replace_validates(self):
        cfg = SentinelConfig()
        with pytest.raises(ValueError):
            dataclasses.replace(cfg, alpha=-1.0)

    def test_replace_multiple_fields(self):
        cfg = SentinelConfig()
        cfg2 = dataclasses.replace(cfg, alpha=0.5, multiplier=2.0, warmup_periods=20)
        assert cfg2.alpha == 0.5
        assert cfg2.multiplier == 2.0
        assert cfg2.warmup_periods == 20
