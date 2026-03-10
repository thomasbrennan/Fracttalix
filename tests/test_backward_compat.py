# tests/test_backward_compat.py
# Tests for backward compatibility

import pytest
from fracttalix import SentinelDetector, SentinelConfig, Detector_7_10, SentinelResult


class TestDetector710Alias:
    def test_detector_710_is_alias(self):
        assert Detector_7_10 is SentinelDetector

    def test_detector_710_instantiates(self):
        det = Detector_7_10()
        assert isinstance(det, SentinelDetector)

    def test_detector_710_updates(self):
        det = Detector_7_10()
        result = det.update_and_check(1.0)
        assert isinstance(result, dict)


class TestLegacyKwargs:
    def test_alpha_kwarg(self):
        det = SentinelDetector(alpha=0.2)
        assert det.config.alpha == 0.2

    def test_multiplier_kwarg(self):
        det = SentinelDetector(multiplier=2.5)
        assert det.config.multiplier == 2.5

    def test_warmup_periods_kwarg(self):
        det = SentinelDetector(warmup_periods=20)
        assert det.config.warmup_periods == 20

    def test_combined_kwargs(self):
        det = SentinelDetector(alpha=0.3, multiplier=2.0, warmup_periods=15)
        assert det.config.alpha == 0.3
        assert det.config.multiplier == 2.0
        assert det.config.warmup_periods == 15

    def test_legacy_kwargs_produce_working_detector(self):
        det = SentinelDetector(alpha=0.2, multiplier=3.0)
        result = det.update_and_check(1.0)
        assert "alert" in result

    def test_rsi_window_maps_to_rpi_window(self):
        """v7.x 'rsi_window' should map to rpi_window."""
        det = SentinelDetector(rsi_window=32)
        assert det.config.rpi_window == 32


class TestProductionDefault:
    def test_production_is_default(self):
        """SentinelConfig.production() should equal the default SentinelConfig()."""
        prod = SentinelConfig.production()
        default = SentinelConfig()
        assert prod == default

    def test_default_detector_uses_production_config(self):
        det = SentinelDetector()
        prod = SentinelConfig.production()
        assert det.config == prod


class TestV9ConvenienceMethods:
    def _warmed_result(self):
        det = SentinelDetector()
        result = None
        for i in range(40):
            result = det.update_and_check(float(i % 5))
        return result

    def test_is_cascade_precursor(self):
        result = self._warmed_result()
        val = result.is_cascade_precursor()
        assert isinstance(val, bool)

    def test_get_channel_status(self):
        result = self._warmed_result()
        status = result.get_channel_status()
        assert isinstance(status, dict)
        assert "structural" in status
        assert "rhythmic_composite" in status

    def test_get_degradation_narrative(self):
        result = self._warmed_result()
        narrative = result.get_degradation_narrative()
        assert isinstance(narrative, str)

    def test_get_primary_carrier_wave(self):
        result = self._warmed_result()
        carrier = result.get_primary_carrier_wave()
        assert isinstance(carrier, str)

    def test_is_reversed_sequence(self):
        result = self._warmed_result()
        val = result.is_reversed_sequence()
        assert isinstance(val, bool)

    def test_get_intervention_signature(self):
        result = self._warmed_result()
        sig = result.get_intervention_signature()
        assert isinstance(sig, dict)
        assert "score" in sig
        assert "sequence_type" in sig

    def test_get_diagnostic_window(self):
        result = self._warmed_result()
        dw = result.get_diagnostic_window()
        assert isinstance(dw, dict)
        assert "steps" in dw
        assert "confidence" in dw

    def test_get_maintenance_burden(self):
        result = self._warmed_result()
        mb = result.get_maintenance_burden()
        assert isinstance(mb, dict)
        assert "mu" in mb
        assert "regime" in mb

    def test_get_pac_status(self):
        result = self._warmed_result()
        pac = result.get_pac_status()
        assert isinstance(pac, dict)
        assert "mean_pac" in pac

    def test_get_phi_kappa_separation(self):
        result = self._warmed_result()
        pks = result.get_phi_kappa_separation()
        assert isinstance(pks, dict)
        assert "separation" in pks
        assert "interpretation" in pks


class TestLegacyKwargsViaConfig:
    def test_legacy_kwargs_to_config_function(self):
        from fracttalix import _legacy_kwargs_to_config
        cfg = _legacy_kwargs_to_config({"alpha": 0.2, "multiplier": 2.5})
        assert isinstance(cfg, SentinelConfig)
        assert cfg.alpha == 0.2
        assert cfg.multiplier == 2.5

    def test_legacy_kwargs_to_config_rsi_window(self):
        from fracttalix import _legacy_kwargs_to_config
        cfg = _legacy_kwargs_to_config({"rsi_window": 32})
        assert cfg.rpi_window == 32
