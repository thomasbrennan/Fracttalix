# tests/test_benchmark.py
# Tests for benchmark package (generate and evaluate functions)

import os
import sys

import pytest

# Ensure benchmark package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmark.archetypes import ARCHETYPES, generate
from benchmark.metrics import evaluate


class TestGenerate:
    def test_generate_returns_tuple(self):
        result = generate("point")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generate_data_is_list(self):
        data, labels = generate("point")
        assert isinstance(data, list)

    def test_generate_labels_is_list(self):
        data, labels = generate("point")
        assert isinstance(labels, list)

    def test_generate_equal_length(self):
        data, labels = generate("point")
        assert len(data) == len(labels)

    def test_generate_default_n_is_1000(self):
        data, labels = generate("point")
        assert len(data) == 1000

    def test_generate_custom_n(self):
        data, labels = generate("point", n=500)
        assert len(data) == 500

    def test_generate_point_has_anomalies(self):
        data, labels = generate("point")
        n_anomalies = sum(labels)
        assert n_anomalies > 0

    def test_generate_point_roughly_2_percent_anomalies(self):
        """Point archetype: ~2% anomaly rate."""
        data, labels = generate("point", n=1000)
        rate = sum(labels) / len(labels)
        # The archetype places 20 anomalies in 1000 points = 2%
        assert 0.01 <= rate <= 0.05

    def test_generate_unknown_archetype_raises(self):
        with pytest.raises(ValueError):
            generate("unknown_archetype")

    def test_generate_seeded_reproducible(self):
        data1, labels1 = generate("point", seed=42)
        data2, labels2 = generate("point", seed=42)
        assert data1 == data2
        assert labels1 == labels2

    def test_generate_different_seeds_differ(self):
        data1, _ = generate("point", seed=42)
        data2, _ = generate("point", seed=99)
        assert data1 != data2


class TestGenerateAllArchetypes:
    @pytest.mark.parametrize("archetype", ARCHETYPES)
    def test_all_archetypes_generate_without_error(self, archetype):
        data, labels = generate(archetype)
        assert isinstance(data, list)
        assert isinstance(labels, list)
        assert len(data) == len(labels)
        assert len(data) == 1000

    @pytest.mark.parametrize("archetype", ARCHETYPES)
    def test_all_archetypes_have_some_anomalies(self, archetype):
        data, labels = generate(archetype)
        assert sum(labels) > 0, f"Archetype {archetype} has no anomalies"

    @pytest.mark.parametrize("archetype", ARCHETYPES)
    def test_all_archetypes_data_is_float(self, archetype):
        data, labels = generate(archetype)
        assert all(isinstance(v, float) for v in data)

    @pytest.mark.parametrize("archetype", ARCHETYPES)
    def test_all_archetypes_labels_are_binary(self, archetype):
        data, labels = generate(archetype)
        assert all(lbl in (0, 1) for lbl in labels)


class TestEvaluate:
    def test_evaluate_returns_dict(self):
        result = evaluate("point", n=200)
        assert isinstance(result, dict)

    def test_evaluate_has_f1_key(self):
        result = evaluate("point", n=200)
        assert "f1" in result

    def test_evaluate_has_auprc_key(self):
        result = evaluate("point", n=200)
        assert "auprc" in result

    def test_evaluate_has_vus_pr_key(self):
        result = evaluate("point", n=200)
        assert "vus_pr" in result

    def test_evaluate_has_mean_lag_key(self):
        result = evaluate("point", n=200)
        assert "mean_lag" in result

    def test_evaluate_f1_in_range(self):
        result = evaluate("point", n=500)
        assert 0.0 <= result["f1"] <= 1.0

    def test_evaluate_auprc_in_range(self):
        result = evaluate("point", n=500)
        assert 0.0 <= result["auprc"] <= 1.0

    def test_evaluate_vus_pr_in_range(self):
        result = evaluate("point", n=500)
        assert 0.0 <= result["vus_pr"] <= 1.0

    def test_evaluate_mean_lag_nonnegative(self):
        result = evaluate("point", n=500)
        lag = result["mean_lag"]
        # may be inf if no detections, but should not be negative
        if lag != float("inf"):
            assert lag >= 0.0

    def test_evaluate_has_n_anomalies(self):
        result = evaluate("point", n=200)
        assert "n_anomalies" in result
        assert result["n_anomalies"] > 0

    def test_evaluate_has_n_detections(self):
        result = evaluate("point", n=200)
        assert "n_detections" in result
        assert isinstance(result["n_detections"], int)

    def test_evaluate_accepts_custom_config(self):
        from fracttalix import SentinelConfig
        cfg = SentinelConfig.fast()
        result = evaluate("point", config=cfg, n=200)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("archetype", ARCHETYPES)
    def test_evaluate_all_archetypes(self, archetype):
        result = evaluate(archetype, n=300)
        assert isinstance(result, dict)
        assert "f1" in result
        assert "auprc" in result
        assert "vus_pr" in result
        assert "mean_lag" in result
