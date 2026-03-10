# tests/test_utils.py
# Unit tests for fracttalix/_utils.py helper functions.

import math
from unittest.mock import patch

from fracttalix._utils import (
    _mean,
    _np_fft,
    _np_ifft,
    _np_rng,
    _phase_randomize_worker,
    _to_np,
)


class TestMean:
    """Tests for _mean() pure-Python fallback."""

    def test_basic(self):
        assert _mean([1, 2, 3]) == 2.0

    def test_single_value(self):
        assert _mean([7.5]) == 7.5

    def test_empty(self):
        assert _mean([]) == 0.0

    def test_generator(self):
        assert _mean(x for x in [2, 4, 6]) == 4.0

    def test_negative_values(self):
        assert _mean([-1, 1]) == 0.0

    def test_floats(self):
        result = _mean([0.1, 0.2, 0.3])
        assert abs(result - 0.2) < 1e-10


class TestNpRng:
    """Tests for _np_rng() random number generator."""

    def test_returns_object_with_uniform(self):
        rng = _np_rng(42)
        vals = rng.uniform(0.0, 1.0, 5)
        assert len(vals) == 5
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_seed_reproducibility(self):
        vals_a = _np_rng(99).uniform(0.0, 1.0, 10)
        vals_b = _np_rng(99).uniform(0.0, 1.0, 10)
        # With same seed, results must match (within float tolerance)
        for a, b in zip(vals_a, vals_b):
            assert abs(a - b) < 1e-10

    def test_different_seeds_differ(self):
        vals_a = _np_rng(1).uniform(0.0, 10.0, 20)
        vals_b = _np_rng(2).uniform(0.0, 10.0, 20)
        # Very unlikely all 20 match with different seeds
        assert any(abs(a - b) > 1e-10 for a, b in zip(vals_a, vals_b))


class TestToNp:
    """Tests for _to_np() array conversion."""

    def test_list_input(self):
        result = _to_np([1, 2, 3])
        assert len(result) == 3

    def test_tuple_input(self):
        result = _to_np((4.0, 5.0))
        assert len(result) == 2

    def test_preserves_values(self):
        result = _to_np([1.5, 2.5, 3.5])
        for i, expected in enumerate([1.5, 2.5, 3.5]):
            assert abs(float(result[i]) - expected) < 1e-10


class TestNpFft:
    """Tests for _np_fft() FFT wrapper."""

    def test_returns_complex(self):
        result = _np_fft(_to_np([1.0, 0.0, 1.0, 0.0]))
        assert len(result) > 0
        # All values should be complex-compatible
        for v in result:
            complex(v)  # should not raise

    def test_output_length(self):
        data = _to_np([1.0, 2.0, 3.0, 4.0])
        result = _np_fft(data)
        assert len(result) >= 1


class TestNpIfft:
    """Tests for _np_ifft() inverse FFT wrapper."""

    def test_roundtrip_shape(self):
        data = _to_np([1.0, 2.0, 3.0, 4.0])
        fft_result = _np_fft(data)
        ifft_result = _np_ifft(fft_result)
        assert len(ifft_result) >= 1


class TestPhaseRandomizeWorker:
    """Tests for _phase_randomize_worker() surrogate generation."""

    def test_returns_list_of_floats(self):
        data = [math.sin(i * 0.1) for i in range(32)]
        result = _phase_randomize_worker((data, 42))
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_output_length(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = _phase_randomize_worker((data, 7))
        assert len(result) > 0

    def test_reproducible_with_seed(self):
        data = [float(i) for i in range(16)]
        a = _phase_randomize_worker((data, 123))
        b = _phase_randomize_worker((data, 123))
        for x, y in zip(a, b):
            assert abs(x - y) < 1e-10

    def test_different_seeds_differ(self):
        data = [float(i) for i in range(16)]
        a = _phase_randomize_worker((data, 1))
        b = _phase_randomize_worker((data, 2))
        assert any(abs(x - y) > 1e-10 for x, y in zip(a, b))

    def test_no_numpy_fallback(self):
        """Verify the function works when numpy is unavailable (the fixed bug)."""
        data = [1.0, 2.0, 3.0, 4.0]
        # Simulate numpy-free environment by patching _NP
        with patch("fracttalix._utils._NP", False), \
             patch("fracttalix._utils.np", None):
            result = _phase_randomize_worker((data, 42))
            assert isinstance(result, list)
            assert all(isinstance(v, (int, float)) for v in result)
