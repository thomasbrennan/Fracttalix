# tests/conftest.py
# Shared pytest fixtures for Fracttalix V12 test suite.

import math
import pytest
from fracttalix import SentinelDetector, SentinelConfig


@pytest.fixture
def default_detector():
    return SentinelDetector()


@pytest.fixture
def fast_detector():
    return SentinelDetector(config=SentinelConfig.fast())


@pytest.fixture
def warmed_detector():
    """Detector past warmup period (30 observations)."""
    det = SentinelDetector()
    for i in range(35):
        det.update_and_check(float(i % 10))
    return det


@pytest.fixture
def normal_stream():
    """100 normal observations."""
    return [math.sin(i * 0.1) + (i % 7) * 0.01 for i in range(100)]


@pytest.fixture
def spike_stream():
    """Normal stream with clear spikes at indices 70, 71, 72."""
    data = [math.sin(i * 0.1) for i in range(100)]
    data[70] = 50.0
    data[71] = 50.0
    data[72] = 50.0
    return data
