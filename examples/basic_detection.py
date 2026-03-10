"""Basic anomaly detection with Fracttalix Sentinel.

Generates a synthetic signal with an injected anomaly and runs
the detector to show how alerts are triggered.
"""
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fracttalix_sentinel_v1200 import SentinelDetector, SentinelConfig

# Create a detector with fast warmup
det = SentinelDetector(SentinelConfig(warmup_periods=10))

# Generate a stable signal, then inject a spike at step 50
for i in range(100):
    if i == 50:
        value = 500.0  # anomaly
    else:
        value = math.sin(i * 0.1) * 2.0  # normal oscillation

    result = det.update_and_check(value)

    if result["alert"]:
        print(f"Step {result['step']:3d} | ALERT | score={result['anomaly_score']:.3f} | {result['alert_reasons']}")
    elif not result["warmup"] and i % 20 == 0:
        print(f"Step {result['step']:3d} |  ok   | score={result['anomaly_score']:.3f}")

print(f"\nTotal observations: {det._n}")
