"""Three-channel monitoring with frequency decomposition and coupling detection.

Demonstrates the v12.0 three-channel architecture:
  Channel 1 (Structural): Statistical properties
  Channel 2 (Rhythmic): Frequency band decomposition and coupling
  Channel 3 (Temporal): Degradation sequence logging
"""
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fracttalix_sentinel_v1200 import SentinelDetector, SentinelConfig

cfg = SentinelConfig(
    warmup_periods=10,
    min_window_for_fft=16,
    coupling_trend_window=5,
    coherence_window=10,
)
det = SentinelDetector(cfg)

# Phase 1: stable oscillation (steps 0-59)
# Phase 2: disrupted signal (steps 60-99)
for i in range(100):
    if i < 60:
        value = math.sin(2 * math.pi * i / 10) * 5.0
    else:
        # Inject noise and frequency shift to trigger coupling degradation
        value = math.sin(2 * math.pi * i / 3) * 50.0 + (i % 7) * 10.0

    result = det.update_and_check(value)

    # Check three-channel status
    if not result["warmup"] and i % 10 == 0:
        fb = result.get("frequency_bands")
        cm = result.get("coupling_matrix")
        cc = result.get("channel_coherence")

        print(f"\n--- Step {result['step']} ---")
        print(f"  Alert: {result['alert']}")

        if fb:
            print(f"  Frequency bands: low={fb.low_power:.3f} mid={fb.mid_power:.3f} high={fb.high_power:.3f}")

        if cm:
            print(f"  Coupling score: {cm.composite_coupling_score:.3f} (trend: {cm.coupling_trend:+.3f})")
            print(f"  Coupling degradation active: {result.get('coupling_degradation_active', False)}")

        if cc:
            print(f"  Channel coherence: {cc.coherence_score:.3f}")
            print(f"  SR decoupling active: {result.get('sr_decoupling_active', False)}")

        print(f"  Cascade precursor: {result.get('cascade_precursor_active', False)}")

    # Report any v12.0 alerts
    if result["alert"]:
        for reason in result.get("alert_reasons", []):
            print(f"  >> {reason}")
