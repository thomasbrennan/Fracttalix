"""
Example 1: Basic single-stream anomaly detection.

Demonstrates the minimal API: create a detector, feed values one at a time,
check for alerts. Shows alert_reasons and anomaly_score.
"""

import math
import random

from fracttalix import SentinelDetector, SentinelConfig

# ──────────────────────────────────────────────────────────────────────────────
# 1. Create detector with production config
# ──────────────────────────────────────────────────────────────────────────────

config = SentinelConfig.production()
det = SentinelDetector(config=config)

print("=" * 60)
print("Example 1: Basic Single-Stream Anomaly Detection")
print("=" * 60)
print(f"Detector: {det}")
print(f"Config:   alpha={config.alpha}, multiplier={config.multiplier}, "
      f"warmup={config.warmup_periods}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Feed 200 normal values: sine wave + small Gaussian noise
# ──────────────────────────────────────────────────────────────────────────────

random.seed(42)
FREQ = 2 * math.pi / 30  # one cycle every 30 steps
AMPLITUDE = 2.0
NOISE_SD = 0.3

print("Phase 1: 200 normal observations (sine + noise)")
alerts_during_normal = 0
for step in range(200):
    value = AMPLITUDE * math.sin(FREQ * step) + random.gauss(0.0, NOISE_SD)
    result = det.update_and_check(value)
    if result["alert"] and not result["warmup"]:
        alerts_during_normal += 1

print(f"  Steps processed: 200")
print(f"  False-positive alerts: {alerts_during_normal}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Inject 3 obvious spikes
# ──────────────────────────────────────────────────────────────────────────────

print("Phase 2: 3 obvious spikes injected")
print("-" * 60)

spike_values = [25.0, -20.0, 30.0]
for i, spike_val in enumerate(spike_values):
    result = det.update_and_check(spike_val)
    step_num = result["step"]
    if result["alert"]:
        reasons = result["alert_reasons"]
        score = result["anomaly_score"]
        z = result["z_score"]
        print(f"  ALERT at step {step_num:>3d}  value={spike_val:>7.2f}  "
              f"z={z:>6.2f}  score={score:.4f}")
        print(f"    alert_reasons: {reasons}")
    else:
        print(f"  No alert at step {step_num} for spike={spike_val}")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 4. Feed 10 more normal values after spikes, show recovery
# ──────────────────────────────────────────────────────────────────────────────

print("Phase 3: 10 normal values after spikes (EWMA recovery)")
for step in range(10):
    value = AMPLITUDE * math.sin(FREQ * (210 + step)) + random.gauss(0.0, NOISE_SD)
    result = det.update_and_check(value)
    if result["alert"]:
        print(f"  Alert at step {result['step']}: {result['alert_reasons']}")

print("  Done — detector recovered to normal baseline\n")

# ──────────────────────────────────────────────────────────────────────────────
# 5. get_diagnostic_window() — collapse time estimate
# ──────────────────────────────────────────────────────────────────────────────

last_result = result  # reuse last result from recovery phase
dw = last_result.get_diagnostic_window()
print("Diagnostic Window (time-to-collapse estimate):")
print(f"  steps (expected):    {dw['steps']}")
print(f"  steps (pessimistic): {dw['steps_pessimistic']}")
print(f"  steps (optimistic):  {dw['steps_optimistic']}")
print(f"  confidence:          {dw['confidence']}")
print(f"  supercompensation:   {dw['supercompensation']}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 6. get_maintenance_burden() — Tainter regime
# ──────────────────────────────────────────────────────────────────────────────

mb = last_result.get_maintenance_burden()
print("Maintenance Burden (Tainter regime):")
print(f"  mu (burden):  {mb['mu']:.4f}  (0=healthy, 1=critical)")
print(f"  regime:       {mb['regime']}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 7. Detector summary
# ──────────────────────────────────────────────────────────────────────────────

print(f"Total observations: {len(det)}")
print(f"History length:     {len(list(det._history))}")
print()
print("Done — Example 1 complete.")
