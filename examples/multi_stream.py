"""Multi-stream monitoring with MultiStreamSentinel.

Monitors three independent streams simultaneously, each with
different signal characteristics.
"""
import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fracttalix_sentinel_v1200 import MultiStreamSentinel, SentinelConfig

cfg = SentinelConfig(warmup_periods=10)
mss = MultiStreamSentinel(config=cfg)

for i in range(80):
    # Stream A: stable sinusoid
    val_a = math.sin(i * 0.1) * 2.0

    # Stream B: trend with anomaly at step 40
    val_b = i * 0.5 + (200.0 if i == 40 else 0.0)

    # Stream C: random-looking via modular arithmetic
    val_c = ((i * 7 + 3) % 13) - 6.0

    for name, val in [("sensor_A", val_a), ("sensor_B", val_b), ("sensor_C", val_c)]:
        result = mss.update(name, val)
        if result["alert"]:
            print(f"Step {i:3d} | {name:10s} | ALERT | {result['alert_reasons']}")

# Print final status for each stream
print("\n--- Stream Status ---")
for name in mss.list_streams():
    status = mss.status(name)
    print(f"  {name}: {status['n']} observations, last_alert={status.get('last_alert', 'none')}")
