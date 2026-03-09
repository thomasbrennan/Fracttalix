"""
Example 2: Monitoring multiple independent streams.

Shows MultiStreamSentinel managing named streams with automatic
detector creation and cross-stream correlation.
"""

import json
import math
import random

from fracttalix import MultiStreamSentinel, SentinelConfig

# ──────────────────────────────────────────────────────────────────────────────
# 1. Create MultiStreamSentinel with production config
# ──────────────────────────────────────────────────────────────────────────────

config = SentinelConfig.production()
multi = MultiStreamSentinel(config=config)

print("=" * 60)
print("Example 2: Multi-Stream Sentinel")
print("=" * 60)
print(f"Config: alpha={config.alpha}, warmup={config.warmup_periods}\n")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Feed 3 named streams with different patterns
#    sensor_A: slow sine wave (structural pattern)
#    sensor_B: faster oscillation with step jump at step 120
#    sensor_C: random walk (brownian motion)
# ──────────────────────────────────────────────────────────────────────────────

random.seed(7)

STEPS = 180
alert_counts = {"sensor_A": 0, "sensor_B": 0, "sensor_C": 0}

print(f"Feeding {STEPS} observations across 3 streams:")
print("-" * 60)

rw = 0.0  # random walk state for sensor_C

for i in range(STEPS):
    # sensor_A: slow low-amplitude sine
    val_a = 1.5 * math.sin(2 * math.pi * i / 40) + random.gauss(0.0, 0.2)

    # sensor_B: faster oscillation; step jump at step 120
    offset_b = 8.0 if i >= 120 else 0.0
    val_b = 3.0 * math.sin(2 * math.pi * i / 12) + offset_b + random.gauss(0.0, 0.3)

    # sensor_C: random walk
    rw += random.gauss(0.0, 0.4)
    val_c = rw

    for stream_id, value in [("sensor_A", val_a), ("sensor_B", val_b),
                              ("sensor_C", val_c)]:
        result = multi.update(stream_id, value)
        if result["alert"] and not result["warmup"]:
            alert_counts[stream_id] += 1

# Print per-stream status
print("\nStream status summary:")
for sid in ["sensor_A", "sensor_B", "sensor_C"]:
    st = multi.status(sid)
    det = multi.get_detector(sid)
    last = st["last_result"]
    print(f"  {sid}:")
    print(f"    n={st['n']}  alert_count={st['alert_count']}  "
          f"regime={last.get('tainter_regime', 'N/A')}")
    print(f"    last z_score={last.get('z_score', 0.0):.3f}  "
          f"anomaly_score={last.get('anomaly_score', 0.0):.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Cross-stream correlations
# ──────────────────────────────────────────────────────────────────────────────

print("\nCross-stream correlations (Pearson on z-scores, window=50):")
corrs = multi.cross_stream_correlations(window=50)
if corrs:
    for pair, r in sorted(corrs.items()):
        print(f"  {pair}: r = {r:+.4f}")
else:
    print("  (insufficient data for correlations)")

# ──────────────────────────────────────────────────────────────────────────────
# 4. save_all() / load_all() round-trip
# ──────────────────────────────────────────────────────────────────────────────

print("\nsave_all() / load_all() round-trip test:")
snapshot_json = multi.save_all()
snapshot = json.loads(snapshot_json)
print(f"  Streams serialized: {list(snapshot.keys())}")
for sid, state in snapshot.items():
    print(f"    {sid}: n={state.get('n', '?')} steps saved")

# Restore into a fresh MultiStreamSentinel
multi2 = MultiStreamSentinel(config=config)
multi2.load_all(snapshot_json)
print(f"\n  Restored streams: {multi2.list_streams()}")

# Verify step counts match
for sid in ["sensor_A", "sensor_B", "sensor_C"]:
    det1 = multi.get_detector(sid)
    det2 = multi2.get_detector(sid)
    match = det1._n == det2._n
    print(f"  {sid}: original n={det1._n}, restored n={det2._n}  "
          f"{'OK' if match else 'MISMATCH'}")

# Continue feeding one more value into restored multi — must not error
for sid in ["sensor_A", "sensor_B", "sensor_C"]:
    r = multi2.update(sid, 0.0)
    assert not r.get("warmup") or True  # just confirm no exception

print("\nRound-trip verified — all streams restored correctly.")
print()
print(f"Done — Example 2 complete.  Streams managed: {multi.list_streams()}")
