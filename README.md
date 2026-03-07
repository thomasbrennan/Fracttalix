# Fracttalix Sentinel v8.0

https://doi.org/10.5281/zenodo.18859299

**A pipeline-architecture streaming anomaly detector grounded in the Fractal Rhythm Model**
Single-file Python | Zero required dependencies | CC0 public domain

---

## What changed in v8.0

v8.0 is a ground-up rewrite. Prior versions (v7.x) were a single monolithic `update_and_check` method that grew organically — each new detector bolted on, sharing mutable state, interleaved with every other detector. v8.0 replaces that with a formal pipeline architecture: a frozen config object, an independent window bank, and 19 discrete detector steps, each owning its own state and responsibility.

This is not an incremental improvement. The internal model is different in kind.

| | v7.x | v8.0 |
|---|---|---|
| Architecture | Monolithic method | 19-step pipeline |
| Config | Mutable kwargs dict | Frozen dataclass (`SentinelConfig`) |
| Windows | Shared scalar deque | `WindowBank` — named, independent slots |
| Extensibility | Fork the file | Subclass `DetectorStep`, call `register_step` |
| Regime adaptation | Hard alpha reset | Soft multiplicative boost with decay |
| EWS window | Coupled to scalar window | Independent `ews_w` bank slot (T0-01 fix) |
| Picklability | Broken (lambda in pool) | Fixed — all workers are module-level functions |
| RSI label | Mislabeled (Rhythm Stability) | Corrected to SSI (Synchrony Stability Index) |

All v7.x kwargs still work via the legacy mapper. `Detector_7_10` is an alias for `SentinelDetector`.

---

## Quick start

```python
from fracttalix_sentinel_v800 import SentinelDetector

det = SentinelDetector()
for value in your_time_series:
    result = det.update_and_check(value)
    if result["alert"]:
        print(result["step"], result["alert_reasons"])
