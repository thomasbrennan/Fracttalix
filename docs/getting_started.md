# Getting Started

## Installation

Install from PyPI:

```bash
pip install fracttalix
```

Optional accelerators (install any or all):

```bash
pip install numpy          # FFT, PAC computation, Hilbert transform
pip install scipy          # scipy.signal.hilbert (falls back to numpy)
pip install numba          # JIT compilation for hot loops
pip install matplotlib     # plot_history() dashboard
pip install tqdm           # progress bars in benchmark runs
```

## Basic Usage

```python
from fracttalix import SentinelDetector, SentinelConfig

det = SentinelDetector(SentinelConfig.production())

for value in my_data_stream:
    result = det.update_and_check(value)
    if result["alert"]:
        print(f"Step {result['step']}: {result['alert_reasons']}")
```

`update_and_check()` accepts a scalar `float` (univariate mode) or a `list[float]` (multivariate mode). It always returns a `SentinelResult` — a `dict` subclass with additional convenience methods.

## Factory Presets

`SentinelConfig` ships with four presets covering the most common deployment profiles:

| Preset | `alpha` | `warmup_periods` | Notes |
|--------|---------|-----------------|-------|
| `SentinelConfig.fast()` | 0.3 | 10 | Fastest reaction; higher false-positive rate |
| `SentinelConfig.production()` | 0.1 | 30 | Balanced defaults; recommended starting point |
| `SentinelConfig.sensitive()` | 0.05 | 50 | Catches subtle anomalies; tighter multiplier (2.5) |
| `SentinelConfig.realtime()` | 0.2 | 15 | Quantile-adaptive thresholds |

```python
det = SentinelDetector(SentinelConfig.sensitive())
```

To customize a preset without modifying other fields, use `dataclasses.replace`:

```python
import dataclasses
cfg = dataclasses.replace(SentinelConfig.production(), multiplier=2.0, warmup_periods=50)
det = SentinelDetector(cfg)
```

`SentinelConfig` is a frozen dataclass; all fields are immutable after construction.

## Multi-Stream Usage

`MultiStreamSentinel` manages any number of named streams under a single shared config. It is thread-safe and supports `async` update calls.

```python
from fracttalix import MultiStreamSentinel, SentinelConfig

mss = MultiStreamSentinel(config=SentinelConfig.production())

# Synchronous update
result = mss.update("sensor_42", 3.14)

# Async update
result = await mss.aupdate("sensor_42", 3.14)

# Management
mss.list_streams()              # list all active stream IDs
mss.get_detector("sensor_42")  # retrieve the underlying SentinelDetector
mss.reset_stream("sensor_42")  # reset to factory state, keep config
mss.delete_stream("sensor_42") # remove stream entirely

# Persist and restore all streams
state_json = mss.save_all()
mss.load_all(state_json)
```

Streams are created automatically on first `update()` call with the shared config.

## Multivariate Mode

```python
cfg = SentinelConfig(multivariate=True, n_channels=3)
det = SentinelDetector(config=cfg)
result = det.update_and_check([v1, v2, v3])
```

Mahalanobis distance replaces the scalar EWMA deviation when `multivariate=True`.

## Reading Results

Every `SentinelResult` supports standard dictionary access and a set of typed convenience methods:

```python
result = det.update_and_check(value)

# Core fields
result["alert"]           # bool — any anomaly triggered
result["alert_reasons"]   # list[str] — which conditions fired
result["anomaly_score"]   # float — normalized composite score
result["step"]            # int — monotonic observation counter
result["warmup"]          # bool — True during warmup period

# Three-channel status (V9+)
result.is_cascade_precursor()     # bool — CRITICAL condition active
result.get_channel_status()       # dict — structural/rhythmic/coupling/coherence
result.get_degradation_narrative() # str — human-readable summary

# Physics metrics (V12)
result.get_maintenance_burden()
# {"mu": 0.82, "regime": "TAINTER_WARNING"}

result.get_pac_status()
# {"mean_pac": 0.35, "degradation_rate": 0.19, "pre_cascade_pac": True}

result.get_diagnostic_window()
# {"steps": 47.3, "confidence": "HIGH", "supercompensation": False}

result.is_reversed_sequence()       # bool
result.get_intervention_signature()
# {"score": 0.74, "sequence_type": "REVERSED",
#  "phi_rate": -0.18, "coupling_rate": -0.01}
```

## State Persistence

The detector state can be serialized to JSON and restored, enabling crash recovery and deployment handoff:

```python
json_str = det.save_state()

det2 = SentinelDetector(cfg)
det2.load_state(json_str)
```

See [State Schema](state_schema.md) for the JSON format and V11-to-V12 migration notes.

## Running the Benchmark

```python
from fracttalix import SentinelBenchmark, SentinelConfig

bench = SentinelBenchmark(n=500, config=SentinelConfig.sensitive())
bench.run_suite()
# Reports F1, AUPRC, VUS-PR, mean detection lag for each of the five archetypes
```

## Auto-Tune from Labeled Data

```python
labeled = [(value, is_anomaly), ...]
det = SentinelDetector.auto_tune(data=[], labeled_data=labeled)
```
