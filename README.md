# Fracttalix Sentinel v8.0

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
```

---

## Config presets

```python
from fracttalix_sentinel_v800 import SentinelConfig, SentinelDetector

det = SentinelDetector(config=SentinelConfig.production())  # default balanced config
det = SentinelDetector(config=SentinelConfig.fast())        # high alpha, low warmup
det = SentinelDetector(config=SentinelConfig.sensitive())   # low alpha, tight multiplier
det = SentinelDetector(config=SentinelConfig.realtime())    # quantile-adaptive thresholds
```

Customize any field via `dataclasses.replace` — the config is immutable so the original is never mutated:

```python
import dataclasses
cfg = dataclasses.replace(SentinelConfig.production(), multiplier=2.5, ews_window=60)
det = SentinelDetector(config=cfg)
```

---

## The pipeline

Each observation passes through 19 `DetectorStep` subclasses in order. Any step can set `ctx.scratch["alert"] = True`. The final step (`AlertReasonsStep`) collects a named list of which detectors fired.

| # | Step | What it detects |
|---|---|---|
| 1 | `CoreEWMAStep` | EWMA baseline + z-score threshold |
| 2 | `CUSUMStep` | Persistent mean shift |
| 3 | `RegimeStep` | Sudden regime change (soft alpha boost) |
| 4 | `VarCUSUMStep` | Volatility explosion |
| 5 | `PageHinkleyStep` | Slow gradual drift |
| 6 | `STIStep` | Shear-turbulence index (fluid dynamics analog) |
| 7 | `TPSStep` | Temporal phase space attractor deformation |
| 8 | `OscDampStep` | Oscillation amplitude shift |
| 9 | `CPDStep` | Two-window change point detection |
| 10 | `RPIStep` | Rhythm Periodicity Index (FRM Axiom 6) |
| 11 | `RFIStep` | Rhythm Fractal Index / Hurst exponent (FRM Axiom 8) |
| 12 | `SSIStep` | Synchrony Stability Index / Kuramoto order (FRM Axiom 10) |
| 13 | `PEStep` | Permutation Entropy — contextual deviation (FRM Axiom 3) |
| 14 | `EWSStep` | Early Warning Signals — critical slowing down (FRM Axiom 9) |
| 15 | `AQBStep` | Adaptive Quantile Baseline |
| 16 | `SeasonalStep` | Per-phase EWMA seasonal baseline |
| 17 | `MahalStep` | Mahalanobis distance (multivariate mode) |
| 18 | `RRSStep` | Rhythm Resonance Score (FRM Axiom 11) |
| 19 | `AlertReasonsStep` | Aggregate named reasons list |

### Result dict keys (selection)

```python
result = det.update_and_check(value)

result["alert"]           # bool — any detector fired
result["alert_reasons"]   # list[str] — which detectors fired
result["warmup"]          # bool — still in warmup period
result["z_score"]         # float
result["anomaly_score"]   # float in [0, 1]
result["ewma"]            # current EWMA baseline
result["ews_regime"]      # "stable" | "approaching" | "critical"
result["rpi"]             # Rhythm Periodicity Index
result["rfi"]             # Rhythm Fractal Index
result["ssi"]             # Synchrony Stability Index
result["pe"]              # Permutation Entropy
result["hurst"]           # Hurst exponent
result["cusum_hi"]        # CUSUM upper arm
result["cpd_score"]       # change point z-score
```

---

## Multi-stream

```python
from fracttalix_sentinel_v800 import MultiStreamSentinel

mss = MultiStreamSentinel()
result = mss.update("btc_usd", 67400.0)
result = mss.update("eth_usd", 3510.0)

mss.status("btc_usd")      # n, alert_count, last_result
mss.reset_stream("btc_usd")
mss.delete_stream("eth_usd")

snapshot = mss.save_all()   # JSON string
mss.load_all(snapshot)      # restore
```

---

## Extending the pipeline

```python
from fracttalix_sentinel_v800 import DetectorStep, SentinelDetector, register_step, StepContext

@register_step
class MyStep(DetectorStep):
    def __init__(self, config): ...
    def update(self, ctx: StepContext) -> None:
        if some_condition(ctx.current):
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
    def reset(self): ...
    def state_dict(self): return {}
    def load_state(self, sd): pass

# Inject into a detector
from fracttalix_sentinel_v800 import _build_default_pipeline, SentinelConfig
cfg = SentinelConfig()
pipeline = _build_default_pipeline(cfg) + [MyStep(cfg)]
det = SentinelDetector(config=cfg, steps=pipeline)
```

---

## Auto-tune

```python
# Grid search alpha/multiplier to maximise F1 on labeled data
det = SentinelDetector.auto_tune(
    data=my_series,
    labeled_data=list(zip(my_series, my_labels))  # (value, bool) pairs
)
```

---

## State persistence

```python
snapshot = det.save_state()          # JSON string — full pipeline state
det2 = SentinelDetector(config=cfg)
det2.load_state(snapshot)            # restore exactly
```

---

## CSV streaming

```python
cfg = SentinelConfig(csv_path="alerts.csv")
det = SentinelDetector(config=cfg)
# every result is written to CSV automatically
det.close()  # flush and close file
```

---

## Async

```python
result = await det.aupdate(value)
result = await mss.aupdate("stream_id", value)
```

---

## HTTP server

```python
from fracttalix_sentinel_v800 import SentinelServer
SentinelServer(host="0.0.0.0", port=8765).run()
```

```
POST /update/<stream_id>    {"value": 1.23}
GET  /status/<stream_id>
GET  /streams
GET  /health
POST /reset/<stream_id>
DELETE /stream/<stream_id>
```

---

## Benchmark

```python
from fracttalix_sentinel_v800 import SentinelBenchmark
SentinelBenchmark().run_suite()
```

Five labeled archetypes — point, contextual, collective, drift, variance — scored on F1, AUPRC, VUS-PR, mean detection lag, and naive 3-sigma baseline comparison.

---

## CLI

```bash
python fracttalix_sentinel_v800.py --file data.csv --alpha 0.1 --multiplier 3.0
python fracttalix_sentinel_v800.py --benchmark
python fracttalix_sentinel_v800.py --serve --port 8765
python fracttalix_sentinel_v800.py --test
```

---

## Theoretical foundation

Sentinel is grounded in the **Fractal Rhythm Model** (Brennan & Grok 4, 2026) — 11 axioms describing how rhythm, periodicity, fractal structure, and synchrony govern the health and stability of complex systems. RPI, RFI, SSI, EWS, RRS, and PE each map to a specific FRM axiom.

See the Papers branch: https://github.com/thomasbrennan/Fracttalix

---

## Requirements

- Python 3.10+ (uses `dataclasses(slots=True)`)
- `numpy` — required for RPI, RFI, SSI, RRS, SeasonalStep, MahalStep, Benchmark
- `matplotlib` — optional, for `plot_history()`
- `numba` — optional, JIT acceleration
- `tqdm` — optional, progress bars

Zero required dependencies for core EWMA + CUSUM operation.

---

## License

CC0 — public domain. No attribution required.
