# Fracttalix Sentinel v9.0

**Streaming anomaly detection grounded in the Three-Channel Model of Dissipative Network Information Transmission.**

Sentinel ingests one scalar (or multivariate) observation at a time and emits a rich result dictionary on every call — no batching, no retraining, no warmup gap once past the configurable warmup window.

> **Theoretical foundation:** Fractal Rhythm Model Papers 1–6
> DOI: [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)
> License: **CC0** — public domain

---

## Table of Contents

1. [Overview](#overview)
2. [Three-Channel Model](#three-channel-model)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [SentinelConfig — Configuration](#sentinelconfig--configuration)
6. [Pipeline Architecture — 26 Steps](#pipeline-architecture--26-steps)
7. [V9.0 New Features](#v90-new-features)
8. [SentinelResult API](#sentinelresult-api)
9. [MultiStreamSentinel](#multistreamssentinel)
10. [SentinelBenchmark](#sentinelbenchmark)
11. [SentinelServer — REST API](#sentinelserver--rest-api)
12. [CLI Reference](#cli-reference)
13. [Backward Compatibility](#backward-compatibility)
14. [Theoretical Foundation](#theoretical-foundation)
15. [Authors & License](#authors--license)

---

## Overview

Fracttalix Sentinel is a single-file Python library (`fracttalix_sentinel_v900.py`) for real-time streaming anomaly detection. Its design priorities are:

- **Zero external dependencies for core operation** — works on the Python standard library alone; numpy, numba, matplotlib, and tqdm are optional accelerators.
- **Immutable, inspectable configuration** — `SentinelConfig` is a frozen dataclass; every parameter is readable and picklable.
- **Composable pipeline** — 26 `DetectorStep` subclasses execute in sequence; custom steps can be inserted via `register_step`.
- **Three-channel anomaly model** — v9.0 monitors structural properties, broadband rhythmicity, and temporal degradation sequences as independent information channels.
- **Full backward compatibility** — all v7.x and v8.0 call patterns continue to work unchanged.

---

## Three-Channel Model

V9.0 implements the three-channel model from Meta-Kaizen Paper 6:

| Channel | Name | What it monitors |
|---------|------|-----------------|
| **1** | Structural | Network topology as active transmitter — mean, variance, skewness, kurtosis, autocorrelation, stationarity of the input stream |
| **2** | Rhythmic | Broadband multiplexed oscillatory transmission — FFT decomposition into five independent carrier-wave bands and cross-frequency phase-amplitude coupling |
| **3** | Temporal | One-way irreversible carrier wave — temporal sequence and ordering of channel degradation events, diagnostic about regime-change type and severity |

**Degradation cascade logic:**

```
Band anomaly detected  →  Cross-frequency coupling degrades  →
  Structural-rhythmic channels decouple  →  CASCADE PRECURSOR (CRITICAL)
```

The cascade precursor requires three simultaneous conditions: coupling degradation active, structural-rhythmic decoupling active, and at least `cascade_ews_threshold` EWS indicators elevated. This compound signature distinguishes a scale-level reversion event from a local anomaly.

---

## Installation

No package installation required — copy the single file into your project:

```bash
cp fracttalix_sentinel_v900.py myproject/
```

**Optional accelerators (install any or none):**

```bash
pip install numpy          # FFT, covariance, phase-amplitude coupling
pip install numba          # JIT compilation for hot loops
pip install matplotlib     # plot_history() dashboard
pip install tqdm           # progress bars in benchmark
```

**Self-test (65 tests, all expected to pass):**

```bash
python fracttalix_sentinel_v900.py
# or
python fracttalix_sentinel_v900.py --test
```

---

## Quick Start

### Basic scalar stream

```python
from fracttalix_sentinel_v900 import SentinelDetector, SentinelConfig

det = SentinelDetector(SentinelConfig.production())

for value in my_data_stream:
    result = det.update_and_check(value)
    if result["alert"]:
        print(f"Step {result['step']}: alert — {result['alert_reasons']}")
```

### V9.0 three-channel status

```python
result = det.update_and_check(value)

# Boolean cascade test
if result.is_cascade_precursor():
    print("CRITICAL: cascade precursor detected")

# Per-channel health
status = result.get_channel_status()
# {"structural": "healthy", "rhythmic_composite": "degrading",
#  "coupling": "healthy", "coherence": "healthy"}

# Human-readable degradation narrative
print(result.get_degradation_narrative())

# Primary carrier wave (e.g. "mid")
print(result.get_primary_carrier_wave())
```

### Multivariate mode

```python
cfg = SentinelConfig(multivariate=True, n_channels=3)
det = SentinelDetector(config=cfg)
result = det.update_and_check([v1, v2, v3])
```

### Async usage

```python
result = await det.aupdate(value)
```

### Auto-tune from labeled data

```python
labeled = [(value, is_anomaly), ...]
det = SentinelDetector.auto_tune(data=[], labeled_data=labeled)
```

---

## SentinelConfig — Configuration

`SentinelConfig` is a frozen dataclass (`slots=True`). All fields are immutable after construction. Use `dataclasses.replace(cfg, field=value)` to derive a new config.

### Factory presets

| Preset | `alpha` | `warmup` | Notes |
|--------|---------|----------|-------|
| `SentinelConfig.fast()` | 0.3 | 10 | Fastest response, higher FP rate |
| `SentinelConfig.production()` | 0.1 | 30 | Balanced defaults |
| `SentinelConfig.sensitive()` | 0.05 | 50 | Catches subtle anomalies; tight multiplier (2.5) |
| `SentinelConfig.realtime()` | 0.2 | 15 | Quantile-adaptive thresholds |

### Parameter groups

#### A — Core EWMA

| Field | Default | Description |
|-------|---------|-------------|
| `alpha` | `0.1` | EWMA smoothing factor (0 < α ≤ 1). Smaller = slower, more stable. |
| `dev_alpha` | `0.1` | EWMA factor for deviation (volatility) estimation. |
| `multiplier` | `3.0` | Alert threshold = EWMA ± multiplier × dev_ewma. |
| `warmup_periods` | `30` | Observations collected before alerts are issued. |

#### B — Regime Detection

| Field | Default | Description |
|-------|---------|-------------|
| `regime_threshold` | `3.5` | Z-score magnitude that triggers a regime change. |
| `regime_alpha_boost` | `2.0` | Multiplicative boost to alpha during regime transitions (v8 fix δ). |
| `regime_boost_decay` | `0.9` | Decay rate of regime boost per observation. |

#### C — Multivariate

| Field | Default | Description |
|-------|---------|-------------|
| `multivariate` | `False` | Enable Mahalanobis distance mode. |
| `n_channels` | `1` | Number of input channels. |
| `cov_alpha` | `0.05` | EWMA factor for covariance matrix (Woodbury rank-1 update). |

#### D — FRM Metrics

| Field | Default | Description |
|-------|---------|-------------|
| `rpi_window` | `64` | Window for Rhythm Periodicity Index FFT. |
| `rfi_window` | `64` | Window for Rhythm Fractal Index (R/S analysis). |
| `rpi_threshold` | `0.6` | Minimum RPI for "rhythm healthy". |
| `rfi_threshold` | `0.4` | RFI alert threshold (higher = more irregular). |

#### E — Complexity & EWS

| Field | Default | Description |
|-------|---------|-------------|
| `pe_order` | `3` | Permutation Entropy embedding dimension. |
| `pe_window` | `50` | Sliding window for PE computation. |
| `pe_threshold` | `0.05` | PE deviation alert threshold. |
| `ews_window` | `40` | EWS rolling window (independent from scalar window — v8 fix T0-01). |
| `ews_threshold` | `0.6` | EWS score threshold for "approaching critical". |

#### F — Fluid Dynamics

| Field | Default | Description |
|-------|---------|-------------|
| `sti_window` | `20` | Shear-Turbulence Index window. |
| `tps_window` | `30` | Temporal Phase Space reconstruction window. |
| `osc_damp_window` | `20` | Oscillation damping detection window. |
| `osc_threshold` | `1.5` | Oscillation damping alert multiplier. |
| `cpd_window` | `30` | Change-Point Detection comparison window. |
| `cpd_threshold` | `2.0` | CPD alert z-score threshold. |

#### G — Drift / Volatility / Seasonal

| Field | Default | Description |
|-------|---------|-------------|
| `ph_delta` | `0.01` | Page-Hinkley incremental sensitivity. |
| `ph_lambda` | `50.0` | Page-Hinkley cumulative threshold. |
| `var_cusum_k` | `0.5` | VarCUSUM allowance. |
| `var_cusum_h` | `5.0` | VarCUSUM decision threshold. |
| `seasonal_period` | `0` | Seasonal period (0 = auto-detect via FFT). |

#### H — AQB / Scoring / IO

| Field | Default | Description |
|-------|---------|-------------|
| `quantile_threshold_mode` | `False` | Use Adaptive Quantile Baseline instead of EWMA ± mult. |
| `aqb_window` | `200` | Rolling window for AQB quantile estimation. |
| `aqb_q_low` | `0.01` | Lower quantile for AQB. |
| `aqb_q_high` | `0.99` | Upper quantile for AQB. |
| `history_maxlen` | `5000` | Maximum result records kept in memory. |
| `csv_path` | `""` | If non-empty, stream results to this CSV file. |
| `log_level` | `"WARNING"` | Python logging level name. |

#### V9.0 — Frequency Decomposition (Channel 2)

| Field | Default | Description |
|-------|---------|-------------|
| `enable_frequency_decomposition` | `True` | Enable FFT decomposition into five carrier-wave bands. |
| `min_window_for_fft` | `32` | Minimum window before FFT decomposition runs. |

#### V9.0 — Cross-Frequency Coupling

| Field | Default | Description |
|-------|---------|-------------|
| `enable_coupling_detection` | `True` | Enable phase-amplitude coupling measurement. |
| `coupling_degradation_threshold` | `0.3` | Composite coupling score below this triggers `COUPLING_DEGRADATION`. |
| `coupling_trend_window` | `10` | Number of `FrequencyBands` snapshots for coupling trend. |

#### V9.0 — Structural-Rhythmic Coherence

| Field | Default | Description |
|-------|---------|-------------|
| `enable_channel_coherence` | `True` | Enable Channel 1–2 coherence measurement. |
| `coherence_threshold` | `0.4` | Coherence score below this triggers `STRUCTURAL_RHYTHMIC_DECOUPLING`. |
| `coherence_window` | `20` | Rolling window for coherence computation. |

#### V9.0 — Cascade Precursor

| Field | Default | Description |
|-------|---------|-------------|
| `enable_cascade_detection` | `True` | Enable `CASCADE_PRECURSOR` detection. |
| `cascade_ews_threshold` | `2` | Minimum EWS indicators elevated for cascade precursor. |

#### V9.0 — Degradation Sequence Logging

| Field | Default | Description |
|-------|---------|-------------|
| `enable_sequence_logging` | `True` | Enable temporal logging of degradation sequences. |
| `sequence_retention` | `1000` | Maximum completed degradation sequences to retain. |

---

## Pipeline Architecture — 26 Steps

Every call to `update_and_check()` runs all 26 steps in order. Each step reads from and writes to a shared `StepContext.scratch` dictionary. The `WindowBank` provides each step with its own named independent deque.

| # | Step | Added | Description |
|---|------|-------|-------------|
| 1 | `CoreEWMAStep` | v8 | EWMA baseline + deviation; must run first |
| 2 | `StructuralSnapshotStep` | **v9** | Channel 1: mean, variance, skewness, kurtosis, autocorrelation (lag 1 & 2), stationarity score |
| 3 | `FrequencyDecompositionStep` | **v9** | Channel 2: FFT decomposition into 5 carrier-wave bands with power and phase |
| 4 | `CUSUMStep` | v8 | CUSUM persistent shift detection |
| 5 | `RegimeStep` | v8 | Regime change detection with soft alpha boost (fix δ) |
| 6 | `VarCUSUMStep` | v8 | CUSUM on variance — detects volatility changes |
| 7 | `PageHinkleyStep` | v8 | Page-Hinkley drift detector |
| 8 | `STIStep` | v8 | Shear-Turbulence Index |
| 9 | `TPSStep` | v8 | Temporal Phase Space reconstruction |
| 10 | `OscDampStep` | v8 | Oscillation damping detection |
| 11 | `CPDStep` | v8 | Change-Point Detection |
| 12 | `RPIStep` | v8 | Rhythm Periodicity Index (FFT-based) |
| 13 | `RFIStep` | v8 | Rhythm Fractal Index (Hurst / R-S analysis) |
| 14 | `SSIStep` | v8 | Synchronization Stability Index — Kuramoto proxy via FFT phase coherence (`rsi` alias preserved) |
| 15 | `PEStep` | v8 | Permutation Entropy (FRM Axiom 3) |
| 16 | `EWSStep` | v8 | Early Warning Signals — rising variance + AC(1) (FRM Axiom 9; uses independent `ews_w` window) |
| 17 | `AQBStep` | v8 | Adaptive Quantile Baseline — distribution-free thresholds |
| 18 | `SeasonalStep` | v8 | Seasonal decomposition with FFT auto-period detection |
| 19 | `MahalStep` | v8 | Mahalanobis distance (multivariate mode) with Woodbury rank-1 covariance update |
| 20 | `RRSStep` | v8 | Robust Residual Score |
| 21 | `BandAnomalyStep` | **v9** | Per-carrier-wave anomaly detection — catches anomalies invisible to composite detection |
| 22 | `CrossFrequencyCouplingStep` | **v9** | Phase-amplitude coupling between adjacent band pairs; declining score precedes regime change |
| 23 | `ChannelCoherenceStep` | **v9** | Structural-rhythmic channel coherence; decoupling is an independent regime-change signal |
| 24 | `CascadePrecursorStep` | **v9** | Detects tipping cascade precursor — CRITICAL severity; requires all three conditions |
| 25 | `DegradationSequenceStep` | **v9** | Logs temporal ordering of channel degradation events (Channel 3 information) |
| 26 | `AlertReasonsStep` | v8 | Must run last — aggregates all alert signals into `alert_reasons` list |

### Custom steps

```python
from fracttalix_sentinel_v900 import DetectorStep, StepContext, register_step

@register_step
class MyStep(DetectorStep):
    def __init__(self, config):
        self.cfg = config

    def update(self, ctx: StepContext) -> None:
        # read from ctx.scratch, write results back
        ctx.scratch["my_metric"] = ctx.current * 2.0

    def reset(self) -> None:
        pass

    def state_dict(self):
        return {}

    def load_state(self, sd):
        pass
```

---

## V9.0 New Features

### New data structures

#### `FrequencyBands`
Channel 2 decomposition into five carrier-wave bands. Frozen dataclass.

```python
result["frequency_bands"]  # FrequencyBands or None
fb = result["frequency_bands"]
fb.ultra_low_power   # trend component
fb.low_power         # slow oscillation
fb.mid_power         # primary rhythmicity
fb.high_power        # fast fluctuation
fb.ultra_high_power  # noise floor
fb.mid_phase         # phase of mid band (radians)
```

#### `StructuralSnapshot`
Channel 1 structural properties. Frozen dataclass.

```python
ss = result["structural_snapshot"]
ss.mean
ss.variance
ss.skewness
ss.kurtosis
ss.autocorrelation_lag1
ss.autocorrelation_lag2
ss.stationarity_score  # 0.0 = non-stationary, 1.0 = stationary
```

#### `CouplingMatrix`
Cross-frequency phase-amplitude coupling. Frozen dataclass.

```python
cm = result["coupling_matrix"]
cm.ultra_low_to_low           # PAC coefficient
cm.low_to_mid
cm.mid_to_high
cm.high_to_ultra_high
cm.composite_coupling_score   # aggregate health score
cm.coupling_trend             # positive = strengthening, negative = degrading
```

#### `ChannelCoherence`
Structural-rhythmic channel coherence. Frozen dataclass.

```python
cc = result["channel_coherence"]
cc.coherence_score          # 0.0 = decoupled, 1.0 = coherent
cc.structural_change_rate
cc.rhythmic_change_rate
cc.decoupling_trend
```

#### `DegradationSequence`
Temporal ordering of channel degradation events. Frozen dataclass.

```python
ds = result["degradation_sequence"]
ds.first_channel_anomaly        # alert type string
ds.first_anomaly_timestamp
ds.second_channel_anomaly
ds.coupling_degradation_timestamp
ds.decoupling_timestamp
ds.cascade_precursor_timestamp
ds.sequence_pattern             # human-readable narrative
```

### New alert types

| `AlertType` | `AlertSeverity` | Trigger |
|-------------|-----------------|---------|
| `BAND_ANOMALY` | WARNING | Per-carrier-wave anomaly in any of the five bands |
| `COUPLING_DEGRADATION` | WARNING | `composite_coupling_score` below `coupling_degradation_threshold` |
| `STRUCTURAL_RHYTHMIC_DECOUPLING` | ALERT | `coherence_score` below `coherence_threshold` |
| `CASCADE_PRECURSOR` | **CRITICAL** | All three: coupling degraded + decoupling + ≥ N EWS elevated |

Structured `Alert` objects are collected in `result["v9_active_alerts"]`.

---

## SentinelResult API

`SentinelResult` is a `dict` subclass. All v8.0 dictionary access patterns work unchanged.

### Core fields (every result)

| Key | Type | Description |
|-----|------|-------------|
| `"step"` | `int` | Monotonic observation counter |
| `"value"` | `float` | Raw input value |
| `"ewma"` | `float` | Current EWMA estimate |
| `"dev_ewma"` | `float` | Current deviation EWMA |
| `"alert"` | `bool` | Any anomaly triggered |
| `"warmup"` | `bool` | True during warmup period |
| `"anomaly_score"` | `float` | Normalized composite score |
| `"z_score"` | `float` | EWMA deviation in sigma units |
| `"alert_reasons"` | `list[str]` | Human-readable list of triggered conditions |
| `"cascade_precursor_active"` | `bool` | True if CASCADE_PRECURSOR active |
| `"v9_active_alerts"` | `list[Alert]` | Structured v9.0 alert objects |

### V9.0 convenience methods

```python
result.is_cascade_precursor() -> bool
result.get_channel_status() -> dict   # keys: structural, rhythmic_composite, coupling, coherence
result.get_degradation_narrative() -> str
result.get_primary_carrier_wave() -> str  # "ultra_low" | "low" | "mid" | "high" | "ultra_high"
```

---

## MultiStreamSentinel

Thread-safe manager for multiple independent named streams. Each stream is lazily created on first observation and maintains fully independent state.

```python
from fracttalix_sentinel_v900 import MultiStreamSentinel, SentinelConfig

mss = MultiStreamSentinel(config=SentinelConfig.production())

# Sync
result = mss.update("sensor_42", 3.14)

# Async
result = await mss.aupdate("sensor_42", 3.14)

# Management
mss.list_streams()              # ["sensor_42", ...]
mss.get_detector("sensor_42")   # SentinelDetector instance
mss.status("sensor_42")         # {"n": ..., "alert_count": ..., "last_result": ...}
mss.reset_stream("sensor_42")
mss.delete_stream("sensor_42")

# Persistence
state_json = mss.save_all()
mss.load_all(state_json)
```

---

## SentinelBenchmark

Built-in evaluation harness. Generates five labeled anomaly archetypes and reports F1, AUPRC, VUS-PR, mean detection lag, and naive 3-sigma baseline comparison.

### Archetypes

| Archetype | Description |
|-----------|-------------|
| `point` | Sparse large spikes (8σ) at fixed intervals |
| `contextual` | Values anomalous given sinusoidal seasonal context |
| `collective` | Extended runs of moderately elevated values |
| `drift` | Slow linear mean drift starting mid-series |
| `variance` | Sudden 4× variance explosion in second half |

### Usage

```python
from fracttalix_sentinel_v900 import SentinelBenchmark, SentinelConfig

bench = SentinelBenchmark(n=500, config=SentinelConfig.sensitive())
bench.run_suite()

# Or evaluate a single archetype
data, labels = bench.generate("drift")
metrics = bench.evaluate(data, labels)
print(metrics)
# {"archetype": "drift", "f1": 0.82, "auprc": 0.79, "vus_pr": 0.74,
#  "mean_lag": 3.2, "baseline_f1": 0.11}
```

---

## SentinelServer — REST API

Asyncio HTTP server wrapping a `MultiStreamSentinel`. No framework dependencies.

```python
from fracttalix_sentinel_v900 import SentinelServer, SentinelConfig

server = SentinelServer(host="0.0.0.0", port=8765, config=SentinelConfig())
server.run()  # blocking
```

Or from CLI:

```bash
python fracttalix_sentinel_v900.py --serve --host 0.0.0.0 --port 8765
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Version, uptime, stream count |
| `GET` | `/streams` | List all stream IDs |
| `POST` | `/update/<id>` | Feed observation: `{"value": 3.14}` |
| `GET` | `/status/<id>` | Stream stats and last result |
| `DELETE` | `/stream/<id>` | Delete stream |
| `POST` | `/reset/<id>` | Reset stream to factory state |

**Example:**

```bash
curl -X POST http://localhost:8765/update/my_sensor \
     -H "Content-Type: application/json" \
     -d '{"value": 42.0}'
```

---

## CLI Reference

```
python fracttalix_sentinel_v900.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--file`, `-f` | — | CSV file to process (reads first column) |
| `--alpha` | `0.1` | EWMA smoothing factor |
| `--multiplier` | `3.0` | Alert threshold multiplier |
| `--warmup` | `30` | Warmup periods |
| `--benchmark` | — | Run benchmark suite and exit |
| `--serve` | — | Start HTTP server |
| `--host` | `0.0.0.0` | Server host |
| `--port` | `8765` | Server port |
| `--version` | — | Print version and exit |
| `--test` | — | Run 65-test smoke suite and exit |

**Examples:**

```bash
# Process a CSV file
python fracttalix_sentinel_v900.py --file data.csv --alpha 0.2 --multiplier 2.5

# Run benchmark suite
python fracttalix_sentinel_v900.py --benchmark

# Start REST server
python fracttalix_sentinel_v900.py --serve --port 9000

# Run tests
python fracttalix_sentinel_v900.py --test
```

---

## Backward Compatibility

V9.0 is a strict superset of v8.0 and v7.x. No existing step is removed. No existing result key is removed. All existing call patterns continue to work.

### V8.0 root-cause fixes (all preserved in v9.0)

| Fix | Label | Description |
|-----|-------|-------------|
| α | Frozen config | `SentinelConfig` is a frozen dataclass — immutable, picklable, inspectable |
| β | WindowBank | Named independent deques; each consumer owns its window slot |
| γ | Pipeline decomposition | 26 `DetectorStep` subclasses |
| δ | Soft regime boost | Replaces hard alpha reset with multiplicative boost + decay |
| ε | SSI naming | `SSIStep` replaces `RSIStep`; `result["rsi"]` alias preserved |

### V7.x flat-kwargs compatibility

```python
# V7.x pattern — still works via _legacy_kwargs_to_config
from fracttalix_sentinel_v900 import Detector_7_10
det = Detector_7_10(alpha=0.1, multiplier=3.0, warmup_periods=30)
```

`Detector_7_10` is an alias for `SentinelDetector`. All v7.x keyword arguments are mapped to their `SentinelConfig` equivalents automatically, including `rsi_window → rpi_window`.

### State persistence across versions

```python
# Save
json_str = det.save_state()

# Load (forward-compatible; new fields get defaults)
det2 = SentinelDetector(config)
det2.load_state(json_str)
```

---

## Theoretical Foundation

Fracttalix Sentinel implements detection algorithms derived from the **Fractal Rhythm Model (FRM)** — a mathematical framework for understanding rhythmicity in dissipative networks.

| FRM Component | Sentinel Implementation |
|---------------|------------------------|
| FRM Axiom 3 (ordinal pattern complexity) | `PEStep` — Permutation Entropy |
| FRM Axiom 9 (critical slowing down) | `EWSStep` — rising variance + lag-1 autocorrelation |
| Rhythm Periodicity Index | `RPIStep` — FFT-based spectral coherence |
| Rhythm Fractal Index | `RFIStep` — Hurst exponent via R/S analysis |
| Synchronization Stability Index | `SSIStep` — Kuramoto synchronization proxy via FFT phase coherence |
| Three-channel model (Paper 6) | `StructuralSnapshotStep`, `FrequencyDecompositionStep`, `ChannelCoherenceStep`, `CascadePrecursorStep`, `DegradationSequenceStep` |

**DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)

---

## Authors & License

**Authors:** Thomas Brennan & Claude (Anthropic) & Grok (xAI)

**License:** [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/) — no restrictions, no attribution required.

**GitHub:** [https://github.com/thomasbrennan/Fracttalix](https://github.com/thomasbrennan/Fracttalix)

**Version history:**

| Version | File | Notes |
|---------|------|-------|
| v9.0.0 | `fracttalix_sentinel_v900.py` | Three-channel model, 26 steps, 65 tests |
| v8.0.0 | `fracttalix_sentinel_v800.py` | Frozen config, WindowBank, 19-step pipeline |
| v7.11 | `fracttalix_sentinel_v711.py` | — |
| v7.10 | `fracttalix_sentinel_v710.py` | — |
| v7.9 | `fracttalix_sentinel_v79.py` | — |
| v7.8 | `fracttalix_sentinel_v78.py` | — |
| v7.7 | `fracttalix_sentinel_v77.py` | — |
| v7.6 | `fracttalix_sentinel_v76.py` | — |
