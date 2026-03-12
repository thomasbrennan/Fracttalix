# Fracttalix Sentinel v12.1

[![Tests](https://github.com/thomasbrennan/Fracttalix/actions/workflows/tests.yml/badge.svg)](https://github.com/thomasbrennan/Fracttalix/actions/workflows/tests.yml)
[![Python 3.9–3.12](https://img.shields.io/badge/python-3.9%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: CC0-1.0](https://img.shields.io/badge/license-CC0--1.0-brightgreen.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![PyPI version](https://img.shields.io/pypi/v/fracttalix.svg)](https://pypi.org/project/fracttalix/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18859299.svg)](https://doi.org/10.5281/zenodo.18859299)

**Streaming anomaly detection grounded in the Three-Channel Model of Dissipative Network Information Transmission — extended with four physics-derived capabilities from Session 36.**

Sentinel ingests one scalar (or multivariate) observation at a time and emits a rich result dictionary on every call — no batching, no retraining, no warmup gap once past the configurable warmup window.

> **Theoretical foundation:** Fractal Rhythm Model Papers 1–6
> DOI: [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)
> License: **CC0** — public domain

**[Quickstart Tutorial](examples/00_quickstart.ipynb)** | **[Full documentation](https://thomasbrennan.github.io/Fracttalix)** | **[Examples](examples/)** | **[CHANGELOG](CHANGELOG.md)**

---

## Table of Contents

1. [Overview](#overview)
2. [Three-Channel Model](#three-channel-model)
3. [V12.1 Changes](#v121-changes)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [SentinelConfig — Configuration](#sentinelconfig--configuration)
7. [Pipeline Architecture — 37 Steps](#pipeline-architecture--37-steps)
8. [Alert Types and Data Structures](#alert-types-and-data-structures)
9. [SentinelResult API](#sentinelresult-api)
10. [MultiStreamSentinel](#multistreamssentinel)
11. [SentinelBenchmark](#sentinelbenchmark)
12. [SentinelServer — REST API](#sentinelserver--rest-api)
13. [CLI Reference](#cli-reference)
14. [Backward Compatibility](#backward-compatibility)
15. [Theoretical Foundation](#theoretical-foundation)
16. [How This Repository Works](#how-this-repository-works--a-guide-to-everything-here)
17. [Authors & License](#authors--license)

---

## Overview

Fracttalix Sentinel is a Python package (`pip install fracttalix`) for real-time streaming anomaly detection. Its design priorities are:

- **Zero external dependencies for core operation** — works on the Python standard library alone; numpy, scipy, numba, matplotlib, and tqdm are optional accelerators.
- **Immutable, inspectable configuration** — `SentinelConfig` is a frozen dataclass; every parameter is readable and picklable.
- **Composable pipeline** — 37 `DetectorStep` subclasses execute in sequence.
- **Three-channel anomaly model** — monitors structural properties, broadband rhythmicity, and temporal degradation sequences as independent information channels.
- **Physics-derived collapse dynamics** — maintenance burden, PAC pre-cascade detection, diagnostic window estimation, and reversed sequence detection derived from the Kuramoto synchronization framework.
- **Full backward compatibility** — all v7.x, v8.0, v9.0, and v10.0 call patterns continue to work unchanged.

---

## Three-Channel Model

Implemented from Meta-Kaizen Paper 6:

| Channel | Name | What it monitors |
|---------|------|-----------------|
| **1** | Structural | Network topology as active transmitter — mean, variance, skewness, kurtosis, autocorrelation, stationarity |
| **2** | Rhythmic | Broadband multiplexed oscillatory transmission — FFT decomposition into five carrier-wave bands and cross-frequency phase-amplitude coupling |
| **3** | Temporal | One-way irreversible carrier wave — temporal sequence and ordering of channel degradation events |

**Degradation cascade logic (v9.0):**

```
Band anomaly detected  →  Cross-frequency coupling degrades  →
  Structural-rhythmic channels decouple  →  CASCADE PRECURSOR (CRITICAL)
```

**Extended diagnostics (v10.0):**

```
PAC pre-cascade detected  →  Δt window opens  →  Maintenance burden μ → 1  →
  Coupling rate dκ̄/dt negative  →  Collapse imminent
```

---

## V12.1 Changes

### Bug Fixes

- **VarCUSUM non-reset** (`VarCUSUMStep`): accumulators `s_hi`/`s_lo` now re-arm
  after each threshold crossing. Normal alert rate: 97% → 35.6%.
- **ChannelCoherence unit mismatch** (`ChannelCoherenceStep`): replaced
  rate-difference formula with Pearson correlation (scale-invariant). Normal data
  now scores ~0.5 above threshold.

### v12.1 Benchmark (n=1000, seed=42)

| Archetype   |  F1   | Normal alert rate |
|-------------|-------|-------------------|
| point       | 0.415 | —                 |
| contextual  | 0.247 | —                 |
| collective  | 0.239 | —                 |
| drift       | 0.723 | —                 |
| variance    | 0.876 | —                 |
| **normal**  | —     | **35.6%**         |

---

## Physics-Derived Capabilities

Four capabilities derived from the Kuramoto synchronization framework (added v10.0):

### 1. Maintenance Burden μ (Tainter Regime Detection)

```
μ = N · κ̄ · E_coupling / P_throughput
```

When μ → 1, the network spends all energy on coupling maintenance with zero adaptive reserve — the Tainter collapse condition.

| μ range | Regime | Meaning |
|---------|--------|---------|
| < 0.5 | `HEALTHY` | Full adaptive reserve |
| 0.5 – 0.75 | `REDUCED_RESERVE` | Adaptive capacity diminishing |
| 0.75 – 0.9 | `TAINTER_WARNING` | Approaching critical burden |
| ≥ 0.9 | `TAINTER_CRITICAL` | Spending all energy on maintenance |

```python
mb = result.get_maintenance_burden()
# {"mu": 0.82, "regime": "TAINTER_WARNING"}
```

### 2. PAC Pre-Cascade Detection (Extended Diagnostic Window)

Phase-Amplitude Coupling (PAC) measures the depth of nonlinear coupling architecture — the structural memory of the network. PAC degrades **before** mean coupling strength κ̄ measurably decreases, providing an earlier warning signal than the v9.0 cascade precursor.

Method: Modulation Index (Tort et al. 2010) across 6 slow-phase/fast-amplitude band pairs.

```python
pac = result.get_pac_status()
# {"mean_pac": 0.41, "degradation_rate": 0.18, "pre_cascade_pac": True}
```

### 3. Diagnostic Window Δt Estimation (Time-to-Collapse)

```
Δt = (κ̄ - κ_c) / |dκ̄/dt|
```

The estimated number of observations remaining before coherence collapse, given current coupling strength and its rate of change. Sentinel stops just detecting that collapse is coming and starts estimating **when**.

- Only active when κ̄ > κ_c and dκ̄/dt < 0
- Confidence graded: `HIGH` / `MEDIUM` / `LOW` based on rate stability
- Detects **supercompensation** (adaptive recovery in progress)

```python
dw = result.get_diagnostic_window()
# {"steps": 47.3, "confidence": "HIGH", "supercompensation": False}
```

### 4. Reversed Sequence Detection (Intervention Signature)

The thermodynamic arrow of network collapse is irreversible: **coupling degrades before coherence collapses**. A reversed sequence — coherence collapsing before coupling degrades — indicates:

1. Measurement error
2. Non-universality class membership
3. **Deliberate external intervention** (a civilization being collapsed vs. one that collapses)

```python
if result.is_reversed_sequence():
    sig = result.get_intervention_signature()
    # {"score": 0.74, "sequence_type": "REVERSED",
    #  "phi_rate": -0.18, "coupling_rate": -0.01}
```

---

## Installation

```bash
pip install fracttalix            # zero dependencies — pure stdlib core
pip install fracttalix[fast]      # + numpy, scipy
pip install fracttalix[full]      # + numpy, scipy, numba, matplotlib, tqdm
```

**Optional accelerators (install any or none):**

```bash
pip install numpy          # FFT, PAC computation, Hilbert transform
pip install scipy          # scipy.signal.hilbert (falls back to numpy)
pip install numba          # JIT compilation for hot loops
pip install matplotlib     # plot_history() dashboard
pip install tqdm           # progress bars in benchmark
```

**Run tests (405 tests, all expected to pass):**

```bash
pip install fracttalix[dev]       # pytest, ruff, mypy, mkdocs
pytest
```

---

## Quick Start

> **New to Sentinel?** Start with the [Quickstart Tutorial](examples/00_quickstart.ipynb) — a hands-on notebook that walks through three-channel detection, physics diagnostics, and tuning tips in under five minutes. GitHub renders it in-browser, no clone required.

### Basic scalar stream

```python
from fracttalix import SentinelDetector, SentinelConfig

det = SentinelDetector(SentinelConfig.production())

for value in my_data_stream:
    result = det.update_and_check(value)
    if result["alert"]:
        print(f"Step {result['step']}: {result['alert_reasons']}")
```

### Collapse dynamics

```python
result = det.update_and_check(value)

# Tainter regime
mb = result.get_maintenance_burden()
if mb["regime"] == "TAINTER_CRITICAL":
    print(f"Tainter critical: μ={mb['mu']:.2f}")

# PAC pre-cascade (earlier warning than cascade precursor)
pac = result.get_pac_status()
if pac["pre_cascade_pac"]:
    print("PAC pre-cascade: coupling architecture degrading")

# Time to collapse estimate
dw = result.get_diagnostic_window()
if dw["steps"] is not None:
    print(f"Δt ≈ {dw['steps']:.0f} steps ({dw['confidence']} confidence)")
if dw["supercompensation"]:
    print("Supercompensation detected — adaptive recovery in progress")

# Intervention signature
if result.is_reversed_sequence():
    sig = result.get_intervention_signature()
    print(f"Reversed sequence — intervention score {sig['score']:.2f}")
```

### V9.0 three-channel status

```python
if result.is_cascade_precursor():
    print("CRITICAL: cascade precursor")

status = result.get_channel_status()
# {"structural": "healthy", "rhythmic_composite": "degrading",
#  "coupling": "healthy", "coherence": "healthy"}

print(result.get_degradation_narrative())
print(result.get_primary_carrier_wave())   # "mid", "low", etc.
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
| `alpha` | `0.1` | EWMA smoothing factor (0 < α ≤ 1) |
| `dev_alpha` | `0.1` | EWMA factor for deviation estimation |
| `multiplier` | `3.0` | Alert threshold = EWMA ± multiplier × dev_ewma |
| `warmup_periods` | `30` | Observations before alerts are issued |

#### B — Regime Detection

| Field | Default | Description |
|-------|---------|-------------|
| `regime_threshold` | `3.5` | Z-score magnitude triggering regime change |
| `regime_alpha_boost` | `2.0` | Multiplicative alpha boost during transitions |
| `regime_boost_decay` | `0.9` | Decay rate of regime boost per observation |

#### C — Multivariate

| Field | Default | Description |
|-------|---------|-------------|
| `multivariate` | `False` | Enable Mahalanobis distance mode |
| `n_channels` | `1` | Number of input channels |
| `cov_alpha` | `0.05` | EWMA factor for covariance (Woodbury rank-1) |

#### D — FRM Metrics

| Field | Default | Description |
|-------|---------|-------------|
| `rpi_window` | `64` | Rhythm Periodicity Index FFT window |
| `rfi_window` | `64` | Rhythm Fractal Index R/S window |
| `rpi_threshold` | `0.6` | Minimum RPI for "rhythm healthy" |
| `rfi_threshold` | `0.4` | RFI alert threshold |

#### E — Complexity & EWS

| Field | Default | Description |
|-------|---------|-------------|
| `pe_order` | `3` | Permutation Entropy embedding dimension |
| `pe_window` | `50` | PE sliding window |
| `pe_threshold` | `0.05` | PE deviation alert threshold |
| `ews_window` | `40` | EWS rolling window (independent from scalar window) |
| `ews_threshold` | `0.6` | EWS "approaching critical" threshold |

#### F — Fluid Dynamics

| Field | Default | Description |
|-------|---------|-------------|
| `sti_window` | `20` | Shear-Turbulence Index window |
| `tps_window` | `30` | Temporal Phase Space window |
| `osc_damp_window` | `20` | Oscillation damping window |
| `osc_threshold` | `1.5` | Oscillation damping alert multiplier |
| `cpd_window` | `30` | Change-Point Detection window |
| `cpd_threshold` | `2.0` | CPD alert z-score threshold |

#### G — Drift / Volatility / Seasonal

| Field | Default | Description |
|-------|---------|-------------|
| `ph_delta` | `0.01` | Page-Hinkley incremental sensitivity |
| `ph_lambda` | `50.0` | Page-Hinkley cumulative threshold |
| `var_cusum_k` | `0.5` | VarCUSUM allowance |
| `var_cusum_h` | `5.0` | VarCUSUM decision threshold |
| `seasonal_period` | `0` | Seasonal period (0 = auto-detect via FFT) |

#### H — AQB / Scoring / IO

| Field | Default | Description |
|-------|---------|-------------|
| `quantile_threshold_mode` | `False` | Use Adaptive Quantile Baseline |
| `aqb_window` | `200` | AQB quantile estimation window |
| `aqb_q_low` | `0.01` | Lower AQB quantile |
| `aqb_q_high` | `0.99` | Upper AQB quantile |
| `history_maxlen` | `5000` | Maximum result records in memory |
| `csv_path` | `""` | Stream results to CSV if non-empty |
| `log_level` | `"WARNING"` | Python logging level |

#### V9.0 — Frequency Decomposition

| Field | Default | Description |
|-------|---------|-------------|
| `enable_frequency_decomposition` | `True` | Enable FFT into five carrier-wave bands |
| `min_window_for_fft` | `32` | Minimum window before FFT runs |

#### V9.0 — Cross-Frequency Coupling

| Field | Default | Description |
|-------|---------|-------------|
| `enable_coupling_detection` | `True` | Enable phase-amplitude coupling |
| `coupling_degradation_threshold` | `0.3` | Composite score below this → `COUPLING_DEGRADATION` |
| `coupling_trend_window` | `10` | FrequencyBands snapshots for coupling trend |

#### V9.0 — Structural-Rhythmic Coherence

| Field | Default | Description |
|-------|---------|-------------|
| `enable_channel_coherence` | `True` | Enable Channel 1–2 coherence |
| `coherence_threshold` | `0.4` | Score below this → `STRUCTURAL_RHYTHMIC_DECOUPLING` |
| `coherence_window` | `20` | Rolling coherence window |

#### V9.0 — Cascade Precursor & Sequence Logging

| Field | Default | Description |
|-------|---------|-------------|
| `enable_cascade_detection` | `True` | Enable `CASCADE_PRECURSOR` detection |
| `cascade_ews_threshold` | `2` | Minimum EWS indicators elevated |
| `enable_sequence_logging` | `True` | Enable temporal degradation sequence logging |
| `sequence_retention` | `1000` | Maximum sequences to retain |

---

## Pipeline Architecture — 37 Steps

Every call to `update_and_check()` runs all 37 steps in order. Steps read from and write to a shared `StepContext.scratch` dictionary.

| # | Step | Version | Description |
|---|------|---------|-------------|
| 1 | `CoreEWMAStep` | v8 | EWMA baseline + deviation; must run first |
| 2 | `StructuralSnapshotStep` | v9 | Channel 1: mean, variance, skewness, kurtosis, autocorrelation, stationarity |
| 3 | `FrequencyDecompositionStep` | v9 | Channel 2: FFT into 5 carrier-wave bands with power and phase |
| 4 | `CUSUMStep` | v8 | CUSUM persistent shift detection |
| 5 | `RegimeStep` | v8 | Regime change with soft alpha boost |
| 6 | `VarCUSUMStep` | v8 | CUSUM on variance |
| 7 | `PageHinkleyStep` | v8 | Page-Hinkley drift detector |
| 8 | `STIStep` | v8 | Shear-Turbulence Index |
| 9 | `TPSStep` | v8 | Temporal Phase Space reconstruction |
| 10 | `OscDampStep` | v8 | Oscillation damping detection |
| 11 | `CPDStep` | v8 | Change-Point Detection |
| 12 | `RPIStep` | v8 | Rhythm Periodicity Index (FFT) |
| 13 | `RFIStep` | v8 | Rhythm Fractal Index (Hurst / R-S) |
| 14 | `SSIStep` | v8 | Synchronization Stability Index — Kuramoto proxy (`rsi` alias preserved) |
| 15 | `PEStep` | v8 | Permutation Entropy (FRM Axiom 3) |
| 16 | `EWSStep` | v8 | Early Warning Signals — variance + AC(1) (FRM Axiom 9) |
| 17 | `AQBStep` | v8 | Adaptive Quantile Baseline |
| 18 | `SeasonalStep` | v8 | Seasonal decomposition with FFT auto-period |
| 19 | `MahalStep` | v8 | Mahalanobis distance (multivariate) |
| 20 | `RRSStep` | v8 | Robust Residual Score |
| 21 | `BandAnomalyStep` | v9 | Per-carrier-wave anomaly invisible to composite |
| 22 | `CrossFrequencyCouplingStep` | v9 | PAC coupling matrix + `COUPLING_DEGRADATION` alert |
| 23 | `ChannelCoherenceStep` | v9 | Structural-rhythmic coherence + `SR_DECOUPLING` alert |
| 24 | `CascadePrecursorStep` | v9 | `CASCADE_PRECURSOR` — CRITICAL; requires all three conditions |
| 25 | `DegradationSequenceStep` | v9 | Channel 3: temporal degradation sequence log |
| 26 | `ThroughputEstimationStep` | **v10** | P_throughput from band amplitudes; populates `band_amplitudes`, `band_powers`, `node_count`, `mean_coupling_strength` |
| 27 | `MaintenanceBurdenStep` | **v10** | μ = N·κ̄ / P_throughput → Tainter regime classification |
| 28 | `PhaseExtractionStep` | **v10** | FFT bandpass + Hilbert transform → instantaneous phase per band |
| 29 | `PACCoefficientStep` | **v10** | Modulation Index (Tort 2010) across 6 slow/fast band pairs |
| 30 | `PACDegradationStep` | **v10** | Rolling PAC history → `pac_degradation_rate`, `pre_cascade_pac` |
| 31 | `CriticalCouplingEstimationStep` | **v10** | κ_c = 2/(π·g(ω₀)) from power-weighted frequency spread |
| 32 | `CouplingRateStep` | **v10** | dκ̄/dt from rolling coupling history |
| 33 | `DiagnosticWindowStep` | **v10** | Δt = (κ̄−κ_c)/|dκ̄/dt|; confidence grading; supercompensation |
| 34 | `KuramotoOrderStep` | **v10** | Φ = |mean(e^iθ_k)| — phase coherence independent of κ̄ |
| 35 | `SequenceOrderingStep` | **v10** | COUPLING_FIRST / COHERENCE_FIRST / SIMULTANEOUS / STABLE per step |
| 36 | `ReversedSequenceStep` | **v10** | Reversed thermodynamic sequence → intervention signature |
| 37 | `AlertReasonsStep` | v8 | Must run last — aggregates all alert signals |

### Custom steps

```python
from fracttalix import DetectorStep, StepContext, register_step

@register_step
class MyStep(DetectorStep):
    def __init__(self, config):
        self.cfg = config

    def update(self, ctx: StepContext) -> None:
        ctx.scratch["my_metric"] = ctx.current * 2.0

    def reset(self) -> None:
        pass

    def state_dict(self):
        return {}

    def load_state(self, sd):
        pass
```

---

## V9.0 Alert Types and Data Structures

### Frozen data structures

| Class | Channel | Description |
|-------|---------|-------------|
| `FrequencyBands` | 2 | Five carrier-wave band powers and phases |
| `StructuralSnapshot` | 1 | Mean, variance, skewness, kurtosis, autocorrelation, stationarity |
| `CouplingMatrix` | 2 | PAC coefficients between adjacent bands + composite score |
| `ChannelCoherence` | 1↔2 | Structural-rhythmic coherence score |
| `DegradationSequence` | 3 | Temporal ordering of channel degradation events |

### Alert types

| `AlertType` | `AlertSeverity` | Trigger |
|-------------|-----------------|---------|
| `BAND_ANOMALY` | WARNING | Per-carrier-wave anomaly |
| `COUPLING_DEGRADATION` | WARNING | `composite_coupling_score` below threshold |
| `STRUCTURAL_RHYTHMIC_DECOUPLING` | ALERT | `coherence_score` below threshold |
| `CASCADE_PRECURSOR` | **CRITICAL** | All three: coupling + decoupling + ≥N EWS |

---

## SentinelResult API

`SentinelResult` is a `dict` subclass. All v8.0/v9.0 dictionary access patterns work unchanged.

### Core fields

| Key | Type | Description |
|-----|------|-------------|
| `"step"` | `int` | Monotonic observation counter |
| `"value"` | `float` | Raw input value |
| `"ewma"` | `float` | Current EWMA estimate |
| `"alert"` | `bool` | Any anomaly triggered |
| `"warmup"` | `bool` | True during warmup period |
| `"anomaly_score"` | `float` | Normalized composite score |
| `"z_score"` | `float` | EWMA deviation in sigma units |
| `"alert_reasons"` | `list[str]` | Triggered conditions |
| `"cascade_precursor_active"` | `bool` | CASCADE_PRECURSOR active |

### V9.0 convenience methods

```python
result.is_cascade_precursor() -> bool
result.get_channel_status() -> dict
result.get_degradation_narrative() -> str
result.get_primary_carrier_wave() -> str
```

### V10.0 convenience methods

```python
result.is_reversed_sequence() -> bool

result.get_maintenance_burden() -> dict
# {"mu": 0.82, "regime": "TAINTER_WARNING"}

result.get_pac_status() -> dict
# {"mean_pac": 0.35, "degradation_rate": 0.19, "pre_cascade_pac": True}

result.get_diagnostic_window() -> dict
# {"steps": 47.3, "confidence": "HIGH", "supercompensation": False}

result.get_intervention_signature() -> dict
# {"score": 0.74, "sequence_type": "REVERSED",
#  "phi_rate": -0.18, "coupling_rate": -0.01}
```

### V10.0 scalar result keys

| Key | Type | Description |
|-----|------|-------------|
| `"maintenance_burden"` | `float` | μ (0.0–1.0) |
| `"tainter_regime"` | `str` | HEALTHY / REDUCED_RESERVE / TAINTER_WARNING / TAINTER_CRITICAL |
| `"mean_pac"` | `float` | Current PAC strength (0.0–1.0) |
| `"pac_degradation_rate"` | `float` | Fractional PAC decline rate |
| `"pre_cascade_pac"` | `bool` | PAC warning before cascade precursor |
| `"diagnostic_window_steps"` | `float\|None` | Steps until coherence collapse |
| `"diagnostic_window_confidence"` | `str` | HIGH / MEDIUM / LOW / NOT_APPLICABLE |
| `"supercompensation_detected"` | `bool` | Adaptive recovery in progress |
| `"kuramoto_order"` | `float` | Φ inter-band phase coherence (0.0–1.0) |
| `"reversed_sequence"` | `bool` | Coherence collapsing before coupling |
| `"intervention_signature_score"` | `float` | 0.0–1.0 confidence of deliberate intervention |
| `"sequence_type"` | `str` | ORGANIC / REVERSED / AMBIGUOUS / INSUFFICIENT_DATA |
| `"coupling_rate"` | `float` | dκ̄/dt (negative = degrading) |
| `"critical_coupling"` | `float` | κ_c estimated from frequency distribution |

---

## MultiStreamSentinel

Thread-safe manager for multiple independent named streams.

```python
from fracttalix import MultiStreamSentinel, SentinelConfig

mss = MultiStreamSentinel(config=SentinelConfig.production())

result = mss.update("sensor_42", 3.14)
result = await mss.aupdate("sensor_42", 3.14)

mss.list_streams()
mss.get_detector("sensor_42")
mss.status("sensor_42")
mss.reset_stream("sensor_42")
mss.delete_stream("sensor_42")

state_json = mss.save_all()
mss.load_all(state_json)
```

---

## SentinelBenchmark

Built-in evaluation harness with five labeled anomaly archetypes.

| Archetype | Description |
|-----------|-------------|
| `point` | Sparse large spikes (8σ) at fixed intervals |
| `contextual` | Values anomalous given sinusoidal seasonal context |
| `collective` | Extended runs of moderately elevated values |
| `drift` | Slow linear mean drift starting mid-series |
| `variance` | Sudden 4× variance explosion in second half |

```python
from fracttalix import SentinelBenchmark, SentinelConfig

bench = SentinelBenchmark(n=500, config=SentinelConfig.sensitive())
bench.run_suite()   # reports F1, AUPRC, VUS-PR, mean lag, 3σ baseline

data, labels = bench.generate("drift")
metrics = bench.evaluate(data, labels)
```

---

## SentinelServer — REST API

Asyncio HTTP server wrapping a `MultiStreamSentinel`. No framework dependencies.

```bash
python -m fracttalix --serve --host 0.0.0.0 --port 8765
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Version, uptime, stream count |
| `GET` | `/streams` | List all stream IDs |
| `POST` | `/update/<id>` | Feed observation: `{"value": 3.14}` |
| `GET` | `/status/<id>` | Stream stats and last result |
| `DELETE` | `/stream/<id>` | Delete stream |
| `POST` | `/reset/<id>` | Reset stream to factory state |

```bash
curl -X POST http://localhost:8765/update/my_sensor \
     -H "Content-Type: application/json" \
     -d '{"value": 42.0}'
```

---

## CLI Reference

```
fracttalix [OPTIONS]
# or: python -m fracttalix [OPTIONS]
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
| `--test` | — | Run smoke suite and exit |

---

## Backward Compatibility

v12.1 is a strict superset of all prior versions. No step is removed. No result key is removed.

### V8.0 root-cause fixes (all preserved)

| Fix | Label | Description |
|-----|-------|-------------|
| α | Frozen config | `SentinelConfig` is a frozen dataclass |
| β | WindowBank | Named independent deques per consumer |
| γ | Pipeline decomposition | 37 `DetectorStep` subclasses |
| δ | Soft regime boost | Multiplicative boost + decay |
| ε | SSI naming | `result["rsi"]` alias preserved |

### V7.x compatibility

```python
from fracttalix import SentinelDetector
det = SentinelDetector(alpha=0.1, multiplier=3.0, warmup_periods=30)  # v7.x kwargs accepted
```

### State persistence

```python
json_str = det.save_state()
det2 = SentinelDetector(config)
det2.load_state(json_str)
```

---

## Theoretical Foundation

| FRM Component | Sentinel Implementation |
|---------------|------------------------|
| FRM Axiom 3 (ordinal pattern complexity) | `PEStep` — Permutation Entropy |
| FRM Axiom 9 (critical slowing down) | `EWSStep` — variance + lag-1 autocorrelation |
| Rhythm Periodicity Index | `RPIStep` — FFT spectral coherence |
| Rhythm Fractal Index | `RFIStep` — Hurst exponent via R/S |
| Synchronization Stability Index | `SSIStep` — Kuramoto proxy via FFT phase coherence |
| Three-channel model (Paper 6) | `StructuralSnapshotStep`, `FrequencyDecompositionStep`, `ChannelCoherenceStep`, `CascadePrecursorStep`, `DegradationSequenceStep` |
| Maintenance burden μ | `ThroughputEstimationStep`, `MaintenanceBurdenStep` |
| PAC pre-cascade (Tort 2010) | `PhaseExtractionStep`, `PACCoefficientStep`, `PACDegradationStep` |
| Diagnostic window Δt | `CriticalCouplingEstimationStep`, `CouplingRateStep`, `DiagnosticWindowStep` |
| Kuramoto order Φ / reversed sequence | `KuramotoOrderStep`, `SequenceOrderingStep`, `ReversedSequenceStep` |

**DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)

---

## How This Repository Works — A Guide to Everything Here

This repo contains three things that are usually kept separate: a streaming
anomaly detector, the physical theory it's derived from (five published papers
plus Meta-Kaizen, seven more planned), and a machine-verifiable registry of
every scientific claim made in that theory. They live together because they
depend on each other — the software implements the theory, the theory justifies
the software, and the claim registry keeps both honest.

Here's what everything is and why it's built this way.

### The Software — Sentinel

Sentinel is the streaming anomaly detector. It's what you `pip install`. It runs
on pure Python stdlib with zero dependencies, processes one observation at a time
in constant memory, and extracts three-channel diagnostics from a single scalar
stream — something no comparable tool does. Everything above this section documents
its API. If you want to see it work, start with the
[Quickstart Tutorial](examples/00_quickstart.ipynb) — five minutes, no clone required.

### The Theory — Fractal Rhythm Model (FRM)

Sentinel isn't a collection of heuristics — it's derived from a physical theory.
The Fractal Rhythm Model describes how any network (biological, organizational,
civilizational) transmits information through coupled oscillatory components, and
how those systems degrade and collapse.

Five papers and the Meta-Kaizen framework are published. Seven more are planned
to complete the corpus:

| Published | What they cover |
|-----------|----------------|
| **Papers 1–4** (Act I) | The law at human scale — organizations, cognition, mathematical form |
| **Paper 5** (Act II) | Scale independence — same mathematics at ocean circulation and civilizational scale |
| **Meta-Kaizen Paper 1** | Continuous improvement framework and KVS scoring methodology |

| Planned | What they will cover |
|---------|---------------------|
| **Papers 6–12** (Act III) | Complete statement, formal proofs, instrumentation, and civilizational application |

The [Build Table](docs/FRM_SeriesBuildTable_v1.5.md) is the living architectural
document — paper status, dependencies, release schedule, referee analysis, and
risk register. If you want to understand the big picture, start there.

### Meta-Kaizen — How We Score Work

Meta-Kaizen is the continuous improvement framework used to evaluate every work
item before and after execution. Every significant task gets a KVS score:

```
KVS = N × I' × C' × T
```

| Component | Measures |
|-----------|----------|
| **N** (Novelty) | Does this exist yet? |
| **I'** (Impact) | How much does it move the project forward? |
| **C'** (Inverse Complexity) | How achievable is it in one session? |
| **T** (Timeliness) | How urgent is it right now? |

Scores above the threshold (κ = 0.75 for the corpus, lower for individual tasks)
justify the work. Scores below suggest doing something else first. You'll see
KVS tables bookending the [Quickstart Tutorial](examples/00_quickstart.ipynb) —
that's Meta-Kaizen in action.

### The Dual Reader Standard — AI Layers

Every paper has a machine-readable AI layer in `ai-layers/`. These are JSON files
that register every scientific claim with:

- **Claim ID** and type (Axiom, Derivation, or Falsification)
- **Derivation sources** — which prior claims this one depends on
- **Falsification predicates** — what would prove the claim wrong, stated precisely
- **Placeholders** — claims that reference future work, tracked until resolved

The point: a human reads the paper, a machine reads the AI layer. Both see the
same claims. If a cross-reference is broken, the
[cross-paper checker](scripts/cross_paper_checker.py) catches it. If a schema
is violated, [CI catches it](.github/workflows/ai-layer-validation.yml)
automatically. This is what "Dual Reader" means — every claim is verifiable by
both audiences, and the verification runs on every commit.

The falsification predicates deserve emphasis. Every Type F claim states exactly
what would prove it wrong, in machine-parseable five-part syntax. This isn't
"we welcome criticism" — it's "here is the specific experiment that would
destroy this claim, stated in advance." 80 claims across 15 layers, 0 cross-reference
errors.

### The Process Graph

`ai-layers/process_graph.json` maps the dependency structure between all papers
and supporting documents. It's the machine-readable version of the Build Table's
dependency diagram — which paper enables which, what's published, what's pending.

### Scripts and Validation

| Script | What it does |
|--------|-------------|
| [`validate_ai_layers.py`](scripts/validate_ai_layers.py) | Schema compliance check — runs in CI on every push |
| [`cross_paper_checker.py`](scripts/cross_paper_checker.py) | Cross-reference integrity — derivation sources, placeholder targets, orphan claims |
| [`corpus_status.py`](scripts/corpus_status.py) | Human-readable and JSON status report across all layers |

### Examples

| File | What it shows |
|------|---------------|
| [`00_quickstart.ipynb`](examples/00_quickstart.ipynb) | Three-channel detection, physics diagnostics, tuning tips |
| [`01_basic_streaming.py`](examples/01_basic_streaming.py) | Minimal API: create, feed, check |
| [`02_multistream.py`](examples/02_multistream.py) | Thread-safe multi-stream monitoring |
| [`03_collapse_detection.py`](examples/03_collapse_detection.py) | PAC degradation and intervention signatures |
| [`04_autotune.py`](examples/04_autotune.py) | Auto-tuning from labeled data |
| [`05_getting_started.ipynb`](examples/05_getting_started.ipynb) | Detailed API walkthrough |

### The Journal

`journal/` contains session notes documenting what was built and why, in
chronological order. Each entry records key actions, validation results, and
session significance. The [journal index](journal/journal_index.md) links them all.

### Reproducibility

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) maps every Type F (falsification)
claim to an executable test, documents the verification pipeline, and provides
step-by-step reproduction instructions. If you want to verify any claim in the
corpus, start there.

### Continuous Integration

Three CI workflows run automatically on every push:

| Workflow | What it checks |
|----------|---------------|
| [`tests.yml`](.github/workflows/tests.yml) | 374 tests across Python 3.9–3.12 |
| [`ai-layer-validation.yml`](.github/workflows/ai-layer-validation.yml) | AI layer schema compliance and cross-reference integrity |
| [`release.yml`](.github/workflows/release.yml) | PyPI release pipeline |

### Contributing and Community

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — how to report bugs, suggest features, and submit pull requests
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) — community standards
- [`CHANGELOG.md`](CHANGELOG.md) — version history following [Keep a Changelog](https://keepachangelog.com/)

### Legal

[`legal/DISCLAIMER.md`](legal/DISCLAIMER.md) — the software is CC0 public domain,
provided as-is for research and educational use. Not intended for safety-critical
or production decision-making without independent validation.

### Directory Map

```
fracttalix/          Python package — Sentinel detector, config, pipeline steps
ai-layers/           Machine-readable claim registries (JSON) + schema
paper/               Software paper + bibliography
docs/                Build table, bootstrap doc, API docs, theory docs
examples/            Tutorials and usage examples
scripts/             Validation and status reporting tools
journal/             Session notes — what was built and why
tests/               374 tests across 12 test files
benchmark/           Performance evaluation suite
legacy/              Pre-refactor archive
legal/               Disclaimer
.github/             CI workflows, issue templates, PR template
```

---

## Channel 2 — AI Layers

Machine-readable falsification layers for the Fracttalix corpus. All layers conform to `ai-layers/ai-layer-schema.json` (v2-S42, Dual Reader Standard).

| ID    | Paper                        | Status      | File                             |
|-------|------------------------------|-------------|----------------------------------|
| P1    | Fractal Rhythm Model (Paper 1) | PHASE-READY | ai-layers/P1-ai-layer.json       |
| MK-P1 | Meta-Kaizen Paper 1          | PHASE-READY | ai-layers/MK-P1-ai-layer.json    |
| DRP-1 | Dependency Resolution Process | PHASE-READY | ai-layers/DRP1-ai-layer.json     |
| SFW-1 | Sentinel v12                 | PHASE-READY | ai-layers/SFW1-ai-layer.json     |

---

## Authors & License

**Authors:** Thomas Brennan & Claude (Anthropic) & Grok (xAI)

**License:** [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/) — no restrictions, no attribution required.

**GitHub:** [https://github.com/thomasbrennan/Fracttalix](https://github.com/thomasbrennan/Fracttalix)

**Version history:**

| Version | Notes |
|---------|-------|
| v10.0.0 | 4 physics-derived capabilities, 37 steps, 98 tests |
| v9.0.0 | Three-channel model, 26 steps, 65 tests |
| v8.0.0 | Frozen config, WindowBank, 19-step pipeline |
| v7.11–v7.6 | Earlier releases |
