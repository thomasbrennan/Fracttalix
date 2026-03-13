# Fracttalix Sentinel v12.3

[![Tests](https://github.com/thomasbrennan/Fracttalix/actions/workflows/tests.yml/badge.svg)](https://github.com/thomasbrennan/Fracttalix/actions/workflows/tests.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](https://www.python.org/)
[![License: CC0-1.0](https://img.shields.io/badge/license-CC0--1.0-brightgreen.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![PyPI version](https://img.shields.io/pypi/v/fracttalix.svg)](https://pypi.org/project/fracttalix/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18859299.svg)](https://doi.org/10.5281/zenodo.18859299)

**Streaming anomaly detection combining EWMA, CUSUM, spectral decomposition, cross-frequency coupling, and Kuramoto-inspired synchrony metrics in a 37-step composable pipeline.**

Sentinel ingests one scalar (or multivariate) observation at a time and emits a rich result dictionary on every call — no batching, no retraining, no warmup gap once past the configurable warmup window.

> **Companion materials:** Fractal Rhythm Model working papers 1–6 (unpublished)
> DOI: [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) (data deposit, not peer-reviewed)
> License: **CC0** — public domain

**[Quickstart Tutorial](examples/00_quickstart.ipynb)** | **[Full documentation](https://thomasbrennan.github.io/Fracttalix)** | **[Examples](examples/)** | **[CHANGELOG](CHANGELOG.md)**

---

## Table of Contents

1. [Overview](#overview)
2. [Three-Channel Model](#three-channel-model)
3. [V12.3 Changes](#v123-changes)
3a. [V12.2 Changes](#v122-changes)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [SentinelConfig — Configuration](#sentinelconfig--configuration)
7. [Pipeline Architecture — 37 Steps](#pipeline-architecture--37-steps-v123)
8. [Alert Types and Data Structures](#alert-types-and-data-structures)
9. [SentinelResult API](#sentinelresult-api)
10. [MultiStreamSentinel](#multistreamssentinel)
11. [SentinelBenchmark](#sentinelbenchmark)
12. [SentinelServer — REST API](#sentinelserver--rest-api)
13. [CLI Reference](#cli-reference)
14. [Backward Compatibility](#backward-compatibility)
15. [Limitations](#limitations)
16. [Algorithms and Techniques](#algorithms-and-techniques)
17. [How This Repository Works](#how-this-repository-works--a-guide-to-everything-here)
18. [Authors & License](#authors--license)

---

## Overview

Fracttalix Sentinel is a Python package (`pip install fracttalix`) for real-time streaming anomaly detection. Its design priorities are:

- **Zero external dependencies for core operation** — works on the Python standard library alone; numpy, scipy, numba, matplotlib, and tqdm are optional accelerators.
- **Immutable, inspectable configuration** — `SentinelConfig` is a frozen dataclass; every parameter is readable and picklable.
- **Composable pipeline** — 37 `DetectorStep` subclasses execute in sequence.
- **Three-channel decomposition** — monitors structural statistics, spectral properties, and temporal degradation sequences as independent signal channels.
- **Signal-processing heuristics** — maintenance burden (μ = 1−κ̄), PAC pre-cascade detection, diagnostic window estimation, and reversed sequence detection. These use standard signal processing techniques (EWMA, FFT, Hilbert transform, phase coherence) with hand-tuned thresholds — not physics derivations.
- **Full backward compatibility** — all v7.x, v8.0, v9.0, and v10.0 call patterns continue to work unchanged.

---

## Three-Channel Decomposition

The detector organizes its 37 steps into three independent signal channels:

| Channel | Name | What it monitors |
|---------|------|-----------------|
| **1** | Structural | Statistical properties — mean, variance, skewness, kurtosis, autocorrelation, stationarity |
| **2** | Spectral | FFT decomposition into five frequency bands, cross-frequency phase-amplitude coupling |
| **3** | Temporal | Ordering of degradation events across channels 1 and 2 |

**Cascade detection (v9.0):** alerts when band anomaly + coupling degradation + structural-spectral decoupling all occur simultaneously.

**Extended heuristics (v10.0):** PAC degradation tracking, time-to-threshold estimation, and sequence ordering classification.

---

## V12.3 Changes

### FPR Elimination & Drift Recovery — Meta Kaisen CBP

v12.3 is a comprehensive recalibration targeting the ~35% FPR floor that
dominated v12.2 performance. FPR dropped 93%, mean F1 rose 25%.

#### New Architecture

- **SeasonalPreprocessStep** (Step 0): FFT-based seasonal decomposition with
  confidence gate `peak_power > 10× mean_power` (<0.1% false detection on
  white noise). All 37 downstream steps receive the deseasonalized residual.
- **Non-adaptive drift CUSUM** in `CUSUMStep`: Accumulates on warmup-frozen
  z-score, detecting slow drift that EWMA adaptation masks. Fires
  `drift_cusum_alert`, resets, and re-fires continuously during ongoing drift.
- **ConsensusGate** in `AlertReasonsStep`: Requires ≥2 soft alerts OR 1
  strong alert (`cusum_mean_shift`, `cusum_variance_spike`, `drift_cusum_shift`,
  `gradual_drift`, `cascade_precursor`) OR |z| ≥ 5σ bypass. Primary FPR
  reduction mechanism.

#### Recalibrated Thresholds (null-distribution calibrated on N(0,1))

- `rfi_threshold`: 0.40 → 0.52; `pe_threshold`: 0.05 → 0.15
- `var_cusum_k`: 0.5 → 1.0 (E[z²]=1.0 under H₀; old k was systematically biased)
- `var_cusum_h`: 5.0 → 10.0; `cusum_k`: 0.5 → 1.0; `cusum_h`: 5.0 → 8.0
- `coherence_threshold`: 0.40 → 0.30; `coupling_degradation_threshold`: 0.30 → 0.24

#### v12.3 Benchmark (n=1000, seed=42, post-warmup)

| Archetype  | v12.2 F1 | v12.3 F1 | Change   |
|------------|----------|----------|----------|
| point      | 0.422    | 0.639    | +51%     |
| contextual | 0.242    | 0.378    | +56%     |
| collective | 0.239    | 0.356    | +49%     |
| drift      | 0.723    | 0.766    | +6%      |
| variance   | 0.876    | 0.987    | +13%     |
| **FPR**    | **35%**  | **2.6%** | **−93%** |
| Mean F1    | 0.500    | 0.625    | **+25%** |

> `SentinelDetector()` (no args) now defaults to `SentinelConfig.production()`
> (multiplier=4.5). The FPR floor has been eliminated by threshold recalibration
> and ConsensusGate — not by multiplier inflation.

---

## V12.2 Changes

### Epistemic Language Corrections

- Replaced "physics-derived" framing with "signal-processing heuristic" throughout
  README and docstrings. The capabilities are real and useful; the claim that they
  are *derived from physics* was overstated and contradicted by code comments.
- Corrected the README maintenance burden formula (was showing the abandoned v10.0
  formula `μ = N·κ̄·E_coupling/P_throughput`; actual implementation is `μ = 1−κ̄`
  as documented in `MaintenanceBurdenStep` since v11.0).
- Reframed "thermodynamic arrow" and "intervention signature" language as signal
  classification labels, not physical or causal claims.

### Default Multiplier Change (Breaking)

- `SentinelConfig.production()` now uses `multiplier=4.5` (was 3.0).
  Normal alert rate: **35.6% → ~6%** on white noise N(0,1).
  Expected F1 improvements at the new default (n=1000, seed=42, estimated):

| Archetype  | v12.1 F1 | v12.2 F1 est. | Notes |
|------------|----------|---------------|-------|
| point      | 0.415    | ~0.38         | Slight drop in marginal spike detection |
| contextual | 0.247    | ~0.35         | Precision gain from lower FPR |
| collective | 0.239    | ~0.45         | Large precision gain from lower FPR |
| drift      | 0.723    | ~0.66         | Moderate recall reduction |
| variance   | 0.876    | ~0.82         | Small precision/recall rebalance |
| **normal** | —        | **~6%**       | Primary improvement target |

> Users who depended on the v12.1 FPR behaviour can restore it with
> `SentinelConfig(multiplier=3.0)` or set any custom multiplier.

## V12.1 Changes

### Bug Fixes

- **VarCUSUM non-reset** (`VarCUSUMStep`): accumulators `s_hi`/`s_lo` now re-arm
  after each threshold crossing. Normal alert rate: 97% → 35.6%.
- **ChannelCoherence unit mismatch** (`ChannelCoherenceStep`): replaced
  rate-difference formula with Pearson correlation (scale-invariant). Normal data
  now scores ~0.5 above threshold.

### v12.1 Benchmark (n=1000, seed=42, multiplier=3.0)

| Archetype   |  F1   | Normal alert rate |
|-------------|-------|-------------------|
| point       | 0.415 | —                 |
| contextual  | 0.247 | —                 |
| collective  | 0.239 | —                 |
| drift       | 0.723 | —                 |
| variance    | 0.876 | —                 |
| **normal**  | —     | **35.6%**         |

---

## Signal-Processing Heuristics (v10.0+)

Four indicators added in v10.0 that track coupling and coherence trends over time.

> **Epistemic status:** These are engineering heuristics with hand-tuned thresholds, not physics derivations. The maintenance burden μ is a simple linear rescaling of coupling strength, not derived from Tainter's socioeconomic model. The regime names (TAINTER_CRITICAL, etc.) are classification labels; thresholds are empirically set.

### 1. Maintenance Burden μ

```
μ = 1 − κ̄
```

A simple inversion of the mean cross-frequency coupling score. When coupling is high (κ̄ → 1), μ is low. When coupling is low (κ̄ → 0), μ is high. That's all it computes — it's `1 minus the coupling score` with regime labels at fixed thresholds.

| μ range | Regime | Meaning |
|---------|--------|---------|
| < 0.5 | `HEALTHY` | Coupling above 0.5 |
| 0.5 – 0.75 | `REDUCED_RESERVE` | Coupling between 0.25–0.5 |
| 0.75 – 0.9 | `TAINTER_WARNING` | Coupling between 0.1–0.25 |
| ≥ 0.9 | `TAINTER_CRITICAL` | Coupling below 0.1 |

> The TAINTER_ prefix in regime names is a label choice, not a claim about Tainter's socioeconomic collapse model. Thresholds are hand-picked, not calibrated from data.

```python
mb = result.get_maintenance_burden()
# {"mu": 0.82, "regime": "TAINTER_WARNING"}
```

### 2. PAC Tracking

Phase-Amplitude Coupling (PAC) measures how strongly low-frequency phase modulates high-frequency amplitude, using a simplified Modulation Index (after Tort et al. 2010) across 6 band pairs with 8 fixed phase bins. When PAC drops over a rolling window, `pre_cascade_pac` fires.

This sometimes precedes a drop in mean coupling κ̄, but the "pre-cascade" framing is aspirational — it's a rolling threshold check on a single metric, not a validated precursor model.

```python
pac = result.get_pac_status()
# {"mean_pac": 0.41, "degradation_rate": 0.18, "pre_cascade_pac": True}
```

### 3. Diagnostic Window Δt Estimation (Time-to-Threshold)

```
Δt = (κ̄ - κ_c) / |dκ̄/dt|
```

A linear extrapolation of how many observations remain before coupling strength crosses the critical threshold, assuming the current rate of change continues.

- Only active when κ̄ > κ_c and dκ̄/dt < 0
- Confidence graded: `HIGH` / `MEDIUM` / `LOW` based on rate stability
- Reports `supercompensation` when dκ̄/dt turns positive (coupling recovering instead of declining)

```python
dw = result.get_diagnostic_window()
# {"steps": 47.3, "confidence": "HIGH", "supercompensation": False}
```

### 4. Reversed Sequence Detection (Sequence Classification)

A heuristic based on the assumption that coupling typically degrades before coherence collapses. When the opposite order is observed, it may indicate noise, a different data-generating process, or an abrupt external change.

> **Note:** "Intervention" here is a signal classification label, not a causal claim about deliberate external action. `sequence_type: "REVERSED"` means the ordering is atypical relative to the heuristic baseline; `intervention_signature_score` is a confidence value for that classification, not a probability of any external cause.

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

**Run tests (434 tests, all expected to pass):**

```bash
pip install fracttalix[dev]       # pytest, ruff, mypy, mkdocs
pytest
```

---

## Quick Start

### Basic use — production defaults (v12.3)

In v12.1, `SentinelConfig.production()` alerted on 35.6% of normal observations.
In v12.3, that default is ~6%. Same API, same one line:

```python
from fracttalix import SentinelDetector, SentinelConfig

det = SentinelDetector(SentinelConfig.production())  # multiplier=4.5, ~6% normal FPR

for value in my_data_stream:
    result = det.update_and_check(value)
    if result["alert"]:
        print(f"Step {result['step']}: {result['alert_reasons']}")
```

If you were on v12.1 and need the old threshold back:

```python
det = SentinelDetector(SentinelConfig(multiplier=3.0))  # restores v12.1 behaviour (~35% FPR)
```

### Choosing sensitivity

The multiplier is the single most important dial. Here is what it means in practice:

```python
# ~6% normal alert rate — good for production dashboards, on-call alerting
det = SentinelDetector(SentinelConfig.production())          # multiplier=4.5

# ~2% normal alert rate — for high-confidence alerting with low tolerance for noise
det = SentinelDetector(SentinelConfig(multiplier=5.0))

# ~35% normal alert rate — v12.1 default; use when missing anomalies is worse
# than chasing false positives, or when downstream filtering is in place
det = SentinelDetector(SentinelConfig(multiplier=3.0))

# ~40-50% normal alert rate — catches the subtlest shifts; pairs with human review
det = SentinelDetector(SentinelConfig.sensitive())           # multiplier=2.5
```

Auto-tune picks the multiplier that maximises F1 on your labeled examples:

```python
labeled = [(value, is_anomaly), ...]
det = SentinelDetector.auto_tune(data=[], labeled_data=labeled)
```

### Signal-processing heuristics

Four indicators track how coupling and coherence metrics are evolving.
Useful for early warning and pattern characterization — these are threshold-based
heuristics, not calibrated physical measurements.

```python
result = det.update_and_check(value)

# Maintenance burden (μ = 1 − κ̄, i.e. inverted coupling score)
# High μ simply means low cross-frequency coupling
mb = result.get_maintenance_burden()
if mb["regime"] in ("TAINTER_WARNING", "TAINTER_CRITICAL"):
    print(f"Coupling fragmented: μ={mb['mu']:.2f} ({mb['regime']})")

# PAC pre-cascade: phase-amplitude coupling degrading before κ̄ drops
# Fires earlier than the cascade precursor — gives more lead time
pac = result.get_pac_status()
if pac["pre_cascade_pac"]:
    print(f"PAC degrading at rate {pac['degradation_rate']:.3f} — early warning")

# Diagnostic window: estimated steps before coupling crosses threshold
# Only active when κ̄ > κ_c and coupling is falling
dw = result.get_diagnostic_window()
if dw["steps"] is not None:
    print(f"Δt ≈ {dw['steps']:.0f} steps ({dw['confidence']} confidence)")
if dw["supercompensation"]:
    print("Coupling rate turned positive — recovering")

# Sequence classification: is coherence collapsing before coupling degrades?
# REVERSED means atypical ordering; it is a classification label, not a causal claim
if result.is_reversed_sequence():
    sig = result.get_intervention_signature()
    print(f"Atypical sequence (score {sig['score']:.2f}) — ordering does not match "
          f"gradual organic decay pattern")
```

### Channel status

```python
# CASCADE_PRECURSOR requires all three conditions simultaneously:
# coupling degradation + structural-spectral decoupling + ≥2 EWS indicators elevated
if result.is_cascade_precursor():
    print("CRITICAL: cascade precursor — all three channels confirming")

status = result.get_channel_status()
# {"structural": "healthy", "rhythmic_composite": "degrading",
#  "coupling": "healthy", "coherence": "healthy"}

print(result.get_degradation_narrative())
print(result.get_primary_carrier_wave())   # "mid", "low", etc.
```

### Multivariate streams

```python
cfg = SentinelConfig(multivariate=True, n_channels=3)
det = SentinelDetector(config=cfg)
result = det.update_and_check([v1, v2, v3])
```

### Async usage

```python
result = await det.aupdate(value)
```

### Multiple streams (thread-safe)

```python
from fracttalix import MultiStreamSentinel

mss = MultiStreamSentinel(config=SentinelConfig.production())

# Each stream ID gets its own independent detector instance
result = mss.update("sensor_42", 3.14)
result = await mss.aupdate("sensor_43", 7.71)
```

---

## SentinelConfig — Configuration

`SentinelConfig` is a frozen dataclass (`slots=True`). All fields are immutable after construction. Use `dataclasses.replace(cfg, field=value)` to derive a new config.

### Factory presets

| Preset | `alpha` | `multiplier` | `warmup` | Normal FPR¹ | Notes |
|--------|---------|-------------|----------|-------------|-------|
| `SentinelConfig.fast()` | 0.3 | 3.0 | 10 | ~60–80% | Fastest response; very high FP rate — use only with downstream filtering |
| `SentinelConfig.production()` | 0.1 | **4.5** | 30 | ~5–8% | Balanced defaults; v12.3 default |
| `SentinelConfig.sensitive()` | 0.05 | 2.5 | 50 | ~40–50% | Catches subtle anomalies; high FP rate |
| `SentinelConfig.realtime()` | 0.2 | 3.0 | 15 | ~30–40% | Quantile-adaptive thresholds |

> ¹ Approximate normal alert rate on white noise N(0,1), empirically measured. FPR is a function of multiplier, alpha, and data distribution — these are indicative values from `benchmark/investigate_fpr_s47.py`.
>
> **Multiplier–FPR trade-off** (white noise, seed=99): multiplier 1.5 → ~90% FPR, 3.0 → ~35%, 4.5 → ~6%, 5.0 → ~2%. Higher multiplier reduces false positives but may miss subtle anomalies.

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

#### D — Spectral Metrics

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

#### F — Signal Analysis Windows

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
| `enable_frequency_decomposition` | `True` | Enable FFT into five frequency bands |
| `min_window_for_fft` | `32` | Minimum window before FFT runs |

#### V9.0 — Cross-Frequency Coupling

| Field | Default | Description |
|-------|---------|-------------|
| `enable_coupling_detection` | `True` | Enable phase-amplitude coupling |
| `coupling_degradation_threshold` | `0.3` | Composite score below this → `COUPLING_DEGRADATION` |
| `coupling_trend_window` | `10` | FrequencyBands snapshots for coupling trend |

#### V9.0 — Structural-Spectral Coherence

| Field | Default | Description |
|-------|---------|-------------|
| `enable_channel_coherence` | `True` | Enable structural-spectral coherence |
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

## Pipeline Architecture — 37 Steps (v12.3)

Every call to `update_and_check()` runs all 37 steps in order. Steps read from and write to a shared `StepContext.scratch` dictionary.

| # | Step | Version | Description |
|---|------|---------|-------------|
| 1 | `CoreEWMAStep` | v8 | EWMA baseline + deviation; must run first |
| 2 | `StructuralSnapshotStep` | v9 | Channel 1: mean, variance, skewness, kurtosis, autocorrelation, stationarity |
| 3 | `FrequencyDecompositionStep` | v9 | Channel 2: FFT into 5 frequency bands with power and phase |
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
| 14 | `SSIStep` | v8 | Synchronization Stability Index — phase coherence via FFT (`rsi` alias preserved) |
| 15 | `PEStep` | v8 | Permutation Entropy |
| 16 | `EWSStep` | v8 | Early Warning Signals — variance + AC(1) |
| 17 | `AQBStep` | v8 | Adaptive Quantile Baseline |
| 18 | `SeasonalStep` | v8 | Seasonal decomposition with FFT auto-period |
| 19 | `MahalStep` | v8 | Mahalanobis distance (multivariate) |
| 20 | `RRSStep` | v8 | Robust Residual Score |
| 21 | `BandAnomalyStep` | v9 | Per-frequency-band anomaly invisible to composite |
| 22 | `CrossFrequencyCouplingStep` | v9 | PAC coupling matrix + `COUPLING_DEGRADATION` alert |
| 23 | `ChannelCoherenceStep` | v9 | Structural-rhythmic coherence + `SR_DECOUPLING` alert |
| 24 | `CascadePrecursorStep` | v9 | `CASCADE_PRECURSOR` — CRITICAL; requires all three conditions |
| 25 | `DegradationSequenceStep` | v9 | Channel 3: temporal degradation sequence log |
| 26 | `ThroughputEstimationStep` | **v10** | P_throughput from band amplitudes; populates `band_amplitudes`, `band_powers`, `node_count`, `mean_coupling_strength` |
| 27 | `MaintenanceBurdenStep` | **v10** | μ = 1−κ̄ (linear rescaling of coupling) → regime classification |
| 28 | `PhaseExtractionStep` | **v10** | FFT bandpass + Hilbert transform → instantaneous phase per band |
| 29 | `PACCoefficientStep` | **v10** | Modulation Index (Tort 2010) across 6 slow/fast band pairs |
| 30 | `PACDegradationStep` | **v10** | Rolling PAC history → `pac_degradation_rate`, `pre_cascade_pac` |
| 31 | `CriticalCouplingEstimationStep` | **v10** | κ_c = 2/(π·g(ω₀)) from power-weighted frequency spread |
| 32 | `CouplingRateStep` | **v10** | dκ̄/dt from rolling coupling history |
| 33 | `DiagnosticWindowStep` | **v10** | Δt = (κ̄−κ_c)/|dκ̄/dt|; confidence grading; supercompensation |
| 34 | `KuramotoOrderStep` | **v10** | Φ = |mean(e^iθ_k)| — phase coherence independent of κ̄ |
| 35 | `SequenceOrderingStep` | **v10** | COUPLING_FIRST / COHERENCE_FIRST / SIMULTANEOUS / STABLE per step |
| 36 | `ReversedSequenceStep` | **v10** | Atypical degradation ordering → sequence classification + intervention_signature_score |
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
| `FrequencyBands` | 2 | Five frequency band powers and phases |
| `StructuralSnapshot` | 1 | Mean, variance, skewness, kurtosis, autocorrelation, stationarity |
| `CouplingMatrix` | 2 | PAC coefficients between adjacent bands + composite score |
| `ChannelCoherence` | 1↔2 | Structural-rhythmic coherence score |
| `DegradationSequence` | 3 | Temporal ordering of channel degradation events |

### Alert types

| `AlertType` | `AlertSeverity` | Trigger |
|-------------|-----------------|---------|
| `BAND_ANOMALY` | WARNING | Per-frequency-band anomaly |
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
| `"diagnostic_window_steps"` | `float\|None` | Steps until coupling crosses critical threshold (linear extrapolation) |
| `"diagnostic_window_confidence"` | `str` | HIGH / MEDIUM / LOW / NOT_APPLICABLE |
| `"supercompensation_detected"` | `bool` | Coupling rate turned positive (recovering) |
| `"kuramoto_order"` | `float` | Φ inter-band phase coherence (0.0–1.0) |
| `"reversed_sequence"` | `bool` | Coherence collapsing before coupling |
| `"intervention_signature_score"` | `float` | 0.0–1.0 confidence of atypical sequence ordering (classification label, not causal claim) |
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

v12.3 is a strict superset of all prior versions. No step is removed. No result key is removed. The only breaking change from v12.1 is the `production()` default multiplier (3.0 → 4.5) — restore with `SentinelConfig(multiplier=3.0)`.

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

## Limitations

- **Benchmarks are synthetic.** The five archetypes (point, contextual, collective, drift, variance) are generated sine waves with injected anomalies. There is no validation on real-world data (network traffic, power grid, financial, etc.).
- **F1 scores are modest.** Point anomaly F1 ~0.64, contextual ~0.42, collective ~0.36 at best. Variance shift (0.99) and drift (0.77) are the strong suits. These are honest numbers, not cherry-picked.
- **FPR is still non-trivial.** At the default multiplier (4.5), ~6% of normal white noise observations trigger alerts. At sensitive settings, 40–50%.
- **No comparison to established baselines.** The benchmark does not compare against PyOD, River, ADTK, or other anomaly detection libraries.
- **Algorithms are simplified.** The Hurst exponent uses a single-pass R/S calculation, not DFA. The PAC uses 8 fixed bins, not surrogate-tested significance. The Kuramoto order parameter computes FFT phase coherence, not a full coupled oscillator model.
- **Thresholds are hand-tuned.** All regime thresholds, coupling thresholds, and alert conditions are empirically set by the author, not learned from data or calibrated against ground truth.
- **The theoretical framework is unpublished.** The Fractal Rhythm Model papers have not been peer-reviewed. The detector works as engineering software regardless, but the three-channel framing is the author's organizational choice, not a validated decomposition.

---

## Algorithms and Techniques

The pipeline combines well-known signal processing and anomaly detection techniques:

| Technique | Sentinel Implementation |
|-----------|------------------------|
| Permutation Entropy (ordinal pattern complexity) | `PEStep` |
| Early Warning Signals (variance + lag-1 AC, per Scheffer et al.) | `EWSStep` |
| FFT spectral coherence | `RPIStep` |
| Hurst exponent via simplified R/S analysis | `RFIStep` |
| Phase coherence (Kuramoto order parameter) | `SSIStep` |
| Statistical moments + stationarity | `StructuralSnapshotStep` |
| FFT band decomposition | `FrequencyDecompositionStep` |
| Phase-amplitude coupling (simplified Tort 2010 Modulation Index) | `PACCoefficientStep` |
| Linear time-to-threshold extrapolation | `DiagnosticWindowStep` |
| Coupling strength μ = 1−κ̄ (linear rescaling) | `MaintenanceBurdenStep` |

These are standard techniques applied in a streaming context. The novelty is in their composition into a single pipeline, not in the individual algorithms.

**Data deposit DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)

---

## How This Repository Works — A Guide to Everything Here

This repo contains three things: a streaming anomaly detector, an unpublished
theoretical framework (the Fractal Rhythm Model), and a machine-readable
registry of claims made in that framework.

### How This Was Built — AI Collaboration

This repo was built collaboratively with Claude (Anthropic) and Grok (xAI).
The theoretical direction and quality control architecture are Brennan's.
The code, tests, documentation, and validation infrastructure were produced
through AI-assisted development. The session journal and commit history
document what was built, when, and with which AI.

### The Software — Sentinel

Sentinel is the streaming anomaly detector. It's what you `pip install`. It runs
on pure Python stdlib with zero dependencies, processes one observation at a time
in constant memory, and combines 37 signal processing steps into a single
streaming pipeline. Everything above this section documents its API. Start with
the [Quickstart Tutorial](examples/00_quickstart.ipynb).

### The Fractal Rhythm Model (FRM) — Unpublished Working Papers

The FRM is an unpublished theoretical framework that motivated the detector's
three-channel decomposition. The working papers exist as markdown documents in
`paper/` and machine-readable claim registries in `ai-layers/`.

**Important:** These papers are not peer-reviewed and have not been published
in any journal or conference proceedings. The Zenodo DOI is a data deposit,
not a publication record. The detector works as engineering software regardless
of whether the theoretical claims hold up under review.

| Status | What they cover |
|--------|----------------|
| **Papers 1–5** (working drafts) | Network information transmission, organizational/cognitive/mathematical framing, scale independence |
| **Meta-Kaizen Paper 1** (working draft) | Continuous improvement framework and KVS scoring methodology |
| **Papers 6–12** (planned) | Formal proofs, instrumentation, applications |

The [Build Table](docs/FRM_SeriesBuildTable_v1.5.md) tracks paper status and
dependencies.

### Meta-Kaizen — Task Prioritization

Meta-Kaizen is the task prioritization framework used in this project. Tasks
are scored using KVS = N x I' x C' x T (Novelty, Impact, Inverse Complexity,
Timeliness). Scores above a threshold justify the work.

### AI Layers — Claim Registry

Every working paper has a machine-readable JSON file in `ai-layers/` that
registers claims with IDs, derivation sources, and falsification predicates.
The [cross-paper checker](scripts/cross_paper_checker.py) validates
cross-references, and [CI](.github/workflows/ai-layer-validation.yml) checks
schema compliance on every push.

### The Process Graph

`ai-layers/process_graph.json` maps the dependency structure between all working
papers and supporting documents — which paper depends on which, and current
draft status.

### Scripts and Validation

| Script | What it does |
|--------|-------------|
| [`validate_ai_layers.py`](scripts/validate_ai_layers.py) | Schema compliance check — runs in CI on every push |
| [`cross_paper_checker.py`](scripts/cross_paper_checker.py) | Cross-reference integrity — derivation sources, placeholder targets, orphan claims |
| [`corpus_status.py`](scripts/corpus_status.py) | Human-readable and JSON status report across all layers |

### Examples

| File | What it shows |
|------|---------------|
| [`00_quickstart.ipynb`](examples/00_quickstart.ipynb) | Three-channel detection, signal diagnostics, tuning tips |
| [`01_basic_streaming.py`](examples/01_basic_streaming.py) | Minimal API: create, feed, check |
| [`02_multistream.py`](examples/02_multistream.py) | Thread-safe multi-stream monitoring |
| [`03_collapse_detection.py`](examples/03_collapse_detection.py) | PAC degradation and sequence classification |
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
| [`tests.yml`](.github/workflows/tests.yml) | 434 tests across Python 3.10–3.12 |
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
paper/               Working paper drafts + bibliography
docs/                Build table, bootstrap doc, API docs, theory docs
examples/            Tutorials and usage examples
scripts/             Validation and status reporting tools
journal/             Session notes — what was built and why
tests/               434 tests across 16 test files
benchmark/           Performance evaluation suite
legacy/              Pre-refactor archive
legal/               Disclaimer
.github/             CI workflows, issue templates, PR template
```

---

## AI Layers — Claim Registry

Machine-readable claim registries for the Fracttalix working papers. All layers conform to `ai-layers/ai-layer-schema.json` (v2-S42).

| ID    | Paper                        | Status      | File                             |
|-------|------------------------------|-------------|----------------------------------|
| P1    | Fractal Rhythm Model (Paper 1) | PHASE-READY | ai-layers/P1-ai-layer.json       |
| MK-P1 | Meta-Kaizen Paper 1          | PHASE-READY | ai-layers/MK-P1-ai-layer.json    |
| DRP-1 | Dependency Resolution Process | PHASE-READY | ai-layers/DRP1-ai-layer.json     |
| SFW-1 | Sentinel v12.3               | PHASE-READY | ai-layers/SFW1-ai-layer.json     |

---

## Authors & License

**Authors:** Thomas Brennan & Claude (Anthropic) & Grok (xAI)

**License:** [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/) — no restrictions, no attribution required.

**GitHub:** [https://github.com/thomasbrennan/Fracttalix](https://github.com/thomasbrennan/Fracttalix)

**Version history:**

| Version | Notes |
|---------|-------|
| v12.3.0 | 8-detector suite (5 core + 3 FRM-derived), Lambda detector, code quality fixes |
| v12.2.0 | Replaced physics-derived framing with signal-processing heuristic language; production() multiplier 3.0→4.5 |
| v12.1.0 | VarCUSUM reset fix, ChannelCoherence Pearson correlation, 374 tests |
| v12.0.0 | Package restructure, PyPI release, review-driven corrections, ablation study |
| v11.0.0 | Corrected overclaimed physics language, added state_dict/load_state, diagnostic window |
| v10.0.0 | 4 signal-processing heuristics (v10 API), 37 steps, 98 tests |
| v9.0.0 | Three-channel model, 26 steps, 65 tests |
| v8.0.0 | Frozen config, WindowBank, 19-step pipeline |
| v7.11–v7.6 | Earlier releases |
