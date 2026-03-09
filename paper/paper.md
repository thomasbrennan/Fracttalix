---
title: 'Fracttalix Sentinel: Real-Time Streaming Anomaly Detection via a Three-Channel Dissipative Network Model'
tags:
  - Python
  - anomaly detection
  - streaming
  - time series
  - online learning
  - phase-amplitude coupling
  - Kuramoto synchronization
authors:
  - name: Thomas Brennan
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2026-03-09
bibliography: paper.bib
---

# Summary

Fracttalix Sentinel is a pure-Python library for real-time streaming anomaly detection grounded in the Three-Channel Dissipative Network Model. It processes one scalar or multivariate observation at a time with no batching, no retraining, and no warmup gap beyond a configurable initialization window. On every call to `update_and_check()`, Sentinel runs a deterministic 37-step pipeline and emits a `SentinelResult` containing a composite anomaly score, per-channel degradation status, and—uniquely among streaming anomaly detectors—an estimated time-to-collapse (`diagnostic_window_steps`) with `HIGH`, `MEDIUM`, or `LOW` confidence bounds derived from the rate of change of inter-band coupling strength.

The library operates without any external dependencies for core use: all 37 pipeline steps are implemented on the Python standard library, with NumPy, SciPy, Numba, Matplotlib, and tqdm treated as optional performance accelerators. Fracttalix Sentinel installs via `pip install fracttalix`, exposes a fully typed public API, and is released under CC0-1.0 (public domain). A built-in benchmark harness (`SentinelBenchmark`) and a dependency-free asyncio HTTP server (`SentinelServer`) are included in the same package.

# Statement of Need

Streaming time-series anomaly detection is a practical requirement across industrial IoT, network operations, financial tick processing, and critical-infrastructure monitoring. Existing software falls into two groups that leave important gaps.

**Batch-first detectors** such as PyOD [@Zhao2019] and scikit-learn's `IsolationForest` [@Pedregosa2011] require the full dataset to be available before fitting. They cannot process data arriving observation by observation without periodic retraining, which introduces latency proportional to retraining frequency and breaks causal isolation.

**Streaming detectors** such as River [@Montiel2021] and ADTK support true one-observation-at-a-time processing, but they model anomalies as statistical deviations from a learned distribution. They do not model the physical dynamics of system degradation, and they provide no diagnostic information about *why* anomaly scores are elevated or *how long* the system is likely to remain coherent.

Neither category provides collapse-precursor diagnostics: the ability to detect that coupling architecture is degrading before the primary anomaly score crosses a threshold, nor to estimate a remaining time window before coherence collapse.

| Library | Streaming | Collapse Physics | Zero Core Deps | Time-to-Collapse |
|---------|-----------|-----------------|----------------|-----------------|
| PyOD | No | No | No | No |
| ADTK | Yes | No | No | No |
| River | Yes | No | No | No |
| **Fracttalix** | **Yes** | **Yes** | **Yes** | **Yes** |

Fracttalix Sentinel addresses this gap for four practitioner communities. Industrial IoT operators can receive a quantified time-to-collapse estimate before a sensor stream reaches alert threshold, allowing pre-emptive maintenance scheduling. Network operations teams can distinguish organic degradation from externally driven disruption via the intervention signature metric. Financial data engineers can monitor market microstructure coupling in real time without batch re-fitting. Researchers studying critical transitions can use the diagnostic window and Kuramoto order outputs as empirical test signals for theoretical collapse models.

# Methodology

## 37-Step Pipeline

Every call to `update_and_check()` runs all 37 `DetectorStep` subclasses in strict sequence. Steps read from and write to a shared `StepContext.scratch` dictionary; no step has side effects outside this context. The pipeline is organized in six functional groups:

1. **Foundation (steps 1–7):** Core EWMA baseline, structural snapshot, FFT frequency decomposition, CUSUM persistent shift, regime change detection with soft alpha boost, variance CUSUM, and Page-Hinkley drift [@Page1954].
2. **Temporal (steps 8–11):** Shear-Turbulence Index, Temporal Phase Space reconstruction, oscillation damping, and change-point detection.
3. **Frequency (steps 12–15):** Rhythm Periodicity Index (FFT spectral coherence), Rhythm Fractal Index (Hurst exponent via R/S analysis), Synchronization Stability Index (Kuramoto proxy), and Permutation Entropy [@BandtPompe2002].
4. **Complexity (steps 16–20):** Early Warning Signals (variance plus lag-1 autocorrelation), Adaptive Quantile Baseline, seasonal decomposition, Mahalanobis distance (multivariate mode), and Robust Residual Score.
5. **Channel (steps 21–25):** Per-band anomaly detection, cross-frequency coupling measurement, structural-rhythmic coherence, cascade precursor detection, and temporal degradation sequence logging.
6. **Physics (steps 26–37):** The four physics-derived capabilities introduced in V12, described below.

## Three-Channel Model

The pipeline implements the Three-Channel Dissipative Network Model from the Fractal Rhythm Model working papers [@FRM2026]:

**Channel 1 (Structural)** monitors the network as an active signal transmitter. `StructuralSnapshotStep` computes mean, variance, skewness, kurtosis, lag-1 autocorrelation, and a stationarity indicator at every step. Elevated variance and autocorrelation together constitute the classical critical-slowing-down signature of an approaching bifurcation.

**Channel 2 (Rhythmic)** monitors broadband multiplexed oscillatory transmission. `FrequencyDecompositionStep` decomposes the signal via FFT into five carrier-wave bands (ultra-low, low, mid, high, ultra-high) with per-band power and instantaneous phase. `CrossFrequencyCouplingStep` computes the phase-amplitude coupling (PAC) Modulation Index [@Tort2010] across six slow-phase/fast-amplitude band pairs, producing a composite coupling matrix and triggering `COUPLING_DEGRADATION` when the composite score falls below threshold.

**Channel 3 (Temporal)** monitors the one-way irreversible ordering of degradation events. `DegradationSequenceStep` logs the temporal sequence in which Channels 1 and 2 degrade. Because thermodynamic irreversibility dictates that coupling degrades before coherence collapses in an organic failure, a reversed ordering constitutes a distinct signal class.

The cascade logic proceeds as: band anomaly detected → cross-frequency coupling degrades → structural-rhythmic channels decouple → `CASCADE_PRECURSOR` (CRITICAL severity). This multi-stage trigger substantially reduces the false-positive rate of the critical alert relative to any single-condition threshold.

## Collapse Dynamics (V12 Physics Group)

V12 adds four steps derived from network collapse physics:

**Maintenance Burden μ** (`MaintenanceBurdenStep`) implements the Tainter collapse condition [@Tainter1988]. The burden μ = N · κ̄ · E_coupling / P_throughput measures the fraction of network energy consumed by coupling maintenance. When μ → 1, adaptive reserve is exhausted. Four regimes are reported: `HEALTHY` (μ < 0.5), `REDUCED_RESERVE` (0.5–0.75), `TAINTER_WARNING` (0.75–0.9), and `TAINTER_CRITICAL` (≥ 0.9).

**PAC Pre-Cascade Detection** (`PACDegradationStep`) exploits the finding that phase-amplitude coupling degrades before mean coupling strength κ̄ measurably decreases [@Tort2010]. This provides an earlier warning signal than the V9 cascade precursor. The `pre_cascade_pac` flag is set when PAC degradation rate exceeds threshold while κ̄ remains above critical.

**Diagnostic Window Δt** (`DiagnosticWindowStep`) estimates the remaining steps before coherence collapse using:

Δt = (κ̄ − κ_c) / |dκ̄/dt|

where κ_c is estimated from the power-weighted frequency distribution (`CriticalCouplingEstimationStep`) following the Kuramoto synchronization framework [@Kuramoto1984], and dκ̄/dt is computed from a rolling coupling history (`CouplingRateStep`). The estimate is only active when κ̄ > κ_c and dκ̄/dt < 0. Confidence is graded `HIGH`, `MEDIUM`, or `LOW` based on rate stability. Supercompensation (adaptive recovery, dκ̄/dt > 0 after prior decline) is also detected and reported.

**Reversed Sequence Detection** (`ReversedSequenceStep`) identifies the intervention signature. The thermodynamic arrow of organic collapse is: coupling degrades first, coherence collapses second. A reversed sequence—Kuramoto order Φ collapsing before κ̄ decreases—indicates measurement error, non-universality class membership, or deliberate external intervention. The `intervention_signature_score` (0.0–1.0) quantifies confidence in this classification.

# Performance

The built-in `SentinelBenchmark` harness evaluates Sentinel against five labeled anomaly archetypes: point anomalies (sparse 8σ spikes), contextual anomalies (values anomalous relative to sinusoidal seasonal context), collective anomalies (extended runs of moderately elevated values), drift (slow linear mean drift starting mid-series), and variance anomalies (sudden 4× variance explosion). Metrics reported are F1 score, area under the precision-recall curve (AUPRC), volume under the surface of precision-recall (VUS-PR), and mean detection lag in observations.

V12 includes comparison against two external baselines: PyOD ECOD [@Zhao2019] and River HalfSpaceTrees [@Montiel2021], run over the same five archetype datasets. An ablation study quantifies the F1 contribution of each step group by sequentially disabling the Foundation, Channel, and Physics groups and reporting the resulting score delta on the collective and drift archetypes, where the multi-channel architecture contributes most visibly. The Physics group (steps 26–37) contributes primarily to earlier detection lag on the collective and variance archetypes, as the PAC pre-cascade signal fires several observations before the composite anomaly score crosses the EWMA threshold.

# Acknowledgements

Development of Fracttalix Sentinel was assisted by Claude (Anthropic) and Grok (xAI) AI coding tools. The theoretical framework underlying the Three-Channel Model and the Fractal Rhythm Model metrics is documented in the Fractal Rhythm Model working papers [@FRM2026].

# References
