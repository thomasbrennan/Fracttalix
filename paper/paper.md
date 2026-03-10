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
    orcid: 0009-0002-6353-7115
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

Every call to `update_and_check()` runs 37 `DetectorStep` subclasses in strict sequence via a shared `StepContext`. Steps are organized in six groups: Foundation (EWMA, CUSUM, Page-Hinkley [@Page1954], regime detection); Temporal (shear-turbulence, phase space, change-point); Frequency (rhythm periodicity/fractal indices, Permutation Entropy [@BandtPompe2002]); Complexity (early warning signals, adaptive quantile baseline, Mahalanobis distance); Channel (three-channel integration); and Physics (collapse dynamics). An ablation study in the benchmark package quantifies each group's F1 contribution.

## Three-Channel Model

The pipeline implements the Three-Channel Dissipative Network Model [@FRM2026]. **Channel 1 (Structural)** computes distributional moments and stationarity at every step; elevated variance and lag-1 autocorrelation are the classical critical-slowing-down signature. **Channel 2 (Rhythmic)** decomposes the signal via FFT into five carrier-wave bands and measures phase-amplitude coupling (PAC) across six band pairs using the Modulation Index [@Tort2010]; declining composite coupling score triggers `COUPLING_DEGRADATION` before any single-band threshold is breached. **Channel 3 (Temporal)** records the ordering of degradation events: because thermodynamic irreversibility dictates that coupling degrades before coherence collapses organically, a reversed ordering is treated as a distinct signal class. The multi-stage cascade trigger (band anomaly → coupling degradation → channel decoupling → `CASCADE_PRECURSOR`) reduces false-positive critical alerts relative to any single threshold.

## Collapse Dynamics

Four physics-derived steps provide collapse forecasting. **Maintenance Burden** μ = 1 − κ̄ [@Tainter1988] encodes the fraction of adaptive reserve consumed; four regimes (`HEALTHY`, `REDUCED_RESERVE`, `TAINTER_WARNING`, `TAINTER_CRITICAL`) are reported continuously. **PAC pre-cascade detection** fires when PAC degrades before mean coupling κ̄ crosses its threshold [@Tort2010], providing an earlier precursor than the cascade logic alone. **Diagnostic window** Δt = (κ̄ − κ_c) / |dκ̄/dt| estimates remaining steps before coherence collapse under the Kuramoto synchronization framework [@Kuramoto1984], with pessimistic, expected, and optimistic bounds and a `HIGH/MEDIUM/LOW` confidence grading based on rate stability. **Reversed-sequence detection** classifies a coherence collapse preceding coupling decline as an intervention signature, quantified by `intervention_signature_score` (0.0–1.0).

# Performance

The built-in `SentinelBenchmark` harness evaluates Sentinel against five labeled anomaly archetypes: point anomalies (sparse 8σ spikes), contextual anomalies (values anomalous relative to sinusoidal seasonal context), collective anomalies (extended runs of moderately elevated values), drift (slow linear mean drift starting mid-series), and variance anomalies (sudden 4× variance explosion). Metrics reported are F1 score, area under the precision-recall curve (AUPRC), volume under the surface of precision-recall (VUS-PR), and mean detection lag in observations.

The following results were produced with seed 42 and n=1000 observations per archetype using `benchmark.run_suite(seed=42)`:

| Archetype          | F1   | AUPRC | VUS-PR | Detection Lag (obs) |
|--------------------|------|-------|--------|---------------------|
| Point anomaly      | 0.36 | 0.29  | 0.32   | 0                   |
| Contextual anomaly | 0.20 | 0.19  | 0.21   | 0                   |
| Collective anomaly | 0.11 | 0.09  | 0.09   | 0                   |
| Drift              | 0.67 | 0.50  | 0.50   | 0                   |
| Variance anomaly   | 0.69 | 0.52  | 0.52   | 0                   |

V12 includes comparison against two external baselines: PyOD ECOD [@Zhao2019] and River HalfSpaceTrees [@Montiel2021], run over the same five archetype datasets. An ablation study quantifies the F1 contribution of each step group by sequentially disabling the Foundation, Channel, and Physics groups and reporting the resulting score delta on the collective and drift archetypes, where the multi-channel architecture contributes most visibly. The Physics group (steps 26–37) contributes primarily to earlier detection lag on the collective and variance archetypes, as the PAC pre-cascade signal fires several observations before the composite anomaly score crosses the EWMA threshold.

# Acknowledgements

Development of Fracttalix Sentinel was assisted by Claude (Anthropic) and Grok (xAI) AI coding tools. The theoretical framework underlying the Three-Channel Model and the Fractal Rhythm Model metrics is documented in the Fractal Rhythm Model working papers [@FRM2026].

# References
