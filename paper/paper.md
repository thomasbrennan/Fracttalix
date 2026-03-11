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

Fracttalix Sentinel is a pure-Python library for real-time streaming anomaly detection grounded in the Three-Channel Dissipative Network Model. It processes one observation at a time with no batching or retraining. On every call to `update_and_check()`, Sentinel runs a deterministic 37-step pipeline and emits a composite anomaly score, per-channel degradation status, and an estimated time-to-collapse with confidence bounds derived from inter-band coupling dynamics.

The library has zero required dependencies: all pipeline steps use the Python standard library, with NumPy, SciPy, and Numba as optional accelerators. It installs via `pip install fracttalix`, exposes a fully typed API, and is released under CC0-1.0.

# Statement of Need

Streaming anomaly detection is required across industrial IoT, network operations, financial tick processing, and critical-infrastructure monitoring. Existing software leaves important gaps.

**Batch-first detectors** such as PyOD [@Zhao2019] and scikit-learn's `IsolationForest` [@Pedregosa2011] require the full dataset before fitting and cannot process data observation-by-observation without periodic retraining.

**Streaming detectors** such as River [@Montiel2021] and ADTK support one-at-a-time processing, but model anomalies as statistical deviations without modeling degradation dynamics or providing diagnostic information about remaining system coherence.

Neither category provides collapse-precursor diagnostics: detecting that coupling architecture is degrading before the anomaly score crosses a threshold, or estimating a time window before coherence collapse.

Fracttalix addresses this gap for practitioners who need quantified time-to-collapse estimates, organic-vs-intervention discrimination, or real-time coupling monitoring without batch re-fitting.

# State of the Field

| Library | Streaming | Collapse Physics | Zero Core Deps | Time-to-Collapse |
|---------|-----------|-----------------|----------------|-----------------|
| PyOD | No | No | No | No |
| ADTK | Yes | No | No | No |
| River | Yes | No | No | No |
| **Fracttalix** | **Yes** | **Yes** | **Yes** | **Yes** |

PyOD [@Zhao2019] provides a comprehensive batch outlier detection toolkit. River [@Montiel2021] offers streaming machine learning including anomaly detection. Neither models physical degradation dynamics or provides collapse forecasting. Fracttalix complements these tools by adding physics-grounded collapse precursor detection to the streaming anomaly detection landscape.

# Software Design

Every call to `update_and_check()` runs 37 `DetectorStep` subclasses in sequence via a shared `StepContext`. Steps are organized in six groups:

- **Foundation**: EWMA, CUSUM, Page-Hinkley [@Page1954], regime detection
- **Temporal**: shear-turbulence, phase space, change-point detection
- **Frequency**: rhythm periodicity, fractal indices, Permutation Entropy [@BandtPompe2002]
- **Complexity**: early warning signals, adaptive quantile baseline, Mahalanobis distance
- **Channel**: three-channel integration implementing the Dissipative Network Model [@FRM2026]
- **Physics**: collapse dynamics and forecasting

The three-channel model decomposes signals into structural (distributional moments), rhythmic (FFT-based phase-amplitude coupling via Modulation Index [@Tort2010]), and temporal (degradation event ordering) channels. A multi-stage cascade trigger reduces false-positive critical alerts relative to single-threshold approaches.

Four physics-derived steps provide collapse forecasting: maintenance burden encoding adaptive reserve [@Tainter1988], PAC pre-cascade detection, diagnostic window estimation under the Kuramoto framework [@Kuramoto1984], and reversed-sequence intervention signature detection.

# Research Impact Statement

Fracttalix Sentinel provides researchers studying critical transitions with empirical test signals (diagnostic window, Kuramoto order parameters) for theoretical collapse models. The built-in `SentinelBenchmark` harness evaluates detection across five labeled archetypes with reproducible results (seed 42, n=1000):

| Archetype          | F1   | AUPRC | VUS-PR |
|--------------------|------|-------|--------|
| Point anomaly      | 0.36 | 0.29  | 0.32   |
| Contextual anomaly | 0.20 | 0.19  | 0.21   |
| Collective anomaly | 0.11 | 0.09  | 0.09   |
| Drift              | 0.67 | 0.50  | 0.50   |
| Variance anomaly   | 0.69 | 0.52  | 0.52   |

Comparison against PyOD ECOD [@Zhao2019] and River HalfSpaceTrees [@Montiel2021] and an ablation study are included in the benchmark package.

# AI Usage Disclosure

Development of Fracttalix Sentinel was assisted by Claude (Anthropic, Claude 3.5/4 Sonnet and Opus models) and Grok (xAI, Grok 3/4 models). AI tools were used for: (1) code generation and refactoring of pipeline step implementations, (2) test suite authoring, (3) documentation and docstring drafting, and (4) paper text drafting and editing. All AI-generated outputs were reviewed, tested against the 405-test automated suite, and verified by the author. The core scientific contributions—the Three-Channel Dissipative Network Model, the 37-step pipeline architecture, channel decomposition strategy, collapse physics formulations, and the Fractal Rhythm Model theoretical framework—were designed by the author. AI tools assisted with implementation, refinement, and iteration on the author's designs.

# Acknowledgements

The theoretical framework underlying the Three-Channel Model is documented in the Fractal Rhythm Model working papers [@FRM2026]. This work received no external funding. The author declares no conflicts of interest.

# References
