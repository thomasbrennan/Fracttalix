# Fracttalix

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18180754.svg)](https://doi.org/10.5281/zenodo.18180754)

**Lightweight open-source toolbox for exploratory fractal and entropy metrics in univariate time series**

Fracttalix is a single-file Python command-line tool designed for quick, cautious screening of long-range correlations, self-similarity, and complexity in time series data. It implements five established monofractal and entropy metrics with built-in phase-randomized surrogates for significance, adaptive detrending, and a synthetic stress-test suite.

Current version: 

Fracttalix  v2.6.5 py "Sentinel"

Overview

Fracttalix Sentinel is a lightweight, high-performance anomaly detection library optimized for early identification of deviations in time-series data. It combines adaptive EWMA thresholding with bidirectional CUSUM-based regime change detection to deliver low-latency alerts while maintaining strong specificity and minimal false positives.

Core Capabilities

•  Adaptive EWMA-based thresholding for sensitive early detection

•  Bidirectional CUSUM for accurate detection of both positive and negative regime shifts

•  Controlled warm-up phase with fixed-threshold fallback for robust initialization

•  Multivariate input support with configurable aggregation function (mean, max, or custom)

•  Optional volatility-adaptive mode for improved performance in high-variance environments

•  Built-in production features: NaN/Inf validation, selective state reset, detailed verbose output

•  Released under CC0 1.0 Universal (public domain) — unrestricted use, modification, and distribution

Target Applications

•  Finance — volatility regime detection, risk signal monitoring, market anomaly identification

•  Healthcare / Medical — real-time vital sign monitoring, early deterioration detection, wearable data analysis

•  Infrastructure, IoT & Security — sensor drift detection, network anomaly identification, subtle failure precursors

•  Research & Analytics — exploratory time-series analysis, reproducible anomaly detection studies

Performance Summary (Simulated Benchmarks)

•  False positive rate in white noise: ~1.7%

•  Early detection latency improvement on persistent drifts: 9–14 points ahead of fixed-threshold methods

•  Regime change reset success rate (up/down spikes): 98% within 8–12 points

•  Volatility-adaptive mode latency reduction: ~27%

Quick Start

from fracttalix_sentinel import Detector_2_6_5



detector = Detector_2_6_5(

    alpha=0.12,

    early_mult=2.75,

    fixed_mult=3.2,

    warm_up_period=60,

    multivariate=False,

    volatility_adaptive=True,

    verbose_explain=True

)

# Process time-series values sequentially

for value in your_time_series:

    result = detector.update_and_check(value)

    if result.get("early_alert"):

        print("Early anomaly signal detected")

    if result.get("confirmed_alert"):

        print("Confirmed anomaly — review recommended")

Installation

Copy fracttalix_sentinel.py into your project directory.

No external dependencies required beyond the Python standard library.

License

CC0 1.0 Universal — Dedicated to the public domain.

No restrictions on use, modification, or redistribution.

Version

Fracttalix Sentinel v2.6.5

Release date: January 2026

Developed in Entwood Hollow research station, Trinity County, California

