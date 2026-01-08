# Fracttalix

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18180754.svg)](https://doi.org/10.5281/zenodo.18180754)

**Lightweight open-source toolbox for exploratory fractal and entropy metrics in univariate time series**

Fracttalix is a single-file Python command-line tool designed for quick, cautious screening of long-range correlations, self-similarity, and complexity in time series data. It implements five established monofractal and entropy metrics with built-in phase-randomized surrogates for significance, adaptive detrending, and a synthetic stress-test suite.

Current version: **v2.6.4** (January 2026) — with improved Higuchi FD robustness  
License: **CC0 1.0 Universal** (public domain) — no rights reserved

## Features

- **Metrics**
  - Hurst exponent (R/S analysis)
  - Higuchi fractal dimension (improved with k_max cap and range warnings)
  - Detrended fluctuation analysis (DFA) exponent
  - Sample entropy
  - Petrosian fractal dimension

- **Statistical caution**
  - Phase-randomized surrogates (default 100) with one-sided p-values and 95% CI
  - Clear interpretation notes (“likely genuine structure” or “consistent with noise”)

- **Preprocessing**
  - Optional linear or wavelet detrending (db4)

- **Built-in validation**
  - Synthetic stress-test suite (white, persistent, periodic, chaotic, pink 1/f)
  - Quick sanity checks on controlled series

- **Usage**
  - Simple CLI: `python fracttalix.py data.csv --col 0 --surrogates 100 --detrend --plot --json`
  - Minimal dependencies (numpy, scipy, optional matplotlib/pywt)

## Installation

```bash
git clone https://github.com/thomasbrennan/Fracttalix.git
cd Fracttalix
# No pip install needed — run directly
python fracttalix.py --help

Optional for full features:
Bashpip install matplotlib pywavelets
Quick Example
Bashpython fracttalix.py bitcoin_daily.csv --col 1 --detrend --surrogates 100
Output includes base metrics, surrogate significance, and warnings for short/unreliable series.
Why Fracttalix?

Convenience: All common monofractal metrics + surrogates in one file.
Caution: Built-in statistical testing to avoid over-interpreting noise.
Teaching & prototyping: Ideal for classrooms, quick checks, or before advanced modeling.
Open: CC0 — use, modify, extend freely.

Papers & Applications
See the /papers branch for exploratory notes applying Fracttalix to landmark datasets:

Dragon Kings in synthetic bubble data
Monofractal screening in PhysioNet HRV (healthy vs. CHF)

More coming weekly.

Limitations

Monofractal only (no multifractal extensions like MF-DFA)
Known biases in finite samples (e.g., Hurst upward bias)
Best for series >500–1000 points
Equivalent to libraries like nolds/pyunicorn — advantage is CLI + surrogates bundle

Future Work

Adaptive refinements
Broader validation
Community contributions welcome

Citation (suggested)
If you find Fracttalix useful, consider citing the repo:
textBrennan, T. G. (2026). Fracttalix: Lightweight toolbox for exploratory fractal and entropy metrics (v2.6.4). 
https://github.com/thomasbrennan/Fracttalix
