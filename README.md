# Fracttalix

**Fracttalix v2.6.2** — Lightweight, open-source (CC0 public domain) Python CLI tool for exploratory fractal and rhythmic metrics in univariate time series.

A "pocket knife" for quick checks of persistence, self-similarity, complexity, and potential regime shifts. Ideal for teaching, prototyping, or sanity-testing before deeper modeling.

## Key Features
- Standard metrics: Hurst (R/S), Higuchi FD, DFA exponent, Sample Entropy, Petrosian FD.
- Optional linear detrending (`--detrend`).
- Built-in plotting (`--plot`).
- JSON output (`--json`).
- **Surrogate significance testing** (`--surrogates N`): Phase-randomized surrogates to distinguish genuine structure from noise (p<0.05 indicates likely real signal). *Especially useful for noisy/short series—provides cautious guidance on when not to over-interpret.*

## Installation
```bash
pip install numpy pandas scipy matplotlib  # matplotlib optional

Quick Usage
Save as fracttalix.py and run:
•  Basic analysis: python fracttalix.py data.csv --col 1
•  With detrend: python fracttalix.py data.csv --detrend
•  Plot series: python fracttalix.py data.csv --plot
•  Surrogate test (recommended for noisy data): python fracttalix.py data.csv --surrogates 100
•  JSON output: python fracttalix.py data.csv --json
•  No file (synthetic demo): python fracttalix.py
Interpretation Tips
•  High Hurst/DFA (>0.5): Persistence/long-memory.
•  Higuchi ~1.5: Fractal roughness (Brownian-like).
•  Low Sample Entropy: More regular/complex.
•  Use surrogates for confidence: p > 0.05 → “consistent with noise—interpret cautiously”.
Optional Heuristic Framework
See Final_Mathematical_Formulations.pdf for falsifiable mathematical versions of 11 conceptual axioms (resilience, rhythm, self-similarity).
Companion Reflective Essay
Fractal_Rhythm_Companion.pdf — Informal parallels with philosophical resilience traditions.
License
CC0 1.0 Universal — public domain. Use, modify, share freely. No rights reserved.
Feedback, forks, and real-data examples welcome! 🚀
Thomas G. Brennan (with contributions from Grok/xAI)
January 2026
