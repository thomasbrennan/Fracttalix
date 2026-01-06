# Fracttalix



**Fracttalix v2.6.3** — Lightweight, open-source (CC0 public domain) Python CLI tool for exploratory fractal and rhythmic metrics in univariate time series.



A "pocket knife" for quick checks of persistence, self-similarity, complexity, and potential regime shifts. Ideal for teaching, prototyping, or sanity-testing before deeper modeling.



## Key Features



- Standard metrics: Hurst (R/S), Higuchi FD, DFA exponent, Sample Entropy, Petrosian FD.

- Optional linear detrending (`--detrend`).

- Built-in plotting (`--plot`).

- JSON output (`--json`).

- **Surrogate significance testing** (`--surrogates N`): Phase-randomized surrogates to distinguish genuine structure from noise (p<0.05 indicates likely real signal). *Especially useful for noisy/short series—provides cautious guidance on when not to over-interpret.*

- Simplified stress-test on synthetic series (`--stress`).



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

•  Synthetic stress-test: python fracttalix.py --stress

•  No file (synthetic demo): python fracttalix.py

Interpretation Tips
•  High Hurst/DFA (>0.5): Persistence/long-memory.

•  Higuchi ~1.5: Fractal roughness (Brownian-like).

•  Low Sample Entropy: More regular/complex.

•  Use surrogates for confidence: p > 0.05 → “consistent with noise—interpret cautiously”.



What's New in V 2.6.3-



1.Replaced placeholders with full implementations:

	•  DFA: Complete detrended fluctuation analysis (Peng et al. 1994 style, log-spaced scales, proper fluctuation averaging).

	•  Sample Entropy: Full Richman & Moorman 2000 implementation (vectorized templates, r=0.2*std default).

	•  Petrosian FD: Exact sign-change formula (Petrosian 1993)—fast and robust.
→ Fully done—no more warnings or dummy returns.

2.  Added surrogate-based validation:

	•  New --surrogates N flag runs phase-randomized surrogates (Theiler 1992).

	•  Reports observed value, p-value, 95% CI, and clear note (“likely genuine structure” or “consistent with noise”).

	•  One-sided test tuned for high values indicating structure.
→ Fully done—directly provides statistical caution for noisy/short series.

3.  Lightweight comparison with change-point methods:

	•  Not a full built-in benchmark (would bloat scope), but:

		•  Surrogates enable user-level change-point flavor (run pre/post windows, compare p-values).

		•  Stress-test and detrend support regime-shift exploration.

		•  Docs/interpretation tips guide “look for jumps” + surrogate validation.


Bonus Improvements 

•  --detrend flag added (critical for real non-stationary data).

•  Warnings captured and output (JSON + pretty-print).

•  Stress-test retained and cleaned.
