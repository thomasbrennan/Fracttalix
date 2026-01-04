# Fracttalix
Fracttalix v2.6 is a lightweight, command-line Python tool designed for rapid exploratory analysis of univariate time series using five well-established fractal and rhythmic metrics:
•  Hurst exponent (R/S method) — measures long-range persistence
•  Higuchi fractal dimension — quantifies roughness and self-similarity
•  Detrended fluctuation analysis (DFA) exponent — robust persistence estimate
•  Sample entropy — assesses rhythmic complexity/irregularity
•  Petrosian fractal dimension — fast complexity proxy
The tool is intentionally minimal (core dependencies: NumPy/SciPy; optional Matplotlib/Pandas), runs directly on CSV input, and provides clean tabular output. New in v2.6: optional plotting, column selection, JSON export, short-series warnings, and improved robustness.
Released under CC0 (public domain), it serves as a simple “pocket knife” for detecting regime shifts and structural changes — ideal for quick sanity checks, teaching, or as an early-warning layer before heavier modeling.
No roadmap, no bloat — just effective, freely reusable diagnostics.
