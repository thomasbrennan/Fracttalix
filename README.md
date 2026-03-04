# Fracttalix Sentinel v7.6

**Lightweight, regime-aware anomaly detection for time series**  
Single-file Python | No dependencies required | CC0 public domain

Sentinel is a simple, fast anomaly detector for univariate or multivariate time series. It uses adaptive EWMA thresholding + bidirectional CUSUM to spot regime shifts early, making it especially useful for:

- Finance (volatility regime changes, BTC/asset monitoring)  
- Physiology (HRV, vital sign monitoring)  
- IoT & infrastructure (sensor drift, network anomalies)  
- Research & exploratory analysis (quick screening with surrogates)

### Current Version: v7.6 (March 2026)
- Two-sided alerts (upper & lower anomalies)
- Per-channel multivariate detection with flexible aggregation
- Soft/full regime reset (preserve baselines during gradual drifts)
- NaN/Inf imputation (skip or mean)
- State persistence (JSON save/load for long-running sessions) — complete including all v7.6 state
- Volatility-adaptive smoothing
- Optional Numba JIT acceleration for fractal/entropy metrics (if installed)
- Optional parallel surrogate generation & tqdm progress
- Proper FFT-based phase randomization for surrogates (Theiler et al. 1992)
- Buffered CSV export (low filesystem overhead: accumulate in memory, flush on interval or manually)
- v7.4–7.6 Meta-Kaizen fluid dynamics improvements:
  - Sentinel Turbulence Index (STI): rolling-window Reynolds number analog. Characterizes flow state (laminar/transitional/turbulent) before applying thresholds. Modulates both alert multipliers AND alpha for deeper integration. All thresholds user-configurable. sti_regime label in result dict.
  - Boundary Layer Warning: fourth alert tier. Fires after tps_periods consecutive steps within tps_proximity_pct of early threshold without crossing. Implemented in BOTH univariate and multivariate paths. Present in all result dict paths.
  - Oscillation Damping Filter: suppresses resonance artifacts (rapid early_alert fire/clear cycles without confirmed_alert). Implemented via extra_mult parameter — no base multiplier mutation, no TypeError risk. Per-channel lookback fully persisted in save_state/load_state.
  - CUSUM Pressure Differential (CPD): optional flag (cpd_enabled). Directional pressure signal in every active result dict. +1.0 upward, -1.0 downward, ~0 bidirectional volatility.

### Quick Start
```python
from fracttalix_sentinel import Detector_7_6

detector = Detector_7_6(
    alpha=0.12,
    early_mult=2.75,
    fixed_mult=3.2,
    two_sided=True,
    per_channel_detection=True,
    volatility_adaptive=True,
    verbose_explain=True,
    turbulence_adaptive=True,
    sti_laminar_threshold=1.0,
    sti_turbulent_threshold=2.0,
    boundary_warning_enabled=True,
    tps_proximity_pct=0.15,
    tps_periods=5,
    oscillation_damping_enabled=True,
    oscillation_count_threshold=3,
    oscillation_periods=20,
    oscillation_mult_bump=0.10,
    cpd_enabled=True,
    csv_output_path="alerts.csv",          # Optional: save alerts to CSV
    csv_flush_interval=60                  # Flush buffer every 60 alerts (0 = manual only)
)

# Feed data (scalar for univariate, list for multivariate)
for value in your_time_series:
    result = detector.update_and_check(value)
    if result.get("early_alert"):
        print("Early alert:", result)
    if result.get("confirmed_alert"):
        print("Confirmed alert:", result)
    if result.get("boundary_warning"):
        print("Boundary warning:", result)
    if result.get("oscillation_damping_active"):
        print("Oscillation damping active:", result)

# Manual flush (optional)
detector.flush_csv()

