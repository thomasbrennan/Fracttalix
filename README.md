# Fracttalix Sentinel v7.3

**Lightweight, regime-aware anomaly detection for time series**  
Single-file Python | No dependencies required | CC0 public domain

Sentinel is a simple, fast anomaly detector for univariate or multivariate time series. It uses adaptive EWMA thresholding + bidirectional CUSUM to spot regime shifts early, making it especially useful for:

- Finance (volatility regime changes, BTC/asset monitoring)  
- Physiology (HRV, vital sign monitoring)  
- IoT & infrastructure (sensor drift, network anomalies)  
- Research & exploratory analysis (quick screening with surrogates)

### Current Version: v7.3 (March 2026)
- Two-sided alerts (upper & lower anomalies)
- Per-channel multivariate detection with flexible aggregation
- Soft/full regime reset (preserve baselines during gradual drifts)
- NaN/Inf imputation (skip or mean)
- State persistence (JSON save/load for long-running sessions)
- Volatility-adaptive smoothing
- Optional Numba JIT acceleration for fractal/entropy metrics (if installed)
- Optional parallel surrogate generation & tqdm progress
- Proper FFT-based phase randomization for surrogates
- Buffered CSV export (low filesystem overhead: accumulate in memory, flush on interval or manually)

### Quick Start
```python
from fracttalix_sentinel import Detector_7_3

detector = Detector_7_3(
    alpha=0.12,
    early_mult=2.75,
    fixed_mult=3.2,
    two_sided=True,
    per_channel_detection=True,
    volatility_adaptive=True,
    verbose_explain=True,
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

# Manual flush (optional)
detector.flush_csv()

