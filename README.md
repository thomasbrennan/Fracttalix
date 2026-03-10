# Fracttalix Sentinel v9.0

https://doi.org/10.5281/zenodo.18859299

**Three-channel streaming anomaly detector grounded in the Fractal Rhythm Model**
Single-file Python | Zero required dependencies | CC0 public domain

---

## What changed in v9.0

v9.0 extends the v8.0 pipeline architecture to implement the **three-channel model of dissipative network information transmission** derived in the Meta-Kaizen Paper 6 theoretical framework.

The v8.0 pipeline treated the input signal as a single composite stream. v9.0 decomposes it into three independent information channels — structural, rhythmic, and temporal — and monitors cross-channel coherence as a first-class detection signal. This produces earlier warnings and richer diagnostics without breaking any existing behavior.

| | v8.0 | v9.0 |
|---|---|---|
| Pipeline steps | 19 | 26 (19 preserved + 7 new) |
| Information channels | 1 (composite) | 3 (structural, rhythmic, temporal) |
| Frequency analysis | Composite RPI/RFI | 5-band carrier wave decomposition |
| Coupling detection | None | Cross-frequency phase-amplitude coupling |
| Channel coherence | None | Structural-rhythmic coherence monitoring |
| Cascade warning | None | CASCADE_PRECURSOR (CRITICAL severity) |
| Alert structure | Dict with string reasons | Structured `Alert` objects with severity levels |
| Degradation logging | None | Temporal sequence logging with diagnostic narratives |

All v8.0 and v7.x APIs remain fully backward compatible. No existing step is removed. No existing behavior changed. All extensions are additive.

---

## Three-channel model

**Channel 1 — Structural.** Network topology as active transmitter. Monitors mean, variance, skewness, kurtosis, autocorrelation, and stationarity of the input stream as an independent information channel.

**Channel 2 — Rhythmic.** Broadband multiplexed oscillatory transmission. Decomposes the composite rhythmicity signal into five independent frequency band carrier waves (ultra-low through ultra-high) and monitors cross-frequency coupling between adjacent bands as a higher-order information channel.

**Channel 3 — Temporal.** One-way irreversible carrier wave. Logs the temporal sequence of channel degradation events as diagnostic information about regime change type and severity. The ordering of degradation is itself informative — coupling degrades before individual channels fail.

---

## New in v9.0

### New alert classes

| Alert type | Severity | Description |
|---|---|---|
| `BAND_ANOMALY` | ALERT | Per-carrier-wave anomaly invisible to composite detection |
| `COUPLING_DEGRADATION` | WARNING | Cross-frequency coupling breakdown (earlier warning signal) |
| `STRUCTURAL_RHYTHMIC_DECOUPLING` | ALERT | Channel 1-2 coherence loss |
| `CASCADE_PRECURSOR` | CRITICAL | Tipping cascade precursor — scale-level reversion risk |

### New pipeline steps (7)

| Step | Channel | Purpose |
|---|---|---|
| `StructuralSnapshotStep` | 1 | Computes structural properties (mean, variance, skewness, kurtosis, autocorrelation, stationarity) |
| `FrequencyDecompositionStep` | 2 | FFT decomposition into 5 frequency band carrier waves with power and phase |
| `BandAnomalyStep` | 2 | Per-band anomaly detection invisible to composite detectors |
| `CrossFrequencyCouplingStep` | 2 | Phase-amplitude coupling between adjacent frequency bands |
| `ChannelCoherenceStep` | 1+2 | Structural-rhythmic coherence measurement and decoupling detection |
| `CascadePrecursorStep` | 1+2+3 | Tipping cascade detection requiring convergence of multiple degradation signals |
| `DegradationSequenceStep` | 3 | Temporal ordering and narrative logging of channel degradation events |

### New data structures

- **`FrequencyBands`** — Five-band power/phase decomposition snapshot
- **`StructuralSnapshot`** — Channel 1 structural properties at current timestep
- **`CouplingMatrix`** — Cross-frequency coupling coefficients with composite score and trend
- **`ChannelCoherence`** — Structural-rhythmic coherence measurement
- **`DegradationSequence`** — Temporal ordering of degradation events with diagnostic narrative
- **`Alert`** — Structured alert with `AlertType`, `AlertSeverity`, score, and message
- **`AlertSeverity`** — INFO, WARNING, ALERT, CRITICAL
- **`AlertType`** — Enumeration of all v8.0 + v9.0 alert classifications

---

## Quick start

```python
from fracttalix_sentinel_v900 import SentinelDetector

det = SentinelDetector()
for value in your_time_series:
    result = det.update_and_check(value)
    if result["alert"]:
        print(result["step"], result["alert_reasons"])
```

v9.0 three-channel features are enabled by default. All new config fields have sensible defaults and can be toggled:

```python
from fracttalix_sentinel_v900 import SentinelDetector, SentinelConfig

# Disable specific v9.0 channels
cfg = SentinelConfig(
    enable_frequency_decomposition=False,
    enable_coupling_detection=False,
    enable_channel_coherence=False,
    enable_cascade_detection=False,
    enable_sequence_logging=False,
)
det = SentinelDetector(cfg)
```

---

## Configuration presets

```python
SentinelConfig.fast()          # alpha=0.3, warmup=10 — react instantly
SentinelConfig.production()    # balanced defaults
SentinelConfig.sensitive()     # alpha=0.05, multiplier=2.5, warmup=50
SentinelConfig.realtime()      # quantile-adaptive thresholds
```

---

## v9.0 configuration parameters

| Parameter | Default | Description |
|---|---|---|
| `enable_frequency_decomposition` | `True` | Enable 5-band FFT carrier wave decomposition |
| `min_window_for_fft` | `32` | Minimum observations before FFT decomposition runs |
| `enable_coupling_detection` | `True` | Enable cross-frequency phase-amplitude coupling |
| `coupling_degradation_threshold` | `0.3` | Composite coupling score below this triggers alert |
| `coupling_trend_window` | `10` | Snapshots used for coupling trend measurement |
| `enable_channel_coherence` | `True` | Enable structural-rhythmic coherence monitoring |
| `coherence_threshold` | `0.4` | Coherence score below this triggers decoupling alert |
| `coherence_window` | `20` | Rolling window for coherence computation |
| `enable_cascade_detection` | `True` | Enable CASCADE_PRECURSOR detection |
| `cascade_ews_threshold` | `2` | Minimum elevated EWS indicators for cascade alert |
| `enable_sequence_logging` | `True` | Enable temporal degradation sequence logging |
| `sequence_retention` | `1000` | Maximum completed degradation sequences retained |

---

## Full pipeline (26 steps)

### v8.0 foundation (steps 1-19)

1. **CoreEWMAStep** — Exponentially weighted moving average baseline
2. **CUSUMStep** — Cumulative sum control chart for mean shifts
3. **RegimeStep** — Regime change detection with soft multiplicative boost
4. **VarCUSUMStep** — Variance change detection via CUSUM
5. **PageHinkleyStep** — Gradual drift detection
6. **STIStep** — Short-term instability
7. **TPSStep** — Trajectory phase space anomaly detection
8. **OscDampStep** — Oscillation damping / amplitude spike detection
9. **CPDStep** — Change point detection via mean comparison
10. **RPIStep** — Rhythmic Power Index (spectral dominance)
11. **RFIStep** — Rhythmic Fluctuation Index (Hurst exponent-based fractal irregularity)
12. **SSIStep** — Synchrony Stability Index (Kuramoto order parameter)
13. **PEStep** — Permutation Entropy (complexity/chaos measure)
14. **EWSStep** — Early Warning Signals (critical slowing down)
15. **AQBStep** — Adaptive Quantile-Based detection (optional mode)
16. **SeasonalStep** — Seasonal anomaly detection (auto-period)
17. **MahalStep** — Mahalanobis distance for multivariate outliers
18. **RRSStep** — Relative Rhythmic Strength (harmonic power)
19. **AlertReasonsStep** — Aggregates alert reasons into human-readable descriptions

### v9.0 three-channel extension (steps 20-26)

20. **StructuralSnapshotStep** — Channel 1 structural properties
21. **FrequencyDecompositionStep** — Channel 2 five-band FFT decomposition
22. **BandAnomalyStep** — Per-band anomaly detection
23. **CrossFrequencyCouplingStep** — Phase-amplitude coupling between bands
24. **ChannelCoherenceStep** — Structural-rhythmic coherence
25. **CascadePrecursorStep** — Tipping cascade detection
26. **DegradationSequenceStep** — Channel 3 temporal degradation logging

---

## CLI

```bash
# Process a CSV file
python3 fracttalix_sentinel_v900.py --file data.csv --alpha 0.1 --multiplier 3.0

# Start HTTP server
python3 fracttalix_sentinel_v900.py --serve --host 0.0.0.0 --port 8765

# Run benchmark suite
python3 fracttalix_sentinel_v900.py --benchmark

# Show version
python3 fracttalix_sentinel_v900.py --version
```

---

## HTTP REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/update/<stream_id>` | Update stream with `{"value": ...}` |
| `GET` | `/streams` | List active streams |
| `GET` | `/status/<stream_id>` | Stream status and statistics |
| `DELETE` | `/stream/<stream_id>` | Remove stream |
| `POST` | `/reset/<stream_id>` | Reset stream state |
| `GET` | `/health` | Health check (version, uptime, stream count) |

---

## Dependencies

**Required:** None. Core detector uses only the Python 3 standard library.

**Optional (graceful degradation):**

| Package | Used for |
|---|---|
| NumPy | FFT spectral analysis, Mahalanobis distance, frequency decomposition |
| Numba | JIT compilation for performance (falls back to pure Python) |
| Matplotlib | Visualization dashboard |
| tqdm | Progress bars for batch processing |

---

## v8.0 root-cause fixes (preserved)

| ID | Fix |
|---|---|
| **alpha** | `SentinelConfig` frozen dataclass — immutable, picklable, inspectable |
| **beta** | `WindowBank` — named independent deques; each consumer owns its slot |
| **gamma** | Pipeline decomposition — `DetectorStep` subclasses (19 in v8.0, 26 in v9.0) |
| **delta** | Soft regime boost (T0-02) — replaces hard alpha reset with multiplicative boost |
| **epsilon** | SSI replaces RSI naming (T0-05); `rsi` alias preserved for backward compatibility |

---

## Version history

| Version | Date | Description |
|---|---|---|
| **v9.0** | 2026-03-03 | Three-channel extension: structural, rhythmic, temporal information channels |
| **v8.0** | 2026-02-19 | Ground-up rewrite: frozen config, WindowBank, 19-step pipeline |
| v7.6 | 2026-02-09 | Last monolithic version (scaling and consistency fixes) |
| v7.x | 2026-01 | Incremental monolithic enhancements |
| v2.x | 2025-12 | Early experimental versions |

---

## Backward compatibility

- All v7.x kwargs work via `_legacy_kwargs_to_config()`
- `Detector_7_10` is an alias for `SentinelDetector`
- All 19 v8.0 pipeline steps preserved with identical behavior
- v9.0 features are additive and enabled by default — disable individually via config

---

## Channel 2 — AI Layers

| ID | Paper | Status | File |
|---|---|---|---|
| P1 | FRM Paper 1 | PHASE-READY | [ai-layers/P1-ai-layer.json](ai-layers/P1-ai-layer.json) |
| MK-P1 | Meta-Kaizen Paper 1 | PHASE-READY | [ai-layers/MK-P1-ai-layer.json](ai-layers/MK-P1-ai-layer.json) |
| DRP-1 | Dual-Reader Publishing | PHASE-READY | [ai-layers/DRP1-ai-layer.json](ai-layers/DRP1-ai-layer.json) |

---

## Theoretical foundation

Fractal Rhythm Model (FRM) — 11 axioms. Papers 1-6.

Introduced by Thomas Brennan and Grok 4.

---

## Authors

Thomas Brennan, Claude (Anthropic), Grok (xAI)

## License

CC0 — public domain. No rights reserved.

## Repository

https://github.com/thomasbrennan/Fracttalix
