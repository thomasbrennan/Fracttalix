# Fracttalix Sentinel v12.0

https://doi.org/10.5281/zenodo.18859299

**Three-channel streaming anomaly detector grounded in the Fractal Rhythm Model**
Single-file Python | Zero required dependencies | CC0 public domain

---

## Three-channel model

**Channel 1 — Structural.** Network topology as active transmitter. Monitors mean, variance, skewness, kurtosis, autocorrelation, and stationarity of the input stream as an independent information channel.

**Channel 2 — Rhythmic.** Broadband multiplexed oscillatory transmission. Decomposes the composite rhythmicity signal into five independent frequency band carrier waves (ultra-low through ultra-high) and monitors cross-frequency coupling between adjacent bands as a higher-order information channel.

**Channel 3 — Temporal.** One-way irreversible carrier wave. Logs the temporal sequence of channel degradation events as diagnostic information about regime change type and severity. The ordering of degradation is itself informative — coupling degrades before individual channels fail.

---

## Alert classes

| Alert type | Severity | Description |
|---|---|---|
| `BAND_ANOMALY` | ALERT | Per-carrier-wave anomaly invisible to composite detection |
| `COUPLING_DEGRADATION` | WARNING | Cross-frequency coupling breakdown (earlier warning signal) |
| `STRUCTURAL_RHYTHMIC_DECOUPLING` | ALERT | Channel 1-2 coherence loss |
| `CASCADE_PRECURSOR` | CRITICAL | Tipping cascade precursor — scale-level reversion risk |

---

## Quick start

```python
from fracttalix_sentinel_v1200 import SentinelDetector

det = SentinelDetector()
for value in your_time_series:
    result = det.update_and_check(value)
    if result["alert"]:
        print(result["step"], result["alert_reasons"])
```

Three-channel features are enabled by default. All config fields have sensible defaults and can be toggled:

```python
from fracttalix_sentinel_v1200 import SentinelDetector, SentinelConfig

# Disable specific channels
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

## Three-channel configuration parameters

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

### v12.0 three-channel extension (steps 20-26)

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
python3 fracttalix_sentinel_v1200.py --file data.csv --alpha 0.1 --multiplier 3.0

# Start HTTP server
python3 fracttalix_sentinel_v1200.py --serve --host 0.0.0.0 --port 8765

# Run benchmark suite
python3 fracttalix_sentinel_v1200.py --benchmark

# Show version
python3 fracttalix_sentinel_v1200.py --version
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

## Performance

**Time complexity per observation:** O(W) where W is the largest window size (default 64). All 26 pipeline steps are O(W) or better. No step requires full history traversal — the WindowBank caps memory and compute.

**Memory:** O(W × S) where S is the number of registered window slots (~30). State is bounded regardless of stream length. For default config, steady-state memory is under 100 KB per detector instance.

**Throughput (pure Python, no NumPy):** ~2,000–5,000 observations/second on modern hardware. With NumPy: ~5,000–15,000 obs/sec depending on pipeline features enabled. Numba JIT (if available) accelerates inner loops further.

**Scaling:** MultiStreamSentinel scales linearly with stream count. Each stream is independent — no cross-stream overhead.

---

## Known limitations

- **Float overflow:** Values near ±1e308 can cause OverflowError in exponential calculations. Keep input values within ±1e100 for safe operation, or normalize inputs.
- **Zero-variance streams:** Constant-value streams may trigger spurious alerts from steps that compute variance-based statistics (division by near-zero variance). This is by design — zero variance is itself anomalous in most real-world contexts.
- **FFT resolution:** Frequency decomposition requires `min_window_for_fft` observations (default 32). Short windows limit frequency resolution in the lower bands.
- **Warmup period:** No alerts are generated during the warmup phase. The detector needs `warmup_periods` observations (default 30) before producing meaningful results.
- **Pure Python FFT:** Without NumPy, frequency decomposition uses a pure Python DFT which is O(N²) rather than O(N log N). For large FFT windows, install NumPy.

---

## Version history

| Version | Date | Description |
|---|---|---|
| **v12.0** | 2026-03-10 | Current release |
| **v11.0** | 2026-03-08 | Incremental pipeline and channel refinements |
| **v10.0** | 2026-03-06 | Incremental pipeline and metric improvements |
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
- Three-channel features are additive and enabled by default — disable individually via config

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
