# DetectorSuite — Modular Five-Detector Suite

`DetectorSuite` runs five independent, specialized anomaly detectors in parallel on a single streaming signal. Unlike `SentinelDetector`, each detector can declare `OUT_OF_SCOPE` when the current data does not match its model — so alerts only come from detectors that understand your data.

**No external dependencies.** Works on stdlib alone.

---

## Installation

```bash
pip install fracttalix
```

---

## The Five Detectors

| Detector | What it watches | Null FPR target | `OUT_OF_SCOPE` when |
|----------|----------------|-----------------|---------------------|
| `HopfDetector` | Critical slowing down (EWS: rising variance + AC(1)) | 0% on N(0,1) | Always in scope (EWS works on any signal) |
| `DiscordDetector` | Point and contextual anomalies (subsequence discord) | ≤ 1% on N(0,1) | Signal too short for discord window |
| `DriftDetector` | Slow mean shift (non-adaptive CUSUM + Page-Hinkley) | ≤ 0.5% on N(0,1) | Always in scope |
| `VarianceDetector` | Sudden volatility changes (CUSUM on z²) | ≤ 1% on N(0,1) | Always in scope |
| `CouplingDetector` | Cross-frequency coordination degrading (PAC) | 0% on N(0,1) | Signal has no oscillatory structure; window too small for FFT |

---

## Quick Start

```python
from fracttalix.suite import DetectorSuite

suite = DetectorSuite()

for value in stream:
    result = suite.update(value)
    print(result.summary())

    if result.any_alert:
        for det in result.alerts:
            print(f"  [{det.detector}] {det.status.name}  score={det.score:.2f}  {det.message}")
```

Example output during a variance explosion:

```
Hopf:ok(0.02) | Disc:ok(0.08) | Drif:ok(0.00) | Vari:ALERT(0.91) | Coup:ok(0.03)
  [VarianceDetector] ALERT  score=0.91  cusum_hi=12.3 z_sq=8.4
```

---

## Reading a `SuiteResult`

Every call to `suite.update(value)` returns a `SuiteResult`:

```python
result = suite.update(value)

# ── Individual detector access ────────────────────────────────
result.hopf      # DetectorResult
result.discord   # DetectorResult
result.drift     # DetectorResult
result.variance  # DetectorResult
result.coupling  # DetectorResult

# ── DetectorResult fields ─────────────────────────────────────
det = result.hopf
det.detector   # str   — detector class name
det.status     # ScopeStatus enum: WARMUP | NORMAL | ALERT | OUT_OF_SCOPE
det.score      # float 0.0–1.0 — higher = more anomalous
det.message    # str   — diagnostic detail
det.step       # int   — observation number when this result was produced
det.is_alert   # bool  — True iff status == ALERT
det.in_scope   # bool  — True iff status in (NORMAL, ALERT)

# ── Collection helpers ────────────────────────────────────────
result.any_alert      # bool — at least one detector alerting
result.alerts         # list[DetectorResult] — only alerting detectors
result.in_scope       # list[DetectorResult] — only in-scope detectors
result.out_of_scope   # list[DetectorResult] — detectors that declared OOS
result.summary()      # one-line dashboard string
```

### `ScopeStatus` values

| Value | Meaning |
|-------|---------|
| `WARMUP` | Not enough observations yet to produce a verdict |
| `NORMAL` | In scope, no anomaly detected |
| `ALERT` | Anomaly detected |
| `OUT_OF_SCOPE` | The detector's model does not apply to the current data |

---

## Detector Details

### HopfDetector — Critical Slowing Down (EWS)

Monitors Early Warning Signals of approaching critical transitions. As a system approaches a bifurcation, variance and lag-1 autocorrelation rise before the transition occurs.

```python
from fracttalix.suite import HopfDetector

hopf = HopfDetector(
    method='ews',      # 'ews' (default) or 'frm' (requires scipy; see FRMSuite)
    warmup=50,         # observations before first verdict (default 50)
    window=40,         # rolling EWS window (default 40)
    var_threshold=0.6, # variance EWS threshold (default 0.6)
    ac_threshold=0.6,  # AC(1) EWS threshold (default 0.6)
)
```

`HopfDetector(method='ews')` is always in scope (no `OUT_OF_SCOPE`). It fires `ALERT` when variance or lag-1 autocorrelation exceeds the EWS threshold — a signal that the system is approaching a tipping point.

> For the FRM physics layer (`method='frm'`), see [FRMSuite](frm_suite.md). The `frm` method requires scipy and fits damping λ directly.

### DiscordDetector — Point and Contextual Anomalies

Detects subsequences that are anomalous relative to their recent context using a discord distance approach (z-normalized Euclidean distance to nearest non-self-match neighbor).

```python
from fracttalix.suite import DiscordDetector

discord = DiscordDetector(
    window=10,           # subsequence length (default 10)
    warmup=50,           # observations before first verdict
    threshold=3.0,       # z-normalized distance threshold (default 3.0)
)
```

Reports `OUT_OF_SCOPE` until there are enough observations to build a comparison pool.

### DriftDetector — Slow Mean Shift

Detects gradual, sustained mean shifts that are too slow to trigger a z-score threshold. Uses non-adaptive CUSUM (the mean is frozen at warmup) plus a Page-Hinkley test.

```python
from fracttalix.suite import DriftDetector

drift = DriftDetector(
    warmup=50,
    cusum_k=0.5,   # allowance (default 0.5σ)
    cusum_h=8.0,   # decision threshold (default 8.0)
    ph_delta=0.01, # Page-Hinkley sensitivity (default 0.01)
    ph_lambda=50,  # Page-Hinkley threshold (default 50)
)
```

**Key design point:** The CUSUM reference mean is frozen after warmup. This means `DriftDetector` correctly fires during ongoing drift that an adaptive detector would track away and miss.

### VarianceDetector — Sudden Volatility Change

Fires when variance explodes or collapses suddenly. Uses CUSUM on the squared z-score (z²). Under the null (N(0,1)), E[z²] = 1.0; a sustained increase above this triggers the alert.

```python
from fracttalix.suite import VarianceDetector

variance = VarianceDetector(
    warmup=50,
    cusum_k=1.0,   # allowance (calibrated to E[z²]=1.0 under null; default 1.0)
    cusum_h=10.0,  # decision threshold (default 10.0)
)
```

### CouplingDetector — Cross-Frequency Coordination

Monitors Phase-Amplitude Coupling (PAC) across FFT frequency bands. When cross-scale coordination degrades — bands decouple — the detector fires. Useful for oscillatory systems (EEG, HRV, power grids, mechanical vibration).

```python
from fracttalix.suite import CouplingDetector

coupling = CouplingDetector(
    warmup=80,                # needs enough history for FFT (default 80)
    window=64,                # FFT window (default 64)
    threshold=0.24,           # coupling score below this → ALERT (default 0.24)
)
```

Reports `OUT_OF_SCOPE` when the signal has no oscillatory structure or the window is too small for FFT.

---

## Using Individual Detectors

All five detectors are independently usable without `DetectorSuite`:

```python
from fracttalix.suite import HopfDetector, DriftDetector, VarianceDetector

hopf = HopfDetector()
drift = DriftDetector()
variance = VarianceDetector()

for value in stream:
    h = hopf.update(value)
    d = drift.update(value)
    v = variance.update(value)

    if h.is_alert:
        print(f"Early warning: {h.message}")
    if d.is_alert:
        print(f"Drift detected: {d.message}")
    if v.is_alert:
        print(f"Variance change: {v.message}")
```

---

## Recommended Domain Configurations

```python
# Power grid / electrical machine monitoring
# HopfDetector catches pre-transition slowing; VarianceDetector catches fault spikes
suite = DetectorSuite()
power_grid = suite  # use result.hopf + result.variance

# API / microservice latency monitoring
# DiscordDetector catches unusual latency spikes; DriftDetector catches regression
suite = DetectorSuite()
api_mon = suite  # use result.discord + result.drift

# Neural / physiological signals (EEG, HRV, muscle EMG)
# HopfDetector tracks pre-seizure / pre-episode slowing; CouplingDetector tracks neural sync
suite = DetectorSuite()
eeg_mon = suite  # use result.hopf + result.coupling

# IoT sensor — unknown anomaly types
# Use all five; let OUT_OF_SCOPE filter irrelevant detectors automatically
suite = DetectorSuite()
# inspect result.in_scope to see which detectors are active on your data
```

---

## State Persistence

```python
import json

# Save
sd = suite.state_dict()
json_str = json.dumps(sd)

# Restore (same config)
suite2 = DetectorSuite()
suite2.load_state(json.loads(json_str))

# Reset to factory state
suite.reset()
```

---

## Comparison with SentinelDetector

| | `SentinelDetector` | `DetectorSuite` |
|-|--------------------|-----------------|
| Architecture | 37-step sequential pipeline | 5 parallel independent detectors |
| Output | Single alert/no-alert + anomaly score | Five independent verdicts + scope status |
| `OUT_OF_SCOPE` | Never — always produces a verdict | Yes — honest about inapplicable models |
| False positive rate | ~2.6% on null (v12.3 with ConsensusGate) | Near 0% on null for most detectors |
| Best for | Unknown signal; broadest coverage | Known signal type; low-FPR domain monitoring |
| FRM physics layer | No | Available via `FRMSuite` wrapper |

> If you know what kind of anomaly you are looking for, `DetectorSuite` is almost always the better choice. If you don't know, or want a single combined score, use `SentinelDetector`.

---

## Next Steps

- [FRMSuite](frm_suite.md) — add FRM physics detectors (Lambda, Omega, Virtu) for oscillatory signals with known `tau_gen`
- [API Reference — Suite](api/suite.md) — full class and method documentation
- [SentinelDetector](../README.md#quick-start) — 37-step monolith for unknown/general signals
