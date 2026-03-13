# Fracttalix v13.0 — FRMSuite

**The first streaming anomaly detection library directly derived from the Fractal Rhythm Model.**

FRMSuite ingests one scalar observation at a time and returns a physics-grounded bifurcation signal — including a time-to-bifurcation estimate — on every call. No batching, no retraining, no warmup gap once past the configurable warmup window.

> **Theoretical foundation:** Fractal Rhythm Model Papers 1–6
> DOI: [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)
> License: **CC0** — public domain

---

[![PyPI](https://img.shields.io/pypi/v/fracttalix)](https://pypi.org/project/fracttalix/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18859299.svg)](https://doi.org/10.5281/zenodo.18859299)
[![Tests](https://github.com/thomasbrennan/Fracttalix/actions/workflows/tests.yml/badge.svg)](https://github.com/thomasbrennan/Fracttalix/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

> **Keywords:** anomaly detection · streaming time-series · online learning · Hopf bifurcation · critical slowing down · phase-amplitude coupling · Fractal Rhythm Model · time-to-bifurcation · change-point detection · multivariate monitoring

## Table of Contents

1. [Overview](#overview)
2. [Which API Should I Use?](#which-api-should-i-use)
3. [v13.0 Changes — FRMSuite as Primary API](#v130-changes)
4. [Installation](#installation)
5. [FRMSuite — Fractal Rhythm Model Physics Layer](#frmsuite--fractal-rhythm-model-physics-layer)
6. [DetectorSuite — Modular Five-Detector Suite](#detectorsuite--modular-five-detector-suite)
7. [Theoretical Foundation](#theoretical-foundation)
8. [Authors & License](#authors--license)
9. [Legacy API — SentinelDetector (Retired)](#legacy-api--sentineldetector-retired)
   - [SentinelConfig — Configuration](#sentinelconfig--configuration)
   - [Pipeline Architecture — 37 Steps](#pipeline-architecture--37-steps-v122)
   - [Alert Types and Data Structures](#v90-alert-types-and-data-structures)
   - [SentinelResult API](#sentinelresult-api)
   - [MultiStreamSentinel](#multistreamssentinel)
   - [SentinelBenchmark](#sentinelbenchmark)
   - [SentinelServer — REST API](#sentinelserver--rest-api)
   - [CLI Reference](#cli-reference)
   - [Backward Compatibility](#backward-compatibility)

---

## Overview

Fracttalix is a Python package (`pip install fracttalix`) for real-time streaming anomaly detection grounded in the **Fractal Rhythm Model (FRM)**. v13.0 is the first release where the primary detection engine is directly derived from FRM physics, not just inspired by it.

The FRM gives every detector a physically-grounded question to answer:

| Detector | FRM Question | What it measures |
|----------|-------------|-----------------|
| **Lambda** (`HopfDetector frm`) | Is λ → 0? | Damping decay rate — the Hopf bifurcation approach |
| **Omega** (`OmegaDetector`) | Is ω = π/(2·τ_gen) still intact? | Frequency integrity — FRM quarter-wave theorem |
| **Virtu** (`VirtuDetector`) | Δt ≈ λ / \|dλ/dt\|? | Time-to-bifurcation estimate |

No generic anomaly detector can answer questions 2 and 3. That requires knowing τ_gen — the FRM generation delay — and having a physical model that predicts what the signal should look like.

**Design priorities:**

- **FRM physics in strong mode** — when `tau_gen` is supplied, Lambda and Omega test specific, falsifiable predictions from the Fractal Rhythm Model
- **Zero hard dependencies** — Layer 1 (DetectorSuite) runs on stdlib; Layer 2 (FRM physics) requires numpy + scipy
- **OUT_OF_SCOPE discipline** — every detector reports `OUT_OF_SCOPE` when its model doesn't fit the input; no spurious verdicts on mismatched signals
- **Modular, independently usable** — each detector works standalone; failure of one does not degrade others

---

## Which API Should I Use?

| API | Best for | Requires | `pip install` |
|-----|----------|----------|---------------|
| **`FRMSuite`** | Oscillatory signals with known FRM generation delay; need time-to-bifurcation and FRM model validation | numpy + scipy | `pip install fracttalix[fast]` |
| **`DetectorSuite`** | Domain-specific monitoring; need to know *which* anomaly type fired; low FPR; no FRM model needed | stdlib only | `pip install fracttalix` |
| ~~`SentinelDetector`~~ | **Retired as of v13.0.** Retained for backward compatibility only. See [Legacy API](#legacy-api--sentineldetector-retired). | stdlib only | `pip install fracttalix` |

**Decision rule:**

- You have an FRM-shaped signal and know `tau_gen` → **FRMSuite** (time-to-bifurcation, model validation)
- You know which anomaly type matters (drift? variance? discord?) → **DetectorSuite**
- You need the old `SentinelDetector` behaviour → use it; it still works, but is no longer developed

---

## v13.0 Changes

### FRMSuite as Primary API — SentinelDetector Retired

v13.0 promotes `FRMSuite` as the primary detection API and formally retires `SentinelDetector`.

#### Why

`SentinelDetector` was a 37-step monolith grounded in generic statistics. It could not answer *why* a signal was changing or *when* a transition would occur. Its FPR floor was driven by channels that applied regardless of signal structure.

`FRMSuite` answers the questions `SentinelDetector` never could:
- **Why is the signal changing?** — λ is declining (damping collapsing)
- **Is the model still valid?** — ω still matches π/(2·τ_gen)?
- **When will the transition occur?** — Δt ≈ λ / |dλ/dt|

#### Benchmark: FRMSuite vs SentinelDetector (N=500, seed=42)

**Null signals (FPR — lower is better)**

| Signal | FRMSuite | SentinelDetector | Verdict |
|--------|----------|-----------------|---------|
| White noise | 1.0% | 3.8% | FRMSuite wins |
| Sustained sinusoid | 3.4% | 4.4% | FRMSuite wins |
| Random walk | 83.0% | 93.6% | FRMSuite wins |
| Slow trend | 15.0% | 23.2% | FRMSuite wins |

**Signal cases (detection rate — higher is better)**

| Signal | FRMSuite | SentinelDetector | Verdict |
|--------|----------|-----------------|---------|
| Hopf approach | 77% | 55% | FRMSuite wins |
| Mean shift | 70% | 66% | FRMSuite wins |
| Variance explosion | 100% | 100% | Tie |
| Omega drift | 99% | 100% | FRMSuite wins |
| Coupling collapse | 35% | 2% | FRMSuite wins |
| Discord anomaly | 100% | 100% | Tie |

FRMSuite wins or ties on every signal class, with lower FPR on all four null signals.

#### Capabilities FRMSuite provides that SentinelDetector cannot

1. **Time-to-bifurcation (Virtu)** — `SentinelDetector` has no TTB capability
2. **ω integrity check (Omega)** — confirms observed frequency matches FRM prediction π/(2·τ_gen)
3. **`frm_confidence` (0–3)** — compound signal from 3 independent FRM-physics detectors
4. **OUT_OF_SCOPE scope reporting** — detectors stay silent when their model doesn't apply
5. **Modular architecture** — each detector usable independently

#### Migration

```python
# Before (v12.x) — SentinelDetector
from fracttalix import SentinelDetector, SentinelConfig
det = SentinelDetector(SentinelConfig.production())
result = det.update_and_check(value)
if result["alert"]:
    print(result["alert_reasons"])

# After (v13.0) — FRMSuite (tau_gen from domain knowledge)
from fracttalix.frm import FRMSuite
suite = FRMSuite(tau_gen=12.5)
result = suite.update(value)
if result.frm_confidence >= 2:
    print(f"FRM bifurcation: confidence={result.frm_confidence}")
    print(result.virtu.message)  # ttb=38.4 steps

# After (v13.0) — DetectorSuite (no tau_gen required)
from fracttalix.suite import DetectorSuite
suite = DetectorSuite()
result = suite.update(value)
if result.any_alert:
    for det in result.alerts:
        print(f"{det.detector}: {det.message}")
```

`SentinelDetector` is still importable and fully functional. No existing code breaks.

---

## Installation

```bash
pip install fracttalix
```

**For FRMSuite (Layer 2 FRM physics — numpy + scipy required):**

```bash
pip install fracttalix[fast]
```

**Optional extras:**

```bash
pip install numpy          # FFT, PAC computation, Hilbert transform
pip install scipy          # curve_fit for Lambda (HopfDetector frm)
pip install numba          # JIT compilation for hot loops
pip install matplotlib     # plot_history() dashboard
pip install tqdm           # progress bars in benchmark
```

**Run tests (444 tests, all expected to pass):**

```bash
pytest
```

---

## FRMSuite — Fractal Rhythm Model Physics Layer

`FRMSuite` is the full two-layer suite: `DetectorSuite` (Layer 1, generic, no scipy) plus three Fractal Rhythm Model physics detectors (Layer 2, scipy required).

```bash
pip install fracttalix[fast]   # includes numpy + scipy for Layer 2
```

### Architecture

```
FRMSuite(tau_gen=τ)
├── Layer 1 — DetectorSuite (5 generic detectors, stdlib + numpy)
│   ├── HopfDetector(ews)    — critical slowing down
│   ├── DiscordDetector      — point / contextual anomalies
│   ├── DriftDetector        — slow mean shift
│   ├── VarianceDetector     — volatility change
│   └── CouplingDetector     — PAC cross-scale decoupling (independent cross-validator)
│
└── Layer 2 — FRM Physics (scipy required; graceful degradation if absent)
    ├── Lambda  (HopfDetector frm) — tracks damping λ → 0
    ├── Omega   (OmegaDetector)    — tracks ω vs π/(2·τ_gen)
    └── Virtu   (VirtuDetector)    — time-to-bifurcation estimate: Δt ≈ λ / |dλ/dt|
```

**Graceful degradation:** If scipy is absent, Layer 2 detectors return `OUT_OF_SCOPE` and `frm_confidence` stays at 0. Layer 1 operates normally.

### Key Concept: `tau_gen`

`tau_gen` is the FRM generation delay — the characteristic delay of your system. The Fractal Rhythm Model predicts:

```
ω = π / (2 · τ_gen)        (quarter-wave theorem)
λ → 0  at Hopf bifurcation  (damping collapse)
Δt ≈ λ / |dλ/dt|           (time-to-bifurcation)
```

When `tau_gen` is supplied:

- **Lambda** (strong mode): uses the fixed predicted frequency ω = π/(2·`tau_gen`) as its reference; curve-fits damping λ against this
- **Omega** (strong mode): checks that the observed dominant frequency matches ω = π/(2·`tau_gen`); a 5% deviation fires ALERT
- **frm_confidence** counts strong-mode Layer 2 detectors in ALERT (0–3)

When `tau_gen=None`: Lambda and Omega run in weak mode (generic frequency tracking). `frm_confidence` stays 0 — the FRM physics test cannot run without the model parameter.

> Get `tau_gen` from domain knowledge. For power grids, it is the generator inertia constant. For EEG alpha rhythms, it is the reciprocal of the cycle frequency. For mechanical oscillators, it is the half-period.

### `frm_confidence` Score

| `frm_confidence` | Meaning |
|-----------------|---------|
| 0 | No strong-mode Layer 2 detector alerting |
| 1 | Lambda alerting (λ declining) |
| 2 | Lambda + Omega alerting (λ declining AND ω drifting from FRM prediction) |
| 3 | Lambda + Omega + Virtu all alerting (full FRM bifurcation signal) |
| `frm_confidence_plus` | `frm_confidence` + 1 if CouplingDetector (Layer 1) also alerting — independent PAC cross-validation |

### Quick Start

```python
from fracttalix.frm import FRMSuite

# tau_gen=12.5: you know your system's generation delay
suite = FRMSuite(tau_gen=12.5)

for value in stream:
    result = suite.update(value)
    print(result.summary())

    if result.frm_confidence >= 2:
        # Lambda + Omega both alerting: strong compound bifurcation signal
        print(f"FRM bifurcation signal: confidence={result.frm_confidence}")

    if result.frm_confidence_plus >= 3:
        # FRM physics + independent PAC structural cross-validation
        print("Cross-validated bifurcation: highest confidence")
        print(result.virtu.message)   # "ttb=38.4 confidence=HIGH safety_factor=1.00 ..."
```

### Without tau_gen

```python
# Weak mode: frequency instability tracking, no FRM physics test
suite = FRMSuite()  # tau_gen=None

for value in stream:
    result = suite.update(value)
    # result.frm_confidence will always be 0 in weak mode
    # Layer 1 detectors still run normally
    if result.layer1.drift.is_alert:
        print("Drift detected")
```

### Reading an FRMSuiteResult

```python
result = suite.update(value)

# Layer 1 (all DetectorSuite fields)
result.layer1.hopf.is_alert
result.layer1.any_alert
result.layer1.summary()

# Layer 2 — FRM physics
result.lambda_.is_alert        # λ declining?
result.lambda_.score           # 0.0–1.0 urgency
result.lambda_.message         # "frm λ=0.18 rate=-0.004 ttb=45.0 ..."

result.omega.is_alert          # ω drifting from FRM prediction?
result.omega.message           # "omega_obs=0.251 omega_pred=0.251 deviation=0.021 ..."

result.virtu.is_alert          # time-to-bifurcation urgency score ≥ threshold?
result.virtu.message           # "ttb=38.4 confidence=HIGH safety_factor=1.00 ..."
result.virtu.score             # urgency: 0 = distant, 1 = imminent

# Compound scores
result.frm_confidence          # int 0–3
result.frm_confidence_plus     # int 0–4 (adds CouplingDetector cross-validation)
result.layer2_available        # bool — False if scipy absent

# Convenience
result.any_alert               # bool — any detector in Layer 1 or Layer 2 firing
result.alerts                  # list[DetectorResult] — all currently alerting
result.summary()               # multi-line dashboard string
```

### Safety Factor (Conservative Estimates)

`VirtuDetector` supports a `safety_factor` for asymmetric-cost applications where acting too late is worse than acting too early:

```python
# Reported ttb = raw_ttb / safety_factor
# safety_factor=2.0 → half the raw estimate → earlier warning
suite = FRMSuite(
    tau_gen=12.5,
    virtu_kwargs={"safety_factor": 2.0}
)
```

### State Persistence

```python
import json
sd = suite.state_dict()
json_str = json.dumps(sd)

suite2 = FRMSuite(tau_gen=12.5)
suite2.load_state(json.loads(json_str))
```

---

## DetectorSuite — Modular Five-Detector Suite

`DetectorSuite` runs five independent specialized detectors in parallel. Unlike `SentinelDetector`, each detector reports `OUT_OF_SCOPE` when the current data does not match its model — so you only get alerts from detectors that understand your signal. No scipy required.

```bash
pip install fracttalix          # stdlib only — no extra deps needed
```

### The Five Detectors

| Detector | Question answered | Null FPR target |
|----------|------------------|-----------------|
| `HopfDetector` | Is critical slowing down occurring? (EWS) | 0% on N(0,1) |
| `DiscordDetector` | Is this point anomalous vs. its context? | ≤ 1% on N(0,1) |
| `DriftDetector` | Has the mean shifted slowly over time? | ≤ 0.5% on N(0,1) |
| `VarianceDetector` | Has volatility suddenly changed? | ≤ 1% on N(0,1) |
| `CouplingDetector` | Is cross-scale coordination degrading? | 0% on N(0,1) |

### Quick Start

```python
from fracttalix.suite import DetectorSuite

suite = DetectorSuite()

for value in stream:
    result = suite.update(value)
    print(result.summary())
    # e.g. "Hopf:ok(0.12) | Disc:ok(0.04) | Drif:ok(0.00) | Vari:ok(0.01) | Coup:ok(0.00)"

    if result.any_alert:
        for det in result.alerts:
            print(f"{det.detector}: {det.message}")
```

### Reading a SuiteResult

```python
result = suite.update(value)

# Individual detector access
result.hopf.is_alert        # bool
result.drift.status         # ScopeStatus.NORMAL / ALERT / OUT_OF_SCOPE / WARMUP
result.variance.score       # float 0.0–1.0
result.discord.message      # diagnostic string

# Collection helpers
result.any_alert            # bool — at least one detector firing
result.alerts               # list[DetectorResult] — only firing detectors
result.in_scope             # list[DetectorResult] — detectors whose model applies
result.out_of_scope         # list[DetectorResult] — detectors that declared OOS
result.summary()            # one-line dashboard string
```

### Recommended Combinations

```python
# Power grid / AC motor monitoring
suite = DetectorSuite(hopf_kwargs={"method": "ews"})
# → HopfDetector catches pre-transition slowing; VarianceDetector catches fault spikes

# API / service latency monitoring
suite = DetectorSuite()
# → DiscordDetector catches unusual request latencies; DriftDetector catches gradual regression

# Neural / physiological signals (EEG, HRV)
suite = DetectorSuite()
# → HopfDetector + CouplingDetector together = compound oscillation-degradation signal

# Use individual detectors independently
from fracttalix.suite import HopfDetector, DriftDetector

hopf = HopfDetector()
drift = DriftDetector()
for v in stream:
    h = hopf.update(v)
    d = drift.update(v)
```

### State Persistence

```python
import json
sd = suite.state_dict()
json_str = json.dumps(sd)

# Restore
suite2 = DetectorSuite()
suite2.load_state(json.loads(json_str))

# Reset to factory state
suite.reset()
```

---

## Theoretical Foundation

| FRM Component | Implementation |
|---------------|---------------|
| Quarter-wave theorem: ω = π/(2·τ_gen) | `OmegaDetector` — strong mode frequency integrity check |
| Damping collapse: λ → 0 at Hopf bifurcation | `HopfDetector(method='frm')` — Lambda |
| Time-to-bifurcation: Δt ≈ λ / \|dλ/dt\| | `VirtuDetector` |
| Cross-scale PAC coordination | `CouplingDetector` — independent cross-validator for Omega |
| Critical slowing down (FRM Concept 9) | `HopfDetector(method='ews')` — variance + lag-1 AC |
| Ordinal pattern complexity (FRM Concept 3) | `PEStep` in SentinelDetector (legacy) |
| Three-channel model (Paper 6) | `StructuralSnapshotStep`, `FrequencyDecompositionStep`, etc. (legacy) |

**DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)

> FRM components refer to concepts in the Fractal Rhythm Model working papers. Layer 2 detectors (Lambda, Omega, Virtu) are directly derived from FRM physics with `tau_gen` supplied. Layer 1 detectors and the retired SentinelDetector use signal-processing heuristics *inspired by* the FRM framework.

---

## Authors & License

**Authors:** Thomas Brennan & Claude (Anthropic) & Grok (xAI)

**License:** [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/) — no restrictions, no attribution required.

**GitHub:** [https://github.com/thomasbrennan/Fracttalix](https://github.com/thomasbrennan/Fracttalix)

**Version history:**

| Version | Notes |
|---------|-------|
| v13.0.0 | FRMSuite promoted as primary API; SentinelDetector retired; first release directly derived from Fractal Rhythm Model physics |
| v12.3.0 | FPR floor eliminated via ConsensusGate + threshold recalibration; SeasonalPreprocessStep |
| v12.2.0 | Epistemic language corrections; production() multiplier 3.0→4.5 |
| v12.1.0 | VarCUSUM non-reset fix; ChannelCoherence unit mismatch fix |
| v10.0.0 | 4 collapse indicators (v10 API), 37 steps, 98 tests |
| v9.0.0 | Three-channel model, 26 steps, 65 tests |
| v8.0.0 | Frozen config, WindowBank, 19-step pipeline |
| v7.11–v7.6 | Earlier releases |

---

## Channel 2 — AI Layers

Machine-readable falsification layers for the Fracttalix corpus. All layers conform to `ai-layers/ai-layer-schema.json` (v2-S42, Dual Reader Standard).

| ID    | Paper                        | Status      | File                             |
|-------|------------------------------|-------------|----------------------------------|
| P1    | Fractal Rhythm Model (Paper 1) | PHASE-READY | ai-layers/P1-ai-layer.json       |
| MK-P1 | Meta-Kaizen Paper 1          | PHASE-READY | ai-layers/MK-P1-ai-layer.json    |
| DRP-1 | Dependency Resolution Process | PHASE-READY | ai-layers/DRP1-ai-layer.json     |
| SFW-1 | Sentinel v12.2               | PHASE-READY | ai-layers/SFW1-ai-layer.json     |

---

---

# Legacy API — SentinelDetector (Retired)

> **SentinelDetector was retired in v13.0.** It is retained for full backward compatibility — all existing code continues to work unchanged. No new features will be added. For new integrations, use [`FRMSuite`](#frmsuite--fractal-rhythm-model-physics-layer) or [`DetectorSuite`](#detectorsuite--modular-five-detector-suite).

The retirement decision is documented in [`RETIREMENT-DECISION.md`](RETIREMENT-DECISION.md).

---

## SentinelConfig — Configuration

`SentinelConfig` is a frozen dataclass (`slots=True`). All fields are immutable after construction. Use `dataclasses.replace(cfg, field=value)` to derive a new config.

### Factory presets

| Preset | `alpha` | `multiplier` | `warmup` | Normal FPR¹ | Notes |
|--------|---------|-------------|----------|-------------|-------|
| `SentinelConfig.fast()` | 0.3 | 3.0 | 10 | ~60–80% | Fastest response; very high FP rate — use only with downstream filtering |
| `SentinelConfig.production()` | 0.1 | **4.5** | 30 | ~35% | Balanced defaults; v12.2 default |
| `SentinelConfig.sensitive()` | 0.05 | 2.5 | 50 | ~40–50% | Catches subtle anomalies; high FP rate |
| `SentinelConfig.realtime()` | 0.2 | 3.0 | 15 | ~30–40% | Quantile-adaptive thresholds |

> ¹ Approximate normal alert rate on white noise N(0,1), empirically measured.

### Parameter groups

#### A — Core EWMA

| Field | Default | Description |
|-------|---------|-------------|
| `alpha` | `0.1` | EWMA smoothing factor (0 < α ≤ 1) |
| `dev_alpha` | `0.1` | EWMA factor for deviation estimation |
| `multiplier` | `3.0` | Alert threshold = EWMA ± multiplier × dev_ewma |
| `warmup_periods` | `30` | Observations before alerts are issued |

#### B — Regime Detection

| Field | Default | Description |
|-------|---------|-------------|
| `regime_threshold` | `3.5` | Z-score magnitude triggering regime change |
| `regime_alpha_boost` | `2.0` | Multiplicative alpha boost during transitions |
| `regime_boost_decay` | `0.9` | Decay rate of regime boost per observation |

#### C — Multivariate

| Field | Default | Description |
|-------|---------|-------------|
| `multivariate` | `False` | Enable Mahalanobis distance mode |
| `n_channels` | `1` | Number of input channels |
| `cov_alpha` | `0.05` | EWMA factor for covariance (Woodbury rank-1) |

#### D — FRM Metrics

| Field | Default | Description |
|-------|---------|-------------|
| `rpi_window` | `64` | Rhythm Periodicity Index FFT window |
| `rfi_window` | `64` | Rhythm Fractal Index R/S window |
| `rpi_threshold` | `0.6` | Minimum RPI for "rhythm healthy" |
| `rfi_threshold` | `0.4` | RFI alert threshold |

#### E — Complexity & EWS

| Field | Default | Description |
|-------|---------|-------------|
| `pe_order` | `3` | Permutation Entropy embedding dimension |
| `pe_window` | `50` | PE sliding window |
| `pe_threshold` | `0.05` | PE deviation alert threshold |
| `ews_window` | `40` | EWS rolling window (independent from scalar window) |
| `ews_threshold` | `0.6` | EWS "approaching critical" threshold |

#### F — Temporal Signal Heuristics

| Field | Default | Description |
|-------|---------|-------------|
| `sti_window` | `20` | Shear-Turbulence Index window (signal heuristic) |
| `tps_window` | `30` | Temporal Phase Space window |
| `osc_damp_window` | `20` | Oscillation damping window |
| `osc_threshold` | `1.5` | Oscillation damping alert multiplier |
| `cpd_window` | `30` | Change-Point Detection window |
| `cpd_threshold` | `2.0` | CPD alert z-score threshold |

#### G — Drift / Volatility / Seasonal

| Field | Default | Description |
|-------|---------|-------------|
| `ph_delta` | `0.01` | Page-Hinkley incremental sensitivity |
| `ph_lambda` | `50.0` | Page-Hinkley cumulative threshold |
| `var_cusum_k` | `0.5` | VarCUSUM allowance |
| `var_cusum_h` | `5.0` | VarCUSUM decision threshold |
| `seasonal_period` | `0` | Seasonal period (0 = auto-detect via FFT) |

#### H — AQB / Scoring / IO

| Field | Default | Description |
|-------|---------|-------------|
| `quantile_threshold_mode` | `False` | Use Adaptive Quantile Baseline |
| `aqb_window` | `200` | AQB quantile estimation window |
| `aqb_q_low` | `0.01` | Lower AQB quantile |
| `aqb_q_high` | `0.99` | Upper AQB quantile |
| `history_maxlen` | `5000` | Maximum result records in memory |
| `csv_path` | `""` | Stream results to CSV if non-empty |
| `log_level` | `"WARNING"` | Python logging level |

#### V9.0 — Frequency Decomposition

| Field | Default | Description |
|-------|---------|-------------|
| `enable_frequency_decomposition` | `True` | Enable FFT into five carrier-wave bands |
| `min_window_for_fft` | `32` | Minimum window before FFT runs |

#### V9.0 — Cross-Frequency Coupling

| Field | Default | Description |
|-------|---------|-------------|
| `enable_coupling_detection` | `True` | Enable phase-amplitude coupling |
| `coupling_degradation_threshold` | `0.3` | Composite score below this → `COUPLING_DEGRADATION` |
| `coupling_trend_window` | `10` | FrequencyBands snapshots for coupling trend |

#### V9.0 — Structural-Rhythmic Coherence

| Field | Default | Description |
|-------|---------|-------------|
| `enable_channel_coherence` | `True` | Enable Channel 1–2 coherence |
| `coherence_threshold` | `0.4` | Score below this → `STRUCTURAL_RHYTHMIC_DECOUPLING` |
| `coherence_window` | `20` | Rolling coherence window |

#### V9.0 — Cascade Precursor & Sequence Logging

| Field | Default | Description |
|-------|---------|-------------|
| `enable_cascade_detection` | `True` | Enable `CASCADE_PRECURSOR` detection |
| `cascade_ews_threshold` | `2` | Minimum EWS indicators elevated |
| `enable_sequence_logging` | `True` | Enable temporal degradation sequence logging |
| `sequence_retention` | `1000` | Maximum sequences to retain |

---

## Pipeline Architecture — 37 Steps (v12.2)

Every call to `update_and_check()` runs all 37 steps in order. Steps read from and write to a shared `StepContext.scratch` dictionary.

| # | Step | Version | Description |
|---|------|---------|-------------|
| 1 | `CoreEWMAStep` | v8 | EWMA baseline + deviation; must run first |
| 2 | `StructuralSnapshotStep` | v9 | Channel 1: mean, variance, skewness, kurtosis, autocorrelation, stationarity |
| 3 | `FrequencyDecompositionStep` | v9 | Channel 2: FFT into 5 carrier-wave bands with power and phase |
| 4 | `CUSUMStep` | v8 | CUSUM persistent shift detection |
| 5 | `RegimeStep` | v8 | Regime change with soft alpha boost |
| 6 | `VarCUSUMStep` | v8 | CUSUM on variance |
| 7 | `PageHinkleyStep` | v8 | Page-Hinkley drift detector |
| 8 | `STIStep` | v8 | Shear-Turbulence Index |
| 9 | `TPSStep` | v8 | Temporal Phase Space reconstruction |
| 10 | `OscDampStep` | v8 | Oscillation damping detection |
| 11 | `CPDStep` | v8 | Change-Point Detection |
| 12 | `RPIStep` | v8 | Rhythm Periodicity Index (FFT) |
| 13 | `RFIStep` | v8 | Rhythm Fractal Index (Hurst / R-S) |
| 14 | `SSIStep` | v8 | Synchronization Stability Index — Kuramoto proxy (`rsi` alias preserved) |
| 15 | `PEStep` | v8 | Permutation Entropy (FRM Axiom 3) |
| 16 | `EWSStep` | v8 | Early Warning Signals — variance + AC(1) (FRM Axiom 9) |
| 17 | `AQBStep` | v8 | Adaptive Quantile Baseline |
| 18 | `SeasonalStep` | v8 | Seasonal decomposition with FFT auto-period |
| 19 | `MahalStep` | v8 | Mahalanobis distance (multivariate) |
| 20 | `RRSStep` | v8 | Robust Residual Score |
| 21 | `BandAnomalyStep` | v9 | Per-carrier-wave anomaly invisible to composite |
| 22 | `CrossFrequencyCouplingStep` | v9 | PAC coupling matrix + `COUPLING_DEGRADATION` alert |
| 23 | `ChannelCoherenceStep` | v9 | Structural-rhythmic coherence + `SR_DECOUPLING` alert |
| 24 | `CascadePrecursorStep` | v9 | `CASCADE_PRECURSOR` — CRITICAL; requires all three conditions |
| 25 | `DegradationSequenceStep` | v9 | Channel 3: temporal degradation sequence log |
| 26 | `ThroughputEstimationStep` | **v10** | P_throughput from band amplitudes |
| 27 | `MaintenanceBurdenStep` | **v10** | μ = 1−κ̄ (coupling overhead heuristic) → regime classification |
| 28 | `PhaseExtractionStep` | **v10** | FFT bandpass + Hilbert transform → instantaneous phase per band |
| 29 | `PACCoefficientStep` | **v10** | Modulation Index (Tort 2010) across 6 slow/fast band pairs |
| 30 | `PACDegradationStep` | **v10** | Rolling PAC history → `pac_degradation_rate`, `pre_cascade_pac` |
| 31 | `CriticalCouplingEstimationStep` | **v10** | κ_c = 2/(π·g(ω₀)) from power-weighted frequency spread |
| 32 | `CouplingRateStep` | **v10** | dκ̄/dt from rolling coupling history |
| 33 | `DiagnosticWindowStep` | **v10** | Δt = (κ̄−κ_c)/\|dκ̄/dt\|; confidence grading; supercompensation |
| 34 | `KuramotoOrderStep` | **v10** | Φ = \|mean(e^iθ_k)\| — phase coherence independent of κ̄ |
| 35 | `SequenceOrderingStep` | **v10** | COUPLING_FIRST / COHERENCE_FIRST / SIMULTANEOUS / STABLE per step |
| 36 | `ReversedSequenceStep` | **v10** | Atypical degradation ordering → sequence classification |
| 37 | `AlertReasonsStep` | v8 | Must run last — aggregates all alert signals |

---

## V9.0 Alert Types and Data Structures

### Frozen data structures

| Class | Channel | Description |
|-------|---------|-------------|
| `FrequencyBands` | 2 | Five carrier-wave band powers and phases |
| `StructuralSnapshot` | 1 | Mean, variance, skewness, kurtosis, autocorrelation, stationarity |
| `CouplingMatrix` | 2 | PAC coefficients between adjacent bands + composite score |
| `ChannelCoherence` | 1↔2 | Structural-rhythmic coherence score |
| `DegradationSequence` | 3 | Temporal ordering of channel degradation events |

### Alert types

| `AlertType` | `AlertSeverity` | Trigger |
|-------------|-----------------|---------|
| `BAND_ANOMALY` | WARNING | Per-carrier-wave anomaly |
| `COUPLING_DEGRADATION` | WARNING | `composite_coupling_score` below threshold |
| `STRUCTURAL_RHYTHMIC_DECOUPLING` | ALERT | `coherence_score` below threshold |
| `CASCADE_PRECURSOR` | **CRITICAL** | All three: coupling + decoupling + ≥N EWS |

---

## SentinelResult API

`SentinelResult` is a `dict` subclass. All v8.0/v9.0 dictionary access patterns work unchanged.

### Core fields

| Key | Type | Description |
|-----|------|-------------|
| `"step"` | `int` | Monotonic observation counter |
| `"value"` | `float` | Raw input value |
| `"ewma"` | `float` | Current EWMA estimate |
| `"alert"` | `bool` | Any anomaly triggered |
| `"warmup"` | `bool` | True during warmup period |
| `"anomaly_score"` | `float` | Normalized composite score |
| `"z_score"` | `float` | EWMA deviation in sigma units |
| `"alert_reasons"` | `list[str]` | Triggered conditions |
| `"cascade_precursor_active"` | `bool` | CASCADE_PRECURSOR active |

### V9.0 convenience methods

```python
result.is_cascade_precursor() -> bool
result.get_channel_status() -> dict
result.get_degradation_narrative() -> str
result.get_primary_carrier_wave() -> str
```

### V10.0 convenience methods

```python
result.is_reversed_sequence() -> bool
result.get_maintenance_burden() -> dict   # {"mu": 0.82, "regime": "TAINTER_WARNING"}
result.get_pac_status() -> dict           # {"mean_pac": 0.35, "degradation_rate": 0.19, ...}
result.get_diagnostic_window() -> dict    # {"steps": 47.3, "confidence": "HIGH", ...}
result.get_intervention_signature() -> dict
```

---

## MultiStreamSentinel

Thread-safe manager for multiple independent named streams.

```python
from fracttalix import MultiStreamSentinel, SentinelConfig

mss = MultiStreamSentinel(config=SentinelConfig.production())
result = mss.update("sensor_42", 3.14)
result = await mss.aupdate("sensor_42", 3.14)

mss.list_streams()
mss.get_detector("sensor_42")
mss.reset_stream("sensor_42")
mss.delete_stream("sensor_42")

state_json = mss.save_all()
mss.load_all(state_json)
```

---

## SentinelBenchmark

Built-in evaluation harness with five labeled anomaly archetypes.

```python
from fracttalix import SentinelBenchmark, SentinelConfig

bench = SentinelBenchmark(n=500, config=SentinelConfig.sensitive())
bench.run_suite()   # reports F1, AUPRC, VUS-PR, mean lag, 3σ baseline

data, labels = bench.generate("drift")
metrics = bench.evaluate(data, labels)
```

---

## SentinelServer — REST API

Asyncio HTTP server wrapping a `MultiStreamSentinel`. No framework dependencies.

```bash
python -m fracttalix --serve --host 0.0.0.0 --port 8765
```

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Version, uptime, stream count |
| `GET` | `/streams` | List all stream IDs |
| `POST` | `/update/<id>` | Feed observation: `{"value": 3.14}` |
| `GET` | `/status/<id>` | Stream stats and last result |
| `DELETE` | `/stream/<id>` | Delete stream |
| `POST` | `/reset/<id>` | Reset stream to factory state |

---

## CLI Reference

```
fracttalix [OPTIONS]
# or: python -m fracttalix [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--file`, `-f` | — | CSV file to process (reads first column) |
| `--alpha` | `0.1` | EWMA smoothing factor |
| `--multiplier` | `3.0` | Alert threshold multiplier |
| `--warmup` | `30` | Warmup periods |
| `--benchmark` | — | Run benchmark suite and exit |
| `--serve` | — | Start HTTP server |
| `--host` | `0.0.0.0` | Server host |
| `--port` | `8765` | Server port |
| `--version` | — | Print version and exit |
| `--test` | — | Run smoke suite and exit |

---

## Backward Compatibility

All v7.x through v12.x call patterns continue to work unchanged. `SentinelDetector` is importable and functional. The only breaking change from v12.3 is that `SentinelDetector` is no longer the primary API — `FRMSuite` is.

### V7.x compatibility

```python
from fracttalix import SentinelDetector
det = SentinelDetector(alpha=0.1, multiplier=3.0, warmup_periods=30)  # v7.x kwargs accepted
```

### State persistence

```python
json_str = det.save_state()
det2 = SentinelDetector(config)
det2.load_state(json_str)
```
