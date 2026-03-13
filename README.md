# Fracttalix Sentinel v12.3

**Streaming anomaly detection grounded in the Three-Channel Model of Dissipative Network Information Transmission — extended with four signal-processing collapse indicators (v10.0+).**

Sentinel ingests one scalar (or multivariate) observation at a time and emits a rich result dictionary on every call — no batching, no retraining, no warmup gap once past the configurable warmup window.

> **Theoretical foundation:** Fractal Rhythm Model Papers 1–6
> DOI: [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)
> License: **CC0** — public domain

---

[![PyPI](https://img.shields.io/pypi/v/fracttalix)](https://pypi.org/project/fracttalix/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18859299.svg)](https://doi.org/10.5281/zenodo.18859299)
[![Tests](https://github.com/thomasbrennan/Fracttalix/actions/workflows/tests.yml/badge.svg)](https://github.com/thomasbrennan/Fracttalix/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

> **Keywords:** anomaly detection · streaming time-series · online learning · Hopf bifurcation · critical slowing down · phase-amplitude coupling · Fractal Rhythm Model · change-point detection · multivariate monitoring · REST API

## Table of Contents

1. [Overview](#overview)
2. [Which API should I use?](#which-api-should-i-use)
3. [Three-Channel Model](#three-channel-model)
4. [V12.3 Changes](#v123-changes)
   - [V12.2 Changes](#v122-changes)
5. [Installation](#installation)
6. [Quick Start — SentinelDetector](#quick-start)
7. [DetectorSuite — Modular Five-Detector Suite](#detectorsuite--modular-five-detector-suite)
8. [FRMSuite — FRM Physics Layer](#frmsuite--frm-physics-layer)
9. [SentinelConfig — Configuration](#sentinelconfig--configuration)
10. [Pipeline Architecture — 37 Steps](#pipeline-architecture--37-steps-v122)
11. [Alert Types and Data Structures](#alert-types-and-data-structures)
12. [SentinelResult API](#sentinelresult-api)
13. [MultiStreamSentinel](#multistreamssentinel)
14. [SentinelBenchmark](#sentinelbenchmark)
15. [SentinelServer — REST API](#sentinelserver--rest-api)
16. [CLI Reference](#cli-reference)
17. [Backward Compatibility](#backward-compatibility)
18. [Theoretical Foundation](#theoretical-foundation)
19. [Authors & License](#authors--license)

---

## Overview

Fracttalix Sentinel is a Python package (`pip install fracttalix`) for real-time streaming anomaly detection. Its design priorities are:

- **Zero external dependencies for core operation** — works on the Python standard library alone; numpy, scipy, numba, matplotlib, and tqdm are optional accelerators.
- **Immutable, inspectable configuration** — `SentinelConfig` is a frozen dataclass; every parameter is readable and picklable.
- **Composable pipeline** — 37 `DetectorStep` subclasses execute in sequence.
- **Three-channel anomaly model** — monitors structural properties, broadband rhythmicity, and temporal degradation sequences as independent information channels.
- **Signal-processing collapse indicators** — maintenance burden (coupling heuristic μ = 1−κ̄), PAC pre-cascade detection, diagnostic window estimation, and reversed sequence detection, architecturally inspired by the Kuramoto synchronization framework. These are engineering heuristics, not physical derivations.
- **Full backward compatibility** — all v7.x, v8.0, v9.0, and v10.0 call patterns continue to work unchanged.

---

## Which API Should I Use?

There are three detection APIs in this package, each optimized for a different context:

| API | Best for | Requires | `pip install` |
|-----|----------|----------|---------------|
| **`SentinelDetector`** | Unknown signal types; broadest coverage; exploratory monitoring | stdlib only | `pip install fracttalix` |
| **`DetectorSuite`** | Domain-specific monitoring; need to know *which* anomaly type fired; low FPR | stdlib only | `pip install fracttalix` |
| **`FRMSuite`** | Oscillatory / physiological / power-grid signals with known generation delay; need time-to-bifurcation | numpy + scipy | `pip install fracttalix[fast]` |

**Decision rule:**

- You know your signal is oscillatory and have a domain parameter (`tau_gen`) → **FRMSuite**
- You know which anomaly type matters (drift? variance? discord?) → **DetectorSuite**
- You're monitoring something unknown or want a single alert/no-alert signal → **SentinelDetector**

---

## Three-Channel Model

Implemented from Meta-Kaizen Paper 6:

| Channel | Name | What it monitors |
|---------|------|-----------------|
| **1** | Structural | Network topology as active transmitter — mean, variance, skewness, kurtosis, autocorrelation, stationarity |
| **2** | Rhythmic | Broadband multiplexed oscillatory transmission — FFT decomposition into five carrier-wave bands and cross-frequency phase-amplitude coupling |
| **3** | Temporal | One-way irreversible carrier wave — temporal sequence and ordering of channel degradation events |

**Degradation cascade logic (v9.0):**

```
Band anomaly detected  →  Cross-frequency coupling degrades  →
  Structural-rhythmic channels decouple  →  CASCADE PRECURSOR (CRITICAL)
```

**Extended diagnostics (v10.0):**

```
PAC pre-cascade detected  →  Δt window opens  →  Maintenance burden μ → 1  →
  Coupling rate dκ̄/dt negative  →  Collapse imminent
```

---

## V12.3 Changes

### FPR Elimination & Drift Recovery — Meta Kaisen CBP

v12.3 is a comprehensive recalibration targeting the ~35% FPR floor that
dominated v12.2 performance. FPR dropped 93%, mean F1 rose 25%.

#### New Architecture

- **SeasonalPreprocessStep** (Step 0): FFT-based seasonal decomposition with
  confidence gate `peak_power > 10× mean_power` (<0.1% false detection on
  white noise). All 37 downstream steps receive the deseasonalized residual.
- **Non-adaptive drift CUSUM** in `CUSUMStep`: Accumulates on warmup-frozen
  z-score, detecting slow drift that EWMA adaptation masks. Fires
  `drift_cusum_alert`, resets, and re-fires continuously during ongoing drift.
- **ConsensusGate** in `AlertReasonsStep`: Requires ≥2 soft alerts OR 1
  strong alert (`cusum_mean_shift`, `cusum_variance_spike`, `drift_cusum_shift`,
  `gradual_drift`, `cascade_precursor`) OR |z| ≥ 5σ bypass. Primary FPR
  reduction mechanism.

#### Recalibrated Thresholds (null-distribution calibrated on N(0,1))

- `rfi_threshold`: 0.40 → 0.52; `pe_threshold`: 0.05 → 0.15
- `var_cusum_k`: 0.5 → 1.0 (E[z²]=1.0 under H₀; old k was systematically biased)
- `var_cusum_h`: 5.0 → 10.0; `cusum_k`: 0.5 → 1.0; `cusum_h`: 5.0 → 8.0
- `coherence_threshold`: 0.40 → 0.30; `coupling_degradation_threshold`: 0.30 → 0.24

#### v12.3 Benchmark (n=1000, seed=42, post-warmup)

| Archetype  | v12.2 F1 | v12.3 F1 | Change   |
|------------|----------|----------|----------|
| point      | 0.422    | 0.639    | +51%     |
| contextual | 0.242    | 0.378    | +56%     |
| collective | 0.239    | 0.356    | +49%     |
| drift      | 0.723    | 0.766    | +6%      |
| variance   | 0.876    | 0.987    | +13%     |
| **FPR**    | **35%**  | **2.6%** | **−93%** |
| Mean F1    | 0.500    | 0.625    | **+25%** |

> `SentinelDetector()` (no args) now defaults to `SentinelConfig.production()`
> (multiplier=4.5). The FPR floor has been eliminated by threshold recalibration
> and ConsensusGate — not by multiplier inflation.

---

## V12.2 Changes

### Epistemic Language Corrections

- Replaced "physics-derived" framing with "signal-processing heuristic" throughout
  README and docstrings. The capabilities are real and useful; the claim that they
  are *derived from physics* was overstated and contradicted by code comments.
- Corrected the README maintenance burden formula (was showing the abandoned v10.0
  formula `μ = N·κ̄·E_coupling/P_throughput`; actual implementation is `μ = 1−κ̄`
  as documented in `MaintenanceBurdenStep` since v11.0).
- Reframed "thermodynamic arrow" and "intervention signature" language as signal
  classification labels, not physical or causal claims.

### Default Multiplier Change (Breaking)

- `SentinelConfig.production()` now uses `multiplier=4.5` (was 3.0).
  Measured normal alert rate on white noise N(0,1), seed=99 (n=1000, post-warmup):
  **36.7% → 35.4%** — a 1.3 pp marginal improvement only.

  > **Important:** The FPR floor (~35%) is not dominated by the EWMA threshold.
  > It is driven by other pipeline channels (coherence, coupling, physics steps).
  > Raising the multiplier alone cannot bring FPR below ~35%. The root cause
  > requires further investigation (see `benchmark/investigate_fpr_s47.py` for
  > channel attribution data).

  Measured benchmark F1 at the new default (n=1000, seed=42):

| Archetype  | v12.1 F1 (mult=3.0) | v12.2 F1 (mult=4.5) | Δ |
|------------|---------------------|---------------------|---|
| point      | 0.415               | 0.422               | +0.007 |
| contextual | 0.247               | 0.242               | −0.005 |
| collective | 0.239               | 0.242               | +0.003 |
| drift      | 0.723               | 0.727               | +0.004 |
| variance   | 0.876               | 0.883               | +0.007 |
| **normal FPR** | **36.7%**       | **35.4%**           | −1.3 pp |

  F1 changes are within noise. The multiplier change is a threshold preference,
  not a performance fix.

> Users who depended on the v12.1 FPR behaviour can restore it with
> `SentinelConfig(multiplier=3.0)` or set any custom multiplier.

## V12.1 Changes

### Bug Fixes

- **VarCUSUM non-reset** (`VarCUSUMStep`): accumulators `s_hi`/`s_lo` now re-arm
  after each threshold crossing. Normal alert rate: 97% → 35.6%.
- **ChannelCoherence unit mismatch** (`ChannelCoherenceStep`): replaced
  rate-difference formula with Pearson correlation (scale-invariant). Normal data
  now scores ~0.5 above threshold.

### v12.1 Benchmark (n=1000, seed=42, multiplier=3.0)

| Archetype   |  F1   | Normal alert rate |
|-------------|-------|-------------------|
| point       | 0.415 | —                 |
| contextual  | 0.247 | —                 |
| collective  | 0.239 | —                 |
| drift       | 0.723 | —                 |
| variance    | 0.876 | —                 |
| **normal**  | —     | **35.6%**         |

---

## Collapse Indicator Capabilities

Four signal-processing indicators architecturally inspired by the Kuramoto synchronization framework (added v10.0).

> **Epistemic status:** These are engineering heuristics, not physical derivations. The maintenance burden μ is explicitly NOT derived from Tainter's socioeconomic model or from any energy-fraction physics — see the full disclaimer in `fracttalix/steps/physics.py` `MaintenanceBurdenStep`. The regime names (TAINTER_CRITICAL, etc.) are descriptive labels; thresholds are empirically set, not calibrated from data.

### 1. Maintenance Burden μ (Coupling Overhead Indicator)

```
μ = 1 − κ̄
```

where κ̄ is the mean cross-frequency coupling score. Low coupling (κ̄ → 0) implies high coordination overhead → high inferred maintenance burden (μ → 1). High coupling (κ̄ → 1) implies efficient coordination → low burden (μ → 0).

**Note:** This is an engineering heuristic. μ is NOT derived from Tainter's socioeconomic collapse model. The regime labels below are classification shortcuts; thresholds are empirically set, not calibrated from data.

| μ range | Regime | Meaning |
|---------|--------|---------|
| < 0.5 | `HEALTHY` | High coupling, low inferred overhead |
| 0.5 – 0.75 | `REDUCED_RESERVE` | Coupling declining |
| 0.75 – 0.9 | `TAINTER_WARNING` | Approaching fragmented state |
| ≥ 0.9 | `TAINTER_CRITICAL` | Very low coupling detected |

```python
mb = result.get_maintenance_burden()
# {"mu": 0.82, "regime": "TAINTER_WARNING"}
```

### 2. PAC Pre-Cascade Detection (Extended Diagnostic Window)

Phase-Amplitude Coupling (PAC) measures the depth of nonlinear coupling architecture — the structural memory of the network. PAC degrades **before** mean coupling strength κ̄ measurably decreases, providing an earlier warning signal than the v9.0 cascade precursor.

Method: Modulation Index (Tort et al. 2010) across 6 slow-phase/fast-amplitude band pairs.

```python
pac = result.get_pac_status()
# {"mean_pac": 0.41, "degradation_rate": 0.18, "pre_cascade_pac": True}
```

### 3. Diagnostic Window Δt Estimation (Time-to-Collapse)

```
Δt = (κ̄ - κ_c) / |dκ̄/dt|
```

The estimated number of observations remaining before coherence collapse, given current coupling strength and its rate of change. Sentinel stops just detecting that collapse is coming and starts estimating **when**.

- Only active when κ̄ > κ_c and dκ̄/dt < 0
- Confidence graded: `HIGH` / `MEDIUM` / `LOW` based on rate stability
- Detects **supercompensation** (adaptive recovery in progress)

```python
dw = result.get_diagnostic_window()
# {"steps": 47.3, "confidence": "HIGH", "supercompensation": False}
```

### 4. Reversed Sequence Detection (Sequence Classification)

The heuristic ordering hypothesis: **coupling typically degrades before coherence collapses** in organic degradation patterns. A reversed sequence — coherence collapsing before coupling degrades — may indicate:

1. Measurement error or noise
2. A different data-generating process
3. **Possible external perturbation** (a pattern distinct from gradual organic decay)

> **Note:** "Intervention" here is a signal classification label, not a causal claim about deliberate external action. `sequence_type: "REVERSED"` means the ordering is atypical relative to the heuristic baseline; `intervention_signature_score` is a confidence value for that classification, not a probability of any external cause.

```python
if result.is_reversed_sequence():
    sig = result.get_intervention_signature()
    # {"score": 0.74, "sequence_type": "REVERSED",
    #  "phi_rate": -0.18, "coupling_rate": -0.01}
```

---

## Installation

```bash
pip install fracttalix
```

**Optional accelerators (install any or none):**

```bash
pip install numpy          # FFT, PAC computation, Hilbert transform
pip install scipy          # scipy.signal.hilbert (falls back to numpy)
pip install numba          # JIT compilation for hot loops
pip install matplotlib     # plot_history() dashboard
pip install tqdm           # progress bars in benchmark
```

**Run tests (374 tests, all expected to pass):**

```bash
pytest
```

---

## Quick Start

### Basic use — production defaults (v12.2)

The API is unchanged from v12.1. The production() preset now uses `multiplier=4.5`
(was 3.0). Measured FPR on white noise: 35.4% (was 36.7%) — marginal.
The primary improvement in v12.2 is epistemic: language corrections to README and
docstrings that previously overstated the physics basis of the heuristics.

```python
from fracttalix import SentinelDetector, SentinelConfig

det = SentinelDetector(SentinelConfig.production())  # multiplier=4.5, ~35% normal FPR

for value in my_data_stream:
    result = det.update_and_check(value)
    if result["alert"]:
        print(f"Step {result['step']}: {result['alert_reasons']}")
```

If you were on v12.1 and need the old threshold back:

```python
det = SentinelDetector(SentinelConfig(multiplier=3.0))  # v12.1 behaviour (~37% FPR)
```

### Choosing sensitivity

The multiplier adjusts only the EWMA z-score threshold. Note that the overall
normal alert rate has a floor of ~35% driven by other pipeline channels —
see `benchmark/investigate_fpr_s47.py` for channel attribution.

```python
# ~35% normal alert rate — v12.2 production default
det = SentinelDetector(SentinelConfig.production())          # multiplier=4.5

# ~37% normal alert rate — v12.1 default; slightly more sensitive EWMA
det = SentinelDetector(SentinelConfig(multiplier=3.0))

# ~40-50% normal alert rate — catches the subtlest shifts; pairs with human review
det = SentinelDetector(SentinelConfig.sensitive())           # multiplier=2.5

# ~67% normal alert rate — maximum sensitivity; use with downstream filtering
det = SentinelDetector(SentinelConfig(multiplier=1.5))
```

Auto-tune picks the multiplier that maximises F1 on your labeled examples:

```python
labeled = [(value, is_anomaly), ...]
det = SentinelDetector.auto_tune(data=[], labeled_data=labeled)
```

### Collapse indicators

Four signal-processing indicators track how coupling and coherence are evolving.
They are heuristics — useful for early warning and pattern characterisation,
not physical measurements.

```python
result = det.update_and_check(value)

# Coupling overhead indicator (μ = 1 − κ̄)
# High μ = low cross-frequency coupling = high inferred coordination cost
mb = result.get_maintenance_burden()
if mb["regime"] in ("TAINTER_WARNING", "TAINTER_CRITICAL"):
    print(f"Coupling fragmented: μ={mb['mu']:.2f} ({mb['regime']})")

# PAC pre-cascade: phase-amplitude coupling degrading before κ̄ drops
# Fires earlier than the cascade precursor — gives more lead time
pac = result.get_pac_status()
if pac["pre_cascade_pac"]:
    print(f"PAC degrading at rate {pac['degradation_rate']:.3f} — early warning")

# Diagnostic window: estimated steps before coherence collapse
# Only active when κ̄ > κ_c and coupling is falling
dw = result.get_diagnostic_window()
if dw["steps"] is not None:
    print(f"Δt ≈ {dw['steps']:.0f} steps ({dw['confidence']} confidence)")
if dw["supercompensation"]:
    print("Coupling recovering — possible adaptive response")

# Sequence classification: is coherence collapsing before coupling degrades?
# REVERSED means atypical ordering; it is a classification label, not a causal claim
if result.is_reversed_sequence():
    sig = result.get_intervention_signature()
    print(f"Atypical sequence (score {sig['score']:.2f}) — ordering does not match "
          f"gradual organic decay pattern")
```

### Three-channel status

```python
# CASCADE_PRECURSOR requires all three conditions simultaneously:
# coupling degradation + structural-rhythmic decoupling + ≥2 EWS indicators elevated
if result.is_cascade_precursor():
    print("CRITICAL: cascade precursor — all three channels confirming")

status = result.get_channel_status()
# {"structural": "healthy", "rhythmic_composite": "degrading",
#  "coupling": "healthy", "coherence": "healthy"}

print(result.get_degradation_narrative())
print(result.get_primary_carrier_wave())   # "mid", "low", etc.
```

### Multivariate streams

```python
cfg = SentinelConfig(multivariate=True, n_channels=3)
det = SentinelDetector(config=cfg)
result = det.update_and_check([v1, v2, v3])
```

### Async usage

```python
result = await det.aupdate(value)
```

### Multiple streams (thread-safe)

```python
from fracttalix import MultiStreamSentinel

mss = MultiStreamSentinel(config=SentinelConfig.production())

# Each stream ID gets its own independent detector instance
result = mss.update("sensor_42", 3.14)
result = await mss.aupdate("sensor_43", 7.71)
```

---

## DetectorSuite — Modular Five-Detector Suite

`DetectorSuite` runs five independent specialized detectors in parallel. Unlike `SentinelDetector`, each detector can report `OUT_OF_SCOPE` when the current data does not match its model — so you only get alerts from detectors that understand your signal. No stdlib-only constraint.

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

## FRMSuite — FRM Physics Layer

`FRMSuite` is the full two-layer suite: `DetectorSuite` (Layer 1, generic) plus three FRM-physics detectors (Layer 2, scipy required).

```bash
pip install fracttalix[fast]   # includes numpy + scipy for Layer 2
```

### Architecture

```
FRMSuite
├── Layer 1 — DetectorSuite (5 generic detectors, no scipy)
│   ├── HopfDetector(ews)    — critical slowing down
│   ├── DiscordDetector      — point / contextual anomalies
│   ├── DriftDetector        — slow mean shift
│   ├── VarianceDetector     — volatility change
│   └── CouplingDetector     — PAC cross-scale decoupling (independent cross-validator)
│
└── Layer 2 — FRM Physics (scipy required; graceful degradation if absent)
    ├── Lambda  (HopfDetector frm) — tracks damping λ → 0
    ├── Omega   (OmegaDetector)    — tracks ω vs π/(2·τ_gen)
    └── Virtu   (VirtuDetector)    — time-to-bifurcation estimate
```

**Graceful degradation:** If scipy is absent, Layer 2 detectors return `OUT_OF_SCOPE` and `frm_confidence` stays at 0. Layer 1 operates normally.

### Key Concept: `tau_gen`

`tau_gen` is the FRM generation delay — the characteristic delay of your system. When supplied:

- **Lambda** (strong mode): uses the fixed predicted frequency ω = π/(2·`tau_gen`) as its reference; curve-fits damping λ against this.
- **Omega** (strong mode): checks that the observed dominant frequency matches ω = π/(2·`tau_gen`). A 5% deviation fires ALERT.
- **frm_confidence** counts strong-mode Layer 2 detectors in ALERT (0–3).

When `tau_gen=None`: Lambda and Omega run in weak mode (generic frequency tracking). `frm_confidence` stays 0 in weak mode — the FRM physics test cannot run without the model parameter.

> Get `tau_gen` from domain knowledge. For power grids, it is the generator inertia constant. For EEG alpha rhythms, it is the reciprocal of the cycle frequency. For mechanical oscillators, it is the half-period.

### `frm_confidence` Score

| `frm_confidence` | Meaning |
|-----------------|---------|
| 0 | No strong-mode Layer 2 detector alerting |
| 1 | Lambda alerting (λ declining) |
| 2 | Lambda + Omega alerting (λ declining AND ω drifting from prediction) |
| 3 | Lambda + Omega + Virtu all alerting (full FRM bifurcation signal) |
| `frm_confidence_plus` | `frm_confidence` + 1 if CouplingDetector (Layer 1) is also alerting — independent cross-validation |

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

result.omega.is_alert          # ω drifting from prediction?
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

## SentinelConfig — Configuration

`SentinelConfig` is a frozen dataclass (`slots=True`). All fields are immutable after construction. Use `dataclasses.replace(cfg, field=value)` to derive a new config.

### Factory presets

| Preset | `alpha` | `multiplier` | `warmup` | Normal FPR¹ | Notes |
|--------|---------|-------------|----------|-------------|-------|
| `SentinelConfig.fast()` | 0.3 | 3.0 | 10 | ~60–80% | Fastest response; very high FP rate — use only with downstream filtering |
| `SentinelConfig.production()` | 0.1 | **4.5** | 30 | ~35% | Balanced defaults; v12.2 default |
| `SentinelConfig.sensitive()` | 0.05 | 2.5 | 50 | ~40–50% | Catches subtle anomalies; high FP rate |
| `SentinelConfig.realtime()` | 0.2 | 3.0 | 15 | ~30–40% | Quantile-adaptive thresholds |

> ¹ Approximate normal alert rate on white noise N(0,1), empirically measured. FPR is a function of multiplier, alpha, and data distribution — these are indicative values from `benchmark/investigate_fpr_s47.py`.
>
> **Multiplier–FPR trade-off** (white noise, seed=99, n=1000): multiplier 1.5 → 66.8%, 2.0 → 47.5%, 2.5 → 39.8%, 3.0 → 36.7%, 4.5 → 35.4%, 6.0 → 35.4%. Note: above ~3.5 the curve flattens — FPR is floor-limited (~35%) by non-EWMA channels.

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
| `sti_window` | `20` | Shear-Turbulence Index window (signal heuristic inspired by turbulence concepts) |
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
| 26 | `ThroughputEstimationStep` | **v10** | P_throughput from band amplitudes; populates `band_amplitudes`, `band_powers`, `node_count`, `mean_coupling_strength` |
| 27 | `MaintenanceBurdenStep` | **v10** | μ = 1−κ̄ (coupling overhead heuristic) → regime classification |
| 28 | `PhaseExtractionStep` | **v10** | FFT bandpass + Hilbert transform → instantaneous phase per band |
| 29 | `PACCoefficientStep` | **v10** | Modulation Index (Tort 2010) across 6 slow/fast band pairs |
| 30 | `PACDegradationStep` | **v10** | Rolling PAC history → `pac_degradation_rate`, `pre_cascade_pac` |
| 31 | `CriticalCouplingEstimationStep` | **v10** | κ_c = 2/(π·g(ω₀)) from power-weighted frequency spread |
| 32 | `CouplingRateStep` | **v10** | dκ̄/dt from rolling coupling history |
| 33 | `DiagnosticWindowStep` | **v10** | Δt = (κ̄−κ_c)/|dκ̄/dt|; confidence grading; supercompensation |
| 34 | `KuramotoOrderStep` | **v10** | Φ = |mean(e^iθ_k)| — phase coherence independent of κ̄ |
| 35 | `SequenceOrderingStep` | **v10** | COUPLING_FIRST / COHERENCE_FIRST / SIMULTANEOUS / STABLE per step |
| 36 | `ReversedSequenceStep` | **v10** | Atypical degradation ordering → sequence classification + intervention_signature_score |
| 37 | `AlertReasonsStep` | v8 | Must run last — aggregates all alert signals |

### Custom steps

```python
from fracttalix import DetectorStep, StepContext, register_step

@register_step
class MyStep(DetectorStep):
    def __init__(self, config):
        self.cfg = config

    def update(self, ctx: StepContext) -> None:
        ctx.scratch["my_metric"] = ctx.current * 2.0

    def reset(self) -> None:
        pass

    def state_dict(self):
        return {}

    def load_state(self, sd):
        pass
```

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

result.get_maintenance_burden() -> dict
# {"mu": 0.82, "regime": "TAINTER_WARNING"}

result.get_pac_status() -> dict
# {"mean_pac": 0.35, "degradation_rate": 0.19, "pre_cascade_pac": True}

result.get_diagnostic_window() -> dict
# {"steps": 47.3, "confidence": "HIGH", "supercompensation": False}

result.get_intervention_signature() -> dict
# {"score": 0.74, "sequence_type": "REVERSED",
#  "phi_rate": -0.18, "coupling_rate": -0.01}
```

### V10.0 scalar result keys

| Key | Type | Description |
|-----|------|-------------|
| `"maintenance_burden"` | `float` | μ (0.0–1.0) |
| `"tainter_regime"` | `str` | HEALTHY / REDUCED_RESERVE / TAINTER_WARNING / TAINTER_CRITICAL |
| `"mean_pac"` | `float` | Current PAC strength (0.0–1.0) |
| `"pac_degradation_rate"` | `float` | Fractional PAC decline rate |
| `"pre_cascade_pac"` | `bool` | PAC warning before cascade precursor |
| `"diagnostic_window_steps"` | `float\|None` | Heuristic estimate of steps until coherence collapse (trajectory extrapolation under current conditions) |
| `"diagnostic_window_confidence"` | `str` | HIGH / MEDIUM / LOW / NOT_APPLICABLE |
| `"supercompensation_detected"` | `bool` | Adaptive recovery in progress |
| `"kuramoto_order"` | `float` | Φ inter-band phase coherence (0.0–1.0) |
| `"reversed_sequence"` | `bool` | Coherence collapsing before coupling |
| `"intervention_signature_score"` | `float` | 0.0–1.0 confidence of atypical sequence ordering (classification label, not causal claim) |
| `"sequence_type"` | `str` | ORGANIC / REVERSED / AMBIGUOUS / INSUFFICIENT_DATA |
| `"coupling_rate"` | `float` | dκ̄/dt (negative = degrading) |
| `"critical_coupling"` | `float` | κ_c estimated from frequency distribution |

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
mss.status("sensor_42")
mss.reset_stream("sensor_42")
mss.delete_stream("sensor_42")

state_json = mss.save_all()
mss.load_all(state_json)
```

---

## SentinelBenchmark

Built-in evaluation harness with five labeled anomaly archetypes.

| Archetype | Description |
|-----------|-------------|
| `point` | Sparse large spikes (8σ) at fixed intervals |
| `contextual` | Values anomalous given sinusoidal seasonal context |
| `collective` | Extended runs of moderately elevated values |
| `drift` | Slow linear mean drift starting mid-series |
| `variance` | Sudden 4× variance explosion in second half |

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

```bash
curl -X POST http://localhost:8765/update/my_sensor \
     -H "Content-Type: application/json" \
     -d '{"value": 42.0}'
```

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

v12.2 is a strict superset of all prior versions. No step is removed. No result key is removed. The only breaking change from v12.1 is the `production()` default multiplier (3.0 → 4.5) — restore with `SentinelConfig(multiplier=3.0)`.

### V8.0 root-cause fixes (all preserved)

| Fix | Label | Description |
|-----|-------|-------------|
| α | Frozen config | `SentinelConfig` is a frozen dataclass |
| β | WindowBank | Named independent deques per consumer |
| γ | Pipeline decomposition | 37 `DetectorStep` subclasses |
| δ | Soft regime boost | Multiplicative boost + decay |
| ε | SSI naming | `result["rsi"]` alias preserved |

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

---

## Theoretical Foundation

> The FRM components below refer to concepts in the Fractal Rhythm Model working papers (DOI: [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)). These are working-paper concepts, not established scientific axioms. The implementations are signal-processing heuristics inspired by the framework.

| FRM Component | Sentinel Implementation |
|---------------|------------------------|
| FRM Concept 3 (ordinal pattern complexity) | `PEStep` — Permutation Entropy |
| FRM Concept 9 (critical slowing down) | `EWSStep` — variance + lag-1 autocorrelation |
| Rhythm Periodicity Index | `RPIStep` — FFT spectral coherence |
| Rhythm Fractal Index | `RFIStep` — Hurst exponent via R/S |
| Synchronization Stability Index | `SSIStep` — Kuramoto proxy via FFT phase coherence |
| Three-channel model (Paper 6) | `StructuralSnapshotStep`, `FrequencyDecompositionStep`, `ChannelCoherenceStep`, `CascadePrecursorStep`, `DegradationSequenceStep` |
| Maintenance burden μ (heuristic) | `ThroughputEstimationStep`, `MaintenanceBurdenStep` |
| PAC pre-cascade (Tort 2010 method) | `PhaseExtractionStep`, `PACCoefficientStep`, `PACDegradationStep` |
| Diagnostic window Δt (heuristic estimate) | `CriticalCouplingEstimationStep`, `CouplingRateStep`, `DiagnosticWindowStep` |
| Kuramoto order Φ / reversed sequence | `KuramotoOrderStep`, `SequenceOrderingStep`, `ReversedSequenceStep` |

**DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)

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

## Authors & License

**Authors:** Thomas Brennan & Claude (Anthropic) & Grok (xAI)

**License:** [CC0 1.0 Universal (Public Domain)](https://creativecommons.org/publicdomain/zero/1.0/) — no restrictions, no attribution required.

**GitHub:** [https://github.com/thomasbrennan/Fracttalix](https://github.com/thomasbrennan/Fracttalix)

**Version history:**

| Version | Notes |
|---------|-------|
| v12.3.0 | FPR floor eliminated via ConsensusGate + threshold recalibration; SeasonalPreprocessStep |
| v12.2.0 | Epistemic language corrections; production() multiplier 3.0→4.5 |
| v12.1.0 | VarCUSUM non-reset fix; ChannelCoherence unit mismatch fix |
| v10.0.0 | 4 collapse indicators (v10 API), 37 steps, 98 tests |
| v9.0.0 | Three-channel model, 26 steps, 65 tests |
| v8.0.0 | Frozen config, WindowBank, 19-step pipeline |
| v7.11–v7.6 | Earlier releases |
