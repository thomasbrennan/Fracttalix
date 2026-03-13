# Fracttalix Sentinel Detector Suite

**Eight modular, scope-aware detectors for real-time anomaly detection in streaming time-series data.**

The suite is not a consensus machine. It is a dashboard. Each detector gives an independent opinion. The user decides which opinions matter for their domain.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Core Concepts](#core-concepts)
3. [Detector Reference](#detector-reference)
   - [HopfDetector — Pre-Transition Early Warning](#hopfdetector--pre-transition-early-warning)
   - [DiscordDetector — Point Anomalies](#discorddetector--point-anomalies)
   - [DriftDetector — Slow Distribution Shifts](#driftdetector--slow-distribution-shifts)
   - [VarianceDetector — Sudden Volatility Changes](#variancedetector--sudden-volatility-changes)
   - [CouplingDetector — Cross-Frequency Decoupling](#couplingdetector--cross-frequency-decoupling)
   - [LambdaDetector — Bifurcation Proximity (FRM)](#lambdadetector--bifurcation-proximity-frm)
   - [OmegaDetector — Timescale Integrity (FRM)](#omegadetector--timescale-integrity-frm)
   - [VirtuDetector — Decision Rationality (FRM)](#virtudetector--decision-rationality-frm)
4. [Quick Start](#quick-start)
5. [Recommended Configurations](#recommended-configurations)
6. [Scope Awareness](#scope-awareness)
7. [Serialisation and State Management](#serialisation-and-state-management)
8. [Dependencies](#dependencies)
9. [Theoretical Foundations](#theoretical-foundations)

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │       Your Data Stream       │
                    │   value₁, value₂, value₃…   │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │       DetectorSuite          │
                    │   (parallel, no blending)    │
                    └──────────────┬───────────────┘
                                   │
          ┌────────┬───────┬───────┼───────┬────────┐
          ▼        ▼       ▼       ▼       ▼        ▼
       ┌──────┐ ┌──────┐ ┌─────┐ ┌────┐ ┌──────┐  ...
       │ Hopf │ │Discord│ │Drift│ │Var.│ │Coupl.│  FRM
       └──┬───┘ └──┬───┘ └──┬──┘ └─┬──┘ └──┬───┘  detectors
          │        │        │      │       │
          ▼        ▼        ▼      ▼       ▼
      DetectorResult  (one per detector per timestep)
       ├── detector: str
       ├── status: ScopeStatus  (NORMAL | ALERT | OUT_OF_SCOPE | WARMUP)
       ├── score: float         (0.0 – 1.0)
       ├── message: str
       └── step: int
```

**Key design decisions:**

- **No blending.** Each detector produces its own score independently.
- **No false consensus.** If 4/5 detectors say OUT_OF_SCOPE and one says ALERT, that one alert is the signal — not a 1/5 vote.
- **Scope awareness.** Every detector knows when its model doesn't apply and says so explicitly.
- **Sliding window.** All detectors operate on a rolling window of recent observations.

---

## Core Concepts

### ScopeStatus

Every result carries exactly one status:

| Status | Meaning | Score meaningful? |
|--------|---------|-------------------|
| `NORMAL` | Detector's model applies, no anomaly | Yes |
| `ALERT` | Detector's model applies, anomaly detected | Yes (≥ threshold) |
| `OUT_OF_SCOPE` | Data doesn't fit this detector's model | No (always 0.0) |
| `WARMUP` | Still collecting baseline data | No (always 0.0) |

### Score

A float in [0.0, 1.0]. Scores at or above the detector's `_alert_threshold` (default 0.5) trigger `ALERT` status. Scores below that threshold produce `NORMAL` status.

The score is **only meaningful** when status is `NORMAL` or `ALERT`. When status is `OUT_OF_SCOPE` or `WARMUP`, the score is fixed at 0.0 and should be ignored.

### BaseDetector Contract

Every detector implements two methods:

1. **`_check_scope(window)`** → `bool` — Does this detector's model apply to this data?
2. **`_compute(window)`** → `(score, message)` — Given in-scope data, what's the anomaly score?

The public API is a single method: **`update(value)`** → `DetectorResult`.

---

## Detector Reference

### HopfDetector — Pre-Transition Early Warning

**What it detects:** Systems approaching a Hopf bifurcation (or any fold/transcritical transition) via critical slowing down — rising variance and rising lag-1 autocorrelation.

**Theoretical basis:** Near a critical transition, the system loses its ability to recover from perturbations. This manifests as rising variance (noise amplified more) and rising AC(1) (recovery takes longer, memory builds). Both signals rise *before* the transition, not at it.

**Algorithm:**
1. Freeze variance and AC(1) baseline during warmup.
2. Compute current variance ratio (current / baseline) and AC(1) delta.
3. EWS score = 0.5 × variance_score + 0.5 × AC1_score.
4. Regime classification: stable → approaching → critical.

**Scope gates:**
- Baseline AC(1) < 0.10 → white noise → `OUT_OF_SCOPE`
- Mean shifted > 3.5σ from warmup → already jumped → `OUT_OF_SCOPE`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup` | 60 | Observations for baseline |
| `window` | 40 | Rolling window size |
| `ews_threshold` | 0.55 | Alert threshold |
| `ac1_min` | 0.10 | Minimum baseline AC(1) for scope |
| `mean_shift_z` | 3.5 | Mean shift z-score for out-of-scope |

**Best at:** Oscillatory or autocorrelated signals approaching a qualitative state change.
**Useless at:** Pure white noise (always `OUT_OF_SCOPE`).

---

### DiscordDetector — Point Anomalies

**What it detects:** Point and contextual anomalies via subsequence discord (DAMP-inspired).

**Theoretical basis:** A discord is a subsequence maximally dissimilar to all other subsequences. For each new subsequence, find its nearest neighbour in history. If that distance >> the distribution of past NN distances, it's a discord.

**Algorithm:**
1. Extract the most recent subsequence (z-normalised).
2. Compare against a random sample of historical subsequences.
3. Track EWMA distribution of nearest-neighbour distances.
4. Score = z-score of current NN distance / 3.0 (3σ → 1.0).

**Scope gates:**
- |linear_trend| > 0.08 → drifting data → `OUT_OF_SCOPE`
- Variance grown > 4× warmup → volatility shift → `OUT_OF_SCOPE`
- Insufficient history (< 2 × subseq_len) → `OUT_OF_SCOPE`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup` | 80 | Observations for baseline |
| `window` | 200 | Rolling window size |
| `subseq_len` | 20 | Subsequence length |
| `n_candidates` | 30 | Random historical comparisons |
| `discord_threshold` | 0.60 | Alert threshold |

**Best at:** Sharp point anomalies, sudden contextual anomalies.
**Useless at:** Slow variance changes.

---

### DriftDetector — Slow Distribution Shifts

**What it detects:** Slow monotonic mean shifts via frozen-baseline CUSUM.

**Theoretical basis:** An EWMA baseline adapts to slow drift, masking it. A frozen (warmup-frozen) baseline does not adapt — slow drift accumulates in the CUSUM statistic and eventually crosses the threshold.

**Algorithm:**
1. Freeze mean/std baseline during warmup.
2. Compute z_raw = (x − warmup_mean) / warmup_std.
3. Bidirectional CUSUM: s_hi = max(0, s_hi + z_raw − k), s_lo = max(0, s_lo − z_raw − k).
4. Alert on threshold crossing; reset CUSUM after crossing.

**Design note:** Page-Hinkley is excluded by design. With a frozen baseline, PH has E[increment] = −δ < 0, causing monotonic accumulation and ~65% false positive rate on stationary data.

**Scope gates:**
- Single-step spike |z_raw| > 5.0 → point anomaly → `OUT_OF_SCOPE`
- 3-step cooldown after spike → `OUT_OF_SCOPE`
- |AC(1)| > 0.35 → oscillatory signal → `OUT_OF_SCOPE`
- Variance grown > 4× → `OUT_OF_SCOPE`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup` | 100 | Observations for frozen baseline |
| `window` | 300 | Rolling window size |
| `cusum_k` | 0.5 | CUSUM allowance (half min detectable shift) |
| `cusum_h` | 8.0 | CUSUM threshold |
| `spike_z` | 5.0 | Spike detection threshold |
| `drift_threshold` | 0.50 | Alert threshold |

**Best at:** Slow monotonic mean shifts (0.5–2σ over 50–500 steps).
**Useless at:** Point anomalies (CUSUM resets on single spikes).

---

### VarianceDetector — Sudden Volatility Changes

**What it detects:** Sudden volatility explosions and regime switches in noise level.

**Theoretical basis:** Variance changes are invisible to mean-tracking statistics. A CUSUM on z² (squared standardised residuals) accumulates when variance has shifted. A single spike raises z² once; a sustained variance explosion raises z² persistently.

**Algorithm:**
1. Freeze variance baseline during warmup.
2. z_raw = (x − warmup_mean) / warmup_std; v² = z_raw².
3. One-sided CUSUM: s_hi = max(0, s_hi + v² − k). Detects variance *increase* only.
4. Complementary sustained-variance check: windowed variance / warmup variance.
5. Score = max(CUSUM score, sustained score).

**Design note:** CUSUM k = 1.848 (derived from log-likelihood ratio for 4× variance detection). Threshold h = 13.5 (calibrated for ~0.43% FPR on N(0,1) null).

**Scope gates:**
- |linear_trend(z_raw)| > 0.06 → mean is drifting → `OUT_OF_SCOPE`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup` | 80 | Observations for baseline |
| `window` | 200 | Rolling window size |
| `var_cusum_k` | 1.848 | CUSUM reference (LLR-derived) |
| `var_cusum_h` | 13.5 | CUSUM threshold |
| `sustained_ratio` | 4.0 | Sustained variance alert ratio |
| `variance_threshold` | 0.50 | Alert threshold |

**Best at:** Sudden volatility explosions, noise-level regime switches.
**Useless at:** Mean shifts without variance change.

---

### CouplingDetector — Cross-Frequency Decoupling

**What it detects:** Loss of cross-scale coordination before system collapse, via phase-amplitude coupling (PAC) degradation.

**Theoretical basis:** In a healthy oscillatory system, lower-frequency bands modulate higher-frequency amplitudes (PAC). This cross-scale coordination degrades before transitions — the heterodyned information channel degrades first.

**Algorithm:**
1. FFT-decompose the window into 5 frequency bands (ultra-low, low, mid, high, ultra-high).
2. Compute PAC strength for adjacent band pairs (low↔mid, mid↔high).
3. Track composite coupling score via EWMA.
4. Score based on coupling drop from baseline and/or declining trend.

**Scope gates:**
- Ultra-high power > 70% of total → pure noise → `OUT_OF_SCOPE`
- Total power < 1e-6 → flat signal → `OUT_OF_SCOPE`
- Max band power / total < 0.40 → no dominant frequency → `OUT_OF_SCOPE`
- PAC bands (low+mid+high) < 30% of total → insufficient coupling bands → `OUT_OF_SCOPE`
- Window < 64 samples → insufficient FFT resolution → `OUT_OF_SCOPE`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup` | 120 | Observations for baseline |
| `window` | 128 | FFT window size |
| `noise_floor_threshold` | 0.70 | Ultra-high fraction for out-of-scope |
| `coupling_drop_threshold` | 0.55 | Coupling ratio for alert |
| `trend_threshold` | -0.05 | Coupling trend threshold |
| `coupling_threshold` | 0.50 | Alert threshold |

**Best at:** Oscillatory systems losing cross-scale coordination (neural signals, power grids, physiological rhythms).
**Useless at:** White noise, non-oscillatory signals.

---

### LambdaDetector — Bifurcation Proximity (FRM)

**What it detects:** How close the system is to a Hopf bifurcation, by fitting the Fractal Rhythm Model parametric form and tracking the decay rate λ toward zero.

**Theoretical basis (unique to Fracttalix):** The FRM form is:

```
f(t) = B + A · exp(-λt) · cos(ωt + φ)
```

where ω = π / (2 · τ_gen) is fixed by the FRM quarter-wave theorem. As λ → 0, the system loses damping and approaches critical transition. No other detection system tracks this directly — generic EWS (Scheffer et al.) watches statistical shadows (variance, AC1); Lambda watches the dynamics themselves.

**Algorithm:**
1. Fix ω from τ_gen (or estimate via FFT if τ_gen unknown).
2. Multi-start nonlinear least-squares fit (scipy.optimize.curve_fit) with 3 initial guesses.
3. Compute R² to assess fit quality (scope awareness).
4. Track λ history; compute dλ/dt via MAD-robust linear regression.
5. Time-to-bifurcation estimate: Δt = λ / |dλ/dt| × fit_interval.

**Scope states:**
- R² < 0.5 → FRM form doesn't fit → `OUT_OF_SCOPE`
- λ × window_length < 0.5 → sustained oscillation (limit cycle) → reported as `LIMIT_CYCLE`
- R² ∈ [0.5, 0.7) → `BOUNDARY` (marginal fit)
- R² ≥ 0.7 → `IN_SCOPE`

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `current_lambda` | `float or None` | Most recent fitted λ value |
| `lambda_rate` | `float` | dλ/dt trend (negative = approaching bifurcation) |
| `time_to_transition` | `float or None` | Estimated steps until λ reaches zero |
| `r_squared` | `float` | Goodness-of-fit of the FRM form |
| `scope_status` | `str` | Current scope state |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_gen` | 0.0 | Generation timescale (0 = estimate from FFT) |
| `fit_window` | 128 | Fitting window size |
| `fit_interval` | 4 | Fit every N steps (performance) |
| `lambda_warning` | 0.05 | λ threshold for critical slowing |
| `r_squared_min` | 0.5 | Minimum R² for in-scope |

**Requires:** `scipy` (for `curve_fit` and `hilbert`).

**Best at:** Oscillatory systems with known generation timescale approaching a Hopf bifurcation.
**Useless at:** Non-oscillatory signals, pure noise.

---

### OmegaDetector — Timescale Integrity (FRM)

**What it detects:** Whether the observed oscillation frequency matches the physics-predicted value ω = π / (2 · τ_gen).

**Theoretical basis (unique to Fracttalix):** The FRM derives an absolute frequency reference from the system's generation timescale. Every other frequency-change detector (BOCPD, spectral CUSUM, wavelet decomposition) can only tell you "frequency changed." Omega can tell you "frequency is *wrong*" — a structural integrity violation.

**Two modes:**

| Mode | Condition | Behaviour |
|------|-----------|-----------|
| **Strong** | `tau_gen > 0` | Alerts when observed ω deviates from the physics-predicted value |
| **Weak** | `tau_gen = 0` | Estimates baseline ω from data, alerts when ω shifts |

**Algorithm:**
1. FFT with Hann window on the sliding data window.
2. Parabolic interpolation around the peak bin for sub-bin frequency accuracy.
3. Spectral SNR check (peak / mean; if below threshold → out of scope).
4. Deviation = |ω_observed − ω_reference| / ω_reference.
5. Score based on fractional deviation exceeding threshold.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `current_omega` | `float or None` | Most recent estimated ω |
| `omega_predicted` | `float or None` | Physics-predicted ω (None in weak mode) |
| `omega_deviation` | `float` | Fractional deviation from reference |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_gen` | 0.0 | Generation timescale (0 = weak mode) |
| `fit_window` | 128 | FFT window size |
| `deviation_threshold` | 0.15 | Fractional deviation for alert (15%) |
| `min_spectral_snr` | 3.0 | Minimum spectral peak / mean ratio |

**Requires:** `numpy` (no scipy needed).

**Best at:** Monitoring oscillation frequency against a known physics reference.
**Useless at:** Signals with no dominant frequency.

---

### VirtuDetector — Decision Rationality (FRM)

**What it detects:** When the optimal decision window is closing — not just *whether* something changed, but *when you should act*.

**Theoretical basis (unique to Fracttalix):** Based on Kramers scaling: σ_τ ~ (μ_c − μ)^(−1/2). As the system approaches bifurcation, the uncertainty in timing diverges. Early action has lower uncertainty but may be premature. Late action has higher urgency but worse odds.

**The Virtu Window:** The interval where:
- λ is declining with confidence.
- Time-to-transition is within the decision horizon.
- Timing uncertainty hasn't yet diverged past the action threshold.

**Four phases:**

| Phase | Decision Quality | Meaning |
|-------|-----------------|---------|
| `WAIT` | < 0.2 | Far from bifurcation, no action needed |
| `MONITOR` | 0.2 – 0.4 | Getting closer, watch carefully |
| `ACT SOON` | 0.4 – 0.7 | Decision window narrowing |
| `ACT NOW` | ≥ 0.7 | Uncertainty about to overwhelm — act immediately |

**Algorithm:**
1. Read λ, dλ/dt, and time-to-transition from the attached LambdaDetector.
2. Kramers scaling: timing_uncertainty = 1 / √λ.
3. Urgency = |dλ/dt| if λ is declining.
4. Decision quality = urgency × time_pressure / (1 + uncertainty).
5. Track Virtu Window state (opens at quality > 0.3, closes at < 0.1).

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `decision_quality` | `float` | Current decision quality (0.0 to 1.0) |
| `virtu_window_open` | `bool` | Whether the Virtu Window is currently open |
| `peak_quality` | `float` | Highest decision quality observed during current window |

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_detector` | None | LambdaDetector to read from (required) |
| `decision_horizon` | 100.0 | Max time-to-transition for relevance |
| `urgency_floor` | 0.001 | Minimum |dλ/dt| for urgency to register |
| `window_size` | 128 | Rolling window size |

**Requires:** An active `LambdaDetector` instance (which requires `scipy`).

**Best at:** Telling you *when* to make a decision about a system approaching bifurcation.
**Useless at:** Systems not approaching bifurcation (will stay in WAIT).

---

## Quick Start

### Using the DetectorSuite (5 Core Detectors)

```python
from fracttalix.suite import DetectorSuite

suite = DetectorSuite()

for value in your_data_stream:
    result = suite.update(value)
    print(result.summary())
    # Output: Hopf:ok(0.12) | Disc:OOS | Drif:ok(0.03) | Vari:ok(0.08) | Coup:WARM

    if result.any_alert:
        for alert in result.alerts:
            print(f"  ALERT from {alert.detector}: {alert.message}")
```

### Using Individual Detectors

```python
from fracttalix.suite import HopfDetector, DriftDetector

hopf = HopfDetector(warmup=60, window=40)
drift = DriftDetector(warmup=100)

for value in your_data_stream:
    h = hopf.update(value)
    d = drift.update(value)

    if h.is_alert:
        print(f"Hopf warning: {h.message}")
    if d.is_alert:
        print(f"Drift detected: {d.message}")
```

### Using FRM Detectors

```python
from fracttalix.suite import LambdaDetector, OmegaDetector, VirtuDetector

# tau_gen = your system's known generation timescale
lam = LambdaDetector(tau_gen=20.0, fit_window=128)
omega = OmegaDetector(tau_gen=20.0, fit_window=128)
virtu = VirtuDetector(lambda_detector=lam, decision_horizon=100.0)

for value in your_data_stream:
    lam_result = lam.update(value)
    omega_result = omega.update(value)
    virtu_result = virtu.update(value)

    # Lambda: how close to bifurcation?
    if lam.current_lambda is not None:
        print(f"lambda={lam.current_lambda:.4f}, rate={lam.lambda_rate:.4f}")
        if lam.time_to_transition is not None:
            print(f"  Estimated steps to transition: {lam.time_to_transition:.0f}")

    # Omega: is the frequency structurally correct?
    if omega_result.is_alert:
        print(f"Frequency deviation: {omega.omega_deviation:.1%}")

    # Virtu: should we act?
    if virtu.virtu_window_open:
        print(f"Decision window OPEN — quality={virtu.decision_quality:.2f}")
    if virtu_result.is_alert:
        print(f"ACT NOW: {virtu_result.message}")
```

---

## Recommended Configurations

| Domain | Detectors | Notes |
|--------|-----------|-------|
| Power grid monitoring | Hopf + Variance + Lambda | Hopf for early warning, Variance for volatility, Lambda for time-to-transition |
| API latency monitoring | Discord + Drift | Discord for spikes, Drift for SLA degradation |
| Neural/physiological | Hopf + Coupling + Omega | Coupling for PAC degradation, Omega for frequency integrity |
| Financial markets | Variance + Drift + Lambda | Volatility regime detection + trend detection + bifurcation proximity |
| "I don't know what I'm looking for" | DetectorSuite (all 5) | The dashboard approach — see what lights up |
| Full FRM analysis | Lambda + Omega + Virtu | Complete physics-aware monitoring with decision support |

---

## Scope Awareness

The single most important design decision in the suite is **scope awareness**. Every detector knows when its model doesn't apply.

### Why This Matters

Consider a white-noise signal. A traditional anomaly detector might report "anomaly score: 0.12" — which is meaningless because white noise has no structure to detect anomalies in. The suite's HopfDetector reports `OUT_OF_SCOPE` instead. This is more honest and more useful.

### How Each Detector Determines Scope

| Detector | In Scope When | Out of Scope When |
|----------|---------------|-------------------|
| Hopf | Baseline AC(1) ≥ 0.10, no mean shift | White noise, post-jump |
| Discord | No strong trend, variance stable | Drifting data, variance explosion |
| Drift | No spikes, no strong oscillation | Point anomalies, oscillatory signals |
| Variance | No mean drift | Mean is shifting |
| Coupling | Dominant frequency band, PAC bands have energy | White noise, flat spectrum |
| Lambda | R² ≥ 0.5 for FRM fit | Non-oscillatory, FRM doesn't fit |
| Omega | Spectral SNR ≥ 3.0, sufficient window | No dominant frequency |
| Virtu | Lambda detector in scope with valid λ | No Lambda detector, Lambda out of scope |

### Domain Separation

The scope gates create natural domain separation between detectors:

- **Spikes** → Discord handles them; Drift declares OUT_OF_SCOPE
- **Slow drift** → Drift handles it; Discord and Variance declare OUT_OF_SCOPE
- **Variance explosion** → Variance handles it; Discord declares OUT_OF_SCOPE
- **Oscillation loss** → Hopf handles it; Drift declares OUT_OF_SCOPE
- **Frequency shift** → Omega handles it; other detectors may or may not notice
- **Approaching bifurcation** → Lambda and Virtu handle it; Hopf provides complementary EWS

---

## Serialisation and State Management

All detectors support save/restore for deployment scenarios (e.g. restart recovery, distributed systems).

```python
import json

# Save state
state = detector.state_dict()
with open("detector_state.json", "w") as f:
    json.dump(state, f)

# Restore state
with open("detector_state.json") as f:
    state = json.load(f)
detector.load_state(state)

# Reset to factory state
detector.reset()
```

The `DetectorSuite` also supports serialisation:

```python
suite_state = suite.state_dict()   # dict with keys: hopf, discord, drift, variance, coupling
suite.load_state(suite_state)
```

---

## Dependencies

| Component | Required | Optional |
|-----------|----------|----------|
| Core detectors (Hopf, Discord, Drift, Variance) | Python 3.8+ | — |
| CouplingDetector | Python 3.8+ | NumPy (falls back to pure-Python DFT) |
| OmegaDetector | NumPy | — |
| LambdaDetector | NumPy, SciPy | — |
| VirtuDetector | (same as LambdaDetector) | — |

---

## Theoretical Foundations

### Critical Slowing Down (Hopf)

Near a bifurcation, the dominant eigenvalue of the linearised system approaches zero. This means perturbations decay more slowly → variance rises and autocorrelation rises. This is the standard Early Warning Signal (EWS) framework (Scheffer et al., 2009).

### Subsequence Discord (Discord)

Based on the DAMP algorithm (Yeh et al., 2022 VLDB). The nearest-neighbour distance of the current subsequence to historical subsequences follows a predictable distribution. Outliers in this distribution are discords — point or contextual anomalies.

### Frozen-Baseline CUSUM (Drift)

Page's CUSUM (1954) with a key insight: the baseline must be frozen at warmup to detect slow drift. An adaptive baseline masks slow drift by tracking it. The bidirectional formulation detects both upward and downward shifts.

### VarCUSUM (Variance)

CUSUM applied to squared standardised residuals z². Under the null hypothesis of constant variance, E[z²] = 1.0 with a χ²(1) distribution. The reference value k = 1.848 is derived from the log-likelihood ratio for detecting a 4× variance increase.

### Phase-Amplitude Coupling (Coupling)

In healthy oscillatory systems, low-frequency phase modulates high-frequency amplitude (Canolty et al., 2006). Degradation of this coupling indicates loss of cross-scale coordination and is an early indicator of system failure.

### Fractal Rhythm Model (Lambda, Omega, Virtu)

The FRM parametric form f(t) = B + A·exp(−λt)·cos(ωt + φ) describes damped oscillatory systems. The quarter-wave theorem ω = π/(2·τ_gen) provides an absolute frequency reference. As the system approaches Hopf bifurcation, λ → 0 (loss of damping). Kramers escape theory provides the scaling for timing uncertainty near the transition: σ_τ ~ 1/√λ.

These three quantities — λ (proximity), ω (integrity), and decision quality (rationality) — form a complete picture that no other detection framework provides.
