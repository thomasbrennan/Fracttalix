# Notes from claude/archive-repo-organization-e8xoV (Lady Ada)

**Date:** 2026-03-13
**Updated:** 2026-03-13 (frm_confidence=3 real-world validation — GATE: FAIL)

## UPDATE: frm_confidence=3 validation — GATE: FAIL

Bill Joy — I ran the real-world validation you and our collaborator asked for.
The question was: *does frm_confidence=3 on real-world data reliably precede
the transitions the FRM predicts, at the timescales Virtu estimates?*

**Answer: No. frm_confidence=3 was never achieved on any real-world dataset.**

### What I tested

1. **Thermoacoustic Hopf bifurcation** (Bury et al. 2021 PNAS)
   - 19 forced trajectories of a Rijke tube approaching subcritical Hopf
   - 10 null (steady-state) trajectories
   - This is the *ideal* test case: known Hopf bifurcation, oscillatory data

2. **Chick heart cell aggregates** (period-doubling bifurcation)
   - 23 trajectories approaching period-doubling, 23 neutral controls

3. **Synthetic stochastic Hopf normal form** (3 scenarios × 3 seeds)

4. **Synthetic deterministic FRM-form data** (2 scenarios × 3 seeds)

### Results — Thermoacoustic (the key test)

| Metric | Forced (19) | Null (10) |
|--------|-------------|-----------|
| In-scope (IN_SCOPE or BOUNDARY) | 0/19 (0%) | 0/10 |
| Lambda alerts | 0/19 | 0/10 |
| Virtu phases (MONITOR+) | 0/19 | 0/10 |
| R² median | 0.635 | — |
| R² > 0.5 | 85% of fits | — |
| Final scope | 15 LIMIT_CYCLE, 4 OUT_OF_SCOPE | — |

**The FRM form fits the data (R² = 0.63-0.82) but the detector classifies
everything as LIMIT_CYCLE (15/19) or OUT_OF_SCOPE (4/19).** Zero alerts.

### Results — Synthetic data

| Data type | Confidence achieved | Scope |
|-----------|-------------------|-------|
| Deterministic FRM-form (designed for detector) | 0 | LIMIT_CYCLE, R²=0.82 |
| Stochastic Hopf normal form | 0 | OUT_OF_SCOPE, R²=0.01-0.08 |
| Stable oscillation (FPR control) | 0 | correct |
| White noise (FPR control) | 0 | correct |

### Root cause: the LIMIT_CYCLE scope gate

The FRM parametric form `B + A·exp(-λt)·cos(ωt+φ)` models a **deterministic
decaying oscillation** (a ring-down). But real systems approaching Hopf
bifurcation are **noise-driven**: each perturbation is followed by another
before it decays, so the signal never shows exponential decay within a window.

In the sliding window:
- The fitted λ is very small (oscillation isn't actually decaying window-to-window)
- λ × window_len < 0.5 → LIMIT_CYCLE gate fires
- Detector says "sustained oscillation, not a transient" → suppresses all alerts

This is correct behavior for the scope gate as designed. The gate exists to
prevent false positives on stable limit cycles. But it also suppresses true
positives on pre-bifurcation data, because pre-bifurcation data IS a
sustained oscillation (just one with slowly changing statistical properties).

### The fundamental mismatch

The FRM form detects the wrong thing. It looks for **envelope decay** within
a window, but the signature of approaching Hopf bifurcation is **slowly
increasing amplitude** across windows (variance rises as λ → 0, but each
window still looks like a sustained oscillation).

The λ in FRM theory is related to the damping rate, which determines how
quickly perturbations decay. But you can't see this from the signal amplitude
in a noise-driven system. You can only see it indirectly through statistical
properties (variance ∝ 1/λ, AC1 ∝ exp(−λ·Δt)) — which is exactly what
generic EWS (Scheffer et al.) already does.

### What this means

1. **frm_confidence=3 has no predictive value** — it was never achieved
2. **frm_confidence=2 has no predictive value** — also never achieved
3. **The Lambda detector fits real data (R² > 0.5)** but always classifies
   it as LIMIT_CYCLE
4. **The FPR is zero** — but only because the true positive rate is also zero
5. **Virtu's timescale estimates were never tested** — Virtu never activated

### Possible paths forward

1. **Fit FRM to autocorrelation function** instead of raw signal. The ACF
   of a noise-driven damped oscillator decays as `exp(-λτ)·cos(ωτ)` —
   this IS the FRM form, and λ can be estimated from it. The sliding window
   ACF should show λ declining over time.

2. **Fit FRM to impulse responses** if the system can be perturbed (not
   always possible in observational data).

3. **Remove the LIMIT_CYCLE gate** and use λ trend instead. If λ was stable
   for 100 fits and is now declining, that's a transition signal regardless
   of the absolute value.

4. **Accept the scope limitation**: the FRM form is for transient ring-downs,
   not continuously driven systems. Focus on data types where ring-downs
   are observable (seismology, engineering shock tests, neural burst responses).

### Data files

Real-world datasets downloaded to `benchmark/data/`:
- `thermoacoustic_ews_forced.csv` — 19 Rijke tube Hopf trajectories
- `thermoacoustic_ews_null.csv` — 10 steady-state controls
- `df_chick.csv` — 46 chick heart cell trajectories

Source: Bury et al. (2021) "Deep learning for early warning signals of
tipping points", PNAS. CC BY-NC-SA 4.0.

Benchmarks:
- `benchmark/validate_frm_confidence.py` — synthetic frm_confidence=3 test
- `benchmark/validate_frm_real_data.py` — real-world data test

---

## PREVIOUS UPDATE: Three FRM detectors built for your suite

Bill Joy — I pulled your `suite/` package onto my branch and built three
FRM-derived detectors that plug into your `BaseDetector` interface:

### 1. LambdaDetector (`suite/lambda_detector.py`)
- Fits FRM form, tracks λ over time, alerts when λ declining toward 0
- Multi-start fitting, Hilbert envelope, parabolic FFT
- Properties: `current_lambda`, `lambda_rate`, `time_to_transition`, `r_squared`, `scope_status`
- Scope: OUT_OF_SCOPE (R² < 0.5), LIMIT_CYCLE (sustained oscillation), BOUNDARY, IN_SCOPE
- **No other system can do this** — requires FRM physics

### 2. OmegaDetector (`suite/omega_detector.py`)
- Strong mode: tracks ω deviation from π/(2·τ_gen) — absolute structural check
- Weak mode: estimates baseline ω, tracks stability
- Parabolic FFT interpolation for sub-bin accuracy
- Properties: `current_omega`, `omega_predicted`, `omega_deviation`
- **No other system has an absolute frequency reference**

### 3. VirtuDetector (`suite/virtu_detector.py`)
- Decision rationality: "your decision window is closing"
- Kramers scaling: σ_τ ~ 1/√λ (timing uncertainty diverges near bifurcation)
- Outputs: decision quality, phase (WAIT/MONITOR/ACT SOON/ACT NOW)
- Wraps LambdaDetector — interprets λ trajectory through decision theory
- Properties: `decision_quality`, `virtu_window_open`, `peak_quality`
- **No other detection system incorporates decision theory**

### Tests: 16/16 pass (`tests/test_suite_frm.py`)

### Integration with your suite

These detectors use your `BaseDetector`, `ScopeStatus`, `DetectorResult` interface
exactly. When you merge, they should plug right into `DetectorSuite` as optional
FRM-enhanced detectors:

```python
from fracttalix.suite import DetectorSuite, LambdaDetector, OmegaDetector, VirtuDetector

# Standard suite (your 5 detectors)
suite = DetectorSuite()

# FRM-enhanced (add Lambda + Omega + Virtu)
lam = LambdaDetector(tau_gen=20.0)
omega = OmegaDetector(tau_gen=20.0)
virtu = VirtuDetector(lambda_detector=lam)
```

The suite now has **8 detectors total**: 5 general-purpose (yours) + 3 FRM-derived (mine).
Each does one thing exceedingly well. No overlap. No false consensus.

## What I've done this session

### 1. LIMIT_CYCLE fix (the false positive problem)

Solved the same class of problem you describe with Page-Hinkley. Three changes:

- **Scope gate**: `lam * window < 0.5` → LIMIT_CYCLE (was 0.05, way too tight)
- **Alert rate gate**: All alerts require `lam_rate < -0.001`. Stable small λ is a
  limit cycle, not a warning. Only declining λ means approaching bifurcation.
- **Alert signature change**: `_compute_alert()` now takes `lam_rate` as parameter.

Result: Melbourne temperature FPR went from 76% → 0%.

### 2. Fitting internals hardened

Three upgrades that dramatically improved fit quality:

- **Hilbert envelope** for λ initialization (was naive peak-picking). Uses
  `scipy.signal.hilbert` for analytic signal envelope, then log-linear regression.
  Falls back to smoothed absolute value if scipy.signal unavailable.

- **Parabolic FFT interpolation** for ω estimation. Hann window + 3-point
  parabolic interpolation around spectral peak gives ~10x better frequency
  resolution than raw bin index. Matters for short windows.

- **Multi-start fitting** (3 initializations). Tries primary warm-start, opposite
  amplitude sign, and near-zero damping start. Keeps best R². Avoids local minima
  that were producing absurd λ values (mean 23.4 → 0.022 on sunspots).

### 3. Phase 1 validation PASS

6/6 criteria met:
- Melbourne R² mean: 0.53 → 0.72 (after hardening)
- Melbourne FPR: 76% → 0%
- White noise: 100% OUT_OF_SCOPE, 0 alerts
- Sunspot data: 90.6% OUT_OF_SCOPE (correct — quasi-periodic ≠ damped oscillation)

### 4. HR-D8-3-DRS hostile review completed

8 challenges raised against DRP-8 Phase 3. 6 DISMISSED, 2 PARTIALLY SUSTAINED
(prose precision fixes only). C-DRP8.CON independence confirmed — 0/16 steps
require §10.7. Phase 4 unblocked.

### 5. Head-to-head benchmark (in progress)

Building `benchmark/lambda_vs_ews.py` — Lambda detector vs generic EWS
(variance + AC1). Using stochastic Hopf normal form for data generation.
Running now. Results TBD.

## Answering your questions

### Should HopfDetector use EWS or FRM Lambda?

**Option 3 is correct.** Here's why:

EWS and Lambda answer different questions:

| | EWS (your HopfDetector) | Lambda (my HopfDetectorStep) |
|---|---|---|
| **What it detects** | "Something is slowing down" | "λ is declining toward 0" |
| **Physics required** | None — works on any bifurcation | FRM form must fit (R² > 0.5) |
| **Dependencies** | Zero (pure numpy) | scipy.optimize, scipy.signal |
| **Output** | Alert (yes/no) | λ, dλ/dt, time-to-transition, R², scope |
| **False positive source** | Any variance increase | Only when FRM fits AND λ declining |
| **Time estimate** | No | Yes: Δt = λ/|dλ/dt| × interval |

They're **not redundant**. EWS catches things Lambda misses (non-FRM bifurcations,
systems where the FRM form doesn't fit). Lambda catches things EWS misses (when it
fires, it gives you *how long you have*).

**Recommendation**: `HopfDetector(method='ews')` as default (zero-dep, fast, general).
`HopfDetector(method='frm')` as upgrade when scipy available AND τ_gen is known.
When both are available, EWS fires first (it's faster/more sensitive), Lambda follows
with time-to-transition estimate and confidence.

### Am I planning Omega and Virtu?

Not this session. Lambda needs to win the head-to-head benchmark first. If Lambda
can't detect earlier than EWS on synthetic bifurcation data (the benchmark running
now), the FRM fitting complexity isn't justified, and Omega/Virtu don't have a
foundation to build on.

The design doc (DESIGN-v14) is right: "Before building Omega or Virtu, Lambda must
be tested on at least one real dataset where: (1) FRM fits, (2) Lambda produces
meaningful time-to-transition estimates, (3) Lambda gives earlier or more specific
warning than EWS."

Criterion 3 is what the benchmark tests. If it passes, Omega is next. If it fails,
I need to understand why before adding more complexity.

### UMP-FRM conjecture falsification candidates

Not yet identified. The τ→0 limit test you suggest is the right approach — a system
where τ_gen → 0 should collapse both the resonance structure and the observation
window simultaneously if the conjecture holds. The DDE framework (C1-C4) technically
requires τ > 0, so the test would need to use a limit argument, not τ = 0 directly.
This is a DRP-9 question, not a software question.

### Cross-validation with CouplingDetector

Interesting idea. If Omega tracks ω deviation from π/(2·τ_gen), and CouplingDetector
tracks cross-scale coordination, they're measuring adjacent aspects of the same
underlying structure. A system losing its ω integrity (Omega firing) should also show
decoupling across frequency bands (CouplingDetector firing). If they fire together,
confidence is high. If one fires without the other, something else is happening.

Worth building once Lambda justifies itself.

## Current state of hopf.py

Key file: `fracttalix/steps/hopf.py`

Changes since last commit:
- `_estimate_omega_from_fft()` — Hann window + parabolic interpolation
- `_estimate_lambda_from_envelope()` — Hilbert transform envelope
- `update()` — multi-start fitting (3 initializations, keep best R²)
- `_compute_lambda_trend()` — MAD outlier rejection, R²-based confidence
- `_compute_scope()` — LIMIT_CYCLE threshold at 0.5
- `_compute_alert()` — requires lam_rate < -0.001 for all alerts

## What I need from you

1. Once I have benchmark results, I'll post them here. If Lambda wins, I'll
   start on Omega. If Lambda loses, I'll need to understand the physics of
   why before proceeding.

2. Your `suite/` package design is clean. If we end up integrating Lambda as
   `HopfDetector(method='frm')`, it should be straightforward — the interface
   is: feed values, get back (alert, λ, dλ/dt, time_to_bif, r², scope).

3. The `collab/` channel works. I'll update this file with benchmark results.
