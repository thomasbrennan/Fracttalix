# Sentinel v14 Design — The FRM Detection Suite

## Status: DESIGN DRAFT — Not Approved

---

## From One Detector to Four

The v13 HopfDetectorStep answers one question: *Is the system's damping
disappearing?* It does this better than anyone else because it fits a
parametric model with derived constants that nobody else has.

But the FRM theorem set contains more than one exploitable result. Each
result answers a radically different question. Each question, answered
well, is something no existing detector can do — because each requires
the FRM functional form and its derived constants.

**The suite:**

| # | Detector | Question | Theorem Source |
|---|----------|----------|----------------|
| 1 | **Lambda** | Is damping disappearing? | D-1.3, F-1.5: lambda = \|alpha\|/(Gamma * tau_gen) |
| 2 | **Omega** | Has the fundamental rhythm changed? | D-1.2, F-1.4: omega = pi/(2 * tau_gen), T = 4 * tau_gen |
| 3 | **Virtu** | Is there still time to act? | MK-P5 Theorems 1-4: W_v, t_trap, Kramers scaling |
| 4 | **Scope** | Does the FRM model apply here at all? | D-2.1, C-3.REG, C-3.DIAG: universality class membership |

Each detector does exactly one thing. Each is self-aware about when it
doesn't apply. Each is derived from a theorem that nobody else has
published.

---

## Detector 1: Lambda (Bifurcation Proximity)

**Already built.** This is the v13 HopfDetectorStep.

### The Question
> Is the system's damping disappearing?

### The Theorem
From F-1.5 (perturbation expansion at Hopf criticality):

```
lambda = |alpha| / (Gamma * tau_gen)
Gamma = 1 + pi^2/4 ~ 3.467
```

When lambda -> 0, alpha -> 0, and the system reaches its Hopf
bifurcation. The damped oscillation stops decaying and begins growing.
The system transitions.

### What It Does
- Fits f(t) = B + A*exp(-lambda*t)*cos(omega*t + phi) to streaming data
- Tracks lambda over time via rolling OLS
- Estimates time-to-bifurcation: dt = lambda / |d_lambda/dt|
- Fires CRITICAL_SLOWING when lambda < threshold
- Fires TRANSITION_APPROACHING when dt < T_decision

### Why Best-in-World
Generic Early Warning Signals (Scheffer et al. 2009) watch for rising
variance and lag-1 autocorrelation — model-free statistical shadows of
the underlying dynamics. The lambda detector watches the dynamics
directly: the decay rate of a fitted parametric model with predicted
frequency. It gives:

- **Earlier warning** (parametric vs statistical)
- **Time estimate** (dt = lambda/|d_lambda/dt|, not available from EWS)
- **Scope awareness** (R^2 tells you when the model doesn't apply)

### Status
Implemented: `fracttalix/steps/hopf.py`. 12 tests passing.

---

## Detector 2: Omega (Timescale Integrity)

### The Question
> Has the system's fundamental rhythm changed?

### The Theorem
From F-1.4 (Hopf quarter-wave theorem):

```
omega = pi / (2 * tau_gen)
T = 4 * tau_gen
```

This is the FRM's strongest prediction: the oscillation period is
exactly four times the generation timescale. T = 4 * 6 hours = 24 hours
(circadian period from generation timescale, zero free parameters).

### What It Does
- Continuously estimates omega_observed from data (FFT dominant peak +
  FRM fit refinement)
- Compares to omega_predicted = pi/(2*tau_gen)
- Tracks the ratio omega_observed / omega_predicted over time
- When the ratio drifts from 1.0, tau_gen itself is changing

### What This Detects
**Timescale regime change.** Not a parameter shift within the same
dynamics — a change in the dynamics themselves. The generation timescale
is the most fundamental property of the substrate. When it changes, the
system has become a different system.

Examples of what this catches that Lambda cannot:
- Lambda watches the *amplitude envelope* decay toward zero
- Omega watches the *frequency* drift away from prediction
- A system can approach bifurcation (lambda -> 0) at constant frequency
  (omega stable) — Lambda catches this, Omega doesn't alert
- A system can shift its fundamental timescale (omega drifts) while
  remaining well-damped (lambda stable) — Omega catches this, Lambda
  doesn't alert

These are orthogonal failure modes. You need both.

### Why Best-in-World
Every existing frequency-change detector (BOCPD, spectral CUSUM,
wavelet decomposition) detects change *relative to the data's own
history*. They answer: "Has the frequency changed compared to what it
was?"

The Omega detector answers a different question: "Has the frequency
changed compared to what the theorem predicts it should be?" This is
possible because ω is *derived*, not fitted. We have a theoretical
prediction to compare against. Nobody else does.

Consequences:
- **Absolute reference**: Detects slow drift that relative methods miss
  (the "boiling frog" — if tau_gen shifts 1% per day, relative detectors
  adapt and lose the baseline; Omega compares against the fixed
  prediction)
- **Immediate scope**: On the first observation window, Omega knows
  whether the data matches the predicted frequency. No burn-in period.
- **Direction of drift**: Omega reports whether tau_gen is increasing
  (system slowing down) or decreasing (system speeding up), with
  physical meaning attached to each direction

### Architecture

```
OmegaDetectorStep(DetectorStep):

    Every N observations:

    1. ESTIMATE omega_observed
       - FFT of sliding window
       - Refine with FRM model fit (reuse Lambda fit if available)

    2. COMPARE to prediction
       omega_ratio = omega_observed / omega_predicted
       tau_gen_implied = pi / (2 * omega_observed)

    3. TRACK drift
       Maintain rolling window of omega_ratio values
       Compute d(omega_ratio)/dt via OLS

    4. ALERT
       |omega_ratio - 1.0| > tolerance → RHYTHM_SHIFT
       d(omega_ratio)/dt significant → RHYTHM_DRIFT
       omega_ratio stable at 1.0 → RHYTHM_LOCKED (healthy)
```

### Result API

```python
result.get_omega_status() -> dict
# {
#     "omega_observed": 0.265,       # measured characteristic frequency
#     "omega_predicted": 0.262,      # pi/(2*tau_gen)
#     "omega_ratio": 1.011,          # observed/predicted (1.0 = perfect)
#     "tau_gen_implied": 5.93,       # what the data says tau_gen is
#     "drift_rate": -0.0003,         # d(ratio)/dt
#     "scope_status": "IN_SCOPE",
#     "alert": False,
#     "alert_type": None,            # RHYTHM_SHIFT / RHYTHM_DRIFT / None
# }
```

### Config

```python
enable_omega_detector: bool = False
omega_fit_window: int = 256          # wider window for frequency resolution
omega_fit_interval: int = 8          # less frequent than Lambda
omega_drift_window: int = 30         # rolling ratio history
omega_ratio_tolerance: float = 0.10  # |ratio - 1| > this → alert
omega_drift_threshold: float = 0.005 # d(ratio)/dt > this → drift alert
```

### Validation: 3 Tests

**Test 1: Frequency Recovery.** Generate signal with known tau_gen.
Supply tau_gen. omega_ratio should be ~1.0.

**Test 2: Drift Detection.** Linearly shift tau_gen over 1000 steps.
RHYTHM_DRIFT should fire.

**Test 3: No False Alarm on Stable Rhythm.** Constant tau_gen for 500
steps. No alerts.

---

## Detector 3: Virtu (Decision Rationality)

### The Question
> Is there still time to act?

### The Theorems
From MK-P5:

**Theorem 1 (Window Rationality):** Intervention is rational iff

```
E[W_v(t)] > T_decision * (1 + C_fp / C_late)
```

**Theorem 2 (Asymmetric Loss):** Optimal detection threshold:

```
delta_c* = C_late / (C_late + C_fp)
```

**Theorem 4 (Self-Generated Friction):** There exists t_trap before
tipping such that the rationality condition fails regardless of cost
structure:

```
sigma_tau ~ (mu_c - mu(t))^(-1/2)     [Kramers scaling]
CV_tau → infinity as mu(t) → mu_c
```

### What It Does
- Takes the Lambda detector's output (lambda, d_lambda/dt, time_to_bifurcation)
  as input
- Computes the Virtu Window: W_v(t) = expected remaining actionable time
- Computes t_trap: the point beyond which rational action is impossible
- Reports whether the decision window is OPEN, CLOSING, or CLOSED

### What This Detects
**Decision deadline.** Not "something bad is coming" (that's Lambda's
job). Not "the frequency has changed" (that's Omega's job). But: *given
that Lambda says a transition is approaching, do you still have time to
do something about it, and when does that window close?*

This is a fundamentally different function. Lambda is physics. Virtu is
decision theory applied to physics.

### Why Best-in-World
No existing detector incorporates decision theory. They all output the
same thing: "alert!" or "no alert." They don't tell you:

- Whether acting now is rational given your cost structure
- When acting will become irrational (t_trap)
- How your false-positive tolerance should change as the transition
  approaches (the asymmetric loss theorem)

The Kramers scaling result (Theorem 4) is particularly powerful: as the
system approaches tipping, the *uncertainty* in tipping time diverges.
This means the closer you get, the harder it is to justify action — the
system generates its own friction against rational intervention. No
generic detector tells you this. The Virtu detector tells you exactly
when the friction wins.

### Architecture

```
VirtuDetectorStep(DetectorStep):

    Requires: Lambda detector output (lambda, d_lambda/dt)

    Every N observations:

    1. ESTIMATE tipping distribution
       mu_tau = time_to_bifurcation from Lambda
       sigma_tau = mu_tau * CV_correction(lambda)
         where CV grows as lambda → 0 via Kramers scaling

    2. COMPUTE Virtu Window
       W_v = mu_tau - sigma_tau * z_alpha
       (conservative estimate of remaining time)

    3. CHECK rationality condition
       W_v > T_decision * (1 + C_fp / C_late) → OPEN
       W_v > T_decision but falling → CLOSING
       W_v <= T_decision → CLOSED (past t_trap)

    4. COMPUTE adaptive threshold
       delta_c = C_late / (C_late + C_fp)
       Apply to Lambda alerts: suppress if P(transition) < delta_c

    5. REPORT
       window_status: OPEN / CLOSING / CLOSED / NOT_APPLICABLE
       time_remaining: W_v in steps
       t_trap_estimate: when window will close
       recommended_action: ACT_NOW / MONITOR / NOT_IN_SCOPE
```

### Result API

```python
result.get_virtu_status() -> dict
# {
#     "window_status": "CLOSING",       # OPEN / CLOSING / CLOSED
#     "virtu_window": 23.5,             # W_v in steps
#     "t_trap_estimate": 47.0,          # steps until window closes
#     "rationality_ratio": 2.35,        # W_v / T_decision (>1 = rational to act)
#     "adaptive_threshold": 0.33,       # delta_c* for current cost structure
#     "recommended_action": "MONITOR",  # ACT_NOW / MONITOR / NOT_IN_SCOPE
#     "uncertainty_cv": 0.45,           # CV of tipping time estimate
#     "scope_status": "IN_SCOPE",
# }
```

### Config

```python
enable_virtu_detector: bool = False
virtu_t_decision: float = 10.0       # minimum intervention lead time
virtu_c_fp: float = 1.0              # cost of false positive
virtu_c_late: float = 1.0            # cost of late/missed transition
virtu_kramers_exponent: float = 0.5  # Kramers scaling exponent
virtu_z_alpha: float = 1.645         # confidence level for W_v (90%)
```

### Validation: 3 Tests

**Test 1: Window Opens and Closes.** Lambda declining → W_v should
transition from OPEN to CLOSING to CLOSED.

**Test 2: Asymmetric Cost.** C_late >> C_fp should produce earlier
ACT_NOW recommendations than symmetric costs.

**Test 3: Stable System.** Constant Lambda → window_status = NOT_APPLICABLE.

---

## Detector 4: Scope (Class Membership)

### The Question
> Does the FRM model apply to this data at all?

### The Theorems
From D-2.1 (universality class definition):

```
Three structural criteria (all necessary):
  (a) Delayed negative feedback with single dominant delay tau > 0
  (b) Characteristic equation with conjugate pair crossing imaginary
      axis at Hopf bifurcation
  (c) tau independently measurable (not inferred from oscillation period)
```

From C-3.REG (measurement protocol):

```
Classification:
  R^2 >= 0.85 → CONFIRMED
  R^2 < 0.85 but fit converges → ANOMALOUS
  Fit fails or R^2 < threshold → EXCLUDED
```

From C-3.DIAG (scope boundary diagnostics):

```
Classification is fully computable (no human judgment required)
```

### What It Does
- Continuously fits the FRM functional form to streaming data
- Monitors R^2, residual structure, and fit stability
- Classifies the data stream into: CONFIRMED / ANOMALOUS / EXCLUDED
- Reports *why* the classification was made (which criterion failed)

### What This Detects
**Model applicability.** This is the anti-detector. It doesn't detect
threats — it detects whether the other three detectors should be
trusted.

When Scope says EXCLUDED, it means: "The FRM functional form does not
fit this data. Lambda, Omega, and Virtu outputs are meaningless. Use a
different tool."

When Scope says ANOMALOUS, it means: "The FRM fits poorly but not
catastrophically. Results are uncertain. Investigate."

When Scope says CONFIRMED, it means: "The FRM model applies. Trust the
other detectors."

### Why Best-in-World
Every other detector is a black box that always outputs a number. Rising
variance? Here's an EWS score. Random noise? Here's an EWS score. The
EWS doesn't know — and can't tell you — when it's producing garbage.

The Scope detector exploits the FRM's parametric form to provide a
principled goodness-of-fit test. The R^2 of the FRM fit is not just a
quality metric — it's a *scope boundary*. When R^2 drops, it's not that
the detector failed; it's that the phenomenon the detector is designed
for isn't present. The detector is working correctly by telling you it
shouldn't be running.

This is possible because the FRM is a specific, falsifiable functional
form. Generic methods (variance, autocorrelation, entropy) can't do
scope detection because they don't have a model to compare against.

### Architecture

```
ScopeDetectorStep(DetectorStep):

    Every N observations:

    1. FIT FRM model
       f(t) = B + A*exp(-lambda*t)*cos(omega*t + phi)
       (Reuse Lambda detector's fit results when available)

    2. COMPUTE fit quality
       R^2 from residuals
       Residual autocorrelation (should be white if model is correct)
       Amplitude significance (|A| / noise_floor)

    3. CLASSIFY
       R^2 >= 0.85 AND residuals white AND amplitude significant
         → CONFIRMED
       R^2 >= r_squared_min AND fit converges
         → ANOMALOUS
       Otherwise
         → EXCLUDED

    4. DIAGNOSE (when not CONFIRMED)
       Report which criterion failed:
         - "LOW_R_SQUARED": model doesn't fit
         - "STRUCTURED_RESIDUALS": model misses systematic pattern
         - "LOW_AMPLITUDE": signal too weak relative to noise
         - "FIT_DIVERGED": optimizer didn't converge
         - "NO_OSCILLATION": data lacks oscillatory structure

    5. GATE other detectors
       When EXCLUDED: suppress Lambda, Omega, Virtu alerts
```

### Result API

```python
result.get_scope_status() -> dict
# {
#     "classification": "CONFIRMED",    # CONFIRMED / ANOMALOUS / EXCLUDED
#     "r_squared": 0.91,
#     "residual_autocorrelation": 0.08, # should be < 0.2
#     "amplitude_snr": 4.2,            # A / noise_floor
#     "diagnosis": None,               # or "LOW_R_SQUARED", etc.
#     "gate_open": True,               # should other detectors' alerts propagate?
#     "confidence": "HIGH",
# }
```

### Config

```python
enable_scope_detector: bool = False
scope_r_squared_confirmed: float = 0.85  # above = CONFIRMED
scope_r_squared_min: float = 0.50        # below = EXCLUDED
scope_residual_ac_max: float = 0.20      # above = structured residuals
scope_amplitude_snr_min: float = 2.0     # below = signal too weak
scope_check_interval: int = 8            # reuse Lambda fits where possible
```

### Validation: 4 Tests

**Test 1: Clean FRM Signal → CONFIRMED.**

**Test 2: White Noise → EXCLUDED with diagnosis "NO_OSCILLATION".**

**Test 3: Linear Trend → EXCLUDED with diagnosis "LOW_R_SQUARED".**

**Test 4: Gate Suppression.** When EXCLUDED, Lambda alerts must not
propagate.

---

## Suite Interactions

The four detectors are designed to work together but operate
independently. No detector depends on another for its core computation.
However, they share information when available:

```
                   ┌─────────┐
                   │  Scope  │  "Does FRM apply?"
                   │ (gate)  │
                   └────┬────┘
                        │ gates alerts from:
            ┌───────────┼───────────┐
            │           │           │
       ┌────▼───┐  ┌────▼───┐  ┌───▼────┐
       │ Lambda │  │ Omega  │  │ Virtu  │
       │ (prox) │  │ (freq) │  │ (time) │
       └────────┘  └────┬───┘  └───▲────┘
                        │          │
                        │    lambda, d_lambda/dt
                        │          │
                        └──────────┘
                      (Virtu reads Lambda)
```

**Data flow:**
- Scope gates all three detectors (EXCLUDED → suppress alerts)
- Virtu reads Lambda's output (needs lambda trajectory to compute W_v)
- Lambda and Omega share the FRM fit (avoid duplicate computation)
- Each detector maintains its own state and history

**Pipeline position:**
All four run after the existing v12.2 pipeline steps, before
AlertReasonsStep. They are opt-in (disabled by default, require scipy).

---

## What Each Detector Is Best At (Summary)

| Detector | Best at | Competitive advantage | What it can't do |
|----------|---------|----------------------|-----------------|
| Lambda | Predicting *that* a transition is coming | Parametric model with derived constants (not fitted) | Doesn't know if you can act in time |
| Omega | Detecting timescale regime change | Compares against theoretical prediction (not historical baseline) | Doesn't detect transitions at constant frequency |
| Virtu | Computing *when* to act | Decision theory + Kramers scaling (no other detector does this) | Needs Lambda output; can't function alone |
| Scope | Knowing when to shut up | Self-aware scope boundary from R^2 of falsifiable model | Only detects model applicability, not threats |

Together they answer the complete question: *Does the model apply?
(Scope) If so, is a transition coming? (Lambda) Has the rhythm itself
changed? (Omega) And is there still time to do something about it?
(Virtu)*

No existing detection system answers all four. Most answer only the
second, and poorly.

---

## What the Suite Replaces

The suite does NOT replace the v12.2 38-step pipeline. That pipeline
does general-purpose streaming anomaly detection (point anomalies,
distribution shift, variance change, etc.). It's competent at many
things.

The suite does something the pipeline cannot: exploit the FRM theorem
to answer questions that require a parametric model with derived
constants. The suite and the pipeline can coexist, with the suite
providing FRM-specific detection alongside the pipeline's general
capabilities.

If the suite validates — if it demonstrably outperforms generic EWS at
transition detection, frequency monitoring, and decision timing — then
the pipeline's role shifts to "everything the FRM doesn't cover" (point
anomalies, distribution shift, etc.), and the suite handles the
FRM-native questions.

---

## Implementation Order

1. **Lambda** — Done (v13, HopfDetectorStep)
2. **Scope** — Next. Partially implemented inside Lambda (R^2 monitoring).
   Extract and strengthen into standalone step with residual diagnostics.
3. **Omega** — After Scope. Requires FFT refinement and ratio tracking.
   Shares fit infrastructure with Lambda.
4. **Virtu** — Last. Requires Lambda to be mature and validated. This is
   the most novel and the hardest to validate.

Each detector ships independently. Each has its own test suite. Each
is opt-in. No detector's absence breaks the others.

---

## Estimated Sizes

| Detector | Lines of code | Test count | Dependencies |
|----------|---------------|------------|--------------|
| Lambda | ~280 (done) | 12 (done) | scipy |
| Scope | ~150 | 6 | scipy (reuses Lambda fit) |
| Omega | ~200 | 6 | scipy, numpy |
| Virtu | ~250 | 8 | numpy (reads Lambda output) |

Total new code: ~600 lines. Total new tests: ~20.

---

## The Thesis

Every detector in the suite is derived from the same theorem set:

```
f(t) = B + A * exp(-lambda * t) * cos(omega * t + phi)
omega = pi / (2 * tau_gen)
lambda = |alpha| / (Gamma * tau_gen)
Gamma = 1 + pi^2/4
```

Four detectors. Four radically different functions. One theorem.

Nobody else has the theorem. Nobody else can build the suite.
