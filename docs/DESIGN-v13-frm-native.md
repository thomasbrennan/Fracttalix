# Sentinel v13 Design — FRM-Native Detector

## Status: DESIGN DRAFT — Not Approved

---

## Problem Statement

Sentinel v12.2 is a competent streaming anomaly detector built from standard
signal processing techniques (EWMA, CUSUM, FFT, Hurst, permutation entropy)
organized in an FRM-inspired architecture. But the FRM's core theoretical
predictions are never used:

- The functional form `f(t) = B + A·exp(-λt)·cos(ωt+φ)` is never fitted
- The characteristic frequency `ω = π/(2·τ_gen)` is never computed
- The decay rate `λ = |α|/(Γ·τ_gen)` is never estimated
- The Hopf bifurcation approach (λ → 0 as early warning) is absent
- The MK-P5 decision theory (Virtù Window, t_trap) is not implemented

v13 would add a parallel detection pathway that actually implements the FRM,
running alongside the existing v12.2 pipeline. The existing steps stay
unchanged — this is additive, not a replacement.

---

## Competitive Context

Current SOTA for streaming anomaly detection (2025-2026):

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| DAMP (Matrix Profile) | Best univariate F1 (0.28 avg), simple, fast | No frequency structure, no early warning |
| HS-Trees (River) | Constant memory, fast | Global anomalies only, no local detection |
| xStream | Handles evolving features | Same limitation as HS-Trees |
| Pi-Transformer | Phase synchrony inductive bias | Batch retraining, heavy compute |
| EWS (Scheffer) | Theoretically grounded pre-transition warning | Model-free, slow to respond, high FPR |

**Gap identified:** No published streaming detector uses cross-frequency
coupling breakdown as an anomaly signal. No detector fits a parametric
pre-bifurcation model for early warning. The FRM approach occupies a
genuinely open niche.

**Honest constraint:** This niche is narrow. FRM-native detection targets
systems approaching critical transitions, not general-purpose anomaly
detection. It would complement DAMP-style methods, not replace them.

---

## Design Principles

1. **Actually fit the model.** The core contribution is fitting
   `f(t) = B + A·exp(-λt)·cos(ωt+φ)` to windowed data and tracking
   parameter evolution. If we don't do this, we're just adding more
   standard techniques with FRM names.

2. **λ is the primary signal.** The FRM predicts that λ → 0 indicates
   approach to Hopf bifurcation (critical transition). This is more
   specific than generic EWS (rising variance + autocorrelation) and
   should provide earlier, more informative warning.

3. **ω grounds the spectral analysis.** Instead of 5 arbitrary frequency
   bands, use the fitted ω to focus analysis on the characteristic
   frequency and its harmonics.

4. **Additive, not replacement.** The v12.2 pipeline stays. The FRM
   pathway runs in parallel and produces its own result keys. Users
   can use either or both.

5. **Validate before shipping.** Every new step needs a synthetic test
   that demonstrates it detects what it claims to detect, and a
   negative test showing it doesn't fire on data it shouldn't.

---

## Architecture: FRM Pathway (New Steps)

### Overview

The FRM pathway adds 6 new steps after the existing 37. These steps
are independent of each other (except where noted) but depend on the
existing pipeline's frequency decomposition and phase extraction.

```
Existing Pipeline (Steps 1-37, unchanged)
    │
    ├── Step 38: FRMModelFitStep
    │       Fits f(t) = B + A·exp(-λt)·cos(ωt+φ) to windowed data
    │       Outputs: λ, ω, A, B, φ, R², residuals
    │
    ├── Step 39: DecayRateTrackingStep (depends on 38)
    │       Tracks λ over time, detects λ → 0 trend
    │       Outputs: λ_history, dλ/dt, bifurcation_proximity
    │
    ├── Step 40: CharacteristicFrequencyStep (depends on 38)
    │       Compares fitted ω to theoretical ω = π/(2·τ_gen)
    │       Adaptive band focusing around fitted ω
    │       Outputs: ω_fitted, ω_theoretical, frequency_drift
    │
    ├── Step 41: ModelQualityStep (depends on 38)
    │       Tracks R² and residual structure over time
    │       Detects when FRM form stops fitting (scope boundary)
    │       Outputs: r_squared, residual_autocorrelation, in_scope
    │
    ├── Step 42: VirtuWindowStep (depends on 39)
    │       MK-P5 decision theory: estimates time remaining for
    │       rational intervention before t_trap
    │       Outputs: virtu_window, t_trap_proximity, intervention_rational
    │
    └── Step 43: FRMAlertStep (depends on 38-42)
            Aggregates FRM pathway signals into alert decisions
            Outputs: frm_alert, frm_alert_reasons
```

---

## Step Specifications

### Step 38: FRMModelFitStep

**Purpose:** Fit the FRM functional form to a sliding window of data.

**Math:**
```
f(t) = B + A·exp(-λt)·cos(ωt + φ)

Parameters to estimate: [B, A, λ, ω, φ]
Fixed: none (all fitted from data)
```

**Algorithm:**
1. Maintain a sliding window of `frm_fit_window` observations (default: 128)
2. On each update, fit the 5-parameter model using nonlinear least squares
   (Levenberg-Marquardt if scipy available, else Nelder-Mead on stdlib)
3. Initialize ω from the dominant FFT peak (from Step 3)
4. Initialize λ from log-envelope decay rate
5. Constrain: λ ≥ 0 (damped only — FRM scope boundary), ω > 0

**Outputs to scratch:**
```python
{
    "frm_B": float,           # baseline
    "frm_A": float,           # amplitude
    "frm_lambda": float,      # decay rate (KEY SIGNAL)
    "frm_omega": float,       # characteristic frequency
    "frm_phi": float,         # phase
    "frm_r_squared": float,   # goodness of fit
    "frm_residuals": list,    # for residual analysis
    "frm_fit_converged": bool  # whether optimizer converged
}
```

**Computational cost:** This is the expensive step. NLS on 128 points
with 5 parameters. With scipy, ~1-5ms per call. Without scipy,
Nelder-Mead will be slower (~10-50ms). Consider:
- Skip fitting when window hasn't changed enough (delta threshold)
- Warm-start from previous fit parameters
- Fit every N steps instead of every step (configurable)

**Config parameters:**
```python
frm_fit_window: int = 128        # sliding window size
frm_fit_interval: int = 1        # fit every N steps (1 = every step)
frm_fit_delta_threshold: float = 0.01  # skip if RMS change < threshold
```

**Tests needed:**
- Synthetic damped oscillation → should recover known λ, ω within 5%
- Pure noise → R² should be low, λ estimate should be unreliable
- Step change → R² should drop, indicating scope boundary
- Undamped oscillation (λ = 0) → should detect bifurcation proximity

---

### Step 39: DecayRateTrackingStep

**Purpose:** Track λ over time and detect approach to Hopf bifurcation.

**Math:**
```
λ → 0  ⟹  system approaching critical transition

dλ/dt estimated from rolling window of λ values
Time-to-zero: Δt_λ = λ / |dλ/dt|  (when dλ/dt < 0)
```

This is analogous to the existing DiagnosticWindowStep (Step 33) but
operates on the fitted decay rate instead of mean coupling strength.
The theoretical grounding is stronger: λ → 0 at the Hopf bifurcation
is a mathematical fact of the DDE, not an engineering heuristic.

**Key distinction from Step 33:**
- Step 33: Δt = (κ̄ − κ_c) / |dκ̄/dt| — heuristic threshold on coupling
- Step 39: Δt_λ = λ / |dλ/dt| — parameter of the fitted physical model

**Outputs to scratch:**
```python
{
    "frm_lambda_rate": float,            # dλ/dt
    "frm_bifurcation_proximity": float,  # λ (lower = closer)
    "frm_time_to_bifurcation": float | None,  # Δt_λ (steps)
    "frm_critical_slowing": bool,        # λ below warning threshold
}
```

**Config parameters:**
```python
frm_lambda_history_window: int = 20   # rolling window for dλ/dt
frm_lambda_warning: float = 0.05      # λ below this → critical_slowing
```

**Tests needed:**
- Linearly decreasing λ → should estimate correct time-to-zero
- Stable λ → should report no critical slowing
- Sudden λ drop → should fire critical_slowing immediately

---

### Step 40: CharacteristicFrequencyStep

**Purpose:** Compare fitted ω to theoretical prediction and focus
spectral analysis on the characteristic frequency.

**Math:**
```
ω_theoretical = π / (2·τ_gen)

If τ_gen is known (user-supplied):
  frequency_drift = |ω_fitted − ω_theoretical| / ω_theoretical

If τ_gen is unknown:
  τ_gen_estimated = π / (2·ω_fitted)
  (informational only — tells user what generation timescale the data implies)
```

**Design decision:** τ_gen is domain-specific (e.g., 6 hours for
circadian, ~20 years for generational). The detector cannot infer it
from data alone. Two modes:

1. **τ_gen supplied** (via config): Compare fitted ω to prediction.
   Drift indicates the system is not behaving as the FRM predicts,
   which is itself diagnostic.

2. **τ_gen unknown** (default): Report estimated τ_gen from fitted ω.
   No comparison, just information extraction. The existing 5 fixed
   bands continue to operate.

**Outputs to scratch:**
```python
{
    "frm_omega_fitted": float,
    "frm_tau_gen_estimated": float,      # π / (2·ω)
    "frm_tau_gen_supplied": float | None,
    "frm_frequency_drift": float | None, # only if τ_gen supplied
    "frm_omega_stable": bool,            # ω not drifting significantly
}
```

**Config parameters:**
```python
frm_tau_gen: float | None = None   # user-supplied generation timescale
frm_omega_drift_threshold: float = 0.15  # fractional drift threshold
```

---

### Step 41: ModelQualityStep

**Purpose:** Track whether the FRM form is actually a good fit for the
current data. This is the scope boundary detector — when R² drops or
residuals show structure, the FRM pathway should be downweighted.

**Math:**
```
R² from Step 38

Residual autocorrelation:
  r_resid = autocorrelation(residuals, lag=1)

  High r_resid means the model is missing systematic structure
  (the residuals aren't random — something is left unexplained)

Scope boundary detection:
  in_scope = (R² > frm_r_squared_threshold) AND (|r_resid| < 0.3)
```

**Why this matters:** The FRM applies to damped systems before the Hopf
bifurcation (μ < 0 in the DDE). When the data doesn't fit the form —
e.g., after a regime change, during limit-cycle oscillation, or when
the signal is just noise — the FRM pathway should flag itself as
out-of-scope rather than producing unreliable estimates.

**Outputs to scratch:**
```python
{
    "frm_in_scope": bool,
    "frm_r_squared": float,                # from Step 38
    "frm_residual_autocorrelation": float,
    "frm_scope_status": str,               # "IN_SCOPE" / "BOUNDARY" / "OUT_OF_SCOPE"
}
```

**Config parameters:**
```python
frm_r_squared_threshold: float = 0.5    # below this → out of scope
frm_r_squared_boundary: float = 0.7     # below this → boundary warning
```

---

### Step 42: VirtuWindowStep

**Purpose:** Implement the MK-P5 decision theory. Estimate the window
of time remaining for rational intervention.

**Math (from MK-P5 Theorem 1):**
```
Intervention is rational iff:
  E[W_v(t)] > T_decision × (1 + C_fp/C_late)

Where:
  W_v(t) = estimated time remaining before critical transition
         = frm_time_to_bifurcation from Step 39

  T_decision = minimum implementation lead time (user-supplied)
  C_fp = cost of false positive intervention (user-supplied or default 1.0)
  C_late = cost of late/missed intervention (user-supplied or default 1.0)
```

**t_trap detection (from MK-P5 Theorem 4):**
```
As λ → 0, the uncertainty in Δt_λ grows (Kramers scaling):
  σ_Δt ~ λ^(-1/2)

t_trap is the point where:
  E[W_v(t)] ≤ T_decision × (1 + C_fp/C_late)

After t_trap, rational intervention is no longer possible
regardless of cost structure.
```

**Implementation:**
- Track variance of Δt_λ estimates over the rolling window
- Compute coefficient of variation: CV = σ_Δt / E[Δt]
- When CV exceeds threshold and Δt is shrinking, approaching t_trap

**Outputs to scratch:**
```python
{
    "frm_virtu_window": float | None,    # steps remaining for rational action
    "frm_intervention_rational": bool,    # W_v > T_decision threshold
    "frm_t_trap_proximity": str,          # "FAR" / "APPROACHING" / "PAST"
    "frm_decision_urgency": str,          # "NONE" / "MONITOR" / "DECIDE_NOW" / "TOO_LATE"
}
```

**Config parameters:**
```python
frm_t_decision: float = 10.0     # minimum lead time for intervention
frm_cost_fp: float = 1.0         # false positive cost
frm_cost_late: float = 1.0       # late intervention cost
```

**Design note:** This step only produces meaningful output when the FRM
pathway is in scope (Step 41) and λ is declining (Step 39). Otherwise
it reports `frm_decision_urgency: "NONE"`.

---

### Step 43: FRMAlertStep

**Purpose:** Aggregate FRM pathway signals into alert decisions,
independent of the v12.2 alert system.

**Alert conditions:**

| Alert | Condition | Severity |
|-------|-----------|----------|
| `FRM_CRITICAL_SLOWING` | λ < warning threshold AND in_scope | WARNING |
| `FRM_BIFURCATION_APPROACHING` | Δt_λ < T_decision AND in_scope | CRITICAL |
| `FRM_SCOPE_EXIT` | R² dropped below threshold | INFO |
| `FRM_FREQUENCY_DRIFT` | ω drifting from theoretical (if τ_gen supplied) | WARNING |
| `FRM_INTERVENTION_WINDOW_CLOSING` | t_trap approaching | CRITICAL |

**Outputs to scratch:**
```python
{
    "frm_alert": bool,
    "frm_alert_reasons": list[str],
    "frm_alert_severity": str,  # "INFO" / "WARNING" / "CRITICAL"
}
```

---

## New SentinelResult Convenience Methods

```python
# FRM model fit
result.get_frm_fit() -> dict
# {"lambda": 0.12, "omega": 0.78, "r_squared": 0.91,
#  "in_scope": True, "tau_gen_estimated": 2.01}

# Bifurcation proximity
result.get_bifurcation_status() -> dict
# {"lambda": 0.03, "critical_slowing": True,
#  "time_to_bifurcation": 42.7, "confidence": "HIGH"}

# Decision window (MK-P5)
result.get_virtu_window() -> dict
# {"window_steps": 32.0, "intervention_rational": True,
#  "t_trap_proximity": "APPROACHING", "urgency": "DECIDE_NOW"}
```

---

## New Config Parameters

```python
@dataclass(frozen=True, slots=True)
class SentinelConfig:
    # ... existing parameters ...

    # FRM Pathway (v13)
    enable_frm_pathway: bool = False       # opt-in (not enabled by default)
    frm_fit_window: int = 128              # sliding window for model fit
    frm_fit_interval: int = 1              # fit every N steps
    frm_fit_delta_threshold: float = 0.01  # skip fit if RMS change small
    frm_lambda_history_window: int = 20    # rolling window for dλ/dt
    frm_lambda_warning: float = 0.05       # λ below this → warning
    frm_tau_gen: float | None = None       # user-supplied generation timescale
    frm_omega_drift_threshold: float = 0.15
    frm_r_squared_threshold: float = 0.5   # below → out of scope
    frm_r_squared_boundary: float = 0.7    # below → boundary
    frm_t_decision: float = 10.0           # minimum intervention lead time
    frm_cost_fp: float = 1.0               # false positive cost
    frm_cost_late: float = 1.0             # late intervention cost
```

**Default: disabled.** The FRM pathway is opt-in because:
1. It requires scipy for practical performance (NLS fitting)
2. It adds computational cost (~5-50ms per step depending on backend)
3. It only produces meaningful output for data that actually fits the FRM form
4. The existing v12.2 pipeline is unaffected

---

## What This Does NOT Do

- **Does not replace the existing pipeline.** Steps 1-37 are unchanged.
- **Does not validate the FRM theory.** It implements the predictions
  and lets users test them on their data. Whether f(t) actually fits
  is an empirical question per dataset.
- **Does not claim universality.** The FRM form fits some data well and
  other data poorly. The ModelQualityStep (Step 41) exists precisely
  to detect when the model doesn't apply.
- **Does not require τ_gen.** If the user doesn't supply a generation
  timescale, the detector still fits ω from data and reports what
  τ_gen the data implies. The theoretical comparison is optional.
- **Does not beat DAMP on general benchmarks.** This targets a specific
  niche: early warning for systems approaching critical transitions.
  For point anomalies, contextual anomalies, etc., the existing
  pipeline (or DAMP) is more appropriate.

---

## Validation Plan

### Synthetic Tests

1. **Known damped oscillation**
   - Generate `f(t) = 5 + 3·exp(-0.1t)·cos(2t) + noise`
   - Verify Step 38 recovers λ=0.1, ω=2 within 5%
   - Verify R² > 0.85

2. **Approaching bifurcation**
   - Generate data with linearly decreasing λ (0.2 → 0.01 over 1000 steps)
   - Verify Step 39 fires critical_slowing before λ reaches 0
   - Verify time-to-bifurcation estimate is within 20% of actual

3. **Scope boundary**
   - Generate 500 steps of damped oscillation, then 500 steps of white noise
   - Verify Step 41 reports in_scope for first half, out_of_scope for second

4. **Non-FRM data**
   - Run on standard benchmark archetypes (point, contextual, collective, drift, variance)
   - FRM pathway should report low R² / out_of_scope for most
   - Existing v12.2 pipeline performance should be unchanged

5. **Virtù window**
   - Generate approaching-bifurcation data with known remaining time
   - Verify Step 42 estimates match within 30%
   - Verify t_trap fires before the transition

### Real-World Validation (Future)

The synthetic tests prove the implementation is correct. Whether the FRM
form actually fits real-world data is a separate question that requires:
- Power grid frequency data (known oscillatory dynamics)
- Heart rate variability (known pre-event signatures)
- Financial volatility (known regime changes)
- Network traffic (known periodicity from human behavior)

This is research, not engineering. Results may show the FRM form doesn't
fit most real-world data, and that's a valid finding.

---

## Implementation Order

1. `FRMModelFitStep` (Step 38) — the core; everything depends on this
2. `ModelQualityStep` (Step 41) — scope boundary needed immediately
3. `DecayRateTrackingStep` (Step 39) — the primary early warning signal
4. `CharacteristicFrequencyStep` (Step 40) — informational
5. `VirtuWindowStep` (Step 42) — decision theory layer
6. `FRMAlertStep` (Step 43) — aggregation

Tests for each step before moving to the next.

---

## Open Questions

1. **Fitting cost.** NLS on 128 points every step may be too expensive
   for high-throughput streams. The `frm_fit_interval` parameter
   mitigates this, but what's the right default? Need benchmarking.

2. **stdlib-only fitting.** Without scipy, Nelder-Mead on 5 parameters
   is slow and unreliable. Should we require scipy for the FRM pathway
   and raise ImportError if it's not available?

3. **Multiple ω.** What if the data has multiple oscillatory components?
   The FRM predicts a single characteristic frequency, but real data
   may have harmonics or multiple modes. Should we fit multiple FRM
   forms and take the dominant one?

4. **Window size.** 128 is arbitrary. Too small → noisy fits. Too large
   → slow to respond to changes. Should this be adaptive based on the
   fitted ω (e.g., window = 4 × period)?

5. **Integration with existing alerts.** Should FRM alerts feed into the
   existing `alert_reasons` list, or stay in a separate namespace?
   Separate namespace is cleaner but means users need to check two places.
