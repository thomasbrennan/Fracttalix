# Sentinel v13 Design — Streaming Cross-Scale Coupling Monitor

## Status: DESIGN DRAFT — Not Approved

---

## The One Thing

**Best-in-world target: detecting that a system is about to fail by
watching how its fast and slow dynamics decouple — in a stream, in
real time, with a time-to-failure estimate.**

This is not a general-purpose anomaly detector. It does one thing:

> Given a stream of observations from a system with oscillatory dynamics
> at multiple timescales, detect when the coupling between those scales
> is degrading, estimate how long before the system transitions, and
> tell you whether it's organic decay or external disruption.

Nobody else does this for streaming data. The neuroscience community has
mature PAC tools but applies them offline. The EWS community does
model-free critical slowing down but doesn't use coupling structure.
DAMP/HS-Trees do general anomaly detection but have no concept of
cross-scale relationships. This is the open niche.

### What we give up

- Point anomaly detection (DAMP is better, and simpler)
- General-purpose streaming anomaly detection (existing v12.2 pipeline, or River)
- Batch anomaly detection (PyOD has 45 algorithms)
- Any claim to universality

### What we claim

- **Streaming PAC monitoring** with proper statistical testing (surrogates)
- **Parametric pre-transition warning** via fitted decay rate λ → 0
- **Time-to-failure estimation** grounded in the Hopf bifurcation model,
  not linear extrapolation of a heuristic
- **Degradation type classification** (organic coupling-first vs external
  coherence-first) — a genuinely novel diagnostic
- **Decision-theoretic intervention window** (MK-P5 Virtù Window / t_trap)
- **Self-aware scope boundary** — knows when it doesn't apply and says so

---

## Why This Could Be Best-in-World

### 1. The gap is real and confirmed

SOTA survey (2025-2026) found:
- Zero papers applying PAC to streaming anomaly detection outside neuroscience
- Zero papers using Kuramoto order parameter as a streaming anomaly score
- Zero streaming detectors that monitor cross-frequency coupling health
- The Pi-Transformer (2025) acknowledges "synchronisation matters more than
  individual values" but uses attention, not coupling analysis

### 2. The theory provides something EWS doesn't

Generic EWS (Scheffer et al.) watches for rising variance and autocorrelation.
These are model-free signals — they work but they're noisy and slow.

The FRM approach fits a specific parametric form and watches for the decay
rate λ approaching zero. This is:
- **More specific** — one parameter to track instead of two noisy statistics
- **Earlier** — λ changes before variance and AC visibly rise (the model
  captures the underlying dynamics, not just their statistical shadow)
- **More informative** — gives you time-to-transition, not just "something
  is changing"

### 3. The implementation barrier is manageable

The core algorithms exist:
- NLS fitting: scipy.optimize.curve_fit or stdlib Nelder-Mead
- PAC: Tort 2010 Modulation Index (already in v12.2, needs surrogate testing)
- Kuramoto Φ: already in v12.2
- Hilbert transform: already in v12.2
- FFT: already in v12.2

What's new is the composition: fitting the FRM form, tracking λ, and using
coupling structure as the primary anomaly signal.

---

## Architecture

### Core Pipeline (3 stages, not 6)

Previous design had 6 steps. That's too many for "one thing." Collapsed to 3:

```
Stage 1: FIT
    Fit f(t) = B + A·exp(-λt)·cos(ωt+φ) to sliding window
    Track λ, ω, R² over time
    Detect scope boundary (R² drop, residual structure)

Stage 2: COUPLE
    Monitor PAC with surrogate-tested significance
    Track Kuramoto Φ independently of κ̄
    Classify degradation sequence (coupling-first vs coherence-first)

Stage 3: DECIDE
    Estimate time-to-bifurcation from λ trajectory
    Compute Virtù Window (rational intervention window)
    Detect t_trap (point of no return)
    Issue alerts with confidence grading
```

These map to 3 new DetectorStep subclasses. They depend on the existing
pipeline's FFT (Step 3) and phase extraction (Step 28) but are otherwise
self-contained.

---

## Stage 1: CouplingFitStep

**The core.** Everything else depends on this.

### What it computes

On every update (or every N updates for performance):

1. **Fit the FRM form** to a sliding window of data:
   ```
   f(t) = B + A·exp(-λt)·cos(ωt + φ)
   ```
   Five parameters: B (baseline), A (amplitude), λ (decay rate),
   ω (characteristic frequency), φ (phase offset).

2. **Track parameter evolution:**
   - λ_history: rolling window of fitted λ values
   - ω_history: rolling window of fitted ω values
   - R²_history: rolling window of fit quality

3. **Detect scope boundary:**
   - R² < 0.5 → OUT_OF_SCOPE (model doesn't fit this data)
   - R² < 0.7 → BOUNDARY (model fit is degrading)
   - Residual autocorrelation > 0.3 → STRUCTURED_RESIDUALS
     (model is missing something systematic)

### Fitting algorithm

**Initialization (critical for convergence):**
- ω_init: dominant FFT peak frequency from Step 3
- λ_init: log-envelope decay rate (fit line to log of peak envelope)
- B_init: mean of last 10% of window (steady-state estimate)
- A_init: max deviation from B_init
- φ_init: phase at window start from FFT

**Warm-starting:** After first successful fit, subsequent fits use the
previous parameters as initial guess. This dramatically improves
convergence speed and reliability.

**Constraint:** λ ≥ 0. The FRM scope boundary is the Hopf bifurcation
(λ = 0). Negative λ means the system is past the bifurcation (growing
oscillation / limit cycle). When the fitter wants λ < 0, report
`scope_status: "PAST_BIFURCATION"` — the system has already transitioned.

**Backend selection:**
- scipy available: `curve_fit` with `method='lm'` (Levenberg-Marquardt)
- scipy unavailable: raise `ImportError` with clear message

**Design decision: require scipy.** The previous design tried to support
stdlib-only fitting with Nelder-Mead. This is wrong for "best in world."
Nelder-Mead on 5 parameters is unreliable, slow, and would produce
worse results than the competition. The FRM pathway requires scipy.
The existing v12.2 pipeline remains stdlib-only.

### Performance

NLS on 128 points with 5 parameters and warm-starting: ~0.5-2ms with scipy.

Mitigation for high-throughput streams:
- `fit_interval`: fit every N steps (default: 4, i.e., every 4th observation)
- `fit_delta_threshold`: skip if RMS window change < threshold
- Between fits, interpolate λ and ω from trend

### Outputs

```python
scratch["coupling_fit"] = {
    "lambda": float,        # decay rate (THE key signal)
    "omega": float,         # characteristic frequency
    "amplitude": float,     # A
    "baseline": float,      # B
    "phase": float,         # φ
    "r_squared": float,     # goodness of fit
    "scope_status": str,    # IN_SCOPE / BOUNDARY / OUT_OF_SCOPE / PAST_BIFURCATION
    "fit_converged": bool,
    "tau_gen_implied": float,  # π/(2ω) — what generation timescale the data implies
}
```

### Config

```python
enable_coupling_monitor: bool = False    # opt-in
coupling_fit_window: int = 128           # sliding window size
coupling_fit_interval: int = 4           # fit every N steps
coupling_fit_delta: float = 0.01         # skip if window barely changed
coupling_r_squared_min: float = 0.5      # below → out of scope
coupling_tau_gen: float | None = None    # user-supplied, for comparison
```

---

## Stage 2: CouplingHealthStep

**Monitor the health of cross-scale coupling with proper statistics.**

This replaces the crude 8-bin PAC in v12.2 with surrogate-tested coupling.

### What it computes

1. **PAC with surrogate testing:**
   - Compute Modulation Index (Tort 2010) for each slow/fast band pair
   - Generate N=200 surrogate PAC values by time-shifting the amplitude
     envelope (Aru et al. 2015 method — preserves spectral content,
     breaks coupling)
   - Compute z-score: z_PAC = (MI_observed - mean(MI_surrogates)) / std(MI_surrogates)
   - PAC is significant at p < 0.05 when z_PAC > 1.96

   **Why this matters:** Without surrogate testing, nonstationary signals
   produce spurious PAC (documented in Aru et al. 2015, van Driel et al.
   2015). The v12.2 raw MI is not trustworthy. Surrogate testing is what
   makes the difference between "we compute a number" and "we detect
   real coupling."

2. **Coupling health score:**
   ```
   H_coupling = fraction of band pairs with significant PAC (z > 1.96)
   ```
   Range: 0.0 (no significant coupling) to 1.0 (all pairs coupled).

   This replaces v12.2's raw `composite_coupling_score` with a
   statistically grounded measure.

3. **Coupling degradation detection:**
   - Track H_coupling over rolling window
   - Degradation = H_coupling declining (OLS slope < 0, p < 0.05)
   - Rate = slope magnitude

4. **Degradation sequence classification:**
   Using the fitted λ from Stage 1 and Φ from existing Step 34:
   ```
   COUPLING_FIRST:  H_coupling declining while λ stable
   DECAY_FIRST:     λ declining while H_coupling stable
   SIMULTANEOUS:    Both declining
   STABLE:          Neither declining
   ```

   The FRM predicts that organic degradation follows COUPLING_FIRST →
   then DECAY_FIRST. External disruption skips to DECAY_FIRST directly.
   This is the sequence classification idea from v12.2, but using
   statistically tested coupling and the fitted decay rate instead
   of raw heuristic scores.

### Surrogate cost

200 surrogates × 6 band pairs = 1200 MI computations per update.
Each MI: ~0.1ms → ~120ms total. Too expensive for every step.

**Mitigation:** Run surrogate testing every `coupling_health_interval`
steps (default: 10). Between tests, track raw MI and flag if it drops
below the last significant threshold. Full retest on flag.

### Outputs

```python
scratch["coupling_health"] = {
    "health_score": float,          # fraction of significant pairs
    "significant_pairs": int,       # count
    "total_pairs": int,             # 6
    "degradation_rate": float,      # OLS slope of H over time
    "degradation_significant": bool,# p < 0.05
    "sequence_type": str,           # COUPLING_FIRST / DECAY_FIRST / SIMULTANEOUS / STABLE
    "pac_z_scores": dict,           # per-pair z-scores
}
```

### Config

```python
coupling_health_interval: int = 10       # surrogate test every N steps
coupling_n_surrogates: int = 200         # surrogate count
coupling_significance: float = 0.05     # p-value threshold
```

---

## Stage 3: TransitionAlertStep

**The decision layer. Answers: "How long do I have, and should I act?"**

### What it computes

1. **Time-to-bifurcation (from λ trajectory):**
   ```
   Δt_λ = λ / |dλ/dt|    (when dλ/dt < 0 and scope_status = IN_SCOPE)
   ```

   Three estimates (pessimistic, expected, optimistic) using the
   variance of the λ rate:
   ```
   Δt_pessimistic = λ / (|dλ/dt| + σ_rate)
   Δt_expected    = λ / |dλ/dt|
   Δt_optimistic  = λ / max(|dλ/dt| - σ_rate, 0.1·|dλ/dt|)
   ```

   Confidence grading from coefficient of variation of recent dλ/dt:
   - CV < 0.3 → HIGH
   - CV < 0.7 → MEDIUM
   - CV ≥ 0.7 → LOW

2. **Virtù Window (MK-P5 Theorem 1):**
   ```
   Intervention is rational iff:
     Δt_expected > T_decision × (1 + C_fp/C_late)

   Where:
     T_decision = user-supplied minimum lead time
     C_fp/C_late = cost asymmetry ratio
   ```

   Default: symmetric costs → intervention rational when Δt > T_decision.

3. **t_trap detection (MK-P5 Theorem 4):**
   ```
   As λ → 0, σ_Δt grows (Kramers scaling):
     CV_Δt → ∞ as λ → 0

   t_trap: the point where Δt_expected ≤ T_decision × (1 + C_fp/C_late)

   After t_trap, the uncertainty is so large and the window so small
   that rational intervention is no longer possible.
   ```

4. **Alert emission:**

   | Alert | Condition | Severity |
   |-------|-----------|----------|
   | `COUPLING_DEGRADING` | H_coupling declining significantly | INFO |
   | `CRITICAL_SLOWING` | λ < warning threshold, in scope | WARNING |
   | `TRANSITION_APPROACHING` | Δt < 2 × T_decision | WARNING |
   | `DECIDE_NOW` | T_decision < Δt < 2 × T_decision | CRITICAL |
   | `WINDOW_CLOSED` | Δt < T_decision (past t_trap) | CRITICAL |
   | `SCOPE_EXIT` | R² dropped, model no longer fits | INFO |

   Every alert includes: confidence level, Δt estimate with bounds,
   coupling health score, sequence classification.

### Outputs

```python
scratch["transition_alert"] = {
    # Time estimate
    "time_to_transition": float | None,      # Δt expected (steps)
    "time_pessimistic": float | None,
    "time_optimistic": float | None,
    "confidence": str,                        # HIGH / MEDIUM / LOW

    # Decision theory
    "intervention_rational": bool,
    "virtu_window": float | None,            # steps of rational action remaining
    "t_trap_status": str,                     # FAR / APPROACHING / PAST

    # Alert
    "alert": bool,
    "alert_type": str | None,
    "alert_severity": str,                    # INFO / WARNING / CRITICAL

    # Context (for the human reading the alert)
    "lambda": float,                          # current decay rate
    "coupling_health": float,                 # current H_coupling
    "sequence_type": str,                     # what kind of degradation
    "scope_status": str,                      # is this estimate trustworthy
}
```

### Config

```python
transition_t_decision: float = 10.0      # minimum intervention lead time
transition_cost_fp: float = 1.0          # false positive cost
transition_cost_late: float = 1.0        # late intervention cost
transition_lambda_warning: float = 0.05  # λ below this → CRITICAL_SLOWING
```

---

## SentinelResult API (New Methods)

Three methods, matching the three stages:

```python
# Stage 1: Is the model fitting? What are the parameters?
result.get_coupling_fit() -> dict
# {"lambda": 0.08, "omega": 0.78, "r_squared": 0.91,
#  "scope_status": "IN_SCOPE", "tau_gen_implied": 2.01}

# Stage 2: Is coupling healthy? What kind of degradation?
result.get_coupling_health() -> dict
# {"health_score": 0.67, "significant_pairs": 4,
#  "degradation_rate": -0.03, "sequence_type": "COUPLING_FIRST"}

# Stage 3: How long do I have? Should I act?
result.get_transition_status() -> dict
# {"time_to_transition": 42.7, "confidence": "HIGH",
#  "intervention_rational": True, "t_trap_status": "APPROACHING",
#  "alert_type": "DECIDE_NOW"}
```

One-liner for dashboards:

```python
if result.get_transition_status()["alert"]:
    status = result.get_transition_status()
    print(f"{status['alert_type']}: ~{status['time_to_transition']:.0f} steps "
          f"({status['confidence']} confidence, {status['sequence_type']})")
```

---

## What Makes This Best-in-World (vs. Just Good)

| Feature | v12.2 (current) | v13 (this design) | Nobody else |
|---------|-----------------|-------------------|-------------|
| PAC computation | 8 bins, no significance testing | Surrogate-tested, z-scored | Correct |
| Coupling health metric | Raw composite score | Fraction of significant pairs | Novel for streaming |
| Pre-transition warning | Heuristic threshold on κ̄ | Fitted λ → 0 (Hopf model) | Novel |
| Time-to-failure | Linear extrapolation of κ̄ | λ trajectory with confidence bounds | Novel |
| Degradation typing | Coupling-first vs coherence-first (raw) | Same idea, statistically tested | Novel |
| Decision support | None | Virtù Window + t_trap | Novel |
| Scope awareness | None | R² + residual structure | Correct |

"Novel for streaming" = published in neuroscience but never applied to
streaming anomaly detection. "Novel" = not published anywhere we could find.

---

## Validation Plan

### Must-pass before shipping

1. **Parameter recovery:** Synthetic damped oscillation with known λ, ω →
   Stage 1 recovers both within 5%. Repeat 100 times with different noise
   levels to establish reliability envelope.

2. **Pre-transition detection:** Generate 1000-step series with λ decreasing
   from 0.2 to 0.01. Stage 3 must fire CRITICAL_SLOWING before step 800
   (i.e., at least 200 steps of warning). Compare lead time against generic
   EWS (variance + AC) on the same data.

3. **False positive rate on stable systems:** Generate 10,000 steps of stable
   damped oscillation (constant λ = 0.15). Stage 3 must fire CRITICAL_SLOWING
   on < 1% of updates.

4. **Scope boundary:** White noise, linear trend, step function, undamped
   sine wave → Stage 1 must report OUT_OF_SCOPE for all of these. The
   detector must know when it doesn't apply.

5. **Surrogate PAC:** Synthetic coupled oscillation (genuine PAC) vs.
   independent oscillations (no PAC) → Stage 2 must distinguish these
   with > 95% accuracy.

6. **Degradation typing:** Generate coupling-first degradation and
   decay-first degradation → Stage 2 must classify correctly.

7. **v12.2 regression:** All 374 existing tests must still pass. The new
   pathway must not affect the existing pipeline's behavior.

### Stretch goals (research, not engineering)

- Run on publicly available power grid frequency data
- Run on MIT-BIH arrhythmia database (ECG with known pre-event signatures)
- Run on Yahoo S5 anomaly detection benchmark
- Compare lead time against Dakos et al. EWS R package on the same data

---

## Implementation Order

1. **CouplingFitStep** — the foundation. Tests 1, 3, 4.
2. **CouplingHealthStep** — surrogate-tested PAC. Tests 5, 6.
3. **TransitionAlertStep** — decision layer. Tests 2, 7.
4. **Integration** — convenience methods, config, documentation.

Each stage is independently useful. Ship Stage 1 alone if needed —
even without the alert logic, streaming λ tracking is novel.

---

## Resolved Design Decisions

| Question | Decision | Rationale |
|----------|----------|-----------|
| Require scipy? | **Yes** | Best-in-world means best results, not broadest compatibility |
| Fit every step? | **No, every 4th** | 0.5ms × 4 = 2ms amortized; warm-start makes this reliable |
| How many new steps? | **3** (not 6) | One thing, done well. Each step = one stage of the pipeline |
| Separate alert namespace? | **Yes** | `frm_alert` separate from `alert`. Different question, different answer |
| Window size? | **128 default, adaptive option** | 4× fitted period when ω is stable; 128 as safe default |
| Multiple ω? | **No** | Fit the dominant mode. If data has multiple modes, report R² degradation — that's useful too |
| stdlib-only fallback? | **No** | Nelder-Mead on 5 params is unreliable. Don't ship something worse than the competition |

---

## What This Is NOT

This is not a general anomaly detector. If someone asks "is this data point
anomalous?", use the v12.2 pipeline or DAMP.

This answers a different question: **"Is this system losing its internal
coherence in a way that predicts failure, and if so, how long do I have?"**

That's a narrower question. But for systems where it applies — oscillatory
systems approaching critical transitions — nobody else answers it in a
streaming context with statistical rigor and time-to-failure estimates.

That's the thing we can be best in the world at.
