# Sentinel v13 Design — The λ Detector

## Status: DESIGN DRAFT — Not Approved

---

## The Theorem

The Fractal Rhythm Model derives, from first principles, that network
information transmission follows:

```
f(t) = B + A·exp(-λt)·cos(ωt + φ)
```

Where two parameters are **derived, not fitted:**

```
ω = π / (2·τ_gen)              — characteristic frequency (Hopf quarter-wave theorem)
λ = |α| / (Γ·τ_gen)            — decay rate (perturbation expansion)
Γ = 1 + π²/4 ≈ 3.467           — universal loop impedance constant
```

The derivation: starting from the delayed differential equation
`dx/ds = α·x(s) − k·x(s−1)`, substituting `λ = iΩ` at the Hopf
critical point `α = 0`, separating real and imaginary parts gives
`cos(Ω) = 0`, therefore `Ω* = π/2`, therefore `T = 4·τ_gen`.

The strongest validation: `T = 4 × 6 hours = 24 hours` (circadian period
from generation timescale, zero free parameters).

**Nobody else has this.** This is original mathematics. The derivation
uses known DDE theory (Hayes 1950, Kuang 1993) but the cross-domain
application and the specific constants β=1/2 and Γ=1+π²/4 are new.

---

## The Detector

When λ → 0, the system approaches its Hopf bifurcation. The damping
disappears. The oscillation stops decaying and starts growing. The
system transitions.

**That's the detector:** fit the FRM form to streaming data, track λ,
and fire when λ approaches zero.

This is fundamentally different from generic Early Warning Signals
(Scheffer et al.), which watch for rising variance and autocorrelation —
model-free statistical shadows of the underlying dynamics. The FRM
detector watches the dynamics directly: the decay rate of the fitted
parametric model.

### Why λ beats generic EWS

| | Generic EWS | λ detector |
|---|-----------|------------|
| What it tracks | Variance + lag-1 autocorrelation | Fitted decay rate of parametric model |
| Theoretical basis | "Systems slow down near transitions" (qualitative) | λ = 0 at Hopf bifurcation (quantitative) |
| Free parameters | Window size, threshold | τ_gen (domain-supplied or estimated) |
| Time-to-transition | Not estimated | Δt = λ / \|dλ/dt\| |
| False positive source | Any variance increase (including benign) | Only when damped oscillation form fits AND λ trends toward zero |
| Scope awareness | None (always produces a number) | R² tells you when the model doesn't apply |

### What we give up

Everything else. This is not a general anomaly detector. It answers
one question:

> **Is this system's damping disappearing, and if so, how fast?**

For point anomalies, use DAMP. For distribution shift, use ADWIN.
For general streaming, use the v12.2 pipeline. This detector does
one thing that none of them can do, because none of them have the
FRM functional form.

---

## Architecture: One Step

Not three stages. Not six steps. One.

### HopfDetectorStep

**Input:** sliding window of observations + τ_gen (supplied or estimated)

**Output:** λ, its trajectory, time-to-bifurcation, scope status

**Algorithm:**

```
Every N observations (default N=4, warm-started):

1. PREDICT ω
   If τ_gen supplied:  ω_predicted = π / (2·τ_gen)
   If τ_gen unknown:   ω_predicted = dominant FFT peak

2. FIT the 3-parameter model (B, A, φ are free; ω is constrained near prediction)
   f(t) = B + A·exp(-λt)·cos(ω_predicted·t + φ)

   This is a 3-parameter fit (B, A, φ) plus λ as the key free parameter,
   with ω constrained within ±15% of prediction. Much easier to fit than
   the full 5-parameter model.

   Constraint: λ ≥ 0 (FRM scope boundary)

3. TRACK λ over time
   Maintain rolling window of fitted λ values (default: 20)
   Compute dλ/dt via OLS slope

4. ESTIMATE time-to-bifurcation (when dλ/dt < 0)
   Δt = λ / |dλ/dt|

   Confidence from coefficient of variation of recent dλ/dt values:
     CV < 0.3 → HIGH
     CV < 0.7 → MEDIUM
     CV ≥ 0.7 → LOW

5. CHECK scope
   R² < 0.5 → OUT_OF_SCOPE (model doesn't fit; don't trust λ)
   R² < 0.7 → BOUNDARY
   λ fitted < 0 → PAST_BIFURCATION (system already transitioned)
   Otherwise → IN_SCOPE

6. ALERT
   λ < warning_threshold AND IN_SCOPE → CRITICAL_SLOWING
   Δt < T_decision AND IN_SCOPE → TRANSITION_APPROACHING
```

### Why one step, not three

The previous design had CouplingFitStep + CouplingHealthStep +
TransitionAlertStep. That's three steps for something that should be
one continuous computation. The surrogate-tested PAC and degradation
typing are interesting engineering but they're not *the thing nobody
else can do*. They dilute focus.

The thing nobody else can do is: **fit a parametric model with derived
constants and watch its decay rate approach zero.** That's one step.

If the λ detector works — if it gives earlier, more specific warning
than generic EWS on real data — then we can add PAC monitoring and
degradation typing later as supporting evidence. But the core must
work first, alone, and be demonstrably superior at the one thing it does.

### Why ω constrained, not free

The previous design fitted all 5 parameters freely. That's wrong.
The FRM's power is that ω is *predicted*: ω = π/(2·τ_gen). If we
fit ω freely, we're just doing generic damped-oscillation fitting —
anyone can do that.

By constraining ω near the predicted value, we're using the theorem.
The fit either works with the predicted ω (FRM applies, λ is
meaningful) or it doesn't (R² drops, scope exit). That's the test.

Two modes:
- **τ_gen supplied:** ω = π/(2·τ_gen) ± 15%. The user knows their
  system's generation timescale. This is the strong mode — the
  frequency is predicted, not fitted.
- **τ_gen unknown:** ω initialized from dominant FFT peak, then
  constrained near it. Weaker — we're estimating ω from data rather
  than predicting it. But λ tracking still works.

---

## Config

```python
@dataclass(frozen=True, slots=True)
class SentinelConfig:
    # ... existing parameters ...

    # Hopf Detector (v13)
    enable_hopf_detector: bool = False     # opt-in; requires scipy
    hopf_tau_gen: float | None = None      # generation timescale (domain-specific)
    hopf_fit_window: int = 128             # sliding window
    hopf_fit_interval: int = 4             # fit every N steps
    hopf_lambda_window: int = 20           # rolling λ history
    hopf_lambda_warning: float = 0.05      # λ below this → alert
    hopf_t_decision: float = 10.0          # minimum intervention lead time
    hopf_omega_tolerance: float = 0.15     # ω constraint band (±15%)
    hopf_r_squared_min: float = 0.5        # below → out of scope
```

8 parameters. Most have sensible defaults. The only one the user
*should* supply is `hopf_tau_gen` — and even that has a fallback.

---

## Result API

One method:

```python
result.get_hopf_status() -> dict
# {
#     "lambda": 0.03,                    # current decay rate
#     "lambda_rate": -0.002,             # dλ/dt (negative = approaching)
#     "time_to_transition": 15.0,        # steps remaining
#     "confidence": "HIGH",
#     "scope_status": "IN_SCOPE",        # model fits the data
#     "r_squared": 0.89,
#     "omega_fitted": 0.26,              # fitted characteristic frequency
#     "tau_gen_implied": 6.04,           # what τ_gen the data implies
#     "alert": True,
#     "alert_type": "TRANSITION_APPROACHING",
# }
```

One-liner:
```python
hopf = result.get_hopf_status()
if hopf["alert"]:
    print(f"{hopf['alert_type']}: λ={hopf['lambda']:.3f}, "
          f"~{hopf['time_to_transition']:.0f} steps remaining "
          f"({hopf['confidence']})")
```

Result keys added to the dict:

| Key | Type | Description |
|-----|------|-------------|
| `"hopf_lambda"` | `float` | Fitted decay rate (0 = at bifurcation) |
| `"hopf_lambda_rate"` | `float` | dλ/dt (negative = approaching transition) |
| `"hopf_time_to_transition"` | `float\|None` | Estimated steps remaining |
| `"hopf_confidence"` | `str` | HIGH / MEDIUM / LOW |
| `"hopf_scope_status"` | `str` | IN_SCOPE / BOUNDARY / OUT_OF_SCOPE / PAST_BIFURCATION |
| `"hopf_r_squared"` | `float` | Model fit quality |
| `"hopf_omega"` | `float` | Fitted characteristic frequency |
| `"hopf_tau_gen_implied"` | `float` | π/(2ω) — implied generation timescale |
| `"hopf_alert"` | `bool` | Whether an alert is active |
| `"hopf_alert_type"` | `str\|None` | CRITICAL_SLOWING / TRANSITION_APPROACHING / None |

---

## Validation: 4 Tests

### Test 1: Parameter Recovery

Generate `f(t) = 5 + 3·exp(-0.1t)·cos(π/(2·6)·t) + ε` where ε ~ N(0, 0.3).

Supply `hopf_tau_gen = 6`.

**Must pass:** Fitted λ within 10% of 0.1. R² > 0.85. Scope = IN_SCOPE.

### Test 2: Transition Detection (The Money Test)

Generate 1000-step series where λ decreases linearly from 0.2 to 0.0:
```python
for t in range(1000):
    lam = 0.2 * (1 - t/1000)
    signal[t] = B + A * exp(-lam * (t % window)) * cos(omega * t) + noise
```

**Must pass:**
- `CRITICAL_SLOWING` fires before step 750 (≥250 steps of warning)
- `time_to_transition` estimate within 30% of actual remaining steps
- On the same data, generic EWS (variance + AC) fires later than λ detector

This is the test that justifies the project. If the λ detector
doesn't give earlier warning than EWS on this synthetic data, the
theoretical advantage is not translating to practice and we need to
understand why before proceeding.

### Test 3: False Positive Rate

Generate 10,000 steps of stable damped oscillation (constant λ = 0.15).

**Must pass:** `CRITICAL_SLOWING` fires on < 1% of fitted windows.

### Test 4: Scope Boundary

Run on four data types that violate FRM assumptions:
- White noise
- Linear trend
- Step function
- Growing oscillation (λ < 0, past bifurcation)

**Must pass:** All four report `OUT_OF_SCOPE` or `PAST_BIFURCATION`.
The detector must never claim confident λ estimates on data that
doesn't fit the model.

### Regression

All 374 existing tests pass unchanged. Enabling `hopf_detector` must
not affect any existing result key.

---

## Implementation

One file: `fracttalix/steps/hopf.py`

One class: `HopfDetectorStep(DetectorStep)`

One dependency: `scipy.optimize.curve_fit` (raise ImportError with
clear message if scipy not installed)

Estimated size: ~200 lines of code.

### Fitting details

The model function:
```python
def frm_model(t, B, A, lam, phi, omega):
    return B + A * np.exp(-lam * t) * np.cos(omega * t + phi)
```

With ω constrained:
```python
# Fix omega near predicted value; fit B, A, λ, φ
def frm_model_fixed_omega(t, B, A, lam, phi):
    return B + A * np.exp(-lam * t) * np.cos(omega_predicted * t + phi)

# Use curve_fit with bounds: λ ∈ [0, ∞), others unconstrained
popt, pcov = curve_fit(
    frm_model_fixed_omega, t, data,
    p0=[B_init, A_init, lambda_init, phi_init],
    bounds=([−np.inf, −np.inf, 0.0, −np.pi],
            [np.inf,   np.inf, np.inf, np.pi]),
    maxfev=1000,
)
```

Initialization:
```python
B_init = np.mean(data[-len(data)//10:])        # steady-state from tail
A_init = np.max(np.abs(data - B_init))          # peak deviation
phi_init = 0.0                                   # or from FFT phase
lambda_init = previous_lambda or 0.1             # warm-start
```

Warm-starting: after first successful fit, use `popt` as `p0` for the
next fit. This is critical for speed and convergence reliability.

---

## What This Is

A streaming detector built on a theorem nobody else has, applied to the
one problem where that theorem gives you something no other method provides:
watching a system's damping disappear in real time and telling you how long
you have before it transitions.

## What This Is Not

A general anomaly detector. A replacement for v12.2. A claim that the
FRM applies to all data. A guarantee that λ → 0 gives earlier warning
than EWS on real-world data (that's what Test 2 is for — and if it
doesn't, we'll know before shipping).
