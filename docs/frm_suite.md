# FRMSuite — FRM Physics Layer

`FRMSuite` is the full two-layer detection suite for **oscillatory signals** where FRM (Fractal Rhythm Model) physics applies. It combines the five generic detectors of `DetectorSuite` (Layer 1) with three FRM-physics detectors (Layer 2) that require scipy.

Use `FRMSuite` when:

- Your signal is oscillatory (EEG, power grid, HRV, mechanical vibration, financial cycles)
- You have or can estimate a characteristic generation delay `tau_gen`
- You need a **time-to-bifurcation** estimate — not just a binary alert

---

## Installation

```bash
pip install fracttalix[fast]   # includes numpy + scipy for Layer 2
# or explicitly:
pip install fracttalix numpy scipy
```

Layer 2 degrades gracefully: if scipy is absent, `FRMSuite` continues running Layer 1 normally. Layer 2 results return `OUT_OF_SCOPE` and `frm_confidence` stays at 0.

---

## Architecture

```
FRMSuite
├── Layer 1 — DetectorSuite (5 generic detectors, no scipy needed)
│   ├── HopfDetector(ews)   — critical slowing down via EWS
│   ├── DiscordDetector     — point / contextual anomalies
│   ├── DriftDetector       — slow mean shift
│   ├── VarianceDetector    — sudden volatility change
│   └── CouplingDetector    — PAC cross-scale coordination (Layer 1 cross-validator)
│
└── Layer 2 — FRM Physics (scipy required)
    ├── Lambda  (HopfDetector frm) — fits damping λ; tracks λ → 0
    ├── Omega   (OmegaDetector)    — checks observed ω vs predicted π/(2·τ_gen)
    └── Virtu   (VirtuDetector)    — synthesizes time-to-bifurcation estimate Δt ≈ λ/|dλ/dt|
```

---

## The Key Parameter: `tau_gen`

`tau_gen` is the **FRM generation delay** — the characteristic oscillation half-period of your system. The FRM quarter-wave theorem predicts:

```
ω = π / (2 · τ_gen)
```

| Domain | `tau_gen` source |
|--------|----------------|
| Power grid (60 Hz) | ~ 1/(4 × 60) ≈ 0.0042 seconds |
| EEG alpha rhythm (~10 Hz) | ~ 1/(4 × 10) = 0.025 seconds |
| HRV (resting ~1 Hz) | ~ 1/(4 × 1) = 0.25 seconds |
| Mechanical oscillator | Half the observed natural period |
| Financial cycle | Domain-specific; estimate from historical periodogram |

When `tau_gen` is supplied (**strong mode**), the FRM physics test runs and `frm_confidence` can reach 2 or 3. Without `tau_gen` (**weak mode**), Lambda and Omega run as generic frequency-change detectors and `frm_confidence` stays at 0.

---

## Quick Start

### Strong mode (tau_gen known)

```python
from fracttalix.frm import FRMSuite

suite = FRMSuite(tau_gen=12.5)   # your system's generation delay in samples

for value in stream:
    result = suite.update(value)
    print(result.summary())

    # frm_confidence: how many Layer 2 detectors confirm the bifurcation signal
    if result.frm_confidence == 1:
        print("Lambda alerting: λ is declining. Early bifurcation signal.")
    elif result.frm_confidence == 2:
        print("Lambda + Omega alerting: λ declining AND ω drifting. Compound signal.")
    elif result.frm_confidence >= 3:
        print("CRITICAL: full FRM bifurcation signal. Time-to-bifurcation estimate active.")

    # Cross-validation with independent structural signal
    if result.frm_confidence_plus >= 3:
        print("FRM signal cross-validated by PAC decoupling (CouplingDetector).")
```

### Weak mode (tau_gen unknown)

```python
suite = FRMSuite()   # tau_gen=None → weak mode

for value in stream:
    result = suite.update(value)
    # frm_confidence is always 0 in weak mode — no FRM physics test
    # Layer 1 still runs all 5 generic detectors normally
    if result.layer1.variance.is_alert:
        print("Variance change detected")
```

---

## Reading an `FRMSuiteResult`

```python
result = suite.update(value)

# ── Layer 1 (DetectorSuite) ───────────────────────────────────
result.layer1                   # SuiteResult
result.layer1.hopf.is_alert     # bool
result.layer1.drift.message     # str
result.layer1.any_alert         # bool
result.layer1.summary()         # one-line string

# ── Layer 2 — Lambda ──────────────────────────────────────────
result.lambda_.is_alert         # bool — λ declining toward 0
result.lambda_.score            # float 0.0–1.0 — urgency
result.lambda_.status           # ScopeStatus enum
result.lambda_.message
# e.g. "frm λ=0.18 rate=-0.004 ttb=45.0 omega_predicted=0.251 mode=strong"

# ── Layer 2 — Omega ───────────────────────────────────────────
result.omega.is_alert           # bool — ω drifting from π/(2·tau_gen)
result.omega.score              # float 0.0–1.0
result.omega.message
# e.g. "omega_obs=0.265 omega_pred=0.251 deviation=0.056 consecutive=7 mode=strong"

# ── Layer 2 — Virtu ───────────────────────────────────────────
result.virtu.is_alert           # bool — urgency score ≥ alert threshold
result.virtu.score              # float 0.0–1.0 (0 = distant, 1 = imminent)
result.virtu.message
# e.g. "ttb=38.4 confidence=HIGH safety_factor=1.00 omega_confirmed=True"

# ── Compound scores ───────────────────────────────────────────
result.frm_confidence           # int 0–3
result.frm_confidence_plus      # int 0–4 (+1 if CouplingDetector alerting)
result.layer2_available         # bool — False if scipy absent

# ── Convenience ───────────────────────────────────────────────
result.any_alert                # bool — any detector (Layer 1 or 2) alerting
result.alerts                   # list[DetectorResult] — all currently alerting
result.summary()                # multi-line dashboard string
```

### `frm_confidence` Reference

| `frm_confidence` | Active signals |
|-----------------|---------------|
| 0 | No strong-mode Layer 2 alert |
| 1 | Lambda only — λ declining (damping weakening) |
| 2 | Lambda + Omega — λ declining AND ω drifting from FRM prediction |
| 3 | Lambda + Omega + Virtu — full compound signal with TTB estimate |

> **Note on correlation:** Lambda and Omega are partially correlated by FRM physics — a declining λ often co-occurs with ω drift at bifurcation. `frm_confidence_plus` adds `CouplingDetector` (Layer 1), which is **independent** of the FRM model, providing the strongest cross-validation.

---

## VirtuDetector — Time-to-Bifurcation

`VirtuDetector` estimates the observation steps remaining before the system crosses the bifurcation point, using the FRM estimate:

```
Δt ≈ λ / |dλ/dt|
```

It is **not a binary detector** — it is a decision-support tool. The urgency `score` (0 = distant, 1 = imminent) rises as the estimated time-to-bifurcation shrinks relative to the planning horizon (200 steps by default).

**Confidence grades:**

| `confidence` | `lam_rate` | Interpretation |
|-------------|-----------|----------------|
| `HIGH` | < −0.01 | λ declining rapidly; estimate reliable |
| `MEDIUM` | < −0.003 | λ declining moderately; estimate indicative |
| `LOW` | ≥ −0.003 | λ barely declining; estimate uncertain |

**Trust gates:**

- Virtu only reports when Lambda shows actively declining λ (`lam_rate < −0.001`).
- With `omega_trust=True` (default): Virtu also requires OmegaDetector to be in scope (FRM structure intact). If Omega is `OUT_OF_SCOPE`, Virtu returns `OUT_OF_SCOPE` — the TTB estimate would be untrustworthy without confirmed FRM structure.
- Set `omega_trust=False` to get TTB estimates from Lambda alone (less reliable).

### Safety Factor

For asymmetric-cost applications (acting too late is worse than too early):

```python
suite = FRMSuite(
    tau_gen=12.5,
    virtu_kwargs={"safety_factor": 2.0}
)
# Reported TTB = raw_TTB / 2.0 → earlier, more conservative warning
```

---

## OmegaDetector — Frequency Integrity

`OmegaDetector` tracks whether the dominant frequency of the signal matches the FRM prediction.

**Strong mode** (`tau_gen` supplied):
- Predicted: `omega_predicted = π / (2 · tau_gen)`
- Fires ALERT after `alert_steps` consecutive observations where `|ω_obs − ω_pred| / ω_pred > deviation_threshold`
- Default `deviation_threshold = 0.05` (5%)
- Uses parabolic interpolation + smoothing to suppress FFT quantization noise

**Weak mode** (`tau_gen=None`):
- Tracks frequency stability via coefficient of variation (CV) over recent history
- Fires ALERT when CV > 20% (frequency wandering)
- Does not test the FRM prediction; does not contribute to `frm_confidence`

Reports `OUT_OF_SCOPE` when the signal has no dominant frequency (flat spectrum, peak-to-mean ratio < 3).

---

## Lambda — FRM Damping Detector

Lambda is `HopfDetector(method='frm')`. It curve-fits the amplitude envelope of the oscillation to an FRM exponential model and extracts the damping parameter λ.

- λ > 0: stable oscillation, damping present
- λ → 0: approaching Hopf bifurcation (critical slowing down in FRM physics terms)
- λ < 0: amplitude growing (post-bifurcation)

In strong mode, Lambda uses the FRM-predicted frequency as the carrier wave reference. In weak mode, it estimates the dominant frequency via FFT.

The message field reports: `frm λ=<value> rate=<dλ/dt> ttb=<estimate> ...`

---

## Customizing Detectors

All Layer 1 and Layer 2 detector parameters are accessible:

```python
suite = FRMSuite(
    tau_gen=12.5,

    # Layer 2 tuning
    lambda_kwargs={"warmup": 100, "fit_interval": 3},
    omega_kwargs={"deviation_threshold": 0.08, "alert_steps": 3},
    virtu_kwargs={"safety_factor": 1.5, "omega_trust": True},

    # Layer 1 tuning
    hopf_ews_kwargs={"window": 60, "var_threshold": 0.7},
    layer1_kwargs={
        "drift": {"cusum_k": 0.3, "cusum_h": 6.0},
        "variance": {"cusum_k": 1.0, "cusum_h": 12.0},
        "coupling": {"threshold": 0.20},
    },
)
```

---

## State Persistence

```python
import json

# Save all detector states
sd = suite.state_dict()
json_str = json.dumps(sd)

# Restore (must use same tau_gen)
suite2 = FRMSuite(tau_gen=12.5)
suite2.load_state(json.loads(json_str))

# Reset to factory state
suite.reset()
```

---

## Worked Example: EEG Alpha Rhythm Monitoring

```python
import random
import math
from fracttalix.frm import FRMSuite

# Alpha rhythm ~10 Hz sampled at 256 Hz
# tau_gen ≈ 1/(4 × 10 Hz) × 256 samples/s = 6.4 samples
suite = FRMSuite(tau_gen=6.4)

# Simulate healthy alpha oscillation
for i in range(200):
    value = math.sin(2 * math.pi * 10 / 256 * i) + random.gauss(0, 0.1)
    result = suite.update(value)
    if result.frm_confidence > 0:
        print(f"Step {i}: frm_confidence={result.frm_confidence}")

# Simulate onset of focal seizure: λ declining, frequency drifting
for i in range(200, 500):
    decay = math.exp(-0.005 * (i - 200))            # amplitude decaying
    freq_drift = 10 + 0.01 * (i - 200)              # frequency slowly rising
    value = decay * math.sin(2 * math.pi * freq_drift / 256 * i) + random.gauss(0, 0.1)
    result = suite.update(value)
    if result.frm_confidence >= 2:
        ttb_msg = result.virtu.message
        print(f"Step {i}: BIFURCATION SIGNAL  confidence={result.frm_confidence}  {ttb_msg}")
```

---

## Next Steps

- [DetectorSuite](detector_suite.md) — Layer 1 standalone (no scipy required)
- [API Reference — FRMSuite](api/frm.md) — full class and method documentation
- [Design Document](DESIGN-FRMSuite-CBT.md) — four-phase CBT v2 build plan
- [README](../README.md#frmsuite--frm-physics-layer) — overview and quick reference
