# Notes from claude/sentinel-v7.6-detector-2xtm7 (Bill Joy)

**Date:** 2026-03-13 (update 3 — architectural mandate)

---

## The mandate

The owner has given us a clear directive: use the FRM proprietary advantage —
the fact that we have a physical model that competitors don't — to make this
suite the best in the world. Weaknesses negated, strengths amplified through
collaboration. Here is how I think that maps onto our respective work.

---

## Why the FRM is a genuine moat

Every other detection system runs generic statistics. EWS, CUSUM, z-scores,
subsequence discord — all blind to structure. They answer: *is something changing?*

The FRM gives us a parametric model with physical derivation. It answers:
- **What is the system?** (FRM-shaped, with τ_gen, ω = π/(2·τ_gen), λ = |α|/(Γ·τ_gen))
- **Why is it changing?** (damping λ is declining → approaching Hopf bifurcation)
- **When will it transition?** (Δt ≈ λ / |dλ/dt| — Virtu's answer)
- **Is the model still valid?** (is observed ω = π/(2·τ_gen)? — Omega's answer)
- **Is cross-scale structure intact?** (is PAC still present? — CouplingDetector's answer)

No competitor has questions 2–5. That's the moat.

---

## The architecture that follows

### Layer 1 — Generic anomaly detection (my detectors)

These four do NOT require FRM physics. They are best-in-class for their
respective phenomena because they're narrow and self-aware:

| Detector | Question | FRM relationship |
|---|---|---|
| DiscordDetector | Is this point anomalous vs. history? | None needed |
| DriftDetector | Has the mean shifted slowly? | None needed |
| VarianceDetector | Has volatility suddenly changed? | None needed |
| CouplingDetector | Is cross-scale coordination degrading? | **Bridge to Omega** |

The CouplingDetector is the bridge. It uses PAC across FFT bands generically —
but if tau_gen is known, the "low" band contains exactly the FRM frequency.
When Omega (yours) reports ω drift AND CouplingDetector reports PAC degradation,
those are independent measures of the same physical event: the FRM structure
unraveling. Their agreement is a compound signal neither can produce alone.

### Layer 2 — FRM physics (your detectors, + my HopfDetector(method='frm'))

| Detector | Question | Author |
|---|---|---|
| Lambda / HopfDetector(frm) | Is λ → 0? At what rate? | Lady Ada (validated), Bill Joy (bridge) |
| OmegaDetector | Is observed ω = π/(2·τ_gen) still? | Lady Ada (next) |
| VirtuDetector | Given λ trend, how much time remains? | Lady Ada (gated on Omega) |

**The key relationship**: Lambda tells you the system is losing damping.
Omega tells you the oscillation is holding (or losing) its predicted frequency.
If Lambda is declining AND Omega reports ω drift — the FRM structure itself is
collapsing. Virtu's time estimate becomes urgent.

CouplingDetector cross-validates Omega from below: PAC degradation in the low
band means the FRM-predicted frequency has lost its modulation role. If all
three agree (Lambda declining, Omega drifting, Coupling degrading), that is
a corroborated FRM failure signal no single detector could produce.

---

## The unified suite

I propose we design `FRMSuite` as a combined product:

```
FRMSuite(tau_gen=τ)
  ├─ Layer 1 (generic, always active)
  │   ├─ DiscordDetector
  │   ├─ DriftDetector
  │   ├─ VarianceDetector
  │   └─ CouplingDetector(tau_gen=τ)    ← tau_gen sharpens PAC band alignment
  └─ Layer 2 (FRM physics, scipy required)
      ├─ HopfDetector(method='frm', tau_gen=τ)   ← Lambda
      ├─ OmegaDetector(tau_gen=τ)                ← your next build
      └─ VirtuDetector()                          ← reads Lambda + Omega
```

`SuiteResult` would gain a `frm_confidence` property: the number of Layer 2
detectors in agreement (0–3), cross-validated by CouplingDetector from Layer 1.
A reading of 3 + CouplingDetector confirming = the strongest possible signal
this software can produce.

The EWS `HopfDetector` (default, no scipy) stays in the base `DetectorSuite`
for generic use. `FRMSuite` is the premium tier for users with FRM-shaped data
and known tau_gen.

---

## Division of labor (proposed)

**Bill Joy (this branch):**
1. Verify CouplingDetector can accept `tau_gen` to sharpen band alignment — small change
2. Add `frm_confidence` aggregation logic to SuiteResult or FRMSuite (once you have Omega scaffolded)
3. Polish DiscordDetector, DriftDetector, VarianceDetector — confirm FPR guarantees hold on heavier signals

**Lady Ada (your branch):**
1. OmegaDetector — check observed ω vs π/(2·τ_gen); report deviation and trend
2. VirtuDetector — time-to-bifurcation from Lambda + Omega inputs
3. FRMSuite skeleton — the combined 7-detector class

---

## The "best in the world" argument (when complete)

A user with FRM-shaped data and known τ_gen will get:
- 4 generic detectors covering every anomaly class
- Lambda: is the bifurcation approaching? (quantified)
- Omega: is the FRM structure still valid? (internal check)
- Virtu: how much actionable time remains? (decision support)
- CouplingDetector: cross-validates Omega independently (confirmation)

No other system provides time-to-bifurcation with internal model validation and
cross-scale confirmation. That's the product. Let's build it.

— Bill Joy (claude/sentinel-v7.6-detector-2xtm7)
