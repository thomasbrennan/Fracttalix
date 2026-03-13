# FRM Detection Suite — CBT v2 Build Plan

**Session:** 2026-03-13
**Authors:** Bill Joy (sentinel branch) · Lady Ada (lambda branch)
**Process:** Canonical Build Track v2 (four phases)
**Owner mandate:** Best-in-class modular suite derived from FRM physics.
  Retirement gate: FPR benchmark + honest scope, both required.

---

## Phase 1: First Build Plan

### 1.1 Suite Identity

| Field | Value |
|-------|-------|
| Suite ID | FRMSuite v1.0 |
| Class | Anomaly + bifurcation detection for oscillatory time series |
| Architecture | Two-layer modular: generic screening + FRM physics |
| Replaces | SentinelDetector (37-step monolith) — conditional on passing retirement gate |
| Depends on | Fractal Rhythm Model (P1–P6, DRP series); scipy optional (Layer 2 only) |

### 1.2 Core Design Goals

1. **Honest scope**: every detector reports OUT_OF_SCOPE when its model doesn't fit. No false verdicts from inapplicable methods.
2. **Modular**: each detector is independently usable. Failure of one does not corrupt others.
3. **FRM-native physics layer**: where FRM applies, detectors use derived constants (ω = π/(2·τ_gen), λ = |α|/(Γ·τ_gen)) — not curve-fit parameters.
4. **Best-in-class per domain**: each detector must outperform the corresponding capability in SentinelDetector on its target phenomenon.
5. **Quantified confidence**: compound signals yield `frm_confidence` — number of independent FRM detectors in agreement.

### 1.3 Component Specification

#### Layer 1 — Generic (no scipy, always active)

| Component | Question answered | Null FPR target | Status |
|-----------|-------------------|-----------------|--------|
| HopfDetector(ews) | Is critical slowing down occurring? | 0% on N(0,1) | DONE |
| DiscordDetector | Is this point anomalous vs. its context? | ≤ 1% on N(0,1) | DONE |
| DriftDetector | Has the mean shifted slowly over time? | ≤ 0.5% on N(0,1) | DONE |
| VarianceDetector | Has volatility suddenly changed? | ≤ 1% on N(0,1) | DONE |
| CouplingDetector | Is cross-scale coordination degrading? | 0% on N(0,1) | DONE (FP fix) |

#### Layer 2 — FRM Physics (scipy required)

| Component | Question answered | Null FPR target | Status |
|-----------|-------------------|-----------------|--------|
| HopfDetector(frm) = Lambda | Is damping λ declining toward 0? | ≤ 10% on limit cycles | DONE (validated) |
| OmegaDetector | Is observed ω = π/(2·τ_gen) still intact? | ≤ 5% on stationary oscillations | NEXT (Lady Ada) |
| VirtuDetector | How much actionable time remains? | N/A (time estimate, not binary) | GATED on Omega |

#### Orchestration

| Component | Purpose | Status |
|-----------|---------|--------|
| DetectorSuite | Runs all 5 Layer 1 detectors in parallel | DONE |
| FRMSuite | Runs all 8 detectors; exposes frm_confidence | NEXT (Bill Joy skeleton) |

### 1.4 Falsifiable Claims (Type F)

| ID | Claim | Falsification condition |
|----|-------|------------------------|
| F-S1 | All Layer 1 detectors: FPR ≤ target on N(0,1) white noise (N=1000) | Any detector exceeds its null FPR target on a fresh random seed |
| F-S2 | Lambda: FPR ≤ 10% on Melbourne-class sustained oscillation (λ≈0, stable) | Alert rate on sinusoid > 10% after fix |
| F-S3 | OmegaDetector: detects ω drift of ≥ 5% within 100 steps of onset | Fails to ALERT on synthetic ω-shift signal within window |
| F-S4 | VirtuDetector: time-to-bifurcation estimate within 2× of true value on synthetic FRM decay | Estimate > 2× true value on ≥ 3/5 synthetic trials |
| F-S5 | frm_confidence=3 requires Lambda + Omega + CouplingDetector all in ALERT simultaneously | frm_confidence=3 achievable without all three firing |
| F-S6 | FRMSuite FPR ≤ SentinelDetector FPR on shared benchmark signals | FRMSuite produces more false positives than Sentinel on any benchmark signal |
| F-S7 | FRMSuite detects ≥ 90% of true positives detected by SentinelDetector on known-anomaly signals | FRMSuite misses > 10% of what Sentinel catches |
| F-S8 | FRMSuite provides time-to-bifurcation estimate that SentinelDetector cannot provide | SentinelDetector produces equivalent time estimate |

### 1.5 Benchmark Suite (Pre-specified)

The retirement gate requires a benchmark suite defined *before* implementation, not after.

**Null signals (FPR testing):**
1. N(0,1) white noise, N=1000
2. Sustained sinusoid f=0.10, N=500 (limit cycle test)
3. Random walk (Brownian motion), N=500
4. Slow trend (linear drift), N=500

**Signal signals (detection testing):**
5. Synthetic Hopf approach: sinusoid with λ declining from 0.5 → 0 over 300 steps
6. Sudden mean shift: N(0,1) → N(3,1) at step 200
7. Variance explosion: N(0,1) → N(0,5) at step 200
8. ω drift: sinusoid with frequency shifting ±10% over 200 steps (OmegaDetector target)
9. Cross-scale coupling collapse: PAC signal with coupling coefficient declining (CouplingDetector target)
10. Subsequence anomaly: injected discord subsequence in stationary signal

**Comparison format:**
For each signal: SentinelDetector alert rate vs. FRMSuite alert rate.
On null signals: lower is better. On signal signals: higher is better.
FRMSuite "plainly outperforms" = lower FPR on all 4 null signals + ≥ 90% detection on signals 5–10.

### 1.6 Dependency Structure

```
FRMSuite
├─ DetectorSuite (Layer 1)
│   ├─ HopfDetector(ews)
│   ├─ DiscordDetector
│   ├─ DriftDetector
│   ├─ VarianceDetector
│   └─ CouplingDetector
└─ FRMLayer (Layer 2, optional — degrades gracefully without scipy)
    ├─ HopfDetector(frm) = Lambda
    ├─ OmegaDetector          ← Lady Ada builds
    └─ VirtuDetector          ← Lady Ada builds (gated on Omega)
```

---

## Phase 2: Hostile Review

### Objection 1: "FRMSuite adds complexity; the 5-detector suite was clean"

**Severity: MEDIUM**

The 5-detector suite has a clear philosophy: parallel, independent, no blending. FRMSuite introduces a "frm_confidence" aggregation and a two-layer architecture. This is more complex. The user now has to understand which layer is active.

### Objection 2: "When tau_gen is unknown, Layer 2 is half-blind"

**Severity: HIGH**

OmegaDetector has two modes: strong (tau_gen supplied → check ω = π/(2·τ_gen)) and weak (tau_gen from FFT → track frequency stability). These are very different products. The strong mode is FRM-native. The weak mode is generic frequency tracking. If you claim FRM physics as the moat, the weak mode doesn't deliver it. Don't conflate them.

### Objection 3: "frm_confidence=3 may never fire in practice"

**Severity: HIGH**

Requiring Lambda + Omega + CouplingDetector all in ALERT simultaneously is three coincident signals. Lambda needs λ declining; Omega needs ω drifting; Coupling needs PAC degrading. Do these actually co-occur on real signals? Or is frm_confidence=3 a theoretically beautiful but empirically empty metric?

### Objection 4: "Retirement of Sentinel is premature — it handles things FRMSuite doesn't"

**Severity: HIGH**

SentinelDetector's 37 steps were built over time to handle specific real-world failure modes. Some of those 37 steps cover phenomena that FRMSuite's 8 detectors may not. Before retiring Sentinel, we need to know: what does Sentinel detect that FRMSuite misses? If the answer is "substantial things," retirement is wrong — coexistence is right.

### Objection 5: "The Layer 1 detectors are good but not individually 'best in class'"

**Severity: MEDIUM**

"Best in class" for CUSUM drift detection means beating the literature's CUSUM implementations. For subsequence discord it means beating MATRIX PROFILE. We haven't benchmarked against external implementations — only against SentinelDetector. That's not the same bar.

### Objection 6: "FPR targets were set without reference to real-world base rates"

**Severity: MEDIUM**

A 1% FPR on N(0,1) white noise sounds good. But if real signals are 90% stationary and 10% anomalous, even 1% FPR produces more false positives than true positives. The FPR targets need to be set relative to expected base rates in the target use case, not just against white noise.

### Objection 7: "The 'compound signal' story (frm_confidence) is correlation, not independence"

**Severity: HIGH**

Lambda, Omega, and CouplingDetector all look at the same underlying signal. If a signal has declining λ, it very likely also has shifting ω (they're coupled by FRM physics). This means frm_confidence=3 is not "3 independent confirmations" — it's "3 correlated views of the same structural change." The independence claim in the moat argument is overstated.

### Objection 8: "No streaming performance benchmark"

**Severity: LOW-MEDIUM**

Accuracy matters, but so does throughput. The 37-step monolith has known performance characteristics. FRMSuite with scipy curve_fit running every 5 steps may be slower. If the use case is real-time streaming, speed matters. We haven't specified performance requirements.

---

## Phase 3: Meta-Kaizen

| # | Objection | Response | Effect |
|---|-----------|----------|--------|
| 1 | Complexity increase | The complexity is layered: Layer 1 alone is as clean as before. Layer 2 is opt-in (requires scipy, degrades gracefully without it). The API surface for simple users is identical: `DetectorSuite()`. FRMSuite is a separate class for users who need FRM physics. | **Resolved** — maintain DetectorSuite as the simple product; FRMSuite is the premium tier |
| 2 | tau_gen unknown / weak mode | Explicitly document: OmegaDetector **strong mode** (tau_gen supplied) is FRM-native and part of the moat. **Weak mode** (FFT-estimated) is generic frequency tracking — useful, but not the differentiator. frm_confidence only increments when ALL detectors are in strong mode. | **Resolved** — strong/weak mode distinction becomes first-class in the API and docs |
| 3 | frm_confidence=3 may not fire | This is an empirical question. We will test on synthetic FRM approach signals. If frm_confidence=3 never fires on real-world signals, it's a design problem — either the thresholds are wrong or the phenomena don't actually co-occur. The benchmark suite (signal 5: synthetic Hopf approach) will answer this before release. | **Empirical gate added** — frm_confidence=3 must fire on at least 3 of 5 synthetic approach signals or threshold design must be revised |
| 4 | Retirement premature | Agree. The right process: build FRMSuite, run head-to-head benchmark, list what Sentinel catches that FRMSuite misses. If the miss list is non-empty and significant, coexistence is the answer. Only retire if: (a) FRMSuite passes all F-S1–F-S8 claims, AND (b) the miss analysis shows Sentinel catches nothing that FRMSuite doesn't. | **Discipline enforced** — add Miss Analysis to retirement decision process |
| 5 | "Best in class" bar | The correct framing: FRMSuite detectors are best-in-class *within the FRM-native constraint*. We are not claiming to outperform MATRIX PROFILE on all subsequence anomaly detection — we are claiming the best MATRIX PROFILE-style detection that (a) is honest about scope and (b) integrates with FRM physics. The claim is: best in class for the FRM use case, not universally best for all time series. | **Scope refined** — "best in class" means "best for FRM-shaped oscillatory streams with known tau_gen" not "universally best for all signals" |
| 6 | FPR vs. base rates | Agreed. The null FPR targets are necessary but not sufficient. We add: expected precision-recall operating point given typical 5% anomaly base rate. With 1% FPR and 5% base rate: PPV = P(true|alert). This should be > 0.5 to be operationally useful. | **Metric added** — benchmark will report PPV at 5% base rate alongside FPR |
| 7 | Correlated signals / not independent | Partially true. Lambda and Omega ARE correlated by FRM physics (declining λ and shifting ω co-occur at bifurcation). But CouplingDetector is NOT correlated — it uses PAC across FFT bands, not FRM model fitting. So frm_confidence=3 is: 2 correlated FRM signals + 1 independent structural signal. The independence claim should be "partially independent" not "fully independent." This is still more informative than frm_confidence=1. | **Claim corrected** — frm_confidence=3 represents "two FRM-physics signals plus one independent structural confirmation" — not three fully independent signals |
| 8 | Streaming performance | Add: `FRMSuite.update()` must complete in < 50ms per observation on standard hardware (Python, no GPU). The scipy curve_fit is the bottleneck. Existing frm_fit_interval=5 (fit every 5 steps) is the mitigation. Benchmark will measure average and worst-case update time. | **Performance gate added** — < 50ms average, < 200ms worst-case per update |

---

## Phase 4: Final Build Plan

### Revised Architecture

```
fracttalix/
├── suite/                   ← DetectorSuite (5 detectors, no scipy) — unchanged
│   ├── hopf.py              ← EWS + FRM method (done)
│   ├── discord.py, drift.py, variance.py, coupling.py
│   └── suite.py             ← DetectorSuite
└── frm/                     ← NEW: FRM physics layer
    ├── __init__.py
    ├── lambda_.py            ← HopfDetector(frm) wrapper (or re-export)
    ├── omega.py              ← OmegaDetector (Lady Ada)
    ├── virtu.py              ← VirtuDetector (Lady Ada, gated)
    └── frm_suite.py          ← FRMSuite orchestrator
```

`FRMSuite` lives in `fracttalix.frm`. It wraps `DetectorSuite` + the FRM physics detectors.

### Revised Claims (Post-Hostile-Review)

| ID | Claim (revised) | Change |
|----|-----------------|--------|
| F-S1 | Layer 1 FPR targets (unchanged) | — |
| F-S2 | Lambda FPR ≤ 10% on limit cycles (unchanged) | — |
| F-S3 | OmegaDetector **strong mode**: ω drift ≥ 5% detected within 100 steps | Added "strong mode" qualifier |
| F-S4 | VirtuDetector TTB within 2× on synthetic (unchanged) | — |
| F-S5 | frm_confidence=3 fires on ≥ 3/5 synthetic Hopf approach signals | Empirical gate added |
| F-S6 | FRMSuite FPR ≤ SentinelDetector FPR (unchanged) | — |
| F-S7 | FRMSuite detection ≥ 90% of Sentinel on known-anomaly signals (unchanged) | — |
| F-S8 | FRMSuite provides TTB estimate Sentinel cannot (unchanged) | — |
| F-S9 | FRMSuite.update() < 50ms average, < 200ms worst-case | Performance gate added |
| F-S10 | PPV > 0.5 at 5% base rate for each alerting detector | Operational utility gate added |

### Division of Labor

**Bill Joy (this branch) — immediate:**
1. Create `fracttalix/frm/` package structure
2. Implement `FRMSuite` skeleton (wraps DetectorSuite + Layer 2 slots)
3. Implement `frm_confidence` aggregation
4. Implement sandbox benchmark harness (`benchmark/frm_suite_sandbox.py`)
5. Run all null FPR tests (signals 1–4) against both FRMSuite and SentinelDetector

**Lady Ada (lambda branch) — once she reads this:**
1. Implement `OmegaDetector` — strong mode first (tau_gen required), weak mode optional
2. Implement `VirtuDetector` — reads Lambda + Omega outputs, reports TTB
3. Validate both against signals 7–9 in the benchmark suite
4. Update collab thread when ready to integrate

**Joint (after both are ready):**
1. Integrate into `FRMSuite`
2. Run full 10-signal benchmark against SentinelDetector
3. Run Miss Analysis (what does Sentinel catch that FRMSuite misses?)
4. Apply retirement gate
5. PR to main

### Retirement Decision Process

1. Run benchmark → FRMSuite must pass F-S1 through F-S10
2. Miss Analysis → list signals where Sentinel alerts but FRMSuite does not
3. If miss list is empty or minor: retire Sentinel, merge FRMSuite to main
4. If miss list is significant: add detectors, or document coexistence rationale
5. Either outcome: document the decision in a RETIREMENT-DECISION.md with data

---

*CBT v2 process complete. Four phases applied. Ready to implement.*
