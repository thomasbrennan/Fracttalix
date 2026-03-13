# P4 Biological Systems — Canonical Build Process

**Session:** S56
**Date:** 2026-03-13
**Process:** Canonical Build (P0 CBT v2)
**Author:** Thomas Brennan · with Claude (Anthropic)

---

## Phase 1: First Build Plan

### 1.1 Paper Identity

| Field | Value |
|-------|-------|
| Paper ID | P4 |
| Title | Biological Systems: FRM Validation Across Biological Oscillatory Substrates |
| Type | application_C |
| Act | II |
| Track | Fracttalix |
| Status | Phase 1 IN BUILD |
| Gate | P3 PHASE-READY ✓ (S48) |
| Depends on | P1 (D-1.1, D-1.2, D-1.3, F-1.4, F-1.6, A-1.5), P2 (D-2.1, C-2.1, C-2.2, C-2.4), P3 (C-3.REG, D-3.1, D-3.2, C-3.ALT, C-3.DIAG) |
| Enables | P6 (integration consistency — Act II claims) |

### 1.2 Core Question

Does the FRM functional form f(t) = B + A·e^(−λt)·cos(ωt + φ) accurately describe
the dynamics of biological oscillatory systems across multiple independent biological
substrate classes, when measured using the P3 standard protocol with zero free parameters?

### 1.3 Thesis (Stated As Falsifiable Claim)

**F-4.1:** The FRM predicts the dynamics of biological oscillators with R²≥0.85
(per C-3.REG R4 threshold) across at least 3 independent biological substrate classes,
with β=1/2 confirmed in each class and T_char = 4·τ_gen matching observed periods,
using zero fitted parameters.

**Falsification condition:** Any biological substrate class satisfying D-2.1 class
criteria where mean R²_frm < 0.85 across ≥3 systems in that class, measured per
P3 C-3.REG protocol.

### 1.4 Scope Boundary (P4 vs P5)

P4 scope: non-neural biological oscillators. All biological systems where the
oscillatory dynamics arise from delayed negative feedback at the cellular,
tissue, organ, or organism level — excluding neural circuits.

P5 scope: neural and pharmacological systems. Neural oscillators (gamma, theta,
alpha rhythms), synaptic feedback circuits, drug response kinetics.

The boundary is substrate-level: if the primary feedback mechanism is synaptic
(neurotransmitter-mediated), the system belongs in P5. If the primary feedback
mechanism is molecular (gene expression, protein concentration, ion channel),
the system belongs in P4 even if it occurs in neural tissue (e.g., circadian
clock in SCN is P4, not P5, because the oscillator is molecular).

### 1.5 Biological Substrate Classes

Five biological substrate classes are proposed for P4 validation:

**Class B1: Circadian oscillators**
- Mammalian SCN circadian clock (τ_gen = 6 hr, CONFIRMED P1)
- Cyanobacterial KaiABC oscillator (τ_gen ≈ 5.5 hr, CONFIRMED P1)
- Drosophila per/tim oscillator
- Neurospora FRQ oscillator
- Plant circadian (Arabidopsis CCA1/LHY)
- Key property: molecular delayed negative feedback (transcription-translation feedback loop, TTFL)
- τ_gen extraction: STRUCTURAL (published gene expression delay times)
- Expected: strong confirmation. P1 F-1.6 already predicts T = 24 hr from τ_gen = 6 hr.

**Class B2: Cell cycle oscillators**
- Xenopus laevis embryonic cell cycle (τ_gen = 30 min, P3 vacuity witness)
- Budding yeast Saccharomyces cerevisiae
- Fission yeast Schizosaccharomyces pombe
- Key property: CDK/cyclin delayed negative feedback (APC-mediated degradation)
- τ_gen extraction: STRUCTURAL (CDK1/APC feedback delay)
- Expected: strong confirmation. Well-characterised oscillatory dynamics.

**Class B3: Cardiac oscillators**
- Sinoatrial node pacemaker cells
- Purkinje fiber oscillations
- Cardiac action potential recovery (APD restitution)
- Key property: ion channel delayed negative feedback (voltage-gated K+ channels)
- τ_gen extraction: STRUCTURAL (refractory period / ion channel recovery time)
- Expected: confirmation with scope boundary note — limit-cycle behaviour (μ>0) in sustained pacemaking may place some regimes OUT of FRM scope. Sub-critical perturbation response (post-stimulus recovery) is IN scope.

**Class B4: Metabolic oscillators**
- Yeast glycolytic oscillations (PFK allosteric feedback)
- Calcium oscillations (IP3R-mediated)
- Key property: enzymatic delayed negative feedback (product inhibition, allosteric regulation)
- τ_gen extraction: STRUCTURAL (enzymatic reaction delay) or SPECTRAL (from oscillation period)
- Expected: partial confirmation. Some metabolic oscillators may be limit cycles (μ>0, OUT of scope).

**Class B5: Musculoskeletal adaptation**
- Supercompensation in resistance training (Rippetoe/Selye model)
- Wound healing temporal dynamics
- Bone remodelling (RANKL/OPG feedback)
- Key property: delayed negative feedback at tissue/organism level (stress → adaptation → recovery)
- τ_gen extraction: STRUCTURAL (recovery time from published exercise science literature)
- Expected: confirmation. Rippetoe's stress-adaptation cycle is a textbook example of damped oscillatory recovery following perturbation. Build Table explicitly cites this.

### 1.6 Inbound Edges Register

| Edge ID | Source | Claim | Label | Status | Received As |
|---------|--------|-------|-------|--------|-------------|
| P1→P4.1 | P1 | D-1.1 | FRM functional form | LIVE | Type A axiom |
| P1→P4.2 | P1 | D-1.2 | ω = π/(2·τ_gen) | LIVE | Type A axiom |
| P1→P4.3 | P1 | D-1.3 | λ ≈ \|α\|/(Γ·τ_gen) | LIVE | Type A axiom |
| P1→P4.4 | P1 | F-1.4 | β = 1/2 (Hopf quarter-wave) | LIVE | Type A axiom |
| P1→P4.5 | P1 | F-1.6 | T = 4·τ_gen circadian prediction | LIVE | Prior confirmation |
| P1→P4.6 | P1 | A-1.5 | Substrate independence | LIVE | Type A axiom |
| P2→P4.1 | P2 | D-2.1 | FRM universality class criteria | LIVE | Type A axiom |
| P2→P4.2 | P2 | C-2.1 | β = 1/2 derivation validity | LIVE | Prior result |
| P2→P4.3 | P2 | C-2.2 | Universality class membership | LIVE | Prior result |
| P2→P4.4 | P2 | C-2.4 | Substrate independence (analytic) | LIVE | Prior result |
| P3→P4.1 | P3 | C-3.REG | FRM regression protocol R1–R9 | LIVE | Measurement standard |
| P3→P4.2 | P3 | D-3.1 | System eligibility criteria | LIVE | Definition |
| P3→P4.3 | P3 | D-3.2 | τ_gen extraction protocol | LIVE | Definition |
| P3→P4.4 | P3 | C-3.ALT | Alternative model comparison | LIVE | Protocol |
| P3→P4.5 | P3 | C-3.DIAG | Scope boundary diagnostics | LIVE | Protocol |

### 1.7 Planned Claim Registry

**Type A (received axioms):**

| Claim ID | Label | Source |
|----------|-------|--------|
| A-4.1 | FRM functional form | P1 D-1.1 |
| A-4.2 | Universal constants (β, k*, Γ) | P1 F-1.4, F-1.5 |
| A-4.3 | Class membership criteria | P2 D-2.1 |
| A-4.4 | Measurement protocol | P3 C-3.REG |

**Type D (definitions):**

| Claim ID | Label | Content |
|----------|-------|---------|
| D-4.1 | Biological substrate class definition | Scope: non-neural biological oscillators with delayed negative feedback. Excludes neural circuits (P5), pharmacological response (P5), non-biological substrates. |
| D-4.2 | Biological validation set | Five classes (B1–B5) with named systems and τ_gen extraction sub-protocols. Pre-specified before data analysis. |
| D-4.3 | Biological τ_gen instantiation | Operationalises P3 D-3.2 for each biological class: which sub-protocol applies, what published delay time to use. |

**Type F (falsifiable claims):**

| Claim ID | Label | Statement |
|----------|-------|-----------|
| F-4.1 | Biological FRM goodness of fit | FRM fits biological oscillators with mean R²≥0.85 across all tested substrate classes (per C-3.REG R4). |
| F-4.2 | Biological β substrate independence | β = 1/2 holds across all five biological substrate classes within 2σ (per C-2.4 criterion, C-3.σ protocol). |
| F-4.3 | Biological T_char prediction | T_char = 4·τ_gen matches observed oscillation periods in biological systems within 10% across all tested classes. |
| F-4.4 | Biological alternative model comparison | FRM performs within Δ≥−0.05 of best alternative model for each biological class (per C-3.ALT). |
| F-4.5 | Supercompensation as FRM instance | The Rippetoe/Selye stress-adaptation supercompensation curve follows f(t) = B + A·e^(−λt)·cos(ωt + φ) with τ_gen = published recovery time, R²≥0.85, and T_char matching observed supercompensation period. |

**Planned claim total:** 12 (4A + 3D + 5F)

### 1.8 Prior Art — Biological Oscillation Modeling

P4 must survey existing biological oscillation models to establish novelty:

**Key contrast sources (pre-specified):**

| Source | Domain | Contrast with FRM |
|--------|--------|-------------------|
| Novak & Tyson (1993, 2008) | Cell cycle | ODE models with fitted parameters. FRM: zero free parameters. |
| Goldbeter (1996, 2002) | Biochemical oscillations | Detailed mechanistic ODE models. FRM: functional form from universality class, not mechanism. |
| Gonze et al. (2005) | Circadian | Multi-variable ODE models (Goodwin oscillator). FRM: single functional form. |
| Rippetoe (2011) | Strength training | Qualitative stress-adaptation model. FRM: quantitative, predictive. |
| Glass & Mackey (1988) | DDE in biology | DDE theory applied to biological oscillations. FRM: specific universality prediction (β=1/2) not in Glass & Mackey. |
| Selye (1936, 1976) | General adaptation | GAS model (qualitative). FRM: mathematical form with derived constants. |
| Winfree (2001) | Biological timing | Topological analysis of biological oscillations. FRM: metric, not topological. |

**Novelty claim:** No prior work derives biological oscillation dynamics from a universality-class argument with zero free parameters and substrate-independent critical exponents. Existing models are either mechanistic (fitted parameters per system) or qualitative (no quantitative predictions).

### 1.9 Paper Structure

| # | Section | Content |
|---|---------|---------|
| 1 | Introduction | FRM applied to biology. Core question. Gate from P3. |
| 2 | Scope and definitions | D-4.1 scope boundary (P4 vs P5). D-4.2 validation set. D-4.3 τ_gen instantiation. |
| 3 | Methods | P3 C-3.REG protocol application. τ_gen extraction per class. Alternative model selection (C-3.ALT Section 5 mapping). |
| 4 | Results: Circadian oscillators (B1) | Per-system fits. R²_frm. T_char vs T_obs. Alternative model comparison. |
| 5 | Results: Cell cycle oscillators (B2) | Per-system fits. R²_frm. T_char vs T_obs. Alternative model comparison. |
| 6 | Results: Cardiac oscillators (B3) | Per-system fits. Scope boundary analysis (μ<0 vs μ>0 regimes). |
| 7 | Results: Metabolic oscillators (B4) | Per-system fits. Scope boundary analysis. |
| 8 | Results: Musculoskeletal adaptation (B5) | Supercompensation fitting. Rippetoe model → FRM mapping. |
| 9 | Cross-class analysis | F-4.2 (β independence). F-4.3 (T_char prediction). Summary table. |
| 10 | Discussion | Scope boundaries confirmed. Anomalous cases flagged. Implications for P5 and P6. |
| 11 | Claim registry | Full AI layer embedded. |

### 1.10 Deliverables

1. P4-ai-layer.json — Phase 1 build (NOT-PHASE-READY)
2. P4-build-process.md — this document (updated through all phases)
3. Updated Build Table entry for P4

### 1.11 Note on Existing P4 AI Layer (S48 Retroactive Build)

The existing P4-ai-layer.json (v1, S48) was a retroactive build with incorrect framing:
- **Wrong title:** "Mathematical Formalization of Structure and Rhythmicity" (should be "Biological Systems")
- **Wrong paper_type:** derivation_B (should be application_C)
- **Wrong claims:** Generic self-similarity claims, not biological application claims
- **Note in file:** "Retroactive AI layer. Paper published but pending framing review."

This Phase 1 build **replaces** the S48 retroactive AI layer entirely. The S48 claims
(A-4.1 self-similarity, D-4.1 fractal property, D-4.2 scale-invariant parameters,
D-4.3 mathematical structure, F-4.1 scale-invariance of β, F-4.2 fractal-like behavior)
are superseded. Self-similarity is a property of the FRM law (P1), not a P4-specific claim.

### 1.12 Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Cardiac oscillators may be limit cycles (μ>0) | MEDIUM | Pre-specify scope check: sustained pacemaking = OUT (μ>0). Post-perturbation recovery = IN (μ<0). C-3.DIAG ANOMALOUS classification handles edge cases. |
| Metabolic oscillators: insufficient data for full validation | LOW | Minimum 3 systems per class for class-level β test. Fewer → EXCLUDED class (registered, not falsifying). |
| Supercompensation data may be qualitative only | MEDIUM | Quantitative datasets from exercise science literature (strength gains over recovery periods). If insufficient quantitative data: PH-4.X placeholder, non-blocking. |
| Overlap with P5 neural scope | LOW | Scope boundary precisely defined (Section 1.4). Molecular oscillators in neural tissue → P4. Synaptic oscillators → P5. |

---

*Phase 1 produced S56. Phases 2–5 pending.*
