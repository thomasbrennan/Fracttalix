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

## Phase 2: Hostile Review

*Adversarial review of P4 Phase 1 build plan. All objections must be addressed
in Phase 3 before the build can proceed.*

### Objection 1: "F-4.2 (β substrate independence) is testing a tautology"

**Attack:** P3 HR-3.1 resolution established that β_measured = ω·τ_gen/π = 1/2
analytically — always, by construction. The C-3.REG R5 step extracts β = 0.5 as
a model-confirmed identity, not an empirical measurement. So F-4.2 claims to test
"β = 1/2 holds across all biological substrate classes within 2σ" — but β is
*defined* to be 1/2 by the FRM. What is being tested? The 2σ test against 0.5
will always pass because the protocol always returns 0.5. F-4.2 as stated has
zero empirical content. It's vacuous.

**Severity:** HIGH — the claim is structurally vacuous. It always returns NOT FALSIFIED
regardless of the system, violating C6.

### Objection 2: "IR numbering collision — P3's IR-12/IR-13 vs schema IR-12"

**Attack:** The P4 AI layer lists IR-12 as "Protocol Specification" and IR-13
as "Statistical Standard Anchoring" — both inherited from P3 (S48). But the
canonical schema (ai-layer-schema.json, updated S56) now defines IR-12 as
"Causal Precedence (DDE derivation class)" from DRP-8. These are completely
different inference rules with the same identifier. P3's local IR-12 and the
schema's IR-12 are incompatible. Any derivation trace citing "IR-12" is
ambiguous — does it mean Protocol Specification or Causal Precedence?

**Severity:** HIGH — IR numbering integrity is foundational. Ambiguous rule
references undermine all derivation traces that use them.

### Objection 3: "Cardiac oscillators (B3) may be an empty class"

**Attack:** The plan acknowledges that sustained pacemaking (SA node, Purkinje
fibres) is a limit cycle (μ>0) and therefore OUT of FRM scope. The only remaining
in-scope system is "cardiac action potential recovery (APD restitution)" — a niche
phenomenon that is a perturbation response, not a natural cardiac oscillation.
If the most prominent systems in a class are excluded by the scope boundary,
the class is effectively empty or trivially small. Including B3 inflates the
class count to make "5 biological substrate classes" look impressive while
potentially delivering only 1 system. This is scope dressing.

**Severity:** MEDIUM-HIGH — inflated class count weakens the cross-class analysis
(F-4.2, F-4.3). If B3 reduces to 0–1 valid systems, it should not count as a class.

### Objection 4: "Circadian class (B1) adds no new empirical content over P1"

**Attack:** P1 already CONFIRMED the mammalian circadian system (F-1.6: T = 24 hr
from τ_gen = 6 hr, no fitting) and the cyanobacterial circadian system. P4
proposes to test the same systems plus 3 more circadian oscillators. But all
circadian oscillators share the same fundamental mechanism (TTFL) — they're not
independent substrates, they're variants of the same system. Adding Drosophila,
Neurospora, and Arabidopsis circadian clocks is scope padding, not independent
validation. The "5 systems in B1" could be viewed as 1 substrate tested 5 times.

**Severity:** MEDIUM — the independence of systems within a class needs formal
justification. Different organisms ≠ independent substrates if the mechanism
is identical.

### Objection 5: "Supercompensation data is qualitative — F-4.5 may be untestable"

**Attack:** Rippetoe's Starting Strength model describes supercompensation
qualitatively: "train, recover, come back stronger." The actual published data
from exercise science consists of (a) pre/post strength measurements at discrete
time points, not continuous time series; (b) highly variable individual responses;
(c) confounded by nutrition, sleep, training history. There may be no published
dataset with sufficient temporal resolution (multiple measurements during the
recovery window at fine-grained intervals) to fit an oscillatory model.
If the data doesn't exist, F-4.5 is not empirical_pending — it's
empirical_impossible. The placeholder PH-4.1 should be flagged as potentially
blocking, not C4-tracking.

**Severity:** MEDIUM — the claim is well-motivated but may lack testable data.
Should be assessed during Phase 3 with a concrete literature survey.

### Objection 6: "n=2 minimum per class has no statistical power"

**Attack:** D-4.2 says "minimum 2 systems per class for inclusion." But F-4.2
tests β within 2σ across classes. With n=2 per class, the standard error σ_β
is computed from 2 data points — the 95% confidence interval will be enormous.
The test has essentially zero statistical power to detect a real deviation from
0.5. A class could have β = 0.3 and still pass the 2σ test because σ is
inflated by small n. This makes F-4.2 non-falsifiable in practice for small classes.

**Severity:** MEDIUM — the minimum should be raised or the predicate should
explicitly acknowledge the power limitation.

### Objection 7: "α parameter uncertainty dominates R² for most biological systems"

**Attack:** The FRM λ expression is λ ≈ |α|/(Γ·τ_gen). C-3.REG R3 specifies
α=−1 as default when α is unavailable from literature. For most biological
systems, α (the normalised distance from Hopf bifurcation) is unknown and
possibly unknowable from experimental data. Using α=−1 for all systems means
the decay rate λ is wrong by an unknown factor. This propagates directly into R²:
a wrong λ means wrong envelope, which means wrong fit. The R² test (F-4.1) is
therefore testing the quality of the α=−1 default, not the quality of the FRM.
Any system where α≠−1 could fail the R² test even though the FRM form is correct.

**Severity:** MEDIUM — the R² test conflates model adequacy with parameter
uncertainty. The note in C-3.REG R3 acknowledges this ("lower R² for systems
with unknown α is expected and correct") but F-4.1 doesn't.

### Objection 8: "P10-GAP-4.1: the 10% T_char threshold is unjustified"

**Attack:** The principle_10_audit flags this: "10% threshold for T_char deviation"
has derivation_path "Conservative threshold for biological variability" but no
formal anchor. The derivation_path says "Natural circadian period ranges
23.5–24.5 hr (±2%) in controlled conditions. 10% = 5× natural variation."
But this logic is circular — it derives the threshold from circadian data and
then applies it to all biological classes. What if cell cycle period variability
is ±20%? Then 10% is too tight and will falsely reject. The threshold needs
to be anchored at a class-independent standard, or be class-specific.

**Severity:** MEDIUM — P10-GAP-4.1 is correctly flagged but the proposed
resolution ("exercise science measurement precision standards") is vague.
Needs a concrete anchoring strategy.

### Objection 9: "Alternative model comparison (F-4.4) is methodologically unfair"

**Attack:** The pre-specified alternative models have fitted parameters:
Goodwin oscillator (multiple), Novak-Tyson ODE (10+), FitzHugh-Nagumo (several),
Goldbeter allosteric (multiple). The FRM has zero fitted parameters. Comparing
R² directly between a zero-parameter model and a multi-parameter model is
methodologically asymmetric — the fitted model will *always* have higher R²
given enough parameters. The threshold Δ≥−0.05 is generous to FRM but the
comparison is still fundamentally flawed. The paper should use AIC or BIC
(which penalise parameter count) or explicitly state that this comparison
tests whether the FRM's *derived* form competes with *fitted* alternatives,
which is a stronger result than R² alone suggests.

**Severity:** LOW-MEDIUM — the comparison is valid as a practical test (does
FRM keep up?) but the interpretation must be framed correctly. The zero-parameter
advantage is a *feature* not a confounder.

### Objection 10: "Build Table inconsistency — P2/P3 status"

**Attack:** The Build Table (v3.3) says P2 is "Phase 1 PHASE-READY (S49)" and
P3 is "QUEUED." But the S55 handoff says both are PHASE-READY, and the
process-graph has P3 as PHASE-READY with AI layer v2 (S48). P4 claims to receive
live edges from P2 (C-2.1, C-2.4) and P3 (C-3.REG). If the Build Table is
correct that P2 is only Phase 1 PHASE-READY (not fully PHASE-READY), then
P2's claims are provisional and should not be received as stable live edges.
The Build Table R-1 still says "P3 is QUEUED" and "Gate NOT yet open." Which
source is authoritative? The handoff, the process-graph, or the Build Table?

**Severity:** HIGH — if P2 or P3 are not fully PHASE-READY, P4's inbound edge
register is built on unverified foundations. This must be reconciled before
P4 can proceed.

---

*Phase 2 produced S56. 10 objections. Phase 3 (Second Meta-Kaizen) pending.*
