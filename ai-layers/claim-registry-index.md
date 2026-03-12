# Fracttalix Claim Registry Index v1.1

**Generated:** 2026-03-12 (Session 53)
**Purpose:** Read-only cross-project citation reference
**Source corpus:** Fracttalix + Meta-Kaizen
**DOI:** 10.5281/zenodo.18859299

## Usage

This file is a **stable snapshot** of all claim IDs in the Fracttalix corpus.
External projects (e.g. the DRP series) cite Fracttalix claims by ID from this index.
Update this file only when Fracttalix claim IDs change (which should not happen post-PHASE-READY).

## Format

```
CLAIM_ID         PAPER_ID     TYPE           LABEL
```

Types: **A** = Axiom, **D** = Definition, **F** = Falsifiable, **C** = Claim/Conjecture

---

### DRP-1: Dual-Reader Scientific Publishing: A Framework for Machine-Verifiable Knowledge Corpora
**Status:** PHASE-READY

```
C-DRP.1          DRP-1        Falsifiable    Syntax Necessity and Sufficiency
C-DRP.2          DRP-1        Falsifiable    Gate Enforcement Requirement
C-DRP.3          DRP-1        Definition     DRS Self-Sufficiency Definition
C-DRP.4          DRP-1        Definition     DRS Core Requirement Definition
C-DRP.5          DRP-1        Falsifiable    Live C5 Instance — MK-P1
C-DRP.6          DRP-1        Definition     PHASE-READY Definition
C-DRP.7          DRP-1        Definition     CORPUS-COMPLETE Definition
```

### DRS-ARCH: The Dual Reader Standard: Architecture Specification
**Status:** NOT-PHASE-READY

```
A-DRSARCH.1      DRS-ARCH     Axiom          Popperian Epistemological Foundation
A-DRSARCH.2      DRS-ARCH     Axiom          FRM as Interpretive Lens
D-DRSARCH.1      DRS-ARCH     Definition     DRS Definition
D-DRSARCH.2      DRS-ARCH     Definition     DRP Definition
D-DRSARCH.3      DRS-ARCH     Definition     GVP Definition
D-DRSARCH.4      DRS-ARCH     Definition     Falsification Kernel Definition
D-DRSARCH.5      DRS-ARCH     Definition     Six Verification Tiers
D-DRSARCH.6      DRS-ARCH     Definition     Three Claim Types
F-DRSARCH.1      DRS-ARCH     Falsifiable    Implementation Sufficiency
F-DRSARCH.2      DRS-ARCH     Falsifiable    Backwards Compatibility
F-DRSARCH.3      DRS-ARCH     Falsifiable    Predicate Binary Determinism
```

### MK-P1: Meta-Kaizen: A General Theory and Algorithmic Framework for the Mathematical Formalization of Self-Evolving Continuous Improvement Across Arbitrary Governance Substrates

**Status:** PHASE-READY
**Correction (S53):** Type F count corrected 9→6. Phantom entries F-MK1.7, F-MK1.8, F-MK1.9 removed (no corresponding prose claims). Claim IDs renamed C-MK1.x→F-MK1.x per DRS type-prefix convention. Crosswalk to prose labels preserved in AI layer.

```
A-MK1.1          MK-P1        Axiom          Conjoint Measurement Axiom A1 — Weak Order
A-MK1.2          MK-P1        Axiom          Conjoint Measurement Axiom A2 — Double Cancellation
A-MK1.3          MK-P1        Axiom          Conjoint Measurement Axiom A3 — Solvability
A-MK1.4          MK-P1        Axiom          Conjoint Measurement Axiom A4 — Archimedean
A-MK1.5          MK-P1        Axiom          Conjoint Measurement Axiom A5 — Essentialness with Veto Power
A-MK1.6          MK-P1        Axiom          Conjoint Measurement Axiom A6 — Marginal Symmetry
D-MK1.1          MK-P1        Definition     Governance Substrate
D-MK1.2          MK-P1        Definition     KVS Formula
D-MK1.3          MK-P1        Definition     Novelty Factor
D-MK1.4          MK-P1        Definition     Normalized Impact
D-MK1.5          MK-P1        Definition     Normalized Inverse Complexity
D-MK1.6          MK-P1        Definition     Timeliness Score
F-MK1.1          MK-P1        Falsifiable    Multiplicative functional form — uniqueness [prose: C-MK1.1]
F-MK1.2          MK-P1        Falsifiable    Equal weights derivation from Axiom A6 [prose: C-MK1.2]
F-MK1.3          MK-P1        Falsifiable    Adoption threshold κ=0.50 derivation under symmetric loss [prose: C-MK1.3]
F-MK1.4          MK-P1        Falsifiable    Simulation recall 73.8% [90% CI: 71.2, 76.2] [prose: C-MK1.4]
F-MK1.5          MK-P1        Falsifiable    KVS boundedness — output in [0,1] for all valid inputs [prose: C-MK1.5]
F-MK1.6          MK-P1        Falsifiable    Self-referential applicability — no logical contradiction [prose: C-MK1.6]
```

### MK-P2: Meta-Kaizen in Networked Organizations: Governance Closure, Privacy Amplification, and Bayesian Calibration Under a Federated Architecture
**Status:** PHASE-READY

```
A-MK2.1          MK-P2        Axiom          Shuffle model privacy amplification
A-MK2.2          MK-P2        Axiom          Club goods governance mechanism
A-MK2.3          MK-P2        Axiom          Beta-Binomial conjugacy
D-MK2.1          MK-P2        Definition     Meta-Kaizen Network (MKN) architecture
C-MK2.1          MK-P2        Falsifiable    Theorem 3.1 — Minimum Network Size
C-MK2.2          MK-P2        Falsifiable    Theorem 7.1 — Temporal Consistency
C-MK2.3          MK-P2        Falsifiable    Theorem 7.2 — Governance Process-Equivalence
C-MK2.4          MK-P2        Falsifiable    Bayesian calibration convergence
```

### MK-P3: The Meta-Kaizen Reasoning Network: A Formal Theory of Bisociative Question Structure, Challenge Taxonomy, and Institutional Memory Propagation
**Status:** PHASE-READY

```
A-MK3.1          MK-P3        Axiom          Aristotle's Topics — predicable partition
A-MK3.2          MK-P3        Axiom          Koestler's bisociation framework
A-MK3.3          MK-P3        Axiom          Institutional memory decay
D-MK3.1          MK-P3        Definition     Question Structure Schema (QSS)
D-MK3.2          MK-P3        Definition     Challenge Taxonomy (Types I-IV)
C-MK3.1          MK-P3        Falsifiable    Proposition 4.1 — Institutional Memory Loss
C-MK3.2          MK-P3        Falsifiable    Proposition 5.1 — Challenge Taxonomy Exhaustiveness
C-MK3.3          MK-P3        Falsifiable    Proposition 5.2 — Minimum Generative Completeness
C-MK3.4          MK-P3        Falsifiable    Theorem 5.3 — Library Quality Convergence (conditional)
```

### MK-P4: The Fractal Rhythm Model: Closed-Loop Governance, Regime-Aware Adaptation, and the Axiom 5 Modification for Dynamic Environments
**Status:** PHASE-READY

```
A-MK4.1          MK-P4        Axiom          Bayesian Online Changepoint Detection (BOCP)
A-MK4.2          MK-P4        Axiom          Axiom 5 — Essentialness with Veto Power (from MK-P1)
D-MK4.1          MK-P4        Definition     Regime Discontinuity Score (RDS)
D-MK4.2          MK-P4        Definition     Complexity Surge Score (CSS)
D-MK4.3          MK-P4        Definition     Axiom 5-prime — Regime-Conditioned Essentialness
D-MK4.4          MK-P4        Definition     KVS-hat Formula
D-MK4.5          MK-P4        Definition     Extinguishing Recursion
C-MK4.1          MK-P4        Falsifiable    Axiom 5-prime Restoration Property
C-MK4.2          MK-P4        Falsifiable    Extinguishing Recursion Convergence (delta_min)
C-MK4.3          MK-P4        Falsifiable    KVS-hat IPS Worked Example
C-MK4.4          MK-P4        Falsifiable    Closed-loop feedback architecture completeness
```

### MK-P5: On the Decision to Act: Strategic Convergence and the Mathematics of Intervention Timing at System Tipping Points
**Status:** PHASE-READY

```
A-MK5.1          MK-P5        Axiom          Sequential decision theory foundation
A-MK5.2          MK-P5        Axiom          Critical slowing down near fold bifurcation
A-MK5.3          MK-P5        Axiom          EWS decision-theoretic gap
D-MK5.1          MK-P5        Definition     Fortuna Process
D-MK5.2          MK-P5        Definition     Virtù Window
C-MK5.1          MK-P5        Falsifiable    Theorem 1 Window Rationality
C-MK5.2          MK-P5        Falsifiable    Theorem 2 Asymmetric Loss Threshold
C-MK5.3          MK-P5        Falsifiable    Theorem 3 Distributed Detection Advantage
C-MK5.4          MK-P5        Falsifiable    Theorem 4 Self-Generated Friction — t_trap existence
```

### MK-P6: The Dual Reader Standard for Software: Measurement-Theoretic Falsification Applied to Executable Systems
**Status:** NOT-PHASE-READY

```
A-MK6.1          MK-P6        Axiom          Popperian epistemological foundation
A-MK6.2          MK-P6        Axiom          DRS Layer 0 kernel applicability
A-MK6.3          MK-P6        Axiom          Design by Contract foundation
D-MK6.1          MK-P6        Definition     Software Claim Taxonomy
D-MK6.2          MK-P6        Definition     Falsification Completeness
D-MK6.3          MK-P6        Definition     Software Phase-Ready Verdict
D-MK6.4          MK-P6        Definition     Three gap categories
F-MK6.1          MK-P6        Falsifiable    Kernel universality — K applies to software without modification
F-MK6.2          MK-P6        Falsifiable    Falsification completeness implies coverage
F-MK6.3          MK-P6        Falsifiable    Coverage does not imply falsification completeness
F-MK6.4          MK-P6        Falsifiable    Gap detection — DRS reveals gaps invisible to coverage
F-MK6.5          MK-P6        Falsifiable    Feasibility — DRS applicable in one session
```

### P1: The Fractal Rhythm Model: A Universal Law of Network Information Transmission
**Status:** PHASE-READY

```
A-1.1            P1           Axiom          Thermodynamic irreversibility
A-1.2            P1           Axiom          Information distinguishability
A-1.3            P1           Axiom          Network definition
A-1.4            P1           Axiom          Non-equilibrium physics
A-1.5            P1           Axiom          Substrate independence
D-1.1            P1           Definition     FRM functional form
D-1.2            P1           Definition     Characteristic frequency
D-1.3            P1           Definition     Decay rate
D-1.4            P1           Definition     Validation set P1
F-1.1            P1           Falsifiable    FRM functional form uniqueness
F-1.2            P1           Falsifiable    36-orders validation
F-1.3            P1           Falsifiable    β = 1/2 substrate independence (empirical)
F-1.4            P1           Falsifiable    β = 1/2 analytic derivation (Hopf quarter-wave theorem)
F-1.5            P1           Falsifiable    λ derivation — leading order
F-1.6            P1           Falsifiable    Circadian period prediction
F-1.7            P1           Falsifiable    Stuart-Landau connection
```

### P2: Derivation and Universality: The β=1/2 Critical Exponent as a Universal Law
**Status:** PHASE-READY

```
D-2.1            P2           Definition     FRM Universality Class — Structural Definition
C-2.1            P2           Falsifiable    β=1/2 Derivation Validity
C-2.2            P2           Falsifiable    Universality Class Membership
C-2.3            P2           Falsifiable    Functional Form Universality
C-2.4            P2           Falsifiable    Substrate Independence (resolves PH-1.1)
C-2.5            P2           Falsifiable    RG Fixed-Point Stability
```

### P3: FRM Measurement and Diagnostics: Standard Protocol for Extracting Parameters, Computing Goodness-of-Fit, and Diagnosing Scope Boundaries
**Status:** PHASE-READY

```
A-3.1            P3           Axiom          Neural substrate is a network
A-3.2            P3           Axiom          Electromagnetic basis of neural association
D-3.1            P3           Definition     FRM System Definition for Measurement
D-3.2            P3           Definition     τ_gen Extraction Protocol
C-3.REG          P3           Falsifiable    FRM Regression Protocol
C-3.ALT          P3           Falsifiable    Alternative Model Comparison Protocol
C-3.DIAG         P3           Falsifiable    Scope Boundary Diagnostics
C-3.sigma        P3           Falsifiable    β Standard Error Protocol
```

### P4: The Fractal Rhythm Model: Mathematical Formalization of Structure and Rhythmicity
**Status:** NOT-PHASE-READY

```
A-4.1            P4           Axiom          Self-similarity across scales
D-4.1            P4           Definition     Fractal property as carrier wave signature
D-4.2            P4           Definition     Scale-invariant parameters
D-4.3            P4           Definition     Mathematical structure of FRM
F-4.1            P4           Falsifiable    Scale-invariance of β
F-4.2            P4           Falsifiable    Fractal-like behavior from carrier wave architecture
```

### P5: On the Decision to Act: Scale Independence and Empirical Validation
**Status:** NOT-PHASE-READY

```
A-5.1            P5           Axiom          Scale independence principle
D-5.1            P5           Definition     AMOC parameter extraction
D-5.2            P5           Definition     Ibn Khaldun ω measurement
F-5.1            P5           Falsifiable    FRM fit to AMOC data
F-5.2            P5           Falsifiable    ω agreement across scales
F-5.3            P5           Falsifiable    Scale independence demonstration
```

### SFW-1: Fracttalix Sentinel v12.1: Software AI Layer — DRS for Software Feasibility Demonstration
**Status:** NOT-PHASE-READY

```
A-SFW.1          SFW-1        Axiom          Python platform requirement
A-SFW.2          SFW-1        Axiom          Zero required dependencies
A-SFW.3          SFW-1        Axiom          numpy FFT correctness
A-SFW.4          SFW-1        Axiom          IEEE 754 floating-point arithmetic
A-SFW.5          SFW-1        Axiom          FRM physics foundation
D-SFW.1          SFW-1        Definition     Package version
D-SFW.2          SFW-1        Definition     SentinelResult structure
D-SFW.3          SFW-1        Definition     AlertType enumeration
D-SFW.4          SFW-1        Definition     SentinelConfig defaults
D-SFW.5          SFW-1        Definition     37-step pipeline architecture
D-SFW.6          SFW-1        Definition     Three-channel model architecture
D-SFW.7          SFW-1        Definition     Maintenance burden definition
D-SFW.8          SFW-1        Definition     Phi-kappa separation metric
F-SFW.1          SFW-1        Falsifiable    Streaming single-observation API
F-SFW.2          SFW-1        Falsifiable    Three-channel detection completeness
F-SFW.3          SFW-1        Falsifiable    Test suite passes
F-SFW.4          SFW-1        Falsifiable    Backward compatibility — v7.x kwargs
F-SFW.5          SFW-1        Falsifiable    Cascade precursor conjunction requirement
F-SFW.6          SFW-1        Falsifiable    Config parameter validation
F-SFW.7          SFW-1        Falsifiable    State persistence round-trip
F-SFW.8          SFW-1        Falsifiable    Warmup period behavior
F-SFW.9          SFW-1        Falsifiable    auto_tune returns valid detector
F-SFW.10         SFW-1        Falsifiable    MultiStreamSentinel cross-stream correlation
F-SFW.11         SFW-1        Falsifiable    Detector reset clears state
F-SFW.12         SFW-1        Falsifiable    numpy fallback warning
```

---

**Total unique claims:** 152
**Papers indexed:** 14
