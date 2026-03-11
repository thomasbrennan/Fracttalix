# MK-P6 Build Process — The Dual Reader Standard for Software

**Session:** 51
**Date:** 2026-03-11
**Process:** Canonical Build (P0 CBT v2)
**Author:** Thomas Brennan · with Claude (Anthropic)

---

## Phase 1: First Build Plan

### 1.1 Paper Identity

| Field | Value |
|-------|-------|
| Paper ID | MK-P6 |
| Title | The Dual Reader Standard for Software: Measurement-Theoretic Falsification Applied to Executable Systems |
| Type | methodology_D |
| Track | Meta-Kaizen (extension: MK 6/6) |
| Status | DRAFT |
| Depends on | MK-P1 (KVS axioms), P14/DRP-1 (DRS specification), SFW-1 (Sentinel — demonstration target), falsification-kernel.md (Layer 0) |
| Enables | P8 (Software and Engineered Systems), P14 v2.0 (DRS extended to software), SFW-1 v2 (software AI layer upgrade) |

### 1.2 Core Question

Can the Dual Reader Standard — the three-layer architecture (Layer 0 semantic kernel, Layer 1 machine-readable claim registry, Layer 2 human-readable prose) — be extended from scientific papers to executable software without modifying the underlying falsification kernel K = (P, O, M, B)?

### 1.3 Thesis (Stated As Falsifiable Claim)

**C-MK6.1:** The falsification kernel K = (P, O, M, B) as defined in falsification-kernel.md v1.1 applies to software behavioral claims without modification. The same grammar, evaluation semantics, and validity constraints that govern scientific Type F claims govern software Type F claims.

**Falsification condition:** Exhibit a software behavioral claim that (a) is clearly falsifiable by empirical test, but (b) cannot be expressed as a well-formed K = (P, O, M, B) 4-tuple under the existing kernel constraints.

### 1.4 Planned Claim Registry (First Pass)

| Claim ID | Type | Name | Description |
|----------|------|------|-------------|
| A-MK6.1 | A | Popperian epistemological foundation | Falsificationism as the basis for software verification: we can falsify but not verify |
| A-MK6.2 | A | Conjoint measurement applicability | Software behavioral claims admit the same measurement-theoretic structure as scientific claims (from MK-P1 A1–A6) |
| A-MK6.3 | A | DRS Layer 0 kernel | The falsification kernel K = (P, O, M, B) as given in falsification-kernel.md v1.1 |
| D-MK6.1 | D | Software Claim Taxonomy | Three-type classification: Assumptions (A), Definitions (D), Falsifiable (F) — same taxonomy as scientific DRS |
| D-MK6.2 | D | Software Assumption (Type A) | Platform requirements, dependency contracts, environmental preconditions |
| D-MK6.3 | D | Software Definition (Type D) | Type signatures, data structures, configuration schemas |
| D-MK6.4 | D | Software Behavioral Claim (Type F) | Correctness guarantees, invariants, performance bounds, API contracts |
| D-MK6.5 | D | Falsification Completeness | A software system is falsification-complete iff every Type F claim has a well-formed predicate and a passing evaluation |
| D-MK6.6 | D | Software Phase-Ready | Extension of DRS phase-ready verdict to software releases: c1–c6 adapted |
| F-MK6.1 | F | Kernel universality | K = (P, O, M, B) applies to software claims without modification |
| F-MK6.2 | F | Falsification completeness implies coverage | If S is falsification-complete, ∃ test suite T from evaluations with L(S,T) ≥ L(S,T') for any T' achieving same behavioral verification |
| F-MK6.3 | F | Coverage does not imply falsification completeness | ∃ systems with L(S,T) = 1.0 that are not falsification-complete |
| F-MK6.4 | F | Assumption propagation | When A_k is invalidated, all dependent F-claims computable in O(|R|) |
| F-MK6.5 | F | Gap detection superiority | Applying DRS to an existing well-tested codebase reveals claims invisible to coverage metrics |
| F-MK6.6 | F | Demonstration: Sentinel v12.1 | Sentinel produces a valid software AI layer with N_A assumptions, N_D definitions, N_F falsifiable claims, with M placeholder claims not covered by existing tests |

### 1.5 Prior Art Structure (Planned Sections)

The paper must survey prior art across ALL major language/cultural traditions:

| Tradition | Key Contributions | Gap Relative to DRS |
|-----------|-------------------|---------------------|
| **Anglo-American** | Design by Contract (Meyer 1992), QuickCheck (Claessen & Hughes 2000), TLA+ (Lamport), Alloy (Jackson), RV conferences | No claim registry; no placeholder honesty; no boundary documentation standard |
| **French** | B-Method (Abrial), Atelier B, CompCert (Leroy), Paris Métro Line 14 | Full formal verification — stronger but less tractable; no "honest accounting of gaps" |
| **Dutch** | Dijkstra structured programming, weakest precondition calculus | Foundational for correctness proofs; does not address claim enumeration |
| **German** | VDI/DIN standards, automotive ISO 26262 | Requirement traceability exists; not machine-readable falsification |
| **Scandinavian** | SIMULA (Dahl & Nygaard), Scandinavian OOP school | Object-oriented contracts implicit but not formalized as falsification |
| **Russian/Soviet** | Ershov programming methodology, GOST standards | Strong theoretical tradition; limited Western integration |
| **Japanese** | Kaizen, Monozukuri, JUSE quality circles, software factories | Quality culture strong but philosophical, not formal-methods based |
| **Chinese** | CertiKOS (Zhong Shao/Yale), ORIENTAIS (ECNU), GB/T standards | Formal verification of OS kernels; no behavioral claim registry standard |
| **Indian** | CMMI Level 5 adoption, STQC | Process maturity models — audits process, not claims |
| **ISO/IEC** | 29119 (testing), 25010 (quality), 12207 (lifecycle), 15026 (assurance) | Requirement traceability but no machine-readable falsification predicates |
| **Safety-critical** | DO-178C (avionics), IEC 62304 (medical), ISO 26262 (automotive) | Closest to DRS in spirit — objective evidence traceability; still prose-based |

### 1.6 Paper Structure

1. Series Orientation
2. Abstract (AI-Reader Header + Human Reader)
3. The Problem: Software Claims Are Implicit and Unaudited
4. Prior Art Survey (11 traditions + ISO + safety-critical)
5. The Software Claim Taxonomy (Type A/D/F mapping)
6. Core Theoretical Results (Theorems 5.1–5.3)
7. What DRS Reveals That Tests Cannot (Three Gap Categories)
8. Demonstration: Fracttalix Sentinel v12.1
9. Implications (Open Source Trust, Dependency Management, AI-Generated Code, Regulatory Compliance)
10. Limitations
11. Conclusion
12. References

### 1.7 Deliverables

1. MK-P6 paper (markdown)
2. MK-P6 AI layer (JSON, conforming to ai-layer-schema.json v2-S50)
3. Sentinel Software AI Layer v2 (comprehensive behavioral claims beyond SFW-1 v1)
4. Build Table update (MK track expanded to 6 papers)
5. Session journal entry

---

## Phase 2: Hostile Review

*To be conducted after First Build Plan is reviewed.*

The hostile reviewer must attack:

1. **"This is just testing with extra steps"** — What does DRS actually add that existing test frameworks don't?
2. **"Prior art is already doing this"** — Design by Contract, formal verification, SBOM — where is the actual novelty?
3. **"Registry completeness is unverifiable"** — Your own Limitation #1 admits you can't know if all claims are enumerated
4. **"The theorems are trivial"** — Falsification completeness implies coverage (Theorem 5.1) is nearly tautological
5. **"No empirical validation"** — One demonstration on your own software is not evidence
6. **"The overhead isn't justified"** — Who will maintain these JSON registries?
7. **"Popper is the wrong epistemology for software"** — Software can be formally verified (Coq, Lean); why settle for falsification?
8. **"You're confusing claims with requirements"** — Requirements engineering already does traceability
9. **"The scope boundary is unclear"** — Does this apply to all software? Libraries only? Safety-critical only?
10. **"Cultural survey is superficial"** — Listing traditions is not engaging with them deeply

---

## Phase 3: Second Meta-Kaizen

*Corrections applied after hostile review. Each objection gets a response and effect classification: Strengthened, Resolved, Discipline Enforced, or Scope Refined.*

---

## Phase 4: Final Build Plan

*Revised architecture incorporating all hostile review corrections. This becomes the paper.*

---

## Process Notes

- This is the first application of the canonical build process to a Meta-Kaizen track extension
- MK-P6 extends the track from 5 to 6 papers
- The paper is self-referential: it describes the DRS for software and its own AI layer is a DRS layer for a paper about software DRS layers
- The Sentinel demonstration is not hypothetical — we will build the actual software AI layer as part of this session
