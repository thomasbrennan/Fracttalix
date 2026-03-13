# DRP-4: The Full Conservation Law: Epistemic Information Vanishes for All Measurement Functions

**Version:** v1
**Session:** S56
**Author:** Thomas Brennan
**Paper type:** Type B (Derivation)
**Series:** DRP Series — Paper 4 of 8
**AI layer:** `ai-layers/DRP4-ai-layer.json` v1 (S56)
**CBT I-9:** 7/7 PASS (S56)
**Phase gate:** PHASE-READY (pending Thomas sign-off)

---

## Abstract

DRP-2 proved that when a falsification predicate's observation class is derived from the proposition under test (O = f(P)), the epistemic information content of any binary test is zero: I(M(O); P) = 0. This paper proves that the I = 0 result holds for *all* measurement functions satisfying condition (b) of T(P, O, M) — not only binary M, but n-ary, continuous, and probabilistic M. The generalisation is structural: the I = 0 mechanism depends solely on the composition M(O) = M(f(P)) = h(P), which makes the test outcome a deterministic function of P regardless of M's output type. The M-class taxonomy (C-DRP4.3) exhaustively classifies all finite deterministic measurement functions by output space, and the conservation law is verified for each sub-class. The central result, the Full Conservation Law (C-DRP4.5), passes forward to DRP-5 as a Type A axiom for the universality proof and auxiliary hypothesis protocol.

**AI-reader annotation:** 7 claims registered. 2A + 2D + 3F. Central claim: C-DRP4.5. Pass-forward: C-DRP4.5 → DRP-5. Inbound: DRP-2 C-DRP2.2, DRP-3 C-DRP3.5. CBT I-9: 7/7 PASS.

---

## 1. Introduction

### 1.1 Problem Statement

DRP-2 §3 proved the epistemic conservation law for binary measurement functions: when O = f(P), the mutual information I(M(O); P) = 0. The proof established that a falsification predicate whose observation class is derived from the proposition under test cannot extract information about that proposition — the test is structurally vacuous.

However, DRP-2 restricted its proof to binary M: functions with output space {FALSIFIED, NOT FALSIFIED}. This is the standard DRS case, but the Falsification Kernel K = (P, O, M, B) admits measurement functions with richer output spaces. A measurement function might return a test statistic (continuous), a multi-valued classification (n-ary), or a confidence value (probabilistic). Does the conservation law hold for these cases?

### 1.2 Central Question

Is the I = 0 result an artefact of binary output spaces, or does it hold structurally for all well-formed measurement functions? If a non-binary M with O = f(P) could produce I > 0, the conservation law would be incomplete and the DRS vacuity detection framework would have a gap.

### 1.3 Result

The I = 0 result holds for all M satisfying condition (b) of T(P, O, M). The proof mechanism — O = f(P) creating a deterministic chain P → O → M(O) — is independent of M's output type. This is the Full Conservation Law (C-DRP4.5).

---

## 2. Scope and Definitions

### 2.1 Scope Boundary

**IN:** All measurement functions M satisfying condition (b) of T(P, O, M) — i.e., all finite deterministic functions from observation classes to verdict spaces. This includes binary, n-ary, continuous, and probabilistic M.

**OUT:**
- Non-terminating M (these fail condition (b) and are not well-formed predicates)
- Stochastic M that are not finite deterministic (these fail condition (b))
- The UMP itself (proved in DRP-1; DRP-4 uses the conservation consequence, not the independence condition)

### 2.2 Notation

- K = (P, O, M, B): Falsification Kernel
- T(P, O, M): Testability relation (DRP-2, C-DRP2.4)
- I(M(O); P): Epistemic information content — mutual information between test outcome and proposition
- H(P): Prior uncertainty about P
- H(P|M(O)): Posterior uncertainty about P given test outcome
- h = M ∘ f: Composed function when O = f(P)

---

## 3. Received Axioms

### 3.1 C-DRP4.1 [A] — Binary Conservation Law (from DRP-2)

**Source:** DRP-2, C-DRP2.2.
**Status:** LIVE.

For any falsification predicate K = (P, O, M, B) where O = f(P) and M is binary (M: O → {FALSIFIED, NOT FALSIFIED}): I(M(O); P) = 0. The mutual information between the test outcome and the proposition is zero when the observation class is derived from the proposition.

This is the starting point for generalisation. DRP-4 identifies the structural mechanism that makes I = 0 hold and shows it is independent of M's output type.

**IR used:** IR-DRP4-1 (Received Axiom).

### 3.2 C-DRP4.2 [A] — Convergence Principle (from DRP-3)

**Source:** DRP-3, C-DRP3.5.
**Status:** LIVE.

M-termination is epistemically equivalent to μ < 0. Both are instances of the same abstract fixed-point convergence requirement. All terminating M functions share the same convergence structure regardless of output type.

This establishes that the class of well-formed M functions (those satisfying condition (b)) share a common structure. The generalisation from binary to all M is structural, not case-by-case.

**IR used:** IR-DRP4-1 (Received Axiom).

---

## 4. M-Class Taxonomy (C-DRP4.3 [D])

The class of all measurement functions M satisfying condition (b) of T(P, O, M) is partitioned into four exhaustive sub-classes by output type:

**(i) Binary M:** O → {FALSIFIED, NOT FALSIFIED}. |range(M)| = 2. This is the standard DRS case.

**(ii) N-ary M:** O → {v₁, ..., vₙ} for finite n > 2. Example: a test with FALSIFIED / NOT FALSIFIED / INDETERMINATE / BOUNDARY.

**(iii) Continuous M:** O → ℝ (or a measurable subset). Example: a test statistic, p-value, or effect size.

**(iv) Probabilistic M:** O → [0,1]. Example: a probability or confidence value.

These four exhaust all possible finite deterministic output spaces: any discrete finite space has cardinality n (cases i-ii); any continuous space embeds in ℝ (case iii) or is bounded in [0,1] (case iv). No well-formed M exists outside this partition.

**Derivation:** From C-DRP4.2 via IR-DRP4-4 (Exhaustive Classification). The convergence principle establishes that all terminating M share common structure; the taxonomy classifies them by output space topology.

---

## 5. Generalised Epistemic Information (C-DRP4.4 [D])

DRP-2 §3 uses Shannon mutual information for binary M:

I(M(O); P) = H(P) − H(P|M(O))

For the general case, the same definition applies with appropriate entropy measure:

- **Binary/n-ary:** Discrete entropy (standard Shannon).
- **Continuous:** Differential entropy (standard extension).
- **Probabilistic:** Differential entropy over [0,1].

In all cases, I measures the reduction in uncertainty about P achieved by observing M(O). When I = 0, observing M(O) provides no information about P. The definition is well-formed for all terminating M because condition (b) guarantees M(O) produces a definite output.

---

## 6. Mechanism Independence (C-DRP4.6 [F])

**Claim:** The I = 0 result in DRP-2 §3 depends on exactly one structural property: O = f(P). It does not invoke any property specific to binary M.

**Derivation:** The DRP-2 §3 proof proceeds:

1. Assume O = f(P).
2. Then M(O) = M(f(P)).
3. Define h = M ∘ f. Then M(O) = h(P).
4. h(P) is a deterministic function of P alone.
5. Observing h(P) does not update the prior on P because the outcome is determined by P's truth value — no independent evidence is introduced.
6. Therefore I(M(O); P) = H(P) − H(P|h(P)) = 0.

**Critical observation:** No step in this chain invokes any property of M's output space. Steps 1-3 use function composition (IR-DRP4-3), valid for any function regardless of output type. Step 4 is a consequence of composition. Step 5 is the epistemic argument — the prior on P is unchanged because h(P) is determined by P. Step 6 applies the information measure.

The cardinality, topology, or measure-theoretic structure of M's output space plays no role. The proof mechanism is: O = f(P) creates a deterministic chain P → O → M(O), making the test outcome a function of P alone. This chain holds whether M outputs {0,1}, {0,...,n}, ℝ, or [0,1].

**Falsification predicate (C-DRP4.6):**

FALSIFIED IF n_binary_steps > 0, where n_binary_steps counts steps in the DRP-2 §3 proof that require |range(M)| = 2 as a necessary premise and cannot be re-derived for general M.

**Vacuity witness:** A step in DRP-2 §3 reading "Since M has exactly two outputs, the probability table has exactly two rows, allowing us to compute I directly."

---

## 7. The Full Conservation Law (C-DRP4.5 [F]) — Central Claim

**Claim:** For any falsification predicate K = (P, O, M, B) where O = f(P) and M satisfies condition (b) of T(P, O, M):

**I(M(O); P) = 0**

The epistemic information content of the test is zero regardless of M's output type.

**Derivation:**

**Premise 1:** I(M(O); P) = 0 for binary M when O = f(P) (C-DRP4.1, received).
**Premise 2:** The proof mechanism depends only on O = f(P), not on M's output type (C-DRP4.6).
**Premise 3:** All terminating M share the same convergence structure (C-DRP4.2, received).
**Premise 4:** The M-class taxonomy (C-DRP4.3) exhausts all terminating M.

By IR-DRP4-2 (Structural Generalisation): The theorem I(M(O); P) = 0 was proved for the binary case. The proof depends only on the structural property O = f(P), which concerns the relationship between O and P, not the structure of M. By the convergence principle, all terminating M share the same abstract structure. By exhaustive classification, the taxonomy covers all cases.

Therefore: for any K = (P, O, M, B) where O = f(P) and M satisfies condition (b): I(M(O); P) = 0.

**Falsification predicate (C-DRP4.5):**

FALSIFIED IF n_counterexamples > 0, where n_counterexamples counts predicates K with O = f(P) and terminating non-binary M yielding I > 0.

**Vacuity witness:** A continuous M computing a test statistic from O = f(P) that yields I > 0. Believed non-existent because h(P) = M(f(P)) is determined by P.

---

## 8. Exhaustive Coverage (C-DRP4.7 [F])

**Claim:** The M-class taxonomy (C-DRP4.3) is exhaustive. The full conservation law holds for each sub-class individually:

- **Binary:** Proved in DRP-2 (C-DRP4.1).
- **N-ary:** Binary is a special case of n-ary with n = 2. The h(P) = M(f(P)) composition is deterministic for any finite output space.
- **Continuous:** h(P) = M(f(P)) ∈ ℝ is still a deterministic function of P. I(h(P); P) = 0.
- **Probabilistic:** Same argument. h(P) ∈ [0,1] is determined by P.

No case produces I > 0 when O = f(P). The taxonomy is exhaustive and the conservation law holds for all M.

**Falsification predicate (C-DRP4.7):**

FALSIFIED IF n_uncovered > 0, where n_uncovered counts finite deterministic M functions that do not fall into any of the four sub-classes.

**Vacuity witness:** An M function whose output space is a non-Hausdorff topological space not embeddable in ℝ or [0,1] and not a finite discrete set.

---

## 9. Inference Rules

| IR | Name | Used by |
|----|------|---------|
| IR-DRP4-1 | Received Axiom | C-DRP4.1, C-DRP4.2, Step 2 |
| IR-DRP4-2 | Structural Generalisation | C-DRP4.5, C-DRP4.6 |
| IR-DRP4-3 | Composition Preservation | C-DRP4.6, Step 3 |
| IR-DRP4-4 | Exhaustive Classification | C-DRP4.3, C-DRP4.5, C-DRP4.7 |
| IR-DRP4-5 | Vacuity Witness | C-DRP4.5, C-DRP4.6, C-DRP4.7 |

All five exercised. No additional rules required.

---

## 10. Relationship to DRP Series

DRP-4 occupies a specific position in the DRP derivation chain:

- **DRP-1** established the Falsification Kernel K = (P, O, M, B) and proved the UMP.
- **DRP-2** proved testability is a ternary relation T(P, O, M) and derived the binary conservation law I = 0.
- **DRP-3** proved the M↔C2 Isomorphism: M-termination ↔ μ < 0 (Convergence Principle).
- **DRP-4** (this paper) generalises I = 0 from binary to all M. The Full Conservation Law.
- **DRP-5** will receive C-DRP4.5 and prove universality + auxiliary hypothesis protocol.

The DRP-4 contribution is the removal of the binary restriction. DRP-2's proof was more general than it appeared: the I = 0 mechanism was never binary-specific, but the restriction was stated explicitly and must be formally removed.

---

## 11. Self-Application — DRP-4 Against Its Own Standard

DRP-4 is a DRP-series paper. It must satisfy its own standards. Three checks:

### Check 1: All claims registered?

7 claims: C-DRP4.1, C-DRP4.2, C-DRP4.3, C-DRP4.4, C-DRP4.5, C-DRP4.6, C-DRP4.7. All appear in AI layer claim_registry. **PASS.**

### Check 2: All Type F claims have full predicates?

C-DRP4.5: FALSIFIED_IF / WHERE / EVALUATION / BOUNDARY / CONTEXT — present.
C-DRP4.6: FALSIFIED_IF / WHERE / EVALUATION / BOUNDARY / CONTEXT — present.
C-DRP4.7: FALSIFIED_IF / WHERE / EVALUATION / BOUNDARY / CONTEXT — present.
All three have UMP checks and vacuity witnesses. **PASS.**

### Check 3: All inbound/outbound edges declared?

Inbound: DRP-2 C-DRP2.2 (LIVE), DRP-3 C-DRP3.5 (LIVE). Both in AI layer.
Outbound: C-DRP4.5 → DRP-5 (pass-forward registered). **PASS.**

Self-application: **3/3 PASS.**

---

## 12. Dependency Map

### Inbound Live Edges

| Edge | Source | Claim | Status |
|------|--------|-------|--------|
| EDGE 4 | DRP-2 | C-DRP2.2 (Binary conservation law) | LIVE |
| EDGE 6 | DRP-3 | C-DRP3.5 (Convergence Principle) | LIVE |

### Outbound Pass-Forward

| Edge | Target | Claim | Status |
|------|--------|-------|--------|
| DRP4→DRP5 | DRP-5 | C-DRP4.5 (Full Conservation Law) | LIVE — pending DRP-4 PHASE-READY |

### Internal Derivation Chain

C-DRP4.1 + C-DRP4.2 → C-DRP4.3 (taxonomy) → C-DRP4.4 (generalised I)
C-DRP4.1 → C-DRP4.6 (mechanism independence) → C-DRP4.5 (full law)
C-DRP4.3 + C-DRP4.5 → C-DRP4.7 (exhaustive coverage)

---

## 13. Phase Gate Record

| Phase | Status |
|-------|--------|
| D4-0.0 Scope & IR Audit | COMPLETE (S56) |
| D4-1.0 Claim Identification | COMPLETE (S56) |
| D4-2.0 Derivation Table | COMPLETE (S56) |
| D4-3.0 Falsification Predicates | COMPLETE (S56) |
| Phase 4 — AI Layer | COMPLETE (S56) |
| Phase 5 — CBT I-9 | PASS 7/7 (S56) |
| Phase 6 — Thomas Sign-off | PENDING |

### Principle 10 Audit Summary

4 entries, all anchored at live edges (DRP-2 C-DRP2.2, DRP-3 C-DRP3.5) or IR axioms (IR-DRP4-3). No orphaned constants. No gaps.

### Claim Summary

| Type | Count | Claims |
|------|-------|--------|
| A | 2 | C-DRP4.1, C-DRP4.2 |
| D | 2 | C-DRP4.3, C-DRP4.4 |
| F | 3 | C-DRP4.5, C-DRP4.6, C-DRP4.7 |
| **Total** | **7** | |

Registry impact: 175 + 7 = 182 total claims, 18 papers.

---

## References

- DRP-1 v1.1. *The Dual Reader Standard: Syntax and Semantics of Machine-Evaluable Falsification.* AI layer: `ai-layers/DRP1-ai-layer.json`.
- DRP-2 v1.2. *Testability as a Relational Property.* AI layer: `ai-layers/DRP2-ai-layer.json`.
- DRP-3 v1. *Evaluation Convergence and Physical Stability: The M↔C2 Isomorphism.* AI layer: `ai-layers/DRP3-ai-layer.json`.
- P2 v4. *FRM Universality and the Hopf Stability Criterion.* AI layer: `ai-layers/P2-ai-layer.json`.

---

*DRP-4 prose draft v1 produced by Claude, Session S56.*
