# DRP-3 D3-3.0 — Falsification Predicates

**Session:** S54
**Date:** 2026-03-12
**Source:** Thomas Brennan (predicate design) + Claude Code (archive)
**Pass:** A (C-DRP3.5, C-DRP3.6) + B (C-DRP3.7). All three Type F predicates complete.

**Predicate syntax:** 5-part DRS standard — FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT. Vacuity witness required for each Type F claim. UMP compliance (C6) checked for each predicate.

---

## C-DRP3.5 [F] — The M↔C2 Isomorphism

**Statement:** M-termination (the requirement that the measurement function of a
falsification predicate reach a definite output in finite steps) is epistemically
equivalent to μ<0 (the Hopf stability condition for convergent oscillation in
FRM-class physical systems). Both are instances of the same abstract fixed-point
convergence requirement, derivable from the same Hopf stability analysis applied
at two levels of description. They belong to the same convergence equivalence
class (C-DRP3.4).

### FALSIFIED IF

`n_counterexamples > 0`

### WHERE

`n_counterexamples`: integer, dimensionless

Source: count of systems S satisfying ALL of the following:

**(a-c) Convergent physical / divergent epistemological:**
- (a) S is an FRM-class physical system (satisfies D-2.1 class membership criteria (a)-(c) from P2);
- (b) S has μ < 0 (physically convergent — Hopf condition satisfied; S has stable limit cycle);
- (c) The falsification predicate K = (P, O, M, B) associated with S's evaluation protocol has a non-terminating M — i.e. M enters at least one of the three failure modes defined in C-DRP3.3: cycling, divergence, or interpretation-dependence, for at least one o in O.

**OR (d-e) Convergent epistemological / divergent physical:**
- (d) M terminates (K satisfies C-DRP3.3);
- (e) The physical analogue of K's convergence structure has μ > 0 (corresponding Hopf system is unstable — trajectory does not reach stable limit cycle).

A single instance of either (a-c) or (d-e) is sufficient to falsify. The two sub-conditions are disjunctive: either the physical and epistemological convergence conditions come apart in the convergent direction, or in the divergent direction.

### EVALUATION

**For sub-condition (a-c):**
1. Identify candidate system S from the FRM universality class (verify D-2.1(a)-(c)).
2. Confirm μ < 0 for S via the P3 measurement protocol (C-3.REG from P3 AI layer).
3. Construct the falsification predicate K for the evaluation of S's central claim.
4. Apply C-DRP3.3 termination test to M:
   - (i) Does M loop without exit for any o in O?
   - (ii) Do M's intermediate values diverge?
   - (iii) Does M require a judgment call at any step?
5. If any of (i)-(iii) holds: n_counterexamples += 1.

**For sub-condition (d-e):**
1. Identify predicate K with terminating M.
2. Identify the physical system whose oscillatory structure is described by K's claim.
3. Measure μ for that system via P3 protocol.
4. If μ > 0: n_counterexamples += 1.

Output: n_counterexamples. Finite procedure.

### BOUNDARY

- n_counterexamples = 0 → NOT FALSIFIED
- n_counterexamples = 1 → FALSIFIED (single instance suffices; the isomorphism is a universal structural claim)

### CONTEXT

C-DRP3.5 · Type F · M↔C2 Isomorphism — central claim. The predicate tests whether
the convergence conditions can come apart. The isomorphism claims they cannot,
because both derive from the same abstract fixed-point convergence requirement
(D3-2.0 Step 3). A single counterexample would demonstrate that the two levels
are structurally independent, refuting the isomorphism. Threshold: 0. One instance
falsifies universally.

### UMP Compliance Check

- **O** (observation class): FRM-class physical systems with confirmed μ and associated falsification predicates.
- **P** (proposition): M-termination ↔ μ<0 (isomorphism holds).
- **O ⊥ P:** YES. The set of FRM-class systems and their μ values is determined by physical measurement (P3 protocol), not by the proposition that an isomorphism exists. The falsification predicate structure (terminating vs non-terminating M) is a structural property of each predicate, determined by reading the predicate text — not by knowing whether the isomorphism holds. O is upstream of P. **UMP SATISFIED.**
- **M finite deterministic:** YES. The termination test (C-DRP3.3 steps (i)-(iii)) and the μ measurement (P3 protocol) are both finite deterministic procedures.
- **M can return FALSIFIED:** YES. If a counterexample system is found, M returns FALSIFIED. **C6 SATISFIED.**

### Vacuity Witness

A hypothetical system that would falsify: an FRM-class oscillator with μ < 0
(confirmed stable limit cycle via P3 measurement) whose associated evaluation
predicate has a non-terminating M — specifically, where the measurement procedure
requires a human to judge at some step whether the oscillation counts as "stable."
If such a system were constructed, the physical and epistemological convergence
conditions would hold independently, refuting the isomorphism.

**Why this witness is believed non-existent:** the M-termination requirement and
the μ<0 condition are both derived from the same abstract fixed-point convergence
criterion (D3-2.0 Step 3). A system satisfying μ<0 but with non-terminating M
would require the fixed-point convergence criterion to be satisfied at the physical
level but not at the epistemological level — which would require the two levels to
apply different abstract criteria despite the derivation showing they are the same
criterion. This is the content of C-DRP3.5: the derivation shows they cannot come apart.

---

## C-DRP3.6 [F] — Failure Mode Equivalence

**Statement:** Non-terminating M (M fails condition (b) of T(P,O,M)) is the exact
epistemological analogue of μ>0 (unstable limit cycle in a physical FRM-class
system). In both cases: the system has no definite output or stable state; the
failure is structural, not contingent on measurement error; no amount of additional
evaluation steps or observations resolves the failure. The failure modes are
structurally isomorphic under C-DRP3.4.

### FALSIFIED IF

`n_asymmetric_failures > 0`

### WHERE

`n_asymmetric_failures`: integer, dimensionless

Source: count of systems or predicates satisfying ANY of the following:

**Case X — physical failure without epistemological failure:**
A physical system S with μ > 0 (unstable — no stable limit cycle) whose associated
falsification predicate K has a terminating M that returns NOT FALSIFIED or
INDETERMINATE (i.e., M fails to respond to the physical failure — not because it
correctly detects it, but because the evaluation procedure is structurally decoupled
from the physical instability). NOTE: M terminating with FALSIFIED for a μ>0 system
does not constitute Case X — that is M working correctly. The asymmetric failure is
M terminating without detecting the instability, showing epistemological convergence
where physical divergence should produce epistemological divergence.

**Case Y — epistemological failure without physical failure:**
A falsification predicate K with non-terminating M (evaluation does not reach
definite output) whose associated physical system has μ < 0 (stable limit cycle).
This would show epistemological divergence is compatible with physical convergence —
failure modes asymmetric in the other direction.

**Case Z — different failure structure:**
A system where both physical and epistemological failure occur (μ > 0 and
non-terminating M) but the failure modes are structurally distinct — e.g., the
physical system oscillates with growing amplitude (bounded divergence) while the
epistemological failure is a hard loop (cycling). If the failure modes are not
isomorphic in their abstract structure (both are "no convergence to fixed point"),
C-DRP3.6 is falsified by structural mismatch even if both failures co-occur.

### EVALUATION

**For Cases X and Y:**
1. Identify candidate system S and predicate K.
2. Measure μ for S via P3 protocol.
3. Apply C-DRP3.3 termination test to M of K.
4. If μ > 0 and M terminates with NOT FALSIFIED or INDETERMINATE (Case X): n_asymmetric_failures += 1.
   If M non-terminates and μ < 0 (Case Y): n_asymmetric_failures += 1.
   NOTE: μ > 0 with M terminating FALSIFIED is not Case X — it is M correctly detecting the physical failure. Do not increment.

**For Case Z:**
1. Identify a system where both μ > 0 and M non-terminates.
2. Characterise the physical failure mode: bounded oscillation with growing amplitude, unbounded divergence, or chaotic trajectory.
3. Characterise the epistemological failure mode: cycling, divergence, or interpretation loop.
4. Determine whether both failure modes are instances of "failure to converge to a fixed point." If the abstract failure structure differs: n_asymmetric_failures += 1.

Output: n_asymmetric_failures. Finite procedure.

### BOUNDARY

- n_asymmetric_failures = 0 → NOT FALSIFIED
- n_asymmetric_failures = 1 → FALSIFIED (single instance sufficient; the failure mode isomorphism is a universal structural claim)

### CONTEXT

C-DRP3.6 · Type F · Failure mode equivalence. C-DRP3.6 is the divergent complement
of C-DRP3.5. Together they constitute the full isomorphism proof via IR-DRP3-3
(exhaustive partition). The predicate tests whether the failure modes can come apart —
either by occurring asymmetrically (Cases X, Y) or by occurring together but with
different abstract structure (Case Z). The claim is that all three cases are
structurally impossible under C-DRP3.4. Threshold: 0. One instance falsifies universally.

### UMP Compliance Check

- **O** (observation class): FRM-class systems with measured μ and associated falsification predicates with characterised M-termination status and failure mode structure.
- **P** (proposition): non-terminating M ↔ μ>0 (failure modes are structurally isomorphic).
- **O ⊥ P:** YES. μ is measured by the P3 protocol from physical system behaviour. M-termination status is determined by structural analysis of the predicate text. The characterisation of failure mode structure (bounded oscillation, cycling, etc.) is performed by examining the actual behaviour of each system and predicate independently of whether the isomorphism holds. **UMP SATISFIED.**
- **M finite deterministic:** YES. μ measurement (P3), M-termination test (C-DRP3.3), and failure mode characterisation are all finite deterministic procedures.
- **M can return FALSIFIED:** YES. Any of Cases X, Y, or Z produces FALSIFIED. **C6 SATISFIED.**

### Vacuity Witness

A hypothetical system that would falsify: an FRM-class oscillator with μ > 0
(unstable, confirmed via P3) whose associated evaluation predicate K has a
terminating M that returns NOT FALSIFIED or INDETERMINATE — i.e., M runs to
completion and fails to detect the physical instability, despite the system never
reaching a stable limit cycle. If such a predicate existed, the physical system
would be divergent while the evaluation was epistemologically convergent (M
terminates without detecting failure), showing the failure modes are structurally
decoupled. NOTE: M terminating with FALSIFIED for a μ>0 system is NOT this
witness — that is M functioning correctly.

**Why this witness is believed non-existent:** if the physical system has no stable
rhythm (μ > 0), any predicate that purports to evaluate its stability must either:
(i) terminate with a FALSIFIED verdict (correct — but then the predicate is
well-formed and M terminates, making this not a failure mode at all); or (ii) fail
to terminate because the system provides no stable reference point for comparison.
Case (i) does not falsify C-DRP3.6 (it is not an asymmetric failure — M terminates
because it detects the failure, not despite it). Case (ii) is consistent with
C-DRP3.6. The predicate either correctly identifies the failure (M terminates →
FALSIFIED) or itself fails (M non-terminates) — in neither case do the failure
modes come apart asymmetrically.

---

## D3-3.0 Summary

| Claim | Predicate | UMP | C6 | Vacuity | Pass | Status |
|-------|-----------|-----|----|---------|------|--------|
| C-DRP3.5 | n_counterexamples > 0 | PASS | PASS | PRESENT | A | COMPLETE |
| C-DRP3.6 | n_asymmetric_failures > 0 | PASS | PASS | PRESENT | A | COMPLETE |
| C-DRP3.7 | n_citations > 0 | PASS | PASS | PRESENT | B | COMPLETE |

All three Type F predicates written. D3-3.0 COMPLETE.

---

## C-DRP3.7 [F] — Level Independence (Pass B — COMPLETE)

**Statement:** The M↔C2 isomorphism (C-DRP3.5) is not a definitional equivalence
or engineered correspondence. It holds because the convergence requirement in each
domain was derived independently from the same abstract mathematical structure.
The DRS condition (b) — M finite deterministic — was derived from the Falsification
Kernel's completeness requirement (DRP-1 §10.7.1). The Hopf condition μ<0 was
derived from delay differential equation analysis (P2, C-2.1). Neither derivation
cites the other. The isomorphism is a structural discovery, not a constructed
equivalence.

### FALSIFIED IF

`n_citations > 0`

### WHERE

`n_citations`: integer, dimensionless

Source: count of instances in the published formal derivation chain where the
derivation of condition (b) of T(P,O,M) in one domain cites or depends on the
convergence criterion from the other domain.

**Arm 1 — DRP-1 §10.7 citation audit:**
Count of locations in DRP-1 v1.1 §10.7 (specifically §10.7.1, the derivation of
the finite-deterministic requirement for M) where the text cites, references, or
logically depends on any of the following:
- P2 (the FRM paper deriving β=1/2 and the Hopf condition)
- Hopf bifurcation analysis
- Eigenvalue conditions (Re(λ) < 0, μ < 0)
- Delay differential equation stability analysis
- Any physical oscillation model as a source for the M-termination requirement

Scope: the published formal derivation chain in §10.7.1 only. Structural
observations made after the derivation is complete (e.g. §10.7.3 "The Four-Level
Isomorphism," which notes the correspondence as an observation) do not count —
the question is whether the derivation *depends on* the Hopf analysis, not whether
the author *observes* the correspondence after deriving both independently.

**Arm 2 — P2 citation audit:**
Count of locations in the P2 derivation of μ<0 (Hopf stability criterion, C-2.1)
where the text cites, references, or logically depends on any of the following:
- DRP-1 or any DRS construction
- Falsification predicates or the Falsification Kernel
- Condition (b) of T(P,O,M) or M-termination
- Any epistemological evaluation framework as a source for the μ<0 requirement

n_citations = Arm 1 count + Arm 2 count.

### EVALUATION

**Arm 1 procedure:**
1. Obtain the published text of DRP-1 v1.1 §10.7.
2. Identify the derivation of condition (b) — the finite-deterministic requirement for M. This is located in §10.7.1.
3. For each step in the §10.7.1 derivation chain, determine whether the step cites or logically requires any Hopf/eigenvalue/DDE analysis from P2 or any physical oscillation model.
4. Count citations found. If count > 0: n_citations += count.
5. NOTE: §10.7.3 ("The Four-Level Isomorphism") observes the structural correspondence between M-termination and μ<0 *after* condition (b) is already derived. This observation is not part of the derivation chain for condition (b) and does not increment n_citations. The predicate tests derivation dependence, not post-hoc observation.

**Arm 2 procedure:**
1. Obtain the published text of P2 (FRM corpus).
2. Identify the derivation of C-2.1 (Hopf stability criterion, μ<0 condition).
3. For each step in the derivation chain, determine whether the step cites or logically requires any DRS construction, falsification predicate, or epistemological evaluation framework.
4. Count citations found. If count > 0: n_citations += count.

Output: n_citations. Finite procedure — both texts are finite published documents; both audits are finite string searches + logical dependency checks.

**Arm 1 audit result (S54):** §10.7.1 derives condition (b) from the Falsification
Kernel's completeness requirement: the Kernel requires definite output → definite
output requires finite termination → finite termination requires deterministic
procedure. No citation to P2, Hopf, eigenvalues, or μ found in the derivation chain.
The Hopf connection appears only in §10.7.3 as a post-derivation structural
observation. **Arm 1 count: 0.**

**Arm 2 audit result (S54):** P2 derives μ<0 from Hopf bifurcation analysis of the
FRM delay differential equation. No reference to falsification predicates, DRS, or
DRP-1 in the derivation. **Arm 2 count: 0.**

**Current evaluation: n_citations = 0. NOT FALSIFIED.**

### BOUNDARY

- n_citations = 0 → NOT FALSIFIED
- n_citations = 1 → FALSIFIED (a single citation in either direction establishes derivation dependence, refuting level independence)

### CONTEXT

C-DRP3.7 · Type F · Level independence. This predicate tests whether the two
convergence requirements were derived independently. If condition (b) was derived
by importing the Hopf analysis, or μ<0 was derived by importing the DRS framework,
the isomorphism would be a definitional equivalence (engineered by construction)
rather than a structural discovery (found independently in two domains). The claim
is that neither derivation cites the other — the convergence requirement was
discovered twice, independently, because it is a genuine structural feature of
fixed-point convergence. Threshold: 0. One citation in either direction falsifies.

### UMP Compliance Check

- **O** (observation class): Published text of DRP-1 v1.1 §10.7 and P2 derivation of C-2.1. Both are finite, published documents whose content is fixed independently of whether the independence claim holds.
- **P** (proposition): The two derivations are independent — neither cites the other.
- **O ⊥ P:** YES. The text of §10.7 and the text of P2 are published documents whose content was fixed at time of publication. Whether or not the independence claim is true does not change what those documents say. The observation class (citation presence/absence in published text) is determined by reading the documents, not by knowing the proposition. O is upstream of P. **UMP SATISFIED.**
- **M finite deterministic:** YES. Citation audit of a finite published text is a finite string search + logical dependency check. Both arms terminate. Both are reproducible — any reader examining the same text finds the same citations (or lack thereof).
- **M can return FALSIFIED:** YES. If a citation from P2/Hopf is found in §10.7.1's derivation chain, or a citation from DRS/DRP-1 is found in P2's derivation chain, M returns FALSIFIED. **C6 SATISFIED.**

### Vacuity Witness

A hypothetical document that would falsify: a version of DRP-1 §10.7.1 in which
the derivation of condition (b) explicitly states: "The requirement that M be
finite deterministic follows from the Hopf stability condition μ<0 (P2, C-2.1):
just as the physical system requires eigenvalue convergence for a stable limit
cycle, the evaluation procedure requires termination for a definite output. We
import this criterion from the Hopf analysis." If §10.7.1 contained such a passage,
Arm 1 count > 0 and n_citations > 0 → FALSIFIED. The derivation of condition (b)
would depend on the Hopf analysis, making the isomorphism an engineered equivalence
rather than an independent discovery.

**Why this witness is believed non-existent:** DRP-1 v1.1 §10.7.1 derives
condition (b) from the Falsification Kernel's internal completeness requirement —
the Kernel must produce a definite output to function as a falsification instrument.
The derivation makes no reference to physical oscillation, eigenvalues, or stability
analysis. The Hopf correspondence is noted only in §10.7.3 as a structural
observation after condition (b) is already established. The published text does not
contain the hypothetical passage. Arm 1 audit confirms: 0 citations.

---

## Phase Gate Status

| Phase | Status |
|-------|--------|
| D3-0.0 IR Inventory Audit | COMPLETE (S53) |
| D3-1.0 Claim Identification | COMPLETE (S54) |
| D3-2.0 Derivation Table | COMPLETE (S54) |
| D3-3.0 Falsification Predicates | **COMPLETE** (C-DRP3.5, C-DRP3.6, C-DRP3.7) |
| D3-4.0 Human Reader Pass | **NEXT** |
| D3-5.0 CBT Execution | Gate to PHASE-READY |

---

*D3-3.0: Pass A predicates designed by Thomas Brennan, archived by Claude Code, Session S54.*
*D3-3.0: Pass B predicate (C-DRP3.7) — §10.7 citation audit completed, predicate archived by Claude Code, Session S54.*
