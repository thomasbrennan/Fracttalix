# DRP-3 D3-2.0 — Derivation Table

**Session:** S54
**Date:** 2026-03-12
**Source:** Thomas Brennan (derivation design) + Claude Code (archive)

---

## Preamble

This document maps every derivation step in DRP-3. Each step identifies: inputs
(received axioms, prior definitions, or prior derived results), the inference rule
applied, and the output. No claim is introduced at D3-3.0 (predicates) or later
that is not grounded in this table.

OQ-1: RESOLVED S54. C-DRP3.7 confirmed Type F.
Claim count: 2A + 2D + 3F = 7 total. No changes from D3-1.0.

Principle 10 compliance: all constants and conditions in the derivation must
terminate at live edges (C-DRP3.1, C-DRP3.2) or IR axioms. Checked at each step.

---

## Step 0 — Received Inputs

**S0-A: C-DRP3.1 [A] — Testability Relation T(P,O,M)**

- Source: DRP-2, C-DRP2.4. LIVE.
- Content relevant to DRP-3: Condition (b): M is a finite deterministic function O → {FALSIFIED, NOT FALSIFIED, INDETERMINATE}. T = FALSE if condition (b) fails.
- P10 anchor: LIVE inbound edge. Terminates at DRP-2 AI layer, C-DRP2.4.

**S0-B: C-DRP3.2 [A] — Hopf stability criterion**

- Source: P2, C-2.1. LIVE.
- Content relevant to DRP-3: The Hopf bifurcation condition requires Re(λ) < 0 (μ < 0) for eigenvalues of the linearised system. μ < 0 → convergent oscillation (stable limit cycle, definite output). μ > 0 → divergent trajectory (unstable, no definite output). μ = 0 → bifurcation point (boundary, not stable operating state).
- P10 anchor: LIVE inbound edge. Terminates at P2 AI layer, C-2.1.

---

## Step 1 — Definition: M-termination (produces C-DRP3.3)

- **Input:** S0-A (C-DRP3.1), condition (b)
- **Rule:** IR-DRP3-2 (Received Axiom) — expand condition (b) into operational components

**Derivation:**

Condition (b) of T(P,O,M) states M is a finite deterministic function. Three failure
modes identified by exhaustive enumeration:

1. **Cycling:** M enters a loop, never exits. Output: undefined.
2. **Divergence:** M's intermediate values grow without bound, never reach output state. Output: undefined.
3. **Interpretation-dependence:** M reaches a step requiring a judgment call — output depends on evaluator, not predicate. Output: indeterminate (not reproducible). This is the DRS-specific case: natural language predicates fail at interpretation steps.

**Output:** C-DRP3.3 [D] — M-termination definition. M terminates iff it avoids all three failure modes for every o ∈ O. Definitional expansion of condition (b); introduces no new claim.

**P10 check:** Three failure modes derived from logical structure of "finite deterministic function" (S0-A). No new constants introduced.

---

## Step 2 — Definition: Convergence Equivalence Class (produces C-DRP3.4)

- **Input:** S0-A (C-DRP3.1), S0-B (C-DRP3.2), C-DRP3.3
- **Rule:** IR-DRP3-1 (Structural Isomorphism) — identify abstract structure common to both inputs before asserting the isomorphism

**Derivation:**

Both T(P,O,M) condition (b) and the Hopf condition impose a convergence requirement.
To assert they are isomorphic, a common abstract framework is required. Three
conditions for class membership:

- **(a)** Both systems independently require condition C for well-defined operation.
- **(b)** The requirement in each is derivable from the same abstract mathematical structure — not by defining one in terms of the other.
- **(c)** The failure mode in each when C is violated is structurally isomorphic: same abstract consequence, different level of description.

This definition is framework-setting only. It does not yet assert that M-termination
and μ<0 are in the same class — that is the content of C-DRP3.5.

**Output:** C-DRP3.4 [D] — Convergence equivalence class.

**P10 check:** Three membership conditions derived from logical requirements of IR-DRP3-1. No new empirical constants.

---

## Step 3 — Convergent Case: M-termination ↔ μ<0 (first half of C-DRP3.5)

- **Input:** C-DRP3.3 (M-termination), S0-B (μ<0 Hopf condition), C-DRP3.4 (convergence equivalence class)
- **Rule:** IR-DRP3-1 (Structural Isomorphism)

**Derivation — checking C-DRP3.4 membership conditions:**

**Condition (a) — independent requirement:**
- Physical: μ<0 required for stable limit cycle. Without it, trajectory diverges or oscillates without settling. System not well-defined as oscillator.
- Epistemological: M-termination required for non-vacuous predicate. Without it, T = FALSE (condition (b) fails), predicate has no definite output.
- Verdict: both require convergence for well-defined operation. **SATISFIED.**

**Condition (b) — common derivation structure:**
- Physical: μ<0 derived from Hopf stability analysis of linearised DDE. Eigenvalue condition Re(λ)<0 is convergence criterion.
- Epistemological: M-termination is convergence criterion for evaluation sequence. Terminating M reaches fixed point (definite output) in finite steps.
- Abstract structure: both are fixed-point convergence criteria. Hopf asks: does trajectory converge to fixed point (limit cycle)? Termination asks: does evaluation converge to fixed point (definite output)? Same abstract mathematical question.
- **SATISFIED** (subject to C-DRP3.7 — independence of derivation — proved at Step 5).

**Condition (c) — isomorphic failure mode:**
- Physical failure (μ>0): trajectory diverges, no stable limit cycle.
- Epistemological failure (non-terminating M): evaluation diverges, no definite output.
- Abstract: system fails to converge to fixed point in both domains.
- **SATISFIED.**

**Output:** Convergent case of C-DRP3.5 established. M-termination ↔ evaluation converges to fixed point. μ<0 ↔ physical trajectory converges to fixed point. Both are instances of the same abstract fixed-point convergence requirement.

**P10 check:** All constants derive from S0-A and S0-B (live edges). Abstract structure extracted by IR-DRP3-1, not imported from uncited source.

---

## Step 4 — Divergent Case: Non-terminating M ↔ μ>0 (produces C-DRP3.6, completes C-DRP3.5)

- **Input:** Step 3 result, S0-A, S0-B, C-DRP3.4
- **Rule:** IR-DRP3-3 (Exhaustive Partition — Convergence)

**Derivation:**

By IR-DRP3-3, the convergence condition partitions each system into exactly two cases:
convergent (terminates / μ<0) and divergent (non-terminating / μ>0). No third case:
μ=0 is the bifurcation point — not a stable operating state for either system.

- **Physical (μ>0):** Eigenvalues have positive real part. Trajectory grows without bound or oscillates with increasing amplitude. No stable limit cycle. Output: undefined as oscillator.
- **Epistemological (non-terminating M):** M enters one of the three failure modes (Step 1). Evaluation sequence does not reach definite output. T=FALSE by condition (b).
- **Abstract isomorphism:** In both cases, failure is structural — not measurement error or boundary condition problem. Systematic divergence that no additional steps can resolve. System does not approach fixed point from either domain.

By IR-DRP3-3: convergent and divergent cases are exhaustive and exclusive. Isomorphism
holds in both (Step 3 for convergent, this step for divergent). Isomorphism is total —
covers entire partition of both systems.

**Output:**
- C-DRP3.6 [F] — Failure mode equivalence. Non-terminating M is epistemological analogue of μ>0. Failure modes structurally isomorphic under C-DRP3.4.
- C-DRP3.5 [F] — M↔C2 Isomorphism **COMPLETE.** Steps 3 and 4 together establish full isomorphism across entire partition.

**P10 check:** Boundary case (μ=0) addressed explicitly. No convergence case omitted. All constants trace to S0-A and S0-B.

---

## Step 5 — Independence Audit (produces C-DRP3.7)

- **Input:** S0-A (C-DRP3.1), S0-B (C-DRP3.2)
- **Rule:** IR-DRP3-4 (Level Distinction), IR-DRP3-2 (Received Axiom — citation audit)

**Derivation:**

C-DRP3.7 is a claim about the historical contingency of the isomorphism, not about
its mathematical validity. It asserts the two derivations were constructed
independently — neither cites the other.

**Arm 1 — DRP-1 §10.7 citation audit:**
The falsification predicate specifies DRP-1 §10.7 text as observation class (OQ-1
resolution). Question: does DRP-1 §10.7 cite P2 or any Hopf analysis in deriving
the finite-deterministic requirement for M?
STATUS: DRS_v1_1_S49.docx not yet available. Arm 1 is outstanding dependency for
C-DRP3.7 predicate at D3-3.0.

**Arm 2 — P2 citation audit:**
P2 (C-2.1) derives μ<0 from Hopf bifurcation analysis of the FRM delay differential
equation. The P2 derivation makes no reference to falsification predicates or the DRS.
STATUS: **CONFIRMED** from available sources. P2 does not cite DRP-1 or any DRS construction.

By IR-DRP3-4: Physical derivation (P2 → μ<0) operates at level of differential equation
analysis. Epistemological derivation (DRP-1 → M-termination) operates at level of logical
requirements for falsification. Both about convergence to fixed point; neither derives
its criterion by citing the other domain. Isomorphism is structural discovery, not
engineered equivalence.

**Output:** C-DRP3.7 [F] — Level independence. Arm 2 confirmed. Arm 1 pending
DRS_v1_1_S49.docx upload for D3-3.0 predicate finalisation.

**D3-3.0 note:** WHERE field of C-DRP3.7 predicate must specify DRP-1 §10.7 as
observation class. FALSIFIED IF citation to P2/Hopf found in that section.
Scoped to published formal derivation chain only (OQ-1 resolution constraint).

**P10 check:** Both arms anchor to live edges: S0-A and S0-B. No uncited constants.

---

## Derivation Table — Summary

| Step | Input(s) | Rule | Output |
|------|----------|------|--------|
| S0 | DRP-2 C-DRP2.4, P2 C-2.1 | IR-DRP3-2 | C-DRP3.1 [A], C-DRP3.2 [A] — received |
| 1 | C-DRP3.1 cond.(b) | IR-DRP3-2 (expand) | C-DRP3.3 [D] — M-termination |
| 2 | C-DRP3.1, C-DRP3.2, C-DRP3.3 | IR-DRP3-1 (framework) | C-DRP3.4 [D] — Convergence equiv. class |
| 3 | C-DRP3.3, C-DRP3.2, C-DRP3.4 | IR-DRP3-1 (isomorphism) | C-DRP3.5 [F] — partial (convergent case) |
| 4 | Step 3, C-DRP3.4, S0-A, S0-B | IR-DRP3-3 (partition) | C-DRP3.6 [F] + C-DRP3.5 [F] COMPLETE |
| 5 | S0-A, S0-B | IR-DRP3-4, IR-DRP3-2 (audit) | C-DRP3.7 [F] — Independence (Arm 2 confirmed; Arm 1 pending §10.7) |

All 7 claims produced. All trace to live edges S0-A and S0-B. No orphaned constants.

---

## Outstanding Dependency for D3-3.0

C-DRP3.7 Arm 1: DRS_v1_1_S49.docx must be uploaded and §10.7 read before C-DRP3.7
falsification predicate can be finalised.

Does not block C-DRP3.5 and C-DRP3.6 predicates. D3-3.0 can proceed in two passes:
- **Pass A:** C-DRP3.5, C-DRP3.6 predicates (no §10.7 needed)
- **Pass B:** C-DRP3.7 predicate (requires §10.7 read)

Or in one pass if DRS_v1_1_S49.docx is uploaded first.

---

## Phase Gate Status

| Phase | Status |
|-------|--------|
| D3-0.0 IR Inventory Audit | COMPLETE (S53) |
| D3-1.0 Claim Identification | COMPLETE (S54) |
| D3-2.0 Derivation Table | COMPLETE (S54, this document) |
| D3-3.0 Falsification Predicates | NEXT — C-DRP3.5/3.6 ready; C-DRP3.7 pending §10.7 |
| D3-4.0 Human Reader Pass | Follows D3-3.0 |
| D3-5.0 CBT Execution | Gate to PHASE-READY |

---

*D3-2.0 derivation table archived by Claude Code, Session S54.*
