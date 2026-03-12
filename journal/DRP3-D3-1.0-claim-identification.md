# DRP-3 D3-1.0 — Claim Identification

**Session:** S54
**Date:** 2026-03-12
**Source:** Thomas Brennan (claim design) + Claude Code (archive)

---

## Paper Identity

| Field | Value |
|-------|-------|
| Paper ID | DRP-3 |
| Working title | Evaluation Convergence and Physical Stability: The M↔C2 Isomorphism |
| Paper type | Type B (Derivation) |
| Protocol | DRP — Dual Reader Protocol |
| Gate cleared | DRP-2 PHASE-READY (satisfied) |
| Blockers | R-12 RESOLVED S54, R-13 RESOLVED S53/S54 |

**Central claim:** M-termination (finite deterministic evaluation) is epistemically
equivalent to μ<0 (Hopf stability condition). The same convergence requirement
operates at the physical level (FRM) and the epistemological level (DRS) for the
same mathematical reason.

---

## Claim Inventory: 7 claims — 2A · 2D · 3F

### Type A — Received Axioms (2)

**C-DRP3.1 [A] Testability Relation — received from DRP-2**

- Source: DRP-2, C-DRP2.4
- Statement: Testability is a ternary relation T(P, O, M). T = TRUE iff: (a) O causally independent of P; (b) M is a finite deterministic function O → {FALSIFIED, NOT FALSIFIED, INDETERMINATE}; (c) M(o) = FALSIFIED for at least one o ∈ O.
- Role in DRP-3: Condition (b) — M finite deterministic — is the epistemological side of the isomorphism.
- IR used: IR-DRP3-2 (Received Axiom)

**C-DRP3.2 [A] Hopf stability criterion — received from P2**

- Source: P2, C-2.1
- Statement: β=1/2 is the universal critical exponent for FRM universality class systems, derived from the Hopf bifurcation condition. μ < 0 is necessary and sufficient for convergent oscillation (stable limit cycle). μ > 0 produces divergent trajectory.
- Role in DRP-3: μ < 0 is the physical side of the isomorphism.
- IR used: IR-DRP3-2 (Received Axiom)

### Type D — Definitions (2)

**C-DRP3.3 [D] M-termination**

- Statement: M terminates iff M reaches a definite output in a finite number of evaluation steps for every input o ∈ O. M terminates ↔ M is a finite deterministic function per C-DRP3.1 condition (b). Failure modes: cycling (infinite loop), divergence (output grows without bound), or interpretation-dependence (requires human judgment).
- Relation to T: M-termination operationalises condition (b) of T(P,O,M).

**C-DRP3.4 [D] Convergence equivalence class**

- Statement: Two systems S1 and S2 belong to the same convergence equivalence class with respect to condition C iff: (a) both independently require C for well-defined operation; (b) the requirement is derivable from the same abstract mathematical structure; (c) the failure modes are structurally isomorphic.
- Role: Framework within which the M↔C2 isomorphism (C-DRP3.5) is proved.
- Outbound: Received by DRP-4 as structural context.

### Type F — Falsifiable Claims (3)

**C-DRP3.5 [F] The M↔C2 Isomorphism — CENTRAL CLAIM**

- Statement: M-termination is epistemically equivalent to μ < 0. Both are instances of the same abstract convergence requirement, derivable from the same Hopf stability analysis at two levels of description. They belong to the same convergence equivalence class (C-DRP3.4).
- Derivation: From C-DRP3.1 condition (b) + C-DRP3.2 [μ < 0]: both require the system trajectory to converge to a definite fixed point.
- IR used: IR-DRP3-1, IR-DRP3-3, IR-DRP3-4
- Outbound: DRP-4 receives as Type A (Convergence Principle). ARCH-2 PLACEHOLDER resolved when PHASE-READY.
- Vacuity witness (provisional): An FRM system with μ < 0 whose M does not terminate, or a predicate with terminating M whose physical analogue has μ > 0.

**C-DRP3.6 [F] Failure mode equivalence**

- Statement: Non-terminating M is the exact epistemological analogue of μ > 0 (unstable limit cycle). In both cases: no definite output or stable state; failure is structural, not contingent on measurement error; no amount of additional evaluation resolves it. Failure modes are structurally isomorphic under C-DRP3.4.
- Derivation: By IR-DRP3-3 (Exhaustive Partition): convergence admits exactly two cases. C-DRP3.5 covers convergent; C-DRP3.6 covers divergent.
- IR used: IR-DRP3-3, IR-DRP3-1
- Vacuity witness (provisional): Non-terminating M with μ < 0, or μ > 0 with terminating M.

**C-DRP3.7 [F] Level independence of the isomorphism**

- Statement: The isomorphism is not a definitional equivalence. It holds because the convergence requirement in each system is derivable independently from the same abstract structure. The DRS condition (b) was derived from machine-evaluability (DRP-1); the Hopf condition from delay differential equations (P2). Neither derivation cites the other.
- Derivation: From IR-DRP3-4 (Level Distinction) + derivation independence verified by citation audit.
- IR used: IR-DRP3-4, IR-DRP3-2
- Vacuity witness (provisional): Evidence that condition (b) of T(P,O,M) was derived by direct analogy with the Hopf condition in DRP-1.

---

## Dependency Map

**Inbound live edges:**
- DRP-2 C-DRP2.4 → C-DRP3.1 [A] (ARCH-2 LIVE)
- P2 C-2.1 → C-DRP3.2 [A] (ARCH-2 LIVE, via ARCH-3)

**Internal derivation chain:**
- C-DRP3.1 + C-DRP3.3 → C-DRP3.5 (convergent case)
- C-DRP3.2 + C-DRP3.3 → C-DRP3.5 (isomorphism)
- C-DRP3.4 → C-DRP3.5, C-DRP3.6, C-DRP3.7 (framework)
- C-DRP3.1 + C-DRP3.2 → C-DRP3.6 (divergent case)
- C-DRP3.1 + C-DRP3.2 → C-DRP3.7 (independence audit)

**Outbound expected edge:**
- C-DRP3.5 → DRP-4 (PLACEHOLDER in ARCH-2 until PHASE-READY)

---

## IR Inventory Confirmation

All five provisional rules from D3-0.0 confirmed in use:

| IR | Name | Used by |
|----|------|---------|
| IR-DRP3-1 | Structural Isomorphism | C-DRP3.5, C-DRP3.6 |
| IR-DRP3-2 | Received Axiom | C-DRP3.1, C-DRP3.2, C-DRP3.7 |
| IR-DRP3-3 | Exhaustive Partition (Convergence) | C-DRP3.5 + C-DRP3.6 (both cases) |
| IR-DRP3-4 | Level Distinction | C-DRP3.7 |
| IR-DRP3-5 | Vacuity Witness | C-DRP3.5, C-DRP3.6, C-DRP3.7 |

All five exercised. No additional rules identified.

---

## Open Question

**OQ-1 — C-DRP3.7 Independence Verification — RESOLVED S54**

C-DRP3.7 requires that condition (b) of T(P,O,M) was derived independently of the
Hopf analysis in P2.

**Resolution:** C-DRP3.7 stays **Type F**. Claim count unchanged at 2A + 2D + 3F = 7.

**Reasoning:**
1. C-DRP3.7 is independently falsifiable from C-DRP3.5. The isomorphism (C-DRP3.5)
   could hold structurally while the independence claim (C-DRP3.7) fails — the
   designer could have consciously imported the Hopf structure into the DRS design.
   The isomorphism would still be valid mathematics; it just wouldn't be an
   independent discovery.
2. The observation class is a citation audit of the published formal derivation
   chain: does DRP-1 §10.7 cite P2 or any Hopf analysis in deriving the
   finite-deterministic requirement for M? This is finite, deterministic, and
   capable of returning FALSIFIED.
3. Derivation independence is a contingent historical fact about how two specific
   documents were written, not a provable structural property of the mathematics.
   It could have gone either way. Therefore C-DRP3.7 is not Type D.

**Scope constraint for D3-3.0 predicate writing:**
The observation class must be scoped to the *published formal derivation chain* only.
"DRP-1 §10.7 cites P2/Hopf in the derivation of condition (b)" is testable.
"The author was thinking about Hopf when writing it" is not — private thought
processes are not observations in the DRS sense. The predicate must be written
against the published text. No handwaving assertions — the falsification path
requires a mathematical chain of causation traceable through the citation graph.

**Status:** RESOLVED S54. Does not block D3-2.0 or D3-3.0.

---

## Phase Gate Status

| Phase | Status |
|-------|--------|
| D3-0.0 IR Inventory Audit | COMPLETE (S53) |
| D3-1.0 Claim Identification | COMPLETE (S54, this document) |
| D3-2.0 Derivation Table | NEXT |
| D3-3.0 Falsification Predicates | Follows D3-2.0 |
| D3-4.0 Human Reader Pass | Follows D3-3.0 |
| D3-5.0 CBT Execution | Gate to PHASE-READY |

OQ-1 to be resolved before D3-3.0.

---

*D3-1.0 claim identification archived by Claude Code, Session S54.*
