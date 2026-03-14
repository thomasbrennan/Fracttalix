# DRP-5 D5-0.0 — IR Inventory Pre-Check

**Paper:** DRP-5 — Underdetermination and the Relational Account
**Session:** S57
**Date:** 2026-03-13
**Purpose:** Audit IR inventory against DRP-5 proof structure before Phase 1 (per MK-OPEN Candidate 3)

---

## Required Proof Steps

DRP-5 must execute the following categories of derivation:

### Category A: Epistemological Ordering
- **Step type:** Prove UMP is epistemically prior to DQ — i.e., UMP must be satisfied before the DQ problem is even well-posed.
- **Argument structure:** If condition A is necessary for condition B to be meaningful, A precedes B in epistemological order.
- **Covering IR:** IR-1 (Modus Ponens) covers the logical structure. But the *ordering* claim needs naming.
- **GAP:** Need IR-DRP5-1: Epistemological Ordering.

### Category B: Causal Independence Transfer
- **Step type:** If T2 is causally independent of P and O is constructed using T2, then O inherits causal independence from P.
- **Argument structure:** Causal independence is preserved under composition when no step in the construction introduces a dependency on P.
- **Covering IR:** IR-DRP4-3 (Composition Preservation) covers function composition. The causal independence claim extends this to causal chains.
- **GAP:** Need IR-DRP5-2: Causal Independence Transfer.

### Category C: Regress Termination
- **Step type:** Prove the T2 → T3 → ... regress terminates at physical measurement.
- **Argument structure:** If T_n is grounded in direct physical measurement independent of P, no further T_{n+1} is required for causal independence. The chain is well-founded.
- **Covering IR:** No existing IR covers well-foundedness of causal chains.
- **GAP:** Need IR-DRP5-3: Causal Grounding (well-founded termination).

### Category D: Protocol Specification
- **Step type:** Derive auxiliary hypothesis protocol steps.
- **Covering IR:** IR-12 (Protocol Specification, from P3) covers step validity. No gap.

### Category E: Standard Operations
- **Step type:** Received Axiom, Definition Expansion, Modus Ponens, Substitution, etc.
- **Covering IR:** IR-1 through IR-8 (schema) plus IR-DRP4-1 (Received Axiom). No gap.

---

## GAP Register

| GAP ID | Required Step | Proposed IR | Justification |
|--------|--------------|-------------|---------------|
| GAP-D5-1 | Epistemological ordering | IR-DRP5-1: Epistemological Ordering | "If A is a necessary condition for B to be well-posed, then A is epistemically prior to B." Derived from IR-1 + IR-4. Not reducible to either alone because the ordering claim is a meta-level statement about the logical structure of the proof, not a step within it. |
| GAP-D5-2 | Causal independence transfer | IR-DRP5-2: Causal Independence Transfer | "If T2 is causally independent of P, and O is constructed solely from T2-governed measurements, then O is causally independent of P." Extends IR-DRP4-3 (composition preservation) from deterministic function composition to causal chain analysis. |
| GAP-D5-3 | Regress termination | IR-DRP5-3: Causal Grounding | "If T_n is grounded in direct physical measurement whose value is determined independently of P, the causal independence chain terminates at T_n. No T_{n+1} is required." Well-foundedness argument. Not reducible to existing IR because it addresses the structure of the chain, not a step within it. |

---

## Proposed IR Inventory for DRP-5

| ID | Name | Form | Source |
|----|------|------|--------|
| IR-DRP5-1 | Epistemological Ordering | If A is necessary for B to be well-posed, A is epistemically prior to B | New — DRP-5 Phase 0 (S57) |
| IR-DRP5-2 | Causal Independence Transfer | If T2 ⊥ P and O is constructed solely via T2, then O ⊥ P | Extends IR-DRP4-3 — DRP-5 Phase 0 (S57) |
| IR-DRP5-3 | Causal Grounding | If T_n is grounded in direct physical measurement independent of P, the independence chain is well-founded and terminates at T_n | New — DRP-5 Phase 0 (S57) |
| IR-DRP4-1 | Received Axiom | A claim proved in a prior paper with LIVE status may be received as Type A without re-derivation | Inherited from DRP-4 |
| IR-DRP5-4 | Vacuity Witness | A falsification predicate is non-vacuous (C6) iff a coherent observation exists that would trigger FALSIFIED | Inherited from DRP-4 IR-DRP4-5 |
| IR-1–IR-8 | Standard schema rules | (see ai-layer-schema.json) | Schema |
| IR-12 | Protocol Specification | A measurement protocol step is valid iff it has named input, deterministic operation, and named output | Inherited from P3 |

---

## Verdict

**3 gaps identified. 3 proposals registered.** All gaps have amendment proposals. No uncovered step without a proposal.

**PASS** — Proceed to D5-0.1 (Claim Init).

---

*IR pre-check produced by Claude Code, Session 57.*
