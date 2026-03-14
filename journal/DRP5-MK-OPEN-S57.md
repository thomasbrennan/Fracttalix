# DRP-5 MK-OPEN — Session 57

**Paper:** DRP-5 — Underdetermination and the Relational Account
**Series position:** DRP-5 of 8
**Paper type:** derivation_B
**Session:** S57
**Date:** 2026-03-13
**Operator:** Claude (Anthropic) + Thomas Brennan
**CBP stage:** MK-OPEN (Canonical Build Process)

---

## Central Question

DRP-5 receives the Full Conservation Law (C-DRP4.5) from DRP-4 as a Type A axiom and must prove:

1. Precise Duhem-Quine boundary conditions (UMP is epistemically prior to DQ)
2. Regress termination (if T2 must satisfy UMP, what governs T3?)
3. C3 = FRM's complete DQ solution (tau_gen independence = T2 causal independence)
4. Auxiliary hypothesis protocol (third-party constructable, handed off to DRP-6)

**Core risk question:** The Duhem-Quine problem has a 70+ year philosophical literature. What is the minimum engagement to state boundary conditions precisely without writing a philosophy dissertation? Is there a regress risk that terminates the entire chain?

---

## Inbound Edges

| Source | Claim | Label | Status |
|--------|-------|-------|--------|
| DRP-4 | C-DRP4.5 | Full Conservation Law: I(M(O);P) = 0 for all M when O = f(P) | LIVE (DRP-4 PHASE-READY S56) |
| DRP-2 | C-DRP2.4 | Testability Relation T(P,O,M) | LIVE (structural reference) |
| DRP-2 | §5.3 proof sketch | UMP-precedes-DQ argument | LIVE (DRP-2 v1.1 content) |
| P3 | D-3.1 | FRM System Definition (C3 = independence requirement) | LIVE (P3 PHASE-READY S48) |
| P3 | D-3.2 | tau_gen Extraction Protocol | LIVE (P3 PHASE-READY S48) |
| P3 | C-3.REG | FRM Regression Protocol | LIVE (P3 PHASE-READY S48) |

## Outbound Edge (Expected)

| Target | Claim | Label | Status |
|--------|-------|-------|--------|
| DRP-6 | C-DRP5.5 (expected) | Auxiliary Hypothesis Protocol | PLACEHOLDER until DRP-5 built |

## FRM Load Map

| FRM Source | Connection | Dep Type | Status |
|------------|-----------|----------|--------|
| P3 | C3 (tau_gen independent measurability) is the FRM's complete DQ solution. P3 provides measurement protocol showing T2 causal independence. | E (empirical) | LIVE — P3 PHASE-READY |

---

## MK-OPEN Candidate Analysis

### Candidate 1: Scope Discipline — Minimum DQ Engagement

**Description:** DRP-5 must engage the DQ literature precisely enough to state boundary conditions, not deeply enough to resolve all philosophical debate. The paper proves UMP is epistemically prior to DQ (already sketched in DRP-2 §5.3). DQ becomes a special case: given UMP compliance (O independent of P), the question shifts to whether background theory T2 also satisfies UMP with respect to P. This is a precise, narrow question — not a survey of Quine vs. Duhem vs. Lakatos.

**Risk if omitted:** Scope creep into philosophy of science literature. DRP-5 becomes a review paper instead of a derivation paper.

**KVS Assessment:**
- Novelty: 0.7 (reframing DQ as UMP-subordinate is novel)
- Impact: 0.9 (controls entire paper scope)
- Inverse Complexity: 0.8 (scope constraint simplifies)
- Timeliness: 0.9 (must be decided before Phase 1)

**KVS = (0.7 × 0.9 × 0.8 × 0.9)^(1/4) = (0.4536)^0.25 = 0.820**

**Verdict: PASS (KVS ≥ 0.50)**

---

### Candidate 2: Regress Termination Strategy — Causal Grounding

**Description:** The regress T2 → T3 → T4 → ... must terminate. Two options identified in the build plan: (a) prove it terminates at some T_n causally upstream; (b) show it is practically bounded without affecting non-vacuity. Option (a) is available via the FRM: in physical systems, the causal chain terminates at independently measurable physical quantities (tau_gen in P3 is measured by molecular biology, not by FRM prediction). The regress terminates wherever T_n is grounded in direct physical measurement independent of P. This is C3 generalised: the same structural condition that resolves DQ for the FRM resolves the regress for the general case.

**Risk if omitted:** HR will target the regress directly. An unaddressed regress kills the paper.

**KVS Assessment:**
- Novelty: 0.8 (linking regress termination to C3-type grounding is novel)
- Impact: 1.0 (paper fails without it)
- Inverse Complexity: 0.6 (regress argument requires careful formulation)
- Timeliness: 1.0 (must be resolved Phase 1)

**KVS = (0.8 × 1.0 × 0.6 × 1.0)^(1/4) = (0.48)^0.25 = 0.832**

**Verdict: PASS (KVS ≥ 0.50)**

---

### Candidate 3: IR Inventory Pre-Check

**Description:** DRP-5 requires inference rules covering: (a) epistemological ordering arguments (UMP precedes DQ), (b) causal independence conditions, (c) protocol specification steps. The existing DRP-series IR inventories include IR-DRP4-1 (Received Axiom), and the standard suite (Modus Ponens, Definition Expansion, etc.). DRP-5 will likely need: (i) Epistemological Ordering — if A is necessary for B to be well-posed, A is epistemically prior to B; (ii) Causal Independence Transfer — if T2 is causally independent of P and O is constructed via T2, then O inherits causal independence from P. Both are derivable from existing rules but should be named explicitly for auditability.

**Risk if omitted:** Mid-build discovery of unnamed inference steps causes build failure (same issue MK-2 Amendment 1 added D3-0.0 to prevent).

**KVS Assessment:**
- Novelty: 0.4 (process safeguard, not intellectual novelty)
- Impact: 0.7 (prevents build failure)
- Inverse Complexity: 0.9 (simple audit)
- Timeliness: 0.9 (must precede Phase 1)

**KVS = (0.4 × 0.7 × 0.9 × 0.9)^(1/4) = (0.2268)^0.25 = 0.690**

**Verdict: PASS (KVS ≥ 0.50)**

---

### Candidate 4: Auxiliary Hypothesis Protocol — Third-Party Constructability Standard

**Description:** The auxiliary hypothesis protocol (D5-2.1) must be third-party constructable: a researcher who has not read DRP-5 must be able to apply it from the description alone. This means every step specifies operator, inputs, and output constituting PASS. No step requiring author judgment. The protocol governs DRP-6 benchmark construction. If the protocol is ambiguous, DRP-6 inherits that ambiguity into its empirical results.

**Risk if omitted:** DRP-6 receives an ambiguous protocol. Empirical results become uncheckable. Type C integrity compromised.

**KVS Assessment:**
- Novelty: 0.5 (protocol design is standard engineering)
- Impact: 0.9 (DRP-6 depends on it entirely)
- Inverse Complexity: 0.6 (third-party standard is hard to achieve)
- Timeliness: 0.8 (deliverable at Phase 2)

**KVS = (0.5 × 0.9 × 0.6 × 0.8)^(1/4) = (0.216)^0.25 = 0.682**

**Verdict: PASS (KVS ≥ 0.50)**

---

### Candidate 5: C-DRP5.4 Type Classification — Historical vs Derivational

**Description:** The build plan warns: "C3 was doing DQ work before DQ was named" is historical framing belonging in §5 as context, not as a Type F claim. C-DRP5.4 (C3 as DQ solution) must be derivational: prove structurally that C3 satisfies the T2-independence requirement derived in D5-1.1/D5-1.2. The historical observation that P3 was written without knowledge of DQ is interesting context but not a claim.

**Risk if omitted:** C-DRP5.4 gets classified Type F with a historical predicate that isn't genuinely falsifiable. HR catches it.

**KVS Assessment:**
- Novelty: 0.3 (classification discipline)
- Impact: 0.7 (prevents misclassification)
- Inverse Complexity: 0.9 (simple type decision)
- Timeliness: 0.7 (Phase 1 decision)

**KVS = (0.3 × 0.7 × 0.9 × 0.7)^(1/4) = (0.1323)^0.25 = 0.603**

**Verdict: PASS (KVS ≥ 0.50)**

---

## MK-OPEN Summary

| # | Candidate | KVS | Verdict |
|---|-----------|-----|---------|
| 1 | Scope Discipline — Minimum DQ Engagement | 0.820 | PASS |
| 2 | Regress Termination — Causal Grounding | 0.832 | PASS |
| 3 | IR Inventory Pre-Check | 0.690 | PASS |
| 4 | Auxiliary Protocol — Third-Party Standard | 0.682 | PASS |
| 5 | C-DRP5.4 Type Classification | 0.603 | PASS |

All 5 candidates pass KVS ≥ 0.50. Zero candidates rejected.

---

## MK-OPEN Decision

**PASS.** Proceed to DRP-5 Phase 0.

Key commitments from MK-OPEN:
1. **Scope:** DRP-5 engages DQ only to state boundary conditions. UMP-prior-to-DQ is the structural result. No philosophy survey.
2. **Regress:** Must be addressed in D5-1.2 with causal grounding argument. C3 generalised.
3. **IR audit:** Named inference rules for epistemological ordering and causal independence transfer before Phase 1.
4. **Protocol:** Third-party constructable auxiliary hypothesis protocol is a hard deliverable.
5. **Type discipline:** C-DRP5.4 is derivational, not historical. Historical context in §5 only.

---

*MK-OPEN produced by Claude Code, Session 57.*
