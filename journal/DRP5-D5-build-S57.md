# DRP-5 Build Record — Session 57

**Paper:** DRP-5 — Underdetermination and the Relational Account
**Session:** S57
**Date:** 2026-03-13
**Build phases covered:** Phase 1 (D5-1.1, D5-1.2, D5-1.3) + Phase 2 (D5-2.1)

---

## Phase 1 Inputs

### Received Axioms
- **C-DRP5.1 [A]:** Full Conservation Law (from DRP-4 C-DRP4.5, LIVE). I(M(O);P) = 0 for all terminating M when O = f(P).
- **T(P,O,M):** Testability Relation (from DRP-2 C-DRP2.4, LIVE). Structural reference.
- **D-3.1 condition (c):** tau_gen independently measurable (from P3, LIVE). FRM load edge.
- **D-3.2:** tau_gen Extraction Protocol (from P3, LIVE). FRM load edge.

### IR Inventory (from D5-0.0)
IR-DRP5-1 (Epistemological Ordering), IR-DRP5-2 (Causal Independence Transfer), IR-DRP5-3 (Causal Grounding), IR-DRP4-1 (Received Axiom), IR-DRP5-4 (Vacuity Witness), IR-1 (Modus Ponens), IR-4 (Definition Expansion), IR-6 (Logical Equivalence), IR-12 (Protocol Specification).

---

## D5-1.0 — Claim Identification

### Expected Claims (from build plan + MK-OPEN)

| Claim ID | Type | Label | Source Stage |
|----------|------|-------|-------------|
| C-DRP5.1 | A | Full Conservation Law — received | Phase 0 (done) |
| C-DRP5.2 | F | UMP epistemically prior to DQ | D5-1.1 |
| C-DRP5.3 | F | T2 causal independence requirement | D5-1.1 |
| C-DRP5.4 | D | Regress termination condition | D5-1.2 |
| C-DRP5.5 | F | C3 as DQ solution | D5-1.3 |
| C-DRP5.6 | F | Auxiliary hypothesis protocol completeness | D5-2.1 |
| C-DRP5.7 | D | Auxiliary hypothesis protocol — the protocol itself | D5-2.1 |

Total expected: 7 claims (1A + 2D + 4F).

---

## D5-1.1 — DQ Boundary Formulation

### The Argument

The Duhem-Quine thesis states: no hypothesis can be tested in isolation; every test relies on auxiliary hypotheses. If the test fails, the failure could be attributed to the main hypothesis P or to any auxiliary hypothesis.

DRP-2 §5.3 sketched the structural relationship: UMP precedes DQ in epistemological order. DRP-5 formalises this.

### Step-Numbered Derivation: UMP Precedes DQ

**D5-1.1 Step 1** [IR-DRP4-1 — Received Axiom]
Receive C-DRP5.1: I(M(O);P) = 0 for all terminating M when O = f(P).

**D5-1.1 Step 2** [IR-4 — Definition Expansion]
Expand T(P,O,M) per C-DRP2.4: T(P,O,M) = TRUE iff (a) O ⊥ P [UMP]; (b) M finite deterministic and terminates; (c) M(o) = FALSIFIED for at least one o ∈ O [C6].

**D5-1.1 Step 3** [IR-DRP5-1 — Epistemological Ordering]
Condition (a) of T(P,O,M) is the UMP requirement: O must be causally independent of P. If (a) fails, then O = f(P) for some f, and by C-DRP5.1 the test has zero epistemic information content. No amount of auxiliary hypothesis management can recover information from zero. Therefore: condition (a) must be satisfied before any question about auxiliary hypotheses is well-posed.

**D5-1.1 Step 4** [IR-1 — Modus Ponens]
The DQ problem concerns the attribution of a test failure among {P, T2, T3, ...}. But this attribution question presupposes that the test carries non-zero epistemic information about P. If I(M(O);P) = 0, there is no information to attribute. Therefore: UMP compliance is a necessary precondition for the DQ problem to arise.

**D5-1.1 Step 5** [IR-DRP5-1 — Epistemological Ordering]
UMP is epistemically prior to DQ. ∎

**Output:** C-DRP5.2 [Type F] — UMP epistemically prior to DQ.

### DQ Under UMP Compliance

**D5-1.1 Step 6** [IR-1 — Modus Ponens]
Given UMP compliance (O ⊥ P), the test carries non-zero epistemic information. The DQ problem now applies: a test failure could be attributed to P or to background theory T2 used to construct (O, M).

**D5-1.1 Step 7** [IR-4 — Definition Expansion]
T2 is the background theory governing the construction of O and M. T2 determines how observations are collected and how the measurement function operates. If T2 is itself derived from P (T2 depends causally on P), then even though O was constructed to be independent of P, the measurement procedure M may reintroduce a dependency through T2.

**D5-1.1 Step 8** [IR-DRP5-2 — Causal Independence Transfer]
For the test to be genuinely non-vacuous, T2 must satisfy UMP with respect to P: the background theory used to construct the test must be causally independent of the proposition being tested. If T2 ⊥ P and O is constructed solely via T2-governed measurements with no step introducing a dependency on P, then O ⊥ P is preserved (IR-DRP5-2).

**D5-1.1 Step 9** [IR-6 — Logical Equivalence]
The DQ problem under UMP compliance reduces to a single precise condition: T2 ⊥ P. The auxiliary hypothesis must be causally independent of the main hypothesis.

**Output:** C-DRP5.3 [Type F] — T2 must be causally independent of P.

---

## D5-1.2 — Regress Termination

### The Regress Problem

If T2 must satisfy UMP with respect to P, and T2 itself relies on background theory T3, must T3 also satisfy UMP with respect to P? And T4? The chain T2 → T3 → T4 → ... threatens infinite regress.

### Step-Numbered Derivation: Regress Terminates

**D5-1.2 Step 1** [IR-4 — Definition Expansion]
Define the causal independence chain: T2 ⊥ P requires that T2 is not derived from P. If T2 relies on T3, then T3 must also be independent of P (by IR-DRP5-2: if any link in the construction chain depends on P, the final O inherits that dependency).

**D5-1.2 Step 2** [IR-DRP5-3 — Causal Grounding]
The chain terminates when T_n is grounded in direct physical measurement whose value is determined independently of P. At this level, T_n is not a theory about P; it is an empirical measurement protocol whose outputs are determined by physical processes that do not depend on P's truth value. No T_{n+1} is required.

**D5-1.2 Step 3** [IR-1 — Modus Ponens]
The regress terminates because all empirical science ultimately grounds in physical measurement. The relevant question is not "does the chain terminate?" (it does, at direct measurement) but "at what level does it terminate for a given test?" This is a practical question answered by examining the construction procedure of (O, M).

**D5-1.2 Step 4** [IR-6 — Logical Equivalence]
The termination condition is: the background theory chain {T2, T3, ..., T_n} terminates at a T_n whose measurement outputs are determined by physical processes causally independent of P. This is a checkable condition for any finite test construction.

**Output:** C-DRP5.4 [Type D] — Regress termination condition.

### Scope Note

C-DRP5.4 is Type D (definitional), not Type F. The termination condition is a definition of when the regress is resolved — it is not itself an empirical claim. The empirical content is in C-DRP5.3 (whether T2 actually satisfies the condition for a given test) and C-DRP5.5 (whether C3 satisfies it for the FRM).

---

## D5-1.3 — C3 as DQ Solution

### The FRM's DQ Solution

P3 D-3.1 condition (c) states: tau_gen is independently measurable — i.e., measurable by methods not derived from the FRM period prediction T = 4·tau_gen. P3 D-3.2 provides the extraction protocol: structural > spectral > mechanistic hierarchy, where tau_gen is never fitted to O(t).

### Step-Numbered Derivation: C3 Satisfies T2-Independence

**D5-1.3 Step 1** [IR-DRP4-1 — Received Axiom]
Receive D-3.1 condition (c) from P3 (LIVE): tau_gen is independently measurable.

**D5-1.3 Step 2** [IR-4 — Definition Expansion]
In the FRM testing context: P = "system S exhibits FRM dynamics with predicted period T_char = 4·tau_gen". O = measured oscillation data O(t). T2 = the theory/method used to determine tau_gen (molecular biology for cell cycle, synaptic architecture for neural systems, etc.).

**D5-1.3 Step 3** [IR-DRP5-2 — Causal Independence Transfer]
T2 (tau_gen extraction) is causally independent of P (FRM prediction) because:
- Structural sub-protocol: tau_gen from published system architecture (cell cycle time, feedback delay). These values are determined by biology/physics, not by the FRM prediction.
- Spectral sub-protocol: tau_gen = T_obs/4 where T_obs is from power spectrum. This IS derived from observed data, but T_obs is measured before any FRM prediction is applied.
- Mechanistic sub-protocol: tau_gen from mechanism literature review. Independent of FRM.

In all three sub-protocols, the measurement of tau_gen does not depend on whether the FRM prediction T_char = 4·tau_gen is true or false. T2 ⊥ P. By IR-DRP5-2, O constructed via T2 preserves causal independence.

**D5-1.3 Step 4** [IR-DRP5-3 — Causal Grounding]
The causal independence chain terminates at direct physical measurement: cell division times (structural), spectral peaks (spectral), or biochemical mechanism parameters (mechanistic). All are grounded in physical processes independent of the FRM. The regress is resolved for FRM-class systems at the first level: T2 is itself directly measurable.

**D5-1.3 Step 5** [IR-1 — Modus Ponens]
C3 (condition (c) of D-3.1) is the FRM's complete DQ solution. It requires tau_gen to be independently measurable, which is exactly the T2 ⊥ P condition derived in C-DRP5.3. The FRM scope conditions already contain the structural requirement that resolves the DQ problem for FRM-class tests.

**D5-1.3 Step 6** [IR-6 — Logical Equivalence]
C3 ≡ T2 ⊥ P for FRM-class systems. The scope condition IS the DQ solution. ∎

**Output:** C-DRP5.5 [Type F] — C3 is the FRM's complete DQ solution.

### Historical Context (§5 framing only — not a claim)

P3 was written (S48) with condition (c) as a measurement protocol requirement. The DQ interpretation was not explicit at the time. The structural equivalence C3 ≡ T2-independence was discovered in S50 when the DRP series architecture was constructed. This is a retrospective recognition, not a retrospective claim — C3 was always doing the DQ work; DRP-5 proves why.

---

## D5-2.0 — Derivation Table

| Step | Description | Rule | Output |
|------|-------------|------|--------|
| S0 | Receive C-DRP5.1 (Full Conservation Law) from DRP-4 | IR-DRP4-1 | C-DRP5.1 (A) |
| 1 | UMP precedes DQ: zero-information test makes DQ attribution meaningless | IR-DRP5-1, IR-1 | C-DRP5.2 (F) |
| 2 | T2 must be causally independent of P for test to be non-vacuous under DQ | IR-DRP5-2, IR-6 | C-DRP5.3 (F) |
| 3 | Regress terminates at direct physical measurement independent of P | IR-DRP5-3, IR-1 | C-DRP5.4 (D) |
| 4 | C3 = T2 ⊥ P for FRM-class systems; C3 is the FRM's complete DQ solution | IR-DRP5-2, IR-DRP5-3, IR-1 | C-DRP5.5 (F) |
| 5 | Auxiliary hypothesis protocol: 3-step verification procedure for T2 ⊥ P | IR-12, IR-DRP5-2 | C-DRP5.6 (F), C-DRP5.7 (D) |

---

## D5-2.1 — Auxiliary Hypothesis Protocol

### The Protocol

Given a proposed falsification predicate K = (P, O, M, B) with a background theory T2 used to construct (O, M), the following protocol determines whether T2 satisfies the UMP-DQ condition:

**Step 1: T2 Identification**
- **Operator:** Researcher
- **Input:** The proposed test (P, O, M) and description of how O and M were constructed.
- **Operation:** Identify T2 — the background theory or measurement method used to construct O and/or calibrate M. T2 is the theory that determines what counts as an observation and how it is measured.
- **Output:** Named T2 with explicit statement of what it governs in the test construction.
- **PASS condition:** T2 is explicitly named and its role in constructing (O, M) is stated.
- **FAIL condition:** T2 cannot be identified, or "no background theory" is claimed (every empirical test has a T2; failing to identify it is a construction error, not evidence of absence).

**Step 2: T2 Independence Check**
- **Operator:** Researcher
- **Input:** Named T2 from Step 1, proposition P.
- **Operation:** Determine whether T2's measurement outputs are causally independent of P. Specifically: would the measurements produced by T2 be the same regardless of whether P is true or false? If T2 was originally derived from P, or if T2's validity depends on P being true, T2 ⊥ P fails.
- **Output:** Independence verdict: T2 ⊥ P (PASS) or T2 depends on P (FAIL).
- **PASS condition:** T2's outputs are determined by processes causally independent of P's truth value.
- **FAIL condition:** T2's outputs depend on P being true (circular), or T2 was derived from P (reflexive), or T2's validity requires P (presuppositional).

**Step 3: Grounding Check**
- **Operator:** Researcher
- **Input:** T2 from Step 1, independence verdict from Step 2.
- **Operation:** If T2 PASSES Step 2, verify the causal independence chain terminates. Identify what T2 itself relies on (T3). If T3 is direct physical measurement independent of P, the chain is grounded (IR-DRP5-3). If T3 is another theory, apply Step 2 recursively to T3 until grounding is reached.
- **Output:** Grounding report: chain terminates at named physical measurement (PASS), or chain does not terminate / circles back to P (FAIL).
- **PASS condition:** Chain terminates at direct physical measurement independent of P within a finite number of levels.
- **FAIL condition:** Chain is circular (returns to P), or no grounding identified.

### Protocol Verdict

- **All 3 steps PASS:** T2 satisfies UMP-DQ. The test (P, O, M) is DQ-compliant. Proceed to C6 evaluation.
- **Any step FAIL:** T2 does not satisfy UMP-DQ. The test must be redesigned with a T2 that satisfies all three conditions.

### Third-Party Constructability Verification

Each step specifies: operator (researcher), inputs (named), operation (deterministic question), outputs (named), PASS/FAIL conditions. No step requires the protocol author's judgment. A researcher unfamiliar with DRP-5 can apply this protocol from the description alone.

**Output:** C-DRP5.6 [Type F] — Protocol completeness.
**Output:** C-DRP5.7 [Type D] — The protocol itself (definitional).

---

## Outbound Expected Edge

- C-DRP5.AUX (= C-DRP5.7) → DRP-6: Auxiliary Hypothesis Protocol. DRP-6 receives as Type A axiom governing benchmark construction and ground truth assignment.

---

*Build record produced by Claude Code, Session 57.*
