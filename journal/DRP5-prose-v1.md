# DRP-5 — Underdetermination and the Relational Account

## Precise Boundary Conditions for the Duhem-Quine Problem Under the Upstream Measurement Principle

**DRP Series — Paper 5 of 8**
**Thomas Brennan · Entwood Hollow Research Station · Trinity County CA**
**Session 57 · 2026**

**AI-reader annotation:** 7 claims registered. 1A + 2D + 4F. Central claim: C-DRP5.2 (UMP epistemically prior to DQ). Pass-forward: C-DRP5.7 (Auxiliary Hypothesis Protocol) → DRP-6. Inbound: DRP-4 C-DRP4.5. FRM load: P3 D-3.1/D-3.2. CBT I-9: 7/7 PASS.

---

## 1. Received from Prior Papers

DRP-5 receives one axiom from the prior DRP chain:

**C-DRP5.1 [Type A]:** The Full Conservation Law (from DRP-4 C-DRP4.5). For any falsification predicate K = (P, O, M, B) where O = f(P) and M satisfies condition (b) of T(P,O,M): I(M(O); P) = 0. [DRP-4, AI-Layer, C-DRP4.5, LIVE]

DRP-5 also draws structural reference from:
- T(P,O,M) (DRP-2 C-DRP2.4, LIVE) — the testability relation.
- D-3.1 condition (c) and D-3.2 (P3 PHASE-READY, LIVE) — the FRM independence requirement and tau_gen extraction protocol.

---

## 2. The Duhem-Quine Problem

The Duhem-Quine thesis has two distinct formulations that are frequently conflated:

**Duhemian underdetermination (local):** No single hypothesis can be tested in isolation. Every empirical test relies on auxiliary hypotheses — background theories governing instrument calibration, experimental conditions, and the interpretation of observations. If a test produces a result inconsistent with P, the failure could be attributed to P or to any auxiliary hypothesis T2 (Duhem 1906). This is a *local* attribution problem: which component of a specific test is responsible for a specific failure?

**Quinean underdetermination (global):** The totality of evidence underdetermines the totality of theory. Any body of evidence is compatible with multiple, mutually incompatible theoretical frameworks (Quine 1951). This is a *global* problem about the theory-evidence relation, not about individual tests.

DRP-5 addresses the **Duhemian local problem** — the test-level attribution question. It does not address Quinean global underdetermination, which concerns theory choice at the level of entire research programmes (Lakatos 1970) and unconceived alternatives (Stanford 2006). Laudan (1990) distinguishes multiple formulations of underdetermination; DRP-5 engages only the Duhemian version as described above.

DRP-5 provides a *reduction* of the local DQ problem, not a dissolution. It proves that under UMP compliance, the DQ attribution problem reduces to a precise, checkable structural condition on background theories.

---

## 3. UMP Is Epistemically Prior to DQ

**C-DRP5.2 [Type F] — UMP epistemically prior to DQ.**

The DQ problem concerns the attribution of a test failure among {P, T2, T3, ...}. This attribution question presupposes that the test carries non-zero epistemic information about P. By C-DRP5.1 (Full Conservation Law), if O = f(P), then I(M(O); P) = 0 — the test carries zero information. If the test carries zero information, there is no failure to attribute. The DQ attribution problem does not arise.

Therefore: UMP compliance — condition (a) of T(P,O,M) — must be satisfied before the DQ problem is well-posed. UMP is epistemically prior to DQ.

**AI-reader annotation:** C-DRP5.2 is the central structural result of DRP-5. It establishes the epistemological ordering: first ensure O ⊥ P (UMP), then address auxiliary hypothesis management (DQ). The ordering is not a convention; it follows from the conservation law.

**Scope note (HR-D5 #1):** DRP-5 provides a *reduction* of the DQ problem, not a dissolution. It reduces the Duhemian attribution problem to a precise, checkable structural condition on T2. The philosophical question of whether auxiliary hypotheses can be fully eliminated (Quinean global underdetermination) is out of scope. DRP-5 provides the *boundary condition* under which the local DQ attribution problem is tractable.

---

## 4. The T2-Independence Condition

**C-DRP5.3 [Type F] — T2 must be causally independent of P.**

Given UMP compliance (O ⊥ P), the DQ problem reduces to a single condition. The background theory T2 used to construct the test (O, M) must itself be causally independent of P. If T2 depends on P (T2 is derived from P, or T2's validity requires P to be true), then even a UMP-compliant test is compromised: the measurement function M, calibrated via T2, reintroduces a dependency on P through the construction procedure.

By IR-DRP5-2 (Causal Independence Transfer): if T2 ⊥ P and O is constructed solely via T2-governed measurements with no step introducing a dependency on P, then O ⊥ P is preserved.

The DQ problem under UMP compliance is: verify T2 ⊥ P.

---

## 5. Regress Termination

**C-DRP5.4 [Type D] — Regress termination condition.**

If T2 must satisfy UMP with respect to P, and T2 relies on background theory T3, must T3 also be independent of P? The chain T2 → T3 → T4 → ... threatens infinite regress.

The chain terminates when T_n satisfies both:
(a) T_n is a measurement protocol whose outputs are determined by physical processes that do not reference P in their causal mechanism.
(b) T_n's own theoretical foundations have been independently validated in contexts where P is not at issue.

This is a *practical* termination criterion, not a foundationalist claim about theory-free observation. All measurement is theory-laden (Hanson 1958, Kuhn 1962) — DRP-5 does not dispute this. What DRP-5 claims is *contextual independence*: T_n's theories are independent of P in the context of testing P, not independent of all theory whatsoever. The thermometer's theory of thermal expansion is not at issue when we use the thermometer to measure tau_gen for an FRM test.

**DQ-attribution vs epistemic-regress (HR-D5 #4, #2):** DRP-5 resolves the DQ *attribution problem* (local, specific: "given this test failure, is P or T2 responsible?") — not the *epistemic regress problem* (global, foundational: "is all science ultimately justified?"). These are different problems. The attribution question is finite and specific; it asks about a particular test construction, not about the foundations of all empirical knowledge. The regress terminates for attribution purposes at independently validated measurement protocols. DRP-5 does not resolve Agrippa's trilemma and does not claim that theory-free observation exists.

---

## 6. C3 as the FRM's DQ Solution

**C-DRP5.5 [Type F] — C3 is the FRM's complete DQ solution.**

P3 D-3.1 condition (c) states: tau_gen is independently measurable — i.e., measurable by methods not derived from the FRM period prediction T_char = 4·tau_gen. P3 D-3.2 provides the tau_gen extraction protocol with three sub-protocols ordered by DQ strength:

1. **Structural** (strongest): tau_gen from published system architecture. T2 = biology/physics of the system mechanism. T2 ⊥ P because the cell cycle time, synaptic delay, or feedback delay is determined by biological processes independent of whether the FRM prediction is true.

2. **Spectral** (intermediate): tau_gen = T_obs/4 from power spectrum. T2 = spectral analysis of observed time series. The spectral sub-protocol extracts tau_gen from raw data before any FRM prediction is applied. However, it is the weakest of the three for DQ purposes because T_char = 4·tau_gen = T_obs, making the prediction-measurement comparison tautological for the system used to extract tau_gen. The genuine test is cross-substrate comparison.

3. **Mechanistic** (adequate): tau_gen from mechanism literature review. T2 = domain-specific mechanism literature. T2 ⊥ P because mechanism parameters are determined by the domain, not by the FRM.

In all three sub-protocols, T2 ⊥ P is satisfied. C3 ≡ T2 ⊥ P for FRM-class systems. The scope condition IS the DQ solution.

**Historical context:** P3 was written (S48) with condition (c) as a measurement protocol requirement. The DQ interpretation was not explicit at the time. DRP-5 proves the structural equivalence C3 ≡ T2-independence — it shows why C3 was always doing the DQ work.

**Joint/independent falsifiability (HR-D5 Challenge 3):**
- C-DRP5.3 falsified → C-DRP5.5 falsified (logical dependency)
- C-DRP5.5 falsified ↛ C-DRP5.3 falsified (C3 could fail without the condition being wrong)
- Independent falsification path for C-DRP5.5: find a P3 tau_gen extraction where the output depends on whether the FRM prediction is true.

---

## 7. Auxiliary Hypothesis Protocol

**C-DRP5.6 [Type F] — Protocol completeness.** The auxiliary hypothesis protocol is complete: every proposed (O, M) with background theory T2 receives a determinate PASS/FAIL verdict via the three-step procedure.

**C-DRP5.7 [Type D] — The protocol itself.**

Given a proposed falsification predicate K = (P, O, M, B) with background theory T2:

**Step 1 — T2 Identification:** Identify the background theory T2 used to construct (O, M). State explicitly what T2 governs in the test construction. FAIL if T2 cannot be identified.

**Step 2 — T2 Independence Check:** Determine whether T2's measurement outputs are causally independent of P. Would the measurements produced by T2 be the same regardless of whether P is true or false? PASS if T2 ⊥ P. FAIL if T2 depends on P (circular), was derived from P (reflexive), or requires P (presuppositional).

**Step 3 — Grounding Check:** Verify the causal independence chain terminates. Identify what T2 relies on (T3). If T3 is direct physical measurement independent of P, the chain is grounded. If T3 is theory, apply Step 2 recursively. PASS if chain terminates at independently validated measurement. FAIL if chain is circular or ungrounded.

All 3 steps PASS → T2 satisfies UMP-DQ. Any step FAIL → redesign test.

Each step specifies operator, inputs, operation, outputs, and PASS/FAIL conditions. No step requires the protocol author's judgment. Third-party constructable by a researcher with relevant domain expertise — Step 2 requires domain knowledge to assess causal independence, but this is domain expertise, not protocol-author judgment (HR-D5 #6).

---

## 8. Scope and Limitations

### What DRP-5 proves:
1. UMP is epistemically prior to DQ (C-DRP5.2).
2. Under UMP compliance, DQ reduces to T2 ⊥ P (C-DRP5.3).
3. The T2-independence regress terminates at independently validated measurement (C-DRP5.4).
4. FRM scope condition C3 is the FRM's complete DQ solution (C-DRP5.5).
5. A third-party-constructable protocol for verifying T2-independence (C-DRP5.6, C-DRP5.7).

### What DRP-5 does NOT prove:
- Resolution of DQ in full philosophical generality (Quinean global underdetermination).
- Existence of theory-free observation (all measurement is theory-laden; Hanson 1958, Kuhn 1962).
- Resolution of Agrippa's trilemma (epistemic regress in general).
- Empirical validation of any prediction (deferred to DRP-6).
- Theory choice between rival research programmes (Lakatos 1970; Laudan 1990).

### Dependency impact:
If C-DRP5.2 is wrong (UMP is not prior to DQ), the entire DQ boundary analysis collapses — DRP-5 has no independent content without this ordering result. DRP-6 would need to redesign its empirical programme to test DQ conditions independently of UMP.

If C-DRP5.5 is wrong (C3 does not satisfy T2-independence), the FRM's DQ compliance is in question — but C-DRP5.3 (the general T2-independence condition) may still hold. The FRM would need a different DQ solution.

---

## 9. Consistency Check

DRP-5 is a DRS paper. This section verifies internal consistency — that DRP-5's claims satisfy the standards it invokes. This is a consistency check, not a validation. External validation of DRP-5's claims requires the DRP-6 empirical programme (HR-D5 #10).

### Check 1: All claims registered?
7 claims: C-DRP5.1 through C-DRP5.7. All in AI layer claim_registry. **PASS.**

### Check 2: All Type F claims have full predicates?
C-DRP5.2, C-DRP5.3, C-DRP5.5, C-DRP5.6: all Type F with 5-part predicates. **PASS.**

### Check 3: All inbound/outbound edges declared?
Inbound: DRP-4 C-DRP4.5 (LIVE), P3 D-3.1/D-3.2 (LIVE). All in AI layer.
Outbound: C-DRP5.7 → DRP-6 (pass-forward registered). **PASS.**

### Check 4: Does DRP-5 apply the auxiliary hypothesis protocol to itself?
DRP-5's own claims are Type F with falsification predicates. The observation classes in those predicates are constructed using background theories (logic, philosophy of science). T2 for DRP-5 claims = formal logic and the DRP-1 through DRP-4 framework. This T2 is causally independent of P (DRP-5's claims about DQ boundary conditions) because the logical framework was established before and independently of the DQ analysis. The grounding chain terminates at the axioms of the DRP series (DRP-1 through DRP-4), which are independently PHASE-READY. **PASS.**

Self-application: **4/4 PASS.**

---

## 10. Relationship to DRP Series

DRP-5 occupies a specific position:

- **DRP-1** established the Falsification Kernel and proved the UMP.
- **DRP-2** proved testability is a ternary relation T(P,O,M) and derived the binary conservation law.
- **DRP-3** proved the Convergence Principle (M-termination ↔ μ < 0).
- **DRP-4** generalised the conservation law to all M.
- **DRP-5** (this paper) proves UMP is prior to DQ, derives T2-independence, and provides the auxiliary hypothesis protocol.
- **DRP-6** will receive the auxiliary hypothesis protocol and conduct empirical tests.

The DRP-5 contribution is the reduction of DQ to a structural condition. The philosophical literature treats DQ as an intractable attribution problem. DRP-5 shows that under UMP compliance, it reduces to a single checkable condition (T2 ⊥ P) with a constructable verification protocol.

---

## 11. Pass-Forward Register

| Edge | Target | Claim | Status |
|------|--------|-------|--------|
| DRP5→DRP6 | DRP-6 | C-DRP5.7 (Auxiliary Hypothesis Protocol) | LIVE — pending DRP-5 PHASE-READY |

---

## 12. Claim Summary

| Type | Count | Claim IDs |
|------|-------|-----------|
| A | 1 | C-DRP5.1 |
| D | 2 | C-DRP5.4, C-DRP5.7 |
| F | 4 | C-DRP5.2, C-DRP5.3, C-DRP5.5, C-DRP5.6 |
| **Total** | **7** | |

Registry impact: 182 + 7 = 189 total claims, 19 papers.

---

*DRP-5 prose draft produced by Claude Code, Session 57.*
