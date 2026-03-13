# DRP-4 Complete Build Record — Phases 0–3

**Session:** S56
**Date:** 2026-03-13
**Source:** Thomas Brennan (architect) + Claude (Anthropic) — build instance
**Paper type:** Type B (Derivation)
**Working title:** The Full Conservation Law: Epistemic Information Vanishes for All Measurement Functions

---

## D4-0.0 — Phase 0: Scope & IR Audit

### Paper Identity

| Field | Value |
|-------|-------|
| Paper ID | DRP-4 |
| Working title | The Full Conservation Law: Epistemic Information Vanishes for All Measurement Functions |
| Paper type | Type B (Derivation) |
| Series | DRP Series — Paper 4 of 8 |
| Protocol | DRP — Dual Reader Protocol |
| Gate cleared | DRP-2 PHASE-READY (S49), DRP-3 PHASE-READY (S56) |

### Inbound Live Edges

**EDGE 4 — DRP-2 C-DRP2.2 (Binary epistemic conservation law)**
- Source: DRP-2 v1.2, C-DRP2.2 (Type F)
- Content: I(M(O); P) = 0 for binary M when O = f(P)
- Scope note: "Binary-scoped in DRP-2; DRP-4 generalises to all M"
- AI layer: DRP-2 AI Layer v1.2 (S54)
- Status: LIVE

**EDGE 6 — DRP-3 C-DRP3.5 (Convergence Principle)**
- Source: DRP-3 v1, C-DRP3.5 (Type F)
- Content: M-termination ↔ μ<0. Both are instances of the same abstract fixed-point convergence requirement.
- AI layer: DRP-3 AI Layer v1 (S56)
- Status: LIVE (promoted from PLACEHOLDER at DRP-3 PHASE-READY, S56)

### Also Received as Structural Vocabulary (not Type A edge)

**C-DRP2.5 — Formal theorem statement**
- "For any K=(P,O,M,B) where O=f(P) and M is binary: I(M(O);P) = 0."
- This is the formal statement of the theorem DRP-4 generalises.
- Not a separate inbound edge — received via C-DRP2.2 which is its falsifiable form.

### Outbound Expected Edge

- C-DRP4.4 → DRP-5: Full conservation law (I=0 for all M)
- DRP-5 receives as Type A and proves universality + auxiliary hypothesis protocol

### IR Inventory

| IR | Name | Form | Source |
|----|------|------|--------|
| IR-DRP4-1 | Received Axiom | A claim proved in a prior paper with LIVE status may be received as Type A without re-derivation. | Standard DRP rule |
| IR-DRP4-2 | Structural Generalisation | If a theorem proved for a restricted case X depends only on structural property S, and S holds for the general case Y ⊃ X, then the theorem holds for Y. | Mathematical induction principle |
| IR-DRP4-3 | Composition Preservation | If f: A→B and g: B→C are both determined by the same upstream variable, then g∘f is determined by that variable. Determinism is preserved under composition. | Function composition axiom |
| IR-DRP4-4 | Exhaustive Classification | If a domain D is partitioned into classes C1...Cn and a property P holds for each Ci, then P holds for all of D. | Proof by exhaustive cases |
| IR-DRP4-5 | Vacuity Witness | A falsification predicate is non-vacuous (C6) iff a coherent observation exists that would trigger FALSIFIED. | Standard DRP rule |

All five rules provisionally accepted. To be confirmed at Phase 1 close.

### Scope Boundary

**IN:** All measurement functions M satisfying condition (b) of T(P,O,M) — i.e., all finite deterministic functions from observation classes to verdict spaces. Includes binary, n-ary, continuous, and probabilistic M.

**OUT:**
- Non-terminating M (these fail condition (b) and are not well-formed predicates)
- Stochastic M that are not finite deterministic (these fail condition (b))
- The UMP itself (proved in DRP-1; DRP-4 uses the conservation consequence, not the independence condition)

---

## D4-1.0 — Phase 1: Claim Identification

### Claim Inventory: 7 claims — 2A · 2D · 3F

### Type A — Received Axioms (2)

**C-DRP4.1 [A] Binary conservation law — received from DRP-2**

- Source: DRP-2, C-DRP2.2 (and formal statement C-DRP2.5)
- Statement: For any falsification predicate K = (P, O, M, B) where O = f(P) and M is binary (M: O → {FALSIFIED, NOT FALSIFIED}): I(M(O); P) = 0. The mutual information between the test outcome and the proposition is zero when the observation class is derived from the proposition.
- Role in DRP-4: The binary case is the starting point. DRP-4 identifies the structural mechanism that makes I=0 hold and shows it is independent of M's output type.
- IR used: IR-DRP4-1 (Received Axiom)

**C-DRP4.2 [A] Convergence Principle — received from DRP-3**

- Source: DRP-3, C-DRP3.5
- Statement: M-termination is epistemically equivalent to μ<0. Both are instances of the same abstract fixed-point convergence requirement. All terminating M functions share the same convergence structure regardless of output type.
- Role in DRP-4: Establishes that the class of well-formed M functions (those satisfying condition (b)) share a common structure. The generalisation from binary to all M is structural, not case-by-case.
- IR used: IR-DRP4-1 (Received Axiom)

### Type D — Definitions (2)

**C-DRP4.3 [D] M-class taxonomy**

- Statement: The class of all measurement functions M satisfying condition (b) of T(P,O,M) (finite deterministic) is partitioned into four exhaustive sub-classes by output type: (i) binary M: O → {FALSIFIED, NOT FALSIFIED}; (ii) n-ary M: O → {v1, ..., vn} for finite n > 2; (iii) continuous M: O → ℝ (or a measurable subset); (iv) probabilistic M: O → [0,1] (interpreted as probability or confidence). All four sub-classes satisfy condition (b) when M terminates. No well-formed M exists outside this partition.
- Role: Framework for the exhaustive generalisation proof (C-DRP4.5).

**C-DRP4.4 [D] Generalised epistemic information**

- Statement: For any M satisfying condition (b) of T(P,O,M), the epistemic information content I(M(O); P) is defined as: the reduction in uncertainty about P achieved by observing M(O), measured as the difference between prior uncertainty H(P) and posterior uncertainty H(P|M(O)). I(M(O); P) = H(P) − H(P|M(O)). This definition applies uniformly across all M-classes (C-DRP4.3) and reduces to Shannon mutual information when M is binary.
- Note: This is a definitional extension of the information measure used in DRP-2 §3 to the general case. The definition is well-formed for all terminating M because M(O) produces a definite output (condition (b)).

### Type F — Falsifiable Claims (3)

**C-DRP4.5 [F] The full conservation law — CENTRAL CLAIM**

- Statement: For any falsification predicate K = (P, O, M, B) where O = f(P) and M satisfies condition (b) of T(P,O,M): I(M(O); P) = 0. The epistemic information content of the test is zero regardless of M's output type. This generalises C-DRP4.1 (binary case) to all terminating M.
- Derivation: The I=0 mechanism in DRP-2 §3 depends on O = f(P) making M(O) = M(f(P)) = h(P) — a deterministic function of P alone. This composition (IR-DRP4-3) holds for any M, not just binary M. The prior on P is unchanged by observing h(P) because h(P) is determined by P's truth value — no independent evidence is introduced. The convergence principle (C-DRP4.2) ensures all terminating M share the same convergence structure, so the generalisation is structural.
- Outbound: DRP-5 receives as Type A (full conservation law).
- Vacuity witness: A non-binary M with O=f(P) that produces I > 0.

**C-DRP4.6 [F] Mechanism independence**

- Statement: The I=0 result in DRP-2 §3 depends on exactly one structural property: O = f(P). It does not invoke any property specific to binary M. The cardinality, topology, or measure-theoretic structure of M's output space plays no role in the proof that I = 0.
- Derivation: Citation audit of DRP-2 §3 proof. The proof proceeds: O = f(P) → M(O) = M(f(P)) → outcome determined by P → prior unchanged → I = 0. No step references |range(M)| = 2 or any binary-specific property.
- Vacuity witness: A step in DRP-2 §3 that uses |range(M)| = 2 as a necessary premise.

**C-DRP4.7 [F] Exhaustive coverage**

- Statement: The M-class taxonomy (C-DRP4.3) is exhaustive: every finite deterministic function from an observation class to a verdict space falls into exactly one of the four sub-classes. The full conservation law (C-DRP4.5) holds for each sub-class individually and therefore for all M.
- Derivation: By IR-DRP4-4 (exhaustive classification). The four classes are defined by output space topology: discrete-finite-2 (binary), discrete-finite-n (n-ary), continuous (ℝ), bounded-continuous ([0,1]). Any finite deterministic function has an output space falling into one of these categories.
- Vacuity witness: A finite deterministic M whose output space is not captured by any of the four sub-classes.

---

## D4-2.0 — Phase 2: Derivation Table

### Step 0 — Received Inputs

**S0-A: C-DRP4.1 [A] — Binary conservation law**
- Source: DRP-2 C-DRP2.2. LIVE.
- Content: I(M(O); P) = 0 for binary M when O = f(P).
- P10 anchor: LIVE inbound edge. Terminates at DRP-2 AI layer.

**S0-B: C-DRP4.2 [A] — Convergence Principle**
- Source: DRP-3 C-DRP3.5. LIVE.
- Content: All terminating M share the same convergence structure. M-termination ↔ μ<0.
- P10 anchor: LIVE inbound edge. Terminates at DRP-3 AI layer.

### Step 1 — M-Class Taxonomy (produces C-DRP4.3)

- **Input:** S0-B (C-DRP4.2 — all terminating M share common structure)
- **Rule:** IR-DRP4-4 (Exhaustive Classification — framework)

**Derivation:**
Condition (b) of T(P,O,M) requires M to be a finite deterministic function. The output space of any such function falls into one of four categories by its topological structure:

1. **Binary:** |range(M)| = 2. Output: {FALSIFIED, NOT FALSIFIED}. This is the DRS standard case.
2. **N-ary:** |range(M)| = n for finite n > 2. Output: {v1, ..., vn}. Example: a test with FALSIFIED / NOT FALSIFIED / INDETERMINATE / BOUNDARY.
3. **Continuous:** range(M) ⊆ ℝ. Output: a real number (e.g., a test statistic, p-value, or effect size).
4. **Probabilistic:** range(M) ⊆ [0,1]. Output: a probability or confidence value.

These four exhaust all possible finite deterministic output spaces: any discrete finite space has cardinality n (cases 1-2); any continuous space embeds in ℝ (case 3) or is bounded in [0,1] (case 4).

**Output:** C-DRP4.3 [D] — M-class taxonomy.
**P10 check:** Four classes derived from topological classification of function output spaces. No new empirical constants.

### Step 2 — Generalised Epistemic Information (produces C-DRP4.4)

- **Input:** S0-A (C-DRP4.1), C-DRP4.3
- **Rule:** IR-DRP4-1 (definition extension)

**Derivation:**
DRP-2 §3 uses Shannon mutual information for binary M: I(M(O); P) = H(P) − H(P|M(O)), where the entropy is computed over the binary output space. For the general case:

I(M(O); P) = H(P) − H(P|M(O))

This definition is well-formed for all four M-classes:
- Binary/n-ary: discrete entropy (standard Shannon).
- Continuous: differential entropy (standard extension).
- Probabilistic: differential entropy over [0,1].

In all cases, I measures the reduction in uncertainty about P achieved by observing M(O). When I = 0, observing M(O) provides no information about P.

**Output:** C-DRP4.4 [D] — Generalised epistemic information.
**P10 check:** Definition extends DRP-2 §3 measure using standard information theory. No stipulated constants.

### Step 3 — Mechanism Independence (produces C-DRP4.6)

- **Input:** S0-A (C-DRP4.1 — binary proof)
- **Rule:** IR-DRP4-2 (Structural Generalisation), IR-DRP4-3 (Composition Preservation)

**Derivation:**
The DRP-2 §3 proof of I(M(O); P) = 0 for binary M proceeds:

1. Assume O = f(P).
2. Then M(O) = M(f(P)).
3. Define h = M ∘ f. Then M(O) = h(P).
4. h(P) is a deterministic function of P alone.
5. Observing h(P) does not update the prior on P because the outcome is determined by P's truth value — no independent evidence is introduced.
6. Therefore I(M(O); P) = H(P) − H(P|h(P)) = 0.

**Critical observation:** No step in this chain invokes any property of M's output space. Steps 1-3 use function composition (IR-DRP4-3), which is valid for any function regardless of output type. Step 4 is a consequence of composition. Step 5 is the epistemic argument — the prior on P is unchanged because h(P) is determined by P. Step 6 applies the information measure.

The proof mechanism is: O = f(P) creates a deterministic chain P → O → M(O), making the test outcome a function of P alone. This chain holds whether M outputs {0,1}, {0,...,n}, ℝ, or [0,1].

**Output:** C-DRP4.6 [F] — Mechanism independence. The I=0 proof does not depend on binary M.
**P10 check:** Analysis of DRP-2 §3 proof structure. No new constants.

### Step 4 — Full Conservation Law (produces C-DRP4.5)

- **Input:** C-DRP4.3 (taxonomy), C-DRP4.4 (generalised I), C-DRP4.6 (mechanism independence), S0-A, S0-B
- **Rule:** IR-DRP4-2 (Structural Generalisation), IR-DRP4-4 (Exhaustive Classification)

**Derivation:**

**Premise 1:** I(M(O); P) = 0 for binary M when O = f(P) (C-DRP4.1, received).
**Premise 2:** The proof mechanism depends only on O = f(P), not on M's output type (C-DRP4.6, Step 3).
**Premise 3:** All terminating M share the same convergence structure (C-DRP4.2, received).
**Premise 4:** The M-class taxonomy (C-DRP4.3) exhausts all terminating M.

By IR-DRP4-2: The theorem I(M(O); P) = 0 was proved for the binary case (Premise 1). The proof depends only on the structural property O = f(P) (Premise 2). This property holds for any M regardless of output type — it concerns the relationship between O and P, not the structure of M. By the convergence principle (Premise 3), all terminating M share the same abstract structure. By exhaustive classification (Premise 4), the taxonomy covers all cases.

Therefore: for ANY K = (P, O, M, B) where O = f(P) and M satisfies condition (b): I(M(O); P) = 0.

**Output:** C-DRP4.5 [F] — Full conservation law.
**P10 check:** All premises trace to live edges (C-DRP4.1 from DRP-2, C-DRP4.2 from DRP-3) or IR axioms. No new constants.

### Step 5 — Exhaustive Coverage (produces C-DRP4.7)

- **Input:** C-DRP4.3 (taxonomy), C-DRP4.5 (full law)
- **Rule:** IR-DRP4-4 (Exhaustive Classification)

**Derivation:**
By C-DRP4.3, every finite deterministic M falls into one of four sub-classes. The full conservation law (C-DRP4.5) holds for each:

- Binary: proved in DRP-2 (C-DRP4.1).
- N-ary: binary is a special case of n-ary with n=2. The I=0 mechanism (Step 3) applies identically: h(P) = M(f(P)) is deterministic for any finite output space.
- Continuous: h(P) = M(f(P)) ∈ ℝ is still a deterministic function of P. I(h(P); P) = 0 because h(P) is determined by P.
- Probabilistic: same argument. h(P) ∈ [0,1] is determined by P.

No case produces I > 0 when O = f(P). The taxonomy is exhaustive. The conservation law holds for all M.

**Output:** C-DRP4.7 [F] — Exhaustive coverage.
**P10 check:** Uses C-DRP4.3 taxonomy and C-DRP4.5 law. No new constants.

### Derivation Table — Summary

| Step | Input(s) | Rule | Output |
|------|----------|------|--------|
| S0 | DRP-2 C-DRP2.2, DRP-3 C-DRP3.5 | IR-DRP4-1 | C-DRP4.1 [A], C-DRP4.2 [A] |
| 1 | C-DRP4.2 | IR-DRP4-4 | C-DRP4.3 [D] — M-class taxonomy |
| 2 | C-DRP4.1, C-DRP4.3 | IR-DRP4-1 | C-DRP4.4 [D] — Generalised epistemic information |
| 3 | C-DRP4.1 | IR-DRP4-2, IR-DRP4-3 | C-DRP4.6 [F] — Mechanism independence |
| 4 | C-DRP4.3, C-DRP4.4, C-DRP4.6, S0-A, S0-B | IR-DRP4-2, IR-DRP4-4 | C-DRP4.5 [F] — Full conservation law |
| 5 | C-DRP4.3, C-DRP4.5 | IR-DRP4-4 | C-DRP4.7 [F] — Exhaustive coverage |

All 7 claims produced. All trace to live edges S0-A and S0-B. No orphaned constants.
n_invalid_steps = 0.

---

## D4-3.0 — Phase 3: Falsification Predicates

### C-DRP4.5 [F] — The Full Conservation Law

**FALSIFIED IF** `n_counterexamples > 0`

**WHERE**
`n_counterexamples`: integer, dimensionless. Count of falsification predicates K = (P, O, M, B) simultaneously satisfying:
(a) O = f(P) — observation class derived from the proposition;
(b) M satisfies condition (b) of T(P,O,M) — M is finite deterministic and terminates;
(c) I(M(O); P) > 0 — the test outcome provides non-zero epistemic information about P, measured as reduction in prior uncertainty.

A single instance of (a)+(b)+(c) suffices to falsify.

**EVALUATION**
1. Construct candidate predicate K with O = f(P) and non-binary M (binary case already covered by C-DRP4.1).
2. Verify M satisfies condition (b): finite, deterministic, terminates.
3. Compute h = M ∘ f. Verify h(P) is a deterministic function of P.
4. Compute I(h(P); P) using the generalised measure (C-DRP4.4).
5. If I > 0: n_counterexamples += 1.
Output: n_counterexamples. Finite procedure.

**BOUNDARY**
n_counterexamples = 0 → NOT FALSIFIED.
n_counterexamples = 1 → FALSIFIED (universal claim; one counterexample suffices).

**CONTEXT**
C-DRP4.5 · Type F · Full conservation law. Generalises DRP-2 C-DRP2.2 from binary to all M. Tests whether a non-binary M with O=f(P) can produce I > 0. Threshold: 0.

**UMP CHECK:** O (candidate predicates with measured I) upstream of P (conservation law). M (computation of I) finite deterministic. M can return FALSIFIED. UMP SATISFIED. C6 SATISFIED.

**VACUITY WITNESS:** A continuous M computing a test statistic from O=f(P) that yields I > 0 — i.e., the test statistic provides information about P despite O being derived from P. Believed non-existent because h(P) = M(f(P)) is determined by P, so H(P|h(P)) = H(P) and I = 0.

---

### C-DRP4.6 [F] — Mechanism Independence

**FALSIFIED IF** `n_binary_steps > 0`

**WHERE**
`n_binary_steps`: integer, dimensionless. Count of steps in the DRP-2 §3 proof of I(M(O); P) = 0 that invoke a property specific to binary M — i.e., steps whose justification requires |range(M)| = 2 or the structure {FALSIFIED, NOT FALSIFIED} as a necessary premise, and which cannot be re-derived for general M.

**EVALUATION**
1. Obtain DRP-2 v1.2 §3 proof text.
2. For each step: identify the justification.
3. Determine whether the justification requires |range(M)| = 2 or any binary-specific property.
4. If a step requires binary M and cannot be re-derived without it: n_binary_steps += 1.
Output: n_binary_steps. Finite procedure (proof is a finite text).

**BOUNDARY**
n_binary_steps = 0 → NOT FALSIFIED.
n_binary_steps = 1 → FALSIFIED (single binary-dependent step would block generalisation).

**CONTEXT**
C-DRP4.6 · Type F · Mechanism independence. Tests whether DRP-2 §3 proof depends on binary M. Threshold: 0.

**UMP CHECK:** O (DRP-2 §3 text) is a published document fixed independently of P (mechanism independence claim). M (step-by-step audit) finite deterministic. C6 SATISFIED.

**VACUITY WITNESS:** A step in DRP-2 §3 reading "Since M has exactly two outputs, the probability table has exactly two rows, allowing us to compute I directly." If such a step existed, the proof would require binary M as a necessary premise.

---

### C-DRP4.7 [F] — Exhaustive Coverage

**FALSIFIED IF** `n_uncovered > 0`

**WHERE**
`n_uncovered`: integer, dimensionless. Count of finite deterministic measurement functions M satisfying condition (b) of T(P,O,M) that do not fall into any of the four sub-classes defined in C-DRP4.3 (binary, n-ary, continuous, probabilistic).

**EVALUATION**
1. Enumerate known M functions used in published falsification predicates across the corpus.
2. For each: classify into C-DRP4.3 sub-class.
3. Attempt to construct an M satisfying condition (b) that does not fit any sub-class.
4. If such an M is found: n_uncovered += 1.
Output: n_uncovered. Finite procedure.

**BOUNDARY**
n_uncovered = 0 → NOT FALSIFIED.
n_uncovered = 1 → FALSIFIED.

**CONTEXT**
C-DRP4.7 · Type F · Exhaustive coverage. Tests whether the M-class taxonomy is complete. Threshold: 0. A single uncovered M would require extending the taxonomy and re-proving the conservation law for the new class.

**UMP CHECK:** O (M functions and their output spaces) determined by examining predicate structure, upstream of P (taxonomy completeness claim). M (classification procedure) finite deterministic. C6 SATISFIED.

**VACUITY WITNESS:** An M function whose output space is a non-Hausdorff topological space not embeddable in ℝ or [0,1] and not a finite discrete set. Such an M would satisfy condition (b) (finite deterministic) but fall outside the taxonomy.

---

## D4-3.0 Summary

| Claim | Predicate | UMP | C6 | Vacuity | Status |
|-------|-----------|-----|----|---------|--------|
| C-DRP4.5 | n_counterexamples > 0 | PASS | PASS | PRESENT | COMPLETE |
| C-DRP4.6 | n_binary_steps > 0 | PASS | PASS | PRESENT | COMPLETE |
| C-DRP4.7 | n_uncovered > 0 | PASS | PASS | PRESENT | COMPLETE |

All three Type F predicates complete. D4-3.0 COMPLETE.

---

## IR Inventory Confirmation

All five rules exercised:

| IR | Name | Used by |
|----|------|---------|
| IR-DRP4-1 | Received Axiom | C-DRP4.1, C-DRP4.2, Step 2 |
| IR-DRP4-2 | Structural Generalisation | C-DRP4.5, C-DRP4.6 |
| IR-DRP4-3 | Composition Preservation | C-DRP4.6, Step 3 |
| IR-DRP4-4 | Exhaustive Classification | C-DRP4.3, C-DRP4.5, C-DRP4.7 |
| IR-DRP4-5 | Vacuity Witness | C-DRP4.5, C-DRP4.6, C-DRP4.7 |

No additional rules required.

---

*DRP-4 Phases 0–3 build record archived by Claude, Session S56.*
