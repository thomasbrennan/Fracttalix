# DRP-3 D3-3.0 — Falsification Predicates

**Session:** S54
**Date:** 2026-03-12
**Source:** Thomas Brennan (claim design) + Claude Code (predicate formalisation)
**Pass:** A (C-DRP3.5, C-DRP3.6). Pass B (C-DRP3.7) pending DRS_v1_1_S49.docx upload.

---

## Preamble

This document produces the full falsification predicates for the three Type F claims
in DRP-3. Each predicate follows the DRS Falsification Kernel 4-tuple (P, O, M, B)
structure from C-DRP.5 and must satisfy C6 (non-vacuity) and the UMP (C-DRP.8:
O causally independent of P).

Pass A covers C-DRP3.5 and C-DRP3.6. Pass B covers C-DRP3.7 (blocked on §10.7 read).

---

## C-DRP3.5 [F] — The M↔C2 Isomorphism

**Statement:** M-termination (the requirement that the measurement function of a
falsification predicate reach a definite output in finite steps) is epistemically
equivalent to μ < 0 (the Hopf stability condition for convergent oscillation in
FRM-class physical systems). Both are instances of the same abstract convergence
requirement, derivable from the same Hopf stability analysis applied at two levels
of description. They belong to the same convergence equivalence class (C-DRP3.4).

**Derivation source:** D3-2.0 Steps 3 and 4.

### Falsification Predicate

```json
{
  "claim_id": "C-DRP3.5",
  "type": "F",
  "label": "M↔C2 Isomorphism — central claim",
  "falsification_predicate": {
    "condition": "n_counterexamples > 0",
    "inputs": [
      {
        "name": "n_counterexamples",
        "type": "integer",
        "units": "dimensionless",
        "source": "count of systems simultaneously satisfying EITHER: (a) system is in the FRM universality class with μ < 0 (physically convergent) AND its associated falsification predicate has a measurement function M that does not terminate (epistemologically divergent); OR (b) a falsification predicate with a terminating M whose associated physical system has μ > 0 (physically divergent). Each candidate must be a system where both the physical and epistemological descriptions are well-defined — i.e., the system admits an FRM parametrisation AND the proposition under test has a stated falsification predicate with explicit M."
      }
    ],
    "evaluation": "for each candidate system: (1) confirm FRM universality class membership per P1 AI layer criteria; (2) extract μ from the Hopf stability analysis of the linearised system; (3) identify the falsification predicate K = (P, O, M, B) for the proposition under test; (4) determine whether M terminates per C-DRP3.3 (finite deterministic, no cycling, no divergence, no interpretation-dependence); (5) check for mismatch: μ < 0 with non-terminating M, or μ > 0 with terminating M; if mismatch found, count as counterexample. Finite evaluation for any finite candidate set.",
    "boundary": "n = 0 → NOT FALSIFIED",
    "context": "C-DRP3.5 · threshold 0 · isomorphism claim: a single system where the convergence status at the physical level (μ sign) disagrees with the convergence status at the epistemological level (M-termination) falsifies the isomorphism. The claim is that these two convergence conditions are structurally locked — they cannot disagree for any system in the FRM universality class.",
    "vacuity_witness": "An FRM-class system (μ < 0, physically stable) whose associated falsification predicate has M that cycles indefinitely on some input o ∈ O — e.g., a natural language predicate requiring unbounded interpretation steps applied to a physically convergent oscillator. This would demonstrate that physical convergence does not guarantee epistemological convergence, breaking the isomorphism."
  }
}
```

### UMP Compliance Check

- **P:** The M↔C2 isomorphism holds for all FRM-class systems.
- **O:** Systems in the FRM universality class examined for convergence mismatch between physical (μ sign) and epistemological (M-termination) levels.
- **O ⊥ P:** The observation class (FRM-class systems and their predicates) exists independently of the isomorphism claim. The systems and their predicates are defined by their own physics and epistemology, not by C-DRP3.5. Examining them does not presuppose the isomorphism.
- **M terminates:** For any finite candidate set, checking μ sign and M-termination status are both finite deterministic operations. M reaches a definite output.
- **C6 (non-vacuity):** The vacuity witness is constructive — a natural language predicate applied to a convergent oscillator is a concrete example of a system where the mismatch could occur. The predicate can return FALSIFIED.

### Joint vs Independent Falsifiability Note

C-DRP3.5 and C-DRP3.6 share the convergence equivalence class framework (C-DRP3.4)
but are independently falsifiable:

- C-DRP3.5 can be falsified while C-DRP3.6 holds: the convergent case fails
  (mismatch at μ < 0) but the divergent case equivalence is still valid.
- C-DRP3.6 can be falsified while C-DRP3.5 holds: the divergent case fails
  (mismatch at μ > 0) but the convergent case equivalence is still valid.

However, if BOTH are falsified, the entire isomorphism collapses (no case holds).
The exhaustive partition (IR-DRP3-3) means C-DRP3.5 in its complete form requires
both cases — but the convergent half of C-DRP3.5 is independently testable from
C-DRP3.6.

---

## C-DRP3.6 [F] — Failure Mode Equivalence

**Statement:** Non-terminating M (M fails condition (b) of T(P,O,M)) is the exact
epistemological analogue of μ > 0 (unstable limit cycle in a physical FRM-class
system). In both cases: the system has no definite output or stable state; the
failure is structural, not contingent on measurement error; no amount of additional
evaluation steps or observations resolves the failure. The failure modes are
structurally isomorphic under C-DRP3.4.

**Derivation source:** D3-2.0 Step 4.

### Falsification Predicate

```json
{
  "claim_id": "C-DRP3.6",
  "type": "F",
  "label": "Failure mode equivalence",
  "falsification_predicate": {
    "condition": "n_structural_mismatches > 0",
    "inputs": [
      {
        "name": "n_structural_mismatches",
        "type": "integer",
        "units": "dimensionless",
        "source": "count of systems where the failure mode structure disagrees between physical and epistemological levels. A structural mismatch is defined as EITHER: (a) a system with μ > 0 (physically divergent) whose associated non-terminating M fails for a CONTINGENT reason — i.e., the non-termination is resolvable by additional evaluation steps, measurement refinement, or error correction, rather than being a structural consequence of the predicate design; OR (b) a system with non-terminating M (epistemologically divergent) whose physical system with μ > 0 reaches a definite state through some mechanism not captured by the Hopf analysis — i.e., the physical divergence is resolvable while the epistemological divergence is not, or vice versa."
      }
    ],
    "evaluation": "for each candidate system with μ > 0 and non-terminating M: (1) classify the non-termination failure mode per C-DRP3.3 (cycling, divergence, or interpretation-dependence); (2) determine whether the failure is structural (inherent in predicate design, not resolvable by more steps) or contingent (resolvable by refinement); (3) determine whether the physical divergence (μ > 0) is structural (inherent in system dynamics) or contingent; (4) if structural status disagrees between levels — one is structural and the other contingent — count as mismatch. Finite evaluation for any finite candidate set.",
    "boundary": "n = 0 → NOT FALSIFIED",
    "context": "C-DRP3.6 · threshold 0 · failure mode equivalence: the claim is not merely that both levels fail simultaneously, but that they fail FOR THE SAME STRUCTURAL REASON — the system does not converge to a fixed point. A single system where one level's failure is structural and the other's is contingent falsifies the equivalence. The structural/contingent distinction is the key test: structural failure = inherent in the system's mathematical description; contingent failure = resolvable by changing parameters, adding steps, or refining measurement.",
    "vacuity_witness": "A physical system with μ > 0 (genuinely divergent trajectory, no stable limit cycle) whose associated falsification predicate has a non-terminating M that can be made to terminate by adding a finite number of additional evaluation steps — e.g., a predicate where interpretation-dependence at step k is resolved by specifying a decision rule at step k+1, yielding a definite output. The physical failure is structural but the epistemological failure is contingent. This would falsify the claim that the failure modes are structurally isomorphic."
  }
}
```

### UMP Compliance Check

- **P:** Non-terminating M and μ > 0 are structurally isomorphic failure modes.
- **O:** Systems with μ > 0 and non-terminating M, examined for structural vs contingent failure classification.
- **O ⊥ P:** The observation class (divergent systems) exists independently of the failure mode equivalence claim. The structural/contingent classification of each failure is determined by the system's own properties, not by C-DRP3.6.
- **M terminates:** Classifying a failure as structural or contingent is a finite analysis of the predicate/system description. Definite output: {structural, contingent}.
- **C6 (non-vacuity):** The vacuity witness is constructive — a resolvable interpretation-dependence failure paired with a genuine physical divergence is a concrete falsification scenario.

---

## C-DRP3.7 [F] — Level Independence (Pass B — PENDING)

**Status:** Blocked on DRS_v1_1_S49.docx upload.

**Predicate structure confirmed at OQ-1 resolution:**
- Observation class: text of DRP-1 §10.7
- FALSIFIED IF: DRP-1 §10.7 cites P2 or any Hopf analysis in deriving the finite-deterministic requirement for M
- Scope: published formal derivation chain only (no private thought processes)
- Arm 2 (P2 citation audit): CONFIRMED — P2 does not cite DRP-1 or DRS

**Predicate to be finalised at Pass B when §10.7 text is available.**

---

## Phase Gate Status

| Phase | Status |
|-------|--------|
| D3-0.0 IR Inventory Audit | COMPLETE (S53) |
| D3-1.0 Claim Identification | COMPLETE (S54) |
| D3-2.0 Derivation Table | COMPLETE (S54) |
| D3-3.0 Falsification Predicates | **PASS A COMPLETE** (C-DRP3.5, C-DRP3.6) |
| D3-3.0 Pass B | PENDING — C-DRP3.7, requires DRS_v1_1_S49.docx §10.7 |
| D3-4.0 Human Reader Pass | Follows D3-3.0 completion |
| D3-5.0 CBT Execution | Gate to PHASE-READY |

---

*D3-3.0 Pass A produced by Claude Code, Session S54.*
