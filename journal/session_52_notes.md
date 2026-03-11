# Session 52 — CorpusArch v10 Reconciliation and UMP–FRM Structural Identity

**Date:** 2026-03-11
**Type:** Build table reconciliation, theoretical discovery.

---

## What Happened

### Build Table Reconciliation (CorpusArch v10)
- P2 status corrected: Phase 1 PHASE-READY (S49), not full PHASE-READY — v9 was speculative
- P3 status corrected: QUEUED, no build plan — v9 was speculative
- P1 bioRxiv REJECTED (S49, scope mismatch) — arXiv cs.DL route active
- Protocol Amendment Log added (S48-A1, S48-A2, S49-A3)
- CorpusArch header updated v9 → v10

### UMP Recovery and Formal Statement

The **Upstream Measurement Principle (UMP)** was recovered from prior session work (named by the claude.ai instance). Formal statement:

> The independence of the observation class O from the proposition P under test is a necessary condition for predicate non-vacuity. Proved by contradiction in DRP-1 v1.1.

Formalized using Shannon mutual information: I(P; O) = 0 when O is derived from (downstream of) P. A test whose observation class is derived from the proposition it tests has exactly zero information content about that proposition.

DRP-2 derives epistemological consequences:
- Replication crisis has a structural cause (UMP violation, not cultural failure)
- Testability is a ternary relation T(P, O, M), not a property of claims in isolation
- Popper's falsifiability criterion is a **correction**, not a refinement

### UMP–FRM Structural Identity (New Discovery — S52)

**Status:** CONJECTURE. Requires formal proof.

**Claim:** UMP is not an independent principle compatible with FRM — it is derivable from the same network conditions that produce β = 1/2. The delay τ in a delayed feedback network simultaneously creates:

1. The quarter-wave resonance (π/2 phase shift → β = 1/2)
2. A temporal window of duration τ where the signal exists before feedback has acted on it

This temporal window **is** the upstream observation point required by UMP. Therefore:

**Conjecture UMP-FRM (S52):** For any system S satisfying the FRM delayed-feedback conditions (D-2.1 criteria a–c), there exists an observation point upstream of the feedback coupling, and this observation point is structurally guaranteed by the delay τ that produces the Hopf bifurcation. Conversely, any system where no upstream observation point exists has τ → 0 and therefore falls outside the FRM universality class.

**Falsification predicate:**

```
FALSIFIED IF:  n_counterexample > 0
WHERE:
  n_counterexample = count of systems S such that:
    (a) S satisfies D-2.1 criteria (a)–(c) [delayed feedback, Hopf-capable, τ independently measurable], AND
    (b) no observation point exists where O is causally independent of the current feedback cycle
      [i.e., I(P; O) > 0 for all accessible observation points, where P is any proposition about x(t)
       and O is the observation class at that point], OR
  count of systems S such that:
    (c) S has τ = 0 [no delay, outside D-2.1], AND
    (d) a UMP-compliant observation point exists [O causally independent of P]
EVALUATION:
  For each candidate system: (1) verify D-2.1 criteria against system description;
  (2) enumerate observation points in the feedback architecture;
  (3) for each point, determine causal independence of O from current-cycle P
  using the signal flow graph; (4) if D-2.1 satisfied and no independent point found,
  OR if D-2.1 not satisfied and independent point found, increment n_counterexample.
  Procedure terminates in finite steps bounded by node count of the feedback graph.
BOUNDARY:  n_counterexample = 0 → NOT FALSIFIED
CONTEXT:  Conjecture UMP-FRM · threshold 0 · structural identity claim:
  a single counterexample in either direction falsifies the biconditional.
```

**Consequences if proved:**

1. **D-3.2 is a theorem, not a methodological choice.** The requirement that τ_gen be independently measurable follows from the network topology. The system guarantees the existence of the upstream point.

2. **UMP violation ↔ FRM inapplicability.** These are the same condition: no delay means no quarter-wave resonance AND no upstream measurement window. A system where UMP cannot be satisfied is a system where β = 1/2 does not hold — not because the law fails, but because the system is outside its scope.

3. **Measurability is structural, not contingent.** The law and the ability to test the law are produced by the same mechanism. This is not a happy coincidence — it is a necessary feature of delayed feedback networks.

4. **The zero-free-parameter constraint (P3) is a corollary of UMP-FRM.** If the upstream point exists and τ_gen is extractable from system architecture, then all model parameters are determined before the data is seen. Free parameters would mean the measurement point has moved downstream.

5. **Self-consistency of the corpus.** The corpus itself is a delayed feedback network (papers → review → revision → papers). By UMP-FRM, the corpus must satisfy UMP (which it does via DRS) and exhibit β = 1/2 (which P9 is designed to test). The measurement protocol for the corpus (DRS + Sentinel) is structurally guaranteed to have an upstream observation point — the AI layer (Channel 1) exists before the prose rendering (Channel 2).

**Proof sketch (informal):**

Let S be a delayed feedback network with characteristic delay τ > 0. The signal x(t) at any node satisfies x(t) = F(x(t), x(t − τ)), where F encodes the feedback coupling. At time t, the value x(t − τ) was determined at time (t − τ) without knowledge of x(t). Therefore x(t − τ) is causally independent of any proposition about x(t). An instrument reading x at time (t − τ) observes a value that is upstream of the current feedback cycle. This observation point exists if and only if τ > 0. When τ → 0, x(t) = F(x(t), x(t)) and no causally independent observation exists. But τ → 0 also eliminates the delay required for Hopf bifurcation and quarter-wave resonance, placing the system outside D-2.1. ∎ (sketch)

**Placement:** This conjecture bridges DRP-1 (where UMP is proved) and P1/P2 (where the FRM network conditions are defined). If proved, it should be registered as a claim in whichever paper formalizes it — likely P2 (as a corollary of universality) or a new DRP-3.

**Open questions:**
- Does the proof require continuous-time DDE specifically, or does it hold for any discrete-delay feedback architecture?
- What is the precise relationship between τ and the size of the "measurement window"? Is the window exactly τ, or is there a fraction (e.g., τ/4 from the quarter-wave)?
- Does UMP-FRM have implications for the measurement decoupling threshold (P11)? The threshold may be the boundary where the upstream window becomes too narrow for practical observation.

---

## Artifacts Produced
- `docs/FRM_SeriesBuildTable_v1.5.md` — updated to CorpusArch v10
- `journal/session_52_notes.md` — this file

## Next Steps
- P2 Phase 2 (hostile review) — in progress on claude.ai
- DRP-2 AI layer and build table entry — pending deposit from claude.ai
- UMP-FRM conjecture — formal proof needed; placement decision pending
