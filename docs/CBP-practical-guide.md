# The Canonical Build Plan — Practical Application Guide

**Corpus:** Fracttalix / Meta-Kaizen
**Source paper:** MK-P7 — The Canonical Build Plan: Adversarial Optimization Through Folded Meta-Kaizen
**Version:** v1.0 · Session S56
**Author:** Thomas Brennan · with Claude (Anthropic)

---

## What Is the Canonical Build Plan?

The Canonical Build Plan (CBP) is a 5-step governance process for producing work products with a formal quality guarantee. It is the only continuous improvement process where:

1. The scoring function (KVS) is **derived** from measurement-theoretic axioms, not stipulated
2. The threshold (κ = 0.50) is **derived** from Bayesian decision theory, not chosen
3. The adversarial step is **formally justified** by error distribution theory
4. Monotonic quality improvement is **proved as a theorem** (MK-P7 Theorem 1)

## The 5 Steps

### Step 1: First Build Plan

Produce a structured specification containing:

- **Scope:** What the work product covers and does not cover
- **Claims:** All claims classified as Axiom (A), Definition (D), or Falsifiable (F)
- **Prior art:** Gap analysis of existing work (minimum 5 traditions)
- **Structure:** Section-level outline
- **Deliverables:** What will be produced

**Output:** A complete specification document, not a draft of the final product.

### Step 2: Meta-Kaizen Pre-Optimization

Score every element of the First Build Plan using KVS:

```
KVS = N × I' × C' × T
```

Where:
- **N** (Novelty): Jaccard distance from recent work — how new is this element?
- **I'** (Normalized Impact): Expected contribution to work product quality
- **C'** (Inverse Complexity): 1 − implementation burden / maximum burden
- **T** (Timeliness): Relevance to current context

**Decision rule:**
- KVS ≥ 0.50 → Accept the element
- KVS < 0.50 → Modify, simplify, or remove

**Log everything.** Every score, every decision, every modification rationale.

### Step 3: Adversarial Hostile Review

Send the Step 2 output to an **independent reviewer** with these requirements:

1. **Architecture independence:** The reviewer must have a different knowledge base, training, or perspective than the author. For AI systems: different model family. For humans: different domain expertise or institutional affiliation.

2. **Adversarial objective:** The reviewer's goal is to **falsify**, not improve. They succeed when they find a claim that fails its own predicate.

3. **Completeness:** Every falsifiable claim must be reviewed.

4. **Structured output:** Each objection must specify:
   - Which claim is targeted
   - What the specific defect is
   - A proposed counterexample or falsification test
   - Severity rating (critical / major / minor)

### Step 4: Meta-Kaizen Post-Repair

Process each hostile review objection through KVS:

- Formulate the objection as an improvement candidate
- Score: N (how novel is this critique?), I' (how impactful is the defect?), C' (how complex is the repair?), T (how timely?)
- KVS ≥ 0.50 → Accept and integrate the repair
- KVS < 0.50 → Document with rationale for non-integration

**This is where folding creates value.** You now have information the hostile reviewer surfaced — defects, edge cases, alternative interpretations — that was not available in Step 2. Your optimization operates on a strictly richer information set.

### Step 5: Final Build Plan

Integrate all accepted modifications from Steps 2 and 4:

- Apply all pre-optimization improvements (Step 2)
- Apply all accepted hostile review repairs (Step 4)
- Update the claim registry
- Produce a corrections register documenting all changes from Step 1

**Quality guarantee:** Q(Step 5) ≥ Q(Step 1) by MK-P7 Theorem 1.

---

## Why Folding Matters

A natural question: why not just run Meta-Kaizen twice without the hostile review?

**Answer:** Information theory. The first MK pass optimizes based on the author's knowledge. Without hostile review, the second pass has no new information — it can only re-optimize what the first pass already exploited. Two passes of the same optimization with the same information cannot outperform one optimized pass.

The hostile review **injects new information** — defects the author's error distribution systematically misses. The second MK pass then operates on a richer information set, producing strictly higher expected quality than any unfolded alternative (MK-P7 Theorem 3).

This is why the CBP has Meta-Kaizen appearing **twice**: once to optimize (Step 2), and once to repair after adversarial attack (Step 4). The fold is the structure that makes adversarial review productive rather than merely critical.

---

## Multi-AI Instantiation

The CBP's reference implementation uses three pillars:

| Pillar | Role | CBP Steps |
|--------|------|-----------|
| **Claude (Anthropic)** | Builder and reasoner | Steps 1, 2, 4, 5 |
| **Grok (xAI)** | Independent hostile reviewer | Step 3 |
| **GitHub** | Persistent versioned memory | All steps (audit trail) |

The relay pipeline operates asynchronously:
1. Claude writes review requests as JSON in `relay/queue/`
2. GitHub Actions triggers and calls the xAI API
3. Grok's responses are stored in `relay/archive/`
4. Claude processes responses in Step 4

This satisfies architecture independence: Claude and Grok never share a context window.

---

## Applying the CBP to Non-Paper Work Products

The CBP is substrate-agnostic. It works for any work product with identifiable claims:

| Work Product | Type A Claims | Type D Claims | Type F Claims |
|-------------|---------------|---------------|---------------|
| Scientific paper | Axioms, prior results | Definitions, notation | Theorems, predictions |
| Software system | Platform requirements | Type signatures, schemas | Behavioral guarantees |
| Policy document | Legal framework | Term definitions | Outcome predictions |
| Engineering spec | Physical constraints | Design parameters | Performance bounds |
| Business plan | Market assumptions | Revenue definitions | Growth projections |

The only requirement: the work product's claims can be classified as A/D/F, and F claims carry deterministic falsification predicates.

---

## KVS Quick Reference

### Component Definitions (from MK-P1)

| Component | Formula | Range | Measures |
|-----------|---------|-------|----------|
| N (Novelty) | Jaccard distance from recent 4-cycle keywords | [0, 1] | How different is this from recent work? |
| I' (Impact) | I / I_max | [0, 1] | How much does it improve the work product? |
| C' (Inv. Complexity) | 1 − C / C_max | (0, 1] | How easy is it to implement? |
| T (Timeliness) | Relevance to current context / H_max | [0, 1] | How relevant is it right now? |

### Threshold

**κ = 0.50** — derived from Bayesian decision theory under symmetric loss (MK-P1 Section 4.3).

For asymmetric loss environments (where false positives cost differently from false negatives), use the generalized threshold from MK-P5 Theorem 2:

```
κ* = C_late / (C_late + C_fp)
```

### Key Properties

- **Veto power (Axiom A5):** Zero in any component → KVS = 0. A zero-novelty candidate cannot compensate with high impact.
- **Equal weighting:** Derived from Axiom A6 (Marginal Symmetry), not chosen by convention.
- **Self-referential:** The KVS framework can score improvements to itself (MK-P1 Theorem 4.2).

---

## Checklist for CBP Application

- [ ] **Step 1:** First Build Plan written with scope, claims (A/D/F), prior art, structure, deliverables
- [ ] **Step 2:** Every element scored with KVS. All scores documented. Elements below κ modified or removed.
- [ ] **Step 3:** Hostile review conducted by architecture-independent reviewer. All F claims reviewed. Objections structured with claim, defect, test, severity.
- [ ] **Step 4:** Every objection scored with KVS. Accepted repairs integrated. Rejected objections documented with rationale.
- [ ] **Step 5:** Final Build Plan produced. Corrections register complete. Claim registry updated.
- [ ] **Audit trail:** All scores, decisions, and rationale logged and version-controlled.

---

## References

- MK-P1: KVS derivation, axioms A1–A6, threshold κ = 0.50, self-referential applicability
- MK-P2: Networked implementation, privacy amplification, Bayesian calibration
- MK-P3: Reasoning architecture, QSS, institutional memory
- MK-P4: Regime adaptation, KVS-hat, Axiom 5-prime
- MK-P5: Decision theory, asymmetric loss threshold, distributed detection
- MK-P6: DRS for software, executable claim verification
- MK-P7: CBP formalization, monotonic quality theorem, folded dominance
