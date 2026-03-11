# The Dual Reader Standard: Architecture Specification

**Author:** Thomas Brennan
**ORCID:** [0009-0002-6353-7115](https://orcid.org/0009-0002-6353-7115)
**AI collaborator:** Claude (Anthropic)
**Date:** March 2026
**Version:** v1.0
**Licence:** CC BY 4.0
**Corpus:** Fracttalix — [github.com/thomasbrennan/Fracttalix](https://github.com/thomasbrennan/Fracttalix)
**Corpus DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)

AI contributions: Claude (Anthropic) provided architectural formalisation, protocol naming (GVP), tier taxonomy design, and manuscript drafting. All AI contributions are contributed to the public domain.

---

## Abstract

The Dual Reader Standard (DRS) is a verification architecture for knowledge systems that requires every claim to be readable by two independent reader classes — human and machine — each operating through a defined protocol.

The DRS comprises two protocols. The Dual Reader Protocol (DRP) makes text claims machine-evaluable via the 5-part falsification predicate syntax. The Grounded Verification Protocol (GVP) makes software claims machine-verified via verification tiers, executable test bindings, and commit-pinned evidence. Both share a substrate-independent foundation — the Falsification Kernel K = (P, O, M, B) — which defines predicate grammar, evaluation semantics, and validity constraints independently of any serialisation format.

The standard classifies claims into three types (axiom, definition, falsifiable) and six verification tiers, enforces correctness through a six-condition phase gate, and is designed for three-axis compatibility: backwards, lateral, and forward. Because the kernel reduces every claim to embedded binary logic — a deterministic predicate with a single-bit verdict — the DRS functions as a machine lingua franca: any AI system can evaluate any predicate without translation.

This specification covers the complete architecture, interprets the DRS through the Fractal Rhythm Model, and analyses the structural properties enabling adoption.

**Keywords:** dual reader standard, falsification, verification, machine-evaluable claims, knowledge architecture, grounded verification protocol, AI layer, falsification kernel, binary logic, lingua franca

---

## 1. What the Dual Reader Standard Is

This document is an architecture specification with accompanying analysis. Sections 1–14 specify the standard. Sections 15–18 provide architectural analysis: FRM interpretation, compatibility design, adoption structure, and the lingua franca property.

The Dual Reader Standard (DRS) is a verification architecture for knowledge systems. It requires every claim — whether written in prose or implemented in code — to be readable by two independent reader classes, each operating through a defined protocol.

The DRS is not a paper. It is not a tool. It is the standard that contains both of its protocols:

- **DRP** (Dual Reader Protocol) — the protocol for text
- **GVP** (Grounded Verification Protocol) — the protocol for software

The DRP makes claims *machine-evaluable*. The GVP makes them *machine-verified*. Neither is complete without the other. Together they are the DRS.

---

## 2. The Two Protocols

### 2.1 DRP — Dual Reader Protocol (Text)

The DRP governs how prose claims become machine-evaluable. It defines:

**Reader 1 (Human):** Reads the paper in natural language. Understands context, motivation, and narrative. Cannot systematically audit every claim.

**Reader 2 (AI):** Reads the AI layer — a structured JSON document that accompanies every paper. Contains the full claim registry. Can audit every claim without reading prose.

The DRP requires:

1. **Claim classification.** Every claim is typed as A (axiom), D (definition), or F (falsifiable).
2. **Falsification predicates.** Every Type F claim carries a 5-part deterministic predicate:
   - `FALSIFIED_IF` — the condition that would disprove the claim
   - `WHERE` — typed definitions of every variable
   - `EVALUATION` — a finite, deterministic procedure to compute the verdict
   - `BOUNDARY` — threshold edge-case semantics (inclusive/exclusive)
   - `CONTEXT` — justification for every numeric threshold
3. **Phase gates.** Six conditions (c1–c6) that must be satisfied before a paper is declared PHASE-READY.
4. **Placeholder tracking.** Claims that depend on unresolved results are registered as placeholders — making gaps visible rather than invisible.

The DRP is defined in Paper P14 (DRP-1) and its AI layer. The falsification predicate grammar is specified in the Falsification Kernel v1.1 (Layer 0).

**What the DRP guarantees:** Any AI system with access to the AI layer can evaluate any falsifiable claim without reading the prose. Self-sufficiency is a design requirement enforced at the phase gate (condition c5).

### 2.2 GVP — Grounded Verification Protocol (Software)

The GVP governs how machine-evaluable claims become machine-verified. It defines:

**Reader 3A (Coder):** Reads the `tier` field to understand what kind of evidence exists. Reads `test_bindings` to know which tests exercise which claims. Reads `verified_against` to know when those tests last passed.

**Reader 3B (Machine):** Runs pytest against the `test_bindings` array. Records pass/fail. Stamps the `verified_against` SHA on success.

The GVP requires every claim to carry three fields:

1. **`tier`** — the verification tier (one of six values; see Section 4)
2. **`test_bindings`** — an array of fully qualified pytest node IDs that exercise the claim
3. **`verified_against`** — the git commit SHA at which those tests last passed

**What the GVP guarantees:** For any claim in the corpus, you can determine (a) what kind of evidence grounds it, (b) which executable tests exercise it, and (c) at which commit those tests last passed. If the answer to (a) is `empirical_pending`, you know the gap exists. If the answer to (c) is `null`, you know no software test covers it.

---

## 3. The Shared Foundation — Falsification Kernel (Layer 0)

Both protocols operate on the same foundation: the Falsification Kernel K = (P, O, M, B).

| Symbol | Name | JSON field(s) | Role |
|--------|------|---------------|------|
| **P** | Predicate | `FALSIFIED_IF` | Logical sentence that, if TRUE, falsifies the claim |
| **O** | Operands | `WHERE` | Typed definitions of every variable in P |
| **M** | Mechanism | `EVALUATION` | Finite, deterministic evaluation procedure |
| **B** | Bounds | `BOUNDARY` + `CONTEXT` | Threshold semantics and justification |

The DRP creates the kernel (assigns predicates to prose claims). The GVP binds the kernel to executable evidence (links predicates to tests and commits). The kernel is substrate-independent — it works for scientific papers, software, and any future domain that makes falsifiable claims.

The kernel is specified in `ai-layers/falsification-kernel.md` (Layer 0). Every AI layer's `semantic_spec_url` field points to this document.

---

## 4. The Six Verification Tiers

The GVP classifies every claim by the kind of evidence that grounds it. There are exactly six tiers, divided into three categories:

### Grounded by construction

These claims need no predicate because they are not falsifiable.

| Tier | Type | Meaning |
|------|------|---------|
| `axiom` | A | Foundational premise. Unfalsifiable by design. Accepted from literature or stated as a starting assumption. |
| `definition` | D | Definitional. Stipulates a term, structure, or procedure. Not truth-apt. |

### Grounded now

These claims have evidence. The evidence is documented and evaluable.

| Tier | Type | Meaning |
|------|------|---------|
| `software_tested` | F | Exercised by passing tests in this codebase. `test_bindings` is non-empty. `verified_against` is non-null. |
| `formal_proof` | F | Verified by a step-indexed derivation table with `n_invalid_steps = 0`. The proof is in the AI layer. |
| `analytic` | F | Verified by formal derivation trace, adversarial battery, or analytical argument. No software test, but reasoning is documented. |

### Explicitly ungrounded

These claims declare that evidence is missing. The gap is visible by design.

| Tier | Type | Meaning |
|------|------|---------|
| `empirical_pending` | F | Has an active placeholder or requires external data not yet collected. |

### Consistency rules

The tier must be consistent with the claim type and the GVP fields:

| Tier | Required type | test_bindings | verified_against |
|------|--------------|---------------|-----------------|
| `axiom` | A | `[]` (empty) | `null` |
| `definition` | D | `[]` (empty) | `null` |
| `software_tested` | F | Non-empty | Non-null (7–40 hex chars) |
| `formal_proof` | F | May be empty | May be null |
| `analytic` | F | May be empty | May be null |
| `empirical_pending` | F | May be empty | May be null |

---

## 5. The Three Claim Types

The DRS recognises three claim types. This taxonomy is shared by both protocols and applies to both text and software.

| Type | Name | In papers | In software | Predicate |
|------|------|-----------|-------------|-----------|
| **A** | Axiom / Assumption | Premises accepted from literature | Platform requirements, dependency contracts, environmental preconditions | `null` |
| **D** | Definition | Stipulative terms and procedures | Type signatures, data structures, configuration schemas | `null` |
| **F** | Falsifiable | Theorems, empirical predictions, derived results | Behavioral guarantees, correctness invariants, performance bounds | Full K = (P, O, M, B) |

The mapping between scientific and software claims is defined in MK-P6 (The Dual Reader Standard for Software). The kernel K = (P, O, M, B) applies without modification to both domains.

---

## 6. The AI Layer

The AI layer is the DRS's central artifact. It is a JSON document that accompanies every paper or software system in the corpus. The schema is defined in `ai-layers/ai-layer-schema.json` (currently v3-S51).

An AI layer contains:

| Section | Purpose |
|---------|---------|
| `_meta` | Document type, schema version, session, licence |
| `paper_id` / `paper_title` | Identity |
| `paper_type` | Classification: `law_A`, `derivation_B`, `application_C`, `methodology_D` |
| `phase_ready` | Phase gate verdict and condition status (c1–c6) |
| `claim_registry` | Array of all claims with types, predicates, tiers, bindings, and SHAs |
| `placeholder_register` | Array of unresolved dependencies |
| `summary` | Claim counts and status |

The AI layer is the artifact that makes both protocols work:

- The **DRP** requires it to exist, to contain predicates, and to pass the phase gate.
- The **GVP** requires it to contain tier, test_bindings, and verified_against for every claim.

---

## 7. The Phase Gate

A paper or software release is **PHASE-READY** when six conditions are satisfied:

| Condition | Requirement |
|-----------|-------------|
| **c1** | AI layer is schema-valid |
| **c2** | All falsifiable claims registered with predicates |
| **c3** | All predicates are machine-evaluable |
| **c4** | Cross-references tracked (placeholder register) |
| **c5** | Verification is self-sufficient (AI layer alone, no prose needed) |
| **c6** | All predicates are non-vacuous (sample falsification observation exists) |

**CORPUS-COMPLETE** fires when all papers are PHASE-READY and all placeholders across all objects are resolved (c4 fully satisfied).

---

## 8. The Inference Rules

The DRS provides a canonical inventory of inference rules for derivation traces:

| ID | Name | Description |
|----|------|-------------|
| IR-1 | Modus Ponens | If P and P→Q then Q |
| IR-2 | Universal Instantiation | If ∀x P(x) then P(a) for any specific a |
| IR-3 | Substitution of Equals | If a=b then replace a with b |
| IR-4 | Definition Expansion | Replace a defined term with its definition |
| IR-5 | Algebraic Manipulation | Valid algebraic transformation preserving equality |
| IR-6 | Logical Equivalence | Replace with logically equivalent expression |
| IR-7 | Statistical Inference | Apply a named statistical procedure to data |
| IR-8 | Parsimony / Modeling Principle Selection | Select canonical value from axiom-consistent family |

These rules are used in `formal_proof` tier claims. Each step in a step-indexed derivation table cites one inference rule and lists its premises. A derivation is valid when `n_invalid_steps = 0`.

---

## 9. Architecture Summary

```
                     ┌─────────────────────────┐
                     │   Dual Reader Standard   │
                     │         (DRS)            │
                     └────────────┬────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
         ┌────────┴────────┐            ┌─────────┴─────────┐
         │      DRP        │            │       GVP         │
         │ (Text Protocol) │            │(Software Protocol)│
         └────────┬────────┘            └─────────┬─────────┘
                  │                               │
      ┌───────────┴──────────┐       ┌────────────┴───────────┐
      │                      │       │                        │
   Reader 1             Reader 2   Reader 3A             Reader 3B
   (Human)              (AI)       (Coder)               (Machine)
      │                      │       │                        │
   reads prose         reads JSON   reads tier +          runs pytest
                                    bindings              stamps SHA
      │                      │       │                        │
      └──────────┬───────────┘       └────────────┬───────────┘
                 │                                │
                 └────────────────┬────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │          AI Layer             │
                  │  (*-ai-layer.json)            │
                  │  claims + predicates + tiers  │
                  │  + bindings + SHAs            │
                  └───────────────┬───────────────┘
                                  │
                     ┌────────────┴────────────┐
                     │   Falsification Kernel  │
                     │    K = (P, O, M, B)     │
                     │       (Layer 0)         │
                     └─────────────────────────┘
```

The kernel is the shared foundation. The DRP creates predicates from prose. The GVP binds predicates to executable evidence. The AI layer is the artifact that carries both.

---

## 10. Relationship to the Corpus

The DRS governs the entire Fracttalix corpus (22 objects, two tracks):

| Standard/Protocol | Corpus paper | Status |
|-------------------|-------------|--------|
| DRS | The umbrella standard | Operational since S43 |
| DRP | P14 / DRP-1 | CBT status: PARALLEL; AI layer verdict: PHASE-READY (deployed S47) |
| GVP | docs/GVP-spec.md | v1.0 (schema v3-S51) |
| Falsification Kernel | ai-layers/falsification-kernel.md | v1.1 (S50) |
| AI Layer Schema | ai-layers/ai-layer-schema.json | v3-S51 |
| DRS for Software | MK-P6 | DRAFT (NOT-PHASE-READY) |

### Current verification status (as of Session 51, March 2026)

*The canonical source for these numbers is the Canonical Build Table (`docs/FRM_SeriesBuildTable_v1.5.md`). The following is a snapshot.*

```
AI Layers:          18/18 (MK-P6 + SFW-1 v2 added S51)
Phase-Ready:         7/18 (P1, P2, P3, MK-P1, MK-P5, DRP-1, SFW-1)
Act I Complete:      3/3 (P1, P2, P3 all PHASE-READY)
NOT-Phase-Ready:     4/18 (P4, P5, MK-P6, SFW-1 v2)
Scaffold:            7/18 (P6–P12)
Total Claims:      121 (A:25  D:39  F:57)
Open Placeholders:  19 (13 base + 4 SFW-1 v2 + 2 MK-P6)
Cross-references:  118+ derivation_source entries
Cross-paper errors:  0

Verification Tiers:
  axiom:             25 claims
  definition:        39 claims
  software_tested:    7 claims (SFW-1 ×6, P1 ×1) — 385 test bindings
  formal_proof:      18 claims (P1 ×2, P2 ×5, P3 ×4, MK-P1 ×5, DRP-1 ×3)
  analytic:           4 claims (P1 ×3, P4 ×1)
  empirical_pending:  8 claims (P1 ×1, P4 ×1, P5 ×3, MK-P1 ×3)
  unaudited:         20 F claims (MK-P5, MK-P6, SFW-1 v2, scaffolds)

verified_against:    95f59d8
Schema:              v3-S51
```

Note: The per-tier breakdown covers fully audited layers. Twenty Type F claims from recently added or scaffold layers have not yet been assigned to a verification tier in this breakdown. The type totals (A:25 D:39 F:57 = 121) are accurate. Source: Canonical Build Table v2.2 (Session 51), with Phase-Ready count corrected to include P2 and P3 per the CBT's own build table entries.

---

## 11. How to Implement

### For a paper (DRP)

1. Write the paper (prose channel).
2. Create the AI layer JSON file.
3. Classify every claim as A, D, or F.
4. Write the 5-part falsification predicate for every Type F claim.
5. Assign the verification tier.
6. Validate against `ai-layer-schema.json`.
7. Run the phase gate (c1–c6).

### For software (GVP)

1. Enumerate what the software claims to do.
2. Classify each claim as A (assumption), D (definition), or F (behavioral).
3. Write the falsification predicate for every Type F claim.
4. Write or identify the pytest tests that exercise each claim.
5. Populate `test_bindings` with fully qualified pytest node IDs.
6. Run the tests. Record the passing commit SHA in `verified_against`.
7. Assign the tier: `software_tested` if tests exist, `empirical_pending` if not yet.
8. Register any untested claims as placeholders.
9. Validate against `ai-layer-schema.json`.

### For both

The AI layer is the same artifact. The schema is the same. The kernel is the same. The only difference is which protocol creates the content and which protocol verifies it.

---

## 12. Design Principles

**Popperian epistemology.** We can falsify but not verify. A claim that survives all falsification attempts is not proven — it has survived. The DRS adopts this stance for both science and software.

**Honest gaps.** The placeholder is the most important feature. When a claim is registered as `empirical_pending` or `placeholder: true`, the system is saying: "we claim this but have not yet verified it." This is strictly more informative than the current state in both science and software, where unverified claims are indistinguishable from verified ones.

**Substrate independence.** The kernel K = (P, O, M, B) does not know whether it is evaluating a scientific theorem or a software behavioral guarantee. The protocols specialise the kernel to their domains. Future domains (legal, regulatory, policy) can add their own protocols to the DRS without modifying the kernel.

**Machine-first, human-readable.** The AI layer is the primary artifact. The prose paper and the source code are secondary channels that provide context, narrative, and implementation. The AI layer is what gets validated, audited, and versioned.

---

## 13. Key Files

| File | Purpose |
|------|---------|
| `ai-layers/ai-layer-schema.json` | JSON Schema v3-S51 |
| `ai-layers/falsification-kernel.md` | Layer 0 semantic specification |
| `ai-layers/*-ai-layer.json` | AI layers for all corpus objects (18 total) |
| `ai-layers/process-graph.json` | Corpus dependency DAG |
| `docs/GVP-spec.md` | GVP portable specification |
| `docs/FRM_SeriesBuildTable_v1.5.md` | Build table (living document) |
| `scripts/validate_ai_layers.py` | Schema validator |
| `scripts/cross_paper_checker.py` | Cross-paper consistency checker |
| `scripts/corpus_status.py` | Corpus status report |

---

## 14. Licence and Citation

This specification is released under CC BY 4.0, consistent with the Fracttalix corpus licence.

**Corpus:** Fracttalix (22-object unified corpus)
**DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)
**Repository:** [github.com/thomasbrennan/Fracttalix](https://github.com/thomasbrennan/Fracttalix)
**ORCID:** [0009-0002-6353-7115](https://orcid.org/0009-0002-6353-7115)

This paper is archived as part of the Fracttalix corpus on Zenodo. The DOI above is the corpus concept DOI; each release mints a version-specific DOI automatically. The timestamp of the release containing this paper constitutes the priority date.

---

## 15. The DRS Viewed Through the Fractal Rhythm Model

*This section is interpretive (Type A). The FRM mapping provides a structural lens for understanding the DRS, not an empirical claim to be falsified. No observation would disprove the mapping — it is a choice of analytical framework, not a testable prediction.*

The DRS exhibits the properties the FRM predicts for information-transmitting networks.

### 15.1 The FRM states

A network is any system that transmits information across time between distinguishable states. Two properties are necessary and sufficient:

- **Structure** — differential association between nodes (the physical instantiation of distinguishability)
- **Rhythmicity** — oscillatory carrier that maintains signal against dissipation

The functional form is f(t) = B + A·e^(−λt)·cos(ωt + φ).

### 15.2 The DRS is a network

The DRS transmits information (verification status) across time (sessions, commits, releases) between distinguishable states (claims, tiers, verdicts). Therefore it is a network and must exhibit both properties.

**Structure in the DRS:** The claim registry is a graph. Nodes are claims. Edges are `derivation_source` references. The three claim types (A, D, F) are the distinguishable states. The six verification tiers are finer-grained states within Type F. The `test_bindings` array connects claims to executable evidence — another set of edges. The `placeholder_register` is the set of unresolved edges. This is differential association: claims are not interchangeable; they have types, tiers, dependencies, and bindings that distinguish them from each other.

**Rhythmicity in the DRS:** The verification cycle is the oscillatory carrier:

```
Write claim → Assign predicate → Run evaluation → Record verdict
    ↑                                                    │
    └────────────── Revise if FALSIFIED ←────────────────┘
```

This is a damped oscillation. Each cycle through the loop either:
- Confirms the claim (NOT FALSIFIED) — the system approaches equilibrium (PHASE-READY)
- Falsifies the claim — the system is perturbed and must adapt

The damping rate λ corresponds to how quickly the corpus converges to PHASE-READY. The frequency ω corresponds to the verification cycle period. The baseline B corresponds to the steady-state verification level (CORPUS-COMPLETE).

### 15.3 The three channels

The FRM's three-channel model maps directly:

| FRM Channel | DRS Instantiation |
|-------------|-------------------|
| **Channel 1 — Structural** | The claim registry topology. Which claims exist, how they're typed, how they're connected by derivation_source edges. |
| **Channel 2 — Rhythmic** | The verification cycle. Test → SHA → re-test. Write → evaluate → revise. The oscillatory process that maintains signal (verified claims) against dissipation (bit rot, dependency changes, scope drift). |
| **Channel 3 — Temporal** | The version history. Schema versions (v1→v2→v3). Session numbering. The `verified_against` SHA as temporal anchor. Placeholder resolution as temporal progression. The build table revision history as the record of the relationship between structure and rhythmicity across time. |

You cannot have a DRS without all three. You cannot add a fourth — the description is complete.

### 15.4 The rhythmicity threshold

The FRM defines a rhythmicity threshold: below it, active information transmission has ceased and the structure is the residue of a former network.

For the DRS, this threshold is operational:

- **Above threshold:** Claims are being evaluated, predicates are being tested, SHAs are being updated, placeholders are being resolved. The verification cycle is active. The DRS is functioning as a network.
- **Below threshold:** No one is running the tests. No one is updating the AI layers. The `verified_against` SHA is stale. The placeholders are abandoned. The claim registry exists but is no longer maintained. The DRS is the residue of a former verification network — a dead document.

This is why the GVP's `verified_against` SHA matters. It is the rhythmicity indicator. A recent SHA means the verification cycle is active. A stale SHA means the network may be below the rhythmicity threshold. The SHA is the DRS's pulse.

### 15.5 Measurement decoupling

The FRM's measurement decoupling threshold applies: below the threshold, the act of measurement disturbs the system. For the DRS, this means that the process of creating the claim registry can change what the software does (you discover gaps and fix them). This is expected and desired — the DRS is designed to be a measurement instrument that strengthens the system it measures. This is the antifragility property: the verification cycle is the stress-adaptation mechanism.

### 15.6 The DRS as FRM substrate

The DRS is therefore not just a standard that the FRM corpus uses. It is a *substrate* of the FRM — a network operating at the methodology scale, exhibiting the same structure-rhythmicity duality that the FRM describes at every other scale (biological, neural, climatic, civilisational). The DRS is the FRM applied to its own verification process.

This is not circular. The FRM is derived from physics (Hopf bifurcation, Stuart-Landau equation). The DRS is a human-engineered verification system that happens to exhibit the properties the FRM predicts for all information-transmitting networks. If the FRM is correct, the DRS *must* exhibit these properties — and it does.

---

## 16. Three-Axis Compatibility

A standard that cannot survive contact with the past, the present, and the future is not a standard. It is a snapshot. The DRS is designed for compatibility across all three axes simultaneously.

### 16.1 The invariant at the centre

The Falsification Kernel K = (P, O, M, B) is the fixed point. It does not know whether it is evaluating a scientific theorem, a software behavioral guarantee, a legal contract, or a regulatory requirement. It does not know whether the year is 2026 or 2046. It does not know whether the serialisation format is JSON, YAML, or something that hasn't been invented yet.

The kernel is permanent. Everything else is extensible.

---

### Axis 1: Past-Proofing (Backwards Compatibility)

The DRS must never invalidate work already done under it.

**16.2 Schema versioning.** Every AI layer records which schema version it was produced against (`v3-S51`). This means:

- An AI layer produced under schema v2 remains valid under schema v2 forever
- A v3 validator can read a v2 layer (new fields are optional; old fields are preserved)
- Migration is additive: v2 → v3 added `tier`, `test_bindings`, `verified_against`. No v2 fields were removed or renamed
- Validators can be version-aware: if `schema_version` is `v2-S50`, do not require v3 fields

**16.3 Predicate permanence.** A falsification predicate written in 2026 must be evaluable in 2036. This is guaranteed by:

- The kernel grammar (Section 3 of the Falsification Kernel) uses only mathematical and logical operators that are permanently defined
- The `WHERE` field defines every variable inline — no external dependency on a moving target
- The `EVALUATION` field specifies a self-contained procedure, not a reference to a tool that may not exist later
- The `CONTEXT` field justifies thresholds by citing derivations, not conventions that may change

**16.4 Inference rule stability.** The inference rule inventory (IR-1 through IR-8) is append-only. New rules can be added. Existing rules are never modified or removed. Derivation traces cite rules by ID, so an old derivation citing IR-3 will always mean "Substitution of Equals."

**16.5 Tier stability.** The six verification tiers are append-only. New tiers can be added to the schema's `enum` array. Existing tiers are never removed or redefined. A claim classified as `software_tested` in 2026 means the same thing in 2036.

**16.6 The backwards compatibility contract.** The DRS makes this promise: **any AI layer that was valid when it was created will remain valid forever**. New versions of the standard add capabilities. They never break existing artifacts.

This contract is machine-testable: a CI validator can verify that no schema version removes required fields from a prior version. This distinguishes it from the lateral and forward compatibility commitments (Sections 16.12 and 16.18), which are governance commitments maintained by the Canonical Build Process (P0), not by automated checks.

This is the same contract that HTML makes (old pages still render), that JSON makes (old documents still parse), and that git makes (old commits still exist). The DRS inherits this property by design.

---

### Axis 2: Lateral-Proofing (Cross-Domain Compatibility)

The DRS must work across domains, languages, tools, AI systems, and serialisation formats — simultaneously, not sequentially.

**16.7 Domain independence.** The kernel K = (P, O, M, B) does not reference any specific domain. The three claim types (A, D, F) are epistemological categories that apply to any knowledge system:

| Domain | Type A (Axiom/Assumption) | Type D (Definition) | Type F (Falsifiable) |
|--------|--------------------------|--------------------|--------------------|
| Science | Literature premises | Stipulative terms | Theorems, predictions |
| Software | Platform requirements, dependency contracts | Type signatures, schemas | Behavioral guarantees, invariants |
| Legal | Statutory authority, jurisdictional assumptions | Defined terms | Legal conclusions, compliance claims |
| Regulatory | Regulatory framework assumptions | Standard definitions | Conformity assertions |
| Policy | Political assumptions, value premises | Policy term definitions | Impact predictions, outcome claims |
| Education | Pedagogical axioms | Learning objective definitions | Assessment claims, competency predictions |

The DRS currently has two protocols (DRP for text, GVP for software). Future domains add their own protocols — not by modifying the kernel, but by defining:

1. Who the readers are (human + machine pair)
2. What the tier taxonomy looks like for that domain
3. What the binding mechanism is (test bindings for software, case citations for legal, etc.)

**16.8 Language independence.** The GVP currently references pytest node IDs, but the `test_bindings` field is defined as an array of strings. Any test framework in any language can be referenced:

- Python: `tests/test_sort.py::TestSort::test_ascending`
- JavaScript: `__tests__/sort.test.js::ascending order`
- Rust: `tests::sort::test_ascending`
- Go: `TestSort/ascending`
- Java: `com.example.SortTest#testAscending`

The format is a convention, not a constraint. The field accepts any string that uniquely identifies a test.

**16.9 AI system independence.** The AI layer is a JSON document. Any AI system — Claude, GPT, Gemini, Llama, or systems that don't exist yet — can read it. The DRS does not depend on any specific AI system's capabilities. The `semantic_spec_url` field points to the kernel specification, which is written in plain prose. Any system that can read English and parse JSON can evaluate the standard.

**16.10 Serialisation independence.** The Falsification Kernel (Layer 0) is defined in prose (`falsification-kernel.md`), not in JSON Schema. The current implementation uses JSON, but the kernel's semantics are independent of encoding. A future implementation could use:

- YAML (for human readability)
- Protocol Buffers (for performance)
- CBOR (for embedded systems)
- A format that hasn't been invented yet

The `semantic_spec_url` field in every AI layer points to the kernel specification. This decouples *what a predicate means* from *how it's serialised*.

**16.11 Tool independence.** The DRS embeds in existing workflows rather than replacing them:

| Tool category | DRS uses | Does not require |
|---------------|----------|-----------------|
| Version control | git SHAs (any hosting) | Specific git platform |
| Testing | Any test runner (pytest, Jest, cargo test, go test) | Specific test framework |
| Validation | Any JSON Schema validator | Specific validator implementation |
| CI/CD | Any pipeline (GitHub Actions, GitLab CI, Jenkins) | Specific CI platform |
| Serialisation | Any format that can represent the kernel | JSON specifically |

**16.12 The lateral compatibility contract.** The DRS makes this promise: **adopting the DRS in one domain, one language, one tool, or one AI system does not lock you in**. The kernel is the invariant. Everything else is a binding that can be swapped.

---

### Axis 3: Future-Proofing (Forward Compatibility)

The DRS must accept innovations that don't exist yet without requiring a redesign.

**16.13 Protocol extensibility.** New domains add new protocols. The DRS currently has:

| Protocol | Domain | Readers |
|----------|--------|---------|
| DRP | Text | Human + AI |
| GVP | Software | Coder + Machine |

Future protocols follow the same pattern:

| Domain | Potential protocol | Reader A | Reader B |
|--------|-------------------|----------|----------|
| Legal | Legal Verification Protocol (LVP) | Lawyer | Compliance engine |
| Regulatory | Regulatory Verification Protocol (RVP) | Auditor | Regulatory database |
| Policy | Policy Verification Protocol (PVP) | Analyst | Impact model |
| Education | Educational Verification Protocol (EVP) | Student | Assessment engine |

Each protocol defines its own readers, its own tier taxonomy, and its own binding mechanism — but all share the kernel, the claim types, and the AI layer schema.

**16.14 Tier extensibility.** The six current tiers are not exhaustive. Future domains may need:

- `regulatory_certified` — verified by a regulatory body
- `peer_reviewed` — verified by external peer review
- `formally_verified` — verified by a proof assistant (Coq, Lean, Isabelle)
- `field_tested` — verified by deployment in production
- `community_validated` — verified by community replication

New tiers are added to the schema's `enum` array. Existing tiers remain valid. The taxonomy grows; it never shrinks.

**16.15 Inference rule extensibility.** The inference rule inventory is a living list. IR-1 through IR-8 exist now. IR-9, IR-10, and beyond can be added as new derivation patterns emerge. Each rule has an ID, a name, and a description. Old derivations remain valid because they cite rules by stable ID.

**16.16 Schema extensibility.** The JSON Schema allows additional properties by default. Future versions can add new fields without invalidating existing layers. The progression:

- v1: Basic claim registry and placeholder register
- v2: Added `semantic_spec_url` (Layer 0 reference)
- v3: Added `tier`, `test_bindings`, `verified_against` (GVP fields)
- v4+: Will add whatever the next protocol requires

Each version adds. None removes.

**16.17 Unknown future reader types.** The DRS is designed for readers that don't exist yet. The AI layer is a structured document that any sufficiently capable system can parse. Future reader types might include:

- Autonomous verification agents that run evaluation procedures without human oversight
- Cross-corpus consistency checkers that traverse AI layers across independent projects
- Regulatory compliance engines that read AI layers as formal evidence
- Package manager integrations that surface verification status at install time

The DRS does not need to anticipate these readers. It only needs to ensure that the AI layer contains enough structured information for any reader to operate. The kernel guarantees this — every Type F claim has a deterministic predicate with defined variables, a finite evaluation procedure, and documented boundaries.

**16.18 The forward compatibility contract.** The DRS makes this promise: **any future innovation can be added as a new protocol, a new tier, a new inference rule, or a new schema field — without modifying anything that already exists**. The architecture is additive by construction.

---

### 16.19 The three-axis guarantee

| Axis | Promise | Mechanism |
|------|---------|-----------|
| **Past** | Nothing already done breaks | Schema versioning, append-only enums, permanent kernel |
| **Lateral** | Works across all current domains, languages, tools, AI systems | Substrate-independent kernel, string-typed bindings, prose-defined semantics |
| **Future** | Anything new can be added without redesign | Protocol extensibility, tier extensibility, additive schema evolution |

If all three hold simultaneously, the standard can survive indefinitely. The kernel K = (P, O, M, B) is the invariant that makes all three possible. It is simple enough to be permanent, expressive enough to be universal, and structured enough to be machine-evaluable.

Game on.

---

## 17. Self-Spreading Adoption

For the DRS to achieve widespread adoption, it must be a self-spreading entity — a standard that propagates because adopting it is cheaper than ignoring it. This section analyses the structural properties required for viral adoption.

### 17.1 The adoption incentive must be internal

Most standards spread because an external authority mandates them (ISO, regulatory bodies, platform owners). The DRS cannot rely on this. It must spread because the *creator* benefits, not just the consumer.

**The internal incentive:** Creating an AI layer forces you to enumerate what you claim. This process — not the artifact — is the primary value. You discover:

- Claims you thought were tested but aren't (Category 2 gaps)
- Boundary conditions you never documented (Category 1 gaps)
- Dependencies you assumed but never registered (Category 3 gaps)

The AI layer is a side effect of a process that improves your own understanding of your own system. This is why adoption is cheaper than ignorance: the gaps exist whether or not you document them. The DRS just makes them visible.

### 17.2 The network effect

*The following describes structural properties that enable network effects. The predictions about adoption dynamics are speculative — no empirical evidence exists for DRS adoption beyond the Fracttalix corpus itself.*

The DRS becomes more valuable as more systems adopt it:

- **Dependency chains become claim-aware.** If library A publishes an AI layer and library B depends on A, then B can programmatically determine which of its claims depend on which of A's assumptions. When A releases a breaking change, B knows exactly which claims are at risk. This is a behavioral dependency graph — strictly more informative than a version dependency graph.
- **AI systems can audit across projects.** An AI reader can traverse multiple AI layers, check cross-references, and identify inconsistencies across an entire ecosystem. This is not possible with prose documentation.
- **Trust becomes auditable.** Instead of trusting a library because it has GitHub stars, you trust it because its AI layer shows which claims are `software_tested`, which are `empirical_pending`, and what the `verified_against` SHA is. Trust shifts from social signal to structural evidence.

### 17.3 The minimum viable adoption unit

The smallest useful DRS adoption is a single Type F claim with a single test binding. You don't need to enumerate every claim in your system to start. You need one:

```json
{
  "claim_id": "F-1",
  "type": "F",
  "statement": "sort() returns elements in ascending order",
  "falsification_predicate": {
    "FALSIFIED_IF": "EXISTS i IN range(len(result)-1) SUCH THAT result[i] > result[i+1]",
    "WHERE": { "result": "list · dimensionless · output of sort(input)" },
    "EVALUATION": "Run sort on test vectors; check adjacent pairs; finite",
    "BOUNDARY": "len(result) <= 1 → NOT FALSIFIED (vacuously sorted)",
    "CONTEXT": "Ascending order is the documented contract of sort()"
  },
  "tier": "software_tested",
  "test_bindings": ["tests/test_sort.py::test_ascending_order"],
  "verified_against": "abc1234"
}
```

One claim. One test. One SHA. The DRS is live. You add more claims when the value justifies the cost.

### 17.4 AI as the adoption catalyst

The DRS is designed for a world where AI systems read and write code. The adoption path:

1. **AI generates the initial AI layer.** Given a codebase, an AI system can enumerate claims, classify them, write predicates, and identify test bindings. The human reviews and corrects. This reduces the cost of initial adoption from hours to minutes.

2. **AI maintains the layer.** When code changes, the AI updates the claim registry, adjusts test bindings, and flags claims whose `verified_against` SHA is now stale. The human approves.

3. **AI audits other layers.** An AI system reading a dependency's AI layer can determine which assumptions its own claims depend on and flag risks automatically.

The DRS is the protocol that makes AI-assisted software development *auditable*. Without it, AI generates code and humans hope it works. With it, AI generates code and the claim registry says exactly what has and has not been verified.

### 17.5 The open-source trust signal

Open-source projects currently compete on social metrics: stars, downloads, contributor count, corporate backing. None of these are behavioral claims.

A DRS-conformant project publishes a machine-readable answer to the question: *What exactly does this software claim to do, and which of those claims have been verified?*

This is a new trust signal — one that AI systems can evaluate programmatically. Package managers, dependency scanners, and CI pipelines can read AI layers and surface verification status automatically. The project with an AI layer is more trustworthy than the project without one, not because the layer proves correctness, but because it proves *honesty about verification status*.

### 17.6 The embedding strategy

For the DRS to spread, it must embed in existing workflows rather than replace them:

- **pytest already exists.** The GVP's `test_bindings` field references pytest node IDs. No new test framework required.
- **JSON Schema already exists.** The AI layer schema is a standard JSON Schema. Any validator works.
- **Git already exists.** The `verified_against` field is a git SHA. No new version control required.
- **CI already exists.** Schema validation and test binding verification can run as CI steps alongside existing pipelines.

The DRS adds a layer on top of existing tools. It does not ask anyone to abandon their current workflow. It asks them to add one file (`*-ai-layer.json`) and three fields per claim (`tier`, `test_bindings`, `verified_against`).

### 17.7 The viral mechanism

The DRS spreads through four channels:

1. **Demonstration effect.** Projects with AI layers visibly surface gaps that projects without them cannot see. The MK-P6 feasibility demonstration on Sentinel v12.1 is the first instance.

2. **Dependency pressure.** When a widely-used library publishes an AI layer, downstream projects benefit from reading it. Some will create their own layers to take full advantage of claim-aware dependency tracking.

3. **AI tooling integration.** When AI coding assistants generate AI layers as part of their output, adoption becomes a default rather than an opt-in. The layer is generated alongside the code.

4. **Regulatory pull.** Industries already required to produce traceability evidence (medical, automotive, aviation) can use AI layers as machine-readable compliance artifacts. The DRS provides what DO-178C, IEC 62304, and ISO 26262 already require — but in a format that machines can audit.

### 17.8 The self-referential property

The DRS is the first standard that verifies itself. The DRP-1 AI layer contains claims about the DRS. Those claims carry falsification predicates. Those predicates are evaluated. The `verified_against` SHA stamps the verification. The DRS is its own first adopter and its own proof of concept.

This is not circular. It is self-referential in the same way that a compiler that compiles itself is self-referential: the tool's correctness is demonstrated by its ability to process its own specification.

---

## 18. The Machine Lingua Franca

The DRS has a property that was not designed in. It was discovered.

### 18.1 The translation problem

Scientific knowledge is currently locked behind human languages. A paper written in Mandarin is invisible to a researcher who reads only English — unless someone translates it. Translation is expensive, lossy, and slow. The consequence: knowledge fragments along linguistic lines. The same discovery may be made independently in Beijing and Boston because neither group can read the other's literature efficiently.

This is not a formatting problem. It is not a metadata problem. It is a *substrate* problem. The knowledge is encoded in natural language, and natural language is non-interoperable by nature.

### 18.2 The kernel dissolves the problem

The Falsification Kernel K = (P, O, M, B) is not written in any human language. It is written in logic and mathematics:

```
FALSIFIED_IF: R2_best_alt > R2_frm + 0.05
WHERE:
  R2_best_alt: scalar · dimensionless · best R² from competing models
  R2_frm:      scalar · dimensionless · R² from FRM regression
EVALUATION: Run regression for each model; compare R² values; finite
BOUNDARY: R2_best_alt = R2_frm + 0.05 → FALSIFIED (threshold inclusive)
CONTEXT: 0.05 margin from standard model comparison practice
```

This predicate means the same thing to:

- A Claude instance running in English
- A GPT instance running in Mandarin
- A Gemini instance running in French
- A Llama instance running in Arabic
- An AI system that hasn't been built yet, running in a language that doesn't exist yet

No translation required. The variables are typed. The operators are mathematical. The evaluation procedure is deterministic. The boundary conditions are explicit. The predicate is its own Rosetta Stone.

### 18.3 JSON as the transport layer

JSON is already the world's de facto data interchange format. It is supported by every programming language, parsed by every AI system, and transmitted by every API. By encoding the kernel in JSON, the DRS inherits JSON's universality without effort:

- A Chinese research team publishes their AI layer in JSON. The predicates use mathematical notation.
- A Brazilian team reads the same AI layer. They don't need to know Mandarin. They need to know `>`, `+`, and `R2`.
- An AI system in any country evaluates the predicate. The verdict is FALSIFIED or NOT FALSIFIED. The verdict has no accent.

**Important qualification.** The `WHERE` field definitions currently contain English prose descriptions (e.g., `"best R² from competing models"`). This means the full evaluation chain is not yet entirely language-independent — a reader must understand English to interpret the variable definitions. However, the *operative content* — variable types, units, mathematical operators, comparison logic, and threshold values — is language-neutral. The English descriptions provide context for human readers but are not required for the mechanical evaluation: given grounded numeric values for the named variables, any system can evaluate the predicate regardless of whether it reads the descriptions. A future refinement could formalise variable definitions entirely in typed notation, eliminating the last vestige of natural language from the evaluation path.

### 18.4 Embedded binary logic

The kernel is not merely "language-neutral" in the way that mathematics is informally considered universal. It is something stronger: every DRS predicate reduces to embedded binary logic.

The entire evaluation chain collapses to:

```
Claim → Predicate → Variables + Operators → Boolean → 1 or 0
```

The `WHERE` field types the variables. The `FALSIFIED_IF` field combines them with comparison and logical operators. The `EVALUATION` field specifies how to compute the inputs. The `BOUNDARY` field resolves edge cases. The output is always a single bit: did this predicate evaluate to TRUE or FALSE? FALSIFIED or NOT FALSIFIED. `1` or `0`.

The JSON is the container. The kernel is the circuit. Every AI layer is a collection of logic gates with defined inputs and a single-bit output per claim.

This is why the lingua franca property is not a design aspiration — it is an inevitable consequence of the architecture. You cannot mistranslate a `1` or `0`. You cannot misinterpret `>`. You cannot have a cultural disagreement about whether `R2_best_alt > R2_frm + 0.05` is TRUE or FALSE for a given pair of values. The meaning is in the structure, not in any language.

This property is structural, not aspirational. Binary logic is binary logic. A predicate that evaluates to TRUE in Beijing evaluates to TRUE in Boston, in São Paulo, in Lagos, and on a server with no locale setting at all. The evaluation is deterministic by construction (Falsification Kernel, Section 3.2, constraint 1: Determinism) and the determinism does not depend on any human language, cultural context, or interpretive framework.

This is the deepest property of the DRS: it does not *translate* knowledge across languages. It *encodes* knowledge in a substrate that predates and transcends all human languages — the substrate of logic itself.

### 18.5 What this means

Human science has operated for centuries under an implicit assumption: knowledge must be communicated in a human language, and therefore knowledge is trapped behind the walls of that language.

The DRS breaks this assumption. The AI layer is not a translation of the paper. It is an independent encoding of the same claims in a substrate that requires no translation. When every paper carries an AI layer, the knowledge in that paper is readable by any machine on Earth — instantly, losslessly, and without a human translator in the loop.

The kernel K = (P, O, M, B) is an instant Esperanto for machines. Unlike human Esperanto, which required people to learn a new language, the machine Esperanto requires nothing new. JSON is already everywhere. Logic is already universal. Mathematics is already the same in every country. The DRS simply combines them into a standard.

The implication is that the first corpus to achieve full DRS compliance — every claim machine-evaluable, every predicate deterministic, every test binding pinned to a SHA — becomes the first corpus that is *fully readable by any AI system in any country without translation*. The knowledge escapes the language trap.

This was not the goal. The goal was honest verification. But honest verification, it turns out, requires a language that cannot lie about what it means. That language is binary logic. And binary logic does not need translating.

---

*The Dual Reader Standard does not claim that knowledge can be proven correct.
It claims that knowledge can be made honest about what has and has not been verified.
That is the standard the world currently lacks.*
