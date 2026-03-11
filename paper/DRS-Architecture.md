# The Dual Reader Standard: Architecture Specification

Thomas Brennan · with AI collaborator Claude (Anthropic)

March 2026

AI contributions: Claude (Anthropic) provided architectural formalisation, protocol naming (GVP), tier taxonomy design, and manuscript drafting. All contributions are contributed to the public domain.

---

## 1. What the Dual Reader Standard Is

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
        │  (Text Protocol)│            │(Software Protocol)│
        └────────┬────────┘            └─────────┬─────────┘
                 │                               │
     ┌───────────┴───────────┐       ┌───────────┴───────────┐
     │                       │       │                       │
  Reader 1              Reader 2   Reader 3A            Reader 3B
  (Human)               (AI)       (Coder)              (Machine)
     │                       │       │                       │
  reads prose          reads JSON   reads tier +         runs pytest
                                    bindings             stamps SHA
                 │                               │
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
| DRP | P14 / DRP-1 | PHASE-READY (AI layer deployed S47) |
| GVP | docs/GVP-spec.md | v1.0 (schema v3-S51) |
| Falsification Kernel | ai-layers/falsification-kernel.md | v1.1 (S50) |
| AI Layer Schema | ai-layers/ai-layer-schema.json | v3-S51 |
| DRS for Software | MK-P6 | DRAFT (NOT-PHASE-READY) |

### Current verification status

```
AI Layers:           18/18
Phase-Ready:          5/18 (P1, MK-P1, MK-P5, DRP-1, SFW-1)
Total Claims:       121 (A:25  D:39  F:57)

Verification Tiers:
  axiom:             25 claims
  definition:        39 claims
  software_tested:    7 claims (385 test bindings)
  formal_proof:      18 claims
  analytic:           4 claims
  empirical_pending:  8 claims

verified_against:    95f59d8
```

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

## 14. Licence

This specification is released under CC BY 4.0, consistent with the Fracttalix corpus licence.

**Corpus:** Fracttalix (22-object unified corpus)
**DOI:** 10.5281/zenodo.18859299
**Repository:** github.com/thomasbrennan/Fracttalix

---

## 15. Derivation from the Fractal Rhythm Model

The DRS is not merely compatible with the FRM. It is an instance of it.

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

## 16. Future-Proofing

### 16.1 The kernel is the invariant

The Falsification Kernel K = (P, O, M, B) is substrate-independent. It does not know whether it is evaluating a scientific theorem, a software behavioral guarantee, a legal contract, or a regulatory requirement. This is by design.

Future-proofing strategy: **never modify the kernel**. Add new protocols to the DRS instead.

### 16.2 Protocol extensibility

The DRS currently has two protocols (DRP for text, GVP for software). Future domains can add their own:

| Domain | Potential protocol | Reader A | Reader B |
|--------|-------------------|----------|----------|
| Legal | Legal Verification Protocol (LVP) | Lawyer | Compliance engine |
| Regulatory | Regulatory Verification Protocol (RVP) | Auditor | Regulatory database |
| Policy | Policy Verification Protocol (PVP) | Analyst | Impact model |
| Education | Educational Verification Protocol (EVP) | Student | Assessment engine |

Each protocol would define its own readers, its own tier taxonomy, and its own binding mechanism — but all would share the kernel and the AI layer schema.

### 16.3 Schema versioning

The AI layer schema is versioned (`v3-S51`). Every AI layer records which schema version it was produced against. This means:

- Old layers remain valid against old schemas
- New fields can be added without breaking existing layers
- Validators can be version-aware
- Migration paths are defined by the version history

### 16.4 Serialisation independence

The Falsification Kernel (Layer 0) is defined independently of JSON. The current implementation uses JSON, but the kernel's semantics are specified in prose (`falsification-kernel.md`). A future implementation could use YAML, Protocol Buffers, or any other serialisation format without changing the kernel.

The `semantic_spec_url` field in every AI layer points to the kernel specification. This decouples evaluation semantics from serialisation format.

### 16.5 Inference rule extensibility

The inference rule inventory (IR-1 through IR-8) is a living list. New rules can be added as new derivation patterns emerge. Each rule has an ID, a name, and a description. Derivation traces cite rules by ID, so adding a new rule does not invalidate existing traces.

### 16.6 Tier extensibility

The six verification tiers are defined in the schema's `enum` array. If a future domain requires a new tier (e.g., `regulatory_certified` for claims verified by a regulatory body), the enum can be extended. Existing tiers remain valid. The tier taxonomy is designed to be extended, not replaced.

### 16.7 The versioning contract

The DRS makes one promise about future versions: **no breaking changes to the kernel**. The 4-tuple K = (P, O, M, B) and the three claim types (A, D, F) are permanent. Everything else — protocols, tiers, schema fields, inference rules — can be added to but not removed.

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

*The Dual Reader Standard does not claim that knowledge can be proven correct.
It claims that knowledge can be made honest about what has and has not been verified.
That is the standard the world currently lacks.*
