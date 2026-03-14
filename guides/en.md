# How to Implement the Dual Reader Standard

**A complete guide to building machine-verifiable claims for scientific papers, software, and knowledge systems.**

> See also: [en.json](en.json) — JSON-LD structured data version of this guide for AI systems and web crawlers.

---

## What Is the Dual Reader Standard?

The **Dual Reader Standard (DRS)** is a verification architecture for knowledge systems. Every claim — whether written in prose or implemented in code — must be readable by two independent reader classes: **human** and **machine**.

The DRS comprises two protocols:

| Protocol | Domain | Function |
|----------|--------|----------|
| **DRP** (Dual Reader Protocol) | Text / Papers | Makes prose claims machine-evaluable via 5-part falsification predicates |
| **GVP** (Grounded Verification Protocol) | Software / Code | Makes machine-evaluable claims machine-verified via test bindings and commit-pinned evidence |

### The Three Readers

| Reader | Channel | Reads | Format |
|--------|---------|-------|--------|
| **Human** | Prose | The paper | Natural language |
| **AI** | JSON | The AI layer | Structured claim registry |
| **CI / test runner** | Executable | Test bindings | Test node IDs + commit SHA |

---

## The Falsification Kernel K = (P, O, M, B)

The shared foundation of both protocols. Every Type F (falsifiable) claim carries a deterministic predicate that evaluates to exactly one of two verdicts: **FALSIFIED** or **NOT FALSIFIED**.

| Symbol | Name | JSON Field | Role |
|--------|------|-----------|------|
| **P** | Predicate | `FALSIFIED_IF` | Logical sentence that, if TRUE, falsifies the claim |
| **O** | Operands | `WHERE` | Typed definitions of every variable in the predicate |
| **M** | Mechanism | `EVALUATION` | Finite, deterministic evaluation procedure |
| **B** | Bounds | `BOUNDARY` + `CONTEXT` | Threshold semantics and justification |

### Example Predicate

```json
{
  "FALSIFIED_IF": "EXISTS i IN range(len(result)-1) SUCH THAT result[i] > result[i+1]",
  "WHERE": {
    "result": "list · dimensionless · output of sort(input)"
  },
  "EVALUATION": "Run sort on test vectors; check adjacent pairs; finite",
  "BOUNDARY": "len(result) <= 1 → NOT FALSIFIED (vacuously sorted)",
  "CONTEXT": "Ascending order is the documented contract of sort()"
}
```

This predicate means the same thing to every AI system in every language. No translation required.

### Predicate Constraints

- **Determinism**: Must evaluate to exactly TRUE or FALSE
- **Finiteness**: Quantifiers range over finite sets only
- **No self-reference**: No circular predicate dependencies
- **Completeness**: Every variable in `FALSIFIED_IF` defined in `WHERE`, and vice versa

---

## The Three Claim Types

| Type | Name | Description | Predicate Required? |
|------|------|-------------|-------------------|
| **A** | Axiom | Foundational premises — unfalsifiable by design | No (`null`) |
| **D** | Definition | Stipulative definitions — not truth-apt | No (`null`) |
| **F** | Falsifiable | Testable claims with deterministic predicates | Yes — full K = (P, O, M, B) |

---

## The Six Verification Tiers

Every claim carries a `tier` field declaring what kind of evidence grounds it.

### Grounded by Construction

| Tier | Applies To | Meaning |
|------|-----------|---------|
| `axiom` | Type A | Foundational, unfalsifiable by design |
| `definition` | Type D | Definitional, no predicate needed |

### Grounded Now

| Tier | Applies To | Meaning |
|------|-----------|---------|
| `software_tested` | Type F | Exercised by passing tests. `test_bindings` non-empty, `verified_against` SHA non-null |
| `formal_proof` | Type F | Step-indexed derivation with `n_invalid_steps = 0` |
| `analytic` | Type F | Verified by formal derivation trace or analytical argument |

### Explicitly Ungrounded

| Tier | Applies To | Meaning |
|------|-----------|---------|
| `empirical_pending` | Type F | Active placeholder or pending data. Gap is visible by design |

---

## The AI Layer — The Central Artifact

The AI layer is a JSON document that accompanies every paper or software system. It is what both protocols operate on.

### Required Sections

| Section | Purpose |
|---------|---------|
| `_meta` | Document type, schema version, session, licence |
| `paper_id` | Unique identifier |
| `paper_title` | Human-readable title |
| `paper_type` | `law_A`, `derivation_B`, `application_C`, or `methodology_D` |
| `phase_ready` | Phase gate verdict and condition status (c1–c6) |
| `claim_registry` | Array of all claims with types, predicates, tiers, bindings |
| `placeholder_register` | Array of unresolved dependencies |

### Schema

The AI layer schema is available at: [`ai-layers/ai-layer-schema.json`](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json)

---

## The Six-Condition Phase Gate

A paper or software release is **PHASE-READY** when all six conditions are satisfied:

| Condition | Requirement |
|-----------|-------------|
| **c1** | AI layer is schema-valid |
| **c2** | All falsifiable claims registered with predicates |
| **c3** | All predicates are machine-evaluable |
| **c4** | Cross-references tracked (placeholder register) |
| **c5** | Verification is self-sufficient (AI layer alone, no prose needed) |
| **c6** | All predicates are non-vacuous (sample falsification observation exists) |

---

## How to Implement

### For a Paper (DRP)

1. Write the paper (prose channel for human readers)
2. Create the AI layer JSON file (machine channel for AI readers)
3. Classify every claim as A (axiom), D (definition), or F (falsifiable)
4. Write the 5-part falsification predicate for every Type F claim
5. Include a `sample_falsification_observation` for each Type F claim (vacuity witness)
6. Assign the verification tier to each claim
7. Validate against `ai-layer-schema.json`
8. Run the phase gate (c1–c6)

### For Software (GVP)

1. Enumerate what the software claims to do
2. Classify each claim as A (assumption), D (definition), or F (behavioral)
3. Write the falsification predicate for every Type F claim
4. Write or identify the tests that exercise each claim
5. Populate `test_bindings` with fully qualified test node IDs
6. Run the tests and record the passing commit SHA in `verified_against`
7. Assign the tier: `software_tested` if tests exist, `empirical_pending` if not yet
8. Register any untested claims as placeholders
9. Validate against `ai-layer-schema.json`

### Minimum Viable Adoption

The smallest useful DRS adoption is **one Type F claim with one test binding**:

```json
{
  "claim_id": "F-1",
  "type": "F",
  "statement": "sort() returns elements in ascending order",
  "falsification_predicate": {
    "FALSIFIED_IF": "EXISTS i IN range(len(result)-1) SUCH THAT result[i] > result[i+1]",
    "WHERE": {
      "result": "list · dimensionless · output of sort(input)"
    },
    "EVALUATION": "Run sort on test vectors; check adjacent pairs; finite",
    "BOUNDARY": "len(result) <= 1 → NOT FALSIFIED (vacuously sorted)",
    "CONTEXT": "Ascending order is the documented contract of sort()"
  },
  "tier": "software_tested",
  "test_bindings": ["tests/test_sort.py::test_ascending_order"],
  "verified_against": "abc1234"
}
```

One claim. One test. One SHA. The DRS is live. Add more claims when the value justifies the cost.

---

## Design Principles

**Popperian epistemology.** We can falsify but not verify. A claim that survives all falsification attempts is not proven — it has survived.

**Honest gaps.** The placeholder is the most important feature. When a claim is `empirical_pending`, the system says: "we claim this but have not yet verified it." This is strictly more informative than the alternative, where unverified claims are indistinguishable from verified ones.

**Substrate independence.** The kernel K = (P, O, M, B) does not know whether it evaluates a scientific theorem or a software guarantee. Future domains (legal, regulatory, policy) can add their own protocols without modifying the kernel.

**Machine lingua franca.** The kernel is written in logic and mathematics, not any human language. A predicate that evaluates to TRUE in Beijing evaluates to TRUE in Boston. JSON is the transport layer. Binary logic is the substrate.

---

## Three-Axis Compatibility

| Axis | Promise | Mechanism |
|------|---------|-----------|
| **Past** | Nothing already done breaks | Schema versioning, append-only enums, permanent kernel |
| **Lateral** | Works across all domains, languages, tools, AI systems | Substrate-independent kernel, string-typed bindings |
| **Forward** | Anything new can be added without redesign | Protocol extensibility, tier extensibility, additive schema evolution |

---

## Resources

- [DRS Architecture Specification](https://github.com/thomasbrennan/Fracttalix/blob/main/paper/DRS-Architecture.md) — Full specification
- [Falsification Kernel v1.1](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/falsification-kernel.md) — Layer 0 semantic spec
- [AI Layer Schema v3](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json) — JSON Schema
- [GVP Specification](https://github.com/thomasbrennan/Fracttalix/blob/main/docs/GVP-spec.md) — Software protocol
- [Example AI Layer (P1)](https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json) — Working example
- [Fracttalix Repository](https://github.com/thomasbrennan/Fracttalix)

---

**License:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Author:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
