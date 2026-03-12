# Grounded Verification Protocol (GVP) v1.0

> Portable specification. Paste into any Claude instance to establish the standard.
> Origin: Fracttalix corpus, Session 51. Schema v3-S51.
> Author: Thomas Brennan. AI collaborator: Claude (Anthropic).

---

## What this is

The **Dual Reader Standard (DRS)** has two protocols:

- **DRP** (Dual Reader Protocol) — the protocol for **text**. Defines how prose claims become machine-evaluable via the 5-part falsification predicate, AI layers, and phase gates. Paper P14/DRP-1.
- **GVP** (Grounded Verification Protocol) — the protocol for **software**. Defines how machine-evaluable claims become machine-verified via verification tiers, executable test bindings, and commit-pinned evidence.

The DRP makes claims *machine-evaluable*. The GVP makes them *machine-verified*.

The GVP requires every claim to declare:

1. **What kind of evidence grounds it** (verification tier)
2. **Which executable tests exercise it** (test bindings)
3. **At which commit those tests last passed** (verified-against SHA)

---

## The three readers

| Reader | Channel | Reads | Format |
|--------|---------|-------|--------|
| **Human** | Prose | The paper | Natural language |
| **AI** | JSON | The AI layer | Structured claim registry |
| **CI / test runner** | Executable | Test bindings | pytest node IDs + commit SHA |

The DRP defines Readers 1 and 2 (text). The GVP adds Reader 3 (software).

---

## The six verification tiers

Every claim in a GVP-compliant AI layer MUST carry a `tier` field with one of these values:

### Grounded by construction (no predicate needed)

| Tier | Applies to | Meaning |
|------|-----------|---------|
| `axiom` | Type A claims | Foundational. Unfalsifiable by design. The starting assumptions of the system. |
| `definition` | Type D claims | Definitional. No predicate because there is nothing to falsify — the claim *defines* a term or procedure. |

### Grounded now (evidence exists)

| Tier | Applies to | Meaning |
|------|-----------|---------|
| `software_tested` | Type F claims | Exercised by passing tests in this codebase. `test_bindings` array is non-empty. `verified_against` SHA is non-null. |
| `formal_proof` | Type F claims | Verified by a step-indexed derivation table with `n_invalid_steps = 0`. The proof is in the AI layer itself. |
| `analytic` | Type F claims | Verified by formal derivation trace, adversarial battery, or analytical argument. No software test, but the reasoning is documented. |

### Explicitly ungrounded

| Tier | Applies to | Meaning |
|------|-----------|---------|
| `empirical_pending` | Type F claims | Has an active placeholder or requires external data not yet collected. The gap is visible by design. |

---

## The three v3 fields

Every claim object in a GVP-compliant AI layer carries these fields (added in schema v3-S49):

```json
{
  "claim_id": "F-1.1",
  "type": "F",
  "statement": "...",
  "falsification_predicate": { ... },
  "tier": "software_tested",
  "test_bindings": [
    "tests/test_benchmark.py::TestBenchmarkEvaluate::test_point_anomaly_detection",
    "tests/test_benchmark.py::TestBenchmarkEvaluate::test_contextual_anomaly_detection"
  ],
  "verified_against": "95f59d8"
}
```

### Field semantics

| Field | Type | Required | Rule |
|-------|------|----------|------|
| `tier` | enum string | Yes | One of the six tiers above. |
| `test_bindings` | array of strings | Yes | Fully qualified pytest node IDs (`file::Class::method`). Empty array `[]` if no software tests. |
| `verified_against` | string or null | Yes | Git commit SHA (7–40 hex chars) at which `test_bindings` last passed. `null` if no software tests. |

### Consistency rules

1. `tier: "axiom"` → `type` MUST be `"A"`, `test_bindings` MUST be `[]`, `verified_against` MUST be `null`
2. `tier: "definition"` → `type` MUST be `"D"`, `test_bindings` MUST be `[]`, `verified_against` MUST be `null`
3. `tier: "software_tested"` → `type` MUST be `"F"`, `test_bindings` MUST be non-empty, `verified_against` MUST be non-null
4. `tier: "formal_proof"` → `type` MUST be `"F"`, `test_bindings` MAY be empty (proofs don't require software tests)
5. `tier: "analytic"` → `type` MUST be `"F"`, `test_bindings` MAY be empty
6. `tier: "empirical_pending"` → `type` MUST be `"F"`, claim SHOULD have `placeholder: true`

---

## The falsification predicate (DRS, unchanged)

Every Type F claim carries a 5-part falsification predicate:

```json
{
  "FALSIFIED_IF": "condition that would disprove the claim",
  "WHERE": { "variable": "definition" },
  "EVALUATION": "how to test it",
  "BOUNDARY": "scope limits",
  "CONTEXT": "what the claim assumes"
}
```

Type A and Type D claims have `falsification_predicate: null`.

The GVP does not change the predicate format. It adds the tier/binding/SHA fields *alongside* the predicate.

---

## Claim types

| Type | Name | Predicate | Tier options |
|------|------|-----------|-------------|
| **A** | Axiom | `null` | `axiom` only |
| **D** | Definition | `null` | `definition` only |
| **F** | Falsifiable | 5-part object | `software_tested`, `formal_proof`, `analytic`, `empirical_pending` |

---

## Schema (JSON Schema v3-S49)

The canonical schema lives at:
```
ai-layers/ai-layer-schema.json
```

Key additions in v3:

```json
"tier": {
  "type": "string",
  "enum": ["axiom", "definition", "software_tested", "analytic",
           "empirical_pending", "formal_proof"]
},
"test_bindings": {
  "type": "array",
  "items": {"type": "string"}
},
"verified_against": {
  "oneOf": [
    {"type": "string", "pattern": "^[0-9a-f]{7,40}$"},
    {"type": "null"}
  ]
}
```

---

## Inference rules (DRS, unchanged)

| ID | Name |
|----|------|
| IR-1 | Modus Ponens |
| IR-2 | Universal Instantiation |
| IR-3 | Substitution of Equals |
| IR-4 | Definition Expansion |
| IR-5 | Algebraic Manipulation |
| IR-6 | Logical Equivalence |
| IR-7 | Statistical Inference |
| IR-8 | Parsimony / Modeling Principle Selection |

---

## Verification pipeline

```
Claim written in prose (Reader 1)
        ↓
AI layer created with tier + predicate + bindings (Reader 2)
        ↓
pytest runs test_bindings at HEAD (Reader 3)
        ↓
If pass → verified_against = HEAD SHA
If fail → tier stays or downgrades to empirical_pending
        ↓
CI validates schema + cross-references on every push
```

---

## Current corpus statistics (Session 49)

```
Total claims:        80 (A:14  D:25  F:41)

Tier distribution:
  axiom:             14 claims
  definition:        25 claims
  software_tested:    7 claims  (385 test bindings)
  formal_proof:      18 claims  (step-indexed, n_invalid=0)
  analytic:           4 claims
  empirical_pending:  8 claims

verified_against:    95f59d8
Schema version:      v3-S49
AI layers:           15/15 passing validation
Cross-paper errors:  0
```

---

## How to implement GVP in a new paper

1. Write the paper (prose channel).
2. Create the AI layer JSON file with all claims.
3. For each claim, assign `type` (A/D/F).
4. For each Type F claim, write the 5-part `falsification_predicate`.
5. For each claim, assign the appropriate `tier`.
6. For `software_tested` claims, populate `test_bindings` with pytest node IDs.
7. Run the tests. Record the passing commit SHA in `verified_against`.
8. For `empirical_pending` claims, register the placeholder.
9. Validate against `ai-layer-schema.json`.

---

## Relationship to other standards

| Standard | What it does | GVP relationship |
|----------|-------------|-----------------|
| **DRS** (Dual Reader Standard) | The standard. Contains both protocols. | GVP is the software half of the DRS |
| **DRP** (Dual Reader Protocol) | The text protocol: predicates, AI layers, phase gates (P14/DRP-1) | GVP is the software protocol; DRP is the text protocol. Together they are the DRS. |
| **CBT** (Canonical Build Table) | Corpus architecture and scheduling | GVP tier data feeds CBT status |
| **KVS** (Knowledge Validation Score) | Corpus quality metric | GVP tier coverage is a KVS input |
| **CBP** (Canonical Build Process, P0) | Governance process | GVP operates under CBP governance |

---

## Origin

- **Corpus**: Fracttalix (21-object unified corpus on the Fractal Rhythm Model)
- **Repo**: github.com/thomasbrennan/Fracttalix
- **DOI**: 10.5281/zenodo.18859299
- **Schema**: ai-layers/ai-layer-schema.json
- **Licence**: CC0 public domain

---

*This specification is self-contained. Any AI instance receiving this document
has sufficient context to create, validate, and maintain GVP-compliant AI layers.*
