# Falsification Kernel v1.1 — Layer 0 Semantic Specification

**Corpus:** Fracttalix + Meta-Kaizen
**Scope:** All AI layers conforming to `ai-layer-schema.json` v2-S50+
**Governing standard:** Dual Reader Standard (DRS) v1.1
**Produced:** Session S50

---

## 1. Purpose

This document is **Layer 0** of the Dual Reader Standard. It defines what a
falsification predicate *means* independently of any serialisation format
(JSON, YAML, or future encodings). An AI layer's `semantic_spec_url` field
points here, making conformance machine-verifiable rather than merely asserted.

Every Type F (falsifiable) claim in the corpus carries a predicate that
evaluates deterministically to one of two verdicts: **FALSIFIED** or
**NOT FALSIFIED**. This kernel specifies the grammar, evaluation semantics,
and validity constraints for those predicates.

---

## 2. The Kernel 4-Tuple K = (P, O, M, B)

Each Type F claim's falsification predicate is an instance of the kernel
4-tuple:

| Symbol | Name | JSON field(s) | Role |
|--------|------|---------------|------|
| **P** | Predicate | `FALSIFIED_IF` | Logical sentence over named variables that, if TRUE, falsifies the claim. |
| **O** | Operands | `WHERE` | Dictionary defining every named variable in P: type, units, source, and derivation. |
| **M** | Mechanism | `EVALUATION` | Finite, deterministic procedure to compute the truth value of P. |
| **B** | Bounds | `BOUNDARY` + `CONTEXT` | Threshold semantics (inclusive/exclusive edge cases) and justification for every numeric threshold. |

A predicate is **well-formed** if and only if all four components are present
and satisfy the constraints in Sections 3–6 below.

---

## 3. P — Predicate (`FALSIFIED_IF`)

### 3.1 Grammar

The predicate is a logical sentence composed of:

- **Named variables** (defined in O)
- **Comparison operators:** `<`, `>`, `<=`, `>=`, `=`, `!=`
- **Logical connectives:** `AND`, `OR`, `NOT`
- **Quantifiers:** `EXISTS ... SUCH THAT`, `FOR ALL ... IN`
- **Parentheses** for grouping
- **Arithmetic operators** on numeric variables: `+`, `-`, `*`, `/`, `^`, `log10()`, `exp()`, `abs()`, `max()`, `min()`
- **Set operators:** `IN`, `∩`, `∪`, `|...|` (cardinality)
- **Function application:** `f(x)` where `f` is defined in O

### 3.2 Constraints

1. **Determinism.** P must evaluate to exactly one of `{TRUE, FALSE}` for any
   valid assignment of the variables in O. Predicates with undefined or
   ambiguous truth values are not well-formed.

2. **Finiteness.** Quantifiers range over finite sets. Unbounded universal
   quantifiers (`FOR ALL x IN R`) are not permitted.

3. **No self-reference.** P must not reference the truth value of the claim
   it belongs to, nor the truth value of any claim whose predicate
   references this claim (no circular dependency).

4. **Verdict mapping.**
   - P evaluates to TRUE → claim verdict is **FALSIFIED**
   - P evaluates to FALSE → claim verdict is **NOT FALSIFIED**

---

## 4. O — Operands (`WHERE`)

### 4.1 Variable specification

Each key in the `WHERE` object names a variable used in P. Its value is a
string with the following semicolon-delimited or `·`-delimited fields:

```
<type> · <units> · <definition or source>
```

| Field | Required | Examples |
|-------|----------|---------|
| **type** | Yes | `scalar`, `integer`, `binary`, `set`, `string`, `function` |
| **units** | Yes (use `dimensionless` if unitless) | `seconds`, `dimensionless`, `bits` |
| **definition** | Yes | `P3 Section 4 regression`, `count of substrates with R2 >= 0.85` |

### 4.2 Constraints

1. **Completeness.** Every free variable in P must appear in O.
2. **No orphans.** Every variable in O must appear in P.
3. **Groundable.** Each variable must be computable from data, parameters,
   or derivations cited in the corpus. Variables that require subjective
   judgment are not permitted in Type F predicates.

---

## 5. M — Mechanism (`EVALUATION`)

### 5.1 Requirements

The evaluation field specifies *how* to compute the truth value of P. It must
satisfy:

1. **Finite.** The procedure terminates in finite steps. The field should
   end with the word `finite` to confirm this.
2. **Deterministic.** The same inputs produce the same verdict.
3. **Reproducible.** A third party with access to the cited data and code
   can execute the procedure independently.

### 5.2 Format

A plain-English procedural description. Example:

> Run P3 regression for each substrate; compare R² values; finite

For computationally complex evaluations, the mechanism may reference a
specific script, test, or algorithm by corpus path.

---

## 6. B — Bounds (`BOUNDARY` + `CONTEXT`)

### 6.1 BOUNDARY

Specifies the threshold edge case explicitly:

- Whether the threshold is **inclusive** or **exclusive**
- What verdict applies at exact equality
- Example: `R2_best_alt = R2_frm + 0.05 → FALSIFIED (threshold inclusive)`

### 6.2 CONTEXT

Justifies every numeric threshold and design choice in P:

- Why this threshold value (not another)?
- What domain knowledge, prior art, or theoretical result grounds the choice?
- If a threshold is a modelling choice (not derived), state so explicitly.

### 6.3 Constraints

1. Every numeric literal in P must have a corresponding justification in
   CONTEXT.
2. Thresholds derived from theory must cite the derivation (paper, section,
   claim ID).
3. Thresholds chosen by convention must state the convention and its source.

---

## 7. Claim Types and Predicate Requirements

| Type | Name | Predicate required? | Notes |
|------|------|-------------------|-------|
| **F** | Falsifiable | Yes — full K = (P, O, M, B) | Core scientific claims |
| **D** | Definition | No (`null`) | Stipulative; not truth-apt |
| **A** | Axiom | No (`null`) | Accepted premises; cited from literature |

Type D and Type A claims carry `"falsification_predicate": null`. This is
correct — definitions and axioms are not falsifiable by design.

---

## 8. Vacuity Witness

Each Type F claim should include a `sample_falsification_observation` field:
a concrete, hypothetical observation that *would* make P evaluate to TRUE.
This serves as a vacuity check — proof that the predicate is not trivially
unfalsifiable.

A predicate that no conceivable observation could satisfy is **vacuously
true** and therefore not a valid Type F claim.

---

## 9. Placeholder Predicates

When a claim depends on results from a paper not yet at PHASE-READY status,
the predicate may contain placeholder references:

- `placeholder: true` in the claim object
- `placeholder_id` linking to the `placeholder_register`
- The predicate text may include `[PLACEHOLDER: pending ...]`

Placeholder claims are valid but **not evaluable** until the dependency
resolves. They do not block PHASE-READY status for the containing paper
unless `blocks_phase_ready: true`.

---

## 10. Validation Checklist

An AI layer conforms to this kernel if and only if:

- [ ] Every Type F claim has all five JSON fields: `FALSIFIED_IF`, `WHERE`,
  `EVALUATION`, `BOUNDARY`, `CONTEXT`
- [ ] Every variable in `FALSIFIED_IF` is defined in `WHERE`
- [ ] Every variable in `WHERE` appears in `FALSIFIED_IF`
- [ ] `EVALUATION` ends with `finite` (or equivalent finiteness assertion)
- [ ] `BOUNDARY` specifies inclusive/exclusive for every threshold
- [ ] `CONTEXT` justifies every numeric literal in `FALSIFIED_IF`
- [ ] `sample_falsification_observation` is non-empty for each Type F claim
- [ ] Type D and Type A claims have `falsification_predicate: null`
- [ ] No circular predicate references exist across claims

---

## 11. Versioning

| Version | Session | Changes |
|---------|---------|---------|
| v1.0 | S48 | Initial specification (implicit in schema) |
| v1.1 | S50 | Explicit Layer 0 document; `semantic_spec_url` added to schema; K = (P, O, M, B) formalized |

---

## 12. Licence

This specification is released under CC BY 4.0, consistent with the
Fracttalix corpus licence.

DOI: 10.5281/zenodo.18859299
