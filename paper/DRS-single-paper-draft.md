# The Dual Reader Standard: A Machine-Readable Annotation Layer for Falsifiable Claims

**Thomas Brennan**

**Draft — March 2026**

---

## Abstract

Scientific papers are written for humans. This means that an AI system
confronting a research corpus cannot determine, without reading and
interpreting prose, what a paper claims, what evidence supports each claim,
or what observation would refute it. We propose the Dual Reader Standard
(DRS): a lightweight JSON annotation layer that makes every falsifiable
claim in a document machine-evaluable. Each claim is encoded as a
falsification predicate — a 4-tuple K = (P, O, M, B) specifying the
logical condition under which the claim is falsified, the operands
involved, the evaluation procedure, and the boundary semantics. We define
a JSON schema, a semantic specification (Layer 0), and a governance gate
(PHASE-READY) for ensuring annotation quality. We demonstrate the format
on claims drawn from three domains: dynamical systems modelling,
organisational theory, and software engineering. The DRS is domain-agnostic,
format-agnostic at Layer 0, and released under CC BY 4.0.

---

## 1. The Problem

A scientific paper makes claims. Some are definitions. Some are accepted
premises. Some are falsifiable — they assert something about the world
that could, in principle, be shown wrong.

A human reader can (usually) identify which claims are which, what evidence
backs them, and what would break them. A machine cannot. The prose is
ambiguous, the claims are scattered across sections, and the falsification
conditions — if stated at all — are buried in natural language.

This matters now because AI systems are increasingly used to evaluate,
summarise, and build on research. If the machine cannot determine what a
paper actually claims, it cannot verify those claims, detect conflicts
between papers, or identify gaps. It can only summarise prose, which is
a different — and much less useful — task.

The Dual Reader Standard addresses this by requiring every falsifiable
claim in a document to carry a structured, machine-evaluable annotation.
The "dual reader" is literal: one channel (prose) serves humans; one
channel (the AI layer) serves machines. Both describe the same claims.
Neither is optional.

---

## 2. The Falsification Kernel

### 2.1 The 4-Tuple

Every falsifiable claim is annotated with a kernel **K = (P, O, M, B)**:

| Symbol | Name | Role |
|--------|------|------|
| **P** | Predicate | Logical sentence: if TRUE, the claim is FALSIFIED |
| **O** | Operands | Dictionary defining every variable in P: type, units, source |
| **M** | Mechanism | Finite, deterministic procedure to compute P's truth value |
| **B** | Bounds | Threshold semantics (inclusive/exclusive) and justification |

A predicate is **well-formed** if and only if:

1. P evaluates deterministically to {TRUE, FALSE}
2. All quantifiers range over finite sets
3. Every variable in P is defined in O, and every variable in O appears in P
4. M terminates in finite steps
5. B justifies every numeric threshold in P
6. No circular predicate references exist

### 2.2 Claim Types

| Type | Name | Predicate? | Purpose |
|------|------|-----------|---------|
| **F** | Falsifiable | Required | Claims about the world |
| **D** | Definition | Null | Stipulative terms |
| **A** | Axiom | Null | Accepted premises (cited) |

Only Type F claims carry predicates. Definitions and axioms are not
truth-apt in the falsificationist sense; they are structural.

### 2.3 Vacuity Witness

Each Type F claim includes a `sample_falsification_observation`: a
concrete, hypothetical observation that *would* trigger FALSIFIED. This
proves the predicate is not trivially unfalsifiable. A predicate that no
conceivable observation could satisfy is vacuously true and therefore
not a valid Type F claim.

---

## 3. The JSON Format

The DRS serialises the kernel as a JSON object conforming to a
JSON Schema (draft-07). The five fields map directly to the 4-tuple,
with B split into two fields for clarity:

```json
{
  "claim_id": "C-1",
  "type": "F",
  "label": "Short descriptive name",
  "statement": "The claim in natural language.",
  "falsification_predicate": {
    "FALSIFIED_IF": "n_failures > 0",
    "WHERE": {
      "n_failures": "count of X satisfying condition Y"
    },
    "EVALUATION": "Step 1; Step 2; ... Finite.",
    "BOUNDARY": "n_failures = 0 -> NOT FALSIFIED (inclusive)",
    "CONTEXT": "Justification for threshold and design choices"
  },
  "sample_falsification_observation": "A concrete example that would trigger FALSIFIED.",
  "tier": "software_tested",
  "test_bindings": ["tests/test_example.py::test_claim_1"],
  "verified_against": "abc1234"
}
```

### 3.1 Verification Tiers

Claims are graded by how they have been verified:

| Tier | Meaning |
|------|---------|
| `axiom` | Type A — accepted, not tested |
| `definition` | Type D — stipulative |
| `software_tested` | Type F with passing automated tests |
| `analytic` | Type F verified by formal derivation |
| `empirical_pending` | Type F awaiting external data |
| `formal_proof` | Type F verified by step-indexed derivation |

### 3.2 Test Bindings

A claim at the `software_tested` tier includes `test_bindings`: an array
of fully qualified test node IDs (e.g. pytest paths) that exercise the
claim. The `verified_against` field records the git SHA at which those
tests last passed. This closes the loop between annotation and code.

---

## 4. The AI Layer Document

An AI layer is a single JSON file that collects all claims from one
document. Its structure:

```json
{
  "_meta": {
    "document_type": "AI_LAYER",
    "schema_version": "v3",
    "semantic_spec_url": "falsification-kernel.md",
    "licence": "CC BY 4.0"
  },
  "paper_id": "PAPER-1",
  "paper_title": "Title of the paper",
  "paper_type": "law_A | derivation_B | application_C | methodology_D",
  "version": "v1",
  "phase_ready": {
    "verdict": "PHASE-READY",
    "c1": "SATISFIED",
    "c2": "SATISFIED",
    "c3": "SATISFIED",
    "c5": "SATISFIED",
    "c6": "SATISFIED",
    "placeholder_count": 0
  },
  "claim_registry": [ ... ],
  "placeholder_register": [ ... ]
}
```

### 4.1 Dependency Tracking

Claims can reference claims in other AI layers. Inbound edges record
axioms received from prior work. Outbound edges record claims passed
forward. Placeholders mark unresolved dependencies. This creates a
typed, auditable dependency graph across a corpus.

---

## 5. The PHASE-READY Gate

An AI layer achieves PHASE-READY status when it satisfies six criteria:

| Gate | Requirement |
|------|-------------|
| C1 | Schema valid — JSON validates against `ai-layer-schema.json` |
| C2 | Complete — every falsifiable claim in the prose is registered |
| C3 | Machine-evaluable — every predicate can be computed without human judgment |
| C4 | Dependencies tracked — all inter-paper references in placeholder register |
| C5 | Self-sufficient — an AI can verify any claim using only the layer |
| C6 | Non-vacuous — every predicate has a coherent falsification witness |

C4 is tracked but non-blocking (dependencies may be open while the
paper itself is ready). All other criteria block.

The gate is the enforcement mechanism. Without it, the annotation layer
becomes optional, and optional metadata is omitted metadata. This is an
empirical prediction (not a philosophical claim): in any corpus where DRS
annotation is optional, the majority of papers will ship without it.

---

## 6. Worked Examples

### 6.1 Dynamical Systems (Fractal Rhythm Model)

**Claim:** The FRM functional form fits oscillatory data from biological,
physical, and social substrates with R² ≥ 0.85 across ≥ 3 substrates
spanning ≥ 3 orders of magnitude in timescale.

```json
{
  "claim_id": "C-1.1",
  "type": "F",
  "label": "Cross-substrate fit",
  "statement": "FRM achieves R² ≥ 0.85 on ≥ 3 substrates spanning ≥ 3 OoM.",
  "falsification_predicate": {
    "FALSIFIED_IF": "n_substrates < 3 OR oom_span < 3",
    "WHERE": {
      "n_substrates": "count of substrates with R² ≥ 0.85 for FRM fit",
      "oom_span": "log10(max_timescale / min_timescale) across qualifying substrates"
    },
    "EVALUATION": "For each substrate: fit FRM model, record R². Count substrates with R² ≥ 0.85. Compute timescale span. Finite.",
    "BOUNDARY": "n_substrates = 3, oom_span = 3 -> NOT FALSIFIED (inclusive)",
    "CONTEXT": "R² ≥ 0.85 is a standard goodness-of-fit threshold. 3 substrates and 3 OoM chosen to demonstrate universality beyond a single domain."
  },
  "sample_falsification_observation": "FRM achieves R² ≥ 0.85 on only 2 substrates."
}
```

### 6.2 Organisational Theory (Meta-Kaizen)

**Claim:** The KVS (Kaizen Value Score) formula is unique up to monotone
transformation given the stated axioms.

```json
{
  "claim_id": "F-MK1.1",
  "type": "F",
  "label": "KVS uniqueness",
  "statement": "KVS_j = N_j × I'_j × C'_j × T_j is the unique multiplicative form satisfying Axioms 1–6.",
  "falsification_predicate": {
    "FALSIFIED_IF": "n_alt > 0",
    "WHERE": {
      "n_alt": "count of alternative multiplicative forms satisfying Axioms 1–6 that are not monotone transformations of KVS"
    },
    "EVALUATION": "Enumerate candidate forms from the axiom constraints. For each: check Axiom 1–6 compliance. If compliant and not a monotone transformation of KVS, increment n_alt. Finite.",
    "BOUNDARY": "n_alt = 0 -> NOT FALSIFIED (inclusive)",
    "CONTEXT": "Uniqueness claim from conjoint measurement theory. Axioms 1–6 cited from Krantz et al. (1971)."
  },
  "sample_falsification_observation": "An alternative form K = N × I' × (C' + T) satisfies all six axioms."
}
```

### 6.3 Software Engineering

**Claim:** The schema validator correctly rejects all malformed AI layers.

```json
{
  "claim_id": "C-SFW.1",
  "type": "F",
  "label": "Validator completeness",
  "statement": "The JSON schema validator rejects any AI layer with a missing or malformed falsification predicate field.",
  "falsification_predicate": {
    "FALSIFIED_IF": "n_false_accepts > 0",
    "WHERE": {
      "n_false_accepts": "count of malformed AI layer inputs accepted by the schema validator"
    },
    "EVALUATION": "Generate test corpus of malformed inputs (missing FALSIFIED_IF, empty WHERE, non-terminating EVALUATION, etc.). Run each through jsonschema.validate(). Count acceptances. Finite.",
    "BOUNDARY": "n_false_accepts = 0 -> NOT FALSIFIED (inclusive)",
    "CONTEXT": "Standard validator completeness criterion. Test corpus must include at least one violation per required field."
  },
  "sample_falsification_observation": "An AI layer with FALSIFIED_IF omitted passes validation.",
  "tier": "software_tested",
  "test_bindings": ["tests/test_schema_validation.py::test_malformed_rejected"],
  "verified_against": "a1b2c3d"
}
```

---

## 7. Machine Evaluation

An AI system consuming a DRS corpus operates as follows:

1. **Parse** the AI layer JSON
2. **Select** a claim by ID
3. **Read** the predicate: FALSIFIED_IF, WHERE, EVALUATION
4. **Execute** the evaluation procedure (or verify it was executed, using test_bindings and verified_against)
5. **Return** FALSIFIED or NOT FALSIFIED

The AI never needs to read the prose. The layer is self-sufficient (C5).
If the claim has `test_bindings`, the AI can re-run the tests at the
recorded SHA and confirm the result independently.

This is the core value proposition: structured claims are verifiable;
prose claims require interpretation.

---

## 8. Scope and Limitations

**What DRS is:**
- A metadata format for annotating documents with machine-evaluable claims
- A governance gate for ensuring annotation quality
- Domain-agnostic — works for any field that makes falsifiable claims

**What DRS is not:**
- A theory of epistemology
- A replacement for peer review
- A guarantee that claims are correct (only that they are testable)

**Known limitations:**
- The format handles falsifiable claims well but says nothing about
  claims that are inherently non-falsifiable (ethical, aesthetic, etc.)
- EVALUATION procedures may reference external data or code that
  becomes unavailable over time
- The PHASE-READY gate requires disciplined authorship; adoption
  depends on tooling and incentives, not just specification quality

---

## 9. Related Work

| Approach | Relationship to DRS |
|----------|-------------------|
| **Semantic Web / RDF** | Encodes relationships between entities, not falsification conditions. DRS is narrower in scope but deeper on testability. |
| **Nanopublications** | Structured scientific assertions with provenance. Closest existing work. DRS adds the falsification predicate and evaluation procedure, which nanopublications lack. |
| **FAIR principles** | Findable, Accessible, Interoperable, Reusable. DRS satisfies FAIR and adds a dimension: *testable*. |
| **Registered Reports** | Pre-registered hypotheses and methods. DRS formalises what registered reports describe in prose. |
| **CML / SciML** | Domain-specific markup for chemistry, science. DRS is domain-agnostic and claim-centric rather than data-centric. |

---

## 10. Conclusion

The DRS is a simple idea: if a paper makes a falsifiable claim, write
down what would break it in a format a machine can read. The 4-tuple
K = (P, O, M, B) is the minimal structure needed. The JSON schema is
the serialisation. The PHASE-READY gate is the enforcement. Everything
else — the philosophical justification, the information-theoretic
analysis, the domain-specific applications — is downstream work that
may or may not be needed, but is not required to use the format.

The specification, schema, and reference implementation are available at
[github.com/thomasbrennan/Fracttalix](https://github.com/thomasbrennan/Fracttalix)
under CC BY 4.0.

---

## Appendix A: Schema Reference

The full JSON Schema is maintained at `ai-layers/ai-layer-schema.json`.
The semantic specification (Layer 0) is at `ai-layers/falsification-kernel.md`.

## Appendix B: Inference Rules

For corpora that include formal derivation traces, DRS defines a
canonical set of inference rules:

| ID | Name | Description |
|----|------|-------------|
| IR-1 | Modus Ponens | If P and P→Q then Q |
| IR-2 | Universal Instantiation | If ∀x P(x) then P(a) |
| IR-3 | Substitution of Equals | If a=b then replace a with b |
| IR-4 | Definition Expansion | Replace term with definition |
| IR-5 | Algebraic Manipulation | Valid transformation preserving equality |
| IR-6 | Logical Equivalence | Replace with logically equivalent expression |
| IR-7 | Statistical Inference | Named statistical procedure on data |
| IR-8 | Parsimony Selection | Canonical choice from axiom-consistent family |

These are optional. A corpus using DRS for annotation only (no formal
proofs) does not need them.
