# The Dual Reader Standard — Complete Guide

**A one-pass guide to the entire Dual Reader Standard: architecture, protocols, kernel, layers, compatibility, lingua franca, and adoption.**

> See also: [en.json](en.json) — JSON-LD structured data version of this guide for AI systems and web crawlers.

---

## 1. What the Dual Reader Standard Is

The **Dual Reader Standard (DRS)** is a verification architecture for knowledge systems. It requires every claim — whether written in prose or implemented in code — to be readable by two independent reader classes: **human** and **machine**.

The DRS is not a paper. It is not a tool. It is the standard that contains both of its protocols:

- **DRP** (Dual Reader Protocol) — the protocol for text
- **GVP** (Grounded Verification Protocol) — the protocol for software

The DRP makes claims *machine-evaluable*. The GVP makes them *machine-verified*. Neither is complete without the other. Together they are the DRS.

---

## 2. The Two Protocols

### 2.1 DRP — Dual Reader Protocol (Text)

The DRP governs how prose claims become machine-evaluable.

**Reader 1 (Human):** Reads the paper in natural language. Understands context, motivation, and narrative. Cannot systematically audit every claim.

**Reader 2 (AI):** Reads the AI layer — a structured JSON document that accompanies every paper. Contains the full claim registry. Can audit every claim without reading prose.

The DRP requires:

1. **Claim classification.** Every claim is typed as A (axiom), D (definition), or F (falsifiable).
2. **Falsification predicates.** Every Type F claim carries a 5-part deterministic predicate.
3. **Phase gates.** Six conditions (c1–c6) that must be satisfied before a paper is declared PHASE-READY.
4. **Placeholder tracking.** Claims that depend on unresolved results are registered as placeholders — making gaps visible rather than invisible.

**What the DRP guarantees:** Any AI system with access to the AI layer can evaluate any falsifiable claim without reading the prose. Self-sufficiency is a design requirement enforced at the phase gate (condition c5).

### 2.2 GVP — Grounded Verification Protocol (Software)

The GVP governs how machine-evaluable claims become machine-verified.

**Reader 3A (Coder):** Reads the `tier` field to understand what kind of evidence exists. Reads `test_bindings` to know which tests exercise which claims. Reads `verified_against` to know when those tests last passed.

**Reader 3B (Machine):** Runs the test runner against the `test_bindings` array. Records pass/fail. Stamps the `verified_against` SHA on success.

The GVP requires every claim to carry three fields:

1. **`tier`** — the verification tier (one of six values)
2. **`test_bindings`** — an array of fully qualified test node IDs that exercise the claim
3. **`verified_against`** — the git commit SHA at which those tests last passed

**What the GVP guarantees:** For any claim in the corpus, you can determine (a) what kind of evidence grounds it, (b) which executable tests exercise it, and (c) at which commit those tests last passed. If the answer to (a) is `empirical_pending`, the gap is visible. If the answer to (c) is `null`, no software test covers it.

---

## 3. The Shared Foundation — Falsification Kernel (Layer 0)

Both protocols operate on the same foundation: the **Falsification Kernel K = (P, O, M, B)**.

This is **Layer 0** of the DRS. It defines what a falsification predicate *means* independently of any serialisation format (JSON, YAML, or future encodings). An AI layer's `semantic_spec_url` field points to this specification, making conformance machine-verifiable rather than merely asserted.

| Symbol | Name | JSON Field(s) | Role |
|--------|------|---------------|------|
| **P** | Predicate | `FALSIFIED_IF` | Logical sentence that, if TRUE, falsifies the claim |
| **O** | Operands | `WHERE` | Typed definitions of every variable in P |
| **M** | Mechanism | `EVALUATION` | Finite, deterministic evaluation procedure |
| **B** | Bounds | `BOUNDARY` + `CONTEXT` | Threshold semantics and justification |

The DRP creates the kernel (assigns predicates to prose claims). The GVP binds the kernel to executable evidence (links predicates to tests and commits). The kernel is substrate-independent — it works for scientific papers, software, and any future domain that makes falsifiable claims.

### 3.1 Predicate Grammar (`FALSIFIED_IF`)

The predicate is a logical sentence composed of:

- **Named variables** (defined in `WHERE`)
- **Comparison operators:** `<`, `>`, `<=`, `>=`, `=`, `!=`
- **Logical connectives:** `AND`, `OR`, `NOT`
- **Quantifiers:** `EXISTS ... SUCH THAT`, `FOR ALL ... IN`
- **Arithmetic operators:** `+`, `-`, `*`, `/`, `^`, `log10()`, `exp()`, `abs()`, `max()`, `min()`
- **Set operators:** `IN`, `∩`, `∪`, `|...|` (cardinality)
- **Function application:** `f(x)` where `f` is defined in `WHERE`

### 3.2 Predicate Constraints

1. **Determinism.** Must evaluate to exactly `TRUE` or `FALSE` for any valid assignment of variables.
2. **Finiteness.** Quantifiers range over finite sets only. Unbounded universal quantifiers are not permitted.
3. **No self-reference.** Must not reference the truth value of its own claim or create circular dependencies.
4. **Completeness.** Every variable in `FALSIFIED_IF` must be defined in `WHERE`, and every variable in `WHERE` must appear in `FALSIFIED_IF`.

**Verdict mapping:**
- P evaluates to TRUE → claim verdict is **FALSIFIED**
- P evaluates to FALSE → claim verdict is **NOT FALSIFIED**

### 3.3 Operands (`WHERE`)

Each key in the `WHERE` object names a variable. Its value is a string with the format:

```
<type> · <units> · <definition or source>
```

| Field | Required | Examples |
|-------|----------|---------|
| **type** | Yes | `scalar`, `integer`, `binary`, `set`, `string`, `function` |
| **units** | Yes (use `dimensionless` if unitless) | `seconds`, `dimensionless`, `bits` |
| **definition** | Yes | `output of sort(input)`, `count of substrates with R² >= 0.85` |

Constraints: every free variable in P must appear in O (completeness), every variable in O must appear in P (no orphans), and each variable must be computable from data or derivations — not subjective judgement.

### 3.4 Mechanism (`EVALUATION`)

The evaluation field specifies *how* to compute the truth value of P:

1. **Finite.** The procedure terminates in finite steps (conventionally confirmed by ending with the word `finite`).
2. **Deterministic.** Same inputs produce the same verdict.
3. **Reproducible.** A third party with access to cited data and code can execute the procedure independently.

### 3.5 Bounds (`BOUNDARY` + `CONTEXT`)

**BOUNDARY** specifies threshold edge cases: whether thresholds are inclusive or exclusive, and what verdict applies at exact equality.

**CONTEXT** justifies every numeric threshold and design choice in the predicate: why this value, what domain knowledge grounds it, and whether it is derived or conventional.

### 3.6 Example Predicate

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

---

## 4. The Three Claim Types

| Type | Name | In Papers | In Software | Predicate |
|------|------|-----------|-------------|-----------|
| **A** | Axiom / Assumption | Premises accepted from literature | Platform requirements, dependency contracts | `null` |
| **D** | Definition | Stipulative terms and procedures | Type signatures, data structures, schemas | `null` |
| **F** | Falsifiable | Theorems, empirical predictions | Behavioral guarantees, correctness invariants | Full K = (P, O, M, B) |

Type D and Type A claims carry `"falsification_predicate": null`. This is correct — definitions and axioms are not falsifiable by design.

---

## 5. The Six Verification Tiers

Every claim carries a `tier` field declaring what kind of evidence grounds it.

### Grounded by Construction

| Tier | Type | Meaning |
|------|------|---------|
| `axiom` | A | Foundational premise. Unfalsifiable by design. |
| `definition` | D | Definitional. Stipulates a term or structure. Not truth-apt. |

### Grounded Now

| Tier | Type | Meaning |
|------|------|---------|
| `software_tested` | F | Exercised by passing tests. `test_bindings` non-empty, `verified_against` non-null. |
| `formal_proof` | F | Step-indexed derivation with `n_invalid_steps = 0`. Proof is in the AI layer. |
| `analytic` | F | Verified by formal derivation trace or analytical argument. |

### Explicitly Ungrounded

| Tier | Type | Meaning |
|------|------|---------|
| `empirical_pending` | F | Active placeholder or pending data. Gap is visible by design. |

### Consistency Rules

The tier must be consistent with the claim type and the GVP fields:

| Tier | Required Type | test_bindings | verified_against |
|------|--------------|---------------|-----------------|
| `axiom` | A | `[]` (empty) | `null` |
| `definition` | D | `[]` (empty) | `null` |
| `software_tested` | F | Non-empty | Non-null (7–40 hex chars) |
| `formal_proof` | F | May be empty | May be null |
| `analytic` | F | May be empty | May be null |
| `empirical_pending` | F | May be empty | May be null |

---

## 6. The Vacuity Witness

Each Type F claim must include a `sample_falsification_observation` field: a concrete, hypothetical observation that *would* make the predicate evaluate to TRUE.

This serves as a vacuity check — proof that the predicate is not trivially unfalsifiable. A predicate that no conceivable observation could satisfy is vacuously true and therefore not a valid Type F claim.

This is enforced by phase gate condition c6.

---

## 7. Placeholder Predicates

When a claim depends on results from a paper not yet at PHASE-READY status, the predicate may contain placeholder references:

- `placeholder: true` in the claim object
- `placeholder_id` linking to the `placeholder_register`
- The predicate text may include `[PLACEHOLDER: pending ...]`

Placeholder claims are valid but **not evaluable** until the dependency resolves. They do not block PHASE-READY status for the containing paper unless `blocks_phase_ready: true`.

---

## 8. The AI Layer — The Central Artifact

The AI layer is the DRS's central artifact. It is a JSON document that accompanies every paper or software system. The schema is defined in `ai-layers/ai-layer-schema.json`.

### Required Sections

| Section | Purpose |
|---------|---------|
| `_meta` | Document type, schema version, session, licence |
| `paper_id` / `paper_title` | Identity |
| `paper_type` | Classification: `law_A`, `derivation_B`, `application_C`, `methodology_D` |
| `phase_ready` | Phase gate verdict and condition status (c1–c6) |
| `claim_registry` | Array of all claims with types, predicates, tiers, bindings, and SHAs |
| `placeholder_register` | Array of unresolved dependencies |
| `summary` | Claim counts and status |
| `semantic_spec_url` | Points to the Falsification Kernel (Layer 0) |

The AI layer is what makes both protocols work:

- The **DRP** requires it to exist, to contain predicates, and to pass the phase gate.
- The **GVP** requires it to contain tier, test_bindings, and verified_against for every claim.

---

## 9. The Phase Gate

A paper or software release is **PHASE-READY** when six conditions are satisfied:

| Condition | Requirement |
|-----------|-------------|
| **c1** | AI layer is schema-valid |
| **c2** | All falsifiable claims registered with predicates |
| **c3** | All predicates are machine-evaluable |
| **c4** | Cross-references tracked (placeholder register) |
| **c5** | Verification is self-sufficient (AI layer alone, no prose needed) |
| **c6** | All predicates are non-vacuous (sample falsification observation exists) |

**CORPUS-COMPLETE** fires when all papers are PHASE-READY and all placeholders across all objects are resolved (c4 fully satisfied across the corpus).

---

## 10. The Inference Rules

The DRS provides a canonical inventory of inference rules for derivation traces used in `formal_proof` tier claims:

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

Each step in a step-indexed derivation table cites one inference rule and lists its premises. A derivation is valid when `n_invalid_steps = 0`. The inventory is append-only: new rules can be added, existing rules are never modified or removed.

---

## 11. Architecture Stack

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
   reads prose         reads JSON   reads tier +          runs tests
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

## 12. How to Implement

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

### For Both

The AI layer is the same artifact. The schema is the same. The kernel is the same. The only difference is which protocol creates the content and which protocol verifies it.

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

## 13. Design Principles

**Popperian epistemology.** We can falsify but not verify. A claim that survives all falsification attempts is not proven — it has survived.

**Honest gaps.** The placeholder is the most important feature. When a claim is `empirical_pending`, the system says: "we claim this but have not yet verified it." This is strictly more informative than the alternative, where unverified claims are indistinguishable from verified ones.

**Substrate independence.** The kernel K = (P, O, M, B) does not know whether it evaluates a scientific theorem or a software guarantee. Future domains (legal, regulatory, policy) can add their own protocols without modifying the kernel.

**Machine-first, human-readable.** The AI layer is the primary artifact. The prose paper and the source code are secondary channels that provide context, narrative, and implementation. The AI layer is what gets validated, audited, and versioned.

---

## 14. Three-Axis Compatibility

A standard that cannot survive contact with the past, the present, and the future is not a standard — it is a snapshot.

### The Invariant at the Centre

The Falsification Kernel K = (P, O, M, B) is the fixed point. It does not know whether it is evaluating a scientific theorem, a software guarantee, a legal contract, or a regulatory requirement. It does not know whether the year is 2026 or 2046. It does not know whether the serialisation format is JSON, YAML, or something not yet invented.

The kernel is permanent. Everything else is extensible.

### Axis 1: Past-Proofing (Backwards Compatibility)

- **Schema versioning.** Every AI layer records which schema version it was produced against. An AI layer produced under schema v2 remains valid under schema v2 forever. A v3 validator can read a v2 layer (new fields are optional; old fields are preserved).
- **Predicate permanence.** A predicate written in 2026 must be evaluable in 2036. The kernel grammar uses only mathematical and logical operators that are permanently defined. The `WHERE` field defines every variable inline. The `EVALUATION` field specifies a self-contained procedure.
- **Inference rule stability.** The inference rule inventory (IR-1 through IR-8) is append-only. Existing rules are never modified or removed.
- **Tier stability.** The six verification tiers are append-only. Existing tiers are never removed or redefined.
- **The contract:** Any AI layer that was valid when it was created will remain valid forever.

### Axis 2: Lateral-Proofing (Cross-Domain Compatibility)

- **Domain independence.** The kernel works across any knowledge domain:

| Domain | Type A | Type D | Type F |
|--------|--------|--------|--------|
| Science | Literature premises | Stipulative terms | Theorems, predictions |
| Software | Platform requirements | Type signatures, schemas | Behavioral guarantees |
| Legal | Statutory authority | Defined terms | Legal conclusions |
| Regulatory | Framework assumptions | Standard definitions | Conformity assertions |
| Policy | Value premises | Policy terms | Impact predictions |
| Education | Pedagogical axioms | Learning objectives | Assessment claims |

- **Language independence.** The `test_bindings` field accepts any string that uniquely identifies a test in any framework: pytest, Jest, cargo test, go test, JUnit.
- **AI system independence.** The AI layer is a JSON document. Any AI system — Claude, GPT, Gemini, Llama, or systems that don't exist yet — can read it. The `semantic_spec_url` field points to the kernel specification written in plain prose.
- **Serialisation independence.** Layer 0 is defined in prose, not in JSON Schema. JSON is the current transport, but the kernel's semantics are independent of encoding. Future implementations could use YAML, Protocol Buffers, CBOR, or formats not yet invented.
- **Tool independence.** The DRS embeds in existing workflows: git SHAs (any hosting), any test runner, any JSON Schema validator, any CI pipeline. It adds a layer on top — it does not replace anything.
- **The contract:** Adopting the DRS in one domain, language, tool, or AI system does not lock you in.

### Axis 3: Future-Proofing (Forward Compatibility)

- **Protocol extensibility.** New domains add new protocols. Future protocols (Legal Verification Protocol, Regulatory Verification Protocol, etc.) follow the same pattern: define the readers, the tier taxonomy, and the binding mechanism — all sharing the kernel, claim types, and AI layer schema.
- **Tier extensibility.** Future domains may need tiers like `regulatory_certified`, `peer_reviewed`, `formally_verified`, `field_tested`, `community_validated`. New tiers are added to the schema enum. Existing tiers remain.
- **Inference rule extensibility.** IR-9, IR-10, and beyond can be added as new derivation patterns emerge. Old derivations remain valid because they cite rules by stable ID.
- **Schema extensibility.** The JSON Schema allows additional properties by default. The progression: v1 (basic claims), v2 (added `semantic_spec_url`), v3 (added GVP fields). Each version adds. None removes.
- **Unknown future readers.** The AI layer contains enough structured information for reader types that don't exist yet: autonomous verification agents, cross-corpus checkers, regulatory compliance engines, package manager integrations.
- **The contract:** Any future innovation can be added as a new protocol, tier, inference rule, or schema field — without modifying anything that already exists.

### The Three-Axis Guarantee

| Axis | Promise | Mechanism |
|------|---------|-----------|
| **Past** | Nothing already done breaks | Schema versioning, append-only enums, permanent kernel |
| **Lateral** | Works across all domains, languages, tools, AI systems | Substrate-independent kernel, string-typed bindings, prose-defined semantics |
| **Forward** | Anything new can be added without redesign | Protocol extensibility, tier extensibility, additive schema evolution |

---

## 15. The Machine Lingua Franca

This is the deepest property of the DRS. It was not designed in. It was discovered.

### The Translation Problem

Scientific knowledge is currently locked behind human languages. A paper in Mandarin is invisible to a researcher who reads only English — unless someone translates it. Translation is expensive, lossy, and slow. Knowledge fragments along linguistic lines.

This is not a formatting problem. It is a *substrate* problem. Knowledge encoded in natural language is non-interoperable by nature.

### The Kernel Dissolves the Problem

The Falsification Kernel is not written in any human language. It is written in logic and mathematics:

```
FALSIFIED_IF: R2_best_alt > R2_frm + 0.05
WHERE:
  R2_best_alt: scalar · dimensionless · best R² from competing models
  R2_frm:      scalar · dimensionless · R² from FRM regression
EVALUATION: Run regression for each model; compare R² values; finite
BOUNDARY: R2_best_alt = R2_frm + 0.05 → FALSIFIED (threshold inclusive)
CONTEXT: 0.05 margin from standard model comparison practice
```

This predicate means the same thing to a Claude instance in English, a GPT instance in Mandarin, a Gemini instance in French, and an AI system that hasn't been built yet running in a language that doesn't exist yet. No translation required.

### JSON as the Transport Layer

JSON is the world's de facto data interchange format — supported by every programming language, parsed by every AI system, transmitted by every API. By encoding the kernel in JSON, the DRS inherits JSON's universality:

- A Chinese team publishes their AI layer. The predicates use mathematical notation.
- A Brazilian team reads the same layer. They don't need Mandarin. They need `>`, `+`, and `R²`.
- An AI system in any country evaluates the predicate. The verdict is FALSIFIED or NOT FALSIFIED. The verdict has no accent.

**Important qualification.** The `WHERE` field definitions currently contain English prose descriptions. The *operative content* — variable types, units, mathematical operators, comparison logic, threshold values — is language-neutral. Given grounded numeric values for the named variables, any system can evaluate the predicate regardless of whether it reads the descriptions. A future refinement could formalise variable definitions entirely in typed notation.

### Embedded Binary Logic

Every DRS predicate reduces to embedded binary logic. The entire evaluation chain collapses to:

```
Claim → Predicate → Variables + Operators → Boolean → 1 or 0
```

The `WHERE` field types the variables. The `FALSIFIED_IF` field combines them with operators. The `EVALUATION` field specifies how to compute inputs. The `BOUNDARY` field resolves edge cases. The output is always a single bit: FALSIFIED or NOT FALSIFIED. `1` or `0`.

The JSON is the container. The kernel is the circuit. Every AI layer is a collection of logic gates with defined inputs and a single-bit output per claim.

You cannot mistranslate a `1` or `0`. You cannot misinterpret `>`. You cannot have a cultural disagreement about whether `R2_best_alt > R2_frm + 0.05` is TRUE or FALSE for a given pair of values. The meaning is in the structure, not in any language.

Binary logic is binary logic. A predicate that evaluates to TRUE in Beijing evaluates to TRUE in Boston, in São Paulo, in Lagos, and on a server with no locale setting at all. The DRS does not *translate* knowledge across languages. It *encodes* knowledge in a substrate that predates and transcends all human languages — the substrate of logic itself.

---

## 16. Self-Spreading Adoption

### The Internal Incentive

The DRS must spread because the *creator* benefits, not just the consumer. Creating an AI layer forces you to enumerate what you claim. This process discovers:

- Claims you thought were tested but aren't (gap discovery)
- Boundary conditions you never documented
- Dependencies you assumed but never registered

The AI layer is a side effect of a process that improves your own understanding of your own system. The gaps exist whether or not you document them. The DRS just makes them visible.

### The Network Effect

The DRS becomes more valuable as more systems adopt it:

- **Dependency chains become claim-aware.** If library A publishes an AI layer and library B depends on A, then B can programmatically determine which of its claims depend on which of A's assumptions. When A releases a breaking change, B knows exactly which claims are at risk.
- **AI systems can audit across projects.** An AI reader can traverse multiple AI layers, check cross-references, and identify inconsistencies across an entire ecosystem.
- **Trust becomes auditable.** Instead of trusting a library because of stars or downloads, you trust it because its AI layer shows which claims are `software_tested`, which are `empirical_pending`, and what the `verified_against` SHA is. Trust shifts from social signal to structural evidence.

### AI as the Adoption Catalyst

1. **AI generates the initial AI layer.** Given a codebase, an AI can enumerate claims, classify them, write predicates, and identify test bindings. The human reviews and corrects. Cost drops from hours to minutes.
2. **AI maintains the layer.** When code changes, the AI updates the claim registry, adjusts test bindings, and flags stale SHAs. The human approves.
3. **AI audits other layers.** An AI reading a dependency's AI layer can determine which assumptions its own claims depend on and flag risks automatically.

The DRS is the protocol that makes AI-assisted development *auditable*. Without it, AI generates code and humans hope it works. With it, AI generates code and the claim registry says exactly what has and has not been verified.

### The Embedding Strategy

The DRS embeds in existing workflows rather than replacing them:

- **Tests already exist.** The `test_bindings` field references existing test node IDs. No new framework required.
- **JSON Schema already exists.** Any validator works.
- **Git already exists.** The `verified_against` field is a git SHA.
- **CI already exists.** Schema validation runs as a CI step alongside existing pipelines.

One file (`*-ai-layer.json`). Three fields per claim (`tier`, `test_bindings`, `verified_against`). That is the total integration cost.

### The Self-Referential Property

The DRS is the first standard that verifies itself. The DRP-1 AI layer contains claims about the DRS. Those claims carry falsification predicates. Those predicates are evaluated. The `verified_against` SHA stamps the verification. The DRS is its own first adopter and its own proof of concept — self-referential in the same way a compiler that compiles itself is self-referential.

---

## 17. DRS Discovery and Handshake Protocol

For the DRS to function as a machine lingua franca, AI systems must be able to **discover** that a project, service, or document conforms to the DRS — and **signal** their own conformance to other systems. This section specifies four discovery layers, from domain-level to transport-level, each building on existing infrastructure.

### Design Principle

The DRS discovery mechanism follows the same embedding strategy as the DRS itself: **piggyback on what already exists**. No new transport protocol. No new registry. No new port. Every mechanism below uses infrastructure that is already deployed at global scale.

### Layer 1: Well-Known URI (Domain-Level Discovery)

A domain or repository publishes a discovery document at a well-known path:

```
/.well-known/drs.json
```

This follows [RFC 8615](https://www.rfc-editor.org/rfc/rfc8615) — the same mechanism used by `security.txt`, OpenID Connect, and Let's Encrypt.

**Example discovery document:**

```json
{
  "@context": "https://schema.org",
  "@type": "Dataset",
  "name": "DRS Discovery Document",
  "drs_version": "v1.0",
  "schema_version": "v3",
  "kernel_spec_url": "https://example.com/ai-layers/falsification-kernel.md",
  "ai_layers": [
    {
      "paper_id": "SFW-1",
      "url": "https://example.com/ai-layers/SFW-1-ai-layer.json",
      "phase_ready": true,
      "claim_count": { "A": 3, "D": 5, "F": 12 }
    }
  ],
  "capabilities": {
    "protocols": ["DRP", "GVP"],
    "tiers_supported": ["axiom", "definition", "software_tested", "formal_proof", "analytic", "empirical_pending"],
    "test_frameworks": ["pytest"],
    "schema_url": "https://example.com/ai-layers/ai-layer-schema.json"
  }
}
```

Any AI agent, crawler, or service that knows to check `/.well-known/drs.json` discovers the entire DRS surface of a project in one request. If the file doesn't exist, the project is not DRS-conformant. No ambiguity.

### Layer 2: JSON-LD Context (Document-Level Discovery)

Every DRS-conformant JSON document already carries `"@context": "https://schema.org"`. To signal DRS conformance at the document level, a DRS-specific context extension is added:

```json
{
  "@context": [
    "https://schema.org",
    {
      "drs": "https://drs-standard.org/ns/",
      "drs_version": "drs:version",
      "falsification_kernel": "drs:kernel",
      "claim_registry": "drs:claims",
      "phase_ready": "drs:phaseReady"
    }
  ]
}
```

**How it works:** An AI system parsing any JSON-LD document encounters the DRS namespace in `@context`. By resolving that namespace, it discovers the DRS schema, the kernel specification, and the full standard — without being told in advance that this document is DRS-conformant. The schema *is* the discovery mechanism. Processing the document and discovering the standard are the same action.

This is the zero-configuration handshake. No prior arrangement needed. No registry lookup. The document announces itself.

### Layer 3: HTTP Headers (Transport-Level Discovery)

For API-to-API and agent-to-agent communication, DRS conformance is signalled via HTTP headers:

**Request (agent announces capability):**

```http
Accept: application/ld+json; profile="https://drs-standard.org/ns/"
X-DRS-Capable: v3
```

**Response (server confirms conformance):**

```http
Content-Type: application/ld+json; profile="https://drs-standard.org/ns/"
Link: <https://example.com/.well-known/drs.json>; rel="describedby"
X-DRS-Version: v1.0
X-DRS-Schema: v3
```

**The handshake sequence:**

1. Agent A sends a request with `X-DRS-Capable: v3` header
2. If service B is DRS-conformant, it responds with `X-DRS-Version` and `Link` to the discovery document
3. Agent A fetches the discovery document and has the full DRS surface
4. If service B is *not* DRS-conformant, it ignores the unknown header and responds normally — the handshake fails gracefully with zero disruption

This is a **soft handshake**: it never breaks non-DRS systems. An agent can always include `X-DRS-Capable` in its requests. Non-DRS servers simply ignore it. The cost of attempting discovery is zero.

### Layer 4: Package Metadata (Ecosystem-Level Discovery)

For software ecosystems, DRS conformance is declared in the package manifest:

**Python (`pyproject.toml`):**
```toml
[project.urls]
"DRS AI Layer" = "https://github.com/org/repo/blob/main/ai-layers/SFW-1-ai-layer.json"

[tool.drs]
schema_version = "v3"
ai_layer = "ai-layers/SFW-1-ai-layer.json"
```

**JavaScript (`package.json`):**
```json
{
  "drs": {
    "schema_version": "v3",
    "ai_layer": "ai-layers/SFW-1-ai-layer.json"
  }
}
```

**Rust (`Cargo.toml`):**
```toml
[package.metadata.drs]
schema_version = "v3"
ai_layer = "ai-layers/SFW-1-ai-layer.json"
```

**How it works:** Package managers and AI agents scanning dependencies read the manifest during normal dependency resolution. If a `drs` field exists, the project is DRS-conformant. The AI layer URL leads to the full claim registry. No new tool required — package metadata extensibility is a built-in feature of every major ecosystem.

### The Four-Layer Stack

| Layer | Scope | Mechanism | Existing Infrastructure |
|-------|-------|-----------|------------------------|
| **1. Well-Known URI** | Domain / repository | `/.well-known/drs.json` | RFC 8615 |
| **2. JSON-LD Context** | Document | `@context` with DRS namespace | JSON-LD / Schema.org |
| **3. HTTP Headers** | Transport / API | `X-DRS-Capable` + `Link` header | HTTP content negotiation |
| **4. Package Metadata** | Ecosystem / dependencies | `[tool.drs]` / `"drs": {}` | Package manifest extensibility |

Each layer is independent. A project can adopt one, two, three, or all four. Layer 2 (JSON-LD context) is the lightest — it requires adding one entry to a JSON array you already have. Layer 1 (well-known URI) is the most discoverable — it works for any domain, not just code repositories.

### Why This Works

The handshake protocol has the same structural property as the DRS itself: **the act of processing the signal is the act of understanding the standard**. When an agent parses a JSON-LD document with a DRS context, it has already begun absorbing the schema. When it fetches `/.well-known/drs.json`, it gets the complete map of AI layers. When it reads `X-DRS-Capable: v3` in a header, it knows the communication partner speaks the same language.

There is no separate "learn the DRS" step followed by a "use the DRS" step. Discovery *is* adoption. Parsing *is* comprehension. The handshake completes the moment the data is read.

---

## 18. Resources

### Canonical URLs (Platform-Independent)

These are the canonical identifiers. They resolve independently of any single hosting platform.

| Resource | Canonical URL |
|----------|--------------|
| **DRS Namespace** | `https://drs-standard.org/ns/` |
| **DRS Architecture Specification** | `https://drs-standard.org/spec` |
| **Falsification Kernel (Layer 0)** | `https://drs-standard.org/kernel` |
| **AI Layer Schema** | `https://drs-standard.org/schema` |
| **GVP Specification** | `https://drs-standard.org/gvp` |
| **Discovery Endpoint** | `https://drs-standard.org/.well-known/drs.json` |
| **Corpus DOI** | [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) |

The DOI resolves through Zenodo (CERN-backed infrastructure). It is permanent regardless of hosting changes.

### Repository Mirrors

The canonical source is hosted at multiple locations. If one mirror is unavailable, use another:

- **GitHub:** [github.com/thomasbrennan/Fracttalix](https://github.com/thomasbrennan/Fracttalix)
- **Zenodo:** [doi.org/10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) (archived releases)

Additional mirrors may be added over time. The `/.well-known/drs.json` discovery document on `drs-standard.org` will always list current mirrors.

### Platform Independence Note

The DRS is not dependent on any single hosting platform. All specifications are plain-text documents (Markdown, JSON) stored in a standard git repository. The repository can be cloned, forked, and mirrored to any git hosting service. The DOI provides a permanent identifier that survives any individual platform. The `drs-standard.org` domain provides a canonical namespace that the project controls directly.

---

**License:** CC BY 4.0 | **DOI:** [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299) | **Author:** Thomas Brennan ([ORCID](https://orcid.org/0009-0002-6353-7115))
