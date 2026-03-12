# MK-P6 Build Process — The Dual Reader Standard for Software

**Session:** 51
**Date:** 2026-03-11
**Process:** Canonical Build (P0 CBT v2)
**Author:** Thomas Brennan · with Claude (Anthropic)

---

## Phase 1: First Build Plan

### 1.1 Paper Identity

| Field | Value |
|-------|-------|
| Paper ID | MK-P6 |
| Title | The Dual Reader Standard for Software: Measurement-Theoretic Falsification Applied to Executable Systems |
| Type | methodology_D |
| Track | Meta-Kaizen (extension: MK 6/6) |
| Status | DRAFT |
| Depends on | MK-P1 (KVS axioms), P14/DRP-1 (DRS specification), SFW-1 (Sentinel — demonstration target), falsification-kernel.md (Layer 0) |
| Enables | P8 (Software and Engineered Systems), P14 v2.0 (DRS extended to software), SFW-1 v2 (software AI layer upgrade) |

### 1.2 Core Question

Can the Dual Reader Standard — the three-layer architecture (Layer 0 semantic kernel, Layer 1 machine-readable claim registry, Layer 2 human-readable prose) — be extended from scientific papers to executable software without modifying the underlying falsification kernel K = (P, O, M, B)?

### 1.3 Thesis (Stated As Falsifiable Claim)

**C-MK6.1:** The falsification kernel K = (P, O, M, B) as defined in falsification-kernel.md v1.1 applies to software behavioral claims without modification. The same grammar, evaluation semantics, and validity constraints that govern scientific Type F claims govern software Type F claims.

**Falsification condition:** Exhibit a software behavioral claim that (a) is clearly falsifiable by empirical test, but (b) cannot be expressed as a well-formed K = (P, O, M, B) 4-tuple under the existing kernel constraints.

### 1.4 Planned Claim Registry (First Pass)

| Claim ID | Type | Name | Description |
|----------|------|------|-------------|
| A-MK6.1 | A | Popperian epistemological foundation | Falsificationism as the basis for software verification: we can falsify but not verify |
| A-MK6.2 | A | Conjoint measurement applicability | Software behavioral claims admit the same measurement-theoretic structure as scientific claims (from MK-P1 A1–A6) |
| A-MK6.3 | A | DRS Layer 0 kernel | The falsification kernel K = (P, O, M, B) as given in falsification-kernel.md v1.1 |
| D-MK6.1 | D | Software Claim Taxonomy | Three-type classification: Assumptions (A), Definitions (D), Falsifiable (F) — same taxonomy as scientific DRS |
| D-MK6.2 | D | Software Assumption (Type A) | Platform requirements, dependency contracts, environmental preconditions |
| D-MK6.3 | D | Software Definition (Type D) | Type signatures, data structures, configuration schemas |
| D-MK6.4 | D | Software Behavioral Claim (Type F) | Correctness guarantees, invariants, performance bounds, API contracts |
| D-MK6.5 | D | Falsification Completeness | A software system is falsification-complete iff every Type F claim has a well-formed predicate and a passing evaluation |
| D-MK6.6 | D | Software Phase-Ready | Extension of DRS phase-ready verdict to software releases: c1–c6 adapted |
| F-MK6.1 | F | Kernel universality | K = (P, O, M, B) applies to software claims without modification |
| F-MK6.2 | F | Falsification completeness implies coverage | If S is falsification-complete, ∃ test suite T from evaluations with L(S,T) ≥ L(S,T') for any T' achieving same behavioral verification |
| F-MK6.3 | F | Coverage does not imply falsification completeness | ∃ systems with L(S,T) = 1.0 that are not falsification-complete |
| F-MK6.4 | F | Assumption propagation | When A_k is invalidated, all dependent F-claims computable in O(|R|) |
| F-MK6.5 | F | Gap detection superiority | Applying DRS to an existing well-tested codebase reveals claims invisible to coverage metrics |
| F-MK6.6 | F | Demonstration: Sentinel v12.1 | Sentinel produces a valid software AI layer with N_A assumptions, N_D definitions, N_F falsifiable claims, with M placeholder claims not covered by existing tests |

### 1.5 Prior Art Structure (Planned Sections)

The paper must survey prior art across ALL major language/cultural traditions:

| Tradition | Key Contributions | Gap Relative to DRS |
|-----------|-------------------|---------------------|
| **Anglo-American** | Design by Contract (Meyer 1992), QuickCheck (Claessen & Hughes 2000), TLA+ (Lamport), Alloy (Jackson), RV conferences | No claim registry; no placeholder honesty; no boundary documentation standard |
| **French** | B-Method (Abrial), Atelier B, CompCert (Leroy), Paris Métro Line 14 | Full formal verification — stronger but less tractable; no "honest accounting of gaps" |
| **Dutch** | Dijkstra structured programming, weakest precondition calculus | Foundational for correctness proofs; does not address claim enumeration |
| **German** | VDI/DIN standards, automotive ISO 26262 | Requirement traceability exists; not machine-readable falsification |
| **Scandinavian** | SIMULA (Dahl & Nygaard), Scandinavian OOP school | Object-oriented contracts implicit but not formalized as falsification |
| **Russian/Soviet** | Ershov programming methodology, GOST standards | Strong theoretical tradition; limited Western integration |
| **Japanese** | Kaizen, Monozukuri, JUSE quality circles, software factories | Quality culture strong but philosophical, not formal-methods based |
| **Chinese** | CertiKOS (Zhong Shao/Yale), ORIENTAIS (ECNU), GB/T standards | Formal verification of OS kernels; no behavioral claim registry standard |
| **Indian** | CMMI Level 5 adoption, STQC | Process maturity models — audits process, not claims |
| **ISO/IEC** | 29119 (testing), 25010 (quality), 12207 (lifecycle), 15026 (assurance) | Requirement traceability but no machine-readable falsification predicates |
| **Safety-critical** | DO-178C (avionics), IEC 62304 (medical), ISO 26262 (automotive) | Closest to DRS in spirit — objective evidence traceability; still prose-based |

### 1.6 Paper Structure

1. Series Orientation
2. Abstract (AI-Reader Header + Human Reader)
3. The Problem: Software Claims Are Implicit and Unaudited
4. Prior Art Survey (11 traditions + ISO + safety-critical)
5. The Software Claim Taxonomy (Type A/D/F mapping)
6. Core Theoretical Results (Theorems 5.1–5.3)
7. What DRS Reveals That Tests Cannot (Three Gap Categories)
8. Demonstration: Fracttalix Sentinel v12.1
9. Implications (Open Source Trust, Dependency Management, AI-Generated Code, Regulatory Compliance)
10. Limitations
11. Conclusion
12. References

### 1.7 Deliverables

1. MK-P6 paper (markdown)
2. MK-P6 AI layer (JSON, conforming to ai-layer-schema.json v2-S50)
3. Sentinel Software AI Layer v2 (comprehensive behavioral claims beyond SFW-1 v1)
4. Build Table update (MK track expanded to 6 papers)
5. Session journal entry

---

## Phase 2: Hostile Review

### Objection 1: "This is just testing with extra steps"

**Attack:** Unit testing frameworks already exist. pytest, JUnit, xUnit — the entire ecosystem. What does wrapping tests in JSON actually accomplish that `pytest --verbose` doesn't? You're adding bureaucratic overhead to a solved problem.

**Severity:** HIGH — must be addressed head-on.

### Objection 2: "Prior art is already doing this — Design by Contract"

**Attack:** Meyer (1992) defined preconditions, postconditions, and invariants as executable specifications 34 years ago. Eiffel enforces them at runtime. How is a "claim registry" different from a contract? You're reinventing DbC with worse tooling.

**Severity:** HIGH — DbC is the most direct ancestor. The distinction must be precise.

### Objection 3: "The French already solved this more rigorously"

**Attack:** The B-Method (Abrial) with Atelier B has run Paris Métro Line 14 since 1998 with zero bugs in version 1.0. CompCert (Leroy) is a Coq-verified C compiler. These are *proofs*, not falsification predicates. Why would anyone settle for "not yet falsified" when "proven correct" exists? Your Popperian stance is a retreat from what formal methods already achieve.

**Severity:** HIGH — this is the strongest intellectual challenge. The answer must explain why falsification is the right level of abstraction for most software, not just a weaker substitute.

### Objection 4: "Registry completeness is unverifiable — your own paper admits it"

**Attack:** Limitation #1 states there is no algorithm to determine whether all behavioral claims have been enumerated. So the registry's value proposition — "here is an honest accounting" — has an unfillable hole. How honest can the accounting be when you can't verify the ledger is complete?

**Severity:** MEDIUM — legitimate but manageable. Science has the same problem.

### Objection 5: "The theorems are trivial"

**Attack:** Theorem 5.1 (falsification completeness implies coverage) is nearly tautological — if you test every claim, you cover the code that implements those claims. Theorem 5.2 (coverage doesn't imply completeness) is a well-known observation. Theorem 5.3 (assumption propagation) is a graph traversal. None of these are deep mathematical results.

**Severity:** MEDIUM — the theorems need better framing or the paper needs to acknowledge they formalize intuitions rather than discovering new ones.

### Objection 6: "No independent empirical validation"

**Attack:** You demonstrate on your own software (Sentinel) which you built. Of course you can enumerate its claims — you wrote the code. The real test is whether a third party can apply DRS to unfamiliar software and find gaps that matter. One self-demonstration is anecdotal, not evidence.

**Severity:** HIGH — the empirical validation story is weak. Must be addressed honestly.

### Objection 7: "Popper is the wrong epistemology for software"

**Attack:** Software *can* be formally verified. Coq, Lean, Isabelle, SPARK Ada — these prove correctness for all inputs, not just "not yet falsified." Settling for Popperian falsification in a domain where verification is achievable is philosophically backwards. Science needs falsification because we can't run all experiments; software can.

**Severity:** HIGH — the epistemological stance must be defended with the tractability argument.

### Objection 8: "Requirements engineering already does traceability"

**Attack:** ISO/IEC 29148 defines requirements traceability. DO-178C requires objective evidence that every requirement has been tested. DOORS, Polarion, and Jama Connect already implement bidirectional traceability. You're claiming novelty in a space that safety-critical industries solved decades ago.

**Severity:** HIGH — this needs careful distinction between requirements traceability (prose-to-test links) and falsification predicates (deterministic evaluation semantics).

### Objection 9: "The scope boundary is unclear"

**Attack:** Does this apply to all software? A one-off script? A 50-line Lambda function? Or only to libraries and frameworks consumed by others? The paper implies universal applicability but acknowledges the overhead may not be justified for simple systems. Where is the line?

**Severity:** MEDIUM — scope should be defined explicitly.

### Objection 10: "Cultural survey is tourism, not scholarship"

**Attack:** Listing 11 traditions in a table with one-sentence summaries is not engaging with those traditions deeply. The Japanese kaizen section doesn't cite Imai (1986). The Russian section doesn't engage with Ershov's 1972 Aesthetics of Programming. The Chinese section doesn't discuss Confucian quality ethics. This reads as box-checking, not genuine intellectual engagement.

**Severity:** MEDIUM-HIGH — the survey must either go deep or acknowledge its breadth-first nature explicitly.

---

## Phase 3: Second Meta-Kaizen

### Corrections Applied

| # | Objection | Response | Effect |
|---|-----------|----------|--------|
| 1 | "Just testing with extra steps" | Testing verifies code runs. DRS verifies *what the code claims*. The distinction: a test suite with 100% coverage tells you every line executed; a claim registry tells you which *promises* have been checked. Three concrete additions: (a) claim enumeration — forces you to list what you're promising, (b) boundary documentation — forces you to justify thresholds, (c) placeholder honesty — forces you to admit what's untested. No test framework provides these. | **Strengthened** — add comparison table: test framework vs. DRS, row by row |
| 2 | "DbC already exists" | DbC embeds contracts in code. DRS extracts them into an auditable registry. Three precise distinctions: (1) DbC is local (contract lives with function) — DRS is global (registry is cross-cutting); (2) DbC asserts conditions but doesn't document *why* the threshold is what it is (no BOUNDARY/CONTEXT equivalent); (3) DbC has no placeholder mechanism — an unannotated function is indistinguishable from one deliberately left without contracts. DRS + DbC is complementary, not competitive. | **Resolved — stronger** — reframe as extending DbC, not replacing it |
| 3 | "French formal methods are stronger" | Correct. Formal verification is stronger. And it costs orders of magnitude more. The B-Method succeeded for Paris Métro (safety-critical, bounded scope, well-funded). CompCert succeeded because it's a compiler (fixed specification). Most software has neither the budget nor the fixed specification for full formal verification. DRS occupies the *tractability gap*: more rigorous than testing, more accessible than formal verification. The positioning is deliberate: Popperian falsification is not a retreat from verification — it is the appropriate level of rigor for the 99% of software that will never be formally verified. | **Strengthened** — add positioning diagram: informal testing → DRS → formal verification. DRS is the middle ground. |
| 4 | "Registry completeness unverifiable" | This is the same epistemological situation as scientific papers. No algorithm can verify that a paper has stated all its claims. The DRS doesn't claim to solve this — it claims to make the *accounting* machine-readable. An incomplete registry is still more informative than no registry. And the placeholder mechanism means you can incrementally improve: every gap discovered can be registered immediately. | **Discipline enforced** — state limitation clearly in Section 10, but note it is inherent to any claim-enumeration system, not specific to DRS |
| 5 | "Theorems are trivial" | Acknowledged. Theorems 5.1–5.3 formalize intuitions that practitioners already hold informally. Their value is not mathematical depth but *precision*: they give exact conditions under which claims about testing and coverage hold. Reframe: the contribution is the *framework* (taxonomy + registry + kernel), not the theorems. Theorems are scaffolding that makes the framework's properties precise. | **Scope refined** — de-emphasize theorems as "core results"; reframe as "formal properties of the framework" |
| 6 | "No independent validation" | Acknowledged fully. One self-demonstration is not evidence. The paper is a *framework proposal*, not an empirical study. The empirical agenda is: (a) apply DRS to 3+ independent open-source projects, (b) measure whether gap detection rate exceeds conventional coverage metrics, (c) measure maintenance overhead. State this explicitly as future work. The Sentinel demonstration proves *feasibility*, not *value*. | **Discipline enforced** — rename Section 8 from "Demonstration" to "Feasibility Demonstration" and add pre-registered empirical agenda |
| 7 | "Popper wrong for software" | The argument isn't that formal verification is wrong — it's that it's inaccessible for most software. Additionally: formal verification proves properties for *all* inputs, but only for the properties you think to verify. It doesn't enumerate claims — you still need to decide *what* to prove. DRS addresses the *what* (claim enumeration); formal verification addresses the *how* (proof vs. testing). They are complementary layers. A formally verified function still benefits from being in a claim registry. | **Strengthened** — reframe DRS as complementary to formal verification, not a substitute. DRS answers "what do we claim?"; formal methods answer "can we prove it?" |
| 8 | "Requirements engineering already does this" | Requirements traceability links natural-language requirements to test procedures. DRS links machine-readable *falsification predicates* to claims. Three precise differences: (1) RE requirements are prose — DRS predicates are deterministic (FALSIFIED_IF evaluates to TRUE/FALSE); (2) RE traceability is bidirectional linking — DRS predicates include EVALUATION (how to compute), BOUNDARY (edge cases), CONTEXT (justification); (3) RE doesn't distinguish Assumptions from Definitions from Falsifiable Claims — everything is a "requirement." The DRS taxonomy (A/D/F) makes the epistemological status explicit. | **Resolved — stronger** — add comparison table: RE traceability vs. DRS falsification predicates |
| 9 | "Scope unclear" | Define explicitly: DRS for Software is most valuable for **libraries and frameworks consumed by others** — where the cost of undocumented assumptions is highest. Secondary value for **safety-critical systems** (where it complements existing standards). Lower value for internal scripts and one-off tools. The overhead/value ratio determines applicability, not a universal mandate. | **Scope refined** — add Section 9.5: Applicability Boundary |
| 10 | "Cultural survey is tourism" | Two options: (a) go genuinely deep into each tradition — but this becomes a 60-page survey paper, not a framework paper; (b) acknowledge the survey is breadth-first and exists to establish that no tradition has built the specific combination DRS provides. Choose option (b): add explicit statement that the survey identifies *what each tradition contributed* and *what gap remains relative to DRS*, not a comprehensive intellectual history. Cite properly: Imai (1986) for kaizen, Ershov (1972) for aesthetics, Abrial (1996) for B-Method. | **Discipline enforced** — add proper citations; reframe as gap analysis, not intellectual history |

---

## Phase 4: Final Build Plan

### Revised Paper Architecture (Post-Hostile-Review)

The hostile review produced 10 objections. All 10 are addressed:
- 3 **Strengthened** (testing distinction, French positioning, Popper complementarity)
- 3 **Resolved — stronger** (DbC extension, RE comparison, overall reframe)
- 2 **Discipline enforced** (registry completeness, cultural survey)
- 2 **Scope refined** (theorem emphasis, applicability boundary)

### Revised Structure

1. **Series Orientation** — MK-P6 extends MK track from 5 to 6 papers
2. **Abstract** (AI-Reader Header + Human Reader)
3. **The Problem** — Software claims are implicit and unaudited (unchanged)
4. **Prior Art Gap Analysis** — RENAMED from "Survey." 11 traditions examined for what they contributed and what gap remains. Proper citations. Explicit breadth-first acknowledgment.
5. **The Software Claim Taxonomy** — Type A/D/F mapping, with DbC comparison table
6. **Positioning: The Tractability Gap** — NEW SECTION. Informal testing → DRS → formal verification. DRS is the middle ground between what most teams do and what most teams can't afford.
7. **Formal Properties of the Framework** — RENAMED from "Core Theoretical Results." Theorems reframed as properties, not discoveries.
8. **What DRS Reveals That Tests Cannot** — Three gap categories + comparison table: test framework vs. DRS
9. **Feasibility Demonstration: Fracttalix Sentinel v12.1** — RENAMED. Explicit acknowledgment that one self-demonstration proves feasibility, not value.
10. **Implications** — Open source trust, dependency management, AI-generated code, regulatory compliance, applicability boundary
11. **Empirical Agenda** — NEW SECTION. Pre-registered validation plan for 3+ independent projects.
12. **Limitations** — Registry completeness, Popperian epistemology, overhead, no independent validation
13. **Conclusion**
14. **References** — With proper citations across traditions

### Revised Claim Registry

No changes to claim IDs. Add:
- **F-MK6.7**: Tractability claim — DRS can be applied to a 37-step pipeline in one session; formal verification of equivalent scope requires specialized tooling and expertise beyond most development teams
- Update F-MK6.5 wording: "reveals claims invisible to coverage metrics" → "reveals at least one claim category invisible to coverage metrics in the feasibility demonstration"

### Key Additions to Paper

1. **Comparison table: Test Framework vs. DRS** (addresses Objection 1)
2. **Comparison table: RE Traceability vs. DRS Predicates** (addresses Objection 8)
3. **Positioning diagram: Testing → DRS → Formal Verification** (addresses Objections 3, 7)
4. **DbC relationship section** (addresses Objection 2)
5. **Applicability Boundary section** (addresses Objection 9)
6. **Empirical Agenda section** (addresses Objection 6)
7. **Proper citations for cultural traditions** (addresses Objection 10)

---

## Process Notes

- This is the first application of the canonical build process to a Meta-Kaizen track extension
- MK-P6 extends the track from 5 to 6 papers
- The paper is self-referential: it describes the DRS for software and its own AI layer is a DRS layer for a paper about software DRS layers
- The Sentinel demonstration is not hypothetical — we will build the actual software AI layer as part of this session
