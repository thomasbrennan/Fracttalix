Meta-Kaizen Series · Paper 6 of 6

## The Dual Reader Standard for Software: Measurement-Theoretic Falsification Applied to Executable Systems

Thomas Brennan · with AI collaborator Claude (Anthropic)

March 2026 · Submitted for peer review

AI contributions: Claude (Anthropic) provided the formal mapping between scientific falsification and software verification, the claim taxonomy, cross-tradition prior art analysis, and manuscript drafting. All theoretical contributions are contributed to the public domain.

## 1. Series Orientation

This is Paper 6 of six — an extension of the Meta-Kaizen series from governance substrates to executable substrates. Papers 1–5 derived a measurement-theoretic governance framework for scientific claims. This paper asks: can the same Dual Reader Standard (DRS) that makes scientific papers machine-verifiable make software machine-auditable — not merely tested, but falsification-complete?

The five prior papers established:
- A scoring framework for evaluating improvement candidates (Paper 1)
- A federated architecture for scaling governance (Paper 2)
- A cognitive infrastructure for institutional memory (Paper 3)
- Regime-adaptive governance under dynamic conditions (Paper 4)
- Decision theory for intervention timing (Paper 5)

This paper extends the DRS itself — the verification layer that underlies all five — from prose claims to executable claims. The contribution is not "software should have tests." The contribution is a formal framework in which every behavioral claim a program makes is registered, classified, and linked to a deterministic falsification predicate — or honestly marked as a placeholder.

## 2. Abstract and AI-Reader Header

## [AI-READER HEADER] — Dual-Reader Standard, Section 2

How to verify this paper without reading it: Load the AI layer JSON at the URL above. Run the schema validator (claim_registry, inference_rule_table, phase_ready fields). Every Type F claim has a deterministic 5-part predicate — FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT — that evaluates to FALSIFIED or NOT FALSIFIED without accessing prose. Placeholders in the register are the only unresolved dependencies.

## Abstract (Human Reader)

Software systems make claims. A function's type signature claims it accepts certain inputs and produces certain outputs. A README claims the software "detects anomalies in streaming data." A performance benchmark claims O(n) complexity. An API specification claims that endpoint X returns status 200 under condition Y. Yet no existing standard systematically enumerates these claims, classifies them by verifiability, links each to a deterministic falsification procedure, or provides an honest accounting of which claims lack verification.

This paper extends the Dual Reader Standard (DRS), originally developed for scientific papers in the Fracttalix and Meta-Kaizen corpora (Papers 1–5), to executable software systems. We define three claim types for software — Assumptions (Type A: platform requirements, dependency contracts, environmental preconditions), Definitions (Type D: type signatures, data structures, configuration schemas), and Falsifiable Claims (Type F: behavioral guarantees, correctness invariants, performance bounds) — and show that the existing falsification kernel K = (P, O, M, B) maps without modification to software verification.

The framework occupies a tractability gap: more rigorous than conventional testing, more accessible than formal verification. It is complementary to both — it answers "what do we claim?" while testing answers "does it pass?" and formal verification answers "can we prove it?"

We present a feasibility demonstration on Fracttalix Sentinel v12.1 (37-step streaming anomaly detector, 374 existing tests), producing a software AI layer that maps behavioral claims to falsification predicates. The demonstration reveals gaps invisible to conventional coverage metrics: documented promises without tests, tested behavior without boundary documentation, and dependency assumptions without explicit registration.

## 3. The Problem: Software Claims Are Implicit and Unaudited

### 3.1 The Current State of Software Verification

Software quality assurance currently relies on four mechanisms, each with a fundamental limitation:

**Unit tests** verify specific input-output pairs but do not enumerate the claims they cover. A project with 374 passing tests may verify 47 behavioral claims or 4 — the test suite itself does not say.

**Code coverage** measures which lines of source code are executed during testing. A function with 100% line coverage may still violate its behavioral contract if the test exercises every line but does not check the output. Coverage measures execution, not verification.

**Type systems** verify structural contracts (this function accepts an integer and returns a string) but not behavioral contracts (this function returns the *correct* string). Types are necessary but not sufficient.

**Documentation** (READMEs, API docs, docstrings) states behavioral claims in natural language that no automated system audits against actual behavior. Documentation can diverge from implementation silently and indefinitely.

### 3.2 The Gap

No existing standard answers the question: *What exactly does this software claim to do, and which of those claims have been verified?*

This is the same gap the DRS closed for scientific papers. A paper before the DRS made claims in prose that a reader had to manually identify and evaluate. A paper under the DRS has a machine-readable claim registry where every claim is classified, and every falsifiable claim carries a deterministic predicate. The reader — human or AI — can audit the paper's honesty without reading the prose.

Software needs the same thing.

### 3.3 What This Paper Adds Beyond Testing

The distinction between testing and the DRS for Software is precise:

| Capability | Test Framework | DRS for Software |
|-----------|----------------|-----------------|
| **Claim enumeration** | Tests exist but are not mapped to behavioral promises | Every promise is a registered claim |
| **Behavioral coverage** | Line/branch coverage (code executed) | Claim coverage (promises verified) |
| **Boundary documentation** | Assertions check conditions | BOUNDARY field justifies *why* each threshold |
| **Gap honesty** | Untested behavior is invisible | Placeholders make gaps explicit |
| **Dependency tracking** | requirements.txt lists packages | Type A registry lists behavioral *assumptions* about packages |
| **Epistemological classification** | All tests are equal | Claims classified as Axiom, Definition, or Falsifiable |

The DRS does not replace testing — it provides the accounting layer that testing lacks.

## 4. Prior Art Gap Analysis

This section examines software verification traditions across major linguistic and cultural traditions. The purpose is not comprehensive intellectual history but systematic gap identification: what has each tradition contributed, and what remains unaddressed relative to the DRS?

### 4.1 Anglo-American Tradition

**Design by Contract** (Meyer, 1992). Preconditions, postconditions, and invariants as executable specifications. Eiffel enforces them at runtime. The DRS extends DbC in three precise ways: (1) DbC is *local* — contracts live with individual functions; the DRS is *global* — a cross-cutting registry enables system-level auditing. (2) DbC asserts conditions but does not require justification for thresholds (no BOUNDARY/CONTEXT equivalent). (3) DbC has no placeholder mechanism — an unannotated function is indistinguishable from one deliberately left without contracts. DRS and DbC are complementary, not competitive.

**Property-Based Testing** (Claessen & Hughes, 2000). QuickCheck generates random inputs to test properties. The DRS's `sample_falsification_observation` (vacuity witness) is philosophically aligned — both ask "what input would break this?" The DRS adds the registry, boundary documentation, and placeholder tracking that property-based testing frameworks do not provide.

**Formal Specification Languages.** TLA+ (Lamport, 1994) specifies concurrent systems. Alloy (Jackson, 2002) models structural constraints. Z notation (Spivey, 1989) and VDM (Jones, 1990) specify behavior formally. These specify *what should hold* — they do not provide machine-readable falsification predicates with boundary justification and gap accounting.

**Runtime Verification.** The RV conference community (Leucker & Schallhart, 2009) monitors systems at runtime against formal specifications. Runtime monitors are evaluation procedures — they could serve as the EVALUATION component of a DRS predicate — but they exist without claim registries, boundary documentation, or gap accounting.

### 4.2 French Tradition

**B-Method** (Abrial, 1996). A stepwise refinement method that proves each transformation correct. Atelier B (ClearSy) has been used for Paris Métro Line 14 since 1998 — no bugs found in version 1.0 for over nine years. This is the strongest existing approach to software correctness.

**CompCert** (Leroy, 2009). A C compiler verified in Coq. The proof guarantees that safety properties verified on source code hold for compiled output.

These French contributions represent *full formal verification* — strictly stronger than the DRS. The DRS is deliberately weaker, occupying a different position in the rigor-tractability tradeoff (see Section 6). The B-Method does not maintain a claim registry or an accounting of what remains unproven — it aims to prove everything. The DRS is designed for the 99% of software that will never receive B-Method treatment.

### 4.3 Dutch Tradition

**Dijkstra** (1968, 1976). Structured programming, weakest precondition calculus, the *Discipline of Programming*. Foundational for correctness reasoning — the intellectual ancestor of all formal approaches. Dijkstra's predicate transformers are conceptual precursors to DRS falsification predicates, but Dijkstra sought proofs of correctness, not falsification accounting.

### 4.4 Scandinavian Tradition

**SIMULA** (Dahl & Nygaard, 1966). The first object-oriented language, developed in Oslo. Classes as behavioral contracts were implicit in SIMULA's design — objects promise behavior via their interfaces — but this was never formalized as a machine-readable claim standard.

### 4.5 German/Austrian Engineering Standards

**ISO 26262** (automotive functional safety, German-led development). Requirements traceability from safety goals through architecture to test evidence. The closest industrial standard in spirit to the DRS. Key difference: ISO 26262 traceability is prose-based (natural language requirements linked to test reports); DRS traceability is predicate-based (deterministic evaluation semantics).

**VDI standards** and DIN norms enforce rigorous documentation but do not specify machine-readable falsification predicates.

### 4.6 Russian/Soviet Tradition

**Ershov** (1972, 1977). *Aesthetics and the Human Factor in Programming* and the Siberian school of informatics at Novosibirsk. Strong theoretical foundations in programming semantics and mixed computation (partial evaluation). Ershov's vision of programs as mathematical objects aligns with the DRS philosophy that software claims should be treated with the same rigor as scientific claims.

**GOST standards.** Soviet and Russian state standards for software documentation and quality (GOST 19.XXX series, GOST 34.XXX series). Process-oriented — they specify *what documentation to produce*, not how to make it machine-verifiable.

### 4.7 Japanese Tradition

**Kaizen** (Imai, 1986). Continuous improvement philosophy. The Meta-Kaizen series name derives from this tradition — "meta" because the framework improves the improvement process itself.

**Monozukuri.** The philosophy of craftsmanship in making. Extended to software via quality circles and the software factory model (Cusumano, 1991; Yasuda, 1989). Japanese software quality assurance emphasizes process discipline and cultural commitment but does not formalize behavioral claims as machine-readable predicates. The DRS formalizes what kaizen leaves implicit: instead of a cultural commitment to quality, a machine-auditable registry of what "quality" means in terms of specific, falsifiable claims.

**JUSE** (Union of Japanese Scientists and Engineers). Administers the Deming Prize and promotes statistical quality control. Quality circles generate improvements but do not produce machine-readable verification artifacts.

### 4.8 Chinese Tradition

**CertiKOS** (Zhong Shao, Yale/ECNU collaboration, 2016). A formally verified concurrent OS kernel. **ORIENTAIS** (East China Normal University). A formally verified RTOS based on OSEK/VDX. Both are full formal verification — powerful but specialized.

**GB/T standards** (Guobiao). Chinese national standards, many adopted from ISO/IEC. GB/T 15532 (software testing), GB/T 8567 (software documentation). Process-oriented like their ISO counterparts; no machine-readable falsification standard.

### 4.9 Indian Tradition

India has the highest concentration of CMMI Level 5 organizations globally. STQC (Standardization, Testing, and Quality Certification) provides conformity assessment. These are *process maturity models* — they audit whether the organization follows good processes, not whether individual software claims have been verified.

### 4.10 Safety-Critical Standards

**DO-178C** (avionics). Defines software levels (A through E) based on failure impact. Requires objective evidence linking requirements to tests. The closest existing standard in spirit to the DRS — but evidence is prose-based and assessment is manual.

**IEC 62304** (medical device software). Lifecycle process standard. Requires traceability from software requirements to verification activities.

**ISO 26262** (automotive). Addressed in Section 4.5.

All three require traceability but none specify deterministic evaluation semantics for verification evidence. The DRS provides a machine-readable substrate for the evidence these standards require.

### 4.11 ISO/IEC Software Standards

**ISO/IEC 29119** (software testing). Comprehensive testing standard but does not define claim registries or falsification predicates.

**ISO/IEC 25010** (software quality model). Defines quality characteristics (reliability, security, etc.) but at the abstract level — no machine-readable claim specification.

**ISO/IEC 12207** (software lifecycle). Process standard. Defines what activities to perform, not how to make their outputs machine-verifiable.

**ISO/IEC 15026** (systems and software assurance). Closest ISO standard to the DRS — addresses assurance cases and evidence. But assurance cases are typically textual or graphical (GSN, CAE), not deterministic predicates.

### 4.12 The Gap — Stated Precisely

No tradition — in any language, in any standard, in any framework — has produced the specific combination that the DRS for Software provides:

1. **A machine-readable claim registry** classifying every software behavioral claim by epistemological type (Assumption / Definition / Falsifiable)
2. **Deterministic falsification predicates** with explicit operands, evaluation procedures, boundary documentation, and threshold justification for every Type F claim
3. **Honest placeholder tracking** that makes unverified claims visible rather than invisible
4. **A phase-ready verdict** that provides a binary, auditable assessment of verification completeness

Individual elements exist in various traditions. The integration is new.

## 5. The Software Claim Taxonomy

### 5.1 Three Claim Types

We extend the DRS claim taxonomy to software. The mapping preserves the original semantics:

| Type | Scientific (Papers 1–5) | Software (This Paper) | Falsifiable? |
|------|------------------------|----------------------|-------------|
| **A** | Axioms: premises accepted from literature | **Assumptions**: platform requirements, dependency contracts, environmental preconditions | No — accepted as given |
| **D** | Definitions: stipulative terms | **Definitions**: type signatures, data structures, configuration schemas, enums | No — stipulative |
| **F** | Falsifiable claims: theorems, empirical predictions | **Behavioral claims**: correctness guarantees, invariants, performance bounds, API contracts | Yes — full K = (P,O,M,B) required |

### 5.2 Type A — Software Assumptions

Software axioms are the preconditions the system assumes but does not verify internally:

- **Platform assumptions**: "Requires Python >= 3.9" (the software does not verify the interpreter version at every function call)
- **Dependency contracts**: "numpy.fft.rfft returns the discrete Fourier transform of a real sequence" (the software trusts numpy's correctness)
- **Environmental preconditions**: "Input data arrives as a sequence of finite floating-point values" (the software does not verify IEEE 754 compliance)

Like scientific axioms, software assumptions are not falsifiable within the system — they are the foundation on which falsifiable claims rest. When an assumption changes (a dependency releases a breaking update, a platform drops support), the assumption shifts and every downstream Type F claim must be re-evaluated. The claim registry makes this dependency explicit.

This is a *behavioral* Software Bill of Materials (SBOM). Where a conventional SBOM lists what libraries are present, a Type A registry lists what behavioral assumptions the software makes about them. When a dependency changes, an SBOM tells you *what updated*; a Type A registry tells you *what claims are now at risk*.

### 5.3 Type D — Software Definitions

Software definitions are structural specifications that are stipulative, not truth-apt:

- **Type signatures**: `def update(self, value: float) -> SentinelResult`
- **Data structures**: `SentinelResult` contains fields `score`, `alert_type`, `reasons`
- **Configuration schemas**: `SentinelConfig` has field `window_size: int` with default 50
- **Enums and constants**: `AlertType` has values `NORMAL`, `WARNING`, `CRITICAL`, `CASCADE_PRECURSOR`

Definitions carry `falsification_predicate: null` — they cannot be falsified because they are what we stipulate, not what we claim about behavior.

### 5.4 Type F — Software Behavioral Claims

These are the claims that matter. Every Type F claim carries the full falsification kernel K = (P, O, M, B):

**Example: Sentinel score range claim**

```json
{
  "claim_id": "F-SW.1",
  "type": "F",
  "name": "Score boundedness",
  "statement": "SentinelResult.score is always in [0.0, 1.0] for any finite input.",
  "falsification_predicate": {
    "FALSIFIED_IF": "EXISTS x IN test_vectors SUCH THAT sentinel.update(x).score < 0.0 OR sentinel.update(x).score > 1.0",
    "WHERE": {
      "x": "scalar · dimensionless · any finite IEEE 754 float64",
      "test_vectors": "set · dimensionless · comprehensive input set including edge cases: 0.0, -0.0, 1e-308, 1e308, NaN-adjacent finite values",
      "sentinel": "object · Sentinel instance · default SentinelConfig"
    },
    "EVALUATION": "Instantiate Sentinel with default config; call update(x) for each x in test_vectors; check score bounds; finite",
    "BOUNDARY": "score = 0.0 → NOT FALSIFIED (inclusive lower bound); score = 1.0 → NOT FALSIFIED (inclusive upper bound)",
    "CONTEXT": "Bounds [0.0, 1.0] derive from the pipeline's final normalization step (Step 37). All intermediate scores are clipped before output."
  },
  "sample_falsification_observation": "sentinel.update(1e308).score = 1.03 would FALSIFY this claim."
}
```

The structure is identical to scientific claim predicates. The only difference is that the EVALUATION field can reference executable tests rather than manual procedures — and this is a strength, not a deviation, because it makes evaluation fully automated.

### 5.5 Relationship to Design by Contract

The DRS for Software extends Design by Contract (Meyer, 1992), not replaces it:

| Aspect | Design by Contract | DRS for Software |
|--------|-------------------|-----------------|
| **Scope** | Per-function (local) | Per-system (global registry) |
| **Assertion** | Precondition/postcondition | FALSIFIED_IF predicate |
| **Variables** | Implicit (function parameters) | Explicit (WHERE defines every operand) |
| **Threshold justification** | None required | BOUNDARY + CONTEXT mandatory |
| **Missing contracts** | Invisible (no annotation = unknown) | Visible (placeholder: true) |
| **Evaluation** | Runtime check or static analysis | Documented finite procedure |

A software system can and should have both: DbC contracts embedded in code *and* a DRS claim registry that audits the system's promises at a higher level. The registry can reference DbC contracts as evaluation evidence.

## 6. Positioning: The Tractability Gap

The DRS for Software occupies a specific position in the rigor-tractability spectrum:

```
Informal Testing ──── DRS for Software ──── Formal Verification
(most software)        (this paper)          (safety-critical)

Low rigor              Medium rigor           Maximum rigor
Low cost               Medium cost            High cost
No claim accounting    Full claim accounting  Full proof
Gaps invisible         Gaps visible           No gaps (in scope)
Any language           Any language           Specialized languages
```

**Informal testing** (pytest, JUnit, manual QA) is what most software uses. It verifies behavior for specific inputs but does not enumerate claims, document boundaries, or track gaps.

**Formal verification** (B-Method, Coq, Lean, SPARK Ada) proves properties for all inputs. It is the gold standard for safety-critical systems (Paris Métro, CompCert, seL4). It requires specialized languages, extensive training, and substantial investment.

**The DRS for Software** occupies the middle ground. It does not prove correctness — it systematically accounts for what the software claims, which claims have survived falsification, and which claims remain unverified. This is the appropriate level of rigor for the vast majority of software that will never receive formal verification.

The DRS is complementary to both:
- For informally tested software, the DRS adds the claim accounting that testing lacks
- For formally verified software, the DRS adds the claim enumeration that even formal methods need (you must decide *what* to prove before you prove it)

## 7. Formal Properties of the Framework

The following properties formalize intuitions that practitioners hold informally. Their value is precision, not mathematical depth.

### 7.1 Definitions

**Definition 7.1 (Claim Registry).** A software claim registry R is a finite set of claims {c_1, ..., c_n} where each c_i is classified as Type A, Type D, or Type F, and each Type F claim carries a well-formed falsification kernel K_i = (P_i, O_i, M_i, B_i).

**Definition 7.2 (Falsification Completeness).** A software system S with claim registry R is *falsification-complete* if and only if:
1. Every publicly documented behavior of S corresponds to at least one Type F claim in R
2. Every Type F claim in R has a well-formed predicate (per the Falsification Kernel v1.1 specification)
3. Every Type F claim evaluates to NOT FALSIFIED under current evaluation
4. Every unverified behavioral claim is registered as a placeholder with `placeholder: true`

**Definition 7.3 (Code Coverage).** Line coverage L(S, T) is the fraction of executable source lines in system S that are executed by test suite T. Branch coverage B(S, T) is the fraction of conditional branches exercised.

### 7.2 Properties

**Property 7.1 (Falsification Completeness Implies Coverage).** If a software system S is falsification-complete with respect to claim registry R, then there exists a test suite T derived from the evaluation procedures in R such that L(S, T) >= L(S, T') for any test suite T' that achieves the same behavioral verification.

*Argument.* Each Type F claim's EVALUATION field specifies a finite procedure that exercises the code path implementing the claimed behavior. The union of all evaluation procedures constitutes a test suite. Since falsification completeness requires every publicly documented behavior to have a claim, and every claim to have an evaluation, the resulting test suite exercises every code path that implements a documented behavior. Lines not covered are either dead code or implementation details exercised by necessity.

**Property 7.2 (Coverage Does Not Imply Falsification Completeness).** There exist software systems with L(S, T) = 1.0 (100% line coverage) that are not falsification-complete.

*Construction.* Consider a function `f(x) = x + 1` with test `assert f(2) == 3`. Line coverage is 100%. But the behavioral claim "f returns its input incremented by 1 for all integers" is not verified — the test checks one input. A falsification-complete registry would require `FALSIFIED_IF EXISTS x IN Z SUCH THAT f(x) != x + 1` with an evaluation procedure that tests representative inputs including boundary cases.

**Property 7.3 (Assumption Propagation).** If assumption A_k in claim registry R is invalidated (e.g., a dependency contract changes), then every Type F claim whose derivation_source includes A_k must be re-evaluated. The claim registry makes this set computable in O(|R|) time via graph traversal of the derivation_source dependency edges.

### 7.3 The Phase-Ready Verdict for Software Releases

The existing DRS defines PHASE-READY as a binary verdict: a paper's AI layer is phase-ready when all conditions (c1–c6) are satisfied. We extend this to software releases:

A software release is **PHASE-READY** if and only if:
- **c1**: Every public API function has at least one Type F claim in the registry
- **c2**: Every Type F claim has a well-formed K = (P, O, M, B) predicate
- **c3**: Every Type F evaluation procedure passes (NOT FALSIFIED)
- **c4**: Cross-references to other claims or external dependencies are resolved or registered as placeholders
- **c5**: Every placeholder is documented with `blocks_phase_ready` status
- **c6**: The claim registry is valid against the schema

This replaces the informal "all tests pass" with a formal "all *claims* are verified or honestly marked as unverified."

## 8. What DRS Reveals That Tests Cannot

### 8.1 Three Categories of Gap

Applying the framework to an existing software system reveals three categories of gap invisible to conventional testing:

**Category 1: Tested but unbounded.** A test verifies behavior for specific inputs but the claim's BOUNDARY field reveals that edge cases are untested. Example: a test checks that `score` is in [0, 1] for typical inputs but never tests the boundary at exactly 0.0 or 1.0, or at extreme input values. The test passes; the claim's boundary documentation exposes the gap.

**Category 2: Documented but untested.** The README or docstring makes a behavioral promise that has no corresponding Type F claim. Example: "Sentinel detects regime changes within 5 observations" — a claim made in documentation but with no test that specifically verifies detection latency. Conventional coverage cannot find this because coverage measures code executed, not promises made.

**Category 3: Assumed but unregistered.** The software depends on external behavior it does not document. Example: the pipeline assumes numpy's FFT implementation is correct, but this assumption is nowhere in the codebase. When numpy changes behavior (as happened in numpy 2.0), the developer has no systematic way to identify which claims are affected. A Type A registry makes these dependencies explicit.

### 8.2 The Honest Placeholder

The most important contribution may be the simplest: the placeholder. When a software system registers a behavioral claim as `placeholder: true`, it is making an honest statement: "we claim this behavior but have not yet verified it." This is strictly more informative than the current state, where unverified claims are indistinguishable from verified ones.

### 8.3 Comparison: Requirements Traceability vs. DRS

Safety-critical standards (DO-178C, IEC 62304, ISO 26262) require requirements traceability — linking requirements to verification evidence. The DRS extends this:

| Aspect | Requirements Traceability | DRS for Software |
|--------|--------------------------|-----------------|
| **Requirement format** | Natural language (prose) | Deterministic predicate (FALSIFIED_IF) |
| **Verification evidence** | Test report reference | EVALUATION procedure (executable) |
| **Edge case documentation** | Optional or implicit | BOUNDARY field (mandatory) |
| **Threshold justification** | Rarely required | CONTEXT field (mandatory) |
| **Claim classification** | All "requirements" | Differentiated: A (assumption) / D (definition) / F (falsifiable) |
| **Machine evaluation** | Manual review | Deterministic: TRUE/FALSE verdict |

The DRS provides a machine-readable substrate for the evidence that safety-critical standards already require.

## 9. Feasibility Demonstration: Fracttalix Sentinel v12.1

### 9.1 Scope and Limitations

We apply the framework to Fracttalix Sentinel v12.1, a 37-step streaming anomaly detector with 374 existing unit tests. This is a *feasibility demonstration*, not an empirical validation: we demonstrate that the DRS can be applied to a real codebase and that it reveals gaps invisible to conventional metrics. The fact that the authors also wrote the software means this demonstration proves applicability but not independent value — that requires the empirical agenda in Section 11.

### 9.2 Results

The demonstration produces a software AI layer (see companion artifact: `ai-layers/SFW1-ai-layer-v2.json`) containing registered claims across the full pipeline.

The existing test suite achieves high code coverage. The claim registry reveals:

1. **Documented but untested claims** (Category 2 gaps): behavioral promises made in README or docstrings that have no corresponding Type F claim with a passing evaluation
2. **Tested but unbounded claims** (Category 1 gaps): tests that verify behavior for typical inputs but lack explicit boundary documentation for edge cases
3. **Assumed but unregistered dependencies** (Category 3 gaps): behavioral contracts assumed about numpy and other optional dependencies that are not documented in any Type A claim

### 9.3 Phase-Ready Assessment

The phase-ready assessment for Sentinel v12.1 under the software DRS is determined by evaluating c1–c6 against the claim registry. Placeholder claims and unbounded claims that remain after the audit determine the verdict. This is not a failure — it is an honest accounting that was previously invisible.

## 10. Implications

### 10.1 For Open Source Trust

Open source software is currently trusted on the basis of social signals: GitHub stars, contributor count, corporate backing, maintenance activity. None of these are behavioral claims. A DRS-conformant project publishes a machine-readable registry of exactly what it claims to do and exactly how each claim can be falsified. Trust shifts from social reputation to auditable evidence.

### 10.2 For Dependency Management

When a library publishes a software AI layer, downstream consumers can programmatically determine which of their own claims depend on which upstream assumptions. Dependency updates become claim-aware: "numpy 2.1 invalidates assumptions A-SW.3 and A-SW.7, which affect claims F-SW.12, F-SW.23, and F-SW.31." This is a behavioral dependency graph, not just a version dependency graph.

### 10.3 For AI-Generated Code

As AI systems generate increasing amounts of software, the question "what does this code claim to do?" becomes critical. AI-generated code with a DRS layer is auditable: every behavioral claim is registered, and any claim the AI could not verify is marked as a placeholder. This is strictly more honest than the current state, where AI-generated code may or may not work as intended and the user has no systematic way to check.

### 10.4 For Regulatory Compliance

Industries subject to software validation requirements (medical devices under IEC 62304, automotive under ISO 26262, financial systems under SOX) currently produce validation documentation manually. A software AI layer is machine-readable validation evidence: every requirement traces to a claim, every claim traces to a falsification procedure, and the phase-ready verdict provides a binary compliance signal.

### 10.5 Applicability Boundary

The DRS for Software is most valuable for **libraries and frameworks consumed by others** — where the cost of undocumented assumptions is highest and the benefit of a machine-readable claim registry extends beyond the original authors.

Secondary value exists for **safety-critical systems** — where the DRS provides a machine-readable substrate for the evidence that existing standards already require.

Lower value exists for **internal scripts and one-off tools** — where the overhead of maintaining a claim registry exceeds the benefit. The overhead/value ratio determines applicability, not a universal mandate.

## 11. Empirical Agenda

This paper presents a framework and a feasibility demonstration. Prospective empirical validation requires:

1. **Independent application.** Apply the DRS to 3+ open-source projects not authored by the framework developers. Measure: time to produce a claim registry, number and categories of gaps discovered.

2. **Gap detection comparison.** For each project, compare gaps discovered by the DRS against gaps discoverable by conventional coverage metrics (line coverage, branch coverage, mutation testing). Measure: does the DRS discover at least one gap category invisible to coverage metrics?

3. **Maintenance overhead.** Track the ongoing cost of maintaining a claim registry across multiple release cycles. Measure: person-hours per release to update the registry vs. defects caught by the registry before release.

4. **Third-party audit.** Provide a software AI layer to a team unfamiliar with the project. Measure: can the team evaluate the phase-ready verdict independently? Does the registry accelerate their understanding of the software's behavioral surface?

## 12. Limitations

1. **Registry completeness is not verifiable.** There is no algorithm to determine whether all behavioral claims have been enumerated — the claim registry depends on human (or AI) judgment about what the software promises. This is the same epistemological situation as scientific papers: no algorithm can verify that a paper has stated all its claims. An incomplete registry is still more informative than no registry, and the placeholder mechanism enables incremental improvement.

2. **Falsification is not verification.** A claim that has survived all falsification attempts is not proven correct. The DRS adopts the Popperian stance: we can falsify but not verify. This is deliberately weaker than formal verification — and deliberately more tractable. For software that can receive formal verification (safety-critical, bounded scope, well-funded), formal methods remain preferable. The DRS serves the vast majority of software that cannot.

3. **Schema overhead.** Maintaining a claim registry adds maintenance burden. The framework is most valuable for libraries and frameworks consumed by others, where the cost of undocumented assumptions is highest.

4. **Single feasibility demonstration.** This paper presents one demonstration on the authors' own software. Independent empirical validation is the next step (Section 11).

## 13. Conclusion

The Dual Reader Standard was designed to make scientific papers machine-verifiable. This paper shows the same framework — the same three claim types, the same falsification kernel K = (P, O, M, B), the same phase-ready verdict, the same honest placeholder mechanism — applies to software without modification.

The contribution is not testing. Testing is necessary but not sufficient. The contribution is *systematic falsification accounting*: an auditable registry of what the software claims, how each claim can be falsified, which claims have survived falsification, and which claims remain unverified. The honest placeholder — the simple act of saying "we claim this but haven't tested it" — may be the most practically valuable element, because it transforms invisible gaps into visible ones.

Software that publishes a DRS layer is not claiming to be correct. It is claiming to be *honest about what it has and has not verified*. That is a standard the field currently lacks.

## References

Abrial, J.-R. (1996). *The B-Book: Assigning Programs to Meanings.* Cambridge University Press.

Claessen, K. & Hughes, J. (2000). QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs. *ICFP*.

Cusumano, M. A. (1991). *Japan's Software Factories: A Challenge to U.S. Management.* Oxford University Press.

Dahl, O.-J. & Nygaard, K. (1966). SIMULA — an ALGOL-based simulation language. *Communications of the ACM*, 9(9), 671–678.

Dijkstra, E. W. (1968). Go To Statement Considered Harmful. *Communications of the ACM*, 11(3), 147–148.

Dijkstra, E. W. (1976). *A Discipline of Programming.* Prentice-Hall.

Ershov, A. P. (1972). Aesthetics and the Human Factor in Programming. *Kibernetika*, 5.

Imai, M. (1986). *Kaizen: The Key to Japan's Competitive Success.* McGraw-Hill.

Jackson, D. (2002). Alloy: A Lightweight Object Modelling Notation. *ACM Transactions on Software Engineering and Methodology*, 11(2), 256–290.

Jones, C. B. (1990). *Systematic Software Development Using VDM.* 2nd ed. Prentice-Hall.

Krantz, D. H., Luce, R. D., Suppes, P. & Tversky, A. (1971). *Foundations of Measurement, Vol. I.* Academic Press.

Lamport, L. (1994). *The Temporal Logic of Actions.* ACM Transactions on Programming Languages and Systems, 16(3), 872–923.

Leroy, X. (2009). Formal Verification of a Realistic Compiler. *Communications of the ACM*, 52(7), 107–115.

Leucker, M. & Schallhart, C. (2009). A Brief Account of Runtime Verification. *Journal of Logic and Algebraic Programming*, 78(5), 293–303.

Luce, R. D. & Tukey, J. W. (1964). Simultaneous conjoint measurement. *Journal of Mathematical Psychology*, 1, 1–27.

Meyer, B. (1992). Applying "Design by Contract." *IEEE Computer*, 25(10), 40–51.

Popper, K. R. (1959). *The Logic of Scientific Discovery.* Hutchinson.

Spivey, J. M. (1989). *The Z Notation: A Reference Manual.* Prentice-Hall.

Yasuda, K. (1989). Software Quality Assurance Activities in Japan. In *Japanese Perspectives in Software Engineering*, Addison-Wesley.

Brennan, T. (2026). Meta-Kaizen: A General Theory and Algorithmic Framework. *Fracttalix Corpus*, MK-P1.

Brennan, T. (2026). Falsification Kernel v1.1 — Layer 0 Semantic Specification. *Fracttalix Corpus*.

Brennan, T. (2026). On the Decision to Act: Strategic Convergence and the Mathematics of Intervention Timing at System Tipping Points. *Fracttalix Corpus*, MK-P5.
