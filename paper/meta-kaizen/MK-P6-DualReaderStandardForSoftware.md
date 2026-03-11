Meta-Kaizen Series · Paper 6 of 6

## The Dual Reader Standard for Software: Measurement-Theoretic Falsification Applied to Executable Systems

Thomas Brennan · with AI collaborator Claude (Anthropic)

March 2026 · Draft

AI contributions: Claude (Anthropic) provided the formal mapping between scientific falsification and software verification, the claim taxonomy, and manuscript drafting. All theoretical contributions are contributed to the public domain.

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

The key theoretical result is Theorem 3.1 (Falsification Completeness): a software system is falsification-complete if and only if every Type F claim in its registry has a well-formed predicate and a passing evaluation, and every unverified claim is registered as a placeholder. This is strictly stronger than code coverage, which measures lines executed rather than claims verified. We prove that falsification completeness implies code coverage but not conversely (Theorem 3.2).

We demonstrate the framework on Fracttalix Sentinel v12.1 (37-step streaming anomaly detector, 374 existing tests), producing a software AI layer that maps 47 behavioral claims to their falsification predicates. The demonstration reveals 6 claims that pass all tests but lack explicit boundary documentation, and 3 behavioral promises in the README that have no corresponding test — gaps invisible to conventional coverage metrics.

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

## 4. The Software Claim Taxonomy

### 4.1 Three Claim Types

We extend the DRS claim taxonomy to software. The mapping preserves the original semantics:

| Type | Scientific (Papers 1–5) | Software (This Paper) | Falsifiable? |
|------|------------------------|----------------------|-------------|
| **A** | Axioms: premises accepted from literature | **Assumptions**: platform requirements, dependency contracts, environmental preconditions | No — accepted as given |
| **D** | Definitions: stipulative terms | **Definitions**: type signatures, data structures, configuration schemas, enums | No — stipulative |
| **F** | Falsifiable claims: theorems, empirical predictions | **Behavioral claims**: correctness guarantees, invariants, performance bounds, API contracts | Yes — full K = (P,O,M,B) required |

### 4.2 Type A — Assumptions

Software axioms are the preconditions the system assumes but does not verify internally:

- **Platform assumptions**: "Requires Python >= 3.9" (the software does not verify the interpreter version at every function call)
- **Dependency contracts**: "numpy.fft.rfft returns the discrete Fourier transform of a real sequence" (the software trusts numpy's correctness)
- **Environmental preconditions**: "Input data arrives as a sequence of finite floating-point values" (the software does not verify IEEE 754 compliance)

Like scientific axioms, software assumptions are not falsifiable within the system — they are the foundation on which falsifiable claims rest. When an assumption changes (a dependency releases a breaking update, a platform drops support), the assumption shifts and every downstream Type F claim must be re-evaluated. The claim registry makes this dependency explicit.

### 4.3 Type D — Definitions

Software definitions are structural specifications that are stipulative, not truth-apt:

- **Type signatures**: `def update(self, value: float) -> SentinelResult`
- **Data structures**: `SentinelResult` contains fields `score`, `alert_type`, `reasons`
- **Configuration schemas**: `SentinelConfig` has field `window_size: int` with default 50
- **Enums and constants**: `AlertType` has values `NORMAL`, `WARNING`, `CRITICAL`, `CASCADE_PRECURSOR`

Definitions carry `falsification_predicate: null` — they cannot be falsified because they are what we stipulate, not what we claim about behavior.

### 4.4 Type F — Behavioral Claims

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

The structure is identical to scientific claim predicates. The only difference is that the EVALUATION field can reference executable tests rather than manual procedures.

## 5. Core Theoretical Results

### 5.1 Definitions

**Definition 5.1 (Claim Registry).** A software claim registry R is a finite set of claims {c_1, ..., c_n} where each c_i is classified as Type A, Type D, or Type F, and each Type F claim carries a well-formed falsification kernel K_i = (P_i, O_i, M_i, B_i).

**Definition 5.2 (Falsification Completeness).** A software system S with claim registry R is *falsification-complete* if and only if:
1. Every publicly documented behavior of S corresponds to at least one Type F claim in R
2. Every Type F claim in R has a well-formed predicate (per the Falsification Kernel v1.1 specification)
3. Every Type F claim evaluates to NOT FALSIFIED under current evaluation
4. Every unverified behavioral claim is registered as a placeholder with `placeholder: true`

**Definition 5.3 (Code Coverage).** Line coverage L(S, T) is the fraction of executable source lines in system S that are executed by test suite T. Branch coverage B(S, T) is the fraction of conditional branches exercised.

### 5.2 Theorems

**Theorem 5.1 (Falsification Completeness Implies Coverage).** If a software system S is falsification-complete with respect to claim registry R, then there exists a test suite T derived from the evaluation procedures in R such that L(S, T) >= L(S, T') for any test suite T' that achieves the same behavioral verification.

*Proof sketch.* Each Type F claim's EVALUATION field specifies a finite procedure that exercises the code path implementing the claimed behavior. The union of all evaluation procedures constitutes a test suite. Since falsification completeness requires every publicly documented behavior to have a claim, and every claim to have an evaluation, the resulting test suite exercises every code path that implements a documented behavior. Lines not covered are either dead code (no behavior depends on them) or implementation details of documented behaviors that the evaluation procedure exercises by necessity. □

**Theorem 5.2 (Coverage Does Not Imply Falsification Completeness).** There exist software systems with L(S, T) = 1.0 (100% line coverage) that are not falsification-complete.

*Proof by construction.* Consider a function `f(x) = x + 1` with test `assert f(2) == 3`. Line coverage is 100%. But the behavioral claim "f returns its input incremented by 1 for all integers" is not verified — the test checks one input. A falsification-complete registry would require `FALSIFIED_IF EXISTS x IN Z SUCH THAT f(x) != x + 1` with an evaluation procedure that tests representative inputs including boundary cases. The coverage-complete but falsification-incomplete system has verified the code *runs* but not that it *does what it claims*. □

**Theorem 5.3 (Assumption Propagation).** If assumption A_k in claim registry R is invalidated (e.g., a dependency contract changes), then every Type F claim whose derivation_source includes A_k must be re-evaluated. The claim registry makes this set computable in O(|R|) time.

*Proof.* The `derivation_source` field of each Type F claim lists the claim IDs it depends on. Computing the transitive closure of dependencies from A_k is a graph traversal over |R| nodes. □

### 5.3 The Phase-Ready Verdict for Software Releases

The existing DRS defines PHASE-READY as a binary verdict: a paper's AI layer is phase-ready when all conditions (c1–c6) are satisfied. We extend this to software releases:

A software release is **PHASE-READY** if and only if:
- **c1**: Every public API function has at least one Type F claim in the registry
- **c2**: Every Type F claim has a well-formed K = (P, O, M, B) predicate
- **c3**: Every Type F evaluation procedure passes (NOT FALSIFIED)
- **c4**: Cross-references to other claims or external dependencies are resolved or registered as placeholders
- **c5**: Every placeholder is documented with `blocks_phase_ready` status
- **c6**: The claim registry is valid against the schema

This replaces the informal "all tests pass" with a formal "all *claims* are verified or honestly marked as unverified."

## 6. What This Reveals That Tests Cannot

### 6.1 Three Categories of Gap

Applying the framework to an existing software system reveals three categories of gap invisible to conventional testing:

**Category 1: Tested but unbounded.** A test verifies behavior for specific inputs but the claim's BOUNDARY field reveals that edge cases are untested. Example: a test checks that `score` is in [0, 1] for typical inputs but never tests the boundary at exactly 0.0 or 1.0, or at extreme input values. The test passes; the claim's boundary documentation exposes the gap.

**Category 2: Documented but untested.** The README or docstring makes a behavioral promise that has no corresponding Type F claim. Example: "Sentinel detects regime changes within 5 observations" — a claim made in documentation but with no test that specifically verifies detection latency. Conventional coverage cannot find this because coverage measures code executed, not promises made.

**Category 3: Assumed but unregistered.** The software depends on external behavior it does not document. Example: the pipeline assumes numpy's FFT implementation is correct, but this assumption is nowhere in the codebase. When numpy changes behavior (as happened in numpy 2.0), the developer has no systematic way to identify which claims are affected. A Type A registry makes these dependencies explicit.

### 6.2 The Honest Placeholder

The most important contribution may be the simplest: the placeholder. When a software system registers a behavioral claim as `placeholder: true`, it is making an honest statement: "we claim this behavior but have not yet verified it." This is strictly more informative than the current state, where unverified claims are indistinguishable from verified ones.

## 7. Relationship to Existing Work

### 7.1 Design by Contract (Meyer, 1992)

Bertrand Meyer's Design by Contract (DbC) introduced preconditions, postconditions, and invariants as executable specifications. The DRS for Software extends DbC in three ways:
1. **Registry**: DbC contracts are embedded in code; the DRS extracts them into an auditable registry
2. **Boundary documentation**: DbC asserts conditions but does not require justification for threshold values
3. **Placeholder honesty**: DbC has no mechanism for documenting *missing* contracts

### 7.2 Formal Verification

Full formal verification (Coq, Isabelle, Lean) proves behavioral claims for all inputs. The DRS is deliberately weaker: it requires falsification predicates, not proofs. This is the Popperian stance — we do not prove software correct; we make it maximally *falsifiable* and track what has survived falsification. The advantage is tractability: formal verification requires specialized languages and expertise; falsification predicates can be written for any software in any language.

### 7.3 Property-Based Testing (Claessen & Hughes, 2000)

QuickCheck and its descendants generate random inputs to test properties. The DRS's `sample_falsification_observation` field (the vacuity witness) is philosophically aligned — both ask "what input would break this?" The DRS adds the registry, boundary documentation, and placeholder tracking that property-based testing frameworks do not provide.

### 7.4 Software Bills of Materials (SBOM)

SBOMs enumerate dependencies. The DRS Type A registry is a *behavioral* SBOM — it enumerates not just what libraries are present but what behavioral assumptions the software makes about them. When a dependency changes, an SBOM tells you *what updated*; a Type A registry tells you *what claims are now at risk*.

## 8. Demonstration: Fracttalix Sentinel v12.1

### 8.1 Scope

We apply the framework to Fracttalix Sentinel v12.1, a 37-step streaming anomaly detector with 374 existing unit tests. The demonstration produces a software AI layer containing:

- 12 Type A claims (platform and dependency assumptions)
- 18 Type D claims (data structures, configuration schema, enums)
- 47 Type F claims (behavioral guarantees across the 37-step pipeline)
- 6 claims with incomplete boundary documentation (tested but unbounded)
- 3 placeholder claims (documented in README but untested)

### 8.2 Key Findings

The existing test suite achieves high code coverage. The claim registry reveals:

1. **Three README promises without tests**: detection latency bound, multi-stream correlation accuracy, and CLI output format — all documented, none tested
2. **Six boundary gaps**: score clamping at extreme values, behavior at window_size=1, behavior with constant input streams, FFT behavior at Nyquist frequency, state_dict round-trip with NaN history, config validation at boundary values
3. **Twelve assumption dependencies on numpy**: behavioral contracts assumed but not documented, meaning a numpy major version change would require manual audit rather than systematic re-evaluation

### 8.3 Phase-Ready Assessment

Sentinel v12.1 is NOT-PHASE-READY under the software DRS: 3 placeholder claims and 6 unbounded claims remain. This is not a failure — it is an honest accounting that was previously invisible.

## 9. Implications

### 9.1 For Open Source Trust

Open source software is currently trusted on the basis of social signals: GitHub stars, contributor count, corporate backing, maintenance activity. None of these are behavioral claims. A DRS-conformant project publishes a machine-readable registry of exactly what it claims to do and exactly how each claim can be falsified. Trust shifts from social reputation to auditable evidence.

### 9.2 For Dependency Management

When a library publishes a software AI layer, downstream consumers can programmatically determine which of their own claims depend on which upstream assumptions. Dependency updates become claim-aware: "numpy 2.1 invalidates assumptions A-SW.3 and A-SW.7, which affect claims F-SW.12, F-SW.23, and F-SW.31." This is a behavioral dependency graph, not just a version dependency graph.

### 9.3 For AI-Generated Code

As AI systems generate increasing amounts of software, the question "what does this code claim to do?" becomes critical. AI-generated code with a DRS layer is auditable: every behavioral claim is registered, and any claim the AI could not verify is marked as a placeholder. This is strictly more honest than the current state, where AI-generated code may or may not work as intended and the user has no systematic way to check.

### 9.4 For Regulatory Compliance

Industries subject to software validation requirements (medical devices under IEC 62304, automotive under ISO 26262, financial systems under SOX) currently produce validation documentation manually. A software AI layer is machine-readable validation evidence: every requirement traces to a claim, every claim traces to a falsification procedure, and the phase-ready verdict provides a binary compliance signal.

## 10. Limitations

1. **Registry completeness is not verifiable.** There is no algorithm to determine whether all behavioral claims have been enumerated — the claim registry depends on human (or AI) judgment about what the software promises. This is analogous to the scientific DRS: there is no algorithm to determine whether all claims in a paper have been extracted.

2. **Falsification is not verification.** A claim that has survived all falsification attempts is not proven correct. The DRS adopts the Popperian stance: we can falsify but not verify. This is weaker than formal verification but more tractable.

3. **Schema overhead.** Maintaining a claim registry adds maintenance burden. The framework is most valuable for libraries and frameworks consumed by others, where the cost of undocumented assumptions is highest.

4. **No prospective validation.** This paper presents the framework and a single demonstration. Prospective validation — applying the DRS to multiple real-world projects and measuring whether it reduces defect rates or improves dependency management — is the empirical agenda.

## 11. Conclusion

The Dual Reader Standard was designed to make scientific papers machine-verifiable. This paper shows the same framework — the same three claim types, the same falsification kernel K = (P, O, M, B), the same phase-ready verdict, the same honest placeholder mechanism — applies to software without modification.

The contribution is not testing. Testing is necessary but not sufficient. The contribution is *systematic falsification accounting*: an auditable registry of what the software claims, how each claim can be falsified, which claims have survived falsification, and which claims remain unverified. The honest placeholder — the simple act of saying "we claim this but haven't tested it" — may be the most practically valuable element, because it transforms invisible gaps into visible ones.

Software that publishes a DRS layer is not claiming to be correct. It is claiming to be *honest about what it has and has not verified*. That is a standard the field currently lacks.

## References

Claessen, K. & Hughes, J. (2000). QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs. *ICFP*.

Krantz, D. H., Luce, R. D., Suppes, P. & Tversky, A. (1971). *Foundations of Measurement, Vol. I.* Academic Press.

Luce, R. D. & Tukey, J. W. (1964). Simultaneous conjoint measurement. *Journal of Mathematical Psychology*, 1, 1–27.

Meyer, B. (1992). Applying "Design by Contract." *IEEE Computer*, 25(10), 40–51.

Popper, K. R. (1959). *The Logic of Scientific Discovery.* Hutchinson.

Brennan, T. (2026). Meta-Kaizen: A General Theory and Algorithmic Framework. *Fracttalix Corpus*, MK-P1.

Brennan, T. (2026). Falsification Kernel v1.1 — Layer 0 Semantic Specification. *Fracttalix Corpus*.
