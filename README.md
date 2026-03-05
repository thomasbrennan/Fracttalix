# Meta-Kaizen

**A General Theory and Algorithmic Framework for the Mathematical Formalization of Self-Evolving Continuous Improvement Across Arbitrary Governance Substrates**

Brennan, Entwood & Claude (Anthropic) — March 2026

-----

## What This Is

Kaizen — the philosophy of continuous incremental improvement — has been practiced across manufacturing, healthcare, software engineering, and public governance for nearly four decades. It has never been formally mathematized. No substrate-agnostic, algorithmically reproducible, self-referential framework has existed for scoring, ranking, and governing improvement candidates across arbitrary structured processes or documents.

This repository contains the research program that closes that gap.

The corpus consists of four papers and a reference software implementation. The papers build on each other in strict logical sequence: each one derives its contributions from the mathematical foundations established by its predecessor.

-----

## The Papers

### Paper 1 — A General Theory of Improvement Scoring

*MetaKaizen GeneralTheory.docx*

The foundational paper. Introduces the **Kaizen Variation Score (KVS)**: a multiplicative priority function over four normalized components — Novelty (N), Impact (I′), inverse Complexity (C′), and Timeliness (T) — each bounded in [0, 1], so that KVS ∈ [0, 1].

The multiplicative form is not stipulated. It is derived axiomatically from six measurement-theoretic axioms using conjoint measurement theory (Luce & Tukey, 1964), including a novel **Essentialness with Veto Power** axiom that formally rules out the additive form. Equal component weighting is derived from a Marginal Symmetry axiom, not assumed. The adoption threshold κ = 0.50 is derived from decision theory under symmetric loss, with explicit substrate-specific calibration guidance.

Three formal properties are proved: boundedness (KVS ∈ [0,1]), monotonicity (KVS non-decreasing in each factor), and self-referential applicability (KVS is well-defined for improvements targeting the Meta-Kaizen process itself). The paper includes a structural simulation analysis — replacing an earlier draft’s circular Monte Carlo validation — and a pre-registered empirical validation path. Four substrate demonstrations span investment governance, clinical nursing, software delivery (DORA), and lean manufacturing.

**Key result:** KVS = N × I′ × C′ × T. Derived, not assumed.

-----

### Paper 2 — Networked Implementation and Governance

*MetaKaizen Paper2.docx*

Derives, necessarily and deductively from Paper 1’s mathematical properties, what a globally networked implementation of the Meta-Kaizen framework must look like. Six formal specifications are produced: a canonical data schema, a federated differential-privacy architecture, a Bayesian threshold calibration mechanism, an adversarial inflation-detection protocol, a club-goods network economics structure, and a governance architecture.

The central contribution is the **Governance Closure Theorem**: the process that governs the Meta-Kaizen network is process-identical to the Meta-Kaizen improvement mechanism itself. No meta-level exception exists. Every prior governance framework in the literature terminates in a set of rules that governs the rules but is not itself governed by the same rules. The theorem proves, for the first time, that this exception is not structurally necessary — it is an artifact of assuming scoring and implementation are simultaneous. Under temporal sequencing, the exception dissolves.

The **No Capture Corollary** follows: institutional capture of the network is detectable by the network’s own improvement process. Not impossible, but not invisible.

A three-phase adoption path specifies the conditions under which KVS becomes a standard reported field in continuous improvement research.

**Key result:** The Governance Closure Theorem. No meta-level exception exists.

-----

### Paper 3 — The Reasoning Network

*MetaKaizen Paper3.docx*

Identifies the structural gap that Papers 1 and 2 left unaddressed. A network that propagates scores, calibration data, and decisions while discarding the question structures that produced those decisions is not a reasoning network. It is an answer archive. And an answer archive systematically discards the most valuable thing institutional reasoning produces.

The unit of propagation in a genuine reasoning network is the **question structure**: the governing assumption that was challenged, the orthogonal domain from which the challenge was drawn, the principle established by their collision, and the conditions under which that principle applies. This four-element schema is grounded in Aristotle’s *Topics*, Roman *ratio decidendi*, Koestler’s bisociation, and the distributed cognition literature. What is novel is its formal specification as a schema element, its identification as the minimum generatively complete unit of institutional memory (Theorem 5.2), and its operationalization at the moment of principal override via a conversational elicitation protocol requiring two to three minutes of practitioner time.

The paper proves that no proper subset of the four elements constitutes a generatively complete unit of institutional memory, and specifies the open problem of sufficient conditions for generativity as a formally stated research agenda with testable predictions.

**Key result:** The four-element question structure schema. Decisions without their question structures are not transmissible institutional memory.

-----

### Integration Paper — Closed-Loop Adaptive Governance

*Integrating Meta-Kaizen with the Fractal Rhythm Model (Fracttalix)*

Proves that Meta-Kaizen and the Fractal Rhythm Model / Fracttalix are not merely complementary — they form a closed-loop adaptive architecture when integrated correctly.

Fracttalix detects long-memory patterns (Hurst exponent H > 0.5), widening multifractal spectra, and rhythmic anomalies in outcome time series — signals that current governance rules are increasingly mismatched to the underlying regime. These signals feed into KVS as bounded, pre-normalized inputs via a **convex combination formulation** that preserves boundedness at every step without renormalization:

```
N̂ = (1 − w_N) × N + w_N × max(RDS, CSS)
Î′ = (1 − w_I) × I′ + w_I × RDS
```

BIBO stability is proved directly from the convex combination construction. Regime-triggered principal overrides invoke the Paper 3 question structure protocol, ensuring that the reasoning behind regime-driven governance changes accumulates as transmissible institutional memory.

Explicit minimum data requirements for Hurst estimation (n ≥ 500 observations) are specified, and RDS/CSS signals are scoped as long-horizon regime indicators rather than short-window triggers.

**Key result:** A closed loop — outcome anomalies → higher KVS pressure → governance refinement → new outcomes → repeat — that is bounded, auditable, reversible, and progressively self-educating.

-----

## The Fractal Rhythm Model / Fracttalix

Fracttalix (Sentinel v7.6) is the anomaly detection engine that powers the regime-sensing layer of the closed-loop architecture. It screens univariate time series for long-range dependence, multifractal spectrum widening, and rhythmic anomalies using adaptive EWMA + bidirectional CUSUM — nonparametric, lightweight, and designed for exploratory screening rather than confirmatory inference.

The v7.4–7.6 improvements (Sentinel Turbulence Index, Boundary Layer Warning, Oscillation Damping Filter, CUSUM Pressure Differential) were developed under a Meta-Kaizen improvement process applied to the Fracttalix codebase itself — making Fracttalix the first software project whose development is documented as a Meta-Kaizen substrate application.

See the `main` branch for the full Fracttalix implementation and documentation.

-----

## The Core Mathematical Object

```
KVS_j = N_j × I′_j × C′_j × T_j

where:
  N_j  = 1 − max Jaccard similarity to prior 4 domain scans       ∈ [0,1]
  I′_j = min(1, μ_j / I_max)                                       ∈ [0,1]
  C′_j = (2.0 − C_j) / 1.0                                        ∈ [0,1]
  T_j  = max(0, 1 − h_j / H_max)                                  ∈ [0,1]

Adoption rule: KVS_j ≥ κ (default κ = 0.50) → surface to principal
All decisions logged. All overrides logged with reason code + rationale.
```

-----

## Formal Properties

|Property                       |Statement                                                            |Status                                     |
|-------------------------------|---------------------------------------------------------------------|-------------------------------------------|
|Boundedness                    |KVS ∈ [0,1]                                                          |Proved (Property 4.1, Paper 1)             |
|Monotonicity                   |KVS non-decreasing in each factor                                    |Proved (Property 4.2, Paper 1)             |
|Self-referentiality            |KVS well-defined on Meta-Kaizen itself                               |Proved (Theorem 4.2, Paper 1)              |
|Governance Closure             |No meta-level exception exists                                       |Proved (Theorem 7.2, Paper 2)              |
|No Capture                     |Institutional capture is detectable                                  |Proved (Corollary 7.3, Paper 2)            |
|Minimum Generative Completeness|4-element schema is the minimum complete unit of institutional memory|Proved (Theorem 5.2, Paper 3)              |
|BIBO Stability (closed loop)   |Bounded inputs → bounded outputs at every period                     |Proved (Proposition 4.1, Integration Paper)|

-----

## What Has Not Yet Been Proved

The empirical question — whether KVS-selected improvements outperform unselected ones on realized outcomes — remains open. The theoretical properties above are properties of the scoring algorithm and governance architecture. They do not guarantee predictive validity. A pre-registered empirical validation protocol is specified in Paper 1 (Section 5.4) and Paper 2 (Section 9.2). The test requires n ≥ 30 adopted improvements per substrate type across ≥ 10 contributing organizations. That data does not yet exist.

The framework is deployable and valuable as a governance tool — transparent, auditable, self-referential — independent of whether the empirical test confirms its predictive validity. If the test fails, the governance contribution stands and the scientific ambition is revised downward. This outcome would be reported.

-----

## Substrate Applications

The framework is substrate-agnostic. It has been demonstrated across:

- **Investment governance** — Investment Policy Statements, portfolio allocation rules
- **Clinical nursing** — Patient safety protocols, adverse event reduction
- **Software delivery** — DORA metrics governance, CI/CD pipeline standards
- **Lean manufacturing** — Process specifications, SMED cycle governance
- **Fracttalix development** — The codebase itself, governed as a Meta-Kaizen substrate

Any structured document that is revisable by identified principals and has measurable outcomes qualifies as a substrate.

-----

## Provenance

This work originated in a practical conversation between Thomas Brennan, a registered nurse and principal at Entwood Hollow Research Station in Douglas City, California, and Sophia Entwood, about the governance of a long-term investment policy statement. Brennan instructed Entwood to approach the IPS problem orthogonally — to bring in principles from Kaizen. In giving that instruction, he recognized simultaneously that he was applying the principle he was describing, and that the principle had never been formally specified. The gap between forty years of Kaizen practice and zero mathematical formalization became visible from exactly the vantage point best suited to see it: outside every single discipline, at the intersection of nursing, farming, investment governance, and systems thinking.

Mathematical formalization, literature synthesis, and co-authorship were contributed by Claude (Anthropic, claude-sonnet-4-6, 2026).

-----

## License

The general theory, all mathematical formulations, and the software library are contributed to the public domain under CC0. The Entwood IPS application described in worked examples is illustrative.

-----

## Citation

```
Brennan, T., Entwood, S., & Claude (Anthropic). (2026). Meta-Kaizen: A General Theory 
and Algorithmic Framework for the Mathematical Formalization of Self-Evolving Continuous 
Improvement Across Arbitrary Governance Substrates. Entwood Hollow Research Station 
Working Paper No. 1. https://github.com/thomasbrennan/Fracttalix/tree/Meta-Kaizen
```
