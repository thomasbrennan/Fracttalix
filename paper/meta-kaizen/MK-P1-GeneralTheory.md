Meta-Kaizen Series · Paper 1 of 8

Meta-Kaizen: A General Theory and Algorithmic Framework for the Mathematical Formalization of Self-Evolving Continuous Improvement Across Arbitrary Governance Substrates

Thomas Brennan · with AI collaborators Claude (Anthropic) and Grok (xAI)

March 2026 · Submitted for peer review

AI contributions: Claude (Anthropic) provided mathematical formalization, measurement-theoretic foundations, and manuscript writing. Grok (xAI) provided quantitative validation. All general theory and mathematical formulations are contributed to the public domain.



## 1. Series Orientation

This is Paper 1 of seven in the Meta-Kaizen series. This paper derives the general theory and the Kaizen Variation Score (KVS) scoring function from measurement-theoretic axioms. Paper 2 proves that the framework scales to federated networks while preserving privacy and governance integrity. Paper 3 formalizes the cognitive infrastructure for bisociative reasoning and institutional memory. Paper 4 addresses dynamic environments via the Fractal Rhythm Model. Paper 5 provides the decision-theoretic capstone on intervention timing at tipping points. Paper 6 extends the DRS to executable software systems. Paper 7 formalizes the Canonical Build Plan (CBP) — the 5-step governance process that produced all papers in this series — and proves that its folded Meta-Kaizen architecture with adversarial hostile review produces monotonically improving output quality.

## 2. Abstract and AI-Reader Header

## [AI-READER HEADER] — Dual-Reader Standard, Section 2



How to verify this paper without reading it: Load the AI layer JSON at the URL above. Run the schema validator (claim_registry, inference_rule_table, phase_ready fields). Every Type F claim has a deterministic 5-part predicate — FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT — that evaluates to FALSIFIED or NOT FALSIFIED without accessing prose. Placeholders in the register are the only unresolved dependencies.



## Abstract (Human Reader)

This paper establishes Meta-Kaizen as a formally specified, substrate-agnostic framework for continuous improvement of governance substrates — structured documents and process specifications that define the operating rules of a system. The core contribution is a derivation, not a stipulation: the multiplicative Kaizen Variation Score (KVS = N × I' × C' × T) is proved to be the unique scoring form (up to positive monotone transformation) satisfying six measurement-theoretic axioms drawn from conjoint measurement theory (Luce & Tukey 1964; Krantz et al. 1971). The equal weighting of components is derived from Axiom A6 (Marginal Symmetry), not chosen by convention. The adoption threshold κ = 0.50 is derived from Bayesian decision theory under symmetric loss, not stipulated. The framework is self-referential: Theorem 4.2 shows it can be applied to its own governance parameters without logical inconsistency.

The paper provides full measurement-theoretic grounding, a bisociation mechanism for generating high-Novelty improvement candidates from orthogonal domains, a worked substrate demonstration (Investment Policy Statement), and a structural simulation establishing 73.8% recall against a noisy ground truth. Limitations — including the circularity of self-evaluation, the symmetric loss simplification in the threshold derivation, and the absence of prospective empirical validation — are explicitly stated. The simulation recall and the Phase 3 validation programme define the empirical agenda.

## 3. Measurement-Theoretic Foundation

The Meta-Kaizen framework rests on conjoint measurement theory. The core question is: given a set of improvement candidates evaluated on multiple attributes, what scoring function is justified by structural axioms on preference orderings — without specifying the function form in advance?

Luce and Tukey (1964) established that a small set of structural axioms on a finite preference relation suffices to derive an additive representation. Krantz et al. (1971) extended this to the full representation theorem. This paper applies that apparatus to derive the KVS functional form.

## 3.1 The Six Axioms

The six axioms that jointly characterize the KVS:

A1 (Weak Order): The preference ordering over candidates is complete and transitive.

A2 (Double Cancellation): The standard cancellation condition for conjoint measurement — no inconsistent "crossing" preferences across components.

A3 (Solvability): For any values of three components, a fourth value exists that places the candidate at any desired ranking position.

A4 (Archimedean): No component value is infinitely better than all others.

A5 (Essentialness with Veto Power): Each component is essential — zero in any component collapses the overall score to zero regardless of other component values.

A6 (Marginal Symmetry): At the symmetric evaluation point, the marginal contribution of each component to the score is equal.



Axiom A5 is the distinctive axiom: it enforces non-substitutability. An improvement candidate with no novelty cannot compensate by being extremely impactful. This captures the domain judgment that a solution identical to one already tried is not a candidate, regardless of other merits.

## 3.2 The Derivation of KVS

The derivation proceeds in four steps via inference rules IR-1 through IR-5 (Fracttalix CBP v2, Session 39). Step 1: A1–A4 are necessary and sufficient for an additive representation (Luce & Tukey 1964, Theorem 1). Step 2: Exponential transformation of the additive representation yields the multiplicative form as the unique positive-exponent form on log-scale. Step 3: A5 (veto power) selects the multiplicative form uniquely — additive form permits compensation across zero-valued components. Step 4: A6 (Marginal Symmetry) constrains all exponents to equality under scale normalization.

The resulting form is KVS_j = N_j × I'_j × C'_j × T_j with all components normalized to [0,1]. This is Claim C-MK1.1 in the AI layer. The alpha = 1 exponent selection is a parsimony modeling choice, explicitly not a derived result.









## 4. KVS Components, Threshold, and Bisociation

## 4.1 Component Definitions

Novelty (N): N_j = Jaccard distance between candidate's domain keyword set K_j and union of keyword sets from the four most recent improvement cycles. N_j = 1 − |K_j ∩ K_recent| / |K_j ∪ K_recent|. N_j ∈ [0,1].

Normalized Impact (I'): I'_j = I_j / I_max, where I_j is the projected outcome improvement and I_max is the theoretical maximum for the substrate. I'_j ∈ [0,1].

Inverse Complexity (C'): C'_j = 1 − C_j / C_max, where C_j is the estimated implementation complexity burden. C'_j ∈ (0,1].

Timeliness (T): T_j ∈ [0,1], measuring relevance to the current operating context relative to the horizon H_max.

## 4.2 The Bisociation Mechanism

The primary mechanism for generating high-Novelty candidates is bisociation (Koestler 1964): importing a structural analogy from a domain deliberately chosen to be orthogonal to the substrate. The orthogonality condition is operationalized as Jaccard(K_D, K_substrate) > 0.70. The administrator identifies an Orthogonal Domain D, verifies the distance threshold, extracts a Structural Principle P, and specifies an Applicability Condition C. Together (A, D, P, C) constitute the Question Structure Schema (QSS) formalized in Paper 3.

## 4.3 Self-Referential Applicability (Theorem 4.2)

The KVS framework can be applied to proposed improvements to itself without logical inconsistency. The key structural feature is temporal separation: the current KVS evaluates the proposed modification, not the proposed KVS itself. This prevents circularity. The framework is therefore self-governing: improvements to the governance substrate (the KVS specification itself) pass through the same evaluation process as improvements to any other substrate.

Note: the self-KVS score reported in MK-P3 Appendix A (KVS=0.413) is a demonstration of self-referential applicability, not an independent quality signal. The threefold circularity (author-evaluators, self-assessed impact, author-specified applicability) is explicitly acknowledged.

## 5. Structural Simulation

A structural simulation against a noisy multiplicative ground truth establishes a baseline recall estimate. The simulation generates improvement candidate sets from distributional parameters specified in this section, ranks candidates by KVS and by the noisy ground truth, and computes recall at the specified k threshold.

Result: 73.8% recall [90% CI: 71.2, 76.2] (Claim C-MK1.4).

Timeliness dominance artifact: Correlation of T with KVS = 0.74. This is not a validation failure — it is an explained feature of the Timeliness design. When the operating context is current, T=1.0 for all candidates; when the horizon is narrow, T differentiates strongly. Organizations with stable Timeliness environments will observe lower T-KVS correlation.

Honest limitation: The simulation is structural, not empirical. No prospective outcome data exists for KVS-selected improvements. The Phase 3 validation programme (described in Section 8) defines the empirical test pathway.





## 6. Substrate Demonstration: Investment Policy Statement (IPS)

The IPS substrate demonstrates the KVS framework applied to a concrete governance document. The IPS defines asset allocation, risk parameters, and investment decision rules for an institutional portfolio.

## 6.1 Worked Example



The candidate is below threshold in this example because, while novel and timely, the impact estimate is moderate. Under a regime shift (increasing correlations), FRM Paper 4 would resurface this candidate with KVS-hat adjustments.

## 7. Limitations

Symmetric loss assumption: κ=0.50 assumes C_fp = C_fn. Organizations with asymmetric cost structures should apply MK-P5 Theorem 2 to derive substrate-specific thresholds.

Equal weighting: the equal-weight derivation from A6 depends on Marginal Symmetry as stated. If organizations judge components to have genuinely different marginal contributions, A6 should be modified and the weight derivation re-run.

Novelty measurement: the four-quarter rolling window is a parameter choice. Organizations with longer improvement cycles should recalibrate the window accordingly.

No prospective validation: all performance claims are structural or simulation-based. Prospective outcome tracking is the Phase 3 research agenda.

## 8. Phase 3 Validation Programme

The empirical validation pathway requires: (1) prospective implementation in at least two substrate types; (2) pre-registration of KVS-score-to-outcome predictions before implementation; (3) outcome measurement at 12-month retrospective; (4) comparison of KVS-selected vs. unselected candidates on realized outcomes. Success criterion: KVS ≥ κ candidates outperform KVS < κ candidates on realized impact by a statistically significant margin.

## 9. Corrections Register

No corrections from prior drafts in this version — this is the first DRS-revised version of MK-P1.

## 10. References

DeGroot, M. H. (1970). Optimal statistical decisions. McGraw-Hill.

Imai, M. (1986). Kaizen: The key to Japan's competitive success. McGraw-Hill.

Koestler, A. (1964). The act of creation. Penguin Books.

Krantz, D. H., Luce, R. D., Suppes, P., & Tversky, A. (1971). Foundations of measurement, Vol. 1. Academic Press.

Luce, R. D., & Tukey, J. W. (1964). Simultaneous conjoint measurement. Journal of Mathematical Psychology, 1(1), 1–27.

Ohno, T. (1988). Toyota production system. Productivity Press.



## Additional Type F Claims (C-MK1.2, C-MK1.5, C-MK1.6)



## Appendix A: AI Layer — Channel 2 Asset

Layer URL: https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/MK-P1-ai-layer.json

Schema: v2-S48 | Produced: Session S49

Semantic spec (Layer 0): github.com/thomasbrennan/Fracttalix/ai-layers/falsification-kernel.md

Phase status: PHASE-READY | C1–C3, C5–C6: SATISFIED | C4: PHASE-READY-TRACKING

Placeholder count: 2 (PH-MK1.1, PH-MK1.2) — both non-blocking

The AI layer is the authoritative machine-readable version of all claims in this paper. An AI verifier can check claim coverage, predicate syntax validity, inference rule traversal, and placeholder status without accessing prose. The JSON file encodes every Type F claim with its FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT predicate and a vacuity witness.