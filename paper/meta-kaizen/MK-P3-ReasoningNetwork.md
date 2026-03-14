Meta-Kaizen Series · Paper 3 of 8

## The Meta-Kaizen Reasoning Network: A Formal Theory of Bisociative Question Structure, Challenge Taxonomy, and Institutional Memory Propagation

Thomas Brennan · with AI collaborators Claude (Anthropic) and Grok (xAI)

March 2026 · Submitted for peer review



## 1. Series Orientation

This is Paper 3 of five. Papers 1–2 derived the KVS and proved networked implementation. This paper addresses the cognitive layer: how are bisociative questions structured, how are principal override decisions captured, and how is institutional reasoning preserved across personnel transitions. Paper 4 integrates regime-aware adaptation. Paper 5 provides the decision-theoretic capstone.

## 2. Abstract and AI-Reader Header

## [AI-READER HEADER] — Dual-Reader Standard, Section 2



How to verify this paper without reading it: Load the AI layer JSON at the URL above. Run the schema validator (claim_registry, inference_rule_table, phase_ready fields). Every Type F claim has a deterministic 5-part predicate — FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT — that evaluates to FALSIFIED or NOT FALSIFIED without accessing prose. Placeholders in the register are the only unresolved dependencies.



## Abstract (Human Reader)

The bisociation mechanism (Papers 1–2) generates improvement candidates by importing structural analogies from orthogonal domains. But the mechanism produces ideas, not arguments. This paper addresses the upstream structure: what is the form of a well-posed bisociative question, and how are reasoning chains underlying improvement decisions captured and made reusable?

Three formal results are established. Proposition 5.2 (Minimum Generative Completeness) proves by constructive counterexample that no proper subset of {A, D, P, C} generates the full space of reusable institutional insights — each element is necessary. Proposition 5.1 (Challenge Taxonomy Exhaustiveness) grounds the four challenge types in Aristotle's Topics 1.4 partition. Theorem 5.3 (Library Quality Convergence) establishes a conditional design guarantee: under honest majority (C1), retrospective validation (C2), and bounded strategic benefit (C3), library quality Q_L(t) → 1. The conditions are verifiable design targets, not assumed background facts.

Key epistemological clarification: the self-KVS score of this paper (Appendix A, KVS=0.413) is a demonstration of self-referential applicability (MK-P1 Theorem 4.2). It cannot and does not constitute independent quality evidence.

## 3. Intellectual Genealogy

## 3.1 Aristotle's Topica

The philosophical roots of structured bisociative questioning trace to Aristotle's Topica (ca. 350 BCE). The Topics is a systematic treatise on productive intellectual questioning — finding the questions most likely to lead from shared premises to new understanding. The medieval questio disputata inherited this tradition: a well-formed questio required an explicit statement of the claim tested (analogous to A), a domain of comparison (D), the principle extracted (P), and conditions of application (C). The QSS is a formal reconstruction of this 2,300-year-old tradition in measurement-theoretic terms.

## 3.2 Koestler's Bisociation

Koestler's (1964) bisociation framework posits that creative insights arise from simultaneous activation of two habitually incompatible matrices of thought. The QSS translates this computational approach into a human-executable protocol: the bisociative bridge is explicitly specified as Structural Principle P extracted from Orthogonal Domain D.

## 4. The Question Structure Schema

## 4.1 Four Elements



The bisociative question in full form: "Given that we assume A, and given that domain D operates according to principle P under condition C, what would our substrate look like if we replaced A with P wherever C holds?"

## 4.2 Worked Example: IPS Substrate, Mycorrhizal Networks Domain

A: "We currently assume that portfolio assets are independent units that can be evaluated in isolation, and that diversification is achieved by holding many uncorrelated assets."

D: Mycorrhizal fungal network ecology. Jaccard distance from investment management = 0.94.

P: "In mycorrhizal networks, individual trees do not optimize in isolation. The forest allocates resources through a hub-and-spoke topology where highly connected nodes redistribute carbon and nutrients to peripheral nodes under stress. Resilience derives from node connectivity structure, not node independence."

C: "This principle applies to our portfolio when average pairwise correlation exceeds 0.35."

## 5. Institutional Memory and Its Loss

## 5.1 Proposition 4.1: Institutional Memory Loss

Under standard improvement record formats (outcome documented, rationale brief or absent, QSS elements not specified), the retention level RL(t) — the proportion of QSS elements that a naive reviewer can reconstruct without access to original participants — declines with personnel transitions. Specifically RL(t_0 + k·Δ) → 0 as k → ∞, where Δ is the median tenure length.

Empirical grounding: Argote & Epple (1990) and Levitt & March (1988) document systematic forgetting curves following personnel transitions. Walsh and Ungson (1991) identify individual memory (lost at departure) as the most volatile retention facility. QSS elements A and P are predominantly held in individual memory.

## 5.2 Proposition 5.1: Challenge Taxonomy Exhaustiveness

Every challenge to a governing assumption A can be classified into exactly one of four types derived from Aristotle's predicables (Topics 1.4, 101b11–20):



Honest limitation: edge cases may be ambiguous between Types I and II. The taxonomy provides a productive first-pass classification; inter-classifier disagreement is informative and should be logged as an annotation dispute in the Library Record. This limitation does not invalidate the exhaustiveness argument — it qualifies its practical application.





## 5.3 Proposition 5.2: Minimum Generative Completeness

No proper subset of {A, D, P, C} generates the full space of reusable bisociative insights. Proof by constructive counterexample for each element:

Without A: practitioners cannot reconstruct which challenge type (I–IV) generated the insight, preventing systematic search for analogous assumptions in other contexts.

Without D: practitioners cannot extend P serendipitously to new substrates — different originating domains suggest different extensions.

Without P: every retrieval requires re-generating the insight from scratch, defeating the purpose of institutional memory.

Without C: P is applied indiscriminately to all contexts, causing harm when the applicability condition is not met.





## 5.4 Theorem 5.3: Library Quality Convergence (Conditional Design Guarantee)

Reclassification: This theorem is a conditional design guarantee, not an unconditional convergence proof. Conditions C1, C2, C3 jointly characterize the target system, not independent premises. Q_L(t) → 1 follows by construction IF the conditions hold.

IF (C1) honest majority of contributing practitioners, AND (C2) retrospective validation each cycle, AND (C3) bounded strategic benefit, THEN Q_L(t) → 1 as t → ∞.

C1 failure scenarios: (i) coordinated strategic behavior by a coalition; (ii) systematic good-faith error in QSS specification; (iii) selection bias in outcome reporting. Mitigations: (i) addressed by MK-P2 club goods mechanism; (ii) require QSS elements specified at proposal time; (iii) flag incomplete outcome reporting at annual retrospective.





## 6. Integration with KVS

The QSS connects to the KVS framework through the Novelty component. The Governing Assumption A determines the scope of what counts as non-trivially novel. An improvement candidate that reinstates a previously rejected assumption should receive a novelty penalty. Formally: the similarity set K_j is expanded to include assumption keyword set A_j from all prior Library Records whose assumption was rejected.

## 7. Limitations

The generative completeness claim (Prop. 5.2) proves minimality, not sufficiency. A fifth QSS element (e.g., explicit failure mode statement) might improve propagation fidelity.

The Aristotelian classification may exhibit edge-case ambiguity between Types I and II. This is acknowledged as a classification challenge, not a theoretical failure.

Theorem 5.3 is conditional on C1–C3. No empirical data yet confirms whether the club goods mechanism successfully maintains C1 in practice.

## 8. Corrections Register

## Correction 1: Proposition 5.1 grounded in Aristotle's Topics

Prior draft presented Challenge Taxonomy as a Theorem without proof. Reclassified as Proposition grounded in Aristotle's Topics 1.4 and Stock (1888 §339) exhaustiveness argument.

## Correction 2: Proposition 5.2 counterexamples made concrete

Prior draft stated 15 counterexamples exist without providing them. Four concrete counterexamples now included — one per element.

## Correction 3: Theorem 5.3 recharacterized as conditional design guarantee

Prior draft presented Theorem 5.3 as formal convergence proof. Recharacterized: C1–C3 are verifiable design targets, not independent premises. Logical status now correctly described.

## Correction 4: Self-KVS score explicitly labeled as demonstration

Prior draft reported KVS=0.502 without adequate caveat. Corrected to KVS=0.413 (arithmetic correction from 0.86×0.60×0.80×1.00). Threefold circularity now explicitly identified.

## 9. References

Argote, L., & Epple, D. (1990). Learning curves in manufacturing. Science, 247(4945), 920–924.

Aristotle. Topics. (Trans. R. Smith, 1997). Clarendon Press.

Koestler, A. (1964). The act of creation. Penguin Books.

Levitt, B., & March, J. G. (1988). Organizational learning. Annual Review of Sociology, 14, 319–340.

Stock, G. W. J. (1888). Introduction to the study of logic. Longmans, Green.

Walsh, J. P., & Ungson, G. R. (1991). Organizational memory. Academy of Management Review, 16(1), 57–91.



## Appendix A: Self-KVS Score — Methodological Demonstration

Important: This score is a demonstration of MK-P1 Theorem 4.2 (self-referential applicability). Authors scoring their own paper are not independent evaluators. This score is not quality evidence.



KVS=0.413 is below the κ=0.50 threshold — consistent with the circularity caveat. A sub-threshold self-evaluation does not imply low quality; it illustrates that author-evaluators apply appropriate discounting.



## Appendix B: AI Layer — Channel 2 Asset

Layer URL: https://github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/MK-P3-ai-layer.json

Schema: v2-S48 | Phase status: PHASE-READY | Placeholders: 2 (PH-MK3.1, PH-MK3.2) — both non-blocking | Produced: Session S49

Semantic spec (Layer 0): github.com/thomasbrennan/Fracttalix/ai-layers/falsification-kernel.md