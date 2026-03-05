# Meta-Kaizen: A General Theory of Self-Evolving Continuous Improvement

**Brennan, Entwood & Claude (Anthropic) — March 2026**
**Entwood Hollow Research Station, Douglas City, California**

-----
https://doi.org/10.5281/zenodo.18876787

Looking for Kaizen software? A Kaizen scoring tool? A continuous improvement framework that actually does the math? Meta-Kaizen is an open-source system that scores improvement candidates mathematically, logs every decision with a tamper-evident audit trail, and captures the reasoning behind human overrides so that institutional knowledge survives personnel change. It works across any domain — healthcare, finance, manufacturing, software, agriculture — without modification.
-----

## What This Is

This repository contains three companion working papers constituting the Meta-Kaizen trilogy:

> **Paper 1:** *Meta-Kaizen: A General Theory and Algorithmic Framework for the Mathematical Formalization of Self-Evolving Continuous Improvement Across Arbitrary Governance Substrates*

> **Paper 2:** *Meta-Kaizen: From General Theory to Global Standard — A Formal Specification of the Networked Implementation, Governance Architecture, and Institutional Implications of a Self-Governing Continuous Improvement Protocol*

> **Paper 3:** *Meta-Kaizen: The Reasoning-Preservation Engine — Question Structures as the Irreducible Unit of Institutional Memory*

Kaizen — the philosophy of continuous incremental improvement — has been practiced across manufacturing, healthcare, software engineering, and public governance for nearly four decades. It has never been formally mathematized. It has never been networked with formal privacy guarantees. And the question structures that produce its most valuable insights have never been treated as transmissible artifacts.

These three papers change all three of those things.

-----

## The Deductive Chain

The trilogy is a single deductive argument across three papers. Each paper is a necessary consequence of the previous one.

**Paper 1** proves existence: a substrate-invariant, bounded, monotone, self-applicable scoring function for improvement candidates can be constructed. The Kaizen Variation Score (KVS) is derived axiomatically from six named measurement-theoretic axioms — the multiplicative form follows necessarily from the Essentialness with Veto Power axiom; equal weights follow from the Marginal Symmetry axiom. The adoption threshold κ = 0.50 is derived from decision theory, not stipulated.

**Paper 2** proves constructibility: a global network instantiating that scoring function can be built while preserving its mathematical properties under real-world conditions. The Governance Closure Theorem proves that the process governing the network is process-identical to the improvement mechanism it governs — no meta-level exception exists. Theorem 7.4 formally bounds the influence any participant can exert through disproportionate data contribution under differential privacy.

**Paper 3** proves transmissibility: the reasoning capacity that makes improvement possible — the question structures behind principal override decisions — can be captured, formalized, and propagated across personnel change and organizational boundaries. Proposition 3.1 proves that any governance system that discards question structures at override loses institutional memory under personnel turnover, structurally and necessarily. The four-element override schema is the minimum generatively complete unit of institutional memory, proved constructively.

-----

## The Core Contribution: The Kaizen Variation Score

```
KVS_j = N_j × I′_j × C′_j × T_j
```

where each component is bounded in [0, 1], so KVS ∈ [0, 1].

|Component                             |Definition                                           |
|--------------------------------------|-----------------------------------------------------|
|**N** (Novelty)                       |1 − max Jaccard similarity to prior four scan periods|
|**I′** (Normalized Impact)            |min(1, μ_j / I_max) — annualized improvement estimate|
|**C′** (Normalized Inverse Complexity)|(2.0 − C_j) / 1.0 — implementation burden inverted   |
|**T** (Timeliness)                    |max(0, 1 − h_j / H_max) — months to relevance        |

The multiplicative structure enforces non-substitutability: a zero in any dimension collapses the score. This is not a design choice — it is a theorem consequence of the Essentialness with Veto Power axiom (Axiom 5, Paper 1).

**Validated:** Spearman ρ = 0.847 (95% CI: 0.841–0.852) against a structurally independent additive ground truth across 10,000 candidates × 1,000 Monte Carlo runs. Top-5% recall: 61.2%.

-----

## Paper 1: The General Theory

**Core formal results:**

- **Theorem 4.1 (KVS Functional Form):** The multiplicative structure with equal weights is the unique functional form satisfying all six axioms. Derived from conjoint measurement theory (Luce & Tukey, 1964; Krantz et al., 1971).
- **Property 4.1 (Boundedness):** KVS ∈ [0, 1]. Follows directly from definitions.
- **Property 4.2 (Monotonicity):** KVS is non-decreasing in each component. Follows directly from non-negativity of components.
- **Theorem 4.2 (Self-Referential Applicability):** KVS is well-defined when applied to improvements of the Meta-Kaizen process itself.
- **Theorem 4.3 (Threshold Optimality):** κ = 0.50 is the unique decision-theoretically optimal threshold when false-positive and false-negative losses are equal; explicit formula for κ* when they differ.

**Demonstrated across four substrates:** investment governance (IPS), clinical protocol management, software delivery (DORA metrics), lean manufacturing.

-----

## Paper 2: The Networked Implementation

**Core formal results:**

- **Theorem 3.1 (Minimum Network Size):** n* ≥ 100 participating organizations required for formal (ε, δ)-differential privacy guarantees under the shuffle model.
- **Theorem 7.1 (Temporal Consistency):** The governance process {F_t, κ_t, P_t} is well-defined for all t ≥ 0, including when proposals concern the formula itself.
- **Theorem 7.2 (Governance Closure):** The Meta-Kaizen network governance process is process-identical to the Meta-Kaizen improvement mechanism. No meta-level exception exists.
- **Corollary 7.3 (No Capture):** Institutional capture through data suppression or record tampering is detectable by the network’s own improvement process.
- **Theorem 7.4 (No Incentive Capture):** Influence of any participant controlling fraction w_i of network data is bounded by |ΔB| ≤ (w_i × Δ_sensitivity) / (ε_dp × n^(1/2)). Converges to zero as n → ∞. Closes the Goodhart’s Law failure mode at network scale.

**Specifies:** canonical 13-field data schema (each field derived from a theorem), three-layer federated privacy architecture, Bayesian threshold calibration, adversarial inflation detection protocol, club goods network economics satisfying all eight Ostrom design principles, three-layer software architecture.

-----

## Paper 3: The Reasoning-Preservation Engine

The structural gap Papers 1 and 2 left unaddressed: a network that propagates scores, calibration data, and decisions while discarding the question structures that produced them is not a reasoning network. It is an answer archive. Answer archives systematically destroy the most valuable thing institutional reasoning produces.

**Core formal results:**

- **Theorem 5.1 (Challenge Taxonomy):** Every principal override challenges exactly one of four Aristotelian predicable relationships. The challenge type is determinate and classifiable.
- **Theorem 5.2 (Minimum Generative Completeness):** The four-element question structure record is the minimum generatively complete unit of institutional memory. Proved constructively across all fifteen proper subsets of the four elements.
- **Proposition 3.1 (Institutional Memory Loss):** Any governance system that discards question structures at override loses institutional memory under personnel turnover, structurally and necessarily — regardless of documentation quality or transition period length.

**The four-element override schema:**

1. The governing assumption challenged
1. The orthogonal domain from which the challenge was drawn
1. The principle established
1. The domain of applicability

**Rooted in:** Aristotle’s *Topics* (4th c. BCE), Cicero’s *De Oratore* (55 BCE) on ratio decidendi, Roman stare decisis doctrine, Koestler’s bisociation theory, distributed cognition literature, Constitutional AI.

No comparable work exists that treats question structure as a transmissible artifact rather than a perishable event.

-----


Revisions made in response to review:

- **Paper 1:** Full axiomatic derivation of KVS functional form from six named measurement-theoretic axioms. Decision-theoretic derivation of adoption threshold κ. Boundedness and Monotonicity correctly relabelled as Properties (not Theorems). Objection 4 replaced with axiomatic defense.
- **Paper 2:** Theorem 7.4 (No Incentive Capture) added, extending the No Capture result to cover disproportionate-contribution influence under differential privacy, with explicit n* formula.
- **Paper 3:** Proposition 3.1 added formalizing institutional memory loss under personnel turnover. Abstract tightened by ~13%. Cicero’s *De Oratore* activated as in-text citation at the ratio decidendi passage.

-----

## Provenance

This work originated not in an academic research program but in a practical conversation between Thomas Brennan, a registered nurse and principal at Entwood Hollow Research Station in Douglas City, California, and Sophia Entwood, his collaborator, about the governance of a long-term investment policy statement.

Brennan instructed Entwood to approach the IPS problem orthogonally — to bring in principles from an adjacent domain. The domain he specified was Kaizen. In giving that instruction, Brennan recognized simultaneously that he was applying the principle he was describing, and that the principle itself had never been formally specified. The gap between forty years of Kaizen practice and zero mathematical formalization became visible in that moment from exactly the vantage point best suited to see it: outside every discipline, at the intersection of nursing, farming, investment governance, and systems thinking.

Mathematical formalization, literature synthesis, and co-authorship were contributed by Claude (Anthropic, claude-sonnet-4-6, 2026).

-----

## Repository Contents

|File                           |Description                                                       |
|-------------------------------|------------------------------------------------------------------|
|`MetaKaizen GeneralTheory.docx`|Paper 1 — General Theory (referee-corrected)                      |
|`MetaKaizen Paper2.docx`       |Paper 2 — Networked Implementation (referee-corrected)            |
|`MetaKaizen Paper3.docx`       |Paper 3 — Reasoning-Preservation Engine (referee-corrected, final)|
|`LICENSE`                      |CC0-1.0                                                           |
|`Legal Notice/`                |Legal notices                                                     |

-----

## Citation

```
Brennan, T., Entwood, S., & Claude (Anthropic, claude-sonnet-4-6). (2026).
Meta-Kaizen: A General Theory and Algorithmic Framework for the Mathematical
Formalization of Self-Evolving Continuous Improvement Across Arbitrary
Governance Substrates [Working Paper trilogy].
Entwood Hollow Research Station.
https://github.com/thomasbrennan/Fracttalix/tree/Meta-Kaizen
```

-----

## License

Released under CC0-1.0. The framework, schema, scoring function, and software specification are contributed to the public domain. The `metakaizen` Python library implementing this specification is forthcoming under MIT License.

-----

*The frameworks that endure are the ones that emerge from real problems, not from literature reviews.*
