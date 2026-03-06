# Meta-Kaizen

**A formal mathematical theory of continuous improvement governance.**

This project began with a single observation made during a software review: we had no formal process for the Kaizen review itself. We were improving things, but the act of improvement had no structure beneath it. The thought followed naturally — if you are trying to construct a process of improvement, you should formalize the construction of that construction.

That observation became this.

-----

## What This Is

Meta-Kaizen is a four-paper working paper series that formalizes continuous improvement as a mathematically specified, substrate-agnostic, self-referential governance framework. Its core contribution is the **Kaizen Variation Score (KVS)** — a multiplicative scoring function derived axiomatically from six measurement-theoretic axioms:

```
KVS = N × I′ × C′ × T
```

Where:

- **N** — Novelty: Jaccard distance of the proposed improvement from the organization’s recent improvement history
- **I′** — Normalized Impact: projected outcome improvement relative to a substrate-calibrated maximum
- **C′** — Inverse Complexity: operational burden, inverted so lower burden scores higher
- **T** — Timeliness: relevance window, decaying linearly to zero at horizon H_max

Candidates scoring above a threshold κ = 0.50 (derived from Bayesian decision theory under symmetric losses) are surfaced for principal approval. The framework is **substrate-agnostic** — it applies to investment policy statements, clinical protocols, engineering standards, software governance documents, and the framework’s own parameters.

The last point is not incidental. Meta-Kaizen is **self-referential by design**: it can be applied to itself (Paper 1, Theorem 4.2). The governance protocol that controls changes to the framework is governed by the same KVS discipline as the improvements it evaluates (Paper 2, Theorem 7.2).

-----

## The Four Papers

### Paper 1 — General Theory and KVS Scoring Function

*The axiomatic foundation.*

Derives the multiplicative functional form from six measurement-theoretic axioms (Luce & Tukey 1964; Krantz et al. 1971). Proves boundedness, monotonicity, and self-referential applicability. Introduces bisociation — importing structural analogies from orthogonal domains — as the primary mechanism for generating high-Novelty improvement candidates. Validates KVS against an additive scorer under three ground-truth outcome functions across 10,000 candidates × 1,000 simulation runs (KVS recall: 73.8% vs. additive: 68.4% under multiplicative ground truth).

### Paper 2 — Networked Implementation, Privacy Amplification, and Governance Closure

*The federated architecture.*

Proves that Meta-Kaizen scales to networks of independent organizations while preserving privacy (Theorem 3.1: minimum network size n* ≈ 100–210 under the shuffle model of differential privacy). Proves temporal consistency — no retroactive modification of prior evaluations is possible under the cryptographic audit log architecture (Theorem 7.1). Proves governance process-equivalence — the network governance protocol is KVS-governed, not a separate process exempt from the framework’s own discipline (Theorem 7.2). Provides full Bayesian calibration justification for threshold adaptation.

### Paper 3 — The Reasoning Network: Bisociative Question Structure, Challenge Taxonomy, and Institutional Memory

*The cognitive infrastructure.*

Introduces the four-element **Question Structure Schema (QSS)**: Governing Assumption (A), Orthogonal Domain (D), Structural Principle (P), and Applicability Condition (C). Proves by constructive counterexample that no proper subset of {A, D, P, C} generates the full space of reusable institutional insights (Proposition 5.2). Grounds the Challenge Taxonomy in Aristotle’s four-fold classification of predicables and proves exhaustiveness (Proposition 5.1). Proves library quality convergence under honest majority and retrospective validation (Theorem 5.3).

### Paper 4 — The Fractal Rhythm Model: Regime-Aware Adaptation

*The dynamic environment layer.*

Addresses what happens when the stationarity assumption breaks — when the operating environment shifts discontinuously and prior calibrations become obsolete. Introduces the Fractal Rhythm Model (FRM): two regime signal detectors (RDS: Bayesian change-point detection; CSS: complexity surge, in clipping and logistic variants), an adjusted scoring formula KVS-hat, and a formal modification of Paper 1’s Axiom 5 (Essentialness with Veto Power) for regime-shift conditions. Derives the minimum extinguishing parameter δ_min for any finite planning horizon. The Axiom 5 departure is explicitly acknowledged, bounded (maximum regime credit ≤ 0.20 under default weights), and fully reversible.

-----

## What This Is Not

- **Not empirical evidence** that KVS-selected improvements outperform unselected ones. That gap is the research agenda, not an oversight.
- **Not a deployed software library.** The `metakaizen` library is specified and planned; it is not yet released.
- **Not a prediction system.** KVS scores a priori improvement quality; it does not predict realized outcomes.
- **Not a substitute for principal judgment.** The threshold gate is a floor for surfacing candidates, not a ceiling for approving them.

-----

## Intellectual Lineage

The bisociation mechanism draws on Koestler (1964). The axiomatic measurement foundation draws on Luce & Tukey (1964) and Krantz et al. (1971). The privacy architecture draws on Erlingsson et al. (2019) and Feldman et al. (2021). The governance durability analysis draws on Ostrom (1990). The challenge taxonomy draws on Aristotle’s *Topica* (ca. 350 BCE). The Bayesian calibration draws on De Finetti (1937) and Gelman et al. (2013). The regime detection draws on Adams & MacKay (2007).

The Corrections Register in each paper documents every material error identified across revision cycles and how each was resolved. This is intentional. A framework for honest improvement governance should itself be governed honestly.

-----

## Repository Structure

```
/
├── README.md                          — this file
├── papers/
│   ├── Paper1_GeneralTheory.docx
│   ├── Paper2_NetworkedImplementation.docx
│   ├── Paper3_ReasoningNetwork.docx
│   └── Paper4_FractalRhythmModel.docx
├── presentation/
│   └── MetaKaizen_Presentation.pptx
└── src/                               — metakaizen library (specification; release pending)
```

-----

## Authors

**Thomas Brennan** · **Sophia Entwood** · **Claude (Anthropic)**

Entwood Hollow Research Station is an independent research organization affiliated with the Entwood Long-Term Fund, a family investment office based in Douglas City, California. Research is conducted independently of commercial activities.

*Conflict of interest: Tge author is affiliated with the Entwood Long-Term Fund. The general theory and all mathematical formulations are contributed to the public domain. The IPS application described in worked examples is illustrative only. The AI co-authors (Claude, Anthropic, Sophia Entwood, Grok) provided methodological review, mathematical formalization, literature synthesis, and manuscript writing assistance.*

-----

## On Publication

These papers are complete, honest about their limitations, and available here. They have not been submitted to journals.

The operating philosophy: don’t seek what you desire — seek to be worthy of what you desire.

*March 2026*
