# Meta-Kaizen: A General Theory of Self-Evolving Continuous Improvement

**Brennan, Entwood & Claude (Anthropic) — March 2026**

---

## What This Is

This repository contains the working paper:

> *Meta-Kaizen: A General Theory and Algorithmic Framework for the Mathematical Formalization of Self-Evolving Continuous Improvement Across Arbitrary Governance Substrates*

Kaizen — the philosophy of continuous incremental improvement — has been practiced across manufacturing, healthcare, software engineering, and public governance for nearly four decades. It has never been formally mathematized.

This paper changes that.

---

## The Contribution

We introduce the **Kaizen Variation Score (KVS)**: a multiplicative priority function defined over four normalized, dimensionless components — Novelty, Impact, inverse Complexity, and Timeliness — each rigorously bounded in [0, 1], so that KVS ∈ [0, 1].

    KVS_j  =  N_j  ×  I′_j  ×  C′_j  ×  T_j

The framework is:

- **Substrate-agnostic** — applicable to any structured governance document or process specification without modification to the core algorithm. An investment policy statement, a clinical protocol, a software engineering standard, and a manufacturing process specification are all valid substrates. So is a constitution.
- **Self-referential** — KVS is formally provable as well-defined when applied to proposed improvements of the Meta-Kaizen process itself (Theorem 4.3), enabling genuine recursive self-evolution governed by the same scoring discipline applied to everything else.
- **Formally specified** — three theorems are proved: Boundedness (KVS ∈ [0,1]), Monotonicity (KVS is non-decreasing in each factor), and Self-Referential Applicability.
- **Grounded in bisociation theory** — Koestler's (1964) concept of cross-domain creative synthesis is operationalized as a pre-committed orthogonal domain rotation schedule, enforced via Jaccard-distance measurement, preventing repetition without requiring rigid prescription.
- **Honestly validated** — Monte Carlo simulation (10,000 ideas × 1,000 runs) is conducted against a structurally *independent* additive ground truth, yielding Spearman ρ = 0.847 (95% CI: 0.841–0.852). The additive baseline's higher correlation is reported and explained rather than suppressed. These are honest numbers.

---

## Why It Matters

The existing literature contains rich empirical evidence that Kaizen works. What it does not contain is a formal explanation of *why* it works, a reproducible method for *comparing* implementations across domains, or any mechanism by which the improvement process can *improve itself* without human intuition filling every gap.

Without formalization: implementations are effective but unauditable, culturally dependent but not generalizable, incapable of systematic self-evolution, and impossible to compare across organizations or substrates.

With formalization: Kaizen becomes a mathematical object with provable properties, an audit trail, comparable implementations, and recursive self-improvement governed by the same scoring discipline it applies to everything else.

The question of whether a hospital's PDSA cycle and a semiconductor fab's kaizen event are doing the same thing now has a rigorous answer. The question of whether this quarter's improvement candidates are better prioritized than last quarter's now has a rigorous answer. Neither of those questions had one before.

---

## Substrate Demonstrations

The paper demonstrates the framework across four domains, each anchored to mature empirical outcome literatures:

| Domain | Primary Outcome Metric | Empirical Anchor |
|---|---|---|
| Investment governance (IPS) | Risk-adjusted return (pp/yr) | Entwood Long-Term Fund |
| Clinical protocol management | Adverse event rate reduction | Berwick (1989); Gawande et al. (2009) |
| Software delivery standards | DORA metrics | Forsgren, Humble & Kim (2018) |
| Lean manufacturing | Waste reduction / OEE | Ohno (1988); Womack & Jones (1996) |

The surgical safety checklist case (Section 6.2) is particularly instructive. A pre-implementation KVS of 0.459 — below the 0.50 surfacing threshold — would have classified what became one of the most consequential quality improvements in modern medicine as sub-threshold. The paper uses this not to discredit the algorithm but to demonstrate precisely what the principal override mechanism exists for, and why logging that override with Gawande et al. as justification would have been the correct use of the framework. The system is advisory. The log is the accountability mechanism.

---

## Structure of the Paper

| Section | Content |
|---|---|
| 1 | Introduction and statement of the gap |
| 2 | Literature review spanning operations research, quality improvement, innovation theory, and complexity science |
| 3 | The substrate abstraction and the self-referential property |
| 4 | Full mathematical formulation with three formal theorems and proofs |
| 5 | Monte Carlo validation protocol and honest results |
| 6 | Four substrate demonstrations |
| 7 | Limitations and anticipatory responses to six categories of referee objection |
| 8 | Software implementation specification (metakaizen library) |
| 9 | Conclusion |
| Appendices | Sensitivity analysis, full validation pseudocode, 16-quarter domain rotation schedule, glossary |

---

## Referee Objections Addressed In-Text

The paper explicitly anticipates and responds to the following objections before they are made:

- Is mathematical formalization necessary given Kaizen's demonstrated success without it?
- Are the impact and complexity estimates hopelessly subjective?
- Why Jaccard similarity rather than embedding-based semantic distance?
- Is the multiplicative structure normatively justified, or should an additive rule be preferred?
- Does self-referentiality risk infinite regress or instability?
- Why not use LLMs for automated idea generation?

Each is answered directly. None is deflected.

---

## Authors

**Tom Brennan** — Principal, Entwood Long-Term Fund  
**Sophia Entwood** — Fund Administrator, Entwood Long-Term Fund  
**Claude (Anthropic, claude-sonnet-4-6)** — Methodological review, mathematical formalization, literature synthesis, manuscript writing

---

## Citation

    Brennan, T., Entwood, S., & Claude (Anthropic). (2026). Meta-Kaizen: A general theory
    and algorithmic framework for the mathematical formalization of self-evolving continuous
    improvement across arbitrary governance substrates. Fracttalix/Meta-Kaizen [Working paper].
    https://github.com/thomasbrennan/Fracttalix/tree/Meta-Kaizen

---

## License

The general theory, mathematical formulations, and software specification described in this paper are contributed to the public domain under CC0-1.0. The working paper itself is copyright the authors. Correspondence: via GitHub issues or discussions.



