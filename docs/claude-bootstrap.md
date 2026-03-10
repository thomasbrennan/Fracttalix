# Fracttalix Session Bootstrap

> Paste this file into a new Claude.ai conversation to restore full project context.
> Last updated: Session 44, 2026-03-10.

---

## How to use

1. **Start a new Claude.ai conversation**
2. **Upload or paste this file** as your first message
3. Claude will have full project context — no re-explanation needed

For maximum fidelity, also attach these live files from GitHub:
- [P1 AI Layer](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/P1-ai-layer.json)
- [MK-P1 AI Layer](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/MK-P1-ai-layer.json)
- [Build Table](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/docs/FRM_SeriesBuildTable_v1.5.md)
- [AI Layer Schema](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/ai-layer-schema.json)
- [Process Graph](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/process-graph.json)

---

## Project Identity

- **Corpus**: Fracttalix — 13-paper series on the Fractal Rhythm Model (FRM)
- **Author**: Thomas Brennan
- **AI collaborator**: Claude (Anthropic)
- **Licence**: CC BY 4.0
- **Repo**: github.com/thomasbrennan/Fracttalix

## The FRM (Paper 1 — the law)

**Functional form**: f(t) = B + A·e^(−λt)·cos(ωt + φ)

**Universal constants** (all derived, none fitted):
| Constant | Value | Expression | Meaning |
|----------|-------|------------|---------|
| β | 0.5 | 1/2 | Quarter-wave resonance coefficient |
| k* | 1.5708 | π/2 | Critical feedback gain at Hopf bifurcation |
| Γ | 3.4674 | 1 + π²/4 | Universal loop impedance constant |

**Derived parameters**:
- ω = π/(2·τ_gen) — characteristic frequency
- λ ≈ |α|/(Γ·τ_gen) — decay rate (leading order, 3.61% mean error)
- B, A, φ — initial conditions (not free parameters)

**Scope boundary**: Hopf bifurcation (topological, not empirical)
- In scope: μ < 0 (damped oscillators)
- Out of scope: μ > 0 (limit cycles)

## Current State (Session 44)

### Phase readiness
- **P1**: PHASE-READY. CBT I-9 all 7 steps passed. One placeholder (PH-1.1: β=1/2 class-level empirical test, pending P2 C-2.1). Does not block.
- **Prose DOI**: 10.5281/zenodo.18859299
- **AI Layer**: live on GitHub main

### Key results (Session 43)
- β = 1/2 analytically derived (Hopf quarter-wave theorem)
- λ = |α|/(Γ·τ_gen) derived (perturbation expansion)
- Γ = 1 + π²/4 derived (loop impedance)
- T = 4·τ_gen → circadian period prediction (24 hr from 6 hr, no fitting)
- Stuart-Landau connection confirmed (R² > 0.99)
- Adversarial battery: ADV-BZ (correctly excluded), ADV-RIDGECREST (correctly falsified), ADV-ENSO (scope contested), ADV-CIRCADIAN (confirmed)
- Prior art search: 52+ queries, 15 languages, max score 1.5/5

### Session 44
- I-4-GITHUB complete: all 4 AI layer files live on main
- P1 AI layer updated: derivation_source arrays added to all F-claims
- CBT I-9 passed

### 13-paper series
| # | Paper | Type | Status |
|---|-------|------|--------|
| 1 | Fractal Rhythm Model | law_A | PHASE-READY |
| 2 | Universality Classes | law_B | Next |
| 3 | Measurement Protocol | protocol | Planned |
| 4 | Scale Coupling | theory | Planned |
| 5 | Information Geometry | theory | Planned |
| 6 | Quantum Substrate | theory | Planned |
| 7 | Consciousness | theory | Planned |
| 8 | Economic Networks | application | Planned |
| 9 | Social Dynamics | application | Planned |
| 10 | Biological Rhythms | application | Planned |
| 11 | Cosmological | application | Planned |
| 12 | Computational | application | Planned |
| 13 | Synthesis | synthesis | Planned |

## Verification Architecture

- **Claim types**: A (axiom), D (definition/derivation), F (falsifiable)
- **Falsification syntax**: I-2 5-part (FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT)
- **Inference rules**: IR-1 through IR-8
- **Verification protocol**: Fracttalix Phase 1-2 Merged, S42, Stage 2.1
- **AI Layer schema**: v2-S42

## Conventions

- **Citation format**: [Fracttalix Paper N, AI-Layer, Claim ID]
- **Session numbering**: S1–S44 (current)
- **Build Table**: tracks all 13 papers, milestones, dependencies
- **Sentinel**: Python script (v12.0) — corpus integrity checker
- **Meta-Kaizen (MK)**: AI layer documenting the verification process itself

## Key files in repo

```
Fracttalix/
├── ai-layers/
│   ├── P1-ai-layer.json          ← Paper 1 claim registry
│   ├── MK-P1-ai-layer.json       ← Meta-Kaizen layer
│   ├── DRP1-ai-layer.json        ← Dual-Reader Publishing layer
│   ├── ai-layer-schema.json       ← Schema definition
│   └── process-graph.json         ← Process dependency graph
├── docs/
│   ├── FRM_SeriesBuildTable_v1.5.md
│   ├── claude-bootstrap.md
│   └── handoff-S44.md
├── journal/
│   ├── journal_index.md
│   ├── session_36_notes.md
│   ├── session_36_theoretical_advance.md
│   └── session_36_complete.md
├── fracttalix_sentinel_v1200.py  ← Sentinel v12.0
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
└── legal/
```

---

*To update this file: edit after each session with new state, commit to main.*
