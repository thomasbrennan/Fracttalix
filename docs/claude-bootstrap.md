# Fracttalix Session Bootstrap

> Paste this file into a new Claude.ai conversation to restore full project context.
> Last updated: Session 49, 2026-03-11.

---

## How to use

1. **Start a new Claude.ai conversation**
2. **Upload or paste this file** as your first message
3. Claude will have full project context — no re-explanation needed

For maximum fidelity, also attach these live files from GitHub:
- [P1 AI Layer](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/P1-ai-layer.json)
- [MK-P1 AI Layer](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/MK-P1-ai-layer.json)
- [SFW-1 AI Layer](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/SFW1-ai-layer.json)
- [Build Table](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/docs/FRM_SeriesBuildTable_v1.5.md)
- [AI Layer Schema](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/ai-layer-schema.json)
- [Process Graph](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/process-graph.json)

---

## Project Identity

- **Corpus**: Fracttalix — 21-object unified corpus on the Fractal Rhythm Model (FRM)
- **Author**: Thomas Brennan
- **AI collaborators**: Claude (Anthropic), Grok (xAI)
- **Licence**: CC0 public domain
- **Repo**: github.com/thomasbrennan/Fracttalix
- **DOI**: 10.5281/zenodo.18859299

## The FRM — The Law

**Core claim**: A network is structure and rhythmicity. No exceptions.

**Functional form**: f(t) = B + A·e^(−λt)·cos(ωt + φ)

**Universal constants** (all derived, none fitted):
| Constant | Value | Expression | Meaning |
|----------|-------|------------|---------|
| β | 0.5 | 1/2 | Quarter-wave resonance coefficient at Hopf criticality |
| k* | 1.5708 | π/2 | Critical feedback gain at Hopf bifurcation |
| Γ | 3.4674 | 1 + π²/4 | Universal loop impedance constant |

**Derived parameters**:
- ω = π/(2·τ_gen) — characteristic frequency
- λ ≈ |α|/(Γ·τ_gen) — decay rate (leading order, 3.61% mean error)
- B, A, φ — initial conditions (not free parameters)

**Scope boundary**: Hopf bifurcation (topological, not empirical)
- In scope: μ < 0 (damped oscillators)
- Out of scope: μ > 0 (limit cycles)

## Current State (Session 48)

### Phase readiness
- **P1**: PHASE-READY. CBT I-9 all 7 steps passed. One placeholder (PH-1.1: β=1/2 class-level empirical test, pending P2 C-2.1). Does not block.
- **Sentinel**: v12.1.0, package refactored (`pip install fracttalix`), 374 tests

### Key results (Session 43)
- β = 1/2 analytically derived (Hopf quarter-wave theorem)
- λ = |α|/(Γ·τ_gen) derived (perturbation expansion)
- Γ = 1 + π²/4 derived (loop impedance)
- T = 4·τ_gen → circadian period prediction (24 hr from 6 hr, no fitting)
- Stuart-Landau connection confirmed (R² > 0.99)
- Adversarial battery: 3 correct rejections, 1 confirmation
- Prior art: 52+ queries, 15 languages, max score 1.5/5

### Session 48 (current)
- **Phase 1**: Merged sentinel branch, fixed 35 schema errors, added CI, corpus status script, process graph v2, journal entries, bootstrap reconciliation
- **Phase 2**: AI layers for all 12 papers (P2-P5 retroactive, P6-P12 scaffolds), MK-P1 schema fixes (phase_ready fields, derivation_source nulls), cross-paper consistency checker, reproducibility manifest, P1 series_position fix (13→12), process graph updated to 15 live assets

### Fracttalix Track (16 objects)

| # | Title | Type | Act | Status |
|---|-------|------|-----|--------|
| P0 | Canonical Build Process Standard | methodology_D | GOV | COMPLETE |
| P1 | The Fractal Rhythm Model: A Universal Law | law_A | I | PHASE-READY |
| P2 | Derivation and Universality: β=1/2 Critical Exponent | derivation_B | I | PHASE-READY |
| P3 | FRM Measurement and Diagnostics | methodology_D | I | PHASE-READY |
| P4 | Biological Systems | application_C | II | QUEUED |
| P5 | Neural and Pharmacological Systems | application_C | II | QUEUED |
| P6 | The Central Paper | derivation_B | II | QUEUED |
| P7 | Climate and Earth Systems | application_C | III | QUEUED |
| P8 | Software and Engineered Systems | application_C | III | QUEUED |
| P9 | Corpus Health and Endogenous Scheduling | synthesis | III | QUEUED |
| P10 | MetaKaizen General Theory | synthesis | III | QUEUED |
| P11 | Civilisational Dataset Fitting | application_C | III | QUEUED |
| P12 | Civilisational Sentinel | synthesis | III | QUEUED |
| P13 | Corpus Completeness Instrument | synthesis | III | QUEUED |
| P14 | Dual Reader Standard (DRP-1) | methodology_D | PARALLEL | v0.3 |
| SFW-1 | Sentinel v12 | software | SOFTWARE | JOSS SUBMITTED |

### Meta-Kaizen Track (5 objects — all PUBLISHED)

| # | Title | Series |
|---|-------|--------|
| MK-P1 | General Theory and Algorithmic Framework | MK 1/5 |
| MK-P2 | Networked Implementation and Governance Closure | MK 2/5 |
| MK-P3 | Reasoning Propagation and Institutional Memory | MK 3/5 |
| MK-P4 | Closed-Loop Governance: FRM Integration | MK 4/5 |
| MK-P5 | On the Decision to Act: Rational Action at Tipping Points | MK 5/5 |

## Verification Architecture

- **Claim types**: A (axiom), D (definition/derivation), F (falsifiable)
- **Falsification syntax**: I-2 5-part (FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT)
- **Inference rules**: IR-1 through IR-8
- **AI Layer schema**: v2-S42
- **KVS Score**: 0.832 (threshold κ = 0.75)

### AI Layers (15 total)
| ID | Paper | Status | File |
|----|-------|--------|------|
| P1 | FRM Paper 1 | PHASE-READY | ai-layers/P1-ai-layer.json |
| P2 | Networked Implementation | NOT-PHASE-READY | ai-layers/P2-ai-layer.json |
| P3 | The Reasoning Network | NOT-PHASE-READY | ai-layers/P3-ai-layer.json |
| P4 | The Fractal Rhythm Model | NOT-PHASE-READY | ai-layers/P4-ai-layer.json |
| P5 | On the Decision to Act | NOT-PHASE-READY | ai-layers/P5-ai-layer.json |
| P6–P12 | Act III papers | Scaffold (v0) | ai-layers/P{6-12}-ai-layer.json |
| MK-P1 | Meta-Kaizen Paper 1 | PHASE-READY | ai-layers/MK-P1-ai-layer.json |
| DRP-1 | Dual-Reader Publishing | PHASE-READY | ai-layers/DRP1-ai-layer.json |
| SFW-1 | Sentinel Software | PHASE-READY | ai-layers/SFW1-ai-layer.json |

## Conventions

- **Citation format**: [Fracttalix Paper N, AI-Layer, Claim ID]
- **Session numbering**: S1–S49 (current)
- **Build Table**: tracks all 21 corpus objects, milestones, dependencies (CorpusArch v9)
- **Sentinel**: Python package (v12.1) — streaming anomaly detector implementing the three-channel model
- **Meta-Kaizen (MK)**: Parallel 5-paper verification corpus

## Key files in repo

```
Fracttalix/
├── ai-layers/
│   ├── P1-ai-layer.json
│   ├── MK-P1-ai-layer.json
│   ├── DRP1-ai-layer.json
│   ├── SFW1-ai-layer.json
│   ├── ai-layer-schema.json
│   └── process-graph.json
├── docs/
│   ├── FRM_SeriesBuildTable_v1.5.md
│   ├── claude-bootstrap.md
│   └── handoff-S44.md
├── journal/
│   ├── journal_index.md
│   ├── session_36_notes.md
│   ├── session_43_notes.md
│   ├── session_44_notes.md
│   └── session_48_notes.md
├── fracttalix/              ← Package (v12.1)
│   ├── __init__.py
│   ├── config.py
│   ├── detector.py
│   ├── steps/
│   └── ...
├── tests/                   ← 374 tests
├── benchmark/               ← Anomaly archetype benchmarks
├── paper/                   ← JOSS paper draft
├── scripts/                 ← Validation, status, and consistency scripts
├── legacy/                  ← Archived monolith versions
├── README.md
├── pyproject.toml
├── CHANGELOG.md
├── LICENSE (CC0)
└── .github/workflows/       ← CI (tests, lint, AI layer validation)
```

---

*To update this file: edit after each session with new state, commit to main.*
