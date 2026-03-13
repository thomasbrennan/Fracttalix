# Fracttalix Session Bootstrap

> Paste this file into a new Claude.ai conversation to restore full project context.
> Last updated: Session 55 (Archivist), 2026-03-12.

---

## How to use

1. **Start a new Claude.ai conversation**
2. **Upload or paste this file** as your first message
3. Claude will have full project context — no re-explanation needed

For maximum fidelity, also attach these live files from GitHub:
- [P1 AI Layer](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/P1-ai-layer.json)
- [MK-P1 AI Layer](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/MK-P1-ai-layer.json)
- [MK-P5 AI Layer](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/MK-P5-ai-layer.json)
- [SFW-1 AI Layer](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/SFW1-ai-layer.json)
- [Build Table](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/docs/FRM_SeriesBuildTable_v1.5.md)
- [AI Layer Schema](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/ai-layer-schema.json)
- [Process Graph](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/process-graph.json)
- [Falsification Kernel](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/falsification-kernel.md)
- [Claim Registry Index](https://raw.githubusercontent.com/thomasbrennan/Fracttalix/main/ai-layers/claim-registry-index.md)

---

## Project Identity

- **Corpus**: Fracttalix — 23-object unified corpus on the Fractal Rhythm Model (FRM)
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

## Current State (Session 54)

### Phase readiness
- **P1**: PHASE-READY (v13). CBT I-9 all 7 steps passed. PH-1.1 resolved via P2 C-2.4.
- **P2**: Phase 1 PHASE-READY (S49). AI layer v4. IR inventory IR-1–IR-11. Phases 2–5 not yet executed.
- **P3**: QUEUED. No build plan. Gated on P2 full PHASE-READY.
- **Sentinel**: v12.2.0, package refactored (`pip install fracttalix`), 434 tests

### Key results (Session 43)
- β = 1/2 analytically derived (Hopf quarter-wave theorem)
- λ = |α|/(Γ·τ_gen) derived (perturbation expansion)
- Γ = 1 + π²/4 derived (loop impedance)
- T = 4·τ_gen → circadian period prediction (24 hr from 6 hr, no fitting)
- Stuart-Landau connection confirmed (R² > 0.99)
- Adversarial battery: 3 correct rejections, 1 confirmation
- Prior art: 52+ queries, 15 languages, max score 1.5/5

### Recent sessions
- **S49**: MK-P5 AI layer deployed (PHASE-READY). Fortuna Process, Virtù Window. 89 claims.
- **S50**: Full DRS deployment. Three-layer architecture operational. Layer 0 (falsification-kernel.md). MK-P2/P3/P4 AI layers built.
- **S51**: MK-P6 written (DRS for Software). DRS-ARCH paper and AI layer. GVP v1.0 spec. Schema v3-S51.
- **S52**: CorpusArch v10 reconciliation. P2/P3 status corrected. Protocol Amendment Log. Build Table v3.0.
- **S53–S54**: DRP-2 AI layer deposited. DRP-3 build process (D3-0.0 through D3-3.0). Claim registry backfill. ARCH-2 pass-forward register. C-DRP.8 UMP correction.

### Fracttalix Track (16 objects)

| # | Title | Type | Act | Status |
|---|-------|------|-----|--------|
| P0 | Canonical Build Process Standard | methodology_D | GOV | COMPLETE |
| P1 | The Fractal Rhythm Model: A Universal Law | law_A | I | PHASE-READY |
| P2 | Derivation and Universality: β=1/2 Critical Exponent | derivation_B | I | Phase 1 PHASE-READY (S49) |
| P3 | FRM Measurement and Diagnostics | methodology_D | I | QUEUED |
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

### Meta-Kaizen Track (6 objects — all PUBLISHED)

| # | Title | Series |
|---|-------|--------|
| MK-P1 | General Theory and Algorithmic Framework | MK 1/6 |
| MK-P2 | Networked Implementation and Governance Closure | MK 2/6 |
| MK-P3 | Reasoning Propagation and Institutional Memory | MK 3/6 |
| MK-P4 | Closed-Loop Governance: FRM Integration | MK 4/6 |
| MK-P5 | On the Decision to Act: Rational Action at Tipping Points | MK 5/6 |
| MK-P6 | The Dual Reader Standard for Software | MK 6/6 |

## Verification Architecture

- **Claim types**: A (axiom), D (definition), F (falsifiable)
- **Falsification syntax**: K = (P, O, M, B) — 5-part (FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT)
- **Inference rules**: IR-1 through IR-13
- **AI Layer schema**: v3-S51
- **Falsification kernel**: Layer 0 — falsification-kernel.md v1.1
- **KVS Score**: 0.832 (threshold κ = 0.75)
- **κ_corpus**: 0.91

### AI Layers (21 total)
| ID | Paper | Status | File |
|----|-------|--------|------|
| P1 | FRM Paper 1 | PHASE-READY | ai-layers/P1-ai-layer.json |
| P2 | Derivation and Universality | PHASE-READY | ai-layers/P2-ai-layer.json |
| P3 | FRM Measurement and Diagnostics | PHASE-READY | ai-layers/P3-ai-layer.json |
| P4 | Biological Systems | NOT-PHASE-READY | ai-layers/P4-ai-layer.json |
| P5 | Neural and Pharmacological Systems | NOT-PHASE-READY | ai-layers/P5-ai-layer.json |
| P6–P12 | Act II/III papers | Scaffold (v0) | ai-layers/P{6-12}-ai-layer.json |
| MK-P1 | Meta-Kaizen General Theory | PHASE-READY | ai-layers/MK-P1-ai-layer.json |
| MK-P2 | Networked Implementation | PHASE-READY | ai-layers/MK-P2-ai-layer.json |
| MK-P3 | Reasoning Propagation | PHASE-READY | ai-layers/MK-P3-ai-layer.json |
| MK-P4 | Closed-Loop Governance | PHASE-READY | ai-layers/MK-P4-ai-layer.json |
| MK-P5 | On the Decision to Act | PHASE-READY | ai-layers/MK-P5-ai-layer.json |
| MK-P6 | DRS for Software | NOT-PHASE-READY | ai-layers/MK-P6-ai-layer.json |
| DRP-1 | Dual-Reader Publishing | PHASE-READY | ai-layers/DRP1-ai-layer.json |
| DRS-ARCH | DRS Architecture Specification | NOT-PHASE-READY | ai-layers/DRS-ARCH-ai-layer.json |
| SFW-1 | Sentinel Software | PHASE-READY | ai-layers/SFW1-ai-layer.json |

### Verification Status
```
AI Layers:          21
Phase-Ready:        10/21
Total Claims:       140 (A:31 D:46 F:63)
Open Placeholders:  17
Cross-references:   225+ derivation_source entries
Schema:             v3-S51
```

## Protocol Amendment Log

| ID | Session | Amendment |
|----|---------|-----------|
| S48-A1 | S48 | `principle_10_audit` array field added to AI layer schema. |
| S48-A2 | S48 | I-9 Step 4 circularity detection added. |
| S49-A3 | S49 | Type B papers must declare own IR inventory. |

## Conventions

- **Citation format**: [Fracttalix Paper N, AI-Layer, Claim ID]
- **Session numbering**: S1–S55 (current)
- **Build Table**: tracks all 23 corpus objects, milestones, dependencies (CorpusArch v10)
- **Sentinel**: Python package (v12.1) — streaming anomaly detector implementing the three-channel model
- **Meta-Kaizen (MK)**: Parallel 6-paper verification and governance corpus

## Key files in repo

```
Fracttalix/
├── ai-layers/
│   ├── ai-layer-schema.json         ← Schema v3-S51
│   ├── falsification-kernel.md      ← Layer 0 semantic spec
│   ├── process-graph.json           ← Paper dependency graph
│   ├── claim-registry-index.md      ← Cross-project claim reference
│   ├── P1-ai-layer.json .. P12-ai-layer.json
│   ├── MK-P1-ai-layer.json .. MK-P6-ai-layer.json
│   ├── DRP1-ai-layer.json
│   ├── DRS-ARCH-ai-layer.json
│   └── SFW1-ai-layer.json
├── docs/
│   ├── FRM_SeriesBuildTable_v1.5.md ← Living architecture document
│   ├── GVP-spec.md                  ← Grounded Verification Protocol
│   ├── claude-bootstrap.md          ← This file
│   └── handoff-S44.md              ← Comprehensive S44 handoff
├── journal/
│   ├── journal_index.md
│   └── session_36_notes.md .. session_52_notes.md
├── paper/
│   ├── DRS-Architecture.md
│   ├── paper.md                     ← JOSS software paper
│   └── meta-kaizen/                 ← MK-P1 through MK-P6
├── fracttalix/                      ← Package (v12.2, 37-step pipeline)
├── tests/                           ← 434 tests
├── benchmark/                       ← Anomaly archetype benchmarks
├── scripts/                         ← Validation, status, consistency scripts
├── legacy/                          ← Archived monolith versions (v7.6–v11.0)
├── examples/                        ← Quickstart notebook + tutorials
├── README.md
├── pyproject.toml
├── CHANGELOG.md
├── REPRODUCIBILITY.md
├── LICENSE (CC0)
└── .github/workflows/               ← CI (tests, AI layer validation, release)
```

---

*To update this file: edit after each session with new state, commit to main.*
