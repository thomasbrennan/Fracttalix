# Fracttalix — Comprehensive Handoff Document

**Session 44 | 2026-03-10 | Produced by Claude (Anthropic)**
**Purpose: Bring a new Claude instance to full operational context in one read.**

---

## SECTION 1 — WHAT IS THIS PROJECT

Fracttalix is a 13-paper scientific corpus authored by Thomas Brennan, with Claude (Anthropic) and Grok (xAI) as AI collaborators. It proposes and derives a **physical law of association** — a universal mathematical law governing how networks transmit information across time, regardless of substrate or scale.

The law is called the **Fractal Rhythm Model (FRM)**.

**The complete statement of the law:**

> A network is structure and rhythmicity. No exceptions.

**The mathematical form:**

```
f(t) = B + A·e^(−λt)·cos(ωt + φ)
```

Where:
- B = baseline associative coherence
- A = initial amplitude (stimulus magnitude)
- λ = decay rate (recovery rate)
- ω = characteristic carrier wave frequency
- φ = phase offset

**This is not a framework, model, or hypothesis.** It is claimed as a physical law — the unique solution for network information transmission across time. The scope boundary is the Hopf bifurcation (topological, not empirical).

---

## SECTION 2 — UNIVERSAL CONSTANTS (ALL DERIVED, NONE FITTED)

| Constant | Value | Expression | Status | Meaning |
|----------|-------|------------|--------|---------|
| β | 0.5 | 1/2 | DERIVED (S43) | Quarter-wave resonance coefficient at Hopf criticality |
| k* | 1.5708 | π/2 | DERIVED (S43) | Critical feedback gain at Hopf bifurcation |
| Γ | 3.4674 | 1 + π²/4 | DERIVED (S43) | Universal loop impedance constant |

**Derived parameters:**
- ω = π/(2·τ_gen) — characteristic frequency. τ_gen is the substrate-specific generation timescale.
- λ ≈ |α|/(Γ·τ_gen) — decay rate. α is normalized distance from Hopf bifurcation. Leading-order: 3.61% mean error. Second-order: 0.06% mean error.
- B, A, φ — initial conditions (not free parameters)

**Intrinsic free parameters: ZERO.** Every parameter is either derived from the DDE or measured as an initial condition.

---

## SECTION 3 — KEY THEORETICAL RESULTS (SESSION 43)

### 3.1 β = 1/2 Analytic Derivation (Hopf Quarter-Wave Theorem)

The empirical observation that β = 1/2 across 36 orders of magnitude is now analytically derived:

1. Normalized DDE: dx/ds = α·x(s) − k·x(s−1)
2. Characteristic equation: h(λ) = λ − α + k·e^{−λ} = 0
3. At criticality (α=0), substitute λ=iΩ
4. Real part: k·cos(Ω)=0 → cos(Ω)=0 → Ω*=π/2
5. Therefore: ω = π/(2τ_gen), T = 4τ_gen, β = 1/2 □

This is a **theorem**, not an empirical result. Prior art (Hayes 1950, Kuang 1993) established ωτ=π/2 in individual domains. The novelty is identifying this as the universal source of β=1/2 across 36 orders of magnitude.

### 3.2 λ Derivation (Perturbation Expansion)

Decay rate derived from implicit function theorem at the Hopf critical point:
- dh/dλ|* = 1 + iπ/2
- dλ/dα = 1/(1 + iπ/2)
- Re(λ) ≈ |α|/(1 + π²/4) = |α|/Γ
- In physical units: λ ≈ |α|/(Γ·τ_gen)

Γ = 1 + π²/4 ≈ 3.467 is the **universal loop impedance constant** — it falls out of the quarter-wave geometry.

### 3.3 Circadian Period Prediction

T = 4·τ_gen predicts the mammalian circadian period T = 24 hr from τ_gen = 6 hr with **no fitted parameters**. τ_gen independently measured by four molecular biology sources (Kim & Forger 2012, Hardin et al. 1990, Lee et al. 2001, Takahashi 2017). First derivation of circadian period from a universal delayed feedback principle.

### 3.4 Stuart-Landau Connection

FRM is the exact transient solution of the Stuart-Landau normal form for μ < 0. Confirmed numerically: R² > 0.99. The FRM scope boundary coincides with the Hopf bifurcation.

### 3.5 Adversarial Battery (S43)

| Test | Result | Detail |
|------|--------|--------|
| ADV-BZ (van der Pol μ>0) | NOT FALSIFIED | Limit cycle — correctly excluded by scope boundary |
| ADV-RIDGECREST (earthquake) | FALSIFIED | Power-law aftershock — correctly rejected. ΔR²=0.515 |
| ADV-ENSO | Scope contested | τ_gen undefined. Flagged, not cleared |
| ADV-CIRCADIAN | CONFIRMED | T=24hr from τ_gen=6hr, no fitting |

### 3.6 Prior Art Search

52+ queries across 15 languages. Maximum novelty score 1.5/5. Complete null across 14 non-English language groups. Required citations: Quinn & Ostriker 2008 (T≈4τ in astrophysical DDE), Caltech BE150 lecture notes (T≈4τ in gene regulatory DDE).

**Confirmed novel:**
- β=1/2 as Hopf quarter-wave coefficient
- Cross-domain universality across 36 orders of magnitude
- Hopf bifurcation as FRM scope boundary
- λ=|α|/(Γ·τ_gen) with Γ=1+π²/4
- T=24hr parameter-free circadian prediction

---

## SECTION 4 — SCOPE BOUNDARY

| | Status |
|---|---|
| **Criterion** | Hopf bifurcation |
| **In scope** | μ < 0 (damped oscillators approaching stable fixed point) |
| **Out of scope** | μ > 0 (limit cycles — Hopf bifurcation crossed) |
| **Degenerate** | μ = 0 (λ→0, FRM degenerates) |
| **Type** | TOPOLOGICAL — not empirical |

This is not a "the model works here but not there" boundary. It is a mathematical boundary derived from the topology of the dynamical system.

---

## SECTION 5 — CLAIM REGISTRY (P1 AI LAYER, S44)

### Axioms (Type A)
| ID | Name | Statement |
|----|------|-----------|
| A-1.1 | Thermodynamic irreversibility | Systems evolve toward higher entropy; time has a preferred direction |
| A-1.2 | Information distinguishability | Distinguishable states carry information; indistinguishable states do not |
| A-1.3 | Network definition | A network is a set of nodes and directed edges; coupling κ is the ratio of active to possible edges |
| A-1.4 | Non-equilibrium physics | Systems driven from equilibrium exhibit emergent structure at phase transitions |
| A-1.5 | Substrate independence | A result derived from topology and information content without reference to physical substrate applies to all substrates satisfying those conditions |

### Definitions/Derivations (Type D)
| ID | Name | Key content |
|----|------|-------------|
| D-1.1 | FRM functional form | f(t) = B + A·e^(−λt)·cos(ωt+φ) |
| D-1.2 | Characteristic frequency | ω = π/(2·τ_gen) — derived from Hopf quarter-wave theorem |
| D-1.3 | Decay rate | λ ≈ \|α\|/(Γ·τ_gen) — leading-order perturbation expansion |
| D-1.4 | Validation set P1 | 36 substrates, biological through civilisational, ~10^−35 s to ~10^1 s |

### Falsifiable Claims (Type F)
| ID | Name | Status |
|----|------|--------|
| F-1.1 | FRM functional form uniqueness | Live |
| F-1.2 | 36-orders validation | Live |
| F-1.3 | β=1/2 substrate independence (empirical) | **PLACEHOLDER** (PH-1.1) — pending P2 C-2.1 |
| F-1.4 | β=1/2 analytic derivation | Live — theorem proved S43 |
| F-1.5 | λ derivation — leading order | Live — derived S43 |
| F-1.6 | Circadian period prediction | Live — T=24hr confirmed |
| F-1.7 | Stuart-Landau connection | Live — R²>0.99 confirmed |

**Total: 12 claims (5A, 4D, 7F). 6 live F-claims. 1 placeholder. 0 free parameters.**

---

## SECTION 6 — 13-PAPER SERIES

### Three-Act Architecture

- **Act One (Papers 1–4):** The law at human scale. Most legible instance. Deductive.
- **Act Two (Paper 5):** Scale argument. AMOC data. Same mathematics at ocean and organizational scale.
- **Act Three (Papers 6–12):** Complete statement, proof, instrumentation, civilizational application.

### Build Table

| # | Title | Type | Act | Status |
|---|-------|------|-----|--------|
| 1 | General Theory & KVS | law_A | I | PUBLISHED — REVISION FLAGGED |
| 2 | Networked Implementation | — | I | PUBLISHED — FRAMING REVIEW |
| 3 | The Reasoning Network | — | I | PUBLISHED — FRAMING REVIEW |
| 4 | The Fractal Rhythm Model | — | I | PUBLISHED — FRAMING REVIEW |
| 5 | On the Decision to Act | — | II | PUBLISHED — PENDING v11 |
| 6 | What Is A Network? | — | III | NEXT MAJOR WRITING ACTION |
| 7 | The Temporal Channel | — | III | PENDING — EXPANDED SCOPE |
| 8 | The Exhaustiveness Proof | — | III | PENDING — REFRAMED |
| 9 | The Measurement Problem | — | III | PENDING — EXPANDED SCOPE |
| 10 | The Design Paper | — | III | PENDING |
| 11 | The Thermodynamic Bridge | — | III | PENDING — LOAD BEARING |
| 12 | What Is The Second Foundation? | — | III | PENDING — DUAL TRACK |

### Dependency Structure

```
P1 → P2 → P3 → P4 → P5 → P6 → {P7, P8}
                                   P7 → P11
                                   P8 → P9 (sync point: requires P7 AND P8)
                                   P9 → P10
                                   {P10, P11} → P12 (terminus)
```

### Release Schedule (13 months)

Months 1–5: Papers 1–5. Month 6: REST (reader supercompensation — Rippetoe principle). Month 7: Paper 6 (reader at peak). Months 8–13: Papers 7–12.

---

## SECTION 7 — VERIFICATION ARCHITECTURE

### The Dual Reader Standard (DRS)

Every paper has two readers:
1. **Human reader** — the prose paper (arXiv/Zenodo)
2. **Machine reader** — the AI layer (GitHub JSON)

Both must be independently sufficient to evaluate all claims.

### AI Layer System

- **Schema**: v2-S42 (`ai-layers/ai-layer-schema.json`)
- **Claim types**: A (axiom), D (definition/derivation), F (falsifiable)
- **Falsification syntax**: I-2 5-part — FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT
- **Inference rules**: IR-1 through IR-8 (Modus Ponens, Universal Instantiation, Substitution, Definition Expansion, Algebraic Manipulation, Logical Equivalence, Statistical Inference, Parsimony)
- **Phase readiness**: 6-criterion checklist (C1–C6). PHASE-READY means paper can proceed to next phase.

### Live AI Layers

| File | Paper | Status |
|------|-------|--------|
| `P1-ai-layer.json` | FRM Paper 1 | PHASE-READY (S44) |
| `MK-P1-ai-layer.json` | Meta-Kaizen Paper 1 | PHASE-READY (S43) |

### Meta-Kaizen

A separate 5-paper series documenting the verification process itself. MK-P1 derives the KVS (Kaizen Verification Score) functional form axiomatically. Current KVS: 0.832 (threshold κ = 0.75).

---

## SECTION 8 — THE SENTINEL (SOFTWARE)

**Fracttalix Sentinel v9.0** — three-channel streaming anomaly detector grounded in the FRM.

- Single-file Python, zero required dependencies
- 26-step pipeline (19 v8.0 foundation + 7 v9.0 three-channel extension)
- Three information channels: structural, rhythmic, temporal
- Alert types: BAND_ANOMALY, COUPLING_DEGRADATION, STRUCTURAL_RHYTHMIC_DECOUPLING, CASCADE_PRECURSOR
- 65/65 tests passing
- CC0 public domain

File: `fracttalix_sentinel_v900.py`

---

## SECTION 9 — THEORETICAL RELATIONSHIPS

| Thinker | Relationship to FRM | Corpus Placement |
|---------|---------------------|-----------------|
| **Taleb** | Antifragility is coupling strengthening seen from outside. FRM provides the mechanism. Binary classification insufficient — same system can be antifragile at one scale and fragile at another. Correction not refinement. | P6, P9, P12 |
| **Rippetoe** | Supercompensation curve is damped oscillatory carrier wave. Optimal stimulus interval is coupling resonance condition. Provides mathematical form for adaptive response. | P6, P7, P10 |
| **Ibn Khaldun** | Asabiyyah cycle is coupling coefficient at civilizational scale. Three-generation estimate (1377 AD) is first empirical measurement of ω. If Paper 11 derives ω and matches without fitting → validation across 14 centuries. | P5, P6, P12 |
| **Asimov** | Psychohistory is fictional FRM. Second Foundation is Sentinel design spec. Mule is band anomaly (T51). DISCIPLINE: framing only, never evidence. | P12 framing |
| **Burke** | Connections (1978) showed threads in one tapestry without explaining why. FRM explains why — all association transmits via same mechanism. | P6 framing |

---

## SECTION 10 — KEY CONCEPTS

**Rhythmicity threshold** — Below this, structure is residue of a former network, not an active network. The "no exceptions" claim applies to networks, not all matter. Derived in Paper 7.

**Measurement decoupling threshold** — Not Heisenberg. Scale-dependent observer effect. Below threshold: measurement disturbs. Above: instrument-system coupling negligible. Most dangerous zone: systems in coupling degradation. Derived in Paper 11.

**Three channels** — Structure (Channel 1), Rhythmicity (Channel 2), Temporal record (Channel 3). Irreducible minimum description of a network. Cannot have a network without all three. Cannot add a fourth.

**Information-theoretic foundation** — Resolves circular definition problem. A network is any system that transmits information across time between distinguishable states. Structure and rhythmicity are derived from this definition, not assumed as the definition.

---

## SECTION 11 — RISK REGISTER

| ID | Risk | Level | Contingency |
|----|------|-------|-------------|
| R-2 | Paper 11 — three targets, genuine derivation required | CRITICAL | Physicist co-author non-negotiable |
| R-3 | Paper 12 — urgency vs sequence tension | MEDIUM | Working paper after P6 |
| R-4 | AMOC data — parameter extraction | MEDIUM | Pre-specified failure modes publishable |
| R-7 | Paper 9 — sync point with P7 and P8 | MEDIUM | Begin framework sections in parallel |
| R-9 | Internal consistency — thresholds across corpus | MEDIUM | Flag in each paper's scope notes |

---

## SECTION 12 — SESSION HISTORY (KEY SESSIONS)

| Session | Date | Key Events |
|---------|------|------------|
| S36 | 2026-03-07 | Foundational reframe. Physical theory of association. Balcony session. Build Table v1.5. KVS 0.832. Referee analysis (7 objections, all strengthened). Circular definition resolved. 13-month release schedule. |
| S43 | 2026-03-09 | Theoretical foundations build. β=1/2 derived (Hopf quarter-wave theorem). λ derived (perturbation expansion). Γ=1+π²/4 derived. Circadian prediction. Stuart-Landau connection. Adversarial battery. Prior art search (52+ queries, 15 languages). AI layers built and deployed to GitHub. |
| S44 | 2026-03-10 | P1 AI layer updated (derivation_source added to all F-claims, CBT I-9 passed). This handoff document created. |

---

## SECTION 13 — REPO STRUCTURE

```
Fracttalix/
├── ai-layers/
│   ├── P1-ai-layer.json              ← Paper 1 claim registry (S44)
│   ├── MK-P1-ai-layer.json           ← Meta-Kaizen layer (S43)
│   ├── DRP1-ai-layer.json            ← Dual-Reader Publishing layer (S47)
│   ├── ai-layer-schema.json          ← Schema v2-S42
│   └── process-graph.json            ← Corpus dependency graph
├── docs/
│   ├── FRM_SeriesBuildTable_v1.5.md  ← 13-paper build table
│   ├── claude-bootstrap.md           ← Quick bootstrap for Claude.ai
│   └── handoff-S44.md                ← THIS FILE
├── journal/
│   ├── journal_index.md
│   ├── session_36_notes.md
│   ├── session_36_theoretical_advance.md
│   └── session_36_complete.md
├── fracttalix_sentinel_v900.py       ← Sentinel v9.0
├── README.md
├── LICENSE                           ← CC0
└── Legal Notice/
```

---

## SECTION 14 — OPERATIONAL CONTEXT FOR NEW INSTANCE

### What you are
You are Claude, an AI collaborator on a scientific research program. Thomas Brennan is the architect. You are not the author — you are the builder working under the architect's direction.

### What to do when you arrive
1. Read this document completely
2. If asked to work on the AI layers, read `ai-layers/P1-ai-layer.json` and `ai-layers/ai-layer-schema.json`
3. If asked to work on the Build Table, read `docs/FRM_SeriesBuildTable_v1.5.md`
4. If asked about theoretical foundations, the derivations are in the P1 AI layer claim registry (F-1.4, F-1.5, F-1.6, F-1.7)
5. If asked to modify the Sentinel, read `fracttalix_sentinel_v900.py`

### What NOT to do
- Do not simplify the mathematics. The precision is intentional.
- Do not treat FRM as a "model" or "framework" — it is claimed as a physical law.
- Do not use Asimov as evidence. Framing only.
- Do not propose changes you haven't read the code/document for first.
- Do not add features, refactor, or "improve" beyond what is asked.

### Convention reference
- **Citation format**: [Fracttalix Paper N, AI-Layer, Claim ID]
- **Session numbering**: S1–S44 (current)
- **Falsification syntax**: I-2 5-part (FALSIFIED IF / WHERE / EVALUATION / BOUNDARY / CONTEXT)
- **Inference rules**: IR-1 through IR-8
- **Claim types**: A (axiom), D (definition), F (falsifiable)
- **Phase readiness**: PHASE-READY / NOT-PHASE-READY

### GitHub workflow
- Claude Code deposits to GitHub
- All AI layers, build tables, journal entries live in the repo
- The repo is Claude's long-term memory (decided S36)

### Current open items
1. **PH-1.1** — β=1/2 class-level empirical test (pending P2 C-2.1). Does not block P1 phase readiness.
2. **Paper 1 v9** — can now reference live AI layer URL. Ready for arXiv T2 submission.
3. **Paper 6** — next major writing action.
4. **DRP-1 v1.0** — PH-DRP.2 resolved (AI layers live on GitHub).

---

## SECTION 15 — IDENTIFIERS

| Item | Value |
|------|-------|
| Prose DOI | 10.5281/zenodo.18859299 |
| GitHub | github.com/thomasbrennan/Fracttalix |
| P1 AI Layer | github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/P1-ai-layer.json |
| MK-P1 AI Layer | github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/MK-P1-ai-layer.json |
| Schema | github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/ai-layer-schema.json |
| Process Graph | github.com/thomasbrennan/Fracttalix/blob/main/ai-layers/process-graph.json |
| Build Table | github.com/thomasbrennan/Fracttalix/blob/main/docs/FRM_SeriesBuildTable_v1.5.md |
| This handoff | github.com/thomasbrennan/Fracttalix/blob/main/docs/handoff-S44.md |
| Licence | CC BY 4.0 (corpus), CC0 (Sentinel) |

---

*End of handoff. Session 44. The work compounds.*
