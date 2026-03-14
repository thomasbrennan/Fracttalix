# Reproducibility Manifest

**Fracttalix v12.1 — Fractal Rhythm Model Corpus**
**Last verified: Session 48, 2026-03-10**

---

## How to reproduce all results

### Prerequisites

```bash
pip install fracttalix          # or: pip install -e .
pip install pytest ruff         # dev dependencies
```

Python 3.9+ required. Zero external runtime dependencies.

### 1. Run the full test suite

```bash
pytest tests/ -v                # 434 tests
python fracttalix/detector.py   # 65 built-in smoke tests
```

Expected: 499/499 pass.

### 2. Validate AI layers

```bash
python scripts/validate_ai_layers.py
```

Expected: All layers VALID (15/15 as of S48).

### 3. Cross-paper consistency check

```bash
python scripts/cross_paper_checker.py
```

Expected: 0 errors. Warnings are informational (orphan claims in self-contained papers, unresolved placeholders for future papers).

### 4. Corpus status report

```bash
python scripts/corpus_status.py
python scripts/corpus_status.py --json    # machine-readable
python scripts/corpus_status.py --check-only  # CI mode
```

### 5. Benchmark anomaly detection

```bash
python -c "
from fracttalix.benchmark import evaluate
for arch in ['point', 'contextual', 'collective', 'drift', 'variance']:
    r = evaluate(arch)
    print(f'{arch:12s} F1={r[\"f1\"]:.2f} AUPRC={r[\"auprc\"]:.2f} VUS-PR={r[\"vus_pr\"]:.2f} lag={r[\"mean_lag\"]}')
"
```

All benchmarks use seed 42 for reproducibility.

---

## Falsification claim test mapping

Each claim in the AI layers carries three v3 fields: `tier`, `test_bindings`, `verified_against`.

**Verification tiers** (v3-S49):

| Tier | Meaning |
|------|---------|
| `axiom` | Type A — foundational, unfalsifiable by design |
| `definition` | Type D — definitional, no predicate |
| `software_tested` | Type F with passing test bindings in this codebase |
| `formal_proof` | Type F verified by step-indexed derivation table (n_invalid=0) |
| `analytic` | Type F verified by analytical argument or adversarial battery |
| `empirical_pending` | Type F with active placeholder or pending external data |

### P1 — The Fractal Rhythm Model

| Claim | Tier | Bindings | File |
|-------|------|----------|------|
| F-1.1 (FRM uniqueness) | `software_tested` | 17 tests | tests/test_benchmark.py |
| F-1.2 (36-orders validation) | `analytic` | — | Requires external data |
| F-1.3 (β=1/2 empirical) | `empirical_pending` | — | Resolved analytically via P2 C-2.4 |
| F-1.4 (β=1/2 analytic) | `formal_proof` | — | 5-step derivation trace in AI layer |
| F-1.5 (λ derivation) | `formal_proof` | — | 4-step derivation trace in AI layer |
| F-1.6 (Circadian prediction) | `analytic` | — | T=4×6=24hr; sources in AI layer |
| F-1.7 (Stuart-Landau) | `analytic` | — | R²>0.99 adversarial battery S43 |

### SFW-1 — Sentinel Software

| Claim | Tier | Bindings | File |
|-------|------|----------|------|
| F-SFW.1 (Three-channel completeness) | `software_tested` | 212 tests | tests/test_steps_*.py |
| F-SFW.2 (Test suite coverage) | `software_tested` | 385 tests | tests/*.py |
| F-SFW.3 (Backward compat v7.x) | `software_tested` | 23 tests | tests/test_backward_compat.py |
| F-SFW.4 (Cascade precursor) | `software_tested` | 6 tests | tests/test_steps_channels.py |
| F-SFW.5 (Config validation) | `software_tested` | 37 tests | tests/test_config.py |
| F-SFW.6 (State persistence) | `software_tested` | 4 tests | tests/test_detector.py |

### P2 — Derivation and Universality

| Claim | Tier | Bindings | Note |
|-------|------|----------|------|
| C-2.1–C-2.5 | `formal_proof` | — | 10-step derivation table, n_invalid=0 |

### P3 — FRM Measurement and Diagnostics

| Claim | Tier | Bindings | Note |
|-------|------|----------|------|
| C-3.REG, C-3.ALT, C-3.DIAG, C-3.σ | `formal_proof` | — | Protocol steps IR-12/13 compliant |

### MK-P1 — Meta-Kaizen

| Claim | Tier | Bindings | Note |
|-------|------|----------|------|
| F-MK1.1–F-MK1.5 (Properties/Theorems) | `formal_proof` | — | Step-indexed derivation traces |
| F-MK1.6–F-MK1.8 (Simulations) | `empirical_pending` | — | Pre-registered protocol in AI layer |
| F-MK1.9 (Empirical H₁) | `empirical_pending` | — | Pending data collection |

### DRP-1 — Dual Reader Standard

| Claim | Tier | Bindings | Note |
|-------|------|----------|------|
| C-DRP.1, C-DRP.2, C-DRP.5 | `formal_proof` | — | Schema validation + MK-P1 live instance |

---

## Verification pipeline

```
Source code → pytest (434 tests)
                ↓
AI layers  → validate_ai_layers.py (schema compliance)
                ↓
Cross-refs → cross_paper_checker.py (derivation source integrity)
                ↓
Corpus     → corpus_status.py (phase readiness, placeholder tracking)
                ↓
CI         → .github/workflows/ (automated on push/PR)
                ↓
DDN        → network-sync.yml (mirror sync, archive snapshot, health check)
```

---

## Distributed Docs Network

The corpus is replicated across multiple independent providers to ensure
persistence beyond any single point of failure. See `network/DDN-theory.md`
for the FRM-grounded theoretical design.

```bash
python -m network.distributed_docs status     # Check network health
python -m network.distributed_docs sync       # Sync all mirrors
python -m network.distributed_docs verify     # Integrity check across nodes
python -m network.distributed_docs snapshot   # Create archival snapshot
python -m network.distributed_docs bootstrap  # First-time network setup
```

Providers: GitHub (primary), GitLab, Codeberg, Bitbucket (git mirrors);
Zenodo, Software Heritage, Internet Archive (archives); IPFS (content-addressed).

Quorum: any 2 of 8 nodes can reconstruct the full repository state.

## Key files

| File | Purpose |
|------|---------|
| `ai-layers/ai-layer-schema.json` | JSON Schema v3-S49 for AI layers |
| `ai-layers/*-ai-layer.json` | Machine-readable claim registries (15 papers) |
| `ai-layers/process-graph.json` | Corpus dependency DAG |
| `scripts/validate_ai_layers.py` | Schema validator |
| `scripts/cross_paper_checker.py` | Cross-paper consistency checker |
| `scripts/corpus_status.py` | Corpus status report generator |
| `network/manifest.json` | DDN topology definition (8 nodes) |
| `network/distributed_docs.py` | DDN management CLI |
| `network/DDN-theory.md` | FRM-grounded network resilience theory |
| `tests/` | 405 pytest tests |
| `benchmark/` | Anomaly archetype benchmarks (seed 42) |

---

## Determinism notes

- All benchmarks use `seed=42` by default
- Sentinel detector is deterministic (no randomness in pipeline)
- AI layer validation is deterministic (JSON parse + schema check)
- Test ordering does not affect results (no shared mutable state)
