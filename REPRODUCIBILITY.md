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
pytest tests/ -v                # 405 tests
python fracttalix/detector.py   # 65 built-in smoke tests
```

Expected: 470/470 pass.

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

Each Type F claim in the AI layers maps to specific executable tests:

### P1 — The Fractal Rhythm Model

| Claim | Test coverage | File |
|-------|--------------|------|
| F-1.1 (FRM uniqueness) | Fit quality tests across archetypes | tests/test_benchmark.py |
| F-1.2 (36-orders validation) | Validation set protocol (requires external data) | — |
| F-1.3 (β=1/2 empirical) | PLACEHOLDER — pending P2 C-2.1 | — |
| F-1.4 (β=1/2 analytic) | Derivation trace in AI layer; numerical confirmation in S43 | — |
| F-1.5 (λ derivation) | Derivation trace in AI layer; numerical confirmation in S43 | — |
| F-1.6 (Circadian prediction) | Analytic: T=4×6=24hr; sources cited in AI layer | — |
| F-1.7 (Stuart-Landau) | R²>0.99 confirmed in S43 adversarial battery | — |

### SFW-1 — Sentinel Software

| Claim | Test coverage | File |
|-------|--------------|------|
| F-SFW.1 (Three-channel completeness) | Channel integration tests | tests/test_steps_channels.py |
| F-SFW.2 (Test suite coverage) | 405 tests covering all 37 steps | tests/*.py |
| F-SFW.3 (Backward compat v7.x) | Legacy kwargs, aliases | tests/test_backward_compat.py |
| F-SFW.4 (Cascade precursor) | Conjunction requirement test | tests/test_steps_channels.py |
| F-SFW.5 (Config validation) | Frozen dataclass, presets, bounds | tests/test_config.py |
| F-SFW.6 (State persistence) | save_state/load_state round-trip | tests/test_detector.py |

### MK-P1 — Meta-Kaizen

| Claim | Test coverage | File |
|-------|--------------|------|
| F-MK1.1–F-MK1.3 (Properties) | Formal proof in AI layer derivation traces | — |
| F-MK1.4–F-MK1.5 (Theorems) | Formal proof in AI layer derivation traces | — |
| F-MK1.6–F-MK1.8 (Simulations) | Pre-registered protocol in AI layer | — |
| F-MK1.9 (Empirical H₁) | PLACEHOLDER — pending data collection | — |

---

## Verification pipeline

```
Source code → pytest (405 tests)
                ↓
AI layers  → validate_ai_layers.py (schema compliance)
                ↓
Cross-refs → cross_paper_checker.py (derivation source integrity)
                ↓
Corpus     → corpus_status.py (phase readiness, placeholder tracking)
                ↓
CI         → .github/workflows/ (automated on push/PR)
```

---

## Key files

| File | Purpose |
|------|---------|
| `ai-layers/ai-layer-schema.json` | JSON Schema v2-S42 for AI layers |
| `ai-layers/*-ai-layer.json` | Machine-readable claim registries (15 papers) |
| `ai-layers/process-graph.json` | Corpus dependency DAG |
| `scripts/validate_ai_layers.py` | Schema validator |
| `scripts/cross_paper_checker.py` | Cross-paper consistency checker |
| `scripts/corpus_status.py` | Corpus status report generator |
| `tests/` | 405 pytest tests |
| `benchmark/` | Anomaly archetype benchmarks (seed 42) |

---

## Determinism notes

- All benchmarks use `seed=42` by default
- Sentinel detector is deterministic (no randomness in pipeline)
- AI layer validation is deterministic (JSON parse + schema check)
- Test ordering does not affect results (no shared mutable state)
