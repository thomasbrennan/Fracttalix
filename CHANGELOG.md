# Changelog

All notable changes to Fracttalix Sentinel are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [12.0.0] — 2026-03-09

### Architecture (Meta-Kaizen V12)

- **Package restructure**: Single-file monolith split into proper Python package
  (`fracttalix/`) with six step submodules, installable via `pip install fracttalix`
- **`pyproject.toml`**: PEP 517/518 compliant; single source of truth for version
- **Version via `importlib.metadata`**: No more dual-source version drift
- **`benchmark/` subpackage**: Promoted from embedded class to top-level subpackage
  with `archetypes.py`, `metrics.py`, `comparison.py`
- **`extras/server.py`**: REST server demoted to optional extras; not in `__all__`
- **`legacy/`**: All v7.x–v11.0 archives moved out of repo root
- **`tests/conftest.py`**: Shared pytest fixtures; test files mirror module structure
- **`docs/`**: MkDocs + mkdocstrings API documentation
- **`examples/`**: Four runnable examples covering key use cases
- **GitHub Actions CI**: Python 3.9–3.12 matrix; zero-dep + fast + full dep runs
- **Release workflow**: Validates CHANGELOG entry before publishing to PyPI

### Hostile-Review Corrections

- **Comparison benchmark**: Added PyOD ECOD and River HalfSpaceTrees baselines
  (`benchmark/comparison.py`) — no longer benchmarks only against naive 3σ
- **Ablation study**: `SentinelBenchmark.ablation_study()` quantifies F1/VUS-PR
  contribution of each step group — answers "are 37 steps justified?"
- **Authorship**: AI tools (Claude, Grok) acknowledged in paper but not listed as authors
- **numpy fallback warning**: Emits `ImportWarning` when numpy absent
- **`auto_tune()` objective**: Added `objective: Literal["f1","auprc"]` parameter
- **`__all__` contract**: Explicit; internal symbols no longer leak

### Breaking Changes

- v10 backward-compat aliases removed (one full release cycle elapsed):
  `kuramoto_order_v10`, `maintenance_burden_v10`, `critical_coupling_v10`
- `SentinelServer` no longer imported from `fracttalix` top-level;
  use `from fracttalix.extras.server import SentinelServer`
- CLI entry point is now `fracttalix` command (via `fracttalix.__main__`)

### New Config Parameters

- `warn_on_numpy_fallback: bool = True` — control degradation warning

---

## [11.0.0] — 2026-03-08

### Meta-Kaizen Corrective Release

**Phase 0 — Architecture:**
- `_STEP_REGISTRY` removed (dead code); pipeline is now fully explicit
- Test file extracted to `tests/test_sentinel_v1100.py`

**Phase 1 — Foundation:**
- `_core_step_ref` side channel eliminated; `RegimeBoostState` shared object introduced
- `state_dict()` / `load_state()` implemented for all 8 previously stateless v10.0 steps

**Phase 2 — Bug fixes:**
- `CUSUMStep` now wired to `config.cusum_k` / `config.cusum_h` (new configurable fields)
- `OscDampStep` amplitude computed from `osc_damp_window` slice, not full bank
- `PhaseExtractionStep` FFT window capped to `rpi_window` (performance)
- Steps renumbered 1–37 cleanly

**Phase 3 — Physics corrections (breaking):**
- `KuramotoOrderStep`: true Φ over per-sample phase vectors across all bands
- `MaintenanceBurdenStep`: μ = 1 − κ̄ (window-size independent)
- `CriticalCouplingEstimationStep`: unified normalized frequency units (0.0–1.0)
- New: `phi_kappa_separation` metric (Φ − κ̄ gap) in `SentinelResult`

**Phase 4 — Refinements:**
- `PACDegradationStep`: linear regression slope over full history window
- `SequenceOrderingStep`: threshold normalized by rolling std
- `ReversedSequenceStep`: dynamic AMBIGUOUS score from count-ratio uncertainty
- `AlertReasonsStep`: per-step configurable cooldown (`alert_cooldown_steps`)

**Phase 5 — New capabilities:**
- `DiagnosticWindowStep`: pessimistic/expected/optimistic Δt triple
- `MultiStreamSentinel.cross_stream_correlations()`: pairwise z-score Pearson

**Breaking output changes (v10 aliases provided for one cycle):**
- `kuramoto_order`, `maintenance_burden`, `critical_coupling` corrected
- Old values preserved as `*_v10` aliases (removed in v12.0)

---

## [10.0.0] — 2026-03-07

### Physics-Derived Collapse Dynamics

- Maintenance burden μ = 1 − κ̄ (Tainter regime classification)
- Phase-Amplitude Coupling pre-cascade warning
- Diagnostic window Δt = (κ̄ − κ_c) / |dκ̄/dt|
- Reversed sequence detection (intervention signature)
- Kuramoto order parameter Φ
- 98 unit tests

---

## [9.0.0] — 2026-03-07

### Three-Channel Dissipative Network Model

- Channel 1 (Structural): distributional moments + stationarity
- Channel 2 (Rhythmic): FFT carrier waves + phase-amplitude coupling
- Channel 3 (Temporal): degradation sequence ordering
- `CASCADE_PRECURSOR` alert type (CRITICAL severity)
- 65 unit tests

---

## [8.0.0] — 2026-03-06

### Foundation Rewrite

- `SentinelConfig` frozen dataclass with `slots=True`
- `WindowBank` (independent named deques — fixes β window collision bug)
- 19-step explicit pipeline
- Soft regime boost (replaces hard alpha override)

---

## [7.11.1] — 2026-03-06

### Hotfixes

- EWS window starvation fix
- Unpicklable lambda replaced with named function
- Hard-reset on `reset()` corrected

---

## [7.6.0] — 2026-03-04

### Initial Stable Release

- EWMA baseline + CUSUM + Page-Hinkley + permutation entropy
- Multi-stream support
- Auto-tune via F1 optimization
- REST server (asyncio)
- CC0-1.0 license
