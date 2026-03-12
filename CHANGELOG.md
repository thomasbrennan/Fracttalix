# Changelog

All notable changes to Fracttalix Sentinel are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [12.2.0] — 2026-03-12

### Epistemic Language Corrections

- **README physics framing removed**: Replaced "physics-derived" and
  "derived from the Kuramoto synchronization framework" throughout README
  with accurate "signal-processing heuristic" language. The capabilities are
  real and useful; the framing implied physical derivation that the code
  explicitly disclaims (see `MaintenanceBurdenStep` docstring: "NOT derived
  from physics...must not be cited as such in publications").
- **Corrected maintenance burden formula**: README line 103 previously showed
  `μ = N·κ̄·E_coupling/P_throughput` (the abandoned v10.0 formula). Corrected
  to the actual v11.0 implementation `μ = 1−κ̄`.
- **Reframed reversed sequence detection**: "Thermodynamic arrow" and
  "civilization being collapsed" language replaced with "heuristic ordering
  hypothesis" and "signal classification label, not a causal claim".
- **Epistemic status block** added to Collapse Indicator Capabilities section.

### Default Multiplier Change — Marginal Behaviour Change

- `SentinelConfig.production()` now uses `multiplier=4.5` (was `3.0`).
  Measured normal alert rate on white noise (seed=99, n=1000, post-warmup):
  **36.7% → 35.4%** — a 1.3 pp marginal change only.
  The FPR floor at ~35% is not dominated by the EWMA threshold; it is driven by
  other pipeline channels (coherence, coupling, physics steps). Raising the
  multiplier above ~3.5 has negligible further effect (see
  `benchmark/investigate_fpr_s47.py` channel attribution for root cause analysis).
  Measured F1 at new default (n=1000, seed=42) vs v12.1 (multiplier=3.0):
  - Point:      0.415 → 0.422  (+0.007)
  - Contextual: 0.247 → 0.242  (−0.005)
  - Collective: 0.239 → 0.242  (+0.003)
  - Drift:      0.723 → 0.727  (+0.004)
  - Variance:   0.876 → 0.883  (+0.007)
  All changes within measurement noise. This release is primarily an epistemic
  language correction; the performance improvements from the multiplier change are
  not material.
  Users who need the v12.1 behaviour: `SentinelConfig(multiplier=3.0)`.

### Documentation

- `fast()` docstring now quantifies expected FPR (~60–80% on white noise).
- Preset table in README now includes FPR column and multiplier–FPR trade-off note.
- `SentinelConfig` group F comment corrected from "Fluid dynamics" to
  "Temporal / oscillatory dynamics (signal-processing parameters)".
- `SFW1-ai-layer.json` version metadata synced to 12.2.0.

---

## [12.1.0] — 2026-03-10

### Bug Fixes

- **VarCUSUM non-reset defect** (`VarCUSUMStep`, `fracttalix/steps/foundation.py`):
  CUSUM accumulators `s_hi` and `s_lo` were never reset after crossing threshold `h`,
  causing a permanent alert state on all post-warmup steps. Fixed by re-arming
  (resetting to 0) after each threshold crossing. Normal alert rate reduced from
  97% → 35.6% on the standard benchmark suite.
  Added companion sustained-variance detector using the warmup-estimated baseline
  (4× ratio threshold) to maintain recall on prolonged volatility regimes without
  re-triggering the permanent-crossing problem.

- **ChannelCoherence unit mismatch** (`ChannelCoherenceStep`, `fracttalix/steps/channels.py`):
  `coherence_score` previously compared `structural_change_rate` (second-order,
  ~0.02) against `rhythmic_change_rate` (first-order, ~0.17) — incompatible units,
  producing a score near 0 on every normal step. Replaced rate-difference formula
  with Pearson correlation between the two change series (scale-invariant). Normal
  data now scores ~0.5 (above threshold); genuine decoupling produces correlated
  divergence.

### Benchmark Results (n=1000, seed=42)

| Metric            | v12.0 | v12.1 |
|-------------------|-------|-------|
| Normal alert rate | 97%   | 35.6% |
| Point F1          | 0.360 | 0.415 |
| Contextual F1     | 0.200 | 0.247 |
| Collective F1     | 0.110 | 0.239 |
| Drift F1          | 0.670 | 0.723 |
| Variance F1       | 0.690 | 0.876 |

374/374 tests passing.

### Investigation

`benchmark/investigate_fpr_s47.py` — S47 false positive rate investigation script.
Reproduces channel attribution, precision-recall curve, normal data characterisation,
and v12 baseline simulation. See report in session S47.

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
