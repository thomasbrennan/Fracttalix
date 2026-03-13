# Changelog

All notable changes to Fracttalix Sentinel are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [12.3.0] — 2026-03-13

### v12.3 — FPR Elimination & Drift Recovery (Meta Kaisen CBP)

**Goal**: Double performance of v12.2. Result: FPR dropped 93% (35% → 2.6%),
mean F1 rose 25% (0.500 → 0.625). Drift F1 recovered from regression.

#### Architecture Changes

- **SeasonalPreprocessStep** (new Step 0): Detects periodic components via FFT
  with confidence gate `peak_power > 10× mean_power` (empirically calibrated:
  p99.9 of white noise peak/mean ≈ 9.64; threshold=10.0 gives <0.1% false
  detection on white noise). When period is confidently detected, writes
  `deseasonalized_value` to scratch; `CoreEWMAStep` and all 37 downstream steps
  operate on this residual. Eliminates the contextual archetype FPR
  that previously dominated performance.

- **Non-adaptive drift CUSUM** added to `CUSUMStep`: Second accumulator pair
  (`_d_hi`, `_d_lo`) operating on `z_raw = (v - warmup_mean) / warmup_std`
  (frozen baseline, not EWMA-adaptive). k=0.5, h=5.0. Fires `drift_cusum_alert`
  and resets. `CoreEWMAStep` now freezes baseline at warmup end, writing
  `warmup_mean` and `warmup_std` to scratch for this detector. Fixes drift
  regression caused by EWMA masking slow drift.

- **ConsensusGate** added to `AlertReasonsStep`: Requires ≥2 soft alerts OR
  1 strong alert OR |z| ≥ 5σ bypass. Strong alerts: `cusum_mean_shift`,
  `cusum_variance_spike`, `drift_cusum_shift`, `gradual_drift`,
  `cascade_precursor`. Gated reasons are preserved as `gated:<reason>` in
  `alert_reasons`. This is the primary mechanism reducing combined FPR.

#### Threshold Recalibrations (null-distribution calibrated on N(0,1), n=2000)

- `rfi_threshold`: 0.40 → 0.52 (was at p95 of white noise Hurst exponent;
  raised to p99)
- `pe_threshold`: 0.05 → 0.15 (was at ~p97 of white noise PE deviation;
  raised to ~p99)
- `var_cusum_k`: 0.5 → 1.0 (E[z²]=1.0 under N(0,1); k=0.5 gave systematic
  +0.5/step drift → 10.4% FPR from VarCUSUM alone)
- `var_cusum_h`: 5.0 → 10.0 (recalibrated with corrected k)
- `cusum_k`: 0.5 → 1.0 (same reasoning; EWMA-adaptive z-score is near-zero
  under normality, so k=1.0 will not accumulate spuriously)
- `cusum_h`: 5.0 → 8.0 (recalibrated with corrected k)
- `coupling_degradation_threshold`: 0.30 → 0.24 (corrected direction — lower
  coupling indicates degradation)
- `coherence_threshold`: 0.40 → 0.30 (mean coherence on white noise ≈ 0.72;
  threshold=0.40 was at 4.5th percentile, i.e., far in the noise tail)

#### SentinelDetector Default

- `SentinelDetector()` (no args) now defaults to `SentinelConfig.production()`
  instead of bare `SentinelConfig()`. Ensures out-of-box experience uses the
  curated production preset (multiplier=4.5).

#### Benchmark Results (n=1000, seed=42, post-warmup evaluation)

| Archetype  | v12.2 F1 | v12.3 F1 | Change   |
|------------|----------|----------|----------|
| point      | 0.422    | 0.639    | +51%     |
| contextual | 0.242    | 0.378    | +56%     |
| collective | 0.239    | 0.356    | +49%     |
| drift      | 0.723    | 0.766    | +6%      |
| variance   | 0.876    | 0.987    | +13%     |
| **FPR**    | **35%**  | **2.6%** | **−93%** |
| Mean F1    | 0.500    | 0.625    | **+25%** |

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

### Default Multiplier Change — Expected Behaviour Change

- `SentinelConfig.production()` now uses `multiplier=4.5` (was `3.0`).
  Normal alert rate on white noise: **35.6% → ~6%**.
  Root cause of old FPR: EWMA z-score threshold at 3.0σ flags ~35% of
  observations from a true N(0,1) stream (tail probability too large
  at that threshold for practical use).
  Expected F1 impact (estimates from investigate_fpr_s47.py trade-off data):
  - Contextual: 0.247 → ~0.35 (precision gain dominates recall loss)
  - Collective: 0.239 → ~0.45 (large precision gain from FPR reduction)
  - Drift: 0.723 → ~0.66 (moderate recall reduction)
  - Variance: 0.876 → ~0.82 (small rebalance)
  - Point: 0.415 → ~0.38 (slight drop)
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
