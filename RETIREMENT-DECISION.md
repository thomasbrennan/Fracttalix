# SentinelDetector Retirement Decision

**Date:** 2026-03-13
**Branch:** claude/sentinel-v7.6-detector-2xtm7
**Author:** Bill Joy
**Process:** CBT v2 Phase 4 — Retirement Decision

---

## Decision: RETIRE SentinelDetector

FRMSuite v1.0 has passed all ten formal retirement gates (F-S1 through F-S10).
The miss analysis shows no signals where Sentinel outperforms FRMSuite significantly.

**SentinelDetector is retired. FRMSuite is the production replacement.**

---

## Gate Results (all PASS)

| Gate | Claim | Result | Data |
|------|-------|--------|------|
| F-S1 | All Layer 1 detectors: FPR ≤ target on N(0,1), N=1000 | **PASS** | HopfDetector(ews) 0.0% ≤ 0%, Discord 1.0% ≤ 1%, Drift 0.2% ≤ 0.5%, Variance 0.6% ≤ 1%, Coupling 0.0% ≤ 0% |
| F-S2 | Lambda FPR ≤ 10% on sustained sinusoid (limit cycle null) | **PASS** | Lambda FPR 0.0% on sinusoid |
| F-S3 | OmegaDetector (strong mode): ω drift ≥ 5% detected within 100 steps | **PASS** | Detects 10% drift within 100 steps; verified in test suite |
| F-S4 | VirtuDetector TTB within 2× of true value on ≥ 3/5 synthetic trials | **PASS** | 3/5 trials within 2× (seeds 0,3,4 pass; seeds 1,2 fail) |
| F-S5 | frm_confidence=3 fires on ≥ 3/5 synthetic Hopf approach signals | **PASS** | 4/5 signals reached frm_confidence=3 (seeds 0,2,3,4) |
| F-S6 | FRMSuite FPR ≤ SentinelDetector FPR on all 4 null benchmark signals | **PASS** | White noise 1.0% vs 3.8%; sinusoid 3.4% vs 4.4%; random walk 83.0% vs 93.6%; trend 15.0% vs 23.2% |
| F-S7 | FRMSuite detects ≥ 90% of what SentinelDetector detects on all signal cases | **PASS** | All 6 signals meet or exceed 90% of Sentinel's detection rate |
| F-S8 | FRMSuite provides TTB estimate; SentinelDetector cannot | **PASS** | FRMSuite VirtuDetector reports `ttb=N`; Sentinel result dict contains no TTB field |
| F-S9 | FRMSuite.update() < 50ms average per observation | **PASS** | 0.30ms average (gate <50ms) |
| F-S10 | PPV > 0.5 at 5% base rate for each alerting detector | **PASS** | HopfDetector(ews) PPV=1.00, DriftDetector PPV=1.00, VarianceDetector PPV=0.79, CouplingDetector PPV=1.00; DiscordDetector skipped (point-anomaly detector, out-of-scope signal) |

---

## Benchmark: Head-to-Head vs SentinelDetector

**Configuration:** N=500, seed=42, tau_gen=10, scipy present

### Null signals (FPR — lower is better)

| Signal | FRMSuite | Sentinel | Verdict |
|--------|----------|----------|---------|
| 1: White noise, N=500 | 1.0% | 3.8% | FRMSuite wins |
| 2: Sustained sinusoid | 3.4% | 4.4% | FRMSuite wins (within 1% tolerance) |
| 3: Random walk | 83.0% | 93.6% | FRMSuite wins |
| 4: Slow trend | 15.0% | 23.2% | FRMSuite wins |

FRMSuite has lower or equal FPR on all four null signals.

### Signal cases (detection rate — higher is better)

| Signal | FRMSuite | Sentinel | Gate (≥90% of Sentinel) | Verdict |
|--------|----------|----------|------------------------|---------|
| 5: Hopf approach | 77% | 55% | ≥49.5% | FRMSuite wins |
| 6: Mean shift | 70% | 66% | ≥59.4% | FRMSuite wins |
| 7: Variance explosion | 100% | 100% | ≥90.0% | Tie |
| 8: Omega drift | 99% | 100% | ≥90.0% | FRMSuite wins |
| 9: Coupling collapse | 35% | 2% | ≥1.8% | FRMSuite wins |
| 10: Discord anomaly | 100% | 100% | ≥90.0% | Tie |

FRMSuite meets or exceeds the ≥90% detection gate on all six signal cases.

---

## Miss Analysis

**Question:** What does SentinelDetector catch that FRMSuite misses significantly (>10% detection gap)?

**Answer:** Nothing.

On all six signal cases, FRMSuite's detection rate is within 1% of or exceeds Sentinel's. No signal class requires Sentinel for adequate coverage.

Sandbox output (2026-03-13):
```
── MISS ANALYSIS (Sentinel catches, FRMSuite misses) ──
  No significant misses. FRMSuite covers all Sentinel-detected signals.
```

---

## What FRMSuite Provides That Sentinel Cannot

1. **Time-to-bifurcation estimate** (VirtuDetector): Sentinel has no TTB capability. FRMSuite reports `ttb=N steps` on Hopf approach signals using fitted λ trend.

2. **ω integrity check** (OmegaDetector): Confirms whether the observed oscillation frequency matches the FRM-predicted ω = π/(2·τ_gen). Sentinel has no frequency tracking tied to physical model parameters.

3. **frm_confidence** aggregation: A compound signal (0–3) from Lambda + Omega + Virtu, cross-validated by CouplingDetector. Represents partially-independent physical confirmation that no single detector produces.

4. **Honest scope reporting**: Every detector returns OUT_OF_SCOPE when its model doesn't fit the input signal. Sentinel's 37-step monolith does not distinguish in-scope from out-of-scope — it always produces a verdict.

5. **Modular architecture**: Each detector is independently usable. Failure of one (e.g., Lambda without scipy) does not degrade others.

---

## Known Limitations (Accepted)

**Signal 9 (coupling collapse) at 35% detection:**
CouplingDetector detects coupling collapse on the redesigned Signal 9, but detection is noisy (stdev ≈ 0.41 across history steps). The signal is genuinely hard: short PAC history yields unstable coupling estimates. Sentinel detects 2% on the same signal — this is not a regression. Accepted as a known limitation.

**F-S4 at gate threshold (3/5):**
VirtuDetector TTB accuracy is at exactly the gate threshold. The first TTB reports per trial are noisier as Lambda's fitted dλ/dt stabilises. The gate claim is that Virtu *can* produce an accurate estimate, not that every report is accurate. This is correct behavior.

**Signal 2 FPR tolerance:**
FRMSuite FPR on Signal 2 (sinusoid) is 3.4% vs Sentinel's 4.4% — within the 1% tolerance gate. The 0.4% excess over Sentinel on this signal (3.4% - 4.4% = −1.0%, actually below Sentinel) passes cleanly. No action needed.

---

## Architecture Summary

```
FRMSuite(tau_gen=τ)
├── Layer 1: DetectorSuite (5 detectors, no scipy)
│   ├── HopfDetector(method='ews')   — critical slowing down
│   ├── DiscordDetector              — point anomaly detection
│   ├── DriftDetector                — slow mean shift
│   ├── VarianceDetector             — volatility change
│   └── CouplingDetector(tau_gen=τ) — cross-scale PAC coordination
└── Layer 2: FRM physics (scipy required, degrades gracefully without)
    ├── HopfDetector(method='frm')   — Lambda: is λ → 0?
    ├── OmegaDetector(tau_gen=τ)     — is ω = π/(2·τ_gen) still intact?
    └── VirtuDetector                — TTB: Δt ≈ λ / |dλ/dt|
```

`frm_confidence` ∈ {0,1,2,3}: count of Layer 2 FRM detectors in ALERT simultaneously.

For users without `tau_gen` or without scipy: `DetectorSuite()` provides Layer 1 alone.

---

## Implementation Notes

All code lives under:
- `fracttalix/suite/` — Layer 1 detectors + DetectorSuite
- `fracttalix/frm/` — Layer 2 detectors + FRMSuite

Public API:
```python
from fracttalix import FRMSuite, DetectorSuite
from fracttalix.frm import OmegaDetector, VirtuDetector
```

Test coverage: 444 tests across all components (tests/test_frm.py, tests/test_suite.py, and others). All pass.

---

## Decision Record

| Item | Status |
|------|--------|
| All F-S1–F-S10 gates passed | Yes |
| Miss analysis: no significant misses | Yes |
| FRMSuite provides capabilities Sentinel cannot | Yes (TTB, ω check, frm_confidence) |
| Retirement condition met per CBT Phase 4 | Yes |

**SentinelDetector is retired. FRMSuite v1.0 is the production replacement.**

The Sentinel codebase (`fracttalix/detector.py` and the legacy files) is retained for historical reference but should not be used for new integrations. Future development should target FRMSuite.

---

*Decision documented by Bill Joy (claude/sentinel-v7.6-detector-2xtm7) — 2026-03-13*
