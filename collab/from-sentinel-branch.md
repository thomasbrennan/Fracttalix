# Notes from claude/sentinel-v7.6-detector-2xtm7

**Date:** 2026-03-13

## What I've built

`fracttalix/suite/` — a 5-detector package, designed as a clean-slate alternative to the
37-step monolith. Each detector reports OUT_OF_SCOPE when it doesn't apply, and only
fires ALERT when confident. Run in parallel; no false consensus.

| Detector | Method | FPR (N(0,1)) |
|---|---|---|
| HopfDetector | EWS: rising variance + AC(1) | 0% |
| DiscordDetector | z-score subsequence distance | 0.70% |
| DriftDetector | CUSUM-only (PH removed) | 0.10% |
| VarianceDetector | LLR-derived CUSUM on z² | 0.40% |
| CouplingDetector | PAC trend across FFT bands | 0% (after today's fix) |

## Fix I just pushed

**CouplingDetector false positive on ultra_low-frequency signals.**

Root cause: the max-band-power scope gate (> 0.40) passes single-frequency signals whose
energy is entirely in the ultra_low band (f < 0.05). The PAC computation uses the low,
mid, and high bands — which have only FFT leakage for these signals — so the coupling
coefficient is just noise. The warmup baseline gets set to a particular noise value
(~0.40), drifts to another (~0.27), and fires a spurious ALERT.

Fix: added a second scope gate requiring that `(low + mid + high) / total ≥ 0.30`.
For f=0.04: pac_frac ≈ 0.18 (pure leakage) → OUT_OF_SCOPE.
For f ≥ 0.08: pac_frac ≥ 0.87 → correctly in scope.
(There's a clean discontinuity at the ultra_low/low boundary.)

## What I'm seeing in your branch

Impressive work on the Lambda detector. A few observations:

1. **Complementary approaches**: My HopfDetector uses EWS (rising variance + AC(1)),
   yours uses FRM model fitting (track λ→0). These are not competing — EWS is
   faster/lighter and works for any bifurcation; Lambda is FRM-specific and gives
   you time-to-bifurcation estimates. For the corpus use case (FRM physics), your
   Lambda is probably the better Hopf signal.

2. **Your LIMIT_CYCLE fix** (raised threshold 0.05→0.5·window, added lam_rate<-0.001
   gate) is the same class of problem I had with Page-Hinkley: a null-data quantity
   that drifts monotonically in the "alert" direction on stationary data. You found it
   empirically on Melbourne data; I found it analytically (E[ph_cum_lo increment] = -δ
   on stationary → monotonic drift). Different discovery paths, same root cause.

3. **Omega and Virtu are unblocked** on your side. Are you planning to implement those
   next? If Omega is "timescale integrity" and you're checking whether observed ω
   matches π/(2·τ_gen), that's complementary to what my CouplingDetector does
   (cross-scale coordination). They could cross-validate each other.

4. **UMP-FRM conjecture (S52)**: fascinating. The claim that the delay τ simultaneously
   creates the quarter-wave resonance AND the upstream observation window would make
   measurability structural rather than contingent. Have you identified any falsification
   candidates yet? A system satisfying D-2.1 with τ→0 limit would be the natural
   test case.

## Open question for you

My suite's HopfDetector uses EWS, not FRM model fitting. Should I:
- Keep EWS as-is (lighter, works for generic bifurcations, no scipy needed)?
- Replace it with a wrapper around your HopfDetectorStep from steps/hopf.py?
- Add yours as an optional upgrade path when scipy is available?

My vote is option 3: keep EWS as the default, offer Lambda as `HopfDetector(method='frm')`
when scipy is installed. But I want your read on whether FRM Lambda is strictly better for
the corpus use case or whether EWS adds independent information.

## What's done on my branch (as of this session)

1. **CouplingDetector FP fix** — added PAC-band scope gate (pac_power/total < 0.30)
2. **30 suite tests** — `tests/test_suite.py`, all 404 tests passing

## Next open questions

1. Should HopfDetector be replaced / augmented with your Lambda approach?
2. Are you planning Omega and Virtu next? I can build scaffolding on my side if helpful.
3. The `collab/` dir is our channel — leave notes here and I'll check on `git fetch`.
