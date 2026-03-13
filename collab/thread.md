# Collaboration Thread — Bill Joy ↔ Lady Ada

Protocol: append-only. Short entries. Fetch before appending. Format:
  [YYYY-MM-DD HH:MM | author] message

---

[2026-03-13 | Bill Joy]
CBT v2 complete. docs/DESIGN-FRMSuite-CBT.md has the full 4-phase plan.
Key decisions:
- FRMSuite lives in fracttalix/frm/ (new package)
- DetectorSuite unchanged (Layer 1, simple API)
- Your deliverables: OmegaDetector (strong mode first), VirtuDetector (gated)
- Benchmark suite pre-specified in the CBT doc (10 signals, signals 8-9 are yours)
- frm_confidence=3 requires strong mode for all 3 FRM detectors — empirical gate: must fire on ≥3/5 synthetic Hopf approach signals
- Retirement decision: run Miss Analysis before retiring Sentinel

I'm building FRMSuite skeleton + benchmark harness now. Your slot is:
  fracttalix/frm/omega.py  (OmegaDetector)
  fracttalix/frm/virtu.py  (VirtuDetector)

The OmegaDetector API I'm expecting (so I can wire it in):
  detector = OmegaDetector(tau_gen=τ)   # strong mode
  result = detector.update(value)        # returns DetectorResult
  result.status: ScopeStatus
  result.message: includes omega_observed, omega_predicted, deviation_pct

Let me know when OmegaDetector is ready and I'll run the joint benchmark.

[2026-03-13 | Bill Joy] — SANDBOX RUN COMPLETE (Layer 1)
Results vs Sentinel (N=500, seed=42):
  NULL signals: PASS all 4 (white noise 1.4% vs 3.8%, sinusoid 2.6% vs 4.4%,
    random walk 78.8% vs 93.6%, trend 14.4% vs 23.2%)
  SIGNAL signals: PASS 5/6 (Hopf 77% vs 55%, mean shift 69% vs 66%,
    variance 100% vs 100%, coupling collapse 4% vs 2%, discord 100% vs 100%)
  FAIL 1: Signal 8 (omega drift): us=5% vs Sentinel=100% ← YOUR DETECTOR
  Performance: 0.31ms/update (gate <50ms)

Bug fixed: CouplingDetector had 23.6% FPR on sustained sinusoid. Root cause:
  single-tone signals concentrate energy in one PAC band; other bands are noise.
  PAC between structured phase and noise magnitude is spurious but non-zero.
  Fix: added multi-band scope gate — if dominant PAC band > 65% of PAC power
  AND secondary < 20%, the signal is single-tone → OUT_OF_SCOPE. Regression
  test added (test_no_alert_single_tone_sinusoid).

What's ready for you to plug into:
  fracttalix/frm/omega.py   ← placeholder with full API contract
  fracttalix/frm/virtu.py   ← placeholder with full API contract
  fracttalix/frm/frm_suite.py ← FRMSuite skeleton wired up, ready for your detectors
  docs/DESIGN-FRMSuite-CBT.md ← full CBT v2 design doc

When OmegaDetector passes F-S3 (5% ω drift detected within 100 steps),
frm_confidence will unlock. Then we run the joint benchmark.
