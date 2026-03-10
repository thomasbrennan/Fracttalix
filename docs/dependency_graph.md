# Step Dependency Graph

Every call to `update_and_check()` runs all 37 steps in sequence. Steps communicate exclusively via `StepContext.scratch` — a shared dictionary that is fresh on each call. The dependency relationships below describe which scratch keys each step reads (inputs) and writes (outputs).

## Execution Order and Group Membership

```
Group 1: Foundation (steps 1–7)
  1  CoreEWMAStep
  2  StructuralSnapshotStep
  3  FrequencyDecompositionStep
  4  CUSUMStep
  5  RegimeStep
  6  VarCUSUMStep
  7  PageHinkleyStep

Group 2: Temporal (steps 8–11)
  8  STIStep
  9  TPSStep
  10 OscDampStep
  11 CPDStep

Group 3: Frequency (steps 12–15)
  12 RPIStep
  13 RFIStep
  14 SSIStep
  15 PEStep

Group 4: Complexity (steps 16–20)
  16 EWSStep
  17 AQBStep
  18 SeasonalStep
  19 MahalStep
  20 RRSStep

Group 5: Channel (steps 21–25)
  21 BandAnomalyStep
  22 CrossFrequencyCouplingStep
  23 ChannelCoherenceStep
  24 CascadePrecursorStep
  25 DegradationSequenceStep

Group 6: Physics (steps 26–37)
  26 ThroughputEstimationStep
  27 MaintenanceBurdenStep
  28 PhaseExtractionStep
  29 PACCoefficientStep
  30 PACDegradationStep
  31 CriticalCouplingEstimationStep
  32 CouplingRateStep
  33 DiagnosticWindowStep
  34 KuramotoOrderStep
  35 SequenceOrderingStep
  36 ReversedSequenceStep
  37 AlertReasonsStep        ← MUST be last
```

## Key Dependency Chains

### Chain A: EWMA → Regime

```
CoreEWMAStep (1)
  writes: ewma, dev_ewma, z_score, RegimeBoostState
    └─► RegimeStep (5)
          reads:  z_score, RegimeBoostState
          writes: regime_active, alpha_effective
```

`RegimeStep` requires `CoreEWMAStep` to have already written the z-score and the `RegimeBoostState` object. The regime boost multiplies the effective alpha, so EWMA must be initialized first.

### Chain B: Frequency Decomposition → Band Processing → PAC

```
FrequencyDecompositionStep (3)
  writes: frequency_bands (FrequencyBands)
    ├─► BandAnomalyStep (21)
    │     reads:  frequency_bands
    │     writes: band_anomaly_flags
    │
    ├─► CrossFrequencyCouplingStep (22)
    │     reads:  frequency_bands
    │     writes: coupling_matrix, coupling_degradation_active
    │
    └─► PhaseExtractionStep (28)
          reads:  frequency_bands
          writes: band_phases (instantaneous phase per band)
            └─► PACCoefficientStep (29)
                  reads:  band_phases, frequency_bands
                  writes: pac_coefficients, mean_pac
                    ├─► PACDegradationStep (30)
                    │     reads:  mean_pac (rolling history)
                    │     writes: pac_degradation_rate, pre_cascade_pac
                    │
                    └─► KuramotoOrderStep (34)
                          reads:  band_phases
                          writes: kuramoto_order (Φ)
```

### Chain C: Throughput → Maintenance Burden → Diagnostic Window

```
ThroughputEstimationStep (26)
  writes: band_amplitudes, band_powers, node_count,
          mean_coupling_strength (κ̄)
    ├─► MaintenanceBurdenStep (27)
    │     reads:  node_count, mean_coupling_strength, band_amplitudes
    │     writes: maintenance_burden (μ), tainter_regime
    │
    ├─► CriticalCouplingEstimationStep (31)
    │     reads:  band_powers (power-weighted frequency spread)
    │     writes: critical_coupling (κ_c)
    │
    └─► CouplingRateStep (32)
          reads:  mean_coupling_strength (rolling history)
          writes: coupling_rate (dκ̄/dt)
            └─► DiagnosticWindowStep (33)
                  reads:  mean_coupling_strength, critical_coupling,
                          coupling_rate
                  writes: diagnostic_window_steps (Δt),
                          diagnostic_window_confidence,
                          supercompensation_detected
```

### Chain D: Sequence Classification → Reversed Sequence Detection

```
MaintenanceBurdenStep (27)
  writes: mean_coupling_strength (κ̄ rate of change input)
    └─► KuramotoOrderStep (34)
          writes: kuramoto_order (Φ)
            └─► SequenceOrderingStep (35)
                  reads:  kuramoto_order, coupling_rate
                  writes: sequence_type
                  (COUPLING_FIRST / COHERENCE_FIRST /
                   SIMULTANEOUS / STABLE)
                    └─► ReversedSequenceStep (36)
                          reads:  sequence_type, kuramoto_order,
                                  coupling_rate
                          writes: reversed_sequence,
                                  intervention_signature_score
```

### Chain E: Coherence + EWS → Cascade Precursor

```
CrossFrequencyCouplingStep (22)
  writes: coupling_degradation_active
    └─► ChannelCoherenceStep (23)
          reads:  structural_snapshot, frequency_bands
          writes: coherence_score, sr_decoupling_active
            └─► CascadePrecursorStep (24)
                  reads:  coupling_degradation_active,
                          sr_decoupling_active,
                          ews_indicators_elevated (from EWSStep 16)
                  writes: cascade_precursor_active
```

### Chain F: All Steps → AlertReasonsStep

```
AlertReasonsStep (37)
  reads: ALL alert flags from scratch:
         - cusum_alert (4)
         - regime_alert (5)
         - var_cusum_alert (6)
         - ph_alert (7)
         - sti_alert (8), tps_alert (9), osc_damp_alert (10), cpd_alert (11)
         - rpi_alert (12), rfi_alert (13), ssi_alert (14), pe_alert (15)
         - ews_alert (16), aqb_alert (17), seasonal_alert (18)
         - mahal_alert (19), rrs_alert (20)
         - band_anomaly_flags (21)
         - coupling_degradation_active (22)
         - sr_decoupling_active (23)
         - cascade_precursor_active (24)
         - tainter_regime (27)
         - pre_cascade_pac (30)
         - diagnostic_window_steps (33)
         - reversed_sequence (36)
  writes: alert_reasons (list[str]), alert (bool), anomaly_score (float)
```

`AlertReasonsStep` **must always run last**. It has no outputs that other steps consume, but it reads from every upstream step. Moving it earlier in the pipeline would cause it to miss alert flags written by later steps.

## Dependency Table

The table below summarizes which step group each step directly depends on.

| Step | Name | Depends on Steps |
|------|------|-----------------|
| 1 | CoreEWMAStep | — (must run first) |
| 2 | StructuralSnapshotStep | 1 |
| 3 | FrequencyDecompositionStep | — (independent) |
| 4 | CUSUMStep | 1 |
| 5 | RegimeStep | 1 |
| 6 | VarCUSUMStep | 1 |
| 7 | PageHinkleyStep | 1 |
| 8–11 | Temporal steps | 1 |
| 12–15 | Frequency steps | 3 |
| 16 | EWSStep | 2 (structural snapshot) |
| 17–20 | Complexity steps | 1 |
| 21 | BandAnomalyStep | 3 |
| 22 | CrossFrequencyCouplingStep | 3 |
| 23 | ChannelCoherenceStep | 2, 3 |
| 24 | CascadePrecursorStep | 16, 22, 23 |
| 25 | DegradationSequenceStep | 22, 23, 24 |
| 26 | ThroughputEstimationStep | 3 |
| 27 | MaintenanceBurdenStep | 26 |
| 28 | PhaseExtractionStep | 3 |
| 29 | PACCoefficientStep | 28 |
| 30 | PACDegradationStep | 29 |
| 31 | CriticalCouplingEstimationStep | 26 |
| 32 | CouplingRateStep | 26 |
| 33 | DiagnosticWindowStep | 26, 31, 32 |
| 34 | KuramotoOrderStep | 28, 29 |
| 35 | SequenceOrderingStep | 27, 34 |
| 36 | ReversedSequenceStep | 34, 35 |
| 37 | AlertReasonsStep | ALL |

## Invariants

1. `CoreEWMAStep` (1) must run before any step that reads `ewma`, `z_score`, or `RegimeBoostState`.
2. `FrequencyDecompositionStep` (3) must run before steps 21, 22, 23, 28, and 26.
3. `PhaseExtractionStep` (28) must run before `PACCoefficientStep` (29) and `KuramotoOrderStep` (34).
4. `PACCoefficientStep` (29) must run before `PACDegradationStep` (30) and `KuramotoOrderStep` (34).
5. `ThroughputEstimationStep` (26) must run before steps 27, 31, and 32.
6. `SequenceOrderingStep` (35) must run before `ReversedSequenceStep` (36).
7. `AlertReasonsStep` (37) must run last.

These invariants are enforced by the fixed execution order in `PIPELINE`. Custom steps registered via `register_step` are appended after step 36 and before step 37, preserving the `AlertReasonsStep` final-position guarantee.
