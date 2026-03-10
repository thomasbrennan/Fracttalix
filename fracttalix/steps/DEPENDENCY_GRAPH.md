# Step Dependency Graph

Inter-step dependencies for the 37-step Fracttalix Sentinel V12 pipeline.
Steps communicate exclusively through `ctx.scratch` (per-observation dict) and
`ctx.bank` (rolling window storage).  One shared mutable object (`RegimeBoostState`)
is wired at construction time between Steps 1 and 5.

---

## Shared State Objects (construction-time wiring)

| Object | Shared Between | Direction |
|--------|---------------|-----------|
| `RegimeBoostState` | `CoreEWMAStep` (Step 1) and `RegimeStep` (Step 5) | Step 5 **writes** `boost.boost`; Step 1 **reads** it on the *next* observation |
| `RegimeStep` instance | `RRSStep` (Step 20) | `RRSStep` receives a reference to `RegimeStep` at construction to query regime state |

---

## Per-Step Scratch Key Dependencies

Each row lists the scratch keys a step **reads** (inputs) and **writes** (outputs).

### Foundation (Steps 1–7)

| Step | Class | Reads from scratch | Writes to scratch |
|------|-------|--------------------|-------------------|
| 1 | `CoreEWMAStep` | *(none — reads from `RegimeBoostState`)* | `ewma`, `dev_ewma`, `z_score`, `anomaly_score`, `anomaly`, `alert`, `warmup` |
| 2 | `StructuralSnapshotStep` | *(none — reads `ctx.bank`)* | `structural_snapshot`, `_structural_snapshot_history` |
| 3 | `FrequencyDecompositionStep` | *(none — reads `ctx.bank`)* | `frequency_bands`, `_bands_history` |
| 4 | `CUSUMStep` | `z_score` | `cusum_pos`, `cusum_neg`, `cusum_alert` |
| 5 | `RegimeStep` | `z_score` | `regime_change` (also writes `RegimeBoostState.boost`) |
| 6 | `VarCUSUMStep` | `z_score` | `var_cusum_pos`, `var_cusum_neg`, `var_cusum_alert` |
| 7 | `PageHinkleyStep` | `z_score` | `ph_cumsum`, `ph_alert` |

### Temporal (Steps 8–11)

| Step | Class | Reads from scratch | Writes to scratch |
|------|-------|--------------------|-------------------|
| 8 | `STIStep` | *(none — reads `ctx.bank`)* | `sti` |
| 9 | `TPSStep` | *(none — reads `ctx.bank`)* | `tps` |
| 10 | `OscDampStep` | *(none — reads `ctx.bank`)* | `osc_damp`, `osc_alert` |
| 11 | `CPDStep` | *(none — reads `ctx.bank`)* | `cpd_score` |

### Frequency (Steps 12–15)

| Step | Class | Reads from scratch | Writes to scratch |
|------|-------|--------------------|-------------------|
| 12 | `RPIStep` | *(none — reads `ctx.bank`)* | `rpi` |
| 13 | `RFIStep` | *(none — reads `ctx.bank`)* | `rfi` |
| 14 | `SSIStep` | *(none — reads `ctx.bank`)* | `ssi` |
| 15 | `PEStep` | *(none — reads `ctx.bank`)* | `pe`, `pe_baseline` |

### Complexity (Steps 16–20)

| Step | Class | Reads from scratch | Writes to scratch |
|------|-------|--------------------|-------------------|
| 16 | `EWSStep` | *(none — reads own `ews_w` bank window)* | `ews_score`, `ews_regime` |
| 17 | `AQBStep` | *(none — reads `ctx.bank`)* | `aqb_alert`, `aqb_lo`, `aqb_hi` |
| 18 | `SeasonalStep` | *(none — reads `ctx.bank`)* | `seasonal_residual`, `seasonal_alert` |
| 19 | `MahalStep` | *(none — reads `ctx.bank`)* | `mahal_dist` |
| 20 | `RRSStep` | *(reads `RegimeStep` state object directly)* | `rrs_score` |

### Channels (Steps 21–25)

| Step | Class | Reads from scratch | Writes to scratch |
|------|-------|--------------------|-------------------|
| 21 | `BandAnomalyStep` | `frequency_bands` | `band_anomalies`, `v9_active_alerts` |
| 22 | `CrossFrequencyCouplingStep` | `_bands_history` | `coupling_matrix`, `coupling_degradation_active`, `v9_active_alerts` |
| 23 | `ChannelCoherenceStep` | `ewma`, `_structural_snapshot_history` | `channel_coherence`, `sr_decoupling_active`, `v9_active_alerts` |
| 24 | `CascadePrecursorStep` | `coupling_degradation_active`, `sr_decoupling_active`, `ews_score`, `anomaly_score`, `rfi`, `v9_active_alerts` | `cascade_precursor_active`, `v9_active_alerts` |
| 25 | `DegradationSequenceStep` | `v9_active_alerts` | `degradation_sequence` |

### Physics (Steps 26–37)

| Step | Class | Reads from scratch | Writes to scratch |
|------|-------|--------------------|-------------------|
| 26 | `ThroughputEstimationStep` | `frequency_bands`, `_bands_history`, `coupling_matrix` | `band_amplitudes`, `band_powers`, `throughput`, `node_count`, `mean_coupling_strength` |
| 27 | `MaintenanceBurdenStep` | `mean_coupling_strength`, `node_count`, `throughput` | `maintenance_burden`, `maintenance_burden_v10`, `tainter_regime` |
| 28 | `PhaseExtractionStep` | *(reads `ctx.bank` scalar window)* | `band_filtered_signals`, `band_phases` |
| 29 | `PACCoefficientStep` | `band_phases`, `band_amplitudes` | `pac_matrix`, `mean_pac` |
| 30 | `PACDegradationStep` | `mean_pac`, `cascade_precursor_active` | `pac_history`, `pac_degradation_rate`, `pre_cascade_pac` |
| 31 | `CriticalCouplingEstimationStep` | `band_powers` | `critical_coupling`, `critical_coupling_v10` |
| 32 | `CouplingRateStep` | `mean_coupling_strength` | `coupling_history`, `coupling_rate` |
| 33 | `DiagnosticWindowStep` | `mean_coupling_strength`, `critical_coupling`, `coupling_rate`, `coupling_history` | `diagnostic_window_steps`, `diagnostic_window_steps_pessimistic`, `diagnostic_window_steps_optimistic`, `diagnostic_window_confidence`, `supercompensation_detected` |
| 34 | `KuramotoOrderStep` | `band_phases` | `kuramoto_order`, `kuramoto_order_v10` |
| 35 | `SequenceOrderingStep` | `kuramoto_order`, `coupling_rate` | `phi_history`, `phi_rate`, `coupling_degrading`, `coherence_degrading`, `sequence_history` |
| 36 | `ReversedSequenceStep` | `sequence_history`, `phi_rate`, `coupling_rate` | `reversed_sequence`, `intervention_signature_score`, `sequence_type` |
| 37 | `AlertReasonsStep` | `cusum_alert`, `var_cusum_alert`, `regime_change`, `ph_alert`, `osc_alert`, `cpd_score`, `z_score`, `rfi`, `ews_regime`, `pe`, `pe_baseline`, `mahal_dist`, `v9_active_alerts`, `structural_snapshot`, `frequency_bands`, `coupling_matrix`, `channel_coherence`, `alert` | `alert_reasons`, `channel_summary` |

---

## Key Dependency Chains

```
Step 1 (CoreEWMA)
  → z_score → Steps 4, 5, 6, 7  (CUSUM, Regime, VarCUSUM, PageHinkley)
  → ewma    → Step 23            (ChannelCoherence)

Step 3 (FrequencyDecomposition)
  → frequency_bands   → Steps 21, 26         (BandAnomaly, Throughput)
  → _bands_history    → Steps 22, 26         (CrossFreqCoupling, Throughput)

Step 16 (EWS)
  → ews_score  → Step 24                     (CascadePrecursor)

Steps 21+22+23 (Channel alerts)
  → v9_active_alerts  → Steps 24, 25, 37     (Cascade, Sequence, AlertReasons)
  → coupling_degradation_active → Step 24, 30
  → sr_decoupling_active        → Step 24

Step 26 (Throughput)
  → mean_coupling_strength → Steps 27, 32, 33, 35
  → band_powers            → Step 31
  → band_amplitudes        → Step 29
  → node_count, throughput → Step 27

Step 28 (PhaseExtraction)
  → band_phases → Steps 29, 34              (PAC, Kuramoto)

Step 29 (PACCoefficient)
  → mean_pac → Step 30                      (PACDegradation)

Step 31 (CriticalCoupling)
  → critical_coupling → Step 33             (DiagnosticWindow)

Step 32 (CouplingRate)
  → coupling_rate, coupling_history → Steps 33, 35, 36

Step 34 (Kuramoto)
  → kuramoto_order → Step 35               (SequenceOrdering)

Step 35 (SequenceOrdering)
  → sequence_history, phi_rate → Step 36   (ReversedSequence)
```

---

## Why AlertReasonsStep (Step 37) Must Be Last

`AlertReasonsStep` is a **read-only aggregator**: it reads alert flags and
metric values from every preceding step and assembles them into the final
`alert_reasons` list and `channel_summary` string.

Specifically it reads:

- `cusum_alert` (Step 4), `var_cusum_alert` (Step 6) — mean/variance shift flags
- `regime_change` (Step 5) — regime transition flag
- `ph_alert` (Step 7) — slow drift flag
- `osc_alert` (Step 10) — oscillation damping flag
- `cpd_score` (Step 11) — change-point detection score
- `z_score` (Step 1) — EWMA z-score for threshold comparison
- `rfi` (Step 13) — fractal irregularity index
- `ews_regime` (Step 16) — EWS criticality regime
- `pe`, `pe_baseline` (Step 15) — permutation entropy comparison
- `mahal_dist` (Step 19) — Mahalanobis distance
- `v9_active_alerts` (Steps 21–24) — structured V9 alert objects
- `structural_snapshot` (Step 2), `frequency_bands` (Step 3),
  `coupling_matrix` (Step 22), `channel_coherence` (Step 23) — channel summaries
- `alert` (Step 1) — base alert flag for cooldown logic

If `AlertReasonsStep` ran before any of these steps, all reads would return
default/empty values, producing an empty `alert_reasons` list and missing
`channel_summary`.  The cooldown suppression logic (Phase 4) also depends on
reading `alert` *after* it has been set by `CoreEWMAStep`, so premature placement
would make cooldown ineffective.

No other step reads `alert_reasons` or `channel_summary`, so placing Step 37 last
creates no downstream dependency violations.
