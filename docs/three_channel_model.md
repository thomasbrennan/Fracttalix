# Three-Channel Dissipative Network Model

Fracttalix Sentinel is built on the Three-Channel Dissipative Network Model, introduced in the Fractal Rhythm Model working papers (DOI: [10.5281/zenodo.18859299](https://doi.org/10.5281/zenodo.18859299)). The model describes how information is transmitted through a network that dissipates energy — a regime in which structural, oscillatory, and temporal properties carry independent diagnostic information about the network's health.

## The Three Channels

### Channel 1 — Structural

Channel 1 treats the network itself as the active transmitter. A network dissipates energy maintaining the topology that allows information to flow. When the network is healthy, its structural properties (mean, variance, skewness, kurtosis, lag-1 autocorrelation, stationarity) remain within characteristic bounds. As the network approaches a critical transition, two classical signatures emerge:

- **Variance amplification** — the system's response to perturbations grows as the restoring force weakens.
- **Critical slowing down** — lag-1 autocorrelation increases as the system takes longer to recover from small disturbances.

`StructuralSnapshotStep` (step 2) computes all six structural statistics at every observation. `EWSStep` (step 16) monitors variance and autocorrelation jointly as Early Warning Signals, firing when both are elevated simultaneously.

### Channel 2 — Rhythmic

Channel 2 treats broadband oscillatory patterns as the information carrier. Real-world signals — sensor readings, network traffic, market prices — are superpositions of carrier waves at different frequencies. Degradation manifests first in the coupling *between* frequency bands before it becomes visible in the aggregate signal.

`FrequencyDecompositionStep` (step 3) decomposes each observation window via FFT into five carrier-wave bands:

| Band | Approximate Frequency Range |
|------|-----------------------------|
| Ultra-low | Bottom 10% of spectrum |
| Low | 10–25% |
| Mid | 25–50% |
| High | 50–75% |
| Ultra-high | 75–100% |

Per-band power and instantaneous phase are stored in a `FrequencyBands` frozen dataclass and passed downstream.

`CrossFrequencyCouplingStep` (step 22) measures phase-amplitude coupling (PAC) between adjacent slow-phase/fast-amplitude band pairs using the Modulation Index method (Tort et al. 2010). A `CouplingMatrix` records the six PAC coefficients and a composite coupling score. When the composite score falls below `coupling_degradation_threshold`, a `COUPLING_DEGRADATION` alert (WARNING severity) is emitted.

### Channel 3 — Temporal

Channel 3 monitors the one-way irreversible ordering of degradation events. In a dissipative network, time has a thermodynamic arrow: coupling architecture degrades before coherence collapses. This sequencing is not an assumption — it is a consequence of the hierarchy of energy scales in the network.

`DegradationSequenceStep` (step 25) maintains a log of the order in which Channel 1 (structural) and Channel 2 (rhythmic) degradation events occur. The temporal ordering is the primary input to the reversed-sequence detection logic in the V12 physics group.

## How the Channels Interact

The channels interact through three inter-channel steps:

**`ChannelCoherenceStep` (step 23)** measures the coherence between Channel 1 and Channel 2 signals over a rolling window. When structural and rhythmic degradation become decorrelated — a sign that the two information pathways are no longer jointly encoding the same system state — the `coherence_score` falls below `coherence_threshold` and a `STRUCTURAL_RHYTHMIC_DECOUPLING` alert (ALERT severity) is emitted.

**`CascadePrecursorStep` (step 24)** fires the `CASCADE_PRECURSOR` alert (CRITICAL severity) when all three conditions are simultaneously satisfied:

1. `COUPLING_DEGRADATION` is active (Channel 2 PAC below threshold)
2. `STRUCTURAL_RHYTHMIC_DECOUPLING` is active (Channel 1–2 coherence below threshold)
3. At least `cascade_ews_threshold` Early Warning Signal indicators are elevated (Channel 1)

This multi-condition gate makes `CASCADE_PRECURSOR` a high-specificity signal. The three conditions correspond to three independent failure modes of the dissipative network, and their simultaneous presence is a strong indicator of imminent coherence collapse.

**`SequenceOrderingStep` (step 35)** classifies the current step's ordering of coupling and coherence changes as one of four types:

| Type | Meaning |
|------|---------|
| `COUPLING_FIRST` | Coupling degraded before coherence — thermodynamically normal |
| `COHERENCE_FIRST` | Coherence collapsed before coupling — anomalous ordering |
| `SIMULTANEOUS` | Both degraded within the same observation window |
| `STABLE` | No degradation event at this step |

## The Cascade Degradation Sequence

The full cascade sequence, as implemented across the 37-step pipeline, proceeds:

```
Band anomaly detected (BandAnomalyStep, step 21)
  → Cross-frequency coupling degrades (CrossFrequencyCouplingStep, step 22)
    → Structural-rhythmic channels decouple (ChannelCoherenceStep, step 23)
      → CASCADE_PRECURSOR: CRITICAL (CascadePrecursorStep, step 24)
```

The V12 physics group extends this with the pre-cascade PAC signature:

```
PAC Modulation Index declines (PACDegradationStep, step 30)
  → pre_cascade_pac flag set (earlier than COUPLING_DEGRADATION)
    → Diagnostic window Δt opens (DiagnosticWindowStep, step 33)
      → Maintenance burden μ → TAINTER_CRITICAL (MaintenanceBurdenStep, step 27)
        → Coherence collapse imminent
```

## Physics Metrics Reference

### Maintenance Burden μ

```
μ = N · κ̄ · E_coupling / P_throughput
```

Operationally, `ThroughputEstimationStep` (step 26) estimates P_throughput from band amplitudes and counts active nodes N. `MaintenanceBurdenStep` (step 27) computes μ and maps it to a Tainter regime. When μ = 1, the network has no adaptive reserve: all energy is consumed by maintaining existing coupling, with nothing left for adaptation or recovery.

### Diagnostic Window Δt

```
Δt = (κ̄ − κ_c) / |dκ̄/dt|
```

- κ̄ is the current mean coupling strength (from `ThroughputEstimationStep`)
- κ_c is the critical coupling threshold below which synchronization cannot be maintained, estimated from the power-weighted frequency distribution by `CriticalCouplingEstimationStep` (step 31) using the Kuramoto formula κ_c = 2 / (π · g(ω₀))
- dκ̄/dt is the rate of change of coupling strength, computed by `CouplingRateStep` (step 32) from a rolling history

`DiagnosticWindowStep` (step 33) also detects supercompensation — when dκ̄/dt becomes positive after a period of decline — which indicates adaptive recovery in progress.

### Kuramoto Order Parameter Φ

```
Φ = |mean(e^{iθ_k})|
```

`KuramotoOrderStep` (step 34) computes the Kuramoto order parameter as the magnitude of the mean complex exponential of instantaneous phases θ_k across the five frequency bands. Φ = 1 indicates full phase synchronization; Φ → 0 indicates complete incoherence. The rate of change of Φ is compared to the rate of change of κ̄ in `ReversedSequenceStep` (step 36) to detect the intervention signature.

### Intervention Signature

When Φ collapses faster than κ̄ decreases, the temporal ordering is reversed. `ReversedSequenceStep` computes an `intervention_signature_score` (0.0–1.0) from the ratio of the two rates. A high score indicates that the degradation sequence does not match the thermodynamically expected organic collapse pattern — consistent with measurement error, a different universality class, or deliberate external disruption of the network.
