# fracttalix/steps/__init__.py
# Imports and exports PIPELINE list (ordered list of all 37 step classes)
# and all step classes.

from fracttalix.config import SentinelConfig
from fracttalix.steps.base import DetectorStep, RegimeBoostState
from fracttalix.steps.channels import (
    BandAnomalyStep,
    CascadePrecursorStep,
    ChannelCoherenceStep,
    CrossFrequencyCouplingStep,
    DegradationSequenceStep,
)
from fracttalix.steps.complexity import (
    AQBStep,
    EWSStep,
    MahalStep,
    RRSStep,
    SeasonalStep,
)
from fracttalix.steps.foundation import (
    CoreEWMAStep,
    CUSUMStep,
    FrequencyDecompositionStep,
    PageHinkleyStep,
    RegimeStep,
    StructuralSnapshotStep,
    VarCUSUMStep,
)
from fracttalix.steps.frequency import (
    PEStep,
    RFIStep,
    RPIStep,
    SSIStep,
)
from fracttalix.steps.hopf import HopfDetectorStep
from fracttalix.steps.physics import (
    AlertReasonsStep,
    CouplingRateStep,
    CriticalCouplingEstimationStep,
    DiagnosticWindowStep,
    KuramotoOrderStep,
    MaintenanceBurdenStep,
    PACCoefficientStep,
    PACDegradationStep,
    PhaseExtractionStep,
    ReversedSequenceStep,
    SequenceOrderingStep,
    ThroughputEstimationStep,
)
from fracttalix.steps.temporal import (
    CPDStep,
    OscDampStep,
    STIStep,
    TPSStep,
)


def _build_default_pipeline(config: SentinelConfig):
    """Return ordered list of DetectorStep instances for a SentinelDetector.

    38 steps. Steps numbered 1-38 consecutively.
    Phase 1.3: CoreEWMAStep and RegimeStep share a RegimeBoostState object —
    no scratch-key side channel.
    Phase 0: @register_step / _STEP_REGISTRY removed; pipeline is explicit.
    """
    boost = RegimeBoostState()                      # Phase 1.3: shared boost state
    core = CoreEWMAStep(config, boost_state=boost)  # Step 1 — MUST be first
    regime = RegimeStep(config, boost_state=boost)  # Step 5 — shares same boost
    rrs = RRSStep(config, regime)
    return [
        core,                                       # Step  1 — EWMA baseline
        StructuralSnapshotStep(config),             # Step  2 — Channel 1
        FrequencyDecompositionStep(config),         # Step  3 — Channel 2
        CUSUMStep(config),                          # Step  4 — mean shift
        regime,                                     # Step  5 — regime detection
        VarCUSUMStep(config),                       # Step  6 — variance shift
        PageHinkleyStep(config),                    # Step  7 — slow drift
        STIStep(config),                            # Step  8 — shear-turbulence
        TPSStep(config),                            # Step  9 — phase space
        OscDampStep(config),                        # Step 10 — osc damping
        CPDStep(config),                            # Step 11 — change point
        RPIStep(config),                            # Step 12 — rhythm periodicity
        RFIStep(config),                            # Step 13 — fractal index
        SSIStep(config),                            # Step 14 — synchrony stability
        PEStep(config),                             # Step 15 — permutation entropy
        EWSStep(config),                            # Step 16 — early warning
        AQBStep(config),                            # Step 17 — adaptive quantile
        SeasonalStep(config),                       # Step 18 — seasonal baseline
        MahalStep(config),                          # Step 19 — Mahalanobis
        rrs,                                        # Step 20 — resonance score
        BandAnomalyStep(config),                    # Step 21 — per-band anomaly
        CrossFrequencyCouplingStep(config),         # Step 22 — coupling matrix
        ChannelCoherenceStep(config),               # Step 23 — coherence
        CascadePrecursorStep(config),               # Step 24 — cascade precursor
        DegradationSequenceStep(config),            # Step 25 — sequence logging
        ThroughputEstimationStep(config),           # Step 26 — throughput
        MaintenanceBurdenStep(config),              # Step 27 — Tainter μ
        PhaseExtractionStep(config),                # Step 28 — band phases
        PACCoefficientStep(config),                 # Step 29 — PAC MI
        PACDegradationStep(config),                 # Step 30 — PAC degradation
        CriticalCouplingEstimationStep(config),     # Step 31 — κ_c
        CouplingRateStep(config),                   # Step 32 — dκ̄/dt
        DiagnosticWindowStep(config),               # Step 33 — Δt triple
        KuramotoOrderStep(config),                  # Step 34 — true Φ
        SequenceOrderingStep(config),               # Step 35 — sequence ordering
        ReversedSequenceStep(config),               # Step 36 — intervention sig
        HopfDetectorStep(config),                   # Step 37 — Hopf λ detector (v13)
        AlertReasonsStep(config),                   # Step 38 — MUST be last
    ]


# Build PIPELINE as a function reference (not a static list — requires config)
PIPELINE = _build_default_pipeline

__all__ = [
    # Base
    "DetectorStep",
    "RegimeBoostState",
    # Foundation (Steps 1-7)
    "CoreEWMAStep",
    "StructuralSnapshotStep",
    "FrequencyDecompositionStep",
    "CUSUMStep",
    "RegimeStep",
    "VarCUSUMStep",
    "PageHinkleyStep",
    # Temporal (Steps 8-11)
    "STIStep",
    "TPSStep",
    "OscDampStep",
    "CPDStep",
    # Frequency (Steps 12-15)
    "RPIStep",
    "RFIStep",
    "SSIStep",
    "PEStep",
    # Complexity (Steps 16-20)
    "EWSStep",
    "AQBStep",
    "SeasonalStep",
    "MahalStep",
    "RRSStep",
    # Channels (Steps 21-25)
    "BandAnomalyStep",
    "CrossFrequencyCouplingStep",
    "ChannelCoherenceStep",
    "CascadePrecursorStep",
    "DegradationSequenceStep",
    # Physics (Steps 26-37)
    "ThroughputEstimationStep",
    "MaintenanceBurdenStep",
    "PhaseExtractionStep",
    "PACCoefficientStep",
    "PACDegradationStep",
    "CriticalCouplingEstimationStep",
    "CouplingRateStep",
    "DiagnosticWindowStep",
    "KuramotoOrderStep",
    "SequenceOrderingStep",
    "ReversedSequenceStep",
    # Hopf detector (Step 37, v13)
    "HopfDetectorStep",
    "AlertReasonsStep",
    # Pipeline builder
    "PIPELINE",
    "_build_default_pipeline",
]
