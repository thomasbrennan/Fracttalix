# fracttalix/__init__.py
# Fracttalix Sentinel V12 — streaming anomaly detector package

try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("fracttalix")
    except PackageNotFoundError:
        __version__ = "12.2.0"
except ImportError:
    __version__ = "12.2.0"

from fracttalix.config import SentinelConfig
from fracttalix.types import (
    FrequencyBands,
    StructuralSnapshot,
    CouplingMatrix,
    ChannelCoherence,
    DegradationSequence,
    AlertSeverity,
    AlertType,
    Alert,
    SentinelResult,
)
from fracttalix.window import WindowBank, StepContext
from fracttalix.steps.base import DetectorStep, RegimeBoostState
from fracttalix.steps import (
    # Foundation
    CoreEWMAStep,
    StructuralSnapshotStep,
    FrequencyDecompositionStep,
    CUSUMStep,
    RegimeStep,
    VarCUSUMStep,
    PageHinkleyStep,
    # Temporal
    STIStep,
    TPSStep,
    OscDampStep,
    CPDStep,
    # Frequency
    RPIStep,
    RFIStep,
    SSIStep,
    PEStep,
    # Complexity
    EWSStep,
    AQBStep,
    SeasonalStep,
    MahalStep,
    RRSStep,
    # Channels
    BandAnomalyStep,
    CrossFrequencyCouplingStep,
    ChannelCoherenceStep,
    CascadePrecursorStep,
    DegradationSequenceStep,
    # Physics
    ThroughputEstimationStep,
    MaintenanceBurdenStep,
    PhaseExtractionStep,
    PACCoefficientStep,
    PACDegradationStep,
    CriticalCouplingEstimationStep,
    CouplingRateStep,
    DiagnosticWindowStep,
    KuramotoOrderStep,
    SequenceOrderingStep,
    ReversedSequenceStep,
    AlertReasonsStep,
    # Pipeline builder
    PIPELINE,
    _build_default_pipeline,
)
from fracttalix.detector import SentinelDetector, Detector_7_10, _legacy_kwargs_to_config
from fracttalix.multistream import MultiStreamSentinel

__all__ = [
    "__version__",
    # Config
    "SentinelConfig",
    # Types
    "FrequencyBands",
    "StructuralSnapshot",
    "CouplingMatrix",
    "ChannelCoherence",
    "DegradationSequence",
    "AlertSeverity",
    "AlertType",
    "Alert",
    "SentinelResult",
    # Window
    "WindowBank",
    "StepContext",
    # Steps base
    "DetectorStep",
    "RegimeBoostState",
    # Foundation steps (1-7)
    "CoreEWMAStep",
    "StructuralSnapshotStep",
    "FrequencyDecompositionStep",
    "CUSUMStep",
    "RegimeStep",
    "VarCUSUMStep",
    "PageHinkleyStep",
    # Temporal steps (8-11)
    "STIStep",
    "TPSStep",
    "OscDampStep",
    "CPDStep",
    # Frequency steps (12-15)
    "RPIStep",
    "RFIStep",
    "SSIStep",
    "PEStep",
    # Complexity steps (16-20)
    "EWSStep",
    "AQBStep",
    "SeasonalStep",
    "MahalStep",
    "RRSStep",
    # Channel steps (21-25)
    "BandAnomalyStep",
    "CrossFrequencyCouplingStep",
    "ChannelCoherenceStep",
    "CascadePrecursorStep",
    "DegradationSequenceStep",
    # Physics steps (26-37)
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
    "AlertReasonsStep",
    # Pipeline
    "PIPELINE",
    "_build_default_pipeline",
    # Detector
    "SentinelDetector",
    "Detector_7_10",
    "_legacy_kwargs_to_config",
    # Multi-stream
    "MultiStreamSentinel",
]
