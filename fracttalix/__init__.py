# fracttalix/__init__.py
# Fracttalix Sentinel V12 — streaming anomaly detector package

try:
    from importlib.metadata import PackageNotFoundError, version
    try:
        __version__ = version("fracttalix")
    except PackageNotFoundError:
        __version__ = "12.1.0"
except ImportError:
    __version__ = "12.1.0"

from fracttalix.config import SentinelConfig
from fracttalix.detector import Detector_7_10, SentinelDetector, _legacy_kwargs_to_config
from fracttalix.multistream import MultiStreamSentinel
from fracttalix.steps import (
    # Pipeline builder
    PIPELINE,
    AlertReasonsStep,
    AQBStep,
    # Channels
    BandAnomalyStep,
    CascadePrecursorStep,
    ChannelCoherenceStep,
    # Foundation
    CoreEWMAStep,
    CouplingRateStep,
    CPDStep,
    CriticalCouplingEstimationStep,
    CrossFrequencyCouplingStep,
    CUSUMStep,
    DegradationSequenceStep,
    DiagnosticWindowStep,
    # Complexity
    EWSStep,
    FrequencyDecompositionStep,
    KuramotoOrderStep,
    MahalStep,
    MaintenanceBurdenStep,
    OscDampStep,
    PACCoefficientStep,
    PACDegradationStep,
    PageHinkleyStep,
    PEStep,
    PhaseExtractionStep,
    RegimeStep,
    ReversedSequenceStep,
    RFIStep,
    # Frequency
    RPIStep,
    RRSStep,
    SeasonalStep,
    SequenceOrderingStep,
    SSIStep,
    # Temporal
    STIStep,
    StructuralSnapshotStep,
    # Physics
    ThroughputEstimationStep,
    TPSStep,
    VarCUSUMStep,
    _build_default_pipeline,
)
from fracttalix.steps.base import DetectorStep, RegimeBoostState
from fracttalix.types import (
    Alert,
    AlertSeverity,
    AlertType,
    ChannelCoherence,
    CouplingMatrix,
    DegradationSequence,
    FrequencyBands,
    SentinelResult,
    StructuralSnapshot,
)
from fracttalix.window import StepContext, WindowBank

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
