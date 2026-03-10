# fracttalix/types.py
# V9.0 data structures — Three-Channel Model
# AlertSeverity, AlertType, Alert
# SentinelResult dict subclass with convenience methods
# _build_sequence_narrative() helper

import dataclasses
import math
from enum import Enum
from typing import Any, Dict, Optional


# ===========================================================================
# V9.0 DATA STRUCTURES — Three-Channel Model
# ===========================================================================


@dataclasses.dataclass(frozen=True)
class FrequencyBands:
    """Channel 2 decomposition into five independent carrier waves.

    Computed via FFT decomposition of windowed signal.
    Five bands: ultra-low (trend), low (slow oscillation),
    mid (primary rhythmicity), high (fast fluctuation),
    ultra-high (noise floor).
    """
    ultra_low_power: float
    low_power: float
    mid_power: float
    high_power: float
    ultra_high_power: float
    ultra_low_phase: float
    low_phase: float
    mid_phase: float
    high_phase: float
    ultra_high_phase: float
    timestamp: int


@dataclasses.dataclass(frozen=True)
class StructuralSnapshot:
    """Channel 1 structural properties at current timestep.

    Treats network topology as active transmitter — not passive substrate.
    Encodes history, identity, and capacity of the input data stream
    at the current moment independently of rhythmic properties.
    """
    mean: float
    variance: float
    skewness: float
    kurtosis: float
    autocorrelation_lag1: float
    autocorrelation_lag2: float
    stationarity_score: float
    timestamp: int


@dataclasses.dataclass(frozen=True)
class CouplingMatrix:
    """Cross-frequency coupling coefficients between adjacent band pairs.

    Phase-amplitude coupling: lower band phase to higher band amplitude.
    Coupling degradation is an earlier warning signal than single-band
    anomaly — the heterodyned information channel degrades first.
    Declining composite_coupling_score precedes regime change.
    """
    ultra_low_to_low: float
    low_to_mid: float
    mid_to_high: float
    high_to_ultra_high: float
    composite_coupling_score: float
    coupling_trend: float       # Positive = strengthening, negative = degrading
    timestamp: int


@dataclasses.dataclass(frozen=True)
class ChannelCoherence:
    """Structural-rhythmic coherence measurement.

    Measures degree to which Channel 1 and Channel 2 remain coupled.
    In a healthy network structural changes are reflected in rhythmic
    changes and vice versa. Decoupling indicates channels have lost
    coherence — itself a regime change signal independent of anomalies
    in either individual channel.
    coherence_score: 0.0 = fully decoupled, 1.0 = fully coherent.
    """
    coherence_score: float
    structural_change_rate: float
    rhythmic_change_rate: float
    decoupling_trend: float
    timestamp: int


@dataclasses.dataclass(frozen=True)
class DegradationSequence:
    """Temporal ordering of channel degradation events (Channel 3 information).

    The sequence and ordering of degradation is diagnostic about regime
    change type and severity.
    """
    first_channel_anomaly: str
    first_anomaly_timestamp: int
    second_channel_anomaly: Optional[str]
    second_anomaly_timestamp: Optional[int]
    coupling_degradation_timestamp: Optional[int]
    decoupling_timestamp: Optional[int]
    cascade_precursor_timestamp: Optional[int]
    sequence_pattern: str       # Human-readable degradation narrative


class AlertSeverity(Enum):
    """Severity levels for v9.0 structured alerts."""
    INFO = 1
    WARNING = 2
    ALERT = 3
    CRITICAL = 4    # Reserved for CASCADE_PRECURSOR only


class AlertType(Enum):
    """Alert type classification for v9.0 structured alerts."""
    # v8.0 alert type identifiers (as string labels)
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"
    DRIFT_ANOMALY = "drift_anomaly"
    VARIANCE_ANOMALY = "variance_anomaly"
    REGIME_CHANGE = "regime_change"
    EWS_WARNING = "ews_warning"
    # V9.0 alert types
    BAND_ANOMALY = "band_anomaly"
    COUPLING_DEGRADATION = "coupling_degradation"
    STRUCTURAL_RHYTHMIC_DECOUPLING = "sr_decoupling"
    CASCADE_PRECURSOR = "cascade_precursor"


@dataclasses.dataclass(frozen=True)
class Alert:
    """Structured alert object for v9.0 three-channel detection."""
    alert_type: AlertType
    severity: AlertSeverity
    score: float
    message: str


def _build_sequence_narrative(alert_types_set: set) -> str:
    """Build human-readable degradation narrative from active alert type set."""
    parts = []
    if AlertType.BAND_ANOMALY in alert_types_set:
        parts.append("carrier wave anomaly detected")
    if AlertType.COUPLING_DEGRADATION in alert_types_set:
        parts.append("cross-frequency coupling degrading")
    if AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING in alert_types_set:
        parts.append("structural-rhythmic channels decoupling")
    if AlertType.CASCADE_PRECURSOR in alert_types_set:
        parts.append("CASCADE PRECURSOR — scale-level reversion risk")
    if not parts:
        parts.append("no active degradation")
    return " \u2192 ".join(parts)


# ===========================================================================
# V9.0 SentinelResult — dict subclass with Three-Channel convenience methods
# ===========================================================================


class SentinelResult(dict):
    """V9.0 result object — dict subclass with Three-Channel convenience methods.

    All existing dict access patterns preserved for backward compatibility::

        result = det.update_and_check(value)
        if result["alert"]:          # v8.0 pattern — still works
            ...
        if result.is_cascade_precursor():  # v9.0 pattern
            ...

    """

    def is_cascade_precursor(self) -> bool:
        """Returns True if CASCADE_PRECURSOR alert is active."""
        return bool(self.get("cascade_precursor_active", False))

    def get_channel_status(self) -> Dict[str, str]:
        """Returns status of each channel: healthy, degrading, anomalous, or unknown."""
        status: Dict[str, str] = {}

        ss = self.get("structural_snapshot")
        if ss is not None:
            if ss.stationarity_score < 0.3:
                status["structural"] = "anomalous"
            elif ss.stationarity_score < 0.6:
                status["structural"] = "degrading"
            else:
                status["structural"] = "healthy"
        else:
            status["structural"] = "unknown"

        score = self.get("anomaly_score", 0.0)
        if score >= 1.0:
            status["rhythmic_composite"] = "anomalous"
        elif score > 0.5:
            status["rhythmic_composite"] = "degrading"
        else:
            status["rhythmic_composite"] = "healthy"

        cm = self.get("coupling_matrix")
        if cm is not None:
            if cm.coupling_trend < -0.1:
                status["coupling"] = "degrading"
            elif cm.composite_coupling_score < 0.3:
                status["coupling"] = "anomalous"
            else:
                status["coupling"] = "healthy"
        else:
            status["coupling"] = "unknown"

        cc = self.get("channel_coherence")
        if cc is not None:
            if cc.coherence_score < 0.2:
                status["coherence"] = "anomalous"
            elif cc.coherence_score < 0.4:
                status["coherence"] = "degrading"
            else:
                status["coherence"] = "healthy"
        else:
            status["coherence"] = "unknown"

        return status

    def get_degradation_narrative(self) -> str:
        """Returns human-readable description of current degradation sequence."""
        ds = self.get("degradation_sequence")
        if ds is not None:
            return ds.sequence_pattern
        return "No active degradation sequence."

    def get_primary_carrier_wave(self) -> str:
        """Returns frequency band currently carrying the most information."""
        fb = self.get("frequency_bands")
        if fb is None:
            return "decomposition_disabled"
        bands = {
            "ultra_low": fb.ultra_low_power,
            "low": fb.low_power,
            "mid": fb.mid_power,
            "high": fb.high_power,
            "ultra_high": fb.ultra_high_power,
        }
        return max(bands, key=bands.get)

    # ------------------------------------------------------------------
    # V10.0 convenience methods
    # ------------------------------------------------------------------

    def is_reversed_sequence(self) -> bool:
        """Returns True if coherence is collapsing before coupling degrades.

        Thermodynamic reversal — indicates possible external intervention
        rather than organic decay.
        """
        return bool(self.get("reversed_sequence", False))

    def get_intervention_signature(self) -> Dict[str, Any]:
        """Returns intervention signature analysis.

        Keys:
            score: 0.0-1.0 confidence of deliberate intervention.
            sequence_type: ORGANIC / REVERSED / AMBIGUOUS / INSUFFICIENT_DATA.
            phi_rate: rate of coherence change.
            coupling_rate: rate of coupling change.
        """
        return {
            "score": self.get("intervention_signature_score", 0.0),
            "sequence_type": self.get("sequence_type", "UNKNOWN"),
            "phi_rate": self.get("phi_rate", 0.0),
            "coupling_rate": self.get("coupling_rate", 0.0),
        }

    def get_diagnostic_window(self) -> Dict[str, Any]:
        """Returns estimated time until coherence collapse.

        Phase 5.1: adds pessimistic/expected/optimistic triple.
        Keys:
            steps: expected steps until collapse (None if not applicable).
            steps_pessimistic: accelerated-rate estimate.
            steps_optimistic: decelerated-rate estimate.
            confidence: HIGH / MEDIUM / LOW / NOT_APPLICABLE / RATE_TOO_SMALL.
            supercompensation: True if adaptive response in progress.
        """
        return {
            "steps": self.get("diagnostic_window_steps", None),
            "steps_pessimistic": self.get("diagnostic_window_steps_pessimistic", None),
            "steps_optimistic": self.get("diagnostic_window_steps_optimistic", None),
            "confidence": self.get("diagnostic_window_confidence", "NOT_APPLICABLE"),
            "supercompensation": self.get("supercompensation_detected", False),
        }

    def get_maintenance_burden(self) -> Dict[str, Any]:
        """Returns maintenance burden μ (Tainter regime detection).

        Keys:
            mu: maintenance burden 0.0-1.0.
            regime: HEALTHY / REDUCED_RESERVE / TAINTER_WARNING / TAINTER_CRITICAL.
        """
        return {
            "mu": self.get("maintenance_burden", 0.0),
            "regime": self.get("tainter_regime", "UNKNOWN"),
        }

    def get_pac_status(self) -> Dict[str, Any]:
        """Returns PAC (Phase-Amplitude Coupling) status.

        Keys:
            mean_pac: current PAC strength 0.0-1.0.
            degradation_rate: rate of PAC decline.
            pre_cascade_pac: True if PAC is warning before cascade precursor fires.
        """
        return {
            "mean_pac": self.get("mean_pac", 0.0),
            "degradation_rate": self.get("pac_degradation_rate", 0.0),
            "pre_cascade_pac": self.get("pre_cascade_pac", False),
        }

    def get_phi_kappa_separation(self) -> Dict[str, Any]:
        """Returns Phi-kappa separation — phase coherence vs coupling strength gap.

        Phase 3.4 new metric.
        Keys:
            separation: Phi - kappa_bar. Positive = coherence exceeds coupling
                        (intervention signature). Negative = organic degradation.
            phi: Kuramoto order parameter (v11.0 true Phi).
            kappa: mean coupling strength.
            interpretation: COHERENCE_LED / COUPLING_LED / BALANCED.
        """
        sep = self.get("phi_kappa_separation", 0.0)
        phi = self.get("kuramoto_order", 0.0)
        kappa = self.get("mean_coupling_strength", 0.0)
        if sep > 0.05:
            interp = "COHERENCE_LED"
        elif sep < -0.05:
            interp = "COUPLING_LED"
        else:
            interp = "BALANCED"
        return {
            "separation": sep,
            "phi": phi,
            "kappa": kappa,
            "interpretation": interp,
        }
