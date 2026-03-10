# fracttalix/steps/channels.py
# Steps 21-25: BandAnomalyStep, CrossFrequencyCouplingStep, ChannelCoherenceStep,
#              CascadePrecursorStep, DegradationSequenceStep

import math
from collections import deque
from typing import Any, Dict, List, Optional

from fracttalix._compat import _mean
from fracttalix.config import SentinelConfig
from fracttalix.steps.base import DetectorStep
from fracttalix.types import (
    Alert, AlertSeverity, AlertType,
    ChannelCoherence, CouplingMatrix, DegradationSequence,
    StructuralSnapshot, _build_sequence_narrative,
)
from fracttalix.window import StepContext

# Phase-amplitude coupling bin count for modulation index computation.
PAC_PHASE_BINS = 8


# ---------------------------------------------------------------------------
# Step 21: BandAnomalyStep
# ---------------------------------------------------------------------------

class BandAnomalyStep(DetectorStep):
    """Per-carrier-wave anomaly detection invisible to composite signal.

    Step 21 — run after existing detection steps.
    Applies independent EWMA anomaly detection to each frequency band's
    power time series. A band-specific anomaly not present in the composite
    signal is invisible to v8.0 but detectable here.
    """

    BAND_NAMES: List[str] = ["ultra_low", "low", "mid", "high", "ultra_high"]

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._band_ewma: Dict[str, float] = {}
        self._band_dev: Dict[str, float] = {}

    def _ewma_anomaly_score(self, power: float, band_name: str) -> float:
        """Return normalized anomaly score (>1.0 = anomalous) for a band."""
        alpha = self.cfg.alpha
        if band_name not in self._band_ewma:
            self._band_ewma[band_name] = power
            self._band_dev[band_name] = 1.0
            return 0.0
        prev = self._band_ewma[band_name]
        self._band_ewma[band_name] = alpha * power + (1 - alpha) * prev
        err = abs(power - prev)
        self._band_dev[band_name] = (
            alpha * err + (1 - alpha) * self._band_dev.get(band_name, 1.0))
        self._band_dev[band_name] = max(self._band_dev[band_name], 1e-10)
        z = abs(power - self._band_ewma[band_name]) / self._band_dev[band_name]
        return z / (self.cfg.multiplier + 1e-10)

    def update(self, ctx: StepContext) -> None:
        fb = ctx.scratch.get("frequency_bands")
        if fb is None:
            ctx.scratch["band_anomalies"] = {}
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        current_powers = [
            fb.ultra_low_power, fb.low_power, fb.mid_power,
            fb.high_power, fb.ultra_high_power,
        ]
        anomalies: Dict[str, float] = {}
        for name, power in zip(self.BAND_NAMES, current_powers):
            score = self._ewma_anomaly_score(power, name)
            if score > 1.0:
                anomalies[name] = min(1.0, score)

        ctx.scratch["band_anomalies"] = anomalies
        alerts: List[Alert] = list(ctx.scratch.get("v9_active_alerts", []))
        if anomalies and not ctx.is_warmup and not ctx.scratch.get("alert", False):
            # Per-band anomaly not present in composite — new information
            alert = Alert(
                alert_type=AlertType.BAND_ANOMALY,
                severity=AlertSeverity.WARNING,
                score=max(anomalies.values()),
                message=(
                    f"Per-band anomaly in bands: {sorted(anomalies.keys())}. "
                    f"Invisible to composite detection."
                ),
            )
            alerts.append(alert)
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        ctx.scratch["v9_active_alerts"] = alerts

    def state_dict(self) -> Dict[str, Any]:
        return {"band_ewma": dict(self._band_ewma), "band_dev": dict(self._band_dev)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._band_ewma = sd.get("band_ewma", {})
        self._band_dev = sd.get("band_dev", {})


# ---------------------------------------------------------------------------
# Step 22: CrossFrequencyCouplingStep
# ---------------------------------------------------------------------------

class CrossFrequencyCouplingStep(DetectorStep):
    """Measure cross-frequency phase-amplitude coupling and detect degradation.

    Step 22 — computes CouplingMatrix and generates COUPLING_DEGRADATION
    alert when composite_coupling_score falls below threshold.
    Earlier warning than single-band anomaly detection.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._coupling_degradation_active: bool = False

    @staticmethod
    def _pac(low_phases: List[float], high_powers: List[float]) -> float:
        """Phase-amplitude coupling: modulation index approximation."""
        n = len(low_phases)
        if n < 2:
            return 0.0
        bins: List[List[float]] = [[] for _ in range(PAC_PHASE_BINS)]
        for ph, pw in zip(low_phases, high_powers):
            bin_idx = int((ph + math.pi) / (2 * math.pi) * PAC_PHASE_BINS) % PAC_PHASE_BINS
            bins[bin_idx].append(pw)
        bin_means = [sum(b) / len(b) if b else 0.0 for b in bins]
        overall_mean = sum(bin_means) / float(PAC_PHASE_BINS)
        variance = sum((m - overall_mean) ** 2 for m in bin_means) / float(PAC_PHASE_BINS)
        best_dev = max(bin_means) - overall_mean
        max_variance = best_dev ** 2 if best_dev > 0 else 1e-10
        return variance / (max_variance + 1e-10)

    def _composite_from_slice(self, bands_slice: list) -> float:
        if len(bands_slice) < 2:
            return 0.0
        return sum([
            self._pac([b.ultra_low_phase for b in bands_slice],
                      [b.low_power for b in bands_slice]),
            self._pac([b.low_phase for b in bands_slice],
                      [b.mid_power for b in bands_slice]),
            self._pac([b.mid_phase for b in bands_slice],
                      [b.high_power for b in bands_slice]),
            self._pac([b.high_phase for b in bands_slice],
                      [b.ultra_high_power for b in bands_slice]),
        ]) / 4.0

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_coupling_detection:
            ctx.scratch["coupling_matrix"] = None
            ctx.scratch["coupling_degradation_active"] = False
            self._coupling_degradation_active = False
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        bands_history = ctx.scratch.get("_bands_history")
        if bands_history is None or len(bands_history) < self.cfg.coupling_trend_window:
            ctx.scratch["coupling_matrix"] = None
            ctx.scratch["coupling_degradation_active"] = self._coupling_degradation_active
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        recent = list(bands_history)[-self.cfg.coupling_trend_window:]
        ul_phases = [b.ultra_low_phase for b in recent]
        l_phases  = [b.low_phase for b in recent]
        m_phases  = [b.mid_phase for b in recent]
        h_phases  = [b.high_phase for b in recent]
        l_powers  = [b.low_power for b in recent]
        m_powers  = [b.mid_power for b in recent]
        h_powers  = [b.high_power for b in recent]
        uh_powers = [b.ultra_high_power for b in recent]

        ul_to_l = self._pac(ul_phases, l_powers)
        l_to_m  = self._pac(l_phases,  m_powers)
        m_to_h  = self._pac(m_phases,  h_powers)
        h_to_uh = self._pac(h_phases,  uh_powers)
        composite = (ul_to_l + l_to_m + m_to_h + h_to_uh) / 4.0

        half = len(recent) // 2
        if half >= 2:
            early = self._composite_from_slice(recent[:half])
            late  = self._composite_from_slice(recent[half:])
            trend = late - early
        else:
            trend = 0.0

        coupling = CouplingMatrix(
            ultra_low_to_low=ul_to_l,
            low_to_mid=l_to_m,
            mid_to_high=m_to_h,
            high_to_ultra_high=h_to_uh,
            composite_coupling_score=composite,
            coupling_trend=trend,
            timestamp=ctx.step,
        )
        ctx.scratch["coupling_matrix"] = coupling

        alerts: List[Alert] = list(ctx.scratch.get("v9_active_alerts", []))
        if composite < self.cfg.coupling_degradation_threshold and not ctx.is_warmup:
            self._coupling_degradation_active = True
            alert = Alert(
                alert_type=AlertType.COUPLING_DEGRADATION,
                severity=AlertSeverity.WARNING,
                score=composite,
                message=(
                    f"Cross-frequency coupling degraded to {composite:.3f} "
                    f"(threshold: {self.cfg.coupling_degradation_threshold}). "
                    f"Trend: {trend:+.3f}. Heterodyned information channel at risk."
                ),
            )
            alerts.append(alert)
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        else:
            self._coupling_degradation_active = False

        ctx.scratch["coupling_degradation_active"] = self._coupling_degradation_active
        ctx.scratch["v9_active_alerts"] = alerts

    def state_dict(self) -> Dict[str, Any]:
        return {"coupling_degradation_active": self._coupling_degradation_active}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._coupling_degradation_active = sd.get("coupling_degradation_active", False)


# ---------------------------------------------------------------------------
# Step 23: ChannelCoherenceStep
# ---------------------------------------------------------------------------

class ChannelCoherenceStep(DetectorStep):
    """Measure structural-rhythmic channel coherence and detect decoupling.

    Step 23 — computes ChannelCoherence and generates
    STRUCTURAL_RHYTHMIC_DECOUPLING alert when coherence_score falls below
    coherence_threshold. Independent of anomalies in either individual channel.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._rhythmic_history: deque = deque(
            maxlen=self.cfg.coherence_window * 2)
        self._sr_decoupling_active: bool = False

    @staticmethod
    def _pearson_coherence(sc: list, rc: list) -> float:
        """Pearson correlation between structural and rhythmic change series.

        Scale-invariant: eliminates the unit mismatch between windowed-variance
        changes (structural) and EWMA changes (rhythmic).  Returns a value in
        [0, 1] where 1 = perfectly co-moving, 0.5 = uncorrelated, 0 = opposing.
        Returns 0.5 (neutral) when either series is constant or too short.
        """
        n = min(len(sc), len(rc))
        if n < 3:
            return 0.5
        sc_n = sc[-n:]
        rc_n = rc[-n:]
        sc_mean = _mean(sc_n)
        rc_mean = _mean(rc_n)
        num = sum((a - sc_mean) * (b - rc_mean) for a, b in zip(sc_n, rc_n))
        denom_s = math.sqrt(sum((a - sc_mean) ** 2 for a in sc_n))
        denom_r = math.sqrt(sum((b - rc_mean) ** 2 for b in rc_n))
        if denom_s < 1e-10 or denom_r < 1e-10:
            return 0.5
        r = num / (denom_s * denom_r)
        return (r + 1.0) / 2.0  # map [-1, 1] → [0, 1]

    def _coherence_score_from_slices(
        self, struct_slice: list, rhythmic_slice: list
    ) -> float:
        if len(struct_slice) < 2 or len(rhythmic_slice) < 2:
            return 0.5
        n = min(len(struct_slice), len(rhythmic_slice))
        ss = struct_slice[-n:]
        rs = rhythmic_slice[-n:]
        sc = [abs(ss[i].variance - ss[i - 1].variance) for i in range(1, len(ss))]
        rc = [abs(rs[i] - rs[i - 1]) for i in range(1, len(rs))]
        return self._pearson_coherence(sc, rc)

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_channel_coherence:
            ctx.scratch["channel_coherence"] = None
            ctx.scratch["sr_decoupling_active"] = False
            self._sr_decoupling_active = False
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        # Accumulate rhythmic proxy (EWMA value)
        self._rhythmic_history.append(ctx.scratch.get("ewma", ctx.current))

        structural_history = ctx.scratch.get("_structural_snapshot_history")
        window = self.cfg.coherence_window

        if (structural_history is None
                or len(structural_history) < window
                or len(self._rhythmic_history) < window):
            ctx.scratch["channel_coherence"] = None
            ctx.scratch["sr_decoupling_active"] = self._sr_decoupling_active
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        recent_struct = list(structural_history)[-window:]
        recent_rhythmic = list(self._rhythmic_history)[-window:]

        sc = [abs(recent_struct[i].variance - recent_struct[i - 1].variance)
              for i in range(1, len(recent_struct))]
        rc = [abs(recent_rhythmic[i] - recent_rhythmic[i - 1])
              for i in range(1, len(recent_rhythmic))]
        structural_change_rate = _mean(sc) if sc else 0.0
        rhythmic_change_rate = _mean(rc) if rc else 0.0
        coherence_score = self._pearson_coherence(sc, rc)

        half = window // 2
        early_coh = self._coherence_score_from_slices(
            recent_struct[:half], recent_rhythmic[:half])
        late_coh = self._coherence_score_from_slices(
            recent_struct[half:], recent_rhythmic[half:])
        decoupling_trend = late_coh - early_coh

        coherence = ChannelCoherence(
            coherence_score=coherence_score,
            structural_change_rate=structural_change_rate,
            rhythmic_change_rate=rhythmic_change_rate,
            decoupling_trend=decoupling_trend,
            timestamp=ctx.step,
        )
        ctx.scratch["channel_coherence"] = coherence

        alerts: List[Alert] = list(ctx.scratch.get("v9_active_alerts", []))
        if coherence_score < self.cfg.coherence_threshold and not ctx.is_warmup:
            self._sr_decoupling_active = True
            alert = Alert(
                alert_type=AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING,
                severity=AlertSeverity.ALERT,
                score=coherence_score,
                message=(
                    f"Channel 1-2 coherence degraded to {coherence_score:.3f} "
                    f"(threshold: {self.cfg.coherence_threshold}). "
                    f"Structural rate: {structural_change_rate:.4f}. "
                    f"Rhythmic rate: {rhythmic_change_rate:.4f}. "
                    f"Channels operating independently."
                ),
            )
            alerts.append(alert)
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        else:
            self._sr_decoupling_active = False

        ctx.scratch["sr_decoupling_active"] = self._sr_decoupling_active
        ctx.scratch["v9_active_alerts"] = alerts

    def state_dict(self) -> Dict[str, Any]:
        return {
            "rhythmic_history": list(self._rhythmic_history),
            "sr_decoupling_active": self._sr_decoupling_active,
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._rhythmic_history = deque(
            sd.get("rhythmic_history", []),
            maxlen=self.cfg.coherence_window * 2,
        )
        self._sr_decoupling_active = sd.get("sr_decoupling_active", False)


# ---------------------------------------------------------------------------
# Step 24: CascadePrecursorStep
# ---------------------------------------------------------------------------

class CascadePrecursorStep(DetectorStep):
    """Detect incipient scale-level reversion cascade precursor.

    Step 24 — requires ALL conditions simultaneously:
      1. COUPLING_DEGRADATION alert active
      2. STRUCTURAL_RHYTHMIC_DECOUPLING alert active
      3. At least cascade_ews_threshold EWS indicators elevated
    This combined signature indicates not a local anomaly but an incipient
    scale-level reversion event.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._cascade_active: bool = False

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_cascade_detection:
            ctx.scratch["cascade_precursor_active"] = False
            ctx.scratch.setdefault("v9_active_alerts", [])
            return

        coupling_active = ctx.scratch.get("coupling_degradation_active", False)
        decoupling_active = ctx.scratch.get("sr_decoupling_active", False)

        ews_indicators = {
            "ews_score": ctx.scratch.get("ews_score", 0.0),
            "anomaly_score": ctx.scratch.get("anomaly_score", 0.0),
            "rfi": ctx.scratch.get("rfi", 0.0),
        }
        elevated_ews = sum(
            1 for v in ews_indicators.values() if v > self.cfg.ews_threshold
        )

        alerts: List[Alert] = list(ctx.scratch.get("v9_active_alerts", []))
        if (coupling_active and decoupling_active
                and elevated_ews >= self.cfg.cascade_ews_threshold
                and not ctx.is_warmup):
            self._cascade_active = True
            alert = Alert(
                alert_type=AlertType.CASCADE_PRECURSOR,
                severity=AlertSeverity.CRITICAL,
                score=1.0,
                message=(
                    f"CASCADE PRECURSOR DETECTED. "
                    f"Cross-frequency coupling degraded. "
                    f"Structural-rhythmic decoupling active. "
                    f"{elevated_ews} EWS indicators elevated. "
                    f"Combined signature indicates incipient scale-level "
                    f"reversion event. Immediate review warranted."
                ),
            )
            alerts.append(alert)
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        else:
            self._cascade_active = False

        ctx.scratch["cascade_precursor_active"] = self._cascade_active
        ctx.scratch["v9_active_alerts"] = alerts

    def state_dict(self) -> Dict[str, Any]:
        return {"cascade_active": self._cascade_active}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._cascade_active = sd.get("cascade_active", False)


# ---------------------------------------------------------------------------
# Step 25: DegradationSequenceStep
# ---------------------------------------------------------------------------

class DegradationSequenceStep(DetectorStep):
    """Log temporal ordering of channel degradation events (Channel 3).

    Step 25 — the sequence and ordering of degradation is diagnostic
    about regime change type and severity.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._active_sequence: Optional[DegradationSequence] = None
        self._sequence_history: deque = deque(maxlen=self.cfg.sequence_retention)

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_sequence_logging:
            ctx.scratch["degradation_sequence"] = None
            return

        active_alerts: List[Alert] = ctx.scratch.get("v9_active_alerts", [])
        alert_types = {a.alert_type for a in active_alerts}
        now = ctx.step

        if not alert_types:
            if self._active_sequence is not None:
                self._sequence_history.append(self._active_sequence)
                self._active_sequence = None
        elif self._active_sequence is None:
            first = list(alert_types)[0]
            self._active_sequence = DegradationSequence(
                first_channel_anomaly=first.value,
                first_anomaly_timestamp=now,
                second_channel_anomaly=None,
                second_anomaly_timestamp=None,
                coupling_degradation_timestamp=(
                    now if AlertType.COUPLING_DEGRADATION in alert_types else None),
                decoupling_timestamp=(
                    now if AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING in alert_types else None),
                cascade_precursor_timestamp=(
                    now if AlertType.CASCADE_PRECURSOR in alert_types else None),
                sequence_pattern=_build_sequence_narrative(alert_types),
            )
        else:
            existing = self._active_sequence
            alert_values = {a.value for a in alert_types}
            new_vals = alert_values - {existing.first_channel_anomaly}
            second = existing.second_channel_anomaly or (
                next(iter(new_vals), None) if new_vals else None
            )
            self._active_sequence = DegradationSequence(
                first_channel_anomaly=existing.first_channel_anomaly,
                first_anomaly_timestamp=existing.first_anomaly_timestamp,
                second_channel_anomaly=second,
                second_anomaly_timestamp=existing.second_anomaly_timestamp or (
                    now if second is not None
                    and existing.second_channel_anomaly is None else None
                ),
                coupling_degradation_timestamp=(
                    existing.coupling_degradation_timestamp or
                    (now if AlertType.COUPLING_DEGRADATION in alert_types else None)
                ),
                decoupling_timestamp=(
                    existing.decoupling_timestamp or
                    (now if AlertType.STRUCTURAL_RHYTHMIC_DECOUPLING in alert_types else None)
                ),
                cascade_precursor_timestamp=(
                    existing.cascade_precursor_timestamp or
                    (now if AlertType.CASCADE_PRECURSOR in alert_types else None)
                ),
                sequence_pattern=_build_sequence_narrative(alert_types),
            )

        ctx.scratch["degradation_sequence"] = self._active_sequence

    def state_dict(self) -> Dict[str, Any]:
        return {"n_sequences": len(self._sequence_history)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass  # in-memory history only
