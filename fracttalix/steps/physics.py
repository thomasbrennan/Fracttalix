# fracttalix/steps/physics.py
# Steps 26-37: ThroughputEstimationStep, MaintenanceBurdenStep, PhaseExtractionStep,
#              PACCoefficientStep, PACDegradationStep, CriticalCouplingEstimationStep,
#              CouplingRateStep, DiagnosticWindowStep, KuramotoOrderStep,
#              SequenceOrderingStep, ReversedSequenceStep, AlertReasonsStep

import math
from collections import deque
from typing import Any, Dict, List

from fracttalix._compat import _NP, _SCIPY, _scipy_signal, np
from fracttalix.config import SentinelConfig
from fracttalix.steps.base import DetectorStep
from fracttalix.window import StepContext

# ---------------------------------------------------------------------------
# Step 26: ThroughputEstimationStep
# ---------------------------------------------------------------------------

class ThroughputEstimationStep(DetectorStep):
    """Estimate network energy throughput P_throughput from carrier wave amplitudes.

    Step 26 — inserted after DegradationSequenceStep.
    Throughput proxy: mean squared amplitude across all active frequency bands.
    P_throughput = (1/T) * integral(A(t)^2 dt)
    Also populates band_amplitudes, band_powers, node_count, and
    mean_coupling_strength for downstream v10 steps.
    """

    BAND_NAMES: List[str] = ["ultra_low", "low", "mid", "high", "ultra_high"]

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        pass

    def update(self, ctx: StepContext) -> None:
        fb = ctx.scratch.get("frequency_bands")
        bands_history = ctx.scratch.get("_bands_history")

        # Current band powers dict
        band_powers: Dict[str, float] = {}
        if fb is not None:
            band_powers = {
                "ultra_low": fb.ultra_low_power,
                "low": fb.low_power,
                "mid": fb.mid_power,
                "high": fb.high_power,
                "ultra_high": fb.ultra_high_power,
            }

        # Band amplitude time-series from history
        band_amplitudes: Dict[str, List[float]] = {}
        if bands_history is not None and len(bands_history) > 0:
            history_list = list(bands_history)
            band_amplitudes = {
                "ultra_low": [b.ultra_low_power for b in history_list],
                "low":       [b.low_power for b in history_list],
                "mid":       [b.mid_power for b in history_list],
                "high":      [b.high_power for b in history_list],
                "ultra_high":[b.ultra_high_power for b in history_list],
            }

        # Throughput: sum of mean-squared amplitudes across bands
        total_power = 0.0
        if _NP:
            for amp_list in band_amplitudes.values():
                if amp_list:
                    arr = np.array(amp_list, dtype=float)
                    total_power += float(np.mean(arr ** 2))
        else:
            for amp_list in band_amplitudes.values():
                if amp_list:
                    total_power += sum(a ** 2 for a in amp_list) / len(amp_list)

        # Node count: scalar window length as proxy for network size
        w = ctx.bank.get("scalar")
        node_count = max(1, len(w))

        # Mean coupling strength from coupling matrix
        cm = ctx.scratch.get("coupling_matrix")
        mean_coupling = cm.composite_coupling_score if cm is not None else 0.0

        ctx.scratch["band_amplitudes"] = band_amplitudes
        ctx.scratch["band_powers"] = band_powers
        ctx.scratch["throughput"] = total_power
        ctx.scratch["node_count"] = node_count
        ctx.scratch["mean_coupling_strength"] = mean_coupling

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Step 27: MaintenanceBurdenStep
# ---------------------------------------------------------------------------

class MaintenanceBurdenStep(DetectorStep):
    """Compute maintenance burden μ — Tainter regime detection.

    Step 27.

    Phase 3 correction: v10.0 formula μ = N·κ̄ / P_throughput mixed a
    window-length count (N≈64) with a unitless coupling score and a power
    value, producing a number dominated by window size rather than network
    state.

    v11.0 formula: μ = 1 − κ̄  (empirical heuristic).

    This is NOT derived from physics or Tainter's socioeconomic model.  It
    is an engineering heuristic: low mean coupling κ̄ is interpreted as high
    coordination overhead and therefore high maintenance burden.  The
    intermediate decomposition in the implementation (coupling_cost = κ̄(1−κ̄),
    productive_surplus = κ̄²) is algebraic bookkeeping only; those functional
    forms are not grounded in any recognised energy or network model and must
    not be cited as such in publications.

    μ ∈ [0, 1], dimensionless and independent of window size.
    Boundary cases: κ̄ → 0 ⟹ μ → 1 (fully fragmented);
                    κ̄ → 1 ⟹ μ → 0 (fully coupled, zero inferred overhead).

    The v10.0 value is preserved as maintenance_burden_v10 for one cycle.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        pass

    def update(self, ctx: StepContext) -> None:
        mean_coupling = ctx.scratch.get("mean_coupling_strength", 0.0)
        kappa = max(0.0, min(1.0, mean_coupling))

        # v10.0 legacy (kept as alias)
        n_nodes = ctx.scratch.get("node_count", 1)
        throughput = ctx.scratch.get("throughput", 1.0)
        mu_v10 = min(1.0, (n_nodes * kappa) / throughput) if throughput > 0 else 1.0
        ctx.scratch["maintenance_burden_v10"] = mu_v10

        # v11.0: μ = 1 − κ̄  (heuristic; algebraic decomposition below is
        # bookkeeping only, not an energy-fraction derivation)
        coupling_cost = kappa * (1.0 - kappa)
        productive_surplus = kappa * kappa
        denom = coupling_cost + productive_surplus
        mu = coupling_cost / denom if denom > 1e-12 else 1.0
        mu = max(0.0, min(1.0, mu))

        # Regime thresholds are empirically set; not calibrated from data.
        if mu >= 0.9:
            regime = "TAINTER_CRITICAL"
        elif mu >= 0.75:
            regime = "TAINTER_WARNING"
        elif mu >= 0.5:
            regime = "REDUCED_RESERVE"
        else:
            regime = "HEALTHY"

        ctx.scratch["maintenance_burden"] = mu
        ctx.scratch["tainter_regime"] = regime

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Step 28: PhaseExtractionStep
# ---------------------------------------------------------------------------

class PhaseExtractionStep(DetectorStep):
    """Extract instantaneous phase from each frequency band via Hilbert transform.

    Step 28 — Uses scipy.signal.hilbert if available; falls back to a
    numpy FFT-based analytic signal construction; falls back to zero phases.
    Band-filtered signals are reconstructed by FFT bandpass.
    """

    BAND_FREQ_RANGES: Dict[str, tuple] = {
        "ultra_low": (0.00, 0.05),
        "low":       (0.05, 0.15),
        "mid":       (0.15, 0.40),
        "high":      (0.40, 0.70),
        "ultra_high":(0.70, 1.00),
    }

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        pass

    def _hilbert_phase(self, arr: Any) -> Any:
        """Instantaneous phase array via Hilbert transform."""
        if _SCIPY:
            analytic = _scipy_signal.hilbert(arr)
            return np.angle(analytic)
        elif _NP:
            # FFT-based analytic signal
            N = len(arr)
            fft = np.fft.fft(arr)
            h = np.zeros(N, dtype=float)
            if N % 2 == 0:
                h[0] = 1.0
                h[1:N // 2] = 2.0
                h[N // 2] = 1.0
            else:
                h[0] = 1.0
                h[1:(N + 1) // 2] = 2.0
            analytic = np.fft.ifft(fft * h)
            return np.angle(analytic)
        else:
            return [0.0] * len(arr)

    def _reconstruct_band(self, data: List[float], lo: float, hi: float) -> Any:
        """Band-pass reconstruct signal via FFT zeroing."""
        if not _NP:
            return list(data)
        N = len(data)
        arr = np.array(data, dtype=float)
        fft = np.fft.rfft(arr)
        freqs = np.fft.rfftfreq(N)
        mask = (freqs >= lo) & (freqs < hi)
        filtered_fft = np.zeros_like(fft)
        filtered_fft[mask] = fft[mask]
        return np.fft.irfft(filtered_fft, N)

    def update(self, ctx: StepContext) -> None:
        # Phase 2: cap window to rpi_window (default 64) — O(N log N) bounded
        w_all = list(ctx.bank.get("scalar"))
        cap = max(self.cfg.rpi_window, 16)
        w = w_all[-cap:] if len(w_all) > cap else w_all
        n = len(w)

        if n < 4:
            ctx.scratch["band_filtered_signals"] = {}
            ctx.scratch["band_phases"] = {}
            return

        band_filtered_signals: Dict[str, Any] = {}
        band_phases: Dict[str, Any] = {}

        for band_name, (lo, hi) in self.BAND_FREQ_RANGES.items():
            filtered = self._reconstruct_band(w, lo, hi)
            band_filtered_signals[band_name] = filtered

            if _NP:
                arr = np.asarray(filtered, dtype=float)
                if len(arr) >= 4:
                    band_phases[band_name] = self._hilbert_phase(arr)
                else:
                    band_phases[band_name] = np.array([])
            else:
                if len(filtered) >= 4:
                    band_phases[band_name] = [0.0] * len(filtered)
                else:
                    band_phases[band_name] = []

        ctx.scratch["band_filtered_signals"] = band_filtered_signals
        ctx.scratch["band_phases"] = band_phases

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Step 29: PACCoefficientStep
# ---------------------------------------------------------------------------

class PACCoefficientStep(DetectorStep):
    """Compute PAC modulation index between slow-phase and fast-amplitude pairs.

    Step 29 — Modulation Index (Tort et al. 2010): KL divergence between
    phase-binned amplitude distribution and uniform distribution.
    Higher MI = stronger PAC = deeper nonlinear coupling = more structural memory.
    Slow bands: ultra_low, low.  Fast bands: mid, high, ultra_high.
    """

    SLOW_BANDS: List[str] = ["ultra_low", "low"]
    FAST_BANDS: List[str] = ["mid", "high", "ultra_high"]
    N_BINS: int = 18

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        pass

    def _compute_MI(self, phase: Any, amplitude: Any) -> float:
        """Modulation Index via KL divergence (requires numpy)."""
        if not _NP:
            return 0.0
        phase = np.asarray(phase, dtype=float)
        amplitude = np.asarray(amplitude, dtype=float)
        if len(phase) == 0 or len(amplitude) == 0:
            return 0.0
        min_len = min(len(phase), len(amplitude))
        phase = phase[:min_len]
        amplitude = amplitude[:min_len]

        bins = np.linspace(-math.pi, math.pi, self.N_BINS + 1)
        amp_by_phase = np.zeros(self.N_BINS)
        for i in range(self.N_BINS):
            idx = np.where((phase >= bins[i]) & (phase < bins[i + 1]))[0]
            if len(idx) > 0:
                amp_by_phase[i] = np.mean(np.abs(amplitude[idx]))

        total = amp_by_phase.sum()
        if total == 0:
            return 0.0
        p = amp_by_phase / total
        p = p + 1e-10
        MI = float(np.sum(p * np.log(p * self.N_BINS)) / np.log(self.N_BINS))
        return float(np.clip(MI, 0.0, 1.0))

    def update(self, ctx: StepContext) -> None:
        band_phases = ctx.scratch.get("band_phases", {})
        band_amplitudes = ctx.scratch.get("band_amplitudes", {})

        pac_matrix: Dict[str, float] = {}
        for slow in self.SLOW_BANDS:
            for fast in self.FAST_BANDS:
                slow_phases = band_phases.get(slow)
                fast_amps = band_amplitudes.get(fast)
                if (slow_phases is not None and fast_amps is not None
                        and len(slow_phases) > 0 and len(fast_amps) > 0):
                    key = f"{slow}_phase_{fast}_amp"
                    mi = self._compute_MI(slow_phases, fast_amps)
                    pac_matrix[key] = mi

        mean_pac = (float(sum(pac_matrix.values()) / len(pac_matrix))
                    if pac_matrix else 0.0)
        ctx.scratch["pac_matrix"] = pac_matrix
        ctx.scratch["mean_pac"] = mean_pac

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Step 30: PACDegradationStep
# ---------------------------------------------------------------------------

class PACDegradationStep(DetectorStep):
    """Track PAC degradation rate — pre-cascade signature.

    Step 30 — PAC degrades before coupling strength κ̄ measurably decreases,
    extending the diagnostic window.  pre_cascade_pac fires when:
    1. PAC degradation rate > PAC_DEGRADATION_THRESHOLD AND
    2. cascade_precursor_active is False (PAC warns BEFORE κ̄ warns).

    Phase 4: degradation_rate now uses linear regression slope over the full
    history window (previously a crude 3-point early/recent comparison).
    Negative slope = degrading; reported as positive rate value for API compat.
    """

    PAC_DEGRADATION_THRESHOLD: float = 0.05
    # Phase 4: threshold lowered from 0.15 to 0.05 to match OLS slope scale.
    # Linear regression slope on [0,1]-bounded PAC data produces slopes in
    # the range [0, ~0.1] per step; 0.05 corresponds to ~5% per-step decline.

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._pac_history: deque = deque(maxlen=10)

    @staticmethod
    def _linreg_slope(y: List[float]) -> float:
        """OLS slope of y over integer x indices. Pure-Python, no numpy needed."""
        n = len(y)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(y) / n
        num = sum((i - x_mean) * (yi - y_mean) for i, yi in enumerate(y))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 1e-12 else 0.0

    def update(self, ctx: StepContext) -> None:
        current_pac = ctx.scratch.get("mean_pac", 0.0)
        self._pac_history.append(current_pac)
        pac_list = list(self._pac_history)
        ctx.scratch["pac_history"] = pac_list

        if len(pac_list) < 3:
            ctx.scratch["pac_degradation_rate"] = 0.0
            ctx.scratch["pre_cascade_pac"] = False
            return

        # Phase 4: linear regression slope (negative = degrading)
        slope = self._linreg_slope(pac_list)
        # Report as positive degradation rate (negative slope = positive rate)
        degradation_rate = max(0.0, -slope)

        cascade_active = ctx.scratch.get("cascade_precursor_active", False)
        pre_cascade_pac = (degradation_rate > self.PAC_DEGRADATION_THRESHOLD
                           and not cascade_active)

        ctx.scratch["pac_degradation_rate"] = float(degradation_rate)
        ctx.scratch["pre_cascade_pac"] = pre_cascade_pac

    def state_dict(self) -> Dict[str, Any]:
        return {"pac_history": list(self._pac_history)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._pac_history = deque(sd.get("pac_history", []), maxlen=10)


# ---------------------------------------------------------------------------
# Step 31: CriticalCouplingEstimationStep
# ---------------------------------------------------------------------------

class CriticalCouplingEstimationStep(DetectorStep):
    """Estimate critical coupling threshold κ_c from frequency distribution.

    Step 31.

    Phase 3 correction: v10.0 used BAND_CENTERS in Hz (0.5, 2.0, 8.0, 32.0,
    128.0) while the band decomposition uses normalized frequencies (0.0-1.0).
    Applying Kuramoto κ_c = 2/(π·g(ω₀)) with Hz-scale statistics to a
    normalized-frequency decomposition is dimensionally inconsistent.

    v11.0 uses normalized band midpoints consistent with BAND_FREQ_RANGES:
      ultra_low:  midpoint 0.025
      low:        midpoint 0.10
      mid:        midpoint 0.275
      high:       midpoint 0.55
      ultra_high: midpoint 0.85

    g(ω₀) is estimated as the normalized frequency spread:
      g(ω₀) = 1 / (1 + weighted_std / weighted_mean)
    This is bounded (0, 1] and increases as frequency content concentrates —
    matching the Kuramoto intuition that a narrow natural-frequency distribution
    is easier to synchronize (lower κ_c).

    The v10.0 value is preserved as critical_coupling_v10 for one cycle.
    """

    # Normalized band midpoints (consistent with PhaseExtractionStep ranges)
    BAND_CENTERS: Dict[str, float] = {
        "ultra_low": 0.025,
        "low":       0.10,
        "mid":       0.275,
        "high":      0.55,
        "ultra_high":0.85,
    }

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        pass

    def update(self, ctx: StepContext) -> None:
        band_powers = ctx.scratch.get("band_powers", {})
        active = {k: v for k, v in band_powers.items()
                  if v > 0 and k in self.BAND_CENTERS}

        # v10.0 legacy with old Hz centers (keep alias)
        OLD_CENTERS = {
            "ultra_low": 0.5, "low": 2.0, "mid": 8.0,
            "high": 32.0, "ultra_high": 128.0,
        }
        if active:
            total_old = sum(active.values())
            w_old = {k: v / total_old for k, v in active.items()}
            wm_old = sum(OLD_CENTERS[k] * w for k, w in w_old.items())
            wv_old = sum(w * (OLD_CENTERS[k] - wm_old) ** 2 for k, w in w_old.items())
            ws_old = math.sqrt(max(wv_old, 0.0))
            g_old = max(0.1, 1.0 - (ws_old / wm_old)) if wm_old > 0 else 1.0
            ctx.scratch["critical_coupling_v10"] = 2.0 / (math.pi * g_old)
        else:
            ctx.scratch["critical_coupling_v10"] = 0.5

        if not active:
            ctx.scratch["critical_coupling"] = 0.5
            return

        total = sum(active.values())
        weights = {k: v / total for k, v in active.items()}

        weighted_mean = sum(self.BAND_CENTERS[k] * w for k, w in weights.items())
        weighted_var = sum(w * (self.BAND_CENTERS[k] - weighted_mean) ** 2
                          for k, w in weights.items())
        weighted_std = (float(np.sqrt(weighted_var)) if _NP
                        else math.sqrt(max(weighted_var, 0.0)))

        # g(ω₀): bounded in (0,1] — higher when frequency content is concentrated
        g_omega = 1.0 / (1.0 + (weighted_std / (weighted_mean + 1e-10)))
        kappa_c = 2.0 / (math.pi * max(g_omega, 0.01))
        ctx.scratch["critical_coupling"] = float(min(kappa_c, 10.0))

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Step 32: CouplingRateStep
# ---------------------------------------------------------------------------

class CouplingRateStep(DetectorStep):
    """Compute rate of change of mean coupling strength dκ̄/dt.

    Step 32 — Negative = coupling degrading; positive = strengthening.
    History maintained in step state across observations.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._coupling_history: deque = deque(maxlen=10)

    def update(self, ctx: StepContext) -> None:
        current = ctx.scratch.get("mean_coupling_strength", 0.0)
        self._coupling_history.append(current)
        history = list(self._coupling_history)
        ctx.scratch["coupling_history"] = history

        if len(history) < 2:
            ctx.scratch["coupling_rate"] = 0.0
            return

        if _NP:
            rate = float(np.mean(np.diff(history)))
        else:
            diffs = [history[i + 1] - history[i] for i in range(len(history) - 1)]
            rate = sum(diffs) / len(diffs)

        ctx.scratch["coupling_rate"] = rate

    def state_dict(self) -> Dict[str, Any]:
        return {"coupling_history": list(self._coupling_history)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._coupling_history = deque(sd.get("coupling_history", []), maxlen=10)


# ---------------------------------------------------------------------------
# Step 33: DiagnosticWindowStep
# ---------------------------------------------------------------------------

class DiagnosticWindowStep(DetectorStep):
    """Estimate time remaining before coherence collapse: Δt = (κ̄ - κ_c) / |dκ̄/dt|.

    Step 33 — Only meaningful when κ̄ > κ_c and dκ̄/dt < 0.
    Confidence: HIGH/MEDIUM/LOW based on history length and rate stability.
    Also detects supercompensation (adaptive response: coupling rising above baseline).
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        pass

    def update(self, ctx: StepContext) -> None:
        kappa_bar = ctx.scratch.get("mean_coupling_strength", 0.0)
        kappa_c = ctx.scratch.get("critical_coupling", 0.5)
        coupling_rate = ctx.scratch.get("coupling_rate", 0.0)
        coupling_history = ctx.scratch.get("coupling_history", [])

        # Supercompensation: coupling rising above recent baseline
        supercompensation = False
        if coupling_rate > 0 and len(coupling_history) >= 5:
            baseline = sum(coupling_history[:3]) / 3.0
            current_avg = sum(coupling_history[-3:]) / 3.0
            supercompensation = current_avg > baseline * 1.05
        ctx.scratch["supercompensation_detected"] = supercompensation

        # Not degrading or already below critical — window not applicable
        if coupling_rate >= 0 or kappa_bar <= kappa_c:
            ctx.scratch["diagnostic_window_steps"] = None
            ctx.scratch["diagnostic_window_confidence"] = "NOT_APPLICABLE"
            return

        margin = kappa_bar - kappa_c
        rate_magnitude = abs(coupling_rate)
        if rate_magnitude < 1e-10:
            ctx.scratch["diagnostic_window_steps"] = None
            ctx.scratch["diagnostic_window_steps_optimistic"] = None
            ctx.scratch["diagnostic_window_steps_pessimistic"] = None
            ctx.scratch["diagnostic_window_confidence"] = "RATE_TOO_SMALL"
            return

        delta_t = margin / rate_magnitude
        ctx.scratch["diagnostic_window_steps"] = float(delta_t)

        # Phase 5.1: pessimistic/expected/optimistic triple
        if len(coupling_history) >= 4 and _NP:
            recent_rates = np.diff(np.array(coupling_history, dtype=float))
            rate_std = float(np.std(recent_rates)) if len(recent_rates) > 1 else 0.0
            rate_cv = rate_std / (rate_magnitude + 1e-10)
            # Pessimistic: rate accelerates by 1 std
            pessimistic_rate = rate_magnitude + rate_std
            # Optimistic: rate decelerates by 1 std (floor at small positive)
            optimistic_rate = max(rate_magnitude - rate_std, rate_magnitude * 0.1)
            ctx.scratch["diagnostic_window_steps_pessimistic"] = float(
                margin / pessimistic_rate) if pessimistic_rate > 1e-10 else None
            ctx.scratch["diagnostic_window_steps_optimistic"] = float(
                margin / optimistic_rate)
            if rate_cv < 0.3:
                confidence = "HIGH"
            elif rate_cv < 0.7:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
        else:
            ctx.scratch["diagnostic_window_steps_pessimistic"] = None
            ctx.scratch["diagnostic_window_steps_optimistic"] = None
            confidence = "LOW"

        ctx.scratch["diagnostic_window_confidence"] = confidence

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Step 34: KuramotoOrderStep
# ---------------------------------------------------------------------------

class KuramotoOrderStep(DetectorStep):
    """Compute Kuramoto order parameter Φ from inter-band phase relationships.

    Step 34 — True Φ = |1/N * sum(e^(i*theta_k))| over ALL per-sample phase
    values pooled across all active frequency bands.

    Phase 3 correction: v10.0 computed the mean phase per band (5 values),
    then averaged those 5 phasors — yielding "inter-band alignment", not true Φ.
    v11.0 pools every sample's instantaneous phase across all bands as individual
    oscillators, producing the standard Kuramoto order parameter.
    Φ=1: perfect phase coherence.  Φ=0: complete phase incoherence.
    The v10.0 value is preserved as kuramoto_order_v10 for one release cycle.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        pass

    def update(self, ctx: StepContext) -> None:
        band_phases = ctx.scratch.get("band_phases", {})

        if not band_phases or not _NP:
            ctx.scratch["kuramoto_order"] = 0.0
            ctx.scratch["kuramoto_order_v10"] = 0.0
            return

        # v10.0 legacy computation (kept as _v10 alias)
        legacy_vectors: List[complex] = []
        for phase_array in band_phases.values():
            arr = np.asarray(phase_array, dtype=float)
            if len(arr) > 0:
                mean_phase = float(np.angle(np.mean(np.exp(1j * arr))))
                legacy_vectors.append(complex(math.cos(mean_phase),
                                              math.sin(mean_phase)))
        phi_v10 = (abs(sum(legacy_vectors) / len(legacy_vectors))
                   if legacy_vectors else 0.0)
        ctx.scratch["kuramoto_order_v10"] = float(phi_v10)

        # v11.0 true Kuramoto Φ: pool all per-sample phases as individual oscillators
        all_phases: List[float] = []
        for phase_array in band_phases.values():
            arr = np.asarray(phase_array, dtype=float)
            if arr.ndim == 1 and len(arr) > 0:
                all_phases.extend(arr.tolist())

        if not all_phases:
            ctx.scratch["kuramoto_order"] = 0.0
            return

        phases_arr = np.array(all_phases, dtype=float)
        Phi = float(np.abs(np.mean(np.exp(1j * phases_arr))))
        ctx.scratch["kuramoto_order"] = Phi

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Step 35: SequenceOrderingStep
# ---------------------------------------------------------------------------

class SequenceOrderingStep(DetectorStep):
    """Track relative degradation sequence of coupling κ̄ and coherence Φ.

    Step 35 — Normal thermodynamic sequence: coupling degrades before
    coherence collapses.  Reversed: coherence collapses before coupling
    degrades.  Records COUPLING_FIRST / COHERENCE_FIRST / SIMULTANEOUS /
    STABLE per observation.

    Phase 4: degradation threshold normalized by each series' rolling std,
    replacing the absolute -0.05 that failed to detect slow degradation.
    Threshold = -0.5 * std(series) — degrades if moving at half a std-dev
    per step in the negative direction.  Falls back to -0.05 if insufficient data.
    """

    DEGRADATION_THRESHOLD_FALLBACK: float = -0.05
    THRESHOLD_SIGMA_FACTOR: float = 0.5

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._phi_history: deque = deque(maxlen=15)
        self._coupling_history_local: deque = deque(maxlen=15)
        self._sequence_history: deque = deque(maxlen=10)

    @staticmethod
    def _adaptive_threshold(series: List[float], factor: float,
                            fallback: float) -> float:
        """Return -factor * std(series), floored at fallback."""
        if len(series) < 4:
            return fallback
        mu = sum(series) / len(series)
        var = sum((x - mu) ** 2 for x in series) / len(series)
        std = math.sqrt(var) if var > 0 else 0.0
        return -factor * std if std > 1e-10 else fallback

    def update(self, ctx: StepContext) -> None:
        current_phi = ctx.scratch.get("kuramoto_order", 1.0)
        self._phi_history.append(current_phi)
        phi_list = list(self._phi_history)
        ctx.scratch["phi_history"] = phi_list

        coupling_rate = ctx.scratch.get("coupling_rate", 0.0)
        self._coupling_history_local.append(coupling_rate)
        coup_list = list(self._coupling_history_local)

        if len(phi_list) < 3:
            ctx.scratch["phi_rate"] = 0.0
            ctx.scratch["coupling_degrading"] = False
            ctx.scratch["coherence_degrading"] = False
            ctx.scratch["sequence_history"] = list(self._sequence_history)
            return

        recent = phi_list[-5:] if len(phi_list) >= 5 else phi_list
        if _NP and len(recent) >= 2:
            phi_rate = float(np.mean(np.diff(recent)))
        elif len(recent) >= 2:
            diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            phi_rate = sum(diffs) / len(diffs)
        else:
            phi_rate = 0.0

        ctx.scratch["phi_rate"] = phi_rate

        # Phase 4: adaptive thresholds normalized by each series' rolling std
        phi_thresh = self._adaptive_threshold(
            phi_list, self.THRESHOLD_SIGMA_FACTOR, self.DEGRADATION_THRESHOLD_FALLBACK)
        coup_thresh = self._adaptive_threshold(
            coup_list, self.THRESHOLD_SIGMA_FACTOR, self.DEGRADATION_THRESHOLD_FALLBACK)

        coupling_degrading = coupling_rate < coup_thresh
        coherence_degrading = phi_rate < phi_thresh
        ctx.scratch["coupling_degrading"] = coupling_degrading
        ctx.scratch["coherence_degrading"] = coherence_degrading

        if coupling_degrading and not coherence_degrading:
            self._sequence_history.append("COUPLING_FIRST")
        elif coherence_degrading and not coupling_degrading:
            self._sequence_history.append("COHERENCE_FIRST")
        elif coupling_degrading and coherence_degrading:
            self._sequence_history.append("SIMULTANEOUS")
        else:
            self._sequence_history.append("STABLE")

        ctx.scratch["sequence_history"] = list(self._sequence_history)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "phi_history": list(self._phi_history),
            "coupling_history_local": list(self._coupling_history_local),
            "sequence_history": list(self._sequence_history),
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._phi_history = deque(sd.get("phi_history", []), maxlen=15)
        self._coupling_history_local = deque(
            sd.get("coupling_history_local", []), maxlen=15)
        self._sequence_history = deque(sd.get("sequence_history", []), maxlen=10)


# ---------------------------------------------------------------------------
# Step 36: ReversedSequenceStep
# ---------------------------------------------------------------------------

class ReversedSequenceStep(DetectorStep):
    """Detect reversed degradation sequence — thermodynamic reversal.

    Step 36 — Reversed sequence (coherence collapses before coupling degrades)
    indicates possible external intervention rather than organic decay.
    In civilizational terms: a civilization being collapsed vs. one that collapses.
    intervention_signature_score: 0.0-1.0 confidence of deliberate intervention.
    sequence_type: ORGANIC / REVERSED / AMBIGUOUS / INSUFFICIENT_DATA.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        pass

    def update(self, ctx: StepContext) -> None:
        sequence_history = ctx.scratch.get("sequence_history", [])
        phi_rate = ctx.scratch.get("phi_rate", 0.0)
        coupling_rate = ctx.scratch.get("coupling_rate", 0.0)

        if len(sequence_history) < 3:
            ctx.scratch["reversed_sequence"] = False
            ctx.scratch["intervention_signature_score"] = 0.0
            ctx.scratch["sequence_type"] = "INSUFFICIENT_DATA"
            return

        coherence_first_count = sequence_history.count("COHERENCE_FIRST")
        coupling_first_count = sequence_history.count("COUPLING_FIRST")

        reversed_seq = (coherence_first_count > coupling_first_count
                        and coherence_first_count >= 2)

        if reversed_seq:
            if coupling_rate == 0:
                rate_ratio = 1.0
            else:
                rate_ratio = abs(phi_rate) / (abs(coupling_rate) + 1e-10)
            n_hist = max(len(sequence_history), 1)
            raw = (coherence_first_count / n_hist) * min(rate_ratio, 2.0) / 2.0
            if _NP:
                score = float(np.clip(raw, 0.0, 1.0))
            else:
                score = max(0.0, min(1.0, raw))
            sequence_type = "REVERSED"
        elif coupling_first_count > coherence_first_count:
            score = 0.0
            sequence_type = "ORGANIC"
        else:
            # Phase 4: dynamic AMBIGUOUS score from ratio uncertainty
            # When counts are equal, score reflects how ambiguous the tie is:
            # small counts = low confidence (near 0.5 max uncertainty)
            # large equal counts = high ambiguity maintained
            n_total = max(len(sequence_history), 1)
            n_active = coherence_first_count + coupling_first_count
            ambiguity = n_active / n_total  # fraction of active (non-STABLE) steps
            score = min(0.5, 0.5 * ambiguity)
            sequence_type = "AMBIGUOUS"

        ctx.scratch["reversed_sequence"] = reversed_seq
        ctx.scratch["intervention_signature_score"] = score
        ctx.scratch["sequence_type"] = sequence_type

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass


# ---------------------------------------------------------------------------
# Step 37: AlertReasonsStep
# ---------------------------------------------------------------------------

class AlertReasonsStep(DetectorStep):
    """Aggregate alert reasons list — must be last step.

    Phase 4: supports per-detector alert cooldown via config.alert_cooldown_steps.
    When cooldown > 0, alerts are suppressed for that many steps after firing,
    preventing repeated alerts from sustained conditions.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._cooldown_remaining: int = 0

    def update(self, ctx: StepContext) -> None:
        # Phase 4: cooldown suppression
        cooldown = self.cfg.alert_cooldown_steps
        if cooldown > 0 and self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            ctx.scratch["alert"] = False
            ctx.scratch["anomaly"] = False
            ctx.scratch["alert_reasons"] = ["cooldown_suppressed"]
            ctx.scratch["channel_summary"] = "cooldown_active"
            return
        reasons: List[str] = []
        s = ctx.scratch
        if s.get("cusum_alert"):
            reasons.append("cusum_mean_shift")
        if s.get("var_cusum_alert"):
            reasons.append("cusum_variance_spike")
        # v12.3: non-adaptive drift CUSUM (warmup-frozen baseline)
        if s.get("drift_cusum_alert"):
            reasons.append("drift_cusum_shift")
        if s.get("regime_change"):
            reasons.append("regime_change")
        if s.get("ph_alert"):
            reasons.append("gradual_drift")
        if s.get("osc_alert"):
            reasons.append("oscillation_damping")
        if s.get("cpd_score", 0) > self.cfg.cpd_threshold:
            reasons.append("change_point")
        if abs(s.get("z_score", 0)) > self.cfg.multiplier:
            reasons.append("ewma_threshold")
        if s.get("rfi", 0) > self.cfg.rfi_threshold:
            reasons.append("high_fractal_irregularity")
        ews = s.get("ews_regime", "stable")
        if ews in ("approaching", "critical"):
            reasons.append(f"ews_{ews}")
        pe = s.get("pe", 0.5)
        pe_base = s.get("pe_baseline", 0.5)
        if abs(pe - pe_base) > self.cfg.pe_threshold:
            if pe < pe_base:
                reasons.append("low_entropy_ordered")
            else:
                reasons.append("high_entropy_chaotic")
        if s.get("mahal_dist", 0) > self.cfg.multiplier * math.sqrt(self.cfg.n_channels):
            reasons.append("mahalanobis_multivariate")
        # V9.0 — include structured alert types in reasons list
        for alert in s.get("v9_active_alerts", []):
            if alert.alert_type.value not in reasons:
                reasons.append(alert.alert_type.value)
        # V9.0 — compute channel summary string
        channel_parts: List[str] = []
        if s.get("structural_snapshot") is not None:
            channel_parts.append("structural:active")
        if s.get("frequency_bands") is not None:
            channel_parts.append("rhythmic:active")
        cm = s.get("coupling_matrix")
        if cm is not None:
            cst = ("degraded" if cm.composite_coupling_score < self.cfg.coupling_degradation_threshold
                   else "healthy")
            channel_parts.append(f"coupling:{cst}")
        cc = s.get("channel_coherence")
        if cc is not None:
            coh = "decoupled" if cc.coherence_score < self.cfg.coherence_threshold else "coherent"
            channel_parts.append(f"coherence:{coh}")
        ctx.scratch["channel_summary"] = (
            " | ".join(channel_parts) if channel_parts else "channels:initializing"
        )
        ctx.scratch["alert_reasons"] = reasons

        # v12.3 — Consensus Gate
        # Classify reasons into strong (fire alone) and soft (require consensus).
        # Strong: statistically robust multi-step accumulators and extremes.
        # Soft:   single-step scores with meaningful per-step FPR; require ≥2.
        _STRONG = frozenset({
            "cusum_mean_shift",
            "cusum_variance_spike",
            "drift_cusum_shift",   # v12.3: non-adaptive drift CUSUM
            "gradual_drift",
            "cascade_precursor",
        })
        strong_active = [r for r in reasons if r in _STRONG]
        soft_active = [r for r in reasons if r not in _STRONG]

        # Hard bypass: z-score > 5× multiplier is always a strong signal.
        hard_z = abs(s.get("z_score", 0.0)) >= 5.0 * self.cfg.multiplier

        gate_passes = hard_z or bool(strong_active) or len(soft_active) >= 2

        if not gate_passes:
            # Suppress alert: not enough evidence consensus.
            ctx.scratch["alert"] = False
            ctx.scratch["anomaly"] = False
            # Retain reasons for diagnostics but mark as gated.
            ctx.scratch["alert_reasons"] = [f"gated:{r}" for r in reasons]
            return

        # Phase 4: start cooldown after any real alert
        if reasons and ctx.scratch.get("alert") and cooldown > 0:
            self._cooldown_remaining = cooldown

    def state_dict(self) -> Dict[str, Any]:
        return {"cooldown_remaining": self._cooldown_remaining}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._cooldown_remaining = sd.get("cooldown_remaining", 0)
