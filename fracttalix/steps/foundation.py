# fracttalix/steps/foundation.py
# Steps 1-7: CoreEWMAStep, StructuralSnapshotStep, FrequencyDecompositionStep,
#            CUSUMStep, RegimeStep, VarCUSUMStep, PageHinkleyStep

import math
from collections import deque
from typing import Any, Dict, List, Optional

from fracttalix._compat import _mean
from fracttalix.config import SentinelConfig
from fracttalix.steps.base import DetectorStep, RegimeBoostState
from fracttalix.types import FrequencyBands, StructuralSnapshot
from fracttalix.window import StepContext, WindowBank

# ---------------------------------------------------------------------------
# Step 1: CoreEWMAStep
# ---------------------------------------------------------------------------

class CoreEWMAStep(DetectorStep):
    """Compute EWMA baseline + anomaly score.  Must be first in pipeline.

    Phase 1.3: accepts a RegimeBoostState object at construction.  The boost
    is written by RegimeStep and read here — no scratch-key side channel.
    """

    def __init__(self, config: SentinelConfig,
                 boost_state: Optional[RegimeBoostState] = None):
        self.cfg = config
        self._boost_state = boost_state or RegimeBoostState()
        self._bank_registered = False
        self.reset()

    def reset(self):
        self._ewma = 0.0
        self._dev_ewma = 1.0
        self._warmup_buf: List[float] = []
        self._initialized = False
        self._n = 0
        # Warmup baseline — frozen after warmup, used by non-adaptive CUSUM
        self._warmup_mean: float = 0.0
        self._warmup_std: float = 1.0
        # per-channel state (multivariate)
        self._ch_ewma: List[float] = []
        self._ch_dev: List[float] = []
        self._ch_init: List[bool] = []
        # AQB rolling buffer
        self._aqb_buf: deque = deque(maxlen=self.cfg.aqb_window)

    def _ensure_bank(self, bank: WindowBank):
        if not self._bank_registered:
            bank.register("scalar", max(self.cfg.pe_window, self.cfg.rpi_window,
                                        self.cfg.rfi_window, self.cfg.sti_window,
                                        self.cfg.tps_window, self.cfg.osc_damp_window,
                                        self.cfg.cpd_window, 64))
            self._bank_registered = True

    def _eff_alpha(self) -> float:
        return min(self.cfg.alpha * self._boost_state.boost, 1.0)

    def _eff_dev_alpha(self) -> float:
        return min(self.cfg.dev_alpha * self._boost_state.boost, 1.0)

    def _scalar_update(self, v: float, ctx: StepContext) -> Dict[str, Any]:
        cfg = self.cfg
        self._n += 1
        self._aqb_buf.append(abs(v - self._ewma) if self._initialized else 0.0)

        if not self._initialized:
            self._warmup_buf.append(v)
            if len(self._warmup_buf) >= cfg.warmup_periods:
                self._ewma = _mean(self._warmup_buf)
                sq = [(x - self._ewma) ** 2 for x in self._warmup_buf]
                self._dev_ewma = math.sqrt(_mean(sq)) or 1.0
                self._initialized = True
                # Freeze warmup baseline for non-adaptive CUSUM (v12.3)
                self._warmup_mean = self._ewma
                self._warmup_std = max(self._dev_ewma, 1e-10)
            return {"ewma": v, "dev_ewma": 1.0, "baseline_mean": v,
                    "baseline_std": 1.0, "z_score": 0.0,
                    "anomaly_score": 0.0, "anomaly": False,
                    "alert": False, "warmup": True,
                    "warmup_mean": 0.0, "warmup_std": 1.0}

        a = self._eff_alpha()
        da = self._eff_dev_alpha()
        prev_ewma = self._ewma
        self._ewma = a * v + (1 - a) * self._ewma
        err = abs(v - prev_ewma)
        self._dev_ewma = da * err + (1 - da) * self._dev_ewma
        self._dev_ewma = max(self._dev_ewma, 1e-10)

        z = (v - self._ewma) / self._dev_ewma
        sigma = self.cfg.multiplier

        if cfg.quantile_threshold_mode and len(self._aqb_buf) >= 10:
            sorted_buf = sorted(self._aqb_buf)
            n = len(sorted_buf)
            hi_idx = min(int(cfg.aqb_q_high * n), n - 1)
            hi_thresh = self._ewma + sorted_buf[hi_idx]
            lo_thresh = self._ewma - sorted_buf[hi_idx]
            alert = v > hi_thresh or v < lo_thresh
        else:
            alert = abs(z) > sigma

        anomaly_score = min(1.0, abs(z) / (sigma + 1e-10))

        # v12.4.2: blend in non-adaptive score from frozen warmup baseline.
        # The adaptive EWMA converges toward sustained mean shifts (collective
        # anomalies), collapsing z-scores mid-block.  The warmup baseline
        # doesn't adapt, so z_raw stays elevated throughout the block.
        if self._warmup_std > 1e-10:
            z_raw = (v - self._warmup_mean) / self._warmup_std
            raw_score = min(1.0, abs(z_raw) / (sigma + 1e-10))
            anomaly_score = max(anomaly_score, raw_score)

        return {
            "ewma": self._ewma,
            "dev_ewma": self._dev_ewma,
            "baseline_mean": self._ewma,
            "baseline_std": self._dev_ewma,
            "z_score": z,
            "anomaly_score": anomaly_score,
            "anomaly": alert,
            "alert": alert,
            "warmup": False,
            # Frozen warmup baseline for non-adaptive drift detection (v12.3)
            "warmup_mean": self._warmup_mean,
            "warmup_std": self._warmup_std,
        }

    def _mv_update(self, vs: List[float], ctx: StepContext) -> Dict[str, Any]:
        cfg = self.cfg
        nc = cfg.n_channels
        if len(self._ch_ewma) < nc:
            self._ch_ewma = [0.0] * nc
            self._ch_dev = [1.0] * nc
            self._ch_init = [False] * nc

        scores = []
        for i, v in enumerate(vs[:nc]):
            if not self._ch_init[i]:
                self._ch_ewma[i] = v
                self._ch_init[i] = True
                scores.append(0.0)
            else:
                a = self._eff_alpha()
                da = self._eff_dev_alpha()
                prev = self._ch_ewma[i]
                self._ch_ewma[i] = a * v + (1 - a) * self._ch_ewma[i]
                err = abs(v - prev)
                self._ch_dev[i] = da * err + (1 - da) * self._ch_dev[i]
                self._ch_dev[i] = max(self._ch_dev[i], 1e-10)
                scores.append(abs(v - self._ch_ewma[i]) / self._ch_dev[i])

        agg = _mean(scores) if scores else 0.0
        z = agg
        sigma = self.cfg.multiplier
        alert = z > sigma
        anomaly_score = min(1.0, z / (sigma + 1e-10))

        # Use last channel as "scalar" representative
        sv = float(vs[-1]) if vs else 0.0
        self._ewma = self._ch_ewma[-1] if self._ch_ewma else sv
        self._dev_ewma = self._ch_dev[-1] if self._ch_dev else 1.0

        return {
            "ewma": self._ewma,
            "dev_ewma": self._dev_ewma,
            "baseline_mean": self._ewma,
            "baseline_std": self._dev_ewma,
            "z_score": z,
            "anomaly_score": anomaly_score,
            "anomaly": alert,
            "alert": alert,
            "warmup": self._n < self.cfg.warmup_periods,
            "ch_ewma": list(self._ch_ewma),
            "ch_dev": list(self._ch_dev),
        }

    def update(self, ctx: StepContext) -> None:
        self._ensure_bank(ctx.bank)
        v = ctx.value
        cfg = self.cfg

        if cfg.multivariate and isinstance(v, (list, tuple, type(None))):
            if v is None:
                v = [0.0] * cfg.n_channels
            vals = [float(x) for x in v]
            result = self._mv_update(vals, ctx)
            sv = float(vals[-1]) if vals else 0.0
        else:
            # v12.3: use deseasonalized residual when SeasonalPreprocessStep
            # has detected a period and provided a residual.  This causes the
            # window bank to accumulate deseasonalized signal, eliminating
            # seasonal false positives in all downstream steps.
            ds_val = ctx.scratch.get("deseasonalized_value")
            sv = float(ds_val) if (ds_val is not None and math.isfinite(float(ds_val))) else float(v)
            result = self._scalar_update(sv, ctx)

        ctx.bank.append(sv)
        ctx.scratch.update(result)

        # Decay boost (Phase 1.3: written to shared RegimeBoostState)
        self._boost_state.boost = max(
            1.0, self._boost_state.boost * self.cfg.regime_boost_decay)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ewma": self._ewma, "dev_ewma": self._dev_ewma,
            "initialized": self._initialized, "n": self._n,
            "warmup_buf": list(self._warmup_buf),
            "warmup_mean": self._warmup_mean,
            "warmup_std": self._warmup_std,
            "ch_ewma": list(self._ch_ewma), "ch_dev": list(self._ch_dev),
            "ch_init": list(self._ch_init),
            "boost": self._boost_state.boost,  # Phase 1.3: persist boost
            "aqb_buf": list(self._aqb_buf),
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._ewma = sd.get("ewma", 0.0)
        self._dev_ewma = sd.get("dev_ewma", 1.0)
        self._initialized = sd.get("initialized", False)
        self._n = sd.get("n", 0)
        self._warmup_buf = sd.get("warmup_buf", [])
        self._warmup_mean = sd.get("warmup_mean", 0.0)
        self._warmup_std = sd.get("warmup_std", 1.0)
        self._ch_ewma = sd.get("ch_ewma", [])
        self._ch_dev = sd.get("ch_dev", [])
        self._ch_init = sd.get("ch_init", [])
        self._boost_state.boost = sd.get("boost", 1.0)  # Phase 1.3
        self._aqb_buf = deque(sd.get("aqb_buf", []), maxlen=self.cfg.aqb_window)


# ---------------------------------------------------------------------------
# Step 2: StructuralSnapshotStep — Channel 1
# ---------------------------------------------------------------------------

class StructuralSnapshotStep(DetectorStep):
    """Compute Channel 1 structural snapshot from current windowed data.

    Step 2 — inserted immediately after CoreEWMAStep.
    Treats network topology as active transmitter: computes statistical
    properties of the input data stream as an independent information channel.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._snapshot_history: deque = deque(
            maxlen=max(self.cfg.coherence_window * 2, 40))

    def update(self, ctx: StepContext) -> None:
        if not (self.cfg.enable_channel_coherence or self.cfg.enable_cascade_detection):
            ctx.scratch["structural_snapshot"] = None
            ctx.scratch["_structural_snapshot_history"] = self._snapshot_history
            return

        w = list(ctx.bank.get("scalar"))
        n = len(w)
        if n < 4:
            ctx.scratch["structural_snapshot"] = None
            ctx.scratch["_structural_snapshot_history"] = self._snapshot_history
            return

        mean = _mean(w)
        variance = _mean([(x - mean) ** 2 for x in w])
        std = math.sqrt(variance) if variance > 1e-20 else 1e-10

        skewness = (sum((x - mean) ** 3 for x in w) / (n * std ** 3)
                    if std > 1e-10 else 0.0)
        kurtosis = (sum((x - mean) ** 4 for x in w) / (n * std ** 4) - 3
                    if std > 1e-10 else 0.0)

        denom = sum((x - mean) ** 2 for x in w)
        ac1 = (sum((w[i] - mean) * (w[i - 1] - mean) for i in range(1, n)) / denom
               if denom > 1e-20 else 0.0)
        ac2 = (sum((w[i] - mean) * (w[i - 2] - mean) for i in range(2, n)) / denom
               if denom > 1e-20 else 0.0)

        half = n // 2
        if half > 1:
            mu1 = _mean(w[:half])
            mu2 = _mean(w[half:])
            v1 = _mean([(x - mu1) ** 2 for x in w[:half]])
            v2 = _mean([(x - mu2) ** 2 for x in w[half:]])
            stationarity = 1.0 - abs(v2 - v1) / (v1 + v2 + 1e-10)
        else:
            stationarity = 1.0

        snapshot = StructuralSnapshot(
            mean=mean,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            autocorrelation_lag1=ac1,
            autocorrelation_lag2=ac2,
            stationarity_score=max(0.0, min(1.0, stationarity)),
            timestamp=ctx.step,
        )
        self._snapshot_history.append(snapshot)
        ctx.scratch["structural_snapshot"] = snapshot
        ctx.scratch["_structural_snapshot_history"] = self._snapshot_history

    def state_dict(self) -> Dict[str, Any]:
        return {"n_snapshots": len(self._snapshot_history)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass  # in-memory history only


# ---------------------------------------------------------------------------
# Step 3: FrequencyDecompositionStep — Channel 2
# ---------------------------------------------------------------------------

class FrequencyDecompositionStep(DetectorStep):
    """Decompose windowed signal into five frequency band carrier waves.

    Step 3 — inserted immediately after StructuralSnapshotStep.
    Implements Channel 2: broadband multiplexed oscillatory transmission.
    Uses numpy FFT if available; falls back to pure-Python DFT.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._bands_history: deque = deque(
            maxlen=max(self.cfg.coupling_trend_window * 4, 60))

    def _pure_python_bands(self, data: List[float]) -> tuple:
        N = len(data)
        spectrum = []
        for k in range(N // 2 + 1):
            re = sum(data[t] * math.cos(2 * math.pi * k * t / N) for t in range(N))
            im = sum(data[t] * math.sin(2 * math.pi * k * t / N) for t in range(N))
            mag = math.sqrt(re ** 2 + im ** 2)
            phase = math.atan2(-im, re)
            spectrum.append((mag, phase))
        total = len(spectrum)

        def pure_band(lo_frac, hi_frac):
            lo_i = int(lo_frac * total)
            hi_i = int(hi_frac * total)
            if hi_i <= lo_i:
                return 0.0, 0.0
            s = spectrum[lo_i:hi_i]
            avg_mag = sum(x[0] for x in s) / len(s)
            avg_phase = sum(x[1] for x in s) / len(s)
            return avg_mag, avg_phase

        ul_p, ul_ph = pure_band(0.00, 0.05)
        l_p,  l_ph  = pure_band(0.05, 0.15)
        m_p,  m_ph  = pure_band(0.15, 0.40)
        h_p,  h_ph  = pure_band(0.40, 0.70)
        uh_p, uh_ph = pure_band(0.70, 1.00)
        return ul_p, ul_ph, l_p, l_ph, m_p, m_ph, h_p, h_ph, uh_p, uh_ph

    def _compute_bands(self, data: List[float], timestamp: int) -> Optional[FrequencyBands]:
        n = len(data)
        try:
            import numpy as _np_local
            arr = _np_local.array(data, dtype=float)
            fft = _np_local.fft.rfft(arr)
            freqs = _np_local.fft.rfftfreq(n)
            magnitudes = _np_local.abs(fft)
            phases = _np_local.angle(fft)

            def band_stats(lo, hi):
                mask = (freqs >= lo) & (freqs < hi)
                if not _np_local.any(mask):
                    return 0.0, 0.0
                return float(_np_local.mean(magnitudes[mask])), float(_np_local.mean(phases[mask]))

            ul_p, ul_ph = band_stats(0.00, 0.05)
            l_p,  l_ph  = band_stats(0.05, 0.15)
            m_p,  m_ph  = band_stats(0.15, 0.40)
            h_p,  h_ph  = band_stats(0.40, 0.70)
            uh_p, uh_ph = band_stats(0.70, 1.00)
        except ImportError:
            ul_p, ul_ph, l_p, l_ph, m_p, m_ph, h_p, h_ph, uh_p, uh_ph = \
                self._pure_python_bands(data)

        return FrequencyBands(
            ultra_low_power=ul_p, low_power=l_p, mid_power=m_p,
            high_power=h_p, ultra_high_power=uh_p,
            ultra_low_phase=ul_ph, low_phase=l_ph, mid_phase=m_ph,
            high_phase=h_ph, ultra_high_phase=uh_ph,
            timestamp=timestamp,
        )

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.enable_frequency_decomposition:
            ctx.scratch["frequency_bands"] = None
            ctx.scratch["_bands_history"] = self._bands_history
            return

        w = list(ctx.bank.get("scalar"))
        n = len(w)
        if n < self.cfg.min_window_for_fft:
            ctx.scratch["frequency_bands"] = None
            ctx.scratch["_bands_history"] = self._bands_history
            return

        bands = self._compute_bands(w, ctx.step)
        if bands is not None:
            self._bands_history.append(bands)
        ctx.scratch["frequency_bands"] = bands
        ctx.scratch["_bands_history"] = self._bands_history

    def state_dict(self) -> Dict[str, Any]:
        return {"n_bands": len(self._bands_history)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        pass  # in-memory history only


# ---------------------------------------------------------------------------
# Step 4: CUSUMStep
# ---------------------------------------------------------------------------

class CUSUMStep(DetectorStep):
    """Bidirectional CUSUM for persistent mean shift detection.

    Two accumulator pairs (v12.3):

    1. *Adaptive CUSUM* — operates on the EWMA z-score.  Detects large,
       fast mean shifts (point anomalies, collective blocks).  Uses
       ``config.cusum_k`` / ``config.cusum_h`` (defaults: 1.0 / 8.0 in v12.3).

    2. *Non-adaptive drift CUSUM* — operates on z_raw = (v − warmup_mean) /
       warmup_std (the warmup-frozen baseline).  Because the EWMA baseline
       adapts to slow trends, the adaptive z-score stays near zero during slow
       drift; z_raw does not.  Uses fixed k=0.5 / h=20.0.  Fires as
       ``drift_cusum_alert`` — a STRONG signal in the consensus gate.
    """

    # Non-adaptive drift CUSUM parameters (v12.3, not user-configurable).
    # k=0.5 (half the minimum 1σ detectable shift); h=5.0 (fires every ~10–15
    # steps during 2σ+ drift, giving dense coverage across the anomaly region).
    # RESETS after each crossing — drift during ongoing shift re-crosses quickly.
    _DRIFT_K = 0.5
    _DRIFT_H = 5.5

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._s_hi = 0.0
        self._s_lo = 0.0
        # Non-adaptive drift accumulators
        self._d_hi = 0.0
        self._d_lo = 0.0

    def update(self, ctx: StepContext) -> None:
        if ctx.is_warmup:
            return
        z = ctx.scratch.get("z_score", 0.0)
        k = self.cfg.cusum_k
        h = self.cfg.cusum_h

        # Adaptive CUSUM (on EWMA z-score)
        self._s_hi = max(0.0, self._s_hi + z - k)
        self._s_lo = max(0.0, self._s_lo - z - k)
        cusum_alert = (self._s_hi > h) or (self._s_lo > h)
        if cusum_alert:
            self._s_hi = 0.0
            self._s_lo = 0.0
        ctx.scratch["cusum_hi"] = self._s_hi
        ctx.scratch["cusum_lo"] = self._s_lo
        ctx.scratch["cusum_alert"] = cusum_alert

        # Non-adaptive drift CUSUM (on warmup-frozen baseline)
        # Operates on z_raw = (v - warmup_mean) / warmup_std so the EWMA
        # cannot mask slow drift by adapting.  Resets after each crossing so
        # it re-arms quickly during ongoing drift (next crossing in ~5–15 steps).
        # v12.4: use deseasonalized value when available, so periodic signals
        # don't systematically accumulate in the drift CUSUM.  When using the
        # deseasonalized residual, the expected mean is zero, so z_raw = residual / std.
        # The warmup std still provides the correct scale since the deseasonalized
        # residual's variance ≈ noise variance captured during warmup.
        wm = ctx.scratch.get("warmup_mean", 0.0)
        ws = max(ctx.scratch.get("warmup_std", 1.0), 1e-10)
        ds_val = ctx.scratch.get("deseasonalized_value")
        if ds_val is not None:
            # Deseasonalized residual: mean-zero by construction
            z_raw = float(ds_val) / ws
        else:
            v_raw = ctx.value if not isinstance(ctx.value, (list, tuple)) else ctx.value[0]
            z_raw = (float(v_raw) - wm) / ws
        dk = self._DRIFT_K
        dh = self._DRIFT_H
        self._d_hi = max(0.0, self._d_hi + z_raw - dk)
        self._d_lo = max(0.0, self._d_lo - z_raw - dk)
        drift_alert = (self._d_hi > dh) or (self._d_lo > dh)
        if drift_alert:
            self._d_hi = 0.0
            self._d_lo = 0.0
        ctx.scratch["drift_cusum_hi"] = self._d_hi
        ctx.scratch["drift_cusum_lo"] = self._d_lo
        ctx.scratch["drift_cusum_alert"] = drift_alert

        if cusum_alert or drift_alert:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"s_hi": self._s_hi, "s_lo": self._s_lo,
                "d_hi": self._d_hi, "d_lo": self._d_lo}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._s_hi = sd.get("s_hi", 0.0)
        self._s_lo = sd.get("s_lo", 0.0)
        self._d_hi = sd.get("d_hi", 0.0)
        self._d_lo = sd.get("d_lo", 0.0)


# ---------------------------------------------------------------------------
# Step 5: RegimeStep
# ---------------------------------------------------------------------------

class RegimeStep(DetectorStep):
    """Detect regime changes; apply soft alpha boost instead of hard reset (δ).

    Phase 1.3: boost is written to a shared RegimeBoostState object passed at
    construction — no fragile scratch-key side channel.
    """

    def __init__(self, config: SentinelConfig,
                 boost_state: Optional[RegimeBoostState] = None):
        self.cfg = config
        self._boost_state = boost_state or RegimeBoostState()
        self.reset()

    def reset(self):
        self._in_regime = False
        self._regime_count = 0

    def update(self, ctx: StepContext) -> None:
        if ctx.is_warmup:
            ctx.scratch["regime_change"] = False
            return
        z = abs(ctx.scratch.get("z_score", 0.0))
        thresh = self.cfg.regime_threshold
        if z > thresh:
            self._in_regime = True
            self._regime_count += 1
            # Soft boost: write to shared state; CoreEWMAStep reads it (Phase 1.3)
            self._boost_state.boost = self.cfg.regime_alpha_boost
            ctx.scratch["regime_change"] = True
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        else:
            if self._in_regime:
                self._in_regime = False
            ctx.scratch["regime_change"] = False
        ctx.scratch["in_regime"] = self._in_regime

    def state_dict(self) -> Dict[str, Any]:
        return {"in_regime": self._in_regime, "regime_count": self._regime_count}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._in_regime = sd.get("in_regime", False)
        self._regime_count = sd.get("regime_count", 0)


# ---------------------------------------------------------------------------
# Step 6: VarCUSUMStep
# ---------------------------------------------------------------------------

class VarCUSUMStep(DetectorStep):
    """CUSUM on squared deviation — catches volatility explosions."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._s_hi = 0.0
        self._s_lo = 0.0
        self._var_ewma = 1.0
        self._warmed = False
        self._var_baseline = 0.0   # warmup-estimated windowed variance
        self._warmup_var_sum = 0.0
        self._warmup_var_count = 0

    def update(self, ctx: StepContext) -> None:
        # Accumulate structural-snapshot variance during warmup to build a fixed
        # pre-anomaly baseline that never adapts away (unlike dev_ewma).
        if ctx.is_warmup:
            ss = ctx.scratch.get("structural_snapshot")
            if ss is not None:
                self._warmup_var_sum += ss.variance
                self._warmup_var_count += 1
            return
        if not self._warmed:
            self._s_hi = 0.0
            self._s_lo = 0.0
            self._var_ewma = 0.0  # seed from real data, not default 1.0
            self._var_baseline = (
                self._warmup_var_sum / self._warmup_var_count
                if self._warmup_var_count > 0 else 1.0
            )
            self._warmed = True
        z = ctx.scratch.get("z_score", 0.0)
        v2 = z * z  # variance proxy
        self._var_ewma = 0.9 * self._var_ewma + 0.1 * v2
        k = self.cfg.var_cusum_k
        self._s_hi = max(0.0, self._s_hi + v2 - k)
        # Only accumulate _s_lo when variance baseline is established; prevents
        # false "variance drop" alerts on data that has always been constant.
        if self._var_ewma > 1e-4:
            self._s_lo = max(0.0, self._s_lo + k - v2)
        else:
            self._s_lo = 0.0
        cusum_alert = self._s_hi > self.cfg.var_cusum_h or self._s_lo > self.cfg.var_cusum_h

        # Sustained-variance detection: compare the current 64-step windowed
        # variance (from StructuralSnapshot) against the fixed warmup baseline.
        # The 64-step window absorbs isolated spikes (a single 8σ spike raises
        # windowed variance by only ~1x, not ~25x), so this does not trigger on
        # point anomalies but fires throughout a prolonged volatility regime.
        ss = ctx.scratch.get("structural_snapshot")
        baseline = max(self._var_baseline, 1e-4)
        # v12.4: threshold raised from 4.0 → 8.0.  With periodic spikes
        # (e.g. point anomalies every 50 steps), the 64-sample window can
        # contain 1-2 spikes raising windowed variance to ~3-4× baseline.
        # True variance explosions (4× amplitude) produce ~16× variance
        # ratios, well above 8.0.  This eliminates sustained-variance FPs
        # on point-anomaly data without affecting variance-archetype recall.
        sustained_alert = (
            ss is not None
            and ss.variance > 8.0 * baseline
        )

        ctx.scratch["var_cusum_hi"] = self._s_hi
        ctx.scratch["var_cusum_lo"] = self._s_lo
        # v12.4: split into two distinct signals.
        # var_cusum_alert: CUSUM-based, fires on individual z² spikes.
        #   Prone to FP runs after legitimate point anomalies.  Soft signal.
        # sustained_variance_alert: windowed-baseline comparison, fires during
        #   genuine prolonged volatility regimes.  Robust to isolated spikes.
        #   Strong signal (treated separately in AlertReasonsStep).
        ctx.scratch["var_cusum_alert"] = cusum_alert
        ctx.scratch["sustained_variance_alert"] = sustained_alert
        if cusum_alert:
            # Re-arm: reset accumulators so the statistic can detect the next event.
            self._s_hi = 0.0
            self._s_lo = 0.0
        if cusum_alert or sustained_alert:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            "s_hi": self._s_hi,
            "s_lo": self._s_lo,
            "var_ewma": self._var_ewma,
            "warmed": self._warmed,
            "var_baseline": self._var_baseline,
            "warmup_var_sum": self._warmup_var_sum,
            "warmup_var_count": self._warmup_var_count,
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._s_hi = sd.get("s_hi", 0.0)
        self._s_lo = sd.get("s_lo", 0.0)
        self._var_ewma = sd.get("var_ewma", 1.0)
        self._warmed = sd.get("warmed", False)
        self._var_baseline = sd.get("var_baseline", 1.0)
        self._warmup_var_sum = sd.get("warmup_var_sum", 0.0)
        self._warmup_var_count = sd.get("warmup_var_count", 0)


# ---------------------------------------------------------------------------
# Step 7: PageHinkleyStep
# ---------------------------------------------------------------------------

class PageHinkleyStep(DetectorStep):
    """Page-Hinkley test for slow gradual mean drift."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._cum_hi = 0.0
        self._cum_lo = 0.0
        self._m_hi = 0.0
        self._m_lo = 0.0
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        if ctx.is_warmup:
            ctx.scratch["ph_alert"] = False
            return
        v = ctx.current
        mu = ctx.ewma
        delta = self.cfg.ph_delta
        lam = self.cfg.ph_lambda

        self._cum_hi += (v - mu - delta)
        self._cum_lo += (mu - v - delta)
        self._m_hi = max(self._m_hi, self._cum_hi)
        self._m_lo = max(self._m_lo, self._cum_lo)
        self._n += 1

        alert = (self._m_hi - self._cum_hi > lam) or (self._m_lo - self._cum_lo > lam)
        ctx.scratch["ph_alert"] = alert
        ctx.scratch["ph_cum_hi"] = self._cum_hi
        ctx.scratch["ph_cum_lo"] = self._cum_lo
        if alert:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
            # reset to avoid continuous triggering
            self._cum_hi = 0.0
            self._cum_lo = 0.0
            self._m_hi = 0.0
            self._m_lo = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {"cum_hi": self._cum_hi, "cum_lo": self._cum_lo,
                "m_hi": self._m_hi, "m_lo": self._m_lo, "n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._cum_hi = sd.get("cum_hi", 0.0)
        self._cum_lo = sd.get("cum_lo", 0.0)
        self._m_hi = sd.get("m_hi", 0.0)
        self._m_lo = sd.get("m_lo", 0.0)
        self._n = sd.get("n", 0)
