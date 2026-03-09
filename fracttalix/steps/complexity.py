# fracttalix/steps/complexity.py
# Steps 16-20: EWSStep, AQBStep, SeasonalStep, MahalStep, RRSStep

import math
from collections import deque
from typing import Any, Dict, List, Optional

from fracttalix._compat import _NP, _mean, np
from fracttalix.config import SentinelConfig
from fracttalix.steps.base import DetectorStep
from fracttalix.window import StepContext, WindowBank


# ---------------------------------------------------------------------------
# Step 16: EWSStep — Early Warning Signals (FRM Axiom 9) — T0-01 fix
# ---------------------------------------------------------------------------

class EWSStep(DetectorStep):
    """EWS: Critical slowing down via rising variance + AC(1).

    T0-01: Uses bank.get("ews_w") — independent window, not scalar_window.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self._bank_registered = False
        self.reset()

    def reset(self):
        self._var_ewma = 0.0
        self._ac1_ewma = 0.0
        self._ews_score = 0.0
        self._n = 0

    def _ensure_bank(self, bank: WindowBank):
        if not self._bank_registered:
            bank.register("ews_w", self.cfg.ews_window)  # T0-01
            self._bank_registered = True

    def _ac1(self, data: List[float]) -> float:
        n = len(data)
        if n < 3:
            return 0.0
        mu = _mean(data)
        num = sum((data[i] - mu) * (data[i+1] - mu) for i in range(n-1))
        den = sum((x - mu) ** 2 for x in data)
        return (num / den) if den > 1e-12 else 0.0

    def update(self, ctx: StepContext) -> None:
        self._ensure_bank(ctx.bank)
        self._n += 1

        w = list(ctx.bank.get("ews_w"))  # T0-01: independent window
        if len(w) < max(10, self.cfg.ews_window // 2):
            ctx.scratch["ews_score"] = 0.0
            ctx.scratch["ews_regime"] = "stable"
            return

        mu = _mean(w)
        var = _mean([(x - mu) ** 2 for x in w])
        ac1 = self._ac1(w)

        alpha = 0.1
        self._var_ewma = alpha * var + (1 - alpha) * self._var_ewma
        self._ac1_ewma = alpha * ac1 + (1 - alpha) * self._ac1_ewma

        # EWS score: combine rising var + rising AC(1)
        # Both should be trending upward near a critical transition
        var_trend = min(1.0, var / (self._var_ewma + 1e-10) if var > 0 else 0.0)
        ac1_clamped = max(0.0, min(1.0, self._ac1_ewma))
        self._ews_score = 0.5 * var_trend + 0.5 * ac1_clamped

        thresh = self.cfg.ews_threshold
        if self._ews_score > thresh * 1.5:
            regime = "critical"
        elif self._ews_score > thresh:
            regime = "approaching"
        else:
            regime = "stable"

        ctx.scratch["ews_score"] = self._ews_score
        ctx.scratch["ews_regime"] = regime
        ctx.scratch["ews_var"] = var
        ctx.scratch["ews_ac1"] = ac1
        if not ctx.is_warmup and regime == "critical":
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"var_ewma": self._var_ewma, "ac1_ewma": self._ac1_ewma,
                "ews_score": self._ews_score, "n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._var_ewma = sd.get("var_ewma", 0.0)
        self._ac1_ewma = sd.get("ac1_ewma", 0.0)
        self._ews_score = sd.get("ews_score", 0.0)
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# Step 17: AQBStep — Adaptive Quantile Baseline (FRM Axiom 1)
# ---------------------------------------------------------------------------

class AQBStep(DetectorStep):
    """AQB: distribution-free quantile thresholds."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._buf: deque = deque(maxlen=self.cfg.aqb_window)

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        self._buf.append(v)
        if len(self._buf) < 10 or not self.cfg.quantile_threshold_mode:
            ctx.scratch["aqb_lo"] = None
            ctx.scratch["aqb_hi"] = None
            return

        s = sorted(self._buf)
        n = len(s)
        lo_idx = max(0, int(self.cfg.aqb_q_low * n))
        hi_idx = min(n - 1, int(self.cfg.aqb_q_high * n))
        lo = s[lo_idx]
        hi = s[hi_idx]
        ctx.scratch["aqb_lo"] = lo
        ctx.scratch["aqb_hi"] = hi
        if not ctx.is_warmup and (v > hi or v < lo):
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"buf": list(self._buf)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._buf = deque(sd.get("buf", []), maxlen=self.cfg.aqb_window)


# ---------------------------------------------------------------------------
# Step 18: SeasonalStep — Seasonal Periodic Baseline
# ---------------------------------------------------------------------------

class SeasonalStep(DetectorStep):
    """Per-phase EWMA baseline; period auto-detected via FFT."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._period: Optional[int] = self.cfg.seasonal_period if self.cfg.seasonal_period > 0 else None
        self._phase_ewma: Dict[int, float] = {}
        self._phase_dev: Dict[int, float] = {}
        self._detect_buf: List[float] = []
        self._n = 0

    def _detect_period(self) -> Optional[int]:
        if not _NP or len(self._detect_buf) < 32:
            return None
        arr = np.array(self._detect_buf)
        arr = arr - arr.mean()
        spec = np.abs(np.fft.rfft(arr)) ** 2
        freqs = np.fft.rfftfreq(len(arr))
        if len(spec) < 2:
            return None
        spec[0] = 0  # skip DC
        peak_idx = int(np.argmax(spec))
        if peak_idx == 0 or freqs[peak_idx] < 1e-6:
            return None
        period = int(round(1.0 / freqs[peak_idx]))
        return period if 2 <= period <= len(self._detect_buf) // 2 else None

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        self._n += 1

        if self._period is None:
            self._detect_buf.append(v)
            if len(self._detect_buf) >= 64:
                self._period = self._detect_period()
            ctx.scratch["seasonal_err"] = 0.0
            return

        phase = self._n % self._period
        alpha = self.cfg.alpha

        if phase not in self._phase_ewma:
            self._phase_ewma[phase] = v
            self._phase_dev[phase] = 1.0
        else:
            prev = self._phase_ewma[phase]
            self._phase_ewma[phase] = alpha * v + (1 - alpha) * prev
            err = abs(v - prev)
            self._phase_dev[phase] = 0.1 * err + 0.9 * self._phase_dev[phase]

        base = self._phase_ewma[phase]
        dev = self._phase_dev[phase] or 1.0
        err = abs(v - base) / dev
        ctx.scratch["seasonal_err"] = err
        ctx.scratch["seasonal_phase"] = phase
        ctx.scratch["seasonal_period"] = self._period

        if not ctx.is_warmup and err > self.cfg.multiplier:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"period": self._period, "phase_ewma": self._phase_ewma,
                "phase_dev": self._phase_dev, "n": self._n,
                "detect_buf": self._detect_buf}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._period = sd.get("period")
        self._phase_ewma = {int(k): v for k, v in sd.get("phase_ewma", {}).items()}
        self._phase_dev = {int(k): v for k, v in sd.get("phase_dev", {}).items()}
        self._n = sd.get("n", 0)
        self._detect_buf = sd.get("detect_buf", [])


# ---------------------------------------------------------------------------
# Step 19: MahalStep — Mahalanobis distance (multivariate)
# ---------------------------------------------------------------------------

class MahalStep(DetectorStep):
    """Rolling EWMA covariance + Woodbury rank-1 update for Mahalanobis distance."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._cov_inv = None
        self._mean_vec = None
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        if not self.cfg.multivariate or not _NP:
            ctx.scratch["mahal_dist"] = 0.0
            return

        v = ctx.value
        if not isinstance(v, (list, tuple)):
            ctx.scratch["mahal_dist"] = 0.0
            return

        x = np.array([float(xi) for xi in v[:self.cfg.n_channels]])
        self._n += 1
        a = self.cfg.cov_alpha

        if self._mean_vec is None:
            self._mean_vec = x.copy()
            d = self.cfg.n_channels
            self._cov_inv = np.eye(d)
            ctx.scratch["mahal_dist"] = 0.0
            return

        # EWMA mean update
        self._mean_vec = a * x + (1 - a) * self._mean_vec
        diff = x - self._mean_vec

        # Woodbury rank-1 covariance inverse update
        # C_new = (1-a)*C + a * diff @ diff.T
        # Use Sherman-Morrison: (A + uv^T)^{-1} = A^{-1} - (A^{-1}u v^T A^{-1})/(1 + v^T A^{-1} u)
        C_inv = self._cov_inv
        u = diff.reshape(-1, 1)
        C_inv_u = C_inv @ u
        denom = (1.0 - a) + a * float(u.T @ C_inv_u)
        if abs(denom) > 1e-12:
            self._cov_inv = (C_inv - a * (C_inv_u @ C_inv_u.T) / denom) / (1.0 - a)
        else:
            d = self.cfg.n_channels
            self._cov_inv = np.eye(d)

        # Mahalanobis distance
        md = float(math.sqrt(max(0.0, float(diff @ self._cov_inv @ diff))))
        ctx.scratch["mahal_dist"] = md

        thresh = self.cfg.multiplier * math.sqrt(self.cfg.n_channels)
        if not ctx.is_warmup and md > thresh:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean_vec": self._mean_vec.tolist() if self._mean_vec is not None and _NP else None,
            "cov_inv": self._cov_inv.tolist() if self._cov_inv is not None and _NP else None,
            "n": self._n,
        }

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)
        mv = sd.get("mean_vec")
        ci = sd.get("cov_inv")
        if mv is not None and _NP:
            self._mean_vec = np.array(mv)
        if ci is not None and _NP:
            self._cov_inv = np.array(ci)


# ---------------------------------------------------------------------------
# Step 20: RRSStep — Rhythm Resonance Score (FRM Axiom 11)
# ---------------------------------------------------------------------------

class RRSStep(DetectorStep):
    """RRS: harmonic resonance score — ratio of harmonic power to total power."""

    def __init__(self, config: SentinelConfig, regime_step=None):
        self.cfg = config
        self._regime = regime_step
        self.reset()

    def reset(self):
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        n = self.cfg.rpi_window
        if len(w) < n or not _NP:
            ctx.scratch["rrs"] = 0.0
            return

        arr = np.array(list(w)[-n:])
        arr = arr - arr.mean()
        spec = np.abs(np.fft.rfft(arr)) ** 2
        total = spec.sum()
        if total < 1e-12:
            ctx.scratch["rrs"] = 0.0
            return

        # Fundamental: peak frequency; harmonics: 2x, 3x
        peak_idx = int(np.argmax(spec[1:])) + 1
        harm_power = spec[peak_idx]
        for mult in [2, 3]:
            hi = peak_idx * mult
            if hi < len(spec):
                harm_power += spec[hi]
        rrs = float(harm_power / total)
        ctx.scratch["rrs"] = rrs

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)
