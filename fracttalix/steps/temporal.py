# fracttalix/steps/temporal.py
# Steps 8-11: STIStep, TPSStep, OscDampStep, CPDStep

import math
from collections import deque
from typing import Any, Dict, Optional

from fracttalix._compat import _mean
from fracttalix.config import SentinelConfig
from fracttalix.steps.base import DetectorStep
from fracttalix.window import StepContext


# ---------------------------------------------------------------------------
# Step 8: STIStep — Shear-Turbulence Index
# ---------------------------------------------------------------------------

class STIStep(DetectorStep):
    """Shear-Turbulence Index — fluid-dynamics anomaly proxy."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._prev = None
        self._diffs: deque = deque(maxlen=self.cfg.sti_window)

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        if self._prev is not None:
            d = abs(v - self._prev)
            self._diffs.append(d)
        self._prev = v

        if len(self._diffs) >= 4:
            mu = _mean(self._diffs)
            sq = [(x - mu) ** 2 for x in self._diffs]
            std = math.sqrt(_mean(sq)) or 1e-10
            sti = mu / std if std > 0 else 0.0
        else:
            sti = 0.0

        ctx.scratch["sti"] = sti
        if not ctx.is_warmup and sti > 2.0:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"prev": self._prev, "diffs": list(self._diffs)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._prev = sd.get("prev")
        self._diffs = deque(sd.get("diffs", []), maxlen=self.cfg.sti_window)


# ---------------------------------------------------------------------------
# Step 9: TPSStep — Temporal Phase Space
# ---------------------------------------------------------------------------

class TPSStep(DetectorStep):
    """Temporal Phase Space reconstruction — detects attractor deformation."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._win: deque = deque(maxlen=self.cfg.tps_window)
        self._ref_radius: Optional[float] = None

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        self._win.append(v)
        if len(self._win) < self.cfg.tps_window:
            ctx.scratch["tps_score"] = 0.0
            return

        w = list(self._win)
        mu = _mean(w)
        # Use pairs (x[i], x[i+1]) as 2D phase space
        radii = [math.sqrt((w[i] - mu) ** 2 + (w[i+1] - mu) ** 2)
                 for i in range(len(w) - 1)]
        r_mean = _mean(radii) if radii else 0.0

        if self._ref_radius is None:
            self._ref_radius = r_mean or 1.0
        score = abs(r_mean - self._ref_radius) / (self._ref_radius + 1e-10)
        ctx.scratch["tps_score"] = score
        if not ctx.is_warmup and score > 0.5:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True
        # Slowly update reference
        self._ref_radius = 0.99 * self._ref_radius + 0.01 * r_mean

    def state_dict(self) -> Dict[str, Any]:
        return {"win": list(self._win), "ref_radius": self._ref_radius}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._win = deque(sd.get("win", []), maxlen=self.cfg.tps_window)
        self._ref_radius = sd.get("ref_radius")


# ---------------------------------------------------------------------------
# Step 10: OscDampStep — Oscillation Damping
# ---------------------------------------------------------------------------

class OscDampStep(DetectorStep):
    """Oscillation damping — detects sudden amplitude shifts."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._amp_buf: deque = deque(maxlen=self.cfg.osc_damp_window)
        self._amp_ewma = 0.0

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        # Phase 2: use osc_damp_window slice, not full bank
        w_all = ctx.bank.get("scalar")
        w = list(w_all)[-self.cfg.osc_damp_window:] if len(w_all) > 0 else []
        if len(w) >= 2:
            amp = max(w) - min(w)
        else:
            amp = 0.0
        self._amp_buf.append(amp)
        alpha = 0.1
        self._amp_ewma = alpha * amp + (1 - alpha) * self._amp_ewma

        if not ctx.is_warmup and self._amp_ewma > 1e-10:
            ratio = amp / self._amp_ewma
            osc_alert = ratio > self.cfg.osc_threshold or ratio < (1.0 / self.cfg.osc_threshold)
        else:
            osc_alert = False

        ctx.scratch["osc_amp"] = amp
        ctx.scratch["osc_alert"] = osc_alert
        if osc_alert:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"amp_buf": list(self._amp_buf), "amp_ewma": self._amp_ewma}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._amp_buf = deque(sd.get("amp_buf", []), maxlen=self.cfg.osc_damp_window)
        self._amp_ewma = sd.get("amp_ewma", 0.0)


# ---------------------------------------------------------------------------
# Step 11: CPDStep — Change Point Detection
# ---------------------------------------------------------------------------

class CPDStep(DetectorStep):
    """Two-window CPD: compare means of recent vs historical windows."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._hist: deque = deque(maxlen=self.cfg.cpd_window * 2)

    def update(self, ctx: StepContext) -> None:
        v = ctx.current
        self._hist.append(v)
        w = self.cfg.cpd_window
        if len(self._hist) < w * 2 or ctx.is_warmup:
            ctx.scratch["cpd_score"] = 0.0
            return

        h = list(self._hist)
        recent = h[-w:]
        historical = h[:w]
        mu_r = _mean(recent)
        mu_h = _mean(historical)
        sq_h = [(x - mu_h) ** 2 for x in historical]
        std_h = math.sqrt(_mean(sq_h)) if sq_h else 1.0
        std_h = max(std_h, 1e-10)
        score = abs(mu_r - mu_h) / std_h
        ctx.scratch["cpd_score"] = score
        if score > self.cfg.cpd_threshold:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"hist": list(self._hist)}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._hist = deque(sd.get("hist", []), maxlen=self.cfg.cpd_window * 2)
