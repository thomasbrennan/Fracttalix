# fracttalix/steps/frequency.py
# Steps 12-15: RPIStep, RFIStep, SSIStep, PEStep

import math
from typing import Any, Dict, List

from fracttalix._compat import _NP, _mean, np
from fracttalix.config import SentinelConfig
from fracttalix.steps.base import DetectorStep
from fracttalix.window import StepContext


# ---------------------------------------------------------------------------
# Step 12: RPIStep — Rhythm Periodicity Index (FRM Axiom 6)
# ---------------------------------------------------------------------------

class RPIStep(DetectorStep):
    """RPI via FFT dominant peak ratio."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        if len(w) < self.cfg.rpi_window or not _NP:
            ctx.scratch["rpi"] = 0.0
            return
        arr = np.array(list(w)[-self.cfg.rpi_window:])
        arr = arr - arr.mean()
        spec = np.abs(np.fft.rfft(arr)) ** 2
        total = spec.sum()
        if total < 1e-12:
            ctx.scratch["rpi"] = 0.0
            return
        rpi = float(spec.max() / total)
        ctx.scratch["rpi"] = rpi
        if not ctx.is_warmup and rpi < self.cfg.rpi_threshold:
            pass  # low RPI is informational, not an alert by itself

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# Step 13: RFIStep — Rhythm Fractal Index (FRM Axiom 8)
# ---------------------------------------------------------------------------

class RFIStep(DetectorStep):
    """RFI: Hurst-like fractal dimension of rhythm irregularity."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._n = 0

    def _hurst(self, ts: List[float]) -> float:
        n = len(ts)
        if n < 8:
            return 0.5
        mu = _mean(ts)
        dev = [x - mu for x in ts]
        # R/S analysis
        cumdev = []
        s = 0.0
        for d in dev:
            s += d
            cumdev.append(s)
        R = max(cumdev) - min(cumdev)
        sq = [(x - mu) ** 2 for x in ts]
        S = math.sqrt(_mean(sq)) or 1e-10
        rs = R / S
        if rs <= 0:
            return 0.5
        return math.log(rs) / math.log(n / 2) if n > 2 else 0.5

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        if len(w) < self.cfg.rfi_window:
            ctx.scratch["rfi"] = 0.0
            ctx.scratch["hurst"] = 0.5
            return
        ts = list(w)[-self.cfg.rfi_window:]
        h = self._hurst(ts)
        rfi = abs(h - 0.5) * 2  # 0=random, 1=perfectly fractal
        ctx.scratch["rfi"] = rfi
        ctx.scratch["hurst"] = h
        if not ctx.is_warmup and rfi > self.cfg.rfi_threshold:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# Step 14: SSIStep — Synchrony Stability Index (FRM Axiom 10) — was RSI (T0-05)
# ---------------------------------------------------------------------------

class SSIStep(DetectorStep):
    """SSI: Kuramoto synchronization proxy via FFT phase coherence.

    Formerly mislabeled 'RSI' in v7.10 (T0-05 fix).  ``rsi`` attribute alias
    preserved for backward compatibility.
    """

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def reset(self):
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        n = self.cfg.rpi_window
        if len(w) < n or not _NP:
            ctx.scratch["ssi"] = 0.0
            ctx.scratch["rsi"] = 0.0  # alias (T0-05)
            return
        arr = np.array(list(w)[-n:])
        arr = arr - arr.mean()
        fft = np.fft.rfft(arr)
        phases = np.angle(fft[1:])  # skip DC
        # Kuramoto order param: |mean(exp(i*phases))|
        order = float(np.abs(np.mean(np.exp(1j * phases))))
        ctx.scratch["ssi"] = order
        ctx.scratch["rsi"] = order  # backward compat alias

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._n = sd.get("n", 0)


# ---------------------------------------------------------------------------
# Step 15: PEStep — Permutation Entropy (FRM Axiom 3)
# ---------------------------------------------------------------------------

class PEStep(DetectorStep):
    """Streaming Permutation Entropy."""

    def __init__(self, config: SentinelConfig):
        self.cfg = config
        self.reset()

    def _ordinal_pattern(self, window: List[float]) -> tuple:
        return tuple(sorted(range(len(window)), key=lambda i: window[i]))

    def _pe(self, data: List[float], m: int) -> float:
        from math import log, factorial
        n = len(data)
        counts: Dict[tuple, int] = {}
        total = 0
        for i in range(n - m + 1):
            pat = self._ordinal_pattern(data[i:i+m])
            counts[pat] = counts.get(pat, 0) + 1
            total += 1
        if total == 0:
            return 0.0
        h = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                h -= p * math.log(p)
        max_h = math.log(math.factorial(m)) if m > 1 else 1.0
        return h / max_h if max_h > 0 else 0.0

    def reset(self):
        self._pe_ewma = 0.5
        self._n = 0

    def update(self, ctx: StepContext) -> None:
        self._n += 1
        w = ctx.bank.get("scalar")
        m = self.cfg.pe_order
        if len(w) < max(self.cfg.pe_window, m + 1):
            ctx.scratch["pe"] = 0.5
            ctx.scratch["pe_baseline"] = self._pe_ewma
            return

        data = list(w)[-self.cfg.pe_window:]
        pe = self._pe(data, m)
        ctx.scratch["pe"] = pe

        # Rolling PE baseline
        alpha = 0.05
        self._pe_ewma = alpha * pe + (1 - alpha) * self._pe_ewma
        ctx.scratch["pe_baseline"] = self._pe_ewma

        # Alert on contextual deviation (not absolute low PE — fixes v7.11 false alarm)
        dev = abs(pe - self._pe_ewma)
        if not ctx.is_warmup and dev > self.cfg.pe_threshold:
            ctx.scratch["alert"] = True
            ctx.scratch["anomaly"] = True

    def state_dict(self) -> Dict[str, Any]:
        return {"pe_ewma": self._pe_ewma, "n": self._n}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self._pe_ewma = sd.get("pe_ewma", 0.5)
        self._n = sd.get("n", 0)
