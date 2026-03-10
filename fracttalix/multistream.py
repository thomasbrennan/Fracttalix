# fracttalix/multistream.py
# MultiStreamSentinel — manages multiple named independent streams.

import json
import math
import threading
from typing import Any, Callable, Dict, List, Optional

from fracttalix.config import SentinelConfig
from fracttalix.detector import SentinelDetector, _legacy_kwargs_to_config


class MultiStreamSentinel:
    """Manage multiple independent named streams, each with its own detector.

    Each stream is lazily initialized on first observation.  Streams share
    a common ``config`` but have fully independent state.
    """

    def __init__(self, config: Optional[SentinelConfig] = None,
                 detector_factory: Optional[Callable] = None,
                 **legacy_kwargs):
        if config is None:
            if legacy_kwargs:
                config = _legacy_kwargs_to_config(legacy_kwargs)
            else:
                config = SentinelConfig()
        self.config = config
        self._factory = detector_factory or (lambda cfg: SentinelDetector(config=cfg))
        self._streams: Dict[str, SentinelDetector] = {}
        self._lock = threading.Lock()

    def update(self, stream_id: str, value) -> Dict[str, Any]:
        """Update a named stream; auto-create detector on first call."""
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = self._factory(self.config)
        return self._streams[stream_id].update_and_check(value)

    async def aupdate(self, stream_id: str, value) -> Dict[str, Any]:
        """Async variant of update."""
        return self.update(stream_id, value)

    def get_detector(self, stream_id: str) -> Optional[SentinelDetector]:
        return self._streams.get(stream_id)

    def list_streams(self) -> List[str]:
        return list(self._streams.keys())

    def reset_stream(self, stream_id: str) -> bool:
        """Reset a named stream; return False if not found."""
        if stream_id in self._streams:
            self._streams[stream_id].reset()
            return True
        return False

    def delete_stream(self, stream_id: str) -> bool:
        """Delete a named stream; return False if not found."""
        with self._lock:
            if stream_id in self._streams:
                self._streams.pop(stream_id)
                return True
        return False

    def status(self, stream_id: str) -> Dict[str, Any]:
        det = self._streams.get(stream_id)
        if det is None:
            return {"error": "stream not found"}
        history = list(det._history)
        alerts = sum(1 for r in history if r.get("alert"))
        return {
            "stream_id": stream_id,
            "n": det._n,
            "alert_count": alerts,
            "last_result": history[-1] if history else None,
        }

    def save_all(self) -> str:
        """Serialize all stream states to JSON string."""
        return json.dumps({
            sid: json.loads(det.save_state())
            for sid, det in self._streams.items()
        })

    def load_all(self, json_str: str) -> None:
        """Restore all stream states from JSON string."""
        data = json.loads(json_str)
        for sid, sd in data.items():
            if sid not in self._streams:
                self._streams[sid] = self._factory(self.config)
            self._streams[sid].load_state(json.dumps(sd))

    def cross_stream_correlations(self, window: int = 50) -> Dict[str, float]:
        """Compute pairwise Pearson correlation of z-scores across named streams.

        Phase 5.2: optional cross-stream coupling detection.
        Returns dict keyed by "stream_a:stream_b" with correlation value [-1, 1].
        Only pairs where both streams have >= window observations are included.

        Parameters
        ----------
        window:
            Number of most-recent z-score observations to use per stream.
        """
        stream_ids = list(self._streams.keys())
        correlations: Dict[str, float] = {}
        if len(stream_ids) < 2:
            return correlations

        # Extract z-score history per stream
        z_series: Dict[str, List[float]] = {}
        for sid in stream_ids:
            det = self._streams[sid]
            hist = list(det._history)
            zs = [r.get("z_score", 0.0) for r in hist if not r.get("warmup", True)]
            if len(zs) >= window:
                z_series[sid] = zs[-window:]

        # Pairwise Pearson
        valid_ids = list(z_series.keys())
        for i in range(len(valid_ids)):
            for j in range(i + 1, len(valid_ids)):
                a, b = valid_ids[i], valid_ids[j]
                ya, yb = z_series[a], z_series[b]
                n = min(len(ya), len(yb))
                ya, yb = ya[-n:], yb[-n:]
                mu_a = sum(ya) / n
                mu_b = sum(yb) / n
                num = sum((ya[k] - mu_a) * (yb[k] - mu_b) for k in range(n))
                den_a = math.sqrt(sum((ya[k] - mu_a) ** 2 for k in range(n)))
                den_b = math.sqrt(sum((yb[k] - mu_b) ** 2 for k in range(n)))
                denom = den_a * den_b
                r = num / denom if denom > 1e-12 else 0.0
                correlations[f"{a}:{b}"] = max(-1.0, min(1.0, r))

        return correlations

    def __repr__(self) -> str:
        return f"MultiStreamSentinel(streams={list(self._streams.keys())})"
