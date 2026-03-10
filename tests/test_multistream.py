# tests/test_multistream.py
# Tests for MultiStreamSentinel

import threading

from fracttalix import MultiStreamSentinel, SentinelConfig, SentinelDetector


class TestMultiStreamBasic:
    def test_instantiate_default(self):
        ms = MultiStreamSentinel()
        assert ms is not None

    def test_instantiate_with_config(self):
        cfg = SentinelConfig.fast()
        ms = MultiStreamSentinel(config=cfg)
        assert ms.config is cfg

    def test_update_creates_stream_on_first_call(self):
        ms = MultiStreamSentinel()
        assert "stream_a" not in ms._streams
        ms.update("stream_a", 1.0)
        assert "stream_a" in ms._streams

    def test_update_returns_dict(self):
        ms = MultiStreamSentinel()
        result = ms.update("s1", 1.0)
        assert isinstance(result, dict)

    def test_update_returns_result_with_alert(self):
        ms = MultiStreamSentinel()
        result = ms.update("s1", 1.0)
        assert "alert" in result

    def test_two_streams_are_independent(self):
        """Two streams should have different internal state after different inputs."""
        ms = MultiStreamSentinel()
        # Feed stream a normal values, stream b different values
        for _ in range(40):
            ms.update("stream_a", 1.0)
        for _ in range(40):
            ms.update("stream_b", 5.0)
        det_a = ms.get_detector("stream_a")
        det_b = ms.get_detector("stream_b")
        # EWMA should differ between the two streams
        r_a = det_a.update_and_check(1.0)
        r_b = det_b.update_and_check(5.0)
        assert r_a["ewma"] != r_b["ewma"]

    def test_list_streams(self):
        ms = MultiStreamSentinel()
        ms.update("s1", 1.0)
        ms.update("s2", 2.0)
        streams = ms.list_streams()
        assert "s1" in streams
        assert "s2" in streams

    def test_get_detector_returns_sentinel_detector(self):
        ms = MultiStreamSentinel()
        ms.update("s1", 1.0)
        det = ms.get_detector("s1")
        assert isinstance(det, SentinelDetector)

    def test_get_detector_unknown_returns_none(self):
        ms = MultiStreamSentinel()
        det = ms.get_detector("nonexistent")
        assert det is None

    def test_reset_stream(self):
        ms = MultiStreamSentinel()
        for _ in range(10):
            ms.update("s1", 1.0)
        result = ms.reset_stream("s1")
        assert result is True
        det = ms.get_detector("s1")
        assert det._n == 0

    def test_reset_stream_unknown_returns_false(self):
        ms = MultiStreamSentinel()
        assert ms.reset_stream("nonexistent") is False

    def test_delete_stream(self):
        ms = MultiStreamSentinel()
        ms.update("s1", 1.0)
        result = ms.delete_stream("s1")
        assert result is True
        assert ms.get_detector("s1") is None

    def test_delete_stream_unknown_returns_false(self):
        ms = MultiStreamSentinel()
        assert ms.delete_stream("nonexistent") is False


class TestMultiStreamCorrelations:
    def test_cross_stream_correlations_returns_dict(self):
        ms = MultiStreamSentinel()
        result = ms.cross_stream_correlations()
        assert isinstance(result, dict)

    def test_cross_stream_correlations_single_stream_empty(self):
        ms = MultiStreamSentinel()
        ms.update("s1", 1.0)
        result = ms.cross_stream_correlations()
        assert result == {}

    def test_cross_stream_correlations_two_streams(self):
        """With two warmed streams, correlations should return at least one key."""
        ms = MultiStreamSentinel()
        for i in range(70):
            ms.update("s1", float(i % 5))
            ms.update("s2", float(i % 5) + 0.5)
        result = ms.cross_stream_correlations(window=30)
        # Should have one correlation key "s1:s2"
        assert "s1:s2" in result
        corr = result["s1:s2"]
        assert -1.0 <= corr <= 1.0


class TestMultiStreamPersistence:
    def test_save_all_returns_str(self):
        ms = MultiStreamSentinel()
        ms.update("s1", 1.0)
        state = ms.save_all()
        assert isinstance(state, str)

    def test_load_all_round_trip(self):
        ms = MultiStreamSentinel()
        for i in range(10):
            ms.update("s1", float(i))
            ms.update("s2", float(i) * 2)
        n_s1 = ms.get_detector("s1")._n
        n_s2 = ms.get_detector("s2")._n
        state = ms.save_all()

        # Load into fresh instance
        ms2 = MultiStreamSentinel()
        ms2.load_all(state)
        assert ms2.get_detector("s1")._n == n_s1
        assert ms2.get_detector("s2")._n == n_s2

    def test_save_all_produces_valid_json(self):
        import json
        ms = MultiStreamSentinel()
        ms.update("s1", 1.0)
        state = ms.save_all()
        parsed = json.loads(state)
        assert "s1" in parsed


class TestMultiStreamThreadSafety:
    def test_concurrent_updates_dont_raise(self):
        """Concurrent updates from multiple threads should not raise exceptions."""
        ms = MultiStreamSentinel()
        errors = []

        def worker(stream_id, values):
            try:
                for v in values:
                    ms.update(stream_id, v)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"stream_{i}", [float(j) for j in range(20)]))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_creates_all_streams(self):
        """All streams created concurrently should exist."""
        ms = MultiStreamSentinel()

        def worker(stream_id):
            ms.update(stream_id, 1.0)

        threads = [threading.Thread(target=worker, args=(f"t_{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ms.list_streams()) == 10


class TestMultiStreamRepr:
    def test_repr_is_string(self):
        ms = MultiStreamSentinel()
        assert isinstance(repr(ms), str)
