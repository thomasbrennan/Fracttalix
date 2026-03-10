# benchmark/__init__.py
# Fracttalix V12 — benchmark subpackage
# Unified evaluation harness for anomaly detection benchmarks.

from benchmark.archetypes import BenchmarkArchetype, generate
from benchmark.metrics import evaluate, run_suite
from benchmark.comparison import compare_baselines
from benchmark.ablation import ablation_study

__all__ = [
    "BenchmarkArchetype",
    "generate",
    "evaluate",
    "run_suite",
    "compare_baselines",
    "ablation_study",
    "SentinelBenchmark",
]


# Convenience alias — unified benchmark interface
class SentinelBenchmark:
    """Unified benchmark interface for Fracttalix V12."""

    def generate(self, archetype, n=1000, seed=42):
        """Generate (data, labels) for the given archetype."""
        return generate(archetype, n=n, seed=seed)

    def evaluate(self, archetype, config=None, n=1000, seed=42):
        """Evaluate Sentinel on one archetype; return metrics dict."""
        return evaluate(archetype, config=config, n=n, seed=seed)

    def run_suite(self, config=None, n=1000, seed=42):
        """Run all 5 archetypes and print formatted table."""
        return run_suite(config=config, n=n, seed=seed)

    def compare_baselines(self, archetype="point", n=1000, seed=42, config=None):
        """Compare Fracttalix against naive and optional baselines."""
        return compare_baselines(archetype=archetype, n=n, seed=seed, config=config)

    def ablation_study(self, archetype="point", n=1000, seed=42, config=None):
        """Run ablation study disabling each step group in turn."""
        return ablation_study(archetype=archetype, n=n, seed=seed, config=config)
