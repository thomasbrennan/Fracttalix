Fracttalix v2.5 - Exploratory Fractal and Rhythmic Metric Tool
# Author: Thomas G. Brennan
# Co-developed with Grok (xAI)
# Implements five standard diagnostic metrics for time series exploration
# plus a synthetic stress-test suite for systematic validation
# This work is dedicated to the public domain under CC0 1.0 Universal.
# No rights reserved. Use, modify, and distribute freely.
# https://creativecommons.org/publicdomain/zero/1.0/

# README-style notes:
# - Lightweight CLI tool for exploratory time series analysis.
# - Requires: numpy, pandas, scipy, matplotlib (optional for stress-test plots).
# - Recommended minimum series length: 100 points for reliable estimates.
# - All metrics are basic implementations of established methods.
# - Strongly recommended: compare results with dedicated libraries (e.g., antropy, nolitsa, hurst).
# - Higuchi FD: k_max=30 for stability; corrected standard formula D = 1 - slope.
#   Note: Higuchi is sensitive to linear trends—detrend series (e.g., subtract linear fit) for more accurate estimates on non-stationary data.
# - Resilience: 5–6 point recovery window after drops >1.5σ (heuristic).
# - Stress test: evaluates metric discrimination on known synthetic processes.
# - Run stress test with: python fracttalix_v2.5.py --stress-test

import numpy as np
import pandas as pd
from scipy.stats import entropy
import json
import time
import argparse
import sys
import matplotlib.pyplot as plt  # Optional for stress-test example plots

__version__ = "2.5"

# --- Core Metrics ---
def higuchi_fd(x, k_max=30):
    """Higuchi fractal dimension (corrected standard formula: D = 1 - slope)."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 100:
        return np.nan

    L = []
    k_vals = np.arange(1, k_max + 1)
    for k in k_vals:
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            diffs = np.diff(x[idx])
            Lmk = np.sum(np.abs(diffs)) * (N - 1) / ((len(idx) - 1) * k)
            Lk.append(Lmk)
        if Lk:
            L.append(np.log(np.mean(Lk)))
    if len(L) < 2:
        return np.nan
    coeffs = np.polyfit(np.log(k_vals[:len(L)]), L, 1)
    return 1 - coeffs[0]  # Corrected standard Higuchi formula

def hurst_rs(x):
    """Basic rescaled-range (R/S) Hurst exponent estimator."""
    x = np.asarray(x, dtype=float)
    if len(x) < 100:
        return np.nan
    cumdev = np.cumsum(x - x.mean())
    R = cumdev.max() - cumdev.min()
    S = np.std(x)
    if S == 0:
        return np.nan
    return np.log(R / S) / np.log(len(x))

def transfer_entropy_self(x, lag=1, bins=12):
    """Univariate (self) transfer entropy with reduced bins for better stability."""
    if len(x) <= lag + 20:
        return np.nan
    past = x[:-lag]
    future = x[lag:]
    joint, _, _ = np.histogram2d(past, future, bins=bins, density=True)
    joint += 1e-12
    p_past = joint.sum(axis=1, keepdims=True)
    p_future = joint.sum(axis=0, keepdims=True)
    te = np.sum(joint * np.log2(joint / (p_past * p_future + 1e-12)))
    return float(te) if te > 0 else 0.0

def integrated_info_approx(x):
    """Simple partition-based approximation of integrated information (Φ)."""
    if len(x) < 100:
        return np.nan
    half = len(x) // 2
    a, b = x[:half], x[half:]
    ha = entropy(np.histogram(a, bins=10, density=True)[0] + 1e-12)
    hb = entropy(np.histogram(b, bins=10, density=True)[0] + 1e-12)
    hab = entropy(np.histogram2d(a, b, bins=10, density=True)[0].ravel() + 1e-12)
    return ha + hb - hab

def resilience(x):
    """Heuristic resilience: average recovery after significant drops."""
    if len(x) < 100:
        return np.nan
    diffs = np.diff(x)
    std_diff = np.std(diffs)
    if std_diff == 0:
        return np.nan
    drops = np.where(diffs < -1.5 * std_diff)[0]
    if len(drops) == 0:
        return 0.0
    recoveries = []
    for d in drops:
        end = min(d + 6, len(diffs))
        if end > d + 1:
            recoveries.append(np.mean(diffs[d + 1:end]))
    return float(np.mean(recoveries)) if recoveries else 0.0

# --- Analysis Function ---
def analyze(series, name="data"):
    s = np.asarray(series, dtype=float)
    if s.size < 100:
        return {"error": "series too short (<100 points recommended for reliable estimates)"}
    m = {
        "D": higuchi_fd(s),
        "H": hurst_rs(s),
        "T": transfer_entropy_self(s),
        "Φ": integrated_info_approx(s),
        "R": resilience(s),
    }
    m["name"] = name
    m["timestamp"] = int(time.time())
    m["note": "Exploratory values only; validate against specialized libraries (antropy, nolitsa, hurst)"
    return m

# --- Synthetic Stress Test Suite ---
np.random.seed(42)  # Ensure reproducibility

def generate_synthetic_series(length: int, series_type: str = 'white'):
    """Generate synthetic time series of known dynamical class."""
    if series_type == 'white':
        return np.random.randn(length)
    elif series_type == 'persistent':
        return np.cumsum(np.random.randn(length))
    elif series_type == 'periodic':
        t = np.linspace(0, 8 * np.pi, length)
        return np.sin(t) + 0.2 * np.random.randn(length)
    elif series_type == 'chaotic':
        x = np.zeros(length)
        x[0] = 0.5
        r = 3.99
        for i in range(1, length):
            x[i] = r * x[i-1] * (1 - x[i-1])
        x = (x - x.mean()) / x.std()
        return x
    elif series_type == 'pink':
        freq = np.fft.rfftfreq(length)
        freq[0] = freq[1] if len(freq) > 1 and freq[1] != 0 else 1.0
        amplitudes = 1 / np.sqrt(np.abs(freq))
        phase = np.random.uniform(0, 2*np.pi, len(freq))
        spectrum = amplitudes * np.exp(1j * phase)
        series = np.fft.irfft(spectrum, n=length)
        return (series - series.mean()) / series.std()
    else:
        raise ValueError(f"Unknown series_type: {series_type}")

def stress_test_fractal_rhythm(
    num_per_type: int = 50,
    lengths: tuple = (100, 500, 2000),
    plot_examples: bool = False,
    save_csv: str = None
) -> pd.DataFrame:
    """Systematically evaluate Fracttalix metrics on synthetic series with known properties."""
    types = ['white', 'persistent', 'periodic', 'chaotic', 'pink']
    results = []

    for length in lengths:
        for typ in types:
            for i in range(num_per_type):
                series = generate_synthetic_series(length, typ)
                metrics = analyze(series, name=f"{typ}_{i}_len{length}")
                if "error" not in metrics:
                    metrics.update({"type": typ, "length": length, "replicate": i})
                    results.append(metrics)

    df = pd.DataFrame(results)

    print("=== Fracttalix Synthetic Stress Test Summary ===")
    print(f"Total valid runs: {len(df)}")
    print("\nMean metrics by series type:")
    print(df.groupby('type')[['D', 'H', 'T', 'Φ', 'R']].mean().round(4))
    print("\nStandard deviation by type:")
    print(df.groupby('type')[['D', 'H', 'T', 'Φ', 'R']].std().round(4))

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"\nResults saved to {save_csv}")

    if plot_examples:
        try:
            fig, axs = plt.subplots(len(types), 1, figsize=(10, 2*len(types)))
            for ax, typ in zip(axs, types):
                example = generate_synthetic_series(500, typ)
                ax.plot(example[:500])
                ax.set_title(f"Example: {typ.capitalize()} Series")
                ax.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting failed (matplotlib optional): {e}")

    return df

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(description=f"Fracttalix v{__version__} – Exploratory time series metrics")
    parser.add_argument("file", nargs="?", help="CSV file path (optional when using stress test)")
    parser.add_argument("--stress-test", action="store_true", help="Run synthetic stress test")
    parser.add_argument("--num-per-type", type=int, default=50, help="Replicates per type/length in stress test (default: 50)")
    parser.add_argument("--lengths", nargs="+", type=int, default=[100, 500, 2000], help="Series lengths for stress test")
    parser.add_argument("--no-plots", action="store_true", help="Disable example plots in stress test")
    parser.add_argument("--save-csv", type=str, help="Save stress-test results to CSV file")
    parser.add_argument("--column", default=None, help="Column name or index for file analysis")
    parser.add_argument("--output", default="pretty", choices=["json", "pretty"], help="Output format for file analysis")
    args = parser.parse_args()

    if args.stress_test:
        stress_test_fractal_rhythm(
            num_per_type=args.num_per_type,
            lengths=tuple(args.lengths),
            plot_examples=not args.no_plots,
            save_csv=args.save_csv
        )
        return

    if not args.file:
        print("Error: Provide a CSV file or use --stress-test")
        sys.exit(1)

    try:
        df_csv = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    try:
        if args.column is None:
            series = pd.to_numeric(df_csv.iloc[:, 0], errors="coerce").dropna().values
            name = df_csv.columns[0]
        else:
            col = int(args.column) if args.column.isdigit() else args.column
            series = pd.to_numeric(df_csv[col], errors="coerce").dropna().values
            name = str(col)
    except Exception as e:
        print(f"Error reading column: {e}")
        sys.exit(1)

    result = analyze(series, name)

    if "error" in result:
        print(result["error"])
    elif args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"Series: {result['name']} (length: {len(series)})")
        print("Exploratory metrics:")
        for k, v in result.items():
            if k not in ["name", "timestamp", "note", "error"]:
                if np.isnan(v):
                    print(f"  {k}: NaN (insufficient data or unstable estimate)")
                else:
                    print(f"  {k}: {v:.4f}")
        print(f"Note: {result.get('note', '')}")

if __name__ == "__main__":
    main()
