import numpy as np
import pandas as pd
from scipy.stats import linregress
import argparse
import json
import sys

def hurst_rs(ts):
    """Hurst exponent via R/S analysis"""
    ts = np.asarray(ts)
    if len(ts) < 20:
        print("Warning: Series too short for reliable Hurst (<20 points)")
        return np.nan
    cumdev = np.cumsum(ts - np.mean(ts))
    R = np.ptp(cumdev)  # range
    S = np.std(ts)
    if S == 0:
        return np.nan
    return np.log(R / S) / np.log(len(ts))

def higuchi_fd(ts, k_max=None):
    """Higuchi fractal dimension"""
    ts = np.asarray(ts)
    N = len(ts)
    if N < 20:
        print("Warning: Series too short for reliable Higuchi (<20 points)")
        return np.nan
    if k_max is None:
        k_max = min(10, N // 4)
    L = []
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            diffs = np.diff(ts[idx])
            Lmk = np.sum(np.abs(diffs)) * (N - 1) / ((len(idx) - 1) * k)
            Lk.append(Lmk)
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return np.nan
    log_L = np.log(L)
    log_k = np.log(np.arange(1, len(L) + 1))
    slope = linregress(log_k, log_L).slope
    return -slope  # Higuchi FD = -slope (1 < D < 2)

# Placeholder for other metrics - replace with real impls if desired
def dfa(ts):
    print("DFA placeholder - implement proper detrended fluctuation")
    return 0.5

def sample_entropy(ts):
    print("Sample entropy placeholder")
    return 0.1

def petrosian_fd(ts):
    print("Petrosian FD placeholder")
    return 1.5

def main():
    parser = argparse.ArgumentParser(description="Fracttalix v2.6 - Exploratory fractal/rhythmic metrics")
    parser.add_argument('file', nargs='?', help='CSV file (single column or with header)')
    parser.add_argument('--col', type=int, default=0, help='Column index to use (0-based, default 0)')
    parser.add_argument('--plot', action='store_true', help='Plot the time series')
    parser.add_argument('--json', action='store_true', help='Output metrics as JSON')
    args = parser.parse_args()

    if args.file:
        try:
            df = pd.read_csv(args.file)
            if args.col >= len(df.columns):
                print(f"Error: Column {args.col} not found")
                sys.exit(1)
            ts = df.iloc[:, args.col].dropna().values
        except Exception as e:
            print(f"Error reading file: {e}")
            sys.exit(1)
    else:
        print("No file provided - using synthetic random walk")
        np.random.seed(42)
        ts = np.cumsum(np.random.randn(1000))

    if len(ts) == 0:
        print("Error: No data")
        sys.exit(1)

    metrics = {
        'Hurst (R/S)': hurst_rs(ts),
        'Higuchi FD': higuchi_fd(ts),
        'DFA exponent': dfa(ts),
        'Sample Entropy': sample_entropy(ts),
        'Petrosian FD': petrosian_fd(ts),
        'Length': len(ts),
    }

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(ts)
            plt.title("Time Series")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.grid(True)
            plt.show()
        except ImportError:
            print("Matplotlib not available - skipping plot")

    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print("\nFracttalix Metrics")
        print("-" * 40)
        for k, v in metrics.items():
            print(f"{k:20}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k:20}: {v}")

if __name__ == "__main__":
    main()
