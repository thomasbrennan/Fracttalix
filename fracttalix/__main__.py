# fracttalix/__main__.py
# CLI entry point: python -m fracttalix

import argparse
import csv

from fracttalix import __version__
from fracttalix.config import SentinelConfig
from fracttalix.detector import SentinelDetector


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fracttalix",
        description=f"Fracttalix Sentinel v{__version__} streaming anomaly detector",
    )
    parser.add_argument("--file", "-f", help="CSV file path (reads first column)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--serve", action="store_true", help="Start HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    cfg = SentinelConfig(alpha=args.alpha, multiplier=args.multiplier,
                         warmup_periods=args.warmup)

    if args.benchmark:
        from benchmark import SentinelBenchmark
        bench = SentinelBenchmark()
        bench.run_suite(config=cfg)
        return

    if args.serve:
        from fracttalix.extras.server import SentinelServer
        server = SentinelServer(host=args.host, port=args.port, config=cfg)
        print(f"Sentinel v{__version__} server on {args.host}:{args.port}")
        server.run()
        return

    if args.file:
        det = SentinelDetector(config=cfg)
        with open(args.file, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    v = float(row[0])
                except (ValueError, IndexError):
                    continue
                r = det.update_and_check(v)
                if r.get("alert"):
                    print(f"[ALERT] step={r['step']} value={v:.4f} "
                          f"z={r.get('z_score', 0):.2f} reasons={r.get('alert_reasons', [])}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
