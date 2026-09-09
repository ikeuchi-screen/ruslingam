"""Execution-speed benchmark: ``lingam.hsic.hsic_test_gamma`` vs
``ruslingam.hsic_test_gamma``.

Both compute the gamma-approximation HSIC independence test with the median-distance
bandwidth (``bw_method="mdbs"``). This script runs the call for both implementations
across a grid of sample sizes and both dependence regimes (an independent pair and a
dependent pair), then prints a table of wall-clock timings and the Rust speed-up
factor.

Usage
-----
    python benchmarks/bench_hsic.py                          # default grid
    python benchmarks/bench_hsic.py --repeat 15 --sizes 200 500 1000 4000
    python benchmarks/bench_hsic.py --relation independent
    python benchmarks/bench_hsic.py --json out.json          # also dump raw results

Requires the reference ``lingam`` package; exits with a message if it is not
importable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np

try:
    from lingam.hsic import hsic_test_gamma as hsic_ref
except ImportError:  # pragma: no cover - benchmark helper
    sys.exit("benchmark needs the reference `lingam` package: pip install lingam")

import ruslingam

hsic_rus = ruslingam.hsic_test_gamma


def make_pair(seed: int, n: int, relation: str) -> tuple[np.ndarray, np.ndarray]:
    """A pair of 1-D non-Gaussian signals.

    ``independent`` : X and Y are drawn independently (cubed uniform noise).
    ``dependent``   : Y is a nonlinear function of X plus noise.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n) ** 3
    if relation == "independent":
        y = rng.uniform(-1.0, 1.0, n) ** 3
    else:
        y = np.tanh(2.0 * x) + 0.3 * (rng.uniform(-1.0, 1.0, n) ** 3)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def time_call(fn, x: np.ndarray, y: np.ndarray, repeat: int) -> list[float]:
    """Return ``repeat`` wall-clock times (seconds) for ``fn(x, y)``."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(x, y)
        times.append(time.perf_counter() - t0)
    return times


def fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:7.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:7.2f} ms"
    return f"{seconds:7.3f} s "


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000, 2000, 4000],
        metavar="N",
        help="sample sizes to benchmark (default: 100 200 500 1000 2000 4000)",
    )
    ap.add_argument("--repeat", type=int, default=10, help="timed runs per config (default: 10)")
    ap.add_argument("--warmup", type=int, default=2, help="untimed warm-up runs per config (default: 2)")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed for the signals")
    ap.add_argument(
        "--relation",
        choices=["independent", "dependent", "both"],
        default="both",
        help="which dependence regime(s) to benchmark (default: both)",
    )
    ap.add_argument("--json", metavar="PATH", help="also write raw timings as JSON to PATH")
    args = ap.parse_args(argv)

    relations = ["independent", "dependent"] if args.relation == "both" else [args.relation]

    print(f"lingam    {__import__('lingam').__version__}")
    print(f"ruslingam {getattr(ruslingam, '__version__', '?')}")
    print(f"numpy     {np.__version__}")
    print(f"repeat={args.repeat}  warmup={args.warmup}  seed={args.seed}")
    print()

    header = (
        f"{'samples':>8} {'relation':>12} | "
        f"{'lingam (best)':>14} {'ruslingam (best)':>16} | {'speedup':>8}  {'stat match':>10}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for n in args.sizes:
        for rel in relations:
            x, y = make_pair(args.seed, n, rel)

            for _ in range(max(0, args.warmup)):
                hsic_ref(x, y)
                hsic_rus(x, y)

            stat_ref, p_ref = hsic_ref(x, y)
            stat_rus, p_rus = hsic_rus(x, y)
            stat_match = np.isclose(stat_ref, stat_rus, rtol=1e-6, atol=1e-9) and np.isclose(
                p_ref, p_rus, rtol=1e-5, atol=1e-12
            )

            ref_times = time_call(hsic_ref, x, y, args.repeat)
            rus_times = time_call(hsic_rus, x, y, args.repeat)

            ref_best, rus_best = min(ref_times), min(rus_times)
            speedup = ref_best / rus_best if rus_best else float("inf")

            print(
                f"{n:>8} {rel:>12} | "
                f"{fmt(ref_best):>14} {fmt(rus_best):>16} | {speedup:>7.1f}x  "
                f"{('yes' if stat_match else 'NO'):>10}"
            )

            records.append(
                {
                    "samples": n,
                    "relation": rel,
                    "lingam": {
                        "best": ref_best,
                        "mean": statistics.fmean(ref_times),
                        "stdev": statistics.pstdev(ref_times),
                        "all": ref_times,
                    },
                    "ruslingam": {
                        "best": rus_best,
                        "mean": statistics.fmean(rus_times),
                        "stdev": statistics.pstdev(rus_times),
                        "all": rus_times,
                    },
                    "speedup_best": speedup,
                    "test_stat": {"lingam": float(stat_ref), "ruslingam": float(stat_rus)},
                    "p_value": {"lingam": float(p_ref), "ruslingam": float(p_rus)},
                    "stat_match": bool(stat_match),
                }
            )

    speedups = [r["speedup_best"] for r in records]
    print("-" * len(header))
    print(
        f"speedup (best-of-{args.repeat}): "
        f"min {min(speedups):.1f}x   median {statistics.median(speedups):.1f}x   max {max(speedups):.1f}x"
    )
    if not all(r["stat_match"] for r in records):
        print("WARNING: test_stat / p_value disagreed on at least one config (see 'stat match' column)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "lingam_version": __import__("lingam").__version__,
                    "repeat": args.repeat,
                    "warmup": args.warmup,
                    "seed": args.seed,
                    "results": records,
                },
                fh,
                indent=2,
            )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
