"""Execution-speed benchmark: ``lingam.DirectLiNGAM`` vs ``ruslingam.DirectLiNGAM``.

Runs ``fit()`` for both implementations on synthetic linear non-Gaussian SEM data
across a grid of sample/feature sizes and both ``adaptive_lasso`` settings, then
prints a table of wall-clock timings and the Rust speed-up factor.

Usage
-----
    python benchmarks/bench_direct_lingam.py                     # default grid
    python benchmarks/bench_direct_lingam.py --repeat 7 --sizes 500x5 2000x10 5000x20
    python benchmarks/bench_direct_lingam.py --adaptive-lasso true
    python benchmarks/bench_direct_lingam.py --json out.json     # also dump raw results

Requires ``lingam`` and ``scikit-learn`` (the reference implementation); exits with
a message if they are not importable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np

try:
    import lingam
except ImportError:  # pragma: no cover - benchmark helper
    sys.exit("benchmark needs the reference `lingam` package: pip install lingam scikit-learn")

import ruslingam


def make_sem(seed: int, n: int, p: int, density: float = 0.4) -> np.ndarray:
    """A random lower-triangular linear SEM with cubed-uniform (sub-Gaussian) noise.

    Mirrors ``tests/test_parity.py`` so timings run on the same style of data the
    parity suite validates.
    """
    rng = np.random.default_rng(seed)
    B = np.zeros((p, p))
    for i in range(p):
        for j in range(i):
            if rng.random() < density:
                B[i, j] = rng.uniform(-2.5, 2.5)
    X = np.zeros((n, p))
    for i in range(p):
        e = rng.uniform(-1.0, 1.0, n) ** 3
        X[:, i] = e + X @ B[i]
    perm = rng.permutation(p)
    return X[:, perm]


def time_fit(cls, X: np.ndarray, adaptive_lasso: bool, repeat: int) -> list[float]:
    """Return ``repeat`` wall-clock times (seconds) for ``cls(...).fit(X)``."""
    times = []
    for _ in range(repeat):
        model = cls(adaptive_lasso=adaptive_lasso)
        t0 = time.perf_counter()
        model.fit(X)
        times.append(time.perf_counter() - t0)
    return times


def parse_sizes(tokens: list[str]) -> list[tuple[int, int]]:
    out = []
    for tok in tokens:
        try:
            n_str, p_str = tok.lower().split("x")
            out.append((int(n_str), int(p_str)))
        except ValueError:
            raise SystemExit(f"bad --sizes token {tok!r}; expected e.g. 2000x10")
    return out


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
        default=["200x5", "500x10", "1000x20", "2000x50", "5000x100"],
        metavar="NxP",
        help="sample x feature grid, e.g. 2000x10 (default: a 5-point grid)",
    )
    ap.add_argument("--repeat", type=int, default=5, help="timed runs per config (default: 5)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed warm-up runs per config (default: 1)")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed for the SEM data")
    ap.add_argument(
        "--adaptive-lasso",
        choices=["false", "true", "both"],
        default="both",
        help="which adaptive_lasso setting(s) to benchmark (default: both)",
    )
    ap.add_argument("--json", metavar="PATH", help="also write raw timings as JSON to PATH")
    args = ap.parse_args(argv)

    sizes = parse_sizes(args.sizes)
    if args.adaptive_lasso == "both":
        al_settings = [False, True]
    else:
        al_settings = [args.adaptive_lasso == "true"]

    print(f"lingam    {lingam.__version__}")
    print(f"ruslingam {getattr(ruslingam, '__version__', '?')}")
    print(f"numpy     {np.__version__}")
    print(f"repeat={args.repeat}  warmup={args.warmup}  seed={args.seed}")
    print()

    header = (
        f"{'samples':>8} {'feats':>5} {'adapt_lasso':>11} | "
        f"{'lingam (best)':>14} {'ruslingam (best)':>16} | {'speedup':>8}  {'order match':>11}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for (n, p) in sizes:
        X = make_sem(args.seed, n, p)
        for al in al_settings:
            # warm-up (JIT-free, but stabilises caches / page faults / BLAS threads)
            for _ in range(max(0, args.warmup)):
                lingam.DirectLiNGAM(adaptive_lasso=al).fit(X)
                ruslingam.DirectLiNGAM(adaptive_lasso=al).fit(X)

            ref_times = time_fit(lingam.DirectLiNGAM, X, al, args.repeat)
            rus_times = time_fit(ruslingam.DirectLiNGAM, X, al, args.repeat)

            ref_order = lingam.DirectLiNGAM(adaptive_lasso=al).fit(X).causal_order_
            rus_order = ruslingam.DirectLiNGAM(adaptive_lasso=al).fit(X).causal_order_
            match = list(map(int, ref_order)) == list(map(int, rus_order))

            ref_best, rus_best = min(ref_times), min(rus_times)
            speedup = ref_best / rus_best if rus_best else float("inf")

            print(
                f"{n:>8} {p:>5} {str(al):>11} | "
                f"{fmt(ref_best):>14} {fmt(rus_best):>16} | {speedup:>7.1f}x  {('yes' if match else 'NO'):>11}"
            )

            records.append(
                {
                    "samples": n,
                    "features": p,
                    "adaptive_lasso": al,
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
                    "causal_order_match": match,
                }
            )

    speedups = [r["speedup_best"] for r in records]
    print("-" * len(header))
    print(
        f"speedup (best-of-{args.repeat}): "
        f"min {min(speedups):.1f}x   median {statistics.median(speedups):.1f}x   max {max(speedups):.1f}x"
    )
    if not all(r["causal_order_match"] for r in records):
        print("WARNING: causal_order_ disagreed on at least one config (see 'order match' column)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "lingam_version": lingam.__version__,
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
