"""Execution-speed benchmark: ``lingam.CAMUV`` vs ``ruslingam.CAMUV``.

Runs ``fit()`` for both implementations on synthetic causal-additive-model data
with unobserved confounders across a grid of sample/feature sizes, then prints a
table of wall-clock timings and the Rust speed-up factor.

CAMUV's combinatorial parent search plus a GAM fit per candidate parent set is
far more expensive per ``fit()`` call than DirectLiNGAM's causal ordering — true
for *both* implementations — so the default grid here is deliberately much
smaller than ``bench_direct_lingam.py``'s.

Usage
-----
    python benchmarks/bench_camuv.py                     # default grid
    python benchmarks/bench_camuv.py --repeat 7 --sizes 200x5 500x8
    python benchmarks/bench_camuv.py --json out.json     # also dump raw results

Requires ``lingam`` (and, transitively, ``pygam``, the reference implementation's
GAM library); exits with a message if they are not importable.
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
    sys.exit("benchmark needs the reference `lingam` package: pip install lingam pygam")

import ruslingam


def make_confounded_sem(seed: int, n: int, p: int, edge_density: float = 0.3) -> np.ndarray:
    """Synthetic data mixing plain direct edges with a pair sharing an
    unobserved latent confounder through two different non-monotonic functions.

    A purely linear shared latent is trivially identified as a direct edge
    instead of flagged as confounded (verified against real ``lingam.CAMUV`` —
    see the ``confounded_data`` fixture in ``tests/conftest.py``), so the last
    two columns always get this non-invertible construction; the rest form a
    random lower-triangular additive chain, mirroring ``bench_direct_lingam.py``'s
    ``make_sem`` in spirit but reusing CAM-UV's own kind of nonlinear structure.
    """
    if p < 3:
        raise ValueError("make_confounded_sem needs p >= 3 (one root plus a confounded pair)")
    rng = np.random.default_rng(seed)

    def noise(scale: float = 1.0) -> np.ndarray:
        return scale * rng.uniform(-1.0, 1.0, n) ** 3

    x = np.zeros((n, p))
    for i in range(p - 2):
        x[:, i] = noise()
        if i > 0 and rng.random() < edge_density:
            j = rng.integers(0, i)
            x[:, i] = 1.5 * x[:, j] + noise()

    latent = rng.uniform(-2.0, 2.0, n)
    x[:, p - 2] = np.sin(2 * latent) + 0.2 * noise()
    x[:, p - 1] = np.cos(3 * latent) + 0.2 * noise()

    perm = rng.permutation(p)
    return x[:, perm]


def time_fit(cls, X: np.ndarray, repeat: int) -> list[float]:
    """Return ``repeat`` wall-clock times (seconds) for ``cls().fit(X)``."""
    times = []
    for _ in range(repeat):
        model = cls()
        t0 = time.perf_counter()
        model.fit(X)
        times.append(time.perf_counter() - t0)
    return times


def adjacency_matrices_match(a: np.ndarray, b: np.ndarray) -> bool:
    """NaN-aware exact equality for CAMUV's discrete (0/1/NaN) adjacency matrix."""
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(np.nan_to_num(a), np.nan_to_num(b)))


def parse_sizes(tokens: list[str]) -> list[tuple[int, int]]:
    out = []
    for tok in tokens:
        try:
            n_str, p_str = tok.lower().split("x")
            out.append((int(n_str), int(p_str)))
        except ValueError:
            raise SystemExit(f"bad --sizes token {tok!r}; expected e.g. 500x8")
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
        default=["100x4", "200x5", "300x6", "500x8"],
        metavar="NxP",
        help="sample x feature grid, e.g. 500x8 (default: a 4-point grid)",
    )
    ap.add_argument("--repeat", type=int, default=5, help="timed runs per config (default: 5)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed warm-up runs per config (default: 1)")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed for the SEM data")
    ap.add_argument("--json", metavar="PATH", help="also write raw timings as JSON to PATH")
    args = ap.parse_args(argv)

    sizes = parse_sizes(args.sizes)

    print(f"lingam    {lingam.__version__}")
    print(f"ruslingam {getattr(ruslingam, '__version__', '?')}")
    print(f"numpy     {np.__version__}")
    print(f"repeat={args.repeat}  warmup={args.warmup}  seed={args.seed}")
    print()

    header = (
        f"{'samples':>8} {'feats':>5} | "
        f"{'lingam (best)':>14} {'ruslingam (best)':>16} | {'speedup':>8}  {'adj. match':>10}"
    )
    print(header)
    print("-" * len(header))

    records = []
    for (n, p) in sizes:
        X = make_confounded_sem(args.seed, n, p)

        for _ in range(max(0, args.warmup)):
            lingam.CAMUV().fit(X)
            ruslingam.CAMUV().fit(X)

        ref_times = time_fit(lingam.CAMUV, X, args.repeat)
        rus_times = time_fit(ruslingam.CAMUV, X, args.repeat)

        ref_adj = lingam.CAMUV().fit(X).adjacency_matrix_
        rus_adj = ruslingam.CAMUV().fit(X).adjacency_matrix_
        match = adjacency_matrices_match(ref_adj, rus_adj)

        ref_best, rus_best = min(ref_times), min(rus_times)
        speedup = ref_best / rus_best if rus_best else float("inf")

        print(
            f"{n:>8} {p:>5} | "
            f"{fmt(ref_best):>14} {fmt(rus_best):>16} | {speedup:>7.1f}x  {('yes' if match else 'NO'):>10}"
        )

        records.append(
            {
                "samples": n,
                "features": p,
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
                "adjacency_matrix_match": match,
            }
        )

    speedups = [r["speedup_best"] for r in records]
    print("-" * len(header))
    print(
        f"speedup (best-of-{args.repeat}): "
        f"min {min(speedups):.1f}x   median {statistics.median(speedups):.1f}x   max {max(speedups):.1f}x"
    )
    if not all(r["adjacency_matrix_match"] for r in records):
        print("WARNING: adjacency_matrix_ disagreed on at least one config (see 'adj. match' column)")

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
