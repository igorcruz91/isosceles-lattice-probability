"""Repeat the 10x10 Monte Carlo experiment on larger grids and compare the results.

Same setup as make_linkedin_gifs.py: N_GAMES games of N_ROUNDS rounds each,
3 distinct uniform points per round, collinear draws redrawn, seeds spawned
from MC_SEED into MC_TASKS tasks. The cache file names and seeding match that
script, so the 10x10 run is reused from .cache/ instead of recomputed.

Writes results/grid_size_comparison.csv and results/grid_size_comparison.png.

Usage::

    python grid_size_experiment.py
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.exact import exact_isosceles_probability  # noqa: E402
from core.monte_carlo import monte_carlo_games  # noqa: E402

GRID_SIZES = (10, 20, 30, 40)
N_ROUNDS = 1_000_000
N_GAMES = 10_000
MC_SEED = 42
MC_TASKS = 200
MC_WORKERS = 10
MC_CHUNK = 2_000_000

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results"


def _mc_task(args: tuple[int, int, np.random.SeedSequence]) -> np.ndarray:
    """Picklable worker: run `n_games` games on a size x size grid."""
    size, n_games, seed = args
    return monte_carlo_games(size, size, n_games, N_ROUNDS, seed=seed, max_draws_per_chunk=MC_CHUNK)


def monte_carlo_estimates(size: int, n_games: int = N_GAMES, seed: int = MC_SEED) -> np.ndarray:
    """Return per-game estimates for a size x size grid, cached on disk."""
    cache = CACHE_DIR / f"mc_{size}x{size}_{N_ROUNDS}r_{n_games}g_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    base, rem = divmod(n_games, MC_TASKS)
    sizes = [base + (1 if i < rem else 0) for i in range(MC_TASKS)]
    seeds = np.random.SeedSequence(seed).spawn(MC_TASKS)
    tasks = [(size, n, s) for n, s in zip(sizes, seeds) if n > 0]
    results: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=MC_WORKERS) as executor:
        for i, res in enumerate(executor.map(_mc_task, tasks), start=1):
            results.append(res)
            if i % 20 == 0:
                print(f"  {size}x{size}: {i}/{len(tasks)} tasks done", flush=True)
    estimates = np.concatenate(results)
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(cache, estimates)
    return estimates


def summarize(size: int, estimates: np.ndarray) -> dict[str, float | int | str]:
    """Return the comparison row for one grid size."""
    exact = exact_isosceles_probability(size, size)
    mean = float(estimates.mean())
    se_of_mean = float(estimates.std(ddof=1) / np.sqrt(len(estimates)))
    return {
        "grid": f"{size}x{size}",
        "n": size,
        "games": len(estimates),
        "rounds_per_game": N_ROUNDS,
        "mc_mean": mean,
        "mc_std_across_games": float(estimates.std(ddof=1)),
        "mc_se_of_mean": se_of_mean,
        "exact": float(exact),
        "exact_fraction": f"{exact.numerator}/{exact.denominator}",
        "z_score": (mean - float(exact)) / se_of_mean,
    }


def plot_comparison(rows: list[dict], path: Path) -> None:
    """Plot the probability and n^2 * probability against grid size."""
    n = np.array([r["n"] for r in rows])
    mc = np.array([r["mc_mean"] for r in rows])
    se = np.array([r["mc_se_of_mean"] for r in rows])
    exact = np.array([r["exact"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.errorbar(n, mc, yerr=1.96 * se, fmt="o", color="#2563eb", label="Monte Carlo mean")
    ax1.plot(n, exact, "x--", color="#1f2933", label="exact count")
    ax1.set_yscale("log")
    ax1.set_xlabel("grid side n (n x n points)")
    ax1.set_ylabel("P(isosceles | triangle)")
    ax1.set_xticks(n)
    ax1.legend(frameon=False)

    ax2.plot(n, mc * n**2, "o-", color="#2563eb", label=r"$n^2 \cdot p$")
    ax2.plot(n, mc * n**2 / np.log(n), "s-", color="#0e9f6e", label=r"$n^2 \cdot p / \ln n$")
    ax2.set_xlabel("grid side n")
    ax2.set_xticks(n)
    ax2.set_title("Scaled probability (what p behaves like)")
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    """Run (or load) every grid size, then write the CSV and the comparison plot."""
    rows = []
    for size in GRID_SIZES:
        start = time.perf_counter()
        estimates = monte_carlo_estimates(size)
        print(f"{size}x{size}: done in {time.perf_counter() - start:.0f}s", flush=True)
        rows.append(summarize(size, estimates))

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "grid_size_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(rows, RESULTS_DIR / "grid_size_comparison.png")

    for r in rows:
        print(
            f"{r['grid']:>6}  MC {r['mc_mean']:.6f} +- {r['mc_se_of_mean']:.6f}  "
            f"exact {r['exact']:.6f} ({r['exact_fraction']})  z={r['z_score']:+.2f}  "
            f"n^2 p = {r['mc_mean'] * r['n'] ** 2:.3f}"
        )


if __name__ == "__main__":
    main()
