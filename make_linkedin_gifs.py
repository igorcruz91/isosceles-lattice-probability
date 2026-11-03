"""Render animated GIFs of the isosceles-triangle experiment for a LinkedIn post.

Each GIF has three panels that evolve together, frame by frame:

* left   - the lattice grid, with the triangle drawn in each round (plus fading
           ghosts of the previous rounds);
* middle - the running isosceles-rate estimate of that single game converging
           to the exact probability;
* right  - the Monte Carlo histogram of the estimates of many independent
           games, filling up as the animation advances.

Three visual styles are rendered. LinkedIn only animates GIFs with fewer than
~400 frames and recommends < 5 MB, so the frame schedule and the colour
quantisation below are tuned to stay well within those limits.

Usage::

    python make_linkedin_gifs.py
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from core.grid import build_grid  # noqa: E402
from core.monte_carlo import classify_batch, monte_carlo_games  # noqa: E402

GRID_SIZE = 10
N_ROUNDS = 1_000_000
N_GAMES = 10_000
SEED = 12  # showcase game: isosceles in rounds 3 and 11, ends at ~7.18%
MC_SEED = 42
MC_TASKS = 200
MC_WORKERS = 10
# Draws per vectorised chunk inside each worker. Kept small so that MC_WORKERS
# processes together stay within a few GB of RAM (~120 bytes per draw).
MC_CHUNK = 2_000_000
SLOW_ROUNDS = 12
FAST_FRAMES = 150
SLOW_MS = 420
FAST_MS = 70
HOLD_MS = 4000
N_GHOSTS = 6
N_COLORS = 96
FIG_SIZE = (12.0, 4.6)
DPI = 100
MAX_FRAMES = 400

EXPORT_STYLES = ("clean", "neon")

OUTPUT_DIR = Path(__file__).resolve().parent
CACHE_DIR = OUTPUT_DIR / ".cache"


@dataclass(frozen=True)
class Style:
    """Colours, fonts and plot variants for one GIF version."""

    name: str
    bg: str
    panel_bg: str
    fg: str
    muted: str
    grid_dot: str
    iso: str
    scalene: str
    accent: str
    band: str
    font: str
    hist_kind: str  # "bars", "step" or "lollipop"
    line_glow: bool
    show_axes_grid: bool


STYLES: tuple[Style, ...] = (
    Style(
        name="clean",
        bg="#ffffff",
        panel_bg="#ffffff",
        fg="#1f2933",
        muted="#7b8794",
        grid_dot="#cbd2d9",
        iso="#0e9f6e",
        scalene="#e4572e",
        accent="#2563eb",
        band="#bfdbfe",
        font="DejaVu Sans",
        hist_kind="bars",
        line_glow=False,
        show_axes_grid=True,
    ),
    Style(
        name="neon",
        bg="#0b1020",
        panel_bg="#0b1020",
        fg="#e6edf7",
        muted="#7c8aa5",
        grid_dot="#2a3552",
        iso="#39ff88",
        scalene="#ff4d8d",
        accent="#22d3ee",
        band="#16324a",
        font="DejaVu Sans",
        hist_kind="step",
        line_glow=True,
        show_axes_grid=False,
    ),
    Style(
        name="editorial",
        bg="#f6f1e7",
        panel_bg="#f6f1e7",
        fg="#2b2118",
        muted="#8a7b6b",
        grid_dot="#cdbfa9",
        iso="#2f6f5e",
        scalene="#c0392b",
        accent="#1d3557",
        band="#e3d7c3",
        font="serif",
        hist_kind="lollipop",
        line_glow=False,
        show_axes_grid=True,
    ),
)


@dataclass
class Experiment:
    """All the precomputed data the animation needs."""

    grid: np.ndarray
    triangles: np.ndarray  # (N_ROUNDS, 3, 2) vertices of each round
    is_isosceles: np.ndarray  # (N_ROUNDS,) bool
    running_rate: np.ndarray  # (N_ROUNDS,) estimate after each round
    game_estimates: np.ndarray  # (N_GAMES,) Monte Carlo game estimates
    exact: float


def exact_isosceles_probability(width: int, height: int) -> float:
    """Return P(isosceles | not collinear) by enumerating every triple of distinct points."""
    grid = build_grid(width, height)
    idx = np.array(list(itertools.combinations(range(len(grid)), 3)))
    result = classify_batch(grid, idx)
    valid = ~result.is_degenerate
    return float(result.is_isosceles[valid].mean())


def running_rate(is_isosceles: np.ndarray) -> np.ndarray:
    """Return the cumulative isosceles rate after each round."""
    return np.cumsum(is_isosceles) / np.arange(1, len(is_isosceles) + 1)


def _mc_task(args: tuple[int, np.random.SeedSequence]) -> np.ndarray:
    """Picklable worker: run `n_games` games with a memory-bounded chunk size."""
    n_games, seed = args
    return monte_carlo_games(
        GRID_SIZE, GRID_SIZE, n_games, N_ROUNDS, seed=seed, max_draws_per_chunk=MC_CHUNK
    )


def monte_carlo_estimates(
    n_games: int = N_GAMES, seed: int = MC_SEED, cache_dir: Path = CACHE_DIR
) -> np.ndarray:
    """Return per-game estimates for n_games games of N_ROUNDS rounds, cached on disk.

    The work is split into MC_TASKS reproducible tasks (seeds spawned from
    `seed`) run on MC_WORKERS processes, printing progress as tasks finish.
    """
    cache = cache_dir / f"mc_{GRID_SIZE}x{GRID_SIZE}_{N_ROUNDS}r_{n_games}g_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    base, rem = divmod(n_games, MC_TASKS)
    sizes = [base + (1 if i < rem else 0) for i in range(MC_TASKS)]
    seeds = np.random.SeedSequence(seed).spawn(MC_TASKS)
    tasks = [(size, s) for size, s in zip(sizes, seeds) if size > 0]
    results: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=MC_WORKERS) as executor:
        for i, res in enumerate(executor.map(_mc_task, tasks), start=1):
            results.append(res)
            print(f"  Monte Carlo: {i}/{len(tasks)} tasks done", flush=True)
    estimates = np.concatenate(results)
    cache_dir.mkdir(exist_ok=True)
    np.save(cache, estimates)
    return estimates


def build_experiment(seed: int = SEED) -> Experiment:
    """Draw one showcase game and the Monte Carlo batch of games."""
    grid = build_grid(GRID_SIZE, GRID_SIZE)
    rng = np.random.default_rng(seed)
    idx = _draw_valid_indices(grid, rng, N_ROUNDS)
    is_iso = classify_batch(grid, idx).is_isosceles
    estimates = monte_carlo_estimates()
    return Experiment(
        grid=grid,
        triangles=grid[idx],
        is_isosceles=is_iso,
        running_rate=running_rate(is_iso),
        game_estimates=estimates,
        exact=exact_isosceles_probability(GRID_SIZE, GRID_SIZE),
    )


def _draw_valid_indices(grid: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw n triples of distinct, non-collinear point indices."""
    out = np.empty((0, 3), dtype=int)
    while len(out) < n:
        idx = rng.choice(len(grid), size=(n, 3))
        distinct = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
        idx = idx[distinct]
        idx = idx[~classify_batch(grid, idx).is_degenerate]
        out = np.vstack([out, idx])
    return out[:n]


def frame_schedule(
    n_rounds: int = N_ROUNDS,
    slow_rounds: int = SLOW_ROUNDS,
    fast_frames: int = FAST_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rounds shown per frame, frame duration in ms).

    The first `slow_rounds` frames show one round each, slowly, so the viewer
    can follow individual triangles; the rest accelerate geometrically up to
    `n_rounds`. The last frame is held for longer before the GIF loops.
    """
    slow = np.arange(1, slow_rounds + 1)
    fast = np.geomspace(slow_rounds + 1, n_rounds, fast_frames).round().astype(int)
    rounds = np.unique(np.concatenate([slow, fast]))
    durations = np.full(len(rounds), FAST_MS)
    durations[:slow_rounds] = SLOW_MS
    durations[-1] = HOLD_MS
    return rounds, durations


def games_for_frame(frame: int, n_frames: int, n_games: int = N_GAMES) -> int:
    """Number of Monte Carlo games revealed in the histogram at a given frame."""
    if n_frames <= 1:
        return n_games
    t = frame / (n_frames - 1)
    return max(1, int(round(n_games * t**1.6)))


def tex_int(n: int) -> str:
    """Format an integer with thousands separators for mathtext, e.g. 10{,}000."""
    return f"{n:,}".replace(",", "{,}")


def tex_pct(x: float, decimals: int) -> str:
    """Format a fraction as a mathtext percentage, e.g. 7.18\\%."""
    return f"{100 * x:.{decimals}f}" + r"\%"


def pct_formatter(decimals: int) -> matplotlib.ticker.FuncFormatter:
    """Tick formatter rendering fractions as LaTeX-style percentages."""
    return matplotlib.ticker.FuncFormatter(lambda v, _: f"${tex_pct(v, decimals)}$")


class GifRenderer:
    """Builds the three-panel figure once and updates its artists per frame."""

    def __init__(self, exp: Experiment, style: Style) -> None:
        self.exp = exp
        self.s = style
        plt.rcParams["font.family"] = style.font
        # LaTeX look for axes (matplotlib's built-in TeX renderer, no TeX install needed).
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["axes.formatter.use_mathtext"] = True
        plt.rcParams["axes.unicode_minus"] = False
        n = len(exp.running_rate)
        self.rate_x = np.unique(np.geomspace(1, n, 2500).round().astype(int))
        self.fig: Figure = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor=style.bg)
        gs = self.fig.add_gridspec(
            1,
            3,
            left=0.03,
            right=0.98,
            bottom=0.14,
            top=0.78,
            wspace=0.28,
            width_ratios=[1.0, 1.25, 1.25],
        )
        self.ax_grid = self.fig.add_subplot(gs[0])
        self.ax_rate = self.fig.add_subplot(gs[1])
        self.ax_hist = self.fig.add_subplot(gs[2])
        self._setup_titles()
        self._setup_grid()
        self._setup_rate()
        self._setup_hist()

    # ---------- static setup ----------

    def _setup_titles(self) -> None:
        s = self.s
        self.fig.text(
            0.03,
            0.93,
            "What are the odds that 3 random grid points form an isosceles triangle?",
            color=s.fg,
            fontsize=16,
            fontweight="bold",
            ha="left",
        )
        self.fig.text(
            0.03,
            0.865,
            f"{GRID_SIZE}×{GRID_SIZE} grid · 3 distinct points per round · "
            f"collinear draws are redrawn · exact value $p = {tex_pct(self.exp.exact, 2)}$",
            color=s.muted,
            fontsize=10.5,
            ha="left",
        )

    def _style_axis(self, ax: plt.Axes) -> None:
        s = self.s
        ax.set_facecolor(s.panel_bg)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(s.muted)
        ax.tick_params(colors=s.muted, labelsize=11)
        ax.xaxis.label.set_color(s.fg)
        ax.yaxis.label.set_color(s.fg)
        if s.show_axes_grid:
            ax.grid(True, color=s.band if s.name == "editorial" else "#e5e7eb", lw=0.8)
            ax.set_axisbelow(True)

    def _setup_grid(self) -> None:
        s = self.s
        ax = self.ax_grid
        ax.set_facecolor(s.panel_bg)
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, GRID_SIZE - 0.4)
        ax.set_ylim(-0.6, GRID_SIZE - 0.4)
        ax.axis("off")
        g = self.exp.grid
        ax.scatter(g[:, 0], g[:, 1], s=14, color=s.grid_dot, zorder=1)
        self.ghosts = PolyCollection([], zorder=2)
        ax.add_collection(self.ghosts)
        self.tri_fill = PolyCollection([], zorder=3, linewidths=0)
        ax.add_collection(self.tri_fill)
        (self.tri_edge,) = ax.plot([], [], lw=2.6, zorder=4, solid_joinstyle="round")
        if s.line_glow:
            (self.tri_glow,) = ax.plot([], [], lw=9, alpha=0.25, zorder=3)
        self.tri_pts = ax.scatter([], [], s=46, zorder=5, edgecolors=s.panel_bg, linewidths=1.2)
        self.round_text = ax.text(
            0.5,
            -0.06,
            "",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=s.fg,
        )
        self.tally_text = ax.text(
            0.5,
            -0.15,
            "",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            color=s.muted,
        )

    def _setup_rate(self) -> None:
        s = self.s
        ax = self.ax_rate
        self._style_axis(ax)
        ax.set_title(
            r"Running estimate $\hat{p}_n$ of one game", color=s.fg, fontsize=12, loc="left"
        )
        ax.set_xlabel(r"$n\ \mathrm{(rounds,\ log\ scale)}$", fontsize=14)
        ax.set_ylabel(r"$\hat{p}_n$", fontsize=16)
        ax.set_xscale("log")
        ax.set_xlim(1, N_ROUNDS * 1.6)
        ax.set_ylim(0, 0.4)
        ax.yaxis.set_major_formatter(pct_formatter(0))
        p = self.exp.exact
        n = np.geomspace(1, N_ROUNDS * 1.6, 400)
        half = 1.96 * np.sqrt(p * (1 - p) / n)
        ax.text(
            0.03,
            0.04,
            r"$\mathrm{shaded:}\ p \pm 1.96\sqrt{p(1-p)/n}$",
            transform=ax.transAxes,
            color=s.muted,
            fontsize=11,
            ha="left",
            va="bottom",
        )
        ax.fill_between(
            n,
            p - half,
            p + half,
            color=s.band,
            alpha=0.9 if s.name != "neon" else 0.6,
            lw=0,
            zorder=0,
            label="expected 95% CI",
        )
        ax.axhline(p, color=s.muted, ls="--", lw=1.2, zorder=1)
        ax.text(
            1.01,
            p,
            f"$p$\n${tex_pct(p, 2)}$",
            transform=ax.get_yaxis_transform(),
            color=s.muted,
            fontsize=10,
            va="center",
            ha="left",
            clip_on=False,
        )
        if s.line_glow:
            self.rate_glow = [
                ax.plot([], [], color=s.accent, lw=w, alpha=a, zorder=2)[0]
                for w, a in ((10, 0.06), (6, 0.12), (3.5, 0.25))
            ]
        (self.rate_line,) = ax.plot([], [], color=s.accent, lw=2.0, zorder=3)
        self.rate_dot = ax.scatter(
            [], [], s=40, color=s.accent, zorder=4, edgecolors=s.panel_bg, linewidths=1.2
        )
        if s.name == "editorial":
            self.rate_markers = ax.scatter([], [], s=10, color=s.accent, zorder=3, alpha=0.5)
        self.rate_text = ax.text(
            0.97,
            0.95,
            "",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=16,
            fontweight="bold",
            color=s.accent,
        )

    def _setup_hist(self) -> None:
        s = self.s
        ax = self.ax_hist
        self._style_axis(ax)
        n_games = len(self.exp.game_estimates)
        ax.set_title(
            rf"Monte Carlo: ${tex_int(n_games)}$ games $\times$ $10^{{{int(np.log10(N_ROUNDS))}}}$"
            " rounds",
            color=s.fg,
            fontsize=12,
            loc="left",
        )
        ax.set_xlabel(r"$\hat{p}\ \mathrm{of\ each\ game}$", fontsize=14)
        ax.set_ylabel(r"$\mathrm{number\ of\ games}$", fontsize=14)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"${tex_int(int(v))}$")
        )
        ax.xaxis.set_major_formatter(pct_formatter(2))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        est = self.exp.game_estimates
        self.bins = np.linspace(est.min(), est.max(), 41)
        step = self.bins[1] - self.bins[0]
        full, _ = np.histogram(est, bins=self.bins)
        ax.set_xlim(self.bins[0] - step, self.bins[-1] + step)
        ax.set_ylim(0, full.max() * 1.15)
        ax.axvline(self.exp.exact, color=s.muted, ls="--", lw=1.2, zorder=4)
        self.centers = (self.bins[:-1] + self.bins[1:]) / 2
        self.width = step
        self.hist_text = ax.text(
            0.97,
            0.95,
            "",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12.5,
            color=s.fg,
            linespacing=1.5,
        )
        self.clt_text = ax.text(
            0.03,
            0.95,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color=s.muted,
        )
        self.hist_artists: list = []

    # ---------- per-frame updates ----------

    def _color(self, iso: bool) -> str:
        return self.s.iso if iso else self.s.scalene

    def _update_grid(self, k: int) -> None:
        s = self.s
        exp = self.exp
        tri = exp.triangles[k - 1]
        iso = bool(exp.is_isosceles[k - 1])
        color = self._color(iso)

        first = max(0, k - 1 - N_GHOSTS)
        ghosts = exp.triangles[first : k - 1]
        ghost_iso = exp.is_isosceles[first : k - 1]
        alphas = np.linspace(0.08, 0.35, len(ghosts)) if len(ghosts) else []
        ghost_colors = [
            matplotlib.colors.to_rgba(self._color(bool(g)), a) for g, a in zip(ghost_iso, alphas)
        ]
        self.ghosts.set_verts(list(ghosts))
        self.ghosts.set_facecolors("none")
        self.ghosts.set_edgecolors(ghost_colors)
        self.ghosts.set_linewidths(1.2)

        self.tri_fill.set_verts([tri])
        self.tri_fill.set_facecolors([matplotlib.colors.to_rgba(color, 0.22)])
        closed = np.vstack([tri, tri[0]])
        self.tri_edge.set_data(closed[:, 0], closed[:, 1])
        self.tri_edge.set_color(color)
        if s.line_glow:
            self.tri_glow.set_data(closed[:, 0], closed[:, 1])
            self.tri_glow.set_color(color)
        self.tri_pts.set_offsets(tri)
        self.tri_pts.set_facecolors([color])

        label = "isosceles" if iso else "scalene"
        self.round_text.set_text(f"Round {k:,}: {label}")
        self.round_text.set_color(color)
        n_iso = int(exp.is_isosceles[:k].sum())
        self.tally_text.set_text(f"isosceles {n_iso:,}  ·  scalene {k - n_iso:,}")

    def _update_rate(self, k: int) -> None:
        s = self.s
        # The full game has up to 10^6 points; plot a log-spaced subsample plus round k.
        x = np.append(self.rate_x[self.rate_x < k], k)
        y = self.exp.running_rate[x - 1]
        self.rate_line.set_data(x, y)
        if s.line_glow:
            for g in self.rate_glow:
                g.set_data(x, y)
        if s.name == "editorial":
            self.rate_markers.set_offsets(np.column_stack([x, y]))
        self.rate_dot.set_offsets([[k, y[-1]]])
        self.rate_text.set_text(rf"$\hat{{p}}_n = {tex_pct(y[-1], 3)}$")

    def _update_hist(self, n_games: int) -> None:
        s = self.s
        ax = self.ax_hist
        for a in self.hist_artists:
            a.remove()
        self.hist_artists = []
        est = self.exp.game_estimates[:n_games]
        counts, _ = np.histogram(est, bins=self.bins)
        if s.hist_kind == "bars":
            bars = ax.bar(
                self.centers, counts, width=self.width * 0.9, color=s.accent, alpha=0.85, zorder=2
            )
            self.hist_artists.extend(bars.patches)
        elif s.hist_kind == "step":
            edges = np.repeat(self.bins, 2)[1:-1]
            heights = np.repeat(counts, 2)
            self.hist_artists.append(
                ax.fill_between(edges, heights, color=s.accent, alpha=0.18, lw=0, zorder=2)
            )
            for w, a in ((7, 0.08), (3.5, 0.2)):
                self.hist_artists.append(
                    ax.plot(edges, heights, color=s.accent, lw=w, alpha=a, zorder=3)[0]
                )
            self.hist_artists.append(ax.plot(edges, heights, color=s.accent, lw=1.6, zorder=3)[0])
        else:
            nz = counts > 0
            self.hist_artists.append(
                ax.vlines(self.centers[nz], 0, counts[nz], color=s.accent, lw=1.6, zorder=2)
            )
            self.hist_artists.append(
                ax.scatter(
                    self.centers[nz],
                    counts[nz],
                    s=22,
                    color=s.accent,
                    zorder=3,
                    edgecolors=s.panel_bg,
                    linewidths=0.8,
                )
            )
        # Normal approximation (CLT) scaled to the games shown so far, as in the app.
        p = self.exp.exact
        se = np.sqrt(p * (1 - p) / N_ROUNDS)
        xs = np.linspace(self.bins[0] - self.width, self.bins[-1] + self.width, 200)
        pdf = np.exp(-0.5 * ((xs - p) / se) ** 2) / (se * np.sqrt(2 * np.pi))
        self.hist_artists.append(
            ax.plot(xs, pdf * n_games * self.width, color=s.fg, ls=":", lw=1.4, zorder=4)[0]
        )
        mean = est.mean()
        self.hist_text.set_text(
            f"${tex_int(n_games)}" + r"\ \mathrm{games}$" + "\n"
            rf"$\bar{{p}} = {tex_pct(mean, 3)}$"
        )
        self.clt_text.set_text(r"$\mathrm{dotted:}\ \mathcal{N}\left(p,\ p(1-p)/n\right)$")

    def render(self, k: int, n_games: int) -> Image.Image:
        """Draw the figure for round k with n_games in the histogram."""
        self._update_grid(k)
        self._update_rate(k)
        self._update_hist(n_games)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return Image.fromarray(buf[:, :, :3].copy())

    def close(self) -> None:
        """Release the matplotlib figure."""
        plt.close(self.fig)


def _swatch(style: Style, width: int, height: int) -> Image.Image:
    """Solid blocks of the style's key colours (and their blends with the background).

    Appended to the palette sample so that colours covering few pixels, like the
    isosceles triangle, still get an exact palette entry instead of a greyish one.
    """
    bg = np.array(matplotlib.colors.to_rgb(style.bg))
    swatches = []
    for c in (style.iso, style.scalene, style.accent, style.fg, style.muted):
        rgb = np.array(matplotlib.colors.to_rgb(c))
        for a in (1.0, 0.6, 0.3, 0.2):
            swatches.append(a * rgb + (1 - a) * bg)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    block = width // len(swatches)
    for i, c in enumerate(swatches):
        img[:, i * block : (i + 1) * block] = (np.array(c) * 255).round()
    return Image.fromarray(img)


def quantize_frames(
    frames: list[Image.Image], style: Style, n_colors: int = N_COLORS
) -> list[Image.Image]:
    """Map every frame to one shared palette so the GIF compresses well and does not flicker."""
    sample = frames[:: max(1, len(frames) // 12)] + [frames[-1]]
    w, h = sample[0].size
    sample.append(_swatch(style, w, h * 2))
    strip = Image.new("RGB", (w, sum(f.size[1] for f in sample)))
    y = 0
    for f in sample:
        strip.paste(f, (0, y))
        y += f.size[1]
    palette = strip.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    return [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]


def make_gif(exp: Experiment, style: Style, out_dir: Path = OUTPUT_DIR) -> Path:
    """Render and save one GIF version; return its path."""
    rounds, durations = frame_schedule()
    if len(rounds) > MAX_FRAMES:
        raise ValueError(f"{len(rounds)} frames exceeds LinkedIn's limit of {MAX_FRAMES}.")
    renderer = GifRenderer(exp, style)
    frames = [
        renderer.render(int(k), games_for_frame(i, len(rounds))) for i, k in enumerate(rounds)
    ]
    renderer.close()
    frames = quantize_frames(frames, style)
    path = out_dir / f"isosceles_triangles_{style.name}.gif"
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=[int(d) for d in durations],
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path


def main() -> None:
    """Render all style versions into the project root."""
    exp = build_experiment()
    for style in STYLES:
        if style.name not in EXPORT_STYLES:
            continue
        path = make_gif(exp, style)
        size_mb = path.stat().st_size / 1e6
        with Image.open(path) as im:
            n_frames = im.n_frames
        print(f"{path.name}: {n_frames} frames, {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
