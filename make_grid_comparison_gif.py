"""Render a GIF comparing the isosceles experiment on 10x10, 20x20 and 30x30 grids.

Companion to ``make_linkedin_gifs.py`` (which shows the 10x10 grid alone). The
three grids play one game each, side by side and in lockstep:

* top row      - each grid with its current triangle (green = isosceles,
                 red = scalene) and its running estimate as a fraction;
* bottom left  - the three running estimates converging to their exact values;
* bottom right - the exact probability p_n against grid size n, with the
                 asymptotic law (24/pi^2 + 4/pi) ln n / n^2 - 1.79 / n^2 and the
                 three live estimates (with their 95% band) homing in on it.

The frame schedule and palette quantisation are shared with
``make_linkedin_gifs.py`` so the GIF stays within LinkedIn's limits
(< 400 frames, < 5 MB).

Usage::

    python make_grid_comparison_gif.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from core.exact import exact_isosceles_probability  # noqa: E402
from core.grid import build_grid  # noqa: E402
from core.monte_carlo import classify_batch  # noqa: E402
from make_linkedin_gifs import (  # noqa: E402
    MAX_FRAMES,
    N_GHOSTS,
    STYLES,
    Style,
    _draw_valid_indices,
    frame_schedule,
    quantize_frames,
    running_rate,
    tex_pct,
)

GRID_SIZES = (10, 20, 30)
# Showcase seeds: 10x10 reuses the first GIF's game (isosceles in rounds 3 and
# 11); the others were picked so one isosceles appears in the slow opening.
SHOWCASE_SEEDS = {10: 12, 20: 43, 30: 51}
N_ROUNDS = 1_000_000
CURVE_SIZES = np.arange(6, 41)  # p_5 = 20.7% would sit above CURVE_YMAX
LAW_SLOPE = 24 / np.pi**2 + 4 / np.pi  # ~3.705, from sum r_2(k)^2 ~ 4 X ln X
LAW_OFFSET = 1.79
FIG_SIZE = (12.0, 7.5)
DPI = 100
RATE_YMAX = 0.20
N_COLORS = 64  # fewer than the 10x10 GIF: three grids per frame cost more bytes
MAX_MB = 5.0  # LinkedIn's recommended GIF size
BAND_START = 100  # rounds; before this the 95% bands are wider than the axis
CURVE_YMAX = 0.16
EXPORT_STYLES = ("clean", "neon")

# One colour per grid size, ordered light -> dark with the grid size.
SERIES_COLORS: dict[str, tuple[str, str, str]] = {
    "clean": ("#3b82f6", "#1d4ed8", "#172554"),
    "neon": ("#a5f3fc", "#22d3ee", "#0ea5e9"),
}

OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass
class GridGame:
    """One showcase game of N_ROUNDS rounds on a size x size grid."""

    size: int
    grid: np.ndarray
    triangles: np.ndarray  # (n_rounds, 3, 2) vertices of each round
    is_isosceles: np.ndarray  # (n_rounds,) bool
    running_rate: np.ndarray  # (n_rounds,) estimate after each round
    exact: float


def play_showcase_game(size: int, seed: int, n_rounds: int = N_ROUNDS) -> GridGame:
    """Play one game of n_rounds non-collinear draws on a size x size grid."""
    grid = build_grid(size, size)
    idx = _draw_valid_indices(grid, np.random.default_rng(seed), n_rounds)
    is_iso = classify_batch(grid, idx).is_isosceles
    return GridGame(
        size=size,
        grid=grid,
        triangles=grid[idx],
        is_isosceles=is_iso,
        running_rate=running_rate(is_iso),
        exact=float(exact_isosceles_probability(size, size)),
    )


def exact_curve(sizes: np.ndarray) -> np.ndarray:
    """Return the exact isosceles probability p_n for every n in sizes."""
    return np.array([float(exact_isosceles_probability(int(n), int(n))) for n in sizes])


def asymptotic_law(n: np.ndarray) -> np.ndarray:
    """Return the asymptotic estimate (LAW_SLOPE ln n - LAW_OFFSET) / n^2."""
    n = np.asarray(n, dtype=float)
    return (LAW_SLOPE * np.log(n) - LAW_OFFSET) / n**2


def ci_half_width(p: float, n_rounds: np.ndarray | int) -> np.ndarray:
    """Half-width of the 95% normal band p +/- 1.96 sqrt(p(1-p)/n)."""
    return 1.96 * np.sqrt(p * (1 - p) / np.asarray(n_rounds, dtype=float))


def dot_area(size: int) -> float:
    """Marker area for grid dots, shrinking so every grid panel looks equally dense."""
    return 1400 / size**2


def grid_label(size: int) -> str:
    """Return the label of a size x size grid, e.g. '20×20'."""
    return f"{size}×{size}"


class ComparisonRenderer:
    """Builds the comparison figure once and updates its artists per frame."""

    def __init__(self, games: list[GridGame], curve: np.ndarray, style: Style) -> None:
        self.games = games
        self.curve = curve
        self.s = style
        self.colors = SERIES_COLORS[style.name]
        plt.rcParams["font.family"] = style.font
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams["axes.formatter.use_mathtext"] = True
        plt.rcParams["axes.unicode_minus"] = False
        self.n_rounds = len(games[0].running_rate)
        self.rate_x = np.unique(np.geomspace(1, self.n_rounds, 2500).round().astype(int))
        self.fig: Figure = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor=style.bg)
        top = self.fig.add_gridspec(1, 3, left=0.03, right=0.97, bottom=0.47, top=0.83, wspace=0.1)
        bottom = self.fig.add_gridspec(
            1, 2, left=0.07, right=0.93, bottom=0.08, top=0.33, wspace=0.32
        )
        self.ax_grids = [self.fig.add_subplot(top[i]) for i in range(len(games))]
        self.ax_rate = self.fig.add_subplot(bottom[0])
        self.ax_curve = self.fig.add_subplot(bottom[1])
        self._setup_titles()
        for ax, game, color in zip(self.ax_grids, games, self.colors):
            self._setup_grid(ax, game, color)
        self._setup_rate()
        self._setup_curve()

    # ---------- static setup ----------

    def _setup_titles(self) -> None:
        s = self.s
        self.fig.text(
            0.03,
            0.95,
            "Bigger grid, rarer isosceles triangles",
            color=s.fg,
            fontsize=17,
            fontweight="bold",
            ha="left",
        )
        self.fig.text(
            0.03,
            0.905,
            "3 distinct random points per round · collinear draws are redrawn · "
            "one game per grid, played side by side",
            color=s.muted,
            fontsize=11,
            ha="left",
        )

    def _style_axis(self, ax: plt.Axes) -> None:
        s = self.s
        ax.set_facecolor(s.panel_bg)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(s.muted)
        ax.tick_params(colors=s.muted, labelsize=10)
        ax.xaxis.label.set_color(s.fg)
        ax.yaxis.label.set_color(s.fg)
        if s.show_axes_grid:
            ax.grid(True, color="#e5e7eb", lw=0.8)
            ax.set_axisbelow(True)

    def _setup_grid(self, ax: plt.Axes, game: GridGame, color: str) -> None:
        s = self.s
        n = game.size
        ax.set_facecolor(s.panel_bg)
        ax.set_aspect("equal")
        pad = 0.04 * n
        ax.set_xlim(-pad, n - 1 + pad)
        ax.set_ylim(-pad, n - 1 + pad)
        ax.axis("off")
        ax.scatter(game.grid[:, 0], game.grid[:, 1], s=dot_area(n), color=s.grid_dot, zorder=1)
        ax.set_title(
            f"{grid_label(n)}  ·  exact $p = {tex_pct(game.exact, 2)}$",
            color=color,
            fontsize=13,
            fontweight="bold",
            pad=6,
        )
        artists: dict = {}
        artists["ghosts"] = PolyCollection([], zorder=2, facecolors="none", linewidths=1.0)
        ax.add_collection(artists["ghosts"])
        artists["fill"] = PolyCollection([], zorder=3, linewidths=0)
        ax.add_collection(artists["fill"])
        lw = 2.4 if n <= 10 else 1.8
        (artists["edge"],) = ax.plot([], [], lw=lw, zorder=4, solid_joinstyle="round")
        if s.line_glow:
            (artists["glow"],) = ax.plot([], [], lw=3.5 * lw, alpha=0.25, zorder=3)
        artists["pts"] = ax.scatter(
            [], [], s=max(14, 4 * dot_area(n)), zorder=5, edgecolors=s.panel_bg, linewidths=0.8
        )
        artists["round"] = ax.text(
            0.5,
            -0.04,
            "",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11.5,
            fontweight="bold",
        )
        artists["rate"] = ax.text(
            0.5, -0.13, "", transform=ax.transAxes, ha="center", va="top", fontsize=11, color=color
        )
        ax.artists_by_name = artists  # type: ignore[attr-defined]

    def _setup_rate(self) -> None:
        s = self.s
        ax = self.ax_rate
        self._style_axis(ax)
        ax.set_title(
            r"Running estimate $\hat{p}$ of each game", color=s.fg, fontsize=12, loc="left"
        )
        ax.set_xlabel(r"$\mathrm{rounds\ (log\ scale)}$", fontsize=12)
        ax.set_xscale("log")
        ax.set_xlim(1, self.n_rounds * 1.6)
        ax.set_ylim(0, RATE_YMAX)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"${tex_pct(v, 0)}$")
        )
        ax.text(
            0.97,
            0.93,
            "dashed: exact p\nshaded: 95% range of p̂ after n rounds",
            transform=ax.transAxes,
            color=s.muted,
            fontsize=10,
            ha="right",
            va="top",
            linespacing=1.6,
        )
        n = np.geomspace(BAND_START, self.n_rounds * 1.6, 300)
        self.rate_lines = []
        self.rate_dots = []
        for game, color in zip(self.games, self.colors):
            half = ci_half_width(game.exact, n)
            ax.fill_between(
                n, game.exact - half, game.exact + half, color=color, alpha=0.13, lw=0, zorder=0
            )
            ax.axhline(game.exact, color=color, ls="--", lw=1.0, alpha=0.8, zorder=1)
            ax.text(
                1.01,
                game.exact,
                f"{grid_label(game.size)}",
                transform=ax.get_yaxis_transform(),
                color=color,
                fontsize=9.5,
                fontweight="bold",
                va="center",
                ha="left",
                clip_on=False,
            )
            (line,) = ax.plot([], [], color=color, lw=1.8, zorder=3)
            dot = ax.scatter([], [], s=30, color=color, zorder=4, edgecolors=s.panel_bg)
            self.rate_lines.append(line)
            self.rate_dots.append(dot)

    def _setup_curve(self) -> None:
        s = self.s
        ax = self.ax_curve
        self._style_axis(ax)
        ax.set_title(r"Exact $p_n$ on an $n \times n$ grid", color=s.fg, fontsize=12, loc="left")
        ax.set_xlabel(r"$\mathrm{grid\ size}\ n$", fontsize=12)
        ax.set_xlim(CURVE_SIZES[0] - 1, CURVE_SIZES[-1] + 1)
        ax.set_ylim(0, CURVE_YMAX)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"${tex_pct(v, 0)}$")
        )
        dense = np.linspace(8, CURVE_SIZES[-1] + 1, 200)
        ax.plot(dense, asymptotic_law(dense), color=s.muted, ls=":", lw=1.4, zorder=1)
        ax.text(
            0.97,
            0.93,
            f"dotted: law ({LAW_SLOPE:.3f} ln n − {LAW_OFFSET}) / n²\n"
            "● exact    ○ this game's estimate, 95% range",
            transform=ax.transAxes,
            color=s.muted,
            fontsize=10,
            linespacing=1.6,
            ha="right",
            va="top",
        )
        ax.plot(CURVE_SIZES, self.curve, color=s.muted, lw=1.0, alpha=0.7, zorder=2)
        ax.scatter(CURVE_SIZES, self.curve, s=9, color=s.muted, zorder=2)
        self.curve_marks = []
        for game, color in zip(self.games, self.colors):
            ax.scatter([game.size], [game.exact], s=60, color=color, zorder=3)
            err = ax.errorbar(
                [game.size + 0.9],
                [game.exact],
                yerr=[[0.0], [0.0]],
                fmt="o",
                ms=5.5,
                mfc=s.panel_bg,
                mec=color,
                mew=1.6,
                ecolor=color,
                elinewidth=1.4,
                capsize=2.5,
                zorder=4,
            )
            self.curve_marks.append(err)

    # ---------- per-frame updates ----------

    def _color(self, iso: bool) -> str:
        return self.s.iso if iso else self.s.scalene

    def _update_grid(self, ax: plt.Axes, game: GridGame, k: int) -> None:
        a = ax.artists_by_name  # type: ignore[attr-defined]
        tri = game.triangles[k - 1]
        iso = bool(game.is_isosceles[k - 1])
        color = self._color(iso)

        first = max(0, k - 1 - N_GHOSTS)
        ghosts = game.triangles[first : k - 1]
        alphas = np.linspace(0.08, 0.35, len(ghosts)) if len(ghosts) else []
        a["ghosts"].set_verts(list(ghosts))
        a["ghosts"].set_edgecolors(
            [
                matplotlib.colors.to_rgba(self._color(bool(g)), al)
                for g, al in zip(game.is_isosceles[first : k - 1], alphas)
            ]
        )
        a["fill"].set_verts([tri])
        a["fill"].set_facecolors([matplotlib.colors.to_rgba(color, 0.22)])
        closed = np.vstack([tri, tri[0]])
        a["edge"].set_data(closed[:, 0], closed[:, 1])
        a["edge"].set_color(color)
        if "glow" in a:
            a["glow"].set_data(closed[:, 0], closed[:, 1])
            a["glow"].set_color(color)
        a["pts"].set_offsets(tri)
        a["pts"].set_facecolors([color])

        a["round"].set_text(f"Round {k:,}: {'isosceles' if iso else 'scalene'}")
        a["round"].set_color(color)
        n_iso = int(game.is_isosceles[:k].sum())
        a["rate"].set_text(f"isosceles {n_iso:,} / {k:,} = {100 * n_iso / k:.3f}%")

    def _update_rate(self, k: int) -> None:
        x = np.append(self.rate_x[self.rate_x < k], k)
        for game, line, dot in zip(self.games, self.rate_lines, self.rate_dots):
            y = game.running_rate[x - 1]
            line.set_data(x, y)
            # No clamping: an off-scale estimate is clipped by the axes, not
            # drawn at the edge as if that were its value.
            dot.set_offsets([[k, y[-1]]])

    def _update_curve(self, k: int) -> None:
        for game, err in zip(self.games, self.curve_marks):
            est = float(game.running_rate[k - 1])
            half = float(ci_half_width(game.exact, k))
            marker, _caps, (bars,) = err.lines
            x = game.size + 0.9
            marker.set_data([x], [est])  # clipped by the axes when off-scale
            lo, hi = est - half, est + half
            bars.set_segments([[[x, lo], [x, hi]]])
            for cap, cy in zip(_caps, (lo, hi)):
                cap.set_data([x], [cy])

    def render(self, k: int) -> Image.Image:
        """Draw the figure for round k."""
        for ax, game in zip(self.ax_grids, self.games):
            self._update_grid(ax, game, k)
        self._update_rate(k)
        self._update_curve(k)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return Image.fromarray(buf[:, :, :3].copy())

    def close(self) -> None:
        """Release the matplotlib figure."""
        plt.close(self.fig)


def make_gif(
    games: list[GridGame], curve: np.ndarray, style: Style, out_dir: Path = OUTPUT_DIR
) -> Path:
    """Render and save the comparison GIF in one style; return its path."""
    rounds, durations = frame_schedule(n_rounds=len(games[0].running_rate))
    if len(rounds) > MAX_FRAMES:
        raise ValueError(f"{len(rounds)} frames exceeds LinkedIn's limit of {MAX_FRAMES}.")
    renderer = ComparisonRenderer(games, curve, style)
    frames = [renderer.render(int(k)) for k in rounds]
    renderer.close()
    frames = quantize_frames(frames, style, n_colors=N_COLORS)
    path = out_dir / f"isosceles_grid_comparison_{style.name}.gif"
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
    """Play the three showcase games and render every exported style."""
    games = [play_showcase_game(n, SHOWCASE_SEEDS[n]) for n in GRID_SIZES]
    curve = exact_curve(CURVE_SIZES)
    for style in STYLES:
        if style.name not in EXPORT_STYLES:
            continue
        path = make_gif(games, curve, style)
        with Image.open(path) as im:
            n_frames = im.n_frames
        size_mb = path.stat().st_size / 1e6
        warn = "" if size_mb < MAX_MB else f"  (over LinkedIn's {MAX_MB} MB)"
        print(f"{path.name}: {n_frames} frames, {size_mb:.2f} MB{warn}")


if __name__ == "__main__":
    main()
