"""Render the figures of the visual explainer docs/isosceles_10x10_visual.tex.

Every figure is saved as a vector PDF in docs/figures/explainer_10x10/, which the
LaTeX file includes. Pass ``--preview DIR`` to also write PNG copies to DIR for
quick viewing.

Usage::

    python make_explainer_figures.py [--preview DIR]
"""

from __future__ import annotations

import argparse
from math import comb, gcd, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from core.exact import collinear_triples, isosceles_triangles, total_triples  # noqa: E402
from core.grid import build_grid  # noqa: E402
from core.monte_carlo import draw_batch, monte_carlo_games  # noqa: E402

N = 10
OUT_DIR = Path("docs/figures/explainer_10x10")

Point = tuple[int, int]

# Colours (validated categorical slots 1-3 and the blue sequential ramp).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b9b8b2"
FAINT = "#e4e3df"
SURFACE = "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }
)


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------


def squared_distance(p: Point, q: Point) -> int:
    """Return the squared Euclidean distance between two lattice points."""
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def points_by_distance(n: int, apex: Point) -> dict[int, list[Point]]:
    """Group the points of the n x n grid (apex excluded) by squared distance to apex."""
    rings: dict[int, list[Point]] = {}
    for x in range(n):
        for y in range(n):
            if (x, y) != apex:
                rings.setdefault(squared_distance(apex, (x, y)), []).append((x, y))
    return dict(sorted(rings.items()))


def equidistant_pairs(n: int, apex: Point) -> int:
    """Return the number of unordered pairs of grid points at equal distance from apex."""
    return sum(comb(len(ring), 2) for ring in points_by_distance(n, apex).values())


def midpoint_pairs_at(n: int, apex: Point) -> int:
    """Return the number of pairs {B, C} in the grid whose midpoint is apex."""
    mx = min(apex[0], n - 1 - apex[0])
    my = min(apex[1], n - 1 - apex[1])
    return ((2 * mx + 1) * (2 * my + 1) - 1) // 2


def apex_triangle_counts(n: int) -> np.ndarray:
    """Return an (n, n) array whose [x, y] entry counts isosceles triangles with apex (x, y)."""
    counts = np.zeros((n, n), dtype=int)
    for x in range(n):
        for y in range(n):
            counts[x, y] = equidistant_pairs(n, (x, y)) - midpoint_pairs_at(n, (x, y))
    return counts


def interior_lattice_points(p: Point, q: Point) -> list[Point]:
    """Return the lattice points strictly between p and q on the segment pq."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    g = gcd(abs(dx), abs(dy))
    return [(p[0] + k * dx // g, p[1] + k * dy // g) for k in range(1, g)]


def running_rate(hits: np.ndarray) -> np.ndarray:
    """Return the running proportion of True values after each round."""
    return np.cumsum(hits) / np.arange(1, len(hits) + 1)


def is_isosceles_triangle(a: Point, b: Point, c: Point) -> bool:
    """Return True if a, b, c form a non-degenerate isosceles triangle."""
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if cross == 0:
        return False
    d = (squared_distance(a, b), squared_distance(b, c), squared_distance(c, a))
    return d[0] == d[1] or d[1] == d[2] or d[2] == d[0]


def apex_of(a: Point, b: Point, c: Point) -> Point:
    """Return the vertex equidistant from the other two (the triangle must be isosceles)."""
    for p, q, r in ((a, b, c), (b, c, a), (c, a, b)):
        if squared_distance(p, q) == squared_distance(p, r):
            return p
    raise ValueError("triangle is not isosceles")


def random_isosceles(n: int, count: int, seed: int) -> list[tuple[Point, Point, Point]]:
    """Draw random triples from the grid until `count` isosceles triangles are found.

    Each triangle is returned as (apex, b, c).
    """
    rng = np.random.default_rng(seed)
    found: list[tuple[Point, Point, Point]] = []
    while len(found) < count:
        idx = rng.choice(n * n, size=3, replace=False)
        a, b, c = (tuple(int(v) for v in divmod(int(i), n)) for i in idx)
        if is_isosceles_triangle(a, b, c):
            apex = apex_of(a, b, c)
            others = [p for p in (a, b, c) if p != apex]
            found.append((apex, others[0], others[1]))
    return found


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------


def board(ax: Axes, n: int = N, ticks: bool = False, dot_color: str = MUTED) -> None:
    """Draw the n x n board of dots on ax."""
    xs, ys = np.meshgrid(range(n), range(n))
    ax.scatter(xs, ys, s=14, color=dot_color, zorder=1, linewidths=0)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_aspect("equal")
    for side in ax.spines.values():
        side.set_visible(False)
    if ticks:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.tick_params(length=0, labelsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def dots(ax: Axes, pts: list[Point], color: str = INK, size: float = 46, z: int = 5) -> None:
    """Mark the given lattice points with large dots."""
    ax.scatter(
        [p[0] for p in pts],
        [p[1] for p in pts],
        s=size,
        color=color,
        zorder=z,
        edgecolors="white",
        linewidths=1.2,
    )


def segment(ax: Axes, p: Point, q: Point, color: str = BLUE, lw: float = 2, **kw: object) -> None:
    """Draw a straight segment from p to q."""
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, solid_capstyle="round", **kw)


def tick_mark(ax: Axes, p: Point, q: Point, count: int = 1, color: str = ORANGE) -> None:
    """Draw `count` short perpendicular ticks at the middle of pq (the 'equal sides' sign)."""
    p_arr, q_arr = np.array(p, float), np.array(q, float)
    mid = (p_arr + q_arr) / 2
    direction = (q_arr - p_arr) / np.linalg.norm(q_arr - p_arr)
    normal = np.array([-direction[1], direction[0]])
    for k in range(count):
        centre = mid + (k - (count - 1) / 2) * 0.22 * direction
        a, b = centre - 0.28 * normal, centre + 0.28 * normal
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.8, zorder=4)


def isosceles(ax: Axes, apex: Point, b: Point, c: Point, label_apex: bool = True) -> None:
    """Draw an isosceles triangle: equal legs in orange with ticks, base in blue."""
    segment(ax, b, c, BLUE)
    segment(ax, apex, b, ORANGE, lw=2.4)
    segment(ax, apex, c, ORANGE, lw=2.4)
    tick_mark(ax, apex, b)
    tick_mark(ax, apex, c)
    dots(ax, [b, c], INK)
    dots(ax, [apex], AQUA, size=70)
    if label_apex:
        away = np.array(apex, float) - (np.array(b, float) + np.array(c, float)) / 2
        away = 15 * away / np.linalg.norm(away)
        ax.annotate(
            "apex",
            apex,
            xytext=(away[0], away[1]),
            textcoords="offset points",
            ha="center",
            va="center",
            color=INK_2,
            fontsize=8.5,
            zorder=6,
        )


def note(ax: Axes, text: str, y: float = -0.02) -> None:
    """Write a short caption line under a panel."""
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", color=INK_2, fontsize=9)


def save(fig: Figure, name: str, preview: Path | None) -> None:
    """Save a figure as PDF (and PNG preview if requested), then close it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f"{name}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_board(preview: Path | None) -> None:
    """The 10 x 10 board of dots."""
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    board(ax, ticks=True, dot_color=INK_2)
    ax.set_title("100 dots, at whole-number positions 0 to 9")
    save(fig, "fig01_board", preview)


def fig_three_draws(preview: Path | None) -> None:
    """Three example draws: isosceles, not isosceles, collinear."""
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    board(axes[0])
    isosceles(axes[0], (1, 1), (5, 2), (2, 5))
    axes[0].set_title("Isosceles: we win")
    note(axes[0], "two sides have the same length")

    a, b, c = (1, 2), (8, 1), (5, 7)
    board(axes[1])
    for p, q in ((a, b), (b, c), (c, a)):
        segment(axes[1], p, q, BLUE)
    dots(axes[1], [a, b, c])
    axes[1].set_title("Not isosceles")
    note(axes[1], "all three sides are different")

    a, b, c = (1, 1), (4, 3), (7, 5)
    board(axes[2])
    segment(axes[2], a, c, MUTED, lw=2, ls="--")
    dots(axes[2], [a, b, c])
    axes[2].set_title("On one line: draw again")
    note(axes[2], "not a triangle at all")
    fig.tight_layout()
    save(fig, "fig02_three_draws", preview)


def fig_isosceles_gallery(preview: Path | None) -> None:
    """Four isosceles triangles of different shapes."""
    examples = [
        ((5, 6), (2, 1), (8, 1), "pointing up"),
        ((2, 2), (7, 2), (2, 7), "with a right angle"),
        ((4, 4), (7, 8), (9, 4), "tilted"),
        ((0, 0), (9, 2), (7, 6), "long and thin"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.7))
    for ax, (apex, b, c, title) in zip(axes, examples):
        board(ax)
        isosceles(ax, apex, b, c)
        ax.set_title(title)
        note(ax, f"legs$^2$ = {squared_distance(apex, b)} and {squared_distance(apex, c)}")
    fig.tight_layout()
    save(fig, "fig03_isosceles_shapes", preview)


def fig_pythagoras(preview: Path | None) -> None:
    """Measuring a side with Pythagoras; squared lengths are whole numbers."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (p, q) in zip(axes, (((1, 1), (5, 4)), ((2, 2), (5, 4)))):
        board(ax)
        corner = (q[0], p[1])
        segment(ax, p, corner, INK_2, lw=1.4, ls="--")
        segment(ax, corner, q, INK_2, lw=1.4, ls="--")
        segment(ax, p, q, BLUE, lw=2.6)
        dots(ax, [p, q])
        dx, dy = q[0] - p[0], q[1] - p[1]
        ax.text((p[0] + q[0]) / 2, p[1] - 0.45, f"{dx} across", ha="center", color=INK_2)
        ax.text(q[0] + 0.25, (p[1] + q[1]) / 2, f"{dy} up", va="center", color=INK_2)
        d2 = dx * dx + dy * dy
        root = (
            f"{int(sqrt(d2))}"
            if int(sqrt(d2)) ** 2 == d2
            else f"$\\sqrt{{{d2}}}\\approx${sqrt(d2):.2f}"
        )
        ax.set_title(f"length$^2$ = {dx}$^2$ + {dy}$^2$ = {d2}")
        note(ax, f"length = {root}")
    fig.tight_layout()
    save(fig, "fig04_pythagoras", preview)


def fig_collinear(preview: Path | None) -> None:
    """Three kinds of collinear triples."""
    cases = [
        (((1, 4), (4, 4), (8, 4)), "along a row"),
        (((1, 1), (4, 4), (7, 7)), "along a diagonal"),
        (((1, 1), (4, 3), (7, 5)), "along a slanted line"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    for ax, (pts, title) in zip(axes, cases):
        board(ax)
        p, q = np.array(pts[0], float), np.array(pts[2], float)
        d = (q - p) / np.linalg.norm(q - p)
        a, b = p - 0.6 * d, q + 0.6 * d
        ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=2, ls="--", zorder=2)
        dots(ax, list(pts))
        ax.set_title(title)
    fig.tight_layout()
    save(fig, "fig05_collinear", preview)


def fig_segment_dots(preview: Path | None) -> None:
    """How many dots lie strictly between two dots: gcd of the steps minus one."""
    cases = [((1, 1), (7, 5)), ((1, 1), (8, 8)), ((1, 1), (6, 4))]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2))
    for ax, (p, q) in zip(axes, cases):
        board(ax)
        segment(ax, p, q, BLUE, lw=2)
        inner = interior_lattice_points(p, q)
        dots(ax, [p, q])
        if inner:
            dots(ax, inner, ORANGE, size=56)
        dx, dy = q[0] - p[0], q[1] - p[1]
        g = gcd(dx, dy)
        ax.set_title(f"steps ({dx}, {dy}),  gcd = {g}")
        word = "dot" if g - 1 == 1 else "dots"
        note(ax, f"{g - 1} {word} in between" if g > 1 else "no dot in between")
    fig.tight_layout()
    save(fig, "fig06_dots_between", preview)


def fig_collinear_bars(preview: Path | None) -> None:
    """The 4448 collinear triples split by the direction of their line."""
    rows = 10 * comb(10, 3)
    diag = 2 * sum((N - a) * (N - a) * (a - 1) for a in range(2, N))
    other = collinear_triples(N, N) - 2 * rows - diag
    labels = ["horizontal", "vertical", "diagonal (45°)", "other slopes"]
    values = [rows, rows, diag, other]
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    bars = ax.barh(labels[::-1], values[::-1], color=BLUE, height=0.6)
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 25, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center", color=INK)
    ax.set_xlim(0, 1450)
    ax.set_xlabel("collinear triples")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"All {sum(values):,} collinear triples, by direction", loc="left")
    fig.tight_layout()
    save(fig, "fig07_collinear_count", preview)


def fig_no_equilateral(preview: Path | None) -> None:
    """The third corner of an equilateral triangle never lands on a dot."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    b, c = (2, 2), (6, 2)
    top = (4.0, 2 + 2 * sqrt(3))
    ax = axes[0]
    board(ax)
    segment(ax, b, c, BLUE)
    ax.plot([b[0], top[0], c[0]], [b[1], top[1], c[1]], color=INK_2, lw=1.4, ls="--")
    ax.scatter(*top, s=80, facecolors="white", edgecolors=ORANGE, linewidths=2, zorder=6)
    dots(ax, [b, c])
    ax.annotate(
        "needed corner\n(4, 5.46...)",
        top,
        xytext=(16, 14),
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
    )
    ax.set_title("Equilateral needs (4, 5.46...)")
    note(ax, "that is not a dot of the board")

    ax = axes[1]
    board(ax)
    segment(ax, b, c, BLUE)
    for apex, colour, dy in (((4, 5), AQUA, -10), ((4, 6), ORANGE, 10)):
        segment(ax, apex, b, colour, lw=2)
        segment(ax, apex, c, colour, lw=2)
        dots(ax, [apex], colour, size=60)
        ax.annotate(
            f"sides$^2$: 16, {squared_distance(apex, b)}, {squared_distance(apex, c)}",
            apex,
            xytext=(-62, dy),
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
            textcoords="offset points",
            color=INK_2,
            fontsize=8.5,
        )
    dots(ax, [b, c])
    ax.set_title("The nearest dots miss")
    note(ax, "isosceles, but never all three equal")
    fig.tight_layout()
    save(fig, "fig08_no_equilateral", preview)


def fig_apex_circle(preview: Path | None) -> None:
    """The apex is the centre of a circle through the two other corners."""
    apex = (3, 3)
    ring = points_by_distance(N, apex)[20]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for k, ax in enumerate(axes):
        board(ax)
        ax.add_patch(
            plt.Circle(
                apex, sqrt(20), fill=False, ls="--", lw=1.3, color=INK_2, zorder=2, clip_on=False
            )
        )
        if k == 0:
            isosceles(ax, apex, (7, 5), (1, 7))
            ax.set_title("The apex is the centre of a circle")
            note(ax, "both other corners sit on it", y=-0.15)
        else:
            for i in range(len(ring)):
                for j in range(i + 1, len(ring)):
                    xs = [apex[0], ring[i][0], ring[j][0], apex[0]]
                    ys = [apex[1], ring[i][1], ring[j][1], apex[1]]
                    ax.fill(xs, ys, color=BLUE, alpha=0.10, zorder=2)
                    ax.plot(xs, ys, color=BLUE, lw=1, alpha=0.7, zorder=3)
            dots(ax, ring, ORANGE, size=56)
            dots(ax, [apex], AQUA, size=70)
            ax.set_title(f"{len(ring)} dots on the circle")
            note(ax, f"every pair is a triangle: {comb(len(ring), 2)} triangles", y=-0.15)
    fig.tight_layout()
    save(fig, "fig09_apex_circle", preview)


def fig_distance_numbers(preview: Path | None) -> None:
    """Each dot labelled with its squared distance from a chosen apex."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))
    for ax, apex, hi in zip(axes, ((0, 0), (4, 4)), (25, 5)):
        board(ax, dot_color="white")
        for x in range(N):
            for y in range(N):
                if (x, y) == apex:
                    dots(ax, [apex], AQUA, size=150)
                    continue
                d = squared_distance(apex, (x, y))
                special = d == hi
                ax.text(
                    x,
                    y,
                    str(d),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if special else INK_2,
                    fontweight="bold" if special else "normal",
                    zorder=5,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        fc=ORANGE if special else "white",
                        ec="none",
                    ),
                )
        k = len(points_by_distance(N, apex)[hi])
        ax.set_title(f"apex {apex}: {k} dots at distance$^2$ {hi}")
    fig.tight_layout()
    save(fig, "fig10_distance_numbers", preview)


def fig_corner_pairs(preview: Path | None) -> None:
    """Corner apex: 45 mirror pairs and the four special circles."""
    apex = (0, 0)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    ax = axes[0]
    board(ax)
    ax.plot([0, 9], [0, 9], color=MUTED, lw=1.2, ls="--", zorder=2)
    above = [(x, y) for x in range(N) for y in range(N) if y > x]
    dots(ax, above, BLUE, size=22, z=3)
    for a, b in ((0, 3), (1, 6), (2, 8), (4, 9)):
        segment(ax, (a, b), (b, a), ORANGE, lw=1.4, ls=":")
        dots(ax, [(a, b), (b, a)], ORANGE, size=50)
    dots(ax, [apex], AQUA, size=70)
    ax.set_title("45 mirror pairs")
    note(ax, "each blue dot (a, b) has a twin (b, a), equally far")

    ax = axes[1]
    board(ax)
    rings = points_by_distance(N, apex)
    theta = np.linspace(0, np.pi / 2, 200)
    for r2 in (25, 50, 65, 85):
        r = sqrt(r2)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=INK_2, lw=1, ls="--", zorder=2)
        dots(ax, rings[r2], ORANGE, size=48)
        ax.text(-0.35, r, str(r2), color=INK_2, fontsize=8, ha="right", va="center")
    dots(ax, [apex], AQUA, size=70)
    ax.set_xlim(-1.6, N - 0.3)
    ax.set_title("4 circles with extra pairs")
    note(ax, "25, 50, 65, 85: 14 more pairs")
    fig.tight_layout()
    save(fig, "fig11_corner_pairs", preview)


def fig_ring_sizes(preview: Path | None) -> None:
    """How many dots sit on each circle, for a corner apex and a central apex."""
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 4.2), sharex=True, sharey=True)
    for ax, apex in zip(axes, ((0, 0), (4, 4))):
        rings = points_by_distance(N, apex)
        r2 = list(rings)
        k = [len(v) for v in rings.values()]
        ax.vlines(r2, 0, k, color=BLUE, lw=2)
        ax.scatter(r2, k, s=12, color=BLUE, zorder=3)
        ax.set_ylim(0, 11)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        pairs = equidistant_pairs(N, apex)
        ax.set_title(f"apex {apex}: {len(rings)} circles, {pairs} equal-distance pairs", loc="left")
        ax.set_ylabel("dots on circle")
    axes[1].set_xlabel("squared radius of the circle")
    fig.tight_layout()
    save(fig, "fig12_ring_sizes", preview)


def fig_midpoint(preview: Path | None) -> None:
    """Equal distances on a straight line are not triangles."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ax = axes[0]
    board(ax)
    a, b, c = (4, 4), (1, 2), (7, 6)
    segment(ax, b, c, ORANGE, lw=2.4)
    tick_mark(ax, a, b)
    tick_mark(ax, a, c)
    dots(ax, [b, c])
    dots(ax, [a], AQUA, size=70)
    ax.set_title("Equal distances, but a straight line")
    note(ax, "the apex is exactly the midpoint")

    ax = axes[1]
    board(ax)
    a = (2, 3)
    mx, my = min(a[0], N - 1 - a[0]), min(a[1], N - 1 - a[1])
    ax.add_patch(
        plt.Rectangle(
            (a[0] - mx - 0.4, a[1] - my - 0.4),
            2 * mx + 0.8,
            2 * my + 0.8,
            color=BLUE,
            alpha=0.12,
            zorder=0,
        )
    )
    for x in range(a[0] - mx, a[0] + mx + 1):
        for y in range(a[1] - my, a[1] + my + 1):
            if (x, y) < a:
                segment(ax, (x, y), (2 * a[0] - x, 2 * a[1] - y), BLUE, lw=0.8, alpha=0.5)
    dots(ax, [a], AQUA, size=70)
    ax.set_title(f"apex {a}: {midpoint_pairs_at(N, a)} straight pairs")
    note(ax, "mirror images through the apex, inside the box")
    fig.tight_layout()
    save(fig, "fig13_midpoint", preview)


def fig_heatmap(preview: Path | None) -> None:
    """Number of isosceles triangles having each dot as apex."""
    counts = apex_triangle_counts(N)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    im = ax.imshow(counts.T, origin="lower", cmap=cmap, vmin=0, vmax=counts.max())
    for x in range(N):
        for y in range(N):
            v = counts[x, y]
            ax.text(
                x,
                y,
                str(v),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if v > 0.55 * counts.max() else INK,
            )
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.tick_params(length=0, labelsize=8)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_label("isosceles triangles with this apex")
    ax.set_title(f"Total over all 100 dots: {counts.sum():,}")
    save(fig, "fig14_apex_heatmap", preview)


def fig_waffle(preview: Path | None) -> None:
    """Where the 161,700 triples go, and the share of isosceles triangles."""
    total = total_triples(N, N)
    coll = collinear_triples(N, N)
    iso = isosceles_triangles(N, N)
    other = total - coll - iso
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    labels = ["not isosceles", "isosceles", "collinear (redrawn)"]
    values = [other, iso, coll]
    colours = [MUTED, ORANGE, INK_2]
    bars = ax.barh(labels[::-1], values[::-1], color=colours[::-1], height=0.6)
    for bar, v in zip(bars, values[::-1]):
        ax.text(v + 1500, bar.get_y() + bar.get_height() / 2, f"{v:,}", va="center")
    ax.set_xlim(0, 185_000)
    ax.set_xticks([0, 50_000, 100_000, 150_000])
    ax.set_xticklabels(["0", "50k", "100k", "150k"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"All {total:,} ways to pick 3 dots", loc="left")

    ax = axes[1]
    share = 100 * iso / (total - coll)
    full = int(share)
    for i in range(100):
        x, y = i % 10, 9 - i // 10
        colour = ORANGE if i < full else (FAINT if i > full else None)
        if colour is None:  # the partly filled square
            ax.add_patch(plt.Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, color=FAINT))
            frac = share - full
            ax.add_patch(plt.Rectangle((x - 0.42, y - 0.42), 0.84 * frac, 0.84, color=ORANGE))
        else:
            ax.add_patch(plt.Rectangle((x - 0.42, y - 0.42), 0.84, 0.84, color=colour))
    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(-0.6, 9.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Out of 100 real triangles: {share:.2f} isosceles")
    fig.tight_layout()
    save(fig, "fig15_where_triples_go", preview)


def fig_running(preview: Path | None) -> None:
    """Running isosceles rate of three simulated games."""
    exact = isosceles_triangles(N, N) / (total_triples(N, N) - collinear_triples(N, N))
    grid = build_grid(N, N)
    rng = np.random.default_rng(2024)
    rounds = 200_000
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for colour, name in zip((BLUE, ORANGE, AQUA), ("game 1", "game 2", "game 3")):
        hits, _ = draw_batch(grid, rng, rounds, resample_degenerate=True)
        rate = running_rate(hits)
        xs = np.unique(np.geomspace(1, rounds, 1500).astype(int))
        ax.plot(xs, rate[xs - 1], color=colour, lw=1.6, label=name)
    ax.axhline(exact, color=INK, lw=1.2, ls="--")
    ax.text(rounds * 1.05, exact, f"exact\n{exact:.4f}", va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(20, rounds)
    ax.set_ylim(0, 0.16)
    ax.set_xlabel("rounds played (log scale)")
    ax.set_ylabel("share that were isosceles")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Each game wobbles at first, then settles on the exact value", loc="left")
    fig.tight_layout()
    save(fig, "fig16_running_rate", preview)


def fig_histogram(preview: Path | None) -> None:
    """Histogram of many game estimates around the exact value."""
    exact = isosceles_triangles(N, N) / (total_triples(N, N) - collinear_triples(N, N))
    games, rounds = 2000, 100_000
    est = monte_carlo_games(N, N, games, rounds, seed=7)
    sd = sqrt(exact * (1 - exact) / rounds)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    bins = np.linspace(exact - 4.5 * sd, exact + 4.5 * sd, 41)
    ax.hist(est, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8)
    xs = np.linspace(bins[0], bins[-1], 400)
    width = bins[1] - bins[0]
    curve = games * width * np.exp(-0.5 * ((xs - exact) / sd) ** 2) / (sd * sqrt(2 * np.pi))
    ax.plot(xs, curve, color=ORANGE, lw=2, label="bell curve predicted by theory")
    ax.axvline(exact, color=INK, lw=1.2, ls="--")
    ax.text(exact, ax.get_ylim()[1] * 1.02, f"exact {exact:.4f}", ha="center", fontsize=8.5)
    ax.set_xlabel(f"estimate from one game of {rounds:,} rounds")
    ax.set_ylabel("number of games")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"{games:,} games, each giving its own estimate", loc="left", pad=16)
    fig.tight_layout()
    save(fig, "fig17_histogram", preview)


def fig_random_gallery(preview: Path | None) -> None:
    """A dozen isosceles triangles found by random draws."""
    found = random_isosceles(N, 12, seed=11)
    fig, axes = plt.subplots(3, 4, figsize=(8.0, 6.3))
    for ax, (apex, b, c) in zip(axes.ravel(), found):
        board(ax)
        isosceles(ax, apex, b, c, label_apex=False)
        ax.set_title(f"legs$^2$ = {squared_distance(apex, b)}", fontsize=9, fontweight="normal")
    fig.tight_layout()
    save(fig, "fig18_random_isosceles", preview)


FIGURES = [
    fig_board,
    fig_three_draws,
    fig_isosceles_gallery,
    fig_pythagoras,
    fig_collinear,
    fig_segment_dots,
    fig_collinear_bars,
    fig_no_equilateral,
    fig_apex_circle,
    fig_distance_numbers,
    fig_corner_pairs,
    fig_ring_sizes,
    fig_midpoint,
    fig_heatmap,
    fig_waffle,
    fig_running,
    fig_histogram,
    fig_random_gallery,
]


def main() -> None:
    """Render every figure of the explainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, default=None, help="also write PNGs here")
    args = parser.parse_args()
    for make in FIGURES:
        make(args.preview)
        print("done:", make.__name__)


if __name__ == "__main__":
    main()
