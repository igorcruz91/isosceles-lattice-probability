"""Re-create the LinkedIn GIFs of the isosceles-triangle experiment with ManimGX.

Same data, same three panels as ``make_linkedin_gifs.py`` (matplotlib), so the two
can be compared side by side:

* left   - the lattice grid, with the triangle of each round morphing into the next
           (plus fading ghosts of the previous rounds);
* middle - the running isosceles-rate estimate of one game converging to the exact p;
* right  - the Monte Carlo histogram of the estimates of many independent games.

ManimGX (https://manimgx.academa.ai) renders on the GPU and writes MP4 only, so each
frame is also captured through ``Scene.render(frames=...)`` and written to a GIF with
Pillow. Frames that the scene holds still are sent once with a ``repeat`` count, which
becomes that GIF frame's duration.

Usage::

    python make_manimgx_gif.py            # both styles
    python make_manimgx_gif.py neon       # one style
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import manimgx as m
import numpy as np
from PIL import Image

from make_linkedin_gifs import (
    GRID_SIZE,
    N_GHOSTS,
    N_ROUNDS,
    OUTPUT_DIR,
    SLOW_ROUNDS,
    STYLES,
    Experiment,
    Style,
    build_experiment,
    quantize_frames,
    tex_pct,
)

PIXEL_WIDTH, PIXEL_HEIGHT = 1200, 460  # same size as the matplotlib GIFs
FPS = 20
INTRO_S = 1.6
SLOW_ROUND_S = 0.45
FAST_S = 7.0
HOLD_S = 3.5
LOG_MAX = np.log10(N_ROUNDS * 1.6)  # right end of the log-scale x-axis
N_BINS = 40
# matplotlib points -> font_size: em = pt * 100/72 px, 57.5 px per unit, em = 0.0131 * font_size
PT = 1.84
# MathTex (Typst, New Computer Modern) sets glyphs about 1.45x smaller than Text at one font_size
MPT = PT * 1.45


@dataclass(frozen=True)
class Box:
    """A panel's rectangle in scene units."""

    left: float
    bottom: float
    width: float
    height: float

    @property
    def right(self) -> float:
        """The x of the right edge."""
        return self.left + self.width

    @property
    def top(self) -> float:
        """The y of the top edge."""
        return self.bottom + self.height


# Panel rectangles, converted from the matplotlib gridspec of make_linkedin_gifs.py.
GRID_BOX = Box(-9.80, -2.705, 4.77, 4.77)
RATE_BOX = Box(-3.47, -2.88, 5.97, 5.12)
HIST_BOX = Box(4.30, -2.88, 5.72, 5.12)
CELL = GRID_BOX.width / GRID_SIZE


# ---------- pure data helpers (unit tested) ----------


def grid_to_scene(xy: np.ndarray) -> np.ndarray:
    """Map lattice coordinates (..., 2) to scene points (..., 3) inside GRID_BOX."""
    xy = np.asarray(xy, dtype=float)
    x = GRID_BOX.left + (xy[..., 0] + 0.6) * CELL
    y = GRID_BOX.bottom + (xy[..., 1] + 0.6) * CELL
    return np.stack([x, y, np.zeros_like(x)], axis=-1)


def round_at(log_k: float) -> int:
    """The round shown when the round tracker holds log10(k): the next whole round."""
    return int(np.clip(np.ceil(10**log_k - 1e-9), 1, N_ROUNDS))


def games_at(fraction: float, n_games: int) -> int:
    """Monte Carlo games revealed at a fraction of the animation (same curve as the GIF)."""
    return max(1, int(round(n_games * float(np.clip(fraction, 0, 1)) ** 1.6)))


def rate_path(rate: np.ndarray, sample: np.ndarray, log_k: float) -> np.ndarray:
    """Points (log10 n, rate) of the running estimate up to a fractional log10 round.

    Uses the log-spaced `sample` of rounds below k, and ends at log_k with the rate
    interpolated between the two neighbouring whole rounds, so the line grows smoothly.
    """
    k = 10**log_k
    lo = int(np.clip(np.floor(k + 1e-9), 1, len(rate)))
    hi = min(lo + 1, len(rate))
    frac = 0.0 if hi == lo else (k - lo)
    end = rate[lo - 1] + (rate[hi - 1] - rate[lo - 1]) * frac
    xs = sample[sample <= lo]
    pts = np.column_stack([np.log10(xs), rate[xs - 1]])
    return np.vstack([pts, [log_k, end]])


def ci_band(p: float, n_points: int = 300, y_max: float = 0.4) -> np.ndarray:
    """Polygon (log10 n, rate) of p ± 1.96 sqrt(p(1-p)/n), clipped to [0, y_max]."""
    log_n = np.linspace(0, LOG_MAX, n_points)
    half = 1.96 * np.sqrt(p * (1 - p) / 10**log_n)
    upper = np.clip(p + half, 0, y_max)
    lower = np.clip(p - half, 0, y_max)
    return np.vstack([np.column_stack([log_n, upper]), np.column_stack([log_n, lower])[::-1]])


def step_outline(bins: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Corner points (x, y) of a histogram drawn as a closed step outline on y = 0."""
    edges = np.repeat(bins, 2)
    heights = np.concatenate([[0], np.repeat(counts, 2), [0]])
    return np.column_stack([edges, heights])


def tex_int(n: int) -> str:
    """Integer with thin-space thousands separators for MathTex, e.g. 10 000.

    Typst sets a math comma with a space after it ("10, 000"), so a thin space is used.
    """
    return f"{n:,}".replace(",", r"\,")


def frame_ms(repeat: int, fps: int = FPS) -> int:
    """GIF duration of a frame shown `repeat` times, in ms (GIF delays are 10 ms steps)."""
    return max(20, int(round(repeat * 1000 / fps / 10)) * 10)


# ---------- scene ----------


class PanelAxes:
    """Axes placed in a Box, with hand-made ticks and labels like the matplotlib panels."""

    def __init__(
        self, box: Box, x_range: tuple[float, float], y_range: tuple[float, float], style: Style
    ) -> None:
        self.box = box
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.s = style

    def c2p(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        """Data coordinates to scene points; arrays give an (n, 3) array."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sx = self.box.left + (x - self.x0) / (self.x1 - self.x0) * self.box.width
        sy = self.box.bottom + (y - self.y0) / (self.y1 - self.y0) * self.box.height
        return np.stack([sx, sy, np.zeros_like(sx)], axis=-1)

    def frame(self, x_ticks: list[tuple[float, str]], y_ticks: list[tuple[float, str]]) -> m.VGroup:
        """Spines, ticks, tick labels and (for the clean style) a light grid."""
        s = self.s
        b = self.box
        parts = m.VGroup()
        if s.show_axes_grid:
            for x, _ in x_ticks:
                p = self.c2p(x, self.y0)[0]
                parts.add(
                    m.Line([p, b.bottom, 0], [p, b.top, 0], color="#e5e7eb", stroke_width=1.4)
                )
            for y, _ in y_ticks:
                q = self.c2p(self.x0, y)[1]
                parts.add(
                    m.Line([b.left, q, 0], [b.right, q, 0], color="#e5e7eb", stroke_width=1.4)
                )
        spines = m.VGroup(
            m.Line([b.left, b.bottom, 0], [b.right, b.bottom, 0]),
            m.Line([b.left, b.bottom, 0], [b.left, b.top, 0]),
        ).set_stroke(s.muted, width=1.7)
        parts.add(spines)
        for x, label in x_ticks:
            p = self.c2p(x, self.y0)
            parts.add(m.Line(p, p + 0.08 * m.DOWN, color=s.muted, stroke_width=1.7))
            parts.add(
                m.MathTex(label, font_size=11 * MPT, color=s.muted).next_to(p, m.DOWN, buff=0.13)
            )
        for y, label in y_ticks:
            p = self.c2p(self.x0, y)
            parts.add(m.Line(p, p + 0.08 * m.LEFT, color=s.muted, stroke_width=1.7))
            parts.add(
                m.MathTex(label, font_size=11 * MPT, color=s.muted).next_to(p, m.LEFT, buff=0.13)
            )
        return parts


def mixed(parts: list[tuple[str, str]], color: str, size: float) -> m.VGroup:
    """A line of plain-text and math pieces: [("t", "words"), ("m", r"\\hat{p}")]."""
    mobs = [
        (
            m.Text(t, font="sans-serif", font_size=size, color=color)
            if kind == "t"
            else m.MathTex(t, font_size=size * MPT / PT, color=color)
        )
        for kind, t in parts
    ]
    return m.VGroup(*mobs).arrange(m.RIGHT, buff=0.1)


def make_scene_class(exp: Experiment, style: Style) -> type[m.Scene]:
    """Build the ManimGX scene for one style (config must already be set)."""
    s = style
    p = exp.exact
    rate = exp.running_rate
    sample = np.unique(np.geomspace(1, N_ROUNDS, 2500).round().astype(int))
    est = exp.game_estimates
    n_games = len(est)
    bins = np.linspace(est.min(), est.max(), N_BINS + 1)
    step = bins[1] - bins[0]
    full_counts, _ = np.histogram(est, bins=bins)

    def color_of(iso: bool) -> str:
        return s.iso if iso else s.scalene

    def triangle(k: int) -> m.VGroup:
        """The triangle of round k: fill, (glow), edge and vertex dots."""
        pts = grid_to_scene(exp.triangles[k - 1])
        c = color_of(bool(exp.is_isosceles[k - 1]))
        group = m.VGroup(m.Polygon(*pts, color=c, fill_opacity=0.22, stroke_width=0))
        if s.line_glow:
            group.add(m.Polygon(*pts, color=c, stroke_width=22, stroke_opacity=0.25))
        group.add(m.Polygon(*pts, color=c, stroke_width=6.3))
        group.add(*(m.Dot(q, radius=0.082, color=c).set_stroke(s.panel_bg, width=2.9) for q in pts))
        return group

    class IsoscelesManimGX(m.Scene):
        def construct(self) -> None:
            log_k = m.ValueTracker(0.0)  # log10 of the round shown
            shown = m.ValueTracker(0.0)  # fraction of Monte Carlo games revealed

            # ----- titles -----
            title = m.Text(
                "What are the odds that 3 random grid points form an isosceles triangle?",
                font="sans-serif",
                weight=m.BOLD,
                font_size=16 * PT,
                color=s.fg,
            )
            title.move_to([-9.80, 3.44, 0], aligned_edge=m.LEFT)
            subtitle = mixed(
                [
                    (
                        "t",
                        f"{GRID_SIZE}×{GRID_SIZE} grid · 3 distinct points per round · "
                        "collinear draws are redrawn · exact value",
                    ),
                    ("m", rf"p = {tex_pct(p, 2)}"),
                ],
                s.muted,
                10.5 * PT,
            )
            subtitle.move_to([-9.80, 2.92, 0], aligned_edge=m.LEFT)

            # ----- grid panel -----
            lattice = grid_to_scene(exp.grid)
            dots = m.VGroup(*(m.Dot(q, radius=0.045, color=s.grid_dot) for q in lattice))

            def ghosts() -> m.VGroup:
                k = round_at(log_k.get_value())
                first = max(0, k - 1 - N_GHOSTS)
                tris = exp.triangles[first : k - 1]
                isos = exp.is_isosceles[first : k - 1]
                alphas = np.linspace(0.08, 0.35, len(tris)) if len(tris) else []
                return m.VGroup(
                    *(
                        m.Polygon(
                            *grid_to_scene(t),
                            color=color_of(bool(i)),
                            stroke_width=2.9,
                            stroke_opacity=float(a),
                        )
                        for t, i, a in zip(tris, isos, alphas)
                    )
                )

            def round_label() -> m.Text:
                k = round_at(log_k.get_value())
                iso = bool(exp.is_isosceles[k - 1])
                label = m.Text(
                    f"Round {k:,}: {'isosceles' if iso else 'scalene'}",
                    font="sans-serif",
                    weight=m.BOLD,
                    font_size=12 * PT,
                    color=color_of(iso),
                )
                return label.move_to([GRID_BOX.left + GRID_BOX.width / 2, -3.15, 0])

            def tally() -> m.Text:
                k = round_at(log_k.get_value())
                n_iso = int(exp.is_isosceles[:k].sum())
                label = m.Text(
                    f"isosceles {n_iso:,}  ·  scalene {k - n_iso:,}",
                    font="sans-serif",
                    font_size=9.5 * PT,
                    color=s.muted,
                )
                return label.move_to([GRID_BOX.left + GRID_BOX.width / 2, -3.58, 0])

            # ----- running-estimate panel -----
            ra = PanelAxes(RATE_BOX, (0, LOG_MAX), (0, 0.4), s)
            rate_frame = ra.frame(
                [(e, f"10^{{{e}}}") for e in range(7)],
                [(v / 100, rf"{v}\%") for v in range(0, 45, 5)],
            )
            band = m.VMobject(stroke_width=0).set_points_as_corners(ra.c2p(*ci_band(p).T))
            band.set_fill(s.band, opacity=0.9 if s.name != "neon" else 0.6).set_stroke(width=0)
            p_line = m.DashedLine(
                ra.c2p(0, p), ra.c2p(LOG_MAX, p), color=s.muted, stroke_width=2.9, dash_length=0.1
            )
            p_label = m.MathTex(rf"p\\{tex_pct(p, 2)}", font_size=10 * MPT, color=s.muted).next_to(
                ra.c2p(LOG_MAX, p), m.RIGHT, buff=0.1
            )
            band_note = m.MathTex(
                r"\mathrm{shaded:}\ p \pm 1.96\sqrt{p(1-p)/n}", font_size=11 * MPT, color=s.muted
            ).move_to(ra.c2p(0, 0) + np.array([0.18, 0.2, 0]), aligned_edge=m.DL)
            rate_title = mixed(
                [("t", "Running estimate"), ("m", r"\hat{p}_n"), ("t", "of one game")],
                s.fg,
                12 * PT,
            ).move_to([RATE_BOX.left, RATE_BOX.top + 0.3, 0], aligned_edge=m.LEFT)
            rate_xlabel = m.MathTex(
                r"n\ \mathrm{(rounds,\ log\ scale)}", font_size=14 * MPT, color=s.fg
            ).move_to([RATE_BOX.left + RATE_BOX.width / 2, -3.62, 0])
            rate_ylabel = (
                m.MathTex(r"\hat{p}_n", font_size=16 * MPT, color=s.fg)
                .rotate(90 * m.DEGREES)
                .move_to([RATE_BOX.left - 1.2, RATE_BOX.bottom + RATE_BOX.height / 2, 0])
            )

            def rate_curve() -> m.VGroup:
                path = ra.c2p(*rate_path(rate, sample, log_k.get_value()).T)
                group = m.VGroup()
                if len(path) < 2 or np.allclose(path[0], path[-1]):
                    path = np.vstack([path[0], path[0] + 1e-4 * m.RIGHT])
                widths = ((16, 0.05), (10, 0.1), (7, 0.2)) if s.line_glow else ()
                for w, a in widths:
                    group.add(
                        m.VMobject(
                            stroke_color=s.accent, stroke_width=w, stroke_opacity=a
                        ).set_points_as_corners(path)
                    )
                group.add(
                    m.VMobject(stroke_color=s.accent, stroke_width=4.8).set_points_as_corners(path)
                )
                group.add(
                    m.Dot(path[-1], radius=0.085, color=s.accent).set_stroke(s.panel_bg, width=2.9)
                )
                return group

            def rate_value() -> m.MathTex:
                y = rate_path(rate, sample, log_k.get_value())[-1, 1]
                return m.MathTex(
                    rf"\hat{{p}}_n = {tex_pct(y, 3)}", font_size=16 * MPT, color=s.accent
                ).move_to([RATE_BOX.right - 0.15, RATE_BOX.top - 0.45, 0], aligned_edge=m.RIGHT)

            # ----- Monte Carlo histogram panel -----
            x_lo, x_hi = bins[0] - step, bins[-1] + step
            y_hi = full_counts.max() * 1.15
            ha = PanelAxes(HIST_BOX, (x_lo, x_hi), (0, y_hi), s)
            x_ticks = [
                (v / 10000, rf"{v / 100:.2f}\%")
                for v in range(int(np.ceil(x_lo * 2000)) * 5, int(x_hi * 10000) + 1, 5)
            ]
            hist_frame = ha.frame(x_ticks, [(v, tex_int(v)) for v in range(0, int(y_hi) + 1, 200)])
            exact_line = m.DashedLine(
                ha.c2p(p, 0), ha.c2p(p, y_hi), color=s.muted, stroke_width=2.9, dash_length=0.1
            )
            hist_title = mixed(
                [
                    ("t", "Monte Carlo:"),
                    ("m", tex_int(n_games)),
                    ("t", "games"),
                    ("m", rf"\times\ 10^{{{int(np.log10(N_ROUNDS))}}}"),
                    ("t", "rounds"),
                ],
                s.fg,
                12 * PT,
            ).move_to([HIST_BOX.left, HIST_BOX.top + 0.3, 0], aligned_edge=m.LEFT)
            hist_xlabel = m.MathTex(
                r"\hat{p}\ \mathrm{of\ each\ game}", font_size=14 * MPT, color=s.fg
            ).move_to([HIST_BOX.left + HIST_BOX.width / 2, -3.62, 0])
            hist_ylabel = (
                m.MathTex(r"\mathrm{number\ of\ games}", font_size=14 * MPT, color=s.fg)
                .rotate(90 * m.DEGREES)
                .move_to([HIST_BOX.left - 1.0, HIST_BOX.bottom + HIST_BOX.height / 2, 0])
            )
            clt_note = m.MathTex(
                r"\mathrm{dotted:}\ \mathcal{N}\left(p,\ p(1-p)/n\right)",
                font_size=9.5 * MPT,
                color=s.muted,
            ).move_to([HIST_BOX.left + 0.18, HIST_BOX.top - 0.3, 0], aligned_edge=m.LEFT)
            se = np.sqrt(p * (1 - p) / N_ROUNDS)
            xs = np.linspace(x_lo, x_hi, 200)
            pdf = np.exp(-0.5 * ((xs - p) / se) ** 2) / (se * np.sqrt(2 * np.pi))

            def histogram() -> m.VGroup:
                n = games_at(shown.get_value(), n_games)
                counts, _ = np.histogram(est[:n], bins=bins)
                group = m.VGroup()
                if s.hist_kind == "bars":
                    for c, left in zip(counts, bins[:-1]):
                        if c == 0:
                            continue
                        lo = ha.c2p(left + 0.05 * step, 0)
                        hi = ha.c2p(left + 0.95 * step, c)
                        group.add(
                            m.Rectangle(
                                width=hi[0] - lo[0],
                                height=max(hi[1] - lo[1], 1e-3),
                                stroke_width=0,
                                fill_color=s.accent,
                                fill_opacity=0.85,
                            ).move_to((lo + hi) / 2)
                        )
                else:
                    outline = ha.c2p(*step_outline(bins, counts).T)
                    group.add(
                        m.VMobject(
                            stroke_width=0, fill_color=s.accent, fill_opacity=0.18
                        ).set_points_as_corners(outline)
                    )
                    for w, a in ((16.9, 0.08), (8.5, 0.2), (3.9, 1.0)):
                        group.add(
                            m.VMobject(
                                stroke_color=s.accent, stroke_width=w, stroke_opacity=a
                            ).set_points_as_corners(outline)
                        )
                curve = m.VMobject(stroke_color=s.fg, stroke_width=3.4).set_points_as_corners(
                    ha.c2p(xs, pdf * n * step)
                )
                group.add(m.DashedVMobject(curve, num_dashes=70))
                return group

            def hist_value() -> m.VGroup:
                n = games_at(shown.get_value(), n_games)
                text = m.VGroup(
                    m.MathTex(tex_int(n) + r"\ \mathrm{games}", font_size=11.5 * MPT, color=s.fg),
                    m.MathTex(
                        rf"\bar{{p}} = {tex_pct(est[:n].mean(), 3)}",
                        font_size=11.5 * MPT,
                        color=s.fg,
                    ),
                ).arrange(m.DOWN, aligned_edge=m.RIGHT, buff=0.18)
                return text.move_to(
                    [HIST_BOX.right - 0.15, HIST_BOX.top - 0.2, 0], aligned_edge=m.UR
                )

            # ----- intro -----
            self.play(
                m.Write(title),
                m.FadeIn(subtitle, shift=0.2 * m.DOWN),
                m.LaggedStart(*(m.FadeIn(d, scale=0.3) for d in dots), lag_ratio=0.01),
                m.FadeIn(band),
                m.Create(rate_frame),
                m.Create(hist_frame),
                m.FadeIn(m.VGroup(rate_title, hist_title, rate_xlabel, rate_ylabel)),
                m.FadeIn(m.VGroup(hist_xlabel, hist_ylabel, band_note, clt_note)),
                m.Create(p_line),
                m.Create(exact_line),
                m.FadeIn(p_label),
                run_time=INTRO_S,
            )

            ghost_mob = m.always_redraw(ghosts)
            label_mob = m.always_redraw(round_label)
            tally_mob = m.always_redraw(tally)
            curve_mob = m.always_redraw(rate_curve)
            value_mob = m.always_redraw(rate_value)
            hist_mob = m.always_redraw(histogram)
            hist_text = m.always_redraw(hist_value)
            tri = triangle(1)
            self.play(
                m.Create(tri),
                m.FadeIn(m.VGroup(label_mob, tally_mob, value_mob, hist_text)),
                m.FadeIn(m.VGroup(curve_mob, hist_mob, ghost_mob)),
                run_time=SLOW_ROUND_S,
            )
            self.add(ghost_mob)

            # ----- slow rounds: the triangle morphs from one round to the next -----
            n_slow_frames = SLOW_ROUNDS + 150  # same schedule as the matplotlib GIF
            for k in range(2, SLOW_ROUNDS + 1):
                self.play(
                    m.Transform(tri, triangle(k)),
                    log_k.animate.set_value(np.log10(k)),
                    shown.animate.set_value((k - 1) / (n_slow_frames - 1)),
                    run_time=SLOW_ROUND_S,
                )

            # ----- fast rounds: accelerate geometrically to 10^6 -----
            self.remove(tri)
            tri_mob = m.always_redraw(lambda: triangle(round_at(log_k.get_value())))
            self.add(tri_mob)
            self.play(
                log_k.animate.set_value(np.log10(N_ROUNDS)),
                shown.animate.set_value(1.0),
                run_time=FAST_S,
                rate_func=m.linear,
            )

            # ----- finale: point at the two estimates agreeing with p -----
            self.play(
                m.Circumscribe(value_mob, color=s.accent, buff=0.12),
                m.Circumscribe(hist_text, color=s.fg, buff=0.12),
                run_time=1.2,
            )
            self.wait(HOLD_S)

    return IsoscelesManimGX


def render_style(exp: Experiment, style: Style, out_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """Render one style to MP4 and GIF; return both paths."""
    m.config.pixel_width, m.config.pixel_height = PIXEL_WIDTH, PIXEL_HEIGHT
    m.config.frame_rate = FPS
    m.config.background_color = style.bg
    scene = make_scene_class(exp, style)()

    frames: list[Image.Image] = []
    repeats: list[int] = []

    def keep(frame: m.Frame) -> None:
        rgba = np.frombuffer(frame.pixels(), dtype=np.uint8)
        img = rgba.reshape(PIXEL_HEIGHT, PIXEL_WIDTH, 4)[:, :, :3]
        frames.append(Image.fromarray(img.copy()))
        repeats.append(frame.repeat)

    mp4 = out_dir / f"isosceles_triangles_manimgx_{style.name}.mp4"
    scene.render(mp4, frames=keep)

    gif = out_dir / f"isosceles_triangles_manimgx_{style.name}.gif"
    quantized = quantize_frames(frames, style)
    quantized[0].save(
        gif,
        save_all=True,
        append_images=quantized[1:],
        duration=[frame_ms(r) for r in repeats],
        loop=0,
        optimize=True,
        disposal=1,
    )
    return mp4, gif


def main(names: list[str]) -> None:
    """Render the requested styles (default: clean and neon) into the project root."""
    wanted = names or ["clean", "neon"]
    exp = build_experiment()
    for style in STYLES:
        if style.name not in wanted:
            continue
        mp4, gif = render_style(exp, style)
        with Image.open(gif) as im:
            n_frames = im.n_frames
        print(f"{mp4.name}: {mp4.stat().st_size / 1e6:.2f} MB")
        print(f"{gif.name}: {n_frames} frames, {gif.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1:])
