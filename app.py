"""Streamlit app: estimate the probability that 3 random grid points form an isosceles triangle."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st

from core.geometry import TriangleType
from core.monte_carlo import monte_carlo_games
from core.simulation import Simulation

st.set_page_config(page_title="Isosceles Triangle Probability", layout="wide")
sns.set_theme(style="whitegrid")

TYPE_COLOR = {
    TriangleType.ISOSCELES: "#2ca02c",
    TriangleType.EQUILATERAL: "#9467bd",
    TriangleType.SCALENE: "#d62728",
    TriangleType.DEGENERATE: "#7f7f7f",
}
TYPE_LABEL = {
    TriangleType.ISOSCELES: "Isosceles",
    TriangleType.EQUILATERAL: "Equilateral",
    TriangleType.SCALENE: "Scalene",
    TriangleType.DEGENERATE: "Degenerate (collinear)",
}


def get_config() -> tuple:
    with st.sidebar:
        st.header("Grid")
        width = st.slider("Width (points)", 2, 60, 10)
        height = st.slider("Height (points)", 2, 60, 10)

        st.header("Randomness")
        use_seed = st.checkbox("Use fixed seed (reproducible)", value=True)
        seed = st.number_input("Seed", value=42, step=1) if use_seed else None

        st.header("Degenerate points (collinear)")
        resample = (
            st.radio(
                "When 3 points are collinear (not a real triangle):",
                options=["Redraw until a real triangle forms", "Count as a non-isosceles round"],
                index=0,
            )
            == "Redraw until a real triangle forms"
        )

    return width, height, seed, resample


def get_simulation(width: int, height: int, seed, resample: bool) -> Simulation:
    config = (width, height, seed, resample)
    if st.session_state.get("config") != config or "sim" not in st.session_state:
        st.session_state.sim = Simulation(
            width=width, height=height, seed=seed, resample_degenerate=resample
        )
        st.session_state.config = config
    return st.session_state.sim


def draw_grid_figure(
    sim: Simulation, highlighted: np.ndarray | None, ttype: TriangleType | None
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sim.grid[:, 0],
            y=sim.grid[:, 1],
            mode="markers",
            marker=dict(size=6, color="#c7c7c7"),
            name="grid points",
            hoverinfo="skip",
        )
    )
    if highlighted is not None:
        closed = np.vstack([highlighted, highlighted[0]])
        color = TYPE_COLOR[ttype]
        fig.add_trace(
            go.Scatter(
                x=closed[:, 0],
                y=closed[:, 1],
                mode="lines+markers",
                line=dict(color=color, width=3),
                marker=dict(size=12, color=color),
                name=TYPE_LABEL[ttype],
            )
        )
    fig.update_layout(
        xaxis=dict(range=[-1, sim.width], scaleanchor="y", scaleratio=1, zeroline=False),
        yaxis=dict(range=[-1, sim.height], zeroline=False),
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=highlighted is not None,
    )
    return fig


def monte_carlo_histogram(estimates: np.ndarray, n_rounds: int) -> plt.Figure:
    """Histogram of per-game P(isosceles) estimates, with a normal-approximation overlay.

    Built with seaborn/matplotlib rather than Plotly: a matplotlib figure carries its
    own explicit colors, so it can't inherit Streamlit's dark Plotly template and end
    up with unreadable low-contrast axis text.
    """
    mean = float(estimates.mean())
    theo_se = (
        float(np.sqrt(mean * (1 - mean) / n_rounds)) if 0.0 < mean < 1.0 else float(estimates.std())
    )
    spread = max(theo_se, 1e-6)

    ink = "#0b0b0b"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(
        estimates,
        stat="density",
        bins=40,
        color="#2a78d6",
        edgecolor="white",
        linewidth=0.3,
        ax=ax,
        label="per-game estimate",
    )

    xs = np.linspace(max(0.0, mean - 4 * spread), min(1.0, mean + 4 * spread), 200)
    ys = np.exp(-0.5 * ((xs - mean) / spread) ** 2) / (spread * np.sqrt(2 * np.pi))
    ax.plot(xs, ys, color=ink, linestyle=":", linewidth=2, label="normal approx (CLT)")

    ax.axvline(mean, color="#52514e", linestyle="--", linewidth=1.5)
    ax.text(
        mean,
        ax.get_ylim()[1] * 0.98,
        f"mean = {mean:.4f}",
        ha="center",
        va="top",
        color=ink,
        fontsize=10,
    )

    ax.set_xlabel("P(isosceles) estimate for one game", color=ink, fontsize=11)
    ax.set_ylabel("density", color=ink, fontsize=11)
    ax.tick_params(colors=ink, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.legend(loc="upper left", frameon=False, labelcolor=ink)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout()
    return fig


st.title("Isosceles Triangle Probability on a Grid")
st.caption(
    "Pick 3 distinct random points on an n×m integer grid each round and check whether they "
    "form an isosceles triangle. Squared side lengths are integers on a lattice, so the "
    "classification below is exact, not float-approximate."
)

width, height, seed, resample = get_config()
sim = get_simulation(width, height, seed, resample)

col_controls, col_bulk = st.columns(2)
with col_controls:
    if st.button("▶ Draw next round", use_container_width=True):
        sim.run_round()
with col_bulk:
    bulk_n = st.number_input("Rounds to run", min_value=1, max_value=200_000, value=1000, step=100)
    if st.button(f"⏩ Run {bulk_n} rounds", use_container_width=True):
        sim.run_rounds(int(bulk_n))

if st.button("↺ Reset simulation"):
    st.session_state.sim = Simulation(
        width=width, height=height, seed=seed, resample_degenerate=resample
    )
    st.rerun()

st.divider()

plot_col, info_col = st.columns([3, 2])

last_round = sim.history[-1] if sim.history else None

with plot_col:
    fig = draw_grid_figure(
        sim,
        last_round.points if last_round else None,
        last_round.result.type if last_round else None,
    )
    st.plotly_chart(fig, use_container_width=True)

with info_col:
    st.subheader("Last round")
    if last_round is None:
        st.info("No rounds yet. Draw one to see the points and triangle here.")
    else:
        r = last_round
        st.markdown(f"**Round #{r.index}** — points {[tuple(p) for p in r.points]}")
        color = "green" if r.result.is_isosceles else "red"
        label = TYPE_LABEL[r.result.type]
        st.markdown(
            f"Squared side lengths: `{r.result.squared_sides}`  \n"
            f"Classification: **:{color}[{label}]**"
        )

    st.subheader("Running statistics")
    total = sim.total_rounds
    if total == 0:
        st.write("Run some rounds to see statistics.")
    else:
        prob = sim.probability_estimate()
        c1, c2, c3 = st.columns(3)
        c1.metric("Rounds", total)
        c2.metric("Valid triangles", sim.valid_triangle_count)
        c3.metric("Degenerate", sim.degenerate_count)
        c4, c5 = st.columns(2)
        c4.metric("Isosceles", sim.isosceles_count)
        c5.metric("P(isosceles)", f"{prob:.4f}" if prob is not None else "n/a")

        if sim.equilateral_count:
            st.caption(
                f"Note: {sim.equilateral_count} equilateral triangle(s) recorded — this "
                "should be impossible on an integer lattice; check for a bug."
            )

st.divider()

st.subheader("Convergence of the probability estimate")
if sim.total_rounds > 0:
    cum_probs = sim.cumulative_probabilities()
    st.line_chart({"P(isosceles | valid triangle)": cum_probs})
else:
    st.write("Run rounds to see how the estimate converges.")

with st.expander("Round history (most recent first)"):
    if sim.history:
        rows = [
            {
                "round": r.index,
                "p1": tuple(r.points[0]),
                "p2": tuple(r.points[1]),
                "p3": tuple(r.points[2]),
                "squared_sides": r.result.squared_sides,
                "type": TYPE_LABEL[r.result.type],
            }
            for r in reversed(sim.history[-200:])
        ]
        st.dataframe(rows, use_container_width=True)
    else:
        st.write("No rounds yet.")

st.divider()

st.subheader("Monte Carlo across many games")
st.caption(
    "Run many independent 'games', each drawing its own batch of rounds, and look at the "
    "spread of the resulting P(isosceles) estimates. This shows how much a single game's "
    "estimate would vary if you reran the whole experiment, and lets you check that spread "
    "against the normal approximation predicted by the central limit theorem — the estimator's "
    "standard deviation across games should land close to the theoretical binomial standard "
    "error, sqrt(p(1-p)/rounds)."
)

mc_col1, mc_col2, mc_col3 = st.columns([1, 1, 1])
with mc_col1:
    n_games = st.number_input(
        "Number of games", min_value=10, max_value=20_000, value=1000, step=10
    )
with mc_col2:
    rounds_per_game = st.number_input(
        "Rounds per game", min_value=10, max_value=50_000, value=1000, step=10
    )
with mc_col3:
    st.write("")
    st.write("")
    run_mc = st.button("🎲 Run Monte Carlo", use_container_width=True)

if run_mc:
    with st.spinner(f"Running {n_games} games of {rounds_per_game} rounds each..."):
        mc_estimates = monte_carlo_games(
            width=width,
            height=height,
            n_games=int(n_games),
            n_rounds=int(rounds_per_game),
            seed=seed,
            resample_degenerate=resample,
        )
    st.session_state.mc_estimates = mc_estimates
    st.session_state.mc_rounds_per_game = int(rounds_per_game)

if "mc_estimates" in st.session_state:
    mc_estimates = st.session_state.mc_estimates
    mc_rounds_per_game = st.session_state.mc_rounds_per_game
    mc_mean = float(mc_estimates.mean())
    mc_std = float(mc_estimates.std())
    mc_theo_se = float(np.sqrt(mc_mean * (1 - mc_mean) / mc_rounds_per_game))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Games run", len(mc_estimates))
    m2.metric("Mean of estimates", f"{mc_mean:.4f}")
    m3.metric("Std across games", f"{mc_std:.4f}")
    m4.metric("Theoretical SE", f"{mc_theo_se:.4f}")

    mc_fig = monte_carlo_histogram(mc_estimates, mc_rounds_per_game)
    st.pyplot(mc_fig)
    plt.close(mc_fig)
else:
    st.write("Run the Monte Carlo above to see the distribution of per-game estimates.")
