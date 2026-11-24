# Isosceles Triangle Probability

Estimates the probability that 3 distinct random points on an integer grid
form an isosceles triangle, via repeated random sampling ("rounds"),
visualized in an interactive Streamlit app.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Tweak grid size, seed, and how collinear (degenerate) draws are handled from
the sidebar; draw single rounds to see the sampled points and the resulting
triangle, or run many rounds at once to watch the probability estimate
converge.

## Run tests

```bash
pytest
black .
flake8
```

## Method notes

- Points are integers, so squared side lengths are integers too — triangle
  classification (scalene / isosceles / equilateral / degenerate) is done
  with exact integer comparisons, no floating-point tolerance needed.
- A triangle is isosceles if at least two of its three squared side lengths
  are equal (equilateral counts as isosceles).
- No triangle with all-integer vertex coordinates can be equilateral (its
  area would be irrational, contradicting Pick's theorem for lattice
  polygons), so the `EQUILATERAL` category should never actually appear —
  it exists for correctness/clarity, and the app flags it as a bug if it
  ever does.
- Collinear (degenerate) draws aren't triangles. The sidebar lets you choose
  whether to redraw until a real triangle forms (probability conditional on
  forming a triangle) or count a collinear draw as a non-isosceles round.

## Exact value and grid-size comparison

`core/exact.py` computes the exact probability without enumerating triples
(collinear triples via a gcd sum, isosceles triangles by counting equidistant
pairs around each apex). `docs/isosceles_10x10.tex` is the hand derivation for
the 10x10 grid: P = 2821/39313 ≈ 0.0717574 (compile it on Overleaf).

```bash
python grid_size_experiment.py
```

reruns the 10,000 games × 1,000,000 rounds Monte Carlo on 10x10, 20x20, 30x30 and
40x40 grids (cached in `.cache/`) and writes `results/grid_size_comparison.{csv,png}`.

## Animations (GIF / MP4)

```bash
python make_linkedin_gifs.py   # matplotlib version: isosceles_triangles_{clean,neon}.gif
python make_manimgx_gif.py     # ManimGX version: isosceles_triangles_manimgx_{clean,neon}.{gif,mp4}
python make_grid_comparison_gif.py  # 10x10 vs 20x20 vs 30x30: isosceles_grid_comparison_{clean,neon}.gif
```

Both scripts use the same experiment (same seeds, same Monte Carlo cache), so the
outputs can be compared frame for frame. The ManimGX one needs Python 3.13+ and a
GPU (Direct3D 12 on Windows); see https://manimgx.academa.ai. ManimGX writes MP4
natively; the GIF is built from the frames it hands back via
`Scene.render(frames=...)`.
