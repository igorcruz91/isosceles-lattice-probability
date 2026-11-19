# Progress log

Read this first in a new session. It records what has been done, where every
piece lives, and what is still open. Last updated: 2026-11-19.

## The question

Pick 3 distinct random points on an n x n integer grid. If they are collinear,
redraw. What is the probability that the triangle is isosceles?

    p_n = I_n / (C(n^2, 3) - L_n)

where I_n is the number of isosceles triangles and L_n is the number of
collinear triples.

## Git state

Everything below is merged into `main` (as of 2026-11-19):

| Branch | Merged via | Contents |
|---|---|---|
| (initial) | commit 77dec4a | Monte Carlo core (`core/`), Streamlit `app.py`, GIF/manim animations, `core/exact.py`, `docs/isosceles_10x10.tex`, `grid_size_experiment.py`, `results/grid_size_comparison.*` |
| `analysis-grid-20-30` | PR #1 | `docs/isosceles_20x20_30x30.tex` |
| `analysis-10x10-visual-explainer` | second PR | `make_explainer_figures.py`, `tests/test_explainer_figures.py`, `docs/isosceles_10x10_visual.tex`, `docs/figures/explainer_10x10/*.pdf`, this file, CLAUDE.md pointer |

Remote: https://github.com/igorcruz91/isosceles-lattice-probability

Project rules (CLAUDE.md): Python 3.11+, Black, flake8, type hints, docstrings,
pytest, branches `analysis-*` / `model-*`, docs in `/docs`. The suite last
passed with 56 tests; black and flake8 were clean.

## Work done, in order

### 1. 10x10 grid (already on `main`, from earlier sessions)
- `docs/isosceles_10x10.tex`: full derivation.
- p_10 = 11,284 / 157,252 = **2821/39313 ≈ 0.0717574**.
- C(100,3) = 161,700; collinear L_10 = 4,448.
- Monte Carlo run of 10,000 games x 1,000,000 rounds gives 0.0717576.

### 2. 20x20 and 30x30 grids (`docs/isosceles_20x20_30x30.tex`, PR #1)

| n | triangles C - L | L_n | I_n | p_n |
|---|---|---|---|---|
| 10 | 157,252 | 4,448 | 11,284 | 2821/39313 ≈ 0.07176 |
| 20 | 10,482,124 | 104,676 | 248,292 | 62073/2620531 ≈ 0.0236872 |
| 30 | 120,464,616 | 630,684 | 1,460,408 | 7937/654699 ≈ 0.0121231 |

**Method.** No lattice equilateral triangles exist, so every isosceles
triangle has exactly one apex. For each apex, count pairs of points equidistant
from it (S), then subtract the pairs collinear with the apex (the midpoint term
M): I = S - M.
- Midpoint term: M_n = n^2 (n^2 - 4)/8 for even n and (n^2 - 1)^2/8 for odd n.
- Collinear triples use the standard gcd formula (matches OEIS A000938).

**Checks.**
- Brute-force enumeration of every triple for n = 10, 20 and 30.
- Monte Carlo runs.
- Recomputed gcd tables.
- A full step-by-step review found no errors.

### 3. General n x n law (section of the same `.tex`)
- There is an exact formula as a sum over vector pairs, but no simple closed
  form, because it depends on r_2(k), the number of ways to write k as a sum
  of two squares.
- Asymptotics: Σ r_2(k)^2 ~ 4 X ln X (Ramanujan) gives
  - **p_n ~ (24/π² + 4/π) · ln n / n² ≈ 3.705 · ln n / n²**
- With a correction term, (3.705 ln n - 1.79)/n² is within 2% of the exact
  value for n ≥ 20 and within 1% for n ≥ 30. A table runs up to n = 150.
- K = (6+π)/(6π²) was identified by PSLQ to 64 digits.
- Caveats: this is not a full proof. It assumes lattice points are spread
  evenly around circles, and the constant was identified numerically rather
  than derived symbolically.

### 4. Visual explainer for 10x10 (uncommitted, `analysis-10x10-visual-explainer`)
- The user is an observational learner and wants simple statements first,
  then gradually more complex ones, with lots of pictures.
- `make_explainer_figures.py` builds 18 figures:
  - Run `python make_explainer_figures.py` to write PDFs to
    `docs/figures/explainer_10x10/`.
  - Add `--preview DIR` to write PNGs instead.
- Palette: blue #2a78d6, orange #eb6834 (equal sides), aqua/green #1baf7a
  (apex), plus a blue sequential ramp for the heat map.
- `tests/test_explainer_figures.py` has 11 tests.
- `docs/isosceles_10x10_visual.tex` is a 16-section picture guide with
  `\keyidea` boxes and uses all 18 figures.

### 5. HTML slide deck (claude.ai artifact)
- "Three Dots, One Question": https://claude.ai/artifact/5UfBbDCXxwBRhW6VokN3J6
  - Private.
  - 27 slides with speaker notes.
  - Uses all 18 figures plus slides on 20x20/30x30 and the general law.
- Source is in `.cache/deck/project/` (gitignored): `deck.json`, `deck.css` and
  `slides/<id>.html`.
- To republish, run Artifact publish with `root: .cache/deck` and the URL
  above.
- Figure PNGs used for the slides are in `.cache/slidepng/` and are
  referenced in the slides as `/_blob/<asset id>`.
- The slides were never visually previewed, because Playwright/Chromium is not
  installed.

### 6. Discussion: related fields (no files)
- **Number theory:** r_2, Gauss circle problem, the 6/π² coprimality constant.
- **Erdős-type geometry:** distinct distances, Pach-Tardos isosceles counts,
  no-three-in-line.
- **Geometric probability:** Lewis Carroll's obtuse-triangle problem.
- **Physics:** box degeneracies, diffraction multiplicities.
- **Computer science:** exact integer predicates, distance ties on pixel grids
  and Voronoi boundaries.
- **Suggested extensions:**
  - 3D grid (equilateral triangles exist)
  - triangular lattice
  - taxicab distance
  - right or Heron triangles

## Open items / possible next steps
1. (Done) Visual explainer committed and merged.
2. (Done) PR #1 merged.
3. Visual guides for 20x20 and 30x30 were offered but not requested yet.
4. Check the slide deck by eye for text overflow, especially the "numbers",
   "win" and "summary" slides.
5. Possible extensions: see section 6.

## Environment notes (Windows)
- Git Bash heredocs collapse `\\`. Write scripts containing regexes or LaTeX
  with the Write tool instead.
- sympy and mpmath are not in the project venv. They were pip-installed into
  the session scratchpad only.
- The Artifact asset upload rejects scratchpad paths. Copy files into the
  project's gitignored `.cache/` first.
