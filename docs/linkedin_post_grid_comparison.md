# LinkedIn post: 10x10 vs 20x20 vs 30x30

Attachment: `isosceles_grid_comparison_clean.gif` (or `_neon.gif`), built by
`python make_grid_comparison_gif.py`. Numbers come from
`docs/isosceles_20x20_30x30.tex` and `results/grid_size_comparison.csv`.

---

A while ago I asked: pick 3 random points on a 10×10 grid. What are the odds
they form an isosceles triangle? Answer: 7.18%.

Then I made the grid bigger. 📐

▪️ 10×10 → 7.18%
▪️ 20×20 → 2.37%
▪️ 30×30 → 1.21%

The surprise: doubling the grid gives 4× as many points, but the odds only drop
about 3×. Tripling it gives 9× as many points, but the odds drop only 6×.

Why? An isosceles triangle needs two points at the same distance from a third.
On a grid, distances are √(a² + b²), and some numbers can be written as a sum of
two squares in many ways (25 = 5² + 0² = 3² + 4²). A bigger grid has more of
these "crowded" distances, which partly offsets the extra space.

Putting it together gives a law for any n×n grid:

p(n) ≈ 3.7 · ln(n) / n²

That ln(n) is the number theory pushing back. (The constant is 24/π² + 4/π. I
pinned it down numerically to 64 digits, but it isn't a full proof yet.)

How I know the numbers are right:
✅ Exact counting: 1,460,408 isosceles out of 120,464,616 triangles on 30×30
✅ Brute force over every triple on all three grids
✅ Monte Carlo: 10,000 games × 1,000,000 rounds per grid, matching the exact
   value to at least 4 significant digits

In the GIF, each grid plays one game side by side. Green = isosceles,
red = scalene. Bottom left: each running estimate settling on its exact value.
Bottom right: those three values sitting on the curve for every grid size.

Next question: a 3D grid, where equilateral triangles are possible. 🤔

#Mathematics #Probability #NumberTheory #MonteCarlo #Python #DataVisualization
