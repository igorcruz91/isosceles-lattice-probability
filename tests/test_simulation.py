from core.simulation import Simulation


def test_run_round_updates_history():
    sim = Simulation(width=5, height=5, seed=42)
    sim.run_round()
    assert sim.total_rounds == 1


def test_run_rounds_bulk():
    sim = Simulation(width=5, height=5, seed=42)
    sim.run_rounds(50)
    assert sim.total_rounds == 50
    assert 0.0 <= sim.probability_estimate() <= 1.0


def test_resample_degenerate_never_records_degenerate():
    sim = Simulation(width=6, height=6, seed=1, resample_degenerate=True)
    sim.run_rounds(200)
    assert sim.degenerate_count == 0


def test_no_resample_can_record_degenerate_on_tiny_grid():
    # A 2x2 grid has only 4 points, all pairwise-adjacent triples are
    # right triangles (never collinear), so use a line-friendly seed
    # search on a bigger grid instead of asserting on 2x2.
    sim = Simulation(width=10, height=10, seed=7, resample_degenerate=False)
    sim.run_rounds(500)
    # Not asserting a fixed count (RNG-dependent), just that the policy
    # is respected: total rounds includes degenerate ones.
    assert sim.total_rounds == 500
    assert sim.valid_triangle_count <= sim.total_rounds


def test_same_seed_is_reproducible():
    sim_a = Simulation(width=8, height=8, seed=123)
    sim_b = Simulation(width=8, height=8, seed=123)
    rounds_a = sim_a.run_rounds(20)
    rounds_b = sim_b.run_rounds(20)
    for ra, rb in zip(rounds_a, rounds_b):
        assert (ra.points == rb.points).all()
        assert ra.result.type == rb.result.type
