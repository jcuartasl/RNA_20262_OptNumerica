"""Minimal tests for src/optimizers/differential_evolution.py."""

import numpy as np

from src.functions.rastrigin import Rastrigin
from src.functions.rosenbrock import Rosenbrock

from src.optimizers.differential_evolution import (
    optimize,
    _pick_distinct_indices,
    _binomial_crossover,
)

FUNCTIONS = [Rastrigin(), Rosenbrock()]
DIMENSIONS = [2, 3]


def test_distinct_indices():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        i = int(rng.integers(10))
        r = _pick_distinct_indices(rng, i, 10)
        assert len(set(r.tolist())) == 3
        assert i not in r


def test_crossover_keeps_dimension_and_takes_at_least_one_from_mutant():
    rng = np.random.default_rng(1)
    target = np.zeros(5)
    mutant = np.ones(5)
    for _ in range(500):
        trial = _binomial_crossover(rng, target, mutant, cr=0.0)
        assert trial.shape == (5,)
        assert trial.sum() == 1.0          # CR = 0 -> exactly j_rand comes from the mutant
        trial = _binomial_crossover(rng, target, mutant, cr=1.0)
        assert np.all(trial == mutant)     # CR = 1 -> everything from the mutant


def test_budget_never_exceeded_and_counts_are_coherent():
    for f in FUNCTIONS:
        for dim in DIMENSIONS:
            for budget in (1, 3, 49, 50, 51, 120, 1000):
                r = optimize(f, f.default_bounds, dim, 0, budget)
                assert r["f_evaluations"] <= budget
                assert r["gradient_evaluations"] == 0
                assert r["equivalent_evaluations"] == r["f_evaluations"]
                assert len(r["best_x"]) == dim


def test_uses_whole_budget_when_it_covers_the_population():
    f = FUNCTIONS[0]
    r = optimize(f, f.default_bounds, 2, 0, 1000)
    assert r["f_evaluations"] == 1000


def test_counter_matches_real_calls():
    calls = {"n": 0}

    class Spy(Rastrigin):
        def evaluate(self, x):
            calls["n"] += 1
            return super().evaluate(x)

    r = optimize(Spy(), (-5.12, 5.12), 3, 1, 777)
    assert calls["n"] == r["f_evaluations"] == 777


def test_same_seed_same_result_different_seed_different_result():
    f = FUNCTIONS[0]
    a = optimize(f, f.default_bounds, 2, 7, 2000)
    b = optimize(f, f.default_bounds, 2, 7, 2000)
    c = optimize(f, f.default_bounds, 2, 8, 2000)
    assert a == b
    assert a["history"] != c["history"]


def test_best_x_inside_bounds_even_with_large_F():
    for f in FUNCTIONS:
        lo, hi = f.default_bounds
        r = optimize(f, f.default_bounds, 3, 0, 2000, {"mutation_factor": 2.0})
        assert all(lo <= v <= hi for v in r["best_x"])


def test_history_starts_at_step_zero_and_never_worsens():
    for f in FUNCTIONS:
        r = optimize(f, f.default_bounds, 2, 3, 3000)
        h = r["history"]
        assert h[0]["step"] == 0
        assert h[0]["equivalent_evaluations"] == 50
        values = [e["best_f"] for e in h]
        assert all(a >= b for a, b in zip(values, values[1:]))
        assert h[-1]["best_f"] == r["best_f"]


def test_invalid_config_raises():
    f = FUNCTIONS[0]
    for cfg in ({"population_size": 3}, {"crossover_rate": 1.5}, {"mutation_factor": 0.0}):
        try:
            optimize(f, f.default_bounds, 2, 0, 100, cfg)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {cfg}")
