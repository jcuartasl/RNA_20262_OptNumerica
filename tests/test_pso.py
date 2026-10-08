"""Unit tests for Particle Swarm Optimization (PSO) optimizer."""

import numpy as np
import pytest
from src.functions.base import ObjectiveFunction
from src.functions.rosenbrock import Rosenbrock
from src.functions.rastrigin import Rastrigin
from src.metrics.evaluation_counter import CountedFunction
from src.optimizers.PSO import optimize


# ---------------------------------------------------------
# Test helpers
# ---------------------------------------------------------
class Sphere(ObjectiveFunction):
    """Convex quadratic f(x) = sum(x_i^2), used only to test PSO correctness."""

    def __init__(self):
        super().__init__(name="Sphere", default_bounds=(-5.0, 5.0), global_minimum_f=0.0)

    def evaluate(self, x):
        return float(np.sum(self._validate_input(x) ** 2))

    def gradient(self, x):
        return 2.0 * self._validate_input(x)

    def global_minimum_x(self, dimension):
        return np.zeros(dimension)


class ShiftedSphere(Sphere):
    """f(x) = sum(x_i^2) + 5, so the global minimum value is f* = 5 (not 0)."""

    def __init__(self):
        ObjectiveFunction.__init__(
            self, name="ShiftedSphere", default_bounds=(-5.0, 5.0), global_minimum_f=5.0
        )

    def evaluate(self, x):
        return float(np.sum(self._validate_input(x) ** 2)) + 5.0


class Spy(ObjectiveFunction):
    """Wraps a function and records EVERY point (and value) where f or grad is requested."""

    def __init__(self, base: ObjectiveFunction):
        super().__init__(base.name, base.default_bounds, base.global_minimum_f)
        self.base = base
        self.f_points = []
        self.f_values = []
        self.grad_points = []

    def evaluate(self, x):
        self.f_points.append(np.array(x, dtype=float))
        value = self.base.evaluate(x)
        self.f_values.append(float(value))
        return value

    def gradient(self, x):
        self.grad_points.append(np.array(x, dtype=float))
        return self.base.gradient(x)

    def global_minimum_x(self, dimension):
        return self.base.global_minimum_x(dimension)


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------
class TestPSO:
    @pytest.mark.parametrize("FunctionClass,dim,bounds", [
        (Rosenbrock, 2, (-5.0, 10.0)),
        (Rosenbrock, 3, (-5.0, 10.0)),
        (Rastrigin, 2, (-5.12, 5.12)),
        (Rastrigin, 3, (-5.12, 5.12)),
    ])
    def test_common_interface_schema(self, FunctionClass, dim, bounds):
        """Verify that the optimizer returns the exact contract required by Phase 0."""
        func = FunctionClass(bounds=bounds)
        result = optimize(func, bounds, dim, seed=42, budget=500,
                          config={"population_size": 20})

        required_keys = [
            "best_x", "best_f", "f_evaluations", "gradient_evaluations",
            "equivalent_evaluations", "iterations", "success", "history"
        ]
        for key in required_keys:
            assert key in result, f"Missing required key '{key}' in result dictionary."

        assert isinstance(result["best_x"], list)
        assert len(result["best_x"]) == dim
        assert isinstance(result["best_f"], float)
        assert np.isfinite(result["best_f"])
        assert isinstance(result["f_evaluations"], int)
        assert isinstance(result["gradient_evaluations"], int)
        assert isinstance(result["equivalent_evaluations"], int)
        assert isinstance(result["iterations"], int)
        assert isinstance(result["success"], bool)
        assert isinstance(result["history"], list)
        assert len(result["history"]) > 0

        for entry in result["history"]:
            assert "step" in entry
            assert "best_f" in entry
            assert "equivalent_evaluations" in entry
            assert "best_x" in entry
            assert len(entry["best_x"]) == dim

        # best_f in history is "best so far": it must never increase
        hist_f = [h["best_f"] for h in result["history"]]
        assert all(b <= a for a, b in zip(hist_f, hist_f[1:]))

        # history is consistent with the run: one entry per iteration plus step 0
        assert len(result["history"]) == result["iterations"] + 1
        assert [h["step"] for h in result["history"]] == list(range(result["iterations"] + 1))
        assert result["history"][-1]["equivalent_evaluations"] == result["equivalent_evaluations"]
        assert result["history"][-1]["best_f"] == result["best_f"]

    def test_pso_does_not_use_gradients(self):
        """PSO is derivative-free: N_grad = 0 and E_eq = N_f."""
        spy = Spy(Rosenbrock())
        result = optimize(spy, (-5.0, 10.0), 3, seed=1, budget=300,
                          config={"population_size": 10})
        assert spy.grad_points == []
        assert result["gradient_evaluations"] == 0
        assert result["equivalent_evaluations"] == result["f_evaluations"]
        assert len(spy.f_points) == result["f_evaluations"]

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_reproducibility(self, seed: int):
        """Same seed -> identical swarm trajectory and result."""
        func = Rosenbrock()
        config = {"population_size": 15, "record_positions": True}
        res1 = optimize(func, (-5.0, 10.0), 2, seed=seed, budget=600, config=config)
        res2 = optimize(func, (-5.0, 10.0), 2, seed=seed, budget=600, config=config)

        assert res1["best_x"] == res2["best_x"]
        assert res1["best_f"] == res2["best_f"]
        assert res1["equivalent_evaluations"] == res2["equivalent_evaluations"]
        assert res1["iterations"] == res2["iterations"]
        assert res1["history"] == res2["history"]

    def test_different_seeds_produce_different_swarms(self):
        func = Rastrigin()
        config = {"population_size": 10, "record_positions": True}
        res1 = optimize(func, (-5.12, 5.12), 2, seed=1, budget=100, config=config)
        res2 = optimize(func, (-5.12, 5.12), 2, seed=2, budget=100, config=config)
        assert res1["history"][0]["positions"] != res2["history"][0]["positions"]
        assert res1["best_x"] != res2["best_x"]

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("budget", [1, 2, 5, 10, 25, 49, 50, 51, 100, 999, 1000, 10000])

    def test_strict_budget_compliance(self, dim: int, budget: int):
        """PSO never exceeds the budget; E_eq = N_f + 2*n*N_grad with N_grad = 0."""
        result = optimize(Rosenbrock(), (-5.0, 10.0), dim, seed=10, budget=budget)

        eq_evals = result["equivalent_evaluations"]
        assert eq_evals <= budget, f"Budget exceeded: allowed {budget}, used {eq_evals} (dim={dim})"

        n_f = result["f_evaluations"]
        n_grad = result["gradient_evaluations"]
        assert eq_evals == n_f + 2 * dim * n_grad

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("budget", [1, 9, 10, 11, 19, 20, 25, 100, 10000])
    def test_budget_used_by_whole_swarm_iterations(self, dim: int, budget: int):
        """PSO evaluates the full swarm each iteration, never a partial batch.

        Cost model: initial swarm n0 = min(P, budget) evaluations, then each
        iteration costs n0. k is the largest number of full iterations that fits.
        """
        pop = 10
        n0 = min(pop, budget)
        k = (budget - n0) // n0
        result = optimize(Rosenbrock(), (-5.0, 10.0), dim, seed=3, budget=budget,
                          config={"population_size": pop})
        assert result["iterations"] == k
        assert result["f_evaluations"] == n0 * (1 + k)
        assert result["gradient_evaluations"] == 0

    def test_initial_swarm_is_reduced_when_budget_is_smaller_than_population(self):
        spy = Spy(Rastrigin())
        result = optimize(spy, (-5.12, 5.12), 2, seed=0, budget=7,
                          config={"population_size": 50})
        assert result["f_evaluations"] == 7
        assert result["iterations"] == 0
        assert len(spy.f_points) == 7
        assert np.isfinite(result["best_f"])

    def test_reused_counter_respects_remaining_budget(self):
        """A counter that already spent evaluations must not let PSO exceed the budget."""
        func = Rosenbrock()
        counter = CountedFunction(func, 2)
        for _ in range(30):
            counter.evaluate(np.array([0.0, 0.0]))

        result = optimize(counter, (-5.0, 10.0), 2, seed=0, budget=100,
                          config={"population_size": 10})
        assert counter.equivalent_evaluations <= 100
        assert result["equivalent_evaluations"] == counter.equivalent_evaluations
        assert counter.equivalent_evaluations == 100  # 30 + 10 initial + 6 iterations of 10

    def test_exhausted_budget_returns_empty_result_without_evaluating(self):
        counter = CountedFunction(Rosenbrock(), 2)
        optimize(counter, (-5.0, 10.0), 2, seed=0, budget=100, config={"population_size": 10})
        assert counter.equivalent_evaluations == 100

        result = optimize(counter, (-5.0, 10.0), 2, seed=0, budget=100,
                          config={"population_size": 10})
        assert counter.equivalent_evaluations == 100  # nothing new was evaluated
        assert result["iterations"] == 0
        assert result["history"] == []
        assert result["best_f"] == float("inf")
        assert result["success"] is False

    def test_initial_swarm_within_bounds(self):
        """Initial positions (step 0) lie inside the domain, with the right shape."""
        pop, dim = 25, 3
        result = optimize(Rastrigin(), (-5.12, 5.12), dim, seed=11, budget=pop,
                          config={"population_size": pop, "record_positions": True})
        positions = np.array(result["history"][0]["positions"])
        assert positions.shape == (pop, dim)
        assert np.all(positions >= -5.12) and np.all(positions <= 5.12)

    @pytest.mark.parametrize("config", [
        {"population_size": 20},
        # aggressive parameters so that clipping is actually exercised
        {"population_size": 20, "inertia": 0.9, "cognitive": 2.0, "social": 2.0,
         "v_max_fraction": 1.0},
    ])
    def test_every_evaluated_point_within_bounds(self, config):
        """ALL evaluated points (not only best_x) stay inside the domain."""
        lower, upper = -5.0, 10.0
        spy = Spy(Rosenbrock())
        result = optimize(spy, (lower, upper), 2, seed=42, budget=2000, config=config)

        points = np.array(spy.f_points)
        assert len(points) == result["f_evaluations"] > 10
        assert np.all(points >= lower) and np.all(points <= upper)
        assert np.all(np.array(result["best_x"]) >= lower)
        assert np.all(np.array(result["best_x"]) <= upper)

    def test_clipping_is_active_with_aggressive_parameters(self):
        spy = Spy(Rosenbrock())
        optimize(spy, (-5.0, 10.0), 2, seed=42, budget=2000,
                 config={"population_size": 20, "inertia": 0.9, "cognitive": 2.0,
                         "social": 2.0, "v_max_fraction": 1.0})
        points = np.array(spy.f_points)
        assert np.any(np.isclose(points, -5.0) | np.isclose(points, 10.0))

    def test_best_is_a_really_evaluated_point_and_the_minimum_seen(self):
        """gbest must be an evaluated solution and equal the min over ALL evaluations."""
        spy = Spy(Rastrigin())
        result = optimize(spy, (-5.12, 5.12), 3, seed=8, budget=1500,
                          config={"population_size": 30})

        assert result["best_f"] == min(spy.f_values)
        best_x = np.array(result["best_x"])
        matches = [np.array_equal(best_x, p) for p in spy.f_points]
        assert any(matches), "best_x was never actually evaluated"
        idx = matches.index(True)
        assert spy.f_values[idx] == result["best_f"]

    def test_first_iteration_matches_update_rule(self):
        """One iteration must follow v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x), x = clip(x+v).

        At iteration 1, pbest = x0 and gbest = the best initial particle. The test
        rebuilds x0, v0, r1, r2 from the same seed and checks the evaluated points.
        """
        pop, dim, seed = 5, 2, 5
        w, c1, c2, vfrac = 0.7, 1.5, 1.5, 0.5
        lower, upper = -5.0, 10.0
        spy = Spy(Rosenbrock())
        result = optimize(spy, (lower, upper), dim, seed=seed, budget=2 * pop,
                          config={"population_size": pop, "inertia": w, "cognitive": c1,
                                  "social": c2, "v_max_fraction": vfrac})
        assert result["iterations"] == 1

        rng = np.random.default_rng(seed)
        v_max = vfrac * (upper - lower)
        x0 = rng.uniform(lower, upper, size=(pop, dim))
        v0 = rng.uniform(-v_max / 2.0, v_max / 2.0, size=(pop, dim))
        r1 = rng.random(size=(pop, dim))
        r2 = rng.random(size=(pop, dim))

        f0 = np.array([Rosenbrock().evaluate(p) for p in x0])
        gbest = x0[np.argmin(f0)]
        v1 = np.clip(w * v0 + c1 * r1 * (x0 - x0) + c2 * r2 * (gbest - x0), -v_max, v_max)
        expected_x1 = np.clip(x0 + v1, lower, upper)

        np.testing.assert_allclose(np.array(spy.f_points[:pop]), x0, rtol=0, atol=1e-15)
        np.testing.assert_allclose(np.array(spy.f_points[pop:]), expected_x1, rtol=0, atol=1e-12)

    def test_velocity_is_clamped(self):
        """Per-step displacement never exceeds v_max (before clipping can only shrink it)."""
        lower, upper, vfrac = -5.0, 10.0, 0.1
        v_max = vfrac * (upper - lower)
        pop = 10
        result = optimize(Rosenbrock(), (lower, upper), 2, seed=3, budget=pop * 30,
                          config={"population_size": pop, "v_max_fraction": vfrac,
                                  "inertia": 0.9, "cognitive": 2.0, "social": 2.0,
                                  "record_positions": True})
        swarms = [np.array(h["positions"]) for h in result["history"]]
        steps = np.abs(np.diff(np.stack(swarms), axis=0))
        assert np.all(steps <= v_max + 1e-12)

    def test_frozen_swarm_is_a_negative_control(self):
        """With w = c1 = c2 = 0 particles never move: best_f cannot improve."""
        result = optimize(Rastrigin(), (-5.12, 5.12), 2, seed=0, budget=500,
                          config={"population_size": 10, "inertia": 0.0,
                                  "cognitive": 0.0, "social": 0.0})
        hist_f = [h["best_f"] for h in result["history"]]
        assert len(set(hist_f)) == 1
        assert result["iterations"] > 0

    def test_converges_on_convex_quadratic(self):
        """On a convex quadratic the swarm must reach the minimum."""
        result = optimize(Sphere(), (-5.0, 5.0), 2, seed=0, budget=5000,
                          config={"population_size": 30, "tolerance": 1e-4})
        assert result["best_f"] < 1e-10
        assert result["success"] is True
        np.testing.assert_allclose(result["best_x"], [0.0, 0.0], atol=1e-4)

    def test_success_uses_global_minimum_f(self):
        """success compares against f* of the function, not against 0."""
        result = optimize(ShiftedSphere(), (-5.0, 5.0), 2, seed=0, budget=5000,
                          config={"population_size": 30, "tolerance": 1e-4})
        assert abs(result["best_f"] - 5.0) <= 1e-4
        assert result["success"] is True

    def test_tolerance_controls_success(self):
        """A very tight budget cannot reach tolerance; a huge tolerance always succeeds."""
        tight = optimize(Rastrigin(), (-5.12, 5.12), 2, seed=0, budget=10,
                         config={"population_size": 10, "tolerance": 1e-12})
        loose = optimize(Rastrigin(), (-5.12, 5.12), 2, seed=0, budget=10,
                         config={"population_size": 10, "tolerance": 1e6})
        assert tight["success"] is False
        assert loose["success"] is True

    def test_pso_escapes_local_basins_on_rastrigin(self):
        """Population-based exploration finds the global basin where local search cannot.

        Rastrigin 2D local minima have f ~ 1 near integer coordinates, so a
        local method started at random usually ends >= 1. PSO should reach
        the global minimum in most runs with the proposed default parameters.
        """
        successes = 0
        for seed in range(10):
            result = optimize(Rastrigin(), (-5.12, 5.12), 2, seed=seed, budget=10000,
                              config={"tolerance": 1e-4})
            successes += int(result["success"])
        assert successes >= 7

    def test_record_positions_option(self):
        pop, dim = 8, 3
        with_pos = optimize(Rastrigin(), (-5.12, 5.12), dim, seed=0, budget=pop * 4,
                            config={"population_size": pop, "record_positions": True})
        without_pos = optimize(Rastrigin(), (-5.12, 5.12), dim, seed=0, budget=pop * 4,
                               config={"population_size": pop})
        for entry in with_pos["history"]:
            assert np.array(entry["positions"]).shape == (pop, dim)
        for entry in without_pos["history"]:
            assert "positions" not in entry
        # recording must not change the optimization itself
        assert with_pos["best_x"] == without_pos["best_x"]
        assert with_pos["best_f"] == without_pos["best_f"]

    @pytest.mark.parametrize("bounds", [
        (-5.0, 10.0),
        [-5.0, 10.0],
        np.array([-5.0, 10.0]),
        [(-5.0, 10.0), (-5.0, 10.0), (-5.0, 10.0)],
        np.array([[-5.0, 10.0]] * 3),
    ])
    def test_accepts_all_bounds_formats(self, bounds):
        result = optimize(Rosenbrock(), bounds, 3, seed=0, budget=200,
                          config={"population_size": 10})
        best_x = np.array(result["best_x"])
        assert best_x.shape == (3,)
        assert np.all(best_x >= -5.0) and np.all(best_x <= 10.0)

    def test_per_dimension_bounds_are_respected(self):
        bounds = [(0.0, 1.0), (-10.0, -5.0)]
        spy = Spy(Rosenbrock())
        optimize(spy, bounds, 2, seed=0, budget=500, config={"population_size": 10})
        points = np.array(spy.f_points)
        assert np.all(points[:, 0] >= 0.0) and np.all(points[:, 0] <= 1.0)
        assert np.all(points[:, 1] >= -10.0) and np.all(points[:, 1] <= -5.0)

    def test_invalid_arguments_raise(self):
        func = Rosenbrock()
        with pytest.raises(ValueError):
            optimize(func, (-5.0, 10.0), 0, seed=0, budget=100)
        with pytest.raises(ValueError):
            optimize(func, (-5.0, 10.0), 2, seed=0, budget=0)
        with pytest.raises(ValueError):
            optimize(func, (-5.0, 10.0), 2, seed=0, budget=100, config={"population_size": 0})
        with pytest.raises(ValueError):
            optimize(func, (10.0, -5.0), 2, seed=0, budget=100)
        with pytest.raises(ValueError):
            optimize(func, np.zeros((5, 2)), 2, seed=0, budget=100)
