"""Unit tests for Gradient Descent optimizer."""

import numpy as np
import pytest
from src.functions.base import ObjectiveFunction
from src.functions.rosenbrock import Rosenbrock
from src.functions.rastrigin import Rastrigin
from src.optimizers.gradient_descent import optimize


# ---------------------------------------------------------
# Test helpers
# ---------------------------------------------------------
class Sphere(ObjectiveFunction):
    """Convex quadratic f(x) = sum(x_i^2), used only to test GD correctness.

    Gradient is 2x, Lipschitz constant L = 2, so fixed-step GD converges for
    0 < lr < 2 / L = 1. With lr = 0.1 each step multiplies x by (1 - 0.2) = 0.8.
    """

    def __init__(self):
        super().__init__(name="Sphere", default_bounds=(-5.0, 5.0), global_minimum_f=0.0)

    def evaluate(self, x):
        return float(np.sum(self._validate_input(x) ** 2))

    def gradient(self, x):
        return 2.0 * self._validate_input(x)

    def global_minimum_x(self, dimension):
        return np.zeros(dimension)


class Spy(ObjectiveFunction):
    """Wraps a function and records EVERY point where f or grad is requested."""

    def __init__(self, base: ObjectiveFunction):
        super().__init__(base.name, base.default_bounds, base.global_minimum_f)
        self.base = base
        self.f_points = []
        self.grad_points = []

    def evaluate(self, x):
        self.f_points.append(np.array(x, dtype=float))
        return self.base.evaluate(x)

    def gradient(self, x):
        self.grad_points.append(np.array(x, dtype=float))
        return self.base.gradient(x)

    def global_minimum_x(self, dimension):
        return self.base.global_minimum_x(dimension)


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------
class TestGradientDescent:
    @pytest.mark.parametrize("FunctionClass,dim,bounds", [
        (Rosenbrock, 2, (-5.0, 10.0)),
        (Rosenbrock, 3, (-5.0, 10.0)),
        (Rastrigin, 2, (-5.12, 5.12)),
        (Rastrigin, 3, (-5.12, 5.12)),
    ])
    def test_common_interface_schema(self, FunctionClass, dim, bounds):
        """Verify that the optimizer returns the exact contract required by Phase 0."""
        func = FunctionClass(bounds=bounds)
        budget = 500
        config = {"learning_rate": 0.0001, "tolerance": 1e-4}
        result = optimize(func, bounds, dim, seed=42, budget=budget, config=config)

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

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_reproducibility(self, seed: int):
        """Same seed -> identical starting point and optimization trajectory."""
        func = Rosenbrock()
        config = {"learning_rate": 0.0001}
        res1 = optimize(func, (-5.0, 10.0), 2, seed=seed, budget=300, config=config)
        res2 = optimize(func, (-5.0, 10.0), 2, seed=seed, budget=300, config=config)

        assert res1["initial_x"] == res2["initial_x"]
        assert res1["best_x"] == res2["best_x"]
        assert res1["best_f"] == res2["best_f"]
        assert res1["equivalent_evaluations"] == res2["equivalent_evaluations"]
        assert res1["iterations"] == res2["iterations"]
        assert res1["history"] == res2["history"]

    def test_different_seeds_produce_different_x0(self):
        func = Rastrigin()
        res1 = optimize(func, (-5.12, 5.12), 2, seed=1, budget=100)
        res2 = optimize(func, (-5.12, 5.12), 2, seed=2, budget=100)
        assert res1["initial_x"] != res2["initial_x"]

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("budget", [1, 2, 5, 10, 25, 50, 100, 200, 1000, 10000])
    def test_strict_budget_compliance(self, dim: int, budget: int):
        """GD never exceeds the budget and E_eq = N_f + 2*n*N_grad."""
        result = optimize(Rosenbrock(), (-5.0, 10.0), dim, seed=10, budget=budget,
                          config={"learning_rate": 0.0001})

        eq_evals = result["equivalent_evaluations"]
        assert eq_evals <= budget, f"Budget exceeded: allowed {budget}, used {eq_evals} (dim={dim})"

        n_f = result["f_evaluations"]
        n_grad = result["gradient_evaluations"]
        assert eq_evals == n_f + 2 * dim * n_grad

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("budget", [1, 4, 5, 6, 7, 50, 10000])
    def test_budget_used_exactly(self, dim: int, budget: int):
        """Without early stopping, GD uses exactly 1 + k*(2n + 1) equivalent evaluations.

        Cost model: 1 initial f(x0), then each iteration = 1 gradient (2n) + 1 f.
        k is the largest number of full iterations that fits in the budget.
        """
        k = (budget - 1) // (2 * dim + 1)
        result = optimize(Rosenbrock(), (-5.0, 10.0), dim, seed=3, budget=budget,
                          config={"learning_rate": 0.0001, "tol_grad": 0.0})
        assert result["iterations"] == k
        assert result["f_evaluations"] == 1 + k
        assert result["gradient_evaluations"] == k
        assert result["equivalent_evaluations"] == 1 + k * (2 * dim + 1)

    def test_single_step_matches_update_rule(self):
        """One iteration must be exactly x1 = clip(x0 - lr * grad(x0), lower, upper)."""
        dim, lr = 3, 0.0001
        spy = Spy(Rosenbrock())
        budget = 1 + (2 * dim + 1)  # f(x0) + exactly one iteration
        result = optimize(spy, (-5.0, 10.0), dim, seed=5, budget=budget,
                          config={"learning_rate": lr})

        x0 = np.array(result["initial_x"])
        expected_x1 = np.clip(x0 - lr * Rosenbrock().gradient(x0), -5.0, 10.0)
        assert result["iterations"] == 1
        np.testing.assert_allclose(spy.grad_points[0], x0)
        np.testing.assert_allclose(spy.f_points[1], expected_x1, rtol=0, atol=1e-15)

    @pytest.mark.parametrize("lr", [0.001, 10.0])
    def test_every_iterate_within_bounds(self, lr: float):
        """ALL evaluated points (not only best_x) stay inside the domain.

        lr=0.001 on Rosenbrock [-5, 10] is the case that bounces between corners,
        so clipping is actually exercised here.
        """
        lower, upper = -5.0, 10.0
        spy = Spy(Rosenbrock())
        optimize(spy, (lower, upper), 2, seed=42, budget=500, config={"learning_rate": lr})

        points = np.array(spy.f_points + spy.grad_points)
        assert len(points) > 10
        assert np.all(points >= lower) and np.all(points <= upper)
        # Clipping must have been active at least once (some coordinate on the boundary)
        assert np.any(np.isclose(points, lower) | np.isclose(points, upper))

    def test_converges_on_convex_quadratic(self):
        """On a convex quadratic with a stable step, GD must reach the minimum.

        Each step multiplies x by 0.8, so after k steps |x_k| = 0.8^k |x0|.
        With budget 2000 in 2D (~400 iterations) f is far below 1e-10.
        """
        result = optimize(Sphere(), (-5.0, 5.0), 2, seed=0, budget=2000,
                          config={"learning_rate": 0.1, "tolerance": 1e-4})
        assert result["best_f"] < 1e-10
        assert result["success"] is True
        np.testing.assert_allclose(result["best_x"], [0.0, 0.0], atol=1e-5)

    def test_unstable_step_does_not_converge_on_quadratic(self):
        """Negative control: with lr > 2/L = 1 fixed-step GD must NOT converge."""
        result = optimize(Sphere(), (-5.0, 5.0), 2, seed=0, budget=2000,
                          config={"learning_rate": 1.5, "tolerance": 1e-4})
        assert result["success"] is False

    def test_rastrigin_gd_stays_in_local_basin(self):
        """With a small stable step GD ends at a stationary point near x0 (local search).

        Rastrigin local minima sit near integer coordinates, so the final point
        should be close to an integer vector and have a (near) zero gradient.
        """
        func = Rastrigin()
        result = optimize(func, (-5.12, 5.12), 2, seed=0, budget=10000,
                          config={"learning_rate": 0.001})
        best_x = np.array(result["best_x"])
        assert np.linalg.norm(func.gradient(best_x)) < 1e-3
        np.testing.assert_allclose(best_x, np.round(best_x), atol=0.05)
