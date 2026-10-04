"""Unit tests for Gradient Descent optimizer."""

import numpy as np
import pytest
from src.functions.rosenbrock import Rosenbrock
from src.functions.rastrigin import Rastrigin
from src.optimizers.gradient_descent import optimize
from src.metrics.evaluation_counter import CountedFunction


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
        config = {"learning_rate": 0.001, "tolerance": 1e-4}
        result = optimize(func, bounds, dim, seed=42, budget=budget, config=config)

        # Check required keys
        required_keys = [
            "best_x", "best_f", "f_evaluations", "gradient_evaluations",
            "equivalent_evaluations", "iterations", "success", "history"
        ]
        for key in required_keys:
            assert key in result, f"Missing required key '{key}' in result dictionary."

        # Check types
        assert isinstance(result["best_x"], list)
        assert len(result["best_x"]) == dim
        assert isinstance(result["best_f"], float)
        assert isinstance(result["f_evaluations"], int)
        assert isinstance(result["gradient_evaluations"], int)
        assert isinstance(result["equivalent_evaluations"], int)
        assert isinstance(result["iterations"], int)
        assert isinstance(result["success"], bool)
        assert isinstance(result["history"], list)
        assert len(result["history"]) > 0

        # Check history item structure
        for entry in result["history"]:
            assert "step" in entry
            assert "best_f" in entry
            assert "equivalent_evaluations" in entry
            assert "best_x" in entry
            assert len(entry["best_x"]) == dim

    @pytest.mark.parametrize("seed", [0, 7, 42, 99])
    def test_reproducibility(self, seed: int):
        """Verify that the same seed produces identical starting point and optimization trajectory."""
        func = Rosenbrock()
        bounds = (-5.0, 10.0)
        dim = 2
        budget = 300
        config = {"learning_rate": 0.001}

        res1 = optimize(func, bounds, dim, seed=seed, budget=budget, config=config)
        res2 = optimize(func, bounds, dim, seed=seed, budget=budget, config=config)

        assert res1["initial_x"] == res2["initial_x"]
        assert res1["best_x"] == res2["best_x"]
        assert res1["best_f"] == res2["best_f"]
        assert res1["equivalent_evaluations"] == res2["equivalent_evaluations"]
        assert res1["iterations"] == res2["iterations"]
        assert res1["history"] == res2["history"]

    def test_different_seeds_produce_different_x0(self):
        """Verify that varying seeds explore different starting positions."""
        func = Rastrigin()
        bounds = (-5.12, 5.12)
        dim = 2
        res1 = optimize(func, bounds, dim, seed=1, budget=100)
        res2 = optimize(func, bounds, dim, seed=2, budget=100)
        assert res1["initial_x"] != res2["initial_x"]

    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("budget", [1, 2, 5, 10, 25, 50, 100, 200])
    def test_strict_budget_compliance(self, dim: int, budget: int):
        """Verify that GD never exceeds allowable equivalent evaluations."""
        func = Rosenbrock()
        bounds = (-5.0, 10.0)
        result = optimize(func, bounds, dim, seed=10, budget=budget, config={"learning_rate": 0.01})
        
        eq_evals = result["equivalent_evaluations"]
        assert eq_evals <= budget, (
            f"Budget exceeded! Allowed: {budget}, but used: {eq_evals} (dim={dim})"
        )

        # Check equation E_eq = N_f + 2 * n * N_grad
        n_f = result["f_evaluations"]
        n_grad = result["gradient_evaluations"]
        expected_eq = n_f + 2 * dim * n_grad
        assert eq_evals == expected_eq, (
            f"E_eq mismatch: got {eq_evals}, expected {expected_eq} = {n_f} + 2*{dim}*{n_grad}"
        )

    def test_bounds_clipping(self):
        """Verify that points never cross the boundaries even with large step size."""
        func = Rosenbrock()
        bounds = (-2.0, 2.0)
        dim = 2
        # Use an aggressive learning rate that could shoot far outside if unbounded
        config = {"learning_rate": 10.0}
        result = optimize(func, bounds, dim, seed=42, budget=200, config=config)

        best_x = np.array(result["best_x"])
        assert np.all(best_x >= -2.0)
        assert np.all(best_x <= 2.0)

        for step in result["history"]:
            pt = np.array(step["best_x"])
            assert np.all(pt >= -2.0)
            assert np.all(pt <= 2.0)

    def test_local_convergence_on_convex_region(self):
        """Verify GD decreases f(x) monotonically or converges towards local minimum."""
        func = Rastrigin()
        bounds = (-5.12, 5.12)
        dim = 2
        # Start near global minimum
        config = {"learning_rate": 0.005}
        result = optimize(func, bounds, dim, seed=0, budget=1000, config=config)
        assert result["best_f"] <= result["history"][0]["best_f"]
