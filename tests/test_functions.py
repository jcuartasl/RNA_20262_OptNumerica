"""Unit tests for objective functions and their analytical gradients."""

import numpy as np
import pytest
from src.functions.rosenbrock import Rosenbrock
from src.functions.rastrigin import Rastrigin
from src.functions import get_function


def numerical_gradient(func, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Compute central difference approximation of the gradient of func at x."""
    n = len(x)
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (func.evaluate(x_plus) - func.evaluate(x_minus)) / (2.0 * h)
    return grad


# ---------------------------------------------------------
# Tests for Rosenbrock
# ---------------------------------------------------------
class TestRosenbrock:
    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_global_minimum_value(self, dim: int):
        rosen = Rosenbrock()
        x_opt = rosen.global_minimum_x(dim)
        assert len(x_opt) == dim
        np.testing.assert_allclose(x_opt, np.ones(dim))
        f_val = rosen.evaluate(x_opt)
        assert abs(f_val - 0.0) < 1e-12

    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_gradient_at_minimum(self, dim: int):
        rosen = Rosenbrock()
        x_opt = rosen.global_minimum_x(dim)
        grad = rosen.gradient(x_opt)
        assert grad.shape == (dim,)
        np.testing.assert_allclose(grad, np.zeros(dim), atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_gradient_vs_central_differences(self, dim: int):
        rosen = Rosenbrock()
        rng = np.random.default_rng(42)
        # 50 random points over the WHOLE experimental domain [-5, 10].
        # Tolerance: f reaches ~1e6 here, so central-difference round-off is about
        # eps * |f| / h ~ 2e-16 * 1e6 / 1e-6 = 2e-4 (absolute). Hence atol=1e-3,
        # while rtol=1e-6 keeps the check strict relative to gradients of ~1e5.
        for _ in range(50):
            x = rng.uniform(-5.0, 10.0, size=dim)
            grad_analytic = rosen.gradient(x)
            grad_numerical = numerical_gradient(rosen, x, h=1e-6)
            np.testing.assert_allclose(grad_analytic, grad_numerical, rtol=1e-6, atol=1e-3)

    def test_invalid_dimension(self):
        rosen = Rosenbrock()
        with pytest.raises(ValueError):
            rosen.evaluate(np.array([1.0]))
        with pytest.raises(ValueError):
            rosen.gradient(np.array([1.0]))


# ---------------------------------------------------------
# Tests for Rastrigin
# ---------------------------------------------------------
class TestRastrigin:
    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_global_minimum_value(self, dim: int):
        rastrigin = Rastrigin()
        x_opt = rastrigin.global_minimum_x(dim)
        assert len(x_opt) == dim
        np.testing.assert_allclose(x_opt, np.zeros(dim))
        f_val = rastrigin.evaluate(x_opt)
        assert abs(f_val - 0.0) < 1e-12

    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_gradient_at_minimum(self, dim: int):
        rastrigin = Rastrigin()
        x_opt = rastrigin.global_minimum_x(dim)
        grad = rastrigin.gradient(x_opt)
        assert grad.shape == (dim,)
        np.testing.assert_allclose(grad, np.zeros(dim), atol=1e-10)

    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_gradient_vs_central_differences(self, dim: int):
        rastrigin = Rastrigin()
        rng = np.random.default_rng(123)
        # 50 random points over the whole domain [-5.12, 5.12]. Here |f| < ~100,
        # so round-off error is ~1e-8 and truncation error O(h^2) is ~1e-9.
        for _ in range(50):
            x = rng.uniform(-5.12, 5.12, size=dim)
            grad_analytic = rastrigin.gradient(x)
            grad_numerical = numerical_gradient(rastrigin, x, h=1e-6)
            np.testing.assert_allclose(grad_analytic, grad_numerical, rtol=1e-6, atol=1e-6)

    def test_central_difference_check_detects_wrong_sign(self):
        """Negative control: the verification must FAIL for a sign-flipped gradient.

        Shows the central-difference test is not vacuous: a gradient written as
        2x - 20*pi*sin(2*pi*x) (wrong sign on the periodic term) is rejected.
        """
        rastrigin = Rastrigin()
        x = np.array([0.25, -1.3, 2.7])
        wrong_grad = 2.0 * x - 20.0 * np.pi * np.sin(2.0 * np.pi * x)
        grad_numerical = numerical_gradient(rastrigin, x, h=1e-6)
        assert not np.allclose(wrong_grad, grad_numerical, rtol=1e-6, atol=1e-6)

    def test_invalid_dimension(self):
        rastrigin = Rastrigin()
        with pytest.raises(ValueError):
            rastrigin.evaluate(np.array([]))


# ---------------------------------------------------------
# Tests for Function Factory
# ---------------------------------------------------------
def test_get_function():
    f1 = get_function("rosenbrock")
    assert isinstance(f1, Rosenbrock)
    f2 = get_function("rastrigin")
    assert isinstance(f2, Rastrigin)
    with pytest.raises(ValueError):
        get_function("non_existent_function")
