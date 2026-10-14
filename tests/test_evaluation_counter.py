"""Unit tests for the CountedFunction evaluation counter."""

import numpy as np
from src.functions.rosenbrock import Rosenbrock
from src.metrics.evaluation_counter import CountedFunction


def test_counters_start_at_zero():
    cf = CountedFunction(Rosenbrock(), dimension=3)
    assert cf.f_evaluations == 0
    assert cf.gradient_evaluations == 0
    assert cf.equivalent_evaluations == 0


def test_each_call_increments_its_own_counter():
    cf = CountedFunction(Rosenbrock(), dimension=3)
    x = np.array([0.5, -1.0, 2.0])
    cf.evaluate(x)
    cf(x)  # __call__ must also count
    cf.gradient(x)
    assert cf.f_evaluations == 2
    assert cf.gradient_evaluations == 1


def test_equivalent_evaluations_formula():
    """E_eq = N_f + 2 * n * N_grad (n = 3 -> each gradient costs 6)."""
    cf = CountedFunction(Rosenbrock(), dimension=3)
    x = np.array([0.5, -1.0, 2.0])
    for _ in range(4):
        cf.evaluate(x)
    for _ in range(5):
        cf.gradient(x)
    assert cf.equivalent_evaluations == 4 + 2 * 3 * 5


def test_wrapper_returns_same_values_as_base():
    base = Rosenbrock()
    cf = CountedFunction(base, dimension=2)
    x = np.array([1.5, -0.3])
    assert cf.evaluate(x) == base.evaluate(x)
    np.testing.assert_array_equal(cf.gradient(x), base.gradient(x))


def test_can_evaluate_checks_remaining_budget():
    cf = CountedFunction(Rosenbrock(), dimension=2)  # gradient costs 4
    x = np.array([0.0, 0.0])
    cf.evaluate(x)                       # E_eq = 1
    assert cf.can_evaluate_gradient(5)   # 1 + 4 = 5 <= 5
    assert not cf.can_evaluate_gradient(4)
    assert cf.can_evaluate_f(2)          # 1 + 1 = 2 <= 2
    assert not cf.can_evaluate_f(1)


def test_reset():
    cf = CountedFunction(Rosenbrock(), dimension=2)
    x = np.array([0.0, 0.0])
    cf.evaluate(x)
    cf.gradient(x)
    cf.reset()
    assert cf.equivalent_evaluations == 0
