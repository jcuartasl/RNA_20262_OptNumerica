"""Evaluation counter and fair-comparison wrapper for objective functions."""

from typing import Tuple, Union
import numpy as np
from ..functions.base import ObjectiveFunction


class CountedFunction:
    """Wrapper that accurately tracks real calls to f(x) and grad(f)(x).
    
    Attributes:
        base_function: The underlying ObjectiveFunction.
        dimension: Problem dimension n.
        f_evaluations: Integer count N_f of function evaluations.
        gradient_evaluations: Integer count N_grad of gradient evaluations.
    """

    def __init__(self, base_function: ObjectiveFunction, dimension: int):
        self.base_function = base_function
        self.dimension = int(dimension)
        self.f_evaluations = 0
        self.gradient_evaluations = 0

    @property
    def name(self) -> str:
        return self.base_function.name

    @property
    def default_bounds(self) -> Tuple[float, float]:
        return self.base_function.default_bounds

    @property
    def global_minimum_f(self) -> float:
        return self.base_function.global_minimum_f

    def global_minimum_x(self, dimension: int) -> np.ndarray:
        return self.base_function.global_minimum_x(dimension)

    @property
    def equivalent_evaluations(self) -> int:
        """Calculate total equivalent evaluations E_eq = N_f + 2 * n * N_grad."""
        return self.f_evaluations + 2 * self.dimension * self.gradient_evaluations

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate f(x) and increment N_f by 1."""
        self.f_evaluations += 1
        return self.base_function.evaluate(x)

    def __call__(self, x: np.ndarray) -> float:
        """Callable interface forwarding to evaluate(x)."""
        return self.evaluate(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute grad(f)(x) and increment N_grad by 1."""
        self.gradient_evaluations += 1
        return self.base_function.gradient(x)

    def can_evaluate_f(self, budget: int) -> bool:
        """Check if one additional f evaluation would fit in budget."""
        return (self.equivalent_evaluations + 1) <= budget

    def can_evaluate_gradient(self, budget: int) -> bool:
        """Check if one additional gradient evaluation would fit in budget."""
        return (self.equivalent_evaluations + 2 * self.dimension) <= budget

    def reset(self):
        """Reset evaluation counters to zero."""
        self.f_evaluations = 0
        self.gradient_evaluations = 0
