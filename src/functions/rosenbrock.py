"""Rosenbrock test function and its analytical gradient."""

from typing import Tuple
import numpy as np
from .base import ObjectiveFunction


class Rosenbrock(ObjectiveFunction):
    """Rosenbrock test function, also known as the Valley or Banana function.
    
    Formula:
        f(x) = sum_{i=1}^{n-1} [ 100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
        
    Properties:
        - Domain: [-5.0, 10.0]^n (proposed experimental domain)
        - Global minimum: x* = (1, 1, ..., 1), f(x*) = 0.0
        - Non-convex, ill-conditioned narrow parabolic valley.
    """

    def __init__(self, bounds: Tuple[float, float] = (-5.0, 10.0)):
        super().__init__(name="Rosenbrock", default_bounds=bounds, global_minimum_f=0.0)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Rosenbrock function for a vector x of length n >= 2."""
        arr = self._validate_input(x)
        if len(arr) < 2:
            raise ValueError(f"Rosenbrock function requires dimension n >= 2, got n={len(arr)}.")
        
        x_curr = arr[:-1]
        x_next = arr[1:]
        return float(np.sum(100.0 * (x_next - x_curr**2)**2 + (1.0 - x_curr)**2))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the analytical gradient of Rosenbrock function for length n >= 2.
        
        Grad equations:
            df/dx_1 = -400 * x_1 * (x_2 - x_1^2) + 2 * (x_1 - 1)
            df/dx_i = 200 * (x_i - x_{i-1}^2) - 400 * x_i * (x_{i+1} - x_i^2) + 2 * (x_i - 1)  [for 2 <= i <= n-1]
            df/dx_n = 200 * (x_n - x_{n-1}^2)
        """
        arr = self._validate_input(x)
        n = len(arr)
        if n < 2:
            raise ValueError(f"Rosenbrock gradient requires dimension n >= 2, got n={n}.")

        grad = np.zeros(n, dtype=np.float64)

        # First component (i = 1, 0-indexed: index 0)
        grad[0] = -400.0 * arr[0] * (arr[1] - arr[0]**2) + 2.0 * (arr[0] - 1.0)

        # Middle components (2 <= i <= n-1, 0-indexed: 1 to n-2)
        if n > 2:
            x_prev = arr[:-2]
            x_curr = arr[1:-1]
            x_next = arr[2:]
            grad[1:-1] = (
                200.0 * (x_curr - x_prev**2)
                - 400.0 * x_curr * (x_next - x_curr**2)
                + 2.0 * (x_curr - 1.0)
            )

        # Last component (i = n, 0-indexed: index -1)
        grad[-1] = 200.0 * (arr[-1] - arr[-2]**2)

        return grad

    def global_minimum_x(self, dimension: int) -> np.ndarray:
        """Return global minimizer vector x* = [1, 1, ..., 1]."""
        if dimension < 2:
            raise ValueError(f"Dimension must be >= 2, got {dimension}.")
        return np.ones(dimension, dtype=np.float64)
