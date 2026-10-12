"""Rastrigin test function and its analytical gradient."""

from typing import Tuple
import numpy as np
from .base import ObjectiveFunction


class Rastrigin(ObjectiveFunction):
    """Rastrigin test function.
    
    Formula:
        f(x) = 10 * n + sum_{i=1}^{n} [ x_i^2 - 10 * cos(2 * pi * x_i) ]
        
    Properties:
        - Domain: [-5.12, 5.12]^n (proposed experimental domain)
        - Global minimum: x* = (0, 0, ..., 0), f(x*) = 0.0
        - Highly multimodal with regularly distributed local minima.
    """

    def __init__(self, bounds: Tuple[float, float] = (-5.12, 5.12)):
        super().__init__(name="Rastrigin", default_bounds=bounds, global_minimum_f=0.0)

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate Rastrigin function for a vector x of length n >= 1."""
        arr = self._validate_input(x)
        n = len(arr)
        if n < 1:
            raise ValueError(f"Rastrigin function requires dimension n >= 1, got n={n}.")

        return float(10.0 * n + np.sum(arr**2 - 10.0 * np.cos(2.0 * np.pi * arr)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the analytical gradient of Rastrigin function.
        
        Grad equation:
            df/dx_i = 2 * x_i + 20 * pi * sin(2 * pi * x_i)
        """
        arr = self._validate_input(x)
        if len(arr) < 1:
            raise ValueError(f"Rastrigin gradient requires dimension n >= 1, got n={len(arr)}.")

        return 2.0 * arr + 20.0 * np.pi * np.sin(2.0 * np.pi * arr)

    def global_minimum_x(self, dimension: int) -> np.ndarray:
        """Return global minimizer vector x* = [0, 0, ..., 0]."""
        if dimension < 1:
            raise ValueError(f"Dimension must be >= 1, got {dimension}.")
        return np.zeros(dimension, dtype=np.float64)
