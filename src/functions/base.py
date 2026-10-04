"""Base class for objective test functions in numerical optimization."""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class ObjectiveFunction(ABC):
    """Abstract base class defining the contract for test objective functions."""

    def __init__(self, name: str, default_bounds: Tuple[float, float], global_minimum_f: float = 0.0):
        self.name = name
        self.default_bounds = default_bounds
        self.global_minimum_f = float(global_minimum_f)

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the objective function at point x.
        
        Args:
            x: 1D numpy array of coordinates.
            
        Returns:
            Scalar objective function value f(x).
        """
        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute the analytical gradient vector of f at point x.
        
        Args:
            x: 1D numpy array of coordinates of length n.
            
        Returns:
            1D numpy array of length n representing grad(f)(x).
        """
        pass

    @abstractmethod
    def global_minimum_x(self, dimension: int) -> np.ndarray:
        """Return the analytical global minimizer x* for the given dimension."""
        pass

    def __call__(self, x: np.ndarray) -> float:
        """Allow calling the function instance directly: f(x)."""
        return self.evaluate(x)

    @staticmethod
    def _validate_input(x: np.ndarray) -> np.ndarray:
        """Ensure x is a 1D float numpy array."""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected a 1D array, got shape {arr.shape} with {arr.ndim} dimensions.")
        return arr
