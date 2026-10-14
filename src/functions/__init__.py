"""Optimization test functions package."""

from typing import Dict, Type
from .base import ObjectiveFunction
from .rosenbrock import Rosenbrock
from .rastrigin import Rastrigin

FUNCTIONS: Dict[str, Type[ObjectiveFunction]] = {
    "rosenbrock": Rosenbrock,
    "rastrigin": Rastrigin,
}


def get_function(name: str, **kwargs) -> ObjectiveFunction:
    """Retrieve an instance of an objective function by name.
    
    Args:
        name: Name of the function ('rosenbrock' or 'rastrigin', case-insensitive).
        **kwargs: Optional arguments passed to the constructor (e.g. custom bounds).
        
    Returns:
        Instance of ObjectiveFunction.
    """
    key = name.strip().lower()
    if key not in FUNCTIONS:
        available = ", ".join(FUNCTIONS.keys())
        raise ValueError(f"Unknown function '{name}'. Available functions: {available}")
    return FUNCTIONS[key](**kwargs)


__all__ = ["ObjectiveFunction", "Rosenbrock", "Rastrigin", "get_function", "FUNCTIONS"]
