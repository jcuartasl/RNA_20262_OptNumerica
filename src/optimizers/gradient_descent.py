"""Gradient Descent optimizer implementing the standard team interface."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ..metrics.evaluation_counter import CountedFunction


def _standardize_bounds(
    bounds: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray],
    dimension: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert bounds specification into standardized lower and upper 1D arrays."""
    if isinstance(bounds, tuple) and len(bounds) == 2 and isinstance(bounds[0], (int, float)):
        lower = np.full(dimension, bounds[0], dtype=np.float64)
        upper = np.full(dimension, bounds[1], dtype=np.float64)
    else:
        bounds_arr = np.asarray(bounds, dtype=np.float64)
        if bounds_arr.shape == (2,):
            lower = np.full(dimension, bounds_arr[0], dtype=np.float64)
            upper = np.full(dimension, bounds_arr[1], dtype=np.float64)
        elif bounds_arr.shape == (dimension, 2):
            lower = bounds_arr[:, 0]
            upper = bounds_arr[:, 1]
        else:
            raise ValueError(
                f"Invalid bounds shape {bounds_arr.shape} for dimension {dimension}. "
                "Expected (2,) or (dimension, 2)."
            )

    if np.any(lower >= upper):
        raise ValueError("All lower bounds must be strictly less than upper bounds.")

    return lower, upper


def optimize(
    function: Any,
    bounds: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray],
    dimension: int,
    seed: int,
    budget: int,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Optimize an objective function using Gradient Descent.
    
    Args:
        function: ObjectiveFunction or CountedFunction instance.
        bounds: Search domain bounds (tuple for all dimensions or list of tuples per dimension).
        dimension: Number of dimensions n.
        seed: Random seed for reproducible initialization of x0.
        budget: Maximum allowable equivalent evaluations E_eq.
        config: Optional configuration dictionary with hyperparameters:
            - learning_rate (float): Step size alpha (default: 0.001).
            - tolerance (float): Threshold for abs(f_best - f*) to consider success (default: 1e-4).
            - tol_grad (float): Optional threshold on ||grad|| for early convergence (default: 1e-8).
            
    Returns:
        Dict conforming to Phase 0 common interface:
            best_x, best_f, f_evaluations, gradient_evaluations,
            equivalent_evaluations, iterations, success, history.
    """
    if dimension < 1:
        raise ValueError(f"Dimension must be positive, got {dimension}.")
    if budget < 1:
        raise ValueError(f"Budget must be >= 1, got {budget}.")

    cfg = config or {}
    learning_rate = float(cfg.get("learning_rate", 0.001))
    tolerance = float(cfg.get("tolerance", 1e-4))
    tol_grad = float(cfg.get("tol_grad", 1e-8))

    lower_bounds, upper_bounds = _standardize_bounds(bounds, dimension)

    # Wrap with CountedFunction if not already wrapped
    if isinstance(function, CountedFunction):
        counter = function
    else:
        counter = CountedFunction(function, dimension)

    # Reproducible random initialization within bounds
    rng = np.random.default_rng(seed)
    x = rng.uniform(lower_bounds, upper_bounds)
    x0 = x.copy()

    history: List[Dict[str, Any]] = []

    # Initial evaluation
    if not counter.can_evaluate_f(budget):
        return {
            "best_x": x.tolist(),
            "best_f": float("inf"),
            "f_evaluations": counter.f_evaluations,
            "gradient_evaluations": counter.gradient_evaluations,
            "equivalent_evaluations": counter.equivalent_evaluations,
            "iterations": 0,
            "success": False,
            "history": [],
            "initial_x": x0.tolist(),
        }

    f_val = counter.evaluate(x)
    best_x = x.copy()
    best_f = f_val
    iteration = 0

    history.append({
        "step": 0,
        "best_f": float(best_f),
        "equivalent_evaluations": int(counter.equivalent_evaluations),
        "best_x": best_x.tolist()
    })

    # Optimization loop
    # One full iteration costs 2n (gradient) + 1 (f at the new point).
    # Only start an iteration if the WHOLE iteration fits in the budget;
    # otherwise a gradient would be charged without ever taking the step.
    iteration_cost = 2 * dimension + 1
    while True:
        if counter.equivalent_evaluations + iteration_cost > budget:
            break

        grad = counter.gradient(x)

        # Optional gradient norm convergence check
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm <= tol_grad:
            break

        # Gradient descent step
        x_next = x - learning_rate * grad

        # Bounds policy: clipping to domain
        x_next = np.clip(x_next, lower_bounds, upper_bounds)

        # Check if evaluating f at the new point would exceed budget (costs 1)
        if not counter.can_evaluate_f(budget):
            break

        f_next = counter.evaluate(x_next)
        iteration += 1
        x = x_next

        # Update best found so far
        if f_next < best_f:
            best_f = f_next
            best_x = x.copy()

        history.append({
            "step": iteration,
            "best_f": float(best_f),
            "equivalent_evaluations": int(counter.equivalent_evaluations),
            "best_x": best_x.tolist()
        })

    # Calculate success against theoretical optimum f*
    f_star = getattr(function, "global_minimum_f", 0.0)
    success = bool(abs(best_f - f_star) <= tolerance)

    return {
        "best_x": best_x.tolist(),
        "best_f": float(best_f),
        "f_evaluations": int(counter.f_evaluations),
        "gradient_evaluations": int(counter.gradient_evaluations),
        "equivalent_evaluations": int(counter.equivalent_evaluations),
        "iterations": int(iteration),
        "success": success,
        "history": history,
        "initial_x": x0.tolist(),
    }
