
"""PSO Optimizer module."""

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
    """Optimize an objective function using Particle Swarm Optimization (PSO).
 
    Update rules (per particle, per iteration; r1, r2 ~ U(0,1) per coordinate):
        v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        v <- clip(v, -v_max, v_max)
        x <- clip(x + v, lower, upper)          # common bounds policy (clipping)
 
    Budget policy: a whole swarm iteration (population_size evaluations of f)
    is started only if it fits entirely in the remaining budget. The initial
    swarm is the exception: if budget < population_size, the swarm is reduced
    to `budget` particles so that a result can always be returned.
 
    Args:
        function: ObjectiveFunction or CountedFunction instance.
        bounds: Search domain bounds (tuple for all dimensions or list of tuples per dimension).
        dimension: Number of dimensions n.
        seed: Random seed (positions, velocities and r1/r2 all come from it).
        budget: Maximum allowable equivalent evaluations E_eq.
        config: Optional configuration dictionary with hyperparameters:
            - population_size (int): Number of particles (default: 50).
            - inertia (float): Inertia weight w (default: 0.7).
            - cognitive (float): Cognitive coefficient c1 (default: 1.5).
            - social (float): Social coefficient c2 (default: 1.5).
            - v_max_fraction (float): v_max = fraction * (upper - lower) (default: 0.5).
            - tolerance (float): Threshold for abs(f_best - f*) to consider success (default: 1e-4).
            - record_positions (bool): Store the full swarm in each history entry,
              useful for animations (default: False).
 
    Returns:
        Dict conforming to Phase 0 common interface:
            best_x, best_f, f_evaluations, gradient_evaluations,
            equivalent_evaluations, iterations, success, history.
    """
    if dimension < 1:
        raise ValueError(f"Dimension must be positive, got {dimension}.")
    if budget < 1:
        raise ValueError(f"Budget must be positive, got {budget}.")
 
    cfg = config or {}
    population_size = int(cfg.get("population_size", 50))
    w = float(cfg.get("inertia", 0.7))
    c1 = float(cfg.get("cognitive", 1.5))
    c2 = float(cfg.get("social", 1.5))
    v_max_fraction = float(cfg.get("v_max_fraction", 0.5))
    tolerance = float(cfg.get("tolerance", 1e-4))
    record_positions = bool(cfg.get("record_positions", False))
 
    if population_size < 1:
        raise ValueError(f"population_size must be positive, got {population_size}.")
 
    lower_bounds, upper_bounds = _standardize_bounds(bounds, dimension)
 
    # Wrap with CountedFunction if not already wrapped
    if isinstance(function, CountedFunction):
        counter = function
    else:
        counter = CountedFunction(function, dimension)
 
    # Never evaluate more particles in the first batch than the remaining budget allows
    remaining = budget - counter.equivalent_evaluations
    if remaining < 1:
        return {
            "best_x": [],
            "best_f": float("inf"),
            "f_evaluations": int(counter.f_evaluations),
            "gradient_evaluations": int(counter.gradient_evaluations),
            "equivalent_evaluations": int(counter.equivalent_evaluations),
            "iterations": 0,
            "success": False,
            "history": [],
        }
    n_particles = min(population_size, remaining)
 
    rng = np.random.default_rng(seed)
    v_max = v_max_fraction * (upper_bounds - lower_bounds)
 
    # Initialization: positions uniform in bounds, velocities uniform in [-v_max/2, v_max/2]
    x = rng.uniform(lower_bounds, upper_bounds, size=(n_particles, dimension))
    v = rng.uniform(-v_max / 2.0, v_max / 2.0, size=(n_particles, dimension))
 
    # Initial evaluation (one counted call to f per particle)
    pbest_x = x.copy()
    pbest_f = np.array([counter.evaluate(p) for p in x], dtype=np.float64)
 
    g_idx = int(np.argmin(pbest_f))
    gbest_x = pbest_x[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])
 
    history: List[Dict[str, Any]] = []
 
    def _log(step: int) -> None:
        entry = {
            "step": step,
            "best_f": float(gbest_f),
            "equivalent_evaluations": int(counter.equivalent_evaluations),
            "best_x": gbest_x.tolist(),
        }
        if record_positions:
            entry["positions"] = x.tolist()
        history.append(entry)
 
    _log(0)
 
    iteration = 0
    while counter.equivalent_evaluations + n_particles <= budget:
        r1 = rng.random(size=x.shape)
        r2 = rng.random(size=x.shape)
 
        v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (gbest_x - x)
        v = np.clip(v, -v_max, v_max)
        x = np.clip(x + v, lower_bounds, upper_bounds)
 
        f_vals = np.array([counter.evaluate(p) for p in x], dtype=np.float64)
        iteration += 1
 
        improved = f_vals < pbest_f
        pbest_x[improved] = x[improved]
        pbest_f[improved] = f_vals[improved]
 
        # gbest only changes if some pbest strictly improves on it (best-so-far)
        idx = int(np.argmin(pbest_f))
        if pbest_f[idx] < gbest_f:
            gbest_f = float(pbest_f[idx])
            gbest_x = pbest_x[idx].copy()
 
        _log(iteration)
 
    # Calculate success against theoretical optimum f*
    f_star = getattr(function, "global_minimum_f", None)
    if f_star is None:
        f_star = getattr(counter, "global_minimum_f", 0.0)
    success = bool(abs(gbest_f - f_star) <= tolerance)
 
    return {
        "best_x": gbest_x.tolist(),
        "best_f": float(gbest_f),
        "f_evaluations": int(counter.f_evaluations),
        "gradient_evaluations": int(counter.gradient_evaluations),
        "equivalent_evaluations": int(counter.equivalent_evaluations),
        "iterations": int(iteration),
        "success": success,
        "history": history,
    }






