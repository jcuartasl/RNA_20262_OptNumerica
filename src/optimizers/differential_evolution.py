"""Differential Evolution (DE/rand/1/bin) optimizer implementing the standard team interface.

Variant implemented (minimization, synchronous update):
    mutation:   v = x_r1 + F * (x_r2 - x_r3),  r1, r2, r3 distinct and != i
    repair:     v <- clip(v, lower, upper)
    crossover:  u_j = v_j if (rand_j < CR or j == j_rand) else x_i,j
    selection:  x_i <- u if f(u) <= f(x_i)  (greedy, ties accepted)

Cost model: DE never calls the gradient, so N_grad = 0 and E_eq = N_f.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ..metrics.evaluation_counter import CountedFunction


def _standardize_bounds(
    bounds: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray],
    dimension: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert bounds specification into standardized lower and upper 1D arrays.

    Same behavior as in gradient_descent.py (candidate to move to a shared utils module).
    """
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


def _pick_distinct_indices(rng: np.random.Generator, i: int, pop_size: int) -> np.ndarray:
    """Pick r1, r2, r3 distinct from each other and different from the target index i."""
    candidates = np.delete(np.arange(pop_size), i)
    return rng.choice(candidates, size=3, replace=False)


def _binomial_crossover(
    rng: np.random.Generator, target: np.ndarray, mutant: np.ndarray, cr: float
) -> np.ndarray:
    """Binomial crossover; coordinate j_rand always comes from the mutant."""
    mask = rng.random(target.size) < cr
    mask[rng.integers(target.size)] = True
    return np.where(mask, mutant, target)


def optimize(
    function: Any,
    bounds: Union[Tuple[float, float], List[Tuple[float, float]], np.ndarray],
    dimension: int,
    seed: int,
    budget: int,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Optimize an objective function using Differential Evolution (rand/1/bin).

    Args:
        function: ObjectiveFunction or CountedFunction instance.
        bounds: Search domain bounds (tuple for all dimensions or list of tuples per dimension).
        dimension: Number of dimensions n.
        seed: Random seed (initial population, indices, crossover masks).
        budget: Maximum allowable equivalent evaluations E_eq (= N_f for DE).
        config: Optional configuration dictionary with hyperparameters:
            - population_size (int): NP, must be >= 4 (default: 50).
            - mutation_factor (float): F, scale of the difference vector (default: 0.8).
            - crossover_rate (float): CR in [0, 1] (default: 0.9).
            - tolerance (float): Threshold for abs(f_best - f*) to consider success (default: 1e-4).

    Returns:
        Dict conforming to Phase 0 common interface:
            best_x, best_f, f_evaluations, gradient_evaluations,
            equivalent_evaluations, iterations, success, history.

    Budget policy: the run never exceeds `budget`. The initial population needs NP
    evaluations; if the budget cannot cover them, the individuals that fit are evaluated
    and the run stops with iterations = 0. Otherwise the budget is checked before EACH
    trial evaluation, so the last generation may be partial. `iterations` counts
    generations started (the last one may be partial). The run does not stop early on
    success: it consumes the whole budget.
    """
    if dimension < 1:
        raise ValueError(f"Dimension must be positive, got {dimension}.")
    if budget < 1:
        raise ValueError(f"Budget must be >= 1, got {budget}.")

    cfg = config or {}
    pop_size = int(cfg.get("population_size", 50))
    F = float(cfg.get("mutation_factor", 0.8))
    CR = float(cfg.get("crossover_rate", 0.9))
    tolerance = float(cfg.get("tolerance", 1e-4))

    if pop_size < 4:
        raise ValueError(f"population_size must be >= 4 for rand/1, got {pop_size}.")
    if not 0.0 <= CR <= 1.0:
        raise ValueError(f"crossover_rate must be in [0, 1], got {CR}.")
    if F <= 0.0:
        raise ValueError(f"mutation_factor must be > 0, got {F}.")

    lower_bounds, upper_bounds = _standardize_bounds(bounds, dimension)

    # Wrap with CountedFunction if not already wrapped
    if isinstance(function, CountedFunction):
        counter = function
    else:
        counter = CountedFunction(function, dimension)

    rng = np.random.default_rng(seed)

    # Initial population, uniform within bounds. Unevaluated individuals keep fitness = inf.
    pop = rng.uniform(lower_bounds, upper_bounds, size=(pop_size, dimension))
    fit = np.full(pop_size, np.inf)

    n_evaluated = 0
    while n_evaluated < pop_size and counter.can_evaluate_f(budget):
        fit[n_evaluated] = counter.evaluate(pop[n_evaluated])
        n_evaluated += 1

    history: List[Dict[str, Any]] = []

    def record(step: int) -> None:
        best_idx = int(np.argmin(fit))
        history.append({
            "step": step,
            "best_f": float(fit[best_idx]),
            "equivalent_evaluations": int(counter.equivalent_evaluations),
            "best_x": pop[best_idx].tolist(),
        })

    record(0)

    generation = 0
    if n_evaluated == pop_size:
        while counter.can_evaluate_f(budget):
            # Synchronous update: trials are built from the current generation only.
            new_pop = pop.copy()
            new_fit = fit.copy()

            for i in range(pop_size):
                if not counter.can_evaluate_f(budget):
                    break

                # Mutation: three distinct indices, all different from i
                r1, r2, r3 = _pick_distinct_indices(rng, i, pop_size)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # Bounds policy: clipping to domain (same as GD / PSO / EA)
                mutant = np.clip(mutant, lower_bounds, upper_bounds)

                # Binomial crossover with j_rand (at least one coordinate from the mutant)
                trial = _binomial_crossover(rng, pop[i], mutant, CR)

                # Greedy selection (ties accepted)
                f_trial = counter.evaluate(trial)
                if f_trial <= fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = f_trial

            pop, fit = new_pop, new_fit
            generation += 1
            record(generation)

    best_idx = int(np.argmin(fit))
    best_x = pop[best_idx].copy()
    best_f = float(fit[best_idx])

    # Calculate success against theoretical optimum f*
    f_star = getattr(function, "global_minimum_f", 0.0)
    success = bool(abs(best_f - f_star) <= tolerance)

    return {
        "best_x": best_x.tolist(),
        "best_f": best_f,
        "f_evaluations": int(counter.f_evaluations),
        "gradient_evaluations": int(counter.gradient_evaluations),
        "equivalent_evaluations": int(counter.equivalent_evaluations),
        "iterations": int(generation),
        "success": success,
        "history": history,
    }
