import numpy as np


def optimize(
    function,
    bounds,
    dimension,
    seed,
    budget,
    config=None
):
    """
    Optimizador simulado utilizado únicamente para desarrollar
    y probar el framework experimental.

    NO forma parte de los experimentos finales.
    """

    rng = np.random.default_rng(seed)

    lower, upper = bounds

    best_x = rng.uniform(lower, upper, size=dimension)

    initial_f = float(rng.uniform(10, 50))

    history = []

    best_f = initial_f

    evaluations = 0
    iteration = 0

    while evaluations < budget:

        improvement = rng.uniform(0.80, 0.98)

        best_f *= improvement

        evaluations += 100

        best_x = best_x * improvement

        history.append(
            {
                "iteration": iteration,
                "best_f": best_f,
                "best_x": best_x.tolist(),
                "equivalent_evaluations": min(evaluations, budget),
            }
        )

        iteration += 1

    return {
        "best_x": best_x.tolist(),
        "best_f": float(best_f),
        "f_evaluations": budget,
        "gradient_evaluations": 0,
        "equivalent_evaluations": budget,
        "iterations": iteration,
        "history": history,
    }