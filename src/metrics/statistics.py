import numpy as np


def calculate_statistics(results, global_minimum=0.0, tolerance=1e-4):
    """
    Calcula las estadísticas requeridas para un conjunto de corridas.

    Parameters
    ----------
    results : list[dict]
        Resultados producidos por múltiples corridas de un optimizador.

    global_minimum : float
        Valor conocido del mínimo global de la función.

    tolerance : float
        Error máximo permitido para considerar una corrida exitosa.

    Returns
    -------
    dict
        Media, desviación estándar, mejor resultado,
        peor resultado, número de éxitos y tasa de éxito.
    """

    if len(results) == 0:
        raise ValueError("No se recibieron resultados para analizar.")

    final_values = np.array(
        [result["best_f"] for result in results],
        dtype=float
    )

    successes = np.abs(final_values - global_minimum) <= tolerance

    return {
        "runs": len(results),
        "mean": float(np.mean(final_values)),
        "std": float(np.std(final_values, ddof=1))
        if len(results) > 1 else 0.0,
        "best": float(np.min(final_values)),
        "worst": float(np.max(final_values)),
        "successes": int(np.sum(successes)),
        "success_rate": float(np.mean(successes) * 100),
    }