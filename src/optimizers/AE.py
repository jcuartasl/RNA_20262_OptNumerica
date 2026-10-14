"""Algoritmo evolutivo (AE) para optimización continua (minimización).

Cada generación: selección por torneo -> crossover BLX-alpha -> mutación
gaussiana por gen -> ajuste de límites -> evaluar hijos -> elitismo.

Los valores por defecto son puntos de partida para los pilotos; NO son
valores universalmente óptimos y deben validarse con experimentos.
"""
import numbers

import numpy as np

from ..functions.base import ObjectiveFunction
from ..metrics.evaluation_counter import CountedFunction


CONFIG_POR_DEFECTO = {
    "population_size": 50,
    "tournament_size": 3,
    "crossover_rate": 0.90,
    "alpha": 0.5,
    "mutation_rate": 0.10,        # probabilidad de mutar cada gen
    "mutation_sigma_rel": 0.10,  # tamaño de la mutación respecto al rango
    "n_elites": 2,
    "max_generations": None,
    "record_population": False,
    "tolerance": 1e-4,
}


# Validación

def es_entero(valor):
    return isinstance(valor, numbers.Integral) and not isinstance(valor, bool)


def preparar_limites(limites, dimension):
    """Convierte los límites en dos vectores.

    Formatos: (inferior, superior) para todas las coordenadas,
    o una lista de `dimension` pares (inferior, superior).
    """
    limites = np.asarray(limites, dtype=float)

    if limites.shape == (2,):
        limites_inferiores = np.full(dimension, limites[0])
        limites_superiores = np.full(dimension, limites[1])
    elif limites.shape == (dimension, 2):
        limites_inferiores = limites[:, 0]
        limites_superiores = limites[:, 1]
    else:
        raise ValueError(f"Forma de límites incorrecta para dimension={dimension}")

    if not np.all(np.isfinite(limites)):
        raise ValueError("Los límites deben ser finitos")

    if np.any(limites_inferiores >= limites_superiores):
        raise ValueError("Cada límite inferior debe ser menor que el superior")

    return limites_inferiores, limites_superiores


def validar_configuracion(configuracion):
    """Combina la configuración recibida con los valores por defecto y la valida."""
    config = dict(CONFIG_POR_DEFECTO)

    if configuracion is not None:
        if not isinstance(configuracion, dict):
            raise ValueError("configuracion debe ser un diccionario")
        desconocidas = set(configuracion) - set(CONFIG_POR_DEFECTO)
        if desconocidas:
            raise ValueError(f"Claves de configuración desconocidas: {sorted(desconocidas)}")

        config.update(configuracion)

    tamano_poblacion = config["population_size"]
    tamano_torneo = config["tournament_size"]
    numero_elites = config["n_elites"]

    if not es_entero(tamano_poblacion) or tamano_poblacion < 2:
        raise ValueError("population_size debe ser un entero >= 2")

    if not es_entero(tamano_torneo) or not 1 <= tamano_torneo <= tamano_poblacion:
        raise ValueError("tournament_size debe estar entre 1 y population_size")

    if not es_entero(numero_elites) or not 0 <= numero_elites < tamano_poblacion:
        raise ValueError("n_elites debe cumplir 0 <= n_elites < population_size")

    # Las comparaciones se escriben con "not" para que un NaN también sea rechazado
    if not 0.0 <= config["crossover_rate"] <= 1.0:
        raise ValueError("crossover_rate debe estar en [0, 1]")

    if not 0.0 <= config["mutation_rate"] <= 1.0:
        raise ValueError("mutation_rate debe estar en [0, 1]")

    if not config["alpha"] >= 0:
        raise ValueError("alpha debe ser >= 0")

    if not config["mutation_sigma_rel"] > 0:
        raise ValueError("mutation_sigma_rel debe ser > 0")

    if not config["tolerance"] >= 0:
        raise ValueError("tolerance debe ser >= 0")

    max_generaciones = config["max_generations"]
    if max_generaciones is not None and (not es_entero(max_generaciones) or max_generaciones < 0):
        raise ValueError("max_generations debe ser None o un entero >= 0")

    return config


# Operadores

def inicializar_poblacion(tamano, limites_inferiores, limites_superiores, generador):
    """Genera una población aleatoria dentro de los límites."""
    dimension = len(limites_inferiores)
    numeros = generador.random((tamano, dimension))
    return limites_inferiores + numeros * (limites_superiores - limites_inferiores)


def evaluar_poblacion(contador, poblacion, presupuesto):
    """Evalúa los individuos sin superar el presupuesto."""
    aptitud = []

    for individuo in poblacion:
        if not contador.can_evaluate_f(presupuesto):
            break

        valor = float(contador.evaluate(individuo.copy()))

        if not np.isfinite(valor):
            valor = np.inf

        aptitud.append(valor)

    return np.asarray(aptitud, dtype=float), len(aptitud)


def seleccion_torneo(aptitud, cantidad_padres, tamano_torneo, generador):
    """Selecciona padres mediante torneos (gana la menor aptitud)."""
    padres = np.empty(cantidad_padres, dtype=int)

    for i in range(cantidad_padres):
        participantes = generador.choice(len(aptitud), size=tamano_torneo, replace=False)
        padres[i] = participantes[np.argmin(aptitud[participantes])]

    return padres


def crossover_blx_alpha(padre_a, padre_b, alpha, tasa_crossover, generador):
    """Genera dos hijos usando BLX-alpha."""
    if generador.random() >= tasa_crossover:
        return padre_a.copy(), padre_b.copy()

    # Por coordenada: hijo ~ U(min - alpha*d, max + alpha*d), con d = |a - b|
    distancia = np.abs(padre_a - padre_b)
    limite_bajo = np.minimum(padre_a, padre_b) - alpha * distancia
    limite_alto = np.maximum(padre_a, padre_b) + alpha * distancia

    hijo_1 = generador.uniform(limite_bajo, limite_alto)
    hijo_2 = generador.uniform(limite_bajo, limite_alto)

    return hijo_1, hijo_2


def mutacion_gaussiana(hijos, tasa_mutacion, sigma, generador):
    """Aplica mutación gaussiana de forma independiente por gen."""
    mascara = generador.random(hijos.shape) < tasa_mutacion
    ruido = generador.normal(0.0, 1.0, hijos.shape) * sigma
    return hijos + ruido * mascara


def ajustar_limites(poblacion, limites_inferiores, limites_superiores):
    """Mantiene los individuos dentro de los límites (clipping)."""
    return np.clip(poblacion, limites_inferiores, limites_superiores)


def elitismo(poblacion, aptitud, numero_elites):
    """Conserva los mejores individuos de la generación."""
    indices = np.argsort(aptitud, kind="stable")[:numero_elites]
    return poblacion[indices].copy(), aptitud[indices].copy()


def actualizar_mejor(mejor_solucion, mejor_valor, poblacion, aptitud):
    """Actualiza la mejor solución encontrada (solo si mejora)."""
    if len(aptitud) == 0:
        return mejor_solucion, mejor_valor

    indice = int(np.argmin(aptitud))

    if aptitud[indice] < mejor_valor:
        return poblacion[indice].copy(), float(aptitud[indice])

    return mejor_solucion, mejor_valor


# Historial

def registrar_historial(historial, paso, mejor_solucion, mejor_valor, contador, poblacion=None):
    """Guarda el avance del algoritmo en cada generación."""
    entrada = {
        "step": int(paso),
        "best_f": float(mejor_valor),
        "equivalent_evaluations": int(contador.equivalent_evaluations),
        "best_x": [float(valor) for valor in mejor_solucion],
        "f_evaluations": int(contador.f_evaluations),
    }

    if poblacion is not None:
        entrada["population"] = poblacion.tolist()

    historial.append(entrada)


# Generación

def generar_hijos(poblacion, aptitud, cantidad_hijos, config, sigma,
                  limites_inferiores, limites_superiores, generador):
    """Selecciona padres y genera los hijos de una generación."""
    # Cada pareja de padres genera dos hijos
    cantidad_padres = cantidad_hijos + cantidad_hijos % 2
    indices_padres = seleccion_torneo(aptitud, cantidad_padres, config["tournament_size"], generador)

    hijos = np.empty((cantidad_padres, poblacion.shape[1]))

    for i in range(0, cantidad_padres, 2):
        padre_a = poblacion[indices_padres[i]]
        padre_b = poblacion[indices_padres[i + 1]]

        hijo_1, hijo_2 = crossover_blx_alpha(
            padre_a, padre_b, config["alpha"], config["crossover_rate"], generador
        )

        hijos[i] = hijo_1
        hijos[i + 1] = hijo_2

    hijos = hijos[:cantidad_hijos]

    # Mutación y ajuste de límites
    hijos = mutacion_gaussiana(hijos, config["mutation_rate"], sigma, generador)
    return ajustar_limites(hijos, limites_inferiores, limites_superiores)


# Algoritmo completo

def optimize(function, bounds, dimension, seed, budget, config=None):
    """Ejecuta el algoritmo evolutivo y devuelve sus resultados."""
    funcion = function
    limites = bounds
    semilla = seed
    presupuesto = budget
    configuracion = config

    if not isinstance(funcion, (ObjectiveFunction, CountedFunction)):
        raise TypeError("funcion debe ser una instancia de ObjectiveFunction o CountedFunction")

    if not es_entero(dimension) or dimension < 1:
        raise ValueError("dimension debe ser un entero >= 1")

    if not es_entero(presupuesto) or presupuesto < 1:
        raise ValueError("presupuesto debe ser un entero >= 1")

    if not es_entero(semilla):
        raise ValueError("semilla debe ser un entero")

    limites_inferiores, limites_superiores = preparar_limites(limites, dimension)
    config = validar_configuracion(configuracion)

    if isinstance(funcion, CountedFunction):
        contador = funcion
        if contador.dimension != dimension:
            raise ValueError("La dimensión del contador no coincide con dimension")
    else:
        contador = CountedFunction(funcion, dimension)

    tamano_poblacion = config["population_size"]
    numero_elites = config["n_elites"]
    max_generaciones = config["max_generations"]
    sigma = config["mutation_sigma_rel"] * (limites_superiores - limites_inferiores)

    # La semilla controla la población y todos los operadores aleatorios
    generador = np.random.default_rng(semilla)
    historial = []

    presupuesto_restante = presupuesto - contador.equivalent_evaluations
    if presupuesto_restante <= 0:
        raise ValueError("No queda presupuesto para iniciar la ejecución")

    # Población inicial (incompleta si el presupuesto es menor que la población)
    tamano_inicial = min(tamano_poblacion, presupuesto_restante)
    poblacion = inicializar_poblacion(tamano_inicial, limites_inferiores, limites_superiores, generador)
    aptitud, cantidad = evaluar_poblacion(contador, poblacion, presupuesto)
    poblacion = poblacion[:cantidad]

    if not np.any(np.isfinite(aptitud)):
        raise ValueError("La función no produjo valores finitos en la población inicial")

    mejor_solucion, mejor_valor = actualizar_mejor(None, np.inf, poblacion, aptitud)
    registrar_historial(historial, 0, mejor_solucion, mejor_valor, contador,
                        poblacion if config["record_population"] else None)

    generacion = 0

    while True:
        presupuesto_restante = presupuesto - contador.equivalent_evaluations
        if presupuesto_restante <= 0:
            break
        if max_generaciones is not None and generacion >= max_generaciones:
            break

        # Las élites pasan sin reevaluarse, así que solo se evalúan los hijos.
        # En la última generación se recorta para no pasar del presupuesto.
        cantidad_hijos = min(tamano_poblacion - numero_elites, presupuesto_restante)

        elites, aptitud_elites = elitismo(poblacion, aptitud, numero_elites)

        hijos = generar_hijos(
            poblacion, aptitud, cantidad_hijos, config, sigma,
            limites_inferiores, limites_superiores, generador
        )

        aptitud_hijos, cantidad = evaluar_poblacion(contador, hijos, presupuesto)
        hijos = hijos[:cantidad]

        if cantidad == 0:  # protección: evita un ciclo infinito si el contador no deja evaluar
            break

        mejor_solucion, mejor_valor = actualizar_mejor(
            mejor_solucion, mejor_valor, hijos, aptitud_hijos
        )

        # Nueva población = élites + hijos
        poblacion = np.vstack([elites, hijos])
        aptitud = np.concatenate([aptitud_elites, aptitud_hijos])

        generacion += 1
        registrar_historial(historial, generacion, mejor_solucion, mejor_valor, contador,
                            poblacion if config["record_population"] else None)

    exito = bool(abs(mejor_valor - contador.global_minimum_f) <= config["tolerance"])

    return {
        "best_x": mejor_solucion.tolist(),
        "best_f": float(mejor_valor),
        "f_evaluations": int(contador.f_evaluations),
        "gradient_evaluations": int(contador.gradient_evaluations),
        "equivalent_evaluations": int(contador.equivalent_evaluations),
        "iterations": int(generacion),
        "success": exito,
        "history": historial,
    }