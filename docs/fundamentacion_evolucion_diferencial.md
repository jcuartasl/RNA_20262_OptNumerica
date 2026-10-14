# Evolución Diferencial (DE): fundamentación e implementación

**Trabajo 1, Parte 1: Optimización numérica** · Redes neuronales y algoritmos bioinspirados · UNAL, Facultad de Minas, 2026-02
**Algoritmo:** Evolución Diferencial (DE / rand / 1 / bin)
**Módulo:** `src/optimizers/differential_evolution.py` · **Tests:** `tests/test_differential_evolution.py`

Este documento recoge la fundamentación matemática de la evolución diferencial, explica cómo se implementó en el proyecto y resume todo lo que se hizo con este algoritmo. La teoría general de los algoritmos evolutivos se desarrolla también en `docs/fundamentacion_ae_de.md`; aquí el foco es la variante exacta que se programó y las decisiones tomadas.

---

## 1. Problema que resuelve el algoritmo

Se busca el mínimo global de una función objetivo $f:\mathbb{R}^n \to \mathbb{R}$ sobre un dominio acotado $[l, u]^n$, usando solo evaluaciones de $f$ (sin gradiente). El equipo escogió dos funciones de prueba, optimizadas en dimensión $n = 2$ y $n = 3$:

| Función | Fórmula | Dominio experimental | Mínimo global |
|---|---|---|---|
| Rosenbrock | $f(x)=\sum_{i=1}^{n-1}\left[100\,(x_{i+1}-x_i^2)^2+(1-x_i)^2\right]$ | $[-5,\,10]^n$ | $x^*=(1,\dots,1)$, $f(x^*)=0$ |
| Rastrigin | $f(x)=10n+\sum_{i=1}^{n}\left[x_i^2-10\cos(2\pi x_i)\right]$ | $[-5.12,\,5.12]^n$ | $x^*=(0,\dots,0)$, $f(x^*)=0$ |

Rosenbrock tiene un único mínimo, ubicado en un valle estrecho y curvo (mal condicionado). Rastrigin es multimodal, con una cuadrícula regular de mínimos locales. Ambas ponen a prueba comportamientos distintos de un optimizador.

---

## 2. Fundamento teórico

### 2.1 Idea general

La evolución diferencial es una metaheurística **poblacional y estocástica** para optimización continua, propuesta por Storn y Price (1997). Mantiene una población de $NP$ candidatos y, en cada generación, intenta mejorar a cada uno construyendo un vector de prueba (*trial*) a partir de **diferencias entre otros individuos de la población**. El trial reemplaza al candidato original solo si no es peor.

Su rasgo distintivo es cómo se perturba: en vez de sumar ruido con una distribución fija, el paso de mutación es proporcional a la diferencia entre dos individuos. Cuando la población está dispersa, los pasos son grandes (exploración); cuando converge, las diferencias se encogen y el algoritmo refina la solución (explotación) sin que haya que ajustar una desviación de mutación.

### 2.2 Notación

- $NP$: tamaño de la población.
- $x_{i,g}\in\mathbb{R}^n$: individuo $i$ en la generación $g$.
- $F>0$: factor de mutación (escala de la diferencia).
- $CR\in[0,1]$: tasa de cruce.
- $l, u$: límites inferior y superior del dominio.

### 2.3 Operadores (variante DE/rand/1/bin)

**Inicialización.** Cada individuo se sortea uniformemente en el dominio:

$$x_{i,0}\sim \mathcal{U}(l,\,u),\qquad i=1,\dots,NP.$$

**Mutación (rand/1).** Para cada individuo objetivo $x_i$ se eligen tres índices $r_1,r_2,r_3$ distintos entre sí y distintos de $i$, y se construye el vector mutante:

$$v_i = x_{r_1} + F\,(x_{r_2}-x_{r_3}).$$

Por eso se exige $NP\ge 4$: hacen falta al menos cuatro individuos distintos.

**Reparación de límites.** Si el mutante sale del dominio, cada coordenada se recorta (*clipping*):

$$v_{i,j}\leftarrow \min\!\big(\max(v_{i,j},\,l_j),\,u_j\big).$$

**Cruce binomial.** Se mezclan el objetivo y el mutante coordenada a coordenada:

$$u_{i,j}=\begin{cases} v_{i,j} & \text{si } \mathrm{rand}_j<CR \ \text{ o } \ j=j_{\text{rand}},\\ x_{i,j} & \text{en otro caso,}\end{cases}$$

donde $\mathrm{rand}_j\sim\mathcal{U}[0,1)$ y $j_{\text{rand}}$ es una coordenada elegida al azar. Esta última garantiza que el trial difiera del objetivo en al menos una coordenada; sin ella, con $CR$ bajo el trial podría ser idéntico al objetivo y se gastaría una evaluación sin información nueva.

**Selección uno a uno (voraz).**

$$x_{i,g+1}=\begin{cases} u_i & \text{si } f(u_i)\le f(x_{i,g}),\\ x_{i,g} & \text{en otro caso.}\end{cases}$$

Como la selección nunca acepta un individuo peor, el mejor valor de la población **nunca empeora** de una generación a otra. Esa propiedad es una consecuencia directa del operador y se verifica en los tests.

### 2.4 Papel de los parámetros

| Parámetro | Qué controla | Efecto esperado (hipótesis a contrastar con los experimentos) |
|---|---|---|
| $F$ | Tamaño del paso de mutación | Valores altos favorecen la exploración pero ralentizan la convergencia; valores bajos contraen la población rápido y aumentan el riesgo de quedar atrapado en un mínimo local |
| $CR$ | Cuántas coordenadas toma el trial del mutante | $CR$ alto cambia muchas coordenadas a la vez, lo que ayuda en funciones con variables acopladas como Rosenbrock; $CR$ bajo explora coordenada por coordenada |
| $NP$ | Diversidad de la población | Más individuos amplían la exploración, pero con presupuesto fijo se pagan con menos generaciones |

### 2.5 Diferencia con el algoritmo evolutivo general

En el algoritmo evolutivo implementado por el equipo, la mutación suma una perturbación aleatoria externa a un individuo, y el reemplazo se decide de forma global (élite y selección). En DE la perturbación sale de la diferencia entre individuos de la propia población, y la competencia es local: cada trial compite únicamente contra el individuo que lo originó.

### 2.6 Garantías

DE es una heurística: **no hay garantía general de convergencia al mínimo global** en tiempo finito. Lo que sí se cumple, por construcción, es que el mejor valor encontrado es no creciente. Cualquier afirmación sobre éxito o calidad se apoya en las corridas independientes del experimento, no en una prueba de optimalidad.

---

## 3. Implementación en el proyecto

### 3.1 Interfaz

El módulo expone una única función pública, con la firma común acordada para los cuatro optimizadores:

```python
optimize(function, bounds, dimension, seed, budget, config=None) -> dict
```

| Argumento | Descripción |
|---|---|
| `function` | Instancia de `ObjectiveFunction` (o ya envuelta en `CountedFunction`) |
| `bounds` | Tupla `(inf, sup)` para todas las dimensiones, o límites por dimensión |
| `dimension` | Dimensión $n$ del problema |
| `seed` | Semilla de la corrida |
| `budget` | Máximo de evaluaciones equivalentes $E_{eq}$ |
| `config` | Diccionario opcional con los hiperparámetros |

**Hiperparámetros** (con sus valores por defecto en el módulo):

| Clave | Significado | Por defecto | Restricción |
|---|---|---|---|
| `population_size` | $NP$ | 50 | $\ge 4$ |
| `mutation_factor` | $F$ | 0.8 | $>0$ |
| `crossover_rate` | $CR$ | 0.9 | en $[0,1]$ |
| `tolerance` | Umbral de éxito | $10^{-4}$ | |

Los valores usados en los experimentos definitivos están en `config/part1.yaml` y se documentan en `docs/experimentos_parte1.md`. Si algún parámetro es inválido, o `dimension` o `budget` son menores que 1, la función lanza `ValueError`.

**Salida.** Un diccionario con las mismas claves que los demás optimizadores:

```
best_x, best_f, f_evaluations, gradient_evaluations,
equivalent_evaluations, iterations, success, history
```

Todos los valores se devuelven como tipos nativos de Python (`float`, `int`, listas), para que se puedan guardar en CSV o JSON sin conversión.

### 3.2 Estructura del módulo

| Elemento | Función |
|---|---|
| `_standardize_bounds` | Convierte los límites recibidos en dos arreglos de longitud $n$ (mismo comportamiento que en `gradient_descent.py`) |
| `_pick_distinct_indices` | Sortea $r_1,r_2,r_3$ distintos entre sí y de $i$ |
| `_binomial_crossover` | Aplica el cruce binomial con $j_{\text{rand}}$ |
| `optimize` | Orquesta la corrida completa |

Las dos funciones auxiliares se separaron de `optimize` para poder probarlas de forma aislada: son las piezas que concentran la lógica del algoritmo.

### 3.3 Integración con el conteo de evaluaciones

Todo cálculo de $f$ pasa por `CountedFunction` (módulo `src/metrics/evaluation_counter.py`). Si `function` no viene envuelta, `optimize` la envuelve. El conteo es real, no reconstruido a partir de generaciones por tamaño de población.

DE nunca evalúa el gradiente, así que:

$$N_{grad}=0,\qquad E_{eq}=N_f+2\,n\,N_{grad}=N_f.$$

Esto permite comparar DE con descenso por gradiente bajo la misma unidad de costo. Una generación de DE cuesta $NP$ evaluaciones, mientras que una iteración de descenso por gradiente cuesta $2n+1$ (el gradiente por diferencias centrales más la evaluación del nuevo punto).

### 3.4 Algoritmo, paso a paso

```
1. Validar dimension, budget y parámetros
2. Envolver la función en CountedFunction (si hace falta)
3. rng = default_rng(seed)
4. Sortear la población inicial uniforme en [l, u]
5. Evaluar individuos mientras quepan en el presupuesto; registrar history[0]
6. Mientras quepa una evaluación más en el presupuesto:
     nueva_poblacion = copia de la población actual
     Para cada individuo i:
        si no cabe una evaluación más: terminar la generación
        elegir r1, r2, r3 distintos entre sí y de i
        mutante = x_r1 + F * (x_r2 - x_r3), recortado a [l, u]
        trial = cruce binomial(x_i, mutante, CR)   # con j_rand
        si f(trial) <= f(x_i): nueva_poblacion[i] = trial
     población = nueva_poblacion; registrar history
7. Devolver el mejor individuo, el éxito y las métricas
```

### 3.5 Decisiones de diseño

| Decisión | Elección | Justificación |
|---|---|---|
| Estrategia | `rand/1/bin` | Es la variante clásica y la que se acordó en la guía del equipo |
| Actualización | **Síncrona**: los trials se construyen leyendo la generación actual y los ganadores se escriben en una población nueva | Es la formulación original; la variante asíncrona (reemplazar en el mismo arreglo) cambiaría el significado de "generación" |
| Empates | `f(trial) <= f(x)` | Regla clásica; permite desplazarse por zonas planas |
| Límites | *Clipping* del mutante | Política común a GD, PSO, AE y DE. Como las coordenadas que no provienen del mutante ya están dentro del dominio, recortar el mutante o el trial da el mismo resultado |
| Aleatoriedad | Un solo `numpy.random.default_rng(seed)` por corrida | Reproducibilidad exacta y sin estado global |
| Fitness inicial | Vector lleno de $+\infty$ | Un individuo no evaluado nunca gana un `argmin` |
| Parada | Solo por presupuesto | Igual que GD: no se detiene al alcanzar el umbral, para que $E_{eq}$ sea comparable entre algoritmos |
| Éxito | $\lvert f_{best}-f^*\rvert\le$ `tolerance`, con $f^*$ tomado de `global_minimum_f` de la función | Sin valores fijos dentro del optimizador |

### 3.6 Política de presupuesto

- El presupuesto se consulta con `can_evaluate_f(budget)` **antes de cada evaluación**, de modo que el algoritmo nunca lo excede y se detiene exactamente al agotarlo.
- Si el presupuesto no alcanza para la población inicial (por ejemplo, 49 con $NP=50$), se evalúan los individuos que caben, se devuelve el mejor de ellos y `iterations = 0`.
- La última generación puede quedar **parcial**: `iterations` cuenta las generaciones iniciadas, no necesariamente completadas.
- Ejemplo: con $B=10\,000$ y $NP=50$ se gastan 50 evaluaciones en la población inicial y alcanzan 199 generaciones completas.

### 3.7 Historial de convergencia

`history` es una lista de registros con `step`, `best_f`, `equivalent_evaluations` y `best_x`. El paso 0 corresponde a la población inicial y cada generación añade un registro. Como la selección es voraz, `best_f` es no creciente a lo largo del historial. Este formato permite graficar $f_{best}$ contra evaluaciones equivalentes y generar las animaciones. Hay que tener presente que solo se guarda el **mejor individuo** de cada generación, no la población completa.

### 3.8 Ejemplo de uso

```python
from src.functions.rastrigin import Rastrigin
from src.optimizers.differential_evolution import optimize

f = Rastrigin()
resultado = optimize(
    f, f.default_bounds, dimension=2, seed=0, budget=10_000,
    config={"population_size": 50, "mutation_factor": 0.8, "crossover_rate": 0.9},
)
print(resultado["best_f"], resultado["equivalent_evaluations"])
```

Con la misma semilla, el resultado es idéntico entre ejecuciones.

---

## 4. Pruebas

Las pruebas están en `tests/test_differential_evolution.py` (9 tests) y se ejecutan desde la raíz del repositorio con:

```
python -m pytest tests/test_differential_evolution.py
```

| Test | Qué verifica |
|---|---|
| `test_distinct_indices` | $r_1,r_2,r_3$ distintos entre sí y de $i$ (2000 repeticiones) |
| `test_crossover_keeps_dimension_and_takes_at_least_one_from_mutant` | Con $CR=0$ cambia exactamente una coordenada ($j_{rand}$); con $CR=1$ el trial es el mutante |
| `test_budget_never_exceeded_and_counts_are_coherent` | Presupuestos 1, 3, 49, 50, 51, 120 y 1000, en ambas funciones y en 2D y 3D: nunca se excede, $N_{grad}=0$ y $E_{eq}=N_f$ |
| `test_uses_whole_budget_when_it_covers_the_population` | Con presupuesto suficiente se consume completo |
| `test_counter_matches_real_calls` | El conteo coincide con las llamadas reales a $f$ (función espía) |
| `test_same_seed_same_result_different_seed_different_result` | Reproducibilidad por semilla |
| `test_best_x_inside_bounds_even_with_large_F` | Con $F=2$ el resultado respeta los límites |
| `test_history_starts_at_step_zero_and_never_worsens` | Historial desde el paso 0 y $f_{best}$ no creciente |
| `test_invalid_config_raises` | Parámetros inválidos lanzan `ValueError` |

---

## 5. Integración con el resto del proyecto

- **Funciones y contador:** DE consume `Rastrigin` y `Rosenbrock` (`src/functions/`) a través de `CountedFunction`, sin conocer su lógica interna.
- **Runner:** el runner del equipo (`src/experiments/run_part1.py`) invoca a DE con la misma interfaz que a GD, PSO y AE, leyendo los hiperparámetros desde `config/part1.yaml`.
- **Resultados:** las corridas de DE se guardan en `results/raw/<función>_<dim>d_de.csv` y sus estadísticas (media, desviación, mejor, peor y tasa de éxito) en `results/summary/<función>_<dim>d_de_summary.csv`. La comparación entre los cuatro métodos está en `results/summary/parte1_comparacion.csv`.
- **Figuras:** gráficas comparativas en `figures/` (`mean_best_f_comparison.png`, `success_rate_comparison.png` y `best_f_distribution_*.png`) y la animación `figures/de_rosenbrock_2d.gif`.
- **Análisis:** la discusión de los resultados está en `docs/experimentos_parte1.md`.

---

## 6. Qué se hizo, en resumen

1. **Diseño:** se definió la variante (rand/1/bin), el contrato común de entrada y salida, la política de límites y el conteo de evaluaciones.
2. **Implementación** del módulo `differential_evolution.py` con control de presupuesto antes de cada evaluación, reproducibilidad por semilla e historial de convergencia.
3. **Refactorización** de los operadores principales (selección de índices y cruce) en funciones auxiliares, para poder probarlos por separado sin alterar el comportamiento.
4. **Pruebas** (9 tests) y verificación con las funciones reales del proyecto, incluyendo presupuestos de borde, reproducibilidad y coincidencia del conteo con las llamadas reales.
5. **Integración** en `develop` mediante dos commits (módulo y tests) y su incorporación al runner, a los experimentos de 30 semillas por configuración y a la comparación con los demás métodos.
6. **Visualización:** animación del proceso de DE sobre Rosenbrock 2D.

---

## 7. Limitaciones

- Se implementa una sola estrategia (rand/1/bin) con $F$ y $CR$ fijos, sin adaptación de parámetros.
- El *clipping* puede acumular individuos sobre la frontera del dominio cuando $F$ es grande. En estas dos funciones el mínimo está en el interior, por lo que no afecta al resultado, pero sí sería relevante en otros problemas.
- La última generación puede quedar incompleta, y `iterations` cuenta generaciones iniciadas.
- El historial solo guarda el mejor individuo, no la población: para animar la población completa habría que ampliar el registro.
- Como en cualquier heurística, los resultados describen el comportamiento en 30 corridas independientes y no demuestran optimalidad global.

---

## Referencias

Price, K., Storn, R. M., & Lampinen, J. A. (2005). *Differential evolution: A practical approach to global optimization*. Springer.

Storn, R., & Price, K. (1997). Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization, 11*(4), 341–359. https://doi.org/10.1023/A:1008202821328
