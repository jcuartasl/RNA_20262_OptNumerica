# Experimentación - Parte 1

## 1. Objetivo

Esta fase corresponde a la ejecución, consolidación y visualización de los experimentos de la Parte 1 del trabajo de optimización numérica.

El objetivo experimental fue comparar el comportamiento de cuatro algoritmos de optimización bajo condiciones comunes de ejecución:

- Descenso por gradiente (GD)
- Optimización por enjambre de partículas (PSO)
- Evolución diferencial (DE)
- Algoritmo evolutivo (AE)

Los métodos se evaluaron sobre las funciones:

- Rosenbrock
- Rastrigin

Cada función se estudió en:

- 2 dimensiones
- 3 dimensiones

De esta forma se obtuvieron 16 configuraciones experimentales:

```text
4 algoritmos x 2 funciones x 2 dimensiones = 16 configuraciones
```

Para cada configuración se realizaron 30 corridas independientes.

El número total de ejecuciones realizadas fue:

```text
16 configuraciones x 30 semillas = 480 corridas
```

---

## 2. Configuración experimental

La configuración oficial de los experimentos se encuentra en:

```text
config/part1.yaml
```

La configuración común utilizada fue:

- Corridas por configuración: 30
- Semillas: 0 a 29
- Presupuesto equivalente máximo: 10000 evaluaciones
- Tolerancia de éxito: 1e-4

Las mismas semillas fueron utilizadas para todos los algoritmos y configuraciones con el fin de facilitar una comparación reproducible.

Los parámetros particulares de cada algoritmo también se encuentran definidos en `config/part1.yaml`.

---

## 3. Criterio de comparación

Para realizar una comparación bajo un presupuesto común se registraron explícitamente:

- Evaluaciones de la función objetivo `f`
- Evaluaciones del gradiente
- Evaluaciones equivalentes
- Número de iteraciones
- Mejor valor encontrado
- Mejor solución encontrada
- Éxito o fracaso de la corrida

Cada ejecución está limitada por el presupuesto equivalente configurado.

El runner registra estos valores de forma independiente para permitir comparar algoritmos que utilizan información diferente durante el proceso de optimización.

El criterio de éxito utilizado consiste en verificar si el mejor valor encontrado se encuentra dentro de la tolerancia definida respecto al mínimo global conocido de la función.

---

## 4. Runner experimental

El punto de entrada utilizado para realizar los experimentos es:

```text
src/experiments/run_part1.py
```

Los optimizadores disponibles son:

```text
gd
pso
de
ae
```

Las funciones disponibles son:

```text
rosenbrock
rastrigin
```

Las dimensiones configuradas son:

```text
2
3
```

---

## 5. Ejecutar una corrida individual

Una ejecución individual se realiza indicando:

- función
- dimensión
- optimizador
- semilla

Ejemplo con evolución diferencial sobre Rosenbrock 2D:

```powershell
python -m src.experiments.run_part1 --function rosenbrock --dim 2 --optimizer de --seed 0
```

Ejemplo con algoritmo evolutivo:

```powershell
python -m src.experiments.run_part1 --function rastrigin --dim 3 --optimizer ae --seed 0
```

Ejemplo con PSO:

```powershell
python -m src.experiments.run_part1 --function rosenbrock --dim 2 --optimizer pso --seed 0
```

Ejemplo con descenso por gradiente:

```powershell
python -m src.experiments.run_part1 --function rosenbrock --dim 2 --optimizer gd --seed 0
```

---

## 6. Ejecutar las 30 semillas de una configuración

Para ejecutar las 30 semillas configuradas se utiliza:

```text
--all-seeds
```

Ejemplo:

```powershell
python -m src.experiments.run_part1 --function rosenbrock --dim 2 --optimizer de --all-seeds
```

Otro ejemplo:

```powershell
python -m src.experiments.run_part1 --function rastrigin --dim 3 --optimizer ae --all-seeds
```

El runner ejecuta automáticamente las semillas definidas en `config/part1.yaml`.

---

## 7. Archivos de resultados

Los resultados individuales de las 30 corridas de cada configuración se almacenan en:

```text
results/raw/
```

Ejemplo:

```text
results/raw/rosenbrock_2d_de.csv
results/raw/rastrigin_3d_ae.csv
```

Los resultados estadísticos consolidados de cada configuración se almacenan en:

```text
results/summary/
```

Ejemplo:

```text
results/summary/rosenbrock_2d_de_summary.csv
results/summary/rastrigin_3d_ae_summary.csv
```

Para cada configuración se reportan:

- Media
- Desviación estándar
- Mejor valor
- Peor valor
- Número de éxitos
- Tasa de éxito
- Presupuesto utilizado
- Tolerancia

---

## 8. Consolidación de los resultados

Los 16 archivos de resumen pueden consolidarse automáticamente utilizando:

```powershell
python -m src.experiments.consolidate_part1_results
```

El script genera:

```text
results/summary/parte1_comparacion.csv
```

Este archivo contiene la comparación conjunta de:

```text
GD
PSO
DE
AE
```

sobre:

```text
Rosenbrock 2D
Rosenbrock 3D
Rastrigin 2D
Rastrigin 3D
```

---

## 9. Resultados obtenidos

### Tasa de éxito

| Función | Dimensión | GD | PSO | DE | AE |
|---|---:|---:|---:|---:|---:|
| Rosenbrock | 2D | 0.00% | 100.00% | 100.00% | 16.67% |
| Rosenbrock | 3D | 0.00% | 6.67% | 100.00% | 6.67% |
| Rastrigin | 2D | 0.00% | 100.00% | 100.00% | 100.00% |
| Rastrigin | 3D | 0.00% | 96.67% | 83.33% | 96.67% |

Los valores estadísticos completos se encuentran disponibles en:

```text
results/summary/parte1_comparacion.csv
```

---

## 10. Observaciones experimentales

Los resultados muestran diferencias importantes entre los algoritmos dependiendo de la función y de la dimensión.

### Rosenbrock

En Rosenbrock 2D, PSO y evolución diferencial alcanzaron una tasa de éxito del 100%.

Evolución diferencial presentó además valores finales extremadamente cercanos al mínimo global.

El algoritmo evolutivo alcanzó una tasa de éxito del 16.67%.

Descenso por gradiente no alcanzó la tolerancia definida en ninguna de las 30 semillas.

En Rosenbrock 3D se observa una diferencia todavía mayor.

Evolución diferencial mantuvo una tasa de éxito del 100%, mientras que PSO y AE alcanzaron 6.67%.

Descenso por gradiente nuevamente obtuvo una tasa de éxito del 0%.

### Rastrigin

En Rastrigin 2D, PSO, evolución diferencial y algoritmo evolutivo alcanzaron una tasa de éxito del 100%.

Los resultados de las tres metaheurísticas estuvieron muy próximos al mínimo global.

En Rastrigin 3D, PSO y AE alcanzaron una tasa de éxito del 96.67%.

Evolución diferencial obtuvo un 83.33%.

El algoritmo evolutivo presentó además una media final muy pequeña, mostrando un comportamiento estable en esta configuración.

Descenso por gradiente no alcanzó la tolerancia definida.

### Comparación general

Los experimentos muestran que el desempeño de un algoritmo depende fuertemente de la geometría de la función objetivo.

Evolución diferencial presentó un desempeño especialmente fuerte en Rosenbrock.

PSO mostró muy buenos resultados en Rosenbrock 2D y en las configuraciones de Rastrigin.

El algoritmo evolutivo presentó un comportamiento particularmente favorable en Rastrigin.

Descenso por gradiente presentó mayores dificultades bajo el presupuesto y la configuración utilizada.

Estos resultados deben interpretarse considerando no solamente el mejor valor obtenido, sino también:

- la tasa de éxito;
- la variabilidad entre semillas;
- la media;
- la desviación estándar;
- el presupuesto equivalente consumido.

---

## 11. Generación de gráficas

El código encargado de generar las figuras comparativas se encuentra en:

```text
src/visualization/plot_part1_results.py
```

Para regenerar las gráficas se utiliza:

```powershell
python -m src.visualization.plot_part1_results
```

Las imágenes se almacenan en:

```text
figures/
```

Actualmente se generan las siguientes figuras:

```text
figures/success_rate_comparison.png
figures/mean_best_f_comparison.png

figures/best_f_distribution_rosenbrock_2d.png
figures/best_f_distribution_rosenbrock_3d.png

figures/best_f_distribution_rastrigin_2d.png
figures/best_f_distribution_rastrigin_3d.png
```

Las gráficas permiten analizar:

- Tasa de éxito de cada algoritmo
- Error final medio
- Distribución de los mejores valores obtenidos durante las 30 semillas
- Variabilidad entre ejecuciones

Para las comparaciones en las que existen diferencias de varios órdenes de magnitud se utiliza escala logarítmica.

---

## 12. Animaciones del proceso de optimización

Para visualizar el proceso de optimización se generaron dos animaciones sobre Rosenbrock 2D:

- Descenso por gradiente
- Evolución diferencial

El código se encuentra en:

```text
src/visualization/animation.py
```

Para generar las dos animaciones con un solo comando:

```powershell
python -m src.visualization.animation --optimizer both --seed 0
```

También pueden generarse individualmente.

### Descenso por gradiente

```powershell
python -m src.visualization.animation --optimizer gd --seed 0
```

### Evolución diferencial

```powershell
python -m src.visualization.animation --optimizer de --seed 0
```

Los archivos generados son:

```text
figures/gd_rosenbrock_2d.gif
figures/de_rosenbrock_2d.gif
```

Las animaciones utilizan el historial generado por cada optimizador y muestran:

- mejor solución encontrada;
- evolución de la posición;
- número de evaluaciones equivalentes;
- mejor valor de la función;
- mínimo global conocido.

---

## 13. Regeneración completa de los artefactos experimentales

Una vez disponibles los CSV de las corridas, la comparación consolidada puede regenerarse con:

```powershell
python -m src.experiments.consolidate_part1_results
```

Las gráficas pueden regenerarse con:

```powershell
python -m src.visualization.plot_part1_results
```

Los GIF pueden regenerarse con:

```powershell
python -m src.visualization.animation --optimizer both --seed 0
```

De esta manera los artefactos experimentales no dependen de edición manual.

---

## 14. Pruebas

Para ejecutar toda la suite de pruebas del proyecto:

```powershell
python -m pytest -q
```

La última ejecución realizada durante la integración experimental produjo:

```text
263 passed
```

Las pruebas incluyen verificaciones relacionadas con:

- presupuesto;
- reproducibilidad;
- límites;
- funciones objetivo;
- algoritmos implementados;
- contadores de evaluación.

---

## 15. Estructura relacionada con experimentación

Los principales archivos relacionados con esta fase son:

```text
config/
└── part1.yaml

src/
├── experiments/
│   ├── run_part1.py
│   └── consolidate_part1_results.py
│
└── visualization/
    ├── animation.py
    └── plot_part1_results.py

results/
├── raw/
└── summary/

figures/

docs/
└── experimentos_parte1.md
```

---

## 16. Reproducibilidad

Para reproducir una configuración experimental se requiere:

1. Instalar las dependencias del proyecto.
2. Utilizar la configuración definida en `config/part1.yaml`.
3. Ejecutar el runner indicando función, dimensión y algoritmo.
4. Utilizar `--all-seeds` para ejecutar las 30 semillas.
5. Consolidar los resultados.
6. Regenerar las gráficas y animaciones mediante los scripts correspondientes.

Ejemplo de ejecución completa de una configuración:

```powershell
python -m src.experiments.run_part1 --function rastrigin --dim 3 --optimizer ae --all-seeds
```

Posteriormente:

```powershell
python -m src.experiments.consolidate_part1_results
```

y:

```powershell
python -m src.visualization.plot_part1_results
```

---

## 17. Resumen de la fase experimental

La fase experimental quedó compuesta por:

```text
4 algoritmos
2 funciones
2 dimensiones
30 semillas
```

Para un total de:

```text
480 corridas
```

Además se generaron:

```text
16 archivos de resultados por configuración
1 tabla consolidada de comparación
6 figuras comparativas
2 GIF de evolución del proceso de optimización
```

Todo el procesamiento experimental puede reproducirse desde el código disponible en el repositorio.