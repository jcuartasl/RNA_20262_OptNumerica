# Fundamentación Matemática y Aporte Técnico: Persona 1
## Funciones de Prueba, Gradientes Analíticos y Descenso por Gradiente

**Autor:** Persona 1 (Funciones + Gradient Descent)  
**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Semestre:** 2026-02 — Universidad Nacional de Colombia  

---

## 1. Fundamentación Matemática de las Funciones de Prueba

### 1.1 Función de Rosenbrock (Valle de Rosenbrock / Banana Function)

La función de Rosenbrock es un clásico problema de prueba no convexo propuesto por Howard H. Rosenbrock en 1960. En dimensión $n \ge 2$, se define matemáticamente como:

$$f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[ 100 \left( x_{i+1} - x_i^2 \right)^2 + (1 - x_i)^2 \right]$$

- **Dominio experimental:** $\mathbf{x} \in [-5.0, 10.0]^n$
- **Mínimo global único:** $\mathbf{x}^* = (1, 1, \dots, 1)^T$
- **Valor en el óptimo:** $f(\mathbf{x}^*) = 0.0$

#### Derivación analítica del gradiente $\nabla f(\mathbf{x})$:
Para un vector $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$, las componentes de la sumatoria que dependen de $x_i$ varían según la posición del índice:

1. **Primera componente ($i = 1$):**  
   Solo interviene en el término $i=1$:
   $$\frac{\partial f}{\partial x_1} = 100 \cdot 2(x_2 - x_1^2)(-2x_1) + 2(1 - x_1)(-1) = -400 x_1 (x_2 - x_1^2) + 2(x_1 - 1)$$

2. **Componentes intermedias ($2 \le i \le n-1$):**  
   Interviene en el sumando anterior (con índice $i-1$) como variable cuadrática adelantada $100(x_i - x_{i-1}^2)^2$ y en el sumando actual como variable base $100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2$:
   $$\frac{\partial f}{\partial x_i} = 200(x_i - x_{i-1}^2) - 400 x_i (x_{i+1} - x_i^2) + 2(x_i - 1)$$

3. **Última componente ($i = n$):**  
   Solo interviene en el sumando final $100(x_n - x_{n-1}^2)^2$:
   $$\frac{\partial f}{\partial x_n} = 200(x_n - x_{n-1}^2)$$

En el mínimo global $\mathbf{x}^* = (1, \dots, 1)^T$, se observa de forma inmediata que $x_{i+1} - x_i^2 = 1 - 1 = 0$ y $x_i - 1 = 0$, resultando exactamente en $\nabla f(\mathbf{x}^*) = \mathbf{0}$.

---

### 1.2 Función de Rastrigin

Propuesta por Leonard Rastrigin (1974) y popularizada en optimización evolutiva por Mühlenbein et al. (1991), es una función no lineal y altamente multimodal generada a partir de una parábola modulada por oscilaciones cosenoidales:

$$f(\mathbf{x}) = 10n + \sum_{i=1}^n \left[ x_i^2 - 10 \cos(2\pi x_i) \right]$$

- **Dominio experimental:** $\mathbf{x} \in [-5.12, 5.12]^n$
- **Mínimo global único:** $\mathbf{x}^* = (0, 0, \dots, 0)^T$
- **Valor en el óptimo:** $f(\mathbf{x}^*) = 0.0$

#### Derivación analítica del gradiente $\nabla f(\mathbf{x})$:
Dado que la función es completamente separable en cada dimensión $x_i$, la derivada parcial respecto a cada componente es independiente:

$$\frac{\partial f}{\partial x_i} = \frac{d}{dx_i} \left[ x_i^2 - 10 \cos(2\pi x_i) \right] = 2x_i - 10 \left( -\sin(2\pi x_i) \cdot 2\pi \right) = 2x_i + 20\pi \sin(2\pi x_i)$$

En el mínimo global $\mathbf{x}^* = \mathbf{0}$:
$$\frac{\partial f}{\partial x_i}\bigg|_{\mathbf{x}^*} = 2(0) + 20\pi \sin(0) = 0 \implies \nabla f(\mathbf{x}^*) = \mathbf{0}$$

---

## 2. Análisis Geométrico y Comportamiento del Descenso por Gradiente

### 2.1 ¿Por qué Rosenbrock es difícil para el Descenso por Gradiente?
1. **Morfología del valle en forma de banana:** La función posee un valle parabólico muy pronunciado y estrecho alrededor de la curva $x_{i+1} \approx x_i^2$. Caer en el valle es relativamente rápido, pero converger hacia el fondo del valle hasta el punto óptimo $(1, \dots, 1)$ resulta sumamente lento.
2. **Mal condicionamiento de la matriz Hessiana ($\kappa(H) \gg 1$):**  
   Al calcular el Hessiano $H = \nabla^2 f(\mathbf{x})$ en el valle, los autovalores presentan una disparidad de magnitud de varios órdenes de magnitud:
   - Las direcciones transversales a las paredes del valle tienen curvatura muy alta (autovalor dominante grande).
   - La dirección a lo largo del suelo del valle tiene una curvatura sumamente baja (autovalor casi nulo).
3. **Comportamiento oscilatorio o divergencia:** Con una tasa de aprendizaje constante $\alpha$, el vector gradiente apunta de forma casi perpendicular a las paredes del valle en lugar de hacia el mínimo global. Si $\alpha$ es ligeramente grande, el algoritmo "rebota" de un lado del valle al otro o diverge; si $\alpha$ se reduce drásticamente para evitar la divergencia, el avance a lo largo del suelo del valle se vuelve infinitesimal, agotando el presupuesto de evaluaciones antes de acercarse a $\mathbf{x}^*$.

### 2.2 ¿Por qué Rastrigin es patológica para métodos basados en gradiente?
1. **Multimodalidad extrema:** En el dominio $[-5.12, 5.12]^n$, la función presenta aproximadamente $11^n$ mínimos locales regulares producidos por el término periódico $20\pi \sin(2\pi x_i)$.
2. **Atrapamiento en mínimos locales:** El método de descenso por gradiente es un optimizador local de búsqueda descendente voraz (*greedy*). Dada una condición inicial $\mathbf{x}_0 \sim U(-5.12, 5.12)$, la trayectoria de descenso seguirá estrictamente la dirección contraria a la pendiente local ($-\nabla f$) y quedará atrapada en el primer pozo de potencial donde $\nabla f(\mathbf{x}) = \mathbf{0}$. Salvo que la semilla aleatoria sitúe a $\mathbf{x}_0$ en la cuenca de atracción inmediata del origen ($|x_{0, i}| < 0.5$), el descenso por gradiente estándar tiene una probabilidad de éxito prácticamente nula ($0\%$), contrastando de forma contundente con la capacidad exploratoria global de los algoritmos bioinspirados (EA, PSO, DE).

---

## 3. Algoritmo de Descenso por Gradiente

### 3.1 Ecuación de Actualización
Dado un punto inicial aleatorio $\mathbf{x}_0 \sim U(\mathbf{l}, \mathbf{u})$ dentro de los límites del problema, el algoritmo actualiza la posición en cada iteración $t$:

$$\mathbf{x}_{t+1} = \operatorname{clip}\left( \mathbf{x}_t - \alpha \nabla f(\mathbf{x}_t), \, \mathbf{l}, \, \mathbf{u} \right)$$

donde:
- $\alpha > 0$ es la tasa de aprendizaje (`learning_rate`).
- $\operatorname{clip}(\mathbf{v}, \mathbf{l}, \mathbf{u})$ es la política obligatoria de límites acordada en la Fase 0:
  $$x_{i} = \min\left( \max\left( x_{i}, l_i \right), u_i \right)$$
- $\mathbf{l}, \mathbf{u}$ representan las cotas inferiores y superiores del dominio de búsqueda.

### 3.2 Regla de Comparación Justa y Presupuesto
Para permitir una comparación rigurosa entre algoritmos basados en gradiente y algoritmos metaheurísticos libres de derivadas, se implementó el conteo contable de costo computacional equivalente:

$$E_{eq} = N_f + 2n \cdot N_{grad}$$

**Justificación:** En optimización en caja negra (*black-box*), el gradiente no se conoce analíticamente de antemano y debe aproximarse mediante diferencias finitas centrales:
$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h \mathbf{e}_i) - f(\mathbf{x} - h \mathbf{e}_i)}{2h}$$
Aproximar las $n$ derivadas parciales cuesta exactamente $2n$ evaluaciones de la función objetivo $f$. Por lo tanto, atribuir $2n$ unidades equivalentes a cada cálculo del gradiente sitúa a todos los métodos bajo el mismo presupuesto contable de información extraída de la función.

---

## 4. Guía para el Video de Contribución Individual (Persona 1)

El video debe durar entre **2 y 3 minutos**, hablando en primera persona y mostrando evidencia concreta en pantalla:

### Estructura recomendada:
1. **Presentación (15-20 s):** Nombre completo, rol de **Persona 1: Funciones de Prueba y Descenso por Gradiente**.
2. **Evidencia de Código y Ecuaciones (45-60 s):**
   - Mostrar el archivo `src/functions/rosenbrock.py` y explicar en 30 segundos la derivación analítica del gradiente en $n$ dimensiones: mostrar cómo se separan la primera componente, las intermedias y la última para evitar la alucinación típica de las IAs.
   - Mostrar el gradiente analítico en `src/functions/rastrigin.py` y el signo $+20\pi \sin(2\pi x_i)$.
3. **Demostración de Pruebas Unitarias en Vivo (30 s):**
   - Ejecutar en consola: `python -m pytest tests/ -v`.
   - Señalar que las 48 pruebas pasan al 100%, destacando la prueba cruzada contra diferencias finitas centrales (`numerical_gradient`) y la prueba de no sobrepaso del presupuesto (`budget`).
4. **Alucinación Cazada (30 s):**
   - Mencionar el hallazgo `H-01` o `H-02` de `docs/hallucination_log.md`: explicar cómo una IA derivó incorrectamente el signo o la componente intermedia y cómo la prueba unitaria lo detectó.
5. **Cierre (15 s):** Indicar que el módulo está listo para la integración en la rama común de `develop`.

---

## 5. Referencias Bibliográficas (Formato APA 7.ª Edición)

- Mühlenbein, H., Schomisch, D., & Born, J. (1991). The parallel genetic algorithm as a global optimizer. *Parallel Computing*, 17(6–7), 619–632. https://doi.org/10.1016/S0167-8191(08)80052-3
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-40065-5
- Rastrigin, L. A. (1974). Systems of extremal control. *Nauka*, Moscow (in Russian).
- Rosenbrock, H. H. (1960). An automatic method for finding the greatest or least value of a function. *The Computer Journal*, 3(3), 175–184. https://doi.org/10.1093/comjnl/3.3.175
